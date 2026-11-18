"""Second pass: find nuclei that are really two cells, and give them two seeds.

Undersegmentation does the most damage to tracking: one blob holds two identities. Each
suspect's pixels are split by k-means on position, run for all suspects at once. K-means
rather than an intensity watershed because overlapping widefield nuclei are brighter
where they overlap, so there is no dim valley to find.

Two sources of suspects:
  * temporal (default): two nuclei of the previous frame now have their centres in one nucleus
  * static (off by default): a shape score. At realistic density, too many false splits
"""
from __future__ import annotations

import torch

from .measure import K, S

# Logistic weights fitted by benchmarks/underseg_features.py on the calibrated synthetic preset
# (held-out AUC 0.94). They reflect the synthetic model, so re-check them on real data.
SCORE_WEIGHTS = {
    "bias": 10.642,
    "area_ratio": 2.248,
    "eccentricity": 0.0,
    "ellipse_ratio": -7.741,
    "circularity": -8.413,
    "radial_rough": 1.743,
    "bimodality": 7.884,
    "extent": -4.766,
}


def score_features(scalars: torch.Tensor, channels: torch.Tensor, nuclear_channel: int = 0) -> dict:
    area = scalars[:, S["nuc_area"]]
    present = area[~torch.isnan(area)]
    median = present.median().clamp(min=1) if present.numel() else torch.tensor(1.0, device=area.device)
    return {
        "area_ratio": area / median,
        "eccentricity": scalars[:, S["eccentricity"]],
        "ellipse_ratio": scalars[:, S["ellipse_ratio"]],
        "circularity": scalars[:, S["circularity"]],
        "radial_rough": scalars[:, S["radial_rough"]],
        "bimodality": channels[:, nuclear_channel, K["bimodality"]],
        "extent": scalars[:, S["extent"]],
    }


def score_undersegmentation(scalars: torch.Tensor, channels: torch.Tensor, nuclear_channel: int = 0) -> torch.Tensor:
    """Probability-like score per detection. NaN features contribute nothing."""
    features = score_features(scalars, channels, nuclear_channel)
    logit = torch.full_like(scalars[:, 0], SCORE_WEIGHTS["bias"])
    for name, value in features.items():
        logit = logit + SCORE_WEIGHTS[name] * torch.nan_to_num(value, nan=0.0)
    return torch.sigmoid(logit)


def _kmeans_split(xs: torch.Tensor, ys: torch.Tensor, seg: torch.Tensor, seeds: torch.Tensor, iters: int) -> torch.Tensor:
    """Pixels (P,) in segments, seeds (n, k, 2) with NaN for unused seeds. Returns cluster (P,)."""
    n, k, _ = seeds.shape
    device = xs.device
    centres = seeds.clone()
    cluster = torch.zeros_like(seg)
    for _ in range(iters):
        dx = xs.unsqueeze(1) - centres[seg, :, 0]
        dy = ys.unsqueeze(1) - centres[seg, :, 1]
        d2 = torch.nan_to_num(dx * dx + dy * dy, nan=float("inf"))
        cluster = d2.argmin(1)
        key = seg * k + cluster
        count = torch.bincount(key, minlength=n * k).float()
        sx = torch.zeros(n * k, device=device).index_add_(0, key, xs)
        sy = torch.zeros(n * k, device=device).index_add_(0, key, ys)
        new = torch.stack([sx / count.clamp(min=1), sy / count.clamp(min=1)], 1).view(n, k, 2)
        centres = torch.where((count.view(n, k, 1) > 0), new, centres)
    return cluster


def _principal_axis_seeds(xs, ys, seg, n) -> torch.Tensor:
    """Two starting centres per segment, placed along the major axis at +/- one sd."""
    device = xs.device
    count = torch.bincount(seg, minlength=n).float().clamp(min=1)
    add = lambda v: torch.zeros(n, device=device).index_add_(0, seg, v)
    mx, my = add(xs) / count, add(ys) / count
    rx, ry = xs - mx[seg], ys - my[seg]
    cxx, cyy, cxy = add(rx * rx) / count, add(ry * ry) / count, add(rx * ry) / count
    theta = 0.5 * torch.atan2(2 * cxy, cxx - cyy)
    spread = (0.5 * (cxx + cyy) + (((cxx - cyy) / 2) ** 2 + cxy**2).sqrt()).sqrt()
    dx, dy = theta.cos() * spread, theta.sin() * spread
    return torch.stack([torch.stack([mx + dx, my + dy], 1), torch.stack([mx - dx, my - dy], 1)], 1)


def split_flagged(labels: torch.Tensor, flagged: torch.Tensor, min_area: float, iters: int) -> tuple[torch.Tensor, int]:
    """Split each flagged label in two. labels (T,1,X,Y) float32; flagged is 1D label ids.

    Returns (labels, number accepted). A split that would leave a child under `min_area`
    is rejected: that nucleus keeps its single label.
    """
    if flagged.numel() == 0:
        return labels, 0
    device = labels.device
    top = int(labels.max().item())
    lut = torch.full((top + 1,), -1, dtype=torch.long, device=device)
    lut[flagged.long()] = torch.arange(flagged.numel(), device=device)
    full = lut[labels.long()]
    nz = (full >= 0).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return labels, 0
    seg = full[nz[:, 0], 0, nz[:, 2], nz[:, 3]]
    xs, ys = nz[:, 2].float(), nz[:, 3].float()
    n = flagged.numel()
    cluster = _kmeans_split(xs, ys, seg, _principal_axis_seeds(xs, ys, seg, n), iters)
    second = cluster == 1
    size_b = torch.bincount(seg[second], minlength=n).float()
    size_a = torch.bincount(seg[~second], minlength=n).float()
    accept = (size_a >= min_area) & (size_b >= min_area)
    new_ids = (top + 1 + torch.arange(n, device=device)).to(labels.dtype)
    moved = second & accept[seg]
    pix = nz[moved]
    labels[pix[:, 0], 0, pix[:, 2], pix[:, 3]] = new_ids[seg[moved]]
    return labels, int(accept.sum())


def _frame_centroids(frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frame (X,Y) float labels -> (ids, x, y, area) for labels present."""
    nz = (frame > 0).nonzero(as_tuple=False)
    if nz.numel() == 0:
        empty = torch.zeros(0, device=frame.device)
        return empty.long(), empty, empty, empty
    ids, inverse = torch.unique(frame[nz[:, 0], nz[:, 1]].long(), return_inverse=True)
    count = torch.bincount(inverse).float()
    sum_x = torch.zeros(ids.numel(), device=frame.device).index_add_(0, inverse, nz[:, 0].float())
    sum_y = torch.zeros(ids.numel(), device=frame.device).index_add_(0, inverse, nz[:, 1].float())
    return ids, sum_x / count, sum_y / count, count


def _split_with_seeds(labels, t, target, seed_xy, min_child, iters):
    """Split label `target` in frame t with the given seeds (k,2). Returns whether it split."""
    frame = labels[t, 0]
    nz = (frame == target).nonzero(as_tuple=False)
    xs, ys = nz[:, 0].float(), nz[:, 1].float()
    seg = torch.zeros_like(xs, dtype=torch.long)
    cluster = _kmeans_split(xs, ys, seg, seed_xy.unsqueeze(0), iters)
    sizes = torch.bincount(cluster, minlength=seed_xy.shape[0]).float()
    if sizes.min() < min_child:
        return False
    next_id = int(labels.max().item()) + 1
    for k in range(1, seed_xy.shape[0]):
        pick = nz[cluster == k]
        frame[pick[:, 0], pick[:, 1]] = float(next_id)
        next_id += 1
    return True


def temporal_split(labels: torch.Tensor, median_area: float, cfg: dict) -> int:
    """Forward sweep: a nucleus holding the centres of 2+ previous nuclei inherits them as seeds.

    Each frame uses the previous frame's result, so one split carries along a whole transient
    overlap. A backward sweep then fixes overlaps already present at the start of the batch.
    Guards: the merged blob must be large, and each predecessor must be a real nucleus. This
    keeps a stray over-split of one nucleus from being propagated, and leaves mitosis alone.
    """
    p = cfg["underseg"]
    T = labels.shape[0]
    min_child = p["min_child_area_frac"] * median_area
    min_blob = 1.3 * median_area
    min_source = 0.5 * median_area
    iters = int(p["kmeans_iters"])
    X, Y = labels.shape[-2:]
    splits = 0

    def sweep(order):
        nonlocal splits
        for prev, t in order:
            ids_p, xp, yp, area_p = _frame_centroids(labels[prev, 0])
            if ids_p.numel() < 2:
                continue
            big_enough = area_p >= min_source
            xp, yp = xp[big_enough], yp[big_enough]
            frame = labels[t, 0]
            hits = frame[xp.round().long().clamp(0, X - 1), yp.round().long().clamp(0, Y - 1)].long()
            occupied = hits > 0
            if not occupied.any():
                continue
            targets, counts = hits[occupied].unique(return_counts=True)
            suspects = targets[counts >= 2]
            for target in suspects.tolist():
                area = (frame == target).sum()
                if area < min_blob:
                    continue
                pick = hits == target
                seeds = torch.stack([xp[pick], yp[pick]], 1)
                if _split_with_seeds(labels, t, target, seeds, min_child, iters):
                    splits += 1

    sweep([(t - 1, t) for t in range(1, T)])
    sweep([(t + 1, t) for t in range(T - 2, -1, -1)])
    return splits
