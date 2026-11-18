"""Second pass: find nuclei that are really two cells, and give them two seeds.

Undersegmentation does the most damage to tracking: one blob holds two identities. When two
nuclei of the previous frame have their centres in one nucleus now, its pixels are split by
k-means on position with those centres as seeds. K-means rather than an intensity watershed
because overlapping widefield nuclei are brighter where they overlap, so there is no dim valley.

`score_undersegmentation` is a shape score saved with the measurements. It does not split
anything: splitting on shape alone made tracking worse on calibrated synthetic data.
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


def _split_round(labels: torch.Tensor, step: int, min_source: float, min_blob: float, min_child: float, iters: int,
                 max_seeds: int = 6) -> int:
    """One vectorised round over the whole batch, in one direction.

    step = +1: a nucleus at t takes seeds from frame t-1. step = -1: from frame t+1.
    Every frame is handled at once from the state at the start of the round.
    """
    T, _, X, Y = labels.shape
    device = labels.device
    nz = (labels > 0).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return 0
    ids, inverse = torch.unique(labels[nz[:, 0], 0, nz[:, 2], nz[:, 3]].long(), return_inverse=True)
    n = ids.numel()
    area = torch.bincount(inverse, minlength=n).float()
    cx = torch.zeros(n, device=device).index_add_(0, inverse, nz[:, 2].float()) / area
    cy = torch.zeros(n, device=device).index_add_(0, inverse, nz[:, 3].float()) / area
    frame = torch.zeros(n, dtype=torch.long, device=device)
    frame[inverse] = nz[:, 0]

    # Where does each real nucleus's centre land in the next frame (in this direction)?
    target_frame = frame + step
    source = (area >= min_source) & (target_frame >= 0) & (target_frame < T)
    src = source.nonzero(as_tuple=True)[0]
    hit_label = labels[target_frame[src], 0, cx[src].round().long().clamp(0, X - 1), cy[src].round().long().clamp(0, Y - 1)].long()
    landed = hit_label > 0
    src, hit_label = src[landed], hit_label[landed]
    if src.numel() == 0:
        return 0
    target = torch.searchsorted(ids, hit_label)  # compact index of the nucleus hit
    hits = torch.bincount(target, minlength=n)
    suspect = (hits >= 2) & (hits <= max_seeds) & (area >= min_blob)
    if not suspect.any():
        return 0

    # Seeds: the centres that landed in each suspect, padded to the most seeds any suspect has.
    keep = suspect[target]
    src, target = src[keep], target[keep]
    suspects = suspect.nonzero(as_tuple=True)[0]
    slot_of = torch.full((n,), -1, dtype=torch.long, device=device)
    slot_of[suspects] = torch.arange(suspects.numel(), device=device)
    seg_of_seed = slot_of[target]
    order = torch.argsort(seg_of_seed, stable=True)
    seg_sorted = seg_of_seed[order]
    first = torch.ones_like(seg_sorted, dtype=torch.bool)
    first[1:] = seg_sorted[1:] != seg_sorted[:-1]
    position = torch.arange(seg_sorted.numel(), device=device)
    rank = position - position.where(first, torch.zeros_like(position)).cummax(0).values
    k = int(hits[suspects].max().item())
    seeds = torch.full((suspects.numel(), k, 2), torch.nan, device=device)
    seeds[seg_sorted, rank, 0] = cx[src[order]]
    seeds[seg_sorted, rank, 1] = cy[src[order]]

    # All suspects' pixels, split together.
    pixel_seg = slot_of[inverse]
    mine = pixel_seg >= 0
    pix, seg = nz[mine], pixel_seg[mine]
    cluster = _kmeans_split(pix[:, 2].float(), pix[:, 3].float(), seg, seeds, iters)
    sizes = torch.bincount(seg * k + cluster, minlength=suspects.numel() * k).view(-1, k).float()
    used = ~torch.isnan(seeds[:, :, 0])
    accept = ((sizes >= min_child) | ~used).all(dim=1)
    if not accept.any():
        return 0

    # New labels for every accepted child after the first, numbered after the batch's highest label.
    new_child = used.clone()
    new_child[:, 0] = False
    new_child &= accept[:, None]
    numbers = torch.cumsum(new_child.view(-1).long(), 0).view(-1, k) - 1
    next_id = labels.max() + 1
    moved = new_child[seg, cluster]
    p = pix[moved]
    labels[p[:, 0], 0, p[:, 2], p[:, 3]] = next_id + numbers[seg[moved], cluster[moved]].to(labels.dtype)
    return int(accept.sum())


def temporal_split(labels: torch.Tensor, median_area: float, cfg: dict) -> int:
    """A nucleus holding the centres of 2+ nuclei from the neighbouring frame inherits them as seeds.

    Runs on the whole batch at once, alternating forward (seeds from t-1) and backward (from t+1).
    Each round lets a split travel one more frame, so rounds repeat until nothing changes, which
    carries a split along a lasting overlap. Guards: the merged blob must be large, and each
    source must be a real nucleus, so a stray over-split is not propagated and mitosis is left alone.
    """
    p = cfg["underseg"]
    args = (0.5 * median_area, 1.3 * median_area, p["min_child_area_frac"] * median_area, int(p["kmeans_iters"]))
    splits = 0
    for _ in range(int(p["max_rounds"])):
        done = _split_round(labels, +1, *args) + _split_round(labels, -1, *args)
        splits += done
        if done == 0:
            break
    return splits
