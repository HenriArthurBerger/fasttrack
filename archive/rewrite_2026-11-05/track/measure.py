"""Per-cell measurements, as flat tensors.

Every pixel of a chunk of frames is scattered at once onto compact label ids, so cost
depends on pixel count and not on cell count or density. Exact medians come from one
sort over (segment, value) keys, whose memory does not scale with the intensity range.

Output: scalars (N, S) float32, one row per cell-frame, and channels (N, C, K) float32.
"""
from __future__ import annotations

import math
import warnings

import torch

from ..core.filters import max_cross

SCALARS = [
    "t", "label", "x", "y", "nuc_area", "cyto_area", "edge_n", "eccentricity", "elongation", "orientation",
    "major_len", "minor_len", "ellipse_ratio", "extent", "bbox_aspect", "radial_rough", "circularity",
    "hu1", "hu2", "concentricity", "corr_01", "corr_02", "corr_12", "local_density", "prev_inside",
    "next_inside", "underseg_score", "touches_border",
]
CHANNELS = [
    "nuc_mean", "nuc_median", "nuc_sum", "nuc_std", "nuc_min", "nuc_max",
    "cyto_mean", "cyto_median", "cyto_sum", "cyto_std", "cyto_min", "cyto_max",
    "nc_ratio", "nuc_sum_frac", "chan_centroid_off", "bimodality",
]
S = {name: i for i, name in enumerate(SCALARS)}
K = {name: i for i, name in enumerate(CHANNELS)}

warnings.filterwarnings("ignore", message="index_reduce")  # beta notice, once per process

_KEY_BITS = 22
_KEY_OFFSET = 1 << 18  # corrected values may be negative
# An 8-connected boundary of a disc has about 4*sqrt(2)*r pixels, against a true perimeter
# of 2*pi*r. This rescales the count so a disc scores a circularity near 1.
_PERIMETER_SCALE = math.pi / (2 * math.sqrt(2))


def _segment_stats(values: torch.Tensor, segments: torch.Tensor, n: int) -> dict:
    """values (P,C), segments (P,) in 0..n-1 -> count, sum, mean, std, min, median, max."""
    device = values.device
    C = values.shape[1]
    count = torch.bincount(segments, minlength=n)
    total = torch.zeros(n, C, device=device, dtype=torch.float64).index_add_(0, segments, values.double())
    square = torch.zeros(n, C, device=device, dtype=torch.float64).index_add_(0, segments, (values * values).double())
    safe = count.clamp(min=1).unsqueeze(1)
    mean = total / safe
    std = (square / safe - mean * mean).clamp(min=0).sqrt()
    starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), count.cumsum(0)[:-1]])
    last = (starts + count - 1).clamp(min=0)
    mid = starts + ((count - 1).clamp(min=0) // 2)
    vmin = torch.full((n, C), torch.nan, device=device)
    vmed = vmin.clone()
    vmax = vmin.clone()
    present = count > 0
    if segments.numel():
        ints = (values.round().long() + _KEY_OFFSET).clamp(0, (1 << _KEY_BITS) - 1)
        for c in range(C):
            keys = torch.sort((segments << _KEY_BITS) | ints[:, c]).values
            decode = lambda i: ((keys[i] & ((1 << _KEY_BITS) - 1)) - _KEY_OFFSET).float()
            vmin[present, c] = decode(starts[present])
            vmed[present, c] = decode(mid[present])
            vmax[present, c] = decode(last[present])
    empty = ~present
    mean_f, std_f, total_f = mean.float(), std.float(), total.float()
    for arr in (mean_f, std_f, total_f):
        arr[empty] = torch.nan
    return {"count": count, "sum": total_f, "mean": mean_f, "std": std_f, "min": vmin, "median": vmed, "max": vmax}


def _measure_chunk(data: torch.Tensor, signed: torch.Tensor, first_label: int, n: int, frame_of: torch.Tensor,
                   peaks_xy: torch.Tensor, t0: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure one chunk. data (t,C,X,Y); signed (t,1,X,Y). Labels are first_label..first_label+n-1."""
    device = data.device
    C = data.shape[1]
    scalars = torch.full((n, len(SCALARS)), torch.nan, device=device)
    channels = torch.full((n, C, len(CHANNELS)), torch.nan, device=device)
    scalars[:, S["t"]] = frame_of.float()
    scalars[:, S["label"]] = (first_label + torch.arange(n, device=device)).float()

    frame = signed[:, 0]
    nz = (frame != 0).nonzero(as_tuple=False)
    tt, xx, yy = nz[:, 0], nz[:, 1], nz[:, 2]
    lab = frame[tt, xx, yy]
    seg = lab.abs().long() - first_label
    inside = (seg >= 0) & (seg < n)  # a ring can only carry labels of this chunk; kept as a safety
    tt, xx, yy, lab, seg = tt[inside], xx[inside], yy[inside], lab[inside], seg[inside]
    values = data[tt, :, xx, yy]
    nuc = lab > 0

    nuc_stats = _segment_stats(values[nuc], seg[nuc], n)
    cyto_stats = _segment_stats(values[~nuc], seg[~nuc], n)
    for prefix, st in (("nuc", nuc_stats), ("cyto", cyto_stats)):
        for name in ("mean", "median", "sum", "std", "min", "max"):
            channels[:, :, K[f"{prefix}_{name}"]] = st[name]
    channels[:, :, K["nc_ratio"]] = nuc_stats["mean"] / cyto_stats["mean"]
    cyto_sum = torch.nan_to_num(cyto_stats["sum"], nan=0.0)
    channels[:, :, K["nuc_sum_frac"]] = nuc_stats["sum"] / (nuc_stats["sum"] + cyto_sum)
    area = nuc_stats["count"].float()
    scalars[:, S["nuc_area"]] = area
    scalars[:, S["cyto_area"]] = cyto_stats["count"].float()

    # --- nuclear shape. Coordinates relative to the seed keep float32 exact.
    seg_n = seg[nuc]
    rx = xx[nuc].float() - peaks_xy[seg_n, 0]
    ry = yy[nuc].float() - peaks_xy[seg_n, 1]
    acc = torch.zeros(n, 5, device=device).index_add_(0, seg_n, torch.stack([rx, ry, rx * rx, ry * ry, rx * ry], 1))
    safe = area.clamp(min=1)
    mx, my = acc[:, 0] / safe, acc[:, 1] / safe
    cxx, cyy, cxy = acc[:, 2] / safe - mx * mx, acc[:, 3] / safe - my * my, acc[:, 4] / safe - mx * my
    cx, cy = peaks_xy[:, 0] + mx, peaks_xy[:, 1] + my
    middle = (cxx + cyy) / 2
    spread = (((cxx - cyy) / 2) ** 2 + cxy * cxy).clamp(min=0).sqrt()
    lam1, lam2 = middle + spread, (middle - spread).clamp(min=0)
    major, minor = 4 * lam1.clamp(min=0).sqrt(), 4 * lam2.sqrt()
    ellipse_area = math.pi * major * minor / 4
    r_nuc = (safe / math.pi).sqrt()
    norm = safe * safe
    eta20, eta02, eta11 = cxx * area / norm, cyy * area / norm, cxy * area / norm
    big = torch.full((n,), 1e9, device=device)
    rx_n, ry_n = xx[nuc].float(), yy[nuc].float()
    xmin = big.clone().index_reduce_(0, seg_n, rx_n, "amin")
    xmax = (-big).index_reduce_(0, seg_n, rx_n, "amax")
    ymin = big.clone().index_reduce_(0, seg_n, ry_n, "amin")
    ymax = (-big).index_reduce_(0, seg_n, ry_n, "amax")
    height, width = (xmax - xmin + 1).clamp(min=1), (ymax - ymin + 1).clamp(min=1)
    X, Y = data.shape[-2:]
    # Clipped by the field: area and shape are underestimates, so analysis can drop these.
    scalars[:, S["touches_border"]] = ((xmin <= 0) | (ymin <= 0) | (xmax >= X - 1) | (ymax >= Y - 1)).float()

    # --- within-nucleus moments on per-cell z-scores: float32 is safe at any brightness
    mu = nuc_stats["mean"][seg_n]
    sd = nuc_stats["std"][seg_n].clamp(min=1e-3)
    z = (values[nuc] - mu) / sd
    moments = torch.zeros(n, C, 2, device=device).index_add_(0, seg_n, torch.stack([z**3, z**4], 2))
    skew = moments[:, :, 0] / safe.unsqueeze(1)
    kurt = moments[:, :, 1] / safe.unsqueeze(1) - 3
    nk = area.clamp(min=4).unsqueeze(1)
    bimodality = (skew**2 + 1) / (kurt + 3 * (nk - 1) ** 2 / ((nk - 2) * (nk - 3))).clamp(min=1e-3)
    bimodality[area < 8] = torch.nan
    channels[:, :, K["bimodality"]] = bimodality
    for name, a, b in (("corr_01", 0, 1), ("corr_02", 0, 2), ("corr_12", 1, 2)):
        if b < C:
            scalars[:, S[name]] = torch.zeros(n, device=device).index_add_(0, seg_n, z[:, a] * z[:, b]) / safe

    # --- how far each channel's bright centre sits from the shape centre
    weights = values[nuc].clamp(min=0)
    wsum = torch.zeros(n, C, device=device).index_add_(0, seg_n, weights).clamp(min=1e-3)
    wx = torch.zeros(n, C, device=device).index_add_(0, seg_n, weights * rx.unsqueeze(1)) / wsum
    wy = torch.zeros(n, C, device=device).index_add_(0, seg_n, weights * ry.unsqueeze(1)) / wsum
    channels[:, :, K["chan_centroid_off"]] = torch.hypot(wx - mx.unsqueeze(1), wy - my.unsqueeze(1)) / r_nuc.unsqueeze(1)

    # --- nuclear boundary: inner edge, so a border shared with a neighbour counts too
    positive = frame.clamp(min=0).float().unsqueeze(1)
    dilate = max_cross(positive)
    erode = -max_cross(-positive)
    edge = ((dilate != positive) | (erode != positive)) & (positive > 0)
    ez = edge[:, 0].nonzero(as_tuple=False)
    if ez.numel():
        seg_e = (positive[ez[:, 0], 0, ez[:, 1], ez[:, 2]].long() - first_label)
        keep = (seg_e >= 0) & (seg_e < n)
        ez, seg_e = ez[keep], seg_e[keep]
        er = torch.hypot(ez[:, 1].float() - cx[seg_e], ez[:, 2].float() - cy[seg_e])
        eacc = torch.zeros(n, 3, device=device).index_add_(0, seg_e, torch.stack([torch.ones_like(er), er, er * er], 1))
        en = eacc[:, 0].clamp(min=1)
        emean = eacc[:, 1] / en
        scalars[:, S["edge_n"]] = eacc[:, 0]
        scalars[:, S["radial_rough"]] = (eacc[:, 2] / en - emean**2).clamp(min=0).sqrt() / emean.clamp(min=1e-3)
        scalars[:, S["circularity"]] = 4 * math.pi * area / (eacc[:, 0] * _PERIMETER_SCALE).clamp(min=1) ** 2

    # --- nuclear versus ring centre: a ring pushed to one side means a crowded neighbourhood
    seg_c = seg[~nuc]
    if seg_c.numel():
        cacc = torch.zeros(n, 2, device=device).index_add_(0, seg_c, torch.stack([xx[~nuc].float(), yy[~nuc].float()], 1))
        cn = cyto_stats["count"].float().clamp(min=1)
        offset = torch.hypot(cacc[:, 0] / cn - cx, cacc[:, 1] / cn - cy) / r_nuc
        offset[cyto_stats["count"] == 0] = torch.nan
        scalars[:, S["concentricity"]] = offset

    updates = {
        "x": cx, "y": cy, "eccentricity": (1 - lam2 / lam1.clamp(min=1e-6)).clamp(0, 1).sqrt(),
        "elongation": (lam1.clamp(min=1e-6) / lam2.clamp(min=1e-6)).sqrt(), "orientation": 0.5 * torch.atan2(2 * cxy, cxx - cyy),
        "major_len": major, "minor_len": minor, "ellipse_ratio": area / ellipse_area.clamp(min=1e-6),
        "extent": area / (height * width), "bbox_aspect": height / width,
        "hu1": eta20 + eta02, "hu2": (eta20 - eta02) ** 2 + 4 * eta11**2,
    }
    for name, value in updates.items():
        scalars[:, S[name]] = value
    scalars[area == 0, 2:] = torch.nan
    return scalars, channels


def _fill_neighbours(scalars: torch.Tensor, signed: torch.Tensor) -> None:
    """Crowding and temporal consistency from centroids.

    prev_inside and next_inside count how many cells of the adjacent frame have their centre
    inside this nucleus. Two predecessors inside one nucleus is the signature of a merge.
    """
    T = signed.shape[0]
    device = scalars.device
    frames = scalars[:, S["t"]].long()
    x, y, area = scalars[:, S["x"]], scalars[:, S["y"]], scalars[:, S["nuc_area"]]
    valid = ~torch.isnan(x)
    rows_of = [((frames == t) & valid).nonzero(as_tuple=True)[0] for t in range(T)]
    X, Y = signed.shape[-2:]
    first_label = scalars[:, S["label"]].min() if scalars.numel() else 0
    for t in range(T):
        rows = rows_of[t]
        if rows.numel() == 0:
            continue
        pts = torch.stack([x[rows], y[rows]], 1)
        d = torch.cdist(pts, pts)
        r = (area[rows] / math.pi).clamp(min=1).sqrt()
        scalars[rows, S["local_density"]] = ((d <= 2 * r.unsqueeze(1)) & (d > 0)).sum(1).float()
        for name, other in (("prev_inside", t - 1), ("next_inside", t + 1)):
            if other < 0 or other >= T or rows_of[other].numel() == 0:
                continue
            o = rows_of[other]
            px = x[o].round().long().clamp(0, X - 1)
            py = y[o].round().long().clamp(0, Y - 1)
            hit = signed[t, 0, px, py]
            hit = hit[hit > 0]
            seg = (hit.long() - first_label.long())
            counts = torch.bincount(seg, minlength=scalars.shape[0]).float()
            scalars[rows, S[name]] = counts[rows]


def measure_batch(data: torch.Tensor, signed: torch.Tensor, frame_of_label: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """data (T,C,X,Y), signed (T,1,X,Y) with compact labels. Returns (scalars, channels)."""
    device = data.device
    N = frame_of_label.numel()
    C = data.shape[1]
    if N == 0:
        return torch.zeros(0, len(SCALARS), device=device), torch.zeros(0, C, len(CHANNELS), device=device)
    T = data.shape[0]
    chunk = int(cfg["measure"]["chunk_frames"])
    # Coordinates are computed relative to each label's bounding-box corner, chosen
    # deterministically so results do not depend on GPU write order.
    frame_counts = torch.bincount(frame_of_label, minlength=T)
    cumulative = torch.cat([torch.zeros(1, device=device, dtype=torch.long), frame_counts.cumsum(0)])
    all_scalars, all_channels = [], []
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        first, last = int(cumulative[t0]), int(cumulative[t1])
        n = last - first
        if n == 0:
            continue
        section = signed[t0:t1]
        positive = section.clamp(min=0)
        nz = positive.nonzero(as_tuple=False)
        seg = positive[nz[:, 0], 0, nz[:, 2], nz[:, 3]].long() - (first + 1)
        big = torch.full((n,), 1e9, device=device)
        anchors = torch.stack([big.clone().scatter_reduce(0, seg, nz[:, 2].float(), "amin"),
                               big.clone().scatter_reduce(0, seg, nz[:, 3].float(), "amin")], dim=1)
        anchors[anchors >= 1e9] = 0.0  # a label whose pixels were all lost
        sc, ch = _measure_chunk(data[t0:t1], section, first + 1, n, frame_of_label[first:last], anchors, t0)
        all_scalars.append(sc)
        all_channels.append(ch)
    scalars = torch.cat(all_scalars)
    channels = torch.cat(all_channels)
    _fill_neighbours(scalars, signed)
    return scalars, channels
