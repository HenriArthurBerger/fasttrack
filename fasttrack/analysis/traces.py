"""Single-cell analysis, ported from the KTR heatmap notebook. Everything takes (cells, T) arrays."""
from __future__ import annotations

import functools

import numpy as np
from scipy.signal import savgol_filter

from ..io.store import metric_matrix


@functools.lru_cache(maxsize=1)
def _sort_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nanquantiles(a: np.ndarray, qs, axis: int = 0) -> np.ndarray:
    """Quantiles ignoring NaN, along an axis: the same values as np.nanpercentile's linear method.
    One parallel sort in torch (on the GPU when there is one) instead of numpy's per-column loop.
    `qs` in 0-1. Returns (len(qs), ...) with NaN where a column has no values."""
    import torch
    a = np.moveaxis(np.asarray(a), axis, -1)          # sort along the last, contiguous axis
    shape = a.shape[:-1]
    x = torch.as_tensor(np.ascontiguousarray(a.reshape(-1, a.shape[-1])), dtype=torch.float64, device=_sort_device())
    valid = ~torch.isnan(x)
    n = valid.sum(1)
    s = torch.sort(torch.where(valid, x, torch.inf), dim=1).values   # NaN (as inf) sorts last
    out = []
    for q in np.atleast_1d(qs):
        pos = (n - 1).clamp(min=0).double() * float(q)
        lo = pos.floor().long()
        hi = torch.minimum(lo + 1, (n - 1).clamp(min=0))
        v_lo = s.gather(1, lo[:, None])[:, 0]
        v_hi = s.gather(1, hi[:, None])[:, 0]
        v = v_lo + (v_hi - v_lo) * (pos - lo)
        out.append(torch.where(n > 0, v, torch.nan).cpu().numpy().reshape(shape))
    return np.stack(out)


def grouped_quantiles(group: np.ndarray, values: np.ndarray, n_groups: int, qs) -> np.ndarray:
    """Quantiles of `values` within each group 0..n_groups-1 (e.g. per frame), linear method as
    np.percentile. Values must be finite. One sort over all rows. Returns (len(qs), n_groups), NaN for empty groups."""
    import torch
    device = _sort_device()
    g = torch.as_tensor(np.asarray(group, dtype=np.int64), device=device)
    v = torch.as_tensor(np.asarray(values, dtype=np.float64), device=device)
    order = torch.sort(v, stable=True).indices
    order = order[torch.sort(g[order], stable=True).indices]   # by group, then by value
    v = v[order]
    n = torch.bincount(g, minlength=n_groups)[:n_groups]
    start = torch.cumsum(n, 0) - n
    out = []
    for q in np.atleast_1d(qs):
        pos = (n - 1).clamp(min=0).double() * float(q)
        lo = pos.floor().long()
        hi = torch.minimum(lo + 1, (n - 1).clamp(min=0))
        if len(v):
            a, b = v[(start + lo).clamp(max=len(v) - 1)], v[(start + hi).clamp(max=len(v) - 1)]
            val = a + (b - a) * (pos - lo)
        else:
            val = torch.zeros(n_groups, dtype=torch.float64, device=device)
        out.append(torch.where(n > 0, val, torch.nan).cpu().numpy())
    return np.stack(out)


def grouped_trailing_mean(track: np.ndarray, t: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Per row, the mean of its track's values over frames t-n+1..t, ignoring missing frames and NaN.
    The row-wise form of moving_average. Returns values in the input row order."""
    # rows usually arrive in frame order; a stable sort by track then gives (track, frame) order cheaply
    order = np.argsort(track, kind="stable") if np.all(np.diff(t) >= 0) else np.lexsort((t, track))
    key = track[order].astype(np.int64) * (int(t.max()) + n + 2) + t[order]
    v = values[order].astype(np.float64)
    ok = np.isfinite(v)
    cs = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    first = np.searchsorted(key, key - (n - 1), side="left")
    last = np.arange(len(key)) + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (cs[last] - cs[first]) / (cn[last] - cn[first])
    out = np.empty_like(mean)
    out[order] = mean
    return out


def moving_average(a: np.ndarray, n: int) -> np.ndarray:
    """Trailing nan-aware mean over n frames, per row."""
    pad = np.full((a.shape[0], n), np.nan)
    stacked = np.hstack([pad, a])
    total = np.nancumsum(stacked, axis=1, dtype=float)
    count = np.nancumsum(~np.isnan(stacked), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total[:, n:] - total[:, :-n]) / (count[:, n:] - count[:, :-n])


def smooth_log2(a: np.ndarray, window: int = 15, order: int = 1) -> np.ndarray:
    """Savitzky-Golay smoothing of log2(a + 1), bridging short gaps. Rows stay NaN outside their span."""
    out = np.full_like(a, np.nan, dtype=float)
    frames = np.arange(a.shape[1])
    for i, row in enumerate(a):
        ok = np.isfinite(row) & (row > -1)
        if ok.sum() < window:
            continue
        span = (frames >= frames[ok][0]) & (frames <= frames[ok][-1])
        filled = np.interp(frames[span], frames[ok], np.log2(row[ok] + 1))
        out[i, span] = savgol_filter(filled, window, order)
    return out


def ktr_ratio(results: dict, channel: int, background_percentile: float = 5.0) -> np.ndarray:
    """Cytoplasm/nucleus ratio of a KTR, per cell per frame: the ERK activity readout.

    Keeps the notebook's per-frame background: a low percentile of nuclear values across cells,
    subtracted from both compartments. This matters: quad correction pins background to its 2nd
    percentile, which leaves a positive level of tens of counts in corrected images.
    """
    nuc = metric_matrix(results, "nuc_median", channel)
    cyto = metric_matrix(results, "cyto_median", channel)
    floor = np.nanpercentile(nuc, background_percentile, axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (cyto - floor) / (nuc - floor)
    ratio[~np.isfinite(ratio)] = np.nan
    return ratio


def percent_high(matrix: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """(percent of present cells above threshold, cells present), per frame."""
    present = np.sum(~np.isnan(matrix), axis=0)
    high = np.nansum(matrix > threshold, axis=0)
    return 100 * high / np.maximum(present, 1), present


def commitment_times(log2_trace: np.ndarray, threshold: float, hold_frames: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Last upward crossing of the threshold, for cells still above it at the end.

    Returns (time per cell, committed mask). Time is -1 if the cell did not commit. Fixes the
    notebook's version, where a cell above threshold from the first frame got commit time 0,
    which also meant "no crossing".
    """
    above = log2_trace > threshold
    n, T = above.shape
    last = np.array([np.max(np.nonzero(~np.isnan(row))[0], initial=-1) for row in log2_trace])
    committed = np.zeros(n, bool)
    times = np.full(n, -1, dtype=int)
    for i in range(n):
        end = last[i]
        if end < hold_frames - 1:
            continue
        if above[i, end - hold_frames + 1:end + 1].all():
            committed[i] = True
            below = np.nonzero(~above[i, :end + 1] & ~np.isnan(log2_trace[i, :end + 1]))[0]
            times[i] = below[-1] + 1 if below.size else 0
    return times, committed


def lineage_matrix(matrix: np.ndarray, mother_track: np.ndarray) -> np.ndarray:
    """Give each daughter its ancestors' history, so every row is a birth-to-end trajectory.

    The notebook's recursive copychildren. Mothers are always processed first, so one pass over
    tracks in order of first appearance suffices, and there is no depth limit.
    """
    out = matrix.copy()
    first = np.array([np.argmax(~np.isnan(row)) if np.any(~np.isnan(row)) else matrix.shape[1] for row in matrix])
    for track in np.argsort(first):
        mother = mother_track[track]
        if mother < 0:
            continue
        before = np.arange(matrix.shape[1]) < first[track]
        out[track, before] = out[mother, before]
    return out


def daughters_of(mother_track: np.ndarray) -> dict[int, list[int]]:
    tree: dict[int, list[int]] = {}
    for track, mother in enumerate(mother_track.tolist()):
        if mother >= 0:
            tree.setdefault(mother, []).append(track)
    return tree


def filter_tracks(results: dict, min_frames: int = 10, drop_border: bool = True) -> np.ndarray:
    """Tracks worth analysing: long enough, and never clipped by the field edge. Cached on lazy results."""
    cache = getattr(results, "_matrices", None)
    key = ("_keep", int(min_frames), bool(drop_border))
    if cache is not None and key in cache:
        return cache[key]
    index = results["trace_index"]
    keep = (index >= 0).sum(1) >= min_frames
    if drop_border:
        border = metric_matrix(results, "touches_border")
        keep &= ~np.any(border > 0, axis=1)
    if cache is not None:
        cache[key] = keep
    return keep
