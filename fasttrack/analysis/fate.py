"""Fate analyses: when do cells commit, from what level, and what happens around it.

A fate is the high state of a marker feature (nuclear PPARG for adipogenesis, any reporter of a
switch). The marker is smoothed (running median) and compared with a threshold, set by hand or
at the valley of the marker's distribution between its two populations.

Committor. For every cell and frame: the marker level now, and whether the marker is high
`horizon_hours` later. P(high later | level now) is the committor curve. A bistable switch gives
a steep curve; the level where it crosses 0.5 (from a logistic fit) is the commitment point, and
an intervention that makes commitment easier shifts it left. Cells can be split by a second
feature averaged over the preceding `split_hours` (e.g. ERK activity): if low ERK favours
commitment, the low-ERK curve sits left of the high-ERK curve at the same marker level.
Only cell-frames whose track is still observed at the horizon count, so cells that leave the
field, die or reach the end of the movie are not counted as "not committed"; the share of such
censored cell-frames is reported.

Aligned to events. Commitment: the first frame where the smoothed marker crosses the threshold
upwards and stays above for `hold_frames`, after at least as long below. Every feature is
averaged around these events, and around the same frames in matched cells that did not commit:
for each committing cell, the non-committing cell of the same position whose marker was most
similar over the window before. Features that differ before the crossing are candidates for
what precedes commitment, not only what follows it. Matching can also use other features
(`match_on`, e.g. nuclear area and integrated H2B, so controls are at a similar cell-cycle phase),
and crossings where a feature is extreme (`exclude_feature`, e.g. H2B texture at mitosis, where a
condensed nucleus can read brighter in every channel) can be left out. Division: features around
observed divisions, the mother before and the mean of her daughters after.

Intervals come from resampling positions (fields of view), the independent unit.

    <results>/Fates/<name>/fate.json   settings, threshold, commitment points, counts
    <results>/Fates/<name>/fate.npz    curves and aligned averages with their intervals
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import math
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from ..experiment import FATE_DEFAULT, PALETTE, merge, normalise, time_unit
from . import features as FT
from .figures import ResultsCache, _window, figure_conditions, safe_name

FOLDER = "Fates"
CLIP = 10.0   # normalised feature values are clipped at ±10 robust spreads before averaging
ALIGN = {"commitment": "Commitment (marker crosses the threshold and stays)", "division": "Division (mother, then her daughters)"}
VIEWS = {"threshold": "Marker distribution and threshold", "committor": "Committor: P(high later | level now)",
         "commit_point": "Commitment point per condition", "aligned": "Features around the event (vs matched cells)",
         "aligned_heatmap": "All features around the event: difference from matched cells"}


# ---------------------------------------------------------------------- small pieces
def running_median(m: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware centred running median along the last axis of (n, W), computed only where m is observed."""
    if k <= 1:
        return m
    h = k // 2
    pad = np.pad(m, ((0, 0), (h, k - 1 - h)), constant_values=np.nan)
    rows, cols = np.nonzero(np.isfinite(m))
    win = np.lib.stride_tricks.sliding_window_view(pad, k, axis=-1)[rows, cols]   # (P, k)
    win = np.sort(win, axis=-1)                                                   # NaN sort last
    c = np.isfinite(win).sum(1)
    lo = np.take_along_axis(win, ((c - 1) // 2)[:, None], 1)[:, 0]
    hi = np.take_along_axis(win, (c // 2)[:, None], 1)[:, 0]
    out = np.full(m.shape, np.nan, np.float32)
    out[rows, cols] = (lo + hi) / 2
    return out


def lineage_future(m: np.ndarray, first: np.ndarray, last: np.ndarray, mother: np.ndarray, divides: np.ndarray) -> np.ndarray:
    """m with each dividing track continued after its division by the mean of its daughters (recursively),
    so a cell's future includes its progeny instead of ending at mitosis."""
    out = m.copy()
    kids: dict[int, list[int]] = {}
    for d, mo in enumerate(mother):
        if mo >= 0:
            kids.setdefault(int(mo), []).append(d)
    for i in sorted(kids, key=lambda i: -first[i]):       # daughters (later starts) are completed first
        if not divides[i]:
            continue
        a = int(last[i]) + 1
        if a >= m.shape[1]:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = np.nanmean(out[kids[i], a:], 0)
        seg = out[i, a:]
        out[i, a:] = np.where(np.isfinite(seg), seg, fill)
    return out


def trailing_mean(v: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware mean of the k frames up to and including each frame."""
    ok = np.isfinite(v)
    c = np.concatenate([np.zeros((len(v), 1)), np.cumsum(np.where(ok, v, 0), 1)], 1)
    n = np.concatenate([np.zeros((len(v), 1)), np.cumsum(ok, 1)], 1)
    idx = np.arange(v.shape[1]) + 1
    lo = np.maximum(idx - k, 0)
    s, cnt = c[:, idx] - c[:, lo], n[:, idx] - n[:, lo]
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def two_populations(values: np.ndarray, iters: int = 200) -> dict:
    """1-D two-Gaussian mixture by EM. Returns weights, means, sds, the separation (difference of means /
    pooled sd), and the threshold: the valley (lowest smoothed density) between the two means. The valley
    does not depend on the populations' shapes, which a density crossing would (it slides towards the
    narrower population)."""
    v = values[np.isfinite(values)]
    if len(v) < 20:
        raise ValueError("Too few marker values to set a threshold.")
    if len(v) > 100_000:
        v = np.random.default_rng(0).choice(v, 100_000, replace=False)
    w, mu = np.array([0.5, 0.5]), np.percentile(v, [25, 85]).astype(float)
    sd = np.array([v.std() / 2] * 2) + 1e-6
    for _ in range(iters):
        before = mu.copy()
        logp = -0.5 * ((v[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w)
        logp -= logp.max(1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        w, mu = nk / len(v), (r * v[:, None]).sum(0) / nk
        sd = np.sqrt((r * (v[:, None] - mu) ** 2).sum(0) / nk) + 1e-6
        if np.abs(mu - before).max() < 1e-5 * (v.std() + 1e-12):
            break
    order = np.argsort(mu)
    w, mu, sd = w[order], mu[order], sd[order]
    edges = np.linspace(mu[0], mu[1], 201)
    counts = np.histogram(v, edges)[0].astype(float)
    kernel = np.exp(-0.5 * (np.arange(-30, 31) / 10.0) ** 2)            # sd = 1/20 of the distance between the means
    smooth = np.convolve(np.pad(counts, 30, mode="edge"), kernel / kernel.sum(), "valid")
    centres = (edges[1:] + edges[:-1]) / 2
    inner = slice(10, 190)                                              # not on either peak
    thr = float(centres[inner][np.argmin(smooth[inner])])
    sep = float((mu[1] - mu[0]) / math.sqrt((sd[0] ** 2 + sd[1] ** 2) / 2))
    return {"weights": w.tolist(), "means": mu.tolist(), "sds": sd.tolist(), "threshold": thr, "separation": sep}


def find_commitments(m: np.ndarray, thr: float, hold: int) -> np.ndarray:
    """Per row: the first column where m crosses thr upwards, having been below for `hold` frames and
    staying above for `hold` frames (60% of each window observed). -1 if never."""
    n, W = m.shape
    out = np.full(n, -1)
    if W < 2 * hold + 1:
        return out
    obs = np.isfinite(m)
    hi, lo = obs & (m >= thr), obs & (m < thr)
    z = np.zeros((n, 1))
    co, ch, cl = (np.concatenate([z, np.cumsum(a, 1)], 1) for a in (obs, hi, lo))
    t = np.arange(hold, W - hold + 1)
    need = math.ceil(0.6 * hold)
    a_obs, a_hi = co[:, t + hold] - co[:, t], ch[:, t + hold] - ch[:, t]
    b_obs, b_lo = co[:, t] - co[:, t - hold], cl[:, t] - cl[:, t - hold]
    cand = hi[:, t] & (a_hi == a_obs) & (a_obs >= need) & (b_lo == b_obs) & (b_obs >= need)
    any_ = cand.any(1)
    out[any_] = t[np.argmax(cand[any_], 1)]
    return out


def _windows(y: np.ndarray, rows: np.ndarray, centre: np.ndarray, before: int, after: int) -> np.ndarray:
    """(E, L, F) slices of y (n, W, F) around each event, NaN outside the movie."""
    L, W = before + after + 1, y.shape[1]
    cols = centre[:, None] + np.arange(-before, after + 1)
    ok = (cols >= 0) & (cols < W)
    out = y[rows[:, None], np.clip(cols, 0, W - 1)]
    out[~ok] = np.nan
    return out.reshape(len(rows), L, y.shape[2])


def match_controls(m: np.ndarray, rows: np.ndarray, cols: np.ndarray, never: np.ndarray, thr: float, before: int, after: int,
                   scale: float = 1.0, extra: np.ndarray | None = None, chunk: int = 32) -> np.ndarray:
    """For each event (row, col): the row among `never` closest over the `before` frames up to col, among
    cells below thr throughout [col - before, col + after] and observed on at least half of the frames both
    before and after (so not cells about to vanish). Closeness: mean squared marker difference in units of
    `scale`, plus, for each feature in `extra` (n, W, F, normalised), the squared difference of its means
    over the same frames (a feature missing for either cell adds nothing). Returns rows of m, -1 if none."""
    n, W = m.shape
    if extra is not None and extra.shape[2]:
        ok_x = np.isfinite(extra)
        zx = np.zeros((n, 1, extra.shape[2]))
        cx_sum = np.concatenate([zx, np.cumsum(np.where(ok_x, extra, 0), 1)], 1)   # (n, W+1, F)
        cx_cnt = np.concatenate([zx, np.cumsum(ok_x, 1)], 1)
        cxN_sum, cxN_cnt = cx_sum[never], cx_cnt[never]
    else:
        extra = None
    mN = m[never]
    obs = np.isfinite(mN)
    z = np.zeros((len(never), 1))
    c_obs = np.concatenate([z, np.cumsum(obs, 1)], 1)
    c_hi = np.concatenate([z, np.cumsum(obs & (mN >= thr), 1)], 1)
    out = np.full(len(rows), -1)
    steps = np.arange(-before, 0)
    for s0 in range(0, len(rows), chunk):
        r, c = rows[s0:s0 + chunk], cols[s0:s0 + chunk]
        a, mid, b = np.maximum(c - before, 0), c, np.minimum(c + after + 1, W)
        ok = (c_hi[:, b] - c_hi[:, a]) == 0                                     # (Nv, E)
        ok &= (c_obs[:, mid] - c_obs[:, a]) >= 0.5 * (mid - a)
        ok &= (c_obs[:, b] - c_obs[:, mid]) >= 0.5 * (b - mid)
        idx = c[:, None] + steps                                                 # (E, before)
        valid = idx >= 0
        idx = np.clip(idx, 0, W - 1)
        mine = np.where(valid, m[r[:, None], idx], np.nan)                       # (E, before)
        theirs = mN[:, idx]                                                      # (Nv, E, before)
        sq = (theirs - mine) ** 2
        both = np.isfinite(sq)
        cnt = both.sum(2)
        d = np.where(both, sq, 0).sum(2) / np.maximum(cnt, 1) / scale ** 2
        if extra is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                mi = (cx_sum[r, mid] - cx_sum[r, a]) / (cx_cnt[r, mid] - cx_cnt[r, a])                  # (E, F)
                mj = (cxN_sum[:, mid] - cxN_sum[:, a]) / (cxN_cnt[:, mid] - cxN_cnt[:, a])              # (Nv, E, F)
            d = d + np.nansum((mj - mi[None]) ** 2, 2)
        need = 0.5 * np.maximum(mid - a, 2)
        d[~ok | (cnt < need) | ((mid - a) < 2)] = np.inf
        j = np.argmin(d, 0)
        has = np.isfinite(d[j, np.arange(len(r))])
        out[s0:s0 + chunk] = np.where(has, never[j], -1)
    return out


def logistic_fit(x: np.ndarray, n: np.ndarray, k: np.ndarray, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Binomial logistic regression P = 1 / (1 + exp(-b (x - x50))) on binned counts, batched over
    the leading axes of n and k (..., bins). Returns x50 and the slope b (NaN when it does not rise)."""
    shape = n.shape[:-1]
    n, k = n.reshape(-1, n.shape[-1]).astype(float), k.reshape(-1, k.shape[-1]).astype(float)
    use = n.sum(0) > 0
    xm = float(np.average(x[use], weights=n.sum(0)[use])) if use.any() else 0.0
    xs = float(np.sqrt(np.average((x[use] - xm) ** 2, weights=n.sum(0)[use]))) if use.any() else 1.0
    xs = xs or 1.0
    z = (x - xm) / xs
    X = np.stack([np.ones_like(z), z], 1)                          # (bins, 2)
    beta = np.zeros((len(n), 2))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(beta @ X.T, -30, 30)))       # (S, bins)
        g = ((k - n * p)[:, :, None] * X).sum(1)                   # (S, 2)
        wgt = n * p * (1 - p) + 1e-9
        H = np.einsum("sb,bi,bj->sij", wgt, X, X) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g[..., None])[..., 0]
        beta += np.clip(step, -5, 5)
    a, b = beta[:, 0], beta[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        x50 = np.where(b > 1e-6, xm - a / b * xs, np.nan)
        slope = np.where(b > 1e-6, b / xs, np.nan)
    enough = (n.sum(1) >= 20) & (k.sum(1) > 0) & (k.sum(1) < n.sum(1))
    x50[~enough], slope[~enough] = np.nan, np.nan
    return x50.reshape(shape), slope.reshape(shape)


def _ci(boot: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(boot, [2.5, 97.5], axis=axis)


# ---------------------------------------------------------------------- the analysis
def fate_folder(exp: dict, name: str) -> Path:
    return Path(exp["paths"]["results"]) / FOLDER / safe_name(name or "fate")


def _frames(hours: float, dt: float) -> int:
    return max(1, int(round(hours * 60 / dt)))


def run_fate(exp: dict, spec: dict, on_event=lambda e: None, stop=None, save: bool = True) -> dict:
    """Run one fate analysis from the experiment file. Returns the report (also saved)."""
    exp = normalise(exp)
    spec = merge(FATE_DEFAULT, spec)
    t0 = time.time()
    dt, unit = time_unit(exp)
    if unit != "min":
        raise ValueError("Fate analyses need the time between frames: set frames per hour (Experiment tab) "
                         "or minutes between frames (Analysis → Dynamics).")
    if not spec["marker"]:
        raise ValueError("Choose the marker feature whose high state is the fate.")
    conditions = figure_conditions(exp, {"conditions": spec["conditions"], "group": spec["group"]})
    if not conditions:
        raise ValueError("No conditions to analyse.")
    align_names = spec["align_features"] or [f["name"] for f in FT.feature_list(exp)]
    extra_names = [*spec["match_on"], *([spec["exclude_feature"]] if spec["exclude_feature"] else [])]
    names = list(dict.fromkeys([spec["marker"], *([spec["split"]] if spec["split"] else []), *align_names, *extra_names]))
    feats = FT.by_name(exp, names)
    A = len(align_names)
    a_idx = [names.index(n) for n in align_names]
    s_idx = names.index(spec["split"]) if spec["split"] else None
    m_idx = [names.index(n) for n in spec["match_on"]]
    x_idx = names.index(spec["exclude_feature"]) if spec["exclude_feature"] else None
    smooth, hold = int(spec["smooth_frames"]), int(spec["hold_frames"])
    H = _frames(spec["horizon_hours"], dt)
    B, Af = _frames(spec["before_hours"], dt), _frames(spec["after_hours"], dt)
    L = B + Af + 1
    S_frames = _frames(spec["split_hours"], dt)
    rng = np.random.default_rng(spec["seed"])
    cache = ResultsCache(exp)
    stats = FT.reference_stats(exp, feats, conditions, cache, seed=spec["seed"])
    ref = stats["reference"]
    on_event({"type": "log", "message": f"Features normalised to {ref}. Horizon {H} frames, window -{B}..+{Af} frames."})

    def block_of(key, slim=False):
        """One position's tracks. slim: only the marker and split feature (first pass)."""
        res = cache.get(key)
        T = int(res["meta"].get("frames", res.frames))
        win = _window(exp, {"frames": list(spec["frames"])}, T)
        use = names[:2 if s_idx is not None else 1] if slim else names
        b = FT.position_block(res, exp, feats[:len(use)], win)
        cache._loaded.pop(key, None)   # one position at a time: memory stays flat
        m = running_median(b["y"][..., 0], smooth)
        z = (b["y"] - np.array([stats["median"][n] for n in use], np.float32)) / np.array([stats["scale"][n] for n in use], np.float32)
        z = np.clip(z, -CLIP, CLIP)   # one bad mask must not dominate an average
        split = trailing_mean(z[..., s_idx], S_frames) if s_idx is not None else None
        return b, m, z, split

    keys_of = {c["name"]: cache.timestacks_of(c) for c in conditions}
    # ---- pass 1: marker distribution (threshold, bins) and split quantiles from the reference
    sample_m: dict[str, list] = {c["name"]: [] for c in conditions}
    split_ref = []
    for c in conditions:
        keys = keys_of[c["name"]]
        pick = keys if c["name"] == ref else list(rng.choice(keys, min(3, len(keys)), replace=False)) if keys else []
        for key in pick:
            if stop is not None and stop.is_set():
                raise InterruptedError("stopped")
            b, m, z, split = block_of(key, slim=True)
            v = m[np.isfinite(m)]
            sample_m[c["name"]].append(rng.choice(v, min(len(v), 50_000), replace=False) if len(v) else v)
            if split is not None and c["name"] == ref:
                s = split[np.isfinite(split)]
                split_ref.append(rng.choice(s, min(len(s), 50_000), replace=False) if len(s) else s)
    pooled = np.concatenate([np.concatenate(v) for v in sample_m.values() if v]) if any(sample_m.values()) else np.zeros(0)
    if not len(pooled):
        raise ValueError(f"No values of the marker {spec['marker']} in these conditions.")
    mixture = two_populations(pooled)
    thr = mixture["threshold"] if spec["threshold_auto"] else float(spec["threshold"])
    if spec["threshold_auto"] and mixture["separation"] < 2:
        on_event({"type": "log", "message": f"Warning: the marker barely looks like two populations (separation {mixture['separation']:.1f} sd)."})
    lo, hi = np.percentile(pooled, [0.5, 99.5])
    edges = np.linspace(lo, hi, int(spec["bins"]) + 1)
    centres = (edges[1:] + edges[:-1]) / 2
    G = int(spec["split_groups"]) if s_idx is not None else 0
    q_edges = np.quantile(np.concatenate(split_ref), np.arange(1, G) / G) if G and split_ref else np.zeros(0)
    hist_edges = np.linspace(lo, hi, 81)
    hist = np.stack([np.histogram(np.concatenate(sample_m[c["name"]]) if sample_m[c["name"]] else np.zeros(0), hist_edges,
                                  density=True)[0] for c in conditions])

    # ---- pass 2: every position
    nb = len(centres)
    per_cond = []
    total_pos = sum(len(v) for v in keys_of.values())
    done = 0
    for c in conditions:
        n_list, k_list, ev_sum, ev_cnt, ct_sum, ct_cnt, n_events, n_tracks, cens = [], [], [], [], [], [], [], [], [0, 0]
        n_excluded = 0
        for key in keys_of[c["name"]]:
            if stop is not None and stop.is_set():
                raise InterruptedError("stopped")
            b, m, z, split = block_of(key)
            n, W = m.shape
            done += 1
            on_event({"type": "progress", "stage": "positions", "done": done, "total": total_pos})
            # committor counts (group 0 = every cell, 1.. = split groups)
            nn = np.zeros((G + 1, nb))
            kk = np.zeros((G + 1, nb))
            if W > H:
                ext = lineage_future(m, b["first"], b["last"], b["mother"], b["divides"])
                x, fut = m[:, :W - H], ext[:, H:]
                ok = np.isfinite(x)
                cens[0] += int(ok.sum())
                cens[1] += int((ok & ~np.isfinite(fut)).sum())
                ok &= np.isfinite(fut)
                bins = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
                high = fut >= thr
                nn[0] = np.bincount(bins[ok], minlength=nb)
                kk[0] = np.bincount(bins[ok & high], minlength=nb)
                if G:
                    s = split[:, :W - H]
                    g = np.digitize(s, q_edges)
                    okg = ok & np.isfinite(s)
                    flat = g * nb + bins
                    nn[1:] = np.bincount(flat[okg], minlength=G * nb).reshape(G, nb)
                    kk[1:] = np.bincount(flat[okg & high], minlength=G * nb).reshape(G, nb)
            n_list.append(nn)
            k_list.append(kk)
            # events
            za = z[..., a_idx]
            if spec["align"] == "division":
                rows = np.nonzero(b["divides"] & (b["last"] >= 0) & (b["last"] < W))[0]
                centre = b["last"][rows].astype(int)
                win = _windows(za, rows, centre, B, Af)
                for e, i in enumerate(rows):
                    kids = np.nonzero(b["mother"] == i)[0]
                    if len(kids):
                        kw = _windows(za, kids, np.full(len(kids), centre[e]), B, Af)[:, B + 1:]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            win[e, B + 1:] = np.nanmean(kw, 0)
                    else:
                        win[e, B + 1:] = np.nan
                ctrl = np.full_like(win, np.nan)
            else:
                e_col = find_commitments(m, thr, hold)
                if x_idx is not None:   # crossings where the exclusion feature is extreme (e.g. mitosis) are left out
                    k = int(spec["exclude_frames"])
                    for i in np.nonzero(e_col >= 0)[0]:
                        w = z[i, max(0, e_col[i] - k):e_col[i] + k + 1, x_idx]
                        if np.isfinite(w).any() and np.nanmax(w) > spec["exclude_above"]:
                            e_col[i] = -2
                    n_excluded += int((e_col == -2).sum())
                rows = np.nonzero(e_col >= 0)[0]
                centre = e_col[rows]
                win = _windows(za, rows, centre, B, Af)
                ctrl = np.full_like(win, np.nan)
                never = np.nonzero(e_col == -1)[0]
                if len(rows) and len(never):
                    best = match_controls(m, rows, centre, never, thr, B, Af, stats["scale"][spec["marker"]],
                                          z[..., m_idx] if m_idx else None)
                    found = best >= 0
                    if found.any():
                        ctrl[found] = _windows(za, best[found], centre[found], B, Af)
            ev_sum.append(np.nansum(win, 0) if len(win) else np.zeros((L, A)))
            ev_cnt.append(np.isfinite(win).sum(0) if len(win) else np.zeros((L, A)))
            ct_sum.append(np.nansum(ctrl, 0) if len(ctrl) else np.zeros((L, A)))
            ct_cnt.append(np.isfinite(ctrl).sum(0) if len(ctrl) else np.zeros((L, A)))
            n_events.append(len(rows))
            n_tracks.append(n)
        per_cond.append({"n": np.array(n_list).reshape(-1, G + 1, nb), "k": np.array(k_list).reshape(-1, G + 1, nb),
                         "ev_sum": np.array(ev_sum).reshape(-1, L, A), "ev_cnt": np.array(ev_cnt).reshape(-1, L, A),
                         "ct_sum": np.array(ct_sum).reshape(-1, L, A), "ct_cnt": np.array(ct_cnt).reshape(-1, L, A),
                         "events": np.array(n_events, int), "tracks": np.array(n_tracks, int), "censored": cens,
                         "excluded": n_excluded})

    # ---- summaries with position-resampled intervals
    C, R = len(conditions), int(spec["bootstrap"])
    P_hat = np.full((C, G + 1, nb), np.nan)
    P_ci = np.full((2, C, G + 1, nb), np.nan)
    x50 = np.full((C, G + 1), np.nan)
    slope = np.full((C, G + 1), np.nan)
    x50_ci = np.full((2, C, G + 1), np.nan)
    ev_mean = np.full((C, L, A), np.nan)
    ct_mean = np.full((C, L, A), np.nan)
    ev_ci = np.full((2, C, L, A), np.nan)
    diff_ci = np.full((2, C, L, A), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for ci, d in enumerate(per_cond):
            Pn = len(d["n"])
            if not Pn:
                continue
            n_all, k_all = d["n"].sum(0), d["k"].sum(0)
            P_hat[ci] = np.where(n_all > 0, k_all / np.maximum(n_all, 1), np.nan)
            x50[ci], slope[ci] = logistic_fit(centres, n_all, k_all)
            ev_mean[ci] = d["ev_sum"].sum(0) / np.where(d["ev_cnt"].sum(0) > 0, d["ev_cnt"].sum(0), np.nan)
            ct_mean[ci] = d["ct_sum"].sum(0) / np.where(d["ct_cnt"].sum(0) > 0, d["ct_cnt"].sum(0), np.nan)
            if R and Pn > 1:
                pick = rng.integers(0, Pn, (R, Pn))
                nb_, kb_ = d["n"][pick].sum(1), d["k"][pick].sum(1)          # (R, G+1, bins)
                P_ci[:, ci] = _ci(np.where(nb_ > 0, kb_ / np.maximum(nb_, 1), np.nan))
                xb, _ = logistic_fit(centres, nb_, kb_)
                x50_ci[:, ci] = _ci(xb)
                es, ec = d["ev_sum"][pick].sum(1), d["ev_cnt"][pick].sum(1)
                cs, cc = d["ct_sum"][pick].sum(1), d["ct_cnt"][pick].sum(1)
                eb = es / np.where(ec > 0, ec, np.nan)
                cb = cs / np.where(cc > 0, cc, np.nan)
                ev_ci[:, ci] = _ci(eb)
                diff_ci[:, ci] = _ci(eb - cb)

    report = {
        "name": spec["name"], "spec": spec, "conditions": [c["name"] for c in conditions], "reference": ref,
        "marker": spec["marker"], "split": spec["split"], "split_groups": G, "split_edges": q_edges.tolist(),
        "threshold": thr, "mixture": mixture, "interval": dt, "horizon_frames": H, "before_frames": B, "after_frames": Af,
        "align": spec["align"], "align_features": align_names,
        "positions": [int(len(d["n"])) for d in per_cond], "tracks": [int(d["tracks"].sum()) for d in per_cond],
        "events": [int(d["events"].sum()) for d in per_cond], "excluded": [int(d["excluded"]) for d in per_cond],
        "censored_share": [d["censored"][1] / max(1, d["censored"][0]) for d in per_cond],
        "x50": x50.tolist(), "x50_ci": x50_ci.tolist(), "slope": slope.tolist(), "normalisation": stats,
        "fitted": _dt.datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1),
        "_arrays": {"centres": centres, "P": P_hat, "P_ci": P_ci, "n": np.stack([d["n"].sum(0) for d in per_cond]),
                    "hist_edges": hist_edges, "hist": hist, "ev_mean": ev_mean, "ct_mean": ct_mean, "ev_ci": ev_ci,
                    "diff_ci": diff_ci, "rel_hours": (np.arange(L) - B) * dt / 60},
    }
    if save:
        folder = fate_folder(exp, spec["name"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "fate.json").write_text(json.dumps({k: v for k, v in report.items() if not k.startswith("_")}, indent=1, default=float))
        np.savez_compressed(folder / "fate.npz", **report["_arrays"])
        render_fate(exp, report, folder)
        on_event({"type": "log", "message": f"Saved to {folder}"})
    return report


def read_fate(exp: dict, name: str) -> dict | None:
    folder = fate_folder(exp, name)
    if not (folder / "fate.json").exists():
        return None
    report = json.loads((folder / "fate.json").read_text())
    with np.load(folder / "fate.npz") as z:
        report["_arrays"] = {k: z[k] for k in z.files}
    return report


def run_all(exp: dict, names: list[str] | None = None, on_event=lambda e: None, stop=None) -> list[dict]:
    exp = normalise(exp)
    out = []
    for spec in exp["analysis"]["fates"]:
        if (names and spec["name"] not in names) or (not names and not spec["enabled"]):
            continue
        out.append(run_fate(exp, spec, on_event, stop))
    return out


# ---------------------------------------------------------------------- figures
def _colors(exp: dict | None, names: list[str]) -> list[str]:
    """Condition colours under the figures palette (plotting.style), the reference first."""
    from ..plotting.style import condition_colors
    return condition_colors(exp, names)


def _group_names(report: dict) -> list[str]:
    G = report["split_groups"]
    if not G:
        return ["every cell"]
    labels = {2: ["low", "high"], 3: ["low", "middle", "high"]}.get(G, [f"q{g + 1}" for g in range(G)])
    return ["every cell"] + [f"{report['split']} {lab}" for lab in labels]


def top_features(report: dict, k: int = 8) -> list[int]:
    """Aligned features ordered by the largest difference from matched cells before the event."""
    a = report["_arrays"]
    B = report["before_frames"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        diff = np.nanmax(np.abs(a["ev_mean"][:, :B + 1] - a["ct_mean"][:, :B + 1]), axis=(0, 1))
        if report["align"] == "division" or not np.isfinite(diff).any():
            diff = np.nanmax(np.abs(a["ev_mean"] - np.nanmean(a["ev_mean"], 1, keepdims=True)), axis=(0, 1))
    order = [int(i) for i in np.argsort(-np.nan_to_num(diff, nan=-1))]
    names = report["align_features"]
    first = [names.index(n) for n in (report["marker"], report["split"]) if n and n in names]
    return list(dict.fromkeys(first + order))[:k]


def draw(fig: Figure, report: dict, view: str, exp: dict | None = None, show: list[str] | None = None) -> Figure:
    """Draw one view. `show`: the conditions to draw (names; empty = all). Colours stay those of the full set.
    The layout follows the figure's size, so the same view fits a window, a single-column or a double-column panel."""
    from ..plotting import panels as P
    with P.screen(fig):
        P.begin(fig)
        _draw(fig, report, view, exp, show, P)
    return fig


def _draw(fig, report, view, exp, show, P) -> None:
    from ..plotting.style import CONTEXT
    a = report["_arrays"]
    conds = report["conditions"]
    colors = _colors(exp, conds)
    idx = P.pick(conds, show)
    names = [conds[c] for c in idx]
    thr = report["threshold"]
    marker = report["marker"]
    horizon = report["spec"]["horizon_hours"]
    legend_w = 8 + 1.6 * max(len(n) for n in names)      # mm the condition legend takes on the right
    keyed = P.line_handles(names, [colors[c] for c in idx])
    if view == "threshold":
        ax = fig.add_subplot()
        e = a["hist_edges"]
        x = (e[1:] + e[:-1]) / 2
        for c in idx:
            ax.plot(x, a["hist"][c], color=colors[c])
        mix = report["mixture"]
        for w, mu, sd in zip(mix["weights"], mix["means"], mix["sds"]):
            ax.plot(x, w * np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)), ls=":", color="0.3")
        ax.axvline(thr, color="k", lw=0.5, ls="--")
        ax.text(thr, 1, f" {thr:.3g}", transform=ax.get_xaxis_transform(), va="top", fontsize="small")
        ax.set_xlabel(f"{marker} (smoothed)")
        ax.set_ylabel("Density")
        P.legend(fig, keyed + P.style_handles([("fitted populations", dict(ls=":", color="0.3"))]))
        how = "the valley between the two populations" if report["spec"]["threshold_auto"] else "set by hand"
        P.finish(fig, VIEWS[view], [f"Dashed: threshold {thr:.3g} ({how}); the populations are {mix['separation']:.1f} SD apart",
                                    "dotted: the two fitted Gaussian populations, all conditions pooled"])
    elif view == "committor":
        groups = _group_names(report)
        axes = P.grid(fig, len(groups), min_w=26, min_h=28, reserve_w=legend_w, what="groups", legend_items=len(idx))
        x = a["centres"]
        for g, ax in enumerate(axes):
            for c in idx:
                ok = a["n"][c, g] >= 20
                ax.plot(x[ok], a["P"][c, g][ok], color=colors[c])
                if np.isfinite(a["P_ci"][0, c, g]).any():
                    ax.fill_between(x[ok], a["P_ci"][0, c, g][ok], a["P_ci"][1, c, g][ok], color=colors[c], alpha=0.2, lw=0)
                xv = report["x50"][c][g]
                if xv is not None and np.isfinite(xv):
                    ax.plot([xv], [0.5], "o", color=colors[c], ms=3, mec="white", mew=0.3, zorder=4)
            ax.axvline(thr, color="k", lw=0.5, ls="--")
            ax.axhline(0.5, color=CONTEXT, lw=0.5, zorder=0)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            if len(groups) > 1:
                ax.set_title(groups[g], loc="left", fontsize="medium")
        fig.supxlabel(f"{marker} now")
        fig.supylabel(f"P(high {horizon:g} h later)")
        P.legend(fig, keyed)
        P.finish(fig, VIEWS[view], ["Dots: commitment point (P = 0.5, logistic fit)",
                                    "bands: 95% interval over position resamples", f"dashed: threshold {thr:.3g}",
                                    "levels with fewer than 20 cells are left out"])
    elif view == "commit_point":
        groups = _group_names(report)
        ax = fig.add_subplot()
        x50, ci = np.array(report["x50"], float), np.array(report["x50_ci"], float)
        width = 0.7 / len(groups)
        markers = ["o", "v", "^", "s", "D", "P"]
        for g in range(len(groups)):
            pos = np.arange(len(idx)) + (g - (len(groups) - 1) / 2) * width
            for p, c in enumerate(idx):
                if np.isfinite(ci[:, c, g]).all():
                    ax.vlines(pos[p], ci[0, c, g], ci[1, c, g], color=colors[c], lw=0.75)
                ax.plot(pos[p], x50[c, g], markers[g % len(markers)], color=colors[c], ms=3.5,
                        mfc=colors[c] if g == 0 else "white", mew=0.75)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanmin(ci[:, idx]), np.nanmax(ci[:, idx])
        if np.isfinite(lo) and lo - 0.15 * (hi - lo) <= thr <= hi + 0.15 * (hi - lo):
            ax.axhline(thr, color="k", lw=0.5, ls="--")
            P.note(fig, f"dashed: threshold {thr:.3g}")
        elif np.isfinite(lo):
            P.note(fig, f"the threshold ({thr:.3g}) lies {'above' if thr > hi else 'below'} every commitment point")
        ax.set_xlim(-0.6, len(idx) - 0.4)
        P.condition_ticks(ax, names)
        ax.set_ylabel(f"{marker} at P = 0.5")
        if len(groups) > 1:
            P.legend(fig, P.style_handles([(g, dict(ls="none", marker=markers[i % len(markers)], ms=3.5,
                                                    mfc="k" if i == 0 else "white", mew=0.75)) for i, g in enumerate(groups)]))
        P.finish(fig, VIEWS[view], [f"The marker level from which half of the cells are high {horizon:g} h later; "
                                    "lower = commitment from a lower level", "bars: 95% interval over position resamples"])
    elif view == "aligned":
        order = top_features(report, k=len(report["align_features"]))
        axes = P.grid(fig, min(len(order), 12), sharey=False, min_w=30, min_h=24, reserve_w=legend_w, what="features",
                      legend_items=len(idx) + 2)
        t = a["rel_hours"]
        commit = report["align"] == "commitment"
        for ax, f in zip(axes, order):
            for c in idx:
                ax.plot(t, a["ev_mean"][c, :, f], color=colors[c])
                if np.isfinite(a["ev_ci"][0, c, :, f]).any():
                    ax.fill_between(t, a["ev_ci"][0, c, :, f], a["ev_ci"][1, c, :, f], color=colors[c], alpha=0.2, lw=0)
                if commit:
                    ax.plot(t, a["ct_mean"][c, :, f], "--", color=colors[c], lw=0.6 * matplotlib.rcParams["lines.linewidth"])
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(report["align_features"][f], loc="left", fontsize="small")
        fig.supxlabel(f"Hours from {report['align']}")
        fig.supylabel("Robust z")
        extra = (P.style_handles([("committing", dict(ls="-")), ("matched, not", dict(ls="--", lw=0.6))]) if commit else [])
        P.legend(fig, keyed + extra)
        what = ("dashed: matched cells that did not commit, at the same time" if commit
                else "the mother before division, the mean of her daughters after")
        P.finish(fig, VIEWS[view], [f"Features ordered by how much they differ before {report['align']}; "
                                    "robust z: reference condition = 0, robust spread = 1", what,
                                    "bands: 95% interval over position resamples",
                                    "events: " + ", ".join(f"{conds[c]} {report['events'][c]}" for c in idx)])
    elif view == "aligned_heatmap":
        t = a["rel_hours"]
        if report["align"] == "commitment":
            M = a["ev_mean"] - a["ct_mean"]
            label = "committing − matched cells"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                M = a["ev_mean"] - np.nanmean(a["ev_mean"], 1, keepdims=True)
            label = "change from each feature's mean over the window"
        order = top_features(report, k=len(report["align_features"]))
        feat = [report["align_features"][i] for i in order]
        label_w = 4 + 1.3 * max(len(n) for n in feat)          # the feature names on the left
        w_mm, h_mm = P.size_mm(fig)
        C = len(idx)
        cols = max(1, min(C, int((w_mm - label_w - 14) // 28)))
        row_mm = P.font_pt("small") * 1.25 / P.PT               # one feature row at the tick label size
        most = max(1, int((h_mm - 10) // (7 + 4 * row_mm)))     # rows of conditions that keep at least 4 features each
        rows = min(math.ceil(C / cols), most)
        if rows * cols < C:
            P.note(fig, f"showing {rows * cols} of {C} conditions (make the panel taller, or choose fewer, to see the rest)")
            idx = idx[:rows * cols]
            C = len(idx)
        cols = math.ceil(C / rows)
        fit = int(((h_mm - 10) / rows - 7) // row_mm)
        k = min(len(order), max(4, fit))
        if k < len(order):
            P.note(fig, f"the {k} of {len(order)} features that differ most (make the panel taller to see more)")
        order, feat = order[:k], feat[:k]
        lim = float(np.nanpercentile(np.abs(M[idx][..., order]), 98)) if np.isfinite(M[idx][..., order]).any() else 1.0
        axes = fig.subplots(rows, cols, sharex=True, sharey=True, squeeze=False).ravel()
        im = None
        for p, ax in enumerate(axes):
            if p >= C:
                ax.set_visible(False)
                continue
            c = idx[p]
            im = ax.imshow(M[c][:, order].T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                           extent=(t[0], t[-1], k - 0.5, -0.5), interpolation="nearest", rasterized=True)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_yticks(range(k), feat)
            ax.tick_params(axis="y", length=0)
            ax.set_title(conds[c], loc="left", fontsize="small")
            if p + cols < C:
                ax.tick_params(labelbottom=False)
            if p % cols:
                ax.tick_params(labelleft=False)
        for ax in axes:
            ax.spines[:].set_visible(False)
        if im is not None:
            fig.colorbar(im, ax=[ax for ax in axes if ax.get_visible()], shrink=0.8, aspect=25, pad=0.02, fraction=0.04,
                         label="Robust spreads")
        fig.supxlabel(f"Hours from {report['align']}")
        P.finish(fig, f"Every feature around {report['align']}", [f"Colour: {label}, in robust spreads; "
                                                                   f"limits at the 98th percentile (±{lim:.2g})"])
    else:
        P.empty(fig, f"Unknown view {view!r}")


def render_fate(exp: dict, report: dict, folder, formats=("png",), dpi: int = 150) -> list:
    out = []
    folder = Path(folder) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for view in VIEWS:
        fig = Figure(figsize=(7.2, 4.5))
        draw(fig, report, view, exp)
        for fmt in formats:
            path = folder / f"{view}.{fmt}"
            fig.savefig(path, dpi=dpi)
            out.append(path)
    return out
