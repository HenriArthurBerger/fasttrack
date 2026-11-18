"""Fit synthetic-data parameters to a real timestack, and check the fit.

`describe` summarises a tracked movie: density over time, nuclear shape, motion, intensities and how
fast each channel changes. `fit_parameters` turns that summary, plus a few raw frames for background
and camera noise, into generator settings. Running `describe` on the synthetic movie's own pipeline
output, and comparing, shows how close the match is. That comparison is what makes a synthetic
benchmark believable; a fit that does not reproduce the real summary should not be trusted.

Channel change is split into persistent change (real biology) and measurement noise using how the
variance of log intensity differences grows with time lag:  var(lag k) = 2 * noise^2 + k * walk^2.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..track import background
from ..track.linking import trace_index
from ..track.measure import K, S


def _tracks(scalars: np.ndarray, links: dict, frames: int, min_len: int = 10) -> np.ndarray:
    index = trace_index(links["track_id"], scalars[:, S["t"]], frames)
    return index[(index >= 0).sum(1) >= min_len]


def _lagged(matrix: np.ndarray, lag: int) -> np.ndarray:
    d = matrix[:, lag:] - matrix[:, :-lag]
    return d[np.isfinite(d)]


def _robust_var(values: np.ndarray) -> float:
    if values.size < 20:
        return float("nan")
    return float((1.4826 * np.median(np.abs(values - np.median(values)))) ** 2)


def _walk_and_noise(matrix: np.ndarray, lags=(1, 2, 3, 4, 6, 8)) -> tuple[float, float]:
    """Persistent per-frame change and measurement noise, both as sd in the matrix's units."""
    v = np.array([_robust_var(_lagged(matrix, k)) for k in lags])
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return 0.0, 0.0
    slope, intercept = np.polyfit(np.array(lags)[ok], v[ok], 1)
    return math.sqrt(max(slope, 0.0)), math.sqrt(max(intercept, 0.0) / 2)


def _switch_rate(matrix: np.ndarray, threshold: float = 1.0, window: int = 3) -> tuple[float, float]:
    """Abrupt, persistent changes: the median of the next frames differs from the previous ones by
    over `threshold` log2 units. A one-frame spike does not count. Returns (rate per frame, size)."""
    n, T = matrix.shape
    if T < 2 * window + 1:
        return 0.0, 0.0
    sizes, chances = [], 0
    for t in range(window, T - window):
        before = np.nanmedian(matrix[:, t - window:t], axis=1)
        after = np.nanmedian(matrix[:, t:t + window], axis=1)
        d = after - before
        ok = np.isfinite(d)
        chances += int(ok.sum())
        sizes.append(np.abs(d[ok & (np.abs(d) > threshold)]))
    sizes = np.concatenate(sizes) if sizes else np.zeros(0)
    # One real switch is seen in about `window` consecutive positions.
    rate = sizes.size / max(chances, 1) / window
    return float(rate), float(np.median(sizes)) if sizes.size else 0.0


def describe(scalars: np.ndarray, channels: np.ndarray, links: dict, frames: int) -> dict:
    """Summary statistics of a tracked movie. Computed the same way for real and synthetic data."""
    t = scalars[:, S["t"]].astype(int)
    counts = np.bincount(t, minlength=frames).astype(float)
    tracks = _tracks(scalars, links, frames)
    drift = np.asarray(links["drift"])
    x = np.where(tracks >= 0, scalars[tracks.clip(0), S["x"]] + drift[np.arange(frames)][None, :, 0], np.nan)
    y = np.where(tracks >= 0, scalars[tracks.clip(0), S["y"]] + drift[np.arange(frames)][None, :, 1], np.nan)
    dx, dy = np.diff(x, axis=1), np.diff(y, axis=1)
    steps = np.hypot(dx, dy)
    steps = steps[np.isfinite(steps)]
    a, b = dx[:, 1:].ravel(), dx[:, :-1].ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    per_track = np.sqrt(np.nanmean(dx**2 + dy**2, axis=1) / 2)
    area = scalars[:, S["nuc_area"]]
    aspect = scalars[:, S["major_len"]] / np.maximum(scalars[:, S["minor_len"]], 1e-3)
    out = {
        "frames": frames,
        "cells_per_frame": [float(np.median(counts[int(f * frames):max(int(f * frames) + 1, int((f + 0.1) * frames))])) for f in np.arange(0, 1, 0.1)],
        "area_q10_50_90": [float(v) for v in np.nanpercentile(area, [10, 50, 90])],
        "aspect_q10_50_90": [float(v) for v in np.nanpercentile(aspect[np.isfinite(aspect)], [10, 50, 90])],
        "step_px_q50_90_99": [float(v) for v in np.percentile(steps, [50, 90, 99])] if steps.size else [0, 0, 0],
        "step_rms_per_axis": float(np.sqrt(np.nanmean(np.concatenate([dx.ravel(), dy.ravel()]) ** 2))) if steps.size else 0.0,
        "persistence": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 20 else 0.0,
        "speed_log_spread": float(np.nanstd(np.log(per_track[per_track > 0]))) if np.any(per_track > 0) else 0.0,
        "channels": [],
    }
    C = channels.shape[1]
    for c in range(C):
        nuc = channels[:, c, K["nuc_median"]]
        cyto = channels[:, c, K["cyto_median"]]
        matrix = np.where(tracks >= 0, np.log2(np.clip(nuc[tracks.clip(0)], 1, None)), np.nan)
        walk, noise = _walk_and_noise(matrix)
        rate, size = _switch_rate(matrix)
        ratio = cyto / np.maximum(nuc, 1)
        cn_matrix = np.where(tracks >= 0, np.log2(np.clip(ratio[tracks.clip(0)], 1e-3, None)), np.nan)
        cn_walk, _ = _walk_and_noise(cn_matrix)
        track_level = np.nanmedian(matrix, axis=1)
        out["channels"].append({
            "nuc_median": float(np.nanmedian(nuc)), "between_cell_log2_sd": float(np.nanstd(track_level)),
            "cn_ratio": float(np.nanmedian(ratio)), "walk_log2_per_frame": walk, "noise_log2": noise,
            "switch_rate": rate, "switch_log2": size, "cn_walk_log2_per_frame": cn_walk,
        })
    out["stage_steps"] = [[int(i) + 1, float(drift[i + 1, 0] - drift[i, 0]), float(drift[i + 1, 1] - drift[i, 1])]
                          for i in np.nonzero(np.abs(np.diff(drift, axis=0)).sum(1))[0]]
    return out


def camera_noise(pairs: np.ndarray) -> tuple[float, float]:
    """Gain and read noise from pairs of consecutive raw frames (P, 2, C, X, Y).

    Uses the darker part of each tile, where cells are absent, and the difference of the two frames,
    which removes fixed structure:  var = gain * mean + read^2.
    """
    means, variances = [], []
    for pair in pairs:
        a, b = pair[0].astype(np.float32), pair[1].astype(np.float32)
        C, X, Y = a.shape
        tile = 64
        for c in range(C):
            for i in range(0, X - tile + 1, tile):
                for j in range(0, Y - tile + 1, tile):
                    ta, tb = a[c, i:i + tile, j:j + tile], b[c, i:i + tile, j:j + tile]
                    keep = (ta < np.quantile(ta, 0.6)) & (tb < np.quantile(tb, 0.6))
                    if keep.sum() < 500:
                        continue
                    d = (ta - tb)[keep]
                    means.append(float((ta[keep] + tb[keep]).mean() / 2))
                    variances.append(float(d.var() / 2))
    m, v = np.array(means), np.array(variances)
    if m.size < 10:
        return 1.5, 6.0
    # Robust line: drop the 10% worst residuals of a first fit, refit.
    g, r2 = np.polyfit(m, v, 1)
    resid = np.abs(v - (g * m + r2))
    keep = resid < np.quantile(resid, 0.9)
    g, r2 = np.polyfit(m[keep], v[keep], 1)
    return float(max(g, 0.05)), float(math.sqrt(max(r2, 1.0)))


def fit_parameters(real: dict, raw_pairs: np.ndarray, frame_shape: tuple, cfg: dict, seed: int = 0) -> dict:
    """Generator settings matching a real movie. `real` is describe() of the real movie."""
    X, Y = frame_shape
    size = int(round(math.sqrt(X * Y) / 16) * 16)
    area = real["area_q10_50_90"][1]
    r0 = math.sqrt(area / (1.53 * math.pi))  # mean area = pi r0^2 * E[scale^2] * E[1 + phase]
    margin = 3 * r0
    visible = (size / (size + 2 * margin)) ** 2
    counts = np.array(real["cells_per_frame"])
    frames = int(real["frames"])

    # Division pace from the count curve, as local doubling times between deciles.
    schedule = []
    positions = (np.arange(10) + 0.5) / 10 * frames
    for i in range(9):
        a, b = max(counts[i], 1.0), max(counts[i + 1], 1.0)
        span = positions[i + 1] - positions[i]
        doubling = span * math.log(2) / math.log(b / a) if b > a * 1.01 else 5000.0
        schedule.append([float((positions[i] + positions[i + 1]) / 2), float(np.clip(doubling, 40.0, 5000.0))])
    base = float(np.median([s[1] for s in schedule]))

    spread = min(real["speed_log_spread"], 1.0)
    speed = real["step_rms_per_axis"] / math.exp(spread**2 / 2)
    lo, _, hi = real["aspect_q10_50_90"]
    chans = real["channels"]
    C = len(chans)

    # Background and illumination from raw frames: one gentle Gaussian matched to the field's falloff.
    raw = torch.as_tensor(raw_pairs[:, 0].astype(np.float32), device=cfg["device"])
    _, diag = background.correct_background(raw, cfg)
    ratio = diag["bg_ratio"].median(dim=0).values.clamp(min=1.01).cpu().numpy()
    corner = math.sqrt(2) * size / 2
    sigma_frac = [float(corner / math.sqrt(2 * math.log(r)) / size) for r in ratio]
    coords = (np.arange(size) - size / 2)
    gain, read = camera_noise(raw_pairs)
    autofluor = []
    for c in range(C):
        s = sigma_frac[c] * size
        field = np.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / (2 * s**2))
        autofluor.append(float(np.median(raw_pairs[:, 0, c]) / field.mean()))

    steps = [[f, -dx, -dy] for f, dx, dy in real["stage_steps"]]  # image motion is minus the correction
    return {
        "seed": seed, "frames": frames, "size": size, "channels": C,
        "cells": int(max(counts[0], 1) / visible), "nuc_radius": round(r0, 2),
        "aspect": (float(max(lo, 1.0)), float(max(hi, lo + 0.05))),
        "speed": round(float(speed), 3), "speed_log_std": round(float(spread), 3),
        "persistence": float(np.clip(real["persistence"], 0.0, 0.95)),
        "cycle_frames": (base, base * 0.25), "cycle_schedule": schedule,
        "illum_sigma": (min(sigma_frac), max(sigma_frac)), "illum_offset": (0.02, 0.05),
        "autofluor_fixed": autofluor, "camera_gain": round(gain, 3), "read_noise": round(read, 2),
        "channel_mean_fixed": [max(ch["nuc_median"], 5.0) for ch in chans],
        "channel_log_std_fixed": [ch["between_cell_log2_sd"] * math.log(2) for ch in chans],
        "cn_fixed": [0.0] + [float(np.clip(ch["cn_ratio"], 0.02, 3.0)) for ch in chans[1:]],
        "abund_step": [ch["walk_log2_per_frame"] * math.log(2) for ch in chans],
        "abund_jump_prob": [min(ch["switch_rate"], 0.05) for ch in chans],
        "abund_jump_log2": [max(ch["switch_log2"], 1.0) for ch in chans],
        "cn_step": [0.0] + [ch["cn_walk_log2_per_frame"] * math.log(2) for ch in chans[1:]],
        "stage_steps": steps,
        # Unknown from a summary; left out so the benchmark isolates segmentation and tracking.
        "debris": 0, "fibers": 0, "blur_frames": 0, "media_change_step": 1.0,
    }


def compare(real: dict, synthetic: dict) -> list[tuple[str, float, float, float]]:
    """(statistic, real, synthetic, ratio) for every number both summaries share."""
    rows = []

    def add(name, a, b):
        rows.append((name, float(a), float(b), float(b / a) if abs(a) > 1e-9 else float("nan")))

    for i, (a, b) in enumerate(zip(real["cells_per_frame"], synthetic["cells_per_frame"])):
        add(f"cells_per_frame@{i * 10}%", a, b)
    for key in ("area_q10_50_90", "aspect_q10_50_90", "step_px_q50_90_99"):
        for q, a, b in zip(("q10", "q50", "q90") if "step" not in key else ("q50", "q90", "q99"), real[key], synthetic[key]):
            add(f"{key.split('_q')[0]}_{q}", a, b)
    for key in ("step_rms_per_axis", "persistence", "speed_log_spread"):
        add(key, real[key], synthetic[key])
    for c, (a, b) in enumerate(zip(real["channels"], synthetic["channels"])):
        for key in ("nuc_median", "between_cell_log2_sd", "cn_ratio", "walk_log2_per_frame", "noise_log2", "switch_rate"):
            add(f"c{c + 1}_{key}", a[key], b[key])
    return rows


def refine(params: dict, real: dict, synthetic: dict) -> dict:
    """One correction step: scale fitted settings by how far the synthetic summary missed.

    The generator's parameters are not the pipeline's measurements (shapes are blurred, dim channels
    are biased by noise and clamping), so a second pass corrects for that empirically.
    """
    out = dict(params)

    def ratio(a, b, floor=1e-6):
        return float(a / b) if abs(b) > floor and abs(a) > floor else 1.0

    out["nuc_radius"] = round(params["nuc_radius"] * math.sqrt(ratio(real["area_q10_50_90"][1], synthetic["area_q10_50_90"][1])), 3)
    out["speed"] = round(params["speed"] * ratio(real["step_rms_per_axis"], synthetic["step_rms_per_axis"]), 3)
    lo, hi = params["aspect"]
    stretch = ratio(real["aspect_q10_50_90"][1] - 1, synthetic["aspect_q10_50_90"][1] - 1, 0.01)
    out["aspect"] = (float(max(1.0, 1 + (lo - 1) * stretch)), float(max(1.05, 1 + (hi - 1) * stretch)))
    for key, stat in (("channel_mean_fixed", "nuc_median"), ("channel_log_std_fixed", "between_cell_log2_sd"),
                      ("cn_fixed", "cn_ratio"), ("abund_step", "walk_log2_per_frame")):
        values = list(params[key])
        for c, (rc, sc) in enumerate(zip(real["channels"], synthetic["channels"])):
            factor = float(np.clip(ratio(rc[stat], sc[stat]), 0.25, 4.0))
            values[c] = values[c] * factor
        out[key] = values
    return out
