"""Find bad positions before tracking: fibers, debris, bright blobs, empty or out-of-focus fields.

For a few frames of every position, each channel is cut into tiles and two numbers are kept
per tile, using the quantile tile function of the background correction:

    low   the 1st percentile: the background floor. A dark fiber pulls it down; a bright
          out-of-focus blob pushes it up.
    mid   the median: what most of the tile shows. Only a drop counts (a fiber or shadow over
          cells); a rise is usually a dense colony, which is biology, not an artifact.

Each map is taken relative to its own frame's median tile, which removes the overall level,
so different expression or density between conditions does not count. A random sample of
positions sets the expected value and spread of every tile (illumination is shared by all
positions). A tile is unusual when it is more than `z_threshold` robust deviations away; a
frame is bad when unusual tiles cover more than `area_threshold` of it, or its background
level is far from the others; a position is flagged when at least `frame_fraction` of its
sampled frames are bad. A high quantile is not used: in sparse cultures it only says whether
a tile holds a nucleus.
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .experiment import QUALITY_DEFAULTS  # noqa: F401  (re-exported)
from .io import layout
from .track.background import quantile_tile_values_with_positions

_MAPS = ("low", "mid")
_THUMB = 160


def sample_frames(frames: int, n: int) -> list[int]:
    return sorted({int(round(v)) for v in np.linspace(0, frames - 1, min(n, frames))})


def _thumb(x: torch.Tensor) -> np.ndarray:
    """(C,X,Y) -> (C,160,160) uint8, log-scaled between the 1st and 99.7th percentile."""
    small = torch.nn.functional.adaptive_avg_pool2d(torch.log2(x + 1).unsqueeze(0), _THUMB)[0]
    flat = small.flatten(1)
    lo = flat.quantile(0.01, dim=1)[:, None, None]
    hi = flat.quantile(0.997, dim=1)[:, None, None]
    return ((small - lo) / (hi - lo + 1e-6)).clamp(0, 1).mul(255).byte().cpu().numpy()


def position_features(timestack: dict, q: dict, device: str = "cuda:0", pool: ThreadPoolExecutor | None = None) -> dict:
    """Tile maps for the sampled frames of one position: {"frames", "low", "mid"} with maps (F,C,k,k) in log2, and thumbnails."""
    k = int(q["tiles_per_edge"])
    frames = sample_frames(timestack["frames"], int(q["frames_per_position"]))
    own = pool is None
    pool = pool or ThreadPoolExecutor(8)
    before = len(timestack.get("unreadable", []))
    try:
        raw = np.concatenate([layout.load_batch(timestack, t, t + 1, pool) for t in frames])
    finally:
        if own:
            pool.shutdown()
    out = {"frames": frames, "low": [], "mid": [], "thumb": [],
           "unreadable": sorted({u["frame"] for u in timestack.get("unreadable", [])[before:]}
                                | ({m["frame"] for m in timestack.get("missing", [])} & set(frames)))}
    with torch.inference_mode():
        for f in range(len(frames)):
            x = torch.from_numpy(raw[f:f + 1].astype(np.float32)).to(device)
            for name, quantile in (("low", q["low_quantile"]), ("mid", q["mid_quantile"])):
                values, _ = quantile_tile_values_with_positions(x, k, float(quantile))
                out[name].append(torch.log2(values[0] + 1).view(-1, k, k).cpu().numpy())
            out["thumb"].append(_thumb(x[0]))
    for name in (*_MAPS, "thumb"):
        out[name] = np.stack(out[name])
    return out


def _robust(values: np.ndarray, axis) -> tuple[np.ndarray, np.ndarray]:
    centre = np.median(values, axis=axis, keepdims=True)
    spread = 1.4826 * np.median(np.abs(values - centre), axis=axis, keepdims=True)
    return centre, spread


def score_positions(features: dict[str, dict], q: dict, reference: list[str] | None = None,
                    channel_names: list[str] | None = None, keep_maps: bool = False) -> dict[str, dict]:
    """Compare every position with the reference sample. Returns {position: result}."""
    ids = sorted(features)
    reference = [r for r in (reference or ids) if r in features]
    C = features[ids[0]]["low"].shape[1]
    names = channel_names if channel_names and len(channel_names) == C else [f"c{i + 1}" for i in range(C)]
    rel, level = {}, {}
    for key in ids:
        f = features[key]
        rel[key] = {m: f[m] - np.median(f[m], axis=(2, 3), keepdims=True) for m in _MAPS}  # (F,C,k,k)
        level[key] = np.median(f["low"], axis=(2, 3))                                       # (F,C)
    ref = {}
    for m in _MAPS:
        stack = np.concatenate([rel[r][m] for r in reference])          # (frames of all reference positions, C, k, k)
        centre, spread = _robust(stack, axis=0)
        floor = np.maximum(np.median(spread, axis=(2, 3), keepdims=True), 0.02)
        ref[m] = (centre, np.maximum(spread, floor))
    lv = np.concatenate([level[r] for r in reference])
    lv_centre, lv_spread = _robust(lv, axis=0)
    lv_spread = np.maximum(lv_spread, 0.02)

    results = {}
    for key in ids:
        frames = features[key]["frames"]
        z = {m: (rel[key][m] - ref[m][0]) / ref[m][1] for m in _MAPS}    # (F,C,k,k)
        z["mid"] = np.minimum(z["mid"], 0)  # a bright median is dense cells; only a dark one is an artifact
        zmax = np.maximum(np.abs(z["low"]), np.abs(z["mid"]))
        unusual = zmax > q["z_threshold"]
        area = unusual.any(axis=1).mean(axis=(1, 2))                     # (F,) any channel
        lz = (level[key] - lv_centre[0]) / lv_spread[0]                 # (F,C)
        bad = (area > q["area_threshold"]) | (np.abs(lz).max(axis=1) > q["level_threshold"])
        unreadable = set(features[key].get("unreadable", []))
        bad |= np.array([f in unreadable for f in frames], bool)
        reasons = [f"missing or unreadable file (frames {', '.join(map(str, sorted(unreadable)))})"] if unreadable else []
        for c in range(C):
            for m in _MAPS:
                bright = ((z[m][:, c] > q["z_threshold"]).mean(axis=(1, 2)) > q["area_threshold"])
                dark = ((z[m][:, c] < -q["z_threshold"]).mean(axis=(1, 2)) > q["area_threshold"])
                for mask, word in ((bright, "bright"), (dark, "dark")):
                    if mask.any():
                        reasons.append(f"{word} region in {names[c]} (frames {', '.join(str(frames[i]) for i in np.nonzero(mask)[0])})")
            for mask, word in ((lz[:, c] > q["level_threshold"], "bright"), (lz[:, c] < -q["level_threshold"], "dark")):
                if mask.any():
                    reasons.append(f"{word} background in {names[c]} (frames {', '.join(str(frames[i]) for i in np.nonzero(mask)[0])})")
        reasons = list(dict.fromkeys(reasons))
        if bad.any() and not reasons:
            reasons = [f"unusual tiles spread over several channels (frames {', '.join(str(frames[i]) for i in np.nonzero(bad)[0])})"]
        worst = int(np.argmax(area))
        flagged = bool(bad.mean() >= q["frame_fraction"]) if len(bad) else False
        results[key] = {
            "flagged": flagged,
            "bad_frames": [frames[i] for i in np.nonzero(bad)[0]],
            "frames": list(frames),
            "worst_area": float(area.max()),
            "worst_level_z": float(np.abs(lz).max()),
            "reason": "; ".join(reasons) if reasons else "",
            "zmap": zmax[worst].max(axis=0),      # (k,k) of the worst frame, for display
            "worst_frame": worst,
        }
        if keep_maps:
            results[key]["zmaps"] = zmax.max(axis=1)  # (F,k,k): every sampled frame
    return results


def check_positions(timestacks: dict, q: dict, device: str = "cuda:0", channel_names: list[str] | None = None,
                    on_event=lambda e: None, stop=None, io_threads: int = 16) -> tuple[dict, dict]:
    """Features for every position, then scores against a random reference sample. Returns (results, features)."""
    ids = sorted(timestacks)
    rng = random.Random(int(q["seed"]))
    reference = sorted(rng.sample(ids, min(int(q["reference_positions"]), len(ids))))
    features = {}
    with ThreadPoolExecutor(io_threads) as pool:
        for i, key in enumerate(ids):
            if stop is not None and stop.is_set():
                break
            features[key] = position_features(timestacks[key], q, device, pool)
            on_event({"kind": "quality_progress", "done": i + 1, "total": len(ids), "id": key})
    if len(features) < 3:
        raise ValueError("The quality check needs at least 3 positions to compare.")
    return score_positions(features, q, reference, channel_names), features


FILE_FLAG = "missing files"  # every file-based flag reason starts with this


def file_problems(timestack: dict) -> tuple[int, str]:
    """(number of bad frames, description) from the files alone: frames missing from the folder,
    files that could not be read (known once read), and frames lacking at the end."""
    missing = {m["frame"] for m in timestack.get("missing", [])}
    unreadable = {u["frame"] for u in timestack.get("unreadable", [])} - missing
    short = int(timestack.get("short_by", 0))
    parts = []
    if missing:
        parts.append(f"{len(missing)} frame{'s' * (len(missing) > 1)} missing ({layout._ranges(sorted(missing))})")
    if unreadable:
        parts.append(f"{len(unreadable)} unreadable ({layout._ranges(sorted(unreadable))})")
    if short:
        parts.append(f"ends {short} frame{'s' * (short > 1)} early")
    return len(missing) + len(unreadable) + short, ", ".join(parts)


def file_flags(timestacks: dict, q: dict) -> dict[str, str]:
    """{position: reason} for positions with more bad frames than quality.max_missing_frames.
    One missing frame is a blank frame that tracks bridge; more usually means something went wrong
    with that position and it is safer left out (tick Track anyway to use it)."""
    limit = int(q.get("max_missing_frames", 1))
    out = {}
    for key, ts in timestacks.items():
        n, text = file_problems(ts)
        if n > limit:
            out[key] = f"{FILE_FLAG}: {text}"
    return out
