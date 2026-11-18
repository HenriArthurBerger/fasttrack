"""Features: many measurements per cell per frame, on one scale, for discovery and fate analyses.

A feature is one measurement with a transform: a cell measurement (nuclear area, shape, local
density), a channel measurement (nuclear mean of PPARG ...), speed, age (frames since the cell's
observed birth), or a readout (readout:<name>, which brings its per-observation error).

Features are named by what they measure (by default "<reporter> <measurement>"), so experiments
that imaged different reporters pool by name: a feature an experiment lacks is simply missing
for its cells, and models skip missing features observation by observation.

Values are not stored: they are computed from the saved measurements when needed (and cached
per position). Each experiment is normalised to its own reference condition: 0 = the reference
median, 1 = its robust spread (1.4826 x median absolute deviation), so plates pool and every
feature reads in the same units.

Lineage comes from the tracker's mother links: a track with a mother was born in the movie
(its first frame is a birth), a track with daughters ends in a division.
"""
from __future__ import annotations

import math

import numpy as np

from ..experiment import FEATURE_DEFAULT, FEATURES_DEFAULTS, merge
from ..track.measure import CHANNELS, SCALARS

TRANSFORMS = ["log", "asinh", "none"]
CLIP = 10.0   # normalised values beyond ±10 robust spreads are clipped when pooled for models
DERIVED = ["speed", "age"]
NOT_FEATURES = {"t", "x", "y", "label", "underseg_score", "touches_border", "orientation"}
CELL_MEASURES = [m for m in SCALARS if m not in NOT_FEATURES] + DERIVED
MEDIAN_SE = math.sqrt(math.pi / 2)
_SE_OF = {"nuc_mean": 1.0, "nuc_median": MEDIAN_SE, "cyto_mean": 1.0, "cyto_median": MEDIAN_SE}  # log transform: se from pixels


def measures() -> list[str]:
    """Every measurement a feature can use (readouts are added as readout:<name>)."""
    return CELL_MEASURES + CHANNELS


def is_channel_measure(measure: str) -> bool:
    return measure in CHANNELS


def default_name(exp: dict, measure: str, channel: str) -> str:
    if measure.startswith("readout:"):
        return measure.split(":", 1)[1]
    if not is_channel_measure(measure):
        return measure
    for ch in exp["channels"]:
        if ch["name"] == channel:
            return f"{ch.get('reporter') or ch['name']} {measure}"
    return f"{channel} {measure}"


def suggest(exp: dict) -> list[dict]:
    """A broad, unbiased set: nuclear size, shape and texture, DNA content, density, movement, age,
    and the level and localisation of every measured reporter. Readouts are added as they are."""
    def f(measure, channel="", transform="none"):
        return merge(FEATURE_DEFAULT, {"name": default_name(exp, measure, channel), "measure": measure, "channel": channel,
                                       "transform": transform})

    out = [f("nuc_area", transform="log"), f("eccentricity"), f("circularity"), f("radial_rough", transform="log"),
           f("local_density", transform="asinh"), f("speed", transform="asinh"), f("age")]
    channels = [c for c in exp["channels"] if c["role"] != "skip"] or [{"name": "c1", "role": "nuclear", "reporter": ""}]
    readout_channels = {r["channel"] for r in exp["analysis"]["readouts"] if r["kind"] in ("ktr", "nuclear_reporter")}
    for ch in channels:
        if ch["role"] == "nuclear":
            out += [f("nuc_sum", ch["name"], "log"), f("nuc_mean", ch["name"], "log"), f("bimodality", ch["name"])]
        elif ch["name"] not in readout_channels:
            out += [f("nuc_mean", ch["name"], "log"), f("nc_ratio", ch["name"], "log")]
    for r in exp["analysis"]["readouts"]:
        if r["name"]:
            out.append(f(f"readout:{r['name']}"))
    return out


def feature_list(exp: dict, enabled_only: bool = True) -> list[dict]:
    feats = [merge(FEATURE_DEFAULT, x) for x in exp["analysis"]["features"]["list"]] or suggest(exp)
    return [x for x in feats if x["enabled"] or not enabled_only]


def by_name(exp: dict, names: list[str]) -> list[dict]:
    feats = {x["name"]: x for x in feature_list(exp, enabled_only=False)}
    missing = [n for n in names if n not in feats]
    if missing:
        raise KeyError(f"no feature called {missing}; features are {list(feats)}")
    return [feats[n] for n in names]


# ---------------------------------------------------------------------- per position
def lineage(results) -> dict:
    """Per track id: first and last frame, mother (-1 = none), born in the movie, ends in a division."""
    cache = results._matrices
    if "_lineage" in cache:
        return cache["_lineage"]
    track = np.asarray(results["track_id"], dtype=np.int64)
    t = results.column("t").astype(np.int64)
    n = int(results.tracks)
    ok = track >= 0
    first = np.full(n, np.iinfo(np.int64).max)
    last = np.full(n, -1)
    np.minimum.at(first, track[ok], t[ok])
    np.maximum.at(last, track[ok], t[ok])
    mother = np.asarray(results["mother_track"], dtype=np.int64)[:n]
    divides = np.zeros(n, bool)
    divides[mother[mother >= 0]] = True
    out = {"first": first, "last": last, "mother": mother, "born": mother >= 0, "divides": divides}
    cache["_lineage"] = out
    return out


def feature_rows(results, exp: dict, feature: dict):
    """(t, track, value, se) of one feature for every measurement row with a finite value. Cached."""
    from .figures import _channel
    from .readouts import readout_rows, speed_column
    f = merge(FEATURE_DEFAULT, feature)
    key = ("_feature", f["measure"], f["channel"], f["transform"])
    readout = None
    if f["measure"].startswith("readout:"):
        name = f["measure"].split(":", 1)[1]
        found = [r for r in exp["analysis"]["readouts"] if r["name"] == name]
        if not found:
            raise KeyError(f"feature {f['name']} uses readout {name!r}, which is not defined")
        readout = found[0]
        key += (repr(sorted((k, repr(v)) for k, v in readout.items())),)
    cache = results._matrices
    if key in cache:
        return cache[key]
    if readout is not None:
        t, track, value, se = readout_rows(results, exp, readout)
        value, se = value.astype(np.float64), se.astype(np.float64)
    else:
        t = results.column("t").astype(np.int64)
        track = np.asarray(results["track_id"], dtype=np.int64)
        se = np.zeros(len(t))
        m = f["measure"]
        if m == "speed":
            value = speed_column(results).copy()
        elif m == "age":
            lin = lineage(results)
            safe = np.maximum(track, 0)
            value = np.where((track >= 0) & lin["born"][safe], t - lin["first"][safe], np.nan).astype(np.float64)
        elif is_channel_measure(m):
            c = _channel(exp, f["channel"])
            value = results.column(m, c).astype(np.float64)
            if m in _SE_OF and f["transform"] == "log":
                with np.errstate(invalid="ignore", divide="ignore"):
                    se = _SE_OF[m] * results.column("nuc_std" if m.startswith("nuc") else "cyto_std", c) / value / \
                        np.sqrt(np.maximum(results.column("nuc_area" if m.startswith("nuc") else "cyto_area"), 1))
        elif m in SCALARS:
            value = results.column(m).astype(np.float64)
        else:
            raise KeyError(f"feature {f['name']}: unknown measurement {m!r}")
        with np.errstate(invalid="ignore", divide="ignore"):
            if f["transform"] == "log":
                value = np.where(value > 0, np.log(np.where(value > 0, value, 1)), np.nan)
            elif f["transform"] == "asinh":
                value = np.arcsinh(value)
        keep = (track >= 0) & np.isfinite(value)
        t, track, value, se = t[keep], track[keep], value[keep], np.nan_to_num(se[keep], nan=0, posinf=0)
    out = (t, track, value, se)
    cache[key] = out
    return out


def position_block(results, exp: dict, features: list[dict], window: slice, stats: dict | None = None) -> dict:
    """Every kept track of one position as arrays over the frame window:
        ids (n,), y (n, W, F) float32 (normalised when `stats` is given), se (n, W, F),
        first, last (n,) frame columns, mother (n,) row in this block or -1, born, divides (n,)."""
    from .figures import _kept_tracks
    kept = _kept_tracks(results, exp)
    parts = [feature_rows(results, exp, f) for f in features]
    W = window.stop - window.start
    present = np.zeros(len(kept), bool)
    for t, track, _, _ in parts:
        sel = (t >= window.start) & (t < window.stop)
        present[track[sel]] = True
    ids = np.nonzero(present & kept)[0]
    row_of = np.full(len(kept), -1)
    row_of[ids] = np.arange(len(ids))
    y = np.full((len(ids), W, len(features)), np.nan, np.float32)
    se = np.zeros((len(ids), W, len(features)), np.float32)
    for k, (t, track, value, err) in enumerate(parts):
        sel = (t >= window.start) & (t < window.stop)
        rows = row_of[track[sel]]
        good = rows >= 0
        cols = t[sel][good] - window.start
        y[rows[good], cols, k] = value[sel][good]
        se[rows[good], cols, k] = err[sel][good]
        if stats is not None:
            mu, sd = stats["median"].get(features[k]["name"], 0.0), stats["scale"].get(features[k]["name"], 1.0)
            y[..., k] = (y[..., k] - mu) / sd
            se[..., k] = se[..., k] / sd
    lin = lineage(results)
    mother_id = lin["mother"][ids]
    mother = np.where(mother_id >= 0, row_of[np.maximum(mother_id, 0)], -1)
    return {"ids": ids, "y": y, "se": se, "first": lin["first"][ids] - window.start, "last": lin["last"][ids] - window.start,
            "mother": mother, "born": lin["born"][ids], "divides": lin["divides"][ids], "names": [f["name"] for f in features]}


# ---------------------------------------------------------------------- normalisation
def reference_condition(exp: dict, conditions: list[dict] | None = None) -> dict:
    name = exp["analysis"]["features"]["reference"]
    for c in exp["plate"]["conditions"]:
        if c["name"] == name:
            return c
    pool = conditions or exp["plate"]["conditions"]
    if not pool:
        raise ValueError("No conditions on the plate.")
    return pool[0]


def reference_stats(exp: dict, features: list[dict], conditions: list[dict] | None = None, cache=None,
                    max_rows: int = 2_000_000, seed: int = 0) -> dict:
    """{median: {name: m}, scale: {name: s}, reference: condition} from the reference condition's kept tracks."""
    from .figures import ResultsCache, _kept_tracks, _window
    cache = cache or ResultsCache(exp)
    ref = reference_condition(exp, conditions)
    keys = cache.timestacks_of(ref)
    if not keys:
        raise ValueError(f"No results for the reference condition {ref['name']}.")
    rng = np.random.default_rng(seed)
    spec = merge(FEATURES_DEFAULTS, exp["analysis"]["features"])
    values = {f["name"]: [] for f in features}
    per = max(1, max_rows // len(keys))
    for key in keys:
        res = cache.get(key)
        T = int(res["meta"].get("frames", res.frames))
        win = _window(exp, {"frames": list(spec["reference_frames"])}, T)
        kept = _kept_tracks(res, exp)
        for f in features:
            t, track, v, _ = feature_rows(res, exp, f)
            sel = kept[track] & (t >= win.start) & (t < win.stop)
            v = v[sel]
            if len(v) > per:
                v = rng.choice(v, per, replace=False)
            values[f["name"]].append(v)
    med, scale = {}, {}
    for name, parts in values.items():
        v = np.concatenate(parts) if parts else np.zeros(0)
        if not len(v):
            med[name], scale[name] = 0.0, 1.0
            continue
        m = float(np.median(v))
        s = 1.4826 * float(np.median(np.abs(v - m)))
        if not s > 0:
            s = float(np.std(v)) or 1.0
        med[name], scale[name] = m, s
    return {"median": med, "scale": scale, "reference": ref["name"]}


# ---------------------------------------------------------------------- pooled tracks
def load_features(experiments: list[dict], names: list[str] | None = None, conditions: list[str] | None = None,
                  frames=(0, 0), max_tracks: int = 0, seed: int = 0, on_event=lambda e: None, stop=None):
    """Tracks of one or more experiments as one TrackData of normalised features.

    Features pool by name: the union over experiments, NaN where an experiment does not have one.
    Conditions pool by name too. Each experiment is normalised to its own reference condition, and values
    beyond ±10 robust spreads are clipped.
    Per-track extras (`rows`): experiment, born, divides, mother (row in the result, -1 if absent)."""
    from ..dynamics.data import TrackData
    from .figures import ResultsCache, _window
    rng = np.random.default_rng(seed)
    per_exp = []
    all_names: list[str] = []
    for exp in experiments:
        feats = feature_list(exp) if names is None else [f for f in feature_list(exp, enabled_only=False) if f["name"] in names]
        per_exp.append(feats)
        all_names += [f["name"] for f in feats if f["name"] not in all_names]
    if names is not None:
        all_names = [n for n in names if n in all_names]
    if not all_names:
        raise ValueError("No features: suggest or define some (Analysis → Features).")
    cond_names: list[str] = []
    for exp in experiments:
        for c in exp["plate"]["conditions"]:
            if (not conditions or c["name"] in conditions) and c["name"] not in cond_names:
                cond_names.append(c["name"])
    ys, ses, conds, poss, tracks, exps, born, divides, mothers, lasts, positions = [], [], [], [], [], [], [], [], [], [], []
    available: dict[str, int] = {}
    T_out = None
    first_frame = 0
    for e, (exp, feats) in enumerate(zip(experiments, per_exp)):
        cache = ResultsCache(exp)
        chosen = [c for c in exp["plate"]["conditions"] if c["name"] in cond_names]
        stats = reference_stats(exp, feats, chosen, cache, seed=seed)
        on_event({"type": "log", "message": f"{exp['experiment']['name'] or e}: normalised to {stats['reference']}"})
        col = [all_names.index(f["name"]) for f in feats]
        for cond in chosen:
            keys = cache.timestacks_of(cond)
            quota = int(np.ceil(max_tracks / max(1, len(keys)))) if max_tracks else 0
            for key in keys:
                if stop is not None and stop.is_set():
                    raise InterruptedError("stopped")
                res = cache.get(key)
                T = int(res["meta"].get("frames", res.frames))
                win = _window(exp, {"frames": list(frames)}, T)
                if T_out is None:
                    T_out, first_frame = win.stop - win.start, win.start
                block = position_block(res, exp, feats, win, stats)
                np.clip(block["y"], -CLIP, CLIP, out=block["y"])   # one bad mask must not create a state of its own
                n = len(block["ids"])
                available[cond["name"]] = available.get(cond["name"], 0) + n
                pick = np.arange(n)
                if quota and n > quota:
                    pick = np.sort(rng.choice(n, quota, replace=False))
                if not len(pick):
                    continue
                W = min(T_out, block["y"].shape[1])
                y = np.full((len(pick), T_out, len(all_names)), np.nan, np.float32)
                se = np.zeros_like(y)
                y[:, :W][..., col] = block["y"][pick, :W]
                se[:, :W][..., col] = block["se"][pick, :W]
                positions.append(f"{exp['experiment']['name']}/{key}" if len(experiments) > 1 else key)
                base = sum(len(a) for a in ys)
                remap = np.full(n, -1)
                remap[pick] = base + np.arange(len(pick))
                m = block["mother"][pick]
                mothers.append(np.where(m >= 0, remap[np.maximum(m, 0)], -1))
                ys.append(y)
                ses.append(se)
                conds.append(np.full(len(pick), cond_names.index(cond["name"])))
                poss.append(np.full(len(pick), len(positions) - 1))
                tracks.append(block["ids"][pick])
                exps.append(np.full(len(pick), e))
                born.append(block["born"][pick])
                divides.append(block["divides"][pick])
                lasts.append(block["last"][pick])
                on_event({"type": "loaded", "position": positions[-1], "tracks": len(pick)})
    if not ys:
        raise ValueError("No tracks with these features in the chosen conditions.")
    data = TrackData(np.concatenate(ys), np.concatenate(ses), np.concatenate(conds), np.concatenate(poss), np.concatenate(tracks),
                     cond_names, positions, all_names, first_frame, available)
    data.rows = {"experiment": np.concatenate(exps), "born": np.concatenate(born), "divides": np.concatenate(divides),
                 "mother": np.concatenate(mothers), "last": np.concatenate(lasts)}
    return data
