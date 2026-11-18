"""Readouts: what a reporter measures, per cell per frame, with each observation's own error.

A readout turns raw measurements into the quantity a model fits, and says how uncertain every
single value is, from the pixels it came from. Kinds:

    ktr               ln(cytoplasm / nucleus) of a kinase translocation reporter (ERK KTR and
                      others): high = kinase active. Per-frame background floor as in the figures.
    nuclear_reporter  ln(nucleus / cytoplasm): a reporter that enters the nucleus when active.
    intensity         ln(nuclear mean): amount reporters (H2B, a fate marker).
    speed             distance moved since the previous frame, px per frame, stage drift removed.
    metric            any figure metric (measure, ratio, log2 ...), with no error estimate.

The error (se) of a median of n pixels with spread s is about 1.25 s / √n; for ln(a/b) the
relative errors add in quadrature. Observations with se above `max_noise` count as missing;
tracks whose mean nuclear signal above background is below `min_signal` are left out (cells
that barely express the reporter carry no information on it).
"""
from __future__ import annotations

import math

import numpy as np

from ..experiment import METRIC_DEFAULT, READOUT_DEFAULT, merge
from . import traces as tr

KINDS = ["ktr", "nuclear_reporter", "intensity", "speed", "metric"]
KIND_TEXT = {"ktr": "Kinase translocation reporter: ln(cytoplasm / nucleus)",
             "nuclear_reporter": "Nuclear translocation: ln(nucleus / cytoplasm)",
             "intensity": "Nuclear intensity: ln(nuclear mean)",
             "speed": "Movement: px per frame, stage drift removed",
             "metric": "Any metric (as in figures), no error estimate"}
MEDIAN_SE = math.sqrt(math.pi / 2)  # se of a median = 1.2533 sd / sqrt(n) for normal pixels


def suggest(exp: dict) -> list[dict]:
    """Readouts for channels whose reporter name says what they are (KTR -> ktr)."""
    out = []
    for ch in exp["channels"]:
        if ch["role"] == "skip":
            continue
        rep = (ch.get("reporter") or "").lower()
        if "ktr" in rep:
            name = (ch["reporter"] or ch["name"]).replace("KTR", "").replace("ktr", "").strip(" _-") or ch["name"]
            out.append(merge(READOUT_DEFAULT, {"name": name, "kind": "ktr", "channel": ch["name"]}))
    return out


def _channel(exp: dict, name: str) -> int:
    from .figures import _channel as ch
    return ch(exp, name)


def drift_corrected_xy(results) -> tuple[np.ndarray, np.ndarray]:
    t = results.column("t").astype(np.int64)
    x, y = results.column("x").astype(np.float64), results.column("y").astype(np.float64)
    drift = np.asarray(results["drift"]) if "drift" in results else np.zeros((1, 2))
    if drift.shape[0] > t.max(initial=0):
        x, y = x + drift[t, 0], y + drift[t, 1]
    return x, y


def speed_column(results) -> np.ndarray:
    """Per measurement row: distance from the same track's previous observation, divided by the
    frames between them (px per frame), stage drift removed. NaN on a track's first frame."""
    cache = results._matrices
    if "_speed" in cache:
        return cache["_speed"]
    t = results.column("t").astype(np.int64)
    track = np.asarray(results["track_id"], dtype=np.int64)
    x, y = drift_corrected_xy(results)
    out = np.full(len(t), np.nan)
    ok = np.nonzero(track >= 0)[0]
    order = ok[np.lexsort((t[ok], track[ok]))]
    same = track[order][1:] == track[order][:-1]
    gap = (t[order][1:] - t[order][:-1]).astype(np.float64)
    step = np.hypot(np.diff(x[order]), np.diff(y[order])) / np.maximum(gap, 1)
    out[order[1:][same]] = step[same]
    cache["_speed"] = out
    return out


def readout_rows(results, exp: dict, readout: dict):
    """(t, track, value, se) for every measurement row with a usable value. se is 0 where unknown.
    Not filtered by track length or border (the caller applies the analysis filters). Cached."""
    r = merge(READOUT_DEFAULT, readout)
    key = ("_readout", r["kind"], r["channel"], float(r["background_percentile"]), float(r["max_noise"]), float(r["min_signal"]),
           repr(sorted(r["metric"].items())) if r["kind"] == "metric" else "")
    cache = results._matrices
    if key in cache:
        return cache[key]
    t = results.column("t").astype(np.int64)
    track = np.asarray(results["track_id"], dtype=np.int64)
    T = int(results["meta"].get("frames", getattr(results, "frames", t.max(initial=0) + 1)))
    se = np.zeros(len(t))
    if r["kind"] in ("ktr", "nuclear_reporter"):
        c = _channel(exp, r["channel"])
        nuc, cyto = results.column("nuc_median", c).astype(np.float64), results.column("cyto_median", c).astype(np.float64)
        ok = np.isfinite(nuc)
        floor = np.zeros(T)
        if r["background_percentile"] > 0 and ok.any():
            floor = tr.grouped_quantiles(t[ok], nuc[ok], T, [r["background_percentile"] / 100])[0]
            floor = np.nan_to_num(floor)
        N, Cy = nuc - floor[t], cyto - floor[t]
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.log(Cy / N) if r["kind"] == "ktr" else np.log(N / Cy)
            rel_n = results.column("nuc_std", c) / N / np.sqrt(np.maximum(results.column("nuc_area"), 1))
            rel_c = results.column("cyto_std", c) / Cy / np.sqrt(np.maximum(results.column("cyto_area"), 1))
            se = MEDIAN_SE * np.sqrt(rel_n ** 2 + rel_c ** 2)
        value[(N <= 0) | (Cy <= 0)] = np.nan
        if r["min_signal"] > 0:
            good = (track >= 0) & np.isfinite(N)
            n_tracks = int(track.max(initial=-1)) + 1
            total = np.bincount(track[good], weights=N[good], minlength=n_tracks)
            count = np.bincount(track[good], minlength=n_tracks)
            mean = total / np.maximum(count, 1)
            value[(track >= 0) & (mean[np.maximum(track, 0)] < r["min_signal"])] = np.nan
    elif r["kind"] == "intensity":
        c = _channel(exp, r["channel"])
        mean = results.column("nuc_mean", c).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.log(mean)
            se = results.column("nuc_std", c) / mean / np.sqrt(np.maximum(results.column("nuc_area"), 1))
        value[mean <= 0] = np.nan
    elif r["kind"] == "speed":
        value = speed_column(results).copy()
    else:
        from .figures import _metric_rows
        rows = _metric_rows(results, exp, merge(METRIC_DEFAULT, r["metric"]))
        out = (rows.t, rows.track, rows.value, np.zeros(len(rows.t)))
        cache[key] = out
        return out
    se = np.where(np.isfinite(se), se, np.inf)
    if r["max_noise"] > 0:
        value[se > r["max_noise"]] = np.nan
    se[~np.isfinite(se)] = 0
    keep = (track >= 0) & np.isfinite(value)
    out = (t[keep], track[keep], value[keep], se[keep])
    cache[key] = out
    return out
