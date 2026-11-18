"""Tracks for model fitting: readouts of the chosen conditions as (tracks, frames, readouts) arrays.

The same track filters as the figures (minimum length, never touching the border) apply, and a
track also needs `min_track_frames` usable readout values. With many cells, at most `max_tracks`
per condition are taken, spread evenly over its positions, so no single field dominates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analysis.figures import ResultsCache, _kept_tracks, _window
from ..analysis.readouts import READOUT_DEFAULT, readout_rows
from ..experiment import merge


@dataclass
class TrackData:
    y: np.ndarray            # (N, T, F) float32, NaN where not observed
    se: np.ndarray           # (N, T, F) float32 measurement error of each value (0 = unknown)
    cond: np.ndarray         # (N,) condition index
    position: np.ndarray     # (N,) index into positions
    track: np.ndarray        # (N,) track id within its position
    conditions: list[str]
    positions: list[str]
    readouts: list[str]
    first_frame: int         # result frame of column 0
    available: dict = field(default_factory=dict)  # condition -> usable tracks before subsampling
    rows: dict = field(default_factory=dict)       # extra per-track arrays (experiment, lineage ...), subset with the tracks

    @property
    def T(self) -> int:
        return self.y.shape[1]

    def subset(self, rows: np.ndarray) -> "TrackData":
        rows = np.asarray(rows)
        extra = {k: v[rows] for k, v in self.rows.items()}
        if "mother" in extra:  # rows of mothers, renumbered; -1 if the mother is not in the subset
            new = np.full(len(self.y), -1)
            new[rows] = np.arange(len(rows))
            m = extra["mother"]
            extra["mother"] = np.where(m >= 0, new[np.maximum(m, 0)], -1)
        return TrackData(self.y[rows], self.se[rows], self.cond[rows], self.position[rows], self.track[rows],
                         self.conditions, self.positions, self.readouts, self.first_frame, self.available,
                         extra)


def load_tracks(exp: dict, readouts: list[dict], conditions: list[dict], frames=(0, 0), max_tracks: int = 0,
                seed: int = 0, cache: ResultsCache | None = None, on_event=lambda e: None, stop=None) -> TrackData:
    """Readouts of every kept track in `conditions` (plate condition dicts)."""
    cache = cache or ResultsCache(exp)
    readouts = [merge(READOUT_DEFAULT, r) for r in readouts]
    min_obs = int(exp["analysis"]["min_track_frames"])
    rng = np.random.default_rng(seed)
    ys, ses, conds, poss, tracks, positions = [], [], [], [], [], []
    available = {}
    window = None
    for ci, cond in enumerate(conditions):
        keys = cache.timestacks_of(cond)
        quota = int(np.ceil(max_tracks / max(1, len(keys)))) if max_tracks else 0
        per_cond = []
        for key in keys:
            if stop is not None and stop.is_set():
                raise InterruptedError("stopped")
            res = cache.get(key)
            T = int(res["meta"].get("frames", res.frames))
            if window is None:
                window = _window(exp, {"frames": list(frames)}, T)
            kept = _kept_tracks(res, exp)
            parts = [readout_rows(res, exp, r) for r in readouts]
            first = parts[0]
            ok = kept[first[1]] if len(first[1]) else np.zeros(0, bool)
            ids = np.unique(first[1][ok])
            if not len(ids):
                continue
            row_of = np.full(int(res.tracks) + 1, -1)
            row_of[ids] = np.arange(len(ids))
            n = len(ids)
            W = window.stop - window.start
            y = np.full((n, W, len(readouts)), np.nan, np.float32)
            se = np.zeros((n, W, len(readouts)), np.float32)
            for f, (t, track, value, err) in enumerate(parts):
                sel = (track < len(row_of)) & (t >= window.start) & (t < window.stop)
                rows = row_of[track[sel]]
                good = rows >= 0
                y[rows[good], t[sel][good] - window.start, f] = value[sel][good]
                se[rows[good], t[sel][good] - window.start, f] = err[sel][good]
            enough = np.isfinite(y[:, :, 0]).sum(1) >= min_obs
            pick = np.nonzero(enough)[0]
            per_cond.append(len(pick))
            if quota and len(pick) > quota:
                pick = np.sort(rng.choice(pick, quota, replace=False))
            if not len(pick):
                continue
            positions.append(key)
            ys.append(y[pick])
            ses.append(se[pick])
            conds.append(np.full(len(pick), ci))
            poss.append(np.full(len(pick), len(positions) - 1))
            tracks.append(ids[pick])
            on_event({"type": "loaded", "position": key, "tracks": len(pick)})
        available[cond["name"]] = int(sum(per_cond))
    if not ys:
        raise ValueError("No usable tracks: check the readout's channel, its noise and signal limits, and the minimum track length.")
    return TrackData(np.concatenate(ys), np.concatenate(ses), np.concatenate(conds), np.concatenate(poss), np.concatenate(tracks),
                     [c["name"] for c in conditions], positions, [r["name"] or r["kind"] for r in readouts],
                     window.start if window else 0, available)
