"""A RAM cache of metrics for the whole experiment, so figures stop re-reading the measurement files.

Reading a metric for every position means opening hundreds of HDF5 files (11.6 GB for a 170-position plate) and
transforming tens of millions of rows; that was most of the time a figure took. Here each metric is computed once
per session and kept in RAM, already filtered by the track filter in use (minimum length, border), as compact
columns for the whole experiment:

- frame int16, track int16 (int32 when needed), value float32: 8 bytes per cell-frame, about 0.45 GB per metric
  for 57 million cell-frames. Nothing is written to disk.
- A small base table holds, per position, the frames, cells per frame, and each track's length and border rows.

Policy: `AUTO` metrics are cached for every experiment (each channel's primary measurement, nuclear area, the
summary heatmaps' log2 versions, then the gates' and saved figures' metrics), plus up to `EXTRA` more that the user
draws, least recently used out. Each position's part is tagged with its files' size and modification time, so a
retracked or newly tracked position is recomputed and the rest kept. Changing the track filter rebuilds the cache.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from hashlib import sha1
from pathlib import Path

import numpy as np

AUTO, EXTRA = 10, 5

_STORES: dict[str, "ColumnStore"] = {}
_STORES_LOCK = threading.Lock()


def store_for(results_dir: str | Path) -> "ColumnStore":
    """The one store per results folder, shared by every figure and the GUI in this session."""
    key = str(Path(results_dir).resolve())
    with _STORES_LOCK:
        if key not in _STORES:
            _STORES[key] = ColumnStore(Path(results_dir))
        return _STORES[key]


def metric_key(metric: dict) -> str:
    from .figures import _canonical
    return sha1(json.dumps(_canonical(metric), sort_keys=True).encode()).hexdigest()[:16]


def _signature(results_dir: Path, key: str) -> tuple:
    out = []
    for sub in ("Measurements", "Traces"):
        try:
            st = (results_dir / sub / f"{key}.h5").stat()
            out += [st.st_size, st.st_mtime_ns]
        except OSError:
            out += [0, 0]
    return tuple(out)


class _Column:
    """One metric for many positions: concatenated rows and each position's slice (views, no copies)."""

    def __init__(self, metric: dict, t, track, value, index: dict):
        self.metric, self.t, self.track, self.value, self.index = metric, t, track, value, index

    def slice(self, key: str):
        a, b, _ = self.index[key]
        return self.t[a:b], self.track[a:b], self.value[a:b]

    @property
    def nbytes(self) -> int:
        return self.t.nbytes + self.track.nbytes + self.value.nbytes


class ColumnStore:
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.lock = threading.RLock()
        self.columns: dict[str, _Column] = {}
        self.base: dict[str, dict] = {}
        self.track_filter: tuple | None = None         # (min_track_frames, drop_border) the columns are filtered by
        self.pinned: list[str] = []                      # AUTO metric keys, in priority order
        self.extra: OrderedDict[str, dict] = OrderedDict()   # metric key -> metric, most recent last
        self.wanted: OrderedDict[str, dict] = OrderedDict()  # drawn but not cached yet: built in the background

    # ------------------------------------------------------------------ lookups used while drawing
    def rows(self, metric: dict, key: str, track_filter: tuple):
        """(t, track, value) of one position's tracks that pass the filter, or None (then note the use, so the
        metric gets cached)."""
        mk = metric_key(metric)
        with self.lock:
            col = self.columns.get(mk)
            if col is not None and track_filter == self.track_filter and key in col.index:
                if mk in self.extra:
                    self.extra.move_to_end(mk)
                return col.slice(key)
            if mk not in self.pinned and mk not in self.extra:
                self.wanted[mk] = metric
                self.wanted.move_to_end(mk)
        return None

    def position(self, key: str) -> dict | None:
        """Frames, cells per frame, and per-track length and border rows of one position."""
        with self.lock:
            return self.base.get(key)

    def ram_bytes(self) -> int:
        with self.lock:
            return sum(c.nbytes for c in self.columns.values())

    # ------------------------------------------------------------------ policy
    def plan(self, exp: dict) -> list[dict]:
        """The AUTO metrics of an experiment, in priority order."""
        from ..experiment import METRIC_DEFAULT, loaded_channels, merge, normalise
        from .summary import primary_metric, summary_specs
        exp = normalise(exp)
        metrics = [primary_metric(exp, c["name"]) for c in loaded_channels(exp)]
        metrics.append(merge(METRIC_DEFAULT, {"measure": "nuc_area"}))
        metrics += [s["metric"] for s in summary_specs(exp) if s["kind"] != "cell_count"]   # incl. log2 heatmap variants
        metrics += [g["metric"] for g in exp["analysis"]["gates"]]
        metrics += [f["metric"] for f in exp["figures"]["list"]
                    if f["enabled"] and f["kind"] in ("timecourse", "heatmap", "track_heatmap", "distribution")]
        unique = OrderedDict()
        for m in metrics:
            if m is not None:
                unique.setdefault(metric_key(m), m)
        return list(unique.values())[:AUTO]

    def build(self, exp: dict, keys: list[str], stop=None, on_event=lambda e: None) -> int:
        """Bring the cache up to date: the base table, the AUTO metrics, then metrics drawn since (up to EXTRA,
        least recently used out). Keeps every position whose files have not changed. Returns metrics built."""
        from ..experiment import normalise
        exp = normalise(exp)
        keys = sorted(set(keys))
        a = exp["analysis"]
        track_filter = (int(a["min_track_frames"]), bool(a["drop_border"]))
        sigs = {k: _signature(self.results_dir, k) for k in keys}
        with self.lock:
            if track_filter != self.track_filter:      # every cached column is filtered the old way
                self.columns.clear()
                self.track_filter = track_filter
        self._build_base(keys, sigs, stop)
        auto = self.plan(exp)
        with self.lock:
            self.pinned = [metric_key(m) for m in auto]
            for mk in self.pinned:
                self.extra.pop(mk, None)
                self.wanted.pop(mk, None)
            for mk, m in list(self.wanted.items()):
                self.extra[mk] = m
                self.extra.move_to_end(mk)
            self.wanted.clear()
            while len(self.extra) > EXTRA:
                old, _ = self.extra.popitem(last=False)
                self.columns.pop(old, None)
            todo = auto + list(self.extra.values())
            keep = set(self.pinned) | set(self.extra)
            for mk in [mk for mk in self.columns if mk not in keep]:   # settings changed: no longer wanted
                self.columns.pop(mk)
        built = 0
        for metric in todo:
            if stop is not None and stop.is_set():
                break
            if self._build_metric(exp, metric, keys, sigs, track_filter, stop):
                built += 1
            on_event({"kind": "column_ready", "metric": metric})
        return built

    # ------------------------------------------------------------------ building
    def _build_base(self, keys: list[str], sigs: dict, stop) -> None:
        from ..io import store
        with self.lock:
            base = dict(self.base)
        for key in keys:
            if stop is not None and stop.is_set():
                return
            if key in base and base[key]["sig"] == sigs[key]:
                continue
            try:
                r = store.load_results(self.results_dir, key)
                t = r.column("t").astype(np.int64)
                track = np.asarray(r["track_id"], dtype=np.int64)
                T = int(r["meta"].get("frames", r.frames))
                ok = track >= 0
                n = int(r.tracks)
                border = r.column("touches_border")[ok] > 0
                base[key] = {"sig": sigs[key], "T": T,
                             "counts": np.bincount(t, minlength=T)[:T].astype(np.int32),
                             "track_len": np.bincount(track[ok], minlength=n)[:n].astype(np.int32),
                             "track_border": np.bincount(track[ok], weights=border, minlength=n)[:n].astype(np.int32)}
            except (OSError, KeyError):
                base.pop(key, None)
        with self.lock:
            self.base = base

    def _build_metric(self, exp: dict, metric: dict, keys: list[str], sigs: dict, track_filter: tuple, stop) -> bool:
        """Recompute the positions that changed or are new, keep the rest, and hold the result in RAM."""
        from ..io import store
        from .figures import _metric_rows
        mk = metric_key(metric)
        with self.lock:
            col = self.columns.get(mk)
            base = self.base
        old = col.index if col is not None else {}
        if col is not None and all(k in old and old[k][2] == sigs[k] for k in keys if k in base):
            return False
        min_frames, drop_border = track_filter
        parts, index, start = [], {}, 0
        for key in keys:
            if stop is not None and stop.is_set():
                return False
            if key not in base:
                continue
            if key in old and old[key][2] == sigs[key]:
                t, track, value = col.slice(key)
            else:
                try:
                    rows = _metric_rows(store.load_results(self.results_dir, key), exp, metric)
                except (OSError, KeyError, ValueError):
                    continue
                kept = base[key]["track_len"] >= min_frames
                if drop_border:
                    kept &= base[key]["track_border"] == 0
                keep = kept[rows.track]
                t, track, value = rows.t[keep], rows.track[keep], rows.value[keep]
            parts.append((t, track, np.asarray(value, dtype=np.float32)))
            index[key] = [start, start + len(t), sigs[key]]
            start += len(t)
        if not parts:
            return False
        T_max = max(base[k]["T"] for k in index)
        n_tracks = max(len(base[k]["track_len"]) for k in index)
        t_type = np.int16 if T_max < 32767 else np.int32
        track_type = np.int16 if n_tracks < 32767 else np.int32
        new = _Column(metric,
                      np.concatenate([p[0].astype(t_type, copy=False) for p in parts]),
                      np.concatenate([p[1].astype(track_type, copy=False) for p in parts]),
                      np.concatenate([p[2] for p in parts]),
                      index)
        with self.lock:
            if self.track_filter == track_filter:
                self.columns[mk] = new
        return True
