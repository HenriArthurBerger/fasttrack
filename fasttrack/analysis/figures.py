"""Figures drawn from the experiment file, so the same file always redraws the same figures.

A figure is a dict in `figures.list` (see experiment.FIGURE_DEFAULT): a kind, a group of
conditions, a metric, and a frame window. `render(exp, spec)` draws one on a matplotlib
Figure; `render_all(exp)` writes every enabled figure, the numbers behind it as CSV, and a
copy of the experiment file into <results>/<figures.folder>.
"""
from __future__ import annotations

import csv
import json
import re
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib.figure import Figure

from ..experiment import METRIC_DEFAULT, merge, normalise, save_experiment
from ..plotting import style as pstyle
from ..io import store
from ..track.measure import CHANNELS
from . import bands as B
from . import traces as tr


HEATMAP_ROWS = 1000  # more rows than a screen or print shows are averaged in groups


class ResultsCache:
    """Loads each timestack's results once while drawing many figures."""

    def __init__(self, exp: dict):
        self.exp = exp
        self.results_dir = Path(exp["paths"]["results"])
        self._loaded: dict[str, dict] = {}
        from .columns import store_for
        self.columns = store_for(self.results_dir)   # metrics cached for the whole experiment (RAM and disk)

    def available(self) -> list[str]:
        return sorted(p.stem for p in (self.results_dir / "Traces").glob("*.h5"))

    def timestacks_of(self, condition: dict) -> list[str]:
        plate, quality = self.exp["plate"], self.exp["quality"]
        wells = set(condition["wells"]) - set(plate["excluded_wells"])
        skip = set(plate["excluded_positions"])
        if quality["enabled"]:
            skip |= set(quality["flagged"]) - set(quality["keep"])
        return [k for k in self.available() if k.split("_xy")[0] in wells and k not in skip]

    def get(self, key: str) -> store.Results:
        if key not in self._loaded:
            path = self.results_dir / "Measurements" / f"{key}.h5"
            try:  # older results: convert once to the per-measurement layout, which reads far faster
                store.upgrade_measurements(path)
            except OSError:
                pass  # read-only folder: read the old layout as it is
            self._loaded[key] = store.load_results(self.results_dir, key)
        return self._loaded[key]


def selected_keys(exp: dict, cache: "ResultsCache") -> list[str]:
    """Every position some condition of the plate draws."""
    return sorted({k for c in exp["plate"]["conditions"] for k in cache.timestacks_of(c)})


def warm(exp: dict, cache: "ResultsCache", stop=None, on_event=lambda e: None) -> int:
    """Bring the metric cache up to date in the background (see columns.py): each channel's primary measurement,
    nuclear area, the saved figures' and gates' metrics, and metrics drawn recently. Returns metrics built."""
    exp = normalise(exp)
    return cache.columns.build(exp, selected_keys(exp, cache), stop, on_event)


def position_rows(cache: "ResultsCache", key: str, exp: dict, metric: dict) -> "Rows":
    """A metric of one position, filtered: from the metric cache when it holds it, else from the files."""
    a = exp["analysis"]
    hit = cache.columns.rows(metric, key, (int(a["min_track_frames"]), bool(a["drop_border"])))
    base = cache.columns.position(key) if hit is not None else None
    if hit is None or base is None:
        return metric_rows(cache.get(key), exp, metric)
    t, track, value = hit
    return Rows(t, track, value, base["T"])


def frame_counts(cache: "ResultsCache", key: str) -> np.ndarray:
    """Cells (rows) per frame of one position."""
    base = cache.columns.position(key)
    if base is not None:
        return base["counts"].astype(float)
    r = cache.get(key)
    return np.bincount(r.column("t").astype(int), minlength=int(r["meta"]["frames"])).astype(float)


# ---------------------------------------------------------------------- metrics
def _channel(exp: dict, name: str) -> int:
    names = [c["name"] for c in exp["channels"] if c["role"] != "skip"]
    if name in names:
        return names.index(name)
    m = re.fullmatch(r"c(\d+)", name or "")
    if not names and m:  # no channels described: c1, c2 ... in file order
        return int(m.group(1)) - 1
    raise KeyError(f"no measured channel called {name!r}; channels are {names or ['c1', 'c2', '...']}")


class Rows(NamedTuple):
    """A metric in long form: one entry per measurement row that has a finite value."""
    t: np.ndarray       # frame, int64
    track: np.ndarray   # track, int64
    value: np.ndarray   # float64
    T: int              # frames in the movie


DERIVED = ["speed"]  # computed from the saved measurements, not stored


def _column(results, exp: dict, measure: str, channel: str) -> np.ndarray:
    if measure == "speed":
        from .readouts import speed_column
        return speed_column(results)
    if measure in CHANNELS:
        return results.column(measure, _channel(exp, channel)).astype(np.float64)
    return results.column(measure).astype(np.float64)


def _canonical(metric: dict) -> dict:
    """A metric spec with every key and fixed types, so 5 and 5.0 (file vs editor) are the same cache entry."""
    m = merge(METRIC_DEFAULT, metric)
    m["background_percentile"] = float(m["background_percentile"])
    m["smooth"] = int(m["smooth"])
    m["log2"] = bool(m["log2"])
    return m


def metric_rows(results, exp: dict, metric: dict, filtered: bool = True) -> Rows:
    """A metric spec for one timestack, computed on measurement rows (no (tracks x frames) matrix).
    The metric is cached per position unfiltered; the track filter is a cheap mask on top, so changing
    the minimum track length or the border rule does not recompute anything heavy."""
    cache = results._matrices
    metric = _canonical(metric)
    channels = tuple(c["name"] for c in exp["channels"] if c["role"] != "skip")
    base_key = ("_rows", json.dumps(metric, sort_keys=True), channels)
    if base_key not in cache:
        cache[base_key] = _metric_rows(results, exp, metric)
    rows = cache[base_key]
    if not filtered:
        return rows
    key = (*base_key, exp["analysis"]["min_track_frames"], exp["analysis"]["drop_border"])
    if key not in cache:
        keep = _kept_tracks(results, exp)[rows.track]
        cache[key] = Rows(rows.t[keep], rows.track[keep], rows.value[keep], rows.T)
    return cache[key]


def _metric_rows(results, exp: dict, metric: dict) -> Rows:
    t = results.column("t").astype(np.int64)
    track = np.asarray(results["track_id"], dtype=np.int64)
    T = int(results["meta"].get("frames", results.frames))
    values = _column(results, exp, metric["measure"], metric["channel"])
    if metric["over_measure"]:
        below = _column(results, exp, metric["over_measure"], metric["over_channel"] or metric["channel"])
        if metric["background_percentile"] > 0:
            ok = np.isfinite(below)
            floor = tr.grouped_quantiles(t[ok], below[ok], T, [metric["background_percentile"] / 100])[0]
            values, below = values - floor[t], below - floor[t]
        with np.errstate(invalid="ignore", divide="ignore"):
            values = values / below
    if metric["log2"]:
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.log2(np.where(values > 0, values, np.nan))
    values[~np.isfinite(values)] = np.nan
    if metric["smooth"] > 1:
        smoothed = tr.grouped_trailing_mean(track, t, values, int(metric["smooth"]))
        values = np.where(np.isnan(values), np.nan, smoothed)
    keep = (track >= 0) & np.isfinite(values)
    return Rows(t[keep], track[keep], values[keep], T)


def _kept_tracks(results, exp: dict) -> np.ndarray:
    """Per track: long enough, and never touching the image border. From rows; cached."""
    a = exp["analysis"]
    key = ("_kept_rows", int(a["min_track_frames"]), bool(a["drop_border"]))
    cache = results._matrices
    if key not in cache:
        track = np.asarray(results["track_id"], dtype=np.int64)
        ok = track >= 0
        n_tracks = int(results.tracks)
        keep = np.bincount(track[ok], minlength=n_tracks)[:n_tracks] >= a["min_track_frames"]
        if a["drop_border"]:
            border = results.column("touches_border")[ok] > 0
            keep &= np.bincount(track[ok], weights=border, minlength=n_tracks)[:n_tracks] == 0
        cache[key] = keep
    return cache[key]


def _dense_ids(track: np.ndarray) -> tuple[np.ndarray, int]:
    """Renumber track ids 0..n-1 in id order. Ids are small integers, so a bincount does it without sorting."""
    if not len(track):
        return track, 0
    present = np.bincount(track) > 0
    new = np.cumsum(present) - 1
    return new[track], int(present.sum())


def rows_to_matrix(rows: Rows) -> np.ndarray:
    """(tracks, T) float matrix of the tracks present in `rows`, NaN where absent (for heatmaps)."""
    row, n = _dense_ids(rows.track)
    m = np.full((n, rows.T), np.nan)
    m[row, rows.t] = rows.value
    return m


def metric_matrix(results, exp: dict, metric: dict, filtered: bool = True) -> np.ndarray:
    """(tracks, T) values of a metric spec, NaN where a track is absent. Only tracks with a value appear."""
    return rows_to_matrix(metric_rows(results, exp, metric, filtered))


def metric_label(exp: dict, metric: dict) -> str:
    from ..metrics_info import label as plain

    def part(measure, channel):
        return f"{channel} {plain(measure).lower()}" if measure in CHANNELS else plain(measure)
    label = part(metric["measure"], metric["channel"])
    if metric["over_measure"]:
        label += f" / {part(metric['over_measure'], metric['over_channel'] or metric['channel'])}"
    if metric["log2"]:
        label = f"log2 {label}"
    if metric["smooth"] > 1:
        label += f" (mean of {metric['smooth']} frames)"
    return label


def _concat(parts: list[Rows]) -> Rows:
    """Pool rows of several positions (tracks renumbered so they stay distinct)."""
    if not parts:
        return Rows(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), 1)
    if len(parts) == 1:
        return parts[0]
    offsets = np.cumsum([0] + [int(p.track.max()) + 1 if len(p.track) else 0 for p in parts[:-1]])
    return Rows(np.concatenate([p.t for p in parts]), np.concatenate([p.track + o for p, o in zip(parts, offsets)]),
                np.concatenate([p.value for p in parts]), max(p.T for p in parts))


def _in_window(rows: Rows, window: slice) -> Rows:
    """Rows of one position inside a frame window. Rows are stored in frame order, so this is a binary search."""
    if window.start == 0 and window.stop >= rows.T:   # the whole movie: nothing to cut
        return Rows(rows.t, rows.track, rows.value, window.stop)
    a, b = np.searchsorted(rows.t, [window.start, window.stop])
    return Rows(rows.t[a:b] - window.start, rows.track[a:b], rows.value[a:b], window.stop - window.start)


def _heatmap_image(rows: Rows, max_rows: int) -> tuple[np.ndarray, int, int]:
    """The heatmap straight from rows: tracks sorted by their mean over the last quarter of the window
    (tracks absent then first), then averaged in groups of consecutive tracks so at most `max_rows`
    remain. Returns (image (rows, T), tracks, group size). No full (tracks x T) matrix is built."""
    ids, n = _dense_ids(rows.track)
    late = rows.t >= rows.T - max(1, rows.T // 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        key = np.bincount(ids[late], weights=rows.value[late], minlength=n) / np.bincount(ids[late], minlength=n)
    rank = np.empty(n, np.int64)
    rank[np.argsort(np.nan_to_num(key, nan=-np.inf), kind="stable")] = np.arange(n)
    group = int(np.ceil(n / max_rows)) if n > max_rows else 1
    bins = (n + group - 1) // group
    cell = (rank[ids] // group) * rows.T + rows.t
    with np.errstate(invalid="ignore", divide="ignore"):
        image = np.bincount(cell, weights=rows.value, minlength=bins * rows.T) / np.bincount(cell, minlength=bins * rows.T)
    return image.reshape(bins, rows.T), n, group


def population_bins(values: np.ndarray, log2: bool, width: float, low=None, high=None) -> np.ndarray:
    """Value bin edges for the population heatmap, shared by every condition of a figure.
    Range: the 1st to 99th percentile of all values (as the notebook did) unless given; width 0 means
    0.1 for log2 metrics (the notebook's bins) and 1/60 of the range otherwise. Edges are multiples of
    the width, so the same settings give the same bins in every figure."""
    if values.size > 1_000_000:  # a fixed, even subsample: same percentiles, a fraction of the time
        values = values[:: values.size // 1_000_000]
    p1, p99 = np.percentile(values, [1, 99]) if values.size else (0.0, 1.0)
    low = p1 if low is None else low
    high = p99 if high is None else high
    if not high > low:
        high = low + 1.0
    width = width or (0.1 if log2 else (high - low) / 60)
    width = max(width, (high - low) / 400)  # at most 400 bins
    start, stop = np.floor(low / width) * width, np.ceil(high / width) * width
    return start + width * np.arange(int(round((stop - start) / width)) + 1)


def population_image(rows: Rows, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bins, T) fraction of the cells in each frame whose value is in each bin: every column sums to 1
    (frames with no cells in range are NaN). Values outside the range are left out of that column, as in
    the notebook. Returns (fractions, cells counted per frame, cells outside the range per frame)."""
    import torch
    n_bins, T = len(edges) - 1, rows.T
    dev = tr._sort_device()
    v = torch.as_tensor(np.asarray(rows.value), device=dev).double()
    t = torch.as_tensor(np.asarray(rows.t), device=dev).long()
    idx = torch.floor((v - float(edges[0])) / float(edges[1] - edges[0])).long()
    inside = (idx >= 0) & (idx < n_bins)
    counts = torch.bincount(idx[inside] * T + t[inside], minlength=n_bins * T)[:n_bins * T]
    counts = counts.reshape(n_bins, T).double().cpu().numpy()
    total = counts.sum(axis=0)
    outside = torch.bincount(t[~inside], minlength=T)[:T].cpu().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return counts / total, total.astype(np.int64), outside


def _frame_quantiles(rows: Rows, qs) -> np.ndarray:
    return tr.grouped_quantiles(rows.t, rows.value, rows.T, qs)


def _over_cells(rows: Rows, band: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Centre and band per frame over the pooled cells of `rows`."""
    if B.uses_median(band):
        lo, mid, hi = _frame_quantiles(rows, list(B.quantiles(band)))
        return (mid, None, None) if band == "none" else (mid, lo, hi)
    T = rows.T
    n = np.bincount(rows.t, minlength=T)[:T].astype(float)
    s1 = np.bincount(rows.t, weights=rows.value, minlength=T)[:T]
    s2 = np.bincount(rows.t, weights=rows.value.astype(float) ** 2, minlength=T)[:T]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s1 / n, np.nan)
        sd = np.sqrt(np.maximum(s2 - n * mean ** 2, 0) / (n - 1))
    lo, hi = B.from_moments(n, mean, sd, band)
    return mean, lo, hi


def _band_series(xs, centre, lo, hi, n: str) -> dict:
    out = {"x": xs, "y": centre, "n": n}
    if lo is not None:
        out.update(lo=lo, hi=hi)
    return out


def _band_table(table: dict, name: str, band: str, centre, lo, hi) -> None:
    c, l, h = B.table_columns(band)
    table[f"{name} {c}"] = list(centre)
    if lo is not None:
        table[f"{name} {l}"], table[f"{name} {h}"] = list(lo), list(hi)


def _pseudo_note(figure, band: str, unit: str) -> None:
    if unit == "cell" and band in ("sem", "ci95"):
        _note(figure, "s.e.m. and CI over pooled cells treat cells as independent, which cells of one well are not; "
                      "use wells or positions as replicates for a real uncertainty")


def _per_cell_means(rows: Rows) -> np.ndarray:
    inv, n = _dense_ids(rows.track)
    return np.bincount(inv, weights=rows.value, minlength=n) / np.bincount(inv, minlength=n)


def _percent_passing(rows: Rows, gate: dict) -> tuple[np.ndarray, np.ndarray]:
    present = np.bincount(rows.t, minlength=rows.T)[:rows.T]
    passed = rows.value > gate["threshold"] if gate["above"] else rows.value < gate["threshold"]
    high = np.bincount(rows.t, weights=passed, minlength=rows.T)[:rows.T]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(present > 0, 100 * high / np.maximum(present, 1), np.nan), present


# ---------------------------------------------------------------------- time axis
def _axis(exp: dict, spec: dict, T: int) -> tuple[np.ndarray, str, float]:
    """x positions of result frames 0..T-1, the axis label, and the movie frame at x = 0."""
    files, time = exp["files"], exp["time"]
    movie = files["first_frame"] + np.arange(T) * files["stride"]
    zero = 0.0
    if spec["time_zero_event"]:
        events = {e["name"]: e["frame"] for e in time["events"]}
        zero = float(events.get(spec["time_zero_event"], 0))
    if time["frames_per_hour"] > 0:
        unit = "hours" + (f" after {spec['time_zero_event']}" if spec["time_zero_event"] else "")
        return (movie - zero) / time["frames_per_hour"], unit, zero
    return movie - zero, "frame" + (f" after {spec['time_zero_event']}" if spec["time_zero_event"] else ""), zero


def _to_x(exp: dict, frame: float, zero: float) -> float:
    fph = exp["time"]["frames_per_hour"]
    return (frame - zero) / fph if fph > 0 else frame - zero


def _window(exp: dict, spec: dict, T: int) -> slice:
    """The spec's movie-frame window as result-frame indices."""
    files = exp["files"]
    start, stop = spec["frames"]
    if start < 0:       # the last frames of the movie, whatever its length
        return slice(max(0, T - int(np.ceil(-start / max(1, files["stride"])))), T)
    a = max(0, int((start - files["first_frame"]) // files["stride"]))
    b = T if not stop else max(a + 1, int((stop - files["first_frame"]) // files["stride"]))
    return slice(a, min(b, T))


def _events(ax, exp: dict, spec: dict, zero: float, label: bool = True) -> None:
    if not spec["show_events"]:
        return
    import matplotlib
    for event in exp["time"]["events"]:
        x = _to_x(exp, event["frame"], zero)
        ax.axvline(x, color="#777777", lw=matplotlib.rcParams["axes.linewidth"], ls="--", zorder=0)
        if label:
            ax.annotate(event["name"], (x, 1), xycoords=("data", "axes fraction"), xytext=(2, -2), textcoords="offset points",
                        va="top", fontsize="small", color="#444444")


def _height_inches(figure) -> float:
    """Height of a Figure or of a panel (SubFigure) in inches."""
    if hasattr(figure, "get_size_inches"):
        return float(figure.get_size_inches()[1])
    root = figure.figure
    return float(root.get_size_inches()[1]) * max(figure.bbox_relative.height, 0.05)


def _auto_columns(n: int) -> int:
    """Small multiples per row: one row up to 4, then a near-square grid that leans wide."""
    if n <= 4:
        return n
    return min(n, int(np.ceil(np.sqrt(n * 1.5))))


def _draw_series(figure, exp: dict, spec: dict, series: dict, styles: dict, ylabel: str, unit: str, zero: float,
                 ylim=None, markers: bool = False) -> list:
    """Lines (with optional bands) per condition, as the figure's comparison mode asks:
    overlay   every condition on one axes;
    reference one small panel per condition, drawn in colour over the reference condition in grey;
    separate  one small panel per condition, alone. Small multiples share both axes."""
    import matplotlib
    from matplotlib.lines import Line2D
    names = list(series)
    ref = spec["reference"] if spec["reference"] in names else names[0]
    mode = spec["compare"]
    ylabel = _label_and_note(figure, ylabel)

    def line(ax, name, color=None, ls=None, alpha=0.2, label=None, z=2):
        s, st = series[name], styles[name]
        c = color or st["color"]
        if s.get("lo") is not None:
            ax.fill_between(s["x"], s["lo"], s["hi"], color=c, alpha=alpha, lw=0, zorder=z - 1)
        return ax.plot(s["x"], s["y"], color=c, ls=ls or st.get("ls", "-"), label=label, zorder=z,
                       marker="o" if markers else None, ms=2.0, mew=0)[0]

    def tag(name):
        return f"{name} ({series[name]['n']})" if series[name].get("n") else name

    if mode == "overlay" or len(names) == 1:
        ax = figure.add_subplot()
        for name in names:
            line(ax, name, label=tag(name), z=3 if name != ref else 2)
        ax.set_ylabel(_fit(ylabel, 0.8 * _height_inches(figure) * 72))
        _finish(ax, exp, spec, unit, zero)
        if ylim:
            ax.set_ylim(*ylim)
        return [ax]
    shown = [n for n in names if n != ref] if mode == "reference" else names
    k = len(shown)
    cols = max(1, min(k, int(spec["columns"]) or _auto_columns(k)))
    rows = int(np.ceil(k / cols))
    grid = figure.subplots(rows, cols, sharex=True, sharey=True, squeeze=False)
    flat = list(grid.ravel())
    for i, name in enumerate(shown):
        ax = flat[i]
        if mode == "reference":
            line(ax, ref, color=pstyle.CONTEXT, ls="-", alpha=0.35, z=1)
        line(ax, name, z=3)
        ax.set_title(tag(name), loc="left")
        _events(ax, exp, spec, zero, label=i == 0)
        if ylim:
            ax.set_ylim(*ylim)
    for ax in flat[k:]:
        ax.set_visible(False)
    for i, ax in enumerate(flat[:k]):
        ax.tick_params(labelbottom=i + cols >= k, labelleft=i % cols == 0)
    if mode == "reference":     # the reference's key goes in an empty grid slot when there is one
        handle = Line2D([], [], color=pstyle.CONTEXT, lw=matplotlib.rcParams["lines.linewidth"] * 2, label=f"{tag(ref)}\n(reference, grey)")
        if len(flat) > k:
            spare = flat[k]
            spare.set_visible(True)
            spare.set_axis_off()
            spare.legend(handles=[handle], loc="center left")
        else:
            figure.legend(handles=[handle], loc="outside upper right")
    figure.supylabel(_fit(ylabel, 0.8 * _height_inches(figure) * 72))
    figure.supxlabel(unit)
    return flat[:k]


def _fit(label: str, length_pt: float) -> str:
    """Wrap a label onto lines no longer than `length_pt` (e.g. a y label along a short panel)."""
    import textwrap
    import matplotlib
    size = matplotlib.font_manager.FontProperties(size=matplotlib.rcParams["axes.labelsize"]).get_size_in_points()
    chars = max(12, int(length_pt / (0.55 * size)))
    return "\n".join(textwrap.wrap(label, chars)) if len(label) > chars else label


# ---------------------------------------------------------------------- drawing
def figure_conditions(exp: dict, spec: dict) -> list[dict]:
    """The conditions a figure draws, in order: its ticked list, else its group, else all."""
    if spec.get("conditions"):
        by_name = {c["name"]: c for c in exp["plate"]["conditions"]}
        return [by_name[n] for n in spec["conditions"] if n in by_name]
    return group_conditions(exp, spec["group"])


def group_conditions(exp: dict, group_name: str) -> list[dict]:
    conditions = {c["name"]: c for c in exp["plate"]["conditions"]}
    if not group_name:
        return list(conditions.values())
    for group in exp["analysis"]["groups"]:
        if group["name"] == group_name:
            return [conditions[c] for c in group["conditions"] if c in conditions]
    raise KeyError(f"no analysis group called {group_name!r}")


def _gate(exp: dict, name: str) -> dict:
    for gate in exp["analysis"]["gates"]:
        if gate["name"] == name:
            return gate
    raise KeyError(f"no gate called {name!r}")


def render(exp: dict, spec: dict, figure=None, cache: ResultsCache | None = None, styled: bool = True) -> tuple[Figure, dict]:
    """Draw one figure. Returns the figure and the numbers it shows, as {column: values} for CSV.
    `figure` may be a matplotlib SubFigure (a panel of an exported figure); `styled=False` leaves the type and
    line settings to the caller (an export sets journal sizes for every panel at once)."""
    exp = normalise(exp)
    spec = normalise({"figures": {"list": [spec]}})["figures"]["list"][0]
    plot = spec["plot"]
    figs = exp["figures"]
    if figure is None:
        size = (plot["width"] or figs["width"], plot["height"] or figs["height"])
        figure = Figure(figsize=size, layout="constrained")
    ctx = pstyle.context("screen", font=plot["font_size"], line=plot["line_width"]) if styled else _nothing()
    with ctx:
        figure, table = _render(exp, spec, figure, cache or ResultsCache(exp))
        _apply_plot(figure, spec)
    return figure, table


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def is_empty(figure) -> bool:
    """True for the 'no results yet' placeholder, which is shown but never saved."""
    return bool(getattr(figure, "_fasttrack_empty", False))


def plot_font(spec: dict) -> float:
    return float(spec.get("plot", {}).get("font_size", 9))


def _apply_plot(figure: Figure, spec: dict) -> None:
    """Titles, labels, limits, scale, legend and grid from spec["plot"], after drawing."""
    plot = spec["plot"]
    axes = [ax for ax in figure.axes if ax.get_label() != "<colorbar>" and ax.get_visible()]
    if not axes:
        return
    if plot["title"]:
        figure.suptitle(plot["title"])
    heatmap = spec["kind"] in ("heatmap", "track_heatmap")
    for ax in axes:
        if not heatmap:
            if plot["log_y"]:
                ax.set_yscale("log")
            if plot["vmin"] is not None or plot["vmax"] is not None:
                ax.set_ylim(plot["vmin"], plot["vmax"])
        if plot["xmin"] is not None or plot["xmax"] is not None:
            ax.set_xlim(plot["xmin"], plot["xmax"])
        if plot["grid"]:
            ax.grid(True, alpha=0.3, lw=0.6)
        legend = ax.get_legend()
        if legend is not None:
            if plot["legend"] == "none":
                legend.remove()
            elif plot["legend"] == "direct":
                legend.remove()
                ax.get_xlim(), ax.get_ylim()     # settle the limits first
                pstyle.direct_labels(ax, [(ln.get_label(), ln, ln.get_color()) for ln in ax.get_lines()
                                         if ln.get_label() and not ln.get_label().startswith("_")])
            else:   # keep the legend's own entries (some are proxies, e.g. the reference in small multiples)
                handles, labels = legend.legend_handles, [t.get_text() for t in legend.get_texts()]
                if plot["legend"] == "outside":
                    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
                else:
                    ax.legend(handles, labels, loc=plot["legend"], frameon=False)
    if plot["ylabel"]:
        if getattr(figure, "_supylabel", None) is not None:
            figure.supylabel(plot["ylabel"])
        else:
            for ax in (axes if not heatmap else axes[len(axes) // 2: len(axes) // 2 + 1]):
                ax.set_ylabel(plot["ylabel"])
    if plot["xlabel"]:
        if getattr(figure, "_supxlabel", None) is not None:
            figure.supxlabel(plot["xlabel"])
        else:
            axes[-1].set_xlabel(plot["xlabel"])


def _render(exp: dict, spec: dict, figure: Figure, cache: "ResultsCache") -> tuple[Figure, dict]:
    figure.clear()
    figure._fasttrack_notes, figure._fasttrack_empty = [], False
    conditions = figure_conditions(exp, spec)
    styles = pstyle.condition_styles(exp, [c["name"] for c in conditions], spec["palette"] or None, spec["reference"] or None)
    per_condition: dict[str, dict] = {}
    for cond in conditions:
        keys = cache.timestacks_of(cond)
        per_condition[cond["name"]] = {"keys": keys, **styles[cond["name"]]}
    if not any(v["keys"] for v in per_condition.values()):
        ax = figure.add_subplot()
        ax.text(0.5, 0.5, "No results for these conditions yet.\nRun tracking first, or check the plate layout.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        figure._fasttrack_empty = True
        return figure, {}
    kind = spec["kind"]
    table: dict[str, list] = {}

    if kind == "event_rate":
        return _event_rate(exp, spec, figure, per_condition, cache)

    if kind == "cell_count":
        # one count per position and frame; replicates are positions, or wells (the mean of their positions)
        rep_unit = "well" if spec["unit"] == "well" else "position"
        band = B.resolve(spec["band"], rep_unit)
        counts = {}
        for name, v in per_condition.items():
            reps: dict[str, list[np.ndarray]] = {}
            for key in v["keys"]:
                c = frame_counts(cache, key)
                reps.setdefault(key.split("_xy")[0] if rep_unit == "well" else key, []).append(c)
            if reps:
                T = max(len(c) for cs in reps.values() for c in cs)
                pad = lambda c: np.pad(c, (0, T - len(c)), constant_values=np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    counts[name] = np.vstack([np.nanmean(np.vstack([pad(c) for c in cs]), axis=0) for cs in reps.values()])
        T = max(c.shape[1] for c in counts.values())
        x, unit, zero = _axis(exp, spec, T)
        table["x"] = list(x)
        series = {}
        for name, stack in counts.items():
            centre, lo, hi = B.across(stack, band)
            k = stack.shape[1]
            series[name] = _band_series(x[:k], centre, lo, hi, f"{stack.shape[0]} {rep_unit}s")
            fill = lambda a: None if a is None else np.concatenate([a, np.full(T - k, np.nan)])
            _band_table(table, name, band, fill(centre), fill(lo), fill(hi))
        _draw_series(figure, exp, spec, series, per_condition,
                     f"Cells per position\n{B.describe(band, rep_unit + 's')}", unit, zero)
        _title(figure, spec["name"])
        return figure, table

    parts: dict[str, list[tuple[str, Rows]]] = {}
    for name, v in per_condition.items():
        if not v["keys"]:
            continue
        metric = _gate(exp, spec["gate"])["metric"] if kind == "percent_high" else spec["metric"]
        parts[name] = [(k, position_rows(cache, k, exp, metric)) for k in v["keys"]]
    T = max((r.T for ps in parts.values() for _, r in ps), default=1)
    x, unit, zero = _axis(exp, spec, T)
    window = _window(exp, spec, T)
    xs = x[window]

    def reps_of(name: str, unit_: str) -> dict[str, Rows]:
        """Replicates of a condition, cut to the window: one per position or per well."""
        groups: dict[str, list[Rows]] = {}
        for key, rows in parts[name]:
            groups.setdefault(key.split("_xy")[0] if unit_ == "well" else key, []).append(rows)
        return {rep: _pooled(rs) for rep, rs in groups.items()}

    def _pooled(rows_list: list[Rows]) -> Rows:
        """Rows of several positions inside the window, pooled (cut first: each position is in frame order)."""
        cut = _concat([_in_window(Rows(*r[:3], T), window) for r in rows_list])
        return Rows(cut.t, cut.track, cut.value, window.stop - window.start)

    band = B.resolve(spec["band"], spec["unit"])

    if kind == "timecourse":
        table["x"] = list(xs)
        series = {}
        for name in parts:
            if spec["unit"] != "cell":
                reps = reps_of(name, spec["unit"])
                medians = np.stack([_frame_quantiles(r, [0.5])[0] for r in reps.values()])
                centre, lo, hi = B.across(medians, band)
                series[name] = _band_series(xs, centre, lo, hi, f"{len(reps)} {spec['unit']}s")
            else:
                pooled = _pooled([r for _, r in parts[name]])
                centre, lo, hi = _over_cells(pooled, band)
                series[name] = _band_series(xs, centre, lo, hi, f"{_dense_ids(pooled.track)[1]} cells")
            _band_table(table, name, band, centre, lo, hi)
        over = "cells" if spec["unit"] == "cell" else f"{spec['unit']} medians"
        _pseudo_note(figure, band, spec["unit"])
        _draw_series(figure, exp, spec, series, per_condition,
                     metric_label(exp, spec["metric"]) + "\n" + B.describe(band, over), unit, zero)

    elif kind == "percent_high":
        gate = _gate(exp, spec["gate"])
        table["x"] = list(xs)
        series = {}
        for name in parts:
            if spec["unit"] != "cell":
                reps = reps_of(name, spec["unit"])
                pcts = np.stack([_percent_passing(r, gate)[0] for r in reps.values()])
                centre, lo, hi = B.across(pcts, band)
                series[name] = _band_series(xs, centre, lo, hi, f"{len(reps)} {spec['unit']}s")
                _band_table(table, name + " %", band, centre, lo, hi)
                continue
            pct, present = _percent_passing(_pooled([r for _, r in parts[name]]), gate)
            lo, hi = B.proportion(pct * present / 100, present, band)
            series[name] = _band_series(xs, pct, lo, hi, "")
            table[f"{name} %"], table[f"{name} cells"] = list(pct), list(present)
            if lo is not None:
                table[f"{name} ci95 low"], table[f"{name} ci95 high"] = list(lo), list(hi)
        side = ">" if gate["above"] else "<"
        if spec["unit"] == "cell":
            what = "% of pooled cells" + (", 95% Wilson interval" if band == "ci95" else "")
        else:
            what = B.describe(band, f"{spec['unit']} percentages")
        _draw_series(figure, exp, spec, series, per_condition,
                     f"% cells {gate['name']}\n({metric_label(exp, gate['metric'])} {side} {gate['threshold']:g})\n{what}",
                     unit, zero, ylim=(0, 100))

    elif kind == "heatmap":
        # the population over time: x time, y binned value, colour the fraction of that frame's cells in the bin
        shown = {name: _pooled([r for _, r in parts[name]]) for name in parts}
        shown = {name: rows for name, rows in shown.items() if len(rows.value)}
        edges = population_bins(np.concatenate([r.value for r in shown.values()]), bool(spec["metric"].get("log2")),
                                float(spec.get("bin_width") or 0), spec["plot"]["vmin"], spec["plot"]["vmax"])
        images = {name: population_image(rows, edges) for name, rows in shown.items()}
        top = max((np.nanpercentile(f, 99.5) for f, _, _ in images.values() if np.isfinite(f).any()), default=1.0)
        axes = figure.subplots(len(shown), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
        image = None
        gate = next((g for g in exp["analysis"]["gates"] if g["name"] == spec["gate"]), None) if spec["gate"] else None
        for ax, (name, (fractions, counted, outside)) in zip(axes, images.items()):
            image = ax.imshow(fractions, aspect="auto", interpolation="nearest", origin="lower", cmap="magma", vmin=0, vmax=top,
                              extent=(xs[0], xs[-1], edges[0], edges[-1]))
            ax.set_ylabel(name)
            ax.text(0.01, 0.97, f"up to {int(counted.max())} cells per frame", transform=ax.transAxes, ha="left", va="top",
                    fontsize="small", color="white")
            if gate is not None and _canonical(gate["metric"]) == _canonical(spec["metric"]):  # the threshold, as the notebook drew it
                ax.axhline(gate["threshold"], color="white", ls="--", lw=0.8, alpha=0.6)
            _events(ax, exp, spec, zero)
            share_out = outside.sum() / max(outside.sum() + counted.sum(), 1)
            if share_out > 0.05:
                ax.text(0.99, 0.97, f"{100 * share_out:.0f}% of values outside the range", transform=ax.transAxes,
                        ha="right", va="top", fontsize="small", color="white")
            table[f"{name} cells counted"] = list(counted)
            for b in range(len(edges) - 1):
                table[f"{name} [{edges[b]:.4g}, {edges[b + 1]:.4g})"] = list(fractions[b])
        table = {unit: list(xs), **table}
        axes[-1].set_xlabel(unit)
        figure.colorbar(image, ax=list(axes), shrink=0.8, aspect=30, label="fraction of cells in the frame")
        figure.supylabel(metric_label(exp, spec["metric"]))

    elif kind == "track_heatmap":
        shown = {}
        for name in parts:
            pooled = _pooled([r for _, r in parts[name]])
            if len(pooled.value):
                shown[name] = pooled
        everything = np.concatenate([r.value for r in shown.values()])
        if everything.size > 1_000_000:  # colour limits from a fixed, even subsample: same limits, a fraction of the time
            everything = everything[:: everything.size // 1_000_000]
        low, high = np.percentile(everything, [2, 98])
        if spec["plot"]["vmin"] is not None:
            low = spec["plot"]["vmin"]
        if spec["plot"]["vmax"] is not None:
            high = spec["plot"]["vmax"]
        axes = figure.subplots(len(shown), 1, sharex=True, squeeze=False)[:, 0]
        image = None
        # two rows per pixel of panel height is all a screen or print can show
        panel_px = _height_inches(figure) * figure.dpi / max(len(shown), 1)
        max_rows = int(min(HEATMAP_ROWS, max(100, 2 * panel_px)))
        for ax, (name, rows) in zip(axes, shown.items()):
            picture, cells, group = _heatmap_image(rows, max_rows)
            image = ax.imshow(picture, aspect="auto", interpolation="nearest", cmap="magma", vmin=low, vmax=high,
                              extent=(xs[0], xs[-1], cells, 0))
            note = f"\nrows: means of {group}" if group > 1 else ""
            ax.set_ylabel(f"{name}\n{cells} cells{note}")
            ax.set_yticks([])
            _events(ax, exp, spec, zero)
        axes[-1].set_xlabel(unit)
        figure.colorbar(image, ax=list(axes), shrink=0.8, aspect=30, label=metric_label(exp, spec["metric"]))
        table = {}  # per-cell matrices are too large for a CSV; the results files hold them

    elif kind == "distribution":
        ax = figure.add_subplot()
        names, cells, reps = [], [], []
        for name in parts:
            per_rep = {}
            for rep, rows in reps_of(name, spec["unit"] if spec["unit"] != "cell" else "position").items():
                v = _per_cell_means(rows)
                per_rep[rep] = v[np.isfinite(v)]
            names.append(name)
            cells.append(np.concatenate(list(per_rep.values())) if per_rep else np.zeros(0))
            reps.append({r: float(np.median(v)) for r, v in per_rep.items() if len(v)})
        rep_unit = spec["unit"]
        values = cells if rep_unit == "cell" else [np.array(list(r.values())) for r in reps]
        colors = [per_condition[n]["color"] for n in names]
        positions = np.arange(len(names))
        style = spec["style"]
        if style == "violin":
            usable = [i for i, v in enumerate(cells) if len(v) > 1]
            if usable:
                from matplotlib.cbook import violin_stats
                stats = violin_stats([cells[i] for i in usable], _binned_kde, points=400)
                body = ax.violin(stats, positions=positions[usable], showmedians=rep_unit == "cell", showextrema=False)
                for part, i in zip(body["bodies"], usable):
                    part.set_facecolor(colors[i])
                    part.set_alpha(0.5 if rep_unit == "cell" else 0.25)
        elif style == "box":
            body = ax.boxplot(values, positions=positions, patch_artist=True, showfliers=False, widths=0.6)
            for patch, color in zip(body["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        else:
            bar_band = B.resolve(spec["band"], rep_unit)
            stats_ = [B.across(v[:, None], bar_band) if len(v) else (np.full(1, np.nan), None, None) for v in values]
            heights = np.array([c[0] for c, _, _ in stats_])
            err = None
            if all(lo is not None for _, lo, _ in stats_):
                err = np.nan_to_num(np.array([[c[0] - lo[0], hi[0] - c[0]] for c, lo, hi in stats_]).T)
            ax.bar(positions, heights, yerr=err, color=colors, alpha=0.8, capsize=3)
            _note(figure, "bars: " + B.describe(bar_band, "cells" if rep_unit == "cell" else f"{rep_unit} values"))
            _pseudo_note(figure, bar_band, rep_unit)
        if rep_unit != "cell":  # every replicate as a dot
            rng = np.random.default_rng(0)
            for i, v in enumerate(values):
                ax.scatter(positions[i] + rng.uniform(-0.12, 0.12, len(v)), v, s=6, color="black", zorder=3, lw=0)
        for i, name in enumerate(names):
            table[f"{name} ({rep_unit}s)"] = list(values[i])
        if rep_unit != "cell":
            for i, name in enumerate(names):
                table[f"{name} {rep_unit} ids"] = list(reps[i])
        if spec["stats"] and len(names) > 1:
            tests = compare_to_reference(names, values, rep_unit)
            for row in tests:  # at the top of the axes, so axis limits never hide them
                i = names.index(row["condition"])
                ax.annotate(_stars(row["p_holm"]), (positions[i], 0.98), xycoords=("data", "axes fraction"), ha="center", va="top")
            table["test"] = [t["test"] for t in tests]
            table["condition"] = [t["condition"] for t in tests]
            table["reference"] = [t["reference"] for t in tests]
            table["n condition"] = [t["n"] for t in tests]
            table["n reference"] = [t["n_reference"] for t in tests]
            table["p"] = [t["p"] for t in tests]
            table["p (Holm)"] = [t["p_holm"] for t in tests]
        ax.set_xticks(positions, [f"{n}\n({len(v)})" for n, v in zip(names, values)], rotation=30 if len(names) > 4 else 0,
                      ha="right" if len(names) > 4 else "center")
        start, stop = xs[0], xs[-1]
        what = "per-cell mean" if rep_unit == "cell" else f"median of per-cell means per {rep_unit}"
        ax.set_ylabel(_label_and_note(figure, metric_label(exp, spec["metric"]) + f"\n{what}, {unit} {start:g} to {stop:g}"))
        if spec["stats"] and len(names) > 1:
            test_name = "Mann-Whitney on cells" if rep_unit == "cell" else f"Welch t-test on {rep_unit}s"
            _note(figure, f"vs {names[0]}: {test_name}, Holm-adjusted; * p < 0.05, ** p < 0.01, *** p < 0.001")
        shown_values = np.concatenate([v for v in (cells if rep_unit == "cell" or style == "violin" else values) if len(v)] or [np.zeros(1)])
        if style == "bar":
            if np.nanmin(np.concatenate([v for v in values if len(v)] or [np.zeros(1)])) >= 0:
                ax.set_ylim(bottom=0)      # bars of positive amounts start at 0
        elif shown_values.size > 20:   # robust limits: a few extreme cells must not flatten every violin
            lo, hi = np.percentile(shown_values, [0.5, 99.5])
            if rep_unit != "cell":
                reps_all = np.concatenate([v for v in values if len(v)] or [np.zeros(1)])
                lo, hi = min(lo, reps_all.min()), max(hi, reps_all.max())
            pad = 0.05 * (hi - lo or 1.0)
            ax.set_ylim(lo - pad, hi + pad)
            clipped = float(np.mean((shown_values < lo) | (shown_values > hi)))
            if clipped > 0:
                _note(figure, f"axis shows the central 99% of cells ({100 * clipped:.1f}% outside)")
        ax.spines[["top", "right"]].set_visible(False)
    else:
        raise ValueError(f"unknown figure kind {kind!r}")
    _title(figure, spec["name"])
    return figure, table


def _event_rate(exp: dict, spec: dict, figure: Figure, per_condition: dict, cache: ResultsCache) -> tuple[Figure, dict]:
    """Events of one category (from Events/<id>.csv) per 100 cells per frame, in bins of frames."""
    from ..events import read_events
    if not spec["event"]:
        raise ValueError("Choose the event category to count.")
    table: dict[str, list] = {}
    width = max(1, int(spec["bin"]))
    band = B.resolve(spec["band"], spec["unit"])
    curves, any_events = {}, False
    for name, v in per_condition.items():
        reps: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for key in v["keys"]:
            r = cache.get(key)
            T = int(r["meta"]["frames"])
            cells = np.bincount(r.column("t").astype(int), minlength=T)[:T].astype(float)
            rows = [e for e in read_events(cache.results_dir, key) if e["category"] == spec["event"]]
            any_events |= bool(read_events(cache.results_dir, key))
            counts = np.bincount([e["t"] for e in rows], minlength=T)[:T].astype(float)
            rep = key.split("_xy")[0] if spec["unit"] == "well" else key
            reps.setdefault(rep, []).append((counts, cells))
        if not reps:
            continue
        T = max(len(c) for parts in reps.values() for c, _ in parts)
        nb = (T + width - 1) // width

        def binned(a):
            a = np.pad(a, (0, nb * width - len(a)))
            return a.reshape(nb, width).sum(1)

        per_rep = []
        pooled_events, pooled_cells = np.zeros(nb), np.zeros(nb)
        for parts in reps.values():
            e = sum(binned(c) for c, _ in parts)
            n = sum(binned(c) for _, c in parts)
            pooled_events += e
            pooled_cells += n
            with np.errstate(invalid="ignore", divide="ignore"):
                per_rep.append(np.where(n > 0, 100 * e / n, np.nan))
        if spec["unit"] == "cell":
            with np.errstate(invalid="ignore", divide="ignore"):
                mean, lo, hi = np.where(pooled_cells > 0, 100 * pooled_events / pooled_cells, np.nan), None, None
        else:
            mean, lo, hi = B.across(np.stack(per_rep), band)
        curves[name] = (mean, lo, hi, len(reps))
    if not any_events:
        raise ValueError("No classified events yet: train the classifier on the Labelling tab and classify the positions.")
    T = max(len(c[0]) for c in curves.values()) * width
    x, unit, zero = _axis(exp, spec, T)
    centres = x[np.minimum(np.arange(len(next(iter(curves.values()))[0])) * width + width // 2, T - 1)]
    table["x"] = list(centres)
    series = {}
    for name, (mean, lo, hi, n) in curves.items():
        series[name] = _band_series(centres[:len(mean)], mean, lo, hi, f"{n} {spec['unit']}s" if spec["unit"] != "cell" else "")
        _band_table(table, name + " %", band, mean, lo, hi)
    what = "pooled cells" if spec["unit"] == "cell" else B.describe(band, f"{spec['unit']}s")
    _draw_series(figure, exp, spec, series, per_condition,
                 f"{spec['event']} events per 100 cells per frame\n(bins of {width} frames; {what})", unit, zero, markers=True)
    _title(figure, spec["name"])
    return figure, table


def _binned_kde(data: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Gaussian KDE with Scott's bandwidth (as matplotlib's violins), from a 2048-bin histogram:
    the same curve to the eye, in milliseconds instead of seconds for 100,000s of cells."""
    data = np.asarray(data, dtype=np.float64)
    bw = data.std(ddof=1) * len(data) ** (-1 / 5) if len(data) > 1 else 1.0
    if not np.isfinite(bw) or bw <= 0:
        bw = 1e-3
    lo, hi = data.min() - 4 * bw, data.max() + 4 * bw
    counts, edges = np.histogram(data, bins=2048, range=(lo, hi))
    width = edges[1] - edges[0]
    half = min(int(np.ceil(4 * bw / width)), len(counts) // 2 - 1)   # the kernel never outgrows the histogram
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * width / bw) ** 2)
    density = np.convolve(counts, kernel, mode="same") / (len(data) * bw * np.sqrt(2 * np.pi))
    return np.interp(coords, (edges[:-1] + edges[1:]) / 2, density, left=0, right=0)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order. NaN stays NaN."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.nonzero(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        out[i] = running
    return out.tolist()


def compare_to_reference(names: list[str], values: list[np.ndarray], unit: str) -> list[dict]:
    """Each condition against the first. Cells: Mann-Whitney U. Replicates: Welch's t-test. Holm-adjusted."""
    from scipy import stats
    ref = values[0]
    rows = []
    for name, v in zip(names[1:], values[1:]):
        if unit == "cell":
            test = "Mann-Whitney U"
            p = stats.mannwhitneyu(v, ref).pvalue if len(v) and len(ref) else np.nan
        else:
            test = "Welch t-test"
            p = stats.ttest_ind(v, ref, equal_var=False).pvalue if len(v) > 1 and len(ref) > 1 else np.nan
        rows.append({"condition": name, "reference": names[0], "test": test, "n": len(v), "n_reference": len(ref), "p": float(p)})
    for row, adj in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = adj
    return rows


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _label_and_note(figure, label: str) -> str:
    """An axis label names the quantity only; what the line or band is (its second line) goes to the caption."""
    first, _, rest = label.partition("\n")
    if rest.strip():
        _note(figure, rest.strip().strip("()"))
    return first


def _note(figure, text: str) -> None:
    notes = getattr(figure, "_fasttrack_notes", None)
    if notes is None:
        notes = figure._fasttrack_notes = []
    if text not in notes:
        notes.append(text)


def _title(figure, name: str) -> None:
    """The figure's name, with the notes (what lines, bands and tests are) in a second, smaller line, wrapped
    to the figure's width. On an exported page both move to the caption."""
    import textwrap
    notes = getattr(figure, "_fasttrack_notes", None) or []
    if not name and not notes:
        return
    import matplotlib
    width_pt = _width_inches(figure) * 72
    size = matplotlib.font_manager.FontProperties(size="small").get_size_in_points()
    text = "; ".join(notes)
    wrapped = "\n".join(textwrap.wrap(text, max(30, int(width_pt / (0.52 * size))))) if text else ""
    figure.suptitle(name + ("\n" if name and wrapped else "") + wrapped, fontsize="medium" if not wrapped else "small")


def _width_inches(figure) -> float:
    if hasattr(figure, "get_size_inches"):
        return float(figure.get_size_inches()[0])
    return float(figure.figure.get_size_inches()[0]) * max(figure.bbox_relative.width, 0.05)


def _finish(ax, exp: dict, spec: dict, unit: str, zero: float) -> None:
    _events(ax, exp, spec, zero)
    ax.set_xlabel(unit)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "figure"


def write_figure(exp: dict, spec: dict, folder: Path, cache: "ResultsCache", on_event=lambda e: None,
                 name: str | None = None) -> list[Path]:
    """Draw one figure into `folder` in every format of the experiment, with the numbers as CSV."""
    name = safe_name(name or spec["name"] or "figure")
    try:
        figure, table = render(exp, spec, cache=cache)  # size from the figure's plot settings
    except (KeyError, ValueError) as exc:
        on_event({"kind": "figure_failed", "name": name, "message": str(exc)})
        return []
    if is_empty(figure):
        on_event({"kind": "figure_failed", "name": name, "message": "no results for these conditions yet"})
        return []
    written = save_figure(exp, spec, figure, table, folder, name)
    on_event({"kind": "figure_done", "name": name})
    return written


def save_figure(exp: dict, spec: dict, figure, table: dict, folder: Path, name: str) -> list[Path]:
    """Write a drawn figure in every format of the experiment, and its numbers as CSV."""
    name = safe_name(name)
    written = []
    with pstyle.context("screen", font=spec["plot"]["font_size"], line=spec["plot"]["line_width"]):
        for fmt in exp["figures"]["formats"]:      # inside the style: fonts embed as TrueType
            path = folder / f"{name}.{fmt}"
            figure.savefig(path, dpi=exp["figures"]["dpi"])
            written.append(path)
    if table:
        path = folder / f"{name}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            columns = list(table)
            writer.writerow(columns)
            length = max(len(v) for v in table.values())
            for row in range(length):
                writer.writerow([table[c][row] if row < len(table[c]) else "" for c in columns])
        written.append(path)
    return written


def render_all(exp: dict, on_event=lambda e: None, summary: bool = True) -> list[Path]:
    """Write every enabled figure in the experiment file, with its numbers as CSV; then the summary figures."""
    exp = normalise(exp)
    folder = Path(exp["paths"]["results"]) / exp["figures"]["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    save_experiment(exp, folder / "experiment.yaml")
    cache = ResultsCache(exp)
    warm(exp, cache)       # compute each metric once for the whole plate, and keep it for next time
    written = []
    for i, spec in enumerate(exp["figures"]["list"]):
        if spec["enabled"]:
            written += write_figure(exp, spec, folder, cache, on_event, name=spec["name"] or f"figure_{i + 1}")
    if summary:
        from .summary import render_summary
        written += render_summary(exp, on_event, cache)
    return written
