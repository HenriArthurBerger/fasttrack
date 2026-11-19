"""The experiment file: one YAML holding every setting and decision, from raw TIFFs to figures.

Running the same file again reproduces the same tracks and the same figures. Saving always
writes every key, defaults included, so a later change of a default cannot change an old
analysis. Sections:

    experiment   name and free notes
    paths        raw data folder (only read) and results folder
    files        filename pattern and which frames to use
    channels     one entry per channel in the files: name, role, reporter, colour
    time         optional frames per hour (figure axes only) and events, e.g. a drug addition
    plate        conditions with colours and wells, excluded wells and positions
    quality      the bad-position check: its settings, what it flagged and why, manual overrides
    labelling    event categories, crop settings and sampling for the event labeller and classifier
    pipeline     every tracking parameter (see config.DEFAULTS)
    analysis     track filters, condition groups for figures, gates (thresholds)
    figures      output formats and the list of figures to draw

The tracking code reads a flat config dict; `pipeline_config` builds it from this file.
"""
from __future__ import annotations

import copy
import datetime as _dt
import re
from pathlib import Path

import yaml

from . import __version__
from .config import DEFAULTS as PIPELINE_DEFAULTS
from .config import merge

ROLES = ["nuclear", "measure", "skip"]
FIGURE_KINDS = ["timecourse", "heatmap", "track_heatmap", "percent_high", "distribution", "cell_count", "event_rate"]
DISTRIBUTION_STYLES = ["violin", "box", "bar"]
UNITS = ["cell", "position", "well"]
PALETTE = ["#2d63e5", "#cc3d3d", "#3a9e4f", "#e08a1e", "#8a4fbf", "#1fa3a3", "#c2457a", "#6b6b6b",
           "#9e7b2f", "#4f7fbf", "#bf4f4f", "#5fae5f"]


def pipeline_defaults() -> dict:
    out = {k: copy.deepcopy(v) for k, v in PIPELINE_DEFAULTS.items() if k not in ("data", "nuclear_channel")}
    out["output"].pop("gui_cache", None)  # the app decides this per run
    return out


METRIC_DEFAULT = {
    "measure": "nuc_median",     # a channel metric (nuc_mean, cyto_median, ...) or a cell metric (nuc_area, ...)
    "channel": "",               # channel name; empty for cell metrics
    "over_measure": "",          # set both over_* to divide, e.g. a FRET or KTR ratio
    "over_channel": "",
    "background_percentile": 0.0,  # >0 subtracts this per-frame percentile of the denominator from both terms
    "log2": False,
    "smooth": 0,                 # trailing moving average over this many frames; 0 = off
}

PLOT_DEFAULT = {
    "title": "",                 # empty = the figure's name
    "xlabel": "",                # empty = automatic
    "ylabel": "",
    "vmin": None,                # value axis (heatmap: its value range; cell heatmap: colour range); None = automatic
    "vmax": None,
    "xmin": None,
    "xmax": None,
    "log_y": False,
    "legend": "best",            # best, upper left, upper right, lower left, lower right, outside, none
    "font_size": 9.0,
    "line_width": 1.5,
    "grid": False,
    "width": 0.0,                # inches; 0 = the figures section's default
    "height": 0.0,
}

LEGEND_PLACES = ["best", "direct", "upper left", "upper right", "lower left", "lower right", "outside", "none"]
COMPARE_MODES = ["overlay", "reference", "separate"]  # all on one axes; each condition over the reference in grey; each alone

FIGURE_DEFAULT = {
    "name": "",
    "kind": "timecourse",        # one of FIGURE_KINDS
    "conditions": [],            # the conditions drawn, in order (the first is the reference); empty = use `group`
    "group": "",                 # an analysis group; used when `conditions` is empty; empty = every condition
    "metric": METRIC_DEFAULT,
    "gate": "",                  # percent_high: the gate to count
    "event": "",                 # event_rate: the classifier category to count, e.g. death
    "bin": 10,                   # event_rate: frames per bin
    "bin_width": 0.0,            # heatmap: width of a value bin; 0 = automatic (0.1 for log2 metrics, else 1/60 of the range)
    "style": "violin",           # distribution: violin, box or bar
    "unit": "well",              # replicates: well (positions in a well are not independent), position, or cell (pooled)
    "band": "auto",              # line and band: auto | iqr | range90 | sd | sem | ci95 | none (analysis/bands.py)
    "stats": True,               # distribution: test each condition against the group's first (reference)
    "frames": [0, 0],            # window in movie frames; 0 at the end = last frame; a negative start counts from the end
    "time_zero_event": "",       # draw time relative to this event
    "show_events": True,
    "compare": "overlay",        # time courses: overlay | reference (small multiples, reference in grey) | separate
    "reference": "",             # the reference condition; empty = the first drawn
    "columns": 0,                # small multiples per row; 0 = automatic
    "palette": "",               # condition colours; empty = the figures section's palette (style.PALETTES)
    "enabled": True,
    "plot": PLOT_DEFAULT,        # titles, axis limits and labels, legend, sizes
}

QUALITY_DEFAULTS = {
    "enabled": True,             # check positions before tracking; flagged ones are skipped unless kept
    "checked": False,            # true once a check has run; its result is in `flagged`
    "frames_per_position": 6,    # evenly spaced, first and last included
    "reference_positions": 48,   # random sample that defines normal; all positions if fewer
    "tiles_per_edge": 16,
    "low_quantile": 0.01,        # per-tile background floor
    "mid_quantile": 0.5,         # per-tile median
    "z_threshold": 6.0,          # robust deviations for a tile to count as unusual
    "area_threshold": 0.02,      # share of a frame's tiles that may be unusual
    "level_threshold": 6.0,      # robust deviations of a frame's background level
    "frame_fraction": 0.34,      # share of sampled frames that must be bad to flag the position
    "seed": 0,
    "max_missing_frames": 1,     # flag a position with more missing, unreadable or cut-off frames than this
    "min_cells_per_frame": 5,    # warn (never stop) when a position has fewer cells per frame
    "max_nuclear_coverage": 0.35,  # warn when nuclei cover more of the field than this
    "flagged": {},               # {position: reason}, written by the check
    "keep": [],                  # flagged positions to track anyway (manual override)
}

OLD_CATEGORIES = ["neutral", "division", "death", "abnormal"]   # before division was split into mitosis and daughter

LABELLING_DEFAULTS = {
    # cell states, keys 1-9 in this order; the first is the default. mitosis: rounding up and dividing, before the
    # split (often odd-looking for a few frames); daughter: a new cell in the first frames after the split;
    # touching: nuclei touching or overlapping, where centres are hardest
    "categories": ["neutral", "mitosis", "daughter", "touching", "death", "abnormal"],
    "window": 2,                 # frames shown and used before and after the event frame
    "crop": 48,                  # px, side of the square crop around the cell
    "channels": [],              # channel names in the crops; empty = the nuclear channel
    "sample": {"random": 0.3, "track_end": 0.3, "track_start": 0.15, "division": 0.25},
    "train_steps": 600,
    "seed": 0,
    "labels_file": "Labels/events.csv",       # inside the results folder
    "model_file": "Labels/event_model.pt",
    "centres": {                 # the nucleus centre labeller, which trains the learned seed finder
        "patch": 96,             # px, side of the box in which every nucleus centre is clicked
        "margin": 32,            # px of context shown (and given to the network) around the box
        "count_sigma": 20.0,     # px; local cell counts are taken over about this distance
        "count_window": 5,       # frames; a frame is compared with the median of this many around it
        "min_change": 0.5,       # smallest local count change that makes a candidate (1 = one nucleus)
        "sample": {"count_change": 0.45, "random": 0.2, "track_end": 0.15, "track_start": 0.05, "division": 0.15},
        "context": 7,            # frames shown before and after the labelled frame (3-15), to tell odd shapes from doubles
        "train_steps": 1500,
        "architecture": "resnet18-half",   # seed finder network (track/centroid_net.py ARCHITECTURES; speed: benchmarks/inference.py)
        "match_distance": 6.0,   # px; a found centre this close to a clicked one counts as the same nucleus
        "labels_file": "Labels/centres.json",
        "model_file": "Labels/centre_model.pt",
    },
    "folds": 5,                  # label check: every label judged by a model trained on the other folds
    "review_file": "Labels/review.json",
}

READOUT_DEFAULT = {
    "name": "",                     # what models and figures call it, e.g. ERK
    "kind": "ktr",                  # ktr, nuclear_reporter, intensity, speed, metric (see analysis/readouts.py)
    "channel": "",
    "background_percentile": 5.0,   # ktr / nuclear_reporter: per-frame floor, a low percentile of nuclear values
    "max_noise": 0.5,               # observations with a larger error estimate (ln units) count as missing; 0 = keep all
    "min_signal": 0.0,              # tracks with a lower mean nuclear signal above the floor are left out
    "metric": METRIC_DEFAULT,       # kind metric: any figure metric
}

FIT_DEFAULT = {
    "name": "",
    "model": "erk_ktr",             # erk_ktr, ou, states, custom (see dynamics/models.py)
    "readouts": [],                 # readout names; SDE models use the first, states uses all; empty = the first readout
    "features": [],                 # states only: feature names (Analysis → Features) instead of readouts, normalised per experiment
    "conditions": [],               # conditions fitted; empty = use `group`; both empty = every condition
    "group": "",
    "frames": [0, 0],               # movie frames; 0 at the end = last frame
    "window_hours": 0.0,            # >0 splits time into windows of this length (parameters may differ per window)
    "split_at_events": False,       # also start a new window at every time event (e.g. a drug addition)
    "options": {"modes": 2, "states": 3, "grid": 48, "topology": "full",
                "rank": 0,              # states: shared directions per state (correlated features); 0 = independent features
                "division": False,      # states: divisions as observations (needs features: tracks with lineage)
                "sweep": [],            # states: numbers of states to compare on held-out positions; the best is fitted
                "flux_feature": "",     # states: routes from the state lowest in this feature to the highest
                "flux_from": [], "flux_to": [],   # ... or these state numbers (1 = first)
                "drift": "theta*(mu - x)", "noise": "s", "parameters": {}},  # custom: formulas, {name: start value}
    "parameters": {},               # {name: {value, fixed, low, high, share}}; share: all, condition, window, condition_window
    "outlier": 0.01,                # share of observations that may be wrong values (tracking swaps, bad masks)
    "max_tracks": 2000,             # per condition, spread over positions; 0 = all
    "heldout": 0.2,                 # share of positions held out when comparing numbers of states
    "steps": 300,
    "batch": 256,                   # tracks per condition per step
    "lr": 0.05,
    "bootstrap": 0,                 # refits on positions resampled with replacement, for confidence intervals
    "recovery_check": False,        # refit data simulated from the fit: are the parameters identifiable at this interval?
    "seed": 0,
    "enabled": True,
}

DYNAMICS_DEFAULTS = {
    "device": "auto",               # auto = the GPU with the most free memory; or cuda:0, cuda:1, cpu
    "interval_minutes": 0.0,        # minutes between movie frames; 0 = from time.frames_per_hour (rates per frame if neither)
    "folder": "Dynamics",           # inside the results folder
    "fits": [],
}

FEATURE_DEFAULT = {
    "name": "",                     # what analyses call it; the same name in two experiments is the same feature
    "measure": "nuc_area",          # a cell or channel measurement, speed, age, or readout:<readout name>
    "channel": "",                  # for channel measurements
    "transform": "log",             # log, asinh or none
    "enabled": True,
}

FEATURES_DEFAULTS = {
    "list": [],                     # [FEATURE_DEFAULT]; empty = suggested from the channels
    "reference": "",                # condition whose cells define 0 and 1 of every feature (median, robust spread); empty = first
    "reference_frames": [0, 0],     # movie frames used for the reference; 0 at the end = last frame
}

FATE_DEFAULT = {
    "name": "",
    "marker": "",                   # feature whose high state is the fate (e.g. nuclear PPARG)
    "threshold": 0.0,               # in the marker's units (after its transform)
    "threshold_auto": True,         # threshold at the valley between the marker's two populations
    "smooth_frames": 5,             # running median over this many frames before thresholding
    "hold_frames": 10,              # a crossing counts once the marker stays on the new side this long
    "horizon_hours": 12.0,          # committor: P(marker high this much later | level now)
    "bins": 30,
    "split": "",                    # optional feature splitting cells into groups (e.g. ERK KTR)
    "split_groups": 2,              # by quantiles of the reference condition
    "split_hours": 1.0,             # the split feature is averaged over this much time before each frame
    "align": "commitment",          # commitment or division
    "align_features": [],           # empty = every enabled feature
    "match_on": [],                 # matched controls also close in these features (e.g. nuc_area, H2B nuc_sum)
    "exclude_feature": "",          # leave out crossings where this feature is extreme (e.g. H2B bimodality at mitosis)
    "exclude_above": 3.0,           # ... above this many robust spreads
    "exclude_frames": 3,            # ... within this many frames of the crossing
    "before_hours": 12.0,
    "after_hours": 6.0,
    "conditions": [],
    "group": "",
    "frames": [0, 0],
    "bootstrap": 200,               # resamples of positions for intervals
    "seed": 0,
    "enabled": True,
}

DEFAULTS: dict = {
    "fasttrack_version": __version__,
    "experiment": {"name": "", "notes": ""},
    "paths": {"raw": "", "results": ""},
    "files": {
        "pattern": PIPELINE_DEFAULTS["data"]["pattern"],
        "first_frame": 0,
        "last_frame": 0,          # 0 = to the end; exclusive otherwise
        "stride": 1,
        "io_threads": PIPELINE_DEFAULTS["data"]["io_threads"],
    },
    "channels": [],               # [{file_channel, name, role, reporter, color}]; empty = all, first is nuclear
    "time": {
        "frames_per_hour": 0.0,   # only for figure axes; 0 = axes in frames
        "events": [],             # [{name, frame, color}]
    },
    "plate": {
        "format": 96,
        "process": "assigned",    # "assigned": only wells with a condition; "all": every well found
        "conditions": [],         # [{name, color, wells: [B02, ...], notes}]
        "excluded_wells": [],
        "excluded_positions": [], # timestack ids, e.g. C05_xy2
    },
    "quality": QUALITY_DEFAULTS,
    "labelling": LABELLING_DEFAULTS,
    "pipeline": pipeline_defaults(),
    "analysis": {
        "min_track_frames": 30,
        "drop_border": True,
        "groups": [],             # [{name, conditions: [...]}]; the first condition is the reference
        "gates": [],              # [{name, metric: {...}, threshold, above}]
        "readouts": [],           # [READOUT_DEFAULT]: what reporters measure, with per-observation error, for models
        "dynamics": DYNAMICS_DEFAULTS,
        "features": FEATURES_DEFAULTS,   # many measurements per cell, normalised per experiment, for discovery
        "fates": [],                     # [FATE_DEFAULT]: committor and event-aligned analyses
    },
    "figures": {
        "folder": "Figures",      # inside the results folder
        "formats": ["pdf", "png"],
        "dpi": 200,
        "width": 6.0,
        "height": 4.0,
        "palette": "experiment",  # condition colours in every figure (style.PALETTES)
        "list": [],
        "exports": [],            # [EXPORT_DEFAULT]: multi-panel figures at journal sizes
        "export_folder": "Figures/Export",
        "summary": {                 # routine figures drawn after every run (analysis/summary.py)
            "enabled": True,
            "folder": "Figures/Summary",
            "kinds": [],             # empty = all: cell_count, nuc_area, timecourse, heatmap, distribution
            "unit": "well",          # replicates
            "band": "auto",          # analysis/bands.py
            "end_frames": 20,        # the distribution's window: this many frames before the end
        },
    },
}

PANEL_DEFAULT = {
    "source": "figure",          # figure (a quick-figure spec, stored here) | fate | dynamics
    "figure": FIGURE_DEFAULT,    # source figure: the full spec, so templates carry it to other experiments
    "ref": "",                   # source fate / dynamics: the fate or fit name
    "view": "",                  # source fate / dynamics: the view
    "condition": 0,              # dynamics tracks view: which condition
    "conditions": [],            # source fate / dynamics: the conditions to draw (names); empty = all
    "span": 1,                   # grid columns this panel spans
    "height": 45.0,              # mm; a row is as tall as its tallest panel
    "label": "",                 # panel letter; empty = automatic
    "caption": "",               # added to the generated caption
}

EXPORT_DEFAULT = {
    "name": "",
    "journal": "nature",         # nature | cell | science (style.JOURNALS): widths and panel-letter case
    "width": "double",           # single | onehalf | double, or a width in mm
    "columns": 2,                # grid columns that panels flow into
    "labels": "auto",            # auto (the journal's case) | lower | upper | none
    "font": 7.0,                 # pt at final size
    "line": 0.75,                # pt, data lines
    "palette": "",               # empty = the figures section's palette
    "formats": ["pdf", "png"],
    "dpi": 300,
    "panels": [],                # [PANEL_DEFAULT]
    "enabled": True,
}

CHANNEL_DEFAULT = {"file_channel": 1, "name": "", "role": "measure", "reporter": "", "color": "",
                   "primary": "auto"}   # what the reporter is read out as (analysis/summary.py PRIMARY); auto: from the reporter
EVENT_DEFAULT = {"name": "", "frame": 0, "color": "#444444"}
CONDITION_DEFAULT = {"name": "", "color": "", "wells": [], "notes": ""}
GROUP_DEFAULT = {"name": "", "conditions": []}
GATE_DEFAULT = {"name": "", "metric": METRIC_DEFAULT, "threshold": 0.0, "above": True}


def _fill_list(items: list | None, default: dict) -> list[dict]:
    return [merge(default, item) for item in (items or [])]


def normalise(exp: dict | None) -> dict:
    """Overlay `exp` on the defaults and fill every list entry, so every key is present."""
    out = merge(DEFAULTS, exp or {})
    out["channels"] = _fill_list(out["channels"], CHANNEL_DEFAULT)
    for i, ch in enumerate(out["channels"]):
        ch["name"] = ch["name"] or f"c{ch['file_channel']}"
        ch["color"] = ch["color"] or PALETTE[i % len(PALETTE)]
    out["time"]["events"] = _fill_list(out["time"]["events"], EVENT_DEFAULT)
    out["plate"]["conditions"] = _fill_list(out["plate"]["conditions"], CONDITION_DEFAULT)
    for i, cond in enumerate(out["plate"]["conditions"]):
        cond["color"] = cond["color"] or PALETTE[i % len(PALETTE)]
        cond["wells"] = sorted({normalise_well(w) for w in cond["wells"]}, key=well_sort_key)
    out["plate"]["excluded_wells"] = sorted({normalise_well(w) for w in out["plate"]["excluded_wells"]}, key=well_sort_key)
    out["analysis"]["groups"] = _fill_list(out["analysis"]["groups"], GROUP_DEFAULT)
    out["analysis"]["gates"] = _fill_list(out["analysis"]["gates"], GATE_DEFAULT)
    out["figures"]["list"] = _fill_list(out["figures"]["list"], FIGURE_DEFAULT)
    out["figures"]["exports"] = _fill_list(out["figures"]["exports"], EXPORT_DEFAULT)
    for export in out["figures"]["exports"]:
        export["panels"] = _fill_list(export["panels"], PANEL_DEFAULT)
    if out["labelling"]["categories"] == OLD_CATEGORIES:      # the earlier default list: take the new one
        out["labelling"]["categories"] = list(LABELLING_DEFAULTS["categories"])
    out["analysis"]["readouts"] = _fill_list(out["analysis"]["readouts"], READOUT_DEFAULT)
    out["analysis"]["dynamics"]["fits"] = _fill_list(out["analysis"]["dynamics"]["fits"], FIT_DEFAULT)
    out["analysis"]["features"]["list"] = _fill_list(out["analysis"]["features"]["list"], FEATURE_DEFAULT)
    out["analysis"]["fates"] = _fill_list(out["analysis"]["fates"], FATE_DEFAULT)
    for fit in out["analysis"]["dynamics"]["fits"]:
        fit["parameters"] = {k: dict(v or {}) for k, v in (fit["parameters"] or {}).items()}
    return out


def new_experiment(**sections) -> dict:
    return normalise(sections)


# ---------------------------------------------------------------------- wells
_WELL = re.compile(r"^([A-Pa-p])0*(\d{1,2})$")


def normalise_well(well) -> str:
    """'b2', 'B02' -> 'B02'. Integers use the old notebook code: 202 -> B02 (row number, two-digit column)."""
    if isinstance(well, int):
        row, col = divmod(well, 100)
        return f"{chr(64 + row)}{col:02d}"
    m = _WELL.match(str(well).strip())
    if not m:
        raise ValueError(f"not a well name: {well!r}")
    return f"{m.group(1).upper()}{int(m.group(2)):02d}"


def well_sort_key(well: str) -> tuple[str, int]:
    return well[0], int(well[1:])


def plate_shape(fmt: int) -> tuple[int, int]:
    return {6: (2, 3), 12: (3, 4), 24: (4, 6), 48: (6, 8), 96: (8, 12), 384: (16, 24)}[int(fmt)]


def wells_outside_plate(plate: dict) -> list[str]:
    """Listed wells that do not exist on the plate format (J12 on a 96-well plate)."""
    rows, cols = plate_shape(plate["format"])
    wells = [w for c in plate["conditions"] for w in c["wells"]] + plate["excluded_wells"]
    return sorted({w for w in wells if ord(w[0]) - 64 > rows or int(w[1:]) > cols}, key=well_sort_key)


def condition_of_well(exp: dict) -> dict[str, str]:
    return {w: c["name"] for c in exp["plate"]["conditions"] for w in c["wells"]}


def select_timestacks(exp: dict, ids: list[str]) -> list[str]:
    """Timestack ids to process: drop excluded wells and positions and flagged positions not kept;
    with process 'assigned', keep only wells with a condition."""
    plate = exp["plate"]
    quality = exp.get("quality", QUALITY_DEFAULTS)
    flagged = set(quality["flagged"]) - set(quality["keep"]) if quality["enabled"] else set()
    assigned = condition_of_well(exp)
    keep = []
    for key in ids:
        well = key.split("_xy")[0]
        if key in plate["excluded_positions"] or well in plate["excluded_wells"] or key in flagged:
            continue
        if plate["process"] == "assigned" and assigned and well not in assigned:
            continue
        keep.append(key)
    return keep


# ---------------------------------------------------------------------- channels
def loaded_channels(exp: dict) -> list[dict]:
    """Channels that are read, in the order they sit in the image tensor."""
    return [c for c in exp["channels"] if c["role"] != "skip"]


def channel_index(exp: dict, name: str) -> int:
    names = [c["name"] for c in loaded_channels(exp)]
    if name not in names:
        raise KeyError(f"no measured channel called {name!r}; channels are {names}")
    return names.index(name)


def pipeline_config(exp: dict) -> dict:
    """The flat config dict the tracking code reads."""
    channels = loaded_channels(exp)
    nuclear = [i for i, c in enumerate(channels) if c["role"] == "nuclear"]
    files = exp["files"]
    data = {
        "root": exp["paths"]["raw"],
        "output": exp["paths"]["results"],
        "pattern": files["pattern"],
        "channels": [int(c["file_channel"]) for c in channels],
        "channel_names": [c["name"] for c in channels],
        "stride": int(files["stride"]),
        "first_frame": int(files["first_frame"]),
        "last_frame": int(files["last_frame"]),
        "max_frames": 0,
        "io_threads": int(files["io_threads"]),
    }
    from .device import resolve as resolve_device
    cfg = merge(PIPELINE_DEFAULTS, {**copy.deepcopy(exp["pipeline"]), "data": data, "nuclear_channel": nuclear[0] if nuclear else 0})
    cfg["device"] = resolve_device(cfg["device"])
    return cfg


def validate(exp: dict) -> list[str]:
    """Problems that stop a run, in plain words. Empty when the file is ready."""
    problems = []
    if not exp["paths"]["raw"]:
        problems.append("Choose the raw data folder.")
    elif not Path(exp["paths"]["raw"]).is_dir():
        problems.append(f"Raw data folder not found: {exp['paths']['raw']}")
    if not exp["paths"]["results"]:
        problems.append("Choose a results folder.")
    elif exp["paths"]["raw"] and _inside(exp["paths"]["results"], exp["paths"]["raw"]):
        problems.append("The results folder must be outside the raw data folder; raw data is only ever read.")
    channels = exp["channels"]
    if channels:
        roles = [c["role"] for c in channels]
        if roles.count("nuclear") != 1:
            problems.append("Mark exactly one channel as nuclear.")
        names = [c["name"] for c in channels]
        if len(set(names)) != len(names):
            problems.append("Channel names must be unique.")
        if len({c["file_channel"] for c in channels}) != len(channels):
            problems.append("Two channels point at the same file channel.")
    outside = wells_outside_plate(exp["plate"])
    if outside:
        problems.append(f"Wells {', '.join(outside)} do not exist on a {exp['plate']['format']}-well plate.")
    seen: dict[str, str] = {}
    for cond in exp["plate"]["conditions"]:
        for w in cond["wells"]:
            if w in seen and seen[w] != cond["name"]:
                problems.append(f"Well {w} is in both {seen[w]} and {cond['name']}.")
            seen[w] = cond["name"]
    return problems


def warnings_for(exp: dict) -> list[str]:
    """Problems that do not stop tracking: the run goes ahead and they are logged.
    Mostly figure settings that point at something no longer there; those figures skip it."""
    out = []
    names = {c["name"] for c in exp["plate"]["conditions"]}
    for cond in exp["plate"]["conditions"]:
        if not cond["wells"]:
            out.append(f"Condition {cond['name']} has no wells.")
    for group in exp["analysis"]["groups"]:
        missing = [c for c in group["conditions"] if c not in names]
        if missing:
            out.append(f"Group {group['name']} names conditions not on the plate ({', '.join(missing)}); its figures leave them out.")
    channels = {c["name"] for c in loaded_channels(exp)}
    if channels:
        for gate in exp["analysis"]["gates"]:
            for key in ("channel", "over_channel"):
                ch = gate["metric"][key]
                if ch and ch not in channels:
                    out.append(f"Gate {gate['name']} uses channel {ch}, which is not measured.")
    groups = {g["name"] for g in exp["analysis"]["groups"]}
    gates = {g["name"] for g in exp["analysis"]["gates"]}
    for fig in exp["figures"]["list"]:
        if fig["group"] and fig["group"] not in groups:
            out.append(f"Figure {fig['name'] or '(unnamed)'} uses group {fig['group']}, which does not exist; it will be skipped.")
        if fig["kind"] == "percent_high" and fig["gate"] not in gates:
            out.append(f"Figure {fig['name'] or '(unnamed)'} uses gate {fig['gate'] or '(none)'}, which does not exist; it will be skipped.")
    return out


def time_unit(exp: dict) -> tuple[float, str]:
    """(time between result frames, unit): minutes when the interval is known, else frames."""
    dyn, files = exp["analysis"]["dynamics"], exp["files"]
    stride = max(1, int(files["stride"]))
    if dyn["interval_minutes"] > 0:
        return float(dyn["interval_minutes"]) * stride, "min"
    if exp["time"]["frames_per_hour"] > 0:
        return 60.0 * stride / float(exp["time"]["frames_per_hour"]), "min"
    return 1.0, "frame"


def rename_condition(exp: dict, old: str, new: str) -> None:
    """Rename a condition in the analysis groups (the plate holds the condition itself)."""
    for group in exp["analysis"]["groups"]:
        group["conditions"] = [new if c == old else c for c in group["conditions"]]


def drop_condition(exp: dict, name: str) -> None:
    for group in exp["analysis"]["groups"]:
        group["conditions"] = [c for c in group["conditions"] if c != name]


def _inside(child: str, parent: str) -> bool:
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------- files
def load_experiment(path: str | Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if is_legacy(raw):
        return import_legacy(raw)
    return normalise(raw)


def save_experiment(exp: dict, path: str | Path) -> None:
    exp = normalise(exp)
    exp["fasttrack_version"] = __version__
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    header = f"# FastTrack experiment, saved {_dt.datetime.now():%Y-%m-%d %H:%M}. Every setting is written out so a rerun gives the same result.\n"
    with open(tmp, "w") as f:
        f.write(header)
        yaml.safe_dump(exp, f, sort_keys=False, allow_unicode=True, width=120)
    tmp.replace(path)


def same_pipeline(a: dict, b: dict) -> bool:
    """True if two experiments would give the same tracks (paths and figures may differ)."""
    keys = ("files", "pipeline")  # quality only picks positions; a finished position is the same either way
    chan = lambda e: [(c["file_channel"], c["role"]) for c in normalise(e)["channels"]]
    return all(normalise(a)[k] == normalise(b)[k] for k in keys) and chan(a) == chan(b)


# ---------------------------------------------------------------------- older formats
_LEGACY_MEASURE = {"MeanNuc": "nuc_mean", "MedianNuc": "nuc_median", "MeanCyto": "cyto_mean", "MedianCyto": "cyto_median",
                   "SumNuc": "nuc_sum", "Area": "nuc_area", "NucArea": "nuc_area"}


def is_legacy(raw: dict) -> bool:
    return any(k in raw for k in ("code_to_condition", "code_to_well", "batch_dict", "condition_wells", "layout", "plate_layout")) \
        and "pipeline" not in raw


def import_legacy(raw: dict) -> dict:
    """Read the notebook settings YAML (code_to_condition, code_to_well, batch_dict, dmi_addition, x/y/z ...)
    or a Plate tab layout (condition_wells / layout + colours). Anything not understood goes into the notes."""
    exp = new_experiment()
    used = set()
    conditions: dict[str, dict] = {}

    def condition(name: str) -> dict:
        return conditions.setdefault(name, {"name": name, "color": "", "wells": []})

    if "code_to_condition" in raw:
        codes = {int(k): str(v) for k, v in raw["code_to_condition"].items()}
        for code, name in codes.items():
            condition(name)["wells"] += [normalise_well(w) for w in (raw.get("code_to_well") or {}).get(code, [])]
        for name, entry in (raw.get("batch_dict") or {}).items():
            members = entry.get("data", entry) if isinstance(entry, dict) else entry
            exp["analysis"]["groups"].append({"name": str(name), "conditions": [codes[int(c)] for c in members if int(c) in codes]})
        used |= {"code_to_condition", "code_to_well", "batch_dict", "batch", "batch_name"}
    groups = raw.get("condition_wells") or raw.get("layout") or {}
    for name, wells in groups.items():
        condition(str(name))["wells"] += [normalise_well(w) for w in wells]
    colours = raw.get("condition_colors") or {}
    if isinstance(raw.get("conditions"), dict):  # Plate tab JSON: {name: [r, g, b]}
        colours = {k: "#%02x%02x%02x" % tuple(v) for k, v in raw["conditions"].items()}
    for name, colour in colours.items():
        condition(str(name))["color"] = colour
    for name in raw.get("conditions") or []:
        if isinstance(name, str):
            condition(name)
    used |= {"condition_wells", "layout", "condition_colors", "conditions", "plate_layout"}
    exp["plate"]["conditions"] = list(conditions.values())

    if "dmi_addition" in raw:
        exp["time"]["events"].append({"name": "DMI addition", "frame": int(raw["dmi_addition"]), "color": "#444444"})
        used.add("dmi_addition")
    names = [raw.get(f"{a}condition") for a in "xyz"]
    if any(names):
        exp["channels"] = [{"file_channel": i + 1, "name": str(n or f"c{i + 1}"), "role": "nuclear" if i == 0 else "measure"}
                           for i, n in enumerate(names)]
        for a, n in zip("xyz", names):
            if n is None or f"{a}threshold" not in raw:
                continue
            measure = _LEGACY_MEASURE.get(raw.get(f"{a}measure", "MeanNuc"), "nuc_mean")
            # The notebook compared log2 values with these thresholds.
            exp["analysis"]["gates"].append({"name": f"{n} high", "threshold": float(raw[f"{a}threshold"]), "above": True,
                                             "metric": {"measure": measure, "channel": str(n), "log2": True}})
        used |= {f"{a}{k}" for a in "xyz" for k in ("condition", "measure", "threshold")}
    if raw.get("data_directory"):
        exp["paths"]["results"] = str(raw["data_directory"])
        used.add("data_directory")
    if raw.get("endframe"):
        exp["files"]["last_frame"] = int(raw["endframe"])
        used.add("endframe")
    rest = {k: v for k, v in raw.items() if k not in used}
    if rest:
        exp["experiment"]["notes"] = "Imported from an older settings file. Not used:\n" + yaml.safe_dump(rest, sort_keys=False)
    return normalise(exp)
