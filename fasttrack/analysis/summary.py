"""Routine summary figures, made without asking: cell counts, nuclear size, and each channel's primary measurement
over time, as a population heatmap, and as a distribution at the end of the movie.

Each channel has a **primary measurement** (`channels[].primary`): what that reporter is read out as.
- Nuclear mean for a nuclear reporter such as PPARG.
- Cytoplasm / nucleus for a KTR such as the ERK KTR.
- Integrated nuclear intensity for DNA content (H2B).

`auto` picks from the reporter's name and the channel's role. Summary figures use it; so do the defaults offered
elsewhere.

The figures are ordinary figure specs (see experiment.FIGURE_DEFAULT) built from the experiment, never stored in it.
After every run they are drawn into `<results>/<figures.summary.folder>`, and the Summary page shows them. Any one
can be copied into the figure list to edit.
"""
from __future__ import annotations

import copy
from pathlib import Path

from ..experiment import FIGURE_DEFAULT, METRIC_DEFAULT, loaded_channels, merge, normalise

# preset: (label, metric fields)
PRIMARY = {
    "nuc_mean": ("Nuclear mean", {"measure": "nuc_mean"}),
    "nuc_median": ("Nuclear median", {"measure": "nuc_median"}),
    "nuc_sum": ("Integrated nuclear intensity (DNA content)", {"measure": "nuc_sum"}),
    "cyto_mean": ("Cytoplasmic mean", {"measure": "cyto_median"}),
    "cyto_nuc": ("Cytoplasm / nucleus (KTR)", {"measure": "cyto_median", "over_measure": "nuc_median",
                                               "background_percentile": 5.0, "log2": True}),
    "nuc_cyto": ("Nucleus / cytoplasm (translocation)", {"measure": "nuc_median", "over_measure": "cyto_median",
                                                         "background_percentile": 5.0, "log2": True}),
    "none": ("None (no summary figures)", {}),
}
PRIMARY_CHOICES = ["auto", *PRIMARY]
SHORT = {"nuc_mean": "nuclear mean", "nuc_median": "nuclear median", "nuc_sum": "integrated intensity",
         "cyto_mean": "cytoplasm mean", "cyto_nuc": "cyto / nuc ratio", "nuc_cyto": "nuc / cyto ratio", "none": "none"}

SUMMARY_KINDS = {
    "cell_count": "Cells per position over time",
    "nuc_area": "Nuclear area over time",
    "timecourse": "Each channel's primary measurement over time",
    "heatmap": "Each channel's primary measurement as a population heatmap",
    "distribution": "Each channel's primary measurement at the end of the movie",
}


def resolve_primary(channel: dict) -> str:
    """The preset a channel uses: its own, or for `auto` one picked from the reporter and the role."""
    choice = channel.get("primary") or "auto"
    if choice != "auto":
        return choice
    reporter = (channel.get("reporter") or channel.get("name") or "").lower()
    if "ktr" in reporter:
        return "cyto_nuc"
    if any(k in reporter for k in ("h2b", "hoechst", "dapi", "histone")):
        return "nuc_sum"
    return "nuc_mean"


def primary_metric(exp: dict, channel_name: str) -> dict | None:
    """The full metric spec of a channel's primary measurement, or None for `none`."""
    channel = next((c for c in loaded_channels(exp) if c["name"] == channel_name), None)
    if channel is None:
        return None
    preset = resolve_primary(channel)
    if preset == "none":
        return None
    fields = PRIMARY[preset][1]
    metric = merge(METRIC_DEFAULT, {**fields, "channel": channel_name})
    if metric["over_measure"]:
        metric["over_channel"] = channel_name
    return metric


def channel_title(channel: dict) -> str:
    """'PPARG (c2)' when the reporter is known, else the channel name."""
    rep = channel.get("reporter") or ""
    return f"{rep} ({channel['name']})" if rep and rep != channel["name"] else channel["name"]


def summary_specs(exp: dict) -> list[dict]:
    """The summary figures for this experiment, in the order they are shown."""
    exp = normalise(exp)
    settings = exp["figures"]["summary"]
    kinds = settings["kinds"] or list(SUMMARY_KINDS)
    conditions = [c["name"] for c in exp["plate"]["conditions"] if c["wells"]]
    compare = "overlay" if len(conditions) <= 4 else "reference"
    base = merge(FIGURE_DEFAULT, {"conditions": conditions, "unit": settings["unit"], "band": settings["band"],
                                  "compare": compare, "stats": True, "show_events": True})
    end = int(settings["end_frames"])
    specs: list[dict] = []

    n = len(conditions)
    grid_rows = -(-max(n - 1, 1) // 4) if compare == "reference" else 1

    def add(name: str, **fields) -> None:
        spec = copy.deepcopy(base)
        spec.update(fields, name=name)
        kind = spec["kind"]
        # sized for what they show: a heatmap row per condition, small multiples 4 per row, violins by count
        if kind == "heatmap":
            spec["plot"].update(width=8.0, height=min(1.4 + 0.75 * n, 12.0))
        elif kind == "distribution":
            spec["plot"].update(width=max(4.0, 1.2 + 0.6 * n), height=4.0)
        elif compare == "reference":
            spec["plot"].update(width=8.0, height=1.4 + 1.8 * grid_rows)
        spec["summary_key"] = fields.get("summary_key", name)
        specs.append(spec)

    if "cell_count" in kinds:
        add("Cells per position", kind="cell_count", summary_key="cell_count")
    if "nuc_area" in kinds:
        add("Nuclear area", kind="timecourse", metric=merge(METRIC_DEFAULT, {"measure": "nuc_area"}), summary_key="nuc_area")
    for channel in loaded_channels(exp):
        metric = primary_metric(exp, channel["name"])
        if metric is None:
            continue
        title = f"{channel_title(channel)}: {PRIMARY[resolve_primary(channel)][0].split(' (')[0].lower()}"
        key = channel["name"]
        if "timecourse" in kinds:
            add(title, kind="timecourse", metric=metric, summary_key=f"{key}:timecourse")
        if "heatmap" in kinds:
            heat = dict(metric, log2=True)       # intensities and ratios span decades: bins on a log scale
            add(f"{title}, population", kind="heatmap", metric=heat, compare="overlay", summary_key=f"{key}:heatmap")
        if "distribution" in kinds:
            add(f"{title}, last {end} frames", kind="distribution", metric=metric, style="violin",
                frames=[-end, 0], summary_key=f"{key}:distribution")
    return specs


def render_summary(exp: dict, on_event=lambda e: None, cache=None) -> list[Path]:
    """Draw every summary figure into the results folder (each in every format, with its CSV)."""
    from .figures import ResultsCache, write_figure
    exp = normalise(exp)
    settings = exp["figures"]["summary"]
    if not settings["enabled"] or not exp["paths"]["results"]:
        return []
    folder = Path(exp["paths"]["results"]) / settings["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    cache = cache or ResultsCache(exp)
    written: list[Path] = []
    for spec in summary_specs(exp):
        written += write_figure(exp, spec, folder, cache, on_event)
    return written
