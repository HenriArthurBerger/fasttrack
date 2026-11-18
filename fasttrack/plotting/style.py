"""The figure standard (docs/FIGURE_STANDARD.md) as code: journal sizes, type and line rules, and colours.

Every FastTrack figure draws inside `context(...)`, so the rules hold for quick figures, fate and dynamics views
and exported multi-panel figures alike. Sizes are in points at final print size, widths in mm."""
from __future__ import annotations

import colorsys
import contextlib
import threading

import matplotlib
import matplotlib.font_manager
from matplotlib.colors import to_hex, to_rgb

MM = 1 / 25.4  # inches per mm

# column widths (mm), maximum height, and how panels are lettered
JOURNALS = {
    "nature": {"single": 89, "onehalf": 120, "double": 183, "max_height": 170, "labels": "lower"},
    "cell": {"single": 85, "onehalf": 114, "double": 174, "max_height": 225, "labels": "upper"},
    "science": {"single": 57, "onehalf": 121, "double": 184, "max_height": 220, "labels": "upper"},
}
WIDTHS = ["single", "onehalf", "double"]

FONTS = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]

# type and lines in points at final size: journal print, and a larger set for the screen
PRESETS = {
    "journal": {"font": 7.0, "small": 6.0, "label": 8.0, "line": 0.75, "axis": 0.5, "tick": 2.5},
    "screen": {"font": 9.0, "small": 8.0, "label": 10.0, "line": 1.2, "axis": 0.6, "tick": 3.0},
}

# ---------------------------------------------------------------- colours
REFERENCE = "#333333"      # the control / reference condition in a comparison
CONTEXT = "#BBBBBB"        # the reference drawn behind another condition
MISSING = "#DDDDDD"

TOL_HIGH_CONTRAST = ["#004488", "#DDAA33", "#BB5566"]
TOL_BRIGHT = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
TOL_MUTED = ["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE", "#882255", "#44AA99", "#999933", "#AA4499"]
TOL_LIGHT = ["#77AADD", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF", "#44BB99", "#BBCC33", "#AAAA00"]

PALETTES = ["experiment", "auto", "okabe_ito", "factorial"]
PALETTE_TEXT = {"experiment": "colours set on the Plate tab",
                "auto": "colour-blind safe, chosen by the number of conditions (Paul Tol)",
                "okabe_ito": "Okabe-Ito (colour-blind safe, up to 7 + reference)",
                "factorial": "combinations: each drug a hue, combinations mix them and draw dashed"}

SEQUENTIAL = "viridis"     # unsigned: counts, densities, occupancy, probability
DIVERGING = "RdBu_r"       # signed around 0: z-scores, fold changes, net flux
HEAT = "magma"             # images of cells x time


def auto_palette(n: int) -> list[str]:
    """n colour-blind-safe colours for the conditions other than the reference."""
    if n <= 3:
        return TOL_HIGH_CONTRAST[:n]
    if n <= 6:
        return TOL_BRIGHT[:n]
    if n <= 9:
        return TOL_MUTED[:n]
    import matplotlib.pyplot as plt  # beyond 9 colour cannot carry identity: cycle, and use small multiples
    cmap = plt.get_cmap("tab20")
    return [to_hex(cmap(i % 20)) for i in range(n)]


def _mix(colors: list[str], darken: float = 0.15) -> str:
    rgb = [sum(to_rgb(c)[k] for c in colors) / len(colors) for k in range(3)]
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return to_hex(colorsys.hls_to_rgb(h, max(0.0, l * (1 - darken)), min(1.0, s * 1.15)))


def split_factors(name: str) -> list[str]:
    """'Tdi + Palbo' -> ['Tdi', 'Palbo']: a combination's parts."""
    import re
    return [p.strip() for p in re.split(r"\s*\+\s*|\s*&\s*", name) if p.strip()]


def condition_styles(exp: dict | None, names: list[str], palette: str | None = None,
                     reference: str | None = None) -> dict[str, dict]:
    """{name: {color, ls}} for the conditions drawn. The reference (default: the first) is dark grey unless the
    palette is `experiment`, which keeps the Plate tab's colours exactly."""
    exp = exp or {}
    palette = palette or exp.get("figures", {}).get("palette", "experiment")
    own = {c["name"]: c.get("color") for c in exp.get("plate", {}).get("conditions", [])}
    if not names:
        return {}
    reference = reference if reference in names else names[0]
    out: dict[str, dict] = {}
    if palette == "experiment":
        from ..experiment import PALETTE
        return {n: {"color": own.get(n) or PALETTE[i % len(PALETTE)], "ls": "-"} for i, n in enumerate(names)}
    others = [n for n in names if n != reference]
    if palette == "factorial":
        ref_parts = set(split_factors(reference))
        singles = sorted({p for n in others for p in split_factors(n) if p not in ref_parts},
                         key=lambda p: min(i for i, n in enumerate(others) if p in split_factors(n)))
        hues = dict(zip(singles, auto_palette(len(singles)) if len(singles) > 6 else (TOL_BRIGHT * 2)[:len(singles)]))
        for n in others:
            parts = [p for p in split_factors(n) if p in hues]
            if not parts:
                out[n] = {"color": REFERENCE, "ls": ":"}
            elif len(parts) == 1:
                out[n] = {"color": hues[parts[0]], "ls": "-"}
            else:
                out[n] = {"color": _mix([hues[p] for p in parts]), "ls": "--"}
    else:
        colors = OKABE_ITO if palette == "okabe_ito" and len(others) <= 7 else auto_palette(len(others))
        out = {n: {"color": c, "ls": "-"} for n, c in zip(others, colors)}
    out[reference] = {"color": REFERENCE, "ls": "-"}
    return {n: out[n] for n in names}


def condition_colors(exp: dict | None, names: list[str], palette: str | None = None, reference: str | None = None) -> list[str]:
    styles = condition_styles(exp, names, palette, reference)
    return [styles[n]["color"] for n in names]


def state_colors(n: int) -> list[str]:
    """Hidden states: Tol light (never the condition palettes) up to 8; beyond that, a sequential map in state order."""
    if n <= len(TOL_LIGHT):
        return TOL_LIGHT[:n]
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("cividis")
    return [to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


# ---------------------------------------------------------------- type and lines
def rc(preset: str = "journal", font: float | None = None, line: float | None = None) -> dict:
    """matplotlib settings for the standard. `font` and `line` override the preset's text size and data line width."""
    p = dict(PRESETS.get(preset, PRESETS["journal"]))
    if font:
        scale = font / p["font"]
        p.update(font=font, small=max(5.0, p["small"] * scale), label=p["label"] * scale)
    if line:
        p["line"] = line
    return {
        "font.family": "sans-serif", "font.sans-serif": FONTS, "font.size": p["font"],
        "axes.titlesize": p["font"], "axes.labelsize": p["font"], "axes.titleweight": "normal",
        "figure.titlesize": p["font"], "figure.labelsize": p["font"],
        "xtick.labelsize": p["small"], "ytick.labelsize": p["small"], "legend.fontsize": p["small"],
        "legend.title_fontsize": p["small"], "legend.frameon": False, "legend.handlelength": 1.5,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
        "axes.linewidth": p["axis"], "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#222222", "axes.labelcolor": "#000000", "axes.titlepad": 3.0, "axes.labelpad": 2.0,
        "xtick.major.width": p["axis"], "ytick.major.width": p["axis"], "xtick.minor.width": p["axis"] * 0.6,
        "ytick.minor.width": p["axis"] * 0.6, "xtick.major.size": p["tick"], "ytick.major.size": p["tick"],
        "xtick.minor.size": p["tick"] * 0.6, "ytick.minor.size": p["tick"] * 0.6,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "xtick.color": "#222222", "ytick.color": "#222222",
        "lines.linewidth": p["line"], "lines.markersize": 3.0, "patch.linewidth": p["axis"],
        "errorbar.capsize": 1.5, "boxplot.flierprops.markersize": 2.0,
        "grid.linewidth": 0.25, "grid.color": "#DDDDDD",
        "image.cmap": SEQUENTIAL, "axes.prop_cycle": matplotlib.cycler(color=TOL_BRIGHT),
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.constrained_layout.w_pad": 1.5 / 72, "figure.constrained_layout.h_pad": 1.5 / 72,
        "figure.constrained_layout.wspace": 0.02, "figure.constrained_layout.hspace": 0.02,
        "savefig.dpi": 300, "savefig.facecolor": "white", "figure.facecolor": "white",
        "_label_size": p["label"],
    }


_LOCK = threading.RLock()   # matplotlib's settings are global: figures drawn in parallel threads take turns


@contextlib.contextmanager
def context(preset: str = "journal", font: float | None = None, line: float | None = None):
    settings = rc(preset, font, line)
    settings.pop("_label_size")
    with _LOCK, matplotlib.rc_context(settings):
        yield


def label_size(preset: str = "journal", font: float | None = None) -> float:
    return rc(preset, font)["_label_size"]


def width_mm(journal: str, width) -> float:
    """A column name (single, onehalf, double) or a number of mm."""
    if isinstance(width, (int, float)) and width > 0:
        return float(width)
    return float(JOURNALS.get(journal, JOURNALS["nature"]).get(str(width), JOURNALS["nature"]["double"]))


def panel_letter(i: int, case: str = "lower") -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s.upper() if case == "upper" else s


def direct_labels(ax, lines: list[tuple[str, object, str]], min_gap: float | None = None) -> None:
    """Label each line at its right end instead of a legend: [(text, Line2D, colour)]. Labels are pushed apart
    vertically (in axes fraction) so they never overlap; text stays dark, a short tick in the line's colour links them."""
    import numpy as np
    ends = []
    for text, line, color in lines:
        x, y = (np.asarray(v, float) for v in line.get_data())
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            continue
        disp = ax.transData.transform((x[ok][-1], y[ok][-1]))
        ends.append([ax.transAxes.inverted().transform(disp)[1], text, color])
    if not ends:
        return
    if min_gap is None:   # one text line (1.3 x the font) as a share of the axes height
        root = ax.figure.figure if hasattr(ax.figure, "figure") and ax.figure.figure is not None else ax.figure
        height_pt = ax.get_position().height * root.get_size_inches()[1] * 72
        if hasattr(ax.figure, "bbox_relative"):
            height_pt *= ax.figure.bbox_relative.height
        size = matplotlib.font_manager.FontProperties(size=matplotlib.rcParams["legend.fontsize"]).get_size_in_points()
        min_gap = min(0.3, 1.6 * size / max(height_pt, 1.0))
    ends.sort(key=lambda e: e[0])
    ys = np.clip([e[0] for e in ends], 0.0, 1.0)
    for k in range(1, len(ys)):          # push up, then pull back inside the axes
        ys[k] = max(ys[k], ys[k - 1] + min_gap)
    over = ys[-1] - 1.0
    if over > 0:
        ys -= over
        for k in range(len(ys) - 2, -1, -1):
            ys[k] = min(ys[k], ys[k + 1] - min_gap)
    from matplotlib.lines import Line2D
    lw = matplotlib.rcParams["lines.linewidth"]
    for (y0, text, color), y in zip(ends, ys):     # a short dash in the line's colour, then the text in black
        ax.add_line(Line2D([1.01, 1.045], [y, y], transform=ax.transAxes, color=color, lw=lw * 1.5, clip_on=False,
                           solid_capstyle="butt"))
        ax.text(1.055, y, text, transform=ax.transAxes, va="center", ha="left", color="#000000",
                fontsize=matplotlib.rcParams["legend.fontsize"], clip_on=False)
