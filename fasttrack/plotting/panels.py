"""Layout helpers shared by the fate and dynamics views, so each view fits whatever it is drawn into: a window
on screen, a single-column panel, or a double-column page.

- `grid` picks how many small plots fit in a row from the panel's size in mm, shares axes, and keeps tick labels
  only on the outer plots.
- `legend` puts one legend for the whole panel outside the plots, on the right, in as many columns as it needs.
- `finish` sets the title and the notes (what lines, bands and markers are). On a page they move to the caption.
- `empty` marks a view that has nothing to draw, so a page says so instead of exporting a blank panel.
"""
from __future__ import annotations

import math
import textwrap

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

from . import style

PT = 72 / 25.4          # points per mm


def standalone(fig) -> bool:
    """A Figure of its own (a window, a saved file) rather than a panel (SubFigure) of a page."""
    return hasattr(fig, "get_size_inches")


def size_mm(fig) -> tuple[float, float]:
    if standalone(fig):
        w, h = fig.get_size_inches()
    else:
        W, H = fig.figure.get_size_inches()
        w, h = W * max(fig.bbox_relative.width, 0.05), H * max(fig.bbox_relative.height, 0.05)
    return float(w) * 25.4, float(h) * 25.4


def begin(fig, exp: dict | None = None) -> None:
    """Clear the figure and give a standalone one the constrained layout (pages lay out their own panels)."""
    fig.clear()
    fig._fasttrack_notes = []
    fig._fasttrack_empty = False
    fig._fasttrack_legend_below = 0
    if standalone(fig) and fig.get_layout_engine() is None:
        fig.set_layout_engine("constrained")


def screen(fig):
    """The screen style for a standalone figure; a page panel is already inside the page's journal style."""
    from ..analysis.figures import _nothing
    return style.context("screen") if standalone(fig) else _nothing()


def font_pt(which: str = "small") -> float:
    return matplotlib.font_manager.FontProperties(size=which).get_size_in_points()


def note(fig, text: str) -> None:
    notes = getattr(fig, "_fasttrack_notes", None)
    if notes is None:
        notes = fig._fasttrack_notes = []
    if text and text not in notes:
        notes.append(text)


def finish(fig, title: str, notes: list[str] | tuple = ()) -> None:
    """Title, then the notes on a second, smaller line wrapped to the width. A page moves both to the caption."""
    layout_notes = list(getattr(fig, "_fasttrack_notes", None) or [])
    fig._fasttrack_notes = []
    for n in list(notes) + layout_notes:          # what the marks are first, then what was left out
        note(fig, n)
    notes = fig._fasttrack_notes
    w, _ = size_mm(fig)
    size = font_pt("small")
    text = "; ".join(n.rstrip(".") for n in notes)
    wrapped = "\n".join(textwrap.wrap(text, max(30, int(w * PT / (0.52 * size))))) if text else ""
    fig.suptitle(title + ("\n" if title and wrapped else "") + wrapped, fontsize="medium" if not wrapped else "small")


def empty(fig, message: str) -> None:
    fig.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig._fasttrack_empty = True
    fig._fasttrack_message = message


def fit_grid(fig, n: int, min_w: float = 30, min_h: float = 24, reserve_w: float = 0, reserve_h: float = 8,
             cols: int | None = None) -> tuple[int, int, int]:
    """(rows, columns, shown): as many small plots as fit at no less than min_w x min_h mm each.
    reserve_w/h: mm kept for a legend, a colour bar, shared axis labels and the title."""
    w, h = size_mm(fig)
    w, h = max(w - reserve_w, min_w), max(h - reserve_h, min_h)
    cols = max(1, min(n, cols or int(w // min_w)))
    rows = min(math.ceil(n / cols), max(1, int(h // min_h)))
    shown = min(n, rows * cols)
    cols = max(1, min(cols, math.ceil(shown / rows)))       # balance: 7 in 2 rows is 4 + 3, not 6 + 1
    return rows, cols, shown


def grid(fig, n: int, sharex: bool = True, sharey: bool = True, min_w: float = 30, min_h: float = 24,
         reserve_w: float = 0, reserve_h: float = 8, cols: int | None = None, what: str = "plots",
         legend_items: int = 0) -> list:
    """Small multiples that fit the figure. Plots that do not fit are left out with a note.
    legend_items: entries of a legend that `reserve_w` keeps room for on the right; when that room costs plots,
    the legend goes under them instead (see `legend`)."""
    if n <= 0:
        return []
    rows, cols_, shown = fit_grid(fig, n, min_w, min_h, reserve_w, reserve_h, cols)
    if legend_items and shown < n:
        w, _ = size_mm(fig)
        per_row = max(1, int(w // max(reserve_w, 10)))
        below = math.ceil(legend_items / per_row) * font_pt(matplotlib.rcParams["legend.fontsize"]) * 1.45 / PT + 2
        r2, c2, s2 = fit_grid(fig, n, min_w, min_h, 0, reserve_h + below, cols)
        if s2 > shown:
            rows, cols_, shown = r2, c2, s2
            fig._fasttrack_legend_below = per_row
    cols = cols_
    axes = fig.subplots(rows, cols, sharex=sharex, sharey=sharey, squeeze=False).ravel()
    for ax in axes[shown:]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:shown]):
        if sharex and i + cols < shown:
            ax.tick_params(labelbottom=False)
            ax.xaxis.label.set_visible(False)
        if sharey and i % cols:
            ax.tick_params(labelleft=False)
            ax.yaxis.label.set_visible(False)
    if shown < n:
        note(fig, f"showing {shown} of {n} {what} (make the panel larger, or choose fewer, to see the rest)")
    return list(axes[:shown])


def outer(axes: list, cols: int | None = None) -> None:
    """Hide tick labels between plots that share axes (for grids made without `grid`)."""
    for ax in axes:
        ss = ax.get_subplotspec()
        if ss is None:
            continue
        if not ss.is_last_row():
            ax.tick_params(labelbottom=False)
        if not ss.is_first_col():
            ax.tick_params(labelleft=False)


def legend(fig, handles: list, labels: list | None = None, title: str | None = None) -> None:
    """One legend for the panel, outside the plots on the right, in columns so that it fits the height."""
    if not handles:
        return
    labels = labels if labels is not None else [h.get_label() for h in handles]
    below = getattr(fig, "_fasttrack_legend_below", 0)
    if below:
        _xlabel_to_axes(fig)         # constrained layout stacks a lower legend over a figure-wide x label
        fig.legend(handles, labels, loc="outside lower center", ncol=min(below, len(handles)), title=title, frameon=False,
                   handlelength=1.3, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.2, labelspacing=0.3)
        return
    _, h = size_mm(fig)
    size = font_pt(matplotlib.rcParams["legend.fontsize"])
    per_col = max(1, int((h * PT * 0.85) / (size * 1.45)) - (1 if title else 0))
    fig.legend(handles, labels, loc="outside right upper", ncol=math.ceil(len(handles) / per_col), title=title,
               frameon=False, handlelength=1.3, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2,
               labelspacing=0.3, title_fontsize=matplotlib.rcParams["legend.fontsize"], alignment="left")


def _xlabel_to_axes(fig) -> None:
    """Move the panel's shared x label onto the middle plot of the bottom row."""
    sup = getattr(fig, "_supxlabel", None)
    if sup is None or not sup.get_text():
        return
    bottom = [ax for ax in fig.axes if ax.get_visible() and ax.get_subplotspec() is not None
              and ax.get_subplotspec().is_last_row()]
    if not bottom:
        return
    bottom.sort(key=lambda ax: ax.get_subplotspec().colspan.start)
    bottom[len(bottom) // 2].set_xlabel(sup.get_text())
    sup.remove()
    fig._supxlabel = None


def line_handles(names: list[str], colors: list[str], ls: str = "-", marker: str | None = None) -> list:
    return [Line2D([], [], color=c, ls=ls if marker is None else "none", marker=marker, label=n,
                   lw=matplotlib.rcParams["lines.linewidth"]) for n, c in zip(names, colors)]


def style_handles(items: list[tuple[str, dict]]) -> list:
    """Black legend entries that explain line styles or markers: [(label, Line2D kwargs)]."""
    return [Line2D([], [], color=kw.pop("color", "k"), label=label, **kw) for label, kw in items]


def pick(conditions: list[str], show: list[str] | None) -> list[int]:
    """Indices of the conditions to draw: those named in `show` (in the report's order), else all."""
    if not show:
        return list(range(len(conditions)))
    idx = [i for i, c in enumerate(conditions) if c in set(show)]
    return idx or list(range(len(conditions)))


def condition_ticks(ax, names: list[str], positions=None) -> None:
    """Condition names on a categorical axis, turned only as far as they need to be to fit."""
    positions = np.arange(len(names)) if positions is None else positions
    w_mm = ax.get_position().width * size_mm(ax.figure)[0]
    space = w_mm * PT / max(len(names), 1)
    longest = max((len(n) for n in names), default=1) * 0.55 * font_pt(matplotlib.rcParams["xtick.labelsize"])
    if longest <= space * 0.95:
        ax.set_xticks(positions, names)
    elif font_pt(matplotlib.rcParams["xtick.labelsize"]) * 1.3 * 1.414 <= space:
        ax.set_xticks(positions, names, rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticks(positions, names, rotation=90)


def log_axis(ax, which: str = "y") -> None:
    """A log axis with plain labels at the decades only (no 6x10^1 clutter between them). Values that span less
    than a factor of 4 stay on a linear axis, where a log scale only adds unreadable ticks."""
    from matplotlib.ticker import FormatStrFormatter, LogFormatterSciNotation, NullFormatter
    lim = ax.dataLim
    lo, hi = (lim.y0, lim.y1) if which == "y" else (lim.x0, lim.x1)
    if lo > 0 and np.isfinite(hi) and hi / lo < 4:
        return
    axis = ax.yaxis if which == "y" else ax.xaxis
    (ax.set_yscale if which == "y" else ax.set_xscale)("log")
    axis.set_minor_formatter(NullFormatter())
    if lo > 0 and hi / lo < 30:
        axis.set_major_formatter(FormatStrFormatter("%.3g"))
    else:
        axis.set_major_formatter(LogFormatterSciNotation())


def log_colorbar(cb) -> None:
    """Readable ticks on a log colour bar: decades when it spans many, else a few plain numbers."""
    from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
    axis = cb.ax.yaxis if cb.orientation == "vertical" else cb.ax.xaxis
    axis.set_minor_formatter(NullFormatter())
    lo, hi = cb.mappable.norm.vmin, cb.mappable.norm.vmax
    if lo and hi and lo > 0 and hi / lo < 30:
        axis.set_major_locator(LogLocator(subs=(1, 2, 3, 5)))
        axis.set_major_formatter(FormatStrFormatter("%.3g"))


def duration(value: float, unit: str) -> str:
    """A time in the unit that reads best: minutes become hours above two hours."""
    if not np.isfinite(value):
        return "–"
    if unit == "min" and value >= 120:
        return f"{value / 60:.3g} h"
    return f"{value:.3g} {unit}"
