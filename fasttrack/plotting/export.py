"""Multi-panel figures at journal sizes, from figure definitions in the experiment file (figures.exports).

A definition is a page (journal, column width), a grid of columns, and panels that flow into it left to right.
A panel is a quick figure (its full spec is stored in the panel, so the definition is self-contained), a fate
view or a dynamics view. Each panel is drawn into its own matplotlib SubFigure under the journal type rules,
gets a letter, and gives up its title to the caption file written next to the figure.

Definitions can be saved as templates (YAML) and applied to other experiments: condition names that do not
exist there are dropped, and the notes say what changed."""
from __future__ import annotations

import copy
import csv
import os
from pathlib import Path

import yaml
from matplotlib.figure import Figure

from ..experiment import EXPORT_DEFAULT, FIGURE_DEFAULT, PANEL_DEFAULT, merge, normalise
from . import style

SOURCES = ["figure", "fate", "dynamics"]
SOURCE_TEXT = {"figure": "Quick figure", "fate": "Fate view", "dynamics": "Dynamics view"}


def blank_export() -> dict:
    return copy.deepcopy(EXPORT_DEFAULT)


def blank_panel(source: str = "figure") -> dict:
    p = copy.deepcopy(PANEL_DEFAULT)
    p["source"] = source
    return p


def _norm(export: dict) -> dict:
    out = merge(EXPORT_DEFAULT, export or {})
    out["panels"] = [merge(PANEL_DEFAULT, p) for p in out["panels"]]
    for p in out["panels"]:
        p["figure"] = merge(FIGURE_DEFAULT, p["figure"])
    return out


# ---------------------------------------------------------------- layout
def layout(export: dict) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Panels flow into the grid left to right: [(panel index, row, column, span)] and each row's height (mm)."""
    export = _norm(export)
    cols = max(1, int(export["columns"]))
    placed, heights = [], []
    row, col = 0, 0
    for i, p in enumerate(export["panels"]):
        span = max(1, min(int(p["span"]), cols))
        if col + span > cols:
            row, col = row + 1, 0
        if row >= len(heights):
            heights.append(0.0)
        heights[row] = max(heights[row], float(p["height"]) or 45.0)
        placed.append((i, row, col, span))
        col += span
    return placed, heights


def page_size(export: dict) -> tuple[float, float]:
    """(width, height) in mm."""
    export = _norm(export)
    _, heights = layout(export)
    return style.width_mm(export["journal"], export["width"]), float(sum(heights)) or 45.0


def label_case(export: dict) -> str:
    if export["labels"] in ("lower", "upper", "none"):
        return export["labels"]
    return style.JOURNALS.get(export["journal"], style.JOURNALS["nature"])["labels"]


# ---------------------------------------------------------------- drawing
def draw_panel(exp: dict, panel: dict, sub, cache=None) -> dict:
    """Draw one panel into a SubFigure. Returns the numbers behind it ({column: values}), if any."""
    source = panel["source"]
    if source == "figure":
        from ..analysis.figures import is_empty, render
        spec = copy.deepcopy(panel["figure"])
        spec["plot"]["title"] = ""
        _, table = render(exp, spec, figure=sub, cache=cache, styled=False)
        if is_empty(sub):
            raise ValueError("no results for these conditions yet")
        return table
    if source == "fate":
        from ..analysis.fate import VIEWS, draw, read_fate
        report = read_fate(exp, panel["ref"])
        if report is None:
            raise ValueError(f"no fate analysis called {panel['ref']!r} has been run")
        draw(sub, report, panel["view"] or next(iter(VIEWS)), exp, show=panel["conditions"])
        _raise_if_empty(sub)
        return {}
    if source == "dynamics":
        from ..dynamics.fit import read_fit, read_states
        from ..dynamics.plots import VIEWS, draw
        report = read_fit(exp, panel["ref"])
        if report is None:
            raise ValueError(f"no dynamics fit called {panel['ref']!r} has been run")
        draw(sub, report, panel["view"] or next(iter(VIEWS)), exp, read_states(exp, panel["ref"]), int(panel["condition"]),
             show=panel["conditions"])
        _raise_if_empty(sub)
        return {}
    raise ValueError(f"unknown panel source {source!r}")


def _raise_if_empty(sub) -> None:
    """A view with nothing to draw (no recovery check, no target...) is an error on a page, not a blank panel."""
    if getattr(sub, "_fasttrack_empty", False):
        raise ValueError(getattr(sub, "_fasttrack_message", "nothing to draw"))


def _take_title(sub) -> str:
    """The panel's own title moves to the caption: on a page, the letter heads the panel instead."""
    title = getattr(sub, "_suptitle", None)
    text = title.get_text() if title is not None else ""
    if title is not None:
        title.remove()
        sub._suptitle = None
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    rest = " ".join(lines[1:])          # the notes, wrapped over several lines on screen
    return lines[0] + (". " + rest[:1].upper() + rest[1:] if rest else "")


def render_export(exp: dict, export: dict, cache=None, dpi: float = 100) -> tuple[Figure, dict]:
    """Draw a figure definition. Returns the Figure and {captions: [(letter, text)], tables: {letter: table},
    notes: [str]} - notes say what went wrong or breaks the standard."""
    exp = normalise(exp)
    export = _norm(export)
    if export["palette"]:
        exp["figures"]["palette"] = export["palette"]
    if cache is None:
        from ..analysis.figures import ResultsCache
        cache = ResultsCache(exp)
    placed, heights = layout(export)
    width, height = page_size(export)
    notes = []
    journal = style.JOURNALS.get(export["journal"], style.JOURNALS["nature"])
    if height > journal["max_height"]:
        notes.append(f"The page is {height:.0f} mm tall; {export['journal']} allows {journal['max_height']} mm with the legend.")
    case = label_case(export)
    captions, tables = [], {}
    with style.context("journal", font=export["font"], line=export["line"]):
        fig = Figure(figsize=(width * style.MM, height * style.MM), dpi=dpi, layout="constrained")
        if not placed:
            fig.text(0.5, 0.5, "Add panels to this figure.", ha="center", va="center")
            return fig, {"captions": [], "tables": {}, "notes": ["No panels."]}
        cols = max(1, int(export["columns"]))
        grid = fig.add_gridspec(len(heights), cols, height_ratios=heights)
        size = style.label_size("journal", export["font"])
        for n, (i, row, col, span) in enumerate(placed):
            panel = export["panels"][i]
            sub = fig.add_subfigure(grid[row, col:col + span])
            sub._fasttrack_notes = []
            letter = panel["label"] or ("" if case == "none" else style.panel_letter(n, case))
            try:
                table = draw_panel(exp, panel, sub, cache)
            except (KeyError, ValueError, OSError, IndexError) as exc:
                sub.clear()
                sub.text(0.5, 0.5, str(exc).strip("'\""), ha="center", va="center", wrap=True, color="#AA3333")
                notes.append(f"Panel {letter or n + 1}: {str(exc).strip(chr(39))}")
                table = {}
            text = _take_title(sub)
            if panel["caption"]:
                text = (text + ". " if text else "") + panel["caption"]
            captions.append((letter, text))
            if table:
                tables[letter or str(n + 1)] = table
            if letter:
                sub.suptitle(letter, x=0.0, ha="left", fontweight="bold", fontsize=size)
        notes += check(fig)      # draws the page: ticks are made now, under the journal sizes
    return fig, {"captions": captions, "tables": tables, "notes": notes}


def save_page(fig: Figure, export: dict, target, **kwargs) -> None:
    """Save under the page's own type settings (tick sizes and TrueType fonts are fixed when saving)."""
    export = _norm(export)
    with style.context("journal", font=export["font"], line=export["line"]):
        fig.savefig(target, **kwargs)


def check(fig: Figure) -> list[str]:
    """What breaks the standard on a drawn page: text below 5 pt, and text cut off at the page edge."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    page = fig.bbox
    small, outside = set(), set()

    def off(box) -> bool:
        return box.x0 < page.x0 - 1 or box.y0 < page.y0 - 1 or box.x1 > page.x1 + 1 or box.y1 > page.y1 + 1

    for t in fig.findobj(Text):
        s = t.get_text().strip()
        if s and t.get_visible() and t.get_fontsize() < 4.95 and (t.axes is None or t.axes.get_visible()):
            small.add(s[:30])
    figures = fig.findobj(lambda a: hasattr(a, "add_subplot"))     # the page and its panels
    for f in figures:          # texts that belong to no axes: titles, axis labels of panels, figure legends
        for t in list(f.texts) + [x for x in (getattr(f, "_suptitle", None), getattr(f, "_supxlabel", None),
                                              getattr(f, "_supylabel", None)) if x is not None]:
            if t.get_visible() and t.get_text().strip() and off(t.get_window_extent(renderer)):
                outside.add(t.get_text().strip()[:30])
        for lg in f.legends:
            if off(lg.get_window_extent(renderer)):
                outside.add("a legend")
    for ax in fig.findobj(lambda a: hasattr(a, "get_tightbbox") and hasattr(a, "xaxis")):
        if not ax.get_visible():
            continue
        box = ax.get_tightbbox(renderer)       # only what is drawn: ticks in range, labels, title, legend, notes
        if box is not None and off(box):
            name = ax.get_title(loc="left") or ax.get_title() or ax.get_ylabel() or "a plot"
            letter = getattr(ax.figure, "_suptitle", None)
            where = f"panel {letter.get_text()}: " if letter is not None and len(letter.get_text()) <= 3 else ""
            outside.add(where + " ".join(name.split())[:30])
    notes = []
    if small:
        notes.append(f"Text below 5 pt: {', '.join(sorted(small)[:4])}{'…' if len(small) > 4 else ''}")
    if outside:
        notes.append(f"Text cut off at the page edge: {', '.join(sorted(outside)[:4])}{'…' if len(outside) > 4 else ''}")
    return notes


def caption_text(export: dict, captions: list[tuple[str, str]]) -> str:
    lines = [export["name"] or "Figure", ""]
    for letter, text in captions:
        lines.append(f"({letter}) {text}" if letter else text)
    return "\n".join(lines) + "\n"


def write_export(exp: dict, export: dict, cache=None, on_event=lambda e: None) -> list[Path]:
    """Save one figure definition: the page in every format, its caption (.txt) and each panel's numbers (.csv)."""
    from ..analysis.figures import safe_name
    exp = normalise(exp)
    export = _norm(export)
    folder = Path(exp["paths"]["results"]) / exp["figures"]["export_folder"]
    folder.mkdir(parents=True, exist_ok=True)
    name = safe_name(export["name"] or "figure")
    fig, info = render_export(exp, export, cache, dpi=export["dpi"])
    written = []
    with style.context("journal", font=export["font"], line=export["line"]):   # TrueType fonts are chosen at save time
        for fmt in export["formats"]:
            path = folder / f"{name}.{fmt}"
            fig.savefig(path, dpi=export["dpi"])
            written.append(path)
    path = folder / f"{name}.txt"
    path.write_text(caption_text(export, info["captions"]) + ("\nNotes:\n" + "\n".join(info["notes"]) + "\n" if info["notes"] else ""))
    written.append(path)
    for letter, table in info["tables"].items():
        path = folder / f"{name}_{letter}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            columns = list(table)
            writer.writerow(columns)
            for row in range(max(len(v) for v in table.values())):
                writer.writerow([table[c][row] if row < len(table[c]) else "" for c in columns])
        written.append(path)
    for note in info["notes"]:
        on_event({"kind": "export_note", "name": name, "message": note})
    return written


def write_all(exp: dict, names: list[str] | None = None, on_event=lambda e: None) -> list[Path]:
    from ..analysis.figures import ResultsCache
    exp = normalise(exp)
    cache = ResultsCache(exp)
    out = []
    for export in exp["figures"]["exports"]:
        if (names and export["name"] not in names) or (not names and not export["enabled"]):
            continue
        out += write_export(exp, export, cache, on_event)
        on_event({"kind": "export_done", "name": export["name"]})
    return out


# ---------------------------------------------------------------- building definitions
def from_quick_figures(exp: dict, compare: str | None = None, columns: int = 2) -> dict:
    """A definition with every enabled quick figure as a panel (heatmaps span the page)."""
    exp = normalise(exp)
    export = blank_export()
    export["name"] = "Quick figures"
    export["columns"] = columns
    for spec in exp["figures"]["list"]:
        if not spec["enabled"]:
            continue
        panel = blank_panel("figure")
        panel["figure"] = copy.deepcopy(spec)
        if compare and spec["kind"] in ("timecourse", "percent_high", "cell_count", "event_rate"):
            panel["figure"]["compare"] = compare
        if spec["kind"] in ("heatmap", "track_heatmap"):
            panel["span"], panel["height"] = columns, 60.0
        export["panels"].append(panel)
    return export


# ---------------------------------------------------------------- templates
def template_dir() -> Path:
    return Path(os.environ.get("FASTTRACK_TEMPLATES") or Path.home() / ".config" / "fasttrack" / "figure_templates")


def save_template(export: dict, path: str | Path | None = None) -> Path:
    """Write a definition as a template. Without a path it goes to the user's template folder, named after it."""
    from ..analysis.figures import safe_name
    export = _norm(export)
    path = Path(path) if path else template_dir() / f"{safe_name(export['name'] or 'template')}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"fasttrack_figure_template": 1, "export": export}, f, sort_keys=False, allow_unicode=True)
    return path


def list_templates() -> list[tuple[str, Path]]:
    folder = template_dir()
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.yaml")):
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            out.append((data.get("export", {}).get("name") or path.stem, path))
        except (OSError, yaml.YAMLError):
            continue
    return out


def load_template(path: str | Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if "export" not in data:
        raise ValueError(f"{path} is not a FastTrack figure template")
    return _norm(data["export"])


def apply_template(exp: dict, export: dict) -> tuple[dict, list[str]]:
    """Fit a template to this experiment. Conditions it names that are not here are dropped (all conditions when
    none is left), groups and references that are missing are reported; nothing else changes."""
    exp = normalise(exp)
    export = _norm(copy.deepcopy(export))
    have = [c["name"] for c in exp["plate"]["conditions"]]
    groups = {g["name"] for g in exp["analysis"]["groups"]}
    fates = {f["name"] for f in exp["analysis"]["fates"]}
    fits = {f["name"] for f in exp["analysis"]["dynamics"]["fits"]}
    notes = []
    for n, panel in enumerate(export["panels"], 1):
        spec = panel["figure"]
        if panel["source"] == "figure":
            missing = [c for c in spec["conditions"] if c not in have]
            if missing:
                spec["conditions"] = [c for c in spec["conditions"] if c in have]
                notes.append(f"Panel {n}: not in this experiment, left out: {', '.join(missing)}"
                             + ("; drawing every condition instead" if not spec["conditions"] else ""))
            if spec["group"] and spec["group"] not in groups:
                notes.append(f"Panel {n}: no group called {spec['group']!r} here; drawing every condition")
                spec["group"] = ""
            if spec["reference"] and spec["reference"] not in have:
                notes.append(f"Panel {n}: reference {spec['reference']!r} is not here; the first condition is used")
                spec["reference"] = ""
        if panel["source"] != "figure" and panel["conditions"]:
            missing = [c for c in panel["conditions"] if c not in have]
            if missing:
                panel["conditions"] = [c for c in panel["conditions"] if c in have]
                notes.append(f"Panel {n}: not in this experiment, left out: {', '.join(missing)}"
                             + ("; drawing every condition instead" if not panel["conditions"] else ""))
        if panel["source"] == "fate" and panel["ref"] not in fates:
            notes.append(f"Panel {n}: no fate analysis called {panel['ref']!r} in this experiment")
        elif panel["source"] == "dynamics" and panel["ref"] not in fits:
            notes.append(f"Panel {n}: no dynamics fit called {panel['ref']!r} in this experiment")
    return export, notes
