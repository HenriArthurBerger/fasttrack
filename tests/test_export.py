"""The figure standard as code: colours, comparison modes, multi-panel export, templates, CLI and the Export tab."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fasttrack.plotting import export as X
from fasttrack.plotting import style


def _exp(tmp_path):
    from test_fates import _experiment
    exp, _ = _experiment(tmp_path, T=60)
    exp["plate"]["conditions"] += [{"name": "siY", "wells": ["C01"], "color": "", "notes": ""}]
    exp["figures"]["list"] = [
        {"name": "PPARG", "kind": "timecourse", "conditions": ["Ctrl", "siX"], "unit": "cell",
         "metric": {"measure": "nuc_mean", "channel": "c2", "log2": True}},
        {"name": "Area", "kind": "distribution", "conditions": ["Ctrl", "siX"], "unit": "cell",
         "metric": {"measure": "nuc_area", "channel": ""}}]
    from fasttrack.experiment import normalise
    return normalise(exp)


def test_palettes_follow_the_number_of_conditions_and_the_design():
    names = ["Ctrl", "A", "B"]
    s = style.condition_styles({}, names, "auto")
    assert s["Ctrl"]["color"] == style.REFERENCE and [s[n]["color"] for n in "AB"] == style.TOL_HIGH_CONTRAST[:2]
    many = ["Ctrl"] + [f"c{i}" for i in range(8)]
    assert [style.condition_styles({}, many, "auto")[n]["color"] for n in many[1:]] == style.TOL_MUTED[:8]
    s = style.condition_styles({}, many[:6], "auto", reference="c2")
    assert s["c2"]["color"] == style.REFERENCE and s["Ctrl"]["color"] != style.REFERENCE
    f = style.condition_styles({}, ["Ctrl", "Tdi", "MekI", "Tdi + MekI"], "factorial")
    assert f["Tdi"]["ls"] == "-" and f["Tdi + MekI"]["ls"] == "--"
    assert f["Tdi + MekI"]["color"] not in (f["Tdi"]["color"], f["MekI"]["color"])
    exp = {"plate": {"conditions": [{"name": "Ctrl", "color": "#123456"}]}}
    assert style.condition_styles(exp, ["Ctrl"], "experiment")["Ctrl"]["color"] == "#123456"   # the Plate tab's own
    assert style.state_colors(5) == style.TOL_LIGHT[:5] and len(set(style.state_colors(12))) == 12
    assert not set(style.state_colors(8)) & set(style.TOL_BRIGHT + style.TOL_MUTED + style.TOL_HIGH_CONTRAST)
    assert style.panel_letter(0) == "a" and style.panel_letter(27, "upper") == "AB"
    assert style.width_mm("nature", "single") == 89 and style.width_mm("cell", "double") == 174 and style.width_mm("nature", 100) == 100


def test_comparison_modes_lay_out_small_multiples(tmp_path):
    from fasttrack.analysis.figures import ResultsCache, render
    exp = _exp(tmp_path)
    cache = ResultsCache(exp)
    spec = copy.deepcopy(exp["figures"]["list"][0])
    shown = lambda f: [ax for ax in f.axes if ax.get_visible() and ax.axison]
    fig, table = render(exp, spec, cache=cache)
    assert len(shown(fig)) == 1 and "Ctrl median" in table
    spec["compare"] = "reference"
    fig, _ = render(exp, spec, cache=cache)
    axes = shown(fig)
    assert len(axes) == 1 and axes[0].get_title(loc="left").startswith("siX")
    greys = [ln for ln in axes[0].get_lines() if ln.get_color() == style.CONTEXT]
    assert len(greys) == 1                                    # the reference behind, in grey
    spec["compare"] = "separate"
    fig, _ = render(exp, spec, cache=cache)
    assert len(shown(fig)) == 2 and fig._supylabel is not None
    spec["compare"], spec["plot"]["legend"] = "overlay", "direct"
    fig, _ = render(exp, spec, cache=cache)
    ax = shown(fig)[0]
    assert ax.get_legend() is None and any(t.get_text().startswith("siX") for t in ax.texts)
    assert "median and quartiles" in fig._suptitle.get_text()   # what the band is: in the title notes, not the axis


def test_export_page_letters_captions_and_files(tmp_path):
    exp = _exp(tmp_path)
    e = X.from_quick_figures(exp, compare="reference", columns=2)
    e["name"], e["formats"] = "Figure 1", ["pdf", "png"]
    fate = X.blank_panel("fate")
    fate.update(ref="nope", view="committor", span=2)
    e["panels"].append(fate)
    placed, heights = X.layout(e)
    assert [(r, c, s) for _, r, c, s in placed] == [(0, 0, 1), (0, 1, 1), (1, 0, 2)] and len(heights) == 2
    assert X.page_size(e) == (183.0, 45.0 + 65.0) or X.page_size(e)[0] == 183.0
    fig, info = X.render_export(exp, e)
    w, h = fig.get_size_inches() / style.MM
    assert abs(w - 183) < 0.5
    assert [c[0] for c in info["captions"]] == ["a", "b", "c"]
    assert info["captions"][0][1].startswith("PPARG") and "Median and quartiles" in info["captions"][0][1]
    assert any("nope" in n for n in info["notes"])            # a missing fate is reported, not a crash
    texts = [t for t in fig.findobj(lambda a: hasattr(a, "get_fontsize") and hasattr(a, "get_text")) if t.get_text().strip()]
    assert min(t.get_fontsize() for t in texts) >= 4.95   # nothing below 5 pt
    exp["figures"]["exports"] = [e]
    written = X.write_all(exp)
    names = {p.name for p in written}
    assert {"Figure_1.pdf", "Figure_1.png", "Figure_1.txt", "Figure_1_a.csv"} <= names
    txt = (Path(exp["paths"]["results"]) / "Figures/Export/Figure_1.txt").read_text()
    assert txt.startswith("Figure 1") and "(b) Area" in txt
    import matplotlib.backends.backend_pdf  # noqa: F401  (fonts embedded as TrueType)
    assert b"/FontFile2" in (Path(exp["paths"]["results"]) / "Figures/Export/Figure_1.pdf").read_bytes()


def test_templates_carry_to_another_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTTRACK_TEMPLATES", str(tmp_path / "templates"))
    exp = _exp(tmp_path)
    e = X.from_quick_figures(exp)
    e["name"] = "Overview"
    e["panels"][0]["figure"]["reference"] = "siX"
    path = X.save_template(e)
    assert [n for n, _ in X.list_templates()] == ["Overview"]
    other = copy.deepcopy(exp)
    other["plate"]["conditions"] = [{"name": "Ctrl", "wells": ["A01"], "color": "", "notes": ""},
                                    {"name": "drug", "wells": ["B01"], "color": "", "notes": ""}]
    got, notes = X.apply_template(other, X.load_template(path))
    assert got["panels"][0]["figure"]["conditions"] == ["Ctrl"] and got["panels"][0]["figure"]["reference"] == ""
    assert any("siX" in n for n in notes)
    fig, info = X.render_export(other, got)
    assert len(info["captions"]) == 2


def test_cli_exports(tmp_path, capsys):
    from fasttrack.cli import main
    from fasttrack.experiment import save_experiment
    exp = _exp(tmp_path)
    e = X.from_quick_figures(exp)
    e["name"], e["formats"] = "Fig", ["png"]
    exp["figures"]["exports"] = [e]
    save_experiment(exp, tmp_path / "exp.yaml")
    main(["export", str(tmp_path / "exp.yaml"), "Fig"])
    assert "exported Fig" in capsys.readouterr().out
    assert (Path(exp["paths"]["results"]) / "Figures/Export/Fig.png").exists()


def test_export_tab_builds_a_definition(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("FASTTRACK_TEMPLATES", str(tmp_path / "templates"))
    from PySide6.QtWidgets import QApplication
    from fasttrack.gui.app import MainWindow
    app = QApplication.instance() or QApplication([])
    exp = _exp(tmp_path)
    window = MainWindow()
    window.load(exp)
    a = window.analysis
    x = a.export
    a.tabs.setCurrentWidget(x)
    x.name.setText("Fig 2")
    x._add_panel("figure", figure=exp["figures"]["list"][0])
    x._add_panel("figure", figure=exp["figures"]["list"][1])
    x.panel_list.setCurrentRow(0)
    x.p_compare.setCurrentIndex(x.p_compare.findData("reference"))
    assert x.panels[0]["figure"]["compare"] == "reference" and x.panels[0]["figure"]["plot"]["legend"] == "direct"
    x.p_span.setValue(2)
    x.list.add()
    saved = window.collect()["figures"]["exports"]
    assert saved[-1]["name"] == "Fig 2" and saved[-1]["panels"][0]["span"] == 2
    assert exp["figures"]["list"][0]["compare"] == "overlay"   # the quick figure itself is untouched
    x._save_template()
    assert [n for n, _ in X.list_templates()] == ["Fig 2"]
    x._add_panel("fate", ref=exp["analysis"]["fates"][0]["name"], view="committor")
    x.panel_list.setCurrentRow(len(x.panels) - 1)
    assert x.p_conditions.isVisibleTo(x) and x.p_conditions.text() == "every condition"
    names = [a.text() for a in x.p_conditions.menu().actions() if a.isCheckable()]
    assert names == ["Ctrl", "siX", "siY"]                     # the fate analysis draws every condition
    next(a for a in x.p_conditions.menu().actions() if a.text() == "siX").setChecked(True)
    assert x.panels[-1]["conditions"] == ["siX"] and x.p_conditions.text() == "siX"
    x.rename_condition("siX", "siZ")
    assert x.panels[-1]["conditions"] == ["siZ"]
    window.collect()
    window.saved = copy.deepcopy(window.exp)
    window.close()
    app.processEvents()


def test_fate_and_dynamics_views_fit_journal_pages(tmp_path):
    """Every fate and states view, at single and double column width: nothing cut off or below 5 pt, titles in the
    caption, and a chosen subset of conditions drawn in their full-set colours."""
    from fasttrack.analysis.fate import VIEWS as FV, run_fate
    from fasttrack.dynamics.fit import run_fit
    from fasttrack.dynamics.plots import VIEWS as DV
    exp = _exp(tmp_path)
    exp["analysis"]["dynamics"]["device"] = "cpu"
    run_fate(exp, exp["analysis"]["fates"][0])
    fit = {"name": "disc", "model": "states", "features": ["PPARG nuc_mean", "ERK", "H2B nuc_sum"],
           "options": {"states": 3, "rank": 1, "division": True, "sweep": [2, 3], "flux_feature": "PPARG nuc_mean"},
           "steps": 30, "batch": 64, "heldout": 0.34}
    from fasttrack.experiment import normalise
    exp["analysis"]["dynamics"]["fits"] = [fit]
    exp = normalise(exp)
    run_fit(exp, exp["analysis"]["dynamics"]["fits"][0])
    fate = exp["analysis"]["fates"][0]["name"]
    jobs = [("fate", fate, v) for v in FV] + [("dynamics", "disc", v) for v in DV if v != "recovery"]
    for source, ref, view in jobs:
        for width, height in (("single", 60), ("double", 70)):
            e = X.blank_export()
            e["width"], e["columns"] = width, 1
            p = X.blank_panel(source)
            p.update(ref=ref, view=view, height=height)
            e["panels"] = [p]
            fig, info = X.render_export(exp, e)
            assert info["notes"] == [], (view, width, info["notes"])
            assert info["captions"][0][1], (view, width)              # the view's title went to the caption
    # an empty view is reported on the page, not exported blank
    e = X.blank_export()
    p = X.blank_panel("dynamics")
    p.update(ref="disc", view="recovery")
    e["panels"] = [p]
    assert any("recovery" in n.lower() for n in X.render_export(exp, e)[1]["notes"])
    # a subset of conditions keeps each condition's colour from the full set
    from matplotlib.figure import Figure
    from fasttrack.analysis.fate import draw, read_fate
    report = read_fate(exp, fate)
    full, part = Figure(figsize=(7, 4)), Figure(figsize=(7, 4))
    draw(full, report, "threshold", exp)
    draw(part, report, "threshold", exp, show=[report["conditions"][1]])
    colour = lambda f: [ln.get_color() for ln in f.axes[0].get_lines() if ln.get_linestyle() == "-"]
    assert colour(part) == colour(full)[1:2]
