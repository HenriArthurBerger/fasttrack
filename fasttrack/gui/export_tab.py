"""Export: multi-panel figures at journal sizes, kept in the experiment file and reusable as templates.

A figure definition is a page (journal, column width, grid columns, type sizes, colours) and a list of panels.
A panel is a copy of a quick figure (so it can be compared differently, e.g. reference in grey, without
touching the quick figure), a fate view or a dynamics view. The preview is the exported page itself, drawn at
its real size and scaled to fit, with a list of what breaks the figure standard."""
from __future__ import annotations

import copy
import io

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QSplitter, QToolButton, QVBoxLayout, QWidget)

from ..experiment import COMPARE_MODES, EXPORT_DEFAULT, LEGEND_PLACES
from ..plotting import export as X
from ..plotting.style import JOURNALS, PALETTE_TEXT, PALETTES, WIDTHS
from .analysis_tab import COMPARE_TEXT, LINE_KINDS, _NamedList
from .worker import FunctionWorker

WIDTH_TEXT = {"single": "single column", "onehalf": "1.5 columns", "double": "double column"}
DEFAULT_HEIGHT = {"figure": 45.0, "fate": 65.0, "dynamics": 65.0}


class ExportPanel(QWidget):
    changed = Signal()

    def __init__(self, get_experiment, get_cache=None):
        super().__init__()
        self.get_experiment = get_experiment
        self.get_cache = get_cache
        self.panels: list[dict] = []
        self.worker = None
        self.pending = None
        self._loading = False
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        # ------------------------------------------------ definition editor
        editor = QWidget()
        el = QVBoxLayout(editor)
        el.setContentsMargins(0, 0, 0, 0)
        page = QGroupBox("Page")
        form = QFormLayout(page)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Figure 2")
        form.addRow("Name", self.name)
        size_row = QHBoxLayout()
        self.journal = QComboBox()
        for j in JOURNALS:
            self.journal.addItem(j.capitalize(), j)
        self.width = QComboBox()
        for w in WIDTHS:
            self.width.addItem(WIDTH_TEXT[w], w)
        self.width.addItem("custom (mm)", "custom")
        self.width_mm = QDoubleSpinBox()
        self.width_mm.setRange(30, 400)
        self.width_mm.setSuffix(" mm")
        self.width_mm.setValue(183)
        for w in (self.journal, self.width, self.width_mm):
            size_row.addWidget(w)
        form.addRow("Journal, width", size_row)
        grid_row = QHBoxLayout()
        self.columns = QSpinBox()
        self.columns.setRange(1, 8)
        self.columns.setPrefix("columns ")
        self.labels = QComboBox()
        for k, t in (("auto", "letters as the journal"), ("lower", "a, b, c"), ("upper", "A, B, C"), ("none", "no letters")):
            self.labels.addItem(t, k)
        grid_row.addWidget(self.columns)
        grid_row.addWidget(self.labels, 1)
        form.addRow("Grid", grid_row)
        type_row = QHBoxLayout()
        self.font = QDoubleSpinBox()
        self.font.setRange(5, 12)
        self.font.setSingleStep(0.5)
        self.font.setSuffix(" pt text")
        self.font.setToolTip("Labels at this size, ticks and legends 6/7 of it (never below 5 pt). Nature: 5 to 7 pt.")
        self.line = QDoubleSpinBox()
        self.line.setRange(0.25, 3)
        self.line.setSingleStep(0.25)
        self.line.setSuffix(" pt lines")
        self.line.setToolTip("Data lines. Axes are 0.5 pt. Nothing may be thinner than 0.25 pt in print.")
        type_row.addWidget(self.font)
        type_row.addWidget(self.line)
        form.addRow("Type", type_row)
        self.palette = QComboBox()
        self.palette.addItem("as in Settings", "")
        for k in PALETTES:
            self.palette.addItem(PALETTE_TEXT[k], k)
        form.addRow("Colours", self.palette)
        out_row = QHBoxLayout()
        self.formats = QLineEdit()
        self.formats.setToolTip("Comma separated. PDF keeps text editable (TrueType fonts) for journals and Illustrator.")
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setSuffix(" dpi")
        out_row.addWidget(self.formats, 1)
        out_row.addWidget(self.dpi)
        form.addRow("Files", out_row)
        el.addWidget(page)

        panels_box = QGroupBox("Panels (drag to reorder; they fill the grid left to right)")
        pl = QVBoxLayout(panels_box)
        self.panel_list = QListWidget()
        self.panel_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.panel_list.model().rowsMoved.connect(self._panels_moved)
        self.panel_list.currentRowChanged.connect(self._show_panel)
        self.panel_list.setMaximumHeight(160)
        pl.addWidget(self.panel_list)
        add_row = QHBoxLayout()
        self.add_buttons = {}
        for source in X.SOURCES:
            b = QToolButton()
            b.setText(f"+ {X.SOURCE_TEXT[source]}")
            b.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            b.setMenu(QMenu(b))
            b.menu().aboutToShow.connect(lambda s=source: self._fill_add_menu(s))
            self.add_buttons[source] = b
            add_row.addWidget(b)
        for text, slot in (("Duplicate", self._duplicate_panel), ("Remove", self._remove_panel)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            add_row.addWidget(b)
        pl.addLayout(add_row)

        self.panel_box = QGroupBox("Selected panel")
        pf = QFormLayout(self.panel_box)
        self.p_source = QLabel("")
        self.p_source.setWordWrap(True)
        pf.addRow(self.p_source)
        place_row = QHBoxLayout()
        self.p_span = QSpinBox()
        self.p_span.setRange(1, 8)
        self.p_span.setPrefix("spans ")
        self.p_span.setSuffix(" col")
        self.p_height = QDoubleSpinBox()
        self.p_height.setRange(15, 250)
        self.p_height.setSuffix(" mm tall")
        self.p_label = QLineEdit()
        self.p_label.setPlaceholderText("letter: auto")
        self.p_label.setMaximumWidth(90)
        for w in (self.p_span, self.p_height, self.p_label):
            place_row.addWidget(w)
        pf.addRow("Place", place_row)
        self.p_compare = QComboBox()
        for m in COMPARE_MODES:
            self.p_compare.addItem(COMPARE_TEXT[m], m)
        compare_row = QHBoxLayout()
        self.p_reference = QComboBox()
        self.p_cols = QSpinBox()
        self.p_cols.setRange(0, 12)
        self.p_cols.setSpecialValueText("auto")
        self.p_cols.setPrefix("per row ")
        compare_row.addWidget(self.p_compare, 1)
        compare_row.addWidget(self.p_reference, 1)
        compare_row.addWidget(self.p_cols)
        pf.addRow("Compare", compare_row)
        self.p_legend = QComboBox()
        self.p_legend.addItems(LEGEND_PLACES)
        self.p_legend.setToolTip("direct: each line labelled at its end, no legend box")
        pf.addRow("Legend", self.p_legend)
        view_row = QHBoxLayout()
        self.p_ref = QComboBox()
        self.p_view = QComboBox()
        self.p_condition = QSpinBox()
        self.p_condition.setRange(0, 100)
        self.p_condition.setPrefix("condition ")
        self.p_condition.setToolTip("Example tracks view: which condition (0 = the first).")
        view_row.addWidget(self.p_ref, 1)
        view_row.addWidget(self.p_view, 2)
        view_row.addWidget(self.p_condition)
        pf.addRow("View", view_row)
        self.panel_form = pf
        self.p_conditions = QToolButton()
        self.p_conditions.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.p_conditions.setMenu(QMenu(self.p_conditions))
        self.p_conditions.setToolTip("Which conditions this view draws. Colours stay those of the full set, "
                                     "so a condition looks the same in every panel.")
        pf.addRow("Conditions", self.p_conditions)
        self.p_caption = QLineEdit()
        self.p_caption.setPlaceholderText("added to the generated caption")
        pf.addRow("Caption", self.p_caption)
        self.p_recopy = QPushButton("Copy again from the quick figure")
        self.p_recopy.setToolTip("Replace this panel's figure settings with the quick figure of the same name "
                                 "(keeps its place, comparison and legend).")
        self.p_recopy.clicked.connect(self._recopy)
        pf.addRow(self.p_recopy)
        for w in (self.p_span, self.p_cols, self.p_condition):
            w.valueChanged.connect(self._panel_edited)
        self.p_height.valueChanged.connect(self._panel_edited)
        for w in (self.p_compare, self.p_reference, self.p_legend, self.p_view):
            w.currentIndexChanged.connect(self._panel_edited)
        self.p_ref.currentIndexChanged.connect(self._ref_changed)
        for w in (self.p_label, self.p_caption):
            w.editingFinished.connect(self._panel_edited)
        pl.addWidget(self.panel_box)
        el.addWidget(panels_box, 1)

        for w in (self.journal, self.width, self.labels, self.palette):
            w.currentIndexChanged.connect(self._page_edited)
        for w in (self.columns, self.dpi):
            w.valueChanged.connect(self._page_edited)
        for w in (self.width_mm, self.font, self.line):
            w.valueChanged.connect(self._page_edited)

        self.list = _NamedList(editor, self._read, self._write, EXPORT_DEFAULT, "figure", checkable=True)
        self.list.changed.connect(self.changed.emit)
        self.list.list.currentRowChanged.connect(lambda _: self.preview())
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(4, 4, 4, 4)
        tpl_row = QHBoxLayout()
        self.templates = QToolButton()
        self.templates.setText("Templates")
        self.templates.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.templates.setMenu(QMenu(self.templates))
        self.templates.menu().aboutToShow.connect(self._fill_templates)
        tpl_row.addWidget(self.templates)
        tpl_row.addStretch(1)
        sl.addLayout(tpl_row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.list)
        sl.addWidget(scroll, 1)
        run_row = QHBoxLayout()
        for text, slot in (("Preview", self.preview), ("Export selected", self.export_selected),
                           ("Export all ticked", self.export_all)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            run_row.addWidget(b)
        sl.addLayout(run_row)
        side.setMinimumWidth(540)
        side.setMaximumWidth(700)
        split.addWidget(side)

        # ------------------------------------------------ preview
        view = QWidget()
        vl = QVBoxLayout(view)
        self.size_label = QLabel("")
        vl.addWidget(self.size_label)
        self.image = QLabel("Preview appears here.")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(300, 200)
        self.image.setStyleSheet("background: palette(mid);")
        vl.addWidget(self.image, 1)
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        vl.addWidget(self.notes)
        self.caption = QPlainTextEdit()
        self.caption.setReadOnly(True)
        self.caption.setMaximumHeight(130)
        self.caption.setPlaceholderText("The caption written next to the figure: each panel's title and what its lines, "
                                        "bands and tests are.")
        vl.addWidget(self.caption)
        split.addWidget(view)
        split.setSizes([600, 900])
        self._pixmap = None
        self._write(copy.deepcopy(EXPORT_DEFAULT))

    # ------------------------------------------------ definition <-> widgets
    def _read(self) -> dict:
        width = self.width.currentData()
        return {"name": self.name.text().strip(), "journal": self.journal.currentData(),
                "width": self.width_mm.value() if width == "custom" else width, "columns": self.columns.value(),
                "labels": self.labels.currentData(), "font": self.font.value(), "line": self.line.value(),
                "palette": self.palette.currentData() or "",
                "formats": [x.strip().lstrip(".") for x in self.formats.text().split(",") if x.strip()] or ["pdf"],
                "dpi": self.dpi.value(), "panels": copy.deepcopy(self.panels)}

    def _write(self, export: dict) -> None:
        export = X._norm(export)
        self._loading = True
        self.name.setText(export["name"])
        self.journal.setCurrentIndex(max(0, self.journal.findData(export["journal"])))
        if isinstance(export["width"], (int, float)):
            self.width.setCurrentIndex(self.width.findData("custom"))
            self.width_mm.setValue(float(export["width"]))
        else:
            self.width.setCurrentIndex(max(0, self.width.findData(export["width"])))
        self.columns.setValue(int(export["columns"]))
        self.labels.setCurrentIndex(max(0, self.labels.findData(export["labels"])))
        self.font.setValue(float(export["font"]))
        self.line.setValue(float(export["line"]))
        self.palette.setCurrentIndex(max(0, self.palette.findData(export["palette"])))
        self.formats.setText(", ".join(export["formats"]))
        self.dpi.setValue(int(export["dpi"]))
        self.panels = copy.deepcopy(export["panels"])
        self._loading = False
        self._fill_panels()
        self._update_width()

    def _page_edited(self, *_) -> None:
        self._update_width()
        if not self._loading:
            self._commit()

    def _update_width(self) -> None:
        self.width_mm.setVisible(self.width.currentData() == "custom")
        self.p_span.setMaximum(self.columns.value())

    def _commit(self) -> None:
        """Edits apply to the selected definition at once (as on the other analysis tabs after Update)."""
        row = self.list.list.currentRow()
        if 0 <= row < len(self.list.items) and self.list.items[row]["name"] == self.name.text().strip():
            item = self._read()
            item["enabled"] = self.list.items[row].get("enabled", True)
            self.list.items[row] = item
            self.changed.emit()
        self.preview()

    # ------------------------------------------------ panels
    def _panel_text(self, i: int, p: dict) -> str:
        if p["source"] == "figure":
            f = p["figure"]
            what = f"{f['name'] or f['kind']}"
            if f["kind"] in LINE_KINDS:
                what += f" ({COMPARE_TEXT[f['compare']]})"
        else:
            what = f"{p['ref']}: {p['view']}"
        return f"{X.SOURCE_TEXT[p['source']]} · {what} · {p['span']} col × {p['height']:.0f} mm"

    def _fill_panels(self, select: int | None = None) -> None:
        self.panel_list.blockSignals(True)
        current = self.panel_list.currentRow() if select is None else select
        self.panel_list.clear()
        for i, p in enumerate(self.panels):
            self.panel_list.addItem(QListWidgetItem(self._panel_text(i, p)))
        self.panel_list.blockSignals(False)
        if self.panels:
            self.panel_list.setCurrentRow(min(max(current, 0), len(self.panels) - 1))
        self._show_panel(self.panel_list.currentRow())

    def _panels_moved(self, *_) -> None:
        texts = [self.panel_list.item(i).text() for i in range(self.panel_list.count())]
        pool = list(zip([self._panel_text(i, p) for i, p in enumerate(self.panels)], self.panels))
        order = []
        for t in texts:
            k = next(j for j, (pt, _) in enumerate(pool) if pt == t)
            order.append(pool.pop(k)[1])
        self.panels = order
        self._commit()

    def _fill_add_menu(self, source: str) -> None:
        menu = self.add_buttons[source].menu()
        menu.clear()
        exp = self.get_experiment()
        if source == "figure":
            figures = exp["figures"]["list"]
            if not figures:
                menu.addAction("(no quick figures yet: make one on the Figures tab)").setEnabled(False)
            for f in figures:
                menu.addAction(f["name"] or f["kind"], lambda f=f: self._add_panel("figure", figure=f))
            if figures:
                menu.addSeparator()
                menu.addAction("Every quick figure", self._add_all_quick)
            return
        from ..analysis.fate import VIEWS as FATE_VIEWS
        from ..dynamics.plots import VIEWS as FIT_VIEWS
        names = [f["name"] for f in (exp["analysis"]["fates"] if source == "fate" else exp["analysis"]["dynamics"]["fits"])]
        views = FATE_VIEWS if source == "fate" else FIT_VIEWS
        if not names:
            menu.addAction("(none defined yet)").setEnabled(False)
        for n in names:
            sub = menu.addMenu(n)
            for v, text in views.items():
                sub.addAction(text, lambda n=n, v=v: self._add_panel(source, ref=n, view=v))

    def _add_panel(self, source: str, figure: dict | None = None, ref: str = "", view: str = "") -> None:
        p = X.blank_panel(source)
        p["height"] = DEFAULT_HEIGHT[source]
        if figure is not None:
            p["figure"] = copy.deepcopy(figure)
            if figure["kind"] in ("heatmap", "track_heatmap"):
                p["span"], p["height"] = self.columns.value(), 60.0
            if p["figure"]["plot"]["legend"] == "best" and figure["kind"] in LINE_KINDS:
                p["figure"]["plot"]["legend"] = "direct"
        p["ref"], p["view"] = ref, view
        if source != "figure":
            p["span"] = self.columns.value()
        self.panels.append(p)
        self._fill_panels(select=len(self.panels) - 1)
        self._commit()

    def _add_all_quick(self) -> None:
        exp = self.get_experiment()
        for p in X.from_quick_figures(exp, columns=self.columns.value())["panels"]:
            self.panels.append(p)
        self._fill_panels(select=len(self.panels) - 1)
        self._commit()

    def _duplicate_panel(self) -> None:
        row = self.panel_list.currentRow()
        if 0 <= row < len(self.panels):
            self.panels.insert(row + 1, copy.deepcopy(self.panels[row]))
            self._fill_panels(select=row + 1)
            self._commit()

    def _remove_panel(self) -> None:
        row = self.panel_list.currentRow()
        if 0 <= row < len(self.panels):
            del self.panels[row]
            self._fill_panels(select=max(0, row - 1))
            self._commit()

    def _show_panel(self, row: int) -> None:
        ok = 0 <= row < len(self.panels)
        self.panel_box.setEnabled(ok)
        if not ok:
            self.p_source.setText("Add a panel: a quick figure, a fate view or a dynamics view.")
            return
        p = self.panels[row]
        self._loading = True
        exp = self.get_experiment()
        fig = p["source"] == "figure"
        f = p["figure"]
        self.p_source.setText(self._panel_text(row, p))
        self.p_span.setValue(int(p["span"]))
        self.p_height.setValue(float(p["height"]))
        self.p_label.setText(p["label"])
        self.p_caption.setText(p["caption"])
        line = fig and f["kind"] in LINE_KINDS
        for w in (self.p_compare, self.p_reference, self.p_cols):
            w.setEnabled(line)
        self.p_legend.setEnabled(fig)
        self.p_recopy.setVisible(fig)
        self.p_compare.setCurrentIndex(max(0, self.p_compare.findData(f["compare"])))
        self.p_reference.clear()
        self.p_reference.addItem("first drawn", "")
        drawn = f["conditions"] or [c["name"] for c in exp["plate"]["conditions"]]
        for n in drawn:
            self.p_reference.addItem(n, n)
        self.p_reference.setCurrentIndex(max(0, self.p_reference.findData(f["reference"])))
        self.p_cols.setValue(int(f["columns"]))
        self.p_legend.setCurrentText(f["plot"]["legend"])
        for w in (self.p_ref, self.p_view, self.p_condition):
            w.setEnabled(not fig)
        self.p_ref.clear()
        self.p_view.clear()
        if not fig:
            from ..analysis.fate import VIEWS as FATE_VIEWS
            from ..dynamics.plots import VIEWS as FIT_VIEWS
            names = [x["name"] for x in (exp["analysis"]["fates"] if p["source"] == "fate" else exp["analysis"]["dynamics"]["fits"])]
            if p["ref"] and p["ref"] not in names:
                names.append(p["ref"])
            self.p_ref.addItems(names)
            self.p_ref.setCurrentText(p["ref"])
            for v, text in (FATE_VIEWS if p["source"] == "fate" else FIT_VIEWS).items():
                self.p_view.addItem(text, v)
            self.p_view.setCurrentIndex(max(0, self.p_view.findData(p["view"])))
            self.p_condition.setValue(int(p["condition"]))
        self.p_condition.setVisible(p["source"] == "dynamics")
        self._fill_conditions(p)
        self._loading = False

    def _report_conditions(self, p: dict) -> list[str]:
        """The conditions of the panel's fate analysis or fit, else of the plate."""
        exp = self.get_experiment()
        if p["source"] == "fate":
            spec = next((f for f in exp["analysis"]["fates"] if f["name"] == p["ref"]), None)
        else:
            spec = next((f for f in exp["analysis"]["dynamics"]["fits"] if f["name"] == p["ref"]), None)
        from ..analysis.figures import figure_conditions
        return [c["name"] for c in figure_conditions(exp, spec)] if spec else [c["name"] for c in exp["plate"]["conditions"]]

    def _fill_conditions(self, p: dict) -> None:
        menu = self.p_conditions.menu()
        menu.clear()
        visible = p["source"] != "figure"
        self.p_conditions.setVisible(visible)
        label = self.panel_form.labelForField(self.p_conditions) if hasattr(self, "panel_form") else None
        if label is not None:
            label.setVisible(visible)
        if not visible:
            return
        names = self._report_conditions(p)
        chosen = set(p["conditions"])
        every = menu.addAction("every condition")
        every.triggered.connect(lambda: self._set_conditions([]))
        menu.addSeparator()
        for n in names:
            act = menu.addAction(n)
            act.setCheckable(True)
            act.setChecked(n in chosen)
            act.toggled.connect(lambda on, n=n: self._toggle_condition(n, on))
        self._conditions_text(p, names)

    def _conditions_text(self, p: dict, names: list[str]) -> None:
        shown = [n for n in names if n in set(p["conditions"])]
        self.p_conditions.setText("every condition" if not shown else
                                  ", ".join(shown) if len(shown) <= 3 else f"{len(shown)} of {len(names)}")

    def _toggle_condition(self, name: str, on: bool) -> None:
        row = self.panel_list.currentRow()
        if self._loading or not 0 <= row < len(self.panels):
            return
        p = self.panels[row]
        names = self._report_conditions(p)
        chosen = set(p["conditions"]) | {name} if on else set(p["conditions"]) - {name}
        self._set_conditions([n for n in names if n in chosen] + [n for n in p["conditions"] if n not in names and n in chosen])

    def _set_conditions(self, names: list[str]) -> None:
        """Store the choice and tick the menu to match. The menu is updated in place, never rebuilt here:
        this runs inside one of its actions' signals."""
        row = self.panel_list.currentRow()
        if not 0 <= row < len(self.panels):
            return
        p = self.panels[row]
        p["conditions"] = list(names)
        for act in self.p_conditions.menu().actions():
            if act.isCheckable():
                act.blockSignals(True)
                act.setChecked(act.text() in set(names))
                act.blockSignals(False)
        self._conditions_text(p, self._report_conditions(p))
        self._commit()

    def _ref_changed(self, *_) -> None:
        self._panel_edited()

    def _panel_edited(self, *_) -> None:
        row = self.panel_list.currentRow()
        if self._loading or not 0 <= row < len(self.panels):
            return
        p = self.panels[row]
        p["span"], p["height"] = self.p_span.value(), self.p_height.value()
        p["label"], p["caption"] = self.p_label.text().strip(), self.p_caption.text().strip()
        if p["source"] == "figure":
            f = p["figure"]
            f["compare"], f["reference"] = self.p_compare.currentData(), self.p_reference.currentData() or ""
            f["columns"] = self.p_cols.value()
            f["plot"]["legend"] = self.p_legend.currentText()
        else:
            ref = self.p_ref.currentText()
            if ref != p["ref"]:
                p["conditions"] = []
            p["ref"], p["view"] = ref, self.p_view.currentData() or ""
            p["condition"] = self.p_condition.value()
            self._loading = True
            self._fill_conditions(p)
            self._loading = False
        self.panel_list.item(row).setText(self._panel_text(row, p))
        self._commit()

    def _recopy(self) -> None:
        row = self.panel_list.currentRow()
        if not 0 <= row < len(self.panels):
            return
        p = self.panels[row]
        source = next((f for f in self.get_experiment()["figures"]["list"] if f["name"] == p["figure"]["name"]), None)
        if source is None:
            QMessageBox.information(self, "No such quick figure", f"There is no quick figure called {p['figure']['name']!r}.")
            return
        keep = {k: p["figure"][k] for k in ("compare", "reference", "columns")}
        legend = p["figure"]["plot"]["legend"]
        p["figure"] = copy.deepcopy(source)
        p["figure"].update(keep)
        p["figure"]["plot"]["legend"] = legend
        self._show_panel(row)
        self._commit()

    # ------------------------------------------------ templates
    def _fill_templates(self) -> None:
        menu = self.templates.menu()
        menu.clear()
        menu.addAction("New from every quick figure", self._from_quick)
        menu.addSeparator()
        saved = X.list_templates()
        apply = menu.addMenu("Apply a saved template")
        apply.setEnabled(bool(saved))
        for name, path in saved:
            apply.addAction(name, lambda p=path: self._apply_template(p))
        menu.addAction("Save this figure as a template", self._save_template)
        menu.addSeparator()
        menu.addAction("Import a template file…", self._import_template)
        menu.addAction("Export this figure as a template file…", self._export_template)
        menu.addSeparator()
        menu.addAction(f"Templates are kept in {X.template_dir()}").setEnabled(False)

    def _load_definition(self, export: dict, notes: list[str]) -> None:
        names = {i["name"] for i in self.list.items}
        base, k = export["name"] or "Figure", 1
        export["name"] = base
        while export["name"] in names:
            k += 1
            export["name"] = f"{base} {k}"
        self._write(export)
        self.list.add()
        if notes:
            QMessageBox.information(self, "Template applied", "\n".join(notes))
        self.preview()

    def _from_quick(self) -> None:
        self._load_definition(X.from_quick_figures(self.get_experiment(), columns=2), [])

    def _apply_template(self, path) -> None:
        try:
            export, notes = X.apply_template(self.get_experiment(), X.load_template(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Template", str(exc))
            return
        self._load_definition(export, notes)

    def _save_template(self) -> None:
        export = self._read()
        if not export["name"]:
            QMessageBox.information(self, "Name needed", "Give the figure a name first.")
            return
        path = X.save_template(export)
        self.notes.setText(f"Saved template {path}")

    def _import_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import figure template", str(X.template_dir()), "Template (*.yaml *.yml)")
        if path:
            self._apply_template(path)

    def _export_template(self) -> None:
        export = self._read()
        path, _ = QFileDialog.getSaveFileName(self, "Export figure template",
                                              str(X.template_dir() / f"{export['name'] or 'template'}.yaml"), "Template (*.yaml)")
        if path:
            X.save_template(export, path)
            self.notes.setText(f"Saved template {path}")

    # ------------------------------------------------ preview and export
    def _experiment(self) -> dict | None:
        exp = copy.deepcopy(self.get_experiment())
        if not exp["paths"]["results"]:
            self.notes.setText("Set the results folder on the Experiment page.")
            return None
        return exp

    def preview(self) -> None:
        if self._loading or not self.isVisible():
            return
        exp = self._experiment()
        if exp is None:
            return
        export = self._read()
        w, h = X.page_size(export)
        self.size_label.setText(f"{w:.0f} × {h:.0f} mm ({export['journal']}), {len(export['panels'])} panels")
        if self.worker is not None and self.worker.isRunning():
            self.pending = (exp, export)
            return
        self._start(exp, export)

    def _start(self, exp: dict, export: dict) -> None:
        cache = self.get_cache(exp) if self.get_cache else None
        target_w, target_h = max(self.image.width(), 300), max(self.image.height(), 200)
        w, h = X.page_size(export)
        dpi = max(40.0, min(target_w / (w / 25.4), target_h / (h / 25.4)) * self.devicePixelRatioF())

        def job(on_event, stop):
            fig, info = X.render_export(exp, export, cache, dpi=dpi)
            buf = io.BytesIO()
            X.save_page(fig, export, buf, format="png", dpi=dpi)
            return buf.getvalue(), info, export

        self.notes.setText("Drawing…")
        self.worker = FunctionWorker(job, self)
        self.worker.finished_ok.connect(self._show)
        self.worker.failed.connect(lambda m: self.notes.setText(m.splitlines()[0]))
        self.worker.finished.connect(self._next)
        self.worker.start()

    def _show(self, out) -> None:
        png, info, export = out
        pix = QPixmap()
        pix.loadFromData(png, "PNG")
        pix.setDevicePixelRatio(self.devicePixelRatioF())
        self._pixmap = pix
        self.image.setPixmap(pix)
        self.notes.setText("\n".join(info["notes"]) if info["notes"] else "Meets the checks: no text below 5 pt, nothing cut off.")
        self.caption.setPlainText(X.caption_text(export, info["captions"]))

    def _next(self) -> None:
        if self.pending is not None:
            request, self.pending = self.pending, None
            self._start(*request)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.preview()

    def export_selected(self) -> None:
        exp = self._experiment()
        if exp is None:
            return
        export = self._read()
        self._run(lambda on_event, stop: X.write_export(exp, export, self.get_cache(exp) if self.get_cache else None, on_event), exp)

    def export_all(self) -> None:
        exp = self._experiment()
        if exp is None:
            return
        exp["figures"]["exports"] = copy.deepcopy(self.list.items)
        self._run(lambda on_event, stop: X.write_all(exp, on_event=on_event), exp)

    def _run(self, fn, exp: dict) -> None:
        self.notes.setText("Exporting…")
        messages = []
        self.export_worker = FunctionWorker(fn, self)
        self.export_worker.event.connect(lambda e: e["kind"] == "export_note" and messages.append(f"{e['name']}: {e['message']}"))
        self.export_worker.finished_ok.connect(lambda written: self.notes.setText(
            f"Wrote {len(written)} files to {exp['paths']['results']}/{exp['figures']['export_folder']}."
            + ("\n" + "\n".join(messages) if messages else "")))
        self.export_worker.failed.connect(lambda m: self.notes.setText(m.splitlines()[0]))
        self.export_worker.start()

    # ------------------------------------------------ experiment file
    def load(self, figures: dict) -> None:
        self.list.load(figures["exports"])

    def collect(self) -> list[dict]:
        return copy.deepcopy(self.list.items)

    def _every_panel(self):
        """The saved definitions' panels and those open in the editor."""
        seen = set()
        for p in [p for e in self.list.items for p in e["panels"]] + list(self.panels):
            if id(p) not in seen:
                seen.add(id(p))
                yield p

    def rename_condition(self, old: str, new: str) -> None:
        for p in self._every_panel():
            f = p["figure"]
            f["conditions"] = [new if c == old else c for c in f["conditions"]]
            if f["reference"] == old:
                f["reference"] = new
            p["conditions"] = [new if c == old else c for c in p.get("conditions", [])]
        self.changed.emit()

    def remove_condition(self, name: str) -> None:
        for p in self._every_panel():
            f = p["figure"]
            f["conditions"] = [c for c in f["conditions"] if c != name]
            if f["reference"] == name:
                f["reference"] = ""
            p["conditions"] = [c for c in p.get("conditions", []) if c != name]
        self.changed.emit()
