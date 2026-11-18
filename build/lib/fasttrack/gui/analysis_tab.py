"""Analysis: build figures from the results, and keep them in the experiment file.

Groups are named sets of conditions (the first is the reference, as in the old batch_dict).
Gates are thresholds on a metric, used to count the % of cells above or below. A figure is a
kind, a group, a metric and a frame window; "Save as new figure" stores it in the file, so
`fasttrack figures experiment.yaml` or "Draw all" redraws exactly the same figures.
"""
from __future__ import annotations

import copy
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..experiment import (COMPARE_MODES, DISTRIBUTION_STYLES, FIGURE_DEFAULT, FIGURE_KINDS, GATE_DEFAULT, LEGEND_PLACES,
                          METRIC_DEFAULT, PLOT_DEFAULT, UNITS, merge)
from ..plotting.style import PALETTE_TEXT, PALETTES
from ..metrics_info import describe as describe_metric
from ..metrics_info import label as plain_label
from ..track.measure import CHANNELS, SCALARS
from ..analysis.bands import BANDS
from .worker import FunctionWorker

CELL_METRICS = [s for s in SCALARS if s not in ("t", "x", "y", "label", "touches_border")] + ["speed"]
LINE_KINDS = ("timecourse", "percent_high", "cell_count", "event_rate")
COMPARE_TEXT = {"overlay": "all on one plot", "reference": "one plot each, reference in grey behind",
                "separate": "one plot each, alone"}
KIND_TEXT = {"timecourse": "Time course (median, quartiles)", "heatmap": "Heatmap: population over time",
             "track_heatmap": "Cells over time (one row per cell)",
             "percent_high": "% of cells passing a gate", "distribution": "Distribution per condition",
             "cell_count": "Cells per position over time", "event_rate": "Event rate (classified events)"}


class MetricEditor(QWidget):
    """measure [channel] [/ measure [channel]], optional background floor, log2 and smoothing."""

    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.measure = QComboBox()
        for m in CHANNELS + CELL_METRICS:
            self.measure.addItem(f"{plain_label(m)}", m)
        self.measure.setMaxVisibleItems(25)
        self.channel = QComboBox()
        row = QHBoxLayout()
        row.addWidget(self.measure, 1)
        row.addWidget(QLabel("of"))
        row.addWidget(self.channel, 1)
        form.addRow("Metric", row)
        self.over_measure = QComboBox()
        self.over_measure.addItem("(no ratio)", "")
        for m in CHANNELS + CELL_METRICS:
            self.over_measure.addItem(plain_label(m), m)
        self.over_channel = QComboBox()
        row = QHBoxLayout()
        row.addWidget(self.over_measure, 1)
        row.addWidget(QLabel("of"))
        row.addWidget(self.over_channel, 1)
        form.addRow("Divided by", row)
        self.background = QDoubleSpinBox()
        self.background.setRange(0, 50)
        self.background.setSpecialValueText("off")
        self.background.setToolTip("Subtract this per-frame percentile of the denominator from both terms "
                                   "(5 reproduces the KTR notebook's background floor).")
        form.addRow("Background percentile", self.background)
        row = QHBoxLayout()
        self.log2 = QCheckBox("log2")
        self.smooth = QSpinBox()
        self.smooth.setRange(0, 500)
        self.smooth.setSpecialValueText("no smoothing")
        self.smooth.setSuffix(" frames")
        row.addWidget(self.log2)
        row.addWidget(QLabel("moving mean"))
        row.addWidget(self.smooth, 1)
        form.addRow(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setStyleSheet("background: palette(alternate-base); padding: 4px; border-radius: 3px")
        form.addRow(self.info)
        self.measure.currentIndexChanged.connect(self._enable)
        self.over_measure.currentIndexChanged.connect(self._enable)
        self.set_channels([])

    def _enable(self) -> None:
        measure = self.measure.currentData() or ""
        self.channel.setEnabled(measure in CHANNELS)
        over = self.over_measure.currentData() or ""
        self.over_channel.setEnabled(over in CHANNELS)
        self.background.setEnabled(bool(over))
        text = describe_metric(measure)
        if over:
            text += "<br><br>divided by " + describe_metric(over)
        self.info.setText(text)

    def set_channels(self, names: list[str]) -> None:
        names = names or ["c1", "c2", "c3"]
        for combo, extra in ((self.channel, None), (self.over_channel, "(same channel)")):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            if extra:
                combo.addItem(extra, "")
            for n in names:
                combo.addItem(n, n)
            if current is not None and combo.findData(current) >= 0:
                combo.setCurrentIndex(combo.findData(current))
            combo.blockSignals(False)
        self._enable()

    def load(self, metric: dict) -> None:
        m = merge(METRIC_DEFAULT, metric)
        self.measure.setCurrentIndex(max(0, self.measure.findData(m["measure"])))
        if m["channel"] and self.channel.findData(m["channel"]) < 0:
            self.channel.addItem(m["channel"], m["channel"])
        if m["channel"]:
            self.channel.setCurrentIndex(self.channel.findData(m["channel"]))
        self.over_measure.setCurrentIndex(max(0, self.over_measure.findData(m["over_measure"])))
        if m["over_channel"] and self.over_channel.findData(m["over_channel"]) < 0:
            self.over_channel.addItem(m["over_channel"], m["over_channel"])
        self.over_channel.setCurrentIndex(max(0, self.over_channel.findData(m["over_channel"])))
        self.background.setValue(m["background_percentile"])
        self.log2.setChecked(m["log2"])
        self.smooth.setValue(m["smooth"])
        self._enable()

    def collect(self) -> dict:
        measure = self.measure.currentData()
        over = self.over_measure.currentData() or ""
        return {"measure": measure, "channel": (self.channel.currentData() or "") if measure in CHANNELS else "",
                "over_measure": over, "over_channel": (self.over_channel.currentData() or "") if over in CHANNELS else "",
                "background_percentile": self.background.value() if over else 0.0,
                "log2": self.log2.isChecked(), "smooth": self.smooth.value()}


class _NamedList(QWidget):
    """A list of named dicts with an editor below it and New / Save / Delete buttons."""

    changed = Signal()

    def __init__(self, editor: QWidget, read, write, blank, noun: str, checkable: bool = False):
        super().__init__()
        self.items: list[dict] = []
        self.read, self.write, self.blank, self.noun, self.checkable = read, write, blank, noun, checkable
        lay = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list.model().rowsMoved.connect(self._reordered)
        self.list.currentRowChanged.connect(self._show)
        self.list.itemChanged.connect(self._checked)
        lay.addWidget(self.list, 1)
        lay.addWidget(editor)
        row = QHBoxLayout()
        self.save_new = QPushButton(f"Save as new {noun}")
        self.save_new.clicked.connect(self.add)
        self.update_button = QPushButton("Update selected")
        self.update_button.clicked.connect(self.update_selected)
        delete = QPushButton("Delete")
        delete.clicked.connect(self.delete)
        for b in (self.save_new, self.update_button, delete):
            row.addWidget(b)
        lay.addLayout(row)

    def _fill(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for item in self.items:
            entry = QListWidgetItem(item["name"] or "(unnamed)")
            if self.checkable:
                entry.setFlags(entry.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                entry.setCheckState(Qt.CheckState.Checked if item.get("enabled", True) else Qt.CheckState.Unchecked)
            self.list.addItem(entry)
        self.list.blockSignals(False)

    def _reordered(self, *_) -> None:
        names = [self.list.item(i).text() for i in range(self.list.count())]
        by_name = {(i["name"] or "(unnamed)"): i for i in self.items}
        self.items = [by_name[n] for n in names if n in by_name]
        self.changed.emit()

    def _checked(self, entry: QListWidgetItem) -> None:
        row = self.list.row(entry)
        if self.checkable and 0 <= row < len(self.items):
            self.items[row]["enabled"] = entry.checkState() == Qt.CheckState.Checked
            self.changed.emit()

    def _show(self, row: int) -> None:
        if 0 <= row < len(self.items):
            self.write(self.items[row])

    def _unique(self, item: dict, skip: int = -1) -> bool:
        if not item["name"]:
            QMessageBox.information(self, "Name needed", f"Give the {self.noun} a name.")
            return False
        if any(i["name"] == item["name"] for k, i in enumerate(self.items) if k != skip):
            QMessageBox.information(self, "Name taken", f"There is already a {self.noun} called {item['name']}.")
            return False
        return True

    def add(self) -> None:
        item = self.read()
        if self._unique(item):
            self.items.append(item)
            self._fill()
            self.list.setCurrentRow(len(self.items) - 1)
            self.changed.emit()

    def update_selected(self) -> None:
        row = self.list.currentRow()
        if not 0 <= row < len(self.items):
            self.add()
            return
        item = self.read()
        if self._unique(item, skip=row):
            item.setdefault("enabled", self.items[row].get("enabled", True))
            self.items[row] = item
            self._fill()
            self.list.setCurrentRow(row)
            self.changed.emit()

    def delete(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self.items):
            del self.items[row]
            self._fill()
            self.changed.emit()

    def load(self, items: list[dict]) -> None:
        self.items = copy.deepcopy(items)
        self._fill()
        if self.items:
            self.list.setCurrentRow(0)
        else:
            self.write(copy.deepcopy(self.blank))


class AnalysisTab(QWidget):
    changed = Signal()
    cache_ready = Signal()          # the metric cache finished building: figures drawn now are fast

    def __init__(self, get_experiment):
        super().__init__()
        self.get_experiment = get_experiment
        self.worker = None
        self.preview_worker = None
        self.warm_worker = None
        self.drawn: dict = {}  # finished previews by settings, until results change
        self.pending_preview = None
        self.results_cache = None
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)
        side = QTabWidget()
        side.setMinimumWidth(420)
        side.setMaximumWidth(560)
        split.addWidget(side)

        # ---------------- figures
        editor = QWidget()
        form = QFormLayout(editor)
        form.setContentsMargins(0, 0, 0, 0)
        self.fig_name = QLineEdit()
        form.addRow("Name", self.fig_name)
        self.kind = QComboBox()
        for k in FIGURE_KINDS:
            self.kind.addItem(KIND_TEXT[k], k)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        form.addRow("Kind", self.kind)
        cond_box = QWidget()
        cl = QVBoxLayout(cond_box)
        cl.setContentsMargins(0, 0, 0, 0)
        self.cond_list = QListWidget()
        self.cond_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.cond_list.setMaximumHeight(150)
        self.cond_list.itemChanged.connect(lambda _: self.preview())
        self.cond_list.model().rowsMoved.connect(lambda *_: self.preview())
        self.cond_list.setToolTip("Tick the conditions to draw; drag to reorder. The first ticked one is the reference for statistics.")
        cl.addWidget(self.cond_list)
        crow = QHBoxLayout()
        for text, state in (("All", True), ("None", False)):
            b = QPushButton(text)
            b.setMaximumWidth(60)
            b.clicked.connect(lambda _=False, st=state: self._tick_all(st))
            crow.addWidget(b)
        self.preset = QComboBox()
        self.preset.setToolTip("Tick the conditions of a saved group (Groups tab).")
        self.preset.activated.connect(self._apply_preset)
        crow.addWidget(self.preset, 1)
        cl.addLayout(crow)
        form.addRow("Conditions", cond_box)
        self.metric = MetricEditor()
        form.addRow(self.metric)
        self.gate = QComboBox()
        self.gate.setToolTip("% of cells passing: the gate to count. Heatmap: its threshold is drawn as a dashed line "
                             "when the gate uses the same metric.")
        form.addRow("Gate", self.gate)
        self.bin_width = QDoubleSpinBox()
        self.bin_width.setRange(0, 10**6)
        self.bin_width.setDecimals(3)
        self.bin_width.setSingleStep(0.05)
        self.bin_width.setSpecialValueText("automatic")
        self.bin_width.setToolTip("Height of a value bin. Automatic: 0.1 for log2 metrics (as in the notebook), otherwise "
                                  "1/60 of the range. The value range is y min / y max in Plot settings (default: 1st to 99th "
                                  "percentile of every condition shown). Each column sums to 1.")
        self.bin_width.editingFinished.connect(self.preview)
        form.addRow("Heatmap bins", self.bin_width)
        event_row = QHBoxLayout()
        self.event = QComboBox()
        self.bin = QSpinBox()
        self.bin.setRange(1, 1000)
        self.bin.setPrefix("bins of ")
        self.bin.setSuffix(" frames")
        event_row.addWidget(self.event, 1)
        event_row.addWidget(self.bin)
        form.addRow("Event", event_row)
        self.style = QComboBox()
        self.style.addItems(DISTRIBUTION_STYLES)
        form.addRow("Style", self.style)
        unit_row = QHBoxLayout()
        self.unit = QComboBox()
        for u in UNITS:
            self.unit.addItem({"cell": "none: pool all cells", "position": "positions", "well": "wells (recommended)"}[u], u)
        self.unit.setToolTip("Replicates: each well's (or position's) cells are summarised first; spread and statistics use those "
                             "values. Positions in one well share a well, so wells are the honest replicate.")
        self.stats = QCheckBox("Test against the first condition")
        unit_row.addWidget(self.unit, 1)
        unit_row.addWidget(self.stats)
        form.addRow("Replicates", unit_row)
        self.band = QComboBox()
        for key, text in BANDS.items():
            self.band.addItem(text, key)
        self.band.setToolTip("What the line and its band show.\n"
                             "Quartiles and 10-90th percentile: the spread of values around the median.\n"
                             "SD: the spread around the mean. s.e.m. and 95% CI: how well the mean is known.\n"
                             "Over replicates (wells, positions) the values are each replicate's median; "
                             "over pooled cells, s.e.m. and CI are misleadingly small.\n"
                             "Distributions: the bar chart's error bars.")
        self.band.currentIndexChanged.connect(lambda *_: self.preview())
        form.addRow("Error band", self.band)
        compare_row = QHBoxLayout()
        self.compare = QComboBox()
        for m in COMPARE_MODES:
            self.compare.addItem(COMPARE_TEXT[m], m)
        self.compare.setToolTip("How conditions are compared. All on one plot shows differences directly; one plot each "
                                "with the reference in grey keeps every condition readable when there are many.")
        self.reference = QComboBox()
        self.reference.setToolTip("The reference condition: grey behind the others, and dark grey in the colour-blind "
                                  "palettes. First drawn = the first ticked condition.")
        self.compare_cols = QSpinBox()
        self.compare_cols.setRange(0, 12)
        self.compare_cols.setSpecialValueText("auto")
        self.compare_cols.setPrefix("per row ")
        self.compare_cols.setToolTip("Small plots per row (one plot each); auto: one row up to 4.")
        for w in (self.compare, self.reference, self.compare_cols):
            compare_row.addWidget(w, 1 if w is not self.compare_cols else 0)
            (w.valueChanged if isinstance(w, QSpinBox) else w.currentIndexChanged).connect(lambda *_: self.preview())
        form.addRow("Compare", compare_row)
        self.fig_palette = QComboBox()
        self.fig_palette.addItem("as in Settings", "")
        for k in PALETTES:
            self.fig_palette.addItem(PALETTE_TEXT[k], k)
        self.fig_palette.currentIndexChanged.connect(lambda *_: self.preview())
        form.addRow("Colours", self.fig_palette)
        frames = QHBoxLayout()
        self.f_start = QSpinBox()
        self.f_start.setRange(-10**6, 10**6)
        self.f_start.setToolTip("First movie frame. Negative: that many frames before the end of the movie.")
        self.f_stop = QSpinBox()
        self.f_stop.setRange(0, 10**6)
        self.f_stop.setSpecialValueText("end")
        frames.addWidget(self.f_start)
        frames.addWidget(QLabel("to"))
        frames.addWidget(self.f_stop)
        form.addRow("Movie frames", frames)
        self.zero_event = QComboBox()
        form.addRow("Time zero at", self.zero_event)
        self.show_events = QCheckBox("Mark events")
        self.show_events.setChecked(True)
        form.addRow(self.show_events)
        preview = QPushButton("Preview")
        preview.clicked.connect(self.preview)
        form.addRow(preview)
        self.figures = _NamedList(editor, self._read_figure, self._write_figure, FIGURE_DEFAULT, "figure", checkable=True)
        self.figures.changed.connect(self.changed.emit)
        self.figures.list.currentRowChanged.connect(lambda _: self.preview())
        fig_page = QWidget()
        fl = QVBoxLayout(fig_page)
        fl.addWidget(self.figures, 1)
        draw_all = QPushButton("Draw all ticked figures into the results folder")
        draw_all.clicked.connect(self.draw_all)
        fl.addWidget(draw_all)
        side.addTab(fig_page, "Figures")

        # ---------------- groups
        g_editor = QWidget()
        gl = QFormLayout(g_editor)
        gl.setContentsMargins(0, 0, 0, 0)
        self.group_name = QLineEdit()
        gl.addRow("Name", self.group_name)
        self.group_members = QListWidget()
        self.group_members.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        gl.addRow(QLabel("Tick the conditions; drag to order. The first ticked is the reference."))
        gl.addRow(self.group_members)
        self.groups = _NamedList(g_editor, self._read_group, self._write_group, {"name": "", "conditions": []}, "group")
        self.groups.changed.connect(self._lists_changed)
        side.addTab(self.groups, "Groups")

        # ---------------- gates
        t_editor = QWidget()
        tl = QFormLayout(t_editor)
        tl.setContentsMargins(0, 0, 0, 0)
        self.gate_name = QLineEdit()
        tl.addRow("Name", self.gate_name)
        self.gate_metric = MetricEditor()
        tl.addRow(self.gate_metric)
        row = QHBoxLayout()
        self.gate_side = QComboBox()
        self.gate_side.addItems(["above", "below"])
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-1e9, 1e9)
        self.threshold.setDecimals(3)
        row.addWidget(QLabel("Cell passes when"))
        row.addWidget(self.gate_side)
        row.addWidget(self.threshold, 1)
        tl.addRow(row)
        self.gates = _NamedList(t_editor, self._read_gate, self._write_gate, GATE_DEFAULT, "gate")
        self.gates.changed.connect(self._lists_changed)
        side.addTab(self.gates, "Gates")

        # ---------------- settings
        settings = QWidget()
        sl = QFormLayout(settings)
        self.min_frames = QSpinBox()
        self.min_frames.setRange(1, 100000)
        sl.addRow("Minimum track length (frames)", self.min_frames)
        self.drop_border = QCheckBox("Leave out cells touching the image border")
        sl.addRow(self.drop_border)
        self.folder = QLineEdit()
        sl.addRow("Figure folder (in results)", self.folder)
        self.formats = QLineEdit()
        self.formats.setToolTip("Comma separated: pdf, png, svg, eps")
        sl.addRow("File formats", self.formats)
        self.dpi = QSpinBox()
        self.dpi.setRange(50, 1200)
        sl.addRow("DPI", self.dpi)
        size = QHBoxLayout()
        self.width_in = QDoubleSpinBox()
        self.height_in = QDoubleSpinBox()
        for w in (self.width_in, self.height_in):
            w.setRange(1, 40)
            w.setSuffix(" in")
            size.addWidget(w)
        sl.addRow("Size (width, height)", size)
        self.palette = QComboBox()
        for k in PALETTES:
            self.palette.addItem(PALETTE_TEXT[k], k)
        self.palette.setToolTip("Condition colours in every figure, fate and dynamics view, unless a figure sets its own.")
        sl.addRow("Condition colours", self.palette)
        self.export_folder = QLineEdit()
        sl.addRow("Export folder (in results)", self.export_folder)
        side.addTab(settings, "Settings")

        plots = QWidget()
        pl = QVBoxLayout(plots)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        pl.addWidget(NavigationToolbar2QT(self.canvas, self))
        pl.addWidget(self.canvas, 1)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        pl.addWidget(self.note)
        split.addWidget(plots)
        plot_box = QGroupBox("Plot settings")
        pbl = QVBoxLayout(plot_box)
        pbl.addWidget(self._build_plot_panel())
        split.addWidget(plot_box)
        split.setSizes([460, 820, 280])
        self._write_plot(PLOT_DEFAULT)
        self._kind_changed()

    # ------------------------------------------------------------------ editors
    def _kind_changed(self) -> None:
        kind = self.kind.currentData()
        self.metric.setEnabled(kind not in ("percent_high", "cell_count", "event_rate"))
        self.event.setEnabled(kind == "event_rate")
        self.bin.setEnabled(kind == "event_rate")
        self.gate.setEnabled(kind in ("percent_high", "heatmap"))
        self.bin_width.setEnabled(kind == "heatmap")
        self.style.setEnabled(kind == "distribution")
        self.stats.setEnabled(kind == "distribution")
        self.unit.setEnabled(kind in ("distribution", "timecourse", "percent_high", "event_rate", "cell_count"))
        self.band.setEnabled(kind in ("distribution", "timecourse", "percent_high", "event_rate", "cell_count"))
        for w in (self.compare, self.compare_cols):
            w.setEnabled(kind in LINE_KINDS)

    def _read_figure(self) -> dict:
        return {"name": self.fig_name.text().strip(), "kind": self.kind.currentData(), "group": "",
                "conditions": self._ticked(), "plot": self._read_plot(),
                "metric": self.metric.collect(), "gate": self.gate.currentText() if self.gate.count() else "",
                "event": self.event.currentText(), "bin": self.bin.value(), "bin_width": self.bin_width.value(),
                "style": self.style.currentText(), "unit": self.unit.currentData(), "band": self.band.currentData() or "auto", "stats": self.stats.isChecked(), "frames": [self.f_start.value(), self.f_stop.value()],
                "time_zero_event": self.zero_event.currentData() or "", "show_events": self.show_events.isChecked(),
                "compare": self.compare.currentData(), "reference": self.reference.currentData() or "",
                "columns": self.compare_cols.value(), "palette": self.fig_palette.currentData() or ""}

    def _write_figure(self, spec: dict) -> None:
        spec = merge(FIGURE_DEFAULT, spec)
        self.fig_name.setText(spec["name"])
        self.kind.setCurrentIndex(max(0, self.kind.findData(spec["kind"])))
        if spec["conditions"]:
            self._set_ticks(spec["conditions"])
        elif spec["group"]:
            group = next((g for g in self.groups.items if g["name"] == spec["group"]), None)
            self._set_ticks(group["conditions"] if group else [])
        else:
            self._set_ticks(None)
        self._write_plot(spec["plot"])
        self.metric.load(spec["metric"])
        self.gate.setCurrentText(spec["gate"])
        if spec["event"] and self.event.findText(spec["event"]) < 0:
            self.event.addItem(spec["event"])
        self.event.setCurrentText(spec["event"])
        self.bin.setValue(spec["bin"])
        self.bin_width.setValue(float(spec["bin_width"]))
        self.style.setCurrentText(spec["style"])
        self.unit.setCurrentIndex(max(0, self.unit.findData(spec["unit"])))
        self.band.blockSignals(True)
        self.band.setCurrentIndex(max(0, self.band.findData(spec["band"])))
        self.band.blockSignals(False)
        self.stats.setChecked(spec["stats"])
        self.f_start.setValue(spec["frames"][0])
        self.f_stop.setValue(spec["frames"][1])
        self.zero_event.setCurrentIndex(max(0, self.zero_event.findData(spec["time_zero_event"])))
        self.show_events.setChecked(spec["show_events"])
        for w, value in ((self.compare, spec["compare"]), (self.fig_palette, spec["palette"])):
            w.blockSignals(True)
            w.setCurrentIndex(max(0, w.findData(value)))
            w.blockSignals(False)
        self.reference.blockSignals(True)
        if spec["reference"] and self.reference.findData(spec["reference"]) < 0:
            self.reference.addItem(spec["reference"], spec["reference"])
        self.reference.setCurrentIndex(max(0, self.reference.findData(spec["reference"])))
        self.reference.blockSignals(False)
        self.compare_cols.blockSignals(True)
        self.compare_cols.setValue(int(spec["columns"]))
        self.compare_cols.blockSignals(False)

    # ---- conditions
    def _set_ticks(self, ticked: list[str] | None, names: list[str] | None = None) -> None:
        """Fill the condition list: `ticked` first in their order, then the rest unticked. None ticks all."""
        if names is None:
            names = [c["name"] for c in self.get_experiment()["plate"]["conditions"]]
        ticked = list(names) if ticked is None else [n for n in ticked if n in names]
        self.cond_list.blockSignals(True)
        self.cond_list.clear()
        for n in ticked + [n for n in names if n not in ticked]:
            item = QListWidgetItem(n)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if n in ticked else Qt.CheckState.Unchecked)
            self.cond_list.addItem(item)
        self.cond_list.blockSignals(False)

    def _ticked(self) -> list[str]:
        return [self.cond_list.item(i).text() for i in range(self.cond_list.count())
                if self.cond_list.item(i).checkState() == Qt.CheckState.Checked]

    def _tick_all(self, state: bool) -> None:
        for i in range(self.cond_list.count()):
            self.cond_list.item(i).setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)

    def _apply_preset(self) -> None:
        name = self.preset.currentData()
        group = next((g for g in self.groups.items if g["name"] == name), None)
        if group:
            self._set_ticks(group["conditions"])
            self.preview()
        self.preset.setCurrentIndex(0)

    # ---- plot settings
    def _build_plot_panel(self) -> QWidget:
        from PySide6.QtWidgets import QScrollArea
        host = QWidget()
        form = QFormLayout(host)
        self.p_title = QLineEdit()
        self.p_title.setPlaceholderText("the figure name")
        self.p_xlabel = QLineEdit()
        self.p_xlabel.setPlaceholderText("automatic")
        self.p_ylabel = QLineEdit()
        self.p_ylabel.setPlaceholderText("automatic")
        form.addRow("Title", self.p_title)
        form.addRow("x label", self.p_xlabel)
        form.addRow("y label", self.p_ylabel)
        self.p_vmin, self.p_vmax, self.p_xmin, self.p_xmax = (QLineEdit() for _ in range(4))
        for w in (self.p_vmin, self.p_vmax, self.p_xmin, self.p_xmax):
            w.setPlaceholderText("auto")
        row = QHBoxLayout()
        row.addWidget(self.p_vmin)
        row.addWidget(QLabel("to"))
        row.addWidget(self.p_vmax)
        form.addRow("y range", row)
        self.p_vmin.setToolTip("Value axis. Heatmap: the lowest value bin. Cells over time: the colour range.")
        self.p_vmax.setToolTip("Value axis. Heatmap: the highest value bin. Cells over time: the colour range.")
        row = QHBoxLayout()
        row.addWidget(self.p_xmin)
        row.addWidget(QLabel("to"))
        row.addWidget(self.p_xmax)
        form.addRow("x range", row)
        self.p_log = QCheckBox("Logarithmic y axis")
        form.addRow(self.p_log)
        self.p_grid = QCheckBox("Grid")
        form.addRow(self.p_grid)
        self.p_legend = QComboBox()
        self.p_legend.addItems(LEGEND_PLACES)
        form.addRow("Legend", self.p_legend)
        self.p_font = QDoubleSpinBox()
        self.p_font.setRange(4, 30)
        self.p_font.setSuffix(" pt")
        form.addRow("Font size", self.p_font)
        self.p_line = QDoubleSpinBox()
        self.p_line.setRange(0.2, 8)
        self.p_line.setSingleStep(0.25)
        form.addRow("Line width", self.p_line)
        self.p_width, self.p_height = QDoubleSpinBox(), QDoubleSpinBox()
        for w in (self.p_width, self.p_height):
            w.setRange(0, 40)
            w.setSuffix(" in")
            w.setSpecialValueText("default")
        row = QHBoxLayout()
        row.addWidget(self.p_width)
        row.addWidget(self.p_height)
        form.addRow("Saved size", row)
        note = QLabel("Changes redraw the preview. Saved size applies to files written by Draw all; the preview fills the window.")
        note.setWordWrap(True)
        note.setProperty("role", "hint")
        form.addRow(note)
        for w in (self.p_title, self.p_xlabel, self.p_ylabel, self.p_vmin, self.p_vmax, self.p_xmin, self.p_xmax):
            w.editingFinished.connect(self.preview)
        for w in (self.p_log, self.p_grid):
            w.toggled.connect(self.preview)
        self.p_legend.currentIndexChanged.connect(self.preview)
        for w in (self.p_font, self.p_line):
            w.setKeyboardTracking(False)
            w.valueChanged.connect(self.preview)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)
        scroll.setMinimumWidth(240)
        return scroll

    @staticmethod
    def _number(edit: QLineEdit):
        text = edit.text().strip()
        try:
            return float(text) if text else None
        except ValueError:
            return None

    def _read_plot(self) -> dict:
        return {"title": self.p_title.text().strip(), "xlabel": self.p_xlabel.text().strip(), "ylabel": self.p_ylabel.text().strip(),
                "vmin": self._number(self.p_vmin), "vmax": self._number(self.p_vmax),
                "xmin": self._number(self.p_xmin), "xmax": self._number(self.p_xmax),
                "log_y": self.p_log.isChecked(), "legend": self.p_legend.currentText(), "font_size": self.p_font.value(),
                "line_width": self.p_line.value(), "grid": self.p_grid.isChecked(),
                "width": self.p_width.value(), "height": self.p_height.value()}

    def _write_plot(self, plot: dict) -> None:
        plot = merge(PLOT_DEFAULT, plot)
        widgets = [self.p_title, self.p_xlabel, self.p_ylabel, self.p_vmin, self.p_vmax, self.p_xmin, self.p_xmax, self.p_log,
                   self.p_grid, self.p_legend, self.p_font, self.p_line, self.p_width, self.p_height]
        for w in widgets:
            w.blockSignals(True)
        self.p_title.setText(plot["title"])
        self.p_xlabel.setText(plot["xlabel"])
        self.p_ylabel.setText(plot["ylabel"])
        for w, key in ((self.p_vmin, "vmin"), (self.p_vmax, "vmax"), (self.p_xmin, "xmin"), (self.p_xmax, "xmax")):
            w.setText("" if plot[key] is None else f"{plot[key]:g}")
        self.p_log.setChecked(plot["log_y"])
        self.p_grid.setChecked(plot["grid"])
        self.p_legend.setCurrentText(plot["legend"])
        self.p_font.setValue(plot["font_size"])
        self.p_line.setValue(plot["line_width"])
        self.p_width.setValue(plot["width"])
        self.p_height.setValue(plot["height"])
        for w in widgets:
            w.blockSignals(False)

    def _read_group(self) -> dict:
        members = [self.group_members.item(i).text() for i in range(self.group_members.count())
                   if self.group_members.item(i).checkState() == Qt.CheckState.Checked]
        return {"name": self.group_name.text().strip(), "conditions": members}

    def _write_group(self, group: dict) -> None:
        self.group_name.setText(group["name"])
        self._fill_members(group["conditions"])

    def _fill_members(self, ticked: list[str]) -> None:
        exp = self.get_experiment()
        names = [c["name"] for c in exp["plate"]["conditions"]]
        order = [n for n in ticked if n in names] + [n for n in names if n not in ticked]
        self.group_members.clear()
        for n in order:
            item = QListWidgetItem(n)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if n in ticked else Qt.CheckState.Unchecked)
            self.group_members.addItem(item)

    def _read_gate(self) -> dict:
        return {"name": self.gate_name.text().strip(), "metric": self.gate_metric.collect(),
                "threshold": self.threshold.value(), "above": self.gate_side.currentText() == "above"}

    def _write_gate(self, gate: dict) -> None:
        gate = merge(GATE_DEFAULT, gate)
        self.gate_name.setText(gate["name"])
        self.gate_metric.load(gate["metric"])
        self.threshold.setValue(gate["threshold"])
        self.gate_side.setCurrentText("above" if gate["above"] else "below")

    def _lists_changed(self) -> None:
        self.refresh_choices()
        self.changed.emit()

    def refresh_choices(self, exp: dict | None = None) -> None:
        """Refill the combos that depend on channels, conditions, events, groups and gates."""
        exp = exp or self.get_experiment()
        channels = [c["name"] for c in exp["channels"] if c["role"] != "skip"]
        self.metric.set_channels(channels)
        self.gate_metric.set_channels(channels)
        names = [c["name"] for c in exp["plate"]["conditions"]]
        if [self.cond_list.item(i).text() for i in range(self.cond_list.count())] != names:
            ticked = self._ticked() if self.cond_list.count() else None
            self._set_ticks(ticked, names)
        for combo, first, values in ((self.preset, "Tick a saved group…", [g["name"] for g in self.groups.items]),
                                     (self.zero_event, "Start of the movie", [e["name"] for e in exp["time"]["events"]]),
                                     (self.reference, "first drawn", names)):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(first, "")
            for v in values:
                combo.addItem(v, v)
            combo.setCurrentIndex(max(0, combo.findData(current)))
            combo.blockSignals(False)
        current = self.event.currentText()
        self.event.clear()
        self.event.addItems(exp["labelling"]["categories"])
        if current:
            self.event.setCurrentText(current)
        current = self.gate.currentText()
        self.gate.clear()
        self.gate.addItems([g["name"] for g in self.gates.items])
        self.gate.setCurrentText(current)
        if not self.groups.items or self.group_members.count() == 0:
            self._fill_members(self._read_group()["conditions"])

    def rename_condition(self, old: str, new: str) -> None:
        for group in self.groups.items:
            group["conditions"] = [new if c == old else c for c in group["conditions"]]
        self.groups.load(self.groups.items)
        self.changed.emit()

    def remove_condition(self, name: str) -> None:
        for group in self.groups.items:
            group["conditions"] = [c for c in group["conditions"] if c != name]
        self.groups.load(self.groups.items)
        self.changed.emit()

    # ------------------------------------------------------------------ experiment file
    def load(self, exp: dict) -> None:
        a, f = exp["analysis"], exp["figures"]
        self.min_frames.setValue(a["min_track_frames"])
        self.drop_border.setChecked(a["drop_border"])
        self.folder.setText(f["folder"])
        self.formats.setText(", ".join(f["formats"]))
        self.dpi.setValue(f["dpi"])
        self.width_in.setValue(f["width"])
        self.height_in.setValue(f["height"])
        self.palette.setCurrentIndex(max(0, self.palette.findData(f["palette"])))
        self.export_folder.setText(f["export_folder"])
        self.groups.items = copy.deepcopy(a["groups"])
        self.gates.items = copy.deepcopy(a["gates"])
        self.refresh_choices(exp)
        self.groups.load(a["groups"])
        self.gates.load(a["gates"])
        self.figures.load(f["list"])

    def collect(self) -> dict:
        return {
            "analysis": {"min_track_frames": self.min_frames.value(), "drop_border": self.drop_border.isChecked(),
                         "groups": copy.deepcopy(self.groups.items), "gates": copy.deepcopy(self.gates.items)},
            "figures": {"folder": self.folder.text().strip() or "Figures",
                        "formats": [x.strip().lstrip(".") for x in self.formats.text().split(",") if x.strip()],
                        "dpi": self.dpi.value(), "width": self.width_in.value(), "height": self.height_in.value(),
                        "palette": self.palette.currentData() or "experiment",
                        "export_folder": self.export_folder.text().strip() or "Figures/Export",
                        "list": copy.deepcopy(self.figures.items)},
        }

    # ------------------------------------------------------------------ drawing
    def refresh_list(self) -> None:
        """Results changed on disk: forget cached results and redraw the current preview."""
        if self.warm_worker is not None and self.warm_worker.isRunning():
            self.warm_worker.stop_flag.set()
            self.warm_worker.wait()
        self.results_cache = None
        self.drawn = {}
        if self.isVisible():
            self.preview()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.warm_up()

    def warm_up(self) -> bool:
        """Bring the metric cache up to date in the background (columns.py). True if a build started or runs."""
        from ..analysis.figures import warm
        if self.warm_worker is not None and self.warm_worker.isRunning():
            return True
        exp = copy.deepcopy(self.get_experiment())
        if not exp["paths"]["results"] or not Path(exp["paths"]["results"]).is_dir():
            return False
        cache = self._cache_for(exp)
        self.warm_worker = FunctionWorker(lambda on_event, stop: warm(exp, cache, stop), self)
        self.warm_worker.finished_ok.connect(lambda _: self.cache_ready.emit())
        self.warm_worker.failed.connect(lambda m: self.note.setText("Metric cache: " + m.splitlines()[0]))
        self.warm_worker.start()
        return True

    def _cache_wanted(self) -> None:
        """A figure just drew a metric the cache does not hold yet: add it in the background."""
        if self.results_cache is not None and self.results_cache.columns.wanted:
            self.warm_up()

    def _cache_for(self, exp: dict):
        from ..analysis.figures import ResultsCache
        if self.results_cache is None or str(self.results_cache.results_dir) != exp["paths"]["results"]:
            self.results_cache = ResultsCache(exp)
        self.results_cache.exp = exp
        return self.results_cache

    def preview(self) -> None:
        """Draw in a background thread; while one draws, only the latest request waits (no queue of stale ones)."""
        exp = copy.deepcopy(self.get_experiment())
        if not exp["paths"]["results"]:
            self.note.setText("Set the results folder on the Experiment page.")
            return
        request = (exp, self._read_figure())
        key = self._preview_key(*request)
        if key in self.drawn:  # seen before with the same settings: show it again at once
            self._show_preview((self.drawn[key], ""))
            return
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.pending_preview = request
            return
        self._start_preview(*request)

    @staticmethod
    def _preview_key(exp: dict, spec: dict) -> str:
        import json
        relevant = {k: exp[k] for k in ("channels", "time", "plate", "quality", "analysis", "files")}
        relevant["palette"] = exp["figures"]["palette"]
        return json.dumps([spec, relevant, exp["paths"]["results"]], sort_keys=True, default=str)

    def _start_preview(self, exp: dict, spec: dict) -> None:
        from ..analysis.figures import render
        cache = self._cache_for(exp)
        key = self._preview_key(exp, spec)
        w, h = max(self.canvas.width(), 200), max(self.canvas.height(), 150)
        dpi = self.figure.dpi

        def job(on_event, stop):
            figure = Figure(figsize=(w / dpi, h / dpi), dpi=dpi, layout="constrained")
            try:
                render(exp, spec, figure=figure, cache=cache)
                self.drawn[key] = figure
                return figure, ""
            except (KeyError, ValueError, OSError) as exc:
                return None, str(exc).strip("'\"")

        self.note.setText("Drawing…")
        self.preview_worker = FunctionWorker(job, self)
        self.preview_worker.finished_ok.connect(self._show_preview)
        self.preview_worker.failed.connect(lambda m: self.note.setText(m.splitlines()[0]))
        self.preview_worker.finished.connect(self._next_preview)
        self.preview_worker.start()

    def _show_preview(self, out) -> None:
        figure, message = out
        if figure is not None:
            figure.set_canvas(self.canvas)
            self.canvas.figure = figure
            self.figure = figure
            self.canvas.draw_idle()
        else:
            self.figure.clear()
            self.canvas.draw_idle()
        self.note.setText(message)

    def _next_preview(self) -> None:
        if self.pending_preview is None:
            self._cache_wanted()
        if self.pending_preview is not None:
            request, self.pending_preview = self.pending_preview, None
            self._start_preview(*request)

    def add_figure(self, spec: dict) -> None:
        """Add a figure made elsewhere (a summary figure) to the list, under a free name, and select it."""
        spec = merge(FIGURE_DEFAULT, spec)
        names = {f["name"] for f in self.figures.items}
        base, k = spec["name"] or "figure", 2
        while spec["name"] in names or not spec["name"]:
            spec["name"] = f"{base} ({k})"
            k += 1
        self.figures.items.append(spec)
        self.figures._fill()
        self.figures.list.setCurrentRow(len(self.figures.items) - 1)
        self.figures.changed.emit()
        self.changed.emit()

    def draw_all(self) -> None:
        from ..analysis.figures import render_all
        exp = self.get_experiment()
        if not exp["paths"]["results"]:
            self.note.setText("Set the results folder on the Experiment page.")
            return
        self.note.setText("Drawing figures…")
        messages = []
        self.worker = FunctionWorker(lambda on_event, stop: render_all(exp, on_event, summary=False), self)
        self.worker.event.connect(lambda e: e["kind"] == "figure_failed" and messages.append(f"{e['name']}: {e['message']}"))
        self.worker.finished_ok.connect(lambda written: self.note.setText(
            f"Wrote {len(written)} files to {exp['paths']['results']}/{exp['figures']['folder']}." + ("\n" + "\n".join(messages) if messages else "")))
        self.worker.failed.connect(lambda m: self.note.setText(m.splitlines()[0]))
        self.worker.start()
