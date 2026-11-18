"""Analysis sections for discovery: Features (many measurements per cell, on one scale) and Fates
(committor curves and event-aligned features: when cells commit, from what level, what precedes it)."""
from __future__ import annotations

import copy

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter, QTextBrowser,
                               QVBoxLayout, QWidget)

from ..analysis import features as FT
from ..analysis.fate import ALIGN, VIEWS
from ..experiment import FATE_DEFAULT, FEATURE_DEFAULT, FEATURES_DEFAULTS, merge
from .analysis_tab import _NamedList
from .worker import FunctionWorker


def _channels(exp: dict) -> list[str]:
    return [c["name"] for c in exp["channels"] if c["role"] != "skip"] or ["c1", "c2", "c3"]


def _dspin(low, high, decimals=1, step=1.0, suffix="", special=None):
    s = QDoubleSpinBox()
    s.setRange(low, high)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    if special:
        s.setSpecialValueText(special)
    return s


def _ispin(low, high, prefix="", suffix="", special=None):
    s = QSpinBox()
    s.setRange(low, high)
    if prefix:
        s.setPrefix(prefix)
    if suffix:
        s.setSuffix(suffix)
    if special:
        s.setSpecialValueText(special)
    return s


def _ticked(lst: QListWidget) -> list[str]:
    return [lst.item(i).text() for i in range(lst.count()) if lst.item(i).checkState() == Qt.CheckState.Checked]


def _set_ticks(lst: QListWidget, names: list[str], ticked: list[str]) -> None:
    lst.blockSignals(True)
    lst.clear()
    for n in names + [t for t in ticked if t not in names]:
        item = QListWidgetItem(n)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if n in ticked else Qt.CheckState.Unchecked)
        lst.addItem(item)
    lst.blockSignals(False)


# ---------------------------------------------------------------------- features
class FeaturesPanel(QWidget):
    changed = Signal()

    def __init__(self, get_experiment):
        super().__init__()
        self.get_experiment = get_experiment
        self.worker = None
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)
        editor = QWidget()
        form = QFormLayout(editor)
        form.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. PPARG nuc_mean")
        form.addRow("Name", self.name)
        self.measure = QComboBox()
        self.measure.currentIndexChanged.connect(self._measure_changed)
        form.addRow("Measurement", self.measure)
        self.channel = QComboBox()
        form.addRow("Channel", self.channel)
        self.transform = QComboBox()
        for t in FT.TRANSFORMS:
            self.transform.addItem(t, t)
        self.transform.setToolTip("log for intensities, areas and other positive amounts; asinh for counts and speeds "
                                  "(like log, but fine at zero); none for shapes and ratios already in log units.")
        form.addRow("Transform", self.transform)
        self.list = _NamedList(editor, self._read, self._write, FEATURE_DEFAULT, "feature", checkable=True)
        self.list.changed.connect(self.changed)
        side = QWidget()
        sl = QVBoxLayout(side)
        note = QLabel("Features are the measurements discovery analyses use: ticked ones go into models and aligned "
                      "averages. Each experiment is put on one scale against its reference condition (0 = its median, "
                      "1 = its robust spread), and features pool across experiments by name.")
        note.setWordWrap(True)
        sl.addWidget(note)
        glob = QFormLayout()
        self.reference = QComboBox()
        self.reference.setToolTip("The condition whose cells define 0 and 1 of every feature (usually the control).")
        glob.addRow("Reference condition", self.reference)
        row = QHBoxLayout()
        self.ref_first = _ispin(0, 10 ** 6)
        self.ref_last = _ispin(0, 10 ** 6, special="end")
        row.addWidget(self.ref_first)
        row.addWidget(QLabel("to"))
        row.addWidget(self.ref_last)
        glob.addRow("Reference frames", row)
        sl.addLayout(glob)
        sl.addWidget(self.list, 1)
        row = QHBoxLayout()
        sug = QPushButton("Suggest a full set")
        sug.setToolTip("Nuclear size, shape and texture, DNA content, density, speed, age since birth, and the level and "
                       "localisation of every reporter; readouts as defined. Replaces the list.")
        sug.clicked.connect(self._suggest)
        prev = QPushButton("Preview")
        prev.setToolTip("Distributions of every ticked feature per condition, and how features correlate (a few positions each).")
        prev.clicked.connect(self.preview)
        row.addWidget(sug)
        row.addWidget(prev)
        sl.addLayout(row)
        side.setMinimumWidth(380)
        side.setMaximumWidth(540)
        split.addWidget(side)
        right = QWidget()
        rl = QVBoxLayout(right)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        rl.addWidget(NavigationToolbar2QT(self.canvas, self))
        rl.addWidget(self.canvas, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        rl.addWidget(self.status)
        split.addWidget(right)
        split.setStretchFactor(1, 1)

    def _measure_changed(self) -> None:
        m = self.measure.currentData() or ""
        self.channel.setEnabled(FT.is_channel_measure(m))

    def refresh(self, exp: dict) -> None:
        cur_m, cur_c, cur_r = self.measure.currentData(), self.channel.currentText(), self.reference.currentData()
        self.measure.blockSignals(True)
        self.measure.clear()
        for m in FT.measures():
            self.measure.addItem(m, m)
        for r in exp["analysis"]["readouts"]:
            if r["name"]:
                self.measure.addItem(f"readout:{r['name']}", f"readout:{r['name']}")
        self.measure.setCurrentIndex(max(0, self.measure.findData(cur_m)))
        self.measure.blockSignals(False)
        self.channel.clear()
        self.channel.addItems(_channels(exp))
        if cur_c:
            self.channel.setCurrentText(cur_c)
        self.reference.clear()
        self.reference.addItem("(first condition)", "")
        for c in exp["plate"]["conditions"]:
            self.reference.addItem(c["name"], c["name"])
        self.reference.setCurrentIndex(max(0, self.reference.findData(cur_r)))
        self._measure_changed()

    def _read(self) -> dict:
        m = self.measure.currentData() or "nuc_area"
        ch = self.channel.currentText() if FT.is_channel_measure(m) else ""
        name = self.name.text().strip() or FT.default_name(self.get_experiment(), m, ch)
        return {"name": name, "measure": m, "channel": ch, "transform": self.transform.currentData()}

    def _write(self, f: dict) -> None:
        f = merge(FEATURE_DEFAULT, f)
        self.name.setText(f["name"])
        if self.measure.findData(f["measure"]) < 0:
            self.measure.addItem(f["measure"], f["measure"])
        self.measure.setCurrentIndex(self.measure.findData(f["measure"]))
        if f["channel"]:
            if self.channel.findText(f["channel"]) < 0:
                self.channel.addItem(f["channel"])
            self.channel.setCurrentText(f["channel"])
        self.transform.setCurrentIndex(max(0, self.transform.findData(f["transform"])))
        self._measure_changed()

    def _suggest(self) -> None:
        self.list.load(FT.suggest(self.get_experiment()))
        self.changed.emit()

    def load(self, spec: dict) -> None:
        s = merge(FEATURES_DEFAULTS, spec)
        self.refresh(self.get_experiment())
        self.reference.setCurrentIndex(max(0, self.reference.findData(s["reference"])))
        self.ref_first.setValue(int(s["reference_frames"][0]))
        self.ref_last.setValue(int(s["reference_frames"][1]))
        self.list.load(s["list"])

    def collect(self) -> dict:
        return {"list": copy.deepcopy(self.list.items), "reference": self.reference.currentData() or "",
                "reference_frames": [self.ref_first.value(), self.ref_last.value()]}

    def remove_condition(self, name: str) -> None:
        if self.reference.currentData() == name:
            self.reference.setCurrentIndex(0)

    def preview(self) -> None:
        exp = copy.deepcopy(self.get_experiment())
        exp["analysis"]["features"] = self.collect()
        if not exp["paths"]["results"] or not exp["plate"]["conditions"]:
            self.status.setText("Choose a results folder and conditions first.")
            return

        def job(on_event, stop):
            from ..analysis.figures import ResultsCache, _window
            cache = ResultsCache(exp)
            feats = FT.feature_list(exp)
            stats = FT.reference_stats(exp, feats, cache=cache)
            out = {}
            for cond in exp["plate"]["conditions"]:
                vals = []
                for key in cache.timestacks_of(cond)[:2]:
                    res = cache.get(key)
                    T = int(res["meta"].get("frames", res.frames))
                    b = FT.position_block(res, exp, feats, _window(exp, {"frames": [0, 0]}, T), stats)
                    y = b["y"].reshape(-1, len(feats))
                    vals.append(y[np.isfinite(y).any(1)][::7])
                    cache._loaded.pop(key, None)
                if vals:
                    out[cond["name"]] = np.concatenate(vals)
            return [f["name"] for f in feats], out, stats["reference"]

        self.status.setText("Reading…")
        self.worker = FunctionWorker(job, self)
        self.worker.finished_ok.connect(self._show_preview)
        self.worker.failed.connect(lambda m: self.status.setText(m.splitlines()[-1]))
        self.worker.start()

    def _show_preview(self, out) -> None:
        names, per_cond, ref = out
        fig = self.figure
        fig.clear()
        if not per_cond:
            self.status.setText("No results for any condition yet.")
            return
        F = len(names)
        exp = self.get_experiment()
        colors = {c["name"]: c["color"] for c in exp["plate"]["conditions"]}
        gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
        ax = fig.add_subplot(gs[0, 0])
        for k, (cond, y) in enumerate(per_cond.items()):
            med = np.nanmedian(y, 0)
            q = np.nanpercentile(y, [25, 75], 0)
            pos = np.arange(F) + (k - (len(per_cond) - 1) / 2) * (0.8 / len(per_cond))
            ax.errorbar(med, pos, xerr=[med - q[0], q[1] - med], fmt="o", ms=3, color=colors.get(cond), label=cond, lw=0.8)
        ax.set_yticks(range(F), names, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(f"median and quartiles, in robust spreads of {ref}", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax2 = fig.add_subplot(gs[0, 1])
        allv = np.concatenate(list(per_cond.values()))
        ok = np.isfinite(allv)
        z = np.where(ok, allv, 0)
        cnt = ok.T.astype(float) @ ok.astype(float)
        mean = z.sum(0) / np.maximum(ok.sum(0), 1)
        zc = np.where(ok, allv - mean, 0)
        cov = zc.T @ zc / np.maximum(cnt, 1)
        sd = np.sqrt(np.diag(cov))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(sd, sd)
        im = ax2.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax2.set_xticks(range(F), names, rotation=90, fontsize=5)
        ax2.set_yticks(range(F), names, fontsize=5)
        fig.colorbar(im, ax=ax2, shrink=0.6)
        ax2.set_title("correlation between features", fontsize=8)
        fig.tight_layout()
        self.canvas.draw_idle()
        self.status.setText("Strongly correlated features carry the same information; unbiased models handle that, "
                            "but untick near-duplicates to make state profiles easier to read.")


# ---------------------------------------------------------------------- fates
class FatePanel(QWidget):
    changed = Signal()

    def __init__(self, get_experiment):
        super().__init__()
        self.get_experiment = get_experiment
        self.worker = None
        self.report = None
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)
        editor = QWidget()
        form = QFormLayout(editor)
        form.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. PPARG commitment")
        form.addRow("Name", self.name)
        self.marker = QComboBox()
        self.marker.setToolTip("The feature whose high state is the fate (e.g. nuclear PPARG).")
        form.addRow("Marker", self.marker)
        row = QHBoxLayout()
        self.auto = QCheckBox("valley between populations")
        self.auto.setToolTip("Put the threshold at the lowest point of the marker distribution between its two populations (low and high).")
        self.threshold = _dspin(-1e6, 1e6, 3, 0.1)
        self.auto.toggled.connect(lambda on: self.threshold.setEnabled(not on))
        row.addWidget(self.auto)
        row.addWidget(self.threshold)
        form.addRow("Threshold", row)
        row = QHBoxLayout()
        self.smooth = _ispin(1, 101, "median of ", " frames")
        self.hold = _ispin(1, 1000, "stays ", " frames")
        self.hold.setToolTip("A crossing counts once the smoothed marker stays on the new side this long (and was on the old side as long).")
        row.addWidget(self.smooth)
        row.addWidget(self.hold)
        form.addRow("Crossing", row)
        self.horizon = _dspin(0.1, 1000, 1, 1, " h")
        self.horizon.setToolTip("Committor: P(marker high this much later | marker level now).")
        form.addRow("Horizon", self.horizon)
        row = QHBoxLayout()
        self.split = QComboBox()
        self.split.setToolTip("Optional: split cells by this feature, averaged over the time before each frame, into groups by "
                              "quantiles of the reference condition (e.g. low and high ERK).")
        self.groups = _ispin(2, 6, "groups ")
        self.split_hours = _dspin(0.1, 100, 1, 0.5, " h")
        self.split_hours.setPrefix("over ")
        row.addWidget(self.split, 1)
        row.addWidget(self.groups)
        row.addWidget(self.split_hours)
        form.addRow("Split by", row)
        self.align = QComboBox()
        for k, text in ALIGN.items():
            self.align.addItem(text, k)
        form.addRow("Align on", self.align)
        row = QHBoxLayout()
        self.before = _dspin(0.1, 1000, 1, 1, " h before")
        self.after = _dspin(0.1, 1000, 1, 1, " h after")
        row.addWidget(self.before)
        row.addWidget(self.after)
        form.addRow("Window", row)
        self.align_features = QListWidget()
        self.align_features.setMaximumHeight(90)
        self.align_features.setToolTip("Features averaged around events. None ticked = every ticked feature.")
        form.addRow("Aligned features", self.align_features)
        self.match_on = QListWidget()
        self.match_on.setMaximumHeight(70)
        self.match_on.setToolTip("Matched cells must also be close in these features over the window before (e.g. nuc_area and "
                                 "H2B nuc_sum, so they are at a similar cell-cycle phase). None = the marker only.")
        form.addRow("Match also on", self.match_on)
        row = QHBoxLayout()
        self.exclude = QComboBox()
        self.exclude.setToolTip("Leave out crossings where this feature is extreme near the crossing, e.g. H2B bimodality at "
                                "mitosis, when a condensed nucleus reads brighter in every channel.")
        self.exclude_above = _dspin(0, 100, 1, 0.5, " spreads")
        self.exclude_above.setPrefix("above ")
        self.exclude_frames = _ispin(0, 100, "±", " frames")
        row.addWidget(self.exclude, 1)
        row.addWidget(self.exclude_above)
        row.addWidget(self.exclude_frames)
        form.addRow("Exclude crossings", row)
        self.conditions = QListWidget()
        self.conditions.setMaximumHeight(100)
        form.addRow("Conditions", self.conditions)
        self.group = QComboBox()
        form.addRow("…or group", self.group)
        row = QHBoxLayout()
        self.first = _ispin(0, 10 ** 6)
        self.last = _ispin(0, 10 ** 6, special="end")
        row.addWidget(self.first)
        row.addWidget(QLabel("to"))
        row.addWidget(self.last)
        form.addRow("Movie frames", row)
        row = QHBoxLayout()
        self.bins = _ispin(5, 200, "bins ")
        self.bootstrap = _ispin(0, 2000, "resamples ")
        self.bootstrap.setToolTip("Positions resampled with replacement, for 95% intervals.")
        self.seed = _ispin(0, 10 ** 6, "seed ")
        for w in (self.bins, self.bootstrap, self.seed):
            row.addWidget(w)
        form.addRow(row)
        self.list = _NamedList(editor, self._read, self._write, FATE_DEFAULT, "analysis", checkable=True)
        self.list.changed.connect(self.changed)
        self.list.list.currentRowChanged.connect(lambda _: self.show_saved())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.list)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.addWidget(scroll, 1)
        row = QHBoxLayout()
        self.run_button = QPushButton("Run selected")
        self.run_button.clicked.connect(self.run_selected)
        self.run_all = QPushButton("Run all ticked")
        self.run_all.clicked.connect(lambda: self._run([copy.deepcopy(f) for f in self.list.items if f.get("enabled", True)]))
        self.stop = QPushButton("Stop")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(lambda: self.worker and self.worker.stop_flag.set())
        for b in (self.run_button, self.run_all, self.stop):
            row.addWidget(b)
        sl.addLayout(row)
        side.setMinimumWidth(440)
        side.setMaximumWidth(600)
        split.addWidget(side)
        right = QWidget()
        rl = QVBoxLayout(right)
        self.view = QComboBox()
        for k, text in VIEWS.items():
            self.view.addItem(text, k)
        self.view.currentIndexChanged.connect(self.redraw)
        rl.addWidget(self.view)
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        rl.addWidget(NavigationToolbar2QT(self.canvas, self))
        rl.addWidget(self.canvas, 3)
        self.summary = QTextBrowser()
        rl.addWidget(self.summary, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(80)
        rl.addWidget(self.log)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self._write(FATE_DEFAULT)

    def _feature_names(self, exp: dict) -> list[str]:
        return [f["name"] for f in FT.feature_list(exp, enabled_only=False)]

    def refresh(self, exp: dict) -> None:
        names = self._feature_names(exp)
        for combo, blank in ((self.marker, None), (self.split, "(no split)"), (self.exclude, "(none)")):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            if blank:
                combo.addItem(blank, "")
            for n in names:
                combo.addItem(n, n)
            combo.setCurrentIndex(max(0, combo.findData(cur)))
            combo.blockSignals(False)
        _set_ticks(self.align_features, names, [t for t in _ticked(self.align_features) if t in names])
        _set_ticks(self.match_on, names, [t for t in _ticked(self.match_on) if t in names])
        conds = [c["name"] for c in exp["plate"]["conditions"]]
        _set_ticks(self.conditions, conds, [t for t in _ticked(self.conditions) if t in conds])
        cur = self.group.currentData()
        self.group.clear()
        self.group.addItem("(every condition)", "")
        for g in exp["analysis"]["groups"]:
            self.group.addItem(g["name"], g["name"])
        self.group.setCurrentIndex(max(0, self.group.findData(cur)))

    def _read(self) -> dict:
        return {"name": self.name.text().strip(), "marker": self.marker.currentData() or "", "threshold": self.threshold.value(),
                "threshold_auto": self.auto.isChecked(), "smooth_frames": self.smooth.value(), "hold_frames": self.hold.value(),
                "horizon_hours": self.horizon.value(), "bins": self.bins.value(), "split": self.split.currentData() or "",
                "split_groups": self.groups.value(), "split_hours": self.split_hours.value(), "align": self.align.currentData(),
                "align_features": _ticked(self.align_features), "match_on": _ticked(self.match_on),
                "exclude_feature": self.exclude.currentData() or "", "exclude_above": self.exclude_above.value(),
                "exclude_frames": self.exclude_frames.value(), "before_hours": self.before.value(), "after_hours": self.after.value(),
                "conditions": _ticked(self.conditions), "group": self.group.currentData() or "",
                "frames": [self.first.value(), self.last.value()], "bootstrap": self.bootstrap.value(), "seed": self.seed.value()}

    def _write(self, spec: dict) -> None:
        s = merge(FATE_DEFAULT, spec)
        exp = self.get_experiment()
        self.refresh(exp)
        self.name.setText(s["name"])
        for combo, value in ((self.marker, s["marker"]), (self.split, s["split"]), (self.exclude, s["exclude_feature"])):
            if value and combo.findData(value) < 0:
                combo.addItem(value, value)
            combo.setCurrentIndex(max(0, combo.findData(value)))
        self.auto.setChecked(bool(s["threshold_auto"]))
        self.threshold.setValue(float(s["threshold"]))
        self.threshold.setEnabled(not s["threshold_auto"])
        self.smooth.setValue(int(s["smooth_frames"]))
        self.hold.setValue(int(s["hold_frames"]))
        self.horizon.setValue(float(s["horizon_hours"]))
        self.bins.setValue(int(s["bins"]))
        self.groups.setValue(int(s["split_groups"]))
        self.split_hours.setValue(float(s["split_hours"]))
        self.align.setCurrentIndex(max(0, self.align.findData(s["align"])))
        self.before.setValue(float(s["before_hours"]))
        self.after.setValue(float(s["after_hours"]))
        _set_ticks(self.align_features, self._feature_names(exp), s["align_features"])
        _set_ticks(self.match_on, self._feature_names(exp), s["match_on"])
        self.exclude_above.setValue(float(s["exclude_above"]))
        self.exclude_frames.setValue(int(s["exclude_frames"]))
        _set_ticks(self.conditions, [c["name"] for c in exp["plate"]["conditions"]], s["conditions"])
        self.group.setCurrentIndex(max(0, self.group.findData(s["group"])))
        self.first.setValue(int(s["frames"][0]))
        self.last.setValue(int(s["frames"][1]))
        self.bootstrap.setValue(int(s["bootstrap"]))
        self.seed.setValue(int(s["seed"]))

    def load(self, items: list[dict]) -> None:
        self.list.load(items)
        self.show_saved()

    def collect(self) -> list[dict]:
        return copy.deepcopy(self.list.items)

    def rename_condition(self, old: str, new: str) -> None:
        for f in self.list.items:
            f["conditions"] = [new if c == old else c for c in f["conditions"]]

    def remove_condition(self, name: str) -> None:
        for f in self.list.items:
            f["conditions"] = [c for c in f["conditions"] if c != name]

    def _experiment(self) -> dict:
        exp = copy.deepcopy(self.get_experiment())
        exp["analysis"]["fates"] = self.collect()
        return exp

    def run_selected(self) -> None:
        self.list.update_selected()
        row = self.list.list.currentRow()
        if 0 <= row < len(self.list.items):
            self._run([copy.deepcopy(self.list.items[row])])

    def _run(self, specs: list[dict]) -> None:
        if not specs or (self.worker is not None and self.worker.isRunning()):
            return
        from ..analysis.fate import run_fate
        exp = self._experiment()

        def job(on_event, stop):
            last = None
            for spec in specs:
                last = run_fate(exp, spec, on_event, stop)
            return last

        self.log.clear()
        self.worker = FunctionWorker(job, self)
        self.worker.event.connect(self._event)
        self.worker.finished_ok.connect(lambda _: (self._finish(), self.show_saved()))
        self.worker.failed.connect(lambda m: (self._finish(), self.log.appendPlainText(m)))
        self.run_button.setEnabled(False)
        self.run_all.setEnabled(False)
        self.stop.setEnabled(True)
        self.worker.start()

    def _event(self, e: dict) -> None:
        if e["type"] == "log":
            self.log.appendPlainText(e["message"])
        elif e["type"] == "progress" and (e["done"] % 10 == 0 or e["done"] == e["total"]):
            self.log.appendPlainText(f"{e['stage']} {e['done']}/{e['total']}")

    def _finish(self) -> None:
        self.run_button.setEnabled(True)
        self.run_all.setEnabled(True)
        self.stop.setEnabled(False)

    def show_saved(self) -> None:
        from ..analysis.fate import read_fate
        row = self.list.list.currentRow()
        exp = self.get_experiment()
        self.report = None
        if 0 <= row < len(self.list.items) and exp["paths"]["results"]:
            self.report = read_fate(exp, self.list.items[row]["name"])
        self.redraw()
        self.summary.setHtml(self._summary_html())

    def redraw(self) -> None:
        from ..analysis.fate import draw
        if not self.report:
            self.figure.clear()
            self.figure.text(0.5, 0.5, "Not run yet", ha="center")
            self.canvas.draw_idle()
            return
        try:
            draw(self.figure, self.report, self.view.currentData(), self.get_experiment())
        except Exception as exc:  # a saved result from an older version, or an empty view
            self.figure.clear()
            self.figure.text(0.5, 0.5, f"Cannot draw this view: {exc}", ha="center", wrap=True)
        self.canvas.draw_idle()

    def _summary_html(self) -> str:
        r = self.report
        if not r:
            return ("<i>Not run yet.</i> Needs the frame interval, a marker feature, and conditions. "
                    "The committor asks how likely a cell at a given marker level is to be high later; aligned averages "
                    "show what every feature does around commitment, compared with matched cells that did not commit.")
        groups = ["every cell"] + ([f"{r['split']} group {g + 1}" for g in range(r["split_groups"])] if r["split_groups"] else [])
        head = "".join(f"<th>{g}</th>" for g in groups)
        rows = []
        for c, name in enumerate(r["conditions"]):
            cells = []
            for g in range(len(groups)):
                v, lo, hi = r["x50"][c][g], r["x50_ci"][0][c][g], r["x50_ci"][1][c][g]
                txt = "–" if v is None or not np.isfinite(v) else f"{v:.3g}"
                if lo is not None and np.isfinite(lo):
                    txt += f" <span style='color:gray'>[{lo:.3g}, {hi:.3g}]</span>"
                cells.append(f"<td>{txt}</td>")
            rows.append(f"<tr><td>{name}</td><td>{r['positions'][c]}</td><td>{r['tracks'][c]}</td><td>{r['events'][c]}</td>"
                        f"<td>{100 * r['censored_share'][c]:.0f}%</td>{''.join(cells)}</tr>")
        mix = r["mixture"]
        return (f"<b>{r['name']}</b>: marker {r['marker']}, threshold {r['threshold']:.3g} "
                f"(two populations {mix['separation']:.1f} sd apart), horizon {r['spec']['horizon_hours']:g} h, "
                f"features normalised to {r['reference']}. Ran {r['fitted']} in {r['seconds']} s."
                f"<table cellpadding=3><tr><th></th><th>positions</th><th>tracks</th><th>{r['align']} events</th>"
                f"<th>censored</th>{head}</tr>{''.join(rows)}</table>"
                + (f"<p>Crossings left out by the exclusion rule: {', '.join(f'{c} {n}' for c, n in zip(r['conditions'], r.get('excluded', [])))}.</p>"
                   if r["spec"].get("exclude_feature") else "")
                + "<p style='color:gray'>Columns after 'censored': the commitment point (marker level where P = 0.5), with 95% "
                "intervals over positions. Censored: cell-frames whose track was gone by the horizon (not counted).</p>")
