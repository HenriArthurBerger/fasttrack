"""Analysis sections beyond figures: Readouts (what each reporter measures, with per-observation
error) and Dynamics (stochastic models fitted to the tracks: ERK switching read through the KTR,
mean-reverting walks, continuous-time hidden Markov models, your own SDE).

`AnalysisSection` holds the analysis pages (Figures, Readouts, Features, Fates, Dynamics, Export). The main window
shows each as a page of its own; this object reads and writes them together so no page drops another's settings.
"""
from __future__ import annotations

import copy
import json

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from ..analysis.readouts import KIND_TEXT, KINDS, suggest
from ..dynamics.engine import FRAME_MINUTES, SHARE_TEXT, SHARES
from ..dynamics.models import MODELS, make_model
from ..dynamics.plots import VIEWS
from ..experiment import DYNAMICS_DEFAULTS, FIT_DEFAULT, READOUT_DEFAULT, merge
from .analysis_tab import AnalysisTab, MetricEditor, _NamedList
from .worker import FunctionWorker


def _channels(exp: dict) -> list[str]:
    return [c["name"] for c in exp["channels"] if c["role"] != "skip"] or ["c1", "c2", "c3"]


def _spin(low, high, decimals=2, step=0.1, special=None, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(low, high)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    if special:
        s.setSpecialValueText(special)
    if suffix:
        s.setSuffix(suffix)
    return s


# ---------------------------------------------------------------------- readouts
class ReadoutsPanel(QWidget):
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
        self.name.setPlaceholderText("e.g. ERK")
        form.addRow("Name", self.name)
        self.kind = QComboBox()
        for k in KINDS:
            self.kind.addItem(KIND_TEXT[k], k)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        form.addRow("Kind", self.kind)
        self.channel = QComboBox()
        form.addRow("Channel", self.channel)
        self.background = _spin(0, 50, 1, 1, "off")
        self.background.setToolTip("Per-frame background floor: this percentile of nuclear values in the channel, "
                                   "subtracted from nucleus and ring (5 = the KTR notebook).")
        form.addRow("Background percentile", self.background)
        self.max_noise = _spin(0, 10, 2, 0.05, "keep all")
        self.max_noise.setToolTip("Observations whose error estimate (from pixel spread and count, in ln units) is larger "
                                  "count as missing. 0.5 drops values more than ~±60% uncertain.")
        form.addRow("Largest error", self.max_noise)
        self.min_signal = _spin(0, 1e6, 1, 5, "keep all")
        self.min_signal.setToolTip("Leave out tracks whose mean nuclear signal above the floor is lower (cells that barely express the reporter).")
        form.addRow("Smallest mean signal", self.min_signal)
        self.metric = MetricEditor()
        self.metric_box = QGroupBox("Metric")
        QVBoxLayout(self.metric_box).addWidget(self.metric)
        form.addRow(self.metric_box)
        self.list = _NamedList(editor, self._read, self._write, READOUT_DEFAULT, "readout")
        self.list.changed.connect(self.changed)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.addWidget(QLabel("Readouts are what models fit. Each value comes with its own error, from the pixels it was measured on, "
                            "so dim, noisy cells count for less instead of adding false dynamics."))
        sl.itemAt(0).widget().setWordWrap(True)
        sl.addWidget(self.list, 1)
        row = QHBoxLayout()
        sug = QPushButton("Suggest from channel reporters")
        sug.setToolTip("Adds a KTR readout for every channel whose reporter name contains KTR (Experiment page).")
        sug.clicked.connect(self._suggest)
        prev = QPushButton("Preview")
        prev.clicked.connect(self.preview)
        row.addWidget(sug)
        row.addWidget(prev)
        sl.addLayout(row)
        side.setMinimumWidth(380)
        side.setMaximumWidth(520)
        split.addWidget(side)
        right = QWidget()
        rl = QVBoxLayout(right)
        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        rl.addWidget(NavigationToolbar2QT(self.canvas, self))
        rl.addWidget(self.canvas, 1)
        self.status = QLabel("Preview: values, their errors and example tracks from a few positions of the first condition.")
        self.status.setWordWrap(True)
        rl.addWidget(self.status)
        split.addWidget(right)
        self._kind_changed()

    def _kind_changed(self) -> None:
        kind = self.kind.currentData()
        self.channel.setEnabled(kind not in ("speed", "metric"))
        self.background.setEnabled(kind in ("ktr", "nuclear_reporter"))
        self.min_signal.setEnabled(kind in ("ktr", "nuclear_reporter"))
        self.max_noise.setEnabled(kind in ("ktr", "nuclear_reporter", "intensity"))
        self.metric_box.setVisible(kind == "metric")

    def refresh(self, exp: dict) -> None:
        current = self.channel.currentText()
        self.channel.clear()
        self.channel.addItems(_channels(exp))
        if current:
            self.channel.setCurrentText(current)
        self.metric.set_channels(_channels(exp))

    def _read(self) -> dict:
        return {"name": self.name.text().strip(), "kind": self.kind.currentData(), "channel": self.channel.currentText(),
                "background_percentile": self.background.value(), "max_noise": self.max_noise.value(),
                "min_signal": self.min_signal.value(), "metric": self.metric.collect()}

    def _write(self, r: dict) -> None:
        r = merge(READOUT_DEFAULT, r)
        self.name.setText(r["name"])
        self.kind.setCurrentIndex(max(0, self.kind.findData(r["kind"])))
        if r["channel"]:
            if self.channel.findText(r["channel"]) < 0:
                self.channel.addItem(r["channel"])
            self.channel.setCurrentText(r["channel"])
        self.background.setValue(r["background_percentile"])
        self.max_noise.setValue(r["max_noise"])
        self.min_signal.setValue(r["min_signal"])
        self.metric.load(r["metric"])
        self._kind_changed()

    def _suggest(self) -> None:
        have = {r["name"] for r in self.list.items}
        new = [r for r in suggest(self.get_experiment()) if r["name"] not in have]
        if not new:
            self.status.setText("No channel has a reporter name with KTR in it, or its readout exists already. Add one by hand.")
            return
        self.list.items += new
        self.list._fill()
        self.list.list.setCurrentRow(len(self.list.items) - 1)
        self.changed.emit()

    def load(self, items: list[dict]) -> None:
        self.list.load(items)

    def collect(self) -> list[dict]:
        return copy.deepcopy(self.list.items)

    def preview(self) -> None:
        from ..analysis.figures import ResultsCache
        from ..analysis.readouts import readout_rows
        exp = copy.deepcopy(self.get_experiment())
        readout = self._read()
        if not exp["paths"]["results"] or not exp["plate"]["conditions"]:
            self.status.setText("Choose a results folder and conditions first.")
            return

        def job(on_event, stop):
            cache = ResultsCache(exp)
            for cond in exp["plate"]["conditions"]:
                keys = cache.timestacks_of(cond)[:3]
                if keys:
                    break
            else:
                raise ValueError("No results for any condition yet.")
            parts = []
            for key in keys:
                res = cache.get(key)
                t, track, value, se = readout_rows(res, exp, readout)
                parts.append((key, t, track, value, se, int(res["meta"].get("frames", res.frames))))
            return cond["name"], parts

        self.status.setText("Reading…")
        self.worker = FunctionWorker(job, self)
        self.worker.finished_ok.connect(self._show_preview)
        self.worker.failed.connect(lambda m: self.status.setText(m.splitlines()[0]))
        self.worker.start()

    def _show_preview(self, out) -> None:
        cond, parts = out
        fig = self.figure
        fig.clear()
        values = np.concatenate([p[3] for p in parts]) if parts else np.zeros(0)
        se = np.concatenate([p[4] for p in parts]) if parts else np.zeros(0)
        a1, a2, a3 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 1, 2)
        if len(values):
            a1.hist(values, 80, color="#2d63e5", alpha=0.7)
            a1.set_title(f"values ({len(values)} observations)", fontsize=8)
            a2.hist(se[se > 0], 80, color="#cc3d3d", alpha=0.7)
            a2.set_title(f"error per observation (median {np.median(se[se > 0]) if (se > 0).any() else 0:.2f})", fontsize=8)
            key, t, track, v, _, T = parts[0]
            counts = np.bincount(track)
            for tr_ in np.argsort(-counts)[:5]:
                sel = track == tr_
                order = np.argsort(t[sel])
                a3.plot(t[sel][order], v[sel][order], ".-", ms=2, lw=0.6)
            a3.set_title(f"{key}: the five longest tracks", fontsize=8)
            a3.set_xlabel("frame", fontsize=7)
        for ax in (a1, a2, a3):
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        self.canvas.draw_idle()
        self.status.setText(f"{cond}: {', '.join(p[0] for p in parts)}. Raise 'Largest error' or lower 'Smallest mean signal' to keep more cells.")


# ---------------------------------------------------------------------- dynamics
class DynamicsPanel(QWidget):
    changed = Signal()

    def __init__(self, get_experiment):
        super().__init__()
        self.get_experiment = get_experiment
        self.worker = None
        self.report = None
        self.states = None
        self._specs: dict = {}
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        editor = QWidget()
        form = QFormLayout(editor)
        form.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. ERK 2 modes")
        form.addRow("Name", self.name)
        self.model = QComboBox()
        for name, cls in MODELS.items():
            self.model.addItem(cls.text, name)
        self.model.currentIndexChanged.connect(self._model_changed)
        form.addRow("Model", self.model)
        self.model_info = QLabel("")
        self.model_info.setWordWrap(True)
        self.model_info.setStyleSheet("background: palette(alternate-base); padding: 4px; border-radius: 3px")
        form.addRow(self.model_info)
        self.readouts = QListWidget()
        self.readouts.setMaximumHeight(70)
        self.readouts.setToolTip("SDE models fit the first ticked readout; the hidden Markov model fits all ticked ones together.")
        form.addRow("Readouts", self.readouts)
        self.features = QListWidget()
        self.features.setMaximumHeight(110)
        self.features.setToolTip("Hidden Markov model only: fit these features (Features page) instead of readouts. "
                                 "Tick many for unbiased discovery of cell states.")
        self.features_label = QLabel("Features")
        form.addRow(self.features_label, self.features)
        self.conditions = QListWidget()
        self.conditions.setMaximumHeight(110)
        form.addRow("Conditions", self.conditions)
        self.group = QComboBox()
        self.group.setToolTip("With no condition ticked: fit this analysis group (empty = every condition).")
        form.addRow("…or group", self.group)
        row = QHBoxLayout()
        self.first = QSpinBox()
        self.first.setRange(0, 10 ** 6)
        self.last = QSpinBox()
        self.last.setRange(0, 10 ** 6)
        self.last.setSpecialValueText("end")
        row.addWidget(self.first)
        row.addWidget(QLabel("to"))
        row.addWidget(self.last)
        form.addRow("Movie frames", row)
        row = QHBoxLayout()
        self.window_hours = _spin(0, 1000, 1, 1, "one window", " h")
        self.window_hours.setToolTip("Split time into windows of this length; parameters shared 'per window' may then change over "
                                     "time (e.g. dwell times through differentiation). Needs the frame interval.")
        self.split_events = QCheckBox("new window at each event")
        row.addWidget(self.window_hours)
        row.addWidget(self.split_events)
        form.addRow("Time windows", row)
        # model options
        row = QHBoxLayout()
        self.modes = QSpinBox()
        self.modes.setRange(2, 6)
        self.modes.setPrefix("ERK modes ")
        self.states_n = QSpinBox()
        self.states_n.setRange(2, 10)
        self.states_n.setPrefix("states ")
        self.grid = QSpinBox()
        self.grid.setRange(16, 160)
        self.grid.setPrefix("grid ")
        self.grid.setToolTip("Cells the readout axis is divided into for SDE models. More = finer, slower (cost grows as cells²).")
        self.topology = QComboBox()
        self.topology.addItem("any state to any", "full")
        self.topology.addItem("chain (neighbours only)", "chain")
        for w in (self.modes, self.states_n, self.grid, self.topology):
            row.addWidget(w)
            if w is not self.topology:
                w.valueChanged.connect(self._options_changed)
        self.topology.currentIndexChanged.connect(self._options_changed)
        form.addRow("Options", row)
        self.states_box = QGroupBox("Hidden Markov model")
        sf = QFormLayout(self.states_box)
        row = QHBoxLayout()
        self.rank = QSpinBox()
        self.rank.setRange(0, 5)
        self.rank.setPrefix("shared directions ")
        self.rank.setToolTip("Correlated features: each state's spread is its own per feature plus this many directions shared "
                             "by features (nuclear area and DNA content move together). 0 = features independent within a state.")
        self.division = QCheckBox("divisions as observations")
        self.division.setToolTip("Each state gets a division rate: frames a cell goes on without dividing and observed divisions "
                                 "are evidence, and newborn cells start from their own state distribution. Needs features.")
        row.addWidget(self.rank)
        row.addWidget(self.division)
        sf.addRow(row)
        row = QHBoxLayout()
        self.sweep = QLineEdit()
        self.sweep.setPlaceholderText("e.g. 3, 4, 5, 6, 8")
        self.sweep.setToolTip("Fit each number of states on most positions and score it on the positions left out; the best "
                              "is then fitted on every position.")
        self.heldout = _spin(0.05, 0.6, 2, 0.05)
        self.heldout.setPrefix("held out ")
        row.addWidget(self.sweep, 1)
        row.addWidget(self.heldout)
        sf.addRow("Try states", row)
        row = QHBoxLayout()
        self.flux_feature = QComboBox()
        self.flux_feature.setToolTip("Routes from the state lowest in this feature to the highest (e.g. PPARG: undifferentiated to "
                                     "differentiated): which states cells pass through, and the chance from each to get there.")
        self.flux_from = QLineEdit()
        self.flux_from.setPlaceholderText("from")
        self.flux_from.setMaximumWidth(70)
        self.flux_to = QLineEdit()
        self.flux_to.setPlaceholderText("to")
        self.flux_to.setMaximumWidth(70)
        row.addWidget(self.flux_feature, 1)
        row.addWidget(self.flux_from)
        row.addWidget(self.flux_to)
        sf.addRow("Routes", row)
        form.addRow(self.states_box)
        self.rank.valueChanged.connect(self._options_changed)
        self.division.toggled.connect(self._options_changed)
        self.features.itemChanged.connect(lambda _: self._options_changed())
        self.drift = QLineEdit()
        self.noise_expr = QLineEdit()
        self.user_params = QLineEdit()
        self.user_params.setPlaceholderText("theta=0.05, mu=0, s=0.1")
        for w in (self.drift, self.noise_expr, self.user_params):
            w.editingFinished.connect(self._options_changed)
        self.custom_box = QGroupBox("Your SDE: du = drift dt + noise dW  (x = readout)")
        cf = QFormLayout(self.custom_box)
        cf.addRow("drift", self.drift)
        cf.addRow("noise", self.noise_expr)
        cf.addRow("parameters", self.user_params)
        form.addRow(self.custom_box)
        # parameters
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Parameter", "Start", "Fixed", "Low", "High", "Shared"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(220)
        self.table.setToolTip("Blank start/low/high = the model's default. Fixed keeps the start value (e.g. a literature KTR rate). "
                              "Shared: one value for everything, or a value per condition and/or per time window. "
                              "Values are in the fit's time unit (minutes when the interval is known).")
        form.addRow(self.table)
        # settings
        row = QHBoxLayout()
        self.max_tracks = QSpinBox()
        self.max_tracks.setRange(0, 10 ** 7)
        self.max_tracks.setSingleStep(500)
        self.max_tracks.setSpecialValueText("all")
        self.max_tracks.setPrefix("tracks/condition ")
        self.steps = QSpinBox()
        self.steps.setRange(10, 100000)
        self.steps.setSingleStep(50)
        self.steps.setPrefix("steps ")
        row.addWidget(self.max_tracks)
        row.addWidget(self.steps)
        form.addRow("Fitting", row)
        row = QHBoxLayout()
        self.bootstrap = QSpinBox()
        self.bootstrap.setRange(0, 200)
        self.bootstrap.setPrefix("bootstrap ")
        self.bootstrap.setToolTip("Refits on positions resampled with replacement: 95% intervals for every parameter. 10-20 is typical.")
        self.recovery = QCheckBox("recovery check")
        self.recovery.setToolTip("Simulate data from the fit (same tracks, frames, noise), refit from the default start, and show "
                                 "which parameters come back: what your frame interval can and cannot resolve.")
        self.outlier = _spin(0, 0.5, 3, 0.005)
        self.outlier.setPrefix("outliers ")
        self.outlier.setToolTip("Share of observations that may be wrong values (tracking swaps, bad masks), spread over the range.")
        row.addWidget(self.bootstrap)
        row.addWidget(self.recovery)
        row.addWidget(self.outlier)
        form.addRow(row)
        row = QHBoxLayout()
        self.batch = QSpinBox()
        self.batch.setRange(8, 100000)
        self.batch.setPrefix("batch ")
        self.lr = _spin(0.0001, 1, 4, 0.01)
        self.lr.setPrefix("step size ")
        self.seed = QSpinBox()
        self.seed.setRange(0, 10 ** 6)
        self.seed.setPrefix("seed ")
        for w in (self.batch, self.lr, self.seed):
            row.addWidget(w)
        form.addRow(row)

        self.list = _NamedList(editor, self._read, self._write, FIT_DEFAULT, "fit", checkable=True)
        self.list.changed.connect(self.changed)
        self.list.list.currentRowChanged.connect(lambda _: self.show_saved())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.list)
        side = QWidget()
        sl = QVBoxLayout(side)
        glob = QFormLayout()
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(["auto", "cuda:0", "cuda:1", "cpu"])
        glob.addRow("Device", self.device)
        self.interval = _spin(0, 10000, 3, 1, "from frames per hour", " min")
        self.interval.valueChanged.connect(self._show_interval)
        glob.addRow("Minutes between frames", self.interval)
        self.interval_note = QLabel("")
        self.interval_note.setWordWrap(True)
        glob.addRow(self.interval_note)
        sl.addLayout(glob)
        sl.addWidget(scroll, 1)
        row = QHBoxLayout()
        self.fit_button = QPushButton("Fit selected")
        self.fit_button.setToolTip("Saves the editor into the selected fit (or a new one), then fits it in the background.")
        self.fit_button.clicked.connect(self.fit_selected)
        self.fit_all = QPushButton("Fit all ticked")
        self.fit_all.clicked.connect(self.fit_ticked)
        self.stop = QPushButton("Stop")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(lambda: self.worker and self.worker.stop_flag.set())
        for b in (self.fit_button, self.fit_all, self.stop):
            row.addWidget(b)
        sl.addLayout(row)
        side.setMinimumWidth(460)
        side.setMaximumWidth(620)
        split.addWidget(side)

        right = QWidget()
        rl = QVBoxLayout(right)
        row = QHBoxLayout()
        self.view = QComboBox()
        for k, text in VIEWS.items():
            self.view.addItem(text, k)
        self.view.currentIndexChanged.connect(self.redraw)
        self.track_condition = QComboBox()
        self.track_condition.currentIndexChanged.connect(self.redraw)
        row.addWidget(self.view, 1)
        row.addWidget(self.track_condition)
        rl.addLayout(row)
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        rl.addWidget(NavigationToolbar2QT(self.canvas, self))
        rl.addWidget(self.canvas, 3)
        self.summary = QTextBrowser()
        rl.addWidget(self.summary, 2)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(90)
        rl.addWidget(self.log)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self._model_changed()

    # ------------------------------------------------------------------ editor
    def _options(self) -> dict:
        params = {}
        for part in self.user_params.text().split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    params[k.strip()] = float(v)
                except ValueError:
                    pass
        def numbers(text):
            return [int(t) for t in text.replace(";", ",").replace(" ", ",").split(",") if t.strip().isdigit()]

        return {"modes": self.modes.value(), "states": self.states_n.value(), "grid": self.grid.value(),
                "topology": self.topology.currentData(), "rank": self.rank.value(), "division": self.division.isChecked(),
                "sweep": numbers(self.sweep.text()), "flux_feature": self.flux_feature.currentData() or "",
                "flux_from": numbers(self.flux_from.text()), "flux_to": numbers(self.flux_to.text()),
                "drift": self.drift.text().strip() or "theta*(mu - x)",
                "noise": self.noise_expr.text().strip() or "s", "parameters": params}

    def _model_changed(self) -> None:
        name = self.model.currentData()
        cls = MODELS[name]
        self.model_info.setText((cls.__doc__ or "").strip().split("\n\n")[0].replace("\n", " "))
        self.modes.setVisible(name == "erk_ktr")
        self.states_n.setVisible(name == "states")
        self.topology.setVisible(name == "states")
        self.grid.setVisible(name != "states")
        self.custom_box.setVisible(name == "custom")
        self.states_box.setVisible(name == "states")
        self.features.setVisible(name == "states")
        self.features_label.setVisible(name == "states")
        self._options_changed()

    def _options_changed(self) -> None:
        overrides = self._read_table()
        try:
            n_in = len(self._ticked(self.features)) if self.model.currentData() == "states" and self._ticked(self.features) else len(self._ticked(self.readouts))
            model = make_model({"model": self.model.currentData(), "options": self._options()}, features=max(1, n_in))
            specs = model.specs()
        except Exception as exc:  # a formula with a typo
            self.model_info.setText(f"Model error: {exc}")
            return
        self._specs = specs
        self._fill_table(overrides)

    def _fill_table(self, overrides: dict) -> None:
        self.table.setRowCount(len(self._specs))
        for i, (name, spec) in enumerate(self._specs.items()):
            o = overrides.get(name, {})
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(f"{spec.text}\ndefault {spec.default:g} (rates per minute), bounds {spec.low:g} to {spec.high:g}")
            self.table.setItem(i, 0, item)
            for col, key in ((1, "value"), (3, "low"), (4, "high")):
                cell = QTableWidgetItem("" if o.get(key) is None else f"{o[key]:g}")
                self.table.setItem(i, col, cell)
            fixed = QTableWidgetItem()
            fixed.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            fixed.setCheckState(Qt.CheckState.Checked if o.get("fixed") else Qt.CheckState.Unchecked)
            self.table.setItem(i, 2, fixed)
            share = QComboBox()
            for s in SHARES:
                share.addItem(SHARE_TEXT[s], s)
            share.setCurrentIndex(SHARES.index(o.get("share") or spec.share))
            share.setEnabled(spec.unit != "logit")
            self.table.setCellWidget(i, 5, share)

    def _read_table(self) -> dict:
        out = {}
        for i in range(self.table.rowCount()):
            name = self.table.item(i, 0).text()
            spec = self._specs.get(name)
            entry = {}
            for col, key in ((1, "value"), (3, "low"), (4, "high")):
                cell = self.table.item(i, col)
                text = cell.text().strip() if cell else ""
                if text:
                    try:
                        entry[key] = float(text)
                    except ValueError:
                        pass
            fixed = self.table.item(i, 2)
            if fixed and fixed.checkState() == Qt.CheckState.Checked:
                entry["fixed"] = True
            combo = self.table.cellWidget(i, 5)
            if combo is not None and spec is not None and combo.currentData() != spec.share:
                entry["share"] = combo.currentData()
            if entry:
                out[name] = entry
        return out

    @staticmethod
    def _ticked(lst: QListWidget) -> list[str]:
        return [lst.item(i).text() for i in range(lst.count()) if lst.item(i).checkState() == Qt.CheckState.Checked]

    @staticmethod
    def _set_ticks(lst: QListWidget, names: list[str], ticked: list[str]) -> None:
        lst.blockSignals(True)
        lst.clear()
        for n in names + [t for t in ticked if t not in names]:
            item = QListWidgetItem(n)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if n in ticked else Qt.CheckState.Unchecked)
            lst.addItem(item)
        lst.blockSignals(False)

    def _read(self) -> dict:
        return {"name": self.name.text().strip(), "model": self.model.currentData(), "readouts": self._ticked(self.readouts),
                "features": self._ticked(self.features) if self.model.currentData() == "states" else [],
                "heldout": self.heldout.value(),
                "conditions": self._ticked(self.conditions), "group": self.group.currentData() or "",
                "frames": [self.first.value(), self.last.value()], "window_hours": self.window_hours.value(),
                "split_at_events": self.split_events.isChecked(), "options": self._options(), "parameters": self._read_table(),
                "outlier": self.outlier.value(), "max_tracks": self.max_tracks.value(), "steps": self.steps.value(),
                "batch": self.batch.value(), "lr": self.lr.value(), "bootstrap": self.bootstrap.value(),
                "recovery_check": self.recovery.isChecked(), "seed": self.seed.value()}

    def _write(self, fit: dict) -> None:
        f = merge(FIT_DEFAULT, fit)
        self.name.setText(f["name"])
        for w in (self.model, self.modes, self.states_n, self.grid, self.topology):
            w.blockSignals(True)
        self.model.setCurrentIndex(max(0, self.model.findData(f["model"])))
        o = f["options"]
        self.modes.setValue(int(o.get("modes", 2)))
        self.states_n.setValue(int(o.get("states", 3)))
        self.grid.setValue(int(o.get("grid", 48)))
        self.topology.setCurrentIndex(max(0, self.topology.findData(o.get("topology", "full"))))
        self.drift.setText(o.get("drift", ""))
        self.noise_expr.setText(o.get("noise", ""))
        self.user_params.setText(", ".join(f"{k}={v:g}" for k, v in (o.get("parameters") or {}).items()))
        self.rank.setValue(int(o.get("rank", 0)))
        self.division.setChecked(bool(o.get("division", False)))
        self.sweep.setText(", ".join(str(k) for k in o.get("sweep") or []))
        self.flux_from.setText(", ".join(str(k) for k in o.get("flux_from") or []))
        self.flux_to.setText(", ".join(str(k) for k in o.get("flux_to") or []))
        for w in (self.model, self.modes, self.states_n, self.grid, self.topology):
            w.blockSignals(False)
        exp = self.get_experiment()
        self._fill_features(exp, f["features"], o.get("flux_feature", ""))
        self.heldout.setValue(float(f["heldout"]))
        self._set_ticks(self.readouts, [r["name"] for r in exp["analysis"]["readouts"]], f["readouts"])
        self._set_ticks(self.conditions, [c["name"] for c in exp["plate"]["conditions"]], f["conditions"])
        self.group.setCurrentIndex(max(0, self.group.findData(f["group"])))
        self.first.setValue(int(f["frames"][0]))
        self.last.setValue(int(f["frames"][1]))
        self.window_hours.setValue(float(f["window_hours"]))
        self.split_events.setChecked(bool(f["split_at_events"]))
        self.outlier.setValue(float(f["outlier"]))
        self.max_tracks.setValue(int(f["max_tracks"]))
        self.steps.setValue(int(f["steps"]))
        self.batch.setValue(int(f["batch"]))
        self.lr.setValue(float(f["lr"]))
        self.bootstrap.setValue(int(f["bootstrap"]))
        self.recovery.setChecked(bool(f["recovery_check"]))
        self.seed.setValue(int(f["seed"]))
        self._model_changed()
        self._fill_table(f["parameters"])

    def _fill_features(self, exp: dict, ticked: list[str], flux: str) -> None:
        from ..analysis.features import feature_list
        names = [f["name"] for f in feature_list(exp, enabled_only=False)]
        self._set_ticks(self.features, names, ticked)
        self.flux_feature.blockSignals(True)
        self.flux_feature.clear()
        self.flux_feature.addItem("(none)", "")
        for n in names:
            self.flux_feature.addItem(n, n)
        if flux and self.flux_feature.findData(flux) < 0:
            self.flux_feature.addItem(flux, flux)
        self.flux_feature.setCurrentIndex(max(0, self.flux_feature.findData(flux)))
        self.flux_feature.blockSignals(False)

    def refresh(self, exp: dict) -> None:
        """Conditions, groups and readouts may have changed on other tabs."""
        readouts = [r["name"] for r in exp["analysis"]["readouts"]]
        ticked = self._ticked(self.readouts) or readouts[:1]
        self._set_ticks(self.readouts, readouts, [t for t in ticked if t in readouts])
        conds = [c["name"] for c in exp["plate"]["conditions"]]
        self._set_ticks(self.conditions, conds, [t for t in self._ticked(self.conditions) if t in conds])
        self._fill_features(exp, self._ticked(self.features), self.flux_feature.currentData() or "")
        current = self.group.currentData()
        self.group.clear()
        self.group.addItem("(every condition)", "")
        for g in exp["analysis"]["groups"]:
            self.group.addItem(g["name"], g["name"])
        self.group.setCurrentIndex(max(0, self.group.findData(current)))
        self._show_interval()

    def _show_interval(self) -> None:
        from ..dynamics.fit import time_unit
        exp = copy.deepcopy(self.get_experiment())
        exp["analysis"]["dynamics"]["interval_minutes"] = self.interval.value()
        dt, unit = time_unit(exp)
        if unit == "min":
            self.interval_note.setText(f"Fits use {dt:g} min between result frames; rates are per minute.")
        else:
            self.interval_note.setText(f"<b>Frame interval unknown</b>: rates come out per frame (starting values assume "
                                       f"{FRAME_MINUTES:g}-minute frames). Set it here or frames per hour on the Experiment page.")

    def load(self, dyn: dict) -> None:
        d = merge(DYNAMICS_DEFAULTS, dyn)
        self.device.setCurrentText(d["device"])
        self.interval.setValue(float(d["interval_minutes"]))
        self._folder = d["folder"]
        self.refresh(self.get_experiment())
        self.list.load(d["fits"])
        self.show_saved()

    def collect(self) -> dict:
        return {"device": self.device.currentText().strip() or "auto", "interval_minutes": self.interval.value(),
                "folder": getattr(self, "_folder", "Dynamics"), "fits": copy.deepcopy(self.list.items)}

    def rename_condition(self, old: str, new: str) -> None:
        for f in self.list.items:
            f["conditions"] = [new if c == old else c for c in f["conditions"]]

    def remove_condition(self, name: str) -> None:
        for f in self.list.items:
            f["conditions"] = [c for c in f["conditions"] if c != name]

    # ------------------------------------------------------------------ fitting
    def _experiment(self) -> dict:
        exp = copy.deepcopy(self.get_experiment())
        exp["analysis"]["dynamics"] = self.collect()
        return exp

    def fit_selected(self) -> None:
        self.list.update_selected()
        row = self.list.list.currentRow()
        if 0 <= row < len(self.list.items):
            self._run([copy.deepcopy(self.list.items[row])])

    def fit_ticked(self) -> None:
        self._run([copy.deepcopy(f) for f in self.list.items if f.get("enabled", True)])

    def _run(self, fits: list[dict]) -> None:
        if not fits or (self.worker is not None and self.worker.isRunning()):
            return
        from ..dynamics.fit import run_fit
        exp = self._experiment()

        def job(on_event, stop):
            from ..analysis.figures import ResultsCache
            cache = ResultsCache(exp)
            last = None
            for fit in fits:
                last = run_fit(exp, fit, on_event, stop, cache)
            return last

        self.log.clear()
        self.worker = FunctionWorker(job, self)
        self.worker.event.connect(self._event)
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.fit_button.setEnabled(False)
        self.fit_all.setEnabled(False)
        self.stop.setEnabled(True)
        self.worker.start()

    def _event(self, e: dict) -> None:
        if e["type"] == "log":
            self.log.appendPlainText(e["message"])
        elif e["type"] == "progress":
            extra = f", loss {e['loss']:.4f}" if "loss" in e else ""
            self.log.appendPlainText(f"{e['stage']} {e['done']}/{e['total']}{extra}")

    def _finish(self) -> None:
        self.fit_button.setEnabled(True)
        self.fit_all.setEnabled(True)
        self.stop.setEnabled(False)

    def _failed(self, message: str) -> None:
        self._finish()
        self.log.appendPlainText(message)

    def _done(self, report) -> None:
        self._finish()
        self.show_saved()

    def show_saved(self) -> None:
        from ..dynamics.fit import read_fit, read_states
        row = self.list.list.currentRow()
        if not (0 <= row < len(self.list.items)):
            return
        exp = self._experiment()
        if not exp["paths"]["results"]:
            return
        name = self.list.items[row]["name"]
        self.report = read_fit(exp, name)
        self.states = read_states(exp, name) if self.report else None
        self.track_condition.blockSignals(True)
        self.track_condition.clear()
        for c in (self.report or {}).get("conditions", []):
            self.track_condition.addItem(c)
        self.track_condition.blockSignals(False)
        self.redraw()
        self.summary.setHtml(self._summary_html(exp))

    def redraw(self) -> None:
        from ..dynamics.plots import draw
        if not self.report:
            self.figure.clear()
            self.figure.text(0.5, 0.5, "Not fitted yet", ha="center")
            self.canvas.draw_idle()
            return
        try:
            draw(self.figure, self.report, self.view.currentData(), self.get_experiment(), self.states,
                 max(0, self.track_condition.currentIndex()))
        except Exception as exc:  # a view the saved fit has no data for
            self.figure.clear()
            self.figure.text(0.5, 0.5, f"Cannot draw this view: {exc}", ha="center", wrap=True)
        self.canvas.draw_idle()

    def _summary_html(self, exp: dict) -> str:
        from ..dynamics.fit import compare
        r = self.report
        if not r:
            return "<i>Not fitted yet. Set it up on the left, then Fit selected.</i>"
        conds = r["conditions"]
        W = len(r["windows"])

        def fmt(entry, c, w):
            v = entry["values"][c][w]
            s = f"{v:.3g}"
            if entry.get("ci"):
                s += f" <span style='color:gray'>[{entry['ci'][0][c][w]:.3g}, {entry['ci'][1][c][w]:.3g}]</span>"
            if entry.get("at_bound") and entry["at_bound"][c][w]:
                s += " <b style='color:#cc3d3d'>bound</b>"
            return s

        head = "".join(f"<th>{c}</th>" for c in conds)
        html = [f"<b>{r['name']}</b>: {r['model_text']}. {r['tracks']} tracks ({', '.join(str(n) for n in r['tracks_per_condition'])}), "
                f"{r['observations']} observations, frame interval {r['interval']:g} {r['time_unit']}, {W} window(s). "
                f"Log-likelihood {r['log_likelihood']:.1f}, BIC {r['bic']:.1f}. Fitted {r['fitted']} in {r['seconds']} s on {r['device']}."]
        w_last = W - 1
        for title, table in (("Derived", r["derived"]), ("Parameters", r["parameters"])):
            html.append(f"<p><b>{title}</b>" + (" (last window)" if W > 1 else "") + f"</p><table cellpadding=3><tr><th></th><th>unit</th>{head}</tr>")
            many = r.get("model") == "states" and len(r.get("feature_names") or r.get("readouts") or []) > 3
            for name, entry in table.items():
                if entry.get("fixed") or (many and name.split("_")[0] in ("mean", "sd", "load")):
                    continue
                cells = "".join(f"<td>{fmt(entry, c, w_last)}</td>" for c in range(len(conds)))
                html.append(f"<tr><td title='{entry.get('text', '')}'>{name}</td><td>{entry.get('unit', '')}</td>{cells}</tr>")
            html.append("</table>")
        if r.get("sweep"):
            best = r.get("states")
            html.append("<p><b>Number of states</b>, scored on positions left out of fitting (higher is better): " +
                        ", ".join(f"{'<b>' if x['states'] == best else ''}{x['states']}: {x['heldout_ll']:.3f}"
                                  f"{'</b>' if x['states'] == best else ''}" for x in r["sweep"]) + "</p>")
        if r.get("flux"):
            fl = r["flux"]
            times = ", ".join(f"{c} {np.asarray(fl['time_to'], float)[i, -1]:.3g}" for i, c in enumerate(conds))
            html.append(f"<p><b>Routes</b> from state(s) {fl['from']} to {fl['to']}: expected time ({r['time_unit']}) {times}. "
                        "See the routes view for the committor of every state.</p>")
        if many:
            html.append("<p style='color:gray'>State means, spreads and loadings are in the profiles view.</p>")
        if r.get("recovery"):
            bad = [n for n, e in r["recovery"].items() if not np.all(e["ok"])]
            html.append("<p><b>Recovery check:</b> " + ("every parameter came back from simulated data." if not bad else
                        f"not recovered from simulated data: {', '.join(bad)}. Treat these as not determined by data like yours.") + "</p>")
        rows = [x for x in compare(exp) if x["data_key"] == r["data_key"]]
        if len(rows) > 1:
            best = min(x["bic"] for x in rows)
            html.append("<p><b>Fits on the same tracks</b> (lower BIC is better; differences above ~10 are strong):</p><table cellpadding=3>")
            for x in rows:
                html.append(f"<tr><td>{x['name']}</td><td>{x['model']}</td><td>ΔBIC {x['bic'] - best:.1f}</td><td>{x['free_numbers']} free</td></tr>")
            html.append("</table>")
        return "".join(html)


# ---------------------------------------------------------------------- the analysis pages
class AnalysisSection(QWidget):
    """Figures, Readouts, Features, Fates, Dynamics and Export, read and written together. Built as sub-tabs; the
    main window moves each into a page of its own and points `tabs` at its navigator."""
    changed = Signal()

    def __init__(self, get_experiment):
        super().__init__()
        self.get_experiment = get_experiment
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.figures = AnalysisTab(get_experiment)
        self.readouts = ReadoutsPanel(get_experiment)
        self.dynamics = DynamicsPanel(lambda: self._with_features())
        from .features_tab import FatePanel, FeaturesPanel
        self.features = FeaturesPanel(self._with_features)
        self.fates = FatePanel(self._with_features)
        from .export_tab import ExportPanel
        self.export = ExportPanel(get_experiment, self.figures._cache_for)
        from .summary_tab import SummaryPanel
        self.summary = SummaryPanel(get_experiment, self.figures._cache_for)
        self.summary.copy_figure.connect(self.figures.add_figure)
        self.summary.warm = self.figures.warm_up
        self.figures.cache_ready.connect(self.summary.cache_built)
        self.tabs.addTab(self.summary, "Summary")
        self.tabs.addTab(self.figures, "Figures")
        self.tabs.addTab(self.readouts, "Readouts")
        self.tabs.addTab(self.features, "Features")
        self.tabs.addTab(self.fates, "Fates")
        self.tabs.addTab(self.dynamics, "Dynamics")
        self.tabs.addTab(self.export, "Export")
        self.tabs.currentChanged.connect(lambda _: self.refresh_choices())
        lay.addWidget(self.tabs)
        for w in (self.figures, self.readouts, self.features, self.fates, self.dynamics, self.export, self.summary):
            w.changed.connect(self.changed)

    def _with_features(self) -> dict:
        """The experiment with this tab's readouts and features as currently edited (the fate editor lists them)."""
        exp = copy.deepcopy(self.get_experiment())
        if hasattr(self, "features"):
            exp["analysis"]["readouts"] = self.readouts.collect()
            exp["analysis"]["features"] = self.features.collect()
        return exp

    def load(self, exp: dict) -> None:
        self.figures.load(exp)
        self.readouts.refresh(exp)
        self.readouts.load(exp["analysis"]["readouts"])
        self.features.load(exp["analysis"]["features"])
        self.fates.load(exp["analysis"]["fates"])
        self.dynamics.load(exp["analysis"]["dynamics"])
        self.export.load(exp["figures"])
        self.summary.load(exp["figures"]["summary"])

    def collect(self) -> dict:
        parts = self.figures.collect()
        parts["analysis"]["readouts"] = self.readouts.collect()
        parts["analysis"]["features"] = self.features.collect()
        parts["analysis"]["fates"] = self.fates.collect()
        parts["analysis"]["dynamics"] = self.dynamics.collect()
        parts["figures"]["exports"] = self.export.collect()
        parts["figures"]["summary"] = self.summary.collect()
        return parts

    def refresh_choices(self, exp: dict | None = None) -> None:
        exp = exp or self.get_experiment()
        self.figures.refresh_choices(exp)
        self.readouts.refresh(exp)
        self.features.refresh(self._with_features())
        self.fates.refresh(self._with_features())
        self.dynamics.refresh(exp)

    def rename_condition(self, old: str, new: str) -> None:
        self.figures.rename_condition(old, new)
        self.fates.rename_condition(old, new)
        self.dynamics.rename_condition(old, new)
        self.export.rename_condition(old, new)

    def remove_condition(self, name: str) -> None:
        self.figures.remove_condition(name)
        self.features.remove_condition(name)
        self.fates.remove_condition(name)
        self.dynamics.remove_condition(name)
        self.export.remove_condition(name)

    def refresh_list(self) -> None:
        self.figures.refresh_list()

    def draw_all(self) -> None:
        self.figures.draw_all()
