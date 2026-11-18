"""Run: find the experiment's timestacks, track the ones its plate selects, watch stage timings."""
from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..experiment import condition_of_well, select_timestacks, validate
from ..io import layout as io_layout
from .worker import FunctionWorker


class RunTab(QWidget):
    results_ready = Signal()
    timestacks_found = Signal(dict)
    cache_ready = Signal(dict)
    figures_requested = Signal()
    run_finished = Signal()          # a run ended normally: summary figures redraw
    quality_checked = Signal(dict)
    files_flagged = Signal(dict)    # {position: reason}, new flags only

    def __init__(self, get_experiment, get_cfg):
        super().__init__()
        self.get_experiment = get_experiment   # collects every tab into the experiment dict
        self.get_cfg = get_cfg
        self.timestacks: dict = {}
        self.chosen: list[str] = []
        self.worker: FunctionWorker | None = None

        root = QVBoxLayout(self)
        self.where = QLabel("")
        self.where.setWordWrap(True)
        self.where.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.where)

        bar = QHBoxLayout()
        self.find_button = QPushButton("Find timestacks")
        self.find_button.clicked.connect(self.discover)
        self.run_button = QPushButton("Run")
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.start)
        self.run_button.setEnabled(False)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.figures_after = QCheckBox("Draw the experiment's figures afterwards")
        self.figures_after.setChecked(True)
        self.summary = QLabel("Set the folders on the Experiment page, then find the timestacks.")
        self.summary_button = QPushButton("Run summary")
        self.summary_button.setToolTip("What to check by hand after the last run, and a link to the detailed log.")
        self.summary_button.clicked.connect(self.show_summary)
        for w in (self.find_button, self.run_button, self.stop_button, self.figures_after, self.summary_button):
            bar.addWidget(w)
        bar.addWidget(self.summary, 1)
        root.addLayout(bar)
        pick = QHBoxLayout()
        pick.addWidget(QLabel("Run now:"))
        for text, slot in (("Tick all", lambda: self._tick(lambda k: True)), ("Untick all", lambda: self._tick(lambda k: False))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            pick.addWidget(b)
        self.condition_pick = QComboBox()
        self.condition_pick.setMinimumWidth(160)
        pick.addWidget(self.condition_pick)
        only = QPushButton("Tick only this condition")
        only.clicked.connect(lambda: self._tick(lambda k: self._condition(k) == self.condition_pick.currentText()))
        pick.addWidget(only)
        hint = QLabel("Untick positions to leave them for later; finished ones stay as they are.")
        hint.setProperty("role", "hint")
        pick.addWidget(hint, 1)
        root.addLayout(pick)
        self.progress = QProgressBar()
        root.addWidget(self.progress)

        split = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Timestack", "Condition", "Frames", "Status", "Cells", "Tracks", "Divisions", "Seconds"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        split.addWidget(self.table)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        split.addWidget(self.log)
        split.setSizes([400, 200])
        root.addWidget(split, 1)

    def show_paths(self) -> None:
        exp = self.get_experiment()
        raw = exp["paths"]["raw"] or "(not set)"
        res = exp["paths"]["results"] or "(not set)"
        self.where.setText(f"<b>Raw data</b> {raw}<br><b>Results</b> {res}")

    def discover(self, quiet: bool = False) -> None:
        """Find the positions. `quiet` (from other tabs): no dialogs, just leave the list empty."""
        self.show_paths()
        cfg = self.get_cfg()
        if not cfg["data"]["root"]:
            if not quiet:
                QMessageBox.information(self, "Raw data folder needed", "Choose the raw data folder on the Experiment page.")
            return
        try:
            self.set_timestacks(io_layout.discover_timestacks(cfg))
        except (OSError, ValueError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Could not read the data folder", str(exc))

    def set_timestacks(self, timestacks: dict) -> None:
        exp = self.get_experiment()
        cfg = self.get_cfg()
        self.timestacks = timestacks
        self.chosen = select_timestacks(exp, list(timestacks))
        conditions = condition_of_well(exp)
        plate = exp["plate"]
        output = Path(cfg["data"]["output"]) if cfg["data"]["output"] else None
        self.table.setRowCount(len(timestacks))
        for row, (key, ts) in enumerate(timestacks.items()):
            if key in self.chosen:
                done = output is not None and (output / "Traces" / f"{key}.h5").exists()
                status = "Done" if done else "Waiting"
            elif key in plate["excluded_positions"] or ts["well"] in plate["excluded_wells"]:
                status = "Excluded"
            elif key in exp["quality"]["flagged"]:
                status = "Flagged by quality check"
            else:
                status = "No condition"
            self._set_row(row, [key, conditions.get(ts["well"], ""), ts["frames"], status])
            item = self.table.item(row, 0)
            if key in self.chosen:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
        self.condition_pick.clear()
        self.condition_pick.addItems([c["name"] for c in exp["plate"]["conditions"]])
        frames = sum(timestacks[k]["frames"] for k in self.chosen)
        self.summary.setText(f"{len(timestacks)} timestacks found; {len(self.chosen)} to process, {frames} frames.")
        self.run_button.setEnabled(bool(self.chosen))
        self.timestacks_found.emit(timestacks)
        if any(self.table.item(r, 3).text() == "Done" for r in range(self.table.rowCount())):
            self.results_ready.emit()

    def show_summary(self) -> None:
        """The last run's summary (things to check by hand), with a button for the detailed log."""
        from pathlib import Path as _P

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        path = getattr(self, "last_summary", None)
        if path is None:
            results = self.get_experiment()["paths"]["results"]
            latest = _P(results) / "Logs" / "latest_summary.txt" if results else None
            path = str(latest) if latest is not None and latest.exists() else None
        if path is None:
            QMessageBox.information(self, "No run yet", "Run tracking first; each run writes a summary and a detailed log to Logs/ in the results folder.")
            return
        text = _P(path).read_text()
        dialog = QDialog(self)
        dialog.setWindowTitle("Run summary")
        dialog.resize(900, 600)
        lay = QVBoxLayout(dialog)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        lay.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        detailed = buttons.addButton("Open detailed log", QDialogButtonBox.ButtonRole.ActionRole)
        folder = buttons.addButton("Open Logs folder", QDialogButtonBox.ButtonRole.ActionRole)
        logs = _P(path).parent
        stamp = _P(path).name.replace("_summary.txt", "")
        candidates = sorted(logs.glob(f"{stamp}_detailed.log")) or sorted(logs.glob("*_detailed.log"))
        detailed.clicked.connect(lambda: candidates and QDesktopServices.openUrl(QUrl.fromLocalFile(str(candidates[-1]))))
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs))))
        buttons.rejected.connect(dialog.reject)
        lay.addWidget(buttons)
        self._summary_dialog = dialog  # kept, so the non-blocking window stays open
        dialog.show()

    def _condition(self, key: str) -> str:
        item = self.table.item(self._row_of(key), 1)
        return item.text() if item else ""

    def _tick(self, keep) -> None:
        for key in self.chosen:
            self.table.item(self._row_of(key), 0).setCheckState(Qt.CheckState.Checked if keep(key) else Qt.CheckState.Unchecked)

    def ticked(self) -> list[str]:
        return [k for k in self.chosen if self.table.item(self._row_of(k), 0).checkState() == Qt.CheckState.Checked]

    def _set_row(self, row: int, values: list) -> None:
        for col, value in enumerate(values):
            self.table.setItem(row, col, QTableWidgetItem(str(value)))

    def _row_of(self, key: str) -> int:
        return list(self.timestacks).index(key)

    def start(self) -> None:
        exp = copy.deepcopy(self.get_experiment())  # the run records quality flags in its own copy
        problems = validate(exp)
        if problems:
            QMessageBox.information(self, "Not ready to run", "\n".join(problems))
            return
        self.progress.setValue(0)
        timestacks = self.timestacks
        only = self.ticked() if self.timestacks else None
        if only is not None and not only:
            QMessageBox.information(self, "Nothing ticked", "Tick the positions to run.")
            return

        def job(on_event, stop):
            from ..track.run import run_experiment
            return run_experiment(exp, on_event, stop, gui_cache=True, timestacks=timestacks or None, only=only)

        self.worker = FunctionWorker(job, self)
        self.worker.event.connect(self.on_event)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished_ok.connect(lambda _: self.figures_after.isChecked() and self.figures_requested.emit())
        self.worker.finished_ok.connect(lambda _: self.run_finished.emit())
        self.worker.finished.connect(self.on_finished)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.start()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop_flag.set()
            self.stop_button.setEnabled(False)
            self.log.appendPlainText("Stopping after the current batch…")

    def on_event(self, e: dict) -> None:
        kind = e["kind"]
        if kind == "experiment_start":
            self.progress.setRange(0, max(len(e["timestacks"]), 1))
        elif kind == "timestack_start":
            self.table.setItem(self._row_of(e["id"]), 3, QTableWidgetItem("Running"))
            for problem in e.get("problems", []):
                self.log.appendPlainText(f"{e['id']}: {problem}")
        elif kind == "gui_cache":
            self.cache_ready.emit(e["cache"])
        elif kind == "batch":
            stages = "  ".join(f"{k} {v:.2f}s" for k, v in e["timings"].items())
            self.log.appendPlainText(f"{e['timestack']}  frames {e['frames_done']}/{e['frames_total']}  {e['cells']} cells  |  {stages}")
        elif kind == "timestack_skipped":
            self.progress.setValue(self.progress.value() + 1)
        elif kind == "timestack_done":
            row = self._row_of(e["id"])
            condition = self.table.item(row, 1).text() if self.table.item(row, 1) else ""
            self._set_row(row, [e["id"], condition, e["frames"], "Done", e["cells"], e["tracks"], e["divisions"], f"{e['seconds']:.1f}"])
            self.progress.setValue(self.progress.value() + 1)
            self.log.appendPlainText(f"{e['id']}: gate {e['max_travel']:.1f} px per frame, batch sizes {e['batch_sizes']}")
            self.results_ready.emit()
        elif kind == "run_summary":
            self.last_summary = e["path"]
            self.show_summary()
        elif kind == "quality_start":
            self.log.appendPlainText(f"Checking {e['positions']} positions for fibers, debris and empty fields…")
        elif kind == "quality_done":
            self.log.appendPlainText(f"Quality check: {len(e['flagged'])} of {e['positions']} positions flagged and skipped.")
            for key, reason in e["flagged"].items():
                self.log.appendPlainText(f"  {key}: {reason}")
            self.quality_checked.emit(e["flagged"])
        elif kind == "files_flagged":
            for key, reason in e["new"].items():
                self.log.appendPlainText(f"Flagged {key}: {reason}")
            self.files_flagged.emit(e["new"])
        elif kind == "warning":
            self.log.appendPlainText(f"Warning: {e['message']}")
        elif kind == "timestack_failed":
            self.table.setItem(self._row_of(e["id"]), 3, QTableWidgetItem("Failed (see log)"))
            self.progress.setValue(self.progress.value() + 1)
        elif kind == "experiment_done":
            self.log.appendPlainText(f"Finished {len(e['summaries'])} timestacks.")

    def on_failed(self, message: str) -> None:
        self.log.appendPlainText(message)
        QMessageBox.critical(self, "Run stopped by an error", message.split("\n")[0])

    def on_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
