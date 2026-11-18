"""Experiment: name, notes, folders, which frames to use, the channels, and events in time.

"Scan data folder" reads the folder once, fills the channel table with the channel numbers
it finds and greys out wells without images on the Plate page. Channels are named and given
a role here; nothing about channels or timing is assumed by the tracking.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..analysis.summary import PRIMARY, PRIMARY_CHOICES, SHORT, resolve_primary
from ..experiment import PALETTE, ROLES, new_experiment, pipeline_config
from ..io import layout as io_layout
from .worker import FunctionWorker


def _color_button(color: str, title: str) -> QPushButton:
    button = QPushButton()
    button.setProperty("color", color)

    def paint():
        button.setStyleSheet(f"background: {button.property('color')}; border: 1px solid #888; min-width: 40px")

    def pick():
        chosen = QColorDialog.getColor(QColor(button.property("color")), button, title)
        if chosen.isValid():
            button.setProperty("color", chosen.name())
            paint()

    button.clicked.connect(pick)
    paint()
    return button


class ExperimentTab(QWidget):
    changed = Signal()
    scanned = Signal(dict)      # {timestack id: timestack}, every channel

    def __init__(self):
        super().__init__()
        self.worker = None
        self.get_quality = None     # set by the app: the Quality page's settings (the missing-frames limit)
        self.suggest_last = 0
        root = QHBoxLayout(self)
        left, right = QVBoxLayout(), QVBoxLayout()
        root.addLayout(left, 1)
        root.addLayout(right, 1)

        about = QGroupBox("Experiment")
        form = QFormLayout(about)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. 2023-01-12 Mosaic MZ6 siRNA")
        form.addRow("Name", self.name)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Cells, reporters, treatments, anything worth keeping with the analysis.")
        self.notes.setMaximumHeight(110)
        form.addRow("Notes", self.notes)
        left.addWidget(about)

        folders = QGroupBox("Folders")
        form = QFormLayout(folders)
        self.raw = self._path_row(form, "Raw data (read only)", "Folder with the TIFFs")
        self.results = self._path_row(form, "Results", "Where tracks, measurements and figures go")
        left.addWidget(folders)

        files = QGroupBox("Files and frames")
        form = QFormLayout(files)
        self.pattern = QLineEdit()
        self.pattern.setToolTip("Regular expression with named groups: well (optional), t (optional), pos, chan.")
        form.addRow("Filename pattern", self.pattern)
        frames = QHBoxLayout()
        self.first = QSpinBox()
        self.first.setRange(0, 10**6)
        self.last = QSpinBox()
        self.last.setRange(0, 10**6)
        self.last.setSpecialValueText("end")
        self.stride = QSpinBox()
        self.stride.setRange(1, 1000)
        for label, w in (("from", self.first), ("to", self.last), ("every", self.stride)):
            frames.addWidget(QLabel(label))
            frames.addWidget(w)
        frames.addStretch()
        form.addRow("Frames", frames)
        self.io_threads = QSpinBox()
        self.io_threads.setRange(1, 128)
        form.addRow("Reader threads", self.io_threads)
        scan_row = QHBoxLayout()
        self.scan_button = QPushButton("Scan data folder")
        self.scan_button.clicked.connect(self.scan)
        self.scan_info = QLabel("")
        self.scan_info.setWordWrap(True)
        scan_row.addWidget(self.scan_button)
        scan_row.addWidget(self.scan_info, 1)
        form.addRow(scan_row)
        self.scan_problems = QPlainTextEdit()
        self.scan_problems.setReadOnly(True)
        self.scan_problems.setMaximumHeight(140)
        self.scan_problems.setToolTip("Missing files are read as blank frames. A position with more missing, unreadable or cut-off "
                                      "frames than the limit on the Quality page is flagged and left out unless you tick Track anyway.")
        self.scan_problems.hide()
        form.addRow(self.scan_problems)
        self.trim_button = QPushButton("")
        self.trim_button.setToolTip("Sets Frames 'to', so every position ends on the same frame. Nothing changes until you press it.")
        self.trim_button.clicked.connect(self._apply_trim)
        self.trim_button.hide()
        form.addRow(self.trim_button)
        left.addWidget(files)
        left.addStretch()

        chans = QGroupBox("Channels")
        lay = QVBoxLayout(chans)
        self.channels = QTableWidget(0, 6)
        self.channels.setHorizontalHeaderLabels(["File ch.", "Name", "Role", "Reporter", "Primary measurement", "Colour"])
        self.channels.horizontalHeaderItem(4).setToolTip(
            "What this reporter is read out as: summary figures and default metrics use it.\n"
            "auto: cytoplasm / nucleus for a KTR, integrated intensity for H2B or DNA dyes, else the nuclear mean.")
        self.channels.itemChanged.connect(self._reporter_edited)
        header = self.channels.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for col, width in ((0, 70), (2, 110), (5, 60)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            self.channels.setColumnWidth(col, width)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setMinimumSectionSize(50)
        self.channels.verticalHeader().hide()
        self.channels.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        lay.addWidget(self.channels)
        row = QHBoxLayout()
        for label, slot in (("Add", lambda: self._add_channel({"file_channel": self.channels.rowCount() + 1})),
                            ("Remove", lambda: self._remove_row(self.channels))):
            b = QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        hint = QLabel("One channel is nuclear (segmentation). 'skip' channels are not read. Empty table: every channel, the first is nuclear.")
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")
        row.addWidget(hint, 1)
        lay.addLayout(row)
        right.addWidget(chans, 1)

        timing = QGroupBox("Time")
        lay = QVBoxLayout(timing)
        form = QFormLayout()
        self.fph = QDoubleSpinBox()
        self.fph.setRange(0, 10000)
        self.fph.setDecimals(3)
        self.fph.setSpecialValueText("not set (axes in frames)")
        self.fph.setToolTip("Only labels figure axes in hours. Tracking never uses it.")
        form.addRow("Frames per hour", self.fph)
        lay.addLayout(form)
        lay.addWidget(QLabel("Events: a change during the movie, e.g. a drug addition. Frames count from the start of the movie."))
        self.events = QTableWidget(0, 3)
        self.events.setHorizontalHeaderLabels(["Event", "Frame", "Colour"])
        self.events.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.events)
        row = QHBoxLayout()
        for label, slot in (("Add event", lambda: self._add_event({"name": "Treatment", "frame": 0, "color": "#444444"})),
                            ("Remove", lambda: self._remove_row(self.events))):
            b = QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        right.addWidget(timing, 1)

    # ------------------------------------------------------------------ helpers
    def _path_row(self, form, label, placeholder) -> QLineEdit:
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(self.changed.emit)
        button = QPushButton("Browse…")

        def browse():
            path = QFileDialog.getExistingDirectory(self, label, edit.text() or str(Path.home()))
            if path:
                edit.setText(path)
                self.changed.emit()

        button.clicked.connect(browse)
        row.addWidget(edit, 1)
        row.addWidget(button)
        form.addRow(label, row)
        return edit

    def _remove_row(self, table: QTableWidget) -> None:
        rows = sorted({i.row() for i in table.selectedIndexes()}, reverse=True) or ([table.rowCount() - 1] if table.rowCount() else [])
        for r in rows:
            table.removeRow(r)
        self.changed.emit()

    def _add_channel(self, ch: dict) -> None:
        r = self.channels.rowCount()
        self.channels.insertRow(r)
        number = QSpinBox()
        number.setRange(0, 99)
        number.setValue(int(ch.get("file_channel", r + 1)))
        self.channels.setCellWidget(r, 0, number)
        self.channels.setItem(r, 1, QTableWidgetItem(ch.get("name") or f"c{number.value()}"))
        role = QComboBox()
        role.addItems(ROLES)
        role.setCurrentText(ch.get("role") or ("nuclear" if r == 0 else "measure"))
        self.channels.setCellWidget(r, 2, role)
        self.channels.setItem(r, 3, QTableWidgetItem(ch.get("reporter", "")))
        primary = QComboBox()
        for key in PRIMARY_CHOICES:
            primary.addItem(key, key)
        primary.setCurrentIndex(max(0, primary.findData(ch.get("primary") or "auto")))
        self.channels.setCellWidget(r, 4, primary)
        self.channels.setCellWidget(r, 5, _color_button(ch.get("color") or PALETTE[r % len(PALETTE)], "Channel colour"))
        self._label_primary(r)

    def _label_primary(self, r: int) -> None:
        """Show what each choice means, and what auto picks for this reporter."""
        combo = self.channels.cellWidget(r, 4)
        if combo is None:
            return
        reporter = self.channels.item(r, 3).text() if self.channels.item(r, 3) else ""
        name = self.channels.item(r, 1).text() if self.channels.item(r, 1) else ""
        auto = SHORT[resolve_primary({"primary": "auto", "reporter": reporter, "name": name})]
        for i in range(combo.count()):
            key = combo.itemData(i)
            combo.setItemText(i, f"auto: {auto}" if key == "auto" else SHORT[key].capitalize())
            combo.setItemData(i, "" if key == "auto" else PRIMARY[key][0], Qt.ItemDataRole.ToolTipRole)

    def _reporter_edited(self, item) -> None:
        if item.column() in (1, 3):
            self._label_primary(item.row())

    def _add_event(self, ev: dict) -> None:
        r = self.events.rowCount()
        self.events.insertRow(r)
        self.events.setItem(r, 0, QTableWidgetItem(ev["name"]))
        frame = QSpinBox()
        frame.setRange(-10**6, 10**6)
        frame.setValue(int(ev["frame"]))
        self.events.setCellWidget(r, 1, frame)
        self.events.setCellWidget(r, 2, _color_button(ev["color"], "Event colour"))

    # ------------------------------------------------------------------ scan
    def scan(self) -> None:
        exp = new_experiment(**self.collect())
        cfg = pipeline_config(exp)
        cfg["data"]["channels"] = []  # find every channel
        if not cfg["data"]["root"] or not Path(cfg["data"]["root"]).is_dir():
            self.scan_info.setText("Choose an existing raw data folder first.")
            return
        self.scan_button.setEnabled(False)
        self.scan_info.setText("Reading the folder…")
        self.worker = FunctionWorker(lambda on_event, stop: io_layout.discover_timestacks(cfg), self)
        self.worker.finished_ok.connect(self._scanned)
        self.worker.failed.connect(lambda m: self.scan_info.setText(f"Could not read the folder: {m.splitlines()[0]}"))
        self.worker.finished.connect(lambda: self.scan_button.setEnabled(True))
        self.worker.start()

    def _scanned(self, timestacks: dict) -> None:
        if not timestacks:
            self.scan_info.setText("No files match the filename pattern.")
            return
        found = sorted({c for ts in timestacks.values() for c in ts["channels"]})
        wells = sorted({ts["well"] for ts in timestacks.values()})
        report = io_layout.scan_report(timestacks)
        self.scan_info.setText(f"{len(timestacks)} positions in {len(wells)} wells, {report['usual_frames']} frames, channels {found}."
                               + (" Some files are missing: see below." if report["missing"] or report["short"] else ""))
        self._show_scan_problems(timestacks, report)
        known = {self.channels.cellWidget(r, 0).value() for r in range(self.channels.rowCount())}
        for c in found:
            if c not in known:
                self._add_channel({"file_channel": c})
        self.scanned.emit(timestacks)
        self.changed.emit()

    def _show_scan_problems(self, timestacks: dict, report: dict) -> None:
        from ..quality import file_flags
        limit = int(self.get_quality().get("max_missing_frames", 1)) if self.get_quality else 1
        flags = file_flags(timestacks, {"max_missing_frames": limit})
        lines = []
        for key, ts in timestacks.items():
            if not ts["problems"]:
                continue
            verdict = "will be flagged and left out" if key in flags else "tracked with blank frames"
            lines.append(f"{key} ({verdict}): " + "; ".join(ts["problems"]))
        for key, n in report["longer"].items():
            lines.append(f"{key}: {n} frames longer than most (extra frames are tracked)")
        if lines:
            head = (f"{len(report['missing'])} positions with missing files, {len(report['short'])} ending early. "
                    f"More than {limit} bad frame{'s' * (limit != 1)} flags a position (limit on the Quality page).")
            self.scan_problems.setPlainText("\n".join([head, ""] + lines))
        self.scan_problems.setVisible(bool(lines))
        self.suggest_last = report["suggest_last"]
        if self.suggest_last:
            self.trim_button.setText(f"End every position at frame {self.suggest_last} "
                                     f"({len(report['short'])} positions lack their last 1-3 frames)")
        self.trim_button.setVisible(bool(self.suggest_last))

    def _apply_trim(self) -> None:
        self.last.setValue(int(self.suggest_last))
        self.trim_button.hide()
        self.scan_info.setText(self.scan_info.text() + f" Frames 'to' set to {self.suggest_last}; scan again to check.")
        self.changed.emit()

    # ------------------------------------------------------------------ experiment file
    def load(self, exp: dict) -> None:
        self.name.setText(exp["experiment"]["name"])
        self.notes.setPlainText(exp["experiment"]["notes"])
        self.raw.setText(exp["paths"]["raw"])
        self.results.setText(exp["paths"]["results"])
        f = exp["files"]
        self.pattern.setText(f["pattern"])
        self.first.setValue(f["first_frame"])
        self.last.setValue(f["last_frame"])
        self.stride.setValue(f["stride"])
        self.io_threads.setValue(f["io_threads"])
        self.channels.setRowCount(0)
        for ch in exp["channels"]:
            self._add_channel(ch)
        self.fph.setValue(exp["time"]["frames_per_hour"])
        self.events.setRowCount(0)
        for ev in exp["time"]["events"]:
            self._add_event(ev)
        self.scan_info.setText("")

    def collect(self) -> dict:
        channels = []
        for r in range(self.channels.rowCount()):
            channels.append({"file_channel": self.channels.cellWidget(r, 0).value(),
                             "name": (self.channels.item(r, 1).text() if self.channels.item(r, 1) else "").strip(),
                             "role": self.channels.cellWidget(r, 2).currentText(),
                             "reporter": (self.channels.item(r, 3).text() if self.channels.item(r, 3) else "").strip(),
                             "primary": self.channels.cellWidget(r, 4).currentData() or "auto",
                             "color": self.channels.cellWidget(r, 5).property("color")})
        events = [{"name": (self.events.item(r, 0).text() if self.events.item(r, 0) else "").strip(),
                   "frame": self.events.cellWidget(r, 1).value(),
                   "color": self.events.cellWidget(r, 2).property("color")} for r in range(self.events.rowCount())]
        return {
            "experiment": {"name": self.name.text().strip(), "notes": self.notes.toPlainText()},
            "paths": {"raw": self.raw.text().strip(), "results": self.results.text().strip()},
            "files": {"pattern": self.pattern.text(), "first_frame": self.first.value(), "last_frame": self.last.value(),
                      "stride": self.stride.value(), "io_threads": self.io_threads.value()},
            "channels": channels,
            "time": {"frames_per_hour": self.fph.value(), "events": events},
        }
