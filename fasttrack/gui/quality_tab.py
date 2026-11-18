"""Quality: find bad positions (fibers, debris, bright blobs, empty fields) before tracking.

"Check positions" reads a few frames of every position the plate selects and compares their
tile maps with a random sample. Flagged positions are skipped by tracking and figures unless
you tick "Track anyway". "Rescore" applies changed thresholds to the frames already read.
The flags, the reasons and your overrides are saved in the experiment file.
"""
from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ..experiment import QUALITY_DEFAULTS, condition_of_well, loaded_channels
from .worker import FunctionWorker

SETTINGS = [  # key, label, kind, range, step
    ("frames_per_position", "Frames read per position", int, (2, 50), 1),
    ("reference_positions", "Positions in the reference sample", int, (3, 2000), 1),
    ("tiles_per_edge", "Tiles per edge", int, (4, 64), 1),
    ("z_threshold", "Unusual tile: robust deviations", float, (1, 50), 0.5),
    ("area_threshold", "Bad frame: share of unusual tiles", float, (0.001, 1), 0.005),
    ("level_threshold", "Bad frame: background level deviations", float, (1, 100), 0.5),
    ("frame_fraction", "Flag: share of bad frames", float, (0.01, 1), 0.05),
    ("max_missing_frames", "Flag: more missing or unreadable frames than", int, (0, 100000), 1),
    ("min_cells_per_frame", "Warn if fewer cells per frame", int, (0, 100000), 1),
    ("max_nuclear_coverage", "Warn if nuclei cover more of the field", float, (0.01, 1), 0.05),
]
COLUMNS = ["Position", "Condition", "Status", "Track anyway", "Unusual tiles", "Level", "Bad frames", "Reason"]


def _pixmap(gray: np.ndarray, rgb: np.ndarray | None = None) -> QPixmap:
    data = np.ascontiguousarray(rgb if rgb is not None else np.repeat(gray[..., None], 3, axis=2))
    h, w = data.shape[:2]
    return QPixmap.fromImage(QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


class QualityTab(QWidget):
    changed = Signal()

    inspect_requested = Signal(str, int)   # position, frame

    def __init__(self, get_experiment, get_timestacks):
        super().__init__()
        self.viewers: list = []
        self.get_experiment = get_experiment
        self.get_timestacks = get_timestacks
        self.q = copy.deepcopy(QUALITY_DEFAULTS)
        self.results: dict = {}
        self.features: dict = {}
        self.worker = None
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        side = QWidget()
        side.setMaximumWidth(400)
        col = QVBoxLayout(side)
        box = QGroupBox("Settings")
        form = QFormLayout(box)
        self.enabled = QCheckBox("Skip flagged positions when tracking and drawing")
        form.addRow(self.enabled)
        self.fields = {}
        for key, label, kind, (lo, hi), step in SETTINGS:
            w = QSpinBox() if kind is int else QDoubleSpinBox()
            if kind is float:
                w.setDecimals(3)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            form.addRow(label, w)
            self.fields[key] = w
        col.addWidget(box)
        row = QHBoxLayout()
        self.check_button = QPushButton("Check positions")
        self.check_button.clicked.connect(self.check)
        self.rescore_button = QPushButton("Rescore")
        self.rescore_button.setToolTip("Apply changed thresholds to the frames already read.")
        self.rescore_button.clicked.connect(self.rescore)
        self.rescore_button.setEnabled(False)
        clear = QPushButton("Clear result")
        clear.clicked.connect(self.clear)
        for b in (self.check_button, self.rescore_button, clear):
            row.addWidget(b)
        col.addLayout(row)
        row = QHBoxLayout()
        self.open_button = QPushButton("Open images")
        self.open_button.setToolTip("Show the selected position's bad frames at full resolution (or double-click a row).")
        self.open_button.clicked.connect(self.open_images)
        self.inspect_button = QPushButton("Open in Test run")
        self.inspect_button.setToolTip("Run the pipeline around the first bad frame on the Test run page.")
        self.inspect_button.clicked.connect(self.open_in_inspect)
        row.addWidget(self.open_button)
        row.addWidget(self.inspect_button)
        col.addLayout(row)
        self.progress = QProgressBar()
        col.addWidget(self.progress)
        self.summary = QLabel("Not checked yet. Tracking runs the check first if it has not been run.")
        self.summary.setWordWrap(True)
        col.addWidget(self.summary)
        help_text = QLabel("Each tile's 1st percentile (background floor) and median are compared with the same tile in a random "
                           "sample of positions, after removing each frame's overall level, so expression and density differences "
                           "between conditions do not count. Bright medians are ignored: they are dense colonies.")
        help_text.setWordWrap(True)
        help_text.setProperty("role", "hint")
        col.addWidget(help_text)
        col.addStretch()
        split.addWidget(side)

        right = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self.show_selected)
        self.table.itemDoubleClicked.connect(lambda _: self.open_images())
        self.table.itemChanged.connect(self._keep_changed)
        right.addWidget(self.table)
        preview = QWidget()
        pl = QVBoxLayout(preview)
        self.preview_title = QLabel("Select a position to see its worst sampled frame and where the unusual tiles are.")
        pl.addWidget(self.preview_title)
        self.images = QHBoxLayout()
        pl.addLayout(self.images)
        pl.addStretch()
        right.addWidget(preview)
        right.setSizes([520, 300])
        split.addWidget(right)
        split.setSizes([380, 1000])

    # ------------------------------------------------------------------ settings
    def _read_settings(self) -> dict:
        q = copy.deepcopy(self.q)
        q["enabled"] = self.enabled.isChecked()
        for key, *_ in SETTINGS:
            q[key] = self.fields[key].value()
        return q

    def load(self, q: dict) -> None:
        self.q = copy.deepcopy(q)
        self.enabled.setChecked(q["enabled"])
        for key, *_ in SETTINGS:
            self.fields[key].setValue(q[key])
        self.results, self.features = {}, {}
        self.rescore_button.setEnabled(False)
        self._fill_from_flags()

    def collect(self) -> dict:
        self.q = self._read_settings()
        return copy.deepcopy(self.q)

    # ------------------------------------------------------------------ checking
    def check(self) -> None:
        exp = copy.deepcopy(self.get_experiment())
        exp["quality"] = self._read_settings()
        timestacks = self.get_timestacks()
        if not timestacks:
            self.summary.setText("Find the timestacks first (Scan on the Experiment page, or Find timestacks on the Run page).")
            return
        if not exp["paths"]["results"]:
            self.summary.setText("Set the results folder first: the report is written there.")
            return
        self.check_button.setEnabled(False)
        self.summary.setText("Reading frames…")

        def job(on_event, stop):
            from ..track.run import run_quality_check
            results, features = run_quality_check(exp, timestacks, on_event, stop)
            return exp["quality"], results, features

        self.worker = FunctionWorker(job, self)
        self.worker.event.connect(self._event)
        self.worker.finished_ok.connect(self._checked)
        self.worker.failed.connect(lambda m: self.summary.setText(m.splitlines()[0]))
        self.worker.finished.connect(lambda: self.check_button.setEnabled(True))
        self.worker.start()

    def _event(self, e: dict) -> None:
        if e["kind"] == "quality_progress":
            self.progress.setRange(0, e["total"])
            self.progress.setValue(e["done"])

    def _checked(self, out) -> None:
        q, results, features = out
        if not results:
            self.summary.setText("Not checked: the check compares positions with each other and needs at least 3.")
            return
        keep = set(self.q["keep"])
        self.q = q
        self.q["keep"] = [k for k in keep if k in q["flagged"]]
        self.results, self.features = results, features
        self.rescore_button.setEnabled(True)
        self._fill()
        self.changed.emit()

    def rescore(self) -> None:
        if not self.features:
            return
        from ..quality import score_positions
        q = self._read_settings()
        exp = self.get_experiment()
        names = [c["name"] for c in loaded_channels(exp)] or None
        import random
        ids = sorted(self.features)
        reference = sorted(random.Random(int(q["seed"])).sample(ids, min(int(q["reference_positions"]), len(ids))))
        self.results = score_positions(self.features, q, reference, names)
        from ..quality import file_flags
        q["flagged"] = {**{k: r["reason"] or "unusual tiles" for k, r in self.results.items() if r["flagged"]},
                        **file_flags({k: v for k, v in (self.get_timestacks() or {}).items() if k in self.results}, q)}
        for k, reason in q["flagged"].items():
            self.results[k] = {**self.results[k], "flagged": True, "reason": reason}
        q["keep"] = [k for k in q["keep"] if k in q["flagged"]]
        q["checked"] = True
        self.q = q
        self._fill()
        self.changed.emit()

    def apply_flags(self, flagged: dict) -> None:
        """A run did the check: take its flags."""
        self.q["flagged"] = dict(flagged)
        self.q["checked"] = True
        self.q["keep"] = [k for k in self.q["keep"] if k in flagged]
        self._fill_from_flags()

    def add_flags(self, new: dict) -> None:
        """A scan or run found positions with too many missing or unreadable files: add them, keep the rest."""
        self.q["flagged"].update(new)
        if self.results:
            for k, reason in new.items():
                self.results[k] = {**self.results.get(k, {"worst_area": float("nan"), "worst_level_z": float("nan"), "bad_frames": []}),
                                   "flagged": True, "reason": reason}
            self._fill()
        elif self.q["checked"]:
            self._fill_from_flags()
        self.changed.emit()

    def clear(self) -> None:
        self.q["flagged"], self.q["keep"], self.q["checked"] = {}, [], False
        self.results, self.features = {}, {}
        self.rescore_button.setEnabled(False)
        self._fill_from_flags()
        self.changed.emit()

    # ------------------------------------------------------------------ table
    def _fill_from_flags(self) -> None:
        if not self.q["checked"]:
            self.table.setRowCount(0)
            self.summary.setText("Not checked yet. Tracking runs the check first if it has not been run.")
            return
        self.results = {k: {"flagged": True, "reason": r, "worst_area": float("nan"), "worst_level_z": float("nan"), "bad_frames": []}
                        for k, r in self.q["flagged"].items()}
        self._fill()

    def _fill(self) -> None:
        exp = self.get_experiment()
        conditions = condition_of_well(exp)
        keep = set(self.q["keep"])
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.results))
        order = sorted(self.results, key=lambda k: (not self.results[k]["flagged"], -np.nan_to_num(self.results[k]["worst_area"]), k))
        for row, key in enumerate(order):
            r = self.results[key]
            if r["flagged"]:
                status, color = ("Kept" if key in keep else "Flagged"), QColor(255, 220, 220)
            elif r["bad_frames"]:
                status, color = "Some bad frames", QColor(255, 245, 210)
            else:
                status, color = "OK", None
            area = "" if np.isnan(r["worst_area"]) else f"{100 * r['worst_area']:.1f}%"
            level = "" if np.isnan(r["worst_level_z"]) else f"{r['worst_level_z']:.1f}"
            values = [key, conditions.get(key.split("_xy")[0], ""), status, "", area, level, ", ".join(map(str, r["bad_frames"])), r["reason"]]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                if color is not None:
                    item.setBackground(color)
                self.table.setItem(row, c, item)
            keep_item = self.table.item(row, 3)
            if r["flagged"]:
                keep_item.setFlags(keep_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                keep_item.setCheckState(Qt.CheckState.Checked if key in keep else Qt.CheckState.Unchecked)
        self.table.blockSignals(False)
        flagged = [k for k, r in self.results.items() if r["flagged"]]
        some = sum(1 for r in self.results.values() if not r["flagged"] and r["bad_frames"])
        total = f" of {len(self.results)} positions" if self.features else ""
        self.summary.setText(f"{len(flagged)}{total} flagged, {len(set(flagged) & keep)} kept anyway"
                             + (f"; {some} more have isolated bad frames." if some else ".")
                             + ("" if self.features else " Run the check again to see images and every position."))

    def _keep_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 3:
            return
        key = self.table.item(item.row(), 0).text()
        if key not in self.q["flagged"]:
            return
        keep = set(self.q["keep"])
        if item.checkState() == Qt.CheckState.Checked:
            keep.add(key)
        else:
            keep.discard(key)
        self.q["keep"] = sorted(keep)
        self.table.blockSignals(True)
        self.table.item(item.row(), 2).setText("Kept" if key in keep else "Flagged")
        self.table.blockSignals(False)
        self.changed.emit()

    def _selected_key(self) -> str | None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        return self.table.item(min(rows), 0).text() if rows else None

    def _bad_frames(self, key: str) -> tuple[list[int], dict]:
        """Bad frames (or all sampled ones) of a position, and the tile maps per frame when this session has them."""
        r = self.results.get(key, {})
        zmaps = {}
        if key in self.features and "frames" in r:
            for i, f in enumerate(r["frames"]):
                zmaps[f] = self._zmap_for(key, i)
        frames = list(r.get("bad_frames") or [])
        if not frames:  # flags loaded from the file: the report in the results folder has the frames
            import yaml
            from pathlib import Path
            report = Path(self.get_experiment()["paths"]["results"]) / "quality_report.yaml"
            if report.exists():
                entry = (yaml.safe_load(report.read_text()) or {}).get("positions", {}).get(key, {})
                frames = list(entry.get("bad_frames") or entry.get("sampled_frames") or [])
        if not frames and "frames" in r:
            frames = list(r["frames"])
        ts = (self.get_timestacks() or {}).get(key, {})  # frames with missing or unreadable files, around them too
        from ..io import layout as io_layout
        for f in io_layout.bad_frames(ts):
            frames += [g for g in (f - 1, f, f + 1) if 0 <= g < ts.get("frames", 0)]
        return sorted(set(frames)) or [0], zmaps

    def _zmap_for(self, key: str, i: int) -> np.ndarray:
        """Unusual-tile strength of sampled frame i (max over maps and channels), recomputed from the features."""
        from ..quality import score_positions
        import random
        q = self._read_settings()
        ids = sorted(self.features)
        if not hasattr(self, "_full_scores") or self._full_scores[0] != (tuple(ids), q["z_threshold"]):
            reference = sorted(random.Random(int(q["seed"])).sample(ids, min(int(q["reference_positions"]), len(ids))))
            self._full_scores = ((tuple(ids), q["z_threshold"]), score_positions(self.features, q, reference, None, keep_maps=True))
        return self._full_scores[1][key]["zmaps"][i]

    def open_images(self) -> None:
        key = self._selected_key()
        if key is None:
            return
        timestacks = self.get_timestacks() or {}
        if key not in timestacks:
            self.summary.setText(f"{key}: its images were not found; check the raw data folder on the Experiment page.")
            return
        frames, zmaps = self._bad_frames(key)
        names = [c["name"] for c in loaded_channels(self.get_experiment())]
        viewer = FrameViewer(key, timestacks[key], frames, names, zmaps, float(self.fields["z_threshold"].value()))
        reason = self.q["flagged"].get(key, "") or self.results.get(key, {}).get("reason", "")
        for i, name in enumerate(names):  # start on the channel the reason names
            if f" in {name} " in f"{reason} ":
                viewer.channel.setCurrentIndex(i)
                break
        self.viewers = [v for v in self.viewers if v.isVisible()] + [viewer]
        viewer.show()

    def open_in_inspect(self) -> None:
        key = self._selected_key()
        if key is not None:
            frames, _ = self._bad_frames(key)
            self.inspect_requested.emit(key, int(frames[0]))

    def show_selected(self) -> None:
        while self.images.count():
            w = self.images.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            return
        key = self.table.item(min(rows), 0).text()
        if key not in self.features or "zmap" not in self.results.get(key, {}):
            self.preview_title.setText(f"{key}: run the check to see its frames.")
            return
        r, f = self.results[key], self.features[key]
        w = r["worst_frame"]
        self.preview_title.setText(f"{key}, frame {r['frames'][w]} (the worst sampled frame). Right: unusual tiles, red = beyond the threshold.")
        names = [c["name"] for c in loaded_channels(self.get_experiment())]
        for c, thumb in enumerate(f["thumb"][w]):
            self._add_image(_pixmap(thumb), names[c] if c < len(names) else f"c{c + 1}")
        z = np.clip(r["zmap"] / float(self.fields["z_threshold"].value()), 0, 2) / 2
        k = z.shape[0]
        big = np.kron(z, np.ones((160 // k + 1, 160 // k + 1)))[:160, :160]
        rgb = (np.stack([big, big * 0.15, big * 0.15], -1) * 255).astype(np.uint8)
        rgb[np.kron(r["zmap"] > self.fields["z_threshold"].value(), np.ones((160 // k + 1, 160 // k + 1), bool))[:160, :160]] = (230, 40, 40)
        self._add_image(_pixmap(None, rgb), "unusual tiles")

    def _add_image(self, pixmap: QPixmap, caption: str) -> None:
        holder = QWidget()
        lay = QVBoxLayout(holder)
        image = QLabel()
        image.setPixmap(pixmap.scaled(220, 220, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        lay.addWidget(image)
        lay.addWidget(QLabel(caption))
        self.images.addWidget(holder)


class FrameViewer(QWidget):
    """A position's frames at full resolution, from the raw TIFFs: the bad frames first, one channel at a
    time, with the unusual tiles outlined when the check's maps are at hand."""

    def __init__(self, key: str, timestack: dict, frames: list[int], channel_names: list[str], zmaps: dict | None, z_threshold: float):
        super().__init__()
        from .image_view import ImageCanvas
        self.setWindowTitle(f"{key}: frames flagged by the quality check")
        self.key, self.timestack, self.zmaps, self.z = key, timestack, zmaps or {}, z_threshold
        self.timestack, self.zmaps, self.z = timestack, zmaps or {}, z_threshold
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        from PySide6.QtWidgets import QComboBox
        self.frame = QComboBox()
        for f in frames:
            self.frame.addItem(f"frame {f}" + ("  (unusual tiles outlined)" if f in self.zmaps else ""), f)
        self.channel = QComboBox()
        self.channel.addItems(channel_names or [f"c{c}" for c in timestack["channels"]])
        self.tiles = QCheckBox("Outline unusual tiles")
        self.tiles.setChecked(True)
        for label, w in (("Frame", self.frame), ("Channel", self.channel)):
            bar.addWidget(QLabel(label))
            bar.addWidget(w)
        bar.addWidget(self.tiles)
        bar.addStretch()
        lay.addLayout(bar)
        self.canvas = ImageCanvas()
        lay.addWidget(self.canvas, 1)
        self.note = QLabel("Scroll to zoom, drag to pan. Contrast: 1st to 99.8th percentile of the frame.")
        lay.addWidget(self.note)
        for w in (self.frame, self.channel):
            w.currentIndexChanged.connect(self.draw)
        self.tiles.toggled.connect(self.draw)
        self.draw()

    def draw(self) -> None:
        from ..io import layout as io_layout
        from .image_view import to_display
        f = self.frame.currentData()
        if f is None:
            return
        files = list(self.timestack["channels"].values())[self.channel.currentIndex()]
        if f >= len(files):
            return
        try:
            image = io_layout.read_frame(files[f]).astype(np.float32)
        except io_layout.UnreadableFrame as exc:
            self.setWindowTitle(f"{self.key}: frame {f} has no readable file ({exc})")
            image = np.zeros(io_layout.frame_shape(self.timestack), np.float32)
        low, high = np.percentile(image, [1, 99.8])
        overlay = None
        if self.tiles.isChecked() and f in self.zmaps:
            z = self.zmaps[f] > self.z
            k = z.shape[0]
            tile = np.kron(z, np.ones((image.shape[0] // k + 1, image.shape[1] // k + 1), bool))[:image.shape[0], :image.shape[1]]
            edge = np.zeros_like(tile)  # tile borders only, so the image stays visible
            edge[:-1] |= tile[:-1] != tile[1:]
            edge[:, :-1] |= tile[:, :-1] != tile[:, 1:]
            overlay = edge
        self.canvas.set_image(to_display(image, low, high), overlay)
