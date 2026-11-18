"""Test run: run the full pipeline on one position (part of it, or the whole movie) and look at every step.

Pick a position (or press Random batch), then Run. Nothing is written to disk: every intermediate is
kept in memory until the next run. Views share zoom, pan and time; outlines of the final nuclei can be
drawn over the images. Keys: left/right step one frame, up/down ten. Independent of the Run page.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
                               QSplitter, QVBoxLayout, QWidget)

from ..config import merge
from ..io import layout as io_layout
from ..io import store
from ..track.measure import S
from .image_view import TRACK_COLORS, ImageCanvas, outline_mask, to_display
from .worker import FunctionWorker

LABEL_LAYERS = {"nuclear_labels", "label_map", "border_labels"}
BINARY_LAYERS = {"rough_mask", "binary_mask"}
HELP = {
    "raw": "The image as read from the TIFF.",
    "corrected": "After background and illumination correction (quadratic fit). Nuclei should sit on an even, dark background.",
    "rough_mask": "First threshold from tile statistics.", "binary_mask": "Final foreground after the LoG cleanup.",
    "distance": "Pseudo distance from the edge of the mask; peaks are nucleus centres.",
    "centroids": "Seeds: one per nucleus. Two seeds in one nucleus split it; one seed in two nuclei merges them.",
    "nuclear_labels": "Nuclei grown from the seeds.", "label_map": "Nuclei and their cytoplasm ring (dim), as measured.",
    "border_labels": "Cells touching the image border (left out of analysis).",
    "underseg_score": "How likely a nucleus is two merged nuclei (0-100).",
}
TITLES = {
    "raw": "Raw", "corrected": "Background corrected", "rough_mask": "Rough mask", "binary_mask": "Binary mask",
    "distance": "Pseudo distance", "centroids": "Centroids", "nuclear_labels": "Nuclear labels",
    "label_map": "Labels and cytoplasm ring", "border_labels": "Border labels", "underseg_score": "Undersegmentation score",
}
DEFAULT_VIEWS = ["corrected", "centroids", "label_map"]
MAX_VIEWS = 4


def colour_labels(labels: np.ndarray) -> np.ndarray:
    """Int label map to RGB. A label's colour is fixed; negative labels (the ring) are dimmer."""
    magnitude = np.abs(labels).astype(np.uint32)
    hashed = (magnitude * 2654435761) & 0xFFFFFFFF
    rgb = np.stack([(hashed >> 8) & 255, (hashed >> 16) & 255, (hashed >> 24) & 255], axis=-1).astype(np.uint8) // 2 + 90
    rgb[labels == 0] = 0
    rgb[labels < 0] = rgb[labels < 0] // 3
    return rgb.astype(np.uint8)


def heat(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Dark blue to yellow, for scores."""
    v = np.clip((values.astype(np.float32) - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.stack([v, v**2 * 0.9, (1 - v) * 0.5 * (values > 0)], axis=-1)
    rgb[values == 0] = 0
    return (rgb * 255).astype(np.uint8)


class StackView(QFrame):
    """One panel: choose a layer and channel, and see it at the shared time and zoom."""

    def __init__(self, tab: "InspectTab", layer: str | None):
        super().__init__()
        self.tab = tab
        self.setFrameShape(QFrame.Shape.StyledPanel)
        col = QVBoxLayout(self)
        col.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.layer = QComboBox()
        self.channel = QComboBox()
        self.low = QDoubleSpinBox()
        self.high = QDoubleSpinBox()
        for spin in (self.low, self.high):
            spin.setRange(-1e5, 1e6)
            spin.setDecimals(0)
            spin.setMaximumWidth(72)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self.draw)
        self.auto = QPushButton("Auto")
        self.auto.setMaximumWidth(50)
        self.auto.clicked.connect(self.auto_contrast)
        row.addWidget(self.layer, 1)
        for w in (self.channel, self.low, self.high, self.auto):
            row.addWidget(w)
        col.addLayout(row)
        self.canvas = ImageCanvas()
        col.addWidget(self.canvas, 1)
        self.value = QLabel(" ")
        col.addWidget(self.value)
        self.preferred = layer
        self.layer.currentIndexChanged.connect(self.layer_changed)
        self.layer.currentIndexChanged.connect(lambda _: self.layer.setToolTip(HELP.get(self.layer.currentData(), "")))
        self.channel.currentIndexChanged.connect(self.auto_contrast)

    def fill_layers(self, names: list[str]) -> None:
        wanted = self.layer.currentData() or self.preferred
        self.layer.blockSignals(True)
        self.layer.clear()
        for name in names:
            self.layer.addItem(TITLES.get(name, name), name)
        index = self.layer.findData(wanted)
        self.layer.setCurrentIndex(max(index, 0))
        self.layer.blockSignals(False)
        self.layer_changed()

    def dataset(self):
        cache = self.tab.cache
        name = self.layer.currentData()
        return cache["layers"].get(name) if cache is not None and name else None

    def layer_changed(self) -> None:
        ds = self.dataset()
        if ds is None:
            return
        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItems([f"c{i + 1}" for i in range(ds.shape[1])])
        self.channel.blockSignals(False)
        self.channel.setVisible(ds.shape[1] > 1)
        continuous = self.layer.currentData() not in LABEL_LAYERS | BINARY_LAYERS | {"centroids"}
        for w in (self.low, self.high, self.auto):
            w.setVisible(continuous)
        self.auto_contrast()

    def frame(self) -> np.ndarray | None:
        ds = self.dataset()
        i = self.tab.index()
        if ds is None or i >= ds.shape[0]:
            return None
        return ds[i, max(self.channel.currentIndex(), 0)]

    def auto_contrast(self) -> None:
        frame = self.frame()
        if frame is None:
            return
        low, high = np.percentile(frame, [1, 99.9])
        if self.layer.currentData() == "underseg_score":
            low, high = 0.0, 100.0
        if high <= low:
            high = low + 1
        for spin, value in ((self.low, low), (self.high, high)):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self.draw()

    def draw(self) -> None:
        frame = self.frame()
        if frame is None:
            return
        name = self.layer.currentData()
        if name in LABEL_LAYERS:
            self.canvas.set_rgb(colour_labels(frame))
        elif name in BINARY_LAYERS:
            self.canvas.set_image((frame > 0).astype(np.uint8) * 255)
        elif name == "centroids":
            # A seed is a single pixel, so widen it to stay visible when zoomed out.
            wide = frame > 0
            for axis in (0, 1):
                wide = wide | np.roll(wide, 1, axis) | np.roll(wide, -1, axis) | np.roll(wide, 2, axis) | np.roll(wide, -2, axis)
            self.canvas.set_image(wide.astype(np.uint8) * 255)
        elif name == "underseg_score":
            self.canvas.set_rgb(heat(frame, self.low.value(), self.high.value()))
        else:
            self.canvas.set_image(to_display(frame, self.low.value(), self.high.value()), self.tab.outlines())
        self.canvas.set_paths(self.tab.overlay_paths())

    def show_value(self, x: int, y: int) -> None:
        frame = self.frame()
        if frame is not None and 0 <= x < frame.shape[0] and 0 <= y < frame.shape[1]:
            self.value.setText(f"{self.layer.currentText()}: {frame[x, y]}")


class InspectTab(QWidget):
    def __init__(self, cfg: dict, collect_cfg, get_experiment=None, ensure_timestacks=None):
        super().__init__()
        self.cfg = cfg
        self.collect_cfg = collect_cfg
        self.get_experiment = get_experiment
        self.ensure_timestacks = ensure_timestacks
        self.timestacks: dict = {}
        self.cache: dict | None = None
        self.tables: dict | None = None
        self.first_frame = 0
        self.worker = None
        self.views: list[StackView] = []
        self._syncing = False

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.condition = QComboBox()
        self.condition.setMinimumWidth(120)
        self.condition.currentIndexChanged.connect(self._fill_positions)
        self.stack = QComboBox()
        self.stack.setMinimumWidth(110)
        self.start = QSpinBox()
        self.start.setRange(0, 10**6)
        self.count = QSpinBox()
        self.count.setRange(2, 2000)
        self.count.setValue(100)
        self.whole = QCheckBox("Whole movie")
        self.whole.setToolTip("Run every frame of the position. Every intermediate is kept in RAM (compressed).")
        self.whole.toggled.connect(lambda on: (self.start.setEnabled(not on), self.count.setEnabled(not on), self._estimate()))
        self.count.valueChanged.connect(lambda _: self._estimate())
        self.stack.currentIndexChanged.connect(lambda _: self._estimate())
        self.random_button = QPushButton("Random batch")
        self.random_button.setToolTip("A random position the plate tracks, and a random start frame.")
        self.random_button.clicked.connect(self.random_batch)
        self.run_button = QPushButton("Run")
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.run_test)
        self.add_button = QPushButton("Add view")
        self.add_button.clicked.connect(lambda: self.add_view(None))
        self.remove_button = QPushButton("Remove view")
        self.remove_button.clicked.connect(self.remove_view)
        for label, w in (("Condition", self.condition), ("Position", self.stack), ("From frame", self.start), ("Frames", self.count)):
            bar.addWidget(QLabel(label))
            bar.addWidget(w)
        bar.addWidget(self.whole)
        bar.addWidget(self.random_button)
        bar.addWidget(self.run_button)
        self.info = QLabel("Pick a position or press Random batch, then Run. The full pipeline runs with the current settings "
                           "(the ones Run will use) and every step stays in memory to inspect. Nothing is saved.")
        self.info.setWordWrap(True)
        bar.addWidget(self.info, 1)
        bar.addWidget(self.add_button)
        bar.addWidget(self.remove_button)
        root.addLayout(bar)

        self.split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.split, 1)

        nav = QHBoxLayout()
        self.back = QPushButton("◀")
        self.back.setMaximumWidth(40)
        self.back.clicked.connect(lambda: self.step(-1))
        self.forward = QPushButton("▶")
        self.forward.setMaximumWidth(40)
        self.forward.clicked.connect(lambda: self.step(1))
        self.time = QSlider(Qt.Orientation.Horizontal)
        self.time.valueChanged.connect(self.time_changed)
        self.time_label = QLabel("Frame -")
        self.time_label.setMinimumWidth(110)
        self.jump_box = QSpinBox()
        self.jump_box.setRange(0, 10**6)
        self.jump_box.setKeyboardTracking(False)
        self.jump_button = QPushButton("Jump")
        self.jump_button.clicked.connect(self.jump)
        self.jump_box.editingFinished.connect(self.jump)
        self.show_outlines = QCheckBox("Outlines")
        self.show_outlines.setChecked(True)
        self.show_outlines.setToolTip("Draw the final nuclei over the image views.")
        self.show_outlines.toggled.connect(self.draw_all)
        self.show_seeds = QCheckBox("Seeds")
        self.show_tracks = QCheckBox("Track tails")
        self.show_seeds.toggled.connect(self.draw_all)
        self.show_tracks.toggled.connect(self.draw_all)
        for w in (self.back, self.forward, self.time_label):
            nav.addWidget(w)
        nav.addWidget(self.time, 1)
        for w in (self.jump_box, self.jump_button, self.show_outlines, self.show_seeds, self.show_tracks):
            nav.addWidget(w)
        root.addLayout(nav)
        self.position = QLabel(" ")
        root.addWidget(self.position)

        for key, delta in ((Qt.Key.Key_Left, -1), (Qt.Key.Key_Right, 1), (Qt.Key.Key_Down, -10), (Qt.Key.Key_Up, 10)):
            shortcut = QShortcut(QKeySequence(key), self)  # active whenever this tab is shown
            shortcut.activated.connect(lambda d=delta: self._typing() or self.step(d))

        for name in DEFAULT_VIEWS:
            self.add_view(name)
        self.update_buttons()

    def _typing(self) -> bool:
        from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QLineEdit
        focus = QApplication.focusWidget()
        return isinstance(focus, (QLineEdit, QAbstractSpinBox, QComboBox))

    def outlines(self) -> np.ndarray | None:
        """Boundaries of the final nuclei at the current frame, for drawing over image views."""
        if not self.show_outlines.isChecked() or self.cache is None:
            return None
        ds = self.cache["layers"].get("label_map")
        if ds is None or self.index() >= ds.shape[0]:
            return None
        return outline_mask(ds[self.index(), 0])

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self.timestacks and self.ensure_timestacks is not None:
            self.ensure_timestacks()

    # ------------------------------------------------------------------ views
    def add_view(self, layer: str | None) -> None:
        if len(self.views) >= MAX_VIEWS:
            return
        view = StackView(self, layer or "raw")
        self.views.append(view)
        self.split.addWidget(view)
        view.canvas.horizontalScrollBar().valueChanged.connect(lambda _, v=view: self.sync_from(v))
        view.canvas.verticalScrollBar().valueChanged.connect(lambda _, v=view: self.sync_from(v))
        view.canvas.zoomed.connect(lambda v=view: self.sync_from(v))
        view.canvas.pixel_hovered.connect(self.on_hover)
        if self.cache is not None:
            view.fill_layers(list(self.cache["layers"]))
        self.update_buttons()

    def remove_view(self) -> None:
        if len(self.views) > 1:
            view = self.views.pop()
            view.setParent(None)
            view.deleteLater()
        self.update_buttons()

    def update_buttons(self) -> None:
        self.add_button.setEnabled(len(self.views) < MAX_VIEWS)
        self.remove_button.setEnabled(len(self.views) > 1)

    def sync_from(self, source: StackView) -> None:
        if self._syncing:
            return
        self._syncing = True
        transform = source.canvas.transform()
        for view in self.views:
            if view is not source:
                view.canvas.user_zoomed()
                view.canvas.setTransform(transform)
                view.canvas.horizontalScrollBar().setValue(source.canvas.horizontalScrollBar().value())
                view.canvas.verticalScrollBar().setValue(source.canvas.verticalScrollBar().value())
        self._syncing = False

    # ------------------------------------------------------------------ time
    def index(self) -> int:
        return self.time.value()

    def step(self, delta: int) -> None:
        self.time.setValue(self.time.value() + delta)

    def jump(self) -> None:
        self.time.setValue(self.jump_box.value() - self.first_frame)

    def time_changed(self) -> None:
        self.time_label.setText(f"Frame {self.first_frame + self.time.value()}")
        self.draw_all()

    def draw_all(self) -> None:
        for view in self.views:
            view.draw()

    def on_hover(self, x: float, y: float) -> None:
        ix, iy = int(x), int(y)
        self.position.setText(f"x {ix}   y {iy}")
        for view in self.views:
            view.show_value(ix, iy)

    def overlay_paths(self) -> list:
        tables = self.tables
        if tables is None:
            return []
        groups = []
        t = self.index()
        sc = tables["scalars"]
        x, y = sc[:, S["x"]], sc[:, S["y"]]
        index = tables["index"]
        if self.show_tracks.isChecked():
            first = max(0, t - 20)
            for track in np.nonzero(index[:, t] >= 0)[0]:
                rows = index[track, first:t + 1]
                rows = rows[rows >= 0]
                if rows.size > 1:
                    groups.append(([(float(x[i]), float(y[i])) for i in rows], TRACK_COLORS[track % len(TRACK_COLORS)], 1.5))
        if self.show_seeds.isChecked():
            for track in np.nonzero(index[:, t] >= 0)[0]:
                i = index[track, t]
                cx, cy = float(x[i]), float(y[i])
                groups.append(([(cx - 3, cy), (cx + 3, cy), (cx, cy), (cx, cy - 3), (cx, cy + 3)], "#ffffff", 1.0))
        return groups

    # ------------------------------------------------------------------ test mode
    def set_timestacks(self, timestacks: dict) -> None:
        self.timestacks = timestacks
        exp = self.get_experiment() if self.get_experiment else None
        current = self.condition.currentText()
        self.condition.blockSignals(True)
        self.condition.clear()
        self.condition.addItem("All positions")
        if exp:
            self.condition.addItems([c["name"] for c in exp["plate"]["conditions"] if c["wells"]])
        self.condition.setCurrentText(current or "All positions")
        self.condition.blockSignals(False)
        self._fill_positions()

    def _positions(self) -> list[str]:
        """Positions the plate tracks (all found if nothing is assigned), filtered by the chosen condition."""
        from ..experiment import condition_of_well, select_timestacks
        exp = self.get_experiment() if self.get_experiment else None
        keys = list(self.timestacks)
        if exp:
            keys = select_timestacks(exp, keys) or keys
            if self.condition.currentIndex() > 0:
                of = condition_of_well(exp)
                keys = [k for k in keys if of.get(k.split("_xy")[0]) == self.condition.currentText()]
        return keys

    def _fill_positions(self) -> None:
        current = self.stack.currentText()
        self.stack.clear()
        self.stack.addItems(self._positions())
        if current:
            self.stack.setCurrentText(current)

    def show_position(self, key: str, frame: int) -> None:
        """Called from the Quality page: put a position and a window around `frame` in place, and run."""
        self.condition.setCurrentIndex(0)
        if self.stack.findText(key) < 0:
            self.stack.addItem(key)
        self.stack.setCurrentText(key)
        self.count.setValue(min(self.count.value(), 40))
        self.start.setValue(max(0, frame - self.count.value() // 2))
        self.run_test()

    def _estimate(self) -> None:
        """RAM a test run will hold, before compression, so a whole-movie run is a choice, not a surprise."""
        key = self.stack.currentText()
        ts = self.timestacks.get(key)
        if ts is None:
            return
        try:
            X, Y = io_layout.frame_shape(ts)
        except OSError:
            return
        frames = ts["frames"] if self.whole.isChecked() else min(self.count.value(), ts["frames"])
        C = len(ts["channels"])
        gb = frames * X * Y * (4 * C + 18) / 1e9   # raw + corrected per channel, masks, distance, labels
        self.whole.setText(f"Whole movie ({ts['frames']} frames)")
        self.info.setText(f"{key}: {frames} frames, up to {gb:.1f} GB of intermediates in RAM (less after compression).")

    def random_batch(self) -> None:
        import random
        keys = self._positions()
        if not keys:
            self.info.setText("No positions found yet: check the raw data folder and filename pattern on the Experiment page.")
            return
        key = random.choice(keys)
        frames = self.timestacks[key]["frames"]
        self.stack.setCurrentText(key)
        self.start.setValue(random.randrange(max(1, frames - self.count.value() + 1)))
        self.info.setText(f"Random batch: {key}, frames {self.start.value()}-{self.start.value() + min(self.count.value(), frames) - 1}. Press Run.")

    def run_test(self) -> None:
        key = self.stack.currentText()
        if key not in self.timestacks:
            self.info.setText("No position chosen. Positions are found from the Experiment page's folder and pattern.")
            return
        ts = self.timestacks[key]
        start = 0 if self.whole.isChecked() else min(self.start.value(), ts["frames"] - 2)
        count = ts["frames"] if self.whole.isChecked() else min(self.count.value(), ts["frames"] - start)
        cfg = merge(self.collect_cfg(), {"output": {"save_measurements": False, "return_tables": True}})
        previous = self.cache
        self.cache = None
        self.tables = None
        store.close_gui_cache(previous)  # overwritten each run: only the latest stays in RAM

        def job(on_event, stop):
            from ..track.run import process_timestack
            cache = store.new_gui_cache(key)
            pool = ThreadPoolExecutor(int(cfg["data"]["io_threads"]))
            part = {**ts, "frames": count}
            try:
                summary = process_timestack(part, cfg, on_event, stop, cache,
                                            reader=lambda a, b: io_layout.load_batch(ts, start + a, start + b, pool))
                store.flush_cache(cache)
            finally:
                pool.shutdown(wait=False)
            return {"cache": cache, "summary": summary, "start": start}

        self.run_button.setEnabled(False)
        self.info.setText(f"Running {key}, frames {start}–{start + count - 1}…")
        self.worker = FunctionWorker(job, self)
        self.worker.event.connect(self.on_event)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(lambda message: self.info.setText("Test mode failed: " + message.splitlines()[0]))
        self.worker.finished.connect(lambda: self.run_button.setEnabled(True))
        self.worker.start()

    def on_event(self, e: dict) -> None:
        if e["kind"] == "batch":
            self.info.setText(f"{e['timestack']}: {e['frames_done']} of {e['frames_total']} frames…")

    def on_done(self, out: dict) -> None:
        summary = out["summary"]
        if summary is None:
            store.close_gui_cache(out["cache"])
            self.info.setText("Stopped.")
            return
        self.cache = out["cache"]
        self.tables = summary["tables"]
        self.first_frame = out["start"]
        names = [n for n in store.ALL_LAYERS if n in self.cache["layers"]]
        for view in self.views:
            view.fill_layers(names)
        size = store.cache_size_bytes(self.cache) / 1e9
        frames = summary["frames"]
        self.time.setRange(0, frames - 1)
        self.jump_box.setRange(self.first_frame, self.first_frame + frames - 1)
        self.time.setValue(0)
        self.time_changed()
        self.info.setText(f"{summary['id']}: {frames} frames, {summary['tracks']} tracks, {summary['divisions']} divisions, "
                          f"{size:.1f} GB in RAM, {summary['seconds']:.1f} s")
