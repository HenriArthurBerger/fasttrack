"""Viewer: spot-check one timestack. Raw frames come from the original TIFFs on demand,
tracks from saved results, and masks and labels from the in-memory cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ..io import layout as io_layout
from ..io import store
from ..track.measure import S
from .image_view import TRACK_COLORS, ImageCanvas, outline_mask, to_display
from .worker import FunctionWorker

TRACE_METRICS = ["nuc_median", "cyto_median", "nc_ratio", "nuc_sum", "nuc_area", "eccentricity"]


class ViewerTab(QWidget):
    def __init__(self, cfg: dict, collect_cfg):
        super().__init__()
        self.cfg = cfg
        self.collect_cfg = collect_cfg
        self.timestacks: dict = {}
        self.results: dict | None = None
        self.cache: dict | None = None
        self.preview: dict | None = None
        self.selected_track: int | None = None
        self.worker = None
        self._frame = None
        self._contrast_key = None

        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.canvas = ImageCanvas()
        self.canvas.pixel_clicked.connect(self.on_click)
        self.canvas.pixel_hovered.connect(self.on_hover)
        left_layout.addWidget(self.canvas, 1)
        slider_row = QHBoxLayout()
        self.time_label = QLabel("t = 0")
        self.time = QSlider(Qt.Orientation.Horizontal)
        self.time.valueChanged.connect(self.refresh)
        slider_row.addWidget(self.time_label)
        slider_row.addWidget(self.time, 1)
        left_layout.addLayout(slider_row)
        self.status = QLabel("")
        left_layout.addWidget(self.status)
        split.addWidget(left)

        side = QWidget()
        side.setMinimumWidth(360)
        side_layout = QVBoxLayout(side)
        box = QGroupBox("Display")
        form = QFormLayout(box)
        self.stack = QComboBox()
        self.stack.currentTextChanged.connect(self.load_timestack)
        form.addRow("Timestack", self.stack)
        self.channel = QComboBox()
        self.channel.currentIndexChanged.connect(self._new_contrast)
        form.addRow("Channel", self.channel)
        self.low = QDoubleSpinBox()
        self.high = QDoubleSpinBox()
        for spin in (self.low, self.high):
            spin.setRange(-1e5, 1e6)
            spin.setDecimals(0)
            spin.valueChanged.connect(self.refresh)
        contrast = QHBoxLayout()
        contrast.addWidget(self.low)
        contrast.addWidget(self.high)
        auto = QPushButton("Auto")
        auto.clicked.connect(self._new_contrast)
        contrast.addWidget(auto)
        form.addRow("Low / high", contrast)
        self.show_tracks = QCheckBox("Track tails")
        self.show_tracks.setChecked(True)
        self.show_tracks.toggled.connect(self.refresh)
        self.show_outlines = QCheckBox("Nuclear outlines (needs preview or latest run)")
        self.show_outlines.setChecked(True)
        self.show_outlines.toggled.connect(self.refresh)
        self.tail = QSpinBox()
        self.tail.setRange(1, 500)
        self.tail.setValue(20)
        self.tail.valueChanged.connect(self.refresh)
        form.addRow(self.show_tracks)
        form.addRow("Tail frames", self.tail)
        form.addRow(self.show_outlines)
        self.preview_button = QPushButton("Preview segmentation here (current parameters)")
        self.preview_button.clicked.connect(self.run_preview)
        form.addRow(self.preview_button)
        side_layout.addWidget(box)

        trace_box = QGroupBox("Selected cell")
        trace_layout = QVBoxLayout(trace_box)
        self.metric = QComboBox()
        self.metric.addItems(TRACE_METRICS)
        self.metric.currentTextChanged.connect(self.draw_trace)
        trace_layout.addWidget(self.metric)
        self.cell_info = QLabel("Click a nucleus to see its timecourse.")
        self.cell_info.setWordWrap(True)
        trace_layout.addWidget(self.cell_info)
        self.figure = Figure(figsize=(4, 3), layout="constrained")
        self.trace_canvas = FigureCanvasQTAgg(self.figure)
        trace_layout.addWidget(self.trace_canvas, 1)
        side_layout.addWidget(trace_box, 1)
        split.addWidget(side)
        split.setSizes([900, 380])

    def set_timestacks(self, timestacks: dict) -> None:
        self.timestacks = timestacks
        current = self.stack.currentText()
        self.stack.blockSignals(True)
        self.stack.clear()
        self.stack.addItems(list(timestacks))
        self.stack.blockSignals(False)
        if current in timestacks:
            self.stack.setCurrentText(current)
        if timestacks:
            self.load_timestack(self.stack.currentText())

    def set_cache(self, cache: dict) -> None:
        self.cache = cache

    def results_updated(self) -> None:
        if self.stack.currentText():
            self._load_results(self.stack.currentText())
            self.refresh()

    def _load_results(self, key: str) -> None:
        output = self.cfg["data"]["output"]
        path = Path(output) / "Traces" / f"{key}.h5" if output else None
        self.results = store.load_results(output, key) if path is not None and path.exists() else None

    def load_timestack(self, key: str) -> None:
        if key not in self.timestacks:
            return
        ts = self.timestacks[key]
        self.selected_track = None
        self.preview = None
        self._load_results(key)
        self.time.setRange(0, ts["frames"] - 1)
        names = self.cfg["data"].get("channel_names") or []
        self.channel.blockSignals(True)
        self.channel.clear()
        for i, chan in enumerate(ts["channels"]):
            self.channel.addItem(names[i] if i < len(names) else f"c{chan}")
        self.channel.blockSignals(False)
        self.draw_trace()
        self._new_contrast()

    def _read_frame(self) -> np.ndarray | None:
        key = self.stack.currentText()
        if key not in self.timestacks:
            return None
        files = list(self.timestacks[key]["channels"].values())[max(self.channel.currentIndex(), 0)]
        try:
            return io_layout.read_frame(files[self.time.value()])
        except io_layout.UnreadableFrame:  # missing or corrupt file: show a blank frame
            return np.zeros(io_layout.frame_shape(self.timestacks[key]), np.uint16)

    def _new_contrast(self) -> None:
        frame = self._read_frame()
        if frame is None:
            return
        low, high = np.percentile(frame, [1, 99.9])
        for spin, value in ((self.low, low), (self.high, high)):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self.refresh()

    def _labels_at(self, t: int) -> np.ndarray | None:
        if self.preview is not None and self.preview["start"] <= t < self.preview["stop"]:
            return self.preview["signed"][t - self.preview["start"], 0]
        cache = self.cache
        if cache is not None and cache["id"] == self.stack.currentText() and "label_map" in cache["layers"]:
            ds = cache["layers"]["label_map"]
            if t < ds.shape[0]:
                return ds[t, 0]
        return None

    def refresh(self) -> None:
        frame = self._read_frame()
        if frame is None:
            return
        t = self.time.value()
        self.time_label.setText(f"t = {t}")
        gray = to_display(frame, self.low.value(), self.high.value())
        overlay = None
        if self.show_outlines.isChecked():
            labels = self._labels_at(t)
            if labels is not None:
                overlay = outline_mask(labels)
        self.canvas.set_image(gray, overlay)
        self._draw_tracks(t)

    def _draw_tracks(self, t: int) -> None:
        groups = []
        r = self.results
        if r is not None and self.show_tracks.isChecked():
            index = r["trace_index"]
            x, y = r["scalars"][:, S["x"]], r["scalars"][:, S["y"]]
            start = max(0, t - self.tail.value())
            for track in np.nonzero(index[:, t] >= 0)[0]:
                rows = index[track, start:t + 1]
                rows = rows[rows >= 0]
                selected = track == self.selected_track
                colour = "#ffffff" if selected else TRACK_COLORS[track % len(TRACK_COLORS)]
                groups.append(([(float(x[i]), float(y[i])) for i in rows], colour, 3.0 if selected else 1.5))
        self.canvas.set_paths(groups)

    def on_hover(self, x: float, y: float) -> None:
        frame = self._read_frame()
        if frame is not None and 0 <= int(x) < frame.shape[0] and 0 <= int(y) < frame.shape[1]:
            self.status.setText(f"x {int(x)}   y {int(y)}   value {frame[int(x), int(y)]}")

    def on_click(self, x: float, y: float) -> None:
        r = self.results
        if r is None:
            self.cell_info.setText("This timestack has no results yet. Run it first.")
            return
        rows = r["trace_index"][:, self.time.value()]
        present = np.nonzero(rows >= 0)[0]
        if present.size == 0:
            return
        sc = r["scalars"][rows[present]]
        distance = np.hypot(sc[:, S["x"]] - x, sc[:, S["y"]] - y)
        nearest = int(distance.argmin())
        if distance[nearest] > 2.5 * max(r["meta"].get("radius", 8), 4):
            self.selected_track = None
        else:
            self.selected_track = int(present[nearest])
        self.draw_trace()
        self.refresh()

    def draw_trace(self) -> None:
        self.figure.clear()
        r = self.results
        track = self.selected_track
        if r is None or track is None:
            self.trace_canvas.draw_idle()
            if r is not None:
                self.cell_info.setText(f"{r['trace_index'].shape[0]} tracks. Click a nucleus to see its timecourse.")
            return
        ax = self.figure.add_subplot(111)
        metric = self.metric.currentText()
        frames = np.arange(r["trace_index"].shape[1])
        if metric in r["channel_names"]:
            for c in range(r["channels"].shape[1]):
                ax.plot(frames, store.metric_matrix(r, metric, c)[track], lw=1.2, label=f"c{c + 1}")
            ax.legend(frameon=False, fontsize=8)
            if metric in ("nuc_median", "nuc_sum"):
                ax.set_yscale("symlog", linthresh=10)
        else:
            ax.plot(frames, store.metric_matrix(r, metric)[track], lw=1.2)
        ax.axvline(self.time.value(), color="0.6", lw=0.8)
        ax.set_xlabel("frame")
        ax.set_ylabel(metric)
        ax.spines[["top", "right"]].set_visible(False)
        self.trace_canvas.draw_idle()
        mother = int(r["mother_track"][track])
        kids = np.nonzero(r["mother_track"] == track)[0].tolist()
        span = np.nonzero(r["trace_index"][track] >= 0)[0]
        self.cell_info.setText(f"Track {track}: frames {span[0]}–{span[-1]}. "
                               f"Mother: {mother if mother >= 0 else 'none'}. Daughters: {kids or 'none'}.")

    def run_preview(self) -> None:
        """Segment 20 frames around the slider with current parameters. Writes nothing to disk."""
        key = self.stack.currentText()
        if key not in self.timestacks:
            return
        ts = self.timestacks[key]
        cfg = self.collect_cfg()
        start = max(0, min(self.time.value() - 10, ts["frames"] - 20))
        stop = min(start + 20, ts["frames"])

        def job(on_event, stop_flag):
            from concurrent.futures import ThreadPoolExecutor

            import torch

            from ..track.batch import process_batch
            with ThreadPoolExecutor(int(cfg["data"]["io_threads"])) as pool:
                raw = io_layout.load_batch(ts, start, stop, pool)
            result = process_batch(torch.from_numpy(raw).to(cfg["device"]), cfg)
            out = {"start": start, "stop": stop, "signed": result["signed"].cpu().numpy(), "cells": result["stats"]["cells"],
                   "radius": result["radius"], "seconds": sum(result["timings"].values())}
            del result
            return out

        self.preview_button.setEnabled(False)
        self.preview_button.setText("Segmenting…")
        self.worker = FunctionWorker(job, self)
        self.worker.finished_ok.connect(self._preview_done)
        self.worker.failed.connect(self._preview_failed)
        self.worker.finished.connect(self._preview_reset)
        self.worker.start()

    def _preview_done(self, out: dict) -> None:
        self.preview = out
        self.show_outlines.setChecked(True)
        self.status.setText(f"Preview frames {out['start']}–{out['stop'] - 1}: {out['cells'] / (out['stop'] - out['start']):.1f} "
                            f"cells per frame, radius {out['radius']:.1f} px, {out['seconds']:.2f} s")
        self.refresh()

    def _preview_failed(self, message: str) -> None:
        self.status.setText(f"Preview failed: {message.splitlines()[0]}")

    def _preview_reset(self) -> None:
        self.preview_button.setEnabled(True)
        self.preview_button.setText("Preview segmentation here (current parameters)")
