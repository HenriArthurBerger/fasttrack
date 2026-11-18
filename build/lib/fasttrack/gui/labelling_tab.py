"""Labelling: click through cell crops, give each a category, train the event classifier.

Each crop is a strip of frames around one cell at one frame (the middle, outlined), one row
per channel, following the cell's track. Keys 1-9 pick a category, Space skips, Backspace goes
back and removes the last label. Labels are saved to the results folder at every click.
"""
from __future__ import annotations

import copy
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ..experiment import LABELLING_DEFAULTS, loaded_channels
from .worker import FunctionWorker

SOURCE_TEXT = {"random": "a random cell", "track_end": "a track that ends", "track_start": "a track that starts",
               "division": "a mother before its daughters"}


def _strip_images(stack: np.ndarray) -> list[list[QPixmap]]:
    """(C, frames, s, s) -> pixmaps, contrast shared along each channel's row, log scaled."""
    out = []
    for row in np.log2(np.maximum(stack, 0) + 1):
        lo, hi = np.percentile(row, [1, 99.7])
        scaled = np.clip((row - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
        pix = []
        for frame in scaled:
            frame = np.ascontiguousarray(frame)
            h, w = frame.shape
            pix.append(QPixmap.fromImage(QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8).copy()))
        out.append(pix)
    return out


class LabellingTab(QWidget):
    changed = Signal()

    def __init__(self, get_experiment, get_timestacks):
        super().__init__()
        self.get_experiment = get_experiment
        self.get_timestacks = get_timestacks
        self.lab = copy.deepcopy(LABELLING_DEFAULTS)
        self.rows: list[dict] = []
        self.history: list[dict] = []
        self.current: dict | None = None
        self.rng = random.Random()
        self.results_cache: dict = {}
        self.reader = None
        self.prefetch = ThreadPoolExecutor(1)
        self.next_future = None
        self.worker = None

        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        side = QWidget()
        side.setMinimumWidth(340)
        side.setMaximumWidth(420)
        col = QVBoxLayout(side)
        box = QGroupBox("Categories and crops")
        form = QFormLayout(box)
        self.categories = QLineEdit()
        self.categories.setToolTip("Comma separated. Keys 1-9 follow this order.")
        self.categories.editingFinished.connect(self._categories_changed)
        form.addRow("Categories", self.categories)
        self.window = QSpinBox()
        self.window.setRange(0, 10)
        form.addRow("Frames before and after", self.window)
        self.crop = QSpinBox()
        self.crop.setRange(16, 256)
        self.crop.setSingleStep(8)
        form.addRow("Crop size (px)", self.crop)
        self.channels = QLineEdit()
        self.channels.setPlaceholderText("nuclear channel only")
        self.channels.setToolTip("Channel names, comma separated, shown and used by the classifier.")
        form.addRow("Channels", self.channels)
        col.addWidget(box)
        box = QGroupBox("What to show, by weight")
        form = QFormLayout(box)
        self.weights = {}
        for source, text in SOURCE_TEXT.items():
            w = QDoubleSpinBox()
            w.setRange(0, 10)
            w.setSingleStep(0.05)
            form.addRow(text.capitalize(), w)
            self.weights[source] = w
        col.addWidget(box)
        box = QGroupBox("Classifier")
        form = QFormLayout(box)
        self.steps = QSpinBox()
        self.steps.setRange(50, 20000)
        self.steps.setSingleStep(100)
        form.addRow("Training steps", self.steps)
        row = QHBoxLayout()
        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.train)
        self.classify_button = QPushButton("Classify all positions")
        self.classify_button.clicked.connect(self.classify)
        row.addWidget(self.train_button)
        row.addWidget(self.classify_button)
        form.addRow(row)
        self.report = QLabel("")
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.report)
        col.addWidget(box)
        col.addStretch()
        split.addWidget(side)

        main = QWidget()
        ml = QVBoxLayout(main)
        top = QHBoxLayout()
        self.start_button = QPushButton("Start labelling")
        self.start_button.clicked.connect(self.next_sample)
        top.addWidget(self.start_button)
        self.info = QLabel("Track some positions first, then start. Labels are saved at every click.")
        self.info.setWordWrap(True)
        top.addWidget(self.info, 1)
        ml.addLayout(top)
        self.strip_host = QWidget()
        self.strip = QGridLayout(self.strip_host)
        self.strip.setSpacing(4)
        ml.addWidget(self.strip_host, 1, Qt.AlignmentFlag.AlignCenter)
        self.buttons_row = QHBoxLayout()
        ml.addLayout(self.buttons_row)
        nav = QHBoxLayout()
        back = QPushButton("Back (Backspace)")
        back.clicked.connect(self.back)
        skip = QPushButton("Skip (Space)")
        skip.clicked.connect(self.next_sample)
        nav.addWidget(back)
        nav.addWidget(skip)
        nav.addStretch()
        self.counts = QLabel("")
        nav.addWidget(self.counts)
        ml.addLayout(nav)
        split.addWidget(main)
        split.setSizes([380, 1000])
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, lambda: self._typing() or self.next_sample())
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, lambda: self._typing() or self.back())
        for n in range(1, 10):
            QShortcut(QKeySequence(str(n)), self, lambda n=n: self._typing() or self._key(n))

    def _typing(self) -> bool:
        """True while a text or number field has focus, so keys type instead of labelling."""
        from PySide6.QtWidgets import QAbstractSpinBox, QApplication
        focus = QApplication.focusWidget()
        return isinstance(focus, (QLineEdit, QAbstractSpinBox))

    # ------------------------------------------------------------------ experiment file
    def load(self, lab: dict) -> None:
        self.lab = copy.deepcopy(lab)
        self.categories.setText(", ".join(lab["categories"]))
        self.window.setValue(lab["window"])
        self.crop.setValue(lab["crop"])
        self.channels.setText(", ".join(lab["channels"]))
        for source, w in self.weights.items():
            w.setValue(float(lab["sample"].get(source, 0)))
        self.steps.setValue(lab["train_steps"])
        self.rows, self.history, self.current = [], [], None
        self.results_cache = {}
        self._close_reader()
        self._category_buttons()
        self._clear_strip()
        self.info.setText("Track some positions first, then start. Labels are saved at every click.")

    def collect(self) -> dict:
        lab = copy.deepcopy(self.lab)
        lab["categories"] = [c.strip() for c in self.categories.text().split(",") if c.strip()] or lab["categories"]
        lab["window"] = self.window.value()
        lab["crop"] = self.crop.value()
        lab["channels"] = [c.strip() for c in self.channels.text().split(",") if c.strip()]
        lab["sample"] = {s: w.value() for s, w in self.weights.items()}
        lab["train_steps"] = self.steps.value()
        self.lab = lab
        return copy.deepcopy(lab)

    def _categories_changed(self) -> None:
        self.collect()
        self._category_buttons()
        self._show_counts()
        self.changed.emit()

    def _category_buttons(self) -> None:
        while self.buttons_row.count():
            w = self.buttons_row.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        for i, c in enumerate(self.lab["categories"][:9]):
            b = QPushButton(f"{i + 1}  {c}")
            b.setMinimumHeight(40)
            b.clicked.connect(lambda _=False, c=c: self.label(c))
            self.buttons_row.addWidget(b)

    # ------------------------------------------------------------------ sampling
    def _close_reader(self) -> None:
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def _ensure(self) -> dict | None:
        from .. import events
        exp = copy.deepcopy(self.get_experiment())
        exp["labelling"] = self.collect()
        if not exp["paths"]["results"]:
            self.info.setText("Set the results folder on the Experiment page.")
            return None
        if self.reader is None or self.reader.exp["labelling"] != exp["labelling"]:
            self._close_reader()
            self.reader = events.FrameReader(exp, self.get_timestacks() or None, int(exp["files"]["io_threads"]))
            self.rows = events.read_labels(exp)
        return exp

    def _draw(self, exp: dict) -> tuple[dict, np.ndarray] | None:
        from .. import events
        done = {(r["position"], r["t"], r["track"]) for r in self.rows}
        obs = events.sample(exp, self.rng, done, self.results_cache)
        if obs is None:
            return None
        results = self.results_cache[obs["position"]][0]
        return obs, self.reader.crop(results, obs)

    def next_sample(self) -> None:
        exp = self._ensure()
        if exp is None:
            return
        try:
            if self.next_future is not None:
                got = self.next_future.result()
            else:
                got = self._draw(exp)
        except (KeyError, OSError, ValueError) as exc:
            self.next_future = None
            self.info.setText(f"Could not read a crop: {exc}")
            return
        self.next_future = self.prefetch.submit(self._draw, exp)
        if got is None:
            self.info.setText("No tracked positions found in the results folder. Run tracking first.")
            return
        self.current, stack = got
        self._show(self.current, stack)

    def _clear_strip(self) -> None:
        while self.strip.count():
            w = self.strip.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _show(self, obs: dict, stack: np.ndarray) -> None:
        self._clear_strip()
        names = [c["name"] for c in loaded_channels(self.get_experiment())]
        chosen = self.lab["channels"] or [next((c["name"] for c in loaded_channels(self.get_experiment()) if c["role"] == "nuclear"), "nuclear")]
        w = self.lab["window"]
        frames = stack.shape[1]
        cell = max(80, min(180, 1100 // max(frames, 1)))
        for r, row in enumerate(_strip_images(stack)):
            label = QLabel(chosen[r] if r < len(chosen) else (names[r] if r < len(names) else f"c{r + 1}"))
            self.strip.addWidget(label, r + 1, 0)
            for c, pix in enumerate(row):
                image = QLabel()
                image.setPixmap(pix.scaled(cell, cell, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
                if c == w:
                    image.setStyleSheet("border: 3px solid #d33")
                self.strip.addWidget(image, r + 1, c + 1)
        for c in range(frames):
            self.strip.addWidget(QLabel(f"{c - w:+d}" if c != w else f"frame {obs['t']}"), 0, c + 1, Qt.AlignmentFlag.AlignCenter)
        previous = next((r["category"] for r in self.rows if (r["position"], r["t"], r["track"]) == (obs["position"], obs["t"], obs["track"])), None)
        self.info.setText(f"{obs['position']}, frame {obs['t']}, track {obs['track']}: {SOURCE_TEXT[obs['source']]}."
                          + (f" Labelled {previous}." if previous else ""))
        self._show_counts()

    def _show_counts(self) -> None:
        counts = {c: 0 for c in self.lab["categories"]}
        for r in self.rows:
            if r["category"] in counts:
                counts[r["category"]] += 1
        self.counts.setText("Labelled: " + ", ".join(f"{c} {n}" for c, n in counts.items()) + "   (aim for 30-50 each)")

    # ------------------------------------------------------------------ labelling
    def _key(self, n: int) -> None:
        if n <= len(self.lab["categories"]):
            self.label(self.lab["categories"][n - 1])

    def label(self, category: str) -> None:
        if self.current is None:
            return
        from .. import events
        exp = self._ensure()
        if exp is None:
            return
        self.rows = events.add_label(exp, self.rows, self.current, category)
        self.history.append(self.current)
        self.changed.emit()
        self.next_sample()

    def back(self) -> None:
        if not self.history:
            return
        from .. import events
        exp = self._ensure()
        obs = self.history.pop()
        key = (obs["position"], obs["t"], obs["track"])
        self.rows = [r for r in self.rows if (r["position"], r["t"], r["track"]) != key]
        events.write_labels(exp, self.rows)
        self.current = obs
        self._show(obs, self.reader.crop(self.results_cache[obs["position"]][0], obs))

    # ------------------------------------------------------------------ classifier
    def train(self) -> None:
        exp = self._ensure()
        if exp is None:
            return
        self.train_button.setEnabled(False)
        self.report.setText("Cutting crops and training…")

        def job(on_event, stop):
            from ..events import train_from_labels
            return train_from_labels(exp, on_event, self.get_timestacks() or None)

        self.worker = FunctionWorker(job, self)
        self.worker.event.connect(lambda e: e["kind"] == "train" and self.report.setText(f"Training: step {e['step']}, loss {e['loss']:.3f}"))
        self.worker.finished_ok.connect(self._trained)
        self.worker.failed.connect(lambda m: self.report.setText(m.splitlines()[0]))
        self.worker.finished.connect(lambda: self.train_button.setEnabled(True))
        self.worker.start()

    def _trained(self, report: dict) -> None:
        text = "Trained on " + ", ".join(f"{c} {n}" for c, n in report["counts"].items()) + "."
        if "holdout_accuracy" in report:
            classes = report["classes"]
            rows = "".join(f"<tr><td>{c}</td>" + "".join(f"<td align=right>{v}</td>" for v in report["confusion"][i]) + "</tr>"
                           for i, c in enumerate(classes))
            text += (f"<br>Held-out accuracy {report['holdout_accuracy']:.2f} on {report['holdout_n']} labelled crops "
                     f"(rows true, columns predicted):<table><tr><td></td>" + "".join(f"<td>{c}</td>" for c in classes)
                     + f"</tr>{rows}</table>The final model uses every label.")
        self.report.setText(text)

    def classify(self) -> None:
        exp = self._ensure()
        if exp is None:
            return
        self.classify_button.setEnabled(False)
        self.report.setText("Classifying…")

        def job(on_event, stop):
            from ..events import classify_positions
            return classify_positions(exp, on_event, stop, self.get_timestacks() or None)

        self.worker = FunctionWorker(job, self)
        self.worker.event.connect(lambda e: e["kind"] == "classify" and self.report.setText(f"Classified {e['done']} of {e['total']} positions"))
        self.worker.finished_ok.connect(lambda counts: self.report.setText(
            f"Wrote Events/<position>.csv for {len(counts)} positions, {sum(counts.values())} candidates."))
        self.worker.failed.connect(lambda m: self.report.setText(m.splitlines()[0]))
        self.worker.finished.connect(lambda: self.classify_button.setEnabled(True))
        self.worker.start()
