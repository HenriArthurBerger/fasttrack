"""Parameters: a form generated from the config dict, so a new key shows up by itself."""
from __future__ import annotations

import copy

import yaml
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ..experiment import pipeline_defaults
TIPS = {
    ("link", "max_travel"): "Largest move in px per frame. 0 estimates it from the data; any other value overrides that.",
    ("link", "method"): "multipass, notebook, or tracks",
    ("centroids", "seed_overlap"): "Higher merges more seeds. Too high merges real neighbours.",
    (None, "batch_auto"): "Grow batches while GPU memory allows. Out-of-memory splits the batch and carries on.",
    ("output", "overwrite"): "Redo timestacks that already have results. Off: finished timestacks are skipped.",
    ("measure", "ring_gap"): "Pixels skipped between the nuclear mask and the cytoplasm ring. 1 skips the border layer; 2 also skips "
                             "the blurred nuclear edge (about 6% nuclear signal one pixel out).",
    ("measure", "ring_width"): "Width of the cytoplasm ring in pixels, beyond the gap.",
    ("measure", "cyto_mask"): "Keep only ring pixels significantly above background in at least one mask channel, so ring pixels "
                              "off the cell do not dilute the cytoplasm values. The same trimmed ring is used for every channel. Saves the kept fraction per cell as cyto_kept.",
    ("measure", "cyto_mask_channel"): "Channels whose signal marks the cell (comma-separated names). Empty: every channel but the nuclear one, "
                                      "whose blur would pass every ring pixel. Each is tested against its own background; a ring pixel is kept if any of them passes.",
    ("measure", "cyto_mask_sigma"): "Threshold in background-noise sigmas above the background level (noise of the dim tail of "
                                    "each background tile after the blur, not of the whole image).",
    ("measure", "cyto_mask_smooth"): "Gaussian blur (px) before the test. Pixel noise is about as large as dim cytoplasm, so "
                                     "without it the mask keeps pixels whose noise was high and biases cytoplasm means up.",
    ("measure", "cyto_min_kept"): "With the mask on: a cell keeping less than this share of its ring gets NaN cytoplasm values. "
                                  "Such cells nearly always do not express the reporter (on the KTR plate about 1 in 5), and "
                                  "their ratio is background over background.",
    ("measure", "cyto_min_pixels"): "A cell with fewer ring pixels left gets NaN cytoplasm values instead of a noisy ratio.",
    (None, "device"): "cuda:0 for the first GPU, cpu to run without one (slow).",
}


def _editor(value):
    if isinstance(value, bool):
        widget = QCheckBox()
        widget.setChecked(value)
        return widget, widget.isChecked
    if isinstance(value, int):
        widget = QSpinBox()
        widget.setRange(-10**9, 10**9)
        widget.setValue(value)
        return widget, widget.value
    if isinstance(value, float):
        widget = QDoubleSpinBox()
        widget.setDecimals(4)
        widget.setRange(-1e9, 1e9)
        widget.setSingleStep(0.05)
        widget.setValue(value)
        return widget, widget.value
    widget = QLineEdit(yaml.safe_dump(value, default_flow_style=True).split("\n...")[0].strip() if not isinstance(value, str) else value)
    if isinstance(value, str):
        return widget, widget.text
    return widget, lambda: yaml.safe_load(widget.text())


class ParametersTab(QWidget):
    """Edits the experiment file's `pipeline` section. Saved with the experiment, never on its own."""

    changed = Signal()

    def __init__(self, params: dict | None = None):
        super().__init__()
        self.params = copy.deepcopy(params) if params is not None else pipeline_defaults()
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self.reset)
        bar.addWidget(reset)
        note = QLabel("Every value here is saved in the experiment file. Hover a field for notes.")
        note.setProperty("role", "hint")
        bar.addWidget(note, 1)
        layout.addLayout(bar)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)
        self.readers: dict[tuple, object] = {}
        self.rebuild()

    def rebuild(self) -> None:
        host = QWidget()
        columns = QHBoxLayout(host)
        left, right = QVBoxLayout(), QVBoxLayout()
        columns.addLayout(left)
        columns.addLayout(right)
        self.readers = {}
        top = {k: v for k, v in self.params.items() if not isinstance(v, dict)}
        groups = [("general", top)] + [(k, v) for k, v in self.params.items() if isinstance(v, dict)]
        for i, (name, values) in enumerate(groups):
            box = QGroupBox(name)
            form = QFormLayout(box)
            for key, value in values.items():
                widget, reader = _editor(value)
                group = None if name == "general" else name
                widget.setToolTip(TIPS.get((group, key), ""))
                form.addRow(key, widget)
                self.readers[(group, key)] = reader
            (left if i % 2 == 0 else right).addWidget(box)
        left.addStretch()
        right.addStretch()
        self.scroll.setWidget(host)

    def collect(self) -> dict:
        """The form as a pipeline dict."""
        for (group, key), reader in self.readers.items():
            value = reader()
            if group is None:
                self.params[key] = value
            else:
                self.params[group][key] = value
        return copy.deepcopy(self.params)

    def load(self, params: dict) -> None:
        self.params = copy.deepcopy(params)
        self.rebuild()

    def reset(self) -> None:
        self.load(pipeline_defaults())
        self.changed.emit()
