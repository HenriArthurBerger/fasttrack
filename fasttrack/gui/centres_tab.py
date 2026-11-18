"""Label cells: one labeller for nucleus centres and cell states, the dataset they make, and a review of the
labels the models disagree with.

Label. A patch opens at a likely mistake or event, with tracking's centres on it. Click a centre to remove it,
click a nucleus without one to add it. Keys 1-9 give the point under the crosshair a state (or add a point
there with that state); points have the first state unless you change it. Every nucleus whose centre is
inside the white box needs exactly one point; the margin is context. Left/Right flip through the frames
before and after (7 each way by default) to tell an odd-shaped nucleus from two, a division from debris.

Dataset. Every labelled patch, or every labelled cell grouped by state. Double-click one to correct it.

Review. After Check labels, every label judged by models that never saw it: patches where the seed finder's
centres differ from yours, cells where the classifier's state does. Double-click to look and correct.

One file holds it all (Labels/centres.json; see fasttrack/dataset.py), and trains both models.
"""
from __future__ import annotations

import copy
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListView, QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
                               QVBoxLayout, QWidget)

from ..experiment import LABELLING_DEFAULTS
from .image_view import ImageCanvas
from .worker import FunctionWorker

# bright, saturated colours that stand out on grey nuclei and the dark background
KEPT, ADDED, BOX, OTHER, CONTEXT, MODEL = "#39ff14", "#ffee00", "#f0f0f0", "#39ff14", "#9aa0a8", "#c77dff"
STATE_COLOURS = ["#39ff14", "#ff3b3b", "#ff4fd8", "#c77dff", "#ff8c00", "#ffffff", "#ffee00", "#00ff9c", "#9dff00"]
RADIUS = 4.0      # image px: the centre circle
PEN = 3.0         # screen px: marker lines, drawn over a dark outline
NEAR = 5.0        # px: a click this close to a point removes it
STATE_NEAR = 8.0  # px: a number key gives its state to the point this close to the crosshair


def _scale(images: list[np.ndarray]) -> list[np.ndarray]:
    """uint8, log scaled, one contrast for all (from the middle frame) so the frames compare."""
    logs = [np.log2(np.maximum(a.astype(np.float32), 0) + 1) for a in images]
    mid = logs[len(logs) // 2]
    lo, hi = np.percentile(mid[mid > 0] if (mid > 0).any() else mid, [1, 99.8])   # the black padding off the image edge does not count
    return [np.clip((a - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8) for a in logs]


def _circle(x: float, y: float, r: float) -> list[tuple[float, float]]:
    return [(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 25)]


def _marked(path: list[tuple[float, float]], colour: str, width: float = PEN) -> list[tuple]:
    """A path in `colour` over a dark outline, so it shows on bright nuclei and dark background alike."""
    return [(path, "#000000", width + 2.5), (path, colour, width)]


def _square(x: float, y: float, r: float) -> list[tuple[float, float]]:
    return [(x - r, y - r), (x - r, y + r), (x + r, y + r), (x + r, y - r), (x - r, y - r)]


def _pixmap(gray: np.ndarray) -> QPixmap:
    h, w = gray.shape
    return QPixmap.fromImage(QImage(np.ascontiguousarray(gray).data, w, h, w, QImage.Format.Format_Grayscale8).copy())


def _thumb(gray: np.ndarray, side: int, box=None, points=(), border: str | None = None) -> QPixmap:
    """A small frame scaled to `side` px, with an optional box (m, s) and points [(x, y, colour)] in its pixels."""
    h, w = gray.shape
    pix = _pixmap(gray).scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    f = pix.width() / w
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    if box is not None:
        m, s = box
        p.setPen(QPen(QColor(BOX), 1))
        p.drawRect(int(m * f), int(m * f), int(s * f), int(s * f))
    for x, y, colour in points:
        p.setPen(QPen(QColor(0, 0, 0, 200), 3.5))
        p.drawEllipse(QPointF(y * f, x * f), 3.5, 3.5)
        p.setPen(QPen(QColor(colour), 2.0))
        p.drawEllipse(QPointF(y * f, x * f), 3.5, 3.5)
    if border:
        p.setPen(QPen(QColor(border), 4))
        p.drawRect(1, 1, pix.width() - 2, pix.height() - 2)
    p.end()
    return pix


class _Thumb(QLabel):
    clicked = Signal(int)

    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.index)


class CentresTab(QWidget):
    """The Label cells page (named for its first job, labelling centres)."""
    changed = Signal()
    use_model = Signal(str, float)  # checkpoint path, threshold

    def __init__(self, get_experiment, get_timestacks):
        super().__init__()
        self.get_experiment = get_experiment
        self.get_timestacks = get_timestacks
        self.lab = copy.deepcopy(LABELLING_DEFAULTS)
        self.rows: list[dict] = []
        self.history: list[dict] = []
        self.skipped: set = set()
        self.current: dict | None = None
        self.points: list[list[float]] = []
        self.states: list[str] = []
        self.tracked: list[list[float]] = []
        self.undo: list[tuple] = []
        self.sampler = None
        self.reader = None
        self.session = None
        self.prefetch = ThreadPoolExecutor(1)
        self.next_future = None
        self.worker = None
        self.thumb_worker = None
        self.report_data = None
        self.view_dt = 0
        self.hover: tuple[float, float] | None = None
        self.overlay: dict | None = None      # review: the model's centres or state for the open patch
        self.review: dict | None = None
        self.review_items: list[dict] = []
        self.thumb_cache: dict = {}

        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)
        split.addWidget(self._build_side())
        self.pages = QTabWidget()
        self.pages.addTab(self._build_label_page(), "Label")
        self.pages.addTab(self._build_dataset_page(), "Dataset")
        self.pages.addTab(self._build_review_page(), "Review")
        self.pages.currentChanged.connect(self._page_changed)
        split.addWidget(self.pages)
        split.setSizes([380, 1100])

        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._shortcut(QKeySequence(key), self.save)
        self._shortcut(QKeySequence(Qt.Key.Key_Space), self.skip)
        self._shortcut(QKeySequence(Qt.Key.Key_Backspace), self.back)
        self._shortcut(QKeySequence.StandardKey.Undo, self.undo_click)
        self._shortcut(QKeySequence(Qt.Key.Key_Left), lambda: self.view_frame(self.view_dt - 1))
        self._shortcut(QKeySequence(Qt.Key.Key_Right), lambda: self.view_frame(self.view_dt + 1))
        self._shortcut(QKeySequence(Qt.Key.Key_Up), lambda: self.view_frame(0))
        for n in range(1, 10):
            self._shortcut(QKeySequence(str(n)), lambda n=n: self.state_key(n))

    def _shortcut(self, seq, fn) -> None:
        """Keys act only on the Label page and never while typing in a settings field."""
        QShortcut(seq, self, lambda: self._typing() or self.pages.currentIndex() != 0 or fn())

    def _typing(self) -> bool:
        from PySide6.QtWidgets import QAbstractSpinBox, QApplication
        return isinstance(QApplication.focusWidget(), (QLineEdit, QAbstractSpinBox))

    # ------------------------------------------------------------------ layout
    def _build_side(self) -> QWidget:
        side = QWidget()
        col = QVBoxLayout(side)

        box = QGroupBox("Patches")
        form = QFormLayout(box)
        self.patch = QSpinBox()
        self.patch.setRange(32, 256)
        self.patch.setSingleStep(16)
        self.patch.setToolTip("Side of the box in which you click every nucleus centre. Smaller boxes are quicker to label.")
        form.addRow("Box size (px)", self.patch)
        self.margin = QSpinBox()
        self.margin.setRange(0, 96)
        self.margin.setSingleStep(8)
        self.margin.setToolTip("Context shown around the box. Nuclei centred there are not labelled.")
        form.addRow("Margin (px)", self.margin)
        self.context = QSpinBox()
        self.context.setRange(3, 15)
        self.context.setToolTip("Frames shown before and after the labelled one (Left/Right flips through them). Seeing a nucleus "
                                "over time tells an odd angle from two nuclei, and a division from debris.")
        form.addRow("Frames before and after", self.context)
        col.addWidget(box)

        box = QGroupBox("Where to look, by weight")
        form = QFormLayout(box)
        note = QLabel("Count changes: the number of cells changes for a frame or two and comes back (a nucleus split, missed or "
                      "merged). Track ends, starts and divisions put an event cell in the centre, so states such as divisions and "
                      "deaths turn up often enough to train on.")
        note.setWordWrap(True)
        note.setProperty("role", "hint")
        form.addRow(note)
        self.weights = {}
        for key, text in (("count_change", "Count changes"), ("random", "Random cells"), ("track_end", "Track ends"),
                          ("track_start", "Track starts"), ("division", "Divisions")):
            w = QDoubleSpinBox()
            w.setRange(0, 10)
            w.setSingleStep(0.05)
            form.addRow(text, w)
            self.weights[key] = w
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(5, 100)
        self.sigma.setToolTip("Cells are counted over about this distance; about one to two nuclear diameters.")
        form.addRow("Count over (px)", self.sigma)
        self.window = QSpinBox()
        self.window.setRange(3, 15)
        self.window.setSingleStep(2)
        self.window.setToolTip("Each frame is compared with the median of this many frames around it. 5 catches errors lasting up to 2 frames.")
        form.addRow("Compare with frames", self.window)
        self.min_change = QDoubleSpinBox()
        self.min_change.setRange(0.1, 5)
        self.min_change.setSingleStep(0.1)
        self.min_change.setToolTip("1 = one nucleus more or less than in the frames around.")
        form.addRow("Smallest change", self.min_change)
        col.addWidget(box)

        box = QGroupBox("Cell states")
        form = QFormLayout(box)
        self.categories = QLineEdit()
        self.categories.setToolTip("Comma separated. Keys 1-9 follow this order; the first is every point's state unless you change it.")
        self.categories.editingFinished.connect(self._categories_changed)
        form.addRow("States", self.categories)
        self.state_window = QSpinBox()
        self.state_window.setRange(0, 10)
        self.state_window.setToolTip("Frames before and after each cell that the state classifier sees.")
        form.addRow("Classifier frames ±", self.state_window)
        self.crop = QSpinBox()
        self.crop.setRange(16, 256)
        self.crop.setSingleStep(8)
        self.crop.setToolTip("Side of the crop around each cell that the state classifier sees.")
        form.addRow("Classifier crop (px)", self.crop)
        self.channels = QLineEdit()
        self.channels.setPlaceholderText("nuclear channel only")
        self.channels.setToolTip("Channel names, comma separated, that the state classifier sees.")
        form.addRow("Classifier channels", self.channels)
        col.addWidget(box)

        box = QGroupBox("Models")
        form = QFormLayout(box)
        self.steps = QSpinBox()
        self.steps.setRange(100, 20000)
        self.steps.setSingleStep(100)
        form.addRow("Seed finder steps", self.steps)
        self.state_steps = QSpinBox()
        self.state_steps.setRange(50, 20000)
        self.state_steps.setSingleStep(100)
        form.addRow("State classifier steps", self.state_steps)
        self.gate = QDoubleSpinBox()
        self.gate.setRange(1, 30)
        self.gate.setToolTip("A found centre this close to a clicked one is the same nucleus when scoring.")
        form.addRow("Match distance (px)", self.gate)
        self.folds = QSpinBox()
        self.folds.setRange(2, 10)
        self.folds.setToolTip("Check labels trains this many models, each on all labels but one share, and judges that share.")
        form.addRow("Check folds", self.folds)
        row = QHBoxLayout()
        self.train_button = QPushButton("Train both")
        self.train_button.setToolTip("Train the seed finder and the state classifier on the dataset.")
        self.train_button.clicked.connect(self.train)
        self.check_button = QPushButton("Check labels")
        self.check_button.setToolTip("Judge every label with models that never saw it, then review where they disagree.")
        self.check_button.clicked.connect(self.check)
        row.addWidget(self.train_button)
        row.addWidget(self.check_button)
        form.addRow(row)
        row = QHBoxLayout()
        self.use_button = QPushButton("Use seed finder for tracking")
        self.use_button.setToolTip("Sets centroids.method = cnn with this model and its best threshold. Check it in a test run.")
        self.use_button.setEnabled(False)
        self.use_button.clicked.connect(self._use)
        self.classify_button = QPushButton("Classify all positions")
        self.classify_button.setToolTip("Give every track end, start and division a state with the classifier: Events/<position>.csv.")
        self.classify_button.clicked.connect(self.classify)
        row.addWidget(self.use_button)
        row.addWidget(self.classify_button)
        form.addRow(row)
        self.report = QLabel("")
        self.report.setWordWrap(True)
        self.report.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.report)
        col.addWidget(box)
        col.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)   # fields fit the width; only scroll down
        side.setMinimumWidth(380)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(480)
        return scroll

    def _build_label_page(self) -> QWidget:
        main = QWidget()
        ml = QVBoxLayout(main)
        top = QHBoxLayout()
        self.start_button = QPushButton("Start labelling")
        self.start_button.clicked.connect(self.next_patch)
        top.addWidget(self.start_button)
        self.info = QLabel("Track some positions first, then start. Patches are saved when you press Enter.")
        self.info.setWordWrap(True)
        top.addWidget(self.info, 1)
        ml.addLayout(top)
        self.canvas = ImageCanvas()
        self.canvas.setMinimumSize(420, 420)
        self.canvas.set_crosshair(True, "#00f0ff")   # cyan: unlike every marker colour
        self.canvas.pixel_clicked.connect(self._clicked)
        self.canvas.pixel_hovered.connect(lambda x, y: setattr(self, "hover", (x, y)))
        ml.addWidget(self.canvas, 1)
        self.film_host = QWidget()
        self.film = QHBoxLayout(self.film_host)
        self.film.setContentsMargins(0, 0, 0, 0)
        self.film.setSpacing(4)
        self.film.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.film_scroll = film_scroll = QScrollArea()          # 15 frames fit a wide window; a narrow one scrolls
        film_scroll.setWidget(self.film_host)
        film_scroll.setWidgetResizable(True)
        film_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        film_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        film_scroll.setFixedHeight(120)
        ml.addWidget(film_scroll)
        self.legend = QLabel("")
        self.legend.setWordWrap(True)
        ml.addWidget(self.legend)
        nav = QHBoxLayout()
        for text, slot in (("Save and next (Enter)", self.save), ("Skip (Space)", self.skip), ("Back (Backspace)", self.back),
                           ("Undo (Ctrl+Z)", self.undo_click), ("Clear box", self.clear_points), ("Reset to tracking", self.reset_points)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            nav.addWidget(b)
        nav.addStretch()
        self.counts = QLabel("")
        nav.addWidget(self.counts)
        ml.addLayout(nav)
        return main

    def _build_dataset_page(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        row = QHBoxLayout()
        self.ds_mode = QComboBox()
        self.ds_mode.addItems(["Patches", "Cells by state"])
        self.ds_filter = QComboBox()
        self.ds_refresh = QPushButton("Refresh")
        for w in (self.ds_mode, self.ds_filter):
            w.currentIndexChanged.connect(lambda _: self.show_dataset())
        self.ds_refresh.clicked.connect(lambda: self.show_dataset(reload=True))
        row.addWidget(QLabel("Show"))
        row.addWidget(self.ds_mode)
        row.addWidget(self.ds_filter)
        row.addWidget(self.ds_refresh)
        self.ds_summary = QLabel("")
        row.addWidget(self.ds_summary, 1)
        col.addLayout(row)
        self.ds_list = QListWidget()
        self.ds_list.setViewMode(QListView.ViewMode.IconMode)
        self.ds_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.ds_list.setMovement(QListView.Movement.Static)
        self.ds_list.setUniformItemSizes(True)
        self.ds_list.itemDoubleClicked.connect(lambda item: self.open_row(item.data(Qt.ItemDataRole.UserRole)))
        col.addWidget(self.ds_list, 1)
        hint = QLabel("Double-click a patch or cell to open it on the Label page; correct it and press Enter to save.")
        hint.setProperty("role", "hint")
        col.addWidget(hint)
        return page

    def _build_review_page(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        row = QHBoxLayout()
        check = QPushButton("Check labels")
        check.clicked.connect(self.check)
        row.addWidget(check)
        self.rv_filter = QComboBox()
        self.rv_filter.addItems(["Every disagreement", "Centres", "States"])
        self.rv_filter.currentIndexChanged.connect(lambda _: self.show_review())
        row.addWidget(self.rv_filter)
        self.rv_hide = QCheckBox("Hide ones relabelled since")
        self.rv_hide.setChecked(True)
        self.rv_hide.toggled.connect(lambda _: self.show_review())
        row.addWidget(self.rv_hide)
        self.rv_status = QLabel("")
        self.rv_status.setWordWrap(True)
        row.addWidget(self.rv_status, 1)
        col.addLayout(row)
        self.rv_list = QListWidget()
        self.rv_list.itemDoubleClicked.connect(self._open_review_item)
        col.addWidget(self.rv_list, 1)
        hint = QLabel("Each label was judged by models trained without it. A disagreement is a weak spot of the model, or a "
                      "label to fix. Double-click to open it: the model's centres are orange squares. Fix it and press Enter, "
                      "or press Enter to keep yours.")
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")
        col.addWidget(hint)
        return page

    # ------------------------------------------------------------------ experiment file
    @property
    def c(self) -> dict:
        return self.lab["centres"]

    def load(self, lab: dict) -> None:
        """lab: the experiment's labelling section (with its centres part)."""
        if "centres" not in lab:          # only the centres part was given
            lab = {**copy.deepcopy(LABELLING_DEFAULTS), "centres": lab}
        self.lab = copy.deepcopy(lab)
        c = self.c
        self.patch.setValue(int(c["patch"]))
        self.margin.setValue(int(c["margin"]))
        context = int(c.get("context", 7))
        self.context.setValue(7 if context == 3 else context)   # 3 was the earlier default; 7 shows a division's run-up
        for key, w in self.weights.items():
            w.setValue(float(c["sample"].get(key, 0)))
        self.sigma.setValue(float(c["count_sigma"]))
        self.window.setValue(int(c["count_window"]))
        self.min_change.setValue(float(c["min_change"]))
        self.steps.setValue(int(c["train_steps"]))
        self.gate.setValue(float(c["match_distance"]))
        self.categories.setText(", ".join(lab["categories"]))
        self.state_window.setValue(int(lab["window"]))
        self.crop.setValue(int(lab["crop"]))
        self.channels.setText(", ".join(lab["channels"]))
        self.state_steps.setValue(int(lab["train_steps"]))
        self.folds.setValue(int(lab.get("folds", 5)))
        self._reset_session()
        self.review, self.review_items, self.thumb_cache = None, [], {}
        self.info.setText("Track some positions first, then start. Patches are saved when you press Enter.")
        self._legend()
        self._show_report()

    def collect(self) -> dict:
        lab = copy.deepcopy(self.lab)
        lab["categories"] = [c.strip() for c in self.categories.text().split(",") if c.strip()] or lab["categories"]
        lab.update({"window": self.state_window.value(), "crop": self.crop.value(), "train_steps": self.state_steps.value(),
                    "channels": [c.strip() for c in self.channels.text().split(",") if c.strip()], "folds": self.folds.value()})
        lab["centres"].update({"patch": self.patch.value(), "margin": self.margin.value(), "context": self.context.value(),
                               "count_sigma": self.sigma.value(), "count_window": self.window.value() | 1, "min_change": self.min_change.value(),
                               "sample": {k: w.value() for k, w in self.weights.items()},
                               "train_steps": self.steps.value(), "match_distance": self.gate.value()})
        self.lab = lab
        return copy.deepcopy(lab)

    def _exp(self) -> dict:
        exp = copy.deepcopy(self.get_experiment())
        exp["labelling"] = self.collect()
        return exp

    def _categories_changed(self) -> None:
        self.collect()
        self._legend()
        self._redraw()
        self._show_counts()
        self.changed.emit()

    def _reset_session(self) -> None:
        if self.reader is not None:
            self.reader.close()
        self.reader = self.sampler = self.session = self.current = self.next_future = None
        self.rows, self.history, self.points, self.states, self.tracked, self.undo = [], [], [], [], [], []
        self.skipped = set()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        exp = self.get_experiment()
        if exp["paths"]["results"]:
            self._show_report()
            if not self.rows:
                from .. import dataset
                self.rows = dataset.read_dataset(self._exp())
                self._show_counts()

    # ------------------------------------------------------------------ sampling
    def _ensure(self) -> dict | None:
        from .. import centres, dataset
        exp = self._exp()
        if not exp["paths"]["results"]:
            self.info.setText("Set the results folder on the Experiment page.")
            return None
        session = (exp["paths"]["results"], exp["paths"]["raw"], *(self.c[k] for k in ("patch", "count_sigma", "count_window", "min_change", "context")),
                   tuple(self.c["sample"].items()))
        if session != self.session:
            self._reset_session()
            self.session = session
            self.reader = centres.NuclearReader(exp, self.get_timestacks() or None, 4)
            self.sampler = centres.Sampler(exp, self.reader.shape, available=self.reader.all())
            self.rows = dataset.read_dataset(exp)
        self.sampler.c = self.c
        self.sampler.exp = exp
        return exp

    def _frame_range(self, t: int) -> list[int]:
        k = max(3, int(self.c.get("context", 7)))
        return list(range(t - k, t + k + 1))

    def _draw(self) -> tuple[dict, list[np.ndarray]] | None:
        obs = self.sampler.draw(self.rows, self.skipped)
        if obs is None:
            return None
        return obs, self.reader.frames(obs["position"], self._frame_range(obs["t"]))

    def next_patch(self) -> None:
        exp = self._ensure()
        if exp is None:
            return
        try:
            got = self.next_future.result() if self.next_future is not None else self._draw()
        except (KeyError, OSError, ValueError) as exc:
            self.next_future = None
            self.info.setText(f"Could not read a patch: {exc}")
            return
        self.next_future = self.prefetch.submit(self._draw)
        if got is None:
            self.info.setText("No patches to show: no tracked position is also in the raw data folder (is the raw drive "
                              "connected?), or every candidate is labelled.")
            return
        obs, frames = got
        self.overlay = None
        self._open(obs, frames, None, None)

    def _open(self, obs: dict, frames: list[np.ndarray], points: list | None, states: list | None) -> None:
        from .. import centres, dataset
        self.current = obs
        self._raw = frames
        m = int(self.c["margin"])
        cuts = [centres.cut(f, obs["x0"], obs["y0"], obs["size"], m, "constant") for f in frames]
        self.k = len(cuts) // 2
        self.grays = _scale(cuts)
        self.tracked = self.sampler.tracked_points(obs)
        self.wide = {**obs, "x0": obs["x0"] - m, "y0": obs["y0"] - m, "size": obs["size"] + 2 * m}
        self.tracked_at = {dt: self.sampler.tracked_points(self.wide, obs["t"] + dt) for dt in range(-self.k, self.k + 1)}
        inside = {tuple(p) for p in self.tracked}
        self.context_pts = [p for p in self.tracked_at[0] if tuple(p) not in inside]
        default = dataset.default_state(self._exp())
        self.points = copy.deepcopy(points if points is not None else self.tracked)
        self.states = list(states) if states is not None else [default] * len(self.points)
        self.undo = []
        self.view_dt = 0
        self.canvas._user_zoomed = False
        self._film()
        self._redraw()
        self.canvas.setFocus()  # so Enter, Space and Backspace label instead of typing into a settings field
        text = f"{obs['position']}, frame {obs['t']}: {centres.SOURCE_TEXT.get(obs['source'], obs['source'])}"
        if obs["source"] == "count_change":
            prof = self.sampler.profile(obs)
            kind = ("more nuclei than in the frames around: look for one split in two, or debris" if obs["change"] > 0
                    else "fewer nuclei than in the frames around: look for one missed, or two merged")
            text += f" ({obs['change']:+.1f}), {kind}. Local count over frames {max(0, obs['t'] - 3)}-{obs['t'] + 3}: {', '.join(map(str, prof))}."
        else:
            text += "."
        if self.overlay:
            text += f"  {self.overlay['text'][0].upper() + self.overlay['text'][1:]}. Enter saves and goes back."
        elif any(centres.patch_key(r) == centres.patch_key(obs) for r in self.rows):
            text += " Already labelled; Enter replaces it."
        self.info.setText(text)
        self._show_counts()

    def _film(self) -> None:
        while self.film.count():
            w = self.film.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        m, s = int(self.c["margin"]), self.current["size"]
        n = 2 * self.k + 1
        side = max(56, min(130, (max(self.canvas.width(), 600) - 4 * n) // n))
        self.film_scroll.setFixedHeight(side + 40)   # thumbnails, their frame numbers, and room for a scroll bar
        self.thumbs = []
        for i, dt in enumerate(range(-self.k, self.k + 1)):
            cell = QWidget()
            lay = QVBoxLayout(cell)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(1)
            thumb = _Thumb(dt)
            thumb.clicked.connect(self.view_frame)
            pts = [(x - self.wide["x0"], y - self.wide["y0"], OTHER) for x, y in self.tracked_at[dt]]
            thumb.setPixmap(_thumb(self.grays[i], side, (m, s), pts, "#d33" if dt == self.view_dt else None))
            title = QLabel(f"frame {self.current['t']}" if dt == 0 else f"{dt:+d}")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(thumb)
            lay.addWidget(title)
            self.film.addWidget(cell)
            self.thumbs.append((thumb, i, pts, side))

    def view_frame(self, dt: int) -> None:
        """Show frame t + dt in the large view. Only the labelled frame (dt = 0) can be edited."""
        if self.current is None:
            return
        dt = int(np.clip(dt, -self.k, self.k))
        if dt == self.view_dt:
            return
        self.view_dt = dt
        m, s = int(self.c["margin"]), self.current["size"]
        for thumb, i, pts, side in self.thumbs:
            thumb.setPixmap(_thumb(self.grays[i], side, (m, s), pts, "#d33" if thumb.index == dt else None))
        self._redraw()

    def _legend(self) -> None:
        cats = self.lab["categories"]
        chip = lambda colour, text: f"<span style='background:#15171c;color:{colour}'>&nbsp;<b>{text}</b>&nbsp;</span>"   # bright colours need a dark chip
        states = " ".join(chip(STATE_COLOURS[i % len(STATE_COLOURS)], f"{i + 1} {c}") for i, c in enumerate(cats[:9]))
        self.legend.setText(f"{chip(KEPT, '○')} found by tracking   {chip(ADDED, '○')} added by you   "
                            f"{chip(BOX, '□')} label every nucleus centred inside.   Click a circle to remove it, a nucleus to add "
                            f"one. States: press the number over a circle: {states}. The number is shown next to the circle; "
                            f"no number = {cats[0] if cats else 'default'}.   ←/→ flip frames, ↑ back to the labelled frame. Wheel zooms, drag pans.")

    def _state_colour(self, state: str) -> str:
        cats = self.lab["categories"]
        return STATE_COLOURS[cats.index(state) % len(STATE_COLOURS)] if state in cats else "#ffffff"

    def _redraw(self) -> None:
        obs = self.current
        if obs is None:
            return
        m, s = int(self.c["margin"]), obs["size"]
        ox, oy = obs["x0"] - m, obs["y0"] - m
        self.canvas.set_image(self.grays[self.k + self.view_dt])
        groups = _marked([(m - .5, m - .5), (m - .5, m + s - .5), (m + s - .5, m + s - .5), (m + s - .5, m - .5), (m - .5, m - .5)], BOX, 1.5)
        texts = []
        default = self.lab["categories"][0] if self.lab["categories"] else ""
        if self.view_dt == 0:
            tracked = {(round(a, 1), round(b, 1)) for a, b in self.tracked}
            cats = self.lab["categories"]
            for x, y in self.context_pts:
                groups += _marked(_circle(x - ox, y - oy, RADIUS), CONTEXT, 1.5)
            for (x, y), st in zip(self.points, self.states):
                colour = KEPT if (round(x, 1), round(y, 1)) in tracked else ADDED
                groups += _marked(_circle(x - ox, y - oy, RADIUS), colour)
                if st != default:
                    label = str(cats.index(st) + 1) if st in cats else st[:4]
                    texts.append((x - ox - RADIUS - 5, y - oy + RADIUS - 1, label, self._state_colour(st)))   # just right of the circle
            if self.overlay and self.overlay.get("model_points"):
                for x, y in self.overlay["model_points"]:
                    groups += _marked(_square(x - ox, y - oy, RADIUS + 1.5), MODEL, 2.0)
            if self.overlay and self.overlay.get("kind") == "state":
                x, y = self.overlay["x"], self.overlay["y"]
                groups += _marked(_circle(x - ox, y - oy, RADIUS * 2.2), MODEL, 2.0)
                texts.append((x - ox + 6, y - oy + 6, f"model: {self.overlay['predicted']}", MODEL))
        else:
            # another frame: tracking's centres there, and your points of the labelled frame as faint marks to compare
            for x, y in self.tracked_at[self.view_dt]:
                groups += _marked(_circle(x - ox, y - oy, RADIUS), OTHER, 2.0)
            for x, y in self.points:
                groups += _marked([(x - ox - 2.5, y - oy), (x - ox + 2.5, y - oy)], CONTEXT, 2.0)
                groups += _marked([(x - ox, y - oy - 2.5), (x - ox, y - oy + 2.5)], CONTEXT, 2.0)
            texts.append((m + 2, m + 2, f"frame {obs['t'] + self.view_dt} ({self.view_dt:+d}): view only, ↑ to edit", BOX))
        self.canvas.set_paths(groups)
        self.canvas.set_texts(texts)

    # ------------------------------------------------------------------ editing
    def _to_image(self, px: float, py: float) -> tuple[float, float]:
        m = int(self.c["margin"])
        return self.current["x0"] - m + px, self.current["y0"] - m + py

    def _inside(self, x: float, y: float) -> bool:
        obs = self.current
        return obs["x0"] <= x < obs["x0"] + obs["size"] and obs["y0"] <= y < obs["y0"] + obs["size"]

    def _push(self) -> None:
        self.undo.append((copy.deepcopy(self.points), list(self.states)))

    def _clicked(self, px: float, py: float) -> None:
        if self.current is None:
            return
        if self.view_dt != 0:
            self.view_frame(0)          # a click on another frame brings the labelled frame back instead of editing
            return
        x, y = self._to_image(px, py)
        if self.points:
            d = [math.hypot(x - a, y - b) for a, b in self.points]
            i = int(np.argmin(d))
            if d[i] <= NEAR:
                self._push()
                self.points.pop(i)
                self.states.pop(i)
                self._redraw()
                return
        if not self._inside(x, y):
            return  # outside the box: context only
        self._push()
        self.points.append([round(x, 1), round(y, 1)])
        self.states.append(self.lab["categories"][0] if self.lab["categories"] else "")
        self._redraw()

    def state_key(self, n: int) -> None:
        """Key n: the point under the crosshair gets state n; with no point there, one is added with that state."""
        cats = self.lab["categories"]
        if self.current is None or n > len(cats) or self.hover is None:
            return
        if self.view_dt != 0:
            self.view_frame(0)
            return
        x, y = self._to_image(*self.hover)
        d = [math.hypot(x - a, y - b) for a, b in self.points]
        if d and min(d) <= STATE_NEAR:
            i = int(np.argmin(d))
            self._push()
            self.states[i] = cats[n - 1]
        elif self._inside(x, y):
            self._push()
            self.points.append([round(x, 1), round(y, 1)])
            self.states.append(cats[n - 1])
        else:
            return
        self._redraw()

    def undo_click(self) -> None:
        if self.undo:
            self.points, self.states = self.undo.pop()
            self._redraw()

    def clear_points(self) -> None:
        if self.current is not None:
            self._push()
            self.points, self.states = [], []
            self._redraw()

    def reset_points(self) -> None:
        if self.current is not None:
            self._push()
            self.points = copy.deepcopy(self.tracked)
            self.states = [self.lab["categories"][0] if self.lab["categories"] else ""] * len(self.points)
            self._redraw()

    def save(self) -> None:
        if self.current is None:
            return
        from .. import centres
        exp = self._ensure()
        if exp is None:
            return
        obs = self.current
        keep = [i for i, p in enumerate(self.points) if self._inside(*p)]
        inside = [self.points[i] for i in keep]
        states = [self.states[i] for i in keep]
        tracked = [p for p in self.tracked if self._inside(*p)]
        self.rows = centres.add_label(exp, self.rows, obs, inside, tracked, states)
        self.history = self.history[-20:]  # Back reaches the last 20 patches; their frames stay in memory
        self.history.append({"obs": obs, "frames": self._raw, "points": inside, "states": states})
        self.changed.emit()
        back_to = self.overlay_return if self.overlay and self.overlay.get("return_to") else None
        self.overlay = None
        if back_to is not None and back_to != 0:
            self.current = None            # opened from the dataset or the review: go back there (it refreshes)
            self.pages.setCurrentIndex(back_to)
            return
        self.next_patch()

    def skip(self) -> None:
        if self.current is not None:
            o = self.current
            self.skipped.add((o["position"], o["t"], o.get("cx", 0), o.get("cy", 0)))
        self.next_patch()

    def back(self) -> None:
        if not self.history:
            return
        from .. import centres
        exp = self._ensure()
        if exp is None:
            return
        last = self.history.pop()
        obs = last["obs"]
        self.rows = [r for r in self.rows if centres.patch_key(r) != centres.patch_key(obs)]
        centres.write_labels(exp, self.rows)
        self.overlay = None
        self._open(obs, last["frames"], last["points"], last["states"])

    def _show_counts(self) -> None:
        from .. import dataset
        s = dataset.summary(self._exp(), self.rows)
        states = ", ".join(f"{c} {n}" for c, n in s["categories"].items())
        self.counts.setText(f"{s['patches']} patches, {s['points']} centres (you added {s['added']}, removed {s['removed']}); "
                            f"states: {states}   (aim for 100+ patches, 30+ cells of each state)")

    # ------------------------------------------------------------------ opening a labelled patch
    def open_row(self, index: int | None, overlay: dict | None = None) -> None:
        """Open labelled patch `index` on the Label page to look at or correct. A state-only row (an earlier event
        label) opens as a full patch around its cell, pre-filled with tracking's centres: saving it labels them all."""
        from .. import centres
        if index is None:
            return
        exp = self._ensure()
        if exp is None or index >= len(self.rows):
            return
        r = self.rows[index]
        obs = {"position": r["position"], "t": int(r["t"]), "x0": int(r["x0"]), "y0": int(r["y0"]), "size": int(r["size"]),
               "source": r["source"], "change": r.get("change", 0.0), "cx": int(r["x0"]) + int(r["size"]) // 2, "cy": int(r["y0"]) + int(r["size"]) // 2}
        default = self.lab["categories"][0] if self.lab["categories"] else ""
        states = centres.point_states(r, default)
        points = copy.deepcopy(r["points"])
        try:
            if r.get("state_only"):
                (px, py), st = r["points"][0], states[0]
                size = int(self.c["patch"])
                obs["x0"], obs["y0"] = centres.box_for(px, py, size, self.reader.shape(r["position"]))
                obs["size"] = size
                points = self.sampler.tracked_points(obs)
                states = [default] * len(points)
                d = [math.hypot(px - a, py - b) for a, b in points]
                if d and min(d) <= 6:
                    states[int(np.argmin(d))] = st
                else:
                    points.append([px, py])
                    states.append(st)
            frames = self.reader.frames(r["position"], self._frame_range(obs["t"]))
        except (KeyError, OSError, ValueError) as exc:
            self.info.setText(f"Could not read that patch: {exc}")
            return
        self.overlay = overlay
        self.overlay_return = self.pages.currentIndex()
        if self.overlay is None:
            self.overlay = {"return_to": True, "text": "opened from the dataset"}
        self.pages.setCurrentIndex(0)
        self._open(obs, frames, points, states)

    def _page_changed(self, index: int) -> None:
        if index == 1:
            self.show_dataset(reload=True)
        elif index == 2:
            self.show_review()

    # ------------------------------------------------------------------ dataset page
    def show_dataset(self, reload: bool = False) -> None:
        from .. import dataset
        exp = self._exp()
        if not exp["paths"]["results"]:
            self.ds_summary.setText("Set the results folder on the Experiment page.")
            return
        if reload or not self.rows:
            self.rows = dataset.read_dataset(exp)
        cats = self.lab["categories"]
        mode = self.ds_mode.currentIndex()
        wanted = ["Everything", *cats] + (["Earlier event labels only"] if mode == 0 else [])
        current = self.ds_filter.currentText()
        if [self.ds_filter.itemText(i) for i in range(self.ds_filter.count())] != wanted:
            self.ds_filter.blockSignals(True)
            self.ds_filter.clear()
            self.ds_filter.addItems(wanted)
            if current in wanted:
                self.ds_filter.setCurrentText(current)
            self.ds_filter.blockSignals(False)
        pick = self.ds_filter.currentText()
        default = cats[0] if cats else ""
        from ..centres import point_states
        if mode == 0:
            items = [(i, r) for i, r in enumerate(self.rows)
                     if pick == "Everything" or (pick == "Earlier event labels only" and r.get("state_only"))
                     or pick in point_states(r, default)]
            jobs = [{"row": i, "position": r["position"], "t": int(r["t"]), "x0": int(r["x0"]), "y0": int(r["y0"]), "size": int(r["size"]),
                     "points": [(x, y, self._state_colour(s) if s != default else KEPT) for (x, y), s in zip(r["points"], point_states(r, default))],
                     "label": f"{r['position']} t{r['t']}" + (" (event label)" if r.get("state_only") else ""), "key": ("p", i, r.get("labelled", ""))}
                    for i, r in items]
            side = 110
        else:
            cells = sorted(dataset.cell_examples(exp, self.rows), key=lambda c: (cats.index(c["state"]) if c["state"] in cats else 99, c["position"], c["t"]))
            cells = [c for c in cells if pick == "Everything" or c["state"] == pick]
            half = 20
            jobs = [{"row": c["patch"], "position": c["position"], "t": c["t"], "x0": int(round(c["x"])) - half, "y0": int(round(c["y"])) - half,
                     "size": 2 * half, "points": [(c["x"], c["y"], self._state_colour(c["state"]))], "label": c["state"],
                     "key": ("c", c["patch"], c["point"], c["state"], self.rows[c["patch"]].get("labelled", ""))} for c in cells]
            side = 72
        s = dataset.summary(exp, self.rows)
        self.ds_summary.setText(f"{len(jobs)} shown. Dataset: {s['patches']} full patches, {s['state_only']} earlier event labels, "
                                f"{s['points']} labelled centres; " + ", ".join(f"{c} {n}" for c, n in s["categories"].items()))
        self.ds_list.clear()
        self.ds_list.setIconSize(QSize(side, side))
        self.ds_items = {}
        todo = []
        for j in jobs:
            item = QListWidgetItem(j["label"])
            item.setData(Qt.ItemDataRole.UserRole, j["row"])
            item.setSizeHint(QSize(side + 16, side + 30))
            if j["key"] in self.thumb_cache:
                item.setIcon(QIcon(self.thumb_cache[j["key"]]))
            else:
                todo.append(j)
            self.ds_list.addItem(item)
            self.ds_items[j["key"]] = (item, j, side)
        self._load_thumbs(todo)

    def _load_thumbs(self, jobs: list[dict]) -> None:
        """Cut thumbnails off the GUI thread, reading each frame once; icons are set as they arrive."""
        if self.thumb_worker is not None:
            self.thumb_worker.stop_flag.set()
        if not jobs:
            return
        exp = self._ensure()
        if exp is None:
            return
        reader = self.reader

        def job(on_event, stop):
            from .. import centres
            by_frame: dict = {}
            for j in jobs:
                by_frame.setdefault((j["position"], j["t"]), []).append(j)
            for (pos, t), js in by_frame.items():
                if stop.is_set():
                    return None
                try:
                    frame = reader.frame(pos, t)
                except (KeyError, OSError, ValueError):
                    continue
                for j in js:
                    m = 8 if j["size"] > 48 else 0
                    on_event({"key": j["key"], "gray": centres.cut(frame, max(0, j["x0"]), max(0, j["y0"]), j["size"], m, "constant"), "m": m})
            return None

        def arrived(e):
            item, j, side = self.ds_items.get(e["key"], (None, None, None))
            if item is None:
                return
            gray = _scale([e["gray"]])[0]
            x0, y0, m = max(0, j["x0"]), max(0, j["y0"]), e["m"]
            pts = [(x - x0 + m, y - y0 + m, c) for x, y, c in j["points"]]
            pix = _thumb(gray, side, (m, j["size"]) if m else None, pts)
            self.thumb_cache[e["key"]] = pix
            item.setIcon(QIcon(pix))

        self.thumb_worker = FunctionWorker(job, self)
        self.thumb_worker.event.connect(arrived)
        self.thumb_worker.start()

    # ------------------------------------------------------------------ review page
    def show_review(self) -> None:
        from .. import dataset
        exp = self._exp()
        if not exp["paths"]["results"]:
            return
        if self.review is None:
            self.review = dataset.read_review(exp)
        self.rows = dataset.read_dataset(exp) if not self.rows else self.rows
        self.rv_list.clear()
        if not self.review:
            self.rv_status.setText("No check yet. Check labels trains a model per fold (a few minutes) and judges every label.")
            return
        self.review_items = dataset.disagreements(self.review, self.rows)
        kind = {0: None, 1: "centres", 2: "state"}[self.rv_filter.currentIndex()]
        shown = 0
        for n, it in enumerate(self.review_items):
            if (kind and it["kind"] != kind) or (self.rv_hide.isChecked() and it["edited"]):
                continue
            item = QListWidgetItem(("✓ relabelled · " if it["edited"] else "") + it["text"])
            item.setData(Qt.ItemDataRole.UserRole, n)
            if it["patch"] is None:
                item.setForeground(QColor(CONTEXT))
            self.rv_list.addItem(item)
            shown += 1
        nc = sum(i["kind"] == "centres" for i in self.review_items)
        ns = len(self.review_items) - nc
        errors = "; ".join(f"{k}: {v}" for k, v in self.review.get("errors", {}).items())
        self.rv_status.setText(f"Checked {self.review['made']} with {self.review['folds']} folds: {nc} patches where the centres differ, "
                               f"{ns} cells where the state differs ({shown} shown)." + (f" Not checked: {errors}" if errors else ""))

    def _open_review_item(self, item: QListWidgetItem) -> None:
        it = self.review_items[item.data(Qt.ItemDataRole.UserRole)]
        if it["patch"] is None:
            self.rv_status.setText("That label has changed since the check, so it cannot be found. Check labels again.")
            return
        self.open_row(it["patch"], {**it, "return_to": True})

    # ------------------------------------------------------------------ training
    def _run(self, button: QPushButton, text: str, fn, done) -> None:
        exp = self._ensure()
        if exp is None:
            return
        button.setEnabled(False)
        self.report.setText(text)
        timestacks = self.get_timestacks() or None

        def job(on_event, stop):
            return fn(exp, on_event, timestacks)

        def progress(e):
            if e.get("kind") in ("patches", "crops"):
                self.report.setText(f"Reading {'frames' if e['kind'] == 'patches' else 'cell crops'}: {e['done']} of {e['total']}")
            elif e.get("kind") == "train":
                self.report.setText(f"Training ({e['phase']}): step {e['step']} of {e['steps']}, loss {e['loss']:.3f}")
            elif e.get("kind") == "classify":
                self.report.setText(f"Classified {e['done']} of {e['total']} positions")

        self.worker = FunctionWorker(job, self)
        self.worker.event.connect(progress)
        self.worker.finished_ok.connect(done)
        self.worker.failed.connect(lambda m: self.report.setText(m.splitlines()[0]))
        self.worker.finished.connect(lambda: button.setEnabled(True))
        self.worker.start()

    def train(self) -> None:
        from ..dataset import train_all
        self._run(self.train_button, "Reading patches…", train_all, lambda _: self._show_report())

    def check(self) -> None:
        from ..dataset import check_labels

        def done(review):
            self.review = review
            self._show_report()
            self.pages.setCurrentIndex(2)
            self.show_review()
        self._run(self.check_button, "Checking labels…", check_labels, done)

    def classify(self) -> None:
        from ..events import classify_positions
        self._run(self.classify_button, "Classifying…", lambda exp, ev, ts: classify_positions(exp, ev, None, ts),
                  lambda counts: self.report.setText(f"Wrote Events/<position>.csv for {len(counts)} positions, {sum(counts.values())} candidates."))

    def _show_report(self) -> None:
        from .. import centres, dataset
        exp = self.get_experiment()
        if not exp["paths"]["results"]:
            self.report.setText("")
            return
        exp = self._exp()
        report = centres.read_report(exp)
        states = dataset.read_state_report(exp)
        self.report_data = report
        self.use_button.setEnabled(bool(report))
        text = ""
        if report:
            def line(name, r):
                return f"<tr><td>{name}</td><td align=right>{r['f1']:.3f}</td><td align=right>{r['missed']}</td><td align=right>{r['extra']}</td></tr>"
            text += (f"<b>Seed finder</b>, trained {report['trained']} on {report['patches']} patches. On {report['holdout_patches']} held out:"
                     "<table><tr><td></td><td>F1</td><td>missed</td><td>extra</td></tr>"
                     + line("tracking now", report["tracking"]) + line(f"model (threshold {report['threshold']})", report["model"]) + "</table>")
        else:
            text += "<b>Seed finder</b>: not trained. Label at least 10 patches (100 or more is better).<br>"
        if states:
            text += f"<b>State classifier</b>, trained {states['trained']} on " + ", ".join(f"{c} {n}" for c, n in states["counts"].items()) + "."
            if "holdout_accuracy" in states:
                classes = states["classes"]
                rows = "".join(f"<tr><td>{c}</td>" + "".join(f"<td align=right>{v}</td>" for v in states["confusion"][i]) + "</tr>"
                               for i, c in enumerate(classes))
                text += (f" Held-out accuracy {states['holdout_accuracy']:.2f} on {states['holdout_n']} (rows true, columns predicted):"
                         "<table><tr><td></td>" + "".join(f"<td>{c[:6]}</td>" for c in classes) + f"</tr>{rows}</table>")
        else:
            text += "<br><b>State classifier</b>: not trained. Give some points a state other than the first."
        text += "<br>Saved models are trained on every label. Check them in a test run before tracking everything."
        self.report.setText(text)

    def _use(self) -> None:
        if self.report_data:
            self.use_model.emit(self.report_data["model_file"], float(self.report_data["threshold"]))
            self.report.setText(self.report.text() + "<br><b>Set: centroids.method = cnn.</b> Run a test run to compare.")


CellLabelTab = CentresTab
