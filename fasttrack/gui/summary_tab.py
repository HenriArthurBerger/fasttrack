"""Summary: routine figures drawn without asking (cell counts, nuclear size, every channel's primary measurement),
as a gallery. They are redrawn in the background after every run and whenever their settings change, and written
to <results>/Figures/Summary. Any one can be copied into the figure list to edit.
"""
from __future__ import annotations

import copy
import io
import json

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ..analysis.bands import BANDS
from ..analysis.summary import SUMMARY_KINDS, summary_specs
from ..experiment import normalise
from . import theme
from .worker import FunctionWorker

UNIT_TEXT = {"well": "wells (recommended)", "position": "positions", "cell": "none: pool all cells"}
FULL_DPI = 110


class SummaryCard(QFrame):
    open_requested = Signal(int)
    copy_requested = Signal(int)

    def __init__(self, index: int, title: str):
        super().__init__()
        self.index = index
        self.setObjectName("HomeCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        self.title = QLabel(title)
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: 600;")
        lay.addWidget(self.title)
        self.image = QLabel("Waiting…")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(220)
        self.image.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image.setToolTip("Click to open at full size")
        self.image.mousePressEvent = lambda e: self.open_requested.emit(self.index)
        theme.hint(self.image)
        lay.addWidget(self.image, 1)
        row = QHBoxLayout()
        self.note = QLabel("")
        self.note.setWordWrap(True)
        theme.hint(self.note)
        row.addWidget(self.note, 1)
        copy_button = QPushButton("Copy to my figures")
        copy_button.setToolTip("Add this figure to the figure list, where every setting can be changed")
        copy_button.clicked.connect(lambda: self.copy_requested.emit(self.index))
        row.addWidget(copy_button)
        lay.addLayout(row)
        self.full: bytes | None = None

    def show_image(self, thumb: bytes, full: bytes, note: str) -> None:
        pix = QPixmap()
        pix.loadFromData(thumb, "PNG")
        self.image.setPixmap(pix.scaled(380, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.full = full
        self.note.setText(note)

    def show_message(self, text: str) -> None:
        self.image.setPixmap(QPixmap())
        self.image.setText(text)


class SummaryPanel(QWidget):
    changed = Signal()
    copy_figure = Signal(dict)

    def __init__(self, get_experiment, cache_for):
        super().__init__()
        self.get_experiment = get_experiment
        self.cache_for = cache_for
        self.worker: FunctionWorker | None = None
        self.pending = False
        self.drawn_key: str | None = None
        self.specs: list[dict] = []
        self.cards: list[SummaryCard] = []
        self.warm = None            # starts the metric cache building; set by the analysis section
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("What to draw")
        form = QGridLayout(box)
        self.enabled = QCheckBox("Draw after every run, into the results folder")
        self.enabled.setToolTip("Also redrawn here whenever these settings or the experiment change.")
        form.addWidget(self.enabled, 0, 0, 1, 2)
        self.kinds = {}
        kinds_row = QHBoxLayout()
        for key, text in SUMMARY_KINDS.items():
            cb = QCheckBox({"cell_count": "Cell counts", "nuc_area": "Nuclear area", "timecourse": "Time courses",
                            "heatmap": "Population heatmaps", "distribution": "End-point distributions"}[key])
            cb.setToolTip(text)
            self.kinds[key] = cb
            kinds_row.addWidget(cb)
        kinds_row.addStretch(1)
        form.addLayout(kinds_row, 1, 0, 1, 4)
        form.addWidget(QLabel("Replicates"), 2, 0)
        self.unit = QComboBox()
        for key, text in UNIT_TEXT.items():
            self.unit.addItem(text, key)
        form.addWidget(self.unit, 2, 1)
        form.addWidget(QLabel("Error band"), 2, 2)
        self.band = QComboBox()
        for key, text in BANDS.items():
            self.band.addItem(text, key)
        form.addWidget(self.band, 2, 3)
        form.addWidget(QLabel("End point"), 3, 0)
        self.end = QSpinBox()
        self.end.setRange(1, 10000)
        self.end.setPrefix("last ")
        self.end.setSuffix(" frames")
        form.addWidget(self.end, 3, 1)
        hint = QLabel("Each channel is drawn as its <b>primary measurement</b>, set in the channel table on the "
                      "Experiment page (e.g. nuclear mean for PPARG, cytoplasm / nucleus for a KTR).")
        hint.setWordWrap(True)
        theme.hint(hint)
        form.addWidget(hint, 3, 2, 1, 2)
        for w in [self.enabled, *self.kinds.values()]:
            w.toggled.connect(self._edited)
        for w in (self.unit, self.band):
            w.currentIndexChanged.connect(self._edited)
        self.end.valueChanged.connect(self._edited)
        root.addWidget(box)

        bar = QHBoxLayout()
        self.draw_button = QPushButton("Draw now")
        theme.primary(self.draw_button)
        self.draw_button.clicked.connect(lambda: self.draw(force=True))
        self.folder_button = QPushButton("Open folder")
        self.folder_button.clicked.connect(self._open_folder)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        theme.hint(self.status)
        bar.addWidget(self.draw_button)
        bar.addWidget(self.folder_button)
        bar.addWidget(self.status, 1)
        root.addLayout(bar)

        self.gallery_host = QWidget()
        self.gallery = QGridLayout(self.gallery_host)
        self.gallery.setSpacing(12)
        self.gallery.setContentsMargins(0, 4, 0, 4)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(self.gallery_host)
        root.addWidget(area, 1)
        self._loading = False

    # ------------------------------------------------------------------ settings
    def load(self, summary: dict) -> None:
        self._loading = True
        self.enabled.setChecked(summary["enabled"])
        chosen = summary["kinds"] or list(SUMMARY_KINDS)
        for key, cb in self.kinds.items():
            cb.setChecked(key in chosen)
        self.unit.setCurrentIndex(max(0, self.unit.findData(summary["unit"])))
        self.band.setCurrentIndex(max(0, self.band.findData(summary["band"])))
        self.end.setValue(int(summary["end_frames"]))
        self.folder = summary["folder"]
        self._loading = False
        self.drawn_key = None

    def collect(self) -> dict:
        kinds = [k for k, cb in self.kinds.items() if cb.isChecked()]
        return {"enabled": self.enabled.isChecked(), "folder": getattr(self, "folder", "Figures/Summary"),
                "kinds": [] if len(kinds) == len(self.kinds) else kinds,
                "unit": self.unit.currentData(), "band": self.band.currentData(), "end_frames": self.end.value()}

    def _edited(self, *_) -> None:
        if self._loading:
            return
        self.changed.emit()
        if self.isVisible():
            self.draw()

    # ------------------------------------------------------------------ drawing
    def _experiment(self) -> dict:
        exp = copy.deepcopy(self.get_experiment())
        exp["figures"]["summary"] = self.collect()
        return normalise(exp)

    @staticmethod
    def _key(exp: dict) -> str:
        relevant = {k: exp[k] for k in ("channels", "time", "plate", "quality", "files")}
        relevant.update(summary=exp["figures"]["summary"], analysis=exp["analysis"]["min_track_frames"],
                        border=exp["analysis"]["drop_border"], palette=exp["figures"]["palette"])
        return json.dumps([relevant, exp["paths"]["results"]], sort_keys=True, default=str)

    def draw(self, force: bool = False, write: bool | None = None) -> None:
        """Redraw in the background if anything changed since the last drawing (or always with `force`).
        `write`: also write the files to the results folder (default: when drawing after a run is on)."""
        exp = self._experiment()
        if not exp["paths"]["results"]:
            self.status.setText("Set the results folder on the Experiment page.")
            return
        key = self._key(exp)
        if not force and key == self.drawn_key:
            return
        if self.worker is not None and self.worker.isRunning():
            self.pending = True
            self.worker.stop_flag.set()
            return
        write = exp["figures"]["summary"]["enabled"] if write is None else write
        self._start(exp, key, write)

    def after_run(self) -> None:
        """Tracking finished: update the metric cache, then redraw (writing the files if drawing after runs is on)."""
        self.drawn_key = None
        self._after_run = True
        if self.warm is not None and self.warm():
            return            # cache_ready → cache_built
        self.cache_built()

    def cache_built(self) -> None:
        if getattr(self, "_after_run", False):
            self._after_run = False
            if self.enabled.isChecked():
                self.draw(force=True, write=True)
        elif self.isVisible():
            self.draw()

    def _start(self, exp: dict, key: str, write: bool) -> None:
        from pathlib import Path

        from ..analysis.figures import is_empty, render, save_figure
        self.specs = summary_specs(exp)
        self._build_cards()
        if not self.specs:
            self.status.setText("Nothing to draw: tick at least one kind, and give the plate conditions with wells.")
            return
        cache = self.cache_for(exp)
        folder = Path(exp["paths"]["results"]) / exp["figures"]["summary"]["folder"]

        def job(on_event, stop):
            written = 0
            for i, spec in enumerate(self.specs):
                if stop.is_set():
                    return {"stopped": True, "written": written}
                try:
                    figure, table = render(exp, spec, cache=cache)
                except (KeyError, ValueError, OSError) as exc:
                    on_event({"index": i, "error": str(exc).strip("'\"")})
                    continue
                if is_empty(figure):
                    on_event({"index": i, "error": "No results for these conditions yet. Run tracking first."})
                    continue
                if write:
                    folder.mkdir(parents=True, exist_ok=True)
                    written += len(save_figure(exp, spec, figure, table, folder, spec["name"]))
                full = io.BytesIO()
                from ..plotting import style as pstyle
                with pstyle.context("screen", font=spec["plot"]["font_size"], line=spec["plot"]["line_width"]):
                    figure.savefig(full, format="png", dpi=FULL_DPI)    # drawn once; the card scales it down
                notes = "; ".join(getattr(figure, "_fasttrack_notes", [])[:1])
                on_event({"index": i, "thumb": full.getvalue(), "full": full.getvalue(), "note": notes})
            return {"stopped": False, "written": written}

        self.status.setText(f"Drawing {len(self.specs)} figures in the background…")
        self.draw_button.setEnabled(True)
        self.worker = FunctionWorker(job, self)
        self.worker.event.connect(self._figure_ready)
        self.worker.finished_ok.connect(lambda out: self._done(out, key, folder, write))
        self.worker.failed.connect(lambda m: self.status.setText(m.splitlines()[0]))
        self.worker.finished.connect(self._next)
        self.worker.start()

    def _build_cards(self) -> None:
        while self.gallery.count():
            w = self.gallery.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.cards = []
        cols = 3
        for i, spec in enumerate(self.specs):
            card = SummaryCard(i, spec["name"])
            card.open_requested.connect(self._open_large)
            card.copy_requested.connect(self._copy)
            self.gallery.addWidget(card, i // cols, i % cols)
            self.cards.append(card)
        for c in range(cols):
            self.gallery.setColumnStretch(c, 1)

    def _figure_ready(self, e: dict) -> None:
        if not 0 <= e["index"] < len(self.cards):
            return
        card = self.cards[e["index"]]
        if "error" in e:
            card.show_message(e["error"])
        else:
            card.show_image(e["thumb"], e["full"], e["note"])

    def _done(self, out: dict, key: str, folder, write: bool) -> None:
        import datetime as dt
        if out.get("stopped"):
            return
        self.drawn_key = key
        where = f" Wrote {out['written']} files to {folder}." if write else " Not written to the results folder (drawing after runs is off)."
        self.status.setText(f"Drawn at {dt.datetime.now():%H:%M}.{where}")

    def _next(self) -> None:
        if self.pending:
            self.pending = False
            self.draw(force=True)

    def _open_large(self, index: int) -> None:
        card = self.cards[index]
        if card.full is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self.specs[index]["name"])
        lay = QVBoxLayout(dialog)
        label = QLabel()
        pix = QPixmap()
        pix.loadFromData(card.full, "PNG")
        label.setPixmap(pix)
        area = QScrollArea()
        area.setWidget(label)
        lay.addWidget(area)
        dialog.resize(min(pix.width() + 40, 1400), min(pix.height() + 40, 950))
        dialog.show()
        self._dialog = dialog

    def _copy(self, index: int) -> None:
        spec = copy.deepcopy(self.specs[index])
        spec.pop("summary_key", None)
        self.copy_figure.emit(spec)

    def _open_folder(self) -> None:
        from pathlib import Path
        exp = self._experiment()
        if not exp["paths"]["results"]:
            return
        folder = Path(exp["paths"]["results"]) / exp["figures"]["summary"]["folder"]
        if folder.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        else:
            self.status.setText("Nothing written yet: press Draw now, or run tracking with drawing after runs on.")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.warm is not None and self.warm():
            self.status.setText("Loading the metric cache in the background; the figures follow…")
            return             # drawn when the cache is ready (cache_ready → draw)
        self.draw()
