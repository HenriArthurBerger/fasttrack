"""Plate: conditions, their colours and wells, and which wells or positions to leave out.

Drag across wells (or Ctrl-click) to select them; what happens to the selection depends on
the mode: assign the chosen condition, exclude the wells, or clear them. Everything here is
saved in the experiment file's `plate` section.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QColorDialog, QComboBox, QFormLayout, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsSimpleTextItem, QGraphicsView, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QSplitter, QVBoxLayout, QWidget)

from ..experiment import PALETTE, normalise_well, plate_shape, well_sort_key

EMPTY = QColor(238, 238, 238)


class WellItem(QGraphicsRectItem):
    def __init__(self, well_id: str, size: float):
        super().__init__(0, 0, size, size)
        self.well_id = well_id
        self.condition: str | None = None
        self.color = EMPTY
        self.excluded = False
        self.found = True
        self.selected = False
        self.label = QGraphicsSimpleTextItem(well_id, self)
        font = QFont()
        font.setPointSize(7)
        self.label.setFont(font)
        r = self.label.boundingRect()
        self.label.setPos((size - r.width()) / 2, (size - r.height()) / 2)
        self.restyle()

    def restyle(self) -> None:
        color = QColor(self.color)
        if self.excluded:
            color = QColor(200, 200, 200)
        self.setBrush(QBrush(color, Qt.BrushStyle.BDiagPattern if self.excluded else Qt.BrushStyle.SolidPattern))
        if self.selected:
            self.setPen(QPen(QColor(220, 30, 30), 3))
        elif self.condition:
            self.setPen(QPen(QColor(40, 40, 40), 1.5))
        else:
            self.setPen(QPen(QColor(150, 150, 150), 1))
        self.label.setBrush(QBrush(QColor(30, 30, 30) if self.found else QColor(160, 160, 160)))
        tip = self.well_id + (f": {self.condition}" if self.condition else "") + (" (excluded)" if self.excluded else "")
        self.setToolTip(tip + ("" if self.found else " (no images found)"))


class PlateView(QGraphicsView):
    """A plate of wells. Drag a rectangle or Ctrl-click to select; `selection_done` fires on release."""

    selection_done = Signal(list)

    def __init__(self, fmt: int = 96):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.wells: dict[str, WellItem] = {}
        self._start = None
        self._band = None
        self._base: set[str] = set()
        self.build(fmt)

    def build(self, fmt: int) -> None:
        scene = self.scene()
        scene.clear()
        self.wells = {}
        rows, cols = plate_shape(fmt)
        size = 34 if cols <= 12 else 18
        step = size + 5
        font = QFont()
        font.setBold(True)
        for r in range(rows):
            label = scene.addSimpleText(chr(65 + r), font)
            label.setPos(4, 30 + r * step + size / 2 - 8)
        for c in range(cols):
            label = scene.addSimpleText(str(c + 1), font)
            label.setPos(30 + c * step + size / 2 - 6, 4)
        for r in range(rows):
            for c in range(cols):
                well = WellItem(f"{chr(65 + r)}{c + 1:02d}", size)
                well.setPos(30 + c * step, 30 + r * step)
                scene.addItem(well)
                self.wells[well.well_id] = well
        scene.setSceneRect(-5, -5, 40 + cols * step, 40 + rows * step)

    def resizeEvent(self, event) -> None:
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(event)

    def _select(self, ids: set[str]) -> None:
        for wid, well in self.wells.items():
            if well.selected != (wid in ids):
                well.selected = wid in ids
                well.restyle()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._start = self.mapToScene(event.position().toPoint())
        additive = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self._base = {w for w, item in self.wells.items() if item.selected} if additive else set()
        self._update(self._start)

    def mouseMoveEvent(self, event) -> None:
        if self._start is not None:
            self._update(self.mapToScene(event.position().toPoint()))

    def _update(self, pos) -> None:
        rect = QRectF(self._start, pos).normalized()
        if self._band is None:
            self._band = self.scene().addRect(rect, QPen(QColor(0, 100, 255), 1, Qt.PenStyle.DashLine), QBrush(QColor(0, 100, 255, 30)))
        self._band.setRect(rect)
        hit = {w for w, item in self.wells.items() if rect.adjusted(-1, -1, 1, 1).intersects(item.sceneBoundingRect())}
        self._select((self._base ^ hit) if self._base else hit)

    def mouseReleaseEvent(self, event) -> None:
        if self._start is None:
            return
        self._start = None
        if self._band is not None:
            self.scene().removeItem(self._band)
            self._band = None
        ids = sorted((w for w, item in self.wells.items() if item.selected), key=well_sort_key)
        self.selection_done.emit(ids)

    def clear_selection(self) -> None:
        self._select(set())


class PlateLayoutTab(QWidget):
    changed = Signal()
    condition_renamed = Signal(str, str)
    condition_removed = Signal(str)

    def __init__(self):
        super().__init__()
        self.colors: dict[str, str] = {}      # condition -> hex, in list order
        self.off_plate: dict[str, list[str]] = {}  # wells not on this plate format, kept so nothing is lost
        self.notes: dict[str, str] = {}
        self.fmt = 96
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split)

        side = QWidget()
        side.setMaximumWidth(380)
        col = QVBoxLayout(side)
        box = QGroupBox("Conditions")
        lay = QVBoxLayout(box)
        add_row = QHBoxLayout()
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("New condition, e.g. DMI MekI")
        self.new_name.returnPressed.connect(self.add_condition)
        add = QPushButton("Add")
        add.clicked.connect(self.add_condition)
        add_row.addWidget(self.new_name, 1)
        add_row.addWidget(add)
        lay.addLayout(add_row)
        self.list = QListWidget()
        self.list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.list.model().rowsMoved.connect(lambda *_: self.changed.emit())
        self.list.itemDoubleClicked.connect(lambda _: self.rename_condition())
        self.list.currentRowChanged.connect(lambda _: self.assign_mode.setChecked(True))
        lay.addWidget(self.list, 1)
        buttons = QHBoxLayout()
        for label, slot in (("Rename", self.rename_condition), ("Colour…", self.recolor_condition), ("Remove", self.remove_condition)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        lay.addLayout(buttons)
        col.addWidget(box, 1)

        mode_box = QGroupBox("Dragging over wells will")
        mode_lay = QVBoxLayout(mode_box)
        self.assign_mode = QRadioButton("assign the selected condition")
        self.exclude_mode = QRadioButton("exclude the wells (toggle)")
        self.clear_mode = QRadioButton("clear the wells")
        self.assign_mode.setChecked(True)
        self.modes = QButtonGroup(self)
        for b in (self.assign_mode, self.exclude_mode, self.clear_mode):
            self.modes.addButton(b)
            mode_lay.addWidget(b)
        col.addWidget(mode_box)

        what = QGroupBox("What to process")
        form = QFormLayout(what)
        self.process = QComboBox()
        self.process.addItem("Only wells with a condition", "assigned")
        self.process.addItem("Every well with images", "all")
        self.process.currentIndexChanged.connect(lambda _: self.changed.emit())
        form.addRow("Wells", self.process)
        self.format = QComboBox()
        for f in (6, 12, 24, 48, 96, 384):
            self.format.addItem(f"{f}-well", f)
        self.format.setCurrentIndex(4)
        self.format.currentIndexChanged.connect(self._format_changed)
        form.addRow("Plate", self.format)
        self.excluded_positions = QLineEdit()
        self.excluded_positions.setPlaceholderText("e.g. C05_xy2, D03_xy1")
        self.excluded_positions.setToolTip("Single positions to leave out, as timestack ids.")
        self.excluded_positions.editingFinished.connect(self.changed.emit)
        form.addRow("Excluded positions", self.excluded_positions)
        col.addWidget(what)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        col.addWidget(self.summary)
        split.addWidget(side)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.view = PlateView(self.fmt)
        self.view.selection_done.connect(self.apply_selection)
        rl.addWidget(self.view, 1)
        hint = QLabel("Drag over wells or Ctrl-click to select. Grey well names have no images in the data folder (after a scan). "
                      "Hatched wells are excluded. Drag conditions in the list to reorder them; the order is used in figures.")
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")
        rl.addWidget(hint)
        split.addWidget(right)
        split.setSizes([340, 800])

    # ------------------------------------------------------------------ conditions
    def _current(self) -> str | None:
        item = self.list.currentItem()
        return item.text() if item else None

    def _add_item(self, name: str, color: str) -> None:
        self.colors[name] = color
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.DecorationRole, QColor(color))
        self.list.addItem(item)

    def add_condition(self) -> None:
        name = self.new_name.text().strip()
        if not name:
            return
        if name in self.colors:
            QMessageBox.information(self, "Already there", f"There is already a condition called {name}.")
            return
        self._add_item(name, PALETTE[len(self.colors) % len(PALETTE)])
        self.list.setCurrentRow(self.list.count() - 1)
        self.new_name.clear()
        self._refresh()

    def rename_condition(self) -> None:
        old = self._current()
        if old is None:
            return
        new, ok = QInputDialog.getText(self, "Rename condition", "Name", text=old)
        new = new.strip()
        if not ok or not new or new == old:
            return
        if new in self.colors:
            QMessageBox.information(self, "Already there", f"There is already a condition called {new}.")
            return
        self.colors = {(new if k == old else k): v for k, v in self.colors.items()}
        self.notes[new] = self.notes.pop(old, "")
        self.list.currentItem().setText(new)
        self.condition_renamed.emit(old, new)
        for well in self.view.wells.values():
            if well.condition == old:
                well.condition = new
                well.restyle()
        self._refresh()

    def recolor_condition(self) -> None:
        name = self._current()
        if name is None:
            return
        color = QColorDialog.getColor(QColor(self.colors[name]), self, f"Colour for {name}")
        if not color.isValid():
            return
        self.colors[name] = color.name()
        self.list.currentItem().setData(Qt.ItemDataRole.DecorationRole, color)
        for well in self.view.wells.values():
            if well.condition == name:
                well.color = color
                well.restyle()
        self._refresh()

    def remove_condition(self) -> None:
        name = self._current()
        if name is None:
            return
        if QMessageBox.question(self, "Remove condition", f"Remove {name} and clear its wells?") != QMessageBox.StandardButton.Yes:
            return
        for well in self.view.wells.values():
            if well.condition == name:
                well.condition, well.color = None, EMPTY
                well.restyle()
        del self.colors[name]
        self.list.takeItem(self.list.currentRow())
        self.condition_removed.emit(name)
        self._refresh()

    # ------------------------------------------------------------------ wells
    def apply_selection(self, ids: list[str]) -> None:
        if not ids:
            return
        if self.assign_mode.isChecked():
            name = self._current()
            if name is None:
                self.summary.setText("Select or add a condition first.")
                return
            for wid in ids:
                well = self.view.wells[wid]
                well.condition, well.color = name, QColor(self.colors[name])
        elif self.exclude_mode.isChecked():
            turn_on = not all(self.view.wells[w].excluded for w in ids)
            for wid in ids:
                self.view.wells[wid].excluded = turn_on
        else:
            for wid in ids:
                well = self.view.wells[wid]
                well.condition, well.color, well.excluded = None, EMPTY, False
        self.view.clear_selection()
        self._refresh()

    def set_found_wells(self, wells: set[str] | None) -> None:
        for wid, well in self.view.wells.items():
            well.found = wells is None or wid in wells
            well.restyle()

    def _format_changed(self) -> None:
        plate = self.collect()
        self.fmt = self.format.currentData()
        self.view.build(self.fmt)
        self._apply_wells(plate)
        self._refresh()

    def _refresh(self) -> None:
        counts = {name: 0 for name in self.colors}
        for well in self.view.wells.values():
            if well.condition in counts:
                counts[well.condition] += 1
        excluded = sum(w.excluded for w in self.view.wells.values())
        parts = [f"{name}: {n} well{'s' if n != 1 else ''}" for name, n in counts.items()]
        text = "; ".join(parts) + (f". {excluded} excluded." if excluded else "")
        outside = sorted({w for ws in self.off_plate.values() for w in ws})
        if outside:
            text += (f"<br><span style='color:#c00'><b>Error:</b> {', '.join(outside)} do not exist on a {self.fmt}-well plate. "
                     "Fix the wells or the plate format before running.</span>")
        self.summary.setText(text)
        self.changed.emit()

    # ------------------------------------------------------------------ experiment file
    def load(self, plate: dict) -> None:
        self.list.clear()
        self.colors, self.notes = {}, {}
        self.fmt = int(plate["format"])
        self.format.blockSignals(True)
        self.format.setCurrentIndex(max(0, self.format.findData(self.fmt)))
        self.format.blockSignals(False)
        self.view.build(self.fmt)
        for cond in plate["conditions"]:
            self._add_item(cond["name"], cond["color"])
            self.notes[cond["name"]] = cond.get("notes", "")
        self._apply_wells(plate)
        self.process.setCurrentIndex(max(0, self.process.findData(plate["process"])))
        self.excluded_positions.setText(", ".join(plate["excluded_positions"]))
        if self.list.count():
            self.list.setCurrentRow(0)
        self._refresh()

    def _apply_wells(self, plate: dict) -> None:
        self.off_plate = {}
        for cond in plate["conditions"]:
            for wid in cond["wells"]:
                if wid not in self.view.wells:
                    self.off_plate.setdefault(cond["name"], []).append(wid)
                if wid in self.view.wells:
                    well = self.view.wells[wid]
                    well.condition, well.color = cond["name"], QColor(cond["color"])
                    well.restyle()
        for wid in plate["excluded_wells"]:
            if wid in self.view.wells:
                self.view.wells[wid].excluded = True
                self.view.wells[wid].restyle()

    def collect(self) -> dict:
        order = [self.list.item(i).text() for i in range(self.list.count())]
        wells = {name: [] for name in order}
        for wid, well in self.view.wells.items():
            if well.condition in wells:
                wells[well.condition].append(wid)
        for name, extra in self.off_plate.items():
            if name in wells:
                wells[name] += extra
        positions = [p.strip() for p in self.excluded_positions.text().replace(";", ",").split(",") if p.strip()]
        return {
            "format": self.fmt,
            "process": self.process.currentData(),
            "conditions": [{"name": n, "color": self.colors[n], "wells": sorted(wells[n], key=well_sort_key), "notes": self.notes.get(n, "")}
                           for n in order],
            "excluded_wells": sorted((w for w, item in self.view.wells.items() if item.excluded), key=well_sort_key),
            "excluded_positions": positions,
        }


__all__ = ["PlateLayoutTab", "PlateView", "normalise_well"]
