"""Zoomable image canvas. Wheel zooms around the cursor, drag pans, click reports pixel coords.

With set_crosshair(True) the cursor is a thin full-view crosshair with a gap at its centre, so the
pixel being clicked stays visible (a hand or arrow cursor covers it)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView

TRACK_COLORS = ["#4cc9f0", "#f72585", "#b5e48c", "#ffd166", "#c77dff", "#ff9e6d", "#80ffdb", "#ff6b6b"]


def to_display(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    """16-bit or float frame -> uint8, without touching the source data."""
    return (np.clip((frame.astype(np.float32) - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)


def outline_mask(labels: np.ndarray) -> np.ndarray:
    """Pixels on a nuclear boundary, including borders shared by touching nuclei."""
    nuclear = np.maximum(labels, 0)
    edge = np.zeros(labels.shape, bool)
    edge[:-1, :] |= nuclear[:-1, :] != nuclear[1:, :]
    edge[:, :-1] |= nuclear[:, :-1] != nuclear[:, 1:]
    return edge & (nuclear > 0)


class ImageCanvas(QGraphicsView):
    pixel_clicked = Signal(float, float)  # (x = first array axis, y = second array axis)
    pixel_hovered = Signal(float, float)
    zoomed = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setBackgroundBrush(QColor("#111318"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self._pixmap = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap)
        self._tracks: list[QGraphicsPathItem] = []
        self._texts: list[QGraphicsSimpleTextItem] = []
        self._user_zoomed = False
        self._press = None
        self._cross = False
        self._cursor = None          # viewport position of the crosshair, None when the mouse is outside
        self._pan_from = None

    def set_crosshair(self, on: bool, colour: str = "#ffd166") -> None:
        """Draw a crosshair instead of the hand cursor; left drag still pans."""
        self._cross = on
        self._cross_colour = QColor(colour)
        self.setDragMode(QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.BlankCursor if on else Qt.CursorShape.OpenHandCursor)
        # the crosshair spans the view, so repaint all of it, not just what changed in the scene
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate if on else QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.viewport().update()

    def drawForeground(self, painter: QPainter, rect) -> None:
        super().drawForeground(painter, rect)
        if not self._cross or self._cursor is None or self._pan_from is not None:
            return
        painter.save()
        painter.resetTransform()
        x, y, gap = int(self._cursor.x()), int(self._cursor.y()), 5
        w, h = self.viewport().width(), self.viewport().height()
        for colour, width in ((QColor(0, 0, 0, 150), 3), (self._cross_colour, 1)):   # dark outline: visible on bright nuclei too
            pen = QPen(colour, width)
            painter.setPen(pen)
            painter.drawLine(0, y, x - gap, y)
            painter.drawLine(x + gap, y, w, y)
            painter.drawLine(x, 0, x, y - gap)
            painter.drawLine(x, y + gap, x, h)
        painter.restore()

    def leaveEvent(self, event) -> None:
        self._cursor = None
        self.viewport().update()
        super().leaveEvent(event)

    def set_texts(self, items: list[tuple[float, float, str, str]]) -> None:
        """Small labels at (x, y) array coordinates: (x, y, text, colour). They keep their size when zooming."""
        for item in self._texts:
            self.scene().removeItem(item)
        self._texts.clear()
        for x, y, text, colour in items:
            item = QGraphicsSimpleTextItem(text)
            item.setBrush(QColor(colour))
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setPos(y + 3, x + 3)
            self.scene().addItem(item)
            self._texts.append(item)

    def set_image(self, gray: np.ndarray, overlay: np.ndarray | None = None, color=(255, 80, 80)) -> None:
        rgb = np.dstack([gray, gray, gray])
        if overlay is not None:
            rgb[overlay] = color
        self.set_rgb(rgb)

    def set_rgb(self, rgb: np.ndarray) -> None:
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._pixmap.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        if not self._user_zoomed:
            self.reset_view()

    def user_zoomed(self) -> None:
        self._user_zoomed = True

    def reset_view(self) -> None:
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_paths(self, groups: list[tuple[list[tuple[float, float]], str, float]]) -> None:
        """groups: (points as (x, y) in array coordinates, colour, pen width)."""
        for item in self._tracks:
            self.scene().removeItem(item)
        self._tracks.clear()
        for points, colour, width in groups:
            if not points:
                continue
            path = QPainterPath()
            path.moveTo(points[0][1], points[0][0])
            for x, y in points[1:]:
                path.lineTo(y, x)
            item = QGraphicsPathItem(path)
            pen = QPen(QColor(colour), width)
            pen.setCosmetic(True)
            item.setPen(pen)
            self.scene().addItem(item)
            self._tracks.append(item)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.reset_view()

    def wheelEvent(self, event) -> None:
        self._user_zoomed = True
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self.zoomed.emit()

    def mousePressEvent(self, event) -> None:
        self._press = event.position()
        if self._cross and event.button() == Qt.MouseButton.LeftButton:
            self._pan_from = None
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if not self._cross:
            super().mouseReleaseEvent(event)
        if self._press is not None and (event.position() - self._press).manhattanLength() < 4:
            p: QPointF = self.mapToScene(event.position().toPoint())
            self.pixel_clicked.emit(p.y(), p.x())
        self._press = None
        self._pan_from = None
        self.viewport().update()

    def mouseMoveEvent(self, event) -> None:
        p: QPointF = self.mapToScene(event.position().toPoint())
        self.pixel_hovered.emit(p.y(), p.x())
        if self._cross:
            self._cursor = event.position()
            if self._press is not None and event.buttons() & Qt.MouseButton.LeftButton:
                if self._pan_from is None and (event.position() - self._press).manhattanLength() >= 4:
                    self._pan_from = self._press
                if self._pan_from is not None:     # pan by hand: the drag modes would bring the hand cursor back
                    d = event.position() - self._pan_from
                    self._pan_from = event.position()
                    self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
                    self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            self.viewport().update()
            return
        super().mouseMoveEvent(event)
