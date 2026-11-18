"""The window's frame: a sidebar that lists every page under the workflow's five stages, a header on every page
(title, what the page is for, How to use, Next), the Home page and the help window. See docs/UI_WORKFLOW.md.

`Navigator` replaces the old tab widget and keeps its interface (count, setCurrentIndex, setCurrentWidget,
currentWidget, widget, tabText, indexOf, currentChanged), so code and tests that switched tabs still work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QStyle, QStyledItemDelegate,
                               QTextBrowser, QVBoxLayout, QWidget)

from . import theme

KIND, KEY, TITLE, STATUS, STATE, OPTIONAL = (Qt.ItemDataRole.UserRole + i for i in range(6))


@dataclass
class PageInfo:
    key: str
    title: str
    subtitle: str
    stage: str = ""
    optional: bool = False
    widget: QWidget | None = None
    page: QWidget | None = None
    item: QListWidgetItem | None = None
    next_key: str | None = None
    extra: dict = field(default_factory=dict)


class NavDelegate(QStyledItemDelegate):
    """Sidebar rows: stage headings, and pages with a status line and a dot (done, needs attention, not started)."""

    def sizeHint(self, option, index) -> QSize:
        return QSize(200, 34 if index.data(KIND) == "stage" else 46)

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = option.rect
        font = QFont(option.font)
        if index.data(KIND) == "stage":
            font.setPointSizeF(font.pointSizeF() * 0.82)
            font.setWeight(QFont.Weight.DemiBold)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
            painter.setFont(font)
            painter.setPen(QColor(theme.SIDEBAR_MUTED))
            painter.drawText(r.adjusted(18, 10, -8, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             index.data(TITLE).upper())
            painter.restore()
            return
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        box = QRectF(r.adjusted(8, 2, -8, -2))
        if selected or hover:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.SIDEBAR_ACTIVE if selected else theme.SIDEBAR_HOVER))
            painter.drawRoundedRect(box, 6, 6)
        if selected:
            painter.setBrush(QColor(theme.ACCENT))
            painter.drawRoundedRect(QRectF(box.left(), box.top() + 8, 3, box.height() - 16), 1.5, 1.5)
        state = index.data(STATE) or "todo"
        dot = QColor(theme.STATE_COLORS.get(state, theme.IDLE))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dot if state not in ("todo", "optional") else Qt.BrushStyle.NoBrush)
        if state in ("todo", "optional"):
            painter.setPen(QColor(theme.SIDEBAR_MUTED))
        cy = box.top() + 15
        painter.drawEllipse(QRectF(box.left() + 13, cy - 4, 8, 8))
        font.setWeight(QFont.Weight.DemiBold if selected else QFont.Weight.Medium)
        painter.setFont(font)
        painter.setPen(QColor("white" if selected else theme.SIDEBAR_TEXT))
        title_rect = QRectF(box.left() + 30, box.top() + 4, box.width() - 36, 20)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, index.data(TITLE))
        status = index.data(STATUS) or ""
        if index.data(OPTIONAL):
            status = f"optional · {status}" if status else "optional"
        small = QFont(option.font)
        small.setPointSizeF(small.pointSizeF() * 0.84)
        painter.setFont(small)
        painter.setPen(QColor(theme.SIDEBAR_MUTED if state != "warn" else "#E6A23C"))
        sub = QRectF(box.left() + 30, box.top() + 22, box.width() - 36, 16)
        painter.drawText(sub, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(status, Qt.TextElideMode.ElideRight, int(sub.width())))
        painter.restore()


class Navigator(QWidget):
    """Sidebar plus page stack. Keeps the tab widget's interface for the pages' content widgets."""
    currentChanged = Signal(int)
    help_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pages: list[PageInfo] = []
        self.by_key: dict[str, PageInfo] = {}
        self._stage = ""
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(236)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(0, 14, 0, 12)
        side.setSpacing(0)
        self.header = QVBoxLayout()
        self.header.setContentsMargins(18, 0, 14, 10)
        self.header.setSpacing(2)
        side.addLayout(self.header)
        self.nav = QListWidget()
        self.nav.setObjectName("Nav")
        self.nav.setItemDelegate(NavDelegate(self.nav))
        self.nav.setMouseTracking(True)
        self.nav.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.nav.currentItemChanged.connect(self._item_changed)
        side.addWidget(self.nav, 1)
        self.footer = QVBoxLayout()
        self.footer.setContentsMargins(18, 8, 14, 0)
        side.addLayout(self.footer)
        lay.addWidget(self.sidebar)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        QShortcut(QKeySequence(Qt.Key.Key_F1), self, activated=lambda: self.help_requested.emit(self.current_key()))

    # ----------------------------------------------------------- building
    def add_stage(self, title: str) -> None:
        self._stage = title
        item = QListWidgetItem()
        item.setData(KIND, "stage")
        item.setData(TITLE, title)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        self.nav.addItem(item)

    def add_page(self, key: str, widget: QWidget, title: str, subtitle: str, optional: bool = False,
                 header: bool = True, scroll: bool = True) -> PageInfo:
        info = PageInfo(key, title, subtitle, self._stage, optional, widget)
        info.page = PageFrame(info, self, header=header, scroll=scroll)
        self.stack.addWidget(info.page)
        item = QListWidgetItem()
        item.setData(KIND, "page")
        item.setData(KEY, key)
        item.setData(TITLE, title)
        item.setData(OPTIONAL, optional)
        item.setData(STATE, "optional" if optional else "todo")
        self.nav.addItem(item)
        info.item = item
        self.pages.append(info)
        self.by_key[key] = info
        return info

    def finish(self) -> None:
        """Link every page to the next required step, and fill in the Next buttons."""
        steps = [p for p in self.pages if not p.optional and p.stage]
        for p in self.pages:
            later = [s for s in steps if self.pages.index(s) > self.pages.index(p)]
            p.next_key = later[0].key if later else None
            p.page.set_next(self.by_key[p.next_key].title if p.next_key else None)
        if self.nav.currentRow() < 0 and self.pages:
            self.nav.setCurrentItem(self.pages[0].item)

    def set_status(self, key: str, text: str, state: str | None = None) -> None:
        info = self.by_key.get(key)
        if info is None:
            return
        info.item.setData(STATUS, text)
        if state is not None:
            info.item.setData(STATE, state)

    # ----------------------------------------------------------- tab-widget interface
    def count(self) -> int:
        return len(self.pages)

    def widget(self, i: int) -> QWidget:
        return self.pages[i].widget

    def tabText(self, i: int) -> str:
        return self.pages[i].title

    def indexOf(self, w: QWidget) -> int:
        return next((i for i, p in enumerate(self.pages) if p.widget is w), -1)

    def currentIndex(self) -> int:
        page = self.stack.currentWidget()
        return next((i for i, p in enumerate(self.pages) if p.page is page), -1)

    def currentWidget(self) -> QWidget | None:
        i = self.currentIndex()
        return self.pages[i].widget if i >= 0 else None

    def current_key(self) -> str:
        i = self.currentIndex()
        return self.pages[i].key if i >= 0 else ""

    def setCurrentIndex(self, i: int) -> None:
        if 0 <= i < len(self.pages):
            self.nav.setCurrentItem(self.pages[i].item)

    def setCurrentWidget(self, w: QWidget) -> None:
        self.setCurrentIndex(self.indexOf(w))

    def go(self, key: str) -> None:
        if key in self.by_key:
            self.nav.setCurrentItem(self.by_key[key].item)

    def _item_changed(self, item: QListWidgetItem | None, _prev=None) -> None:
        if item is None or item.data(KIND) != "page":
            return
        info = self.by_key[item.data(KEY)]
        self.stack.setCurrentWidget(info.page)
        self.currentChanged.emit(self.pages.index(info))


class PageFrame(QWidget):
    """A page: the header (stage, title, what it is for, How to use, Next) above the page's own widget."""

    def __init__(self, info: PageInfo, nav: Navigator, header: bool = True, scroll: bool = True):
        super().__init__()
        self.info, self.nav = info, nav
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.next_button = None
        if header:
            bar = QFrame()
            bar.setObjectName("PageHeader")
            row = QHBoxLayout(bar)
            row.setContentsMargins(24, 12, 20, 12)
            text = QVBoxLayout()
            text.setSpacing(1)
            if info.stage:
                kicker = QLabel(info.stage.upper() + ("  ·  OPTIONAL" if info.optional else ""))
                kicker.setProperty("role", "kicker")
                text.addWidget(kicker)
            title = QLabel(info.title)
            title.setProperty("role", "pagetitle")
            text.addWidget(title)
            sub = QLabel(info.subtitle)
            sub.setProperty("role", "pagesub")
            sub.setWordWrap(True)
            text.addWidget(sub)
            row.addLayout(text, 1)
            help_button = QPushButton("How to use")
            help_button.setProperty("role", "help")
            help_button.setToolTip("A guide to this page: what it is for, every control, and what is saved (F1)")
            help_button.clicked.connect(lambda: nav.help_requested.emit(info.key))
            row.addWidget(help_button, 0, Qt.AlignmentFlag.AlignVCenter)
            self.next_button = QPushButton("")
            self.next_button.setProperty("role", "ghost")
            self.next_button.clicked.connect(lambda: info.next_key and nav.go(info.next_key))
            row.addWidget(self.next_button, 0, Qt.AlignmentFlag.AlignVCenter)
            lay.addWidget(bar)
        body = QWidget()
        inner = QVBoxLayout(body)
        inner.setContentsMargins(16, 12, 16, 12)
        inner.addWidget(info.widget)
        info.widget.show()               # a widget taken out of a tab widget is still marked hidden
        if scroll:
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setWidget(body)
            lay.addWidget(area, 1)
        else:
            lay.addWidget(body, 1)

    def set_next(self, title: str | None) -> None:
        if self.next_button is None:
            return
        self.next_button.setVisible(bool(title))
        self.next_button.setText(f"Next: {title}  →" if title else "")


class HelpDialog(QDialog):
    """One window, reused: the guide of the page asked for."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("How to use")
        self.resize(760, 820)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        head = QFrame()
        head.setObjectName("PageHeader")
        row = QHBoxLayout(head)
        row.setContentsMargins(24, 14, 16, 14)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.kicker = QLabel("HOW TO USE")
        self.kicker.setProperty("role", "kicker")
        self.title = QLabel("")
        self.title.setProperty("role", "pagetitle")
        col.addWidget(self.kicker)
        col.addWidget(self.title)
        row.addLayout(col, 1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        lay.addWidget(head)
        self.text = QTextBrowser()
        self.text.setObjectName("Help")
        self.text.setOpenExternalLinks(True)
        self.text.document().setDocumentMargin(18)
        lay.addWidget(self.text, 1)

    def show_page(self, title: str, stage: str, markdown: str) -> None:
        from .help_text import to_html
        self.kicker.setText(f"HOW TO USE  ·  {stage.upper()}" if stage else "HOW TO USE")
        self.title.setText(title)
        self.setWindowTitle(f"How to use: {title}")
        self.text.setHtml(to_html(markdown))
        self.text.verticalScrollBar().setValue(0)
        self.show()
        self.raise_()
        self.activateWindow()


STAGES = [
    ("1 · Set up", "What is the data, and what is each well?", "Name the channels, set the folders and the frame "
     "range, mark events, and assign conditions to wells.", "experiment"),
    ("2 · Check & tune", "Are the images usable, and does tracking work?", "Flag bad positions, try the pipeline on one "
     "position, adjust parameters until segmentation and seeds look right.", "quality"),
    ("3 · Track", "Track every position", "Run the whole plate in the background, then look at tracks and "
     "single-cell timecourses on the images.", "run"),
    ("4 · Analyse", "What do the cells do, per condition?", "Quick figures, readouts and features, commitment to a "
     "fate, and stochastic models of the dynamics.", "figures"),
    ("5 · Publish", "A figure page for the paper", "Lay out panels at journal size, check them against the figure "
     "standard, and export PDF with the caption.", "export"),
]


class HomePage(QWidget):
    """Start here: open or create an experiment, and the five stages of the workflow."""
    new_requested = Signal()
    open_requested = Signal()
    open_path = Signal(str)
    synthetic_requested = Signal()

    def __init__(self, nav: Navigator):
        super().__init__()
        self.nav = nav
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(18)
        hero = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(4)
        kicker = QLabel("FASTTRACK")
        kicker.setProperty("role", "kicker")
        col.addWidget(kicker)
        self.headline = QLabel("Track every cell, then ask what it did.")
        self.headline.setStyleSheet("font-size: 22pt; font-weight: 650;")
        col.addWidget(self.headline)
        sub = QLabel("Everything you set lives in one experiment file. Work down the sidebar: each stage needs the one "
                     "before it. Every page has a How to use button (or press F1).")
        sub.setWordWrap(True)
        theme.hint(sub)
        col.addWidget(sub)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        new = QPushButton("New experiment")
        theme.primary(new)
        new.clicked.connect(self.new_requested)
        opn = QPushButton("Open…")
        opn.clicked.connect(self.open_requested)
        synth = QPushButton("Try it on synthetic data…")
        synth.setProperty("role", "ghost")
        synth.clicked.connect(self.synthetic_requested)
        for b in (new, opn, synth):
            buttons.addWidget(b)
        buttons.addStretch(1)
        col.addSpacing(8)
        col.addLayout(buttons)
        hero.addLayout(col, 3)
        recent_card = QFrame()
        recent_card.setObjectName("HomeCard")
        rc = QVBoxLayout(recent_card)
        rc.setContentsMargins(14, 12, 14, 12)
        head = QLabel("Recent experiments")
        head.setStyleSheet("font-weight: 600;")
        rc.addWidget(head)
        self.recent = QListWidget()
        self.recent.setObjectName("RecentList")
        self.recent.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recent.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.recent.setMaximumHeight(190)
        self.recent.itemActivated.connect(lambda it: self.open_path.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.recent.itemClicked.connect(lambda it: self.open_path.emit(it.data(Qt.ItemDataRole.UserRole)))
        rc.addWidget(self.recent)
        hero.addSpacing(16)
        hero.addWidget(recent_card, 2)
        outer.addLayout(hero)
        outer.addSpacing(6)

        self.current = QLabel("")
        self.current.setWordWrap(True)
        outer.addWidget(self.current)
        grid = QGridLayout()
        grid.setSpacing(12)
        for i, (stage, question, text, key) in enumerate(STAGES):
            card = StageCard(stage, question, text)
            card.clicked.connect(lambda k=key: nav.go(k))
            grid.addWidget(card, 0, i, Qt.AlignmentFlag.AlignTop)
            grid.setColumnStretch(i, 1)
            self._cards = getattr(self, "_cards", []) + [card]
        outer.addLayout(grid)
        outer.addStretch(1)

    def set_recent(self, paths: list[str]) -> None:
        from pathlib import Path
        self.recent.clear()
        paths = [p for p in paths if Path(p).exists()]
        for p in paths:
            item = QListWidgetItem(f"{Path(p).name}\n{Path(p).parent}")
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            self.recent.addItem(item)
        if not paths:
            item = QListWidgetItem("No recent files yet.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent.addItem(item)

    def set_pages(self, stage_pages: dict[str, list[PageInfo]]) -> None:
        for card, (stage, *_rest) in zip(self._cards, STAGES):
            card.set_pages(stage_pages.get(stage, []))

    def set_current(self, name: str, path: str) -> None:
        self.current.setText(f"<span style='color:{theme.MUTED}'>Open now:</span> <b>{name}</b>"
                             f"<span style='color:{theme.MUTED}'>  ·  {path}</span>")


class StageCard(QFrame):
    clicked = Signal()

    def __init__(self, stage: str, question: str, text: str):
        super().__init__()
        self.setObjectName("StageCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 14)
        lay.setSpacing(4)
        k = QLabel(stage.upper())
        k.setProperty("role", "kicker")
        lay.addWidget(k)
        q = QLabel(question)
        q.setWordWrap(True)
        q.setStyleSheet("font-weight: 600; font-size: 11pt;")
        lay.addWidget(q)
        t = QLabel(text)
        t.setWordWrap(True)
        theme.hint(t)
        lay.addWidget(t)
        lay.addSpacing(6)
        self.list = QLabel("")
        self.list.setWordWrap(True)
        self.list.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.list)
        lay.addStretch(1)

    def set_pages(self, pages: list[PageInfo]) -> None:
        rows = []
        for p in pages:
            state = p.item.data(STATE) or "todo"
            colour = theme.STATE_COLORS.get(state, theme.IDLE)
            dot = "●" if state not in ("todo", "optional") else "○"
            status = p.item.data(STATUS) or ""
            name = f"{p.title}" + (f" <span style='color:{theme.FAINT}'>(optional)</span>" if p.optional else "")
            rows.append(f"<div style='margin:2px 0'><span style='color:{colour}'>{dot}</span>&nbsp; {name}"
                        f"<br><span style='color:{theme.MUTED}; font-size:8.5pt'>&nbsp;&nbsp;&nbsp;&nbsp;{status}</span></div>")
        self.list.setText("".join(rows))

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)
