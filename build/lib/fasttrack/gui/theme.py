"""The app's look in one place: colour tokens, the Qt palette and one stylesheet.

Tabs never set colours themselves. They mark the button that does the page's main job with `primary(button)` and
secondary explanatory text with `hint(label)`; everything else follows from the widget type.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QWidget

# Colour tokens. Data colours (conditions, states) come from the experiment and never use these.
BG = "#F3F4F7"          # window
SURFACE = "#FFFFFF"     # cards, inputs, tables
SUNKEN = "#F7F8FA"      # table headers, read-only fields
BORDER = "#DFE2E8"
BORDER_STRONG = "#C9CED8"
TEXT = "#1D2433"
MUTED = "#5F6B7E"
FAINT = "#9AA3B2"
ACCENT = "#4F46E5"
ACCENT_HOVER = "#4338CA"
ACCENT_SOFT = "#ECEBFD"
ACCENT_TEXT = "#3730A3"
OK = "#16A34A"
WARN = "#D97706"
BAD = "#DC2626"
IDLE = "#8792A6"

SIDEBAR = "#171C28"
SIDEBAR_HOVER = "#222939"
SIDEBAR_ACTIVE = "#2A3247"
SIDEBAR_TEXT = "#C8CFDC"
SIDEBAR_MUTED = "#7D879A"

STATE_COLORS = {"done": OK, "warn": WARN, "todo": IDLE, "optional": IDLE, "bad": BAD}


def primary(*buttons) -> None:
    """The one button that does the page's main job: filled with the accent colour."""
    for b in buttons:
        b.setProperty("role", "primary")
        _restyle(b)


def danger(*buttons) -> None:
    for b in buttons:
        b.setProperty("role", "danger")
        _restyle(b)


def hint(*labels) -> None:
    """Secondary, explanatory text."""
    for label in labels:
        label.setProperty("role", "hint")
        label.setStyleSheet("")
        _restyle(label)


def _restyle(w: QWidget) -> None:
    w.style().unpolish(w)
    w.style().polish(w)


def palette() -> QPalette:
    p = QPalette()
    roles = {QPalette.ColorRole.Window: BG, QPalette.ColorRole.WindowText: TEXT, QPalette.ColorRole.Base: SURFACE,
             QPalette.ColorRole.AlternateBase: SUNKEN, QPalette.ColorRole.Text: TEXT, QPalette.ColorRole.Button: SURFACE,
             QPalette.ColorRole.ButtonText: TEXT, QPalette.ColorRole.Highlight: ACCENT,
             QPalette.ColorRole.HighlightedText: "#FFFFFF", QPalette.ColorRole.ToolTipBase: TEXT,
             QPalette.ColorRole.ToolTipText: "#FFFFFF", QPalette.ColorRole.PlaceholderText: FAINT,
             QPalette.ColorRole.Mid: BORDER_STRONG, QPalette.ColorRole.Midlight: BORDER, QPalette.ColorRole.Light: SURFACE,
             QPalette.ColorRole.Dark: BORDER_STRONG, QPalette.ColorRole.Link: ACCENT}
    for role, colour in roles.items():
        p.setColor(role, QColor(colour))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(FAINT))
    return p


STYLESHEET = f"""
* {{ outline: none; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {TEXT}; color: white; border: none; padding: 6px 8px; border-radius: 4px; }}

/* cards */
QGroupBox {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    margin-top: 24px; padding: 12px 10px 10px 10px; font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left; left: 2px; top: 2px;
    padding: 0 2px; color: {TEXT};
}}
QGroupBox QLabel, QGroupBox QCheckBox, QGroupBox QRadioButton {{ font-weight: normal; }}
QGroupBox QGroupBox {{ background: {SUNKEN}; }}

/* text */
QLabel[role="hint"] {{ color: {MUTED}; }}
QLabel[role="pagetitle"] {{ font-size: 17pt; font-weight: 600; color: {TEXT}; }}
QLabel[role="pagesub"] {{ color: {MUTED}; }}
QLabel[role="kicker"] {{ color: {ACCENT}; font-weight: 600; font-size: 8.5pt; letter-spacing: 1px; }}

/* inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QDateEdit {{
    background: {SURFACE}; border: 1px solid {BORDER_STRONG}; border-radius: 6px;
    padding: 3px 6px; min-height: 22px; selection-background-color: {ACCENT};
}}
QPlainTextEdit, QTextEdit {{ padding: 6px 8px; }}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QPlainTextEdit:hover, QTextEdit:hover {{
    border-color: {FAINT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {SUNKEN}; color: {FAINT}; border-color: {BORDER};
}}
QLineEdit[readOnly="true"] {{ background: {SUNKEN}; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 18px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 18px; border: none; background: transparent; subcontrol-origin: border;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; margin: 2px 2px 0 0; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; margin: 0 2px 2px 0; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {SUNKEN}; border-radius: 3px; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url(@UP@); width: 9px; height: 9px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url(@DOWN@); width: 9px; height: 9px; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: none; }}
QComboBox {{ padding-right: 18px; }}
QComboBox::drop-down {{ border: none; width: 22px; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url(@DOWN@); width: 10px; height: 10px; margin-right: 4px; }}
QComboBox::drop-down {{ width: 16px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px;
    selection-background-color: {ACCENT_SOFT}; selection-color: {ACCENT_TEXT}; outline: none;
}}

/* buttons */
QPushButton, QToolButton {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER_STRONG}; border-radius: 6px;
    padding: 5px 14px; min-height: 20px;
}}
QPushButton:hover, QToolButton:hover {{ background: {SUNKEN}; border-color: {FAINT}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {BORDER}; }}
QPushButton:checked, QToolButton:checked {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {ACCENT_TEXT}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {FAINT}; background: {SUNKEN}; border-color: {BORDER}; }}
QPushButton[role="primary"] {{ background: {ACCENT}; color: white; border: 1px solid {ACCENT}; font-weight: 600; }}
QPushButton[role="primary"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[role="primary"]:disabled {{ background: #B9B6F3; border-color: #B9B6F3; color: #F4F4FF; }}
QPushButton[role="danger"] {{ color: {BAD}; }}
QPushButton[role="danger"]:hover {{ background: #FDECEC; border-color: {BAD}; }}
QPushButton[role="ghost"] {{ background: transparent; border: 1px solid transparent; color: {MUTED}; }}
QPushButton[role="ghost"]:hover {{ background: {SUNKEN}; color: {TEXT}; }}
QPushButton[role="help"] {{
    background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; border: 1px solid transparent; font-weight: 600;
    padding: 5px 14px;
}}
QPushButton[role="help"]:hover {{ border-color: {ACCENT}; }}
QToolButton::menu-indicator {{ subcontrol-position: right center; right: 6px; }}
QToolButton[popupMode="2"] {{ padding-right: 22px; }}

/* check boxes and radios */
QCheckBox, QRadioButton {{ spacing: 7px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {BORDER_STRONG}; background: {SURFACE};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
    image: url(@CHECK@);
}}
QRadioButton::indicator:checked {{ background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
    stop:0 white, stop:0.35 white, stop:0.4 {ACCENT}, stop:1 {ACCENT}); border-color: {ACCENT}; }}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ background: {SUNKEN}; border-color: {BORDER}; }}

/* lists, tables, trees */
QListView, QListWidget, QTableView, QTableWidget, QTreeView, QTreeWidget {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
    alternate-background-color: {SUNKEN}; gridline-color: #EDEFF3;
    selection-background-color: {ACCENT_SOFT}; selection-color: {ACCENT_TEXT};
}}
QListView::item, QListWidget::item {{ padding: 3px 4px; border-radius: 4px; }}
QListView::item:hover, QListWidget::item:hover {{ background: {SUNKEN}; }}
QListView::item:selected, QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; }}
QTableView::item:selected, QTableWidget::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; }}
QHeaderView {{ background: {SUNKEN}; border: none; }}
QHeaderView::section {{
    background: {SUNKEN}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    border-right: 1px solid #EDEFF3; padding: 5px 8px; font-weight: 600;
}}
QHeaderView::section:last {{ border-right: none; }}
QTableCornerButton::section {{ background: {SUNKEN}; border: none; border-bottom: 1px solid {BORDER}; }}

/* tabs (inside pages) */
QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; top: -1px; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; border: none; border-bottom: 2px solid transparent;
    padding: 7px 14px; margin-right: 2px;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {ACCENT_TEXT}; border-bottom: 2px solid {ACCENT}; font-weight: 600; }}

/* scroll bars */
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{ background: #C5CBD6; border-radius: 3px; }}
QScrollBar::handle:vertical {{ min-height: 28px; }}
QScrollBar::handle:horizontal {{ min-width: 28px; }}
QScrollBar::handle:hover {{ background: #A5AEBD; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* misc */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 8px; }}
QSplitter::handle:vertical {{ height: 8px; }}
QProgressBar {{ background: {BORDER}; border: none; border-radius: 4px; height: 8px; text-align: center;
               color: transparent; max-height: 8px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: white; border: 2px solid {ACCENT}; width: 12px; height: 12px;
                              margin: -6px 0; border-radius: 8px; }}
QMenuBar {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; padding: 2px 4px; }}
QMenuBar::item {{ padding: 4px 10px; border-radius: 4px; background: transparent; }}
QMenuBar::item:selected {{ background: {SUNKEN}; }}
QMenu {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
QStatusBar {{ background: {SURFACE}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QToolBar QToolButton {{ background: transparent; border: 1px solid transparent; padding: 3px; border-radius: 6px; }}
QToolBar QToolButton:hover {{ background: {SUNKEN}; border-color: {BORDER}; }}
QToolBar QToolButton:checked {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; }}
QToolBar {{ background: {SURFACE}; border: none; border-bottom: 1px solid {BORDER}; spacing: 4px; padding: 3px; }}

/* shell */
#Sidebar {{ background: {SIDEBAR}; }}
#Sidebar QLabel {{ color: {SIDEBAR_TEXT}; }}
#Brand {{ color: white; font-size: 14pt; font-weight: 700; }}
#BrandSub {{ color: {SIDEBAR_MUTED}; }}
#FileName {{ color: white; font-weight: 600; }}
#FilePath {{ color: {SIDEBAR_MUTED}; font-size: 8.5pt; }}
#Nav {{ background: transparent; border: none; }}
#Nav::item {{ border: none; padding: 0; }}
#Nav::item:hover, #Nav::item:selected {{ background: transparent; }}
#Sidebar QPushButton {{ background: {SIDEBAR_HOVER}; color: {SIDEBAR_TEXT}; border: 1px solid #2E3648; }}
#Sidebar QPushButton:hover {{ background: {SIDEBAR_ACTIVE}; color: white; }}
#Sidebar QPushButton[role="primary"] {{ background: {ACCENT}; color: white; border-color: {ACCENT}; }}
#Sidebar QPushButton[role="primary"]:hover {{ background: {ACCENT_HOVER}; }}
#Sidebar QScrollBar::handle:vertical {{ background: #333C50; }}
#PageHeader {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
#HomeCard {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
#StageCard {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; }}
#StageCard:hover {{ border-color: {ACCENT}; }}
#RecentList {{ border: none; background: transparent; }}
#RecentList::item {{ padding: 6px 8px; border-radius: 6px; }}
QTextBrowser#Help {{ border: none; background: {SURFACE}; padding: 4px 18px; }}
"""


CHECK_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 15 15">'
             '<path d="M3.5 7.8 L6.2 10.4 L11.5 4.8" fill="none" stroke="white" stroke-width="1.9" '
             'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _chevron(points: str) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
            f'<path d="{points}" fill="none" stroke="{MUTED}" stroke-width="1.6" stroke-linecap="round" '
            'stroke-linejoin="round"/></svg>')


ICONS = {"check": CHECK_SVG, "up": _chevron("M3 7.5 L6 4.5 L9 7.5"), "down": _chevron("M3 4.5 L6 7.5 L9 4.5")}


def _icon_dir():
    """Icons the stylesheet needs as files (Qt style sheets can't take inline images)."""
    import os
    import tempfile
    from pathlib import Path
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "fasttrack" / "ui"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        base = Path(tempfile.gettempdir()) / "fasttrack-ui"
        base.mkdir(exist_ok=True)
    for name, svg in ICONS.items():
        f = base / f"{name}.svg"
        if not f.exists() or f.read_text() != svg:
            f.write_text(svg)
    return base


def stylesheet() -> str:
    d = _icon_dir()
    css = STYLESHEET
    for name in ICONS:
        css = css.replace(f"@{name.upper()}@", (d / f"{name}.svg").as_posix())
    return css


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setPalette(palette())
    font = app.font()
    if font.pointSizeF() < 9.5 or font.pointSizeF() > 11:
        font.setPointSizeF(10)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)
    app.setStyleSheet(stylesheet())
