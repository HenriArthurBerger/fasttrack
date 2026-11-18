"""Runs the pipeline off the GUI thread. Events cross into Qt as signals."""
from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal


class FunctionWorker(QThread):
    """Calls fn(on_event, stop). Any exception is reported, never lost in a background thread."""

    event = Signal(dict)
    failed = Signal(str)
    finished_ok = Signal(object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
        self.stop_flag = threading.Event()

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._fn(self.event.emit, self.stop_flag))
        except Exception as exc:  # shown to the user in the run log
            import traceback
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
