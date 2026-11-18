import os

# GUI tests never open windows on the desktop, whatever the shell sets (e.g. QT_QPA_PLATFORM=wayland;xcb).
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import torch

from fasttrack.config import make_config


@pytest.fixture(scope="session")
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture()
def cfg(device):
    return make_config({"device": device})


@pytest.fixture(autouse=True, scope="session")
def _private_qt_settings(tmp_path_factory):
    """GUI tests must not add their temporary files to the user's recent-files list."""
    import os
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path_factory.mktemp("config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(tmp_path_factory.mktemp("cache")))
    try:
        from PySide6.QtCore import QSettings
        QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, os.environ["XDG_CONFIG_HOME"])
    except ImportError:
        pass
    yield


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """A message box in a test would wait for a click that never comes: answer it at once instead."""
    try:
        from PySide6.QtWidgets import QMessageBox
    except ImportError:
        return
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard))
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
