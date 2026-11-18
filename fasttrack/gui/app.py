"""FastTrack main window. Everything you set is part of one experiment file (File > Save)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox)

import yaml

from ..experiment import is_legacy, load_experiment, new_experiment, normalise, pipeline_config, save_experiment
from .dynamics_tab import AnalysisSection
from .centres_tab import CentresTab
from .experiment_tab import ExperimentTab
from .inspect_tab import InspectTab
from .parameters_tab import ParametersTab
from .plate_tab import PlateLayoutTab
from .quality_tab import QualityTab
from .run_tab import RunTab
from .viewer_tab import ViewerTab
from . import theme
from .shell import HelpDialog, HomePage, Navigator
from .status import page_status
from .worker import FunctionWorker

FILTER = "Experiment (*.yaml *.yml)"


class MainWindow(QMainWindow):
    def __init__(self, path: str | None = None):
        super().__init__()
        self.exp = new_experiment()
        self._loading = True  # until the first load: tabs are being built
        self.path: Path | None = None
        self.saved = copy.deepcopy(self.exp)
        self.cfg = pipeline_config(self.exp)   # shared, updated in place, read by the tracking tabs
        self.resize(1500, 920)

        self.experiment = ExperimentTab()
        self.plate = PlateLayoutTab()
        self.parameters = ParametersTab(self.exp["pipeline"])
        self.quality = QualityTab(self.collect, self.ensure_timestacks)
        self.run = RunTab(self.collect, self.sync_cfg)
        self.inspect = InspectTab(self.cfg, self.sync_cfg, self.collect, self.ensure_timestacks)
        self.viewer = ViewerTab(self.cfg, self.sync_cfg)
        self.centres = CentresTab(self.collect, self.ensure_timestacks)
        self.analysis = AnalysisSection(self.collect)
        self.analysis.setParent(self)
        self.analysis.hide()
        a = self.analysis
        a.tabs.clear()                       # its panels become pages of their own
        self.analysis_pages = (a.summary, a.figures, a.readouts, a.features, a.fates, a.dynamics, a.export)

        nav = Navigator()
        self.home = HomePage(nav)
        nav.add_page("home", self.home, "Home", "", header=False)
        nav.add_stage("1 · Set up")
        nav.add_page("experiment", self.experiment, "Experiment", "Where the images are, how files are named, what each channel is, and when things happened.")
        nav.add_page("plate", self.plate, "Plate", "Which wells are which condition. Conditions decide what is tracked and how figures group cells.")
        nav.add_stage("2 · Check & tune")
        nav.add_page("quality", self.quality, "Quality", "Find positions with fibers, debris or odd backgrounds before tracking them.")
        nav.add_page("inspect", self.inspect, "Test run", "Run the pipeline on part of one position and look at every step. Nothing is saved.")
        nav.add_page("parameters", self.parameters, "Parameters", "Every tracking parameter. Change one at a time and check it in a test run.")
        nav.add_page("centres", self.centres, "Label cells", "Click nucleus centres and cell states on patches; train the seed finder and the state classifier, and review their disagreements.", optional=True)
        nav.add_stage("3 · Track")
        nav.add_page("run", self.run, "Run", "Track every selected position in the background, then draw the figures.")
        nav.add_page("viewer", self.viewer, "Viewer", "Tracked positions on the images: track tails, outlines, and any cell's timecourse.")
        nav.add_stage("4 · Analyse")
        nav.add_page("summary", a.summary, "Summary", "Routine figures drawn for you after every run: cell counts, nuclear size, and each channel's primary measurement.")
        nav.add_page("figures", a.figures, "Figures", "Quick figures per condition: time courses, heatmaps, % passing a gate, distributions.")
        nav.add_page("readouts", a.readouts, "Readouts", "What each reporter measures, with a per-observation error, for the dynamics models.")
        nav.add_page("features", a.features, "Features", "The measurements discovery analyses use, normalised to a reference condition.")
        nav.add_page("fates", a.fates, "Fates", "When cells commit to a fate, and what other features do around commitment.")
        nav.add_page("dynamics", a.dynamics, "Dynamics", "Stochastic models fitted to every track: switching rates, hidden states, routes.")
        nav.add_stage("5 · Publish")
        nav.add_page("export", a.export, "Export", "Figure pages at journal size, checked against the figure standard, with a caption.")
        nav.finish()
        a.tabs = nav                         # code that switched analysis sub-tabs switches pages now
        self.nav = self.tabs = nav
        self._build_sidebar()
        self.help_dialog = None
        nav.help_requested.connect(self.show_help)
        self.home.new_requested.connect(self.new)
        self.home.open_requested.connect(self.open)
        self.home.open_path.connect(lambda p: self.maybe_save() and self.open_path(p) and self.nav.go("experiment"))
        self.home.synthetic_requested.connect(self.write_synthetic)
        self.setCentralWidget(nav)
        nav.currentChanged.connect(self._tab_changed)
        self._mark_primary()

        self.experiment.scanned.connect(self._scanned)
        self.plate.condition_renamed.connect(self.analysis.rename_condition)
        self.plate.condition_removed.connect(self.analysis.remove_condition)
        self.run.timestacks_found.connect(self.viewer.set_timestacks)
        self.run.timestacks_found.connect(self.inspect.set_timestacks)
        self.run.results_ready.connect(self.viewer.results_updated)
        self.run.results_ready.connect(self.analysis.refresh_list)
        self.run.cache_ready.connect(self.viewer.set_cache)
        self.run.figures_requested.connect(self.analysis.draw_all)
        self.run.run_finished.connect(self.analysis.summary.after_run)
        self.analysis.summary.copy_figure.connect(lambda _: self.nav.go("figures"))
        self.run.quality_checked.connect(self.quality.apply_flags)
        self.run.files_flagged.connect(self.quality.add_flags)
        self.centres.use_model.connect(self._use_centre_model)
        self.experiment.get_quality = lambda: self.quality.collect()
        self.quality.inspect_requested.connect(lambda key, frame: (self.tabs.setCurrentWidget(self.inspect), self.inspect.show_position(key, frame)))
        self.quality.changed.connect(lambda: self.run.timestacks and self.run.set_timestacks(self.run.timestacks))
        self._build_menu()
        self.load(self.exp)
        if path and self.open_path(path):
            self.nav.go("experiment")

    # ------------------------------------------------------------------ menu
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, key, slot, tip in (("&New experiment", QKeySequence.StandardKey.New, self.new, "Start a new experiment file"),
                                     ("&Open…", QKeySequence.StandardKey.Open, self.open, "Open an experiment file (older settings files are converted)"),
                                     ("&Save", QKeySequence.StandardKey.Save, self.save, "Save every setting to the experiment file"),
                                     ("Save &as…", QKeySequence.StandardKey.SaveAs, self.save_as, "Save to a new experiment file")):
            action = QAction(text, self)
            action.setShortcut(key)
            action.setToolTip(tip)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        self.recent_menu = file_menu.addMenu("Open &recent")
        self._fill_recent()
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        tools = self.menuBar().addMenu("&Tools")
        synth = QAction("Write a &synthetic experiment…", self)
        synth.triggered.connect(self.write_synthetic)
        tools.addAction(synth)
        help_menu = self.menuBar().addMenu("&Help")
        page_help = QAction("How to use this page", self)
        page_help.setShortcut(QKeySequence(Qt.Key.Key_F1))
        page_help.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        page_help.triggered.connect(lambda: self.show_help(self.nav.current_key()))
        help_menu.addAction(page_help)
        overview = QAction("The workflow, start to finish", self)
        overview.triggered.connect(lambda: self.show_help("home"))
        help_menu.addAction(overview)

    def _build_sidebar(self) -> None:
        """App name, the open experiment and its file, and Save, above the page list."""
        head = self.nav.header
        brand = QLabel("FastTrack")
        brand.setObjectName("Brand")
        head.addWidget(brand)
        sub = QLabel("cell tracking and analysis")
        sub.setObjectName("BrandSub")
        head.addWidget(sub)
        head.addSpacing(14)
        self.file_name = QLabel("")
        self.file_name.setObjectName("FileName")
        self.file_name.setWordWrap(True)
        head.addWidget(self.file_name)
        self.file_label = QLabel("")
        self.file_label.setObjectName("FilePath")
        self.file_label.setWordWrap(True)
        head.addWidget(self.file_label)
        row = QHBoxLayout()
        row.setContentsMargins(0, 8, 0, 4)
        self.save_button = QPushButton("Save")
        self.save_button.setToolTip("Save every setting to the experiment file (Ctrl+S)")
        self.save_button.clicked.connect(self.save)
        open_button = QPushButton("Open…")
        open_button.clicked.connect(self.open)
        row.addWidget(self.save_button)
        row.addWidget(open_button)
        head.addLayout(row)
        tip = QLabel("F1: how to use this page")
        tip.setObjectName("FilePath")
        self.nav.footer.addWidget(tip)

    def show_help(self, key: str) -> None:
        from .help_text import HELP
        info = self.nav.by_key.get(key) or self.nav.by_key["home"]
        if self.help_dialog is None:
            self.help_dialog = HelpDialog(self)
        title = "FastTrack" if info.key == "home" else info.title
        self.help_dialog.show_page(title, info.stage, HELP.get(info.key, HELP["home"]))

    def _mark_primary(self) -> None:
        """The one button per page that does its main job."""
        a = self.analysis
        buttons = [self.experiment.scan_button, self.quality.check_button, self.inspect.run_button, self.run.run_button,
                   self.centres.start_button, a.fates.run_button, a.dynamics.fit_button]
        for page, text in ((a.figures, "Preview"), (a.export, "Export selected"), (a.features, "Suggest a full set"),
                           (a.readouts, "Suggest from channel reporters")):
            buttons += [b for b in page.findChildren(QPushButton) if b.text() == text][:1]
        theme.primary(*buttons)
        # combo boxes with long entries must not force their panel wider than the window; they show the full text
        # in their drop-down anyway
        for combo in self.findChildren(QComboBox):
            if combo.sizeHint().width() > 260:
                combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(14)

    def _update_status(self) -> None:
        """The line under each page's name in the sidebar, and its dot."""
        for key, (text, state) in page_status(self.exp, self).items():
            self.nav.set_status(key, text, state)
        self.nav.nav.viewport().update()
        stages: dict = {}
        for p in self.nav.pages:
            if p.stage:
                stages.setdefault(p.stage, []).append(p)
        self.home.set_pages(stages)

    def _recent(self) -> list[str]:
        value = QSettings("FastTrack", "FastTrack").value("recent", [])
        return [value] if isinstance(value, str) else list(value or [])

    def _remember(self, path: Path) -> None:
        recent = [str(path)] + [p for p in self._recent() if p != str(path)]
        QSettings("FastTrack", "FastTrack").setValue("recent", recent[:8])
        self._fill_recent()

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for p in self._recent():
            action = QAction(p, self)
            action.triggered.connect(lambda _=False, p=p: self.maybe_save() and self.open_path(p))
            self.recent_menu.addAction(action)
        self.recent_menu.setEnabled(bool(self._recent()))
        if hasattr(self, "home"):
            self.home.set_recent(self._recent())

    # ------------------------------------------------------------------ experiment state
    def collect(self) -> dict:
        """Read every tab into the experiment dict and return it. While a file is loading, the tabs
        are half old, half new, so the loaded dict is returned as it is."""
        if self._loading:
            return self.exp
        parts = {**self.experiment.collect(), "plate": self.plate.collect(), "quality": self.quality.collect(),
                 "pipeline": self.parameters.collect(), "labelling": self.centres.collect(),
                 **self.analysis.collect()}
        fresh = normalise({**self.exp, **parts})
        self.exp.clear()
        self.exp.update(fresh)
        self._title()
        return self.exp

    def sync_cfg(self) -> dict:
        """Collect, then rebuild the shared tracking config in place. Returns it."""
        cfg = pipeline_config(self.collect())
        self.cfg.clear()
        self.cfg.update(cfg)
        return self.cfg

    def load(self, exp: dict) -> None:
        exp = normalise(exp)
        self.exp.clear()
        self.exp.update(exp)
        self._loading = True
        try:
            self._load_tabs()
        finally:
            self._loading = False
        self.saved = copy.deepcopy(self.exp)
        self.sync_cfg()
        self.run.show_paths()

    def _load_tabs(self) -> None:
        self.experiment.load(self.exp)
        self.plate.load(self.exp["plate"])
        self.parameters.load(self.exp["pipeline"])
        self.quality.load(self.exp["quality"])
        self.centres.load(self.exp["labelling"])
        self.analysis.load(self.exp)
        self.plate.set_found_wells(None)

    def dirty(self) -> bool:
        return normalise(self.exp) != normalise(self.saved)

    def _title(self) -> None:
        name = self.path.name if self.path else "untitled experiment"
        dirty = normalise(self.exp) != normalise(self.saved)
        self.setWindowTitle(f"FastTrack — {name}{' •' if dirty else ''}")
        if hasattr(self, "file_label"):
            label = self.exp["experiment"]["name"] or "Untitled experiment"
            self.file_name.setText(label)
            where = str(self.path) if self.path else "not saved yet"
            self.file_label.setText(f"{'● unsaved changes · ' if dirty else ''}{where}")
            self.save_button.setProperty("role", "primary" if dirty else "")
            theme._restyle(self.save_button)
            self.home.set_current(label, where)
            self._update_status()

    def _tab_changed(self, _: int) -> None:
        self.collect()
        widget = self.tabs.currentWidget()
        if widget in self.analysis_pages:
            self.analysis.refresh_choices()
        elif widget is self.run:
            self.run.show_paths()

    def _use_centre_model(self, path: str, threshold: float) -> None:
        """Point tracking at the trained seed finder (Label cells tab)."""
        params = self.parameters.collect()
        params["centroids"].update({"method": "cnn", "cnn_checkpoint": path, "cnn_threshold": threshold})
        self.parameters.load(params)
        self.sync_cfg()

    def ensure_timestacks(self) -> dict:
        """The experiment's positions, found once and shared by every tab (Run, Inspect, Quality, Labelling)."""
        if not self.run.timestacks:
            self.run.discover(quiet=True)
        return self.run.timestacks

    def _scanned(self, timestacks: dict) -> None:
        self.plate.set_found_wells({ts["well"] for ts in timestacks.values()})
        self.sync_cfg()
        self.run.discover()

    # ------------------------------------------------------------------ file actions
    def maybe_save(self) -> bool:
        self.collect()
        if not self.dirty():
            return True
        answer = QMessageBox.question(self, "Unsaved changes", "Save the changes to this experiment first?",
                                      QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def new(self) -> None:
        if self.maybe_save():
            self.path = None
            self.load(new_experiment())
            self.nav.go("experiment")

    def open(self) -> None:
        if not self.maybe_save():
            return
        start = str(self.path.parent) if self.path else (str(Path(self._recent()[0]).parent) if self._recent() else "")
        path, _ = QFileDialog.getOpenFileName(self, "Open experiment", start, FILTER + ";;All files (*)")
        if path and self.open_path(path):
            self.nav.go("experiment")

    def open_path(self, path: str | Path) -> bool:
        path = Path(path)
        try:
            exp = load_experiment(path)
        except Exception as exc:  # any unreadable file: say so, keep the current experiment
            QMessageBox.warning(self, "Could not open", f"{path}\n\n{exc}")
            return False
        with open(path) as f:
            imported = is_legacy(yaml.safe_load(f) or {})
        self.path = None if imported else path
        self.load(exp)
        if imported:
            QMessageBox.information(self, "Imported", "This is an older settings or plate file. It was converted; "
                                    "anything not understood is in the notes. Save it as a new experiment file.")
            self.saved = new_experiment()
        else:
            self._remember(path)
        self._title()
        return True

    def save(self) -> bool:
        if self.path is None:
            return self.save_as()
        save_experiment(self.collect(), self.path)
        self.saved = copy.deepcopy(self.exp)
        self._remember(self.path)
        self._title()
        return True

    def save_as(self) -> bool:
        exp = self.collect()
        suggestion = (exp["experiment"]["name"] or "experiment").replace(" ", "_") + ".yaml"
        start = str(self.path) if self.path else str(Path(exp["paths"]["results"] or Path.home()) / suggestion)
        path, _ = QFileDialog.getSaveFileName(self, "Save experiment", start, FILTER)
        if not path:
            return False
        self.path = Path(path if path.endswith((".yaml", ".yml")) else path + ".yaml")
        return self.save()

    def closeEvent(self, event) -> None:
        if not self.maybe_save():
            event.ignore()
            return
        self.stop_workers()
        event.accept()

    def stop_workers(self) -> None:
        """Ask every background job to stop and wait for it, so no thread outlives the window."""
        for worker in self.findChildren(FunctionWorker):
            if worker.isRunning():
                worker.stop_flag.set()
                worker.wait(5000)

    # ------------------------------------------------------------------ tools
    def write_synthetic(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Write a synthetic experiment")
        form = QFormLayout(dialog)
        folder = QLineEdit(str(Path.home() / "fasttrack_synthetic"))
        form.addRow("Empty folder for the images", folder)
        preset = QComboBox()
        preset.addItems(["mosaic (sparse, like 1 in 5 labelled)", "populated", "dense"])
        form.addRow("Density", preset)
        frames = QSpinBox()
        frames.setRange(10, 2000)
        frames.setValue(100)
        form.addRow("Frames", frames)
        wells = QLineEdit("B02, B03")
        form.addRow("Wells", wells)
        positions = QSpinBox()
        positions.setRange(1, 16)
        positions.setValue(1)
        form.addRow("Positions per well", positions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        root = Path(folder.text().strip())
        if root.exists() and any(root.iterdir()):
            QMessageBox.information(self, "Folder not empty", "Choose an empty or new folder.")
            return
        if not self.maybe_save():
            return
        from ..synthetic.generate import DENSE_PRESET, POPULATED_PRESET
        well_list = [w.strip().upper() for w in wells.text().split(",") if w.strip()]
        updates = {"frames": frames.value(), **[{}, POPULATED_PRESET, DENSE_PRESET][preset.currentIndex()]}
        device = self.exp["pipeline"]["device"]

        def job(on_event, stop):
            from ..synthetic.write import write_experiment
            write_experiment(root / "raw", well_list, list(range(1, positions.value() + 1)), updates, device=device)
            exp = new_experiment(
                experiment={"name": f"Synthetic {preset.currentText().split()[0]}",
                            "notes": f"Written by Tools > Write a synthetic experiment. Generator settings: {updates}. "
                                     "Ground truth is in raw/truth_<well>_xy<pos>.pkl."},
                paths={"raw": str(root / "raw"), "results": str(root / "results")},
                channels=[{"file_channel": 1, "name": "nuclear", "role": "nuclear"},
                          {"file_channel": 2, "name": "c2", "role": "measure"},
                          {"file_channel": 3, "name": "c3", "role": "measure"}],
                plate={"conditions": [{"name": "synthetic", "wells": well_list}]})
            save_experiment(exp, root / "experiment.yaml")
            return root / "experiment.yaml"

        self.statusBar().showMessage(f"Writing synthetic images to {root}…")
        self._synth_worker = FunctionWorker(job, self)
        self._synth_worker.finished_ok.connect(lambda path: (self.statusBar().showMessage(f"Wrote {path}", 8000), self.open_path(path)))
        self._synth_worker.failed.connect(lambda m: QMessageBox.critical(self, "Could not write", m.splitlines()[0]))
        self._synth_worker.start()


def main(path: str | None = None) -> None:
    app = QApplication(sys.argv[:1])
    app.setApplicationName("FastTrack")
    theme.apply(app)
    window = MainWindow(path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
