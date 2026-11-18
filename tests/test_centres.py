"""Nucleus centre labelling: candidates from local count changes, the labels file, training, and the tab."""
import os
import pickle
import random

import numpy as np
import pytest

from fasttrack import centres
from fasttrack.experiment import LABELLING_DEFAULTS, new_experiment
from fasttrack.track.measure import S, SCALARS

C = LABELLING_DEFAULTS["centres"]


def _fake_results(points):
    """A results-like dict from (t, x, y) rows."""
    points = np.array(points, float)
    scalars = np.full((len(points), len(SCALARS)), np.nan, np.float32)
    scalars[:, S["t"]], scalars[:, S["x"]], scalars[:, S["y"]] = points.T
    frames = int(points[:, 0].max()) + 1
    return {"scalars": scalars, "meta": {"frames": frames}, "drift": np.zeros((frames, 2))}


def test_a_one_frame_extra_nucleus_is_a_candidate_and_a_division_is_not():
    rng = np.random.default_rng(0)
    cells = rng.uniform(40, 470, (12, 2))
    rows = [(t, x, y) for t in range(20) for x, y in cells]
    rows.append((7, 300.0, 100.0))                       # a spurious seed for one frame
    rows += [(t, 120.0, 400.0) for t in range(12, 20)]   # a daughter from frame 12 on: the count stays up
    rows = [r for r in rows if not (r[0] == 15 and abs(r[1] - cells[3, 0]) < 1e-6)]  # one nucleus missed at frame 15
    cand = centres.count_changes(_fake_results(rows), (512, 512), C)
    found = {(o["t"], round(o["x"] / 40), round(o["y"] / 40)): o["change"] for o in cand}
    assert found.get((7, round(300 / 40), round(100 / 40)), 0) > 0.8
    assert any(o["t"] == 15 and o["change"] < -0.8 and abs(o["x"] - cells[3, 0]) < 16 and abs(o["y"] - cells[3, 1]) < 16 for o in cand)
    assert not any(10 <= o["t"] <= 14 and abs(o["x"] - 120) < 30 and abs(o["y"] - 400) < 30 for o in cand)   # the division
    assert abs(cand[0]["change"]) >= abs(cand[-1]["change"])                                                  # strongest first


def test_boxes_stay_inside_the_image_and_cuts_keep_their_size():
    assert centres.box_for(5, 1000, 96, (1022, 1024)) == (0, 1024 - 96)
    frame = np.arange(100 * 120, dtype=np.uint16).reshape(100, 120)
    a = centres.cut(frame, 0, 120 - 32, 32, 16)
    assert a.shape == (64, 64) and a[16, 16] == frame[0, 88]


def test_labels_file_and_summary(tmp_path):
    exp = new_experiment(paths={"results": str(tmp_path)})
    obs = {"position": "B02_xy1", "t": 3, "x0": 10, "y0": 20, "size": 64, "source": "count_change", "change": 1.2}
    rows = centres.add_label(exp, [], obs, [[20, 30], [40, 50]], [[20.5, 30], [60, 70], [61, 72]])
    rows = centres.add_label(exp, rows, obs, [[20, 30], [40, 50], [50, 50]], [[20.5, 30], [60, 70], [61, 72]])  # replaces
    assert centres.read_labels(exp) == rows and len(rows) == 1
    s = centres.label_summary(rows)
    assert (s["kept"], s["added"], s["removed"]) == (1, 2, 2)
    assert centres.covered(rows, "B02_xy1", 3, 40, 50) and not centres.covered(rows, "B02_xy1", 4, 40, 50)
    assert centres.prf(centres.match(np.array([[0, 0], [10, 10]]), np.array([[1, 0]]), 4), 2, 1)["recall"] == 0.5


def test_label_train_and_track_with_the_learned_seeds(tmp_path, device):
    """Track, draw patches, label them from the synthetic truth, train, and track again with method cnn."""
    from fasttrack.synthetic.write import write_experiment
    from fasttrack.track.run import run_experiment
    write_experiment(tmp_path / "raw", ["B02", "B03"], [1], {"frames": 16, "size": 192, "cells": 14, "stage_jumps": 0, "fibers": 0}, device=device)
    exp = new_experiment(paths={"raw": str(tmp_path / "raw"), "results": str(tmp_path / "out")},
                         plate={"conditions": [{"name": "A", "wells": ["B02", "B03"]}]},
                         quality={"enabled": False}, pipeline={"device": device, "batch_auto": False},
                         labelling={"centres": {"patch": 64, "margin": 16, "train_steps": 40}})
    run_experiment(exp)
    reader = centres.NuclearReader(exp, None, 2)
    sampler = centres.Sampler(exp, reader.shape, seed=1, available=reader.all())
    rows, sources = [], set()
    for _ in range(14):
        obs = sampler.draw(rows, set())
        assert obs is not None and 0 <= obs["x0"] <= 192 - 64 and 0 <= obs["y0"] <= 192 - 64
        with open(tmp_path / "raw" / f"truth_{obs['position'].replace('_', '_')}.pkl", "rb") as f:
            truth = pickle.load(f)["truth"]
        m = (truth["t"] == obs["t"]) & (truth["x"] >= obs["x0"]) & (truth["x"] < obs["x0"] + 64) \
            & (truth["y"] >= obs["y0"]) & (truth["y"] < obs["y0"] + 64)
        rows = centres.add_label(exp, rows, obs, np.stack([truth["x"][m], truth["y"][m]], 1).tolist(), sampler.tracked_points(obs))
        sources.add(obs["source"])
    reader.close()
    assert sources == {"count_change", "random"} or len(rows) == 14
    report = centres.train_from_labels(exp)
    assert report["patches"] == 14 and report["threshold"] in centres.THRESHOLDS
    assert {"f1", "missed", "extra"} <= set(report["model"]) and {"f1"} <= set(report["tracking"])
    assert centres.read_report(exp)["model_file"] == str(centres.model_path(exp))
    exp["pipeline"]["centroids"].update(method="cnn", cnn_checkpoint=report["model_file"], cnn_threshold=0.05)
    exp["pipeline"]["output"]["overwrite"] = True
    done = run_experiment(exp)                                   # the learned seeds go through the whole pipeline
    assert len(done) == 2


def test_centres_tab_edits_and_saves(tmp_path, device):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from fasttrack.gui.centres_tab import CentresTab
    from fasttrack.synthetic.write import write_experiment
    from fasttrack.track.run import run_experiment
    app = QApplication.instance() or QApplication([])
    write_experiment(tmp_path / "raw", ["B02"], [1], {"frames": 10, "size": 160, "cells": 10, "stage_jumps": 0, "fibers": 0}, device=device)
    exp = new_experiment(paths={"raw": str(tmp_path / "raw"), "results": str(tmp_path / "out")},
                         plate={"conditions": [{"name": "A", "wells": ["B02"]}]}, quality={"enabled": False},
                         pipeline={"device": device, "batch_auto": False}, labelling={"centres": {"patch": 64, "margin": 16}})
    run_experiment(exp)
    tab = CentresTab(lambda: exp, lambda: None)
    tab.load(exp["labelling"])
    tab.show()
    tab.next_patch()
    assert tab.current is not None and tab.points == tab.tracked
    obs, m = tab.current, 16
    before = len(tab.points)
    tab._clicked(m + 5.0, m + 5.0)                      # inside the box, away from points: adds (unless a point is there)
    tab._clicked(2.0, 2.0)                              # in the margin: ignored
    assert len(tab.points) in (before + 1, before - 1)
    tab.undo_click()
    assert len(tab.points) == before
    tab.clear_points()
    tab._clicked(m + 30.0, m + 31.0)
    assert tab.points == [[obs["x0"] + 30.0, obs["y0"] + 31.0]]
    tab.save()
    saved = centres.read_labels({**exp, "labelling": tab.collect()})
    assert len(saved) == 1 and saved[0]["points"] == [[obs["x0"] + 30.0, obs["y0"] + 31.0]] and saved[0]["states"] == ["neutral"]
    tab.back()
    assert tab.current["t"] == obs["t"] and tab.points == [[obs["x0"] + 30.0, obs["y0"] + 31.0]]
    assert centres.read_labels({**exp, "labelling": tab.collect()}) == []
    # at least 3 frames each way, flipped with the film strip; other frames are view only
    assert tab.k >= 3 and len(tab.thumbs) == 2 * tab.k + 1
    tab.view_frame(-2)
    tab._clicked(m + 40.0, m + 40.0)                    # brings the labelled frame back instead of editing
    assert tab.view_dt == 0 and len(tab.points) == 1
    # number keys: state for the point under the crosshair, or a new point with that state
    tab.hover = (m + 30.5, m + 31.0)
    tab.state_key(2)
    assert tab.states == ["division"]
    tab.hover = (m + 50.0, m + 10.0)
    tab.state_key(3)
    assert tab.points[-1] == [obs["x0"] + 50.0, obs["y0"] + 10.0] and tab.states[-1] == "death"
    tab.undo_click()
    assert len(tab.points) == 1
    tab.save()
    saved = centres.read_labels({**exp, "labelling": tab.collect()})
    assert saved[0]["states"] == ["division"]
    # dataset page lists it, in both views; double-click opens it for correction, Enter goes back
    tab.pages.setCurrentIndex(1)
    assert tab.ds_list.count() == 1
    tab.ds_mode.setCurrentIndex(1)
    assert tab.ds_list.count() == 1 and tab.ds_list.item(0).text() == "division"
    tab.open_row(tab.ds_list.item(0).data(Qt.ItemDataRole.UserRole))
    assert tab.pages.currentIndex() == 0 and tab.states == ["division"]
    tab.save()
    assert tab.pages.currentIndex() == 1
    # review page without a check says so; with one, a disagreement opens with the model's centres
    tab.pages.setCurrentIndex(2)
    assert "No check yet" in tab.rv_status.text()
    tab.review = {"made": "2000-01-01 00:00:00", "folds": 2, "errors": {}, "classes": ["neutral", "division"],
                  "patches": [{"key": [obs["position"], obs["t"], obs["x0"], obs["y0"]], "model_points": [], "matched": 0, "missed": 1, "extra": 0}],
                  "cells": []}
    tab.rv_hide.setChecked(False)
    tab.show_review()
    assert tab.rv_list.count() == 1
    tab._open_review_item(tab.rv_list.item(0))
    assert tab.pages.currentIndex() == 0 and tab.overlay["kind"] == "centres"
    tab.close()
    app.processEvents()
