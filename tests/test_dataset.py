"""One labelled dataset for both models: states on points, event candidates as patches, the label check, migration."""
import csv
import pickle

import numpy as np
import pytest

from fasttrack import centres, dataset, events
from fasttrack.experiment import new_experiment


@pytest.fixture(scope="module")
def tracked(tmp_path_factory, device):
    from fasttrack.synthetic.write import write_experiment
    from fasttrack.track.run import run_experiment
    tmp = tmp_path_factory.mktemp("ds")
    write_experiment(tmp / "raw", ["B02", "B03"], [1], {"frames": 16, "size": 192, "cells": 14, "stage_jumps": 0, "fibers": 0}, device=device)
    exp = new_experiment(paths={"raw": str(tmp / "raw"), "results": str(tmp / "out")},
                         plate={"conditions": [{"name": "A", "wells": ["B02", "B03"]}]},
                         quality={"enabled": False}, pipeline={"device": device, "batch_auto": False},
                         labelling={"train_steps": 30, "crop": 32, "window": 1, "folds": 2,
                                    "centres": {"patch": 64, "margin": 16, "train_steps": 30}})
    run_experiment(exp)
    return tmp, exp


def _label(tmp, exp, n, seed=1):
    """Draw n patches and label every centre from the synthetic truth: every third point in mitosis, every third neutral, the rest no state."""
    reader = centres.NuclearReader(exp, None, 2)
    sampler = centres.Sampler(exp, reader.shape, seed=seed, available=reader.all())
    rows, sources = dataset.read_dataset(exp), set()
    for _ in range(n):
        obs = sampler.draw(rows, set())
        with open(tmp / "raw" / f"truth_{obs['position']}.pkl", "rb") as f:
            truth = pickle.load(f)["truth"]
        m = (truth["t"] == obs["t"]) & (truth["x"] >= obs["x0"]) & (truth["x"] < obs["x0"] + 64) \
            & (truth["y"] >= obs["y0"]) & (truth["y"] < obs["y0"] + 64)
        pts = np.stack([truth["x"][m], truth["y"][m]], 1).tolist()
        states = ["mitosis" if (len(rows) + i) % 3 == 0 else "neutral" if (len(rows) + i) % 3 == 1 else "" for i in range(len(pts))]
        rows = centres.add_label(exp, rows, obs, pts, sampler.tracked_points(obs), states)
        sources.add(obs["source"])
    reader.close()
    return rows, sources


def test_states_are_saved_with_points_and_old_files_get_none(tracked):
    tmp, exp = tracked
    rows, sources = _label(tmp, exp, 12)
    assert sources & set(centres.EVENT_SOURCES), sources          # event candidates are drawn as patches too
    saved = centres.read_labels(exp)
    assert all(len(r["states"]) == len(r["points"]) for r in saved)
    assert all(s in ("neutral", "mitosis", centres.NONE) for r in saved for s in r["states"])
    old = {k: v for k, v in saved[0].items() if k != "states"}
    assert centres.point_states(old) == [centres.NONE] * len(old["points"])
    cells = dataset.cell_examples(exp, saved)
    assert len(cells) == sum(len(r["points"]) for r in saved)
    assert dataset.summary(exp, saved)["categories"]["mitosis"] > 0


def test_events_csv_is_copied_in_once_as_state_only_rows(tracked):
    tmp, exp = tracked
    before = centres.read_labels(exp)
    r = before[0]
    x, y = r["points"][0] if r["points"] else (r["x0"] + 10.0, r["y0"] + 10.0)
    path = events.labels_path(exp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, events.LABEL_COLUMNS)
        w.writeheader()
        w.writerow({"position": "B02_xy1", "t": 5, "track": 3, "x": 50.0, "y": 60.0, "source": "track_end", "category": "death", "labelled": ""})
    rows = dataset.read_dataset(exp)
    assert len(rows) == len(before) + 1 and rows[-1]["state_only"] and rows[-1]["states"] == ["death"]
    assert len(dataset.read_dataset(exp)) == len(rows)                       # only once
    assert len(centres.centre_rows(rows)) == len(before)                     # not used for the seed finder
    # labelling a full patch around that cell absorbs the state-only row
    obs = {"position": "B02_xy1", "t": 5, "x0": 20, "y0": 30, "size": 64, "source": "random"}
    rows = centres.add_label(exp, rows, obs, [[51.0, 59.0]], [], ["death"])
    assert not any(r.get("state_only") for r in rows)


def test_both_models_train_and_every_label_is_checked(tracked):
    tmp, exp = tracked
    _label(tmp, exp, 6, seed=2)
    out = dataset.train_all(exp)
    assert "error" not in out["centres"] and "error" not in out["states"], out
    assert set(out["states"]["classes"]) >= {"neutral", "mitosis"}
    assert events.load_model(events.model_path(exp), exp["pipeline"]["device"])[1]["settings"]["follow"] is False
    review = dataset.check_labels(exp)
    rows = dataset.read_dataset(exp)
    assert not review["errors"], review["errors"]
    assert len(review["patches"]) == len(centres.centre_rows(rows)) and all(p is not None for p in review["patches"])
    stated = [c for c in dataset.cell_examples(exp, rows) if c["state"] in exp["labelling"]["categories"]]
    assert len(review["cells"]) == len(stated)                     # cells with no state are not judged
    items = dataset.disagreements(review, rows)
    assert all(i["kind"] in ("centres", "state") for i in items)
    assert dataset.read_review(exp)["made"] == review["made"]
    # the state model classifies every candidate with crops at a fixed centre
    counts = events.classify_positions(exp)
    assert sum(counts.values()) > 0 and events.read_events(exp["paths"]["results"], "B02_xy1")


def test_points_without_a_state_are_centres_but_not_state_examples(tracked):
    tmp, exp = tracked
    rows, _ = _label(tmp, exp, 3)
    obs = {k: rows[0][k] for k in ("position", "t", "x0", "y0", "size", "source")}
    pts = rows[0]["points"]
    rows = centres.add_label(exp, rows, obs, pts, [], ["mitosis"] + [None] * (len(pts) - 1))
    r = rows[-1]
    assert r["states"][0] == "mitosis" and all(s == centres.NONE for s in r["states"][1:])
    assert centres.point_states({"points": pts}) == [centres.NONE] * len(pts)   # older files without states: none
    assert dataset.summary(exp, rows)["no_state"] >= len(pts) - 1
    trained_on = [c for c in dataset.cell_examples(exp, rows) if c["state"] in exp["labelling"]["categories"]]
    assert all(c["state"] for c in trained_on)
