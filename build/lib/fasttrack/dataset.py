"""One labelled dataset for two models: nucleus centres (the seed finder) and cell states (the classifier).

You label patches in Label cells: every nucleus centred in the box gets one point, and every point a state
(labelling.categories, e.g. neutral, division, death, abnormal; the first is the default). The same file,
Labels/centres.json, trains both models:

  - the seed finder (fasttrack/centres.py) on the points of every full patch;
  - the state classifier (the ResNet of fasttrack/events.py) on a stack of frames cut around every point,
    `window` frames before and after, at a fixed centre (points you added have no track to follow).

Earlier event labels (Labels/events.csv, one cell each) are copied in once as state-only rows: they train
the classifier but not the seed finder, because the other nuclei near them were never labelled. Opening one
in the labeller and saving it turns it into a full patch.

Label check. `check_labels` splits the labels into folds and judges every label with models trained on the
other folds, so each judgement comes from a model that never saw that label. Where it disagrees with you,
either the model is weak there or the label is wrong: Labels/review.json lists both kinds for review.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from . import centres, events
from .device import resolve as resolve_device


# ---------------------------------------------------------------------- the dataset
def default_state(exp: dict) -> str:
    cats = exp["labelling"]["categories"]
    return cats[0] if cats else ""


def migrate_events(exp: dict, rows: list[dict]) -> tuple[list[dict], int]:
    """Copy Labels/events.csv into the patch file once, as state-only rows. Returns (rows, number added).
    events.csv itself is left as it is."""
    path = events.labels_path(exp)
    flag = Path(exp["paths"]["results"]) / "Labels" / ".events_migrated"
    if not path.exists() or flag.exists():
        return rows, 0
    size = int(exp["labelling"]["centres"]["patch"])
    added = 0
    for r in events.read_labels(exp):
        x, y = float(r["x"]), float(r["y"])
        if any(c.get("state_only") and c["position"] == r["position"] and c["t"] == r["t"] and c["points"] == [[round(x, 1), round(y, 1)]]
               for c in rows):
            continue
        rows.append({"position": r["position"], "t": int(r["t"]), "x0": int(round(x - size / 2)), "y0": int(round(y - size / 2)),
                     "size": size, "source": "events_csv", "change": 0.0, "points": [[round(x, 1), round(y, 1)]],
                     "states": [r["category"]], "tracked": [], "state_only": True, "track": int(r["track"]),
                     "labelled": r.get("labelled", "")})
        added += 1
    centres.write_labels(exp, rows)
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(f"events.csv copied into {exp['labelling']['centres']['labels_file']} on {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n")
    return rows, added


def read_dataset(exp: dict) -> list[dict]:
    """Every labelled patch (full and state-only), after copying in any earlier event labels."""
    rows = centres.read_labels(exp)
    return migrate_events(exp, rows)[0]


def cell_examples(exp: dict, rows: list[dict] | None = None) -> list[dict]:
    """One row per labelled point: {position, t, x, y, state, patch (index into rows), point (index in it), state_only}."""
    rows = read_dataset(exp) if rows is None else rows
    default = default_state(exp)
    out = []
    for i, r in enumerate(rows):
        for j, ((x, y), st) in enumerate(zip(r["points"], centres.point_states(r, default))):
            out.append({"position": r["position"], "t": int(r["t"]), "x": float(x), "y": float(y), "state": st,
                        "patch": i, "point": j, "state_only": bool(r.get("state_only"))})
    return out


def summary(exp: dict, rows: list[dict]) -> dict:
    s = centres.label_summary(rows, default_state(exp))
    s["categories"] = {c: s["states"].get(c, 0) for c in exp["labelling"]["categories"]}
    return s


# ---------------------------------------------------------------------- state classifier
def state_crops(exp: dict, cells: list[dict], timestacks: dict | None = None, on_event=lambda e: None) -> np.ndarray:
    """Normalised crop stacks (N, channels*frames, crop, crop), each frame read once per position."""
    lab = exp["labelling"]
    w, size = int(lab["window"]), int(lab["crop"])
    reader = events.FrameReader(exp, timestacks, int(exp["files"]["io_threads"]))
    out: list[np.ndarray | None] = [None] * len(cells)
    try:
        by_pos: dict[str, list[int]] = {}
        for i, c in enumerate(cells):
            by_pos.setdefault(c["position"], []).append(i)
        done = 0
        for key, idx in by_pos.items():
            T = reader._ts(key)["frames"]
            frames, _ = reader.frames(key, {min(max(cells[i]["t"] + d, 0), T - 1) for i in idx for d in range(-w, w + 1)})
            padded = events.pad_frames(frames, size)
            for i in idx:
                c = cells[i]
                out[i] = events.normalise_crop(events.crop_stack(padded, lambda f, c=c: (c["x"], c["y"]), c["t"], w, size, T))
                done += 1
            on_event({"kind": "crops", "done": done, "total": len(cells)})
            del frames, padded
    finally:
        reader.close()
    return np.stack(out) if out else np.zeros((0, 1, size, size), np.float32)


def _state_data(exp: dict, timestacks, on_event):
    lab = exp["labelling"]
    cells = [c for c in cell_examples(exp) if c["state"] in lab["categories"]]
    classes = [c for c in lab["categories"] if any(e["state"] == c for e in cells)]
    if len(classes) < 2:
        raise ValueError("Give at least two states before training the state classifier "
                         "(press a number key over a point; most points are the first state).")
    x = state_crops(exp, cells, timestacks, on_event)
    y = np.array([classes.index(c["state"]) for c in cells])
    return cells, classes, x, y


def train_states(exp: dict, on_event=lambda e: None, timestacks: dict | None = None) -> dict:
    """Train the state classifier on every labelled point and save it where Classify reads it. Returns the report."""
    lab = exp["labelling"]
    cells, classes, x, y = _state_data(exp, timestacks, on_event)
    device = resolve_device(exp["pipeline"]["device"])
    net, report = events.train_classifier(x, y, classes, int(lab["train_steps"]), device, seed=int(lab["seed"]),
                                          on_step=lambda s, l: on_event({"kind": "train", "phase": "states", "step": s,
                                                                         "steps": int(lab["train_steps"]), "loss": l}))
    report["trained"] = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
    report["cells"] = len(cells)
    events.save_model(events.model_path(exp), net, report, {**{k: lab[k] for k in ("window", "crop", "channels")}, "follow": False})
    with open(events.model_path(exp).with_suffix(".json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def read_state_report(exp: dict) -> dict | None:
    path = events.model_path(exp).with_suffix(".json")
    if not exp["paths"]["results"] or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def train_all(exp: dict, on_event=lambda e: None, timestacks: dict | None = None) -> dict:
    """Both models from the one dataset. A model that cannot train yet (too few labels) says why instead."""
    out = {}
    for name, fn in (("centres", centres.train_from_labels), ("states", train_states)):
        try:
            out[name] = fn(exp, on_event, timestacks)
        except ValueError as exc:
            out[name] = {"error": str(exc)}
    return out


# ---------------------------------------------------------------------- label check
def review_path(exp: dict) -> Path:
    return Path(exp["paths"]["results"]) / exp["labelling"]["review_file"]


def check_labels(exp: dict, on_event=lambda e: None, timestacks: dict | None = None, folds: int | None = None) -> dict:
    """Judge every label with models that never saw it (k-fold). Writes and returns the review:
    {made, folds, patches: [...from centres.cross_check], cells: [{position, t, x, y, state, predicted, p_state,
    p_predicted, probabilities, patch, point}]}. A part that cannot run yet (too few labels) says why in `errors`."""
    folds = int(folds or exp["labelling"]["folds"])
    review: dict = {"made": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}", "folds": folds, "patches": [], "cells": [], "errors": {}}
    try:
        review["patches"] = centres.cross_check(exp, folds, on_event, timestacks)
    except ValueError as exc:
        review["errors"]["centres"] = str(exc)
    try:
        lab = exp["labelling"]
        cells, classes, x, y = _state_data(exp, timestacks, on_event)
        device = resolve_device(exp["pipeline"]["device"])
        fold = np.random.default_rng(int(lab["seed"])).permutation(len(y)) % folds
        prob = np.zeros((len(y), len(classes)), np.float32)
        for k in range(folds):
            train = fold != k
            if len(np.unique(y[train])) < 2 or train.all():
                continue
            net, _ = events.train_classifier(x[train], y[train], classes, int(lab["train_steps"]), device, holdout=0.0, seed=int(lab["seed"]) + k,
                                             on_step=lambda s, l, k=k: on_event({"kind": "train", "phase": f"states, fold {k + 1} of {folds}",
                                                                                 "step": s, "steps": int(lab["train_steps"]), "loss": l}))
            prob[~train] = events.predict(net, x[~train], device)
            del net
        review["classes"] = classes
        for c, p, truth in zip(cells, prob, y):
            pred = int(p.argmax())
            review["cells"].append({**{k: c[k] for k in ("position", "t", "x", "y", "state", "patch", "point", "state_only")},
                                    "predicted": classes[pred], "p_state": round(float(p[truth]), 3), "p_predicted": round(float(p[pred]), 3),
                                    "probabilities": [round(float(v), 3) for v in p]})
    except ValueError as exc:
        review["errors"]["states"] = str(exc)
    path = review_path(exp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(review, f, indent=1)
    return review


def read_review(exp: dict) -> dict | None:
    path = review_path(exp)
    if not exp["paths"]["results"] or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def disagreements(review: dict, rows: list[dict]) -> list[dict]:
    """Review items worst first: patches where the model's centres differ from yours, and cells where its state
    differs. Each item: {kind: centres|state, patch (index into rows or None), text, score, edited (relabelled
    since the check), ...}."""
    keys = {centres.patch_key(r): i for i, r in enumerate(rows)}
    made = review.get("made", "")
    items = []
    for p in review.get("patches", []):
        if not p or not (p["missed"] or p["extra"]):
            continue
        i = keys.get(tuple(p["key"][:1]) + tuple(int(v) for v in p["key"][1:]))
        items.append({"kind": "centres", "patch": i, "model_points": p["model_points"], "score": p["missed"] + p["extra"],
                      "text": f"{p['key'][0]} frame {p['key'][1]}: model missed {p['missed']}, found {p['extra']} you did not",
                      "edited": i is not None and rows[i].get("labelled", "") > made})
    for c in review.get("cells", []):
        if c["predicted"] == c["state"]:
            continue
        i = c["patch"] if c["patch"] is not None and c["patch"] < len(rows) and rows[c["patch"]]["position"] == c["position"] else None
        items.append({"kind": "state", "patch": i, "point": c["point"], "score": c["p_predicted"], "x": c["x"], "y": c["y"],
                      "text": f"{c['position']} frame {c['t']}: you said {c['state']}, model says {c['predicted']} ({c['p_predicted']:.2f})",
                      "predicted": c["predicted"], "edited": i is not None and rows[i].get("labelled", "") > made})
    # centre disagreements first by size, then confident state disagreements
    items.sort(key=lambda d: (d["kind"] != "centres", -d["score"]))
    return items

