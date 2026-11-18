"""Event classifier against synthetic ground truth, through the same path as real data.

    python benchmarks/event_classifier.py [settings.yaml]   # default: benchmarks/settings/event_classifier.yaml

1. Writes synthetic positions with deaths and divisions, and an experiment file for them.
2. Tracks them from that file.
3. Labels `per_class` candidates per category on the training positions from the ground truth,
   as a person would on the Labelling tab (division, death, neutral).
4. Trains the classifier from the labels file, classifies every candidate of every position.
5. Scores the held-out positions: precision and recall per category, and a confusion matrix.

Truth for a candidate (a track at a frame): the true cell nearest to it. Division if that cell
ends there and has children; death if it ends there early, away from the border, without
children; neutral otherwise.
"""
from __future__ import annotations

import pickle
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import load_settings  # noqa: E402  (benchmarks/settings.py)

from fasttrack import events  # noqa: E402
from fasttrack.experiment import new_experiment, save_experiment  # noqa: E402
from fasttrack.io import store  # noqa: E402
from fasttrack.synthetic.write import write_experiment  # noqa: E402
from fasttrack.track.run import run_experiment  # noqa: E402


def truth_label(truth: dict, t: int, x: float, y: float, size: int, T: int, reach: float = 8.0) -> str | None:
    at = np.nonzero(truth["t"] == t)[0]
    if not len(at):
        return None
    d = np.hypot(truth["x"][at] - x, truth["y"][at] - y)
    if d.min() > reach:
        return None
    cell = int(truth["cell"][at[np.argmin(d)]])
    track = truth["tracks"][cell]
    has_children = any(v["parent"] == cell for v in truth["tracks"].values())
    if track["end"] == t and has_children:
        return "division"
    margin = 20
    if track["end"] == t and t < T - 1 and margin < x < size - margin and margin < y < size - margin:
        return "death"
    return "neutral"


def main() -> None:
    args = load_settings(__file__, {"out": "runs/event_classifier", "train_positions": 3, "test_positions": 2, "frames": 120,
                                    "size": 512, "cells": 60, "death_rate": 0.004, "per_class": 40, "seed": 0})
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    n = args.train_positions + args.test_positions
    synth = {"frames": args.frames, "size": args.size, "cells": args.cells, "death_rate": args.death_rate, "stage_jumps": 0, "fibers": 0}
    print(f"writing {n} synthetic positions", flush=True)
    write_experiment(out / "raw", ["B02"], list(range(1, n + 1)), synth, device=args.cfg["device"])
    exp = new_experiment(paths={"raw": str(out / "raw"), "results": str(out / "results")},
                         plate={"conditions": [{"name": "synthetic", "wells": ["B02"]}]},
                         quality={"enabled": False}, pipeline={"device": args.cfg["device"]},
                         labelling={"categories": ["neutral", "division", "death"]})
    save_experiment(exp, out / "experiment.yaml")
    run_experiment(exp)
    train = [f"B02_xy{i}" for i in range(1, args.train_positions + 1)]
    test = [f"B02_xy{i}" for i in range(args.train_positions + 1, n + 1)]
    truths = {k: pickle.load(open(out / "raw" / f"truth_{k}.pkl", "rb"))["truth"] for k in train + test}

    # label from truth, as a person would: candidates by source, until per_class of each
    rng = random.Random(args.seed)
    pools: dict[str, list] = {c: [] for c in exp["labelling"]["categories"]}
    for key in train:
        r = store.load_results(out / "results", key)
        T = r["trace_index"].shape[1]
        for source, pairs in events.candidates(r).items():
            pairs = pairs if source != "random" else pairs[rng.sample(range(len(pairs)), min(300, len(pairs)))]
            for track, t in pairs:
                x, y = events.track_xy(r, int(track), int(t))
                label = truth_label(truths[key], int(t), x, y, args.size, T)
                if label:
                    pools[label].append({"position": key, "t": int(t), "track": int(track), "x": x, "y": y, "source": source, "category": label})
    rows = []
    for c, pool in pools.items():
        rng.shuffle(pool)
        rows += pool[:args.per_class]
        print(f"  {c}: {len(pool)} candidates in training positions, labelled {min(len(pool), args.per_class)}")
    events.write_labels(exp, rows)
    report = events.train_from_labels(exp, on_event=lambda e: e["kind"] == "train" and e["step"] % 200 == 0 and print(f"  step {e['step']} loss {e['loss']:.3f}", flush=True))
    print(f"held-out accuracy within the labels: {report.get('holdout_accuracy', float('nan')):.3f} on {report.get('holdout_n', 0)} crops")
    events.classify_positions(exp)

    classes = report["classes"]
    confusion = np.zeros((len(classes), len(classes)), int)
    for key in test:
        r = store.load_results(out / "results", key)
        T = r["trace_index"].shape[1]
        for row in events.read_events(out / "results", key):
            label = truth_label(truths[key], row["t"], float(row["x"]), float(row["y"]), args.size, T)
            if label in classes:
                confusion[classes.index(label), classes.index(row["category"])] += 1
    print("\nheld-out positions, track ends / starts / divisions classified (rows true, columns predicted):")
    print(f"{'':>10s}" + "".join(f"{c:>10s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>10s}" + "".join(f"{v:10d}" for v in confusion[i]))
    for i, c in enumerate(classes):
        recall = confusion[i, i] / max(confusion[i].sum(), 1)
        precision = confusion[i, i] / max(confusion[:, i].sum(), 1)
        print(f"  {c:9s} recall {recall:.2f}  precision {precision:.2f}  (n = {confusion[i].sum()})")


if __name__ == "__main__":
    main()
