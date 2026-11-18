"""Which measurements detect undersegmented nuclei? AUC per feature, then a logistic fit.

A detection is positive when the centres of 2+ true nuclei fall inside it. Fits on some
synthetic movies at realistic density and reports on held-out ones. The fitted weights go in
fasttrack/track/undersegmentation.py. They reflect the synthetic model, so re-check on real data.

    python benchmarks/underseg_features.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="index_reduce")

from fasttrack.config import make_config
from fasttrack.synthetic.evaluate import match_detections, roc_auc
from fasttrack.synthetic.generate import generate
from fasttrack.track.batch import process_batch
from fasttrack.track.measure import K, S, SCALARS
from fasttrack.track.undersegmentation import score_features


def first_pass_table(seed, frames=100, cells=60, size=1024):
    data, truth, scfg = generate({"seed": seed, "frames": frames, "cells": cells, "size": size})
    result = process_batch(data, make_config({"underseg": {"enabled": False}}))
    sc = result["scalars"].cpu().numpy()
    ev = match_detections(sc[:, S["t"]], sc[:, S["x"]], sc[:, S["y"]], truth, size, gate=max(scfg["nuc_radius"] * 1.5, 8.0),
                          signed_labels=result["signed"].cpu().numpy(), det_labels=sc[:, S["label"]])
    fitted = {k: v.cpu().numpy() for k, v in score_features(result["scalars"], result["channels"]).items()}
    return sc, result["channels"].cpu().numpy(), ev["merged_rows"], fitted


def fit_logistic(x, y, steps=1500):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(np.nan_to_num(x), device=device, dtype=torch.float32)
    Y = torch.as_tensor(y, device=device, dtype=torch.float32)
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    pos_weight = torch.tensor([min(20.0, float((1 - Y).sum() / Y.sum().clamp(min=1)))], device=device)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, Y, pos_weight=pos_weight) + 1e-3 * w.abs().sum()
        loss.backward()
        opt.step()
    return w.detach().cpu().numpy(), float(b.item())


def main():
    tables = [first_pass_table(s) for s in range(1, 9)]
    sc = np.concatenate([t[0] for t in tables])
    ch = np.concatenate([t[1] for t in tables])
    merged = np.concatenate([t[2] for t in tables])
    print(f"detections {sc.shape[0]}, undersegmented {merged.sum()} ({100 * merged.mean():.1f}%)\n")

    rows = [(n, sc[:, S[n]]) for n in SCALARS if n not in ("t", "label", "x", "y", "underseg_score")]
    for name in ("bimodality", "chan_centroid_off", "nuc_std"):
        for c in range(ch.shape[1]):
            rows.append((f"{name}_c{c}", ch[:, c, K[name]]))
    rows.append(("nuc_sum_c0", ch[:, 0, K["nuc_sum"]]))
    print("feature             AUC   (|AUC-0.5| ranks usefulness; <0.5 = lower when merged)")
    aucs = sorted(((roc_auc(v, merged), n) for n, v in rows), key=lambda r: -abs(r[0] - 0.5) if r[0] == r[0] else 0)
    for auc, name in aucs:
        print(f"{name:20s}{auc:.3f}")

    names = list(tables[0][3].keys())
    train = list(range(6))
    x_train = np.concatenate([np.stack([tables[i][3][n] for n in names], 1) for i in train])
    y_train = np.concatenate([tables[i][2] for i in train])
    w, b = fit_logistic(x_train, y_train)
    print("\nweights:", {"bias": round(b, 3), **{n: round(float(v), 3) for n, v in zip(names, w)}})
    x_test = np.nan_to_num(np.concatenate([np.stack([tables[i][3][n] for n in names], 1) for i in (6, 7)]))
    score = 1 / (1 + np.exp(-(x_test @ w + b)))
    y_test = np.concatenate([tables[i][2] for i in (6, 7)])
    print(f"held-out AUC {roc_auc(score, y_test):.3f}")
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
        pick = score > thr
        print(f"  score>{thr}: flagged {pick.sum():6d}  precision {y_test[pick].mean() if pick.any() else 0:.3f}  recall {y_test[pick].sum() / max(y_test.sum(), 1):.3f}")


if __name__ == "__main__":
    main()
