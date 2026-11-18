"""Train the ResNet seed finder on a few dozen labelled frames and compare it with distance seeds.

Trains on `--label-frames` frames of one synthetic movie and tests on an unseen one. Then runs
the checkpoint on real data, reporting agreement with classical seeds. That is agreement, not
accuracy: real data has no ground truth.

    python benchmarks/centroid_net.py [settings.yaml]   # default: benchmarks/settings/centroid_net.yaml

The real data comes from the experiment file named in the settings (optional).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import load_settings  # noqa: E402  (benchmarks/settings.py)
warnings.filterwarnings("ignore")

from fasttrack.config import make_config
from fasttrack.io import layout
from fasttrack.synthetic.evaluate import match_detections
from fasttrack.synthetic.generate import generate
from fasttrack.track import background, centroid_net, centroids, masking
from concurrent.futures import ThreadPoolExecutor


def seeds_f1(corrected, truth, scfg, cfg):
    mask, _ = masking.get_binary_mask(corrected, cfg)
    radius = masking.compute_radius(mask)
    if cfg["centroids"]["method"] == "cnn":
        peaks = centroid_net.predict_peaks(corrected, mask, radius, cfg)
    else:
        peaks, _ = centroids.get_centroids(mask, radius, cfg)
    p = peaks.cpu().numpy().astype(float)
    ev = match_detections(p[:, 0], p[:, 1], p[:, 2], truth, scfg["size"], gate=max(scfg["nuc_radius"] * 1.5, 8.0))
    return ev["precision"], ev["recall"], ev["f1"], peaks.shape[0]


def main():
    args = load_settings(__file__, {"out": "runs/seednet.pt", "label_frames": 40, "steps": 800})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = make_config()
    train_data, truth_a, scfg_a = generate({"seed": 50, "frames": args.label_frames, "cells": 80})
    with torch.inference_mode():
        corrected_a, _ = background.correct_background(train_data.float(), cfg)
        x = centroid_net.normalize(corrected_a[:, 0:1])
    keep = (truth_a["x"] > 0) & (truth_a["x"] < 1024) & (truth_a["y"] > 0) & (truth_a["y"] < 1024)
    points = torch.as_tensor(np.stack([truth_a["t"][keep], truth_a["x"][keep], truth_a["y"][keep]], 1), device=x.device, dtype=torch.float32)
    print(f"training on {args.label_frames} frames, {points.shape[0]} labelled nuclei")
    t0 = time.perf_counter()
    net = centroid_net.train(x.clone(), points, steps=args.steps,
                             on_step=lambda s, l: print(f"  step {s} loss {l:.3f}", flush=True) if s % 200 == 0 else None)
    print(f"trained in {time.perf_counter() - t0:.0f}s, {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters")
    centroid_net.save_checkpoint(net, args.out)

    test_cfg = make_config({"centroids": {"method": "cnn", "cnn_checkpoint": str(args.out)}})
    for seed in (60, 61):
        data, truth, scfg = generate({"seed": seed, "frames": 100})
        with torch.inference_mode():
            corrected, _ = background.correct_background(data.float(), cfg)
            t0 = time.perf_counter()
            a = seeds_f1(corrected, truth, scfg, cfg)
            t1 = time.perf_counter()
            b = seeds_f1(corrected, truth, scfg, test_cfg)
            t2 = time.perf_counter()
        print(f"seed {seed}: distance P/R/F1 {a[0]:.3f} {a[1]:.3f} {a[2]:.3f} ({t1 - t0:.2f}s) | cnn {b[0]:.3f} {b[1]:.3f} {b[2]:.3f} ({t2 - t1:.2f}s per 100 frames)")

    if args.raw is not None:
        rcfg = args.cfg
        stacks = layout.discover_timestacks(rcfg)
        with ThreadPoolExecutor(16) as pool:
            for key, ts in stacks.items():
                raw = layout.load_batch(ts, 0, ts["frames"], pool)
                with torch.inference_mode():
                    corrected, _ = background.correct_background(torch.from_numpy(raw).to(rcfg["device"]), rcfg)
                    mask, _ = masking.get_binary_mask(corrected, rcfg)
                    radius = masking.compute_radius(mask)
                    classic, _ = centroids.get_centroids(mask, radius, rcfg)
                    learned = centroid_net.predict_peaks(corrected, mask, radius, test_cfg)
                c, l = classic.cpu().numpy().astype(float), learned.cpu().numpy().astype(float)
                both = 0
                for t in np.unique(c[:, 0]):
                    a, b = c[c[:, 0] == t, 1:], l[l[:, 0] == t, 1:]
                    if a.size and b.size:
                        both += int((np.linalg.norm(a[:, None] - b[None], axis=2).min(1) < 8).sum())
                print(f"{key}: classic {c.shape[0]} cnn {l.shape[0]}; classic seeds with a cnn seed within 8 px: {both / max(c.shape[0], 1):.3f}")


if __name__ == "__main__":
    main()
