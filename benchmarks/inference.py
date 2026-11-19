"""How much slower is tracking with learned models than with the classical pipeline? Budget: at most 2x.

    python benchmarks/inference.py experiment.yaml [--checkpoint Labels/centre_model.pt] [--positions 2] [--frames 100]

Reads `--frames` frames of `--positions` real positions into memory once (disk time is the same for every
method, so it is left out), then times `process_batch` on them with:

  classical      centroids.method = notebook (LoG mask, distance peaks)
  seed finder    centroids.method = cnn with the checkpoint (the learned seeds; mask still classical)

and the network alone at the real frame size, for the variants that decide whether a fully learned pipeline
(seeds and masks from one encoder) fits the budget:

  every preset in track/centroid_net.ARCHITECTURES
  + mask head          resnet18-half with a second output (a foreground mask): what a learned mask adds

Each run appends one row per method to benchmarks/results/inference.csv (date, commit, GPU, frame size,
seconds per 100 frames, ratio to classical), so a slowdown shows up in the history. Exits with status 1
if the seed finder is over budget.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from fasttrack.device import resolve
from fasttrack.experiment import load_experiment, pipeline_config
from fasttrack.io import layout
from fasttrack.track import centroid_net
from fasttrack.track.batch import process_batch

BUDGET = 2.0
RESULTS = ROOT / "benchmarks" / "results" / "inference.csv"
COLUMNS = ["date", "commit", "gpu", "frame", "channels", "frames", "method", "seconds_per_100", "ratio", "stages"]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(fn, device, repeat: int = 3) -> float:
    fn()                                   # warm up: allocations, cudnn algorithm choice, model load
    _sync(device)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        best = min(best, time.perf_counter() - t0)
    return best


class SeedMaskNet(centroid_net.SeedNet):
    """SeedNet with two outputs: the centre map and a foreground mask, from one encoder and decoder."""

    def __init__(self, stride: int = centroid_net.STRIDE):
        super().__init__(stride)
        self.up1[-1] = nn.Conv2d(32, 2, 1)


def _net_time(net, x, device, chunk: int, channels_last: bool = False) -> float:
    net = net.to(device).eval()
    if channels_last:
        net = net.to(memory_format=torch.channels_last)

    def run():
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            for s in range(0, x.shape[0], chunk):
                part = x[s:s + chunk]
                if channels_last:
                    part = part.contiguous(memory_format=torch.channels_last)
                torch.sigmoid(net(part))
    return _time(run, device)


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "-"
    except OSError:
        return "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("experiment")
    ap.add_argument("--checkpoint", default="", help="seed finder; default: the experiment's Labels/centre_model.pt")
    ap.add_argument("--positions", type=int, default=2)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--device", default="")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    exp = load_experiment(args.experiment)
    cfg = pipeline_config(exp)
    device = torch.device(resolve(args.device or cfg["device"]))
    cfg["device"] = str(device)
    ckpt = args.checkpoint or str(Path(exp["paths"]["results"]) / exp["labelling"]["centres"]["model_file"])
    if not Path(ckpt).exists():
        print(f"No seed finder at {ckpt}: train one on the Label cells page, or pass --checkpoint.")
        return 2
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    stacks = layout.discover_timestacks(cfg)
    keys = [k for k in sorted(stacks) if stacks[k]["frames"] >= args.frames][:args.positions]
    print(f"{gpu}; positions {', '.join(keys)}; {args.frames} frames each, read once into memory")
    with ThreadPoolExecutor(int(cfg["data"]["io_threads"])) as pool:
        batches = [layout.load_batch(stacks[k], 0, args.frames, pool) for k in keys]
    T, C, X, Y = batches[0].shape
    n = len(batches) * T

    rows, base = [], None
    for method, extra in (("classical", {"method": "notebook"}), ("seed finder", {"method": "cnn", "cnn_checkpoint": ckpt})):
        c = {**cfg, "centroids": {**cfg["centroids"], **extra}}
        stages: dict[str, float] = {}

        def run():
            stages.clear()
            for b in batches:
                r = process_batch(torch.from_numpy(b).to(device), c)
                for k, v in r["timings"].items():
                    stages[k] = stages.get(k, 0.0) + v
                del r
        sec = _time(run, device, repeat=2) * 100 / n
        base = base or sec
        if method == "seed finder":
            method += f", {centroid_net.load_checkpoint(ckpt, device).arch}"
        rows.append((method, sec, {k: round(v * 100 / n, 2) for k, v in stages.items()}))
        print(f"  {method}: {sec:.2f} s / 100 frames", flush=True)
        gc.collect()
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # the network alone on the nuclear channel at full frame size
    x = centroid_net.normalize(torch.from_numpy(batches[0][:, cfg["nuclear_channel"]:cfg["nuclear_channel"] + 1].astype(np.float32)).to(device))
    nets = [(f"net alone, {name}", centroid_net.build(name)) for name in centroid_net.ARCHITECTURES]
    for method, net in nets + [("net alone, resnet18-half + mask head", SeedMaskNet(2))]:
        chunk = max(1, centroid_net.CHUNK_PIXELS * net.stride ** 2 // (X * Y))
        try:
            sec = _net_time(net, x, device, chunk, True) * 100 / T
        except torch.OutOfMemoryError:
            print(f"  {method}: out of memory", flush=True)
            continue
        finally:
            del net
            gc.collect()
            torch.cuda.empty_cache() if device.type == "cuda" else None
        rows.append((method, sec, {}))
        print(f"  {method}: {sec:.2f} s / 100 frames", flush=True)

    print(f"\n{'method':40s} {'s / 100 frames':>15s} {'x classical':>12s}   stages (s / 100 frames)")
    for method, sec, stages in rows:
        print(f"{method:40s} {sec:15.2f} {sec / base:12.2f}   " + " ".join(f"{k}={v}" for k, v in stages.items()))
    seed = rows[1][1] / base
    verdict = "within" if seed <= BUDGET else "OVER"
    print(f"\nSeed finder: {seed:.2f}x classical, {verdict} the {BUDGET:.0f}x budget.")

    if not args.no_save:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        new = not RESULTS.exists()
        with open(RESULTS, "a", newline="") as f:
            w = csv.DictWriter(f, COLUMNS)
            if new:
                w.writeheader()
            for method, sec, stages in rows:
                w.writerow({"date": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "commit": _commit(), "gpu": gpu, "frame": f"{X}x{Y}",
                            "channels": C, "frames": n, "method": method, "seconds_per_100": round(sec, 3),
                            "ratio": round(sec / base, 3), "stages": " ".join(f"{k}={v}" for k, v in stages.items())})
        print(f"Saved to {RESULTS.relative_to(ROOT)}")
    return 0 if seed <= BUDGET else 1


if __name__ == "__main__":
    sys.exit(main())
