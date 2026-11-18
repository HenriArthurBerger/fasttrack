"""Batch throughput at the acquisition sizes that matter, against the 30 s per 100 frames goal.

    python benchmarks/speed.py
"""
import gc
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from fasttrack.config import make_config
from fasttrack.synthetic.generate import DENSE_PRESET, generate
from fasttrack.track.batch import process_batch
from fasttrack.track.linking import link_tracks


def main():
    name = torch.cuda.get_device_name(0)
    print(f"{name}: 100 frames, 3 channels")
    for label, params in (("1024 px, mosaic", {"size": 1024}), ("2048 px, mosaic", {"size": 2048, "cells": 140}),
                          ("1024 px, dense stress", {"size": 1024, **DENSE_PRESET})):
        frames, _, _ = generate({"frames": 100, **params})
        cfg = make_config()
        process_batch(frames[:4], cfg)  # warm up allocations, so the timed run is steady state
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r = process_batch(frames, cfg)
        batch = time.perf_counter() - t0
        t0 = time.perf_counter()
        links = link_tracks(r["scalars"], r["channels"], r["radius"], cfg, r["frame_shift"])
        link = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{label:24s} {batch:6.2f}s batch  {link:5.2f}s link  {peak:5.1f} GB peak  {r['stats']['cells']:7d} cells  {links['tracks']:6d} tracks  "
              + " ".join(f"{k}={v:.2f}" for k, v in r["timings"].items()), flush=True)
        del r, frames, links
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
