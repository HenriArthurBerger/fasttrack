"""Timestack and experiment orchestration.

Double-buffered: while the GPU processes one batch a thread reads the next, so the slower of
disk and GPU sets the pace, not their sum. Progress goes out through one `on_event` callback,
so the same code serves the CLI and the GUI.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..config import save_config
from ..io import layout, store
from .batch import process_batch
from .linking import link_tracks, trace_index
from .measure import S

Event = Callable[[dict], None]


def _noop(_: dict) -> None:
    pass


def process_timestack(timestack: dict, cfg: dict, on_event: Event = _noop, stop: threading.Event | None = None,
                      gui_cache: dict | None = None) -> dict | None:
    """Measure and link one timestack. Returns a summary, or None if stopped."""
    device = torch.device(cfg["device"])
    frames = timestack["frames"]
    ranges = layout.batch_ranges(frames, int(cfg["batch_size"]))
    io_pool = ThreadPoolExecutor(int(cfg["data"]["io_threads"]), thread_name_prefix="tiff")
    read_ahead = ThreadPoolExecutor(1, thread_name_prefix="readahead")
    scalars_all, channels_all, radii = [], [], []
    next_label = 0
    timings: dict[str, float] = {}
    started = time.perf_counter()
    wait_io = 0.0
    try:
        pending = read_ahead.submit(layout.load_batch, timestack, *ranges[0], io_pool)
        for b, (start, end) in enumerate(ranges):
            if stop is not None and stop.is_set():
                pending.cancel()
                return None
            t0 = time.perf_counter()
            raw = pending.result()
            wait_io += time.perf_counter() - t0
            if b + 1 < len(ranges):
                pending = read_ahead.submit(layout.load_batch, timestack, *ranges[b + 1], io_pool)
            tensor = torch.from_numpy(raw).to(device, non_blocking=True)  # stays uint16 until corrected

            def on_layer(name, value, offset=start):
                if gui_cache is not None and name in cfg["output"]["gui_cache_layers"]:
                    store.add_layer(gui_cache, name, value, offset)

            result = process_batch(tensor, cfg, on_layer if gui_cache is not None else None)
            sc = result["scalars"].cpu().numpy()
            sc[:, S["t"]] += start
            sc[:, S["label"]] += next_label
            next_label = int(np.nanmax(sc[:, S["label"]])) if sc.shape[0] else next_label
            scalars_all.append(sc)
            channels_all.append(result["channels"].cpu().numpy())
            radii.append(result["radius"])
            for key, value in result["timings"].items():
                timings[key] = timings.get(key, 0.0) + value
            on_event({"kind": "batch", "timestack": timestack["id"], "batch": b + 1, "batches": len(ranges),
                      "cells": sc.shape[0], "radius": result["radius"], "timings": result["timings"], "stats": result["stats"]})
            del result, tensor
    finally:
        io_pool.shutdown(wait=False, cancel_futures=True)
        read_ahead.shutdown(wait=False, cancel_futures=True)

    scalars = np.concatenate(scalars_all)
    channels = np.concatenate(channels_all)
    radius = float(np.median(radii)) if radii else 0.0
    t0 = time.perf_counter()
    with torch.inference_mode():
        links = link_tracks(torch.as_tensor(scalars, device=device), torch.as_tensor(channels, device=device), radius, cfg)
    timings["link"] = time.perf_counter() - t0
    index = trace_index(links["track_id"], scalars[:, S["t"]], frames)
    meta = {"id": timestack["id"], "well": timestack["well"], "pos": timestack["pos"], "frames": frames, "radius": radius}
    if cfg["output"]["save_measurements"]:
        out = Path(cfg["data"]["output"])
        store.save_measurements(out / "Measurements" / f"{timestack['id']}.h5", scalars, channels, meta)
        store.save_traces(out / "Traces" / f"{timestack['id']}.h5", links, index, meta)
    elapsed = time.perf_counter() - started
    summary = {"id": timestack["id"], "frames": frames, "cells": scalars.shape[0], "tracks": links["tracks"],
               "divisions": links["divisions"], "radius": radius, "seconds": elapsed, "wait_io": wait_io, "timings": timings}
    on_event({"kind": "timestack_done", **summary})
    return summary


def process_experiment(cfg: dict, on_event: Event = _noop, stop: threading.Event | None = None, select: list[str] | None = None) -> list[dict]:
    """Process every timestack under data.root. A finished timestack is skipped on re-run."""
    timestacks = layout.discover_timestacks(cfg)
    if select:
        timestacks = {k: v for k, v in timestacks.items() if k in select}
    output = Path(cfg["data"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output / "Experiment_Config.yaml")
    on_event({"kind": "experiment_start", "timestacks": list(timestacks)})
    summaries = []
    cache = None
    for i, (key, timestack) in enumerate(timestacks.items()):
        if stop is not None and stop.is_set():
            break
        if (output / "Traces" / f"{key}.h5").exists() and not cfg["output"].get("overwrite", False):
            on_event({"kind": "timestack_skipped", "id": key})
            continue
        on_event({"kind": "timestack_start", "id": key, "index": i, "total": len(timestacks)})
        store.close_gui_cache(cache)
        cache = store.new_gui_cache(key) if cfg["output"]["gui_cache"] else None
        if cache is not None:
            on_event({"kind": "gui_cache", "cache": cache})
        summary = process_timestack(timestack, cfg, on_event, stop, cache)
        if summary is not None:
            summaries.append(summary)
    on_event({"kind": "experiment_done", "summaries": summaries})
    return summaries
