"""Timestack and experiment orchestration.

Two threads: one loads frames, and the caller's thread launches tracking on each batch as soon
as it is fully loaded. A bounded queue between them means the slower of disk and GPU sets the
pace, not their sum. Stage shifts chain across batches, so all are referenced to frame 0.

Progress goes out through one `on_event` callback, so the same code serves the CLI and the GUI.
"""
from __future__ import annotations

import gc
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ..config import save_config
from ..device import resolve as resolve_device
from ..io import layout, store
from .batch import process_batch
from .linking import link_tracks, trace_index
from .measure import S

Event = Callable[[dict], None]
MIN_TAIL = 20  # a batch this short gives poor background and radius estimates
MIN_SPLIT = 10  # below this, an out-of-memory error is real and is raised


def _noop(_: dict) -> None:
    pass


def _next_range(start: int, size: int, frames: int) -> tuple[int, int]:
    """Next batch, absorbing a short tail so a final batch is never only a few frames."""
    stop = min(start + size, frames)
    if frames - stop < MIN_TAIL:
        stop = frames
    return start, stop


def _load_loop(reader, frames: int, plan: dict, out: queue.Queue, stop: threading.Event) -> None:
    """Loader thread: read batches, queue each when complete, and pass on any failure."""
    start = 0
    try:
        while start < frames and not stop.is_set():
            begin, end = _next_range(start, plan["size"], frames)
            t0 = time.perf_counter()
            raw = reader(begin, end)
            item = ("batch", begin, end, raw, time.perf_counter() - t0)
            while not stop.is_set():
                try:
                    out.put(item, timeout=0.2)
                    break
                except queue.Full:
                    pass
            start = end
    except Exception as exc:  # re-raised on the tracking thread, where it can be reported
        out.put(("error", exc))
        return
    out.put(("done",))


def _grow_batch(plan: dict, frames_done: int, device: torch.device, cfg: dict) -> None:
    """Raise the batch size if the last batch's peak memory says a larger one fits."""
    if not cfg["batch_auto"] or device.type != "cuda" or frames_done == 0:
        return
    per_frame = torch.cuda.max_memory_allocated(device) / frames_done
    total = torch.cuda.get_device_properties(device).total_memory
    fits = int(cfg["vram_fraction"] * total / per_frame)
    plan["size"] = max(plan["size"], min(int(cfg["batch_max"]), fits // 25 * 25))


def _process_with_retry(tensor, cfg, on_layer, previous_mask, plan, on_part):
    """Process a batch. If the card runs out of memory, split it in half and carry on.

    The retry happens after leaving the except block: a caught exception keeps the failed
    attempt's tensors alive, so retrying inside it would start with the memory still full.
    """
    failed = False
    try:
        result = process_batch(tensor, cfg, on_layer, previous_mask)
    except torch.OutOfMemoryError:
        failed = True
    if failed:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        T = tensor.shape[0]
        if T < MIN_SPLIT * 2:
            raise torch.OutOfMemoryError(f"out of GPU memory even on a batch of {T} frames")
        plan["size"] = max(T // 2, MIN_SPLIT)
        plan["auto_stopped"] = True
        first = _process_with_retry(tensor[: T // 2], cfg, on_layer, previous_mask, plan, on_part)
        return _process_with_retry(tensor[T // 2:], cfg, on_layer, first, plan, on_part)
    on_part(result, tensor.shape[0])
    last = result["mask"][-1, 0].float().clone()
    del result
    return last


def process_timestack(timestack: dict, cfg: dict, on_event: Event = _noop, stop: threading.Event | None = None,
                      cache: dict | None = None, reader=None) -> dict | None:
    """Measure and link one timestack. Returns a summary, or None if stopped.

    `reader(start, stop)` returns (T,C,X,Y) frames. By default it reads the timestack's TIFFs.
    """
    device = torch.device(resolve_device(cfg["device"]))
    frames = timestack["frames"]
    stop = stop or threading.Event()
    plan = {"size": int(cfg["batch_size"]), "auto_stopped": False}
    io_pool = ThreadPoolExecutor(int(cfg["data"]["io_threads"]), thread_name_prefix="tiff")
    if reader is None:
        reader = lambda a, b: layout.load_batch(timestack, a, b, io_pool)
    ready: queue.Queue = queue.Queue(maxsize=max(1, int(cfg["prefetch_batches"])))
    halt = threading.Event()
    loader = threading.Thread(target=_load_loop, args=(reader, frames, plan, ready, halt), name="frame-loader", daemon=True)

    scalars_all, channels_all, shifts_all, radii = [], [], [], []
    timings: dict[str, float] = {}
    started = time.perf_counter()
    wait_loader = read_seconds = 0.0
    frames_done = 0
    previous_mask = None
    batch_sizes = []
    pixels = 1
    bytes_read = 0

    def on_layer(name, value, offset=None):
        if cache is not None and name in cache["layers_wanted"]:
            store.add_layer(cache, name, value, cursor["start"])

    cursor = {"start": 0}

    def on_part(result, count):
        nonlocal frames_done
        sc = result["scalars"].cpu().numpy()
        sc[:, S["t"]] += cursor["start"]
        scalars_all.append(sc)
        channels_all.append(result["channels"].cpu().numpy())
        shifts_all.append(result["frame_shift"].cpu())
        radii.append(result["radius"])
        for key, value in result["timings"].items():
            timings[key] = timings.get(key, 0.0) + value
        batch_sizes.append(count)
        frames_done += count
        on_event({"kind": "batch", "timestack": timestack["id"], "start": cursor["start"], "stop": cursor["start"] + count,
                  "frames_done": frames_done, "frames_total": frames, "cells": sc.shape[0], "radius": result["radius"],
                  "timings": result["timings"], "stats": result["stats"]})
        cursor["start"] += count

    loader.start()
    try:
        while True:
            if stop.is_set():
                return None
            t0 = time.perf_counter()
            try:
                item = ready.get(timeout=0.2)
            except queue.Empty:
                wait_loader += time.perf_counter() - t0
                continue
            wait_loader += time.perf_counter() - t0
            if item[0] == "done":
                break
            if item[0] == "error":
                raise item[1]
            _, begin, end, raw, seconds = item
            pixels = raw.shape[-2] * raw.shape[-1]
            read_seconds += seconds
            bytes_read += raw.nbytes
            cursor["start"] = begin
            tensor = torch.from_numpy(raw).to(device, non_blocking=True)  # stays uint16 until corrected
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            previous_mask = _process_with_retry(tensor, cfg, on_layer if cache is not None else None, previous_mask, plan, on_part)
            del tensor
            if not plan["auto_stopped"]:
                _grow_batch(plan, end - begin, device, cfg)
    finally:
        halt.set()
        io_pool.shutdown(wait=False, cancel_futures=True)

    if not scalars_all:
        return None
    scalars = np.concatenate(scalars_all)
    channels = np.concatenate(channels_all)
    frame_shift = torch.cat(shifts_all)
    radius = float(np.median(radii))
    per_frame = np.bincount(scalars[:, S["t"]].astype(int), minlength=frames)[:frames]
    area = np.bincount(scalars[:, S["t"]].astype(int), weights=scalars[:, S["nuc_area"]], minlength=frames)[:frames]
    density = {"cells_per_frame": float(np.median(per_frame)), "empty_frames": int((per_frame == 0).sum()),
               "nuclear_coverage": float(np.median(area) / pixels)}
    t0 = time.perf_counter()
    with torch.inference_mode():
        links = link_tracks(torch.as_tensor(scalars, device=device), torch.as_tensor(channels, device=device), radius, cfg, frame_shift)
    timings["link"] = time.perf_counter() - t0
    index = trace_index(links["track_id"], scalars[:, S["t"]], frames)
    meta = {"id": timestack["id"], "well": timestack["well"], "pos": timestack["pos"], "frames": frames, "radius": radius}
    if cfg["output"]["save_measurements"]:
        out = Path(cfg["data"]["output"])
        store.save_measurements(out / "Measurements" / f"{timestack['id']}.h5", scalars, channels, meta)
        store.save_traces(out / "Traces" / f"{timestack['id']}.h5", links, index, meta)
    summary = {"id": timestack["id"], "frames": frames, "cells": scalars.shape[0], "tracks": links["tracks"], "divisions": links["divisions"],
               "max_travel": links.get("max_travel", 0.0), "radius": radius, **density, "seconds": time.perf_counter() - started, "wait_io": wait_loader,
               "read_seconds": read_seconds, "batch_sizes": batch_sizes, "timings": timings,
               "peak_gpu_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
               "bytes_read": int(bytes_read), "stage_steps": _stage_steps(links.get("drift")),
               "unreadable": list(timestack.get("unreadable", [])), "retried": list(timestack.get("retried", [])),
               "missing": list(timestack.get("missing", [])), "short_by": int(timestack.get("short_by", 0))}
    on_event({"kind": "timestack_done", **summary})
    if cfg["output"].get("return_tables"):
        summary["tables"] = {"scalars": scalars, "channels": channels, "frame_shift": frame_shift.numpy(), "links": links, "index": index}
    return summary


def _stage_steps(drift) -> list[tuple[int, float]]:
    """(frame, jump in px) wherever the corrected stage position changes between frames."""
    if drift is None:
        return []
    d = np.asarray(drift.cpu() if hasattr(drift, "cpu") else drift, dtype=float)
    if len(d) < 2:
        return []
    jumps = np.hypot(*np.diff(d, axis=0).T)
    return [(int(i) + 1, round(float(jumps[i]), 1)) for i in np.nonzero(jumps > 0)[0]]


def process_experiment(cfg: dict, on_event: Event = _noop, stop: threading.Event | None = None, select: list[str] | None = None,
                       timestacks: dict | None = None, keep_going: bool = False) -> list[dict]:
    """Process every timestack under data.root. A finished timestack is skipped on re-run.
    With keep_going, a timestack that raises is reported as `timestack_failed` and the others still run."""
    timestacks = timestacks if timestacks is not None else layout.discover_timestacks(cfg)
    if select is not None:
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
        on_event({"kind": "timestack_start", "id": key, "index": i, "total": len(timestacks), "problems": timestack.get("problems", [])})
        store.close_gui_cache(cache)
        cache = store.new_gui_cache(key, cfg["output"]["gui_cache_layers"]) if cfg["output"].get("gui_cache") else None
        if cache is not None:
            on_event({"kind": "gui_cache", "cache": cache})
        try:
            summary = process_timestack(timestack, cfg, on_event, stop, cache)
        except Exception as exc:  # one bad position must not cost the rest of the plate
            if not keep_going:
                raise
            import traceback
            on_event({"kind": "timestack_failed", "id": key, "message": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        if cache is not None:
            store.flush_cache(cache)
        if summary is not None:
            summaries.append(summary)
    on_event({"kind": "experiment_done", "summaries": summaries})
    return summaries


class RunLog:
    """The two logs of a run, in <results>/Logs/.

    <time>_detailed.log   every event, written as it happens (settings, batches, timings, warnings, tracebacks)
    <time>_summary.txt    only what a person should look at, written at the end (also when a run fails or stops)
    latest_summary.txt    a copy of the last summary
    """

    def __init__(self, results: Path, exp: dict, on_event: Event):
        import datetime as dt
        import platform
        self.started = dt.datetime.now()
        folder = results / "Logs"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = f"{self.started:%Y-%m-%d_%H%M%S}"
        self.detailed_path = folder / f"{stamp}_detailed.log"
        self.summary_path = folder / f"{stamp}_summary.txt"
        self.latest = folder / "latest_summary.txt"
        self.on_event = on_event
        self.exp = exp
        self.checks: dict[str, list[str]] = {}       # section -> lines to check by hand
        self.messages: list[str] = []                # every warning, for run_log.yaml
        self.done: list[dict] = []
        self.failed: list[str] = []
        self.skipped: list[str] = []
        self.selected: list[str] = []
        self.flagged: dict = {}
        self.status = "running"
        self._file = open(self.detailed_path, "a", buffering=1)
        from .. import __version__
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        self.write(f"FastTrack {__version__}, Python {platform.python_version()}, torch {torch.__version__}, GPU {gpu}")
        self.write(f"experiment: {exp['experiment']['name'] or '(unnamed)'}")
        self.write(f"raw data: {exp['paths']['raw']}")
        self.write(f"results: {exp['paths']['results']}")

    def write(self, line: str) -> None:
        import datetime as dt
        self._file.write(f"{dt.datetime.now():%H:%M:%S}  {line}\n")

    def check(self, section: str, line: str) -> None:
        self.checks.setdefault(section, []).append(line)

    def warn(self, message: str, timestack: str = "", section: str = "Warnings") -> None:
        line = f"{timestack}: {message}" if timestack else message
        self.messages.append(line)
        self.write(f"WARNING {line}")
        self.check(section, line)
        self.on_event({"kind": "warning", "message": line})

    def event(self, e: dict) -> None:
        """Record any pipeline event in the detailed log; pick out what needs a person."""
        kind = e["kind"]
        if kind == "batch":
            stages = " ".join(f"{k}={v:.2f}s" for k, v in e["timings"].items())
            self.write(f"{e['timestack']} frames {e['start']}-{e['stop'] - 1} ({e['frames_done']}/{e['frames_total']}): "
                       f"{e['cells']} cells, radius {e['radius']:.1f} px | {stages}")
        elif kind == "timestack_start":
            self.write(f"{e['id']}: start ({e['index'] + 1} of {e['total']})")
            for problem in e.get("problems", []):
                self.write(f"{e['id']}: {problem}")
                self.check("Missing files or short movies (tracked: blank frames, tracks bridge them; consider excluding)",
                           f"{e['id']}: {problem}")
        elif kind == "timestack_done":
            self.done.append(e)
            mb = e.get("bytes_read", 0) / 1e6
            self.write(f"{e['id']}: done in {e['seconds']:.1f}s: {e['frames']} frames, {e['cells']} cells, {e['tracks']} tracks, "
                       f"{e['divisions']} divisions, gate {e['max_travel']:.1f} px, radius {e['radius']:.1f} px, "
                       f"{e['cells_per_frame']:.0f} cells/frame, coverage {100 * e['nuclear_coverage']:.1f}%, "
                       f"read {mb:.0f} MB in {e['read_seconds']:.1f}s, GPU waited {e['wait_io']:.1f}s for the disk, "
                       f"batches {e['batch_sizes']}, peak GPU {e['peak_gpu_gb']:.1f} GB")
            for bad in e.get("unreadable", []):
                self.write(f"{e['id']}: {bad['error']} -> frame {bad['frame']} channel {bad['channel']} left blank after 3 tries")
                self.check("Unreadable files (corrupt or truncated; left blank after 3 tries)", f"{e['id']}: frame {bad['frame']}, {bad['path']}")
            for again in e.get("retried", []):
                self.write(f"{e['id']}: frame {again['frame']} channel {again['channel']} read on try {again['tries']}: {again['path']}")
                self.check("Files that only read on a second or third try (the disk may be failing; copy the data soon)",
                           f"{e['id']}: frame {again['frame']}, {again['path']}")
            steps = e.get("stage_steps", [])
            if steps:
                self.write(f"{e['id']}: stage steps (frame, px): {steps}")
                big = [(f, j) for f, j in steps if j >= 2 * max(e["radius"], 1)]
                if big:
                    self.check("Large stage jumps (check the shift correction in a test run)",
                               f"{e['id']}: " + ", ".join(f"frame {f}: {j:.0f} px" for f, j in big[:8]) + (" ..." if len(big) > 8 else ""))
        elif kind == "timestack_failed":
            self.failed.append(e["id"])
            self.write(f"{e['id']}: FAILED {e['message']}\n{e.get('trace', '')}")
        elif kind == "timestack_skipped":
            self.skipped.append(e["id"])
            self.write(f"{e['id']}: already done, skipped")
        elif kind == "quality_done":
            self.flagged = dict(e["flagged"])
            self.write(f"quality check: {len(e['flagged'])} of {e['positions']} positions flagged")
            for key, reason in e["flagged"].items():
                self.write(f"  {key}: {reason}")
        elif kind == "quality_position":
            self.write(f"quality {e['id']}: unusual tiles {100 * e['worst_area']:.1f}%, level {e['worst_level_z']:.1f}, "
                       f"bad frames {e['bad_frames']} {e['reason']}")
            if not e["flagged"] and e["bad_frames"]:
                self.check("Isolated bad frames (tracked; have a look)", f"{e['id']}: frames {e['bad_frames']}: {e['reason']}")
        elif kind == "files_flagged":
            for key, reason in e["new"].items():
                self.write(f"{key}: flagged: {reason}")
                self.flagged[key] = reason
        elif kind == "experiment_start":
            self.write(f"tracking {len(e['timestacks'])} positions")

    def finish(self, status: str) -> None:
        import datetime as dt
        self.status = status
        elapsed = (dt.datetime.now() - self.started).total_seconds()
        self.write(f"run {status} after {elapsed:.0f}s")
        self._file.close()
        lines = [f"FastTrack run summary: {self.exp['experiment']['name'] or '(unnamed experiment)'}",
                 f"{self.started:%Y-%m-%d %H:%M}, {status}, {elapsed / 60:.1f} min. Details: {self.detailed_path.name}", ""]
        n_sel = len(self.selected)
        lines.append(f"Positions: {n_sel} selected, {len(self.done)} tracked now, {len(self.skipped)} already done, "
                     f"{len(self.failed)} failed, {len(self.flagged)} left out by the quality check.")
        if self.done:
            read = sum(e["read_seconds"] for e in self.done)
            wait = sum(e["wait_io"] for e in self.done)
            total = sum(e["seconds"] for e in self.done)
            mb = sum(e.get("bytes_read", 0) for e in self.done) / 1e6
            frames = sum(e["frames"] for e in self.done)
            bound = "the disk" if wait > 0.3 * total else "the GPU"
            lines.append(f"Speed: {frames} frames in {total / 60:.1f} min ({frames / max(total, 1e-9):.1f} frames/s). "
                         f"Read {mb / 1e3:.1f} GB at {mb / max(read, 1e-9):.0f} MB/s; the GPU waited for the disk "
                         f"{100 * wait / max(total, 1e-9):.0f}% of the time, so the limit was {bound}.")
        lines.append("")
        if self.failed:
            self.checks.setdefault("Failed (see the detailed log for the error)", []).extend(self.failed)
        if self.flagged:
            self.checks["Left out by the quality check (open them on the Quality tab; tick Track anyway to use one)"] = \
                [f"{k}: {v}" for k, v in self.flagged.items()]
        if not self.checks:
            lines.append("Nothing needs checking by hand.")
        for section, items in self.checks.items():
            lines.append(f"{section} ({len(items)}):")
            lines += [f"  - {item}" for item in items[:200]]
            if len(items) > 200:
                lines.append(f"  ... and {len(items) - 200} more in the detailed log")
            lines.append("")
        text = "\n".join(lines) + "\n"
        self.summary_path.write_text(text)
        self.latest.write_text(text)
        self.on_event({"kind": "run_summary", "path": str(self.summary_path), "text": text})


def density_warnings(summary: dict, q: dict) -> list[str]:
    """Plain-language warnings when a position is unusually sparse or crowded. Never stops a run."""
    out = []
    if summary["cells_per_frame"] < q["min_cells_per_frame"]:
        out.append(f"sparse: a median of {summary['cells_per_frame']:.0f} cells per frame (below {q['min_cells_per_frame']}); "
                   "tracks and the travel estimate rest on few cells")
    if summary["nuclear_coverage"] > q["max_nuclear_coverage"]:
        out.append(f"crowded: nuclei cover {100 * summary['nuclear_coverage']:.0f}% of the field (above {100 * q['max_nuclear_coverage']:.0f}%); "
                   "expect more merged nuclei and linking errors")
    if summary["empty_frames"]:
        out.append(f"{summary['empty_frames']} frames with no cells detected (out of focus, blank or missing images?)")
    return out


def run_experiment(exp: dict, on_event: Event = _noop, stop: threading.Event | None = None, gui_cache: bool = False,
                   timestacks: dict | None = None, only: list[str] | None = None) -> list[dict]:
    """Track an experiment file from raw data: its plate picks the wells, its pipeline section the parameters.

    `only` limits this run to some timestacks (a partial run); the rest are left as they are.
    Only problems that make tracking impossible stop the run. Everything else (figure settings that
    point nowhere, sparse or crowded positions, a position that fails) is logged and the run carries on.

    Writes <results>/experiment.yaml (the file as run), <results>/Logs/<time>_detailed.log and
    <time>_summary.txt (see RunLog), and appends what the data decided per timestack (gate, radius,
    batch sizes, density) to <results>/run_log.yaml.
    """
    from ..experiment import pipeline_config, validate

    problems = validate(exp)
    if problems:
        raise ValueError("\n".join(problems))
    cfg = pipeline_config(exp)
    cfg["output"]["gui_cache"] = gui_cache
    results = Path(cfg["data"]["output"])
    results.mkdir(parents=True, exist_ok=True)
    log = RunLog(results, exp, on_event)
    try:
        return _run_logged(exp, cfg, results, log, on_event, stop, timestacks, only)
    except BaseException as exc:
        log.write(f"run stopped by an error: {type(exc).__name__}: {exc}")
        import traceback
        log.write(traceback.format_exc())
        log.check("The run stopped", f"{type(exc).__name__}: {exc}")
        log.finish("failed")
        raise


def _run_logged(exp, cfg, results, log, on_event, stop, timestacks, only):
    import datetime as dt

    import yaml

    from .. import __version__
    from ..experiment import load_experiment, same_pipeline, save_experiment, select_timestacks, warnings_for

    warn = log.warn
    for message in warnings_for(exp):
        warn(message, section="Settings that point at nothing (figures leave them out)")
    snapshot = results / "experiment.yaml"
    if snapshot.exists() and not same_pipeline(load_experiment(snapshot), exp):
        kept = results / f"experiment.{dt.datetime.now():%Y%m%d-%H%M%S}.yaml"
        snapshot.replace(kept)
        if not cfg["output"]["overwrite"]:
            warn(f"Tracking settings differ from the ones that made the existing results (kept as {kept.name}). "
                 "Finished timestacks are not redone unless overwrite is on.", section="Settings changed since earlier results")
    timestacks = timestacks if timestacks is not None else layout.discover_timestacks(cfg)
    log_scan(log, timestacks)
    q = exp["quality"]
    if q["enabled"] and not q["checked"]:
        run_quality_check(exp, timestacks, lambda e: (log.event(e), on_event(e)), stop)
    elif q["enabled"]:
        log.flagged = {k: v for k, v in q["flagged"].items() if k not in q["keep"]}
        flag_files(exp, timestacks, lambda e: (log.event(e), on_event(e)))
    save_experiment(exp, snapshot)
    log.write("experiment file as run: experiment.yaml")
    chosen = select_timestacks(exp, list(timestacks))
    if only is not None:
        skipped = sorted(set(only) - set(chosen))
        if skipped:
            warn(f"Not run because the plate or the quality check leaves them out: {', '.join(skipped)}", section="Asked for but not run")
        chosen = [k for k in chosen if k in set(only)]
    if not chosen:
        flagged = set(exp["quality"]["flagged"]) - set(exp["quality"]["keep"]) if exp["quality"]["enabled"] else set()
        raise ValueError(f"No timestacks to process: {len(timestacks)} found, {len(flagged & set(timestacks))} left out by the quality check "
                         f"(see the Quality tab), the rest by the plate layout, exclusions or selection.")
    log.selected = chosen
    started = dt.datetime.now()

    def on_timestack(event):
        log.event(event)
        if event["kind"] == "timestack_done":
            for message in density_warnings(event, exp["quality"]):
                warn(message, event["id"], section="Sparse, crowded or empty (tracked; check the results)")
            if q["enabled"] and event["unreadable"]:  # unreadable files are only known once read
                late = flag_files(exp, {event["id"]: timestacks[event["id"]]}, lambda e: on_event(e))
                for key, reason in late.items():
                    log.check("Left out of figures after tracking: too many unreadable files (results kept; tick Track anyway "
                              "on the Quality tab to use them)", f"{key}: {reason}")
                    log.write(f"{key}: flagged after tracking: {reason}")
        elif event["kind"] == "timestack_failed":
            log.messages.append(f"{event['id']}: failed and was skipped: {event['message']}")
            on_event({"kind": "warning", "message": f"{event['id']}: failed and was skipped: {event['message']}"})
        on_event(event)

    summaries = process_experiment(cfg, on_timestack, stop, select=chosen, timestacks=timestacks, keep_going=True)
    entry = {"started": f"{started:%Y-%m-%d %H:%M:%S}", "fasttrack_version": __version__,
             "device": torch.cuda.get_device_name(torch.device(cfg["device"])) if str(cfg["device"]).startswith("cuda") and torch.cuda.is_available() else str(cfg["device"]),
             "selected": chosen,
             "timestacks": {s["id"]: {"frames": int(s["frames"]), "cells": int(s["cells"]), "tracks": int(s["tracks"]),
                                      "divisions": int(s["divisions"]), "max_travel_px": round(float(s["max_travel"]), 3),
                                      "radius_px": round(float(s["radius"]), 3), "cells_per_frame": round(s["cells_per_frame"], 1),
                                      "nuclear_coverage": round(s["nuclear_coverage"], 4), "empty_frames": s["empty_frames"],
                                      "batch_sizes": [int(b) for b in s["batch_sizes"]],
                                      "blank_frames": sorted({m["frame"] for m in s["missing"]} | {u["frame"] for u in s["unreadable"]}),
                                      "short_by": s["short_by"],
                                      "seconds": round(float(s["seconds"]), 1)} for s in summaries},
             "warnings": log.messages, "summary": log.summary_path.name}
    with open(results / "run_log.yaml", "a") as f:
        yaml.safe_dump([entry], f, sort_keys=False)
    save_experiment(exp, snapshot)  # again: flags added while tracking
    log.finish("stopped" if stop is not None and stop.is_set() else "finished")
    return summaries


def run_quality_check(exp: dict, timestacks: dict, on_event: Event = _noop, stop: threading.Event | None = None) -> tuple[dict, dict]:
    """Check the positions the plate selects, record the flags in `exp["quality"]` (changed in place),
    and write <results>/quality_report.yaml. Returns (results, features)."""
    import yaml

    from ..experiment import loaded_channels, select_timestacks
    from ..quality import check_positions

    q = exp["quality"]
    unflagged = {**exp, "quality": {**q, "flagged": {}}}
    candidates = {k: timestacks[k] for k in select_timestacks(unflagged, list(timestacks))}
    names = [c["name"] for c in loaded_channels(exp)] or None
    cfg_channels = pipeline_config_channels(exp)
    if cfg_channels:  # read only the channels that will be tracked
        candidates = {k: {**ts, "channels": {c: ts["channels"][c] for c in cfg_channels if c in ts["channels"]}} for k, ts in candidates.items()}
    if len(candidates) < 3:
        on_event({"kind": "warning", "message": f"Quality check skipped: it compares positions with each other and needs at least 3 ({len(candidates)} selected)."})
        flag_files(exp, candidates, on_event)
        return {}, {}
    on_event({"kind": "quality_start", "positions": len(candidates)})
    results, features = check_positions(candidates, q, resolve_device(exp["pipeline"]["device"]), names, on_event, stop, int(exp["files"]["io_threads"]))
    from ..quality import file_flags
    q["flagged"] = {**{k: r["reason"] or "unusual tiles" for k, r in results.items() if r["flagged"]}, **file_flags({k: candidates[k] for k in results}, q)}
    for k, reason in q["flagged"].items():
        results[k].update(flagged=True, reason=reason)
    for k, r in results.items():
        on_event({"kind": "quality_position", "id": k, **{f: r[f] for f in ("flagged", "bad_frames", "worst_area", "worst_level_z", "reason")}})
    q["keep"] = [k for k in q["keep"] if k in q["flagged"]]
    q["checked"] = True
    report = {k: {"flagged": k in q["flagged"], "kept": k in q["keep"], "bad_frames": r["bad_frames"], "sampled_frames": r["frames"],
                  "worst_area": round(r["worst_area"], 4), "worst_level_z": round(r["worst_level_z"], 2),
                  "reason": q["flagged"].get(k, r["reason"])}
              for k, r in results.items()}
    results_dir = Path(exp["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "quality_report.yaml", "w") as f:
        yaml.safe_dump({"settings": {k: v for k, v in q.items() if k not in ("flagged", "keep")}, "positions": report}, f, sort_keys=False)
    on_event({"kind": "quality_done", "flagged": dict(q["flagged"]), "positions": len(results)})
    return results, features


def log_scan(log: "RunLog", timestacks: dict) -> dict:
    """Write what the scan found about missing files and movie lengths to the run log."""
    report = layout.scan_report(timestacks)
    log.write(f"scan: {len(timestacks)} positions, most with {report['usual_frames']} frames; "
              f"{len(report['missing'])} with missing files, {len(report['short'])} ending early, {len(report['longer'])} longer")
    for key, text in report["missing"].items():
        log.write(f"  {key}: missing {text}")
    for key, n in report["short"].items():
        log.write(f"  {key}: ends {n} frames early")
    for key, n in report["longer"].items():
        log.write(f"  {key}: {n} frames longer than most (extra frames are tracked)")
    if report["suggest_last"]:
        log.check("Movie length (your choice; nothing was changed)",
                  f"{len(report['short'])} position{'s' * (len(report['short']) > 1)} end{'s' * (len(report['short']) == 1)} 1-3 frames early. To give every position the same length, set Frames to "
                  f"= {report['suggest_last']} on the Experiment tab (files.last_frame) and run again with overwrite on.")
    return report


def flag_files(exp: dict, timestacks: dict, on_event: Event = _noop) -> dict:
    """Add file-based flags (too many missing, unreadable or cut-off frames) to exp["quality"]["flagged"].
    Keeps every existing flag and the manual keeps. Returns the new flags."""
    from ..quality import file_flags
    q = exp["quality"]
    new = {k: v for k, v in file_flags(timestacks, q).items() if k not in q["flagged"]}
    if new:
        q["flagged"].update(new)
        on_event({"kind": "files_flagged", "new": new, "flagged": dict(q["flagged"])})
    return new


def pipeline_config_channels(exp: dict) -> list[int]:
    from ..experiment import loaded_channels
    return [int(c["file_channel"]) for c in loaded_channels(exp)]
