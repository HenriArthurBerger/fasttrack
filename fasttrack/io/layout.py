"""Find timestacks on disk and read frames fast.

A timestack is one position of one well: {channel: [one file per timepoint]}.
"""
from __future__ import annotations

import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def discover_timestacks(cfg: dict) -> dict:
    """Walk data.root and group files by data.pattern. Returns {id: timestack}.

    Frame i means the same timepoint in every position: the timeline starts at the experiment's first
    timepoint, and a file that is missing inside it stays in the list as None (read as a blank frame,
    listed in timestack["missing"]) instead of being dropped, which would shift every later frame.
    A position whose files stop early ends early; timestack["short_by"] says by how many frames,
    compared with most positions. timestack["problems"] says all of this in words."""
    d = cfg["data"]
    pattern = re.compile(d["pattern"])
    groups: dict[tuple, dict] = {}
    stack = [str(d["root"])]
    while stack:
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                if entry.is_dir():
                    stack.append(entry.path)
                    continue
                match = pattern.search(entry.name)
                if match is None:
                    continue
                g = match.groupdict()
                well = g.get("well") or Path(entry.path).parent.name
                time_index = int(g["t"]) if g.get("t") else None
                key = (well, int(g["pos"]))
                groups.setdefault(key, {}).setdefault(int(g["chan"]), []).append((time_index, entry.name, entry.path))
    times_seen = [r[0] for chans in groups.values() for recs in chans.values() for r in recs if r[0] is not None]
    t0 = min(times_seen) if times_seen else 0
    # first_frame / last_frame count timepoints of the full movie (from its first timepoint), before the stride.
    span = slice(int(d.get("first_frame", 0)), int(d.get("last_frame", 0)) or None, int(d["stride"]))
    timestacks = {}
    for (well, pos), channels in sorted(groups.items()):
        selected = d["channels"] or sorted(channels)
        problems: list[str] = []
        missing: list[dict] = []
        by_time = {chan: {r[0]: r[2] for r in channels.get(chan, []) if r[0] is not None} for chan in selected}
        if all(by_time.values()):
            ends = {chan: max(v) for chan, v in by_time.items()}
            end = min(ends.values())  # a channel that stops early ends the position there
            if max(ends.values()) > end:
                problems.append(f"channel {min(ends, key=ends.get)} has no files after timepoint {end}; "
                                f"the position ends there ({max(ends.values()) - end} frames of other channels unused)")
            timeline = list(range(t0, end + 1))
            index = list(range(len(timeline)))[span]
            times = [timeline[i] for i in index]
            lists = {chan: [by_time[chan].get(t) for t in times] for chan in selected}
            for i, t in enumerate(times):
                absent = [chan for chan in selected if lists[chan][i] is None]
                if absent:
                    missing.append({"frame": i, "time": t, "channels": absent})
        else:
            ordered = {chan: [r[2] for r in sorted(channels.get(chan, []), key=lambda r: r[1])][span] for chan in selected}
            shortest = min(len(v) for v in ordered.values())
            if len({len(v) for v in ordered.values()}) > 1:
                problems.append(f"channels have unequal frame counts and the filenames have no time index, so gaps cannot be "
                                f"placed; kept the first {shortest} frames")
            lists = {chan: v[:shortest] for chan, v in ordered.items()}
            times = list(range(shortest))
            timeline = range(min(len(channels.get(chan, [])) for chan in selected))
        frames = len(times)
        if frames == 0:
            continue
        if d.get("max_frames"):
            frames = min(frames, int(d["max_frames"]))
            missing = [m for m in missing if m["frame"] < frames]
        if missing:
            problems.append(f"missing files, read as blank frames: {describe_missing(missing, len(selected))}")
        timestacks[f"{well}_xy{pos}"] = {"id": f"{well}_xy{pos}", "well": well, "pos": pos, "channels": lists,
                                         "frames": frames, "times": times[:frames], "problems": problems,
                                         "missing": missing, "short_by": 0,
                                         "movie_frames": len(timeline)}
    if timestacks:
        usual = Counter(ts["frames"] for ts in timestacks.values()).most_common(1)[0][0]
        for ts in timestacks.values():
            if ts["frames"] < usual:
                ts["short_by"] = usual - ts["frames"]
                ts["problems"].append(f"ends {ts['short_by']} frame{'s' * (ts['short_by'] > 1)} early "
                                      f"({ts['frames']} frames; most positions have {usual})")
    return timestacks


def _ranges(frames: list[int]) -> str:
    """[3, 4, 5, 9] -> '3-5, 9'."""
    out, run = [], []
    for f in sorted(frames):
        if run and f == run[-1] + 1:
            run.append(f)
            continue
        if run:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
        run = [f]
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ", ".join(out)


def describe_missing(missing: list[dict], n_channels: int) -> str:
    """'frames 57-58 (all channels); frame 120 (channel 2)'."""
    whole = [m["frame"] for m in missing if len(m["channels"]) == n_channels]
    parts = [f"frame{'s' * (len(whole) > 1)} {_ranges(whole)} (all channels)"] if whole else []
    some: dict[tuple, list[int]] = {}
    for m in missing:
        if len(m["channels"]) < n_channels:
            some.setdefault(tuple(m["channels"]), []).append(m["frame"])
    for chans, frames in some.items():
        parts.append(f"frame{'s' * (len(frames) > 1)} {_ranges(frames)} (channel{'s' * (len(chans) > 1)} {', '.join(map(str, chans))})")
    return "; ".join(parts)


def bad_frames(timestack: dict) -> list[int]:
    """Frames with a missing or unreadable file in any channel (known so far)."""
    return sorted({m["frame"] for m in timestack.get("missing", [])} | {u["frame"] for u in timestack.get("unreadable", [])})


def scan_report(timestacks: dict) -> dict:
    """What a scan found that a person should know, for the Experiment tab and the run log.

    usual_frames   the movie length most positions have
    missing        {position: description} of files missing inside the movie
    short          {position: frames it lacks at the end}
    longer         {position: extra frames} (tracked as they are)
    suggest_last   a 'Frames to' value that ends every position on the same frame, when the short positions
                   lack only a few frames (cutting a few frames from all beats losing positions); else 0
    """
    if not timestacks:
        return {"usual_frames": 0, "missing": {}, "short": {}, "longer": {}, "suggest_last": 0}
    usual = Counter(ts["frames"] for ts in timestacks.values()).most_common(1)[0][0]
    short = {k: ts["short_by"] for k, ts in timestacks.items() if ts.get("short_by")}
    longer = {k: ts["frames"] - usual for k, ts in timestacks.items() if ts["frames"] > usual}
    missing = {k: describe_missing(ts["missing"], len(ts["channels"])) for k, ts in timestacks.items() if ts.get("missing")}
    suggest = 0
    if short and max(short.values()) <= 3:
        suggest = min(timestacks[k]["movie_frames"] for k in short)  # 'Frames to' counts the full movie, before the stride
    return {"usual_frames": usual, "missing": missing, "short": short, "longer": longer, "suggest_last": suggest}


class UnreadableFrame(OSError):
    """A TIFF that cannot be decoded (empty, truncated or corrupt)."""


def _usable(a, shape) -> bool:
    return isinstance(a, np.ndarray) and a.ndim == 2 and a.size > 0 and (shape is None or a.shape == tuple(shape))


def read_frame(path: str | None, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read one 16-bit frame. PIL's C decoder is 22x faster than tifffile's pure-Python PackBits path,
    measured on real files; tifffile covers anything PIL cannot open. A truncated file can make PIL
    return an empty array instead of failing, so the result is checked; if neither reader gives a
    2D image (of `shape`, when given), UnreadableFrame names the file."""
    if path is None:
        raise UnreadableFrame("no file for this frame (missing from the folder)")
    errors = []
    try:
        with Image.open(path) as image:
            image.load()
            a = np.asarray(image)
        if _usable(a, shape):
            return a
        errors.append(f"PIL gave shape {getattr(a, 'shape', None)}")
    except Exception as exc:  # any decoder failure: try the other reader
        errors.append(f"PIL: {exc}")
    try:
        a = tifffile.imread(path)
        if _usable(a, shape):
            return a
        errors.append(f"tifffile gave shape {getattr(a, 'shape', None)}")
    except Exception as exc:
        errors.append(f"tifffile: {exc}")
    raise UnreadableFrame(f"unreadable file {path} ({'; '.join(errors)})")


def frame_shape(timestack: dict) -> tuple[int, int]:
    """(X, Y) of the position's frames, from the first file that opens. Cached on the timestack."""
    if "shape" in timestack:
        return timestack["shape"]
    files = [f for f in next(iter(timestack["channels"].values())) if f is not None]
    for path in files[:50]:
        try:
            with Image.open(path) as image:
                timestack["shape"] = (image.size[1], image.size[0])
                return timestack["shape"]
        except Exception:
            continue
    raise UnreadableFrame(f"none of the first {min(50, len(files))} files of {timestack.get('id', '?')} can be opened")


RETRY_DELAYS = (0.5, 2.0)  # seconds before the second and third try of a file that failed to read


def read_frame_retry(path: str | None, shape=None, delays=RETRY_DELAYS) -> tuple[np.ndarray, int]:
    """read_frame, tried again after a pause when it fails: a USB disk that is spinning up or busy can
    fail a read that works a moment later. Returns (image, tries needed). A file that is not there
    (missing, or the drive was unplugged) is not retried."""
    for attempt in range(len(delays) + 1):
        try:
            return read_frame(path, shape), attempt + 1
        except UnreadableFrame:
            if path is None or attempt == len(delays) or not os.path.exists(path):
                raise
            time.sleep(delays[attempt])
    raise AssertionError("unreachable")


class DriveUnavailable(OSError):
    """No file of a whole batch could be read: the drive is probably unplugged or failing."""


def load_batch(timestack: dict, start: int, stop: int, pool: ThreadPoolExecutor) -> np.ndarray:
    """Read frames [start, stop) into (T, C, X, Y) uint16. One thread per file: reads are I/O bound.

    A file missing from the folder (None, known from the scan) is a blank frame. A file that fails to
    read is tried again twice; if it still fails it becomes a blank frame instead of stopping the run,
    listed in timestack["unreadable"]. Files that only read on a later try go in timestack["retried"].
    If not one file of the batch can be read, DriveUnavailable stops this position."""
    X, Y = frame_shape(timestack)
    channels = list(timestack["channels"].values())
    out = np.empty((stop - start, len(channels), X, Y), dtype=np.uint16)
    unreadable = timestack.setdefault("unreadable", [])
    retried = timestack.setdefault("retried", [])
    failures = []

    def read(job):
        t, c = job
        path = channels[c][t]
        if path is None:
            out[t - start, c] = 0
            return
        try:
            out[t - start, c], tries = read_frame_retry(path, (X, Y))
            if tries > 1:
                retried.append({"frame": t, "channel": c, "path": str(path), "tries": tries})
        except UnreadableFrame as exc:
            out[t - start, c] = 0
            gone = not os.path.exists(path)
            failures.append(gone)
            unreadable.append({"frame": t, "channel": c, "path": str(path),
                               "error": f"file disappeared (drive unplugged?): {path}" if gone else str(exc)})

    jobs = [(t, c) for t in range(start, stop) for c in range(len(channels)) if channels[c][t] is not None]
    list(pool.map(read, ((t, c) for t in range(start, stop) for c in range(len(channels)))))
    if jobs and len(failures) == len(jobs) and (all(failures) or stop - start > 1):
        raise DriveUnavailable(f"none of the {len(jobs)} files for frames {start}-{stop - 1} of {timestack.get('id', '?')} could be read"
                               + (" and they no longer exist: is the drive still connected?" if all(failures) else
                                  ": the drive may be failing (see the detailed log)"))
    return out


def batch_ranges(frames: int, batch_size: int, min_tail: int = 20) -> list[tuple[int, int]]:
    """Split a movie into batches, folding a short tail into the previous batch.

    A batch of a few frames gives poor background and radius estimates. 101 frames with
    batch size 100 is one batch of 101, not 100 plus 1.
    """
    edges = list(range(0, frames, batch_size))
    ranges = [(s, min(s + batch_size, frames)) for s in edges]
    if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] < min_tail:
        ranges[-2] = (ranges[-2][0], ranges[-1][1])
        ranges.pop()
    return ranges
