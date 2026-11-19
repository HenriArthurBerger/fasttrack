"""Nucleus centre labelling: correct the seeds on small patches, train the learned seed finder on them.

Where to look. A segmentation mistake rarely lasts: a nucleus split in two for one frame, or one
missed for a frame, shows up as a short bump in the number of cells in that part of the field.
So for every tracked position the detections are smoothed into a local cell count (a Gaussian
of `count_sigma` px around every point, drift removed), and each frame is compared with the
median of the frames around it (`count_window`). Where the count differs by at least
`min_change`, there is a candidate. A division does not count: after it the count stays up, so
the median follows. Random patches around cells are mixed in (`sample.random`) so the model also
sees ordinary nuclei.

Event candidates. Where a track ends early, starts late, or a mother divides, a patch is centred on
that cell (sources track_end, track_start, division), so rare states such as divisions and deaths
turn up often enough to train the state classifier (see fasttrack/dataset.py).

Labelling. A patch shows one frame around the candidate, pre-filled with the centres that tracking
found there, with `context` frames before and after to flip through. Click a centre to remove it,
click a nucleus without one to add it. Every nucleus whose centre lies inside the box must have
exactly one point; the margin around the box is only there to see what is cut. An empty box is a
fine answer. Every point also has a state (labelling.categories; the first, e.g. neutral, unless
you change it).

Files, inside the results folder:
    Labels/centres.json        your patches: position, frame, box, the points in it and their states
    Labels/centre_model.pt     the trained seed finder (centroids.method = "cnn" uses it)
    Labels/centre_model.json   what it was trained on and how it did on held-out patches
"""
from __future__ import annotations

import datetime as dt
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .device import resolve as resolve_device
from .experiment import pipeline_config
from .io import layout, store
from .track.measure import S

SOURCES = ["count_change", "random", "track_end", "track_start", "division"]
LEGACY_STATES = {"division": "mitosis"}   # states renamed since: read as the new name
EVENT_SOURCES = ("track_end", "track_start", "division")
SOURCE_TEXT = {"count_change": "a local change in cell count", "random": "a random cell",
               "track_end": "a track that ends (the cell in the centre)", "track_start": "a track that starts (the cell in the centre)",
               "division": "a mother just before it divides (the cell in the centre)", "events_csv": "an earlier event label"}


# ---------------------------------------------------------------------- where to look
def detections(results) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every detected nucleus: frame, x, y (raw image coordinates)."""
    t = results.column("t") if hasattr(results, "column") else results["scalars"][:, S["t"]]
    x = results.column("x") if hasattr(results, "column") else results["scalars"][:, S["x"]]
    y = results.column("y") if hasattr(results, "column") else results["scalars"][:, S["y"]]
    ok = np.isfinite(x) & np.isfinite(y)
    return t[ok].astype(int), x[ok].astype(np.float32), y[ok].astype(np.float32)


def local_counts(t, x, y, frames: int, shape: tuple[int, int], sigma: float, step: int = 8) -> np.ndarray:
    """(frames, X/step, Y/step): about how many nuclei lie within `sigma` px of each grid point.
    A single nucleus adds 1 at its own place, so a count difference of 1 is one nucleus."""
    gx, gy = -(-shape[0] // step), -(-shape[1] // step)
    h = np.zeros((frames, gx, gy), np.float32)
    keep = (t >= 0) & (t < frames)
    np.add.at(h, (t[keep], np.clip((x[keep] // step).astype(int), 0, gx - 1), np.clip((y[keep] // step).astype(int), 0, gy - 1)), 1)
    s = sigma / step
    return ndimage.gaussian_filter(h, (0, s, s), mode="constant", truncate=3.0) * (2 * np.pi * s * s)


def count_changes(results, shape: tuple[int, int], c: dict, step: int = 8) -> list[dict]:
    """Candidates for one position, strongest first: {t, x, y, change}. x, y in raw image coordinates.

    `change` > 0: more nuclei than in the frames around (a nucleus split, or debris);
    < 0: fewer (a nucleus missed, or two merged). One candidate per place and few frames."""
    t, x, y = detections(results)
    frames = int(results["meta"]["frames"]) if "meta" in results and "frames" in results["meta"] else int(t.max()) + 1 if len(t) else 0
    if frames < 3 or not len(t):
        return []
    drift = np.asarray(results["drift"]) if "drift" in results else np.zeros((frames, 2))
    if drift.shape[0] < frames:
        drift = np.zeros((frames, 2))
    # counted in frame-0 coordinates, so stage drift does not look like cells coming and going
    counts = local_counts(t, x + drift[t, 0], y + drift[t, 1], frames, shape, float(c["count_sigma"]), step)
    w = max(3, int(c["count_window"]) | 1)
    dev = counts - ndimage.median_filter(counts, size=(w, 1, 1), mode="nearest")
    size = max(3, int(c["patch"]) // step)
    strength = np.abs(dev)
    peak = (strength >= float(c["min_change"])) & (strength == ndimage.maximum_filter(strength, size=(w, size, size), mode="constant"))
    out = []
    for tt, gx, gy in zip(*np.nonzero(peak)):
        cx = gx * step + step / 2 - drift[tt, 0]
        cy = gy * step + step / 2 - drift[tt, 1]
        out.append({"t": int(tt), "x": float(cx), "y": float(cy), "change": round(float(dev[tt, gx, gy]), 2)})
    out.sort(key=lambda o: -abs(o["change"]))
    return out


def change_profile(results, shape, c, at: dict, step: int = 8) -> list[float]:
    """The local count at the candidate's place over the frames around it (for the info line)."""
    t, x, y = detections(results)
    frames = int(results["meta"]["frames"])
    drift = np.asarray(results["drift"])
    counts = local_counts(t, x + drift[t, 0], y + drift[t, 1], frames, shape, float(c["count_sigma"]), step)
    gx = int(np.clip((at["x"] + drift[at["t"], 0]) // step, 0, counts.shape[1] - 1))
    gy = int(np.clip((at["y"] + drift[at["t"], 1]) // step, 0, counts.shape[2] - 1))
    lo, hi = max(0, at["t"] - 3), min(frames, at["t"] + 4)
    return [round(float(v), 1) for v in counts[lo:hi, gx, gy]]


def box_for(cx: float, cy: float, size: int, shape: tuple[int, int]) -> tuple[int, int]:
    """Top-left corner of a size x size box centred near (cx, cy), kept inside the image."""
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    return int(np.clip(x0, 0, max(0, shape[0] - size))), int(np.clip(y0, 0, max(0, shape[1] - size)))


def covered(rows: list[dict], position: str, t: int, x: float, y: float) -> bool:
    """True if a labelled patch of this position and frame already contains the point."""
    return any(r["position"] == position and r["t"] == t and r["x0"] <= x < r["x0"] + r["size"] and r["y0"] <= y < r["y0"] + r["size"]
               for r in rows)


class Sampler:
    """Draws patches to label. Candidates of a position are computed the first time it is drawn."""

    def __init__(self, exp: dict, shape_of, seed: int | None = None, available=None):
        """available: positions that can be read from the raw folder; tracked positions outside it are left out."""
        self.exp = exp
        self.c = exp["labelling"]["centres"]
        self.results_dir = Path(exp["paths"]["results"])
        self.shape_of = shape_of            # position -> (X, Y)
        self.rng = random.Random(seed)
        self.cache: dict[str, tuple] = {}
        self.positions = sorted(p.stem for p in (self.results_dir / "Traces").glob("*.h5")) if exp["paths"]["results"] else []
        if available is not None:
            self.positions = [k for k in self.positions if k in set(available)]

    def position(self, key: str) -> tuple:
        if key not in self.cache:
            r = store.load_results(self.results_dir, key)
            shape = self.shape_of(key)
            from .events import candidates
            try:
                events = candidates(r)
            except (KeyError, ValueError):
                events = {s: np.zeros((0, 2), int) for s in EVENT_SOURCES}
            self.cache[key] = (r, shape, count_changes(r, shape, self.c), detections(r), events)
        return self.cache[key]

    def draw(self, rows: list[dict], skipped: set) -> dict | None:
        if not self.positions:
            return None
        weights = [max(0.0, float(self.c["sample"].get(s, 0))) for s in SOURCES]
        size = int(self.c["patch"])
        for _ in range(60):
            source = self.rng.choices(SOURCES, weights)[0] if sum(weights) else "random"
            key = self.rng.choice(self.positions)
            r, shape, cands, (t, x, y), events = self.position(key)
            if source in EVENT_SOURCES:
                from .events import track_xy
                pool = events.get(source, np.zeros((0, 2), int))
                if not len(pool):
                    continue
                track, tt = (int(v) for v in pool[self.rng.randrange(len(pool))])
                cx, cy = track_xy(r, track, tt)
                change = 0.0
                if covered(rows, key, tt, cx, cy) or (key, tt, round(cx), round(cy)) in skipped:
                    continue
            elif source == "count_change":
                # strongest first, with some randomness so one position is not labelled twenty times in a row
                fresh = [o for o in cands if not covered(rows, key, o["t"], o["x"], o["y"]) and (key, o["t"], round(o["x"]), round(o["y"])) not in skipped]
                if not fresh:
                    continue
                o = fresh[min(int(self.rng.expovariate(1 / 5)), len(fresh) - 1)]
                cx, cy, tt, change = o["x"], o["y"], o["t"], o["change"]
            else:
                if not len(t):
                    continue
                i = self.rng.randrange(len(t))
                tt = int(t[i])
                cx = float(x[i]) + self.rng.uniform(-size / 4, size / 4)
                cy = float(y[i]) + self.rng.uniform(-size / 4, size / 4)
                change = 0.0
                if covered(rows, key, tt, cx, cy) or (key, tt, round(cx), round(cy)) in skipped:
                    continue
            x0, y0 = box_for(cx, cy, size, shape)
            if any(patch_key(r) == (key, tt, x0, y0) for r in rows):
                continue
            return {"position": key, "t": tt, "x0": x0, "y0": y0, "size": size, "source": source, "change": change,
                    "cx": round(cx), "cy": round(cy)}
        return None

    def tracked_points(self, obs: dict, t: int | None = None) -> list[list[float]]:
        """Centres tracking found in the box at frame t (the patch's frame by default), in image coordinates."""
        _, _, _, (tt, x, y), _ = self.position(obs["position"])
        t = obs["t"] if t is None else t
        m = (tt == t) & (x >= obs["x0"]) & (x < obs["x0"] + obs["size"]) & (y >= obs["y0"]) & (y < obs["y0"] + obs["size"])
        return [[round(float(a), 1), round(float(b), 1)] for a, b in zip(x[m], y[m])]

    def profile(self, obs: dict) -> list[float]:
        r, shape, _, _, _ = self.position(obs["position"])
        return change_profile(r, shape, self.c, {"t": obs["t"], "x": obs["cx"], "y": obs["cy"]})


# ---------------------------------------------------------------------- frames
def nuclear_index(exp: dict) -> int:
    return int(pipeline_config(exp)["nuclear_channel"])


class NuclearReader:
    """Reads the nuclear channel of single frames from the raw TIFFs (only ever read)."""

    def __init__(self, exp: dict, timestacks: dict | None = None, io_threads: int = 4):
        self.exp = exp
        self.cfg = pipeline_config(exp)
        self.timestacks = timestacks
        self.channel = int(self.cfg["nuclear_channel"])
        self.pool = ThreadPoolExecutor(io_threads)

    def all(self) -> dict:
        if not self.timestacks:
            self.timestacks = layout.discover_timestacks(self.cfg)
        return self.timestacks

    def ts(self, key: str) -> dict:
        if key not in self.all():
            raise KeyError(f"{key} has results but is not in the raw data folder")
        return self.timestacks[key]

    def shape(self, key: str) -> tuple[int, int]:
        return tuple(layout.frame_shape(self.ts(key)))

    def frame(self, key: str, t: int, channel: int | None = None) -> np.ndarray:
        """(X, Y) uint16 of the nuclear channel, or `channel` (an index into the loaded channels); blank if the
        file is missing or unreadable, as in tracking."""
        ts = self.ts(key)
        paths = list(ts["channels"].values())[self.channel if channel is None else channel]
        shape = layout.frame_shape(ts)
        if not 0 <= t < ts["frames"]:
            return np.zeros(shape, np.uint16)
        try:
            return layout.read_frame(paths[t], shape)
        except layout.UnreadableFrame:
            return np.zeros(shape, np.uint16)

    def frames(self, key: str, ts: list[int], channel: int | None = None) -> list[np.ndarray]:
        return list(self.pool.map(lambda t: self.frame(key, t, channel), ts))

    def close(self) -> None:
        self.pool.shutdown(wait=False)


def cut(frame: np.ndarray, x0: int, y0: int, size: int, margin: int, mode: str = "reflect") -> np.ndarray:
    """The box plus a margin on every side. Where it runs off the image: reflected (for the network), or
    mode="constant" for black (for people: a mirrored strip looks like real cells)."""
    padded = np.pad(frame, margin, mode=mode) if margin else frame
    return padded[x0:x0 + size + 2 * margin, y0:y0 + size + 2 * margin]


def model_input(frame: np.ndarray, cfg: dict, device: str) -> torch.Tensor:
    """(X,Y) raw nuclear frame -> (X,Y) the network's input: background corrected, then normalised over the whole frame,
    exactly as tracking does."""
    from .track import background, centroid_net
    with torch.inference_mode():
        x = torch.from_numpy(frame.astype(np.float32))[None, None].to(device)
        corrected, _ = background.correct_background(x, cfg)
        return centroid_net.normalize(corrected)[0, 0]


# ---------------------------------------------------------------------- labels file
def labels_path(exp: dict) -> Path:
    return Path(exp["paths"]["results"]) / exp["labelling"]["centres"]["labels_file"]


def read_labels(exp: dict) -> list[dict]:
    path = labels_path(exp)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("patches", [])


def write_labels(exp: dict, rows: list[dict]) -> None:
    path = labels_path(exp)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"about": "Nucleus centres clicked in FastTrack. Every nucleus centred inside a box has one point; "
                            "x is the first image axis (row), y the second.", "patches": rows}, f, indent=1)
    tmp.replace(path)


def patch_key(r: dict) -> tuple:
    return r["position"], int(r["t"]), int(r["x0"]), int(r["y0"])


def add_label(exp: dict, rows: list[dict], obs: dict, points: list, tracked: list, states: list | None = None) -> list[dict]:
    """Save one patch (replacing an earlier label of the same box). states: one category per point; missing or
    empty means the default (first) category. A state-only row (an earlier event label) is replaced too when the
    new patch has a point within 6 px of its cell: the cell is now labelled in full."""
    default = exp["labelling"]["categories"][0] if exp["labelling"]["categories"] else ""
    states = list(states or [])
    states = [str(states[i]) if i < len(states) and states[i] else default for i in range(len(points))]
    pts = [[round(float(a), 1), round(float(b), 1)] for a, b in points]

    def absorbed(r):
        return (r.get("state_only") and r["position"] == obs["position"] and int(r["t"]) == int(obs["t"])
                and any(math.hypot(r["points"][0][0] - a, r["points"][0][1] - b) <= 6 for a, b in pts))
    rows = [r for r in rows if patch_key(r) != patch_key(obs) and not absorbed(r)]
    rows.append({"position": obs["position"], "t": int(obs["t"]), "x0": int(obs["x0"]), "y0": int(obs["y0"]), "size": int(obs["size"]),
                 "source": obs["source"], "change": obs.get("change", 0.0), "points": pts, "states": states,
                 "tracked": [[round(float(a), 1), round(float(b), 1)] for a, b in tracked],
                 "labelled": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"})
    write_labels(exp, rows)
    return rows


def point_states(row: dict, default: str) -> list[str]:
    """One state per point of a patch; older files without states get the default."""
    s = [LEGACY_STATES.get(v, v) for v in row.get("states") or []]
    return [s[i] if i < len(s) and s[i] else default for i in range(len(row["points"]))]


def centre_rows(rows: list[dict]) -> list[dict]:
    """Patches whose every nucleus is labelled (state-only rows mark one cell and do not count for centres)."""
    return [r for r in rows if not r.get("state_only")]


def label_summary(rows: list[dict], default: str = "neutral") -> dict:
    """How much the labels changed what tracking found, and how many points have each state."""
    states: dict[str, int] = {}
    for r in rows:
        for st in point_states(r, default):
            states[st] = states.get(st, 0) + 1
    state_only = sum(bool(r.get("state_only")) for r in rows)
    rows = centre_rows(rows)
    added = removed = kept = 0
    for r in rows:
        a = match(np.array(r["points"], float).reshape(-1, 2), np.array(r.get("tracked", []), float).reshape(-1, 2), 4.0)
        kept += a
        added += len(r["points"]) - a
        removed += len(r.get("tracked", [])) - a
    return {"patches": len(rows), "points": sum(len(r["points"]) for r in rows), "added": added, "removed": removed, "kept": kept,
            "count_change": sum(r["source"] == "count_change" for r in rows), "random": sum(r["source"] == "random" for r in rows),
            "states": states, "state_only": state_only}


# ---------------------------------------------------------------------- scoring
def match(truth: np.ndarray, found: np.ndarray, gate: float) -> int:
    """Number of one-to-one pairs closer than `gate` px (optimal assignment)."""
    if not len(truth) or not len(found):
        return 0
    from scipy.optimize import linear_sum_assignment
    d = np.linalg.norm(truth[:, None] - found[None], axis=2)
    rows, cols = linear_sum_assignment(np.where(d < gate, d, 1e6))
    return int((d[rows, cols] < gate).sum())


def prf(tp: int, n_truth: int, n_found: int) -> dict:
    p = tp / n_found if n_found else 1.0
    r = tp / n_truth if n_truth else 1.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "missed": n_truth - tp, "extra": n_found - tp}


# ---------------------------------------------------------------------- training
def model_path(exp: dict) -> Path:
    return Path(exp["paths"]["results"]) / exp["labelling"]["centres"]["model_file"]


def build_patches(exp: dict, rows: list[dict], timestacks: dict | None, device: str, on_event=lambda e: None):
    """Network inputs for every labelled patch: images (N,1,P,P), inside (N,1,P,P), points (M,3) [patch, x, y], all in patch coordinates."""
    c = exp["labelling"]["centres"]
    margin = int(c["margin"])
    cfg = pipeline_config(exp)
    reader = NuclearReader(exp, timestacks, int(exp["files"]["io_threads"]))
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["position"], int(r["t"])), []).append(i)
    images, inside, points = [None] * len(rows), [None] * len(rows), []
    try:
        for n, ((key, t), idx) in enumerate(groups.items()):
            full = model_input(reader.frame(key, t), cfg, device).cpu().numpy()
            for i in idx:
                r = rows[i]
                s = int(r["size"])
                images[i] = cut(full, int(r["x0"]), int(r["y0"]), s, margin)
                box = np.zeros_like(images[i], bool)
                box[margin:margin + s, margin:margin + s] = True
                inside[i] = box
            on_event({"kind": "patches", "done": n + 1, "total": len(groups)})
    finally:
        reader.close()
    P = max(a.shape[0] for a in images)
    pad = lambda a: np.pad(a, ((0, P - a.shape[0]), (0, P - a.shape[1])))
    for i, r in enumerate(rows):
        for px, py in r["points"]:
            points.append([i, px - r["x0"] + margin, py - r["y0"] + margin])
    return (torch.as_tensor(np.stack([pad(a) for a in images]))[:, None].to(device),
            torch.as_tensor(np.stack([pad(a) for a in inside]))[:, None].to(device),
            torch.as_tensor(np.array(points, np.float32).reshape(-1, 3)).to(device))


@torch.inference_mode()
def patch_scores(net, images: torch.Tensor) -> torch.Tensor:
    out = []
    with torch.inference_mode():
        for i in range(0, images.shape[0], 16):
            out.append(torch.sigmoid(net(images[i:i + 16])).float())
    return torch.cat(out)


def evaluate(scores: torch.Tensor, inside: torch.Tensor, rows: list[dict], margin: int, threshold: float, gate: float) -> dict:
    """Seeds from the score maps against the clicked centres, inside each box only."""
    from .track.centroid_net import find_peaks
    peaks = find_peaks(scores, threshold).cpu().numpy()
    tp = n_truth = n_found = 0
    for i, r in enumerate(rows):
        p = peaks[peaks[:, 0] == i][:, 1:].astype(float)
        p = p[inside[i, 0].cpu().numpy()[p[:, 0].astype(int), p[:, 1].astype(int)]] if len(p) else p.reshape(-1, 2)
        truth = np.array(r["points"], float).reshape(-1, 2) - [r["x0"] - margin, r["y0"] - margin]
        tp += match(truth, p, gate)
        n_truth += len(truth)
        n_found += len(p)
    return prf(tp, n_truth, n_found)


def tracked_score(rows: list[dict], gate: float) -> dict:
    """What tracking found in the same boxes, against the clicked centres."""
    tp = n_truth = n_found = 0
    for r in rows:
        truth = np.array(r["points"], float).reshape(-1, 2)
        found = np.array(r.get("tracked", []), float).reshape(-1, 2)
        tp += match(truth, found, gate)
        n_truth += len(truth)
        n_found += len(found)
    return prf(tp, n_truth, n_found)


THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7)


def train_from_labels(exp: dict, on_event=lambda e: None, timestacks: dict | None = None) -> dict:
    """Cut every labelled patch from the raw TIFFs, hold out a fifth to measure, train, then train the
    final model on every patch. Saves the model and a report next to it; returns the report."""
    from .track import centroid_net
    c = exp["labelling"]["centres"]
    rows = centre_rows(read_labels(exp))
    if len(rows) < 10:
        raise ValueError(f"Label at least 10 patches before training ({len(rows)} so far).")
    device = resolve_device(exp["pipeline"]["device"])
    margin = int(c["margin"])
    images, inside, points = build_patches(exp, rows, timestacks, device, on_event)
    steps, seed = int(c["train_steps"]), int(exp["labelling"]["seed"])
    gate = float(c["match_distance"])
    report = {"patches": len(rows), "centres": int(points.shape[0]), "summary": label_summary(rows)}

    def fit(mask: np.ndarray, phase: str):
        idx = torch.as_tensor(np.nonzero(mask)[0], device=device)
        remap = torch.full((len(rows),), -1, dtype=torch.long, device=device)
        remap[idx] = torch.arange(len(idx), device=device)
        pts = points[mask[points[:, 0].long().cpu().numpy()]] if points.shape[0] else points
        pts = pts.clone()
        if pts.shape[0]:
            pts[:, 0] = remap[pts[:, 0].long()].float()
        return centroid_net.train_patches(images[idx], inside[idx], pts, steps=steps, seed=seed, arch=c["architecture"],
                                          on_step=lambda s, l: s % 50 == 0 and on_event({"kind": "train", "phase": phase, "step": s, "steps": steps, "loss": l}))

    rng = np.random.default_rng(seed)
    test = np.zeros(len(rows), bool)
    test[rng.choice(len(rows), max(2, len(rows) // 5), replace=False)] = True
    net = fit(~test, "held-out check")
    held = [r for r, k in zip(rows, test) if k]
    scores = patch_scores(net, images[torch.as_tensor(np.nonzero(test)[0], device=device)])
    sweep = {th: evaluate(scores, inside[torch.as_tensor(np.nonzero(test)[0], device=device)], held, margin, th, gate) for th in THRESHOLDS}
    best = max(sweep, key=lambda th: (sweep[th]["f1"], -abs(th - 0.35)))
    report.update({"holdout_patches": int(test.sum()), "threshold": best, "model": sweep[best], "tracking": tracked_score(held, gate),
                   "thresholds": {str(k): v["f1"] for k, v in sweep.items()},
                   "count_change_patches": {"model": None, "tracking": None}})
    cc = [i for i, r in enumerate(held) if r["source"] == "count_change"]
    if cc:
        sel = torch.as_tensor(cc, device=device)
        ins = inside[torch.as_tensor(np.nonzero(test)[0], device=device)][sel]
        report["count_change_patches"] = {"n": len(cc), "model": evaluate(scores[sel], ins, [held[i] for i in cc], margin, best, gate),
                                          "tracking": tracked_score([held[i] for i in cc], gate)}
    net = fit(np.ones(len(rows), bool), "final model")
    path = model_path(exp)
    path.parent.mkdir(parents=True, exist_ok=True)
    centroid_net.save_checkpoint(net, str(path))
    report["trained"] = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
    report["model_file"] = str(path)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def cross_check(exp: dict, folds: int = 5, on_event=lambda e: None, timestacks: dict | None = None, threshold: float | None = None) -> list[dict]:
    """Every labelled patch judged by a seed finder that never saw it: split the patches into `folds`, train on the
    others, find centres in the held-out ones. Returns per patch {key, model_points (image coordinates), missed,
    extra, matched}. Disagreements are where the model is weak, or where a label is wrong."""
    from .track import centroid_net
    from .track.centroid_net import find_peaks
    c = exp["labelling"]["centres"]
    rows = centre_rows(read_labels(exp))
    if len(rows) < max(folds, 5):
        raise ValueError(f"Label at least {max(folds, 5)} patches before checking them ({len(rows)} so far).")
    device = resolve_device(exp["pipeline"]["device"])
    margin, gate, steps, seed = int(c["margin"]), float(c["match_distance"]), int(c["train_steps"]), int(exp["labelling"]["seed"])
    if threshold is None:
        report = read_report(exp)
        threshold = float(report["threshold"]) if report else 0.35
    images, inside, points = build_patches(exp, rows, timestacks, device, on_event)
    fold = np.random.default_rng(seed).permutation(len(rows)) % folds
    out: list[dict | None] = [None] * len(rows)
    for k in range(folds):
        train = fold != k
        idx = torch.as_tensor(np.nonzero(train)[0], device=device)
        remap = torch.full((len(rows),), -1, dtype=torch.long, device=device)
        remap[idx] = torch.arange(len(idx), device=device)
        pts = points[train[points[:, 0].long().cpu().numpy()]].clone() if points.shape[0] else points
        if pts.shape[0]:
            pts[:, 0] = remap[pts[:, 0].long()].float()
        net = centroid_net.train_patches(images[idx], inside[idx], pts, steps=steps, seed=seed + k, arch=c["architecture"],
                                         on_step=lambda s, l, k=k: s % 50 == 0 and on_event({"kind": "train", "phase": f"centres, fold {k + 1} of {folds}",
                                                                                               "step": s, "steps": steps, "loss": l}))
        held = np.nonzero(~train)[0]
        sel = torch.as_tensor(held, device=device)
        peaks = find_peaks(patch_scores(net, images[sel]), threshold).cpu().numpy()
        for j, i in enumerate(held):
            r = rows[i]
            p = peaks[peaks[:, 0] == j][:, 1:].astype(float)
            if len(p):
                p = p[inside[i, 0].cpu().numpy()[p[:, 0].astype(int), p[:, 1].astype(int)]]
            found = p.reshape(-1, 2) + [r["x0"] - margin, r["y0"] - margin]
            truth = np.array(r["points"], float).reshape(-1, 2)
            tp = match(truth, found, gate)
            out[i] = {"key": list(patch_key(r)), "model_points": np.round(found, 1).tolist(), "matched": tp,
                      "missed": len(truth) - tp, "extra": len(found) - tp}
        del net
    return out


def read_report(exp: dict) -> dict | None:
    path = model_path(exp).with_suffix(".json")
    if not exp["paths"]["results"] or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
