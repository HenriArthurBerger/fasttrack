"""Cell events: label a few dozen crops per category, train a small network, classify every candidate.

An observation is one cell at one frame of one position. Its crop is a small stack of frames
around that frame (`window` before and after), centred on the cell's track, in the chosen
channels. Categories are yours (neutral, division, death, abnormal by default).

Candidates come from the tracks: where a track ends early or starts late, and where a mother
splits, are where events happen, so the labeller samples those more often than random cells.

Files, inside the results folder:
    Labels/events.csv        your labels, one row per labelled crop
    Labels/event_model.pt    the trained classifier and what it was trained on
    Events/<id>.csv          classifier output per position: every candidate with class probabilities
"""
from __future__ import annotations

import csv
import datetime as dt
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .experiment import loaded_channels, pipeline_config
from .io import layout, store
from .track.measure import S

SOURCES = ["random", "track_end", "track_start", "division"]
LABEL_COLUMNS = ["position", "t", "track", "x", "y", "source", "category", "labelled"]


# ---------------------------------------------------------------------- candidates
def candidates(results: dict, min_track_frames: int = 3) -> dict[str, np.ndarray]:
    """(track, t) pairs per source, from a position's traces. Track ends at the last frame and starts at frame 0 are not events."""
    index = results["trace_index"]
    present = index >= 0
    T = index.shape[1]
    length = present.sum(1)
    first = np.argmax(present, axis=1)
    last = T - 1 - np.argmax(present[:, ::-1], axis=1)
    ok = length >= min_track_frames
    mothers = results["mother_track"]
    has_daughters = np.zeros(len(index), bool)
    has_daughters[mothers[mothers >= 0]] = True
    tracks = np.arange(len(index))
    out = {
        "track_end": np.stack([tracks, last], 1)[ok & (last < T - 1) & ~has_daughters],
        "track_start": np.stack([tracks, first], 1)[ok & (first > 0) & (mothers < 0)],
        "division": np.stack([tracks, last], 1)[has_daughters],
    }
    rows_t, rows_track = np.nonzero(present.T)
    out["random"] = np.stack([rows_track, rows_t], 1)[ok[rows_track]]
    return out


def track_xy(results: dict, track: int, t: int) -> tuple[float, float]:
    """The track's position at frame t, or at its nearest frame if it is absent then."""
    row = results["trace_index"][track]
    frames = np.nonzero(row >= 0)[0]
    nearest = frames[np.argmin(np.abs(frames - t))]
    r = row[nearest]
    return float(results["scalars"][r, S["x"]]), float(results["scalars"][r, S["y"]])


def sample(exp: dict, rng: random.Random, done: set, cache: dict) -> dict | None:
    """One random observation to label, by the source weights. `done` holds (position, t, track) already labelled."""
    lab = exp["labelling"]
    results_dir = Path(exp["paths"]["results"])
    positions = sorted(p.stem for p in (results_dir / "Traces").glob("*.h5"))
    if not positions:
        return None
    weights = [max(0.0, float(lab["sample"].get(s, 0))) for s in SOURCES]
    for _ in range(50):
        source = rng.choices(SOURCES, weights)[0] if sum(weights) else "random"
        key = rng.choice(positions)
        if key not in cache:
            r = store.load_results(results_dir, key)
            cache[key] = (r, candidates(r))
        r, cand = cache[key]
        pool = cand[source]
        if not len(pool):
            continue
        track, t = (int(v) for v in pool[rng.randrange(len(pool))])
        if (key, t, track) in done:
            continue
        x, y = track_xy(r, track, t)
        return {"position": key, "t": t, "track": track, "x": round(x, 1), "y": round(y, 1), "source": source}
    return None


# ---------------------------------------------------------------------- crops
def _channels(exp: dict) -> list[int]:
    """Indices into the loaded channels used for crops: the named ones, or the nuclear channel."""
    names = [c["name"] for c in loaded_channels(exp)]
    chosen = exp["labelling"]["channels"]
    if chosen:
        return [names.index(c) for c in chosen if c in names]
    nuclear = [i for i, c in enumerate(loaded_channels(exp)) if c["role"] == "nuclear"]
    return nuclear[:1] or [0]


def pad_frames(frames: dict[int, np.ndarray], size: int) -> dict[int, np.ndarray]:
    """Reflect-pad every frame by half a crop once, so crops near the border keep their size."""
    half = size // 2
    return {f: np.pad(a, ((0, 0), (half, half), (half, half)), mode="reflect") for f, a in frames.items()}


def crop_stack(padded: dict[int, np.ndarray], xy_of_t, t: int, window: int, size: int, T: int) -> np.ndarray:
    """(channels, 2*window+1, size, size) float32 from padded frames {frame: (C,X+size,Y+size)}.
    Frames beyond the movie repeat the edge frame. xy_of_t gives the crop's centre per frame (a track, or a fixed point)."""
    out = []
    for dt_ in range(-window, window + 1):
        f = min(max(t + dt_, 0), T - 1)
        x, y = xy_of_t(f)
        xi, yi = int(round(x)), int(round(y))
        out.append(padded[f][:, xi:xi + size, yi:yi + size])
    return np.stack(out, 1).astype(np.float32)


def normalise_crop(stack: np.ndarray) -> np.ndarray:
    """log2, then per channel: subtract the median, divide by a robust spread. Shape (C*frames, size, size)."""
    a = np.log2(np.maximum(stack, 0) + 1)
    C = a.shape[0]
    flat = a.reshape(C, -1)
    med = np.median(flat, axis=1)[:, None, None, None]
    spread = (np.percentile(flat, 99, axis=1) - np.percentile(flat, 50, axis=1))[:, None, None, None] + 0.1
    return ((a - med) / spread).reshape(-1, a.shape[2], a.shape[3]).astype(np.float32)


class FrameReader:
    """Reads the frames a set of crops needs from one position's TIFFs, each frame once."""

    def __init__(self, exp: dict, timestacks: dict | None = None, io_threads: int = 16):
        self.exp = exp
        self.cfg = pipeline_config(exp)
        self.timestacks = timestacks
        self.pool = ThreadPoolExecutor(io_threads)
        self.channel_idx = _channels(exp)

    def _ts(self, key: str) -> dict:
        if self.timestacks is None:
            self.timestacks = layout.discover_timestacks(self.cfg)
        return self.timestacks[key]

    def frames(self, key: str, wanted: set[int]) -> tuple[dict[int, np.ndarray], int]:
        ts = self._ts(key)
        chans = list(ts["channels"].values())
        sel = [chans[i] for i in self.channel_idx]
        wanted = sorted(f for f in wanted if 0 <= f < ts["frames"])

        shape = layout.frame_shape(ts)

        def one(path):
            try:
                return layout.read_frame(path, shape)
            except layout.UnreadableFrame:  # missing or corrupt: blank, as in tracking
                return np.zeros(shape, np.uint16)

        def read(f):
            return f, np.stack([one(c[f]) for c in sel])

        return dict(self.pool.map(read, wanted)), ts["frames"]

    def crop_at(self, key: str, t: int, x: float, y: float) -> np.ndarray:
        """(channels, 2*window+1, crop, crop) centred on (x, y) in every frame: the state classifier's crop. It does
        not follow a track, so points you added (which have none) are cut the same way as tracked cells."""
        lab = self.exp["labelling"]
        w, size = int(lab["window"]), int(lab["crop"])
        T = self._ts(key)["frames"]
        frames, _ = self.frames(key, {min(max(t + d, 0), T - 1) for d in range(-w, w + 1)})
        return crop_stack(pad_frames(frames, size), lambda f: (x, y), t, w, size, T)

    def crop(self, results: dict, obs: dict) -> np.ndarray:
        lab = self.exp["labelling"]
        w, size = int(lab["window"]), int(lab["crop"])
        T = results["trace_index"].shape[1]
        frames, _ = self.frames(obs["position"], {min(max(obs["t"] + d, 0), T - 1) for d in range(-w, w + 1)})
        return crop_stack(pad_frames(frames, size), lambda f: track_xy(results, obs["track"], f), obs["t"], w, size, T)

    def close(self) -> None:
        self.pool.shutdown(wait=False)


# ---------------------------------------------------------------------- labels file
def labels_path(exp: dict) -> Path:
    return Path(exp["paths"]["results"]) / exp["labelling"]["labels_file"]


def read_labels(exp: dict) -> list[dict]:
    path = labels_path(exp)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["t"], r["track"] = int(r["t"]), int(r["track"])
        r["x"], r["y"] = float(r["x"]), float(r["y"])
    return rows


def write_labels(exp: dict, rows: list[dict]) -> None:
    path = labels_path(exp)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, LABEL_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in LABEL_COLUMNS})
    tmp.replace(path)


def add_label(exp: dict, rows: list[dict], obs: dict, category: str) -> list[dict]:
    rows = [r for r in rows if (r["position"], r["t"], r["track"]) != (obs["position"], obs["t"], obs["track"])]
    rows.append({**obs, "category": category, "labelled": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"})
    write_labels(exp, rows)
    return rows


# ---------------------------------------------------------------------- classifier
class EventNet(nn.Module):
    """ResNet18 on a crop stack (channels x frames as input planes), global pooled, one logit per category."""

    def __init__(self, in_planes: int, classes: int):
        super().__init__()
        from torchvision.models import resnet18
        base = resnet18(weights=None, num_classes=classes)
        base.conv1 = nn.Conv2d(in_planes, 64, 7, 2, 3, bias=False)
        self.net = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _augment(x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        x = x.flip(-1)
    if random.random() < 0.5:
        x = x.flip(-2)
    x = torch.rot90(x, random.randrange(4), (-2, -1))
    return x * (1 + 0.1 * torch.randn(x.shape[0], 1, 1, 1, device=x.device)) + 0.05 * torch.randn_like(x)


def train_classifier(x: np.ndarray, y: np.ndarray, classes: list[str], steps: int = 600, device: str = "cuda:0",
                     holdout: float = 0.2, seed: int = 0, on_step=None) -> tuple[EventNet, dict]:
    """Train on crops x (N, planes, s, s) and integer labels y. Holds out a stratified share to report accuracy,
    then trains a final model on everything. Returns (model, report)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    test = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        n = int(round(holdout * len(idx))) if len(idx) >= 5 else 0
        test[rng.choice(idx, n, replace=False)] = True
    report = {"classes": classes, "counts": {classes[c]: int((y == c).sum()) for c in range(len(classes))}}

    def fit(mask):
        net = EventNet(x.shape[1], len(classes)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
        xs = torch.as_tensor(x[mask], device=device)
        ys = torch.as_tensor(y[mask], device=device)
        counts = torch.bincount(ys, minlength=len(classes)).float()
        weights = (1.0 / counts.clamp(min=1))[ys]  # class-balanced batches
        net.train()
        for step in range(steps):
            batch = torch.multinomial(weights, min(64, len(ys)), replacement=True)
            loss = F.cross_entropy(net(_augment(xs[batch])), ys[batch], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if on_step and step % 50 == 0:
                on_step(step, loss.item())
        return net.eval()

    if test.any() and (~test).sum() >= len(classes):
        net = fit(~test)
        pred = predict(net, x[test], device).argmax(1)
        truth = y[test]
        confusion = np.zeros((len(classes), len(classes)), int)
        np.add.at(confusion, (truth, pred), 1)
        report["holdout_accuracy"] = float((pred == truth).mean())
        report["holdout_n"] = int(test.sum())
        report["confusion"] = confusion.tolist()  # rows: true, columns: predicted
    net = fit(np.ones(len(y), bool))
    return net, report


@torch.inference_mode()
def predict(net: EventNet, x: np.ndarray, device: str = "cuda:0", chunk: int = 256) -> np.ndarray:
    """Class probabilities, averaged over the 8 flips and rotations."""
    out = []
    for i in range(0, len(x), chunk):
        xs = torch.as_tensor(x[i:i + chunk], device=device)
        p = 0
        for k in range(4):
            r = torch.rot90(xs, k, (-2, -1))
            p = p + F.softmax(net(r), 1) + F.softmax(net(r.flip(-1)), 1)
        out.append((p / 8).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, net.net.fc.out_features))


def model_path(exp: dict) -> Path:
    return Path(exp["paths"]["results"]) / exp["labelling"]["model_file"]


def save_model(path: Path, net: EventNet, report: dict, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "report": report, "settings": settings,
                "in_planes": net.net.conv1.in_channels}, path)


def load_model(path: Path, device: str = "cuda:0") -> tuple[EventNet, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    net = EventNet(blob["in_planes"], len(blob["report"]["classes"])).to(device)
    net.load_state_dict(blob["state"])
    return net.eval(), blob


def train_from_labels(exp: dict, on_event=lambda e: None, timestacks: dict | None = None) -> dict:
    """Read the labels, cut their crops from the raw TIFFs, train, and save the model. Returns the report."""
    lab = exp["labelling"]
    rows = [r for r in read_labels(exp) if r["category"] in lab["categories"]]
    classes = [c for c in lab["categories"] if any(r["category"] == c for r in rows)]
    if len(classes) < 2:
        raise ValueError("Label at least two categories before training.")
    reader = FrameReader(exp, timestacks, int(exp["files"]["io_threads"]))
    xs, ys, loaded = [], [], {}
    try:
        for i, r in enumerate(rows):
            if r["position"] not in loaded:
                loaded[r["position"]] = store.load_results(exp["paths"]["results"], r["position"])
            xs.append(normalise_crop(reader.crop(loaded[r["position"]], r)))
            ys.append(classes.index(r["category"]))
            on_event({"kind": "crops", "done": i + 1, "total": len(rows)})
    finally:
        reader.close()
    device = exp["pipeline"]["device"]
    net, report = train_classifier(np.stack(xs), np.array(ys), classes, int(lab["train_steps"]), device, seed=int(lab["seed"]),
                                   on_step=lambda s, l: on_event({"kind": "train", "step": s, "loss": l}))
    settings = {k: lab[k] for k in ("window", "crop", "channels")}
    report["trained"] = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
    save_model(model_path(exp), net, report, settings)
    return report


def classify_positions(exp: dict, on_event=lambda e: None, stop=None, timestacks: dict | None = None,
                       sources: tuple = ("track_end", "track_start", "division")) -> dict[str, int]:
    """Classify every candidate of every tracked position; write Events/<id>.csv. Returns candidates per position."""
    device = exp["pipeline"]["device"]
    net, blob = load_model(model_path(exp), device)
    classes = blob["report"]["classes"]
    settings = blob["settings"]
    follow = settings.get("follow", True)          # state models from the cell labeller crop at a fixed centre
    run_exp = {**exp, "labelling": {**exp["labelling"], **settings}}
    results_dir = Path(exp["paths"]["results"])
    positions = sorted(p.stem for p in (results_dir / "Traces").glob("*.h5"))
    reader = FrameReader(run_exp, timestacks, int(exp["files"]["io_threads"]))
    w, size = int(settings["window"]), int(settings["crop"])
    counts = {}
    try:
        for i, key in enumerate(positions):
            if stop is not None and stop.is_set():
                break
            r = store.load_results(results_dir, key)
            cand = candidates(r)
            obs = [(src, int(tr), int(t)) for src in sources for tr, t in cand[src]]
            if not obs:
                counts[key] = 0
                continue
            frames, T = reader.frames(key, {min(max(t + d, 0), r["trace_index"].shape[1] - 1) for _, _, t in obs for d in range(-w, w + 1)})
            frames = pad_frames(frames, size)
            if follow:
                at = lambda tr, t: (lambda f: track_xy(r, tr, f))
            else:
                at = lambda tr, t: (lambda f, xy=track_xy(r, tr, t): xy)
            crops = np.stack([normalise_crop(crop_stack(frames, at(tr, t), t, w, size, T)) for _, tr, t in obs])
            prob = predict(net, crops, device)
            out = results_dir / "Events" / f"{key}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t", "track", "x", "y", "source", "category", *[f"p_{c}" for c in classes]])
                for (src, tr, t), p in zip(obs, prob):
                    x, y = track_xy(r, tr, t)
                    writer.writerow([t, tr, round(x, 1), round(y, 1), src, classes[int(p.argmax())], *np.round(p, 4)])
            counts[key] = len(obs)
            del frames
            on_event({"kind": "classify", "done": i + 1, "total": len(positions), "id": key, "candidates": len(obs)})
    finally:
        reader.close()
    return counts


def read_events(results_dir: str | Path, key: str) -> list[dict]:
    path = Path(results_dir) / "Events" / f"{key}.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{**r, "t": int(r["t"]), "track": int(r["track"])} for r in csv.DictReader(f)]
