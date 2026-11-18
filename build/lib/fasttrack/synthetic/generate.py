"""Synthetic timestacks with complete ground truth.

    image = illumination_field * (protein_abundance + background_autofluorescence)

plus shot and read noise. Channel 0 is nuclear only. Other channels mix nuclear and
cytoplasmic signal, and the last one behaves like a KTR with a time-varying C/N ratio.

Signal from overlapping cells adds, as it does in widefield, so an overlap is *brighter*
than either nucleus. That is what makes undersegmentation hard: there is no dim neck.

Simulation runs on the CPU once; rendering is batched on the GPU.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree

from ..core.filters import gaussian_blur
from ..config import merge

# Calibrated to a real mosaic H2B dataset: about 30 labelled nuclei per 1024 px field,
# about 18 x 11 px nuclei, median step 2.4 px/frame, reporters of tens of counts on backgrounds
# of 650-1650, gentle illumination falloff of 1.2-1.5x, dark fibres, an unlabelled monolayer.
SYNTH_DEFAULTS: dict = {
    "seed": 0,
    "frames": 100,
    "size": 1024,
    "channels": 3,
    "cells": 35,
    "nuc_radius": 6.0,       # geometric-mean radius of a newborn nucleus; area doubles per cycle
    "aspect": (1.1, 1.9),    # long axis / short axis
    "cyto_scale": (1.7, 2.6),
    "speed": 2.2,            # rms px/frame per axis, multiplied by a per-cell lognormal
    "speed_log_std": 0.5,
    "persistence": 0.8,
    "contact_tolerance": 1.0,  # repulsion starts at this fraction of the summed radii
    "repulsion": 0.3,
    "overlap_fraction": 0.25,  # cells that ignore repulsion, the source of real overlaps
    "cycle_frames": (140, 40), # mean, std
    "mitosis_frames": 3,
    "death_rate": 0.0003,
    "illum_sigma": (1.0, 1.6),    # fraction of field size
    "illum_offset": (0.03, 0.12), # fraction of field size
    "autofluor": ((550.0, 900.0), (500.0, 800.0), (1200.0, 2000.0)),  # per channel; last repeats
    "autofluor_structure": 0.01,
    "monolayer_texture": 0.004,   # fine texture of unlabelled cells
    "nuclear_mean": (800.0, 2500.0),
    "nuclear_log_std": (0.55, 0.8),
    "reporter_mean": (40.0, 150.0),
    "reporter_log_std": (0.4, 1.0),
    "cn_ratio": (0.3, 0.9),       # concentration cytoplasm / nucleus for reporter channels
    "camera_gain": 1.5,           # counts per photon
    "read_noise": 6.0,
    "debris": 2,
    "fibers": 1,                  # dark curved fibres shadowing the field
    "media_change_frame": None,   # None = pick one in the second half of the movie
    "media_change_step": 1.1,
    "stage_jumps": 1,
    "stage_jump_px": 25.0,
    "blur_frames": 1,             # frames slightly out of focus
    # --- Optional overrides, set by synthetic.fit to match a real timestack. None = use the ranges above.
    "channel_mean_fixed": None,     # per channel, mean nuclear abundance
    "channel_log_std_fixed": None,  # per channel, between-cell log spread
    "cn_fixed": None,               # per channel, cytoplasm / nucleus concentration
    "autofluor_fixed": None,        # per channel
    "abund_step": 0.012,            # per-frame log change of abundance: one value, or one per channel
    "abund_jump_prob": 0.0,         # per channel and frame, chance of an abrupt switch (FUCCI-like)
    "abund_jump_log2": 1.5,         # size of a switch, in log2 units, up or down
    "cn_step": None,                # per-frame log change of C/N per channel; None = 0.05 on the last channel only
    "cycle_schedule": None,         # [[frame, mean cycle frames], ...]: division pace changing over the movie
    "stage_steps": None,            # [[frame, dx, dy], ...]: exact stage moves, replacing stage_jumps
}

# Calibrated to a second real well: about 400 smaller nuclei per 1152 px field, 4% of pixels in mask.
POPULATED_PRESET: dict = {"size": 1152, "cells": 480, "nuc_radius": 5.0, "aspect": (1.05, 1.6)}

# Far denser than the pipeline is designed for. A stress test, not a target.
DENSE_PRESET: dict = {"cells": 600, "nuc_radius": 11.0, "aspect": (1.0, 1.8), "speed": 1.2, "repulsion": 0.6,
                      "contact_tolerance": 0.75, "overlap_fraction": 0.12}


def _channel_specs(cfg: dict, rng: np.random.Generator) -> dict:
    C, size = cfg["channels"], cfg["size"]
    offsets = rng.uniform(*cfg["illum_offset"], C) * rng.choice([-1, 1], C)
    offsets_y = rng.uniform(*cfg["illum_offset"], C) * rng.choice([-1, 1], C)
    log_nuc = np.log(cfg["nuclear_mean"])
    log_rep = np.log(cfg["reporter_mean"])
    mean = np.exp(rng.uniform(log_rep[0], log_rep[1], C))
    mean[0] = np.exp(rng.uniform(log_nuc[0], log_nuc[1]))
    log_std = rng.uniform(*cfg["reporter_log_std"], C)
    log_std[0] = rng.uniform(*cfg["nuclear_log_std"])
    auto_ranges = cfg["autofluor"]
    auto = [rng.uniform(*auto_ranges[min(c, len(auto_ranges) - 1)]) for c in range(C)]
    cn = np.concatenate([[0.0], rng.uniform(*cfg["cn_ratio"], max(C - 1, 0))])
    for key, target in (("channel_mean_fixed", mean), ("channel_log_std_fixed", log_std), ("cn_fixed", cn)):
        if cfg[key] is not None:
            target[:] = np.asarray(cfg[key], dtype=float)[:C]
    if cfg["autofluor_fixed"] is not None:
        auto = list(np.asarray(cfg["autofluor_fixed"], dtype=float)[:C])
    return {
        "mean": mean,
        "log_std": log_std,
        "autofluor": np.array(auto),
        "illum_sigma": rng.uniform(*cfg["illum_sigma"], C) * size,
        "illum_cx": (0.5 + offsets) * size,
        "illum_cy": (0.5 + offsets_y) * size,
        # concentration in cytoplasm / concentration in nucleus, per channel
        "cn_base": cn,
    }


def simulate_cells(cfg: dict) -> dict:
    """Run the cell simulation. Returns one row per nucleus per frame, plus track table."""
    rng = np.random.default_rng(cfg["seed"])
    specs = _channel_specs(cfg, rng)
    T, C, size = cfg["frames"], cfg["channels"], cfg["size"]
    r0 = cfg["nuc_radius"]
    margin = 3 * r0

    def spawn(n, phase=None):
        a = np.pi * r0**2
        return {
            "id": np.zeros(n, np.int64),
            "parent": np.full(n, -1, np.int64),
            "x": rng.uniform(-margin, size + margin, n),
            "y": rng.uniform(-margin, size + margin, n),
            "vx": rng.normal(0, cfg["speed"], n),
            "vy": rng.normal(0, cfg["speed"], n),
            "pace": np.exp(rng.normal(-cfg["speed_log_std"] ** 2 / 2, cfg["speed_log_std"], n)),
            "scale": np.exp(rng.normal(0, 0.10, n)),
            "theta": rng.uniform(0, np.pi, n),
            "ecc": np.sqrt(rng.uniform(*cfg["aspect"], n)),
            "h2": rng.uniform(0, 0.05, n),
            "h3": rng.uniform(0, 0.05, n),
            "p2": rng.uniform(0, 2 * np.pi, n),
            "p3": rng.uniform(0, 2 * np.pi, n),
            "phase": rng.uniform(0, 1, n) if phase is None else phase,
            "cycle": np.maximum(40, rng.normal(*cfg["cycle_frames"], n)),
            "abund": np.exp(rng.normal(0, 1, (n, C)) * specs["log_std"]) * specs["mean"],
            "cn": np.tile(specs["cn_base"], (n, 1)) * np.exp(rng.normal(0, 0.15, (n, C))),
            "cyto": rng.uniform(*cfg["cyto_scale"], n),
            "tex": rng.integers(0, 4096, (n, 2)),
            "free": rng.random(n) < cfg["overlap_fraction"],
        }

    state = spawn(cfg["cells"])
    n_cells = cfg["cells"]
    state["id"] = np.arange(n_cells)
    next_id = n_cells

    stage = np.zeros((T, 2))
    offset = np.zeros(2)
    if cfg["stage_steps"] is not None:
        steps = {int(f): np.array([dx, dy], dtype=float) for f, dx, dy in cfg["stage_steps"]}
        jump_frames = sorted(f for f in steps if f < T)
    else:
        jump_frames = sorted(rng.choice(np.arange(5, T - 1), min(cfg["stage_jumps"], max(T - 6, 0)), replace=False).tolist()) if T > 7 else []
        steps = {f: rng.normal(0, cfg["stage_jump_px"], 2) for f in jump_frames}
    for t in range(T):
        if t in steps:
            offset = offset + steps[t]
        stage[t] = offset

    def per_channel(value):
        return np.broadcast_to(np.asarray(value, dtype=float), (C,)) if np.ndim(value) else np.full(C, float(value))

    abund_step = per_channel(cfg["abund_step"])
    jump_prob = per_channel(cfg["abund_jump_prob"])
    jump_size = per_channel(cfg["abund_jump_log2"])
    cn_step = per_channel(cfg["cn_step"]) if cfg["cn_step"] is not None else np.array([0.0] * (C - 1) + [0.05])
    schedule = np.asarray(cfg["cycle_schedule"], dtype=float) if cfg["cycle_schedule"] else None

    columns = {k: [] for k in ("t", "cell", "parent", "x", "y", "true_x", "true_y", "rx", "ry", "theta", "h2", "h3",
                               "p2", "p3", "cyto", "mitotic")}
    abund_rows, cn_rows, tex_rows = [], [], []
    tracks: dict[int, dict] = {int(i): {"parent": -1, "start": 0, "end": 0} for i in range(n_cells)}

    for t in range(T):
        n = state["x"].size
        # --- motion: persistent random walk plus contact repulsion
        fx = np.zeros(n)
        fy = np.zeros(n)
        radius_now = r0 * state["scale"] * np.sqrt(1 + state["phase"])
        if n > 1:
            pts = np.stack([state["x"], state["y"]], 1)
            pairs = cKDTree(pts).query_pairs(float(cfg["contact_tolerance"] * 2.8 * r0 * 2), output_type="ndarray")
            if pairs.size:
                i, j = pairs[:, 0], pairs[:, 1]
                dx, dy = state["x"][i] - state["x"][j], state["y"][i] - state["y"][j]
                d = np.hypot(dx, dy) + 1e-3
                reach = cfg["contact_tolerance"] * (radius_now[i] + radius_now[j])
                push = np.clip(reach - d, 0, None) / reach * cfg["repulsion"] * r0
                keep = ~(state["free"][i] | state["free"][j])
                push = push * keep
                np.add.at(fx, i, push * dx / d)
                np.add.at(fy, i, push * dy / d)
                np.add.at(fx, j, -push * dx / d)
                np.add.at(fy, j, -push * dy / d)
        rho = cfg["persistence"]
        noise = cfg["speed"] * math.sqrt(1 - rho**2)
        state["vx"] = rho * state["vx"] + rng.normal(0, noise, n)
        state["vy"] = rho * state["vy"] + rng.normal(0, noise, n)
        state["x"] = state["x"] + state["vx"] * state["pace"] + fx
        state["y"] = state["y"] + state["vy"] * state["pace"] + fy
        state["theta"] = state["theta"] + rng.normal(0, 0.04, n)
        # --- signalling: drift, abrupt switches (FUCCI-like), and C/N pulses (KTR-like)
        state["abund"] = state["abund"] * np.exp(rng.normal(0, 1, (n, C)) * abund_step)
        switch = rng.random((n, C)) < jump_prob
        if switch.any():
            state["abund"] = state["abund"] * np.where(switch, np.exp2(rng.choice([-1.0, 1.0], (n, C)) * jump_size), 1.0)
        for c in range(1, C):
            if cn_step[c] > 0:
                log_cn = np.log(state["cn"][:, c] + 1e-3)
                log_base = np.log(max(specs["cn_base"][c], 1e-3))
                state["cn"][:, c] = np.exp(log_cn + 0.05 * (log_base - log_cn) + rng.normal(0, cn_step[c], n))
        # --- cell cycle, with an optional pace that changes over the movie
        pace = 1.0
        if schedule is not None:
            pace = cfg["cycle_frames"][0] / np.interp(t, schedule[:, 0], schedule[:, 1])
        state["phase"] = state["phase"] + pace / state["cycle"]
        frames_left = (1 - state["phase"]) * state["cycle"] / pace
        mitotic = frames_left < cfg["mitosis_frames"]

        # --- record this frame
        radius = r0 * state["scale"] * np.sqrt(1 + np.minimum(state["phase"], 1.0))
        radius_draw = np.where(mitotic, radius * 0.8, radius)
        ecc = np.where(mitotic, np.maximum(state["ecc"], 1.3), state["ecc"])
        visible = (state["x"] > -margin) & (state["x"] < size + margin) & (state["y"] > -margin) & (state["y"] < size + margin)
        idx = np.nonzero(visible)[0]
        columns["t"].append(np.full(idx.size, t))
        columns["cell"].append(state["id"][idx])
        columns["parent"].append(state["parent"][idx])
        columns["x"].append(state["x"][idx] + stage[t, 0])
        columns["y"].append(state["y"][idx] + stage[t, 1])
        columns["true_x"].append(state["x"][idx])
        columns["true_y"].append(state["y"][idx])
        columns["rx"].append(radius_draw[idx] * ecc[idx])
        columns["ry"].append(radius_draw[idx] / ecc[idx])
        for key in ("theta", "h2", "h3", "p2", "p3", "cyto"):
            columns[key].append(state[key][idx])
        columns["mitotic"].append(mitotic[idx].astype(np.int8))
        boost = np.where(mitotic, 1 / 0.8**2, 1.0)[idx, None]  # condensed chromatin is brighter
        abund_rows.append(state["abund"][idx] * boost)
        cn_rows.append(state["cn"][idx])
        tex_rows.append(state["tex"][idx])
        for cid in state["id"][idx]:
            tracks[int(cid)]["end"] = t

        # --- divisions and deaths take effect from the next frame
        dead = rng.random(n) < cfg["death_rate"]
        dividing = state["phase"] >= 1.0
        keep = ~(dead | dividing)
        div_idx = np.nonzero(dividing & ~dead)[0]
        if div_idx.size:
            m = div_idx.size
            kids = spawn(2 * m, phase=np.zeros(2 * m))
            ids = np.arange(next_id, next_id + 2 * m)
            next_id += 2 * m
            kids["id"] = ids
            kids["parent"] = np.repeat(state["id"][div_idx], 2)
            theta = np.repeat(state["theta"][div_idx], 2)
            sign = np.tile([1.0, -1.0], m)
            sep = r0 * 1.2
            for key, trig in (("x", np.cos), ("y", np.sin)):
                kids[key] = np.repeat(state[key][div_idx], 2) + sign * sep * trig(theta)
            kids["vx"] = sign * 1.5 * np.cos(theta) + rng.normal(0, 0.3, 2 * m)
            kids["vy"] = sign * 1.5 * np.sin(theta) + rng.normal(0, 0.3, 2 * m)
            for key in ("scale", "cyto", "free", "pace"):
                kids[key] = np.repeat(state[key][div_idx], 2)
            kids["abund"] = np.repeat(state["abund"][div_idx], 2, axis=0) * np.exp(rng.normal(0, 0.04, (2 * m, C)))
            kids["cn"] = np.repeat(state["cn"][div_idx], 2, axis=0)
            for i in range(2 * m):
                tracks[int(ids[i])] = {"parent": int(kids["parent"][i]), "start": t + 1, "end": t + 1}
        state = {k: v[keep] for k, v in state.items()}
        if div_idx.size:
            state = {k: np.concatenate([state[k], kids[k]]) for k in state}

    truth = {k: np.concatenate(v) for k, v in columns.items()}
    truth["abund"] = np.concatenate(abund_rows)
    truth["cn"] = np.concatenate(cn_rows)
    truth["tex"] = np.concatenate(tex_rows)
    truth["stage"] = stage
    truth["jump_frames"] = np.array(jump_frames)
    truth["tracks"] = tracks
    truth["specs"] = specs

    debris_n = cfg["debris"]
    truth["debris"] = np.stack([
        rng.uniform(0, size, debris_n), rng.uniform(0, size, debris_n), rng.uniform(5, 14, debris_n),
        rng.uniform(0.5, 2, debris_n),
    ], 1) if debris_n else np.zeros((0, 4))
    truth["fibers"] = [rng.uniform(0, size, (3, 2)) for _ in range(cfg["fibers"])]
    media = cfg["media_change_frame"]
    truth["media_change_frame"] = int(media if media is not None else rng.integers(T // 2, max(T // 2 + 1, T)))
    jitter = rng.choice(np.arange(T), min(cfg["blur_frames"], T), replace=False) if cfg["blur_frames"] else np.array([], int)
    truth["blur_frames"] = jitter
    return truth


def _illumination(specs: dict, size: int, device) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    fields = []
    for c in range(len(specs["mean"])):
        cx, cy, s = (float(specs[k][c]) for k in ("illum_cx", "illum_cy", "illum_sigma"))
        fields.append(torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s**2)))
    return torch.stack(fields)


def _fiber_shadow(fibers: list, size: int, device) -> torch.Tensor:
    """Dark curved fibres, as a multiplicative field. Real fields show these."""
    if not fibers:
        return torch.ones(size, size, device=device)
    paint = torch.zeros(1, 1, size, size, device=device)
    s = torch.linspace(0, 1, 2000, device=device)[:, None]
    for p in fibers:
        a, b, c = (torch.as_tensor(p[i], device=device, dtype=torch.float32)[None] for i in range(3))
        curve = ((1 - s) ** 2 * a + 2 * s * (1 - s) * b + s**2 * c).long().clamp(0, size - 1)
        paint[0, 0, curve[:, 0], curve[:, 1]] = 1.0
    blurred = gaussian_blur(paint, 3.0)[0, 0]
    return 1 - 0.75 * (blurred / blurred.max().clamp(min=1e-6))


def _stamp_frame(truth: dict, rows: np.ndarray, cfg: dict, texture: torch.Tensor, device, chunk: int = 300):
    """Paint one frame's protein abundance. Returns (protein (C,X,Y), labels (X,Y))."""
    size, C = cfg["size"], cfg["channels"]
    protein = torch.zeros(C, size * size, device=device)
    label_out = torch.zeros(size * size, dtype=torch.int32, device=device)
    tex_n = texture.shape[0]
    n_rows = rows.size
    for start in range(0, n_rows, chunk):
        sel = rows[start:start + chunk]
        get = lambda k: torch.as_tensor(truth[k][sel], device=device, dtype=torch.float32)
        x, y, rx, ry, th, h2, h3, p2, p3, cyto = (get(k) for k in ("x", "y", "rx", "ry", "theta", "h2", "h3", "p2", "p3", "cyto"))
        reach = int(np.ceil((truth["rx"][sel] * truth["cyto"][sel]).max() * 1.25)) + 3
        grid = torch.arange(-reach, reach + 1, device=device)
        cu, cv = torch.meshgrid(grid, grid, indexing="ij")
        ix = x.round().long()[:, None, None] + cu
        iy = y.round().long()[:, None, None] + cv
        u = ix.float() - x[:, None, None]
        v = iy.float() - y[:, None, None]
        cos, sin = th.cos()[:, None, None], th.sin()[:, None, None]
        a = u * cos + v * sin
        b = -u * sin + v * cos
        phi = torch.atan2(b, a)
        wobble = 1 + h2[:, None, None] * torch.cos(2 * phi + p2[:, None, None]) + h3[:, None, None] * torch.cos(3 * phi + p3[:, None, None])
        rho = torch.sqrt((a / rx[:, None, None]) ** 2 + (b / ry[:, None, None]) ** 2) / wobble
        mean_r = torch.sqrt(rx * ry)[:, None, None]
        nuc = torch.sigmoid((1 - rho) * mean_r / 0.9)
        rho_c = rho / (cyto[:, None, None] * (1 + 0.5 * h3[:, None, None] * torch.cos(2 * phi + p3[:, None, None])))
        cell = torch.sigmoid((1 - rho_c) * mean_r * cyto[:, None, None] / 2.5)
        inside = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
        flat = (ix * size + iy)[inside]
        tex_xy = torch.as_tensor(truth["tex"][sel], device=device)
        tex = texture[(cu[None] + tex_xy[:, 0, None, None]) % tex_n, (cv[None] + tex_xy[:, 1, None, None]) % tex_n]
        abund = torch.as_tensor(truth["abund"][sel], device=device, dtype=torch.float32)
        cn = torch.as_tensor(truth["cn"][sel], device=device, dtype=torch.float32)
        for c in range(C):
            nuc_part = abund[:, c, None, None] * nuc * (1 + (0.22 if c == 0 else 0.08) * tex)
            cyto_part = abund[:, c, None, None] * cn[:, c, None, None] * cell * (1 - nuc)
            protein[c].index_add_(0, flat, (nuc_part + cyto_part)[inside])
        cell_id = torch.as_tensor(truth["cell"][sel], device=device, dtype=torch.int32)[:, None, None].expand_as(nuc)
        body = inside & (nuc > 0.5)
        label_out[(ix * size + iy)[body]] = cell_id[body] + 1
    return protein.view(C, size, size), label_out.view(size, size)


def render(truth: dict, cfg: dict, t0: int = 0, t1: int | None = None, device="cuda:0", return_labels: bool = False):
    """Render frames [t0, t1) to a uint16 tensor (T, C, X, Y) on `device`."""
    t1 = cfg["frames"] if t1 is None else t1
    size, C = cfg["size"], cfg["channels"]
    specs = truth["specs"]
    gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 1 + t0)
    illum = _illumination(specs, size, device)
    tex_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 7)
    texture = gaussian_blur(torch.randn(1, 1, 512, 512, device=device, generator=tex_gen), 2.0)[0, 0]
    texture = (texture / texture.std()).clamp(-2.5, 2.5)
    structure = gaussian_blur(torch.randn(1, C, size, size, device=device, generator=tex_gen), size / 12)[0]
    structure = structure / structure.flatten(1).std(1)[:, None, None]
    auto = torch.as_tensor(specs["autofluor"], device=device, dtype=torch.float32)[:, None, None]
    auto = auto * (1 + cfg["autofluor_structure"] * structure)
    fine = gaussian_blur(torch.randn(1, C, size, size, device=device, generator=tex_gen), 2.5)[0]
    auto = auto * (1 + cfg["monolayer_texture"] * fine / fine.flatten(1).std(1)[:, None, None])
    transmission = _fiber_shadow(truth["fibers"], size, device)

    frames = torch.empty(t1 - t0, C, size, size, dtype=torch.uint16, device=device)
    labels = torch.empty(t1 - t0, size, size, dtype=torch.int32, device=device) if return_labels else None
    order = np.argsort(truth["t"], kind="stable")
    starts = np.searchsorted(truth["t"][order], np.arange(cfg["frames"] + 1))
    debris = torch.as_tensor(truth["debris"], device=device, dtype=torch.float32)
    coords = torch.arange(size, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    for t in range(t0, t1):
        rows = order[starts[t]:starts[t + 1]]
        protein, lab = _stamp_frame(truth, rows, cfg, texture, device)
        off = torch.as_tensor(truth["stage"][t], device=device, dtype=torch.float32)
        mean = torch.as_tensor(specs["mean"], device=device, dtype=torch.float32)[:, None, None]
        for d in debris:
            blob = torch.sigmoid((d[2] - torch.hypot(xx - d[0] - off[0], yy - d[1] - off[1])) / 1.5)
            protein = protein + blob * mean * d[3]
        step = cfg["media_change_step"] if t >= truth["media_change_frame"] else 1.0
        shadow = transmission
        if off.abs().sum() > 0:
            shadow = torch.roll(transmission, (int(off[0].round()), int(off[1].round())), (0, 1))
        expected = illum * shadow * (protein + auto * step)
        if t in truth["blur_frames"]:
            expected = gaussian_blur(expected[None], 2.5)[0]
        gain, read = cfg["camera_gain"], cfg["read_noise"]
        noisy = torch.poisson((expected / gain).clamp(min=0), generator=gen) * gain
        noisy = noisy + torch.randn_like(noisy) * read
        frames[t - t0] = noisy.round().clamp(0, 65535).to(torch.uint16)
        if return_labels:
            labels[t - t0] = lab
    return (frames, labels) if return_labels else frames


def make_synth_config(updates: dict | None = None) -> dict:
    return merge(SYNTH_DEFAULTS, updates)


def generate(updates: dict | None = None, device="cuda:0", return_labels: bool = False):
    """One call: returns (frames, truth, cfg), or (frames, labels, truth, cfg)."""
    cfg = make_synth_config(updates)
    truth = simulate_cells(cfg)
    out = render(truth, cfg, device=device, return_labels=return_labels)
    if return_labels:
        return out[0], out[1], truth, cfg
    return out, truth, cfg
