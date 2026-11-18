"""Label maps. +label is nucleus, -label is the cytoplasmic ring. 0 is unassigned.

Labels are compact integers 1..N in a batch, ordered by frame. That one ordering replaces
the notebook's 49 sub-section local labels: measurement can scatter every pixel of every
frame at once, chunks are plain label ranges, and no cells-per-tile limit can overflow.
"""
from __future__ import annotations

import torch

from ..core.filters import max_octagon


def seed_labels(peaks: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Write peaks (P,3) as labels 1..P. Float32 because max_pool needs it (exact to 2**24)."""
    if peaks.shape[0] >= 2**24:
        raise ValueError("Too many cells in one batch for float32 labels; reduce batch_size")
    labels = torch.zeros(shape, device=peaks.device, dtype=torch.float32)
    labels[peaks[:, 0], 0, peaks[:, 1], peaks[:, 2]] = torch.arange(1, peaks.shape[0] + 1, device=peaks.device, dtype=torch.float32)
    return labels


def grow_geodesic(labels: torch.Tensor, mask: torch.Tensor, iterations: int) -> torch.Tensor:
    """Grow seeds through the mask at equal speed (seed-distance Voronoi)."""
    for i in range(iterations):
        dilated = max_octagon(labels, i)
        new = mask & (labels == 0) & (dilated > 0)
        if not new.any():
            break
        labels = torch.where(new, dilated, labels)
    return labels


def grow_watershed(labels: torch.Tensor, mask: torch.Tensor, distance: torch.Tensor, steps_per_level: int = 2) -> torch.Tensor:
    """Flood from the deepest pixels outward, so boundaries settle in necks.

    Seed-distance growth would cut two touching nuclei of unequal size midway between seeds.
    """
    top = int(distance.max().item())
    level_step = 2
    for level in range(top, 0, -level_step):
        allowed = mask & (distance >= level)
        for i in range(steps_per_level):
            dilated = max_octagon(labels, i)
            new = allowed & (labels == 0) & (dilated > 0)
            labels = torch.where(new, dilated, labels)
    return grow_geodesic(labels, mask, top * 2)


def grow_labels(peaks: torch.Tensor, mask: torch.Tensor, distance: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Seeds -> nuclear labels. Frames are independent, so this runs in chunks to bound memory."""
    seeds = seed_labels(peaks, mask.shape)
    chunk = int(cfg["measure"]["chunk_frames"])
    for start in range(0, mask.shape[0], chunk):
        s, m, d = seeds[start:start + chunk], mask[start:start + chunk], distance[start:start + chunk]
        if cfg["labels"]["growth"] == "watershed":
            seeds[start:start + chunk] = grow_watershed(s, m, d)
        else:
            seeds[start:start + chunk] = grow_geodesic(s, m, int(d.max().item()) * 2)
    return seeds


def add_cytoplasm_ring(labels: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Return signed int32 labels with a -label ring around each nucleus.

    The ring starts `cyto_gap` pixels out, which keeps nuclear blur out of the cytoplasm
    estimate, and is `cyto_width` wide. Rings never enter any nucleus.
    """
    gap, width = int(cfg["labels"]["cyto_gap"]), int(cfg["labels"]["cyto_width"])
    out = labels.to(torch.int32)
    chunk = int(cfg["measure"]["chunk_frames"])
    for start in range(0, labels.shape[0], chunk):
        section = labels[start:start + chunk]
        nuclear = section > 0
        grown = section
        result = out[start:start + chunk]
        for i in range(gap + width):
            dilated = max_octagon(grown, i)
            grown = torch.where(grown == 0, dilated, grown)
            if i >= gap:
                ring = (result == 0) & (grown > 0) & ~nuclear
                result = torch.where(ring, -grown.to(torch.int32), result)
        out[start:start + chunk] = result
    return out


def compact_labels(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Renumber labels (T,1,X,Y) to 1..N ordered by frame. Empty labels disappear.

    Returns (labels, frame_of_label (N,)). Only `labels` positive entries are considered.
    """
    device = labels.device
    top = int(labels.max().item())
    if top == 0:
        return labels, torch.zeros(0, dtype=torch.long, device=device)
    lut_time = torch.full((top + 1,), -1, dtype=torch.long, device=device)
    nz = (labels > 0).nonzero(as_tuple=False)
    lut_time[labels[nz[:, 0], 0, nz[:, 2], nz[:, 3]].long()] = nz[:, 0]  # labels are unique to a frame
    lut_time[0] = -1
    present = (lut_time >= 0).nonzero(as_tuple=True)[0]
    order = present[torch.argsort(lut_time[present] * (top + 1) + present)]
    lut = torch.zeros(top + 1, dtype=labels.dtype, device=device)
    lut[order] = torch.arange(1, order.numel() + 1, device=device, dtype=labels.dtype)
    return lut[labels.long()], lut_time[order]
