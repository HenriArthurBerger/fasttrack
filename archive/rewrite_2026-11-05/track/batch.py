"""Process one batch of frames: raw pixels in, measurement tensors out."""
from __future__ import annotations

import time
from typing import Callable

import torch

from . import background, centroids, labels as labelling, masking, measure, undersegmentation as us


class _Clock:
    """Stage timer. Synchronises the GPU, so each stage's time is its real time."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.times: dict[str, float] = {}
        self._last = self._now()

    def _now(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def lap(self, name: str) -> None:
        now = self._now()
        self.times[name] = self.times.get(name, 0.0) + now - self._last
        self._last = now


def segment(corrected: torch.Tensor, cfg: dict, clock: _Clock | None = None) -> dict:
    """corrected -> mask, radius, seeds, nuclear labels (first pass)."""
    clock = clock or _Clock(corrected.device)
    mask, thresholds = masking.get_binary_mask(corrected, cfg)
    radius = masking.estimate_radius(mask)
    clock.lap("mask")
    if cfg["centroids"]["method"] == "cnn":
        from . import centroid_net
        peaks = centroid_net.predict_peaks(corrected, mask, radius, cfg)
        from ..core.filters import pseudo_distance
        distance = pseudo_distance(mask, max(4, int(round(cfg["centroids"]["max_distance_r"] * radius))))
    else:
        peaks, distance = centroids.get_centroids(mask, radius, cfg)
    clock.lap("centroids")
    nuclear = labelling.grow_labels(peaks, mask, distance, cfg)
    clock.lap("labels")
    return {"mask": mask, "thresholds": thresholds, "radius": radius, "peaks": peaks, "distance": distance, "nuclear": nuclear}


def second_pass(corrected: torch.Tensor, nuclear: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict]:
    """Find and split undersegmented nuclei. Modifies and returns the nuclear label map."""
    p = cfg["underseg"]
    stats = {"static_splits": 0, "temporal_splits": 0}
    for _ in range(int(p["rounds"]) if p["static"] else 0):
        compact, frames = labelling.compact_labels(nuclear)
        nuclear = compact
        scalars, channels = measure.measure_batch(corrected, nuclear.to(torch.int32), frames, cfg)
        score = us.score_undersegmentation(scalars, channels, cfg["nuclear_channel"])
        area = scalars[:, measure.S["nuc_area"]]
        median = area.nanmedian().clamp(min=1)
        flagged = scalars[:, measure.S["label"]][score > p["score_threshold"]]
        nuclear, accepted = us.split_flagged(nuclear, flagged, float(p["min_child_area_frac"] * median), int(p["kmeans_iters"]))
        stats["static_splits"] += accepted
        if accepted == 0:
            break
    if p["temporal"]:
        compact, frames = labelling.compact_labels(nuclear)
        nuclear = compact
        counts = torch.bincount(nuclear.long().view(-1))[1:]
        median = counts[counts > 0].float().median().item() if counts.numel() else 1.0
        stats["temporal_splits"] = us.temporal_split(nuclear, median, cfg)
    return nuclear, stats


def process_batch(raw: torch.Tensor, cfg: dict, on_layer: Callable[[str, torch.Tensor], None] | None = None) -> dict:
    """raw (T,C,X,Y) -> {scalars (N,S), channels (N,C,K), radius, stats, timings, ...}.

    `on_layer` receives intermediates by name for the GUI cache. It is the only hook, so
    no figure or file output can slow the pipeline when nothing is listening.
    """
    with torch.inference_mode():
        clock = _Clock(raw.device)
        corrected, diag = background.correct_background(raw, cfg)
        clock.lap("background")
        seg = segment(corrected, cfg, clock)
        nuclear = seg.pop("nuclear")
        seg.pop("distance")  # only needed for label growth; a full batch's worth of memory
        stats = {"first_pass_cells": seg["peaks"].shape[0]}
        if cfg["underseg"]["enabled"]:
            nuclear, split_stats = second_pass(corrected, nuclear, cfg)
            stats.update(split_stats)
            clock.lap("second_pass")
        nuclear, frames = labelling.compact_labels(nuclear)
        signed = labelling.add_cytoplasm_ring(nuclear, cfg)
        clock.lap("ring")
        scalars, channels = measure.measure_batch(corrected, signed, frames, cfg)
        scalars[:, measure.S["underseg_score"]] = us.score_undersegmentation(scalars, channels, cfg["nuclear_channel"])
        clock.lap("measure")
        stats["cells"] = int(scalars.shape[0])
        if on_layer is not None:
            on_layer("raw", raw)
            on_layer("corrected", corrected)
            on_layer("binary_mask", seg["mask"])
            on_layer("label_map", signed)
            on_layer("centroids", scalars[:, [measure.S["t"], measure.S["x"], measure.S["y"]]])
            clock.lap("gui_cache")
        return {
            "scalars": scalars, "channels": channels, "radius": seg["radius"], "stats": stats,
            "timings": clock.times, "diag": diag, "corrected": corrected, "mask": seg["mask"], "signed": signed,
        }
