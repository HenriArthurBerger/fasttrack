"""Process one batch of frames: raw pixels in, measurement tensors out."""
from __future__ import annotations

import time
from typing import Callable

import torch

from .background import mask_channels  # noqa: F401  (re-exported)
from . import background, centroids, labels as labelling, linking, masking, measure, undersegmentation as us


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


def get_frame_shifts(mask: torch.Tensor, cfg: dict, previous: torch.Tensor | None = None) -> torch.Tensor:
    """Pixel shift of each frame from the one before, (T, 2). `previous` links across batches.

    Runs on the binary mask. On corrected images the camera's fixed pattern pins the peak at
    (0, 0): real stage jumps were missed in 2 of 6 real positions. Shifts under `min_shift` are
    cell motion rather than stage motion, and are set to zero.
    """
    frames = mask[:, 0].float()
    first = torch.zeros(1, 2, device=frames.device) if previous is None else linking.find_shifts(previous[None], frames[:1])
    shifts = first if frames.shape[0] == 1 else torch.cat([first, linking.find_shifts(frames[:-1], frames[1:])])
    small = shifts.abs().amax(dim=1) < cfg["link"]["min_shift"]
    shifts[small] = 0
    return shifts


def segment(corrected: torch.Tensor, cfg: dict, clock: _Clock | None = None) -> dict:
    """corrected -> mask, radius, seeds, compact nuclear labels (first pass)."""
    clock = clock or _Clock(corrected.device)
    rough, thresholds = masking.get_rough_binary_mask(corrected, cfg)
    mask = masking.get_clean_binary_mask(corrected, rough, cfg)
    radius = masking.compute_radius(mask)
    clock.lap("mask")
    if cfg["centroids"]["method"] == "cnn":
        from . import centroid_net
        peaks = centroid_net.predict_peaks(corrected, mask, radius, cfg)
        distance = centroids.get_pseudo_distance(mask)
    else:
        peaks, distance = centroids.get_centroids(mask, radius, cfg)
    clock.lap("centroids")
    nuclear = labelling.get_label_map(peaks, mask, distance)
    clock.lap("labels")
    return {"mask": mask, "rough": rough, "thresholds": thresholds, "radius": radius, "peaks": peaks, "nuclear": nuclear, "distance": distance}


def _send_layers(on_layer, raw, corrected, seg, nuclear, signed, border_labels, scalars) -> None:
    """Hand every intermediate to the listener. Ordered as the pipeline runs."""
    peaks = torch.zeros_like(seg["mask"], dtype=torch.uint8)
    p = seg["peaks"]
    peaks[p[:, 0], 0, p[:, 1], p[:, 2]] = 1
    on_layer("raw", raw)
    on_layer("corrected", corrected)
    on_layer("rough_mask", seg["rough"])
    on_layer("binary_mask", seg["mask"])
    on_layer("distance", seg["distance"])
    on_layer("centroids", peaks)
    on_layer("nuclear_labels", nuclear)
    on_layer("label_map", signed)
    on_layer("border_labels", border_labels)
    on_layer("underseg_score", _score_map(signed, scalars))
    on_layer("table", scalars[:, [measure.S["t"], measure.S["x"], measure.S["y"]]])


def _score_map(signed: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
    """Paint each nucleus with its undersegmentation score, 0-100."""
    out = torch.zeros_like(signed, dtype=torch.uint8)
    if scalars.shape[0] == 0:
        return out
    T = signed.shape[0]
    span = int(scalars[:, measure.S["label"]].max().item()) + 1
    lut = torch.zeros(T * span, device=signed.device, dtype=torch.uint8)
    keys = scalars[:, measure.S["t"]].long() * span + scalars[:, measure.S["label"]].long()
    lut[keys] = (torch.nan_to_num(scalars[:, measure.S["underseg_score"]]) * 100).clamp(0, 100).to(torch.uint8)
    nz = (signed > 0).nonzero(as_tuple=False)
    out[nz[:, 0], 0, nz[:, 2], nz[:, 3]] = lut[nz[:, 0] * span + signed[nz[:, 0], 0, nz[:, 2], nz[:, 3]].long()]
    return out


def process_batch(raw: torch.Tensor, cfg: dict, on_layer: Callable[[str, torch.Tensor], None] | None = None,
                  previous_mask: torch.Tensor | None = None) -> dict:
    """raw (T,C,X,Y) -> {scalars (N,S), channels (N,C,K), frame_shift (T,2), radius, ...}.

    `on_layer` receives intermediates by name for the GUI cache. It is the only hook, so
    no figure or file output can slow the pipeline when nothing is listening.
    """
    with torch.inference_mode():
        clock = _Clock(raw.device)
        corrected, diag = background.correct_background(raw, cfg)
        clock.lap("background")
        seg = segment(corrected, cfg, clock)
        shift = get_frame_shifts(seg["mask"], cfg, previous_mask)
        clock.lap("shift")
        nuclear = seg["nuclear"]
        stats = {"first_pass_cells": seg["peaks"].shape[0], "temporal_splits": 0}
        if cfg["underseg"]["enabled"]:
            counts = torch.bincount(nuclear.long().view(-1))[1:]
            median = counts[counts > 0].float().median().item() if (counts > 0).any() else 1.0
            stats["temporal_splits"] = us.temporal_split(nuclear, median, cfg)
            clock.lap("second_pass")
        encoded, section_max = labelling.encode_local_labels(nuclear, int(cfg["measure"]["subsections"]))
        m = cfg["measure"]
        label_map, border_labels = labelling.get_border_map(encoded, seg["mask"], int(m.get("ring_gap", 1)), int(m.get("ring_width", 2)))
        ring_labels = None
        if "significant" in diag:
            label_map, ring_labels = labelling.mask_ring(label_map, diag.pop("significant"))
        clock.lap("ring")
        scalars, channels = measure.measure_batch(corrected, label_map, border_labels, section_max, seg["radius"], cfg, ring_labels)
        scalars[:, measure.S["underseg_score"]] = us.score_undersegmentation(scalars, channels, cfg["nuclear_channel"])
        clock.lap("measure")
        stats["cells"] = int(scalars.shape[0])
        signed = label_map.to(torch.int32)
        if on_layer is not None:
            _send_layers(on_layer, raw, corrected, seg, nuclear, signed, border_labels, scalars)
            clock.lap("gui_cache")
        return {
            "scalars": scalars, "channels": channels, "radius": seg["radius"], "stats": stats, "timings": clock.times,
            "diag": diag, "corrected": corrected, "mask": seg["mask"], "signed": signed, "frame_shift": shift,
        }
