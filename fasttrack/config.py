"""Tracking parameters: one plain nested dict passed through the tracking code.

Users never edit this directly: the experiment file (experiment.py) holds these values under
`pipeline`, and `experiment.pipeline_config` builds this dict from it.

A dict (not a class) on purpose, so a new feature only needs a new key here. Every
key has a default; a YAML file only has to state what differs. Values that scale with
nuclear size are multiples of the *measured* radius so they survive a change of
objective or binning.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "device": "cuda:0",
    "batch_size": 100,          # first batch; grows if batch_auto and memory allows
    "batch_auto": True,
    "batch_max": 500,
    "vram_fraction": 0.75,      # of the card's memory that auto batching aims to use
    "prefetch_batches": 1,      # batches read ahead of the one being processed
    "nuclear_channel": 0,
    "seed": 0,
    "data": {
        "root": "",
        "output": "",
        # Per-frame TIFFs. Named groups: well (optional), t (optional), pos, chan.
        # With no t group, frames are ordered by sorted filename.
        "pattern": r"(?P<well>[A-H]\d{2})_Time(?P<t>\d+)_xy(?P<pos>\d+)c(?P<chan>\d+)\.tiff?$",
        "channels": [],        # empty = every channel found, in sorted order
        "channel_names": [],
        "stride": 1,           # keep every Nth timepoint
        "first_frame": 0,      # timepoints of the full movie, before the stride
        "last_frame": 0,       # exclusive; 0 = to the end
        "max_frames": 0,       # 0 = all
        "io_threads": 16,
    },
    "background": {
        "tiles_per_edge": 21,
        "tile_quantile": 0.01,
        "scale_quantile": 0.02,
        "chunk_frames": 25,      # bounds quantile memory; the result is unchanged
    },
    "mask": {
        "tiles_per_edge": 21,
        "rough_std": 5.0,
        "foreground_quantile": 0.01,
        "background_quantile": 0.98,
        "threshold_scale": 0.6,
        "log_sigma": 2.0,
        "log_kernel": 11,
    },
    "centroids": {
        "method": "notebook",    # "notebook" | "cnn"
        "blur_sigma": 1.5,       # px; the notebook's blur drew this at random from 0.1-2.0
        "seed_overlap": 1.75,    # drop a seed inside this many inscribed radii of a deeper one; 0 = off
        "cnn_checkpoint": "",
        "cnn_threshold": 0.35,
    },
    "measure": {
        "subsections": 7,
        "even_median": "lower",
        "ring_gap": 1,           # px skipped between the nuclear mask and the cytoplasm ring (1 = the border layer)
        "ring_width": 2,         # px of cytoplasm ring beyond the gap
        "cyto_mask": False,      # keep only ring pixels significantly above background in the mask channel
        "cyto_mask_channel": "", # channel name(s), comma-separated; empty = all but the nuclear channel. Kept if any passes (OR)
        "cyto_mask_sigma": 2.0,  # threshold: background level + this many background-noise sigmas
        "cyto_mask_smooth": 1.5, # px Gaussian blur before the test, so it follows the cell, not pixel noise
        "cyto_min_pixels": 5,    # fewer ring pixels left than this: the cell's cytoplasm values are NaN
        "cyto_min_kept": 0.5,    # with the mask: less of the ring kept than this (reporter not expressed): NaN too
    },
    "underseg": {
        "enabled": True,
        "min_child_area_frac": 0.2,  # of the median nuclear area
        "kmeans_iters": 6,
        "max_rounds": 25,        # each round lets a split travel one frame along an overlap
    },
    "link": {
        "method": "multipass",   # "multipass" | "notebook" (one pass) | "tracks" (voting, gaps)
        "correct_shift": True,   # FFT pixel shift, referenced to frame 0, removed before distances
        "min_shift": 3,          # px; smaller common shifts are cell motion
        "max_travel": 0.0,       # px per frame; 0 = estimate from the data
        "max_travel_factor": 2.0,  # estimate = 99th percentile of confident moves * this
        "max_travel_floor": 15.0,
        "move_weight": 5.0,      # adds move / max_travel * weight to the similarity cost
        "division": True,
        # Division evidence
        "division_mass_tol": 0.7,    # |log2((d1 + d2) / mother)|
        "division_sister_tol": 1.0,  # |log2(d1 / d2)|
        "division_net_count": True,  # the local cell count must rise
        "division_radius": 25.0,     # px, around the mother, for the local count
        # Cleanup pass
        "max_gap": 3,            # frames a track may stay undetected
        "gap_penalty": 0.5,
        "cleanup_cost_quantile": 0.995,  # of accepted link costs; dearer pairs stay unlinked
        # Options for method "tracks"
        "history": 5,
        "feature_clip": 6.0,
        "distance_weight": 1.0,
        "max_cost": 8.0,
        "assignment": "lsa",
        "lsa_max_cells": 1000,
        "division_distance": 35.0,
    },
    "output": {
        "save_measurements": True,
        "overwrite": False,     # re-process timestacks that already have results
        "gui_cache": False,     # in-memory HDF5 intermediates for the last timestack
        "gui_cache_layers": ["binary_mask", "label_map"],  # raw frames are read from the TIFFs
    },
}


def merge(base: dict, updates: dict | None) -> dict:
    """Recursively overlay `updates` on a deep copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def make_config(updates: dict | None = None) -> dict:
    return merge(DEFAULTS, updates)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return make_config(yaml.safe_load(f))


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
