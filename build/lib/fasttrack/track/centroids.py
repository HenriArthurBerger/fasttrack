"""Centroids from the notebook: pseudo distance map, blur, local maxima."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.filters import gaussian_blur


def get_pseudo_distance(mask: torch.Tensor) -> torch.Tensor:
    """Peel the mask from alternating sides. Each pixel records the step that removed it."""
    erode = mask.float()
    distance = torch.zeros_like(erode)
    T, C, X, Y = mask.shape
    pad_x = torch.zeros(T, C, 1, Y, device=mask.device)
    pad_y = torch.zeros(T, C, X, 1, device=mask.device)
    i = 1
    while erode.max() == 1:
        if i % 2 == 0:
            dx = torch.diff(erode, dim=-2, prepend=pad_x)
            dy = torch.diff(erode, dim=-1, prepend=pad_y)
        else:
            dx = torch.diff(erode, dim=-2, append=pad_x)
            dy = torch.diff(erode, dim=-1, append=pad_y)
        edges = (dx.abs() + dy.abs()) > 0
        distance[edges] = i
        erode[edges] = 0
        i += 1
    return distance


def get_peaks(distance: torch.Tensor, mask: torch.Tensor, radius: int, cfg: dict) -> torch.Tensor:
    """Blur the distance map twice, then take local maxima deep enough to be nuclei.

    The notebook's TVT.GaussianBlur(kernel_size) draws a random sigma from 0.1-2.0 on every
    call. `blur_sigma` fixes it, so a run is reproducible.
    """
    sigma = cfg["centroids"]["blur_sigma"]
    blurred = gaussian_blur(gaussian_blur(distance, sigma), sigma)
    pooled = F.max_pool2d(blurred, 5, stride=1, padding=2)
    return (blurred == pooled) & mask & (blurred > radius // 3), blurred


def merge_nested_seeds(peaks: torch.Tensor, depth: torch.Tensor, overlap: float) -> torch.Tensor:
    """Keep-mask over peaks: drop a seed inside `overlap` inscribed radii of a deeper seed.

    A nucleus with a bumpy distance ridge gets several close seeds, and a second seed on one nucleus
    is the largest single cause of broken tracks. The lobes of a true doublet are farther apart
    than their depth, so they survive.
    """
    keep = torch.ones(peaks.shape[0], dtype=torch.bool, device=peaks.device)
    for t in peaks[:, 0].unique().tolist():
        rows = (peaks[:, 0] == t).nonzero(as_tuple=True)[0]
        if rows.numel() < 2:
            continue
        d = torch.cdist(peaks[rows, 1:].float(), peaks[rows, 1:].float())
        deep = depth[rows]
        deeper = (deep[None, :] > deep[:, None]) | ((deep[None, :] == deep[:, None]) & (rows[None, :] < rows[:, None]))
        keep[rows[((d < overlap * deep[None, :]) & deeper).any(1)]] = False
    return keep


def get_centroids(mask: torch.Tensor, radius: int, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """mask (T,1,X,Y) -> (peaks (P,3) [t,x,y], distance (T,1,X,Y))."""
    distance = get_pseudo_distance(mask)
    peak_map, blurred = get_peaks(distance, mask, radius, cfg)
    peaks = peak_map.nonzero(as_tuple=False)[:, [0, 2, 3]]
    overlap = float(cfg["centroids"]["seed_overlap"])
    if overlap > 0 and peaks.shape[0]:
        peaks = peaks[merge_nested_seeds(peaks, blurred[peaks[:, 0], 0, peaks[:, 1], peaks[:, 2]], overlap)]
    return peaks, distance
