"""Seed finding on the pseudo distance map, by prominence (h-maxima).

A window-based peak finder has a built-in failure: any window big enough to suppress false
peaks inside a nucleus also deletes the small neighbour next to a big one. Prominence has no
window. A peak becomes a seed if it rises at least `h` above the saddle linking it to any
higher peak, so a separate blob always gets a seed and a bump on a flat top never does.

Everything is morphological reconstruction by dilation, a loop of GPU max-pools that stops
as soon as nothing changes.
"""
from __future__ import annotations

import torch

from ..core.filters import gaussian_blur, max_octagon, max_square, pseudo_distance


def reconstruct_by_dilation(marker: torch.Tensor, ceiling: torch.Tensor, max_iters: int) -> torch.Tensor:
    current = marker
    for i in range(max_iters):
        grown = torch.minimum(max_octagon(current, i), ceiling)
        if i % 4 == 3 and torch.equal(grown, current):
            break
        current = grown
    return current


def _component_ids(core: torch.Tensor, max_iters: int) -> torch.Tensor:
    """Connected components of a sparse bool map: each pixel takes the largest id it can reach."""
    T, _, X, Y = core.shape
    # Ids only need to be unique within a frame, which keeps them exact in float32.
    ids = torch.arange(1, X * Y + 1, device=core.device, dtype=torch.float32).view(1, 1, X, Y) * core
    for _ in range(max_iters):
        grown = max_square(ids) * core
        if torch.equal(grown, ids):
            break
        ids = grown
    return ids


def find_h_maxima(score: torch.Tensor, h: float, max_iters: int, chunk: int = 25) -> torch.Tensor:
    """One (t, x, y) per peak with prominence >= h. Returns (P,3) long, sorted by frame.

    Runs in chunks of frames to bound memory: frames are independent.
    """
    parts = []
    for start in range(0, score.shape[0], chunk):
        peaks = _h_maxima_chunk(score[start:start + chunk], h, max_iters)
        peaks[:, 0] += start
        parts.append(peaks)
    return torch.cat(parts) if parts else torch.zeros(0, 3, dtype=torch.long, device=score.device)


def _h_maxima_chunk(score: torch.Tensor, h: float, max_iters: int) -> torch.Tensor:
    rec = reconstruct_by_dilation((score - h).clamp(min=0), score, max_iters)
    core = (score > 0) & ((score - rec) >= h * 0.999)
    # A top can be several connected pixels. Keep the highest of each component.
    ids = _component_ids(core, 12)
    nz = core.nonzero(as_tuple=False)
    if nz.numel() == 0:
        return torch.zeros(0, 3, dtype=torch.long, device=score.device)
    pix_id = nz[:, 0].double() * (score.shape[-2] * score.shape[-1] + 1) + ids[nz[:, 0], 0, nz[:, 2], nz[:, 3]].double()
    values = score[nz[:, 0], 0, nz[:, 2], nz[:, 3]].double()
    order = torch.argsort(values)  # ascending, so the highest pixel is written last
    unique_ids, inverse = torch.unique(pix_id, return_inverse=True)
    winner = torch.zeros(unique_ids.numel(), dtype=torch.long, device=score.device)
    winner[inverse[order]] = order
    picked = nz[winner][:, [0, 2, 3]]
    return picked[torch.argsort(picked[:, 0], stable=True)]


def merge_nested_seeds(peaks: torch.Tensor, depth: torch.Tensor, overlap: float) -> torch.Tensor:
    """Drop a seed lying within `overlap` times the inscribed radius of a deeper seed.

    Two lobes of a dumbbell are farther apart than their depths; bumps along the ridge of a
    thin elongated nucleus are not. Returns a keep mask over `peaks`.
    """
    keep = torch.ones(peaks.shape[0], dtype=torch.bool, device=peaks.device)
    for t in peaks[:, 0].unique().tolist():
        rows = (peaks[:, 0] == t).nonzero(as_tuple=True)[0]
        if rows.numel() < 2:
            continue
        xy = peaks[rows, 1:].float()
        d = torch.cdist(xy, xy)
        depths = depth[rows]
        # Row i is dropped if some strictly deeper seed j (ties by index) has i in its circle.
        deeper = (depths[None, :] > depths[:, None]) | ((depths[None, :] == depths[:, None]) & (rows[None, :] < rows[:, None]))
        keep[rows[((d < overlap * depths[None, :]) & deeper).any(1)]] = False
    return keep


def get_centroids(mask: torch.Tensor, radius: float, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """mask (T,1,X,Y) -> (seeds (P,3) [t,x,y], distance (T,1,X,Y) float32)."""
    p = cfg["centroids"]
    cap = max(4, int(round(p["max_distance_r"] * radius)))
    distance = pseudo_distance(mask, cap)
    blurred = gaussian_blur(distance, p["blur_sigma"])
    peaks = find_h_maxima(blurred, p["h_prominence"], cap * 2)
    depth = blurred[peaks[:, 0], 0, peaks[:, 1], peaks[:, 2]]
    keep = depth > p["min_peak_r"] * radius
    peaks, depth = peaks[keep], depth[keep]
    if peaks.shape[0]:
        peaks = peaks[merge_nested_seeds(peaks, depth, p["seed_overlap"])]
    return peaks, distance
