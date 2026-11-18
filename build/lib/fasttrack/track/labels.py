"""Label maps from the notebook.

label_map: +label is nucleus, -label is cytoplasm ring. border_labels: +label on the ring of
pixels just outside the nucleus. Labels are local to a frame and a sub-section:

    label = 1000 * section + index, with index counted from 1 within (frame, section)

That encoding is what lets the measurement step work one sub-section at a time.
"""
from __future__ import annotations

import torch

from ..core.filters import max_square


def get_local_labels(x: int, y: int, peak_t: torch.Tensor, peak_x: torch.Tensor, peak_y: torch.Tensor,
                     subsections: int = 7) -> tuple[torch.Tensor, int]:
    """Local label per peak, plus the largest index in any (frame, section)."""
    xbin = x // subsections + 1
    ybin = y // subsections + 1
    section = (peak_y // ybin * subsections + peak_x // xbin).long()
    region = peak_t.long() * subsections**2 + section
    order = torch.argsort(region, stable=True)
    sorted_region = region[order]
    first = torch.ones_like(sorted_region, dtype=torch.bool)
    first[1:] = sorted_region[1:] != sorted_region[:-1]
    position = torch.arange(region.numel(), device=region.device)
    start = position.where(first, torch.zeros_like(position)).cummax(0).values
    local_sorted = position - start + 1
    local = torch.zeros_like(local_sorted)
    local[order] = local_sorted
    section_max = int(local.max().item()) if local.numel() else 0
    if section_max >= 1000:
        raise ValueError(f"{section_max} cells in one frame's sub-section; the label encoding allows 999. Raise measure.subsections.")
    return local + 1000 * section, section_max


def get_label_map(peaks: torch.Tensor, mask: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """Grow compact labels 1..P from peaks through the mask with a 3x3 max pool."""
    labels = torch.zeros(mask.shape, device=mask.device, dtype=torch.float32)
    labels[peaks[:, 0], 0, peaks[:, 1], peaks[:, 2]] = torch.arange(1, peaks.shape[0] + 1, device=mask.device, dtype=torch.float32)
    free = labels == 0
    for _ in range(int(distance.max().item())):
        dilated = max_square(labels)
        grow = (dilated > 0) & free & mask
        if not grow.any():
            break
        labels = torch.where(grow, dilated, labels)
        free = free & ~grow
    return labels


def encode_local_labels(nuclear: torch.Tensor, subsections: int) -> tuple[torch.Tensor, int]:
    """Convert any unique positive labels to the local encoding, by each nucleus's centroid."""
    T, _, X, Y = nuclear.shape
    nz = (nuclear > 0).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return nuclear, 0
    ids, inverse = torch.unique(nuclear[nz[:, 0], 0, nz[:, 2], nz[:, 3]].long(), return_inverse=True)
    n = ids.numel()
    device = nuclear.device
    count = torch.bincount(inverse, minlength=n).float()
    cx = torch.zeros(n, device=device).index_add_(0, inverse, nz[:, 2].float()) / count
    cy = torch.zeros(n, device=device).index_add_(0, inverse, nz[:, 3].float()) / count
    frame = torch.zeros(n, device=device, dtype=torch.long)
    frame[inverse] = nz[:, 0]
    local, section_max = get_local_labels(X, Y, frame, cx.long(), cy.long(), subsections)
    out = torch.zeros_like(nuclear)
    out[nz[:, 0], 0, nz[:, 2], nz[:, 3]] = local[inverse].to(nuclear.dtype)
    return out, section_max


def get_border_map(labels: torch.Tensor, mask: torch.Tensor, gap: int = 1, width: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (label_map with -label on the cytoplasm ring, border_labels), as float32.

    Grows labels gap + width pixels out from the nuclear mask. The first `gap` pixel layers are skipped
    (their first layer is the nuclear border) and the next `width` layers are the cytoplasm ring. The
    defaults (1, 2) are the notebook's ring. A max pool is exact here only because labels stay below 2**24.
    """
    binary = mask.float()
    erode = 1 - binary
    border = max_square(binary) - binary
    grown = labels
    outside_gap = erode
    for step in range(gap + width):
        if step == gap:
            outside_gap = erode
        grown = grown + max_square(grown) * erode
        erode = -max_square(-erode)
    if width == 0:
        outside_gap = erode
    border_labels = border * grown
    ring = outside_gap * (1 - erode)
    return grown * binary - grown * ring, border_labels


def mask_ring(label_map: torch.Tensor, significant: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop cytoplasm-ring pixels that are not significantly above background.

    Returns (masked label_map, ring_labels): ring_labels holds every ring pixel's label before masking
    (0 elsewhere), so the kept fraction of each ring can be counted afterwards."""
    ring = label_map < 0
    ring_labels = torch.where(ring, -label_map, torch.zeros_like(label_map))
    return torch.where(ring & ~significant, torch.zeros_like(label_map), label_map), ring_labels
