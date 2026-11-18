"""Binary nuclear mask: rough threshold, then Laplacian-of-Gaussian refinement."""
from __future__ import annotations

import math

import torch

from ..core.filters import box_blur, conv_same, gaussian_blur, log_kernel, max_octagon


def noise_floor(image: torch.Tensor, k: float, stride: int = 2) -> torch.Tensor:
    """Per-frame threshold: median + k * robust sd. image (T,1,X,Y). Returns (T,).

    Opinionated: assumes most pixels are background, which sparse and mosaic cultures
    satisfy by a wide margin. Otsu has no second mode to find in such a field, and cuts
    the background noise in two. The sd comes from the half below the median, which
    cells cannot inflate.
    """
    sample = image[:, 0, ::stride, ::stride].flatten(1)
    median = sample.median(dim=1).values
    below = (median[:, None] - sample).clamp(min=0)
    # Median absolute deviation of the lower half: the 50th percentile of |dev| below the
    # median is the 75th of the half-normal scaled by 0.6745.
    lower = torch.where(sample <= median[:, None], below, torch.nan)
    sd = 1.4826 * lower.nanmedian(dim=1).values
    return median + k * sd


def rough_mask(nuclei: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Deliberately permissive: the LoG step, not this threshold, sets the final boundary."""
    p = cfg["mask"]
    smooth = gaussian_blur(nuclei, p["smooth_sigma"])
    thr = noise_floor(smooth, p["noise_k"])
    # One blurred or bright frame must not lose its cells: keep cuts near the batch's.
    centre = thr.median()
    thr = thr.clamp(centre * 0.5, centre * 2.0) if centre > 0 else thr
    return smooth > thr[:, None, None, None], thr


def clean_mask(mask: torch.Tensor) -> torch.Tensor:
    """Remove single-pixel specks, then close single-pixel holes."""
    mask = box_blur(mask, 3) > 0.7
    return box_blur(mask, 3) > 0.45


def restore_enclosed(rough: torch.Tensor, keep: torch.Tensor, iterations: int) -> torch.Tensor:
    """Separate necks from texture holes among the pixels the LoG cut.

    A neck between two nuclei reaches the outside of the rough mask, so it stays cut. A dark
    chromatin spot is enclosed, so it is refilled. Found by flooding inward from the outside
    through cut pixels only.
    """
    cut = rough & ~keep
    flood = ~rough
    for i in range(iterations):
        grown = max_octagon(flood.float(), i).bool() & cut
        new = grown & ~flood
        if not new.any():
            break
        flood = flood | new
    return rough & ~(cut & flood)


def get_binary_mask(corrected: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """corrected (T,C,X,Y) -> (mask (T,1,X,Y) bool, thresholds (T,))."""
    p = cfg["mask"]
    nuclei = corrected[:, cfg["nuclear_channel"]:cfg["nuclear_channel"] + 1]
    rough, thr = rough_mask(nuclei, cfg)
    response = conv_same(nuclei, log_kernel(int(p["log_kernel"]), p["log_sigma"], nuclei.device))
    # Cut only where the curvature is strongly negative relative to local brightness. Dim
    # necks between touching nuclei go; flat, textured nuclear interiors stay. The relative
    # test needs no intensity threshold, so dim and bright nuclei are treated alike.
    local = box_blur(nuclei, int(p["log_kernel"])).clamp(min=0)
    keep = response >= -p["neck_alpha"] * local
    return clean_mask(restore_enclosed(rough, keep, int(p["hole_fill_iters"]))), thr


def estimate_radius(mask: torch.Tensor) -> float:
    """Typical nuclear radius from area over chord count along both axes.

    For an ellipse this is 2ab / (a + b) whatever its orientation. Using one axis alone made
    a lone thin nucleus lying along that axis read as three times too large.
    """
    m = mask.bool()
    along_x = (m[:, :, 1:, :] & ~m[:, :, :-1, :]).sum() + m[:, :, 0, :].sum()
    along_y = (m[:, :, :, 1:] & ~m[:, :, :, :-1]).sum() + m[:, :, :, 0].sum()
    chords = along_x + along_y
    if chords == 0:
        return 8.0
    return float(m.sum() / chords) * 4 / math.pi
