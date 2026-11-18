"""Binary nuclear mask from the notebook: rough threshold, then Laplacian-of-Gaussian cleanup."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..core.filters import box_blur
from .background import median_tile_values_with_positions


def generate_log_kernel(size: int, sigma: float) -> np.ndarray:
    """Laplacian of Gaussian, normalised by its sum as in the notebook.

    The sum is a truncation residue, so the kernel's behaviour depends strongly on `size`
    and `sigma`. The pipeline is tuned for size 11, sigma 2: change them together, and check.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    c = size // 2
    x, y = np.mgrid[-c:c + 1, -c:c + 1]
    r2 = (x**2 + y**2) / (2 * sigma**2)
    kernel = -(1 / (np.pi * sigma**4)) * (1 - r2) * np.exp(-r2)
    return kernel / np.sum(kernel)


def get_rough_binary_mask(corrected: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    p = cfg["mask"]
    nuc = corrected[:, cfg["nuclear_channel"]:cfg["nuclear_channel"] + 1]
    T = nuc.shape[0]
    values, _ = median_tile_values_with_positions(nuc, int(p["tiles_per_edge"]))
    rough = values.mean(dim=-1) + p["rough_std"] * values.std(dim=-1)
    flat = nuc.view(T, 1, -1)
    background = torch.where(flat < rough.unsqueeze(-1), flat, torch.nan)
    foreground = torch.where(flat > rough.unsqueeze(-1), flat, torch.nan)
    better = (_nanquantile(foreground, p["foreground_quantile"]) + _nanquantile(background, p["background_quantile"])) * p["threshold_scale"]
    mask = nuc > better[:, :, None, None]
    return box_blur(mask, 3) > 0.7, better[:, 0]


def _nanquantile(x: torch.Tensor, q: float) -> torch.Tensor:
    """torch.nanquantile on (T,1,N), tolerating a frame with no pixels on one side."""
    out = torch.nanquantile(x, q, dim=-1)
    return torch.nan_to_num(out, nan=0.0)


def get_clean_binary_mask(corrected: torch.Tensor, rough: torch.Tensor, cfg: dict) -> torch.Tensor:
    p = cfg["mask"]
    nuc = corrected[:, cfg["nuclear_channel"]:cfg["nuclear_channel"] + 1]
    size = int(p["log_kernel"])
    kernel = torch.as_tensor(generate_log_kernel(size, p["log_sigma"]), device=nuc.device, dtype=torch.float32)[None, None]
    response = F.conv2d(nuc, kernel, padding=size // 2)
    mask = rough & ~(response < 0)
    mask = box_blur(mask, 3) > 0.7
    return box_blur(mask, 3) > 0.55


def get_binary_mask(corrected: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """corrected (T,C,X,Y) -> (mask (T,1,X,Y) bool, thresholds (T,))."""
    rough, thresholds = get_rough_binary_mask(corrected, cfg)
    return get_clean_binary_mask(corrected, rough, cfg), thresholds


def compute_radius(mask: torch.Tensor) -> int:
    """Odd integer size scale: area / chord count along X. For a disc, about pi * r / 2."""
    m = mask.bool()
    starts = (m[:, :, 1:, :] & ~m[:, :, :-1, :]).sum() + m[:, :, 0, :].sum()
    if starts == 0:
        return 9
    return int(m.sum() // starts) // 2 * 2 + 1
