"""Batched image filters on (T, C, X, Y) tensors. All run on the tensor's own device."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def box_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """Mean over a k*k window with reflected edges. Uses avg_pool, so cost ignores k."""
    p = k // 2
    return F.avg_pool2d(F.pad(x.float(), (p, p, p, p), mode="reflect"), k, stride=1)


def gaussian_kernel_1d(sigma: float, device, truncate: float = 3.0) -> torch.Tensor:
    radius = max(1, int(math.ceil(truncate * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian, deterministic (torchvision's default sigma is random)."""
    if sigma <= 0:
        return x.float()
    g = gaussian_kernel_1d(sigma, x.device)
    p = g.numel() // 2
    channels = x.shape[1]
    y = F.pad(x.float(), (p, p, p, p), mode="reflect")
    y = F.conv2d(y, g.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)
    return F.conv2d(y, g.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)


def max_square(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, stride=1, padding=1)
