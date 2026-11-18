"""Batched image filters on (T, C, X, Y) tensors. All run on the tensor's own device."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def odd(value: float, minimum: int = 1) -> int:
    k = max(minimum, int(round(value)))
    return k if k % 2 == 1 else k + 1


def box_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """Mean over a k*k window with reflected edges. Uses avg_pool, so cost ignores k."""
    p = k // 2
    return F.avg_pool2d(F.pad(x.float(), (p, p, p, p), mode="reflect"), k, stride=1)


def box_sum(x: torch.Tensor, k: int) -> torch.Tensor:
    return box_blur(x, k) * (k * k)


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


def log_kernel(size: int, sigma: float, device) -> torch.Tensor:
    """Zero-mean negative Laplacian of Gaussian: positive on blobs, negative outside.

    The notebook divided the kernel by its own sum. That sum is a truncation residue
    near zero, so the kernel's gain swung from 0.84 to 11791 with kernel size and it
    picked up a DC term. Subtracting the mean makes the sign of the response meaningful.
    """
    if size % 2 == 0:
        raise ValueError("LoG kernel size must be odd")
    c = size // 2
    coords = torch.arange(-c, c + 1, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    r2 = (xx**2 + yy**2) / (2 * sigma**2)
    k = (1 - r2) * torch.exp(-r2)
    k = k - k.mean()
    return k / k.abs().sum()


def conv_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    p = kernel.shape[-1] // 2
    return F.conv2d(F.pad(x.float(), (p, p, p, p), mode="reflect"), kernel[None, None])


def max_square(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, stride=1, padding=1)


def max_cross(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(
        F.max_pool2d(x, (1, 3), stride=1, padding=(0, 1)),
        F.max_pool2d(x, (3, 1), stride=1, padding=(1, 0)),
    )


def max_octagon(x: torch.Tensor, step: int) -> torch.Tensor:
    """Alternating cross/square dilation: a much rounder metric than either alone."""
    return max_cross(x) if step % 2 == 0 else max_square(x)


def pseudo_distance(mask: torch.Tensor, max_iters: int, chunk: int = 25) -> torch.Tensor:
    """Pseudo distance map: the number of octagonal erosions each pixel survives.

    The edge of the frame counts as inside, so cells clipped by the field are not
    pulled toward the border. Capped at `max_iters` so one large aggregate cannot
    set the cost of the whole batch.
    """
    distance = torch.zeros_like(mask, dtype=torch.float32)
    for start in range(0, mask.shape[0], chunk):
        section = mask[start:start + chunk].float()
        current = section
        total = section.clone()
        for i in range(max_iters - 1):
            current = -max_octagon(-current, i) * section
            if not current.any():
                break
            total += current
        distance[start:start + chunk] = total
    return distance


def tiles(x: torch.Tensor, k: int) -> torch.Tensor:
    """(T, C, X, Y) -> (T, C, k*k, tile_pixels), dropping any remainder."""
    T, C, X, Y = x.shape
    tx, ty = X // k, Y // k
    t = x[:, :, : k * tx, : k * ty].view(T, C, k, tx, k, ty).permute(0, 1, 2, 4, 3, 5)
    return t.contiguous().view(T, C, k * k, tx * ty)
