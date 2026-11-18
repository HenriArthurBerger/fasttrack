"""Background and illumination correction.

Model: raw = I(x, y) * (protein + autofluorescence). Background pixels therefore measure
I * autofluorescence, a smooth surface fitted as a polynomial in log space. Degree 2 is exact
for a Gaussian field, off-centre included; real optics need more. Dividing by the surface
after subtracting it recovers protein on a common scale:

    corrected = (raw - bg) * bg_max / bg  =  protein * I_max

Assumes at least `rescale_quantile` of each frame is background.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.filters import tiles


def _monomials(x: torch.Tensor, y: torch.Tensor, degree: int) -> torch.Tensor:
    """All x^i y^j with i + j <= degree, stacked on the last dim."""
    return torch.stack([x**i * y**j for i in range(degree + 1) for j in range(degree + 1 - i)], dim=-1)


def _solve_weighted(A: torch.Tensor, Z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Batched WLS. A (n,p); Z, w (C,n). Returns (C,p). Zero weights drop a row."""
    AtA = torch.einsum("np,cn,nq->cpq", A, w, A)
    AtZ = torch.einsum("np,cn->cp", A, w * Z)
    eye = 1e-6 * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(AtA + eye, AtZ.unsqueeze(-1)).squeeze(-1)


def fit_background(raw: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit the smooth background. Returns (shape (C,X,Y), level (T,C), diag); bg = shape * level.

    The optics do not change over a batch, so one shape per channel is fitted to low tile
    quantiles pooled over time. Pooling also removes transient debris. Only the overall
    level is per frame, which follows background steps from a media change.
    """
    p = cfg["background"]
    T, C, X, Y = raw.shape
    device = raw.device
    d = int(p["downsample"])
    small = F.avg_pool2d(raw.float(), d, stride=d)  # averaging reduces the shot-noise bias of a low quantile
    k = int(p["tiles_per_edge"])
    tiled = tiles(small, k)
    kth = max(1, int(round(p["tile_quantile"] * tiled.shape[-1])))
    values = torch.kthvalue(tiled, kth, dim=-1).values  # (T,C,k*k)

    logs = torch.log2(values.clamp(min=1.0))
    Z = (logs - logs.median(dim=-1, keepdim=True).values).median(dim=0).values  # (C,k*k)

    sx, sy = small.shape[-2:]
    tx, ty = sx // k, sy // k
    ix = torch.arange(k, device=device, dtype=torch.float32)
    cx = ((ix + 0.5) * tx * d).repeat_interleave(k) / X * 2 - 1
    cy = ((ix + 0.5) * ty * d).repeat(k) / Y * 2 - 1
    degree = int(p["degree"])
    A = _monomials(cx, cy, degree)

    w = (Z < 1.0).float()  # more than double the field's level: an aggregate, not background
    for _ in range(int(p["robust_iters"])):
        coef = _solve_weighted(A, Z, w)
        resid = Z - coef @ A.T
        masked = torch.where(w > 0, resid, torch.nan)
        centre = masked.nanmedian(dim=-1, keepdim=True).values
        mad = (1.4826 * (masked - centre).abs().nanmedian(dim=-1, keepdim=True).values).clamp(min=1e-3)
        w = ((resid - centre).abs() < p["mad_k"] * mad).float()
        w = torch.where(w.sum(-1, keepdim=True) < A.shape[1] * 2, torch.ones_like(w), w)
    coef = _solve_weighted(A, Z, w)

    gx = (torch.arange(X, device=device, dtype=torch.float32) + 0.5) / X * 2 - 1
    gy = (torch.arange(Y, device=device, dtype=torch.float32) + 0.5) / Y * 2 - 1
    xx, yy = torch.meshgrid(gx, gy, indexing="ij")
    shape = torch.exp2((_monomials(xx, yy, degree) @ coef.T).permute(2, 0, 1))  # (C,X,Y)

    # Level per frame: pin to the frame's own low quantile of raw / shape.
    ratio = (small / F.avg_pool2d(shape, d, stride=d).unsqueeze(0)).view(T, C, -1)
    kth_global = max(1, int(round(p["rescale_quantile"] * ratio.shape[-1])))
    level = torch.kthvalue(ratio, kth_global, dim=-1).values.clamp(min=1e-6)
    diag = {
        "coef": coef,
        "level": level,
        "tiles_used_frac": w.mean(-1),
        "bg_ratio": shape.amax(dim=(-2, -1)) / shape.amin(dim=(-2, -1)).clamp(min=1e-6),
    }
    return shape, level, diag


def correct_background(raw: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict]:
    """Returns corrected (T,C,X,Y) float32, which may be negative: background noise is
    symmetric around zero and clamping would bias every dim measurement upward.

    Works frame by frame in one copy of the batch, so a 2048 px batch fits on a 24 GB card.
    """
    shape, level, diag = fit_background(raw, cfg)
    gain = shape.amax(dim=(-2, -1), keepdim=True) / shape.clamp(min=1e-6)  # (C,X,Y), since level cancels
    corrected = raw.float() if raw.dtype != torch.float32 else raw.clone()
    for t in range(corrected.shape[0]):
        corrected[t] -= shape * level[t].view(-1, 1, 1)
        corrected[t] *= gain
    return corrected, diag


def flatness(corrected: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Median nuclear intensity at the field edge over the centre, per channel. 1 = flat.

    The notebook's QC question, asked with the real mask instead of a bright-tail proxy.
    """
    T, C, X, Y = corrected.shape
    xx = torch.linspace(-1, 1, X, device=corrected.device)[:, None]
    yy = torch.linspace(-1, 1, Y, device=corrected.device)[None, :]
    r2 = (xx**2 + yy**2) / 2
    inner, outer = r2 < 0.15, r2 > 0.5
    frame = mask[:, 0].bool()
    out = []
    for c in range(C):
        a = corrected[:, c][frame & inner]
        b = corrected[:, c][frame & outer]
        out.append(b[:: max(1, b.numel() // 1000000)].median() / a[:: max(1, a.numel() // 1000000)].median().clamp(min=1e-6))
    return torch.stack(out)
