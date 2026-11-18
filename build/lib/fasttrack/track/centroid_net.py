"""Optional learned seed finder: a ResNet-18 encoder with a light decoder.

Trained from point labels (a click on each nucleus centre, on a few dozen frames) to predict
a Gaussian centre map. Seeds are then the h-maxima of that map, and everything downstream is
unchanged. The input is the nuclear channel only, on a log scale in noise units, so the model
does not depend on a microscope's absolute intensity.

Select it with centroids.method = "cnn" and centroids.cnn_checkpoint = <path>.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODELS: dict[tuple, nn.Module] = {}


class SeedNet(nn.Module):
    """Encoder stride reduced at the stem, because nuclei here are only about 15 px across."""

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        base = resnet18(weights=None)
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 5, padding=2, bias=False), base.bn1, base.relu)
        self.enc1, self.enc2, self.enc3 = base.layer1, base.layer2, base.layer3
        self.up2 = nn.Sequential(nn.Conv2d(256 + 128, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.Conv2d(64 + 64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.enc1(self.stem(x))
        m = self.enc2(s)
        d = self.enc3(m)
        d = F.interpolate(d, size=m.shape[-2:], mode="bilinear", align_corners=False)
        u = self.up2(torch.cat([d, m], 1))
        u = F.interpolate(u, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return self.up1(torch.cat([u, s], 1))


def normalize(nuclei: torch.Tensor) -> torch.Tensor:
    """(T,1,X,Y) corrected -> log signal-to-noise, about 0..1. Independent of absolute scale."""
    sample = nuclei[:, 0, ::4, ::4].flatten(1)
    median = sample.median(dim=1).values
    sd = (1.4826 * (sample - median[:, None]).abs().median(dim=1).values).clamp(min=1.0)
    snr = (nuclei - median.view(-1, 1, 1, 1)) / sd.view(-1, 1, 1, 1)
    return torch.log1p(snr.clamp(min=0)) / 6.0


def centre_map(points: torch.Tensor, shape: tuple, sigma: float) -> torch.Tensor:
    """Target for points (P,3) [t,x,y]: a Gaussian bump on each. Returns (T,1,X,Y)."""
    T, _, X, Y = shape
    device = points.device
    target = torch.zeros(T, 1, X, Y, device=device)
    reach = int(3 * sigma)
    grid = torch.arange(-reach, reach + 1, device=device)
    du, dv = torch.meshgrid(grid, grid, indexing="ij")
    bump = torch.exp(-(du**2 + dv**2).float() / (2 * sigma**2))
    px = points[:, 1].round().long()[:, None, None] + du
    py = points[:, 2].round().long()[:, None, None] + dv
    tt = points[:, 0].long()[:, None, None].expand_as(px)
    ok = (px >= 0) & (px < X) & (py >= 0) & (py < Y)
    flat = (tt[ok] * X * Y + px[ok] * Y + py[ok])
    out = target.view(-1)
    out.scatter_reduce_(0, flat, bump.expand_as(px)[ok], "amax")
    return out.view(T, 1, X, Y)


def train(frames: torch.Tensor, points: torch.Tensor, sigma: float = 3.0, steps: int = 800, crop: int = 256,
          batch: int = 12, lr: float = 2e-3, seed: int = 0, on_step=None) -> SeedNet:
    """frames (T,1,X,Y) normalized; points (P,3) [t,x,y]. Returns a trained model.

    Sparse fields are mostly empty, so most crops are centred on a labelled nucleus.
    """
    device = frames.device
    torch.manual_seed(seed)
    net = SeedNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=steps)
    target = centre_map(points, frames.shape, sigma)
    T, _, X, Y = frames.shape
    half = crop // 2
    weights = 1 + 40 * target
    net.train()
    for step in range(steps):
        picks = points[torch.randint(points.shape[0], (batch,), device=device)]
        jitter = torch.randint(-half + 8, half - 8, (batch, 2), device=device)
        cx = (picks[:, 1] + jitter[:, 0]).clamp(half, X - half).long() - half
        cy = (picks[:, 2] + jitter[:, 1]).clamp(half, Y - half).long() - half
        tt = picks[:, 0].long()
        x = torch.stack([frames[t, :, a:a + crop, b:b + crop] for t, a, b in zip(tt, cx, cy)])
        y = torch.stack([target[t, :, a:a + crop, b:b + crop] for t, a, b in zip(tt, cx, cy)])
        w = torch.stack([weights[t, :, a:a + crop, b:b + crop] for t, a, b in zip(tt, cx, cy)])
        if torch.rand(1) < 0.5:
            x, y, w = x.flip(-1), y.flip(-1), w.flip(-1)
        if torch.rand(1) < 0.5:
            x, y, w = x.flip(-2), y.flip(-2), w.flip(-2)
        k = int(torch.randint(4, (1,)))
        x, y, w = x.rot90(k, (-2, -1)), y.rot90(k, (-2, -1)), w.rot90(k, (-2, -1))
        loss = (w * F.binary_cross_entropy_with_logits(net(x), y, reduction="none")).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if on_step is not None:
            on_step(step, float(loss))
    return net.eval()


def train_patches(images: torch.Tensor, inside: torch.Tensor, points: torch.Tensor, sigma: float = 3.0, steps: int = 1500,
                  crop: int = 128, batch: int = 16, lr: float = 2e-3, seed: int = 0, on_step=None) -> SeedNet:
    """Train on labelled patches. images (N,1,P,P) normalized (see `normalize`, taken over the whole frame);
    inside (N,1,P,P) bool, the box in which every nucleus centre was clicked; points (M,3) [patch, x, y].

    Outside its box a patch is context only: it carries no loss, because nuclei there were not labelled.
    """
    device = images.device
    torch.manual_seed(seed)
    net = SeedNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=steps)
    target = centre_map(points, images.shape, sigma) if points.shape[0] else torch.zeros_like(images)
    weights = (1 + 20 * target) * inside.float()
    N, _, P, _ = images.shape
    crop = min(crop, P)
    net.train()
    for step in range(steps):
        idx = torch.randint(N, (batch,), device=device)
        off = torch.randint(0, P - crop + 1, (batch, 2), device=device)
        x = torch.stack([images[i, :, a:a + crop, b:b + crop] for i, (a, b) in zip(idx.tolist(), off.tolist())])
        y = torch.stack([target[i, :, a:a + crop, b:b + crop] for i, (a, b) in zip(idx.tolist(), off.tolist())])
        w = torch.stack([weights[i, :, a:a + crop, b:b + crop] for i, (a, b) in zip(idx.tolist(), off.tolist())])
        if torch.rand(1) < 0.5:
            x, y, w = x.flip(-1), y.flip(-1), w.flip(-1)
        if torch.rand(1) < 0.5:
            x, y, w = x.flip(-2), y.flip(-2), w.flip(-2)
        k = int(torch.randint(4, (1,)))
        x, y, w = x.rot90(k, (-2, -1)), y.rot90(k, (-2, -1)), w.rot90(k, (-2, -1))
        x = x * (1 + 0.1 * torch.randn(batch, 1, 1, 1, device=device))  # brightness differs between wells
        loss = (w * F.binary_cross_entropy_with_logits(net(x), y, reduction="none")).sum() / w.sum().clamp(min=1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if on_step is not None:
            on_step(step, loss.item())
    return net.eval()


def save_checkpoint(net: SeedNet, path: str) -> None:
    torch.save(net.state_dict(), path)


def load_checkpoint(path: str, device) -> SeedNet:
    import os
    key = (path, str(device), os.path.getmtime(path))  # a retrained file is read again
    if key not in _MODELS:
        net = SeedNet()
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        _MODELS[key] = net.to(device).eval()
    return _MODELS[key]


def score_map(net: SeedNet, nuclei: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    x = normalize(nuclei)
    parts = []
    with torch.inference_mode(), torch.autocast(x.device.type, enabled=x.device.type == "cuda"):
        for start in range(0, x.shape[0], chunk):
            parts.append(torch.sigmoid(net(x[start:start + chunk])).float())
    return torch.cat(parts)


def find_peaks(score: torch.Tensor, threshold: float, window: int = 9) -> torch.Tensor:
    """(T,1,X,Y) score -> (P,3) [t,x,y] long: local maxima within `window` px at or above threshold.
    A plateau (equal neighbours) keeps one pixel: the first in raster order."""
    pooled = F.max_pool2d(score, window, stride=1, padding=window // 2)
    peak = (score == pooled) & (score >= threshold)
    peaks = peak.nonzero(as_tuple=False)[:, [0, 2, 3]]
    if peaks.shape[0] < 2:
        return peaks
    # drop the later of two equal maxima closer than the window (flat tops)
    keep = torch.ones(peaks.shape[0], dtype=torch.bool, device=peaks.device)
    for t in peaks[:, 0].unique().tolist():
        rows = (peaks[:, 0] == t).nonzero(as_tuple=True)[0]
        if rows.numel() < 2:
            continue
        d = torch.cdist(peaks[rows, 1:].float(), peaks[rows, 1:].float())
        close = torch.triu(d < window / 2, diagonal=1)
        keep[rows[close.any(0)]] = False
    return peaks[keep]


def predict_peaks(corrected: torch.Tensor, mask: torch.Tensor, radius: int, cfg: dict) -> torch.Tensor:
    """Seeds from the model. Only seeds inside the mask are kept, since labels grow through it."""
    p = cfg["centroids"]
    if not p["cnn_checkpoint"]:
        raise ValueError('centroids.method is "cnn" but centroids.cnn_checkpoint is empty')
    net = load_checkpoint(p["cnn_checkpoint"], corrected.device)
    nuclei = corrected[:, cfg["nuclear_channel"]:cfg["nuclear_channel"] + 1]
    score = score_map(net, nuclei)
    return find_peaks(score * mask, float(p["cnn_threshold"]))
