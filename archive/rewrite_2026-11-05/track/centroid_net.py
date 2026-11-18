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

from .centroids import find_h_maxima

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


def save_checkpoint(net: SeedNet, path: str) -> None:
    torch.save(net.state_dict(), path)


def load_checkpoint(path: str, device) -> SeedNet:
    key = (path, str(device))
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


def predict_peaks(corrected: torch.Tensor, mask: torch.Tensor, radius: float, cfg: dict) -> torch.Tensor:
    """Seeds from the model. Only seeds inside the mask are kept, since labels grow through it."""
    p = cfg["centroids"]
    if not p["cnn_checkpoint"]:
        raise ValueError('centroids.method is "cnn" but centroids.cnn_checkpoint is empty')
    net = load_checkpoint(p["cnn_checkpoint"], corrected.device)
    nuclei = corrected[:, cfg["nuclear_channel"]:cfg["nuclear_channel"] + 1]
    score = score_map(net, nuclei)
    peaks = find_h_maxima(score * mask, float(p["cnn_threshold"]) * 0.5, 40)
    keep = score[peaks[:, 0], 0, peaks[:, 1], peaks[:, 2]] >= p["cnn_threshold"]
    return peaks[keep]
