"""Optional learned seed finder: a small encoder-decoder network that draws a centre map.

Trained from point labels (a click on each nucleus centre, on a few dozen frames) to predict
a Gaussian centre map. Seeds are then the h-maxima of that map, and everything downstream is
unchanged. The input is the nuclear channel only, on a log scale in noise units, so the model
does not depend on a microscope's absolute intensity.

Select it with centroids.method = "cnn" and centroids.cnn_checkpoint = <path>.

Architectures (ARCHITECTURES, chosen by name with labelling.centres.architecture; checkpoints record theirs):

  resnet18-full   ResNet-18 layers 1-3 at full resolution + light decoder (the first version; over the speed budget)
  resnet18-half   the same with the stem at stride 2: about 4x faster, the map upsampled back to full size
  resnet18-half-hr  resnet18-half plus a two-layer full-resolution branch before the head, for sub-pixel precision
  resnet34-half-hr  deeper encoder (ResNet-34 layers 1-3), same layout
  unet16 / unet32 / unet16-5  plain U-Nets at full resolution, widths 16-128, 32-256, or 16-256 over 5 levels

Speed budget: learned tracking may cost at most 2x the classical pipeline (benchmarks/inference.py, tests/test_speed.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODELS: dict[tuple, nn.Module] = {}
STRIDE = 2                 # SeedNet() default: encoder at half resolution
CHUNK_PIXELS = 8 * 1024 * 1024 // 2   # frame pixels per forward pass at stride 1 (about 4 GB); twice that at stride 2


def _up(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)


def _conv(i: int, o: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SeedNet(nn.Module):
    """ResNet-18 layers 1-3 with a light decoder. The stem keeps full resolution (stride 1) or halves it (stride 2);
    either way the output is a full-size map. ResNet's own stem (stride 4) would leave a 15 px nucleus 4 px wide."""

    def __init__(self, stride: int = STRIDE, depth: int = 18, hires: int = 0):
        super().__init__()
        from torchvision.models import resnet18, resnet34
        base = (resnet18 if depth == 18 else resnet34)(weights=None)
        self.stride, self.depth, self.hires_width = int(stride), int(depth), int(hires)
        self.arch = {"name": "resnet", "stride": self.stride, "depth": self.depth, "hires": self.hires_width}
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 5, stride=self.stride, padding=2, bias=False), base.bn1, base.relu)
        self.enc1, self.enc2, self.enc3 = base.layer1, base.layer2, base.layer3
        self.up2 = nn.Sequential(nn.Conv2d(256 + 128, 64, 3, padding=1), nn.ReLU(inplace=True))
        if hires:   # a shallow full-resolution branch joins the upsampled decoder before the head
            self.up1 = nn.Sequential(nn.Conv2d(64 + 64, 32, 3, padding=1), nn.ReLU(inplace=True))
            self.hires = nn.Sequential(_conv(1, hires, 5), _conv(hires, hires))
            self.head = nn.Sequential(nn.Conv2d(32 + hires, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
        else:       # (layer names as in the first version, so its checkpoints still load)
            self.up1 = nn.Sequential(nn.Conv2d(64 + 64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.enc1(self.stem(x))
        m = self.enc2(s)
        d = self.enc3(m)
        u = self.up2(torch.cat([_up(d, m), m], 1))
        out = self.up1(torch.cat([_up(u, s), s], 1))
        if self.hires_width:
            return self.head(torch.cat([_up(out, x), self.hires(x)], 1))
        return _up(out, x) if out.shape[-2:] != x.shape[-2:] else out


class UNet(nn.Module):
    """Plain U-Net at full resolution: two conv-BN-ReLU per level, max-pool down, bilinear up, skip concatenation.
    Cheap at full resolution because the first level is narrow (16-32 channels, against ResNet's 64)."""

    def __init__(self, widths=(16, 32, 64, 128)):
        super().__init__()
        widths = [int(w) for w in widths]
        self.stride = 1
        self.arch = {"name": "unet", "widths": widths}
        self.down = nn.ModuleList()
        c = 1
        for w in widths:
            self.down.append(nn.Sequential(_conv(c, w), _conv(w, w)))
            c = w
        self.ups = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.ups.append(nn.Sequential(_conv(c + w, w), _conv(w, w)))
            c = w
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for block in self.ups:
            s = skips.pop()
            x = block(torch.cat([_up(x, s), s], 1))
        return self.head(x)


ARCHITECTURES = {
    "resnet18-full": {"name": "resnet", "stride": 1},
    "resnet18-half": {"name": "resnet", "stride": 2},
    "resnet18-half-hr": {"name": "resnet", "stride": 2, "hires": 24},
    "resnet34-half-hr": {"name": "resnet", "stride": 2, "depth": 34, "hires": 24},
    "unet16": {"name": "unet", "widths": [16, 32, 64, 128]},
    "unet32": {"name": "unet", "widths": [32, 64, 128, 256]},
    "unet16-5": {"name": "unet", "widths": [16, 32, 64, 128, 256]},
}
DEFAULT_ARCHITECTURE = "resnet18-half"


def build(arch: str | dict | None = None) -> nn.Module:
    """A network from a preset name (ARCHITECTURES) or a saved arch dict."""
    spec = dict(ARCHITECTURES[arch or DEFAULT_ARCHITECTURE] if not isinstance(arch, dict) else arch)
    name = spec.pop("name", "resnet")
    return UNet(**spec) if name == "unet" else SeedNet(**spec)


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
          batch: int = 12, lr: float = 2e-3, seed: int = 0, on_step=None, arch: str | dict | None = None) -> nn.Module:
    """frames (T,1,X,Y) normalized; points (P,3) [t,x,y]. Returns a trained model.

    Sparse fields are mostly empty, so most crops are centred on a labelled nucleus.
    """
    device = frames.device
    torch.manual_seed(seed)
    net = build(arch).to(device)
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
                  crop: int = 128, batch: int = 16, lr: float = 2e-3, seed: int = 0, on_step=None, arch: str | dict | None = None) -> nn.Module:
    """Train on labelled patches. images (N,1,P,P) normalized (see `normalize`, taken over the whole frame);
    inside (N,1,P,P) bool, the box in which every nucleus centre was clicked; points (M,3) [patch, x, y].

    Outside its box a patch is context only: it carries no loss, because nuclei there were not labelled.
    """
    device = images.device
    torch.manual_seed(seed)
    net = build(arch).to(device)
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


def save_checkpoint(net: nn.Module, path: str) -> None:
    torch.save({"arch": net.arch, "state": net.state_dict()}, path)


def load_checkpoint(path: str, device) -> nn.Module:
    import os
    key = (path, str(device), os.path.getmtime(path))  # a retrained file is read again
    if key not in _MODELS:
        saved = torch.load(path, map_location=device, weights_only=True)
        if "state" not in saved:                 # first-version files: a bare state dict, full-resolution encoder
            saved = {"arch": {"stride": 1}, "state": saved}
        net = build({"name": "resnet", **saved["arch"]})
        net.load_state_dict(saved["state"])
        _MODELS[key] = net.to(device).eval().to(memory_format=torch.channels_last)
    return _MODELS[key]


def score_map(net: nn.Module, nuclei: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
    """(T,1,X,Y) corrected nuclear channel -> (T,1,X,Y) centre score 0..1. Frames go through in chunks sized to
    keep memory bounded at any frame size; channels_last is about 20% faster on tensor cores."""
    x = normalize(nuclei)
    if chunk is None:
        chunk = max(1, CHUNK_PIXELS * getattr(net, "stride", 1) ** 2 // (x.shape[-2] * x.shape[-1]))
    parts = []
    with torch.inference_mode(), torch.autocast(x.device.type, enabled=x.device.type == "cuda"):
        for start in range(0, x.shape[0], chunk):
            part = x[start:start + chunk].contiguous(memory_format=torch.channels_last)
            parts.append(torch.sigmoid(net(part)).float().contiguous())
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
