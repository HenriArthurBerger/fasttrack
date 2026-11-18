"""Quadratic background and illumination correction, from the FastTrack notebook.

Assumes the images are raw and that at least 2% of the pixels in each image are background.
Each frame's 1st-percentile tile values are fitted with a 2D quadratic in log2 space, scaled
to the frame's 2nd percentile, subtracted, and then corrected for excitation bias:

    relative protein = measured fluorescence * max illumination / measured illumination
"""
from __future__ import annotations

import torch


def _tiles(images: torch.Tensor, k: int) -> torch.Tensor:
    """(T, C, k*k, pixels per tile): the image cut into a k*k grid, row-major."""
    T, C, n, m = images.shape
    tile_n, tile_m = n // k, m // k
    tiles = images[:, :, : k * tile_n, : k * tile_m].reshape(T, C, k, tile_n, k, tile_m)
    return tiles.permute(0, 1, 2, 4, 3, 5).reshape(T, C, k * k, tile_n * tile_m)


def quantile_tile_values_with_positions(images: torch.Tensor, k: int, quantile: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantile of each tile on a k*k grid, and the position of the pixel nearest that value."""
    T, C, n, m = images.shape
    device = images.device
    tile_n, tile_m = n // k, m // k
    tiles = _tiles(images, k)
    values = torch.quantile(tiles, quantile, dim=-1)
    indices = torch.argmin(torch.abs(tiles - values.unsqueeze(-1)), dim=-1)
    rows, cols = indices // tile_m, indices % tile_m
    positions = torch.stack([
        rows + (torch.arange(k, device=device).repeat_interleave(k) * tile_n).view(1, 1, k * k),
        cols + (torch.arange(k, device=device).repeat(k) * tile_m).view(1, 1, k * k),
    ], dim=-1)
    return values, positions


def median_tile_values_with_positions(images: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    return quantile_tile_values_with_positions(images, k, 0.5)


def fit_2d_quadratic(positions: torch.Tensor, values: torch.Tensor, xsize: int, ysize: int) -> torch.Tensor:
    """Least-squares quadratic per (T, C). A NaN value drops that tile from its frame's fit."""
    x = (positions[..., 0] - xsize // 2).float()
    y = (positions[..., 1] - ysize // 2).float()
    A = torch.stack([x**2, y**2, x * y, x, y, torch.ones_like(x)], dim=-1).double()
    keep = ~torch.isnan(values)
    Z = torch.nan_to_num(values, nan=0.0).unsqueeze(-1).double()
    # Filtered tiles contribute nothing. Passing their NaN to the solver would blank the frame.
    A = A * keep.unsqueeze(-1)
    Z = Z * keep.unsqueeze(-1)
    At = A.transpose(-2, -1)
    AtA = At @ A
    # A blank or unreadable frame leaves fewer tiles than parameters: regularise only those frames so the
    # solve cannot fail. Frames with enough tiles are solved exactly as before.
    few = (keep.sum(-1) < 6).unsqueeze(-1).unsqueeze(-1)
    AtA = AtA + few * torch.eye(6, dtype=AtA.dtype, device=AtA.device)
    return torch.linalg.solve(AtA, At @ Z).squeeze(-1).float()


def get_quadratic_background(shape: tuple, parameters: torch.Tensor) -> torch.Tensor:
    _, _, xsize, ysize = shape
    device = parameters.device
    x, y = torch.meshgrid(torch.arange(xsize, device=device), torch.arange(ysize, device=device), indexing="ij")
    x, y = (x - xsize // 2).float(), (y - ysize // 2).float()
    p = parameters[:, :, :, None, None]
    return p[:, :, 0] * x**2 + p[:, :, 1] * y**2 + p[:, :, 2] * x * y + p[:, :, 3] * x + p[:, :, 4] * y + p[:, :, 5]


# the dim-tail percentiles the noise is read from, and their z-scores for a normal distribution
# (inside the 2% of every tile that FastTrack requires to be background)
_P_LO, _P_HI = 0.002, 0.015
_Z_LO, _Z_HI = -2.8782, -2.1701


def _median3(grid: torch.Tensor) -> torch.Tensor:
    """3x3 median over a (T, C, k, k) tile grid, edges replicated: removes single-tile outliers."""
    import torch.nn.functional as F
    T, C, k, _ = grid.shape
    padded = F.pad(grid.reshape(T * C, 1, k, k), (1, 1, 1, 1), mode="replicate")
    patches = F.unfold(padded, 3)                       # (T*C, 9, k*k)
    return patches.nanmedian(dim=1).values.view(T, C, k, k)


def background_noise(resid: torch.Tensor, k: int, with_fraction: bool = False):
    """Background level and noise per tile of a residual image (raw - fitted background): two
    (T, C, k, k) tensors in the residual's units.

    FastTrack requires at least 2% of every tile to be background (the same rule the background fit's
    tile minima rely on), so each tile is estimated from its own background pixels, the same way on every
    frame:
    1. seed from the dim tail: sigma = (q1.5 - q0.2) / (z1.5 - z0.2) and level = q0.2 - z0.2 * sigma
       (percentiles q, normal z-scores), with the z-scores corrected for the share of the tile that cells
       cover (pixels below the level are nearly all background, so that share is about twice theirs);
    2. refine with many pixels: level = median of the pixels below level + 2 sigma, sigma = 1.4826 x the
       median distance of the pixels below the level (half the background and nearly no cell pixels);
    Medians throughout, never means, so bright cell clusters and debris cannot pull the estimate.
    3. a 3x3 median over tiles removes single-tile outliers.
    with_fraction=True also returns each tile's estimated background fraction.
    This is pixel-to-pixel noise, not the spread of the tile minima across the field (that is the
    background's shape).
    """
    T, C = resid.shape[:2]
    tiles = _tiles(resid, k)
    q = torch.quantile(tiles, torch.tensor([_P_LO, _P_HI], device=resid.device), dim=-1)
    z1, z2 = torch.full_like(q[0], _Z_LO), torch.full_like(q[0], _Z_HI)
    for _ in range(4):
        # Where cells cover part of a tile, its percentiles are deeper background percentiles (p / f for a
        # background fraction f). Pixels below the background level are nearly all background, so f is
        # about twice their share; refine z1, z2 with it.
        sigma = ((q[1] - q[0]) / (z2 - z1)).clamp(min=0)
        level = q[0] - z1 * sigma
        f = (2 * (tiles < level.unsqueeze(-1)).float().mean(dim=-1)).clamp(0.02, 1.0)
        z1, z2 = torch.special.ndtri(_P_LO / f), torch.special.ndtri(_P_HI / f)
    sigma = ((q[1] - q[0]) / (z2 - z1)).clamp(min=0)
    level = q[0] - z1 * sigma
    for _ in range(2):
        # Refine with many pixels instead of two percentiles: the level is the median of the pixels below
        # level + 2 sigma (nearly all of the background; +0.0286 sigma undoes the cut), and sigma from the
        # median distance of the pixels below the level (half the background, and nearly no cell pixels).
        near = torch.where(tiles < (level + 2 * sigma).unsqueeze(-1), tiles, torch.nan)
        level = torch.nanmedian(near, dim=-1).values + 0.0286 * sigma
        # median, never mean: bright cells and debris must not pull the estimate (1.4826 * median distance
        # below the level is sigma for normal noise)
        below = torch.where(tiles < level.unsqueeze(-1), level.unsqueeze(-1) - tiles, torch.nan)
        sigma = 1.4826 * torch.nanmedian(below, dim=-1).values
    level, sigma = _median3(level.view(T, C, k, k)).flatten(2), _median3(sigma.view(T, C, k, k)).flatten(2)
    out = torch.nan_to_num(level, nan=0.0).view(T, C, k, k), torch.nan_to_num(sigma, nan=0.0).view(T, C, k, k)
    return (*out, f.view(T, C, k, k)) if with_fraction else out


def _map(grid: torch.Tensor, shape) -> torch.Tensor:
    """A (T, C, k, k) tile grid interpolated to (T, C, *shape) between tile centres."""
    import torch.nn.functional as F
    return F.interpolate(grid, size=tuple(shape), mode="bilinear", align_corners=False)


def blur_noise_factor(resid: torch.Tensor, k: int, smooth: float) -> float:
    """How much a Gaussian blur of `smooth` px shrinks the background noise: one number for a batch.

    Real background noise is not white (on the KTR plate a 1.5 px blur leaves 0.32-0.42 of it, where
    white noise would leave 0.19), so it is measured: the median over tiles of blurred / unblurred noise,
    from tiles that are at least half background (there the blurred estimate is reliable; thin gaps between
    cells blur away). Pooled over every frame of the batch, so all its frames get the same factor. Without
    any such tile it falls back to the white-noise factor.
    """
    from ..core.filters import gaussian_blur, gaussian_kernel_1d
    _, raw_sigma, f = background_noise(resid, k, with_fraction=True)
    _, blur_sigma = background_noise(gaussian_blur(resid, smooth), k)
    open_ = (f >= 0.5) & (raw_sigma > 0)
    if not open_.any():
        g = gaussian_kernel_1d(smooth, resid.device)
        return float((g ** 2).sum())
    return float((blur_sigma[open_] / raw_sigma[open_]).median())


def significant_pixels(image: torch.Tensor, background: torch.Tensor, k: int, k_sigma: float, smooth: float,
                       factor: float | None = None):
    """Pixels significantly above background: (significant (T,C,X,Y) bool, sigma (T,C), factor).

    The test runs after background subtraction, on raw - background without the clamp at 0, blurred by a
    Gaussian of `smooth` px: significant = blurred > level + k_sigma * factor * sigma. The background level
    and pixel noise are read per tile on the unblurred image by `background_noise` (robust down to the 2%
    background FastTrack requires), and `factor` is how much the blur shrinks that noise (`blur_noise_factor`,
    measured on these frames unless given). The flat-field step only multiplies by a smooth positive factor
    (max / background), so the same pixels pass on the corrected image. Real images have pixel noise about
    as large as dim cytoplasm: a per-pixel test would keep the pixels whose noise happened to be high and
    bias their mean upwards, while the blur makes it a test of where the cell is. The same rule runs on
    every frame; sigma is the frame's median noise after the blur, in raw counts.
    """
    from ..core.filters import gaussian_blur
    resid = image - background
    level, sigma = background_noise(resid, k)
    if smooth > 0:
        if factor is None:
            factor = blur_noise_factor(resid, k, smooth)
        sigma = sigma * factor
        resid = gaussian_blur(resid, smooth)
    else:
        factor = 1.0
    significant = resid > _map(level + k_sigma * sigma, resid.shape[-2:])
    return significant, sigma.flatten(2).median(dim=-1).values, factor


def mask_channels(cfg: dict) -> list[int]:
    """Indices of the channels whose significance decides which cytoplasm-ring pixels are kept.

    measure.cyto_mask_channel: empty = every channel except the nuclear one; otherwise one name or a
    comma-separated list. Each channel is tested against its own background, and a pixel is kept if any of
    them passes (OR), so the mask is as lax as the most permissive channel: a cell counts if any reporter
    shows it. The nuclear channel is left out by default because its optical blur reaches past the nuclear
    mask: a few percent of a bright nucleus 1-3 px out is far above its background noise, so it would pass
    every ring pixel and switch the mask off (seen on the KTR plate)."""
    raw = cfg["measure"].get("cyto_mask_channel", "") or ""
    wanted = [str(n).strip() for n in (raw if isinstance(raw, (list, tuple)) else str(raw).split(",")) if str(n).strip()]
    names = [str(n) for n in cfg["data"].get("channel_names", [])]
    if not wanted:
        nuclear = int(cfg.get("nuclear_channel", 0))
        others = [i for i in range(len(names)) if i != nuclear]
        return others or [nuclear]
    missing = [n for n in wanted if n not in names]
    if missing:
        raise ValueError(f"cyto_mask_channel {missing} not loaded channels; channels are {names}")
    return [names.index(n) for n in wanted]


def _fit(image: torch.Tensor, p: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(background (T,C,X,Y) in raw counts, quadratic params) of each frame."""
    T, C = image.shape[:2]
    values, positions = quantile_tile_values_with_positions(image, int(p["tiles_per_edge"]), p["tile_quantile"])
    # Illumination should be no more than 2-3x brighter at the centre, so a tile brighter than
    # twice the median holds a bright artifact or aggregate, not background.
    filtered = torch.where(values < 2 * values.median(dim=-1).values.unsqueeze(-1), values, torch.nan)
    params = fit_2d_quadratic(positions, torch.log2(filtered), image.shape[-2], image.shape[-1])
    background = torch.exp2(get_quadratic_background(image.shape, params))
    scale_image = torch.quantile(image.view(T, C, -1), p["scale_quantile"], dim=-1)
    scale_quad = torch.quantile(background.view(T, C, -1), p["scale_quantile"], dim=-1)
    background = background * (scale_image / scale_quad)[:, :, None, None]
    return background, params


def _correct_chunk(image: torch.Tensor, p: dict, mask: dict | None = None) -> tuple:
    background, params = _fit(image, p)
    maxes = background.amax(dim=(-2, -1), keepdim=True)
    corrected = ((image - background) * maxes / background).clamp(min=0)
    # blank frames (an unreadable file left as zeros) and other degenerate fits give zeros, never NaN
    blank = (image.amax(dim=(-2, -1)) <= 0) | ~torch.isfinite(maxes[..., 0, 0]) | (background.amin(dim=(-2, -1)) <= 0)
    corrected = torch.where(blank[:, :, None, None], torch.zeros_like(corrected), torch.nan_to_num(corrected, nan=0.0, posinf=0.0))
    extra = {}
    if mask is not None:
        # each channel against its own background and noise; a pixel is kept if any channel passes
        anyp, sigmas = None, []
        for c, factor in zip(mask["channels"], mask["factors"]):
            sig, sigma, _ = significant_pixels(image[:, c:c + 1], background[:, c:c + 1], int(p["tiles_per_edge"]),
                                               mask["k_sigma"], mask["smooth"], factor)
            sig = sig & ~blank[:, c, None, None, None]
            anyp = sig if anyp is None else anyp | sig
            sigmas.append(sigma[:, 0])
        extra = {"significant": anyp, "mask_sigma": torch.stack(sigmas, dim=1)}
    return corrected, params, maxes[..., 0, 0] / background.amin(dim=(-2, -1)), extra


def correct_background(raw: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict]:
    """raw (T,C,X,Y) -> (corrected float32, diagnostics). Negative values are set to 0.

    Diagnostics: params (quadratic fits) and bg_ratio. With measure.cyto_mask on, also significant
    (T,1,X,Y): pixels significantly above background in any of the mask channels (`mask_channels`), each
    tested against its own background (see `significant_pixels`); mask_sigma (T, channels): each one's
    noise after the blur, in raw counts; mask_factor: per channel, how much the blur shrank its background
    noise (one number per batch, see `blur_noise_factor`); mask_channels: the channel indices.

    Each frame is fitted on its own, so chunking frames only bounds the memory of the
    quantile sorts. The result is the same.
    """
    p = cfg["background"]
    m = cfg.get("measure", {})
    mask = None
    if m.get("cyto_mask", False):
        channels = [c for c in mask_channels(cfg) if c < raw.shape[1]]
        mask = {"channels": channels, "k_sigma": float(m.get("cyto_mask_sigma", 2.0)),
                "smooth": float(m.get("cyto_mask_smooth", 1.5)), "factors": [1.0] * len(channels)}
        if mask["smooth"] > 0:
            # one blur-noise factor per channel for the whole batch, from up to 20 frames spread across it
            pick = torch.linspace(0, raw.shape[0] - 1, min(20, raw.shape[0]), device=raw.device).round().long().unique()
            for j, c in enumerate(channels):
                sample = torch.stack([raw[int(i), c:c + 1].float() for i in pick])   # uint16 cannot be fancy-indexed
                mask["factors"][j] = blur_noise_factor(sample - _fit(sample, p)[0], int(p["tiles_per_edge"]), mask["smooth"])
    chunk = int(p["chunk_frames"])
    parts = [_correct_chunk(raw[s:s + chunk].float(), p, mask) for s in range(0, raw.shape[0], chunk)]
    diag = {"params": torch.cat([x[1] for x in parts]), "bg_ratio": torch.cat([x[2] for x in parts])}
    for key in parts[0][3]:
        diag[key] = torch.cat([x[3][key] for x in parts])
    if mask is not None:
        diag["mask_factor"] = mask["factors"]
        diag["mask_channels"] = mask["channels"]
    return torch.cat([x[0] for x in parts]), diag
