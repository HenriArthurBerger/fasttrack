"""Cytoplasm ring: gap and width, the background-noise estimate, and the significance mask."""
import numpy as np
import torch

from fasttrack.config import make_config
from fasttrack.core.filters import max_square
from fasttrack.synthetic.generate import generate
from fasttrack.track import background, labels
from fasttrack.track.batch import mask_channels, process_batch
from fasttrack.track.measure import K, S, measure_batch


def _notebook_border_map(labels_, mask):
    """The ring before gap and width were settings, kept as the reference."""
    binary = mask.float()
    erode = 1 - binary
    border = max_square(binary) - binary
    grown = labels_
    for _ in range(3):
        grown = grown + max_square(grown) * erode
        erode = -max_square(-erode)
    border_labels = border * grown
    grown = grown * (1 - border)
    return grown * (binary + (binary + erode - 1)), border_labels


def _nuclei(device):
    nuclear = torch.zeros(2, 1, 120, 120, device=device)
    yy, xx = torch.meshgrid(torch.arange(120.0, device=device), torch.arange(120.0, device=device), indexing="ij")
    for t in range(2):
        for i, (cx, cy) in enumerate(((30, 30), (30, 52), (90, 70))):
            nuclear[t, 0][torch.hypot(yy - cx - t, xx - cy).lt(9)] = i + 1
    return labels.encode_local_labels(nuclear, 7)[0], nuclear > 0


def test_default_ring_is_the_notebook_ring(device):
    encoded, mask = _nuclei(device)
    new, new_border = labels.get_border_map(encoded, mask)
    old, old_border = _notebook_border_map(encoded, mask)
    assert torch.equal(new, old) and torch.equal(new_border, old_border)


def test_gap_and_width_set_which_layers_are_ring(device):
    nuclear = torch.zeros(1, 1, 40, 40, device=device)
    nuclear[0, 0, 15:25, 15:25] = 1
    encoded, _ = labels.encode_local_labels(nuclear, 7)
    # chessboard distance from the square: the max-pool grows one square layer per step
    ii, jj = np.mgrid[:40, :40]
    dist = np.maximum(np.maximum(15 - ii, ii - 24), np.maximum(15 - jj, jj - 24))
    for gap, width in ((1, 2), (2, 2), (0, 3), (2, 1)):
        lm, _ = labels.get_border_map(encoded, nuclear > 0, gap, width)
        ring = (lm[0, 0] < 0).cpu().numpy()
        assert np.array_equal(ring, (dist > gap) & (dist <= gap + width)), (gap, width)
        assert np.array_equal((lm[0, 0] > 0).cpu().numpy(), dist <= 0)


def _field(device, cover, period=40, X=1152, T=2):
    """Flat 500 background, pixel noise sigma 24, and cells with smooth texture covering `cover` of the field."""
    g = torch.Generator(device=device).manual_seed(0)
    yy, xx = torch.meshgrid(torch.arange(X, device=device), torch.arange(X, device=device), indexing="ij")
    side = int(round(period * cover ** 0.5))
    cells = ((yy % period) < side) & ((xx % period) < side)
    raw = 500 + 24 * torch.randn(T, 1, X, X, device=device, generator=g)
    from fasttrack.core.filters import gaussian_blur
    texture = gaussian_blur(torch.randn(T, 1, X, X, device=device, generator=g), 3)   # smooth, as real cells
    raw = raw + cells * 300 * (1 + 0.3 * texture / texture.std())
    return raw, cells.expand(T, -1, -1)


def _mask_cfg(**measure):
    return make_config({"measure": {"cyto_mask": True, **measure}})


def test_noise_is_pixel_noise_and_the_mask_follows_cells(device):
    """The blurred noise matches theory for white noise, and the 2-sigma mask keeps cells and drops nearly
    all background, from sparse to near-confluent fields."""
    from fasttrack.core.filters import gaussian_kernel_1d
    g = gaussian_kernel_1d(1.5, device)
    expected = 24 * float((g ** 2).sum())            # noise of a separable blur of white noise, sigma 24
    for cover in (0.0, 0.4, 0.85, 0.95):
        raw, cells = _field(device, cover)
        _, diag = background.correct_background(raw, _mask_cfg())
        assert abs(diag["mask_sigma"].median().item() - expected) < 0.15 * expected, cover
        passed = diag["significant"][:, 0]
        # the blur spreads cells by about a pixel, so judge background more than 3 px from any cell
        near = torch.nn.functional.max_pool2d(cells[:, None].float(), 7, stride=1, padding=3)[:, 0] > 0
        if (~near).any():
            assert passed[~near].float().mean() < 0.05, cover      # about 2.3% for 2 sigma
        if cells.any():
            assert passed[cells].float().mean() > 0.98, cover


def test_noise_estimate_holds_down_to_a_few_percent_background(device):
    """FastTrack requires 2% background in every tile; at 5% (1 px gaps between cells) every tile's own
    estimate is still right."""
    raw, cells = _field(device, 0.95)
    level, sigma = background.background_noise(raw - 500, 21)
    assert ((sigma - 24).abs() < 0.2 * 24).float().mean() > 0.95      # median-based: a little noisier per tile
    assert ((level - 0).abs() < 0.5 * 24).float().mean() > 0.95


def test_masked_ring_drops_dim_pixels_and_counts_the_kept_fraction(cfg, device):
    encoded, mask = _nuclei(device)
    label_map, border = labels.get_border_map(encoded, mask)
    data = torch.full((2, 2, 120, 120), 100.0, device=device)
    data[:, :, :, 60:] = 0                       # right half dark: ring pixels there fail
    significant = data[:, 1:2] > 50
    masked, ring_labels = labels.mask_ring(label_map, significant)
    sc, ch = measure_batch(data, masked, border, int(encoded.abs().max() % 1000), 9, cfg, ring_labels)
    lm, ms = label_map.cpu().numpy(), masked.cpu().numpy()
    for row in range(sc.shape[0]):
        t, lab = int(sc[row, S["t"]]), int(sc[row, S["label"]])
        total, kept = (lm[t, 0] == -lab).sum(), (ms[t, 0] == -lab).sum()
        assert sc[row, S["cyto_area"]] == kept
        assert np.isclose(sc[row, S["cyto_kept"]].item(), kept / total)
        if kept >= 5 and kept / total >= 0.5:
            assert ch[row, 1, K["cyto_mean"]] == 100     # only bright pixels were measured
        else:
            assert torch.isnan(ch[row, 1, K["cyto_mean"]])
    # the cell centred at x=70 straddles the dark half, so it keeps only part of its ring
    assert ((sc[:, S["cyto_kept"]] > 0) & (sc[:, S["cyto_kept"]] < 1)).any()


def test_cells_with_too_few_ring_pixels_get_nan_cytoplasm(device):
    cfg = make_config({"device": device, "measure": {"cyto_min_pixels": 5}})
    encoded, mask = _nuclei(device)
    label_map, border = labels.get_border_map(encoded, mask)
    data = torch.full((2, 1, 120, 120), 100.0, device=device)
    masked, ring_labels = labels.mask_ring(label_map, torch.zeros_like(mask))   # nothing significant
    sc, ch = measure_batch(data, masked, border, int(encoded.abs().max() % 1000), 9, cfg, ring_labels)
    assert (sc[:, S["cyto_area"]] == 0).all() and (sc[:, S["cyto_kept"]] == 0).all()
    for name in ("cyto_mean", "cyto_median", "nc_ratio", "nuc_sum_frac"):
        assert torch.isnan(ch[:, :, K[name]]).all()
    assert not torch.isnan(ch[:, :, K["nuc_mean"]]).any()
    # without a mask every ring is whole
    sc, _ = measure_batch(data, label_map, border, int(encoded.abs().max() % 1000), 9, cfg)
    assert (sc[:, S["cyto_kept"]] == 1).all()


def test_pipeline_with_the_mask(device):
    frames, _, _ = generate({"seed": 5, "frames": 6, "size": 256, "cells": 10}, device=device)
    off = process_batch(frames, make_config({"device": device}))
    on = process_batch(frames, make_config({"device": device, "measure": {"cyto_mask": True}}))
    assert "significant" not in on["diag"]          # used and dropped, not kept for the whole batch
    assert torch.equal(off["scalars"][:, S["nuc_area"]], on["scalars"][:, S["nuc_area"]])
    kept = on["scalars"][:, S["cyto_kept"]]
    assert ((kept >= 0) & (kept <= 1)).all()
    assert (on["scalars"][:, S["cyto_area"]] <= off["scalars"][:, S["cyto_area"]]).all()


def test_mask_channels_default_to_every_channel_but_the_nuclear_one():
    from fasttrack.experiment import new_experiment, pipeline_config
    exp = new_experiment()
    exp["channels"] = [{"file_channel": 1, "name": "h2b", "role": "nuclear", "reporter": "H2B"},
                       {"file_channel": 2, "name": "pparg", "role": "measure", "reporter": "PPARG"},
                       {"file_channel": 3, "name": "erk", "role": "measure", "reporter": "ERK-KTR"}]
    cfg = pipeline_config(exp)
    assert mask_channels(cfg) == [1, 2]
    cfg["measure"]["cyto_mask_channel"] = "erk, pparg"
    assert mask_channels(cfg) == [2, 1]


def test_mask_is_the_or_of_each_channels_own_test(device):
    """Each channel is judged against its own background: a cell visible in only one channel is kept, and a
    channel with a much higher background does not raise the bar for the others."""
    raw0, cells0 = _field(device, 0.3, T=1)
    raw1, cells1 = _field(device, 0.3, period=56, T=1)
    raw = torch.cat([raw0, 4000 + 3 * (raw1 - 500)], dim=1)       # channel 1: other cells, 8x background, 3x noise
    names = {"data": {"channel_names": ["a", "b"]}, "measure": {"cyto_mask": True, "cyto_mask_channel": "a,b"}}
    both = background.correct_background(raw, make_config(names))[1]["significant"][0, 0]
    only_a = background.correct_background(raw, make_config({**names, "measure": {"cyto_mask": True, "cyto_mask_channel": "a"}}))[1]["significant"][0, 0]
    for cells in (cells0[0], cells1[0]):
        assert both[cells].float().mean() > 0.98
    assert only_a[cells0[0]].float().mean() > 0.98
    grow = lambda m: torch.nn.functional.max_pool2d(m[None].float(), 7, stride=1, padding=3)[0, 0] > 0   # 3 px of blur
    assert only_a[cells1[0] & ~grow(cells0)].float().mean() < 0.05
    near = grow(cells0 | cells1)
    assert both[~near].float().mean() < 0.08           # two independent 2-sigma tests: about 4.5%


def test_cells_keeping_little_of_their_ring_get_nan_cytoplasm(device):
    cfg = make_config({"device": device, "measure": {"cyto_min_pixels": 1, "cyto_min_kept": 0.5}})
    encoded, mask = _nuclei(device)
    label_map, border = labels.get_border_map(encoded, mask)
    data = torch.full((2, 1, 120, 120), 100.0, device=device)
    significant = torch.zeros_like(mask)
    significant[..., :, :52] = True                    # left part of the field only
    masked, ring_labels = labels.mask_ring(label_map, significant)
    sc, ch = measure_batch(data, masked, border, int(encoded.abs().max() % 1000), 9, cfg, ring_labels)
    kept = sc[:, S["cyto_kept"]]
    low = kept < 0.5
    assert low.any() and (~low).any()
    assert torch.isnan(ch[low][:, :, K["cyto_mean"]]).all()
    assert not torch.isnan(ch[~low][:, :, K["cyto_mean"]]).any()
