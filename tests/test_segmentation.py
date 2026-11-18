"""Background, mask, seeds and labels, as ported from the notebook."""
import numpy as np
import torch

from fasttrack.core import filters
from fasttrack.track import background, centroids, labels, masking


def _ellipse(X, Y, cx, cy, a, b, theta=0.0, device="cpu"):
    xx, yy = torch.meshgrid(torch.arange(X, device=device).float(), torch.arange(Y, device=device).float(), indexing="ij")
    u = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    v = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
    return (u / a) ** 2 + (v / b) ** 2 <= 1


def test_gaussian_blur_is_deterministic(device):
    # torchvision's GaussianBlur defaults to a random sigma on every call.
    x = torch.zeros(1, 1, 21, 21, device=device)
    x[0, 0, 10, 10] = 1
    assert torch.equal(filters.gaussian_blur(x, 2.0), filters.gaussian_blur(x, 2.0))
    assert abs(filters.gaussian_blur(x, 2.0).sum().item() - 1) < 1e-4


def test_pseudo_distance_peaks_near_radius(device):
    disc = _ellipse(101, 101, 50, 50, 20, 20, device=device)[None, None]
    d = centroids.get_pseudo_distance(disc)
    assert 20 <= d.max().item() <= 35  # peeling reads about 1.4x the radius
    # The deepest point is at the centre. Pixels just outside can be marked too, which is
    # harmless: peaks are always restricted to the mask.
    assert d[0, 0, 50, 50] == d.max()


def test_quadratic_background_flattens_offset_gaussian(cfg, device):
    X = Y = 512
    torch.manual_seed(0)
    xx, yy = torch.meshgrid(torch.arange(X, device=device).float(), torch.arange(Y, device=device).float(), indexing="ij")
    illum = torch.exp(-((xx - 300) ** 2 + (yy - 200) ** 2) / (2 * 400.0**2))
    protein = 2000 * _ellipse(X, Y, 100, 400, 12, 8, device=device).float()
    raw = (illum * (protein + 500))[None, None] + torch.randn(1, 1, X, Y, device=device) * 3
    corrected, diag = background.correct_background(raw, cfg)
    background_px = corrected[0, 0][protein == 0]
    assert background_px.quantile(0.99) < 25  # the raw background spans over 100 counts
    assert abs(corrected[0, 0, 100, 400].item() - 2000 * illum.max().item()) < 300


def test_bright_tile_does_not_blank_the_frame(cfg, device):
    # Filtered tiles are NaN in the module version; passed to the solver, they blank the fit.
    raw = torch.full((2, 1, 420, 420), 500.0, device=device) + torch.randn(2, 1, 420, 420, device=device)
    raw[:, :, :60, :60] = 20000
    corrected, _ = background.correct_background(raw, cfg)
    assert torch.isfinite(corrected).all()
    assert corrected[:, :, 300:, 300:].median() < 10


def test_mask_finds_sparse_nuclei(cfg, device):
    torch.manual_seed(0)
    image = torch.randn(3, 1, 400, 400, device=device) * 20 + 40
    truth = _ellipse(400, 400, 150, 150, 14, 9, device=device) | _ellipse(400, 400, 300, 220, 12, 12, device=device)
    image[:, :, truth] += 1000
    image = filters.gaussian_blur(image, 1.0)
    mask, _ = masking.get_binary_mask(image, cfg)
    iou = (mask[0, 0] & truth).sum() / (mask[0, 0] | truth).sum()
    assert iou > 0.6
    assert mask.float().mean() < 0.01


def test_two_nuclei_get_two_seeds(cfg, device):
    mask = (_ellipse(200, 200, 60, 60, 14, 9, device=device) | _ellipse(200, 200, 140, 140, 12, 12, device=device))[None, None]
    peaks, _ = centroids.get_centroids(mask, masking.compute_radius(mask), cfg)
    assert peaks.shape[0] == 2


def test_local_labels_encode_section_and_index(device):
    t = torch.tensor([0, 0, 0, 1], device=device)
    x = torch.tensor([10, 20, 500, 10], device=device)
    y = torch.tensor([10, 20, 10, 10], device=device)
    local, section_max = labels.get_local_labels(700, 700, t, x, y, 7)
    assert section_max == 2
    assert local[0] == 1 and local[1] == 2       # same frame and section: counted up
    assert local[3] == 1                         # another frame restarts the count
    assert local[2] // 1000 == 500 // 101        # section index for the third peak


def test_border_map_makes_nucleus_border_and_ring(device):
    nuclear = torch.zeros(1, 1, 60, 60, device=device)
    nuclear[0, 0, 20:30, 20:30] = 5
    label_map, border = labels.get_border_map(nuclear, nuclear > 0)
    assert (label_map[nuclear > 0] == 5).all()
    assert (border[0, 0, 19, 25] == 5) and (label_map[0, 0, 19, 25] == 0)  # first ring is the border
    assert (label_map[0, 0, 18, 25] == -5) and (label_map[0, 0, 17, 25] == -5)  # next two are cytoplasm
    assert label_map[0, 0, 16, 25] == 0


def test_label_growth_fills_the_mask(device):
    mask = _ellipse(100, 100, 50, 50, 20, 20, device=device)[None, None]
    peaks = torch.tensor([[0, 50, 50]], device=device)
    grown = labels.get_label_map(peaks, mask, centroids.get_pseudo_distance(mask))
    assert (grown[mask] == 1).all() and (grown[~mask] == 0).all()
