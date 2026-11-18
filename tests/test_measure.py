"""measure_subsections checked against a brute-force NumPy loop."""
import numpy as np
import torch

from fasttrack.track import labels
from fasttrack.track.measure import K, M, S, flatten_measurements, mark_border_cells, measure_batch, measure_subsections


def _scene(device, seed=0):
    rng = np.random.default_rng(seed)
    T, C, X, Y = 3, 3, 140, 140
    data = rng.normal(100, 30, (T, C, X, Y)).round().astype(np.float32)
    nuclear = np.zeros((T, 1, X, Y), np.float32)
    n = 1
    for t in range(T):
        for cx, cy in ((30, 30), (30, 90), (100, 60)):
            cx += int(rng.integers(-4, 4))
            cy += int(rng.integers(-4, 4))
            yy, xx = np.ogrid[:X, :Y]
            inside = np.hypot(yy - cx, xx - cy) < 10
            nuclear[t, 0][inside] = n
            data[t, :, inside] *= rng.uniform(2, 5)
            n += 1
    nuclear = torch.as_tensor(nuclear, device=device)
    encoded, section_max = labels.encode_local_labels(nuclear, 7)
    label_map, border = labels.get_border_map(encoded, nuclear > 0)
    return torch.as_tensor(data, device=device), label_map, border, section_max, data


def test_intensities_match_brute_force(cfg, device):
    data, label_map, border, section_max, data_np = _scene(device)
    scalars, channels = measure_batch(data, label_map, border, section_max, 9, cfg)
    sc, ch = scalars.cpu().numpy(), channels.cpu().numpy()
    lab = label_map.cpu().numpy()
    assert sc.shape[0] == 9
    for row in range(sc.shape[0]):
        t, label = int(sc[row, S["t"]]), int(sc[row, S["label"]])
        nuc, cyto = lab[t, 0] == label, lab[t, 0] == -label
        assert sc[row, S["nuc_area"]] == nuc.sum()
        assert sc[row, S["cyto_area"]] == cyto.sum()
        px, py = np.nonzero(nuc)
        assert abs(sc[row, S["x"]] - px.mean()) < 1e-2 and abs(sc[row, S["y"]] - py.mean()) < 1e-2
        for c in range(3):
            v, w = data_np[t, c][nuc], data_np[t, c][cyto]
            assert np.isclose(ch[row, c, K["nuc_mean"]], v.mean(), rtol=1e-3)
            assert np.isclose(ch[row, c, K["nuc_sum"]], v.sum(), rtol=1e-3)
            assert abs(ch[row, c, K["nuc_median"]] - np.sort(v)[(v.size - 1) // 2]) <= 1
            assert abs(ch[row, c, K["nuc_min"]] - v.min()) <= 1 and abs(ch[row, c, K["nuc_max"]] - v.max()) <= 1
            assert abs(ch[row, c, K["cyto_median"]] - np.sort(w)[(w.size - 1) // 2]) <= 1


def test_local_density_runs_with_several_channels(cfg, device):
    # Regression: the reshape in _fill_local_density dropped the channel axis.
    data, label_map, border, section_max, _ = _scene(device)
    out = measure_subsections(data.round().to(torch.int32), label_map, border, section_max, 9)
    scalars, _ = flatten_measurements(out)
    assert not torch.isnan(scalars[:, S["local_density"]]).any()


def test_values_above_int16_are_kept(cfg, device):
    # Regression: the notebook cast to int16, wrapping 40000 to -25536.
    nuclear = torch.zeros(1, 1, 60, 60, device=device)
    nuclear[0, 0, 20:30, 20:30] = 1
    encoded, section_max = labels.encode_local_labels(nuclear, 7)
    label_map, border = labels.get_border_map(encoded, nuclear > 0)
    data = torch.full((1, 1, 60, 60), 40000.0, device=device)
    _, channels = measure_batch(data, label_map, border, section_max, 5, cfg)
    assert channels[0, 0, K["nuc_median"]] == 40000


def test_border_cells_are_flagged(cfg, device):
    nuclear = torch.zeros(1, 1, 70, 70, device=device)
    nuclear[0, 0, 0:8, 20:30] = 1
    nuclear[0, 0, 40:50, 40:50] = 2
    encoded, section_max = labels.encode_local_labels(nuclear, 7)
    label_map, border = labels.get_border_map(encoded, nuclear > 0)
    scalars, _ = measure_batch(torch.ones(1, 1, 70, 70, device=device) * 100, label_map, border, section_max, 5, cfg)
    assert sorted(scalars[:, S["touches_border"]].tolist()) == [0.0, 1.0]


def test_too_many_cells_in_a_section_is_an_error(device):
    import pytest
    t = torch.zeros(1000, device=device, dtype=torch.long)
    x = torch.zeros(1000, device=device, dtype=torch.long)
    with pytest.raises(ValueError):
        labels.get_local_labels(700, 700, t, x, x, 7)
