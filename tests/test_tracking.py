"""Linking, drift, divisions and temporal splitting on tiny controlled inputs."""
import numpy as np
import pytest
import torch

from fasttrack.config import make_config
from fasttrack.track import linking
from fasttrack.track.measure import CHANNELS, K, S, SCALARS
from fasttrack.track.undersegmentation import temporal_split


def table(rows, device, C=2):
    """rows: (t, x, y, brightness, area). Sorted by t, as the pipeline guarantees."""
    rows = sorted(rows, key=lambda r: r[0])
    N = len(rows)
    sc = torch.full((N, len(SCALARS)), 0.0, device=device)
    ch = torch.full((N, C, len(CHANNELS)), 0.0, device=device)
    for i, (t, x, y, b, a) in enumerate(rows):
        sc[i, S["t"]], sc[i, S["x"]], sc[i, S["y"]], sc[i, S["nuc_area"]] = t, x, y, a
        sc[i, S["eccentricity"]] = 0.5
        sc[i, S["major_len"]], sc[i, S["minor_len"]], sc[i, S["edge_n"]] = 18.0, 11.0, 40.0
        for c in range(C):
            ch[i, c, K["nuc_median"]] = b
            ch[i, c, K["nuc_std"]] = b / 10
            ch[i, c, K["nuc_mean"]] = b
            ch[i, c, K["cyto_mean"]] = b / 4
            ch[i, c, K["cyto_median"]] = b / 4
            ch[i, c, K["nuc_sum"]] = b * a
    return sc, ch


def lines(n_frames, specs, jitter=0.3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for (x, y, vx, vy, b) in specs:
        for t in range(n_frames):
            rows.append((t, x + vx * t + rng.normal(0, jitter), y + vy * t + rng.normal(0, jitter),
                         b * np.exp(rng.normal(0, 0.02)), 150 * np.exp(rng.normal(0, 0.02))))
    return rows


VOTING = {"link": {"method": "tracks"}}
NOTEBOOK = {"link": {"method": "notebook"}}


@pytest.mark.parametrize("updates", [{}, NOTEBOOK, VOTING], ids=["multipass", "notebook", "voting"])
def test_separate_cells_make_separate_tracks(device, updates):
    sc, ch = table(lines(30, [(100, 100, 2, 0, 500), (300, 300, 0, 2, 1000), (500, 100, -1, 1, 2000)]), device)
    result = linking.link_tracks(sc, ch, 7, make_config(updates))
    assert result["tracks"] == 3
    assert result["divisions"] == 0


@pytest.mark.parametrize("updates", [{}, NOTEBOOK, VOTING], ids=["multipass", "notebook", "voting"])
def test_no_candidate_starts_a_new_track(device, updates):
    # Regression: the notebook's argmin over all-inf costs linked to row 0.
    rows = lines(10, [(100, 100, 0, 0, 500)]) + [(t, 800.0, 800.0, 500.0, 150.0) for t in range(5, 10)]
    sc, ch = table(rows, device)
    assert linking.link_tracks(sc, ch, 7, make_config(updates))["tracks"] == 2


def test_voting_keeps_identity_through_a_close_pass(device):
    sc, ch = table(lines(40, [(100, 100, 2, 0, 400), (180, 103, -2, 0, 3000)]), device)
    result = linking.link_tracks(sc, ch, 7, make_config(VOTING))
    assert result["tracks"] == 2
    bright = ch[:, 0, K["nuc_median"]].cpu().numpy() > 1000
    assert len(set(result["track_id"][bright])) == 1


def test_voting_closes_a_gap(device):
    rows = [r for r in lines(30, [(100, 100, 1, 0, 500), (400, 400, 0, 1, 1500)]) if not (r[0] in (15, 16) and r[3] < 800)]
    sc, ch = table(rows, device)
    assert linking.link_tracks(sc, ch, 7, make_config(VOTING))["tracks"] == 2


def test_voting_declares_division_with_mass_conserved(device):
    mother = [(t, 100.0 + t * 0.2, 100.0, 1000.0, 300.0) for t in range(15)]
    d1 = [(t, 106.0 + 1.5 * (t - 15), 103.0 + 1.5 * (t - 15), 1000.0, 150.0) for t in range(15, 30)]
    d2 = [(t, 106.0 - 1.5 * (t - 15), 97.0 - 1.5 * (t - 15), 1000.0, 150.0) for t in range(15, 30)]
    others = lines(30, [(500, 500, 1, 0, 300), (700, 200, 0, 1, 700)])
    sc, ch = table(mother + d1 + d2 + others, device)
    result = linking.link_tracks(sc, ch, 7, make_config(VOTING))
    assert result["divisions"] == 1
    assert (result["mother_track"] >= 0).sum() == 2


def test_notebook_link_makes_shared_mother_a_division(device):
    mother = [(t, 100.0, 100.0, 1000.0, 300.0) for t in range(5)]
    kids = [(t, 100.0 + s * 6 * (t - 4), 100.0, 1000.0, 150.0) for t in range(5, 10) for s in (1, -1)]
    sc, ch = table(mother + kids, device)
    result = linking.link_tracks(sc, ch, 7, make_config(NOTEBOOK))
    assert result["divisions"] == 1 and (result["mother_track"] >= 0).sum() == 2


def _division_rows():
    mother = [(t, 100.0 + t * 0.2, 100.0, 1000.0, 300.0) for t in range(15)]
    d1 = [(t, 106.0 + 1.5 * (t - 15), 103.0 + 1.5 * (t - 15), 1000.0, 150.0) for t in range(15, 30)]
    d2 = [(t, 106.0 - 1.5 * (t - 15), 97.0 - 1.5 * (t - 15), 1000.0, 150.0) for t in range(15, 30)]
    return mother + d1 + d2 + lines(30, [(500, 500, 1, 0, 300), (700, 200, 0, 1, 700)])


def test_multipass_accepts_division_with_evidence(device):
    sc, ch = table(_division_rows(), device)
    result = linking.link_tracks(sc, ch, 7, make_config({"link": {"max_travel": 30.0}}))
    assert result["divisions"] == 1
    assert (result["mother_track"] >= 0).sum() == 2
    assert result["tracks"] == 5  # mother, two daughters, two others


def test_multipass_rejects_shared_mother_when_mass_is_not_conserved(device):
    # A much brighter cell appears beside another: its mass rules out being a daughter.
    rows = lines(30, [(100, 100, 0.2, 0, 1000)]) + [(t, 112.0, 100.0, 3000.0, 150.0) for t in range(15, 30)]
    sc, ch = table(rows, device)
    result = linking.link_tracks(sc, ch, 7, make_config({"link": {"max_travel": 30.0}}))
    assert result["divisions"] == 0
    assert result["tracks"] == 2
    assert (result["mother_track"] >= 0).sum() == 0


def test_multipass_cleanup_gives_a_loser_its_free_mother(device):
    # At frame 10 both cells prefer mother A. The loser must recover mother B in cleanup.
    a = [(t, 100.0, 100.0, 1000.0, 150.0) for t in range(20)]
    b = [(t, 112.0, 100.0, 1200.0, 150.0) for t in range(10)] + [(t, 112.0, 100.0, 1020.0, 150.0) for t in range(10, 20)]
    others = lines(20, [(500, 500, 0, 0, 300), (700, 200, 0, 0, 3000)])
    sc, ch = table(a + b + others, device)
    result = linking.link_tracks(sc, ch, 7, make_config({"link": {"max_travel": 30.0, "cleanup_cost_quantile": 1.0}}))
    assert result["divisions"] == 0
    assert result["tracks"] == 4


def test_multipass_closes_a_gap(device):
    rows = [r for r in lines(30, [(100, 100, 1, 0, 500), (400, 400, 0, 1, 1500)]) if not (r[0] in (15, 16) and r[3] < 800)]
    sc, ch = table(rows, device)
    result = linking.link_tracks(sc, ch, 7, make_config({"link": {"max_travel": 30.0, "cleanup_cost_quantile": 1.0}}))
    assert result["tracks"] == 2


def test_max_travel_estimate_and_override(device):
    sc, ch = table(lines(40, [(100 + 120 * i, 100 + 100 * (i % 4), 3, 2, 500 + 300 * i) for i in range(10)]), device)
    auto = linking.link_tracks(sc, ch, 7, make_config())
    assert 5 <= auto["max_travel"] <= 30
    assert linking.link_tracks(sc, ch, 7, make_config({"link": {"max_travel": 11.0}}))["max_travel"] == 11.0


def _textured_frames(shift, device, T=4):
    """Random blobs; from frame 2, the content moves by `shift`."""
    torch.manual_seed(0)
    base = (torch.rand(1, 256, 256, device=device) > 0.995).float()
    base = torch.nn.functional.max_pool2d(base, 9, stride=1, padding=4)[0]
    frames = base.repeat(T, 1, 1)
    frames[2:] = torch.roll(frames[2:], shift, dims=(1, 2))
    return frames


def test_fft_shift_sign_and_correction(device):
    # Regression: x - shift, as in the notebook's lineage cell, doubles a stage jump.
    frames = _textured_frames((5, -3), device)
    d = linking.find_shifts(frames[:-1], frames[1:])
    assert d[1].tolist() == [-5.0, 3.0] and d[0].tolist() == [0.0, 0.0] and d[2].tolist() == [0.0, 0.0]
    aligned = 100.0 + 5 + d.cumsum(0)[1, 0].item()
    assert aligned == 100.0


@pytest.mark.parametrize("updates", [{}, NOTEBOOK, VOTING], ids=["multipass", "notebook", "voting"])
def test_stage_jump_is_removed_before_distances(device, updates):
    specs = [(100 + 80 * i, 100 + 60 * (i % 3), 0, 0, 500 * (1 + i)) for i in range(8)]
    rows = [(t, x + (45 if t >= 10 else 0), y + (-40 if t >= 10 else 0), b, a) for t, x, y, b, a in lines(20, specs)]
    sc, ch = table(rows, device)
    frame_shift = torch.zeros(20, 2, device=device)
    frame_shift[10] = torch.tensor([-45.0, 40.0], device=device)
    result = linking.link_tracks(sc, ch, 7, make_config(updates), frame_shift)
    assert result["tracks"] == 8
    assert np.allclose(result["drift"][-1], [-45, 40])
    # The same jump is beyond max_travel when it is not corrected.
    assert linking.link_tracks(sc, ch, 7, make_config(updates))["tracks"] > 8


def test_trace_index_round_trip():
    index = linking.trace_index(np.array([0, 1, 0, 1, -1]), np.array([0, 0, 1, 2, 2]), 3)
    assert index.tolist() == [[0, 2, -1], [1, -1, 3]]


def _disc(frame, cx, cy, r, value):
    X, Y = frame.shape
    xx, yy = torch.meshgrid(torch.arange(X, device=frame.device).float(), torch.arange(Y, device=frame.device).float(), indexing="ij")
    frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= r**2] = value


def test_temporal_split_recovers_a_merge(cfg, device):
    labels = torch.zeros(3, 1, 100, 100, device=device)
    _disc(labels[0, 0], 50, 35, 8, 1)
    _disc(labels[0, 0], 50, 65, 8, 2)
    _disc(labels[1, 0], 50, 42, 8, 3)  # merged blob in frames 1 and 2
    _disc(labels[1, 0], 50, 58, 8, 3)
    labels[1, 0, 46:54, 42:58] = 3
    _disc(labels[2, 0], 50, 44, 8, 4)
    _disc(labels[2, 0], 50, 56, 8, 4)
    labels[2, 0, 46:54, 44:56] = 4
    splits = temporal_split(labels, 200.0, cfg)
    assert splits == 2
    assert labels[1].unique().numel() == 3  # background + 2
    assert labels[2].unique().numel() == 3
    # A new label must be unique across the batch, not only within its frame.
    per_frame = [set(labels[t].unique().tolist()) - {0.0} for t in range(3)]
    assert not (per_frame[1] & per_frame[2])


def test_temporal_split_leaves_a_small_nucleus_alone(cfg, device):
    labels = torch.zeros(2, 1, 100, 100, device=device)
    _disc(labels[0, 0], 50, 44, 5, 1)  # an over-split nucleus, in two parts
    _disc(labels[0, 0], 50, 56, 5, 2)
    _disc(labels[1, 0], 50, 50, 8, 3)  # one normal nucleus
    assert temporal_split(labels, 200.0, cfg) == 0
