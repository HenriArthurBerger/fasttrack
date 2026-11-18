"""Stochastic dynamics: the generator, recovery of switching faster than the frame interval, the
continuous-time hidden Markov model, readouts with per-observation error, a full fit from an
experiment file to saved reports and figures, and the Analysis tab sections."""
import copy
import json
import math
import os

import numpy as np
import pytest
import torch

from fasttrack.dynamics import engine as E
from fasttrack.dynamics.models import ErkKtr, OrnsteinUhlenbeck, States, fokker_planck, stationary
from fasttrack.experiment import new_experiment, save_experiment


def test_fokker_planck_grid_matches_the_ou_stationary_law():
    m = OrnsteinUhlenbeck(grid=80)
    m.set_range(-3, 3)
    p = {"theta": torch.tensor([0.2], dtype=torch.float64), "mu": torch.tensor([0.5], dtype=torch.float64),
         "sigma": torch.tensor([0.4], dtype=torch.float64), "noise": torch.tensor([0.1], dtype=torch.float64)}
    Q, _, _ = m.build(p)
    assert torch.allclose(Q.sum(-1), torch.zeros(1, 80, dtype=torch.float64), atol=1e-10)
    assert (Q[0] - torch.diag(torch.diag(Q[0])) >= 0).all()
    pi = stationary(Q)[0]
    x = m.centres("cpu")
    mean = float((pi * x).sum())
    var = float((pi * (x - mean) ** 2).sum())
    assert abs(mean - 0.5) < 0.01
    assert abs(var - 0.4 ** 2 / (2 * 0.2)) < 0.02          # σ² / 2θ = 0.4
    # strong drift: no overflow, still a generator
    Qs = fokker_planck(torch.full((1, 9), 1e4, dtype=torch.float64), torch.tensor([1e-6], dtype=torch.float64), 0.1)
    assert torch.isfinite(Qs).all() and torch.allclose(Qs.sum(-1), torch.zeros(1, 10, dtype=torch.float64), atol=1e-6)


def test_erk_switching_faster_than_the_frames_is_recovered(device):
    """6-minute frames, ERK dwell times of 10 (off) and 5 (on) minutes, simulated in continuous time."""
    m = ErkKtr(grid=40, modes=2)
    m.set_range(-3.5, 2.5)
    truth = {"k_on": 0.1, "k_off": 0.2, "level_1": -1.0, "level_2": 0.8, "k_import": 0.5, "sigma": 0.05, "noise": 0.15}
    gen = torch.Generator().manual_seed(1)
    y = m.simulate(truth, 1200, 80, 6.0, gen, substeps=30).numpy()
    rng = np.random.default_rng(0)
    se = rng.uniform(0.05, 0.3, y.shape)
    y = y + rng.normal(size=y.shape) * np.sqrt(truth["noise"] ** 2 + se ** 2)
    y[np.arange(80)[None] < rng.integers(0, 30, len(y))[:, None]] = np.nan   # cells appear at different times
    b = E.Batches(y, se, np.zeros(len(y), int), 1, device)
    lay = E.Layout(m, {}, 1, 1, minutes=True)
    z, _ = E.fit(m, lay, b, np.zeros(80, int), 6.0, steps=200, batch=600, start=m.start_values(torch.as_tensor(y[np.isfinite(y)])))
    d = m.derived(lay.values(z))
    assert abs(float(d["dwell_1"][0][0]) / 10 - 1) < 0.25
    assert abs(float(d["dwell_2"][0][0]) / 5 - 1) < 0.25
    v = lay.values(z)
    assert abs(float(v["level_1"][0]) + 1.0) < 0.15 and abs(float(v["level_2"][0]) - 0.8) < 0.15


def test_continuous_time_hmm_recovers_states_and_rates(device):
    m = States(states=2, features=2)
    m.set_range(-4, 4)
    truth = {"rate_1_2": 0.05, "rate_2_1": 0.1, "mean_1_1": -1.0, "mean_1_2": 0.5, "mean_2_1": 1.0, "mean_2_2": -0.5,
             "sd_1_1": 0.4, "sd_1_2": 0.4, "sd_2_1": 0.4, "sd_2_2": 0.4, "init_2": 0.0}
    y = m.simulate(truth, 600, 60, 6.0, torch.Generator().manual_seed(2)).numpy()
    b = E.Batches(y, np.zeros_like(y), np.zeros(len(y), int), 1, device)
    lay = E.Layout(m, {}, 1, 1, minutes=True)
    z, _ = E.fit(m, lay, b, np.zeros(60, int), 6.0, steps=250, batch=600, start=m.start_values(torch.as_tensor(y.reshape(-1, 2))))
    v = {k: float(x[0]) for k, x in lay.values(z).items()}
    assert abs(v["mean_1_1"] + 1) < 0.15 and abs(v["mean_2_1"] - 1) < 0.15 and abs(v["mean_1_2"] - 0.5) < 0.15
    assert abs(math.log(v["rate_1_2"] / 0.05)) < math.log(1.4) and abs(math.log(v["rate_2_1"] / 0.1)) < math.log(1.4)


# ---------------------------------------------------------------------- a small results folder
def _write_results(folder, key, ktr, nuc_level=400.0, rng=None):
    """Measurements and traces of tracks whose ln(C/N) follows `ktr` (tracks, T); channel 0 = KTR."""
    from fasttrack.io import store
    from fasttrack.track.measure import CHANNELS, SCALARS
    rng = rng or np.random.default_rng(0)
    n, T = ktr.shape
    rows = n * T
    track = np.repeat(np.arange(n), T)
    t = np.tile(np.arange(T), n)
    scalars = np.full((rows, len(SCALARS)), np.nan, np.float32)
    channels = np.full((rows, 1, len(CHANNELS)), np.nan, np.float32)
    S = {s: i for i, s in enumerate(SCALARS)}
    K = {c: i for i, c in enumerate(CHANNELS)}
    scalars[:, S["t"]] = t
    scalars[:, S["x"]] = 50 + 30 * track + t * 0.5
    scalars[:, S["y"]] = 50.0
    scalars[:, S["touches_border"]] = 0
    scalars[:, S["nuc_area"]] = 200
    scalars[:, S["cyto_area"]] = 150
    nuc = nuc_level * (1 + 0.02 * rng.normal(size=rows))
    channels[:, 0, K["nuc_median"]] = nuc
    channels[:, 0, K["cyto_median"]] = nuc * np.exp(ktr.reshape(-1))
    channels[:, 0, K["nuc_std"]] = 40
    channels[:, 0, K["cyto_std"]] = 40
    store.save_measurements(folder / "Measurements" / f"{key}.h5", scalars, channels, {"frames": T, "id": key})
    index = np.arange(rows).reshape(n, T)
    links = {"track_id": track, "mother_track": np.full(n, -1), "drift": np.zeros((T, 2))}
    store.save_traces(folder / "Traces" / f"{key}.h5", links, index, {"frames": T})


def _experiment(tmp_path):
    rng = np.random.default_rng(3)
    m = ErkKtr(grid=32)
    m.set_range(-4, 3)
    gen = torch.Generator().manual_seed(4)
    fast = {"k_on": 0.15, "k_off": 0.15, "level_1": -1.5, "level_2": 0.5, "k_import": 0.5, "sigma": 0.05, "noise": 0.1}
    slow = {**fast, "k_on": 0.02}
    for key, p in (("A01_xy1", fast), ("A01_xy2", fast), ("B01_xy1", slow), ("B01_xy2", slow)):
        ktr = m.simulate(p, 60, 50, 6.0, gen, substeps=20)[..., 0].numpy() + 0.05 * rng.normal(size=(60, 50))
        _write_results(tmp_path / "res", key, ktr, rng=rng)
    exp = new_experiment(
        paths={"raw": str(tmp_path / "raw"), "results": str(tmp_path / "res")},
        channels=[{"file_channel": 1, "name": "ktr", "role": "measure", "reporter": "ERKKTR"}],
        time={"frames_per_hour": 10},
        plate={"conditions": [{"name": "Ctrl", "wells": ["A01"]}, {"name": "MekI", "wells": ["B01"]}]},
        analysis={"min_track_frames": 10, "readouts": [{"name": "ERK", "kind": "ktr", "channel": "ktr", "background_percentile": 0.0}],
                  "dynamics": {"device": "cpu", "fits": [
                      {"name": "erk", "model": "erk_ktr", "steps": 60, "batch": 60, "bootstrap": 2, "recovery_check": True,
                       "options": {"grid": 32}},
                      {"name": "ou", "model": "ou", "steps": 40, "batch": 60, "options": {"grid": 32}}]}})
    return exp


def test_readout_error_and_speed(tmp_path):
    from fasttrack.analysis.readouts import readout_rows, speed_column, suggest
    from fasttrack.io import store
    exp = _experiment(tmp_path)
    assert suggest(exp)[0]["channel"] == "ktr"
    res = store.load_results(exp["paths"]["results"], "A01_xy1")
    t, track, value, se = readout_rows(res, exp, exp["analysis"]["readouts"][0])
    expected = math.sqrt(math.pi / 2) * 40 / math.sqrt(200) / 400        # nucleus term; ring term on top
    assert np.all(se > expected * 0.9) and np.all(se < 1)
    assert np.allclose(np.nanmedian(speed_column(res)), 0.5, atol=1e-6)   # 0.5 px per frame along x


def test_fit_from_experiment_file_saves_reports_and_figures(tmp_path):
    from fasttrack.dynamics.fit import compare, read_fit, read_states, run_all
    exp = _experiment(tmp_path)
    reports = run_all(exp)
    erk = read_fit(exp, "erk")
    assert erk["time_unit"] == "min" and erk["interval"] == 6.0
    assert erk["conditions"] == ["Ctrl", "MekI"] and erk["tracks"] == 240
    k_on = np.asarray(erk["parameters"]["k_on"]["values"])[:, 0]
    assert k_on[0] > k_on[1]                               # the slow condition switches on less often
    assert erk["parameters"]["k_on"]["ci"] is not None and erk["recovery"]
    assert set(erk["_checks"]) >= {"hist_obs", "hist_pred", "pit", "acf_obs", "acf_sim"}
    states = read_states(exp, "erk")
    assert states["modes"].shape == (240, 50, 2) and np.allclose(np.nansum(states["modes"].astype(float), -1)[:, 5], 1, atol=0.01)
    assert (tmp_path / "res" / "Dynamics" / "erk" / "figures" / "derived.png").exists()
    rows = compare(exp)
    assert {r["name"] for r in rows} == {"erk", "ou"} and rows[0]["data_key"] == rows[1]["data_key"]
    # the command line runs a named fit
    from fasttrack.cli import main
    path = tmp_path / "exp.yaml"
    save_experiment(exp, path)
    main(["dynamics", str(path), "ou"])
    assert json.loads((tmp_path / "res" / "Dynamics" / "ou" / "fit.json").read_text())["model"] == "ou"


def test_windows_follow_hours_and_events(tmp_path):
    from fasttrack.dynamics.fit import windows
    exp = _experiment(tmp_path)
    exp["time"]["events"] = [{"name": "drug", "frame": 25, "color": "#000"}]
    w, spans = windows(exp, {"window_hours": 2.0, "split_at_events": True}, 50, 0, 6.0, "min")
    assert spans == [(0, 20), (20, 25), (25, 40), (40, 50)]
    assert w[24] == 1 and w[25] == 2


def test_analysis_tab_sections_keep_readouts_and_fits(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from fasttrack.gui.app import MainWindow
    app = QApplication.instance() or QApplication([])
    exp = _experiment(tmp_path)
    window = MainWindow()
    window.load(exp)
    a = window.analysis
    pages = [a.tabs.tabText(a.tabs.indexOf(w)) for w in (a.figures, a.readouts, a.features, a.fates, a.dynamics, a.export)]
    assert pages == ["Figures", "Readouts", "Features", "Fates", "Dynamics", "Export"]
    d = a.dynamics
    assert [f["name"] for f in d.collect()["fits"]] == ["erk", "ou"]
    d.list.list.setCurrentRow(1)
    assert d.model.currentData() == "ou" and d.table.rowCount() == 4
    d.model.setCurrentIndex(d.model.findData("states"))
    assert d.table.rowCount() == 3 * 3 - 3 + 3 * 2 + 2       # rates, means and sds, initial odds
    d.name.setText("hmm")
    d.list.add()
    out = window.collect()
    assert [f["name"] for f in out["analysis"]["dynamics"]["fits"]] == ["erk", "ou", "hmm"]
    assert out["analysis"]["readouts"][0]["name"] == "ERK" and out["analysis"]["min_track_frames"] == 10
    assert "6 min" in d.interval_note.text()
    window.saved = copy.deepcopy(window.exp)  # no "save changes?" dialog on close
    window.close()
    app.processEvents()
