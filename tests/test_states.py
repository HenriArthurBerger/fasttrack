"""The states model on many features: correlated emissions, missing features, divisions as
observations, births, held-out choice of the number of states, and routes between states."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fasttrack.dynamics import engine as E
from fasttrack.dynamics.models import States


def _cycle(rng, n_roots=150, T=120, dt=6.0, missing=0.3):
    """A 3-state cycle 1 -> 2 -> 3 (dwell 60, 60, 40 min), division from state 3 (rate 1/40 per min) and a
    little from 2, daughters born into state 1. 3 features: state means plus noise sharing one direction."""
    K = 3
    Q = np.zeros((K, K))
    Q[0, 1], Q[1, 2], Q[2, 0] = 1 / 60, 1 / 60, 1 / 400
    Q -= np.diag(Q.sum(1))
    P = __import__("scipy.linalg", fromlist=["expm"]).expm(Q * dt)
    hazard = np.array([0.0, 0.002, 1 / 40])
    means = np.array([[-1.0, -1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 1.2, -0.5]])
    load = np.array([0.6, 0.6, 0.0])                   # features 1 and 2 move together (e.g. area and DNA)
    tracks, mother, born, divides = [], [], [], []
    queue = [(int(rng.integers(0, K)), int(rng.integers(0, 20)), -1, False) for _ in range(n_roots)]
    while queue:
        state, t0, mo, b = queue.pop()
        y = np.full((T, 3), np.nan)
        t, s, div = t0, state, False
        while t < T:
            y[t] = means[s] + 0.3 * rng.normal(size=3) + load * rng.normal()
            if t + 1 >= T:
                break
            if rng.random() < 1 - np.exp(-hazard[s] * dt):
                div = True
                break
            s = rng.choice(K, p=P[s])
            t += 1
        me = len(tracks)
        tracks.append(y)
        mother.append(mo)
        born.append(b)
        divides.append(div)
        if div and t + 1 < T and len(tracks) < 3000:
            queue += [(0, t + 1, me, True), (0, t + 1, me, True)]
    y = np.stack(tracks)
    y[rng.random(y.shape) < missing] = np.nan
    return y, np.array(born), np.array(divides), Q, hazard, means


def test_states_with_divisions_and_correlated_features_recover_the_cycle(device):
    rng = np.random.default_rng(0)
    y, born, divides, Q, hazard, means = _cycle(rng)
    ev = E.track_events(y, divides, born)
    seen = np.isfinite(y).any((1, 2))      # a track whose every value went missing carries no event
    assert ((ev & E.DIVIDE) > 0).sum() == (divides & seen).sum() and ((ev & E.BORN) > 0).sum() == (born & seen).sum()
    m = States(states=3, features=3, rank=1, division=True, topology="full")
    m.set_range(-4, 4)
    lay = E.Layout(m, {}, 1, 1, minutes=True)
    b = E.Batches(y, np.zeros_like(y), np.zeros(len(y), int), 1, device, ev=ev)
    z, hist = E.fit(m, lay, b, np.zeros(y.shape[1], int), 6.0, steps=400, batch=512, lr=0.05,
                    start=m.start_values(torch.as_tensor(y[np.isfinite(y).any(-1)])))
    p = {k: v.detach().cpu().numpy()[0] for k, v in lay.values(z.to(device)).items()}
    got = np.array([[p[f"mean_{k + 1}_{f + 1}"] for f in range(3)] for k in range(3)])
    order = [int(np.argmin(((got - mu) ** 2).sum(1))) for mu in means]     # match fitted states to the true ones
    assert sorted(order) == [0, 1, 2]
    assert np.abs(got[order] - means).max() < 0.25
    h = np.array([p[f"division_{k + 1}"] for k in order])
    assert h[2] > 5 * h[1] and abs(np.log(h[2] * 40)) < np.log(1.6)        # divisions come from the last phase
    birth = np.array([1.0] + [np.exp(p[f"birth_{k + 1}"]) for k in range(1, 3)])
    birth /= birth.sum()
    assert birth[order[0]] > 0.8                                           # newborn cells start in the first phase
    for i, j, true in ((0, 1, 1 / 60), (1, 2, 1 / 60)):
        rate = p[f"rate_{order[i] + 1}_{order[j] + 1}"]
        assert abs(np.log(rate / true)) < np.log(1.6)
    loads = np.array([p[f"load_{order[0] + 1}_{f + 1}_1"] for f in range(3)])
    assert abs(loads[0]) > 0.3 and abs(loads[1]) > 0.3 and np.sign(loads[0]) == np.sign(loads[1])


def test_missing_features_are_left_out_exactly(device):
    """A frame with only feature 1 must score as the 1-D marginal of the correlated Gaussian."""
    em = {"mean": torch.zeros(1, 1, 1, 2, device=device), "var": torch.full((1, 1, 1, 2), 0.5, device=device),
          "load": torch.tensor([[[[[1.0], [0.8]]]]], device=device), "dt": 1.0}
    y = torch.tensor([[[[0.7, float("nan")]]]], device=device)
    loge, info = E.emission(em, y, torch.zeros_like(y), torch.zeros(1, dtype=torch.long, device=device), outlier=0.0)
    var1 = 0.5 + 1.0
    assert abs(float(loge[0, 0, 0, 0]) + 0.5 * (0.49 / var1 + np.log(var1) + np.log(2 * np.pi))) < 1e-5
    y2 = torch.tensor([[[[0.7, -0.2]]]], device=device)
    loge2, _ = E.emission(em, y2, torch.zeros_like(y2), torch.zeros(1, dtype=torch.long, device=device), outlier=0.0)
    S = np.array([[1.5, 0.8], [0.8, 0.5 + 0.64]])
    v = np.array([0.7, -0.2])
    exact = -0.5 * (v @ np.linalg.solve(S, v) + np.log(np.linalg.det(S)) + 2 * np.log(2 * np.pi))
    assert abs(float(loge2[0, 0, 0, 0]) - exact) < 1e-5


def test_routes_committor_and_flux():
    from fasttrack.dynamics.fit import routes
    Q = np.array([[-1.0, 1.0, 0.0], [0.5, -1.5, 1.0], [0.0, 0.0, 0.0]])   # 1 -> 2 -> 3, 2 can fall back
    r = routes(Q, [0], [2])
    assert np.allclose(r["committor"], [0, 2 / 3, 1])
    assert np.isclose(r["net"][0, 1], 1.0) and np.isclose(r["net"][1, 2], 1.0)  # one net passage per cell
    assert np.isclose(r["time"], 2.5)          # 1.5 visits to state 1 (1 each) and to state 2 (1/1.5 each)


def test_states_fit_on_features_with_sweep_and_routes(tmp_path):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_fates import _experiment
    from fasttrack.dynamics.fit import read_fit, run_fit
    exp, _ = _experiment(tmp_path)
    exp["analysis"]["dynamics"]["device"] = "cpu"
    fit = {"name": "discovery", "model": "states", "features": ["PPARG nuc_mean", "ERK", "H2B nuc_sum"],
           "options": {"states": 2, "rank": 1, "division": True, "sweep": [2, 3], "flux_feature": "PPARG nuc_mean"},
           "steps": 80, "batch": 64, "heldout": 0.34}
    r = run_fit(exp, fit)
    assert [s["states"] for s in r["sweep"]] == [2, 3] and r["states"] in (2, 3)
    assert r["inputs"] == "features" and r["feature_names"] == fit["features"]
    assert r["flux"]["committor"][0][0][r["flux"]["to"][0] - 1] == 1.0
    saved = read_fit(exp, "discovery")
    figs = {p.stem for p in (tmp_path / "res" / "Dynamics" / "discovery" / "figures").glob("*.png")}
    assert {"profiles", "graph", "routes", "sweep", "occupancy"} <= figs and saved["states"] == r["states"]


def test_dynamics_editor_sets_up_a_discovery_fit(tmp_path):
    import copy
    import os
    import sys
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from test_fates import _experiment
    from fasttrack.gui.app import MainWindow
    app = QApplication.instance() or QApplication([])
    exp, _ = _experiment(tmp_path)
    window = MainWindow()
    window.load(exp)
    a = window.analysis
    a.features._suggest()
    d = a.dynamics
    a.tabs.setCurrentWidget(d)
    d.model.setCurrentIndex(d.model.findData("states"))
    assert d.states_box.isVisibleTo(d) and d.features.count() > 5
    for i in range(d.features.count()):
        if d.features.item(i).text() in ("PPARG nuc_mean", "ERK", "nuc_area"):
            d.features.item(i).setCheckState(Qt.CheckState.Checked)
    d.states_n.setValue(4)
    d.rank.setValue(1)
    assert d.table.rowCount() == 4 * 3 + 4 * 3 * 3 + 3        # rates, means + sds + loadings, initial odds
    d.division.setChecked(True)
    d.sweep.setText("3, 4, 6")
    d.flux_feature.setCurrentIndex(d.flux_feature.findData("PPARG nuc_mean"))
    d.name.setText("discovery")
    d.list.add()
    fit = window.collect()["analysis"]["dynamics"]["fits"][-1]
    assert fit["features"] == ["nuc_area", "PPARG nuc_mean", "ERK"] or set(fit["features"]) == {"nuc_area", "PPARG nuc_mean", "ERK"}
    assert fit["options"]["sweep"] == [3, 4, 6] and fit["options"]["division"] and fit["options"]["rank"] == 1
    assert fit["options"]["flux_feature"] == "PPARG nuc_mean"
    d.list.list.setCurrentRow(len(d.list.items) - 1)
    assert d.sweep.text() == "3, 4, 6" and d.rank.value() == 1
    window.saved = copy.deepcopy(window.exp)
    window.close()
    app.processEvents()
