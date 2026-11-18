"""Features (many measurements, normalised per experiment, pooled by name) and fate analyses
(committor, commitment points, features aligned to commitment or division)."""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest

from fasttrack.experiment import new_experiment

CH = ["H2B", "PPARG", "KTR"]


def _write(folder, key, tracks, T):
    """tracks: list of dicts {first, values: {(channel, measure): (L,) array} or {scalar: (L,)}, mother}."""
    from fasttrack.io import store
    from fasttrack.track.measure import CHANNELS, SCALARS
    S = {s: i for i, s in enumerate(SCALARS)}
    K = {c: i for i, c in enumerate(CHANNELS)}
    rows = sum(len(next(iter(tr["values"].values()))) for tr in tracks)
    scalars = np.full((rows, len(SCALARS)), np.nan, np.float32)
    channels = np.full((rows, len(CH), len(CHANNELS)), np.nan, np.float32)
    index = np.full((len(tracks), T), -1, np.int64)
    track_id = np.zeros(rows, np.int64)
    r = 0
    for i, tr in enumerate(tracks):
        L = len(next(iter(tr["values"].values())))
        sl = slice(r, r + L)
        t = tr["first"] + np.arange(L)
        scalars[sl, S["t"]] = t
        scalars[sl, S["x"]] = 40.0 + 20 * i
        scalars[sl, S["y"]] = 40.0 + 0.3 * t
        scalars[sl, S["touches_border"]] = 0
        scalars[sl, S["nuc_area"]] = 200
        scalars[sl, S["cyto_area"]] = 150
        channels[sl, :, K["nuc_std"]] = 10
        channels[sl, :, K["cyto_std"]] = 10
        for name, v in tr["values"].items():
            if isinstance(name, tuple):
                channels[sl, CH.index(name[0]), K[name[1]]] = v
            else:
                scalars[sl, S[name]] = v
        index[i, t] = np.arange(r, r + L)
        track_id[sl] = i
        r += L
    store.save_measurements(folder / "Measurements" / f"{key}.h5", scalars, channels, {"frames": T, "id": key})
    links = {"track_id": track_id, "mother_track": np.array([tr.get("mother", -1) for tr in tracks]), "drift": np.zeros((T, 2))}
    store.save_traces(folder / "Traces" / f"{key}.h5", links, index, {"frames": T})


def _cells(rng, n, T, dip: bool):
    """Cells whose PPARG switches from low (e^4) to high (e^5.5) at a random frame; with `dip`, ERK (KTR
    ln(C/N)) drops by 1 for the 2 h before the switch. Returns the track dicts and the switch frames."""
    out, taus = [], []
    t = np.arange(T)
    for _ in range(n):
        tau = int(rng.integers(20, 2 * T))   # about half switch within the movie
        log_p = 4 + 1.5 / (1 + np.exp(-(t - tau) / 1.5)) + 0.05 * rng.normal(size=T)
        ktr = 0.1 * rng.normal(size=T) - (1.0 * ((t >= tau - 20) & (t < tau)) if dip else 0)
        nuc = 400 * (1 + 0.01 * rng.normal(size=T))
        out.append({"first": 0, "values": {("H2B", "nuc_mean"): 1000 * (1 + 0.05 * rng.normal(size=T)),
                                           ("H2B", "nuc_sum"): 2e5 * (1 + 0.05 * rng.normal(size=T)),
                                           ("PPARG", "nuc_mean"): np.exp(log_p),
                                           ("KTR", "nuc_median"): nuc, ("KTR", "cyto_median"): nuc * np.exp(ktr)}})
        taus.append(tau)
    return out, np.array(taus)


def _experiment(tmp_path, T=160):
    rng = np.random.default_rng(1)
    taus = {}
    for key, dip in (("A01_xy1", True), ("A01_xy2", True), ("A01_xy3", True), ("B01_xy1", False), ("B01_xy2", False), ("B01_xy3", False)):
        cells, taus[key] = _cells(rng, 70, T, dip)
        _write(tmp_path / "res", key, cells, T)
    exp = new_experiment(
        paths={"raw": str(tmp_path / "raw"), "results": str(tmp_path / "res")},
        channels=[{"file_channel": 1, "name": "c1", "role": "nuclear", "reporter": "H2B"},
                  {"file_channel": 2, "name": "c2", "role": "measure", "reporter": "PPARG"},
                  {"file_channel": 3, "name": "c3", "role": "measure", "reporter": "ERKKTR"}],
        time={"frames_per_hour": 10},
        plate={"conditions": [{"name": "Ctrl", "wells": ["A01"]}, {"name": "siX", "wells": ["B01"]}]},
        analysis={"min_track_frames": 10,
                  "readouts": [{"name": "ERK", "kind": "ktr", "channel": "c3", "background_percentile": 0.0, "max_noise": 0}],
                  "fates": [{"name": "pparg", "marker": "PPARG nuc_mean", "split": "ERK", "split_hours": 1.0, "horizon_hours": 2.0,
                             "before_hours": 4.0, "after_hours": 2.0, "bootstrap": 50, "hold_frames": 8}]})
    return exp, taus


# ---------------------------------------------------------------------- features
def test_features_suggest_transform_and_lineage(tmp_path):
    from fasttrack.analysis import features as FT
    from fasttrack.io import store
    T = 30
    tracks = [{"first": 0, "values": {("PPARG", "nuc_mean"): np.full(15, np.e ** 2), "local_density": np.arange(15.0)}},
              {"first": 15, "mother": 0, "values": {("PPARG", "nuc_mean"): np.full(15, np.e ** 3), "local_density": np.zeros(15)}},
              {"first": 15, "mother": 0, "values": {("PPARG", "nuc_mean"): np.full(15, np.e ** 4), "local_density": np.zeros(15)}}]
    _write(tmp_path / "res", "A01_xy1", tracks, T)
    exp = new_experiment(paths={"raw": "", "results": str(tmp_path / "res")},
                         channels=[{"file_channel": 1, "name": "c1", "role": "nuclear", "reporter": "H2B"},
                                   {"file_channel": 2, "name": "c2", "role": "measure", "reporter": "PPARG"}],
                         plate={"conditions": [{"name": "Ctrl", "wells": ["A01"]}]}, analysis={"min_track_frames": 5})
    names = [f["name"] for f in FT.suggest(exp)]
    assert {"nuc_area", "speed", "age", "H2B nuc_sum", "PPARG nuc_mean", "PPARG nc_ratio"} <= set(names)
    res = store.load_results(exp["paths"]["results"], "A01_xy1")
    lin = FT.lineage(res)
    assert lin["divides"].tolist() == [True, False, False] and lin["born"].tolist() == [False, True, True]
    f = {"name": "p", "measure": "nuc_mean", "channel": "c2", "transform": "log"}
    t, track, value, se = FT.feature_rows(res, exp, f)
    assert np.allclose(value[track == 1], 3) and np.all(se > 0)                 # log, with the pixel error
    age = FT.feature_rows(res, exp, {"name": "age", "measure": "age", "transform": "none"})
    assert set(age[1].tolist()) == {1, 2} and age[2].max() == 14                # only cells born in the movie
    b = FT.position_block(res, exp, [f], slice(0, T), {"median": {"p": 3.0}, "scale": {"p": 0.5}})
    assert b["mother"].tolist() == [-1, 0, 0] and np.isclose(np.nanmax(b["y"][2]), 2.0)   # (4 - 3) / 0.5


def test_features_pool_across_experiments_by_name(tmp_path):
    from fasttrack.analysis import features as FT
    exp, _ = _experiment(tmp_path / "a")
    other = copy.deepcopy(exp)
    other["experiment"]["name"] = "geminin"
    other["channels"][2]["reporter"] = "Geminin"   # same channel, another reporter: different feature names
    other["analysis"]["readouts"] = []
    exp["experiment"]["name"] = "ktr"
    data = FT.load_features([exp, other], conditions=["Ctrl"], max_tracks=60)
    assert "ERK" in data.readouts and "Geminin nuc_mean" in data.readouts and "PPARG nuc_mean" in data.readouts
    e = data.rows["experiment"]
    erk, gem = data.readouts.index("ERK"), data.readouts.index("Geminin nuc_mean")
    assert np.isnan(data.y[e == 1][..., erk]).all() and np.isfinite(data.y[e == 0][..., erk]).any()
    assert np.isnan(data.y[e == 0][..., gem]).all()
    p = data.readouts.index("PPARG nuc_mean")
    ref = data.y[(e == 0)][..., p]
    assert abs(np.nanmedian(ref)) < 0.3                       # normalised to the reference condition
    sub = data.subset(np.arange(10))
    assert len(sub.rows["mother"]) == 10


# ---------------------------------------------------------------------- fate pieces
def test_threshold_crossing_and_logistic_pieces():
    from fasttrack.analysis.fate import find_commitments, logistic_fit, running_median, two_populations
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(4, 0.3, 3000), rng.normal(6, 0.3, 1500)])
    mix = two_populations(v)
    assert 4.6 < mix["threshold"] < 5.4 and mix["separation"] > 5
    m = np.full((3, 60), 4.0)
    m[0, 30:] = 6
    m[1, 30:33] = 6                                   # a blip: not a commitment
    m[2, :] = 6                                       # high from the start: no crossing seen
    m[0, 40] = np.nan
    assert find_commitments(running_median(m, 3), 5.0, 8).tolist() == [30, -1, -1]
    x = np.linspace(-3, 3, 30)
    n = np.full(30, 2000)
    k = np.round(n / (1 + np.exp(-2 * (x - 0.7)))).astype(int)
    x50, slope = logistic_fit(x, n, k)
    assert abs(x50 - 0.7) < 0.05 and abs(slope - 2) < 0.2


def test_fate_analysis_finds_commitment_and_what_precedes_it(tmp_path):
    from fasttrack.analysis.fate import fate_folder, read_fate, run_fate
    exp, taus = _experiment(tmp_path)
    report = run_fate(exp, exp["analysis"]["fates"][0])
    assert 4.4 < report["threshold"] < 5.2
    T, hold = 160, 8
    for c, wells in enumerate(("A01", "B01")):
        expected = sum(int(((v >= hold + 2) & (v <= T - hold - 2)).sum()) for k, v in taus.items() if k.startswith(wells))
        assert abs(report["events"][c] - expected) <= 0.15 * expected
    a = report["_arrays"]
    P = a["P"][:, 0]
    assert np.all(P[:, :3] < 0.3) and np.all(P[:, -3:] > 0.9)          # low now: rarely high later; high stays high
    assert np.all(np.isfinite(np.array(report["x50"], float)[:, 0]))
    erk = report["align_features"].index("ERK")
    pre = (a["rel_hours"] > -1.8) & (a["rel_hours"] < -0.2)
    diff = (a["ev_mean"] - a["ct_mean"])[:, pre, erk].mean(1)
    assert diff[0] < -2 and abs(diff[1]) < 1                           # the dip before commitment, only in Ctrl
    assert np.all(np.array(report["diff_ci"] if "diff_ci" in report else a["diff_ci"])[1, 0, pre, erk] < 0)
    # matching on more features and excluding crossings both run; exclusion by a feature that is never extreme drops nothing
    spec2 = dict(exp["analysis"]["fates"][0], name="pparg2", match_on=["H2B nuc_sum"], exclude_feature="H2B nuc_mean",
                 exclude_above=50.0, bootstrap=0)
    r2 = run_fate(exp, spec2, save=False)
    assert r2["excluded"] == [0, 0] and r2["events"] == report["events"]
    spec3 = dict(spec2, exclude_feature="PPARG nuc_mean", exclude_above=0.0)   # the marker is above 0 at every crossing
    assert run_fate(exp, spec3, save=False)["events"] == [0, 0]
    saved = read_fate(exp, "pparg")
    assert saved["events"] == report["events"] and saved["_arrays"]["P"].shape == a["P"].shape
    figs = sorted(p.name for p in (fate_folder(exp, "pparg") / "figures").glob("*.png"))
    assert figs == ["aligned.png", "aligned_heatmap.png", "commit_point.png", "committor.png", "threshold.png"]


def test_division_alignment_follows_daughters_and_cli(tmp_path):
    from fasttrack import cli
    from fasttrack.analysis.fate import run_fate
    from fasttrack.experiment import save_experiment
    rng = np.random.default_rng(2)
    T = 80
    tracks = []
    for i in range(30):
        m = len(tracks)
        tracks.append({"first": 0, "values": {("PPARG", "nuc_mean"): np.exp(4 + 0.05 * rng.normal(size=40)),
                                              ("H2B", "nuc_mean"): np.full(40, 1000.0), "nuc_area": np.full(40, 300.0)}})
        for _ in range(2):
            tracks.append({"first": 40, "mother": m, "values": {("PPARG", "nuc_mean"): np.exp(4 + 0.05 * rng.normal(size=40)),
                                                                 ("H2B", "nuc_mean"): np.full(40, 1000.0), "nuc_area": np.full(40, 150.0)}})
    _write(tmp_path / "res", "A01_xy1", tracks, T)
    _write(tmp_path / "res", "A01_xy2", tracks, T)
    exp = new_experiment(paths={"raw": "", "results": str(tmp_path / "res")}, time={"frames_per_hour": 10},
                         channels=[{"file_channel": 1, "name": "c1", "role": "nuclear", "reporter": "H2B"},
                                   {"file_channel": 2, "name": "c2", "role": "measure", "reporter": "PPARG"}],
                         plate={"conditions": [{"name": "Ctrl", "wells": ["A01"]}]}, analysis={"min_track_frames": 5})
    spec = {"name": "div", "marker": "PPARG nuc_mean", "align": "division", "threshold_auto": False, "threshold": 5.0,
            "align_features": ["nuc_area"], "before_hours": 2, "after_hours": 2, "bootstrap": 10}
    r = run_fate(exp, spec)
    assert r["events"] == [60]
    area = r["_arrays"]["ev_mean"][0, :, 0]
    B = r["before_frames"]
    assert area[B - 5] > area[B + 5] + 1                         # the nucleus halves at division: mother, then daughters
    exp["analysis"]["fates"] = [spec]
    path = tmp_path / "exp.yaml"
    save_experiment(exp, path)
    cli.main(["fates", str(path)])


def test_analysis_tab_has_features_and_fates(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from fasttrack.gui.app import MainWindow
    app = QApplication.instance() or QApplication([])
    exp, _ = _experiment(tmp_path)
    window = MainWindow()
    window.load(exp)
    a = window.analysis
    a.features._suggest()
    names = [f["name"] for f in a.features.collect()["list"]]
    assert "ERK" in names and "PPARG nuc_mean" in names
    a.tabs.setCurrentWidget(a.fates)
    fp = a.fates
    assert fp.marker.findData("PPARG nuc_mean") >= 0 and fp.split.findData("ERK") >= 0
    fp.list.list.setCurrentRow(0)
    assert fp.marker.currentData() == "PPARG nuc_mean" and fp.horizon.value() == 2.0
    fp.name.setText("pparg 2")
    fp.align.setCurrentIndex(fp.align.findData("division"))
    fp.list.add()
    out = window.collect()
    assert [f["name"] for f in out["analysis"]["fates"]] == ["pparg", "pparg 2"]
    assert out["analysis"]["fates"][1]["align"] == "division"
    assert len(out["analysis"]["features"]["list"]) == len(names)
    window.saved = copy.deepcopy(window.exp)
    window.close()
    app.processEvents()
