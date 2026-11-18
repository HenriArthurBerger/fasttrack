"""The experiment file: defaults, round trips, older formats, plate selection, and raw-to-figures runs."""
import os

import pytest
import yaml

from fasttrack.experiment import (import_legacy, load_experiment, new_experiment, normalise, normalise_well, pipeline_config,
                                  same_pipeline, save_experiment, select_timestacks, validate)

LEGACY = """
batch: [0,1,2]
batch_name: 'All'
batch_dict: {'All': {'data': [0,1,2]}, 'DMI': {'data': [0,1]}}
code_to_condition: {0: "NS", 1: "DMI", 2: "ROSI"}
dmi_addition: 12
xcondition: 'H2B'
ycondition: 'PPARY'
zcondition: 'Geminin'
xmeasure: 'MeanNuc'
ymeasure: 'MeanNuc'
zmeasure: 'MeanNuc'
xthreshold: 10
ythreshold: 12.25
zthreshold: 8.5
plothist: True
data_directory: '/Volumes/x/tracked'
endframe: 350
code_to_well: {0: [202, 203], 1: [205], 2: [1012]}
"""


def test_saved_file_has_every_key_and_round_trips(tmp_path):
    exp = new_experiment(paths={"raw": "/data/raw"}, plate={"conditions": [{"name": "A", "wells": ["b2", "B3"]}]})
    save_experiment(exp, tmp_path / "e.yaml")
    text = yaml.safe_load((tmp_path / "e.yaml").read_text())
    assert set(text) == set(normalise({}))
    assert "max_travel" in text["pipeline"]["link"]          # defaults are written, not implied
    assert text["plate"]["conditions"][0]["wells"] == ["B02", "B03"]
    assert load_experiment(tmp_path / "e.yaml") == normalise(exp)


def test_old_notebook_settings_are_imported(tmp_path):
    (tmp_path / "old.yaml").write_text(LEGACY)
    exp = load_experiment(tmp_path / "old.yaml")
    conditions = {c["name"]: c["wells"] for c in exp["plate"]["conditions"]}
    assert conditions == {"NS": ["B02", "B03"], "DMI": ["B05"], "ROSI": ["J12"]}
    assert [g["conditions"] for g in exp["analysis"]["groups"]] == [["NS", "DMI", "ROSI"], ["NS", "DMI"]]
    assert exp["time"]["events"][0]["frame"] == 12
    assert [c["name"] for c in exp["channels"]] == ["H2B", "PPARY", "Geminin"]
    assert exp["channels"][0]["role"] == "nuclear"
    gate = exp["analysis"]["gates"][1]
    assert gate["threshold"] == 12.25 and gate["metric"]["channel"] == "PPARY" and gate["metric"]["log2"]
    assert exp["files"]["last_frame"] == 350
    assert "plothist" in exp["experiment"]["notes"]


def test_plate_tab_layouts_are_imported():
    exp = import_legacy({"condition_colors": {"Ctrl": "#cc3d3d"}, "condition_wells": {"Ctrl": ["B02", "C02"]}, "conditions": ["Ctrl"]})
    assert exp["plate"]["conditions"][0] == {"name": "Ctrl", "color": "#cc3d3d", "wells": ["B02", "C02"], "notes": ""}
    exp = import_legacy({"conditions": {"Ctrl": [255, 0, 0]}, "layout": {"Ctrl": ["B02"]}})
    assert exp["plate"]["conditions"][0]["color"] == "#ff0000"


def test_wells_outside_the_plate_format_are_an_error():
    exp = new_experiment(plate={"conditions": [{"name": "A", "wells": ["J12"]}]})
    assert exp["plate"]["format"] == 96
    assert any("J12 do not exist on a 96-well plate" in p for p in validate(exp))
    exp["plate"]["format"] = 384
    assert not any("plate" in p for p in validate(exp))


def test_wells():
    assert normalise_well("b2") == "B02" and normalise_well(705) == "G05" and normalise_well("H12") == "H12"
    with pytest.raises(ValueError):
        normalise_well("Z")


def test_plate_picks_the_timestacks():
    exp = new_experiment(plate={"conditions": [{"name": "A", "wells": ["B02", "B03"]}], "excluded_wells": ["B03"],
                                "excluded_positions": ["B02_xy2"]})
    ids = ["B02_xy1", "B02_xy2", "B03_xy1", "C04_xy1"]
    assert select_timestacks(exp, ids) == ["B02_xy1"]
    exp["plate"]["process"] = "all"
    assert select_timestacks(exp, ids) == ["B02_xy1", "C04_xy1"]


def test_pipeline_config_maps_channels_and_frames():
    exp = new_experiment(paths={"raw": "/r", "results": "/o"}, files={"first_frame": 5, "last_frame": 50, "stride": 2},
                         channels=[{"file_channel": 2, "name": "ERK", "role": "measure"},
                                   {"file_channel": 1, "name": "H2B", "role": "nuclear"},
                                   {"file_channel": 3, "name": "bf", "role": "skip"}])
    exp["pipeline"]["link"]["max_travel"] = 22.0
    cfg = pipeline_config(exp)
    assert cfg["data"]["channels"] == [2, 1] and cfg["nuclear_channel"] == 1
    assert cfg["data"]["channel_names"] == ["ERK", "H2B"]
    assert (cfg["data"]["first_frame"], cfg["data"]["last_frame"], cfg["data"]["stride"]) == (5, 50, 2)
    assert cfg["link"]["max_travel"] == 22.0 and cfg["data"]["root"] == "/r"


def test_validate_catches_mistakes(tmp_path):
    exp = new_experiment(paths={"raw": str(tmp_path), "results": str(tmp_path / "out")},
                         channels=[{"file_channel": 1, "name": "a", "role": "measure"}],
                         plate={"conditions": [{"name": "A", "wells": ["B02"]}, {"name": "B", "wells": ["B02"]}]},
                         analysis={"groups": [{"name": "g", "conditions": ["A", "nope"]}]})
    problems = " ".join(validate(exp))
    assert "results folder must be outside" in problems
    assert "exactly one channel as nuclear" in problems
    assert "B02 is in both" in problems
    assert "nope" not in problems  # figure settings never block tracking ...
    from fasttrack.experiment import warnings_for
    assert any("nope" in w for w in warnings_for(exp))  # ... they are warnings


def test_same_pipeline_ignores_figures_and_paths():
    a = new_experiment()
    b = new_experiment(paths={"results": "/elsewhere"}, figures={"dpi": 600})
    assert same_pipeline(a, b)
    b["pipeline"]["link"]["move_weight"] = 1.0
    assert not same_pipeline(a, b)


def test_frame_range_applies_before_stride(tmp_path):
    from fasttrack.io import layout
    folder = tmp_path / "WellB02"
    folder.mkdir()
    for t in range(10):
        (folder / f"B02_Time{t:05d}_xy1c1.tif").write_bytes(b"")
    exp = new_experiment(paths={"raw": str(tmp_path)}, files={"first_frame": 2, "last_frame": 9, "stride": 3})
    ts = layout.discover_timestacks(pipeline_config(exp))["B02_xy1"]
    assert ts["times"] == [2, 5, 8]


def test_cli_takes_only_the_experiment_file(capsys):
    from fasttrack.cli import main
    with pytest.raises(SystemExit) as stop:
        main(["run", "--root", "/x"])
    assert stop.value.code == 2
    assert "exactly one experiment file" in capsys.readouterr().out


@pytest.mark.gpu
def test_raw_to_figures_from_one_file(tmp_path, device):
    from fasttrack.analysis.figures import render_all
    from fasttrack.synthetic.write import write_experiment
    from fasttrack.track.run import run_experiment
    write_experiment(tmp_path / "raw", ["B02", "B03", "C02"], [1], {"frames": 30, "size": 256, "cells": 8, "stage_jumps": 0}, device=device)
    metric = {"measure": "cyto_median", "channel": "ERK", "over_measure": "nuc_median", "background_percentile": 5}
    exp = new_experiment(
        paths={"raw": str(tmp_path / "raw"), "results": str(tmp_path / "out")},
        channels=[{"file_channel": 1, "name": "H2B", "role": "nuclear"}, {"file_channel": 2, "name": "ERK"},
                  {"file_channel": 3, "name": "c3", "role": "skip"}],
        time={"frames_per_hour": 6, "events": [{"name": "drug", "frame": 10}]},
        plate={"conditions": [{"name": "NS", "wells": ["B02"]}, {"name": "DMI", "wells": ["C02"]}]},
        pipeline={"device": device, "batch_auto": False},
        analysis={"min_track_frames": 5, "gates": [{"name": "ERK high", "metric": metric, "threshold": 1.0}]},
        figures={"formats": ["png"], "list": [
            {"name": "ktr", "kind": "timecourse", "metric": metric, "time_zero_event": "drug"},
            {"name": "heat", "kind": "heatmap", "metric": {"measure": "nuc_median", "channel": "ERK", "log2": True}},
            {"name": "high", "kind": "percent_high", "gate": "ERK high"},
            {"name": "area", "kind": "distribution", "metric": {"measure": "nuc_area"}, "style": "violin"},
            {"name": "count", "kind": "cell_count"},
            {"name": "off", "kind": "cell_count", "enabled": False}]})
    summaries = run_experiment(exp)
    assert sorted(s["id"] for s in summaries) == ["B02_xy1", "C02_xy1"]   # B03 has no condition
    out = tmp_path / "out"
    assert load_experiment(out / "experiment.yaml") == normalise(exp)
    log = yaml.safe_load((out / "run_log.yaml").read_text())
    assert log[0]["timestacks"]["B02_xy1"]["max_travel_px"] > 0
    from fasttrack.io import store
    assert store.load_results(out, "B02_xy1")["channels"].shape[1] == 2  # the skipped channel was not read
    written = {p.name for p in render_all(exp)}
    assert {"ktr.png", "heat.png", "high.png", "area.png", "count.png", "ktr.csv"} <= written
    assert "off.png" not in written
    # Changing a tracking setting and running again keeps the old snapshot and warns.
    exp["pipeline"]["link"]["move_weight"] = 1.0
    events = []
    run_experiment(exp, events.append)
    assert any(e["kind"] == "warning" for e in events)
    assert len(list(out.glob("experiment.*.yaml"))) == 1


def test_gui_round_trips_an_experiment(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from fasttrack.gui.app import MainWindow
    app = QApplication.instance() or QApplication([])
    (tmp_path / "old.yaml").write_text(LEGACY)
    window = MainWindow()
    exp = load_experiment(tmp_path / "old.yaml")
    exp["figures"]["list"] = [{"name": "f", "kind": "percent_high", "gate": "PPARY high"}]
    window.load(exp)
    for i in range(window.tabs.count()):
        window.tabs.setCurrentIndex(i)
        app.processEvents()
    assert window.collect() == normalise(exp)
    window.path = tmp_path / "new.yaml"
    window.save()
    assert load_experiment(tmp_path / "new.yaml") == normalise(exp)
    # Regression: opening a second file must not carry the first one's groups across. Tabs that read
    # the experiment while loading (a checked quality result does) used to re-collect half-loaded tabs.
    other = new_experiment(plate={"conditions": [{"name": "Z", "wells": ["C03"]}]},
                           quality={"checked": True, "flagged": {"C03_xy1": "fiber"}},
                           analysis={"groups": [{"name": "zz", "conditions": ["Z"]}]})
    window.load(other)
    assert window.collect()["analysis"]["groups"] == [{"name": "zz", "conditions": ["Z"]}]
    window.load(exp)
    assert window.collect() == normalise(exp)
    window.saved = window.exp  # skip the unsaved-changes prompt
    window.close()
    app.processEvents()


@pytest.mark.gpu
def test_quality_check_flags_a_fiber_and_respects_keep(tmp_path, device):
    import numpy as np
    import tifffile
    from fasttrack.io import layout
    from fasttrack.quality import check_positions
    from fasttrack.synthetic.write import write_experiment
    write_experiment(tmp_path / "raw", ["B02"], [1, 2, 3, 4, 5, 6], {"frames": 12, "size": 256, "cells": 10, "stage_jumps": 0, "fibers": 0,
                                                                  "illum_sigma": (1.3, 1.3), "illum_offset": (0.05, 0.05)}, device=device)
    for path in sorted((tmp_path / "raw" / "WellB02").glob("B02_Time*_xy3c*.tif")):
        image = tifffile.imread(path)
        rr = np.arange(40, 220)
        for w in range(-3, 4):  # a dark curved fiber across the field
            image[rr, np.clip((100 + 40 * np.sin(rr / 30)).astype(int) + w, 0, 255)] = 0
        tifffile.imwrite(path, image)
    exp = new_experiment(paths={"raw": str(tmp_path / "raw"), "results": str(tmp_path / "out")},
                         plate={"conditions": [{"name": "A", "wells": ["B02"]}]}, pipeline={"device": device},
                         quality={"frames_per_position": 4, "tiles_per_edge": 8})
    timestacks = layout.discover_timestacks(pipeline_config(exp))
    results, features = check_positions(timestacks, exp["quality"], device)
    assert [k for k, r in results.items() if r["flagged"]] == ["B02_xy3"]
    assert "dark region" in results["B02_xy3"]["reason"]
    assert features["B02_xy1"]["thumb"].shape[-1] == 160
    exp["quality"].update(checked=True, flagged={"B02_xy3": "fiber"})
    assert "B02_xy3" not in select_timestacks(exp, list(timestacks))
    exp["quality"]["keep"] = ["B02_xy3"]
    assert "B02_xy3" in select_timestacks(exp, list(timestacks))
    exp["quality"]["enabled"] = False
    exp["quality"]["keep"] = []
    assert "B02_xy3" in select_timestacks(exp, list(timestacks))


def test_holm_and_reference_tests():
    import numpy as np
    from fasttrack.analysis.figures import compare_to_reference, holm
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.isnan(holm([float("nan"), 0.01])[0])
    rng = np.random.default_rng(0)
    ref, same, shifted = rng.normal(0, 1, 6), rng.normal(0, 1, 6), rng.normal(5, 1, 6)
    rows = compare_to_reference(["ref", "same", "shifted"], [ref, same, shifted], "position")
    assert rows[0]["test"] == "Welch t-test" and rows[0]["p"] > 0.05 and rows[1]["p_holm"] < 0.001
    single = compare_to_reference(["ref", "one"], [ref, np.array([1.0])], "well")
    assert np.isnan(single[0]["p"])  # one replicate: no test


def test_row_based_statistics_match_dense_numpy():
    import warnings
    import numpy as np
    from fasttrack.analysis import traces as tr
    from fasttrack.analysis.figures import Rows, _binned_kde, _heatmap_image, _per_cell_means, rows_to_matrix
    warnings.simplefilter("ignore", RuntimeWarning)
    rng = np.random.default_rng(0)
    m = rng.normal(size=(400, 40))
    m[rng.random(m.shape) < 0.5] = np.nan
    track, t = np.nonzero(~np.isnan(m))
    order = np.argsort(t, kind="stable")
    rows = Rows(t[order], track[order], m[track, t][order], 40)
    assert np.allclose(tr.grouped_quantiles(rows.t, rows.value, 40, [0.25, 0.5]), np.nanpercentile(m, [25, 50], axis=0), equal_nan=True)
    assert np.allclose(tr.nanquantiles(m, [0.1, 0.9]), np.nanpercentile(m, [10, 90], axis=0), equal_nan=True)
    assert np.allclose(tr.grouped_trailing_mean(track, t, m[track, t], 4), tr.moving_average(m, 4)[track, t])
    dense = rows_to_matrix(rows)
    present = ~np.all(np.isnan(m), axis=1)
    assert np.allclose(dense, m[present], equal_nan=True)
    assert np.allclose(_per_cell_means(rows), np.nanmean(m[present], axis=1))
    image, cells, group = _heatmap_image(rows, 100)
    assert cells == present.sum() and image.shape == (int(np.ceil(cells / group)), 40)
    from matplotlib import mlab
    data = rng.normal(size=5000)
    xs = np.linspace(-3, 3, 50)
    assert np.allclose(_binned_kde(data, xs), mlab.GaussianKDE(data, "scott").evaluate(xs), atol=1e-3)


def test_results_read_lazily_and_old_files_upgrade(tmp_path):
    import h5py
    import numpy as np
    from fasttrack.io import store
    from fasttrack.track.measure import CHANNELS, SCALARS, S
    N, C, T = 50, 2, 10
    scalars = np.random.default_rng(0).normal(size=(N, len(SCALARS))).astype(np.float32)
    scalars[:, S["t"]] = np.repeat(np.arange(T), 5)
    channels = np.random.default_rng(1).normal(size=(N, C, len(CHANNELS))).astype(np.float32)
    path = tmp_path / "Measurements" / "B02_xy1.h5"
    path.parent.mkdir()
    with h5py.File(path, "w") as f:  # the old block layout
        f.create_dataset("scalars", data=scalars)
        f.create_dataset("channels", data=channels)
        f["scalars"].attrs["names"] = np.array(SCALARS, dtype=h5py.string_dtype())
        f["channels"].attrs["names"] = np.array(CHANNELS, dtype=h5py.string_dtype())
        f.attrs["frames"] = T
    old = store.load_results(tmp_path, "B02_xy1")
    assert np.array_equal(old.column("nuc_area"), scalars[:, S["nuc_area"]])
    assert store.upgrade_measurements(path) and not store.upgrade_measurements(path)
    new = store.load_results(tmp_path, "B02_xy1")
    assert new.format == 2 and new["meta"]["frames"] == T
    assert np.array_equal(new.column("nuc_median", 1), channels[:, 1, CHANNELS.index("nuc_median")])
    assert np.array_equal(new["scalars"], scalars) and np.array_equal(new["channels"], channels)


def test_event_candidates_and_labels_file(tmp_path):
    import numpy as np
    from fasttrack import events
    # tracks: 0 spans all 6 frames; 1 ends at 3 (early); 2 starts at 2; 3 is a mother ending at 2 with daughters 4, 5
    index = -np.ones((6, 6), int)
    index[0, :] = 0
    index[1, :4] = 0
    index[2, 2:] = 0
    index[3, :3] = 0
    index[4, 3:] = 0
    index[5, 3:] = 0
    results = {"trace_index": index, "mother_track": np.array([-1, -1, -1, -1, 3, 3])}
    cand = events.candidates(results)
    assert cand["track_end"].tolist() == [[1, 3]]
    assert cand["division"].tolist() == [[3, 2]]
    assert sorted(cand["track_start"][:, 0].tolist()) == [2]   # daughters have a mother, so they are not 'starts'
    exp = new_experiment(paths={"results": str(tmp_path)})
    obs = {"position": "B02_xy1", "t": 3, "track": 1, "x": 10.0, "y": 12.0, "source": "track_end"}
    rows = events.add_label(exp, [], obs, "death")
    rows = events.add_label(exp, rows, obs, "neutral")   # relabelling replaces
    back = events.read_labels(exp)
    assert len(back) == 1 and back[0]["category"] == "neutral" and back[0]["t"] == 3


def test_crops_follow_the_track_and_keep_their_size():
    import numpy as np
    from fasttrack import events
    frames = {f: np.zeros((1, 64, 64), np.float32) for f in range(3)}
    for f in range(3):
        frames[f][0, 10 + 5 * f, 20] = 100.0   # a bright pixel moving down
    padded = events.pad_frames(frames, 16)
    stack = events.crop_stack(padded, lambda f: (10 + 5 * f, 20), 1, 1, 16, 3)
    assert stack.shape == (1, 3, 16, 16)
    assert all(stack[0, i, 8, 8] == 100.0 for i in range(3))  # centred in every frame
    edge = events.crop_stack(padded, lambda f: (0, 0), 0, 1, 16, 3)
    assert edge.shape == (1, 3, 16, 16)
    assert events.normalise_crop(stack).shape == (3, 16, 16)


@pytest.mark.gpu
def test_event_classifier_learns_separable_classes(device):
    import numpy as np
    from fasttrack import events
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (60, 3, 24, 24)).astype(np.float32)
    y = np.repeat([0, 1], 30)
    x[y == 1, :, 8:16, 8:16] += 3.0   # class 1: a bright centre
    net, report = events.train_classifier(x, y, ["a", "b"], steps=150, device=device)
    assert report["holdout_accuracy"] >= 0.9
    assert events.predict(net, x[:4], device).shape == (4, 2)


@pytest.mark.gpu
def test_partial_runs_warnings_and_failures_do_not_stop_the_plate(tmp_path, device, monkeypatch):
    from fasttrack.synthetic.write import write_experiment
    from fasttrack.track import run
    write_experiment(tmp_path / "raw", ["B02", "B03"], [1, 2], {"frames": 12, "size": 192, "cells": 4, "stage_jumps": 0}, device=device)
    exp = new_experiment(paths={"raw": str(tmp_path / "raw"), "results": str(tmp_path / "out")},
                         plate={"conditions": [{"name": "A", "wells": ["B02", "B03"]}]},
                         quality={"enabled": False, "min_cells_per_frame": 1000},     # everything counts as sparse
                         analysis={"groups": [{"name": "old", "conditions": ["renamed away"]}]},
                         pipeline={"device": device, "batch_auto": False})
    events = []
    done = run.run_experiment(exp, events.append, only=["B02_xy1", "B03_xy2"])
    assert sorted(s["id"] for s in done) == ["B02_xy1", "B03_xy2"]           # a partial run
    summary = (tmp_path / "out" / "Logs" / "latest_summary.txt").read_text()
    assert "renamed away" in summary and "B02_xy1: sparse" in summary       # warned, still ran
    assert "Speed:" in summary and "frames/s" in summary
    detailed = sorted((tmp_path / "out" / "Logs").glob("*_detailed.log"))[-1].read_text()
    assert "B02_xy1 frames 0-11" in detailed and "GPU waited" in detailed     # every batch, every timing
    assert done[0]["cells_per_frame"] > 0 and 0 < done[0]["nuclear_coverage"] < 1

    real = run.process_timestack

    def flaky(ts, *args, **kwargs):
        if ts["id"] == "B02_xy2":
            raise RuntimeError("unreadable frame")
        return real(ts, *args, **kwargs)

    monkeypatch.setattr(run, "process_timestack", flaky)
    events = []
    done = run.run_experiment(exp, events.append)
    assert [s["id"] for s in done] == ["B03_xy1"]                            # the others were finished before
    assert any(e["kind"] == "timestack_failed" and e["id"] == "B02_xy2" for e in events)
    summary = (tmp_path / "out" / "Logs" / "latest_summary.txt").read_text()
    assert "Failed (see the detailed log for the error) (1):" in summary and "B02_xy2" in summary
    detailed = sorted((tmp_path / "out" / "Logs").glob("*_detailed.log"))[-1].read_text()
    assert "B02_xy2: FAILED RuntimeError: unreadable frame" in detailed and "Traceback" in detailed


def test_population_heatmap_matches_the_notebook():
    import numpy as np
    """x time, y log2 value in bins of 0.1, colour the fraction of that frame's cells: every column sums to 1."""
    from fasttrack.analysis.figures import Rows, population_bins, population_image
    rng = np.random.default_rng(0)
    T, cells = 40, 300
    values = rng.normal(8 + np.linspace(0, 2, T)[None, :], 0.5, (cells, T))   # the population drifts up
    values[rng.random(values.shape) < 0.2] = np.nan                          # cells absent in some frames
    t, track = np.nonzero(np.isfinite(values.T))[0], np.nonzero(np.isfinite(values.T))[1]
    order = np.argsort(t, kind="stable")
    rows = Rows(t[order], track[order], values.T[np.isfinite(values.T)][order], T)
    edges = population_bins(rows.value, log2=True, width=0)
    assert np.allclose(np.diff(edges), 0.1)
    frac, counted, outside = population_image(rows, edges)
    assert frac.shape == (len(edges) - 1, T)
    assert np.allclose(np.nansum(frac, axis=0), 1)
    # the notebook: cumulative counts below each right edge, differenced, divided by the column sum
    lo = int(round(edges[0] * 10))
    cum = np.stack([np.nansum(values < (b + 1) / 10, axis=0) for b in range(lo - 1, lo + len(edges) - 1)], 1)
    hist = np.diff(cum, axis=1).T
    assert np.allclose(frac, hist / hist.sum(axis=0))
    assert (counted + outside == np.isfinite(values).sum(axis=0)).all()
    mean_bin = (frac * ((edges[:-1] + edges[1:]) / 2)[:, None]).sum(axis=0)
    assert mean_bin[-1] - mean_bin[0] > 1.5                                  # the drift shows
