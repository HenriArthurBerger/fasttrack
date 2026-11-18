"""End to end: synthetic movie through the pipeline, disk layout, storage, analysis and GUI."""
import copy
import os

import numpy as np
import pytest
import torch

from fasttrack.analysis import plate, traces
from fasttrack.config import make_config
from fasttrack.io import layout, store
from fasttrack.synthetic.evaluate import evaluate_tracking, match_detections
from fasttrack.synthetic.generate import generate
from fasttrack.synthetic.write import write_experiment
from fasttrack.track.batch import process_batch
from fasttrack.track.linking import link_tracks
from fasttrack.track.measure import S
from fasttrack.track.run import process_experiment, process_timestack


@pytest.fixture(scope="module")
def movie(device):
    return generate({"seed": 3, "frames": 40, "size": 512, "cells": 20}, device=device)


def test_accuracy_floor_on_calibrated_synthetic(movie, device):
    """Guards against regressions. Numbers are for the calibrated preset, not a real-data claim."""
    frames, truth, scfg = movie
    cfg = make_config({"device": device})
    result = process_batch(frames, cfg)
    links = link_tracks(result["scalars"], result["channels"], result["radius"], cfg, result["frame_shift"])
    sc = result["scalars"].cpu().numpy()
    ev = match_detections(sc[:, S["t"]], sc[:, S["x"]], sc[:, S["y"]], truth, scfg["size"], 9.0)
    tr = evaluate_tracking(links["track_id"], sc[:, S["t"]], ev["det_cell"], truth)
    assert ev["f1"] > 0.7
    assert tr["link_accuracy"] > 0.75


def test_runs_are_deterministic(movie, device):
    # Regression: the notebook's random blur sigma changed centroids from run to run.
    cfg = make_config({"device": device})
    a = process_batch(movie[0], cfg)
    b = process_batch(movie[0], cfg)
    assert torch.equal(a["signed"], b["signed"])  # the segmentation is exactly reproducible
    # Measurements differ only by the GPU's float32 summation order.
    assert torch.allclose(torch.nan_to_num(a["scalars"]), torch.nan_to_num(b["scalars"]), atol=1e-3)


def test_empty_field_does_not_crash(device):
    torch.manual_seed(0)
    raw = (700 + 10 * torch.randn(12, 2, 256, 256, device=device)).round()
    cfg = make_config({"device": device})
    result = process_batch(raw, cfg)
    link_tracks(result["scalars"], result["channels"], max(result["radius"], 4), cfg, result["frame_shift"])


def test_batch_ranges_fold_short_tails():
    from fasttrack.track.run import _next_range
    assert _next_range(0, 100, 101) == (0, 101)
    assert _next_range(100, 100, 250) == (100, 200)
    assert _next_range(200, 100, 250) == (200, 250)
    assert _next_range(0, 100, 50) == (0, 50)


def test_real_filename_pattern():
    import re
    from fasttrack.config import DEFAULTS
    g = re.search(DEFAULTS["data"]["pattern"], "A02_Time00017_xy3c2.tif").groupdict()
    assert g == {"well": "A02", "t": "00017", "pos": "3", "chan": "2"}


def test_disk_round_trip_and_resume(tmp_path, device):
    write_experiment(tmp_path / "raw", ["B03"], [1], {"frames": 50, "size": 256, "cells": 6, "stage_jumps": 0}, device=device)
    cfg = make_config({"device": device, "batch_size": 25, "batch_auto": False, "data": {"root": str(tmp_path / "raw"), "output": str(tmp_path / "out")}})
    events = []
    summaries = process_experiment(cfg, events.append)
    assert len(summaries) == 1 and summaries[0]["frames"] == 50
    assert sum(e["kind"] == "batch" for e in events) == 2
    assert events[-1]["kind"] == "experiment_done"
    results = store.load_results(tmp_path / "out", "B03_xy1")
    assert results["meta"]["frames"] == 50
    # frames and labels continue across the batch boundary
    t = results["scalars"][:, S["t"]]
    assert t.max() >= 25 and np.all(np.diff(t) >= 0)
    keys = results["scalars"][:, S["t"]] * 100000 + results["scalars"][:, S["label"]]
    assert np.unique(keys).size == keys.size  # labels are local to a frame
    assert results["trace_index"].shape[1] == 50
    assert store.metric_matrix(results, "nuc_median", 0).shape == results["trace_index"].shape
    assert process_experiment(cfg) == []  # a finished timestack is skipped


def test_stop_flag_stops_between_batches(tmp_path, device):
    import threading
    write_experiment(tmp_path / "raw", ["B03"], [1], {"frames": 20, "size": 128, "cells": 3, "stage_jumps": 0}, device=device)
    cfg = make_config({"device": device, "data": {"root": str(tmp_path / "raw"), "output": str(tmp_path / "out")}})
    stop = threading.Event()
    stop.set()
    assert process_experiment(cfg, stop=stop) == []
    assert not (tmp_path / "out" / "Traces" / "B03_xy1.h5").exists()


def test_gui_cache_holds_layers(device):
    cache = store.new_gui_cache("x")
    store.add_layer(cache, "label_map", torch.tensor([[[[0, -1], [2, 70000]]]], device=device), 0)
    store.add_layer(cache, "label_map", torch.tensor([[[[5, 5], [5, 5]]]], device=device), 1)
    store.flush_cache(cache)
    assert cache["layers"]["label_map"].shape == (2, 1, 2, 2)
    assert cache["layers"]["label_map"][0, 0, 1, 1] == 70000
    assert store.cache_size_bytes(cache) > 0
    store.close_gui_cache(cache)


def test_lineage_gives_daughters_their_history():
    m = np.full((3, 6), np.nan)
    m[0, :3] = [1, 2, 3]
    m[1, 3:] = [4, 5, 6]
    m[2, 3:] = [7, 8, 9]
    out = traces.lineage_matrix(m, np.array([-1, 0, 0]))
    assert out[1].tolist() == [1, 2, 3, 4, 5, 6]
    assert out[2].tolist() == [1, 2, 3, 7, 8, 9]


def test_commitment_distinguishes_early_from_never():
    a = np.zeros((3, 40))
    a[0, 20:] = 10
    a[1, :] = 10   # above from the start: commits at 0
    times, committed = traces.commitment_times(a, 5, hold_frames=10)
    assert times.tolist() == [20, 0, -1]
    assert committed.tolist() == [True, True, False]


def test_moving_average_ignores_gaps():
    a = np.array([[1.0, np.nan, 3.0, 5.0]])
    assert np.allclose(traces.moving_average(a, 2)[0], [1, 1, 3, 4])


def test_plate_layout_reads_both_formats(tmp_path):
    (tmp_path / "a.yaml").write_text("condition_wells:\n  Ctrl: [B02, C02]\n")
    (tmp_path / "b.json").write_text('{"layout": {"Ctrl": ["B02"]}}')
    assert plate.wells_by_condition(tmp_path / "a.yaml") == {"Ctrl": ["B02", "C02"]}
    assert plate.wells_by_condition(tmp_path / "b.json") == {"Ctrl": ["B02"]}


def test_gui_builds_and_edits_parameters():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from fasttrack.experiment import new_experiment
    from fasttrack.gui.app import MainWindow
    from fasttrack.gui.shell import STATUS
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert titles == ["Home", "Experiment", "Plate", "Quality", "Test run", "Parameters", "Label cells", "Run", "Viewer",
                      "Summary", "Figures", "Readouts", "Features", "Fates", "Dynamics", "Export"]
    # every page has a guide, and the guides render
    from fasttrack.gui.help_text import HELP, to_html
    assert {p.key for p in window.nav.pages} <= set(HELP)
    window.show_help("fates")
    assert "Commitment" in window.help_dialog.text.toPlainText() or "commit" in window.help_dialog.text.toPlainText()
    assert all("<table" in to_html(HELP[k]) for k in ("experiment", "run", "export"))
    # Next goes to the next required step, past optional pages
    assert window.nav.by_key["parameters"].next_key == "run" and window.nav.by_key["dynamics"].next_key == "export"
    # the sidebar says where each step stands
    window.load(new_experiment())
    assert window.nav.by_key["experiment"].item.data(STATUS) == "choose the raw data folder"
    window.help_dialog.close()
    # summary settings round-trip, and a copied summary figure lands in the figure list under a free name
    summ = window.analysis.summary
    assert summ.collect() == new_experiment()["figures"]["summary"]
    window.analysis.figures.add_figure({"name": "Cells per position", "kind": "cell_count"})
    window.analysis.figures.add_figure({"name": "Cells per position", "kind": "cell_count"})
    assert [f["name"] for f in window.collect()["figures"]["list"]][-2:] == ["Cells per position", "Cells per position (2)"]
    window.saved = copy.deepcopy(window.collect())     # nothing left unsaved when the window closes
    reader_keys = set(window.parameters.readers)
    assert ("link", "history") in reader_keys and ("data", "root") not in reader_keys
    params = window.parameters.collect()
    assert params["link"]["history"] == 5 and params["centroids"]["method"] == "notebook"
    window.close()
    app.processEvents()


def _reader(frames):
    return lambda a, b: frames[a:b].cpu().numpy()


def test_shifts_chain_across_batches_from_frame_0(device):
    # A stage jump inside the second batch must still count from frame 0.
    frames, truth, scfg = generate({"seed": 5, "frames": 90, "size": 512, "cells": 25, "stage_jumps": 0, "fibers": 0}, device=device)
    jumped = frames.cpu().numpy().copy()
    jumped[60:] = np.roll(jumped[60:], (20, -15), axis=(2, 3))
    jumped = torch.from_numpy(jumped)
    ts = {"id": "s", "well": "A01", "pos": 1, "frames": 90, "channels": {}}
    cfg = make_config({"device": device, "batch_size": 40, "batch_auto": False, "output": {"save_measurements": False, "return_tables": True}})
    summary = process_timestack(ts, cfg, reader=_reader(jumped))
    drift = summary["tables"]["links"]["drift"]
    assert summary["batch_sizes"] == [40, 50]
    assert np.allclose(drift[59], drift[0], atol=8)
    assert np.allclose(drift[-1] - drift[59], [-20, 15], atol=6)  # x_aligned = x + cumulative shift


def test_batching_matches_a_single_batch(device):
    frames, truth, scfg = generate({"seed": 6, "frames": 80, "size": 512, "cells": 20, "stage_jumps": 0}, device=device)
    ts = {"id": "s", "well": "A01", "pos": 1, "frames": 80, "channels": {}}
    outs = {"save_measurements": False, "return_tables": True}
    one = process_timestack(ts, make_config({"device": device, "batch_size": 80, "batch_auto": False, "output": outs}), reader=_reader(frames))
    two = process_timestack(ts, make_config({"device": device, "batch_size": 40, "batch_auto": False, "output": outs}), reader=_reader(frames))
    a, b = one["cells"], two["cells"]
    assert abs(a - b) / a < 0.05


def test_auto_batch_grows_when_memory_allows(device):
    if device == "cpu":
        pytest.skip("needs a GPU")
    frames, _, _ = generate({"seed": 7, "frames": 90, "size": 256, "cells": 5, "stage_jumps": 0}, device=device)
    ts = {"id": "s", "well": "A01", "pos": 1, "frames": 90, "channels": {}}
    cfg = make_config({"device": device, "batch_size": 25, "batch_auto": True, "prefetch_batches": 1, "output": {"save_measurements": False}})
    summary = process_timestack(ts, cfg, reader=_reader(frames))
    assert summary["batch_sizes"][0] == 25 and max(summary["batch_sizes"]) > 25
    assert sum(summary["batch_sizes"]) == 90


def test_out_of_memory_splits_the_batch(device, monkeypatch):
    from fasttrack.track import run
    frames, _, _ = generate({"seed": 8, "frames": 60, "size": 256, "cells": 5, "stage_jumps": 0}, device=device)
    real = run.process_batch

    def tight(tensor, *args, **kwargs):
        if tensor.shape[0] > 30:
            raise torch.OutOfMemoryError("simulated")
        return real(tensor, *args, **kwargs)

    monkeypatch.setattr(run, "process_batch", tight)
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    ts = {"id": "s", "well": "A01", "pos": 1, "frames": 60, "channels": {}}
    cfg = make_config({"device": device, "batch_size": 60, "batch_auto": False, "output": {"save_measurements": False}})
    summary = run.process_timestack(ts, cfg, reader=_reader(frames))
    assert summary["batch_sizes"] == [30, 30]


def test_loader_failure_reaches_the_caller(device):
    def broken(a, b):
        raise OSError("drive unplugged")

    ts = {"id": "s", "well": "A01", "pos": 1, "frames": 20, "channels": {}}
    with pytest.raises(OSError, match="drive unplugged"):
        process_timestack(ts, make_config({"device": device, "output": {"save_measurements": False}}), reader=broken)


def test_incomplete_positions_are_trimmed_and_reported(tmp_path, device):
    write_experiment(tmp_path, ["C02"], [1], {"frames": 25, "size": 64, "cells": 1, "stage_jumps": 0}, device=device)
    (tmp_path / "WellC02" / "C02_Time00024_xy1c2.tif").unlink()
    (tmp_path / "WellC02" / "C02_Time00010_xy1c1.tif").unlink()
    ts = layout.discover_timestacks(make_config({"data": {"root": str(tmp_path)}}))["C02_xy1"]
    # the gap stays in place as a blank frame; the position ends where its shortest channel ends
    assert ts["frames"] == 24 and ts["times"] == list(range(24))
    assert ts["channels"][1][10] is None and ts["missing"] == [{"frame": 10, "time": 10, "channels": [1]}]
    assert len(ts["problems"]) == 2
    assert all(len(v) == 24 for v in ts["channels"].values())


def test_test_mode_keeps_every_intermediate(device):
    frames, _, _ = generate({"seed": 9, "frames": 30, "size": 256, "cells": 8, "stage_jumps": 0}, device=device)
    cache = store.new_gui_cache("t")
    ts = {"id": "t", "well": "A01", "pos": 1, "frames": 30, "channels": {}}
    cfg = make_config({"device": device, "batch_size": 15, "batch_auto": False, "output": {"save_measurements": False}})
    process_timestack(ts, cfg, cache=cache, reader=_reader(frames))
    store.flush_cache(cache)
    expected = {"raw", "corrected", "rough_mask", "binary_mask", "distance", "centroids", "nuclear_labels", "label_map", "border_labels", "underseg_score"}
    assert set(cache["layers"]) == expected
    assert all(d.shape[0] == 30 for d in cache["layers"].values())
    assert cache["layers"]["raw"].shape[1] == 3
    assert cache["layers"]["label_map"][:].min() < 0  # the cytoplasm ring
    store.close_gui_cache(cache)


def test_inspect_colours_are_stable_and_the_ring_is_dim():
    pytest.importorskip("PySide6")
    from fasttrack.gui.inspect_tab import colour_labels
    a = colour_labels(np.array([[0, 7, -7, 8]]))
    assert a[0, 0].sum() == 0
    assert a[0, 1].sum() > a[0, 2].sum() > 0  # one cell's ring is a dimmer version of its colour
    assert (a[0, 1] != a[0, 3]).any()         # different cells differ
    assert (colour_labels(np.array([[7]]))[0, 0] == a[0, 1]).all()  # a label's colour never changes


def test_unreadable_files_are_blank_frames_not_crashes(tmp_path, device):
    """A truncated TIFF (PIL returns an empty array) and a 0-byte TIFF: reported, left blank, the run finishes."""
    from fasttrack.io.layout import UnreadableFrame, read_frame
    write_experiment(tmp_path / "raw", ["B03"], [1], {"frames": 12, "size": 128, "cells": 3, "stage_jumps": 0}, device=device)
    folder = tmp_path / "raw" / "WellB03"
    truncated = folder / "B03_Time00004_xy1c1.tif"
    truncated.write_bytes(truncated.read_bytes()[:200])
    (folder / "B03_Time00008_xy1c2.tif").write_bytes(b"")
    with pytest.raises(UnreadableFrame):
        read_frame(str(truncated))
    cfg = make_config({"device": device, "batch_auto": False, "data": {"root": str(tmp_path / "raw"), "output": str(tmp_path / "out")}})
    summaries = process_experiment(cfg)
    assert len(summaries) == 1 and summaries[0]["frames"] == 12
    assert sorted((u["frame"], u["channel"]) for u in summaries[0]["unreadable"]) == [(4, 0), (8, 1)]


def test_missing_files_keep_time_aligned_and_too_many_flag_the_position(tmp_path, device):
    """A gap stays a blank frame at its own time; a short movie ends early; more bad frames than the limit flags."""
    from fasttrack.experiment import new_experiment
    from fasttrack.io.layout import discover_timestacks, scan_report
    from fasttrack.track import run
    write_experiment(tmp_path / "raw", ["B02", "B03"], [1, 2], {"frames": 12, "size": 128, "cells": 3, "stage_jumps": 0, "fibers": 0}, device=device)
    raw = tmp_path / "raw"
    (raw / "WellB02" / "B02_Time00004_xy1c2.tif").unlink()             # one channel at one time
    for t in (2, 3, 5):
        for c in (1, 2, 3):
            (raw / "WellB02" / f"B02_Time{t:05d}_xy2c{c}.tif").unlink()   # three whole frames
    for c in (1, 2, 3):
        (raw / "WellB03" / f"B03_Time00011_xy1c{c}.tif").unlink()        # last frame
    exp = new_experiment(paths={"raw": str(raw), "results": str(tmp_path / "out")},
                         plate={"conditions": [{"name": "A", "wells": ["B02", "B03"]}]},
                         quality={"enabled": True, "checked": True}, pipeline={"device": device, "batch_auto": False})
    from fasttrack.experiment import pipeline_config
    ts = discover_timestacks(pipeline_config(exp))
    one = ts["B02_xy1"]
    assert one["frames"] == 12 and one["channels"][2][4] is None and one["channels"][1][4] is not None
    assert one["missing"] == [{"frame": 4, "time": 4, "channels": [2]}]
    assert ts["B02_xy2"]["times"] == list(range(12))                   # time stays aligned
    assert ts["B03_xy1"]["frames"] == 11 and ts["B03_xy1"]["short_by"] == 1
    report = scan_report(ts)
    assert report["usual_frames"] == 12 and report["short"] == {"B03_xy1": 1} and report["suggest_last"] == 11
    assert "frames 2-3, 5 (all channels)" in report["missing"]["B02_xy2"]

    events = []
    done = run.run_experiment(exp, events.append)
    assert sorted(s["id"] for s in done) == ["B02_xy1", "B03_xy1", "B03_xy2"]   # B02_xy2 is left out
    assert exp["quality"]["flagged"]["B02_xy2"].startswith("missing files: 3 frames missing")
    summary = (tmp_path / "out" / "Logs" / "latest_summary.txt").read_text()
    assert "B02_xy2: missing files" in summary and "Track anyway" in summary
    assert "set Frames to = 11" in summary                                 # suggested, not applied
    assert "B02_xy1 (" not in summary and "B02_xy1: missing files, read as blank frames: frame 4 (channel 2)" in summary

    import copy
    fresh = copy.deepcopy(exp)
    fresh["quality"].update(checked=False, flagged={}, keep=[])
    results, _ = run.run_quality_check(fresh, ts)                         # the image check also flags files
    assert results["B02_xy2"]["flagged"] and fresh["quality"]["flagged"]["B02_xy2"].startswith("missing files")
    assert not fresh["quality"]["flagged"].get("B02_xy1", "").startswith("missing files")  # one missing frame is forgiven

    exp["quality"]["keep"] = ["B02_xy2"]                                     # the user keeps it anyway
    done = run.run_experiment(exp, events.append)
    assert [s["id"] for s in done] == ["B02_xy2"]
    assert done[0]["frames"] == 12


def test_reads_are_retried_and_a_vanished_drive_fails_the_position(tmp_path, device, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from fasttrack.io import layout
    write_experiment(tmp_path / "raw", ["B02"], [1], {"frames": 4, "size": 64, "cells": 2, "stage_jumps": 0, "fibers": 0}, device=device)
    cfg = make_config({"device": device, "data": {"root": str(tmp_path / "raw"), "output": str(tmp_path / "out")}})
    ts = layout.discover_timestacks(cfg)["B02_xy1"]
    real, calls = layout.read_frame, {}

    def flaky(path, shape=None):
        calls[path] = calls.get(path, 0) + 1
        if path.endswith("Time00001_xy1c1.tif") and calls[path] == 1:
            raise layout.UnreadableFrame("busy disk")
        return real(path, shape)

    monkeypatch.setattr(layout, "read_frame", flaky)
    monkeypatch.setattr(layout, "RETRY_DELAYS", (0.01, 0.01))
    monkeypatch.setattr(layout.read_frame_retry, "__defaults__", (None, (0.01, 0.01)))
    with ThreadPoolExecutor(4) as pool:
        out = layout.load_batch(ts, 0, 4, pool)
        assert out[1, 0].any() and ts["unreadable"] == []
        assert [(r["frame"], r["tries"]) for r in ts["retried"]] == [(1, 2)]
        for f in (tmp_path / "raw" / "WellB02").glob("*.tif"):                # the drive goes away
            f.unlink()
        with pytest.raises(layout.DriveUnavailable, match="still connected"):
            layout.load_batch(ts, 0, 4, pool)


def test_figures_leave_out_wells_removed_from_conditions(tmp_path):
    """Editing the experiment file is enough: a well taken out of its condition, or excluded, is not drawn."""
    from fasttrack.analysis.figures import ResultsCache
    from fasttrack.experiment import new_experiment
    out = tmp_path / "out" / "Traces"
    out.mkdir(parents=True)
    for key in ("B02_xy1", "B02_xy2", "B03_xy1", "B04_xy1"):
        (out / f"{key}.h5").write_bytes(b"")
    exp = new_experiment(paths={"raw": str(tmp_path), "results": str(tmp_path / "out")},
                         plate={"conditions": [{"name": "A", "wells": ["B02", "B03"]}, {"name": "B", "wells": ["B04"]}],
                                "excluded_positions": ["B02_xy2"]})
    cache = ResultsCache(exp)
    assert cache.timestacks_of(exp["plate"]["conditions"][0]) == ["B02_xy1", "B03_xy1"]
    exp["plate"]["conditions"][0]["wells"] = ["B02"]                         # B03 removed from A
    assert cache.timestacks_of(exp["plate"]["conditions"][0]) == ["B02_xy1"]
    exp["plate"]["excluded_wells"] = ["B04"]
    assert cache.timestacks_of(exp["plate"]["conditions"][1]) == []
