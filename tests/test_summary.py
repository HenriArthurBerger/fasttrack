"""Error bands, primary measurements and the summary figures drawn after every run."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fasttrack.analysis import bands as B
from fasttrack.analysis import summary as S


def test_bands_compute_what_they_say():
    stack = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, np.nan]])     # 3 replicates, 2 frames; one missing
    mean, lo, hi = B.across(stack, "sd")
    assert np.allclose(mean, [2.0, 3.0]) and np.allclose(hi - mean, [1.0, np.sqrt(2)])
    mean, lo, hi = B.across(stack, "sem")
    assert np.allclose(hi[0] - mean[0], 1 / np.sqrt(3))
    mean, lo, hi = B.across(stack, "ci95")
    assert np.isclose(hi[0] - mean[0], 4.302653 / np.sqrt(3), atol=1e-5)      # t(0.975, 2 df)
    assert np.isclose(hi[1] - mean[1], 12.706205 * np.sqrt(2) / np.sqrt(2), atol=1e-4)
    med, lo, hi = B.across(np.arange(1, 6, dtype=float)[:, None], "iqr")
    assert (med[0], lo[0], hi[0]) == (3.0, 2.0, 4.0)
    centre, lo, hi = B.across(stack, "none")
    assert lo is None and np.allclose(centre, [2.0, 3.0])
    # one replicate: no band rather than a band of zero width
    _, lo, hi = B.across(np.array([[5.0]]), "sem")
    assert np.isnan(lo[0]) and np.isnan(hi[0])
    lo, hi = B.proportion(np.array([30.0]), np.array([100.0]), "ci95")     # Wilson: 21.9% to 39.6%
    assert np.isclose(lo[0], 21.94, atol=0.05) and np.isclose(hi[0], 39.62, atol=0.05)
    assert B.resolve("auto", "cell") == "iqr" and B.resolve("auto", "well") == "sem"
    with pytest.raises(ValueError):
        B.resolve("stdev", "well")


def test_primary_measurement_follows_the_reporter():
    assert S.resolve_primary({"reporter": "ERK KTR"}) == "cyto_nuc"
    assert S.resolve_primary({"reporter": "H2B-mTurq"}) == "nuc_sum"
    assert S.resolve_primary({"reporter": "PPARG"}) == "nuc_mean"
    assert S.resolve_primary({"reporter": "PPARG", "primary": "nuc_median"}) == "nuc_median"
    from fasttrack.experiment import new_experiment
    exp = new_experiment(channels=[{"file_channel": 1, "name": "H2B", "role": "nuclear"},
                                   {"file_channel": 2, "name": "erk", "reporter": "ERKKTR"},
                                   {"file_channel": 3, "name": "x", "primary": "none"}])
    m = S.primary_metric(exp, "erk")
    assert (m["measure"], m["over_measure"], m["over_channel"], m["background_percentile"], m["log2"]) == \
        ("cyto_median", "nuc_median", "erk", 5.0, True)
    assert S.primary_metric(exp, "x") is None and S.primary_metric(exp, "missing") is None
    assert exp["channels"][0]["primary"] == "auto"         # every file gets the key


def test_figures_draw_every_band_and_summary_figures_are_written(tmp_path):
    from test_export import _exp
    from fasttrack.analysis.figures import ResultsCache, render, render_all
    exp = _exp(tmp_path)
    cache = ResultsCache(exp)
    base = exp["figures"]["list"][0]
    for unit in ("cell", "position"):
        for band in B.BANDS:
            spec = dict(copy.deepcopy(base), unit=unit, band=band)
            fig, table = render(exp, spec, cache=cache)
            b = B.resolve(band, unit)
            centre, low, high = B.table_columns(b)
            assert f"Ctrl {centre}" in table
            assert (f"Ctrl {low}" in table) == (b != "none")
            notes = " ".join(fig._fasttrack_notes)
            assert ("treat cells as independent" in notes) == (unit == "cell" and b in ("sem", "ci95"))
            assert B.describe(b, "cells" if unit == "cell" else "position medians") in notes
    for kind in ("cell_count", "distribution"):
        spec = dict(copy.deepcopy(base), kind=kind, style="bar", band="sd", unit="position")
        fig, table = render(exp, spec, cache=cache)
        assert table
    # a negative start counts from the end of the movie
    spec = dict(copy.deepcopy(exp["figures"]["list"][1]), frames=[-10, 0])
    fig, _ = render(exp, spec, cache=cache)
    assert "hours 5 to 5.9" in " ".join(fig._fasttrack_notes)     # frames 50 to 59 at 10 frames per hour

    specs = S.summary_specs(exp)
    names = [s["name"] for s in specs]
    assert names[:2] == ["Cells per position", "Nuclear area"]
    assert "ERKKTR (c3): cyto / nuc ratio" not in names and "ERKKTR (c3): cytoplasm / nucleus" in names
    assert "PPARG (c2): nuclear mean, last 20 frames" in names
    exp["figures"]["summary"]["kinds"] = ["cell_count", "timecourse"]
    assert len(S.summary_specs(exp)) == 1 + 3              # counts, and one time course per channel
    exp["figures"]["summary"]["kinds"] = []
    written = render_all(exp)
    summary = Path(exp["paths"]["results"]) / "Figures" / "Summary"
    assert (summary / "Cells_per_position.png").exists() and (summary / "Cells_per_position.csv").exists()
    assert len([p for p in written if p.parent == summary and p.suffix == ".csv"]) == len(specs)
    exp["figures"]["summary"]["enabled"] = False
    assert S.render_summary(exp) == []


def test_metric_cache_matches_the_files_and_keeps_recent_metrics(tmp_path, monkeypatch):
    import os
    import time
    from test_export import _exp
    from fasttrack.analysis import columns as C
    from fasttrack.analysis.figures import ResultsCache, metric_rows, position_rows, render, selected_keys, warm
    exp = _exp(tmp_path)
    cache = ResultsCache(exp)
    store = cache.columns
    keys = selected_keys(exp, cache)
    plan = store.plan(exp)
    assert 0 < len(plan) <= C.AUTO
    assert warm(exp, cache) == len(plan)
    assert not (Path(exp["paths"]["results"]) / "Cache").exists()          # RAM only
    for metric in (S.primary_metric(exp, "c2"), {"measure": "nuc_sum", "channel": "c1"}):
        for key in keys:      # the cache gives the files' values (stored as float32)
            a = position_rows(cache, key, exp, metric) if C.metric_key(metric) in store.columns else None
            b = metric_rows(cache.get(key), exp, metric)
            if a is None:
                continue
            assert np.array_equal(a.t, b.t) and np.array_equal(a.track, b.track)
            assert np.allclose(a.value, b.value, rtol=1e-6)
    # a retracked position is recomputed, the others kept; a new track filter rebuilds everything
    path = Path(exp["paths"]["results"]) / "Measurements" / f"{keys[0]}.h5"
    os.utime(path, ns=(time.time_ns(), time.time_ns()))
    assert warm(exp, cache) == len(plan) and warm(exp, cache) == 0
    exp["analysis"]["min_track_frames"] += 1
    assert warm(exp, cache) == len(plan)
    # metrics drawn beyond the automatic ones are cached too, least recently used out
    monkeypatch.setattr(C, "EXTRA", 1)
    metric = S.primary_metric(exp, "c2")
    extra = [dict(metric, smooth=3), dict(metric, smooth=5)]
    for m in extra:
        render(exp, dict(exp["figures"]["list"][0], metric=m), cache=cache)
        assert C.metric_key(m) in store.wanted
        warm(exp, cache)
    assert C.metric_key(extra[1]) in store.columns and C.metric_key(extra[0]) not in store.columns
