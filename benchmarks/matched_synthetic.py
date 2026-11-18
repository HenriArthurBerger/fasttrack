"""Lineage benchmark on synthetic data matched to a real timestack.

    python benchmarks/matched_synthetic.py [settings.yaml]   # default: benchmarks/settings/matched_synthetic.yaml

The settings name an experiment file (raw folder, tracking parameters), and either timestack
ids such as [G02_xy1] or a number of random ones.

For each chosen timestack:
  1. Track the real movie, and summarise it (density over time, shape, motion, intensities, channel dynamics).
  2. Fit generator settings to that summary and to raw frames (background, camera noise, stage steps).
  3. Generate a synthetic movie of the same length with full ground truth, track it, and summarise it.
  4. Compare the two summaries. Large differences mean the benchmark does not represent the real data.
  5. Score detection and lineage against ground truth, for each linker.

Real data is only read. Settings, comparison and scores go to `out`.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import load_settings, well_folders  # noqa: E402  (benchmarks/settings.py)
warnings.filterwarnings("ignore", message="index_reduce")

from fasttrack.config import merge
from fasttrack.io import layout
from fasttrack.synthetic.evaluate import evaluate_divisions, evaluate_tracking, match_detections
from fasttrack.synthetic.fit import compare, describe, fit_parameters, refine
from fasttrack.synthetic.generate import make_synth_config, render, simulate_cells
from fasttrack.track.linking import link_tracks
from fasttrack.track.measure import S
from fasttrack.track.run import process_timestack

LINKERS = {"multipass": {}, "notebook": {"link": {"method": "notebook"}}, "voting": {"link": {"method": "tracks"}}}


def track(ts: dict, cfg: dict, reader=None) -> dict:
    run_cfg = merge(cfg, {"output": {"save_measurements": False, "return_tables": True}})
    summary = process_timestack(ts, run_cfg, reader=reader)
    return summary


def score(tables: dict, truth: dict, scfg: dict, cfg: dict, radius: float) -> dict:
    sc, ch = tables["scalars"], tables["channels"]
    device = torch.device(cfg["device"])
    gate = max(scfg["nuc_radius"] * 1.5, 8.0)
    ev = match_detections(sc[:, S["t"]], sc[:, S["x"]], sc[:, S["y"]], truth, scfg["size"], gate)
    out = {"detection": {k: round(ev[k], 4) for k in ("precision", "recall", "f1")}, "linkers": {}}
    sct, cht = torch.as_tensor(sc, device=device), torch.as_tensor(ch, device=device)
    shift = torch.as_tensor(tables["frame_shift"], device=device)
    for name, update in LINKERS.items():
        with torch.inference_mode():
            links = link_tracks(sct, cht, radius, merge(cfg, update), shift)
        tr = evaluate_tracking(links["track_id"], sc[:, S["t"]], ev["det_cell"], truth, scfg["size"])
        dv = evaluate_divisions(links["track_id"], sc[:, S["t"]], ev["det_cell"], truth, links["mother_track"])
        out["linkers"][name] = {**{k: round(v, 4) if isinstance(v, float) else v for k, v in tr.items()},
                                **{k: round(v, 4) if isinstance(v, float) else v for k, v in dv.items()}}
    return out


def main() -> None:
    args = load_settings(__file__, {"out": "runs/matched", "timestacks": [], "random": 1, "frames": 0, "seed": None, "refine": 1})
    if args.raw is None:
        raise SystemExit("Name an experiment file with a raw data folder in the settings.")
    root, out = args.raw, args.out
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else random.randrange(10**6)
    rng = random.Random(seed)
    wells = well_folders(args) or [root]
    picks = []
    if args.timestacks:
        for key in args.timestacks:
            well = next(w for w in wells if w.name.lower().endswith(key.split("_xy")[0].lower()))
            picks.append((well, key))
    else:
        for well in rng.sample(wells, min(args.random or 1, len(wells))):
            picks.append((well, None))
    cfg = merge(args.cfg, {"data": {"max_frames": args.frames}})
    print(f"seed {seed}", flush=True)
    report = []
    for well, key in picks:
        wcfg = merge(cfg, {"data": {"root": str(well)}})
        stacks = layout.discover_timestacks(wcfg)
        key = key or rng.choice(sorted(stacks))
        ts = stacks[key]
        print(f"\n== {key}: {ts['frames']} frames", flush=True)

        t0 = time.perf_counter()
        real_run = track(ts, wcfg)
        real_tables = real_run["tables"]
        real = describe(real_tables["scalars"], real_tables["channels"], real_tables["links"], ts["frames"])
        times = np.linspace(0, ts["frames"] - 2, 6).astype(int)
        with ThreadPoolExecutor(16) as pool:
            pairs = np.stack([layout.load_batch(ts, t, t + 2, pool) for t in times])
        print(f"   real tracked in {time.perf_counter() - t0:.0f}s", flush=True)

        params = fit_parameters(real, pairs, pairs.shape[-2:], wcfg, seed=seed)
        for round_ in range(args.refine + 1):
            scfg = make_synth_config(params)
            t0 = time.perf_counter()
            truth = simulate_cells(scfg)
            frames = render(truth, scfg, device=wcfg["device"]).cpu()
            gen_seconds = time.perf_counter() - t0
            synth_ts = {"id": f"{key}_synthetic", "well": "S", "pos": 0, "frames": ts["frames"], "channels": {}}
            synth_run = track(synth_ts, wcfg, reader=lambda a, b: frames[a:b].numpy())
            synthetic = describe(synth_run["tables"]["scalars"], synth_run["tables"]["channels"], synth_run["tables"]["links"], ts["frames"])
            off = sum(1 for _, a, b, r in compare(real, synthetic) if np.isfinite(r) and abs(math.log(max(r, 1e-6))) > math.log(1.3) and max(abs(a), abs(b)) > 0.02)
            print(f"   fit round {round_}: generated in {gen_seconds:.0f}s, {(truth['t'] == 0).sum()} cells at frame 0, "
                  f"{(truth['t'] == ts['frames'] - 1).sum()} at the end; {off} statistics off by over 30%", flush=True)
            if round_ < args.refine:
                params = refine(params, real, synthetic)
        (out / f"{key}_synthetic.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(params, default=float)), sort_keys=False))
        rows = compare(real, synthetic)
        scores = score(synth_run["tables"], truth, scfg, wcfg, synth_run["radius"])

        print(f"   {'statistic':34s}{'real':>10s}{'synthetic':>11s}{'ratio':>8s}")
        for name, a, b, r in rows:
            flag = "  <-- off by over 30%" if np.isfinite(r) and abs(math.log(max(r, 1e-6))) > math.log(1.3) and max(abs(a), abs(b)) > 0.02 else ""
            print(f"   {name:34s}{a:10.3f}{b:11.3f}{r:8.2f}{flag}")
        print(f"   detection {scores['detection']}")
        for name, s in scores["linkers"].items():
            print(f"   {name:10s} true_link_recall {s['true_link_recall']:.3f}  link_accuracy {s['link_accuracy']:.3f}  "
                  f"tracks {s['tracks']} (true {s['true_cells']})  divisions P/R {s['division_precision']:.2f}/{s['division_recall']:.2f}", flush=True)
        report.append({"id": key, "seed": seed, "params": params, "real": real, "synthetic": synthetic, "comparison": rows, "scores": scores,
                       "real_seconds": real_run["seconds"], "synthetic_seconds": synth_run["seconds"]})
        (out / f"matched_{seed}.json").write_text(json.dumps(report, indent=1, default=float))
        del frames, real_tables, synth_run
        torch.cuda.empty_cache()
    print(f"\nreport: {out / f'matched_{seed}.json'}")


if __name__ == "__main__":
    main()
