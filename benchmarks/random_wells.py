"""Run the full pipeline on random wells of a real experiment, read-only.

    python benchmarks/random_wells.py [settings.yaml]     # default: benchmarks/settings/random_wells.yaml

Picks wells at random, processes every frame, and reports speed and descriptive numbers. Real data
has no ground truth, so nothing here measures lineage accuracy. For that, use synthetic data matched
to these wells (benchmarks/matched_synthetic.py) or human-annotated movies.

  tracks_per_cell   tracks / median cells per frame
  in_long_tracks    share of detections in tracks of 30+ frames
  divisions         count, and per 100 cells per 100 frames

The raw folder and tracking parameters come from the experiment file named in the settings.
Results go to `out` and never to the raw data folder.
"""
from __future__ import annotations

import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import load_settings, well_folders  # noqa: E402  (benchmarks/settings.py)
warnings.filterwarnings("ignore", message="index_reduce")

from fasttrack.config import merge
from fasttrack.io import layout
from fasttrack.track.linking import link_tracks, trace_index
from fasttrack.track.measure import S
from fasttrack.track.run import process_timestack

LINKERS = {"multipass": {}, "notebook": {"link": {"method": "notebook"}}, "voting": {"link": {"method": "tracks"}}}


def quality(scalars, channels, links, frames, nuclear=0) -> dict:
    t = scalars[:, S["t"]].astype(int)
    per_frame = np.bincount(t, minlength=frames)
    index = trace_index(links["track_id"], scalars[:, S["t"]], frames)
    lengths = (index >= 0).sum(1)
    cells = float(np.median(per_frame))
    return {
        "cells_per_frame": cells, "tracks": int(links["tracks"]), "tracks_per_cell": round(links["tracks"] / max(cells, 1), 2),
        "in_long_tracks": round(float(lengths[lengths >= 30].sum() / max(lengths.sum(), 1)), 3),
        "divisions": int(links["divisions"]),
        "divisions_per_100cells_100frames": round(100 * 100 * links["divisions"] / max(per_frame.sum(), 1), 2),
    }


def main() -> None:
    args = load_settings(__file__, {"out": "runs/random_wells", "wells": 3, "positions": 1, "frames": 0, "seed": None})
    if args.raw is None:
        raise SystemExit("Name an experiment file with a raw data folder in the settings.")
    out = args.out
    seed = args.seed if args.seed is not None else random.randrange(10**6)
    rng = random.Random(seed)
    wells = well_folders(args)
    chosen = sorted(rng.sample(wells, min(args.wells, len(wells))))
    print(f"seed {seed}: {len(wells)} wells, chose {[w.name for w in chosen]}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for well in chosen:
        cfg = merge(args.cfg, {"data": {"root": str(well), "max_frames": args.frames},
                               "output": {"save_measurements": False, "return_tables": True}})
        t0 = time.perf_counter()
        stacks = layout.discover_timestacks(cfg)
        scan = time.perf_counter() - t0
        for key in rng.sample(sorted(stacks), min(args.positions, len(stacks))):
            ts = stacks[key]
            print(f"\n{key}: {ts['frames']} frames, found in {scan:.1f}s. {ts['problems'] or ''}", flush=True)

            def on_event(e):
                if e["kind"] == "batch":
                    print(f"  {e['frames_done']}/{e['frames_total']}  {e['cells']} cells  gpu {sum(e['timings'].values()):.1f}s", flush=True)

            summary = process_timestack(ts, cfg, on_event)
            tables = summary.pop("tables")
            row = {"id": key, "seed": seed, **{k: summary[k] for k in ("frames", "radius", "max_travel", "seconds", "read_seconds", "wait_io", "batch_sizes", "peak_gpu_gb")},
                   "gpu_seconds": round(sum(v for k, v in summary["timings"].items() if k != "link"), 1), "link_seconds": round(summary["timings"]["link"], 1),
                   "timings": {k: round(v, 1) for k, v in summary["timings"].items()}, "problems": ts["problems"], "linkers": {}}
            drift = tables["links"]["drift"]
            row["stage_steps"] = [(int(i) + 1, drift[i + 1, 0].item() - drift[i, 0].item(), drift[i + 1, 1].item() - drift[i, 1].item())
                                  for i in np.nonzero(np.abs(np.diff(drift, axis=0)).sum(1))[0]][:40]
            device = torch.device(cfg["device"])
            sc, ch = torch.as_tensor(tables["scalars"], device=device), torch.as_tensor(tables["channels"], device=device)
            shift = torch.as_tensor(tables["frame_shift"], device=device)
            for name, update in LINKERS.items():
                link_cfg = merge(cfg, update)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    links = link_tracks(sc, ch, summary["radius"], link_cfg, shift)
                row["linkers"][name] = {"seconds": round(time.perf_counter() - t0, 1), **quality(tables["scalars"], tables["channels"], links, ts["frames"])}
                print(f"  {name:10s}", row["linkers"][name], flush=True)
            print(f"  total {summary['seconds']:.0f}s: read {summary['read_seconds']:.0f}s, gpu {row['gpu_seconds']}s, batches {summary['batch_sizes']}, peak {summary['peak_gpu_gb']:.1f} GB, gate {summary['max_travel']:.1f} px, {len(row['stage_steps'])} stage steps", flush=True)
            report.append(row)
            (out / f"report_{seed}.json").write_text(json.dumps(report, indent=1, default=float))
            del sc, ch, tables
            torch.cuda.empty_cache()
    print(f"\nreport: {out / f'report_{seed}.json'}")


if __name__ == "__main__":
    main()
