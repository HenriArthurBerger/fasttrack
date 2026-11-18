"""End-to-end accuracy and speed against synthetic ground truth.

    python benchmarks/validate.py [settings.yaml]   # default: benchmarks/settings/validate.yaml

preset: mosaic (calibrated to the sparse example well), populated, or dense (beyond the design range).
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import load_settings  # noqa: E402  (benchmarks/settings.py)
warnings.filterwarnings("ignore", message="index_reduce")

from fasttrack.config import make_config
from fasttrack.synthetic.evaluate import evaluate_divisions, evaluate_tracking, instance_iou, match_detections
from fasttrack.synthetic.generate import DENSE_PRESET, POPULATED_PRESET, generate
from fasttrack.track.batch import process_batch
from fasttrack.track.linking import link_tracks
from fasttrack.track.measure import K, S


def run(synth: dict, overrides: dict, device="cuda:0", iou=True) -> dict:
    cached = synth.get("_cache")
    if cached is None:
        params = {k: v for k, v in synth.items() if not k.startswith("_")}
        cached = generate(params, device=device, return_labels=True)
    frames, truth_labels, truth, scfg = cached
    cfg = make_config({"device": device, **overrides})
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    result = process_batch(frames, cfg)
    t_batch = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    t0 = time.perf_counter()
    links = link_tracks(result["scalars"], result["channels"], result["radius"], cfg, result["frame_shift"])
    t_link = time.perf_counter() - t0

    sc = result["scalars"].cpu().numpy()
    ch = result["channels"].cpu().numpy()
    signed = result["signed"].cpu().numpy()
    gate = max(scfg["nuc_radius"] * 1.5, 8.0)
    ev = match_detections(sc[:, S["t"]], sc[:, S["x"]], sc[:, S["y"]], truth, scfg["size"], gate, signed, sc[:, S["label"]])
    tr = evaluate_tracking(links["track_id"], sc[:, S["t"]], ev["det_cell"], truth, scfg["size"])
    dv = evaluate_divisions(links["track_id"], sc[:, S["t"]], ev["det_cell"], truth, links["mother_track"])
    jumps = truth["stage"]
    # x_aligned = x + cumulative shift, so a perfect estimate is the negative of the stage path.
    out_travel = links.get("max_travel", 0.0)
    drift_err = float(np.abs(links["drift"] + (jumps - jumps[0])).max())
    out = {
        "batch_s": round(t_batch, 2), "link_s": round(t_link, 2), "peak_gb": round(peak, 1), "radius": round(result["radius"], 1),
        "stages": {k: round(v, 2) for k, v in result["timings"].items()}, "stats": result["stats"],
        "precision": round(ev["precision"], 4), "recall": round(ev["recall"], 4), "f1": round(ev["f1"], 4),
        "merged_detections": ev["merged_detections"], "gt_total": ev["gt_total"], "gt_in_merge": ev["gt_in_merge"],
        "drift_max_error_px": round(drift_err, 1), "max_travel": round(out_travel, 1),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in tr.items()},
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in dv.items()},
    }
    if iou:
        out["iou"] = round(instance_iou(signed, truth_labels.cpu().numpy(), frames=range(0, scfg["frames"], 10)), 4)
    matched = (ev["det_cell"] >= 0) & ~ev["merged_rows"]
    if matched.any() and scfg["channels"] > 1:
        lookup = dict(zip(((truth["t"].astype(np.int64) << 32) + truth["cell"]).tolist(), truth["cn"][:, -1].tolist()))
        gt = np.array([lookup[(int(a) << 32) + int(b)] for a, b in zip(sc[matched, S["t"]], ev["det_cell"][matched])])
        est = ch[matched, -1, K["cyto_median"]] / ch[matched, -1, K["nuc_median"]]
        ok = np.isfinite(est) & (est > 0)
        out["cn_log_corr"] = round(float(np.corrcoef(np.log(gt[ok]), np.log(est[ok]))[0, 1]), 3)
    return out


def main():
    args = load_settings(__file__, {"preset": "mosaic", "seeds": 3, "frames": 100, "size": 1024})
    args.dense, args.populated = args.preset == "dense", args.preset == "populated"
    variants = {
        "multipass": {},
        "multipass,no_seed_merge": {"centroids": {"seed_overlap": 0.0}},
        "multipass,no_split": {"underseg": {"enabled": False}},
        "notebook": {"link": {"method": "notebook"}},
        "voting": {"link": {"method": "tracks"}},
    }
    for seed in range(args.seeds):
        synth = {"seed": seed, "frames": args.frames, **({"size": args.size} if not args.populated else {}), **(DENSE_PRESET if args.dense else {}), **(POPULATED_PRESET if args.populated else {})}
        synth["_cache"] = generate({k: v for k, v in synth.items() if k != "_cache"}, return_labels=True)
        for name, overrides in variants.items():
            print(json.dumps({"seed": seed, "variant": name, **run(synth, overrides)}))
        del synth["_cache"]


if __name__ == "__main__":
    main()
