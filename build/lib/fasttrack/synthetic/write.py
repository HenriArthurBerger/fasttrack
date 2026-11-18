"""Write a synthetic timestack to disk in the same layout as real data, for pipeline tests."""
from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tifffile

from .generate import make_synth_config, render, simulate_cells


def write_experiment(root: str | Path, wells=("A02",), positions=(1,), updates: dict | None = None, device="cuda:0") -> None:
    """Writes <root>/Well<well>/<well>_Time<t>_xy<pos>c<chan>.tif and truth_<well>_xy<pos>.pkl."""
    root = Path(root)
    seed = (updates or {}).get("seed", 0)
    with ThreadPoolExecutor(16) as pool:
        for w, well in enumerate(wells):
            folder = root / f"Well{well}"
            folder.mkdir(parents=True, exist_ok=True)
            for pos in positions:
                cfg = make_synth_config({**(updates or {}), "seed": seed + 1000 * w + pos})
                truth = simulate_cells(cfg)
                frames = render(truth, cfg, device=device).cpu().numpy()
                jobs = [(folder / f"{well}_Time{t:05d}_xy{pos}c{c + 1}.tif", frames[t, c])
                        for t in range(frames.shape[0]) for c in range(frames.shape[1])]
                list(pool.map(lambda j: tifffile.imwrite(j[0], j[1]), jobs))
                with open(root / f"truth_{well}_xy{pos}.pkl", "wb") as f:
                    pickle.dump({"truth": truth, "config": cfg}, f)
