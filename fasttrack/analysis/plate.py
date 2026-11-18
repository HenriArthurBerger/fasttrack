"""Group timestacks by condition from an experiment file (or an older plate layout JSON / YAML)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ..io.store import load_results


def wells_by_condition(layout_path: str | Path) -> dict[str, list[str]]:
    with open(layout_path) as f:
        layout = yaml.safe_load(f)
    # JSON is a subset of YAML, so one loader reads both formats.
    if "plate" in layout:  # an experiment file
        return {c["name"]: list(c["wells"]) for c in layout["plate"].get("conditions", [])}
    groups = layout.get("condition_wells") or layout.get("layout") or {}
    return {cond: list(wells) for cond, wells in groups.items()}


def timestacks_for_wells(output: str | Path, wells: list[str]) -> list[str]:
    found = sorted(p.stem for p in (Path(output) / "Traces").glob("*.h5"))
    return [name for name in found if name.split("_xy")[0] in wells]


def condition_matrix(output: str | Path, wells: list[str], metric_fn) -> np.ndarray:
    """Stack `metric_fn(results)` over the timestacks of some wells. Pads to the longest movie."""
    matrices = [metric_fn(load_results(output, name)) for name in timestacks_for_wells(output, wells)]
    if not matrices:
        return np.zeros((0, 0))
    T = max(m.shape[1] for m in matrices)
    padded = [np.pad(m, ((0, 0), (0, T - m.shape[1])), constant_values=np.nan) for m in matrices]
    return np.vstack(padded)
