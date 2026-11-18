"""Benchmark settings come from a YAML file, not command-line flags.

    python benchmarks/<name>.py                     # uses benchmarks/settings/<name>.yaml
    python benchmarks/<name>.py my_settings.yaml    # or another settings file

A settings file may name an experiment file (`experiment:`). The benchmark then reads the raw
data folder and every tracking parameter from it, so a benchmark runs exactly the analysis
that experiment would run. Relative paths are relative to the refactor folder. A missing
settings file is written with the defaults, so it can be filled in.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from fasttrack.config import make_config  # noqa: E402
from fasttrack.experiment import load_experiment, pipeline_config  # noqa: E402


def _path(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT / p).resolve()


def load_settings(script: str, defaults: dict) -> SimpleNamespace:
    """Returns the settings as attributes, plus `cfg` (tracking config), `raw` (data folder or None), `out` (Path)."""
    name = Path(script).stem
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "settings" / f"{name}.yaml"
    template = {"experiment": "", **defaults}
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Settings for benchmarks/{name}.py\n" + yaml.safe_dump(template, sort_keys=False))
        raise SystemExit(f"Wrote {path} with the defaults. Fill it in (at least `experiment:` if the benchmark needs real data) and run again.")
    with open(path) as f:
        given = yaml.safe_load(f) or {}
    unknown = set(given) - set(template)
    if unknown:
        raise SystemExit(f"{path}: unknown settings {sorted(unknown)}; known are {sorted(template)}")
    opts = {**template, **given}
    exp = load_experiment(_path(opts["experiment"])) if opts["experiment"] else None
    cfg = pipeline_config(exp) if exp else make_config()
    raw = _path(exp["paths"]["raw"]) if exp and exp["paths"]["raw"] else None
    out = _path(opts["out"]) if opts.get("out") else None
    if raw is not None and out is not None and (out == raw or raw in out.parents):
        raise SystemExit("`out` must be outside the raw data folder: raw data is only read")
    print(f"settings: {path}" + (f", experiment: {opts['experiment']}" if exp else ""), flush=True)
    return SimpleNamespace(**{**opts, "cfg": cfg, "raw": raw, "out": out, "experiment_file": exp})


def well_folders(args: SimpleNamespace) -> list[Path]:
    """Well folders under the raw data folder that the experiment's plate would process."""
    from fasttrack.experiment import select_timestacks
    folders = sorted(p for p in args.raw.iterdir() if p.is_dir() and p.name.lower().startswith("well"))
    if args.experiment_file is None:
        return folders
    return [p for p in folders if select_timestacks(args.experiment_file, [f"{p.name[4:].upper()}_xy0"])]
