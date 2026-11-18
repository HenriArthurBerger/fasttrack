"""Which torch device to run on, so one experiment file works on any machine.

  auto      the CUDA GPU with the most free memory, else the CPU
  cuda:N    that GPU; if this machine has no such GPU, the CPU (with one warning)
  mps       Apple-silicon GPU, when asked for by name (not every operation the pipeline uses is
            supported there, so `auto` does not pick it); the CPU if it is not available
  cpu       the CPU
"""
from __future__ import annotations

import warnings

import torch

_warned: set[str] = set()


def _fallback(name: str, why: str) -> str:
    if name not in _warned:
        _warned.add(name)
        warnings.warn(f"Device {name!r}: {why}; running on the CPU instead (slow).", RuntimeWarning, stacklevel=3)
    return "cpu"


def resolve(name: str | None = "auto") -> str:
    """A device name torch accepts on this machine."""
    name = str(name or "auto").strip().lower()
    if name == "auto":
        if not torch.cuda.is_available():
            return "cpu"
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        return f"cuda:{max(range(len(free)), key=free.__getitem__)}"
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            return _fallback(name, "no CUDA GPU on this machine")
        index = int(name.split(":")[1]) if ":" in name else 0
        if index >= torch.cuda.device_count():
            return _fallback(name, f"this machine has {torch.cuda.device_count()} CUDA GPU(s)")
        return f"cuda:{index}"
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else _fallback(name, "no Apple GPU available")
    return name


def describe() -> str:
    """One line for the About box and the CLI: torch version and what it can run on."""
    parts = [f"torch {torch.__version__}"]
    if torch.cuda.is_available():
        parts += [f"CUDA {torch.version.cuda}", ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))]
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        parts.append("Apple GPU (mps) available")
    else:
        parts.append("CPU only")
    return " · ".join(parts)
