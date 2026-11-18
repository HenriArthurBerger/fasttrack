"""Results on disk: one HDF5 per timestack, and the in-memory intermediates behind the GUI.

    <output>/Experiment_Config.yaml
    <output>/Measurements/<id>.h5   scalars/<name> (N,) and channels/<name> (N,C), one dataset per measurement
    <output>/Traces/<id>.h5         track_id (N), trace_index (tracks,T), mother_track, drift

trace_index holds measurement rows, so any metric becomes a (tracks, T) matrix by indexing
and nothing is stored twice.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

from ..track.measure import CHANNELS, SCALARS


def _names(values: list[str]) -> np.ndarray:
    return np.array(values, dtype=h5py.string_dtype())


FORMAT = 2  # 1: (N,S) and (N,C,K) blocks. 2: one dataset per measurement, so a figure reads only what it needs.


def save_measurements(path: Path, scalars: np.ndarray, channels: np.ndarray, meta: dict) -> None:
    """One dataset per measurement: scalars/<name> (N,) and channels/<name> (N, C)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        _write_columns(f, scalars, channels, SCALARS, CHANNELS)
        for key, value in meta.items():
            f.attrs[key] = value
    tmp.replace(path)  # a stopped run can never leave a half-written result under the final name


def _write_columns(f, scalars: np.ndarray, channels: np.ndarray, scalar_names: list[str], channel_names: list[str]) -> None:
    f.attrs["format"] = FORMAT
    f.attrs["rows"] = scalars.shape[0]
    f.attrs["channel_count"] = channels.shape[1]
    sg, cg = f.create_group("scalars"), f.create_group("channels")
    sg.attrs["names"], cg.attrs["names"] = _names(scalar_names), _names(channel_names)
    chunk = max(1, min(scalars.shape[0], 1 << 16))
    for i, name in enumerate(scalar_names):
        sg.create_dataset(name, data=np.ascontiguousarray(scalars[:, i]), chunks=(chunk,) if scalars.shape[0] else None, compression="lzf")
    for k, name in enumerate(channel_names):
        cg.create_dataset(name, data=np.ascontiguousarray(channels[:, :, k]), chunks=(chunk, channels.shape[1]) if scalars.shape[0] else None,
                          compression="lzf")


def save_traces(path: Path, links: dict, index: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("track_id", data=links["track_id"], compression="lzf")
        f.create_dataset("trace_index", data=index, chunks=True, compression="lzf")
        f.create_dataset("mother_track", data=links["mother_track"])
        f.create_dataset("drift", data=links["drift"])
        for key, value in meta.items():
            f.attrs[key] = value
    tmp.replace(path)


def _decode(values) -> list[str]:
    return [n.decode() if isinstance(n, bytes) else n for n in values]


class Results:
    """One timestack's results, read lazily: a measurement column is read from disk the first time it
    is asked for, and metric matrices are cached. Indexing like a dict still works
    (results["scalars"], results["trace_index"] ...) for code that wants whole arrays."""

    def __init__(self, output: str | Path, timestack_id: str):
        output = Path(output)
        self.id = timestack_id
        self.measurements = output / "Measurements" / f"{timestack_id}.h5"
        self._data: dict = {}
        self._columns: dict = {}
        self._matrices: dict = {}
        with h5py.File(self.measurements) as f:
            self.format = int(f.attrs.get("format", 1))
            self._data["meta"] = {k: v for k, v in f.attrs.items() if k not in ("format", "rows", "channel_count")}
            self._data["scalar_names"] = _decode(f["scalars"].attrs["names"])
            self._data["channel_names"] = _decode(f["channels"].attrs["names"])
        self.traces = output / "Traces" / f"{timestack_id}.h5"
        if self.traces.exists():
            with h5py.File(self.traces) as f:
                for key in ("track_id", "mother_track", "drift"):
                    self._data[key] = f[key][()]
                self.tracks, self.frames = f["trace_index"].shape  # the matrix itself is read only when asked for

    # dict-style access, for whole arrays
    def __getitem__(self, key: str):
        if key in self._data:
            return self._data[key]
        if key == "trace_index" and self.traces.exists():
            with h5py.File(self.traces) as f:
                self._data[key] = f[key][()]
            return self._data[key]
        if key in ("scalars", "channels"):
            self._load_blocks()
            return self._data[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data or key in ("scalars", "channels") or (key == "trace_index" and self.traces.exists())

    def get(self, key: str, default=None):
        return self[key] if key in self else default

    def keys(self):
        return [*dict.fromkeys([*self._data, "trace_index", "scalars", "channels"])]

    def _load_blocks(self) -> None:
        with h5py.File(self.measurements) as f:
            if self.format == 1:
                self._data["scalars"], self._data["channels"] = f["scalars"][()], f["channels"][()]
            else:
                self._data["scalars"] = np.stack([f["scalars"][n][()] for n in self._data["scalar_names"]], 1)
                self._data["channels"] = np.stack([f["channels"][n][()] for n in self._data["channel_names"]], 2)

    # lazy columns
    def column(self, name: str, channel: int | None = None) -> np.ndarray:
        """One measurement for every row: a cell metric, or a channel metric of one channel."""
        key = (name, None) if channel is None else (name, "all")
        names = self._data["scalar_names"] if channel is None else self._data["channel_names"]
        if key not in self._columns and name not in names:
            # a measurement added after this file was written (e.g. cyto_kept): blank, not an error
            from ..track.measure import CHANNELS, SCALARS
            if name not in (SCALARS if channel is None else CHANNELS):
                raise KeyError(name)
            if channel is None:
                shape = self.column(names[0]).shape
            else:
                self.column(names[0], 0)
                shape = self._columns[(names[0], "all")].shape
            self._columns[key] = np.full(shape, np.nan, dtype=np.float32)
        if key not in self._columns:
            if "scalars" in self._data or self.format == 1:
                block = self["scalars"] if channel is None else self["channels"]
                names = self._data["scalar_names"] if channel is None else self._data["channel_names"]
                self._columns[key] = block[:, names.index(name)]
            else:
                with h5py.File(self.measurements) as f:
                    self._columns[key] = f["scalars" if channel is None else "channels"][name][()]
        values = self._columns[key]
        return values if channel is None else values[:, channel]

    def matrix(self, name: str, channel: int | None = None) -> np.ndarray:
        """(tracks, T) float32 of a measurement, NaN where the track is absent. Cached."""
        key = (name, channel)
        if key not in self._matrices:
            index = self._data["trace_index"]
            self._matrices[key] = np.where(index >= 0, self.column(name, channel)[index.clip(0)], np.nan).astype(np.float32)
        return self._matrices[key]


def load_results(output: str | Path, timestack_id: str) -> Results:
    """A timestack's measurements and traces. Columns are read when first used."""
    return Results(output, timestack_id)


def upgrade_measurements(path: Path) -> bool:
    """Rewrite a format-1 measurements file in the per-measurement layout, in place and atomically.
    Returns True if it was converted. Results are unchanged; only reading them gets faster."""
    with h5py.File(path) as f:
        if int(f.attrs.get("format", 1)) >= FORMAT:
            return False
        scalars, channels = f["scalars"][()], f["channels"][()]
        scalar_names, channel_names = _decode(f["scalars"].attrs["names"]), _decode(f["channels"].attrs["names"])
        meta = dict(f.attrs)
    tmp = path.with_suffix(".upgrade.tmp")
    with h5py.File(tmp, "w") as f:
        _write_columns(f, scalars, channels, scalar_names, channel_names)
        for key, value in meta.items():
            f.attrs[key] = value
    tmp.replace(path)
    return True


def metric_matrix(results, name: str, channel: int | None = None) -> np.ndarray:
    """A metric as (tracks, T) float32, NaN where the track is absent."""
    if isinstance(results, Results):
        return results.matrix(name, channel)
    index = results["trace_index"]
    if channel is None:
        column = results["scalars"][:, results["scalar_names"].index(name)]
    else:
        column = results["channels"][:, channel, results["channel_names"].index(name)]
    return np.where(index >= 0, column[index.clip(0)], np.nan)


# ---------------------------------------------------------------------- in-memory intermediates
# Every layer of test mode is an HDF5 dataset in RAM (no file on disk), in `cache["layers"]`.
# Compression and writing run on their own thread, so inspecting does not slow the GPU.
_ENCODING = {
    "raw": np.uint16, "corrected": np.int16, "rough_mask": np.uint8, "binary_mask": np.uint8, "distance": np.uint16,
    "centroids": np.uint8, "nuclear_labels": np.int32, "label_map": np.int32, "border_labels": np.int32, "underseg_score": np.uint8,
}
ALL_LAYERS = [*_ENCODING, "table"]


def new_gui_cache(timestack_id: str, layers: list[str] | None = None) -> dict:
    handle = h5py.File(f"{timestack_id}.gui.h5", "w", driver="core", backing_store=False)
    return {"id": timestack_id, "file": handle, "layers": {}, "table": [], "layers_wanted": set(layers or ALL_LAYERS),
            "pool": ThreadPoolExecutor(1, thread_name_prefix="h5-cache"), "pending": [], "first_t": None}


def _write_layer(cache: dict, name: str, array: np.ndarray) -> None:
    layers = cache["layers"]
    if name not in layers:
        layers[name] = cache["file"].create_dataset(name, data=array, maxshape=(None, *array.shape[1:]),
                                                    chunks=(1, *array.shape[1:]), compression="lzf")
    else:
        ds = layers[name]
        ds.resize(ds.shape[0] + array.shape[0], axis=0)
        ds[-array.shape[0]:] = array


def add_layer(cache: dict, name: str, tensor: torch.Tensor, offset_t: int = 0) -> None:
    """Queue one batch of a layer. The copy off the GPU is made here; the rest is deferred."""
    if cache["first_t"] is None:
        cache["first_t"] = offset_t
    if name == "table":
        table = tensor.cpu().numpy().copy()
        table[:, 0] += offset_t
        cache["table"].append(table)
        return
    kind = _ENCODING[name]
    data = tensor.float() if tensor.dtype == torch.uint16 else tensor
    if kind is np.int16:
        data = data.round().clamp(-32768, 32767)
    array = data.to(torch.int64 if kind is np.int32 else torch.float32).cpu().numpy().astype(kind)
    cache["pending"].append(cache["pool"].submit(_write_layer, cache, name, array))


def flush_cache(cache: dict) -> None:
    """Wait for queued writes, and raise any error they hit."""
    for future in cache["pending"]:
        future.result()
    cache["pending"].clear()


def cache_size_bytes(cache: dict) -> int:
    return cache["file"].id.get_filesize()


def close_gui_cache(cache: dict | None) -> None:
    if cache is not None:
        cache["pool"].shutdown(wait=True)
        cache["file"].close()
