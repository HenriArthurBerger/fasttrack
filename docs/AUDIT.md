# Audit of the original code

What I found reviewing the notebooks and `Track/`, and what happened to each item. "Verified" means I
ran the code and saw the behaviour. "By reading" means I found it in the source and did not run it.

## Bugs that affect results

| # | Where | Problem | Evidence | Now |
|---|-------|---------|----------|-----|
| 1 | Cell 168 | `data_clone - shift_tensor` has the wrong sign for what `find_shift` returns, so a stage jump is doubled. | Verified on real data: xy1 frame 98 cells move +11 in Y, `find_shift` returns −11. | `x + cumulative shift`, with a test. |
| 2 | Cell 151 | `shift[0] != 0 and shift[1] != 0` ignores a jump along one axis only. | By reading. | Removed. |
| 3 | Cells 150–151 | On the corrected image the phase-correlation peak often stays at (0, 0). | Verified: no shift found in xy3 and xy4, though cells move about (−3, +10). Synthetic error up to 142 px. | Same FFT, on the binary mask: synthetic error under 6 px, frame 98 found in all 6 positions. Cause not fully established; fixed camera pattern is my best guess. |
| 4 | `Get_Centroids.py`, cell 97 | `TVT.GaussianBlur(kernel_size)` draws a **random sigma from 0.1–2.0 on every call**, so centroids differ between runs. | Verified: 3 calls gave centre values 0.34, 0.40, 0.07. | Fixed sigma, chosen by sweep (1.5). Label maps are now identical between runs (tested). |
| 5 | Cells 129–134, `Meausrement_V2.py` | `.type(torch.int16)` wraps values above 32767. | Verified: 40000 becomes −25536. | int32, tested at 40000. |
| 6 | `Meausrement_V2.py`, `_fill_local_density` | The final reshape drops the channel axis, so it crashes with more than one channel. | Verified. | One-token fix. |
| 7 | Cell 168 | `torch.min(total_diff, axis=1)` gives one entry per *t0* cell but indexes `t0_label`, stored against `t1_label`. `torch.max(torch.min(...))` above it would also raise. | By reading. | argmin per t1 cell. |
| 8 | Cell 168 | A cell with no candidate has all-`inf` costs, so `argmin` links it to row 0. NaN features make costs NaN. | By reading. | New track; NaN counts as no difference. |
| 9 | `Quadratic_Background.py` | Filtered tiles are NaN and go into `linalg.solve`, which can blank the frame's fit. Cell 29 avoids this by fitting unfiltered values, so debris tiles are kept. | By reading, plus a regression test. | NaN tiles get zero weight. |

## Fragile code that happened to work, kept as is

**The LoG kernel.** `kernel / np.sum(kernel)` divides by a truncation residue near zero, so its gain depends
strongly on kernel size (verified):

| Size, sigma | 11, 2 | 13, 2 | 21, 2 | 11, 1.5 | 15, 3 |
|---|---|---|---|---|---|
| Centre value | 0.84 | 3.27 | 11792 | 24.5 | 0.21 |

It works at (11, 2), because the DC term that division leaves behind keeps flat nuclear interiors positive. A
"correct" zero-mean kernel shreds nuclei; I tried it. Kept unchanged. Change size and sigma only together, and check.

## Things in the code that could not run

Every module in `Track/` was a partial extraction and none could be imported:

- `Filters_2D.py` has a syntax error: `from Track.Kernels_2D import ` with nothing after it.
- `Batch_Track.py` has `from __future__` after other imports, which is a SyntaxError.
- Wrong module names: `Track.Load_Timestrack`, `Tack.Get_Centroids`, `Track.QuadraticBackground`, `Track.Filter_2D`, `Track.Filters2D`, `Track.Hue_Cluster`.
- Undefined globals: `t`, `x`, `y`, `radius`, `device`, `max_distance`, `labeled_peaks`, `image_stack`.
- `main.py` imports `Gui.*`, but the folder is `GUI/`, which fails on Linux.
- `Track/import math` (a file with that name) is a copy of `Old/measurement2vec.py`.

The logic was sound; the modules were simply unfinished extractions.

## Performance

| Issue | Measured | Fix |
|---|---|---|
| `tifffile` decodes PackBits in pure Python without `imagecodecs` | 14.9 s for 300 real files vs 0.68 s with PIL (22×, identical pixels) | `layout.read_frame` |
| Figure output (`show_tensor`) in the pipeline | You identified this as the slowest step | One `on_layer` hook; nothing runs when nothing listens |
| Python dict and `deepcopy` per frame while building traces (cell 171) | By reading: O(T²N) | `trace_index` array, one assignment |
| `get_local_labels` loops over groups | By reading | Same output, vectorised |

## What is your code, and what is not

Ported with only the fixes listed above: quadratic background, rough and LoG mask, pseudo distance, peaks, label
growth, border and ring map, local labels, `measure_subsections`, FFT shift, and the lineage cell.

Built on your core: nested-seed merge after your peaks, and the multi-pass linker around your backward similarity.

New: synthetic generator and evaluator, temporal second pass, voting linker (optional), ResNet seeds (optional),
storage, orchestration, GUI. My earlier rewrites of the core are in `archive/`, unused.

The event detector in cell 102 (merge, death, exit classes) is not ported.

## Analysis notebook

The `AnalysisFunctions` / `LiveTools` module it imports is not in the folder, so I could not run the
notebook. Ported from its visible usage: moving average, Savitzky–Golay smoothing, KTR ratio, percent-high,
commitment times, copying mother history to daughters, and condition grouping. Not ported: the many one-off
figure cells, `.mat` loading. One fix: `commitment_times` used to give 0 both for "committed from frame 0"
and "never crossed"; those are now distinct.
