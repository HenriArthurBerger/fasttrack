# FastTrack

GPU-native segmentation, measurement and lineage tracking for terabyte-scale timelapse fluorescence
microscopy of 2D cultures. From raw 16-bit frames to single-cell traces, with a PySide GUI.

**Opinionated by design.** It targets sparse and mosaic cultures, where most pixels are background and nuclei
are mostly separate. It is not a general segmenter, and very dense fields are out of scope.

**Start here:** [docs/GUIDE.md](docs/GUIDE.md) covers running and reviewing everything yourself, and [docs/REVIEW.md](docs/REVIEW.md) assesses what works.

The core is the original notebook pipeline, repackaged: quadratic background, LoG mask, pseudo distance peaks,
local labels, `measure_subsections`, FFT shift correction. [`docs/AUDIT.md`](docs/AUDIT.md) lists each fix.

## Install

One line installs the app, with its own Python and the PyTorch build that fits the machine (CUDA with an NVIDIA
GPU, CPU-only otherwise, Apple GPU support on a Mac), and adds FastTrack to your applications menu. Run it again
to update.

**macOS and Linux** (in a terminal):

```bash
curl -LsSf https://raw.githubusercontent.com/HenriArthurBerger/fasttrack/main/installer/install.sh | bash
```

**Windows** (in PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/HenriArthurBerger/fasttrack/main/installer/install.ps1 | iex"
```

The first install downloads 1–3 GB (mostly PyTorch). Tracking a full plate needs an NVIDIA GPU; without one
FastTrack runs on the CPU, which is fine for looking at results, figures and labelling, but slow for tracking.
Options (a specific version, forcing the CPU build, uninstalling) are listed at the top of
[`installer/install.sh`](installer/install.sh) and [`installer/install.ps1`](installer/install.ps1).

**Into your own Python environment** instead:

```bash
pip install "fasttrack[gui] @ git+https://github.com/HenriArthurBerger/fasttrack"
```

## Quick start

For development, from a checkout:

```bash
pip install -e ".[gui,dev]"
```

```bash
fasttrack gui examples/example_data.yaml
```

```bash
fasttrack run examples/example_data.yaml
```

```bash
python -m pytest tests
```

**One experiment file holds everything.** Folders, channels, frame range, events such as a drug addition, the plate
layout and conditions, every tracking parameter, groups, gates and the list of figures all live in one YAML. You make it
in the GUI (File > Save), and running the same file again gives the same tracks and the same figures. The command line
takes nothing but that file:

| Command | What it does |
|---|---|
| `fasttrack gui [experiment.yaml]` | Open the app, optionally with a file |
| `fasttrack run experiment.yaml` | Track from raw data, then draw the figures |
| `fasttrack figures experiment.yaml` | Redraw the figures from existing results |

Before tracking, a quality check compares tile statistics across positions and skips those with fibers, debris or
odd backgrounds (you can override each one). Figures can treat positions or wells as replicates and test each condition
against a reference.

Event categories are yours: label crops with the keyboard on the Labelling tab, train a small classifier, and count
classified events per condition (section 2b of the guide).

Older notebook settings files (`code_to_condition`, `batch_dict`, `dmi_addition` ...) and plate layouts open directly
and are converted; [`examples/0811_converted.yaml`](examples/0811_converted.yaml) is `Config/0811_All_Batches.yaml`
converted. The format is described in [docs/GUIDE.md](docs/GUIDE.md#2-the-experiment-file) and
[`fasttrack/experiment.py`](fasttrack/experiment.py).

## The pipeline

| Step | Module | What it does |
|---|---|---|
| Background | `track/background.py` | Per-frame log2 quadratic fit to 1% tile values, scaled to the 2nd percentile |
| Mask | `track/masking.py` | Rough threshold from tile statistics, then LoG cleanup |
| Seeds | `track/centroids.py` | Pseudo distance map, blur, local maxima, then merge of nested seeds (or `track/centroid_net.py`) |
| Labels | `track/labels.py` | Max-pool growth, local labels, border ring and −label cytoplasm ring |
| Second pass | `track/undersegmentation.py` | Two nuclei of the previous frame inside one nucleus become two seeds |
| Measure | `track/measure.py` | `measure_subsections`: 40 metrics, one sub-section at a time |
| Shift | `track/linking.py` | FFT phase correlation on the mask, removed from coordinates before distances |
| Link | `track/linking.py` | Multi-pass, on the notebook's backward similarity core (below) |
| Orchestrate | `track/run.py` | Loader thread and tracking thread, growing batches, HDF5 output, resume |

**The multi-pass linker** (`link.method: multipass`):

1. Backward, each cell picks its most similar cell in the frame before, within `link.max_travel`. Each mother keeps only its top child.
2. Mothers that lost children are checked for division: mass conserved, sisters alike, local cell count up. If so, the mother's track ends and both children become daughters.
3. Remaining orphans are matched one-to-one to unused track ends, across gaps of up to `link.max_gap` frames.

`link.max_travel` is estimated per movie when 0 (99th percentile of unambiguous moves, times 2); any other value overrides it.

**Long movies.** One thread loads and the tracking thread launches each batch once it is fully loaded. Batches start at
`batch_size` and grow while GPU memory allows; if the card runs out, the batch splits and carries on. Stage shifts chain across
batches, all referenced to frame 0.

**Test mode** (the Inspect tab). Runs a chosen frame range and keeps every intermediate as an in-memory HDF5 dataset, overwritten on
each run. Up to 4 synced views, each with a layer dropdown. Left/right steps one frame, up/down steps ten, and there is a jump box.

**Testing on real wells** (read-only; results go to the `out` set in `benchmarks/settings/random_wells.yaml`, which names
an experiment file):

```bash
python benchmarks/random_wells.py
```

Results: `Measurements/<id>.h5`, `Traces/<id>.h5`, `experiment.yaml` (the file as run), `run_log.yaml` (what the data decided per timestack: travel gate, radius, batch sizes) and `Figures/`. See [`fasttrack/io/store.py`](fasttrack/io/store.py).

```python
from fasttrack.io.store import load_results, metric_matrix
r = load_results("results", "A02_xy1")
c3 = metric_matrix(r, "nuc_median", channel=2)   # (tracks, T), NaN where absent
```

## Measured results

### Full-length real movies (`benchmarks/random_wells.py`, RTX 3090, USB drive)

8 random positions from `2023-01-12_Mosaic_MZ6` (576 frames, 3 channels, 1022 × 1024): **54–94 s each**, of which about 50 s is
reading the drive and 15–50 s is GPU, at about 7–8 GB peak. Reading and tracking overlap, so totals are below their sum.
The second pass was then vectorised: on the densest position (G02_xy1), 42 s became 2.5 s, and the whole movie took 21 s of
compute with files cached.

Cells per frame rise a lot during these movies (for example 16 to 529), so the first 100 frames do not represent them.

| Check, on 8 positions | Multi-pass | Notebook single pass |
|---|---|---|
| Tracks per median cell count | 7–36 | 17–102 |
| Detections in tracks of 30+ frames | 0.83–0.95 | 0.57–0.88 |

These counts describe the output; they do not measure accuracy. Reporters and divisions change intensity legitimately, so only
synthetic or human-annotated data can score lineage. See the matched benchmark below.

Stage shifts at frames 100 and 345 appear across wells at about (+9, −9). Moving the batch boundaries does not change them, so they
are real events, not batching artifacts.

Test mode on 100 late frames: 1.1 GB in RAM (raw 597 MB, corrected 451 MB, int maps 1–28 MB each), about 30 s, 0.2 s per frame step.

### Matched synthetic benchmark (`benchmarks/matched_synthetic.py`)

Fits the generator to each real position, then scores against ground truth. On 3 random full-length positions, the fit was not
close enough in crowded late frames: the generator's nuclei overlap more than real ones, so it under-detects them. Treat its crowded-frame
scores as pessimistic until nuclear packing is fitted as well. Details and numbers are in [docs/REVIEW.md](docs/REVIEW.md).

### Accuracy on calibrated synthetic data (100 frames)

"Link accuracy" scores only links where both cells were found. "True link recall" counts every true link, so missed and
merged cells count as failures, and it cannot be inflated by merging. Use it for the 95–98% target.

| Mosaic preset, 5 movies | True link recall | Link accuracy | ID breaks | Tracks (true: 82) | Division precision / recall |
|---|---|---|---|---|---|
| **Multi-pass** | **0.938** | **0.979** | 86 | 145 | 0.39 / 0.89 |
| Multi-pass, no seed merge | 0.927 | 0.962 | 158 | 164 | 0.31 / 0.82 |
| Multi-pass, no second pass | 0.917 | 0.976 | 96 | 175 | 0.29 / 0.87 |
| Notebook single pass | 0.922 | 0.962 | 154 | 238 | 0.16 / 0.88 |
| Voting | 0.940 | 0.981 | 78 | 120 | 0.54 / 0.88 |

| Populated preset, 3 movies | True link recall | Link accuracy | ID breaks | Tracks (true: 1146) | Division precision / recall |
|---|---|---|---|---|---|
| **Multi-pass** | **0.877** | **0.982** | 939 | 1991 | 0.40 / 0.65 |
| Multi-pass, no seed merge | 0.894 | 0.975 | 1338 | 2078 | 0.36 / 0.64 |
| Multi-pass, no second pass | 0.819 | 0.978 | 1073 | 2469 | 0.29 / 0.64 |
| Notebook single pass | 0.842 | 0.943 | 3008 | 4635 | 0.12 / 0.74 |
| Voting | 0.878 | 0.983 | 875 | 1750 | 0.46 / 0.65 |

What this says:

- Linking among found cells meets the target at 98%. **True link recall, 88–94%, does not**, and the gap is now mostly detection:
  merged or missed nuclei, not choices made by the linker.
- **Division precision (0.39–0.40) is the weakest number.** It is the reason to build the event classifier.
- The populated preset's overlaps may be harsher than real data. That preset is only roughly calibrated.

Speed on the 3090: 1024 px, 100 frames takes about 2 s per batch. 2048 px needs `batch_size: 50` or auto batching, which recovers from
running out of memory by splitting.

### FFT shift correction

On the mask, the FFT recovers the real stage moves (see above), plus occasional small single-frame shifts. Synthetic error is about 1 px over a movie.

### Quadratic background on real data

Residual tile-median spread (p5–p95) is 15, 11 and 45 counts for c1–c3, with a background level of 26, 25 and 38
counts. The level is comparable to the c2 and c3 nuclear signal (about 50–70), so subtract it before ratios, as
`analysis.traces.ktr_ratio` does, following the notebook. This only measures flatness in background tiles.

### ResNet seed finder (`benchmarks/centroid_net.py`)

Measured against my earlier seeds, and needs re-running against the notebook's: F1 0.945 and 0.972 vs 0.926 and 0.958
on unseen synthetic movies, trained on 40 labelled frames. No labelling tool yet; training is by script.

## Synthetic data

```python
from fasttrack.synthetic.generate import generate
frames, truth, cfg = generate({"seed": 1, "frames": 100})   # calibrated mosaic preset
```

`image = illumination * (protein + autofluorescence)`, with shot and read noise, overlap adding signal, an
offset Gaussian field, divisions with mass conservation, stage jumps, media changes, debris, dark fibres and
out-of-focus frames. `DENSE_PRESET` is a stress test, not a target. Tools > Write a synthetic experiment in the GUI writes any preset to disk, with an experiment file.

## Known limitations

- A merge cannot inherit seeds across a batch boundary. The backward sweep recovers most of these.
- At most 999 cells per frame in one sub-section; raise `measure.subsections` beyond that.
- `underseg_score` is saved and shown in the Inspect tab, but its weights were fitted on synthetic data with earlier seeds. On real
  data it saturates: most nuclei score high. Refit on labelled real data before relying on it.
- Horizontal seam of 3–4 counts at row 510 in c1 and c2 on your camera. It is not corrected.
- Parameters were checked on one well. There is no ground truth on real data, so real-data results are
  consistency checks, not accuracy scores.
- Plate conditions feed `analysis/plate.py`, but the Analysis tab does not group by condition yet.


