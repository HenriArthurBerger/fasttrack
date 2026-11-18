# FastTrack: how to run it and review it yourself

This guide covers installing, running on your data, using the GUI, reproducing every number in the README, and
reading the code. For an assessment of what works and what does not, see [REVIEW.md](REVIEW.md).

All commands run from the `refactor/` folder.

## 1. Install and check

```bash
pip install -e ".[gui,learned,analysis,dev]"
```

```bash
python -m pytest tests
```

That runs 79 tests in about 15 s on the GPU (they fall back to the CPU without one). A pass means the pieces behave as
specified on small controlled inputs. It says nothing about accuracy on your data; section 5 covers that.

## 2. The experiment file

Everything about an analysis lives in one YAML: which folders, which frames, what each channel is, when things happened,
which wells are which condition, every tracking parameter, and which figures to draw. Make it in the GUI and save it
(File > Save). Then:

```bash
fasttrack run experiment.yaml
```

That tracks every selected timestack from the raw TIFFs and draws the figures. Running it again skips finished timestacks
and redraws the figures. `fasttrack figures experiment.yaml` only redraws. Nothing is set on the command line.

Saving writes every key, defaults included, so a later change to a default cannot change an old analysis.

| Section | What it holds |
|---|---|
| `experiment` | Name and free notes. |
| `paths` | `raw` (only ever read) and `results`. The results folder must be outside the raw folder. |
| `files` | Filename `pattern` (regex with groups `well`, `t`, `pos`, `chan`), `first_frame`, `last_frame` (exclusive, 0 = end), `stride`, reader threads. Frame numbers count timepoints of the full movie. |
| `channels` | One entry per file channel: `name`, `role` (`nuclear`, `measure` or `skip`), `reporter`, `color`, and `primary`, what the reporter is read out as (`nuc_mean`, `nuc_median`, `nuc_sum`, `cyto_mean`, `cyto_nuc` for a KTR, `nuc_cyto`, `none`; `auto` picks from the reporter's name). Exactly one is nuclear. Empty list: every channel, the first nuclear. |
| `time` | `frames_per_hour` (optional, only labels figure axes; 0 = axes in frames) and `events`: `name`, `frame`, `color`. An event is any change during the movie; `dmi_addition` becomes one. |
| `plate` | `conditions` (`name`, `color`, `wells`), `excluded_wells`, `excluded_positions` (e.g. `C05_xy2`), and `process`: `assigned` tracks only wells with a condition, `all` tracks every well found. |
| `quality` | The bad-position check (section 2a): settings, `checked`, `flagged` (position: reason, written by the check) and `keep` (flagged positions you track anyway). |
| `labelling` | Event categories (yours; neutral, division, death, abnormal by default), crop size, frames before and after, channels, how often to show each kind of candidate, training steps, and the labels and model file names (inside results). `labelling.centres` holds the centre labeller's box size, margin, count-change settings, sampling and training steps (section 2b). |
| `pipeline` | Every tracking parameter. The Parameters tab edits it; see `fasttrack/config.py` for comments. |
| `analysis` | `min_track_frames`, `drop_border`, `groups` (named lists of conditions, first is the reference; `batch_dict` becomes these), `gates` (a metric, a threshold, above or below; `xthreshold` etc. become these). |
| `figures` | Output `folder` (inside results), `formats`, `dpi`, size, the `list` of figures, and `summary`: the routine figures drawn after every run (section 2c). |

**Metrics** (in gates and figures): a `measure` and `channel` (`nuc_median` of `ERK`), or a cell metric with no channel
(`nuc_area`). Set `over_measure` / `over_channel` to divide: a FRET ratio is `nuc_mean` of YFP over `nuc_mean` of CFP; a
KTR readout is `cyto_median` over `nuc_median` of the same channel with `background_percentile: 5` (the notebook's
background floor). Optional `log2` and `smooth` (trailing moving mean in frames).

**Figures:** `kind` is `timecourse` (median and quartiles per condition), `heatmap` (the population over time,
as in the KTR notebook: x is time, y the metric in value bins, colour the fraction of that frame's cells in each bin, so
every column sums to 1; bins are `bin_width` high, 0.1 for log2 metrics by default; the value range is the plot's
`vmin`/`vmax`, by default the 1st to 99th percentile of all conditions shown, shared by every panel; a `gate` on the same
metric is drawn as a dashed line), `track_heatmap` (one row per cell, sorted by its late value), `percent_high`
(the % of cells passing a `gate`), `distribution` (per-cell mean over the `frames` window, as `violin`, `box` or `bar`) or
`cell_count`, or `event_rate` (classified events of one `event` category per 100 cells per frame, in `bin`-frame bins).
`group` picks conditions (empty = all). `time_zero_event` draws time relative to an event. Untick a figure
(`enabled: false`) to keep it in the file without drawing it. Each figure is written in every format plus a CSV of the
numbers drawn; `Figures/experiment.yaml` records the file that drew them.

**Error bands:** `band` sets what a line and its band show (`fasttrack/analysis/bands.py`):

| `band` | Centre | Band |
|---|---|---|
| `iqr` | median | 25th to 75th percentile |
| `range90` | median | 10th to 90th percentile |
| `sd` | mean | ± 1 SD |
| `sem` | mean | ± 1 s.e.m. |
| `ci95` | mean | 95% confidence interval (Student t) |
| `none` | median | no band |
| `auto` | | `iqr` over pooled cells, `sem` over replicates |

The band is over replicate values (each well's or position's median) when `unit` is well or position, and over
cells when it is `cell`. s.e.m. and CI over pooled cells treat cells as independent, so those figures say so in
their notes. % passing with pooled cells shows a 95% Wilson interval for `ci95`. Bar charts use the band for their
error bars, and start at 0.

**Replicates and statistics:** `unit` sets what one point is: `cell` (pooled), `position` or `well`. With position or well,
each replicate's cells are summarised first (median of per-cell values), time courses show mean ± s.e.m. across
replicates, and distributions draw each replicate as a dot. With `stats: true`, a distribution tests every condition
against the group's first: Welch's t-test on replicate values, or Mann-Whitney U on cells (pseudoreplicated: cells of one
well are not independent, so prefer replicates). p-values are Holm-adjusted and written to the figure's CSV; with one
replicate the test is reported as n/a.

**What stops a run, and what does not.** Only problems that make tracking impossible stop it: no raw or results folder,
results inside the raw folder, no single nuclear channel, duplicate channel names, a well in two conditions, a well that
does not exist on the plate format. Everything else is a warning, shown in the log and appended to `warnings.log`, and the
run carries on:

- figure settings that point at something missing (a group naming a condition no longer on the plate, a figure using a
  deleted group or gate); those figures leave it out or are skipped;
- a sparse position (median cells per frame below `quality.min_cells_per_frame`, 5) or a crowded one (nuclei covering
  more than `quality.max_nuclear_coverage`, 35%, of the field), and frames with no cells;
- a position that fails outright: it is skipped and the rest of the plate is tracked.

**Missing and broken files.** Frame *i* means the same timepoint in every position, so nothing is ever dropped or shifted:

- *Missing from the folder* (found by the scan): the frame stays in place as a blank frame. Tracks bridge a single blank
  frame. The Experiment tab lists each one after **Scan data folder**, and the run summary repeats them.
- *Present but unreadable* (0 bytes, truncated, corrupt): the read is tried three times, 0.5 s and 2 s apart, because a
  busy or spinning-up USB disk can fail a read that works a moment later. A file that only reads on a later try is listed
  in the summary (the disk may be failing). One that never reads becomes a blank frame.
- *Gone mid-run* (the drive was unplugged): if not one file of a batch can be read, that position fails with "is the drive
  still connected?" and the run moves on, so the summary shows exactly what to redo.
- *Movie ends early*: the position ends early; its length is never padded. If some positions lack only their last 1 to 3
  frames, the Experiment tab offers **End every position at frame N** and the summary suggests the same value. Nothing
  changes until you press it, since cutting frames from every position is your decision.
- *Too many*: a position with more missing, unreadable or cut-off frames than `quality.max_missing_frames` (1) is flagged
  like a fiber: left out of tracking and figures unless ticked Track anyway. Files that turn out unreadable during
  tracking flag the position afterwards: its results are kept but figures leave it out.

**Leaving wells out of figures.** Figures draw only wells that are in a condition and not excluded, and skip excluded or
flagged positions. Editing the experiment file (removing a well from its condition, adding it to `plate.excluded_wells`,
or adding a position to `plate.excluded_positions`) and redrawing is enough; nothing needs to be tracked again.

**Partial runs.** On the Run tab, untick positions to leave them for later, or tick only one condition. Finished
positions are skipped anyway, so running the rest later completes the plate. Wells you never want go in the plate's
excluded wells or get no condition. Each run records which positions it was asked to do in `run_log.yaml`.

**Every channel is measured**, the nuclear one included: all per-channel metrics (mean, median, sum, cytoplasm ring,
ratios ...) for every channel, plus shape metrics (area, eccentricity, axis lengths ...) per cell. The nuclear role only
says which channel segmentation uses.

**What a run writes** into the results folder:

- `Measurements/<id>.h5` and `Traces/<id>.h5` per timestack.
- `experiment.yaml`, the file as run. If tracking settings change between runs, the old copy is kept as
  `experiment.<date>.yaml` and the log warns that finished timestacks were not redone (turn on `pipeline.output.overwrite`).
- `quality_report.yaml`: every checked position, its bad frames, worst unusual-tile share and level, and the reason.
- `Logs/<time>_detailed.log`: everything about the run as it happened. `Logs/<time>_summary.txt` (and
  `Logs/latest_summary.txt`): only what to check by hand (flagged, failed, unreadable files, sparse or crowded positions,
  large stage jumps) and a speed line saying whether the disk or the GPU was the limit. The Run tab shows it at the end.
- `run_log.yaml`, appended each run: date, version, GPU, and per timestack what the data decided (travel gate, nuclear
  radius, batch sizes) with counts and timings.
- `Experiment_Config.yaml`, the flat parameter dict the tracking code read.

**Older files.** Opening a notebook settings YAML (`code_to_condition`, `code_to_well`, `batch_dict`, `dmi_addition`,
`x/y/zcondition`, thresholds, `endframe`) or a plate layout converts it. Well codes like `202` become `B02`. Thresholds
become log2 gates, as the notebook compared them. Anything not understood goes into the notes. Save it as a new file.
`examples/0811_converted.yaml` is your `Config/0811_All_Batches.yaml` converted, with figures that mirror the notebook's.

## 2a. Bad positions (the quality check)

Before tracking, a run reads a few evenly spaced frames of every position the plate selects (6 by default) and cuts each
channel into 16 × 16 tiles. Per tile it keeps the 1st percentile (the background floor, from the same quantile tile function
as the background correction) and the median. Each map is taken relative to its frame's median tile, which removes the
overall level, so expression or density differences between conditions do not count. A random sample of positions (48)
defines each tile's normal value and spread; illumination is shared by all positions of a plate, so this works.

- A tile is unusual beyond 6 robust deviations. Floor maps count in both directions: a dark fiber pulls the floor down, a
  bright out-of-focus blob lifts it. Median maps count only when darker: a brighter median is a dense colony, which is
  biology.
- A frame is bad if unusual tiles cover over 2% of it, or its background level is 6 robust deviations off.
- A position is flagged if a third or more of its sampled frames are bad. Positions with an isolated bad frame are listed
  but tracked.

Flagged positions are skipped by tracking and figures unless ticked "Track anyway" on the Quality tab (`quality.keep`);
untick "Skip flagged positions" (`quality.enabled`) to ignore the check. The result is stored in the experiment file, so a
rerun of the file makes the same choice. On ExampleData it flags the two A02 positions with a drifting fiber and nothing
else; on 60 random positions of the 2023-01-12 HDD dataset it flags none, and reports one isolated bad frame (a bright
streak in channel 3 of E05_xy2, frame 345). The check takes about 0.6 s per position from the USB drive.

The Quality tab shows every position, its status and reason, and for a selected position its worst frame per channel
beside the map of unusual tiles. **Rescore** applies changed thresholds without reading again.

## 2b. Labelling nucleus centres and the learned seed finder

Seeds (one point per nucleus) decide everything after them: a nucleus with two seeds is cut in two, a nucleus with
none is lost for that frame and its track breaks. On the **Label centres** tab you correct seeds on small patches,
and a ResNet seed finder is trained on your corrections.

**Where it looks.** Segmentation mistakes rarely last. A nucleus split in two for one frame, or missed for a frame,
is a short bump in the number of cells in that part of the field. For every tracked position the detections are
turned into a local cell count (within about `count_sigma` px, stage drift removed), and every frame is compared
with the median of the `count_window` frames around it. Where the count differs by at least `min_change` (1 = one
nucleus) there is a candidate, strongest first. Divisions do not count, because the count stays up after them, so
the median follows. On the 2024-01-03 plate this gives about 40 candidates per position, and the strongest show
nuclei missed in one frame, clusters cut into too many pieces, and seeds on out-of-focus debris. A quarter of the
patches (`sample.random`) are around random cells, so the model also sees ordinary nuclei.

**Labelling.** **Start labelling** opens a patch at the candidate's frame, with tracking's centres already on it
(blue), and the frames before and after on the right for context. Click a point to remove it. Click a nucleus
without one to add a point (pink). Every nucleus whose centre lies inside the yellow box needs exactly one point.
The margin around the box is context only, and an empty box is a valid answer. Enter saves and moves on, Space
skips, Backspace goes back, Ctrl+Z undoes a click. Patches are saved to `Labels/centres.json` at every Enter. With
the default 96 px box, a patch takes a few seconds. Aim for 100 or more.

**Training.** **Train** reads the labelled frames from the raw TIFFs, background-corrects and normalises them
exactly as tracking does, and trains the seed finder on the boxes. Outside its box a patch carries no loss. A fifth
of the patches is held out first: the report compares tracking's seeds and the model's with your clicks on those
patches (F1, nuclei missed, extra seeds), picks the best threshold, and then trains the final model on every patch
(`Labels/centre_model.pt`, report in `Labels/centre_model.json`). **Use for tracking** sets `centroids.method: cnn`,
the checkpoint and the threshold. Check the result in a test run before tracking the plate.

How well can it work? On synthetic movies, 150 patches labelled from the known truth (as a perfect labeller
would), 1500 steps, 3 minutes on one GPU: on the 30 held-out patches, tracking's seeds scored F1 0.68 (53 missed,
13 extra) and the model 0.98 (3 missed, 3 extra). Synthetic nuclei are easier than yours: judge it by the
held-out numbers from your own labels.

## 2c. Labelling events and the event classifier

On the Label events tab, **Start labelling** shows one cell at a time as a strip of frames around one frame (outlined), one row
per channel, following the cell's track. Press 1-9 for a category, Space to skip, Backspace to go back and undo. Labels save
to `Labels/events.csv` in the results folder at every click, so you can stop any time.

The cells shown are drawn from the tracks, weighted by kind: random cells, tracks that end early, tracks that start late,
and mothers just before their daughters appear. Events hide in the last three, so they come up far more often than random
sampling would give. Aim for 30-50 per category.

**Train** cuts every labelled crop from the raw TIFFs and trains a ResNet18 on the stack (channels × frames as input planes),
with flips, rotations and class-balanced batches. It first holds out a fifth of the labels to report accuracy and a confusion
matrix, then trains the final model on all of them (`Labels/event_model.pt`). **Classify all positions** scores every track
end, track start and division candidate and writes `Events/<position>.csv` with class probabilities. Figures of kind
`event_rate` count them per condition over time.

How well does it work? `benchmarks/event_classifier.py` runs the whole path on synthetic movies with known deaths and
divisions, with 40 labels per category, and scores held-out positions:

| Category | Recall | Precision | n |
|---|---|---|---|
| division | 0.98 | 0.67 | 51 |
| death | 0.90 | 0.45 | 31 |
| neutral | 0.86 | 0.99 | 416 |

Precision for rare events is limited by how many neutral candidates there are (most track ends are tracking breaks or cells
leaving the field). Synthetic deaths simply vanish, unlike real ones, so this measures the machinery, not death detection on
your cells: judge that with the held-out confusion matrix from your own labels.

### Summary figures, drawn for you

After every run, and whenever their settings change, the **Summary** page draws the routine figures in the
background and writes them to `<results>/Figures/Summary`, with a CSV each (`fasttrack/analysis/summary.py`):

- cells per position;
- nuclear area;
- for every channel, its **primary measurement** over time, as a population heatmap, and as a distribution over the
  last `end_frames` frames.

The primary measurement is set per channel on the Experiment page. `auto` picks cytoplasm / nucleus (log2,
background-corrected) for a KTR, integrated intensity for H2B, Hoechst or DAPI, and the nuclear mean otherwise.

- **Settings** (`figures.summary`): which kinds, the replicates, the error band, the end-point window, and whether to
  draw after runs.
- **Copy to my figures** puts one in the figure list to edit.
- `fasttrack figures experiment.yaml` draws them too.

### How figures stay fast

Figures never build (tracks × frames) matrices; they work on measurement rows (frame, track, value). Those are up
to 10,000 × 600 and 86% empty on real data.

**The metric cache** (`fasttrack/analysis/columns.py`). Reading one metric for a 170-position plate means opening
170 HDF5 files (11.6 GB) and transforming 57 million rows. So each metric is computed once per session for the
whole experiment and kept in **RAM only**; nothing is written to disk. It is held already filtered by the track
filter in use, as compact columns of 8 bytes per cell-frame: frame int16, track int16, value float32.

- **What is cached:** 10 metrics automatically:
  - each channel's primary measurement;
  - nuclear area;
  - the summary heatmaps' log2 versions;
  - the gates' and saved figures' metrics, in that order.

  Up to 5 more metrics you draw are cached as you go, least recently used out.
- **When:** in the background, when the analysis pages open, after every run, and after drawing a metric that
  isn't cached yet. A retracked position is recomputed and the rest kept. Changing the track filter rebuilds the
  cache.
- **Cost and speed**, example plate: building 9 metrics takes about 45 s in the background, once per session, and
  uses about 2.9 GB of RAM. The 11 summary figures then take 4.4 s instead of 30 s.

**What's left of the time:**
- **Per-frame statistics:** quantiles, moments, and the heatmap's bin counts are one sort or one bincount on the
  GPU (torch), when there is one.
- **matplotlib:** most of the rest, about 0.2-0.5 s per multi-panel figure.

Cached values match the files' values to float32 precision (checked in `tests/test_summary.py`).

### Comparing conditions, and figures for a paper

There is no single best way to compare conditions, so every time course (time course, % passing a gate, cells per
position, event rate) offers three, under **Compare** on the Figures tab:

| Compare | Draws | Use it when |
|---|---|---|
| all on one plot | every condition on one axes | a few conditions whose differences you want to see directly |
| one plot each, reference in grey behind | a small plot per condition, the reference (first ticked, or the one chosen) in grey behind it, axes shared | many conditions, or overlapping bands |
| one plot each, alone | a small plot per condition, axes shared | the shape of each condition matters more than the comparison |

*Legend: direct* labels each line at its end instead of a legend box. **Colours** come from the Settings palette (or
the figure's own):

- **Plate tab colours:** exactly the colours set on the Plate tab.
- **Colour-blind safe:** chosen by the number of conditions. The reference is dark grey; 1–3 others use Tol high-contrast, 4–6 Tol bright, 7–9 Tol muted.
- **Okabe–Ito.**
- **Combinations:** each drug gets a hue, and combinations such as "Tdi + MekI" mix their parts' hues and draw dashed.

An axis label names the quantity only. What a line or band is (median and quartiles, mean ± s.e.m. of wells) and the
test go into the figure's title notes, and on an exported page into the caption.

The **Export** page builds multi-panel figures at journal size:

- **Page:** the journal (Nature 89/120/183 mm, Cell 85/114/174, Science 57/121/184), the column width, the grid columns, the text size (7 pt by default) and the data line width (0.75 pt).
- **Panels:** each is a copy of a quick figure (so it can be compared differently without touching the quick figure), a fate view or a dynamics view. Panels fill the grid left to right; each has a span (in columns) and a height (in mm). Fate and dynamics views lay themselves out for the panel's size: as many small plots as fit, one legend outside the plots, and a note in the caption on anything left out. **Conditions** picks which conditions a fate or dynamics panel draws; colours stay the same as with every condition.
- **Preview:** the exported page itself, at its real proportions. Under it is a list of what breaks the standard (text below 5 pt, anything cut off at the page edge) and the caption.
- **Files:** **Export** writes the page in each format, with TrueType fonts so text stays editable in Illustrator. Next to it go the caption (`.txt`) and each quick-figure panel's numbers (`<name>_<letter>.csv`), all in `<results>/Figures/Export`.
- **Templates:** save a definition and apply it to another experiment. Conditions that are not there are left out, and you are told which. They are kept in `~/.config/fasttrack/figure_templates` (or `$FASTTRACK_TEMPLATES`).
- **Command line:** `fasttrack export experiment.yaml [name ...]` does the same.

The rules behind all of this, with sources, are in [FIGURE_STANDARD.md](FIGURE_STANDARD.md).

### Loading results in Python

```python
from fasttrack.io.store import load_results, metric_matrix
r = load_results("/path/to/results", "A02_xy1")  # reads columns lazily
r["scalar_names"], r["channel_names"]          # what was measured
r.column("nuc_area"), r.column("nuc_median", 1) # one measurement for every row (channel 1)
nuc = metric_matrix(r, "nuc_median", channel=1)  # (tracks, frames), NaN where a track is absent
mothers = r["mother_track"]                      # mother track of each track, -1 if none
r["drift"]                                       # cumulative stage shift per frame, from frame 0
```

`fasttrack/analysis/traces.py` has the notebook's analysis functions (smoothing, KTR ratio, percent-high, commitment times,
mother history copied into daughters, filtering of short and edge-clipped tracks).

## 2d. Stochastic dynamics: what many cells say about fast switching

The Analyse stage has five pages: **Figures**, **Readouts**, **Features**, **Fates** and **Dynamics**. Readouts say what a
reporter measures; Dynamics fits stochastic models to every track at once. Features and Fates are in section 2e.

**Why a population model.** ERK pulses last minutes and the KTR follows ERK within minutes. Frames 6-12 minutes apart
cannot resolve a single pulse in a single cell. A model in continuous time can still learn about it from many cells:
between two frames the hidden state moves by P = expm(Q Δt), so hidden switching faster than the frames is part of the
likelihood. With many cells, the values seen at Δt, 2Δt, 3Δt..., and how far apart they spread, constrain rates above 1/Δt
wherever the reporter still carries a trace of them. On synthetic data with 6-minute frames, ERK dwell times of 10 and 5
minutes come back as 9.7 and 4.9 (`tests/test_dynamics.py` checks this). Where the data hold no such trace, the **recovery
check** says so: it refits data simulated from your fit, so you see which parameters your frame interval can and cannot
pin down.

**Readouts** (`analysis.readouts`). A readout is a kind applied to a channel:
- `ktr`: ln(cytoplasm / nucleus), with the per-frame background floor used in figures.
- `nuclear_reporter`: ln(nucleus / cytoplasm).
- `intensity`: ln(nuclear mean).
- `speed`: px per frame, stage drift removed.
- `metric`: any figure metric.

Each value carries its own **error estimate** from the pixels it was measured on: the median of n pixels with spread s has
error about 1.25 s/√n, and the relative errors of ring and nucleus add. On the example ERK KTR data the median error of
ln(C/N) is 0.34. That is most of the frame-to-frame jitter, and the model then does not mistake it for ERK dynamics.

`Largest error` drops single noisy observations. `Smallest mean signal` drops cells that barely express the reporter.
**Suggest from channel reporters** adds a KTR readout for every channel whose reporter name contains KTR. **Preview** shows
values, errors and example tracks.

**Models** (`analysis.dynamics.fits`; code in `fasttrack/dynamics/`):

| Model | Hidden state | Parameters |
|---|---|---|
| `erk_ktr` | ERK in 2-6 activity modes (birth-death chain) × the KTR ln(C/N) on a grid | switching rates, the ln(C/N) each mode settles at, KTR import rate (response time), intrinsic noise, extra measurement noise |
| `ou` | the readout as a mean-reverting random walk | reversion rate (correlation time), mean, noise |
| `states` | K discrete states over one or more readouts or many features (a hidden Markov model in continuous time; section 2e) | rates between states, each state's profile (mean, sd, shared directions), starting share per condition, optionally division rates and where newborn cells start |
| `custom` | your SDE, du = drift(x) dt + noise(x) dW, as formulas in the file | your parameters |

The KTR in `erk_ktr`: nuclear import k_i and ERK-driven export k_e,m = k_i e^{L_m} give dr/dt = k_e,m (1 − r) − k_i r for the
cytoplasmic fraction r. So ln(C/N) = logit(r) relaxes towards L_m, at rate k_i (1 + e^{L_m}). Continuous variables are put on
a grid with a Scharfetter-Gummel Fokker-Planck scheme, which stays positive and does not overflow for any drift.

**Fitting.**
- **Likelihood.** Exact, by the forward algorithm, for every track. Cells that appear later start from the population's state
  at that time.
- **Robust.** Observations may be outliers (1% by default, spread over the range), so a tracking swap does not dominate a track.
- **Optimiser.** Adam on bounded parameters, with minibatches from every condition, on the GPU with the most free memory.
- **Sharing.** Each parameter is one value overall, or one per condition, per time window (`window_hours`, `split_at_events`),
  or both. Comparing a fit with a rate shared against one with it free per condition asks whether a drug changes that rate.
- **Intervals.** `bootstrap` refits on positions resampled with replacement: positions, not cells, are the independent units.
- **Fixed parameters.** Fix a parameter, for example a literature KTR import rate, in the parameter table.
- **Time unit.** Set **Minutes between frames**, or frames per hour on the Experiment tab, so rates are per minute. Without
  either, rates are per frame.

**What a fit saves**, in `<results>/Dynamics/<fit>/`:
- `fit.json`: every parameter and derived quantity (dwell times, share of time in each mode, KTR response time, cycle time)
  per condition and window, with intervals, likelihood, AIC and BIC.
- `checks.npz`.
- `states.h5`: the probability of each hidden state for every fitted track and frame.
- `figures/`.

**Checks.** Look at them before trusting a fit:
- **Distributions:** the observed population against the model's one-step-ahead prediction.
- **PIT:** where each observation fell in its prediction from the cell's past. Flat means calibrated.
- **Autocorrelation:** the data against data simulated from the fit with the same frames and noise.

**Comparing models.** Fits on the same tracks are listed with ΔBIC under the summary; a difference above about 10 is strong.

```bash
fasttrack dynamics experiment.yaml            # every ticked fit
fasttrack dynamics experiment.yaml "ERK 2"    # just this one
```

A first fit on the example data (Ctrl and MekI, 800 tracks each, no frame interval set) took 140 s on one 3090. It found an
"off" mode at ln(C/N) ≈ −3.0 and an "on" mode at ≈ 0.2. The model's autocorrelation (0.80, 0.71, 0.56 at lags 1, 5, 10 frames)
is close to the data's (0.73, 0.63, 0.53).

## 2e. Features and fates: many measurements, and when cells commit

**Features** (`analysis.features`; code in `fasttrack/analysis/features.py`). A feature is one measurement with a transform
(`log`, `asinh` or `none`). It can be:
- a cell measurement: nuclear area, shape, texture, local density;
- a channel measurement, such as the nuclear mean of PPARG;
- `speed`;
- `age`: frames since the cell's observed birth;
- a readout, written `readout:<name>`, which brings its per-observation error.

**Suggest a full set** fills in a broad, unbiased list:
- nuclear size, shape and roughness;
- integrated H2B (DNA content) and its texture;
- density, speed and age;
- the level and localisation of every other reporter;
- every defined readout.

Features aren't stored. They are computed from the saved measurements when needed (about 0.7 s for every feature of a
position), because a table of every feature for every cell and frame would be gigabytes.

**One scale, many experiments.** Each experiment is normalised to its own **reference condition**: 0 is the reference median
and 1 its robust spread (1.4826 × median absolute deviation). Features are named by what they measure, by default
`<reporter> <measurement>`, so experiments pool by name.

An experiment that imaged Geminin in the red channel has `Geminin nuc_mean`, and one that imaged the ERK KTR has `ERK`. When
both are loaded together (`features.load_features([exp1, exp2])`), each cell simply lacks the other's feature. Lineage comes
along: a track with a mother was born in the movie, and a track with daughters ends in a division.

**Fates** (`analysis.fates`; code in `fasttrack/analysis/fate.py`). A fate is the high state of a **marker** feature, for
example nuclear PPARG. The marker is smoothed (running median of `smooth_frames`). The threshold is set by hand or, by
default, placed at the valley of the marker's distribution between its two populations.

- **Committor.** For each cell and frame: the marker level now, and whether the marker is high `horizon_hours` later.
  - A bistable switch gives a steep curve.
  - The **commitment point** is where the curve crosses 0.5 (logistic fit). An intervention that makes commitment harder
    moves it right.
  - A cell's future follows its lineage: after a division it is the mean of her daughters.
  - Cell-frames whose track is gone by the horizon are not counted as "not committed"; their share is reported as *censored*.
- **Split by a second feature.** Cells are split by a feature averaged over the preceding `split_hours`, into groups by
  quantiles of the reference condition (e.g. low and high ERK). This asks whether that feature changes commitment *at the same
  marker level*.
- **Aligned to commitment.** Commitment is the first frame where the smoothed marker crosses the threshold and stays above
  for `hold_frames`, having been below as long.
  - Every feature is averaged around these events, and around the same frames in a **matched cell** that did not commit:
    from the same position, with the most similar marker over the window before, staying below the threshold, and observed
    both before and after.
  - **Match also on** (`match_on`): controls must also be close in these features over the window before. Use nuclear
    area and integrated H2B, so controls are at a similar cell-cycle phase. Without them, controls on the example data were
    older cells with more DNA.
  - **Exclude crossings** (`exclude_feature`, `exclude_above`, `exclude_frames`): leave out crossings where a feature is
    extreme close by. On the example data, 13% of crossings coincided with an H2B texture spike: at mitosis a condensed
    nucleus reads brighter in every channel.
  - A feature that departs from its matched cells *before* the crossing is a candidate for what precedes commitment.
  - Near the threshold, almost every cell crosses, so matched cells that never cross sit lower on the marker. Compare the
    other features, not the marker itself.
  - Values are clipped at ±10 robust spreads, so one bad mask can't dominate.
- **Aligned to division.** Features around observed divisions: the mother before, the mean of her daughters after.
- **Intervals.** 95% intervals come from resampling positions.

Fates need the frame interval: frames per hour on the Experiment tab, or minutes between frames under Dynamics.

**What a fate analysis saves**, in `<results>/Fates/<name>/`:
- `fate.json`: settings, threshold, counts, commitment points with intervals, normalisation;
- `fate.npz`: curves and aligned averages;
- `figures/`: threshold, committor, commitment points, aligned features, and a heatmap of every feature around the event.

On the example data (all 10 conditions, 176 positions), the run took 190 s on CPU. Settings: marker nuclear PPARG, split by the
ERK KTR, controls matched on nuclear area and integrated H2B, crossings at H2B texture spikes excluded. Frames were taken as
6 min apart (the file has no interval), so "12 h" means 120 frames.
- **Two populations.** Nuclear PPARG has two clear populations, 3.0 sd apart; the threshold sits at the valley (ln 5.17).
- **Commitment points** (ln nuclear PPARG, 95% over positions):

  | Condition | Commitment point |
  |---|---|
  | Ctrl | 4.65 [4.61, 4.68] |
  | MekI | 4.58 [4.56, 4.61] |
  | Gne | 4.65 |
  | Palbo | 4.75 |
  | Tdi | 4.90 [4.87, 4.93] |
  | Tdi + Palbo | 4.93 |
  | Tdi + MekI | 4.74 |

- **Track loss.** 33-65% of cell-frames lost their track before the horizon.

```bash
fasttrack fates experiment.yaml               # every ticked fate analysis
fasttrack fates experiment.yaml "PPARG"       # just this one
```

**Discovering cell states** (the `states` model on features; the Dynamics page). Tick many features instead of
readouts and the hidden Markov model looks for cell states without being told what they are. Then it measures how
interventions change their occupancy and the transitions between them.

- **One dictionary of states.** Each state's profile is shared by all conditions by default; rates and starting shares
  are per condition. So every condition is described with the same states, and a drug or knockdown can only change
  how often cells visit them and how they move between them.
- **Correlated features** (`rank`). Each state's spread is its own per feature plus `rank` directions shared by
  features. Nuclear area and DNA content then don't count twice as independent evidence.
- **Missing features.** A feature missing for a cell at a frame is left out of that observation exactly. So experiments
  with different reporters fit together, each cell using what was measured.
- **Divisions as observations** (`division`). Each state gets a division rate.
  - Every frame a cell goes on without dividing is evidence against dividing in its current state; an observed division
    is evidence for it. States with high division rates are the cycle's late phases (G2/M).
  - Newborn cells start from their own state distribution (post-mitotic), not the population's.
  - This needs features, because they carry the lineage.
- **How many states** (`sweep`, `heldout`). Each number in the list is fitted on most positions and scored on the
  positions left out; the best is then fitted on every position. More states always fit the fitted positions better,
  but only real structure carries over to unseen positions.
- **Routes** (`flux_feature`, or `flux_from` / `flux_to`). These go from the state lowest in a feature to the highest,
  e.g. undifferentiated to differentiated by PPARG.
  - The **committor** of each state is the chance a cell there reaches the target before going back to the start.
  - The **net flux** gives the expected passages along each edge on the way.
  - A state with an intermediate committor that carries much of the flux is a transition state.
  - The expected time to the target is given per condition.
- **Views:** state profiles (heatmap of every state's mean in every feature, dwell per condition, division rates), the
  transition graph per condition, routes, the comparison of state counts, and occupancy over time.
- **Speed.** With 16 states or fewer, the likelihood is computed as a balanced product of per-frame matrices: about
  10 steps for 600 frames instead of 600. It gives the same value as the forward recursion to float precision, and a
  training step on 4 conditions × 64 tracks × 600 frames × 13 features takes about 0.2 s on a 3090.

## 3. The GUI

```bash
fasttrack gui
```

The window has a sidebar on the left and the current page on the right. The sidebar lists every page under the
five stages of the workflow, in the order you use them: **Set up**, **Check & tune**, **Track**, **Analyse** and
**Publish**. Under each page's name, a short line and a dot show where that step stands: green done, amber needs
attention, an open circle not started. Optional pages (Label centres, Label events) are marked as such.
[UI_WORKFLOW.md](UI_WORKFLOW.md) explains the layout.

- **Home** is the first page: New, Open, recent files, and the five stages with each page's status.
- **Every page** has a header with what the page is for, a **How to use** button (or F1) that opens a guide to every
  control on it, and **Next →**, which goes to the next required step.
- **File** > New, Open, Save and Save as work on the experiment file. The sidebar shows the experiment's name and
  file, and Save turns blue when there are unsaved changes. File > Open recent lists recent files. Every page below
  edits part of the same file.

| Page | What to do there |
|---|---|
| **Experiment** | Name and notes; raw and results folders; filename pattern and frame range. **Scan data folder** reads the folder once, fills the channel table with the channel numbers found, and greys out wells without images on the Plate tab. Name each channel, pick its role, note the reporter. Add events (drug additions and the like) with their frame. |
| **Plate** | Add conditions (double-click to rename, Colour… to recolour, drag to reorder; the order is used in figures). Drag over wells to assign the selected condition. Switch the mode to exclude wells (hatched) or clear them. Choose whether to track only wells with a condition, and list single positions to leave out. |
| **Quality** | **Check positions** finds fibers, debris, bright blobs and odd backgrounds (section 2a). Flagged positions are listed first with the reason; tick *Track anyway* to override. Select a row to see its worst frame and the unusual tiles. |
| **Parameters** | Every tracking parameter. Hover a field for notes. Reset to defaults. |
| **Run** | **Find timestacks**: every position found, its condition, and whether it will be tracked (excluded and unassigned ones are listed but skipped). **Run** tracks them, then draws the figures if ticked. The log shows stage timings, the estimated travel gate and any missing frames. |
| **Test run** | Pick a condition and position (or Random batch), part of the movie or all of it, and Run: the full pipeline with the current settings, every intermediate in memory, outlines over the images. Nothing is saved. |
| **Viewer** | Pick a processed timestack: frames from the TIFFs, track tails, and outlines. Click a nucleus to plot its timecourse. **Preview segmentation here** segments 20 frames with the current parameters, without saving. |
| **Label centres** | Correct nucleus centres on patches picked from local cell-count changes, train the seed finder, use it for tracking (section 2b). |
| **Label events** | Label cell crops into your event categories with the keyboard, train the event classifier, classify every position (section 2c). |
| **Figures**, **Readouts**, **Features**, **Fates**, **Dynamics**, **Export** | The Analyse and Publish pages. **Export**: multi-panel figures at journal size, and templates (section 2c, "Comparing conditions"). **Readouts** and **Dynamics**: section 2d. **Features** and **Fates**: section 2e. **Figures**: the figure builder. *Groups*: named sets of conditions. *Gates*: thresholds on a metric. *Figures*: pick a kind, conditions, metric and frame window; selecting a figure previews it. **Save as new figure** keeps it in the file; **Draw all ticked figures** writes them to the results folder. *Settings*: track filters and output formats. |

Tools > Write a synthetic experiment writes synthetic images with ground truth into a new folder, writes an experiment file
for them, and opens it.

**Test run layers:** raw, background corrected, rough mask, binary mask, pseudo distance, centroids, nuclear labels, labels with
cytoplasm ring, border labels, undersegmentation score. Expect about 1 GB of RAM per 100 frames of 1024 px, 3 channels, and
about 30 s to run on dense late frames.

A good first review session: run test mode on the last 100 frames of a dense position. Compare *Background corrected*,
*Centroids* and *Labels and cytoplasm ring*, and step through looking for merged, missed and split nuclei.

## 4. Parameters you are most likely to change

| Parameter | Default | When to change it |
|---|---|---|
| `link.max_travel` | 0 = estimate | `run_log.yaml` records the estimate per timestack. Set it if cells clearly move further than that between frames. |
| `link.method` | `multipass` | `notebook` is your original single pass; `tracks` is the alternative voting linker. |
| `centroids.seed_overlap` | 1.75 | Raise it if single nuclei get two seeds; lower it if touching neighbours become one. Check in a test run. |
| `centroids.blur_sigma` | 1.5 | Your blur's sigma, fixed instead of random. Higher means fewer seeds. |
| `mask.threshold_scale` | 0.6 | Your notebook's rough-threshold factor. |
| `underseg.enabled` | true | The temporal split of merged nuclei. |
| `batch_size`, `batch_auto` | 100, true | Batches grow while GPU memory allows, and split on out-of-memory. Lower `batch_size` if the GPU is shared. |
| `device` | `cuda:0` | On your machine, GPU 1 drives the display. |

## 5. Reproducing the numbers

Each benchmark prints its tables and writes JSON. Results are reproducible with the same seed, except for tiny float differences
from GPU summation order.

Benchmarks also take no flags. Each reads `benchmarks/settings/<name>.yaml` (or a settings file given as the only
argument). A settings file names an experiment file, and the benchmark takes the raw folder and every tracking parameter
from it, so it measures exactly the analysis that file runs. Outputs go to `out`, relative to `refactor/`, and must be
outside the raw data. A missing settings file is written with the defaults.

### Lineage on synthetic data matched to a real dataset (the benchmark to trust)

Make an experiment file for the dataset (raw folder on the external drive, results on a local disk), name it in
`benchmarks/settings/matched_synthetic.yaml` with `random: 3`, then:

```bash
python benchmarks/matched_synthetic.py
```

For each picked position, it:
1. Tracks the real movie and summarises it.
2. Fits generator settings.
3. Generates a synthetic movie of the same length with full ground truth, and refines the fit once.
4. Prints real against synthetic statistics, flagging any off by over 30%.
5. Scores detection and every linker against ground truth.

Takes about 3–5 minutes per full-length position. **Read the comparison table before the scores**: a score only counts if the synthetic
movie resembles the real one. Settings land in `runs/matched/<id>_synthetic.yaml` and can be regenerated with
`fasttrack.synthetic.generate.generate(params)`.

Two lineage numbers are reported:

- `link_accuracy`: correct links among true links whose two cells were both detected. It can look good by merging cells.
- `true_link_recall`: correct links among all true links. Missed and merged cells count as failures. Use this one.

### Lineage on the fixed synthetic presets

```bash
python benchmarks/validate.py
```

Set `preset` to `mosaic`, `populated` or `dense` and `seeds` in `benchmarks/settings/validate.yaml`.

These compare variants (no seed merge, no second pass, the notebook linker, the voting linker) on presets roughly calibrated to
two wells. They are quicker than matched data, but less faithful to it.

### Speed and descriptive numbers on real wells

```bash
python benchmarks/random_wells.py
```

Picks random wells that the experiment's plate would track. Times the full pipeline per position, including reading. It reports track and division counts, but **these are not accuracy**.
Fast reporters and divisions change intensities legitimately, so no real-data number here can tell a correct link from a wrong one.

### Others

| Script | What it measures |
|---|---|
| `benchmarks/speed.py` | Batch time and peak memory at 1024 and 2048 px |
| `benchmarks/underseg_features.py` | Which measurements separate merged from single nuclei, by AUC, and a logistic fit |
| `benchmarks/event_classifier.py` | The event classifier on synthetic deaths and divisions, through the real files and labels path |
| `benchmarks/centroid_net.py` | Trains the ResNet seed finder on 40 labelled synthetic frames; compares with your seeds (the Label centres tab trains on patches instead) |

## 6. Reading the code

Suggested order, following the data:

1. `fasttrack/experiment.py`: the experiment file, and `pipeline_config`, which turns it into the flat dict below.
   `fasttrack/config.py`: every tracking parameter in one dict, with defaults and comments.
2. `fasttrack/track/run.py`: loader thread, batches, out-of-memory recovery, saving.
3. `fasttrack/track/batch.py`: one batch, end to end. `process_batch` lists every stage in order.
4. Your ported core:
   - `background.py`, `masking.py` and `centroids.py`: background, mask and seeds.
   - `labels.py`: local labels and the cytoplasm ring.
   - `measure.py`: `measure_subsections`, verbatim plus packaging.
5. `fasttrack/track/undersegmentation.py`: the temporal split.
6. `fasttrack/track/linking.py`: FFT shift, travel estimate, then `_link_multipass` with `_pass_divisions` and `_pass_cleanup`.
7. `fasttrack/synthetic/`: the generator, evaluation, and fitting to real data.
   `fasttrack/analysis/figures.py`: draws a figure spec from the experiment file.
   `fasttrack/plotting/style.py`: the figure standard as code (journal sizes, type, lines, palettes).
   `fasttrack/plotting/export.py`: multi-panel pages, captions, checks and templates.
   `fasttrack/plotting/panels.py`: layout of the fate and dynamics views for the space they get (grids, legends, caption notes).
8. `fasttrack/analysis/readouts.py` (readouts with per-observation error), `features.py` (features, lineage, normalisation, pooling experiments), `fate.py` (committor and event-aligned analyses), and `fasttrack/dynamics/` (`models.py` generators, `engine.py` likelihood and fitting, `fit.py` runs and saves a fit, `plots.py`).
9. `fasttrack/quality.py` (bad positions), `fasttrack/centres.py` (count-change candidates, centre labels, seed finder training) and `fasttrack/events.py` (candidates, crops, labels, classifier).
10. `fasttrack/gui/`: one file per page (`dynamics_tab.py` holds the Readouts and Dynamics pages and `AnalysisSection`, which reads and writes the analysis pages together; `features_tab.py` the Features and Fates pages; `export_tab.py` the Export page). `shell.py` is the window's frame (sidebar, page headers, Home, the help window), `help_text.py` the How-to-use guide of every page, `status.py` the sidebar's status lines, and `theme.py` the colours and stylesheet.

Conventions worth knowing before you change anything:

- **Axes:** arrays are (T, C, X, Y), and `x` is the first spatial axis, as in your notebooks.
- **Labels:** positive is nucleus, negative is its cytoplasm ring. Labels are local to a frame: `1000 * section + index`. A cell is
  identified by (t, label), never by label alone.
- **Shift sign:** `find_shifts` returns minus the content's motion, so aligned coordinates are `x + cumulative shift`. Your
  notebook's lineage cell subtracted it, which doubles a jump. A test guards this.
- **Rows:** measurement rows are ordered by frame. `trace_index` maps (track, frame) to a row, so any metric becomes a matrix by indexing.

[AUDIT.md](AUDIT.md) lists every change to your code and why.

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| "found 0 timestacks" | Filenames do not match the pattern on the Experiment tab. Test the regex on one filename. |
| A good position is flagged | Look at it on the Quality tab. Raise `z_threshold` or `area_threshold` and Rescore, or tick Track anyway. |
| "Wells ... do not exist on a 96-well plate" | A well such as J12 is listed but the plate format is 96. Fix the well or set the format to 384. |
| Warnings about groups naming unknown conditions | A condition was renamed or removed outside the Plate tab (renames there update the groups). Fix the group on Figures → Groups; tracking runs regardless. |
| "No timestacks to process" | The plate tracks only wells with a condition, and none of the found wells has one. Assign conditions or choose "Every well with images". |
| A timestack has fewer frames than expected | Its files stop early (or one channel's do). The scan and the run summary say by how many frames; see "Missing and broken files" in section 2. |
| "is the drive still connected?" | Every file of a batch failed and the files are gone. Reconnect the drive and run again: finished positions are skipped. |
| Out of GPU memory | It recovers by splitting the batch. A repeat on a batch of under 20 frames means the GPU is shared or the frame is huge. |
| Slow on a USB drive | Reading is the limit, at about 50 s per 576-frame position here. Tracking overlaps it. |
| Too many or too few seeds | Check in a test run, then adjust `centroids.seed_overlap` or `centroids.blur_sigma`. |
