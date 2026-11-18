# FastTrack review: state of the refactor

An honest assessment of each component: where it came from, how it was checked, and what is still unknown. Evidence levels:

- **Test**: covered by a unit or integration test (`tests/`).
- **Synthetic**: scored against ground truth on generated data.
- **Matched**: scored on synthetic data fitted to a real timestack, with the fit checked against the real statistics.
- **Real, by eye**: inspected on real data. No accuracy number is possible there.
- **Unverified**: implemented, not yet checked.

## Before running on real experiments (state on 2026-11-07)

**Ready to use**

- One experiment file per experiment, built in the GUI (Open / Save on the toolbar). It holds folders, channels, events,
  plate, every tracking parameter and every figure; `fasttrack run experiment.yaml` repeats an analysis exactly.
- Tracking from raw TIFFs, read-only on the data drive, results on a local disk. Robust to what real plates contain:
  unreadable or empty TIFFs become blank frames (listed in the summary), a position that fails is skipped and logged,
  sparse or crowded positions are warned about, and only problems that make tracking impossible stop a run.
- Logs per run in `results/Logs/`: a detailed log with every batch and timing, and a short summary of what to check by hand,
  including whether the run was limited by the disk or the GPU.
- Quality check before tracking (fibers, debris, odd backgrounds), with flagged positions opened at full resolution or
  in a test run from the Quality tab.
- Test run tab: the full pipeline on any position (part or whole movie), every intermediate in memory, outlines over the
  image, before committing to a run.
- Partial runs (tick positions or one condition), resumable: finished positions are skipped.
- Figures with conditions ticked per figure, wells as replicates by default, statistics against a reference, plain-language
  metric names with formulas, and per-figure plot settings. Fast on full plates (0.1-1.5 s per figure once loaded).

**Known limits, in order of how much they matter for a comparison**

1. **Lineage accuracy on your data is unknown.** On synthetic presets the linker is right ~98% of the time among detected
   cells; matched to real crowded positions the benchmark is not yet faithful (detection recall there 0.65-0.71 is a
   generator artefact). Division calls are the weakest part (precision 0.3-0.5). Expect fewer, longer tracks than the
   notebook's single pass, and treat division counts with care until checked against labels.
2. **Throughput is disk-bound and not yet streamed across positions.** Within a position reading overlaps the GPU; between
   positions linking and saving still run on the tracking thread, so the disk idles for a few seconds per position. The
   summary reports the disk speed and GPU wait per run, so this is measurable on your drives. A ~1 TB experiment from a USB
   HDD will take roughly as long as reading 1 TB from it.
3. **Thresholds of the quality check** were set on two datasets (ExampleData, 60 HDD positions). Check its first flags.
4. **Event classifier and labeller** are tested on synthetic data only; train them on your own labels before trusting counts.
5. **Analysis** covers time courses, heatmaps, % passing a gate, distributions, cell counts and event rates. Lineage plots,
   single-cell traces in figures, and FRET/KTR presets are not built yet.

**Suggested workflow for each experiment**

1. Open or convert its old settings YAML, set raw and results folders, scan, name channels, check the plate.
2. Quality tab: Check positions, look at the flagged ones (Open images), keep any that are fine.
3. Test run: two or three positions (Random batch), look at outlines in late, crowded frames. Adjust parameters if needed.
4. Run. Read the summary when it finishes.
5. Analysis: figures; Draw all writes PDFs, PNGs and CSVs with the experiment file that made them.

**Comparing with the existing pipeline**

The fair comparisons, given that no ground truth exists on real data:

- **Per-condition population curves** (median nuclear intensity, % above a threshold, cell counts per frame): these should
  agree closely, since segmentation and measurement are your code. Disagreement points at the background or mask fixes listed
  in AUDIT.md, or at different frames or wells being included.
- **Detections per frame** per position: the same order of magnitude, with differences where the nested-seed merge and the
  temporal split change merged or split nuclei. Look at those in a test run.
- **Track length distributions and division counts**: expected to differ (multi-pass linker, cleanup pass, max-travel gate);
  this is the part to judge on a few human-checked movies, not on agreement.

## Summary (benchmarks, earlier)

The pipeline core is your notebook code, repackaged with only the fixes it needed to run correctly. Around it are the parts
that did not exist: batching for long movies, a multi-pass linker, a temporal split for merged nuclei, test mode, a GUI, and
synthetic data with ground truth. A full 576-frame real position takes about 20 s of compute with the files cached, and about
60–90 s from the USB drive, where reading is the limit.

Lineage accuracy can only be trusted on synthetic or human-annotated data. Fast reporters and divisions change intensity
legitimately, so real-data consistency checks cannot tell a correct link from a wrong one. Among detected cells, synthetic
link accuracy is about 98%. Counting missed and merged cells as failures, it is about 88–94% on the presets, and the gap is
mostly detection, not the linker. Division precision, 0.38–0.50 on the presets, is the weakest number.

**Matched benchmark on 3 random full-length positions (576 frames): the fit is not yet close enough to trust the scores.**
11–21 statistics stayed off by over 30% after refinement. The main gap is late, crowded frames. Real movies show 800–900 detected
nuclei per frame; the synthetic ones contain 800–1,100 true nuclei, but the pipeline detects only about 430 of them (detection
recall 0.65–0.71). So the generator's nuclei overlap more than real ones, and its scores in crowded frames are pessimistic, not
representative. The dim third channel is also poorly matched (level and spread off by 2–3×).

| Position (matched, refined once) | Detection recall | True link recall | Link accuracy | Division precision / recall |
|---|---|---|---|---|
| C05_xy2 (sparse, 9–17 cells) | 0.99 | 0.964 | 0.978 | 0.08 / 0.69 |
| G02_xy3 (17 to about 800 cells) | 0.71 | 0.654 | 0.974 | 0.28 / 0.45 |
| H06_xy6 (12 to about 900 cells) | 0.65 | 0.595 | 0.973 | 0.29 / 0.41 |

These are multi-pass numbers; the notebook pass and voting linker results are in the benchmark output. Among detected cells, linking
stays at about 97% even in crowded frames. The next fix belongs in the generator: fit how closely nuclei pack (nearest-neighbour
distances) instead of using fixed overlap settings, so crowded frames match.

## Components

| Component | Origin | Evidence | Status and risks |
|---|---|---|---|
| File discovery (`io/layout.py`) | New | Test, real (80 wells) | Missing timepoints are dropped and reported. Assumes one folder per well and a regex for names. |
| Frame reading | New | Real | PIL's decoder, 22× faster than `tifffile` here. On the USB drive, about 50 s per 576-frame position. |
| Loader and batches (`track/run.py`) | New | Test, real | Separate loader thread; batches grow with spare GPU memory, and split on out-of-memory. Growth uses the first batch's memory, which underestimates dense late frames, so the split does the work then. |
| Stage shift | Yours, fixed | Test, synthetic, real | Your FFT, on the binary mask instead of the corrected image, with the sign fixed. Real stage steps at frames 100 and 345 are found across wells, and do not move when batch boundaries move. Occasional false small steps. |
| Background | Yours | Test, real | Quadratic in log2, per frame. Flatness of background tiles on real data: 11–45 counts spread, with a level of 25–38 counts left in, about half of a dim reporter's signal. The bright rim seen in B04 c2/c3 was not investigated. |
| Mask | Yours | Real, by eye | Rough threshold, then your LoG cleanup, unchanged, LoG kernel included (size and sigma must change together). |
| Seeds | Yours, plus one filter | Synthetic, real by eye | Your distance peaks at a fixed sigma, then nested seeds merged. That filter is the biggest single accuracy gain: link accuracy 0.966 to 0.985 on the mosaic preset. |
| Labels and ring | Yours | Test | Local labels, `-label` cytoplasm ring, border labels. At most 999 cells per frame in one sub-section. |
| Measurement | Yours | Test (against brute force) | `measure_subsections` verbatim, plus two fixes (int32, a reshape). About 0.5 s per batch. |
| Temporal split (second pass) | New | Test, synthetic | Vectorised over the batch: 42 s to 2.5 s on the densest real movie. Raises true-link recall, for example 0.83 to 0.89 on the populated preset. |
| Undersegmentation score | From your metrics | Synthetic | Saved per cell and viewable, but **saturates on real data**: its weights came from synthetic data with earlier seeds. Refit on labelled real data before using it. |
| Multi-pass linker | Your design | Test, synthetic, matched | Pass 1 (your backward pick, top child per mother), pass 2 (division evidence, vectorised), pass 3 (one-to-one cleanup over gaps). The travel gate is estimated per movie, or set by you. |
| Voting linker | New, optional | Synthetic | About as accurate as multi-pass, with slightly better division precision, but 3–24 s per real movie instead of 1–3 s. Kept for comparison. |
| ResNet seed finder | New, optional | Synthetic | Trained by script; beat the earlier seed finder on synthetic data. Not re-run against the current seeds. No labelling tool yet. |
| Storage | New | Test | One HDF5 per timestack, written atomically. `trace_index` turns any metric into a (tracks × frames) matrix. |
| Test mode and Inspect tab | New | Test, real | Every intermediate in RAM: about 1.1 GB per 100 frames. Driven by script on real data, not yet used by hand. |
| Experiment file and GUI | New | Tests, scripted GUI runs, ExampleData from the file alone | One YAML holds folders, channels, events, plate, every parameter and the figure list; load, edit in every tab, save and reload are lossless (tested). Old notebook YAMLs and plate layouts convert. Plate picks wells to track. |
| Quality check | New | ExampleData (fibers flagged), 60 random HDD positions (none flagged), synthetic fiber test | Per-tile floor and median compared across positions; flags, reasons and overrides stored in the file. Thresholds were set on these two datasets only. |
| Event labeller and classifier | New | Synthetic deaths and divisions through the real path; unit tests | Works end to end; on synthetic data with 40 labels per class: division recall 0.98 / precision 0.67, death 0.90 / 0.45. Not yet used as division evidence by the linker. Not tested on real labels. |
| Figure builder | New | Rendered on synthetic and ExampleData | Time course, heatmap, % passing a gate, distributions (violin, box, bar), cell counts; ratios (FRET, KTR with background floor); event lines and time zero. Replicates (position, well) with mean ± s.e.m.; Welch or Mann-Whitney against the reference, Holm-adjusted. |
| Synthetic generator | New | Test | Physics as you specified, plus fast switches, C/N dynamics, changing division pace, exact stage steps. Overlap and texture models are guesses. |
| Fitting to real data (`synthetic/fit.py`) | New | Matched | Fits shape, motion, density over time, channel levels and dynamics, background, camera noise and stage steps. Refines once. See the matched results above for how close it gets. |

## What is not verified

1. **No human-annotated ground truth yet.** Every accuracy number comes from synthetic data, made by a model I wrote. The labelling
   tool is the way to fix this.
2. **Divisions.** Precision of 0.38–0.50 means about half of declared divisions are wrong on synthetic data.
3. **Late, crowded frames.** Real movies reach 500–900 cells per frame. The populated preset approximates this, but its overlap
   model is a guess.
4. **Real overlap and merge behaviour.** The temporal split helps on synthetic data; on real data it has only been looked at.
5. **Bad wells and anomalies** (fibers, aggregates) are not detected yet.
6. **The GUI has not been used by hand** on your display. Everything was driven by script, offscreen.

## Changes from your code

Every one is listed in [AUDIT.md](AUDIT.md). In short:

- The shift sign, the shift input, and the random blur sigma.
- The `int16` cast and the `_fill_local_density` reshape.
- The NaN tiles in the background fit.
- The lineage cell's argmin axis and its no-candidate case.

## Suggested next steps

1. A labelling tool: real ground truth for seeds, events and tracks, and training data for the event classifier.
2. An event classifier as division evidence in linker pass 2: the most promising route to better division precision.
3. One experiment file, edited in the GUI: plate layout, conditions, colours, channels, paths, exclusions.
4. Anomaly detection with default exclusion and manual override.
5. A figure builder.
