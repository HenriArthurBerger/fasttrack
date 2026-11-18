"""The How-to-use guide for every page, as Markdown, and a small Markdown-to-HTML converter for the help window.

Each guide follows the same order: what the page is for and where it sits in the workflow, the steps, every
control, what is saved, then tips. Keep them in step with docs/GUIDE.md when a page changes.
"""
from __future__ import annotations

import html
import re

from . import theme

HELP: dict[str, str] = {}

HELP["home"] = """
FastTrack turns a plate of timelapse movies into tracked cells, and then into figures and models of what the cells
did. Everything you set is stored in **one experiment file** (YAML). Save it with [[Ctrl+S]]. `fasttrack run
experiment.yaml` on the command line does exactly what the app does with that file.

## The workflow

Work down the sidebar. Each stage needs the one before it, and inside a stage you can go back and forth.

| Stage | Pages | You are done when |
|---|---|---|
| **1 · Set up** | Experiment, Plate | the folders are set, the channels are named, and every well you want has a condition |
| **2 · Check & tune** | Quality, Test run, Parameters, *Label cells* | a test run on a dense, late part of the movie shows few merged, missed or split nuclei |
| **3 · Track** | Run, Viewer | every position is *Done*, and tracks in the Viewer follow the cells |
| **4 · Analyse** | Summary, Figures, Readouts, Features, Fates, Dynamics | you have the figures and fits you need |
| **5 · Publish** | Export | the page check shows no problems and the PDF is written |

*Italic* pages are optional. They improve a step but nothing needs them.

## The sidebar

- The **dot** next to each page shows where it stands: green done, amber needs attention, open circle not started.
- The **line under the name** gives the details, for example "10 conditions · 30 wells" or "2 flagged".
- **Next →** at the top right of every page goes to the next required step.

## Getting started

1. **New experiment**, or **Open…** a file (older notebook settings files are converted when opened).
2. On **Experiment**, choose the raw data folder (only ever read) and a results folder outside it, then press
   **Scan data folder**.
3. Follow the sidebar from there.

To try the app without your own data, use **Try it on synthetic data…**. It writes a small synthetic movie with
known ground truth and opens an experiment file for it.

> Every page has a **How to use** button at its top right, and [[F1]] opens the same guide.
"""

HELP["experiment"] = """
**What this page is for:** tell FastTrack where the images are, how the files are named, what each channel is, and
when things happened during the movie. It is the first step; nothing else works until the raw folder and the
channels are set.

## Steps

1. **Name** the experiment and add **notes**: cells, reporters, treatments, anything worth keeping with the analysis.
2. **Folders.** Choose **Raw data**, which is only ever read and never written to. Choose **Results**, which must
   be outside the raw folder.
3. **Filename pattern.** The default reads names like `B02_Time0001_xy1c2.tiff`. Change it only if your files are
   named differently (see below).
4. Press **Scan data folder**. It reads the folder once, fills the channel table with the channels it found, greys
   out wells without images on the Plate page, and lists missing frames.
5. **Channels.** Give each a short name and pick its role. Write the reporter in *Reporter / marker*: it names
   features (e.g. "PPARG nuc_mean") and lets Readouts suggest KTR readouts.
6. **Time.** Set **Frames per hour** if you know it, and add **events** (drug additions and the like) with their frame.

## Controls

| Control | What it does |
|---|---|
| Filename pattern | A regular expression with named groups `well` (optional), `t` (optional), `pos` and `chan`. Hover it for an example. |
| Frames from / to / every | The part of the movie to use. *to* is exclusive, *end* = the last frame. Frame numbers always count from the start of the full movie, so events keep their frames. |
| Reader threads | Parallel file reads. 16 suits a USB disk; more rarely helps. |
| Scan data folder | Reads the folder, fills the channels, finds missing or short positions. If some positions lack only their last 1–3 frames, it offers **End every position at frame N**. Nothing changes until you press it. |
| Primary measurement | What the reporter is read out as: nuclear mean, cytoplasm / nucleus (KTR), integrated intensity (DNA)… Summary figures use it. *auto* picks from the reporter's name. |
| Channel role | **nuclear**: segmentation runs on it. There must be exactly one. **measure**: measured in every cell. **skip**: never read. Every channel that is read is measured, the nuclear one included. |
| Colour | The channel's display colour in the viewers. |
| Frames per hour | Only labels figure axes in hours, and gives Fates and Dynamics their time unit. Tracking never uses it. |
| Events | A name, the frame it happened at, and a colour. Figures can mark events and put time zero at one. |

## What is saved

The `experiment`, `paths`, `files`, `channels` and `time` sections of the experiment file.

## Tips

- An empty channel table means: every channel is read, and the first is nuclear.
- **Missing frames** stay in place as blank frames, so frame *i* is the same time in every position. Tracks bridge a
  single blank frame.
- If the raw folder is on a USB disk that goes to sleep, the first scan can take a few seconds.
"""

HELP["plate"] = """
**What this page is for:** say which wells are which condition. Conditions decide what is tracked (by default only
wells with a condition), how figures group cells, and every colour in every figure.

## Steps

1. Type a condition name (e.g. `DMI MekI`) and press **Add**.
2. With the condition selected, **drag over wells** (or [[Ctrl]]-click) to assign them.
3. Repeat for every condition. **Drag conditions in the list** to set their order. Figures use this order, and the
   first condition is the usual reference.
4. Optionally switch *Dragging over wells will* to **exclude** wells (hatched) you never want, such as a well with
   a bubble.

## Controls

| Control | What it does |
|---|---|
| Rename / double-click | Renames the condition everywhere it is used: figures, groups, fates, fits and export panels. |
| Colour… | The condition's colour. Figures can use these, or a colour-blind-safe palette (Figures → Settings). |
| Remove | Removes the condition; its wells become unassigned. |
| Dragging over wells will | **assign** the selected condition, **exclude** (toggle, hatched), or **clear**. |
| Wells | *Only wells with a condition* tracks just those; *All wells found* tracks every well with images. |
| Plate | The plate format: 6, 12, 24, 48, 96 or 384 wells. |
| Excluded positions | Single positions to leave out, as timestack ids, e.g. `C05_xy2, D03_xy1`. |

Grey well names have no images in the data folder (after a scan on the Experiment page). The summary under the
settings counts the wells per condition.

## What is saved

The `plate` section: conditions (name, colour, wells), excluded wells and positions, and what to process.

## Tips

- Excluding a well or position later doesn't need re-tracking. Figures leave it out the next time they are drawn.
- For colour-blind-safe figures, keep the control first. In the colour-blind palettes it is drawn dark grey.
- Combination names such as `Tdi + MekI` let the *Combinations* palette mix the colours of their parts.
"""

HELP["quality"] = """
**What this page is for:** find positions whose images are not usable (a fiber across the field, debris, a
bright out-of-focus blob, an odd background) **before** spending GPU time tracking them. Flagged positions are left
out of tracking and figures unless you say otherwise.

## Steps

1. Press **Check positions**. It reads a few evenly spaced frames of every position the plate selects (6 by
   default), about 0.6 s per position from a USB disk.
2. Flagged positions are listed first, with the reason. **Select a row** to see its worst frame with the unusual
   tiles outlined.
3. For a position that looks fine to you, tick **Track anyway**.
4. Use **Open images** (or double-click) to step through its bad frames at full resolution, or **Open in Test run**
   to run the pipeline around the first bad frame.

## How the check works

Each channel is cut into 16 × 16 tiles. For each tile it keeps the background floor (1st percentile) and the
median, relative to the frame's typical tile, so overall brightness and density differences between conditions
don't count. A random sample of positions defines each tile's normal value.

- A **tile** is unusual beyond 6 robust deviations. A brighter median doesn't count, because that is a dense colony.
- A **frame** is bad if unusual tiles cover over 2% of it, or its whole background is far off.
- A **position** is flagged if a third or more of its sampled frames are bad. An isolated bad frame is listed but
  tracked.
- A position is also flagged when it has more missing or unreadable frames than allowed.

## Controls

| Control | What it does |
|---|---|
| Skip flagged positions | Untick to ignore the check entirely. |
| Settings | Frames per position, reference positions, tiles, thresholds. The defaults suit most plates. |
| Rescore | Applies changed thresholds to the frames already read, without reading them again. |
| Clear result | Forgets the check, so the next run checks again. |
| Outline unusual tiles | Draws the tiles that made a frame bad. |

## What is saved

The `quality` section: the settings, the flagged positions with reasons, and the ones you keep. A rerun of the file
makes the same choice. Details are in `quality_report.yaml` in the results folder.

## Tips

- If you skip this page, Run does the check first anyway.
- On the example data the check flags only the two A02 positions with a drifting fiber.
"""

HELP["inspect"] = """
**What this page is for:** run the full pipeline on part of one position with the current settings, and look at
every step: background correction, masks, seeds, labels and tracks. Nothing is saved. Use it to check settings
before tracking the plate, and come back after each change on the Parameters page.

## Steps

1. Pick a **condition** and **position**, or press **Random batch** for a random position and start frame.
2. Choose the frames, or tick **Whole movie**. Start with about 100 frames from late in the movie, where cells are
   densest.
3. Press **Run**.
4. Step through the frames with the slider, or [[←]] [[→]] (one frame) and [[↓]] [[↑]] (ten frames). Compare **Background corrected**, **Centroids** and
   **Labels and cytoplasm ring**, and look for merged, missed and split nuclei.
5. Change something on **Parameters**, then run again.

## Views and layers

**Add view** puts several layers side by side; they zoom and pan together. Each view shows one layer of one channel:

- raw and background corrected images;
- rough mask, binary mask and pseudo distance;
- centroids (the seeds);
- nuclear labels, labels with the cytoplasm ring, and border labels;
- the undersegmentation score.

| Control | What it does |
|---|---|
| Low / high, Auto | The contrast of a view. Auto uses the 1st to 99.8th percentile. |
| Outlines | The final nuclei drawn over the image views. |
| Seeds | The seed points over the image views. |
| Track tails | Where each cell was over the last frames. |
| Frame / Jump | Go to a frame number. |

## Tips

- Expect about 1 GB of RAM per 100 frames of 1024 px with 3 channels, and up to about 30 s for dense late frames.
- **Two seeds in one nucleus** (it gets cut in two): raise `centroids.seed_overlap` or `centroids.blur_sigma`.
- **Touching nuclei merged:** lower `centroids.seed_overlap`.
- If seeds are often wrong, the learned seed finder (Label cells) usually fixes it.
- The Quality page's **Open in Test run** opens a flagged frame here.
"""

HELP["parameters"] = """
**What this page is for:** every tracking parameter, saved in the experiment file. Most never need changing.
Change one at a time and check the result on the **Test run** page.

## Parameters you are most likely to change

| Parameter | Default | When to change it |
|---|---|---|
| `link.max_travel` | 0 = estimate | Each run records the estimate per position in `run_log.yaml`. Set it if cells clearly move further than that between frames. |
| `link.method` | `multipass` | `notebook` is the original single pass; `tracks` is the alternative voting linker. |
| `centroids.seed_overlap` | 1.75 | Raise it if single nuclei get two seeds; lower it if touching neighbours become one. |
| `centroids.blur_sigma` | 1.5 | Higher means fewer seeds. |
| `centroids.method` | classic | `cnn` uses the learned seed finder (set by *Use seed finder for tracking* on Label cells). |
| `mask.threshold_scale` | 0.6 | The rough threshold factor. |
| `underseg.enabled` | true | The temporal split of nuclei that merge for a few frames. |
| `batch_size`, `batch_auto` | 100, true | Batches grow while GPU memory allows. Lower `batch_size` if the GPU is shared. |
| `device` | `cuda:0` | Which GPU to use. |
| `output.overwrite` | false | Retrack positions that are already done (after changing settings). |

## Controls

- **Hover a field** for what it does.
- **Reset to defaults** sets every parameter back to its default.

## What is saved

The `pipeline` section. Every key is written, defaults included, so a later change to a default can't change an
old analysis.

## Tips

- Changing settings doesn't retrack finished positions. Run warns about this, and the old experiment file is kept
  next to the results. Turn on `output.overwrite` to redo them.
- The sidebar shows how many parameters differ from the defaults.
"""

HELP["centres"] = """
**What this page is for (optional):** one labelling pass teaches FastTrack two things. **Centres**: seeds, one point
per nucleus, decide everything after them (two seeds cut a nucleus in two, none loses it for a frame). **States**:
what each cell is doing, such as dividing, dying or abnormal. You click both on the same patches. One dataset trains
two models: a seed finder for tracking and a state classifier that scores every track end, start and division.

## Steps

1. Track a few positions first (Run).
2. Press **Start labelling**. A patch opens at a likely mistake or event, with tracking's centres in blue.
3. Make sure every nucleus whose centre lies **inside the yellow box** has exactly one point:
   - click a point to remove it, click a nucleus without one to add a point (pink);
   - press [[1]]–[[9]] with the crosshair on a point to give it that state (or add a point there with that state).
     Points are the first state (e.g. neutral) unless you change them.

   The margin around the box is only context, and an empty box is a valid answer.
4. Use [[←]] / [[→]] (or click the film strip) to flip through at least 3 frames before and after. An odd angle that
   looks like two nuclei is one if it stays one; a division splits and stays split. [[↑]] returns to the labelled
   frame, the only one you edit.
5. [[Enter]] saves and moves on, [[Space]] skips, [[Backspace]] goes back, [[Ctrl+Z]] undoes. Aim for 100+ patches
   and 30+ cells of each state.
6. Press **Train both**. The report compares tracking's seeds and the seed finder's with your clicks on held-out
   patches, and gives the state classifier's held-out accuracy and confusion matrix.
7. Press **Check labels** now and then: see Review below.
8. **Use seed finder for tracking** (check it in a Test run first), and **Classify all positions** for the states.

## Dataset

Every labelled patch, or every labelled cell grouped by state, so you can see whether your divisions look like
divisions. Double-click one to open it on the Label page, correct it, and press [[Enter]].

## Review

**Check labels** splits the labels into folds and judges each fold with models trained on the others, so every
label is judged by a model that never saw it. The list shows patches where the seed finder's centres differ from
yours, then cells where the classifier's state differs, worst first. Each is either a weak spot of the model or a
label to fix. Double-click to open it: the model's centres are orange squares. Fix it and press [[Enter]], or press
[[Enter]] to keep yours.

## Where it looks

| Source | Why |
|---|---|
| Count changes | The local cell count changes for a frame or two and comes back: a nucleus split, missed or merged. Divisions do not count, because the count stays up. |
| Random cells | Ordinary nuclei, so both models see the common case. |
| Track ends / starts / divisions | An event cell in the centre, so rare states turn up often enough to train on. |

| Control | What it does |
|---|---|
| Box size / Margin / Frames before and after | The area you label, its context, and how far the film strip reaches (at least 3). |
| Count over / Compare with frames / Smallest change | How count changes are found. |
| States | Comma separated; keys 1–9 follow the order; the first is the default. |
| Classifier frames ± / crop / channels | What the state classifier sees around each point (empty channels = the nuclear one). |
| Steps, Match distance, Check folds | Training length, how close a found centre counts as the same nucleus, and how many folds Check labels uses. |

## What is saved

- Your labels, centres and states: `Labels/centres.json`. Earlier event labels (`Labels/events.csv`) are copied in
  once as single-cell labels; opening one and saving labels its whole patch.
- Seed finder: `Labels/centre_model.pt` (+ `.json` report). State classifier: `Labels/event_model.pt` (+ `.json`).
- Label check: `Labels/review.json`. Classified events: `Events/<position>.csv`.
- **Use seed finder for tracking** sets `centroids.method: cnn` on Parameters.
"""

HELP["run"] = """
**What this page is for:** track every selected position from the raw images, in the background, and then draw the
experiment's figures. You can stop at any time. Finished positions are skipped the next time, so a later run
completes the plate.

## Steps

1. Press **Find timestacks**. It lists every position, its condition, and whether it will be tracked:
   - *Waiting*: will be tracked;
   - *Done*: already tracked;
   - *Excluded*, *No condition* or *Flagged by quality check*: skipped.
2. Untick positions to leave for later, or use **Tick only this condition**.
3. Press **Run**. The table fills in cells, tracks, divisions and seconds per position, and the log shows stage
   timings, the estimated travel gate and any missing frames.
4. When it finishes, press **Run summary**. It lists only what to check by hand: flagged or failed positions,
   unreadable files, sparse or crowded positions, large stage jumps, and whether the disk or the GPU was the limit.

## What stops a run, and what does not

Only problems that make tracking impossible stop it:

- no raw or results folder, or results inside the raw folder;
- not exactly one nuclear channel, or duplicate channel names;
- a well in two conditions.

Everything else is a warning in the log, and the run carries on. A position that fails outright is skipped and the
rest of the plate is tracked.

## What a run writes, in the results folder

| File | What |
|---|---|
| `Measurements/<id>.h5`, `Traces/<id>.h5` | every cell's measurements and tracks |
| `Logs/<time>_summary.txt` | what to check by hand (shown by Run summary) |
| `Logs/<time>_detailed.log` | everything that happened |
| `run_log.yaml` | per position: travel gate, nuclear radius, counts, timings |
| `experiment.yaml` | the file as run; an older copy is kept if settings changed |

## Tips

- Tick **Draw the experiment's figures afterwards** to draw the ticked figures (Figures page) when tracking ends.
- Run uses the saved settings of every page, saved file or not. Save the experiment so you can rerun it later.
"""

HELP["viewer"] = """
**What this page is for:** look at tracked positions: the raw frames with track tails and nuclear outlines, and the
timecourse of any cell you click. Use it to check that tracks follow the cells before you analyse them.

## Steps

1. Pick a **Timestack**. Positions appear once they are tracked.
2. Pick a **Channel** and set the contrast (**Low / high**, or **Auto**).
3. Move through time with the slider.
4. **Click a nucleus** to plot its timecourse under *Selected cell*.

## Controls

| Control | What it does |
|---|---|
| Track tails / Tail frames | Where each cell was over the last frames. |
| Nuclear outlines | The nuclei from the latest run, or from a preview. |
| Preview segmentation here | Segments 20 frames around the current one with the current parameters, without saving. A quick check after changing a parameter. |

## Tips

- Scroll to zoom, and drag to pan.
- Broken tracks show as a tail that starts again in the middle of the field. If you see many, check
  `link.max_travel` on Parameters and run a Test run.
"""

HELP["summary"] = """
**What this page is for:** the routine figures you'd otherwise make by hand for every experiment. They are drawn
for you, in the background, after every run:

- cells per position over time;
- nuclear area over time;
- for every channel, its **primary measurement** over time, as a population heatmap, and as a distribution over the
  last frames of the movie.

They are written to `Figures/Summary` in the results folder, with the numbers behind each one as CSV.

## Primary measurements

Each channel has one, set in the channel table on the **Experiment** page:

| Primary measurement | Use it for |
|---|---|
| Nuclear mean | a nuclear reporter, e.g. PPARG |
| Nuclear median | the same, robust to a few bright pixels |
| Integrated nuclear intensity | DNA content (H2B, Hoechst) |
| Cytoplasmic mean | a cytoplasmic reporter |
| Cytoplasm / nucleus | a kinase translocation reporter (KTR), e.g. ERK KTR. log2, background-corrected |
| Nucleus / cytoplasm | a reporter that moves into the nucleus when active |
| None | no summary figures for this channel |

**auto** picks cytoplasm / nucleus when the reporter's name contains "KTR", integrated intensity for H2B, Hoechst
or DAPI, and the nuclear mean otherwise. The menu shows what auto picks for each channel.

## Steps

1. Set each channel's reporter and primary measurement on the Experiment page.
2. Choose what to draw, the **replicates** and the **error band** (see below).
3. The gallery redraws itself whenever these settings or the experiment change. **Draw now** redraws and writes the
   files.
4. Click a figure to see it full size. **Copy to my figures** adds it to the Figures page, where every setting can
   be changed (conditions, compare mode, axis ranges…).

## Error bands

| Band | Centre | Shows |
|---|---|---|
| Quartiles (IQR) | median | where the middle half of the values lie |
| 10th-90th percentile | median | the spread, including most of the tails |
| SD | mean | how much values vary |
| s.e.m. | mean | how well the mean is known (about a 68% interval) |
| 95% CI | mean | how well the mean is known, at 95% (Student t) |
| automatic | | quartiles over pooled cells, s.e.m. over replicates |

With **wells** or **positions** as replicates, each replicate is summarised by its median first. The band is then
over those values. s.e.m. and CI over pooled cells are always tiny, because cells of one well aren't independent,
and the figure notes this.

## Tips

- Untick **Draw after every run** to keep the gallery but write nothing to the results folder.
- With more than 4 conditions, time courses are drawn as small multiples with the first condition in grey behind.
- The command line draws them too: `fasttrack figures experiment.yaml`.
"""

HELP["figures"] = """
**What this page is for:** quick figures of the tracked cells, per condition: time courses, heatmaps of the
population, % of cells passing a gate, distributions, cell counts and event rates. Selecting a figure previews it.
Saved figures are drawn into the results folder after each run.

## Steps

1. Pick a **Kind** and a **Metric** (e.g. `nuc_median` of PPARG, or a ratio such as cytoplasm over nucleus for a KTR).
2. Tick the **Conditions** to draw and drag them into order, or pick a saved **group**. The first is the reference.
3. Choose how to **Compare**:
   - all on one plot;
   - one plot each with the reference in grey behind;
   - one plot each, alone.
4. Choose **Replicates**: what one point is (cells pooled, or each position or well summarised first). Then the
   **Error band**: quartiles, 10th-90th percentile, SD, s.e.m., 95% CI or none. The Summary page's guide explains
   when to use which.
5. Press **Preview**. When it looks right, press **Save as new figure**. **Draw all ticked figures** writes every
   ticked one to the results folder.

## The sub-pages

| Sub-page | What it holds |
|---|---|
| **Figures** | The figure list and the editor. |
| **Groups** | Named sets of conditions, e.g. "MekI series". The first is the reference. |
| **Gates** | A threshold on a metric, e.g. PPARG high, used by *% passing* and drawn on heatmaps. |
| **Settings** | Track filters (minimum length, border), output formats and size, condition colours. |

## Figure kinds

| Kind | Shows |
|---|---|
| timecourse | the median and quartiles per condition over time |
| heatmap | the population over time: each column is the distribution of that frame's cells |
| track_heatmap | one row per cell, sorted by its late value |
| percent_high | the % of cells passing a gate |
| distribution | the per-cell mean over a frame window, as a violin, box or bar |
| cell_count / event_rate | cells per position, or classified events per 100 cells |

## Statistics

With replicates set to position or well, each replicate is summarised first. Time courses then show the error band
across replicates: s.e.m. unless you pick another. Bar charts use the same band for their error bars. **Test against the first condition** uses Welch's t-test on replicate values, or Mann-Whitney on
cells, with Holm-adjusted p-values in the figure's CSV.

## What is saved

The `figures` and `analysis` (groups, gates) sections. Each drawn figure is written in every format, with a CSV of
the numbers drawn.

## Tips

- *Plot settings* (title, axis labels and ranges, legend, font) only change this figure.
- For a paper, copy figures into a page on **Export** instead of tuning sizes here.
"""

HELP["readouts"] = """
**What this page is for:** say what a reporter measures, for the models on the Dynamics page. A readout is a kind
applied to a channel (an ERK KTR, a nuclear reporter, an intensity, speed). Each value comes with its own error,
estimated from the pixels it was measured on, so models don't mistake measurement noise for biology.

## Steps

1. Press **Suggest from channel reporters** to add a KTR readout for every channel whose reporter name contains
   "KTR". Or fill in the editor by hand.
2. Press **Preview** to see the values, their errors, and example tracks from a few positions.
3. Press **Save as new** to keep it (**Update selected** changes the selected one).

## Kinds

| Kind | Value |
|---|---|
| ktr | ln(cytoplasm / nucleus), above the per-frame background floor |
| nuclear_reporter | ln(nucleus / cytoplasm) |
| intensity | ln(nuclear mean) |
| speed | px per frame, with stage drift removed |
| metric | any figure metric |

## Controls

| Control | What it does |
|---|---|
| Background percentile | The per-frame floor subtracted from the channel. |
| Largest error | Drops single observations whose error is larger. |
| Smallest mean signal | Drops cells that barely express the reporter. |

## What is saved

`analysis.readouts` in the experiment file. Values are computed when needed, never stored.

## Tips

- On the example ERK KTR data, the median error of ln(C/N) is 0.34, which is most of the frame-to-frame jitter.
- A readout can also be used as a feature (`readout:<name>`), carrying its error with it.
"""

HELP["features"] = """
**What this page is for:** the measurements that discovery analyses use: the states model on Dynamics, and the
aligned features on Fates. A feature is one measurement with a transform. All features are normalised to the same
reference condition, so experiments can be pooled by feature name.

## Steps

1. Press **Suggest a full set**. It fills in a broad, unbiased list:
   - nuclear size, shape and texture;
   - DNA content;
   - density, speed and age;
   - the level and localisation of every reporter;
   - every readout.
2. Untick the features you don't want, or add your own (name, measurement, channel, transform).
3. Set the **Reference condition** (usually the control) and the **reference frames**.
4. Press **Preview** for each feature's distribution per condition, and how the features correlate.

## Controls

| Control | What it does |
|---|---|
| Measurement / Channel | A cell measurement (area, shape, texture, density), a channel measurement, `speed`, `age` (frames since birth), or `readout:<name>`. |
| Transform | `log` for intensities, areas and other positive amounts; `asinh` for counts and speeds that can be 0; `none`. |
| Reference condition | Its median is 0 and its robust spread is 1, for every feature. |

## What is saved

`analysis.features`. Features are computed from the saved measurements when needed (about 0.7 s per position),
never stored.

## Tips

- Name features by what they measure, e.g. `PPARG nuc_mean`. Experiments with different reporters then pool by
  name, and each cell simply lacks the features its experiment didn't image.
"""

HELP["fates"] = """
**What this page is for:** when and how cells commit to a fate. A fate is the high state of a marker (e.g. nuclear
PPARG). The page measures:

- **how likely** a cell is to end up high, given its marker level now (the committor);
- **where** that crosses 50% (the commitment point), per condition;
- **what other features do** around the moment of commitment, compared with matched cells that didn't commit.

## Steps

1. Define features first (Features page). The marker and everything you align must be features.
2. Name the analysis and pick the **Marker**. Leave **valley between populations** ticked to put the threshold at the
   dip between the low and high populations.
3. Set the **Horizon**: how far ahead "ends up high" looks (e.g. 12 h).
4. Optionally **Split by** a second feature (e.g. the ERK KTR) into low/high groups. This asks whether it changes
   commitment *at the same marker level*.
5. Pick the conditions, then press **Run selected**. Look at the views on the right.

## Controls

| Control | What it does |
|---|---|
| Crossing | How long the smoothed marker must stay on the new side (and have been on the old one) for a crossing to count. |
| Align on | Commitment, or division (mother before, daughters after). |
| Window | Hours before and after the event. |
| Aligned features | Features averaged around events. None ticked = every ticked feature. |
| Match also on | Matched cells must also be close in these features, e.g. nuclear area and integrated H2B, so controls are in a similar cell-cycle phase. |
| Exclude crossings | Leave out crossings where a feature is extreme nearby, e.g. the H2B texture spike at mitosis. |
| Bootstrap | Positions resampled for 95% intervals. |

## Views

| View | What it shows |
|---|---|
| threshold | the marker distribution, the fitted populations and the threshold |
| committor | P(high later \\| marker now) per condition, with the commitment point |
| commit_point | commitment points with 95% intervals, per condition and group |
| aligned | features around commitment, solid for cells that committed and dashed for their matched cells |
| aligned_heatmap | every feature around the event at once |

## What is saved

The settings in `analysis.fates`. Results go to `<results>/Fates/<name>/`: `fate.json` (threshold, counts,
commitment points), `fate.npz` and `figures/`.

## Tips

- Fates need the frame interval: frames per hour (Experiment page) or minutes between frames (Dynamics page).
- A feature that departs from its matched cells *before* the crossing is a candidate for what precedes commitment.
- Cell-frames whose track ends before the horizon are counted as *censored*, not as "not committed".
"""

HELP["dynamics"] = """
**What this page is for:** fit stochastic models to every track at once, to learn about dynamics faster than the
frames. Examples are ERK switching between activity modes read through a KTR, a mean-reverting walk, or hidden cell
states found from many features. A model in continuous time can learn about switching faster than the frame
interval from many cells, and the **recovery check** says which parameters your data actually pin down.

## Steps

1. Set **Minutes between frames** (or frames per hour on Experiment), so rates are per minute.
2. Name the fit and pick a **Model**. The text under it explains the model.
3. Tick the **Readouts** to fit (Readouts page), or for the states model, the **Features** (Features page).
4. Tick the conditions, choose the frames, and decide which parameters are shared across conditions or windows.
5. Press **Fit selected**. Fits run in the background on the GPU; **Stop** ends one early.
6. Read the **checks** before trusting a fit. Then use the views on the right.

## Models

| Model | Hidden state | Main parameters |
|---|---|---|
| ERK modes + KTR | ERK in 2–6 activity modes × the KTR on a grid | switching rates, the level of each mode, KTR response time, noise |
| Mean-reverting walk (OU) | the readout itself | correlation time, mean, noise |
| Hidden states (HMM) | K discrete states over readouts or many features | rates between states, each state's profile, starting share per condition, optional division rates |
| Your SDE | du = drift dt + noise dW, as formulas | your parameters |

## Controls

| Control | What it does |
|---|---|
| Time windows / new window at each event | Lets a parameter shared "per window" change over time. |
| Parameter table | Start value, Fixed (keep a literature value), bounds, and Shared (overall, per condition, per window). |
| Try states | Fits each number of states on most positions and scores it on the rest. The best is kept. |
| divisions as observations | Each state gets a division rate, and newborn cells start from their own state distribution. |
| Routes | From the state lowest in a feature to the highest: committors, net flux, time to target. |
| Bootstrap | Refits on resampled positions for 95% intervals. 10–20 is typical. |
| recovery check | Simulates data from the fit, refits it, and shows which parameters come back. |

## Checks and views

- **Distributions, PIT, autocorrelation:** does the model reproduce the data? A flat PIT means calibrated.
- **Parameters, derived:** rates, dwell times and levels per condition, with intervals.
- **States model:** profiles (each state's features), the transition graph, routes, occupancy over time, and the
  comparison of state counts.
- Fits on the same tracks are listed with **ΔBIC**. A difference above about 10 is strong.

## What is saved

The settings in `analysis.dynamics`. Results go to `<results>/Dynamics/<fit>/`:

- `fit.json`: parameters with intervals, AIC and BIC;
- `checks.npz`;
- `states.h5`: the state probabilities per cell and frame;
- `figures/`.

## Tips

- To ask whether a drug changes a rate, fit it **shared** and again **per condition**, and compare the ΔBIC.
- The command line does the same: `fasttrack dynamics experiment.yaml "fit name"`.
"""

HELP["labelling"] = HELP["centres"]   # event labelling is part of Label cells now

HELP["export"] = """
**What this page is for:** build a figure page for a paper at the journal's exact size, from quick figures, fate
views and dynamics views. The page is checked against the figure standard (text size, nothing cut off) and exported
with a caption.

## Steps

1. **Page:** name it, pick the journal and the width (single, 1.5 or double column), and the number of grid columns.
2. **Add panels** with the **+** buttons:
   - a copy of a quick figure, which you can then change without touching the original;
   - a fate view;
   - a dynamics view.

   Panels fill the grid left to right. Drag them to reorder.
3. For the **selected panel**, set its span (columns) and height (mm), how conditions are compared, the legend, and
   for fate and dynamics views, which **Conditions** to draw.
4. Check the **preview**. It is the exported page at its real proportions, with a list of problems under it and the
   generated caption.
5. Press **Export selected** (or **Export all ticked**).

## Controls

| Control | What it does |
|---|---|
| Journal, width | Nature 89/120/183 mm, Cell 85/114/174, Science 57/121/184. |
| Type | Text size (7 pt by default; ticks and legends a bit smaller, never below 5 pt) and data line width. |
| Colours | The palette for this page: Plate colours, colour-blind safe, Okabe–Ito, or combinations. |
| Files | Formats to write. PDF keeps text editable in Illustrator. |
| Letter | The panel letter (auto: a, b, c in reading order). |
| Caption | Extra text added to this panel's part of the caption. |
| Copy again from the quick figure | Replaces the panel's settings with the quick figure's current ones. |
| Templates | Save the page as a template and apply it to another experiment. Conditions that aren't there are left out, and you are told which. |

## What is saved

The page definition is saved in `figures.exports`. Files go to `<results>/Figures/Export`:

- the page in every format;
- the caption (`.txt`);
- the numbers behind every quick-figure panel (`<name>_<letter>.csv`).

## Tips

- Fate and dynamics views lay themselves out for the panel's size. Whatever doesn't fit is named in the caption, so
  make the panel larger or pick fewer conditions.
- The command line does the same: `fasttrack export experiment.yaml "Figure 2"`.
- The rules behind the page check are in `docs/FIGURE_STANDARD.md`.
"""


# ------------------------------------------------------------------ Markdown to HTML (just what the guides use)
CSS = f"""
body {{ color: {theme.TEXT}; font-size: 10pt; }}
h2 {{ font-size: 12.5pt; font-weight: 600; margin-top: 18px; margin-bottom: 6px; color: {theme.TEXT}; }}
h3 {{ font-size: 11pt; font-weight: 600; margin-top: 12px; margin-bottom: 4px; }}
p {{ margin-top: 4px; margin-bottom: 8px; line-height: 145%; }}
li {{ margin-bottom: 4px; line-height: 140%; }}
code {{ font-family: monospace; background-color: {theme.SUNKEN}; color: #8A2D5F; }}
.kbd {{ font-family: monospace; background-color: {theme.SUNKEN}; border: 1px solid {theme.BORDER_STRONG};
        color: {theme.TEXT}; }}
table {{ border-collapse: collapse; margin-top: 4px; margin-bottom: 10px; }}
th {{ background-color: {theme.SUNKEN}; color: {theme.MUTED}; font-weight: 600; text-align: left;
      padding: 5px 9px; border-bottom: 1px solid {theme.BORDER}; }}
td {{ padding: 5px 9px; border-bottom: 1px solid #EDEFF3; vertical-align: top; }}
.tip {{ background-color: {theme.ACCENT_SOFT}; color: {theme.ACCENT_TEXT}; padding: 8px 12px; }}
.lead {{ font-size: 10.5pt; }}
"""


def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\[\[(.+?)\]\]", r"<span class='kbd'>&nbsp;\1&nbsp;</span>", text)
    text = re.sub(r"`([^`]+)`", r"<code>&nbsp;\1&nbsp;</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"<a href='\2'>\1</a>", text)
    return text.replace("\\|", "|")


def _cells(line: str) -> list[str]:
    parts = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [p.strip() for p in parts]


def to_html(markdown: str) -> str:
    """Headings, paragraphs, nested bullet and numbered lists, tables, `> ` tips and inline marks."""
    lines = markdown.strip("\n").splitlines()
    out: list[str] = []
    stack: list[tuple[int, str]] = []     # open lists: (indent, tag)
    para: list[str] = []
    first_para = True

    def flush_para():
        nonlocal first_para
        if para:
            cls = " class='lead'" if first_para and not out else ""
            out.append(f"<p{cls}>{_inline(' '.join(para))}</p>")
            para.clear()
            first_para = False

    def close_lists(indent: int = -1):
        while stack and stack[-1][0] > indent:
            out.append(f"</li></{stack.pop()[1]}>")

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if not stripped:
            flush_para()
            i += 1
            continue
        if stripped.startswith("#"):
            flush_para()
            close_lists()
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{level + 1 if level < 3 else 4}>{_inline(stripped.lstrip('#').strip())}</h{level + 1 if level < 3 else 4}>")
        elif stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            flush_para()
            close_lists()
            head = _cells(stripped)
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            t = ["<table width='100%' cellspacing='0'>", "<tr>" + "".join(f"<th>{_inline(h)}</th>" for h in head) + "</tr>"]
            t += ["<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows]
            out.append("".join(t) + "</table>")
            continue
        elif stripped.startswith(">"):
            flush_para()
            close_lists()
            out.append(f"<table width='100%' cellspacing='0'><tr><td class='tip'>{_inline(stripped.lstrip('> '))}"
                       "</td></tr></table>")
        elif m:
            flush_para()
            indent, marker, body = len(m.group(1)), m.group(2), m.group(3)
            tag = "ol" if marker[0].isdigit() else "ul"
            if stack and stack[-1][0] == indent:
                out.append("</li>")
            elif not stack or stack[-1][0] < indent:
                out.append(f"<{tag}>")
                stack.append((indent, tag))
            else:
                close_lists(indent)
                out.append("</li>")
            out.append(f"<li>{_inline(body)}")
        elif stack and line.startswith(" "):
            # a continuation of the open list item: the same paragraph, or after a blank line a new one inside it
            blank_before = i > 0 and not lines[i - 1].strip()
            if blank_before:
                close_lists(len(line) - len(line.lstrip()) - 1)    # text at a list's own indent belongs to its parent item
            gap = "" if out[-1].endswith(("</ul>", "</ol>")) else "<br><br>"
            out.append((gap if blank_before else " ") + _inline(stripped))
        else:
            close_lists()
            para.append(stripped)
        i += 1
    flush_para()
    close_lists()
    return f"<html><head><style>{CSS}</style></head><body>{''.join(out)}</body></html>"
