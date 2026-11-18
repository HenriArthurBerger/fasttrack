# FastTrack figure standard

These are the rules every FastTrack figure follows, whether it is saved to a file or shown in the GUI. They are based on the Nature research figure guide, Cell Press and Science guidance, Paul Tol's and Okabe–Ito's colour-blind-safe palettes, Crameri et al. 2020 (colour misuse), Rougier et al. 2014 (ten rules), Weissgerber et al. 2015 (show the data), and Bang Wong's *Points of View* columns. Sources are listed at the end.

## 1. Size: draw at final print size

| Format | Width | Use |
|---|---|---|
| single | 89 mm (3.50 in) | one plot, or 2 small panels |
| 1.5 column | 120 mm (4.72 in) | 3 panels in a row |
| double | 183 mm (7.20 in) | multi-panel figures |
| height | ≤ 170 mm (6.69 in) | leaves room for the legend under the figure |

- Make the figure at its final size and never scale it afterwards. Text sizes and line widths only mean something at 100%.
- Cell Press widths are 85, 114 and 174 mm. A `journal` setting changes the widths; the type rules stay the same.
- Keep the page size exact: `layout="constrained"` with small pads. Don't use `bbox_inches="tight"`, which shrinks the canvas.

## 2. Text

- **Font:** Arial or Helvetica, falling back to Liberation Sans or DejaVu Sans if neither is installed. One family throughout.
- **Sizes at final size:** 7 pt axis labels and titles; 6 pt ticks, legends and annotations; never below 5 pt.
- **Panel labels:** 8 pt bold lowercase **a, b, c** (Nature) or uppercase (Cell). Place them top-left, outside the axes and aligned across the row.
- **Colour:** text is black or dark grey (contrast > 4.5:1). Never encode meaning in coloured text.
- **No figure titles.** The message goes in the caption. A panel may carry a short heading of 7 pt or less.
- **Axis labels:** "Quantity (unit)" in sentence case, e.g. "Nuclear area (µm²)" or "PPARG, nuclear mean (log a.u.)". Never a raw column name such as `nuc_area`.
- **Fonts stay editable:** `pdf.fonttype = 42`, `ps.fonttype = 42` and `svg.fonttype = "none"`, so text can be edited in Illustrator.

## 3. Lines and marks (at final size)

| Element | Width |
|---|---|
| axes spines, ticks | 0.5 pt (ticks 2.5 pt long, pointing out) |
| data lines | 0.75–1.0 pt (1.25 pt for the one highlighted series) |
| reference lines (0, threshold, event) | 0.5 pt, grey, dashed |
| error bars, box whiskers | 0.5 pt |
| scatter markers | 2–3 pt, alpha to taste when dense, no edge |
| minimum anywhere | 0.25 pt; thinner may not print |

- Remove the top and right spines. No grid, unless the reader has to look up values; then use 0.25 pt light grey behind the data.
- Shaded bands such as CIs use the line's colour at alpha 0.2 with no edge.

## 4. Colour

### 4a. Categorical: conditions

- One colour per condition, the same in every figure, set on the Plate tab and stored in the experiment.
- The control is always **black or dark grey**.
- Choose the palette by the number of other conditions:

| Other conditions | Palette | Notes |
|---|---|---|
| 1–3 | Tol *high-contrast* `#004488 #DDAA33 #BB5566` | also survives greyscale |
| 4–7 | Tol *bright* `#4477AA #EE6677 #228833 #CCBB44 #66CCEE #AA3377` (+ grey `#BBBBBB`) or Okabe–Ito | colour-blind safe |
| 8–9 | Tol *muted* `#CC6677 #332288 #DDCC77 #117733 #88CCEE #882255 #44AA99 #999933 #AA4499` | the practical limit for telling colours apart |
| > 9 | don't rely on colour alone, see below | |

- **Combinations and factorial designs:** give each drug a hue family and shade the combinations. For example Tdi = blue, MekI = orange, Tdi + MekI = a darker blend or blue with an orange marker. Or code the two factors separately, colour for one and line style or marker for the other. Our plate (Ctrl, Palbo, MekI, Gne, Tdi and their pairwise combinations) fits two factors: base drug × co-treatment.
- **Too many conditions for one panel:** use small multiples, one panel per condition with the control drawn as a grey line in each, or direct labels at line ends. Never a 10-entry legend over 20 lines.
- Never put red against green. Never use matplotlib's default `tab10` and `tab20` for conditions; they aren't colour-blind safe.
- Missing or bad data: light grey `#DDDDDD`.

### 4b. Categorical: hidden states

- States use a palette that **does not overlap** with condition colours, so the two can't be confused.
- Up to 8 states: Tol *light* or *pale*. Beyond 8, order states by a meaningful axis (e.g. the PPARG mean) and use a sequential map, so colour also carries information.

### 4c. Continuous colour maps

| Data | Map | Rule |
|---|---|---|
| unsigned (counts, density, occupancy, probability) | `viridis`, `cividis`, or Crameri `batlow` | perceptually uniform, starts light or dark |
| signed around a meaningful 0 (robust z, log fold change, net flux) | Tol *sunset* / `BuRd`, Crameri `vik`, or `RdBu_r` | centre fixed at 0, symmetric limits |
| cyclic (angle, phase) | Crameri `romaO` | |
| images | grey, or the channel's own colour | never pseudo-colour with `jet` |

- No rainbow or `jet`, ever.
- Clip colour limits at robust percentiles (1–99) and say so on the colourbar.
- Colourbars are thin (about 3 mm), as tall as the axes they belong to, with 6 pt labels and units.

## 5. Layout and whitespace

- Pack panels tightly: about 2 mm between them and 4 mm where an axis label has to fit. Use `constrained_layout` with `w_pad`/`h_pad` of 1–2 pt and no `tight_layout` (it turns constrained layout off).
- Panels that are compared **share axes**. Inner tick labels are dropped (`sharex`/`sharey` plus `label_outer()`), and the axis label appears once per row or column.
- Size each panel by what it shows; not every panel needs to be the same size. Heatmaps get square cells, time courses get a wide aspect (about 1.6:1).
- Legends: label lines directly at their ends or in one shared legend outside the data area. Never cover data.
- The reading order runs left to right, then top to bottom, and matches the panel letters.
- Leave no blank grid slots: an odd number of panels gets a wider panel or a legend panel.
- Avoid a colourbar per panel: one shared colourbar when the panels share a scale.

## 6. Statistics and honesty

- Show the data:
  - distributions (violin, box with points, histogram or ECDF) instead of bars of means;
  - points for n < 20;
  - when there are thousands of cells, show per-position or per-replicate summaries as points on top of the distribution.
- Every band or bar says what it is: **SD** (spread), **SEM** (uncertainty of the mean, rarely what you want), or **95% CI** (state how, e.g. bootstrap over positions). Every FastTrack figure has a `band` setting: quartiles, 10th-90th percentile, SD, s.e.m., 95% CI (t) or none. The default is quartiles for pooled cells and s.e.m. over wells, because wells are the replicates, not cells. The figure's notes, and an exported page's caption, name the band.
- **n** goes in the panel or caption: cells, tracks, positions and wells.
- Axes start at 0 for bar charts and counts. For other plots, limits are chosen for the data and kept the same across panels that are compared.
- Log axes are marked as log, with real values on the ticks (10¹, 10², …), not ln(x).
- Time axes run in hours from a stated origin (drug addition, commitment); "frames" only when the interval is unknown, with a warning.

## 7. Export

| Purpose | Format |
|---|---|
| journal, Illustrator | PDF (vector, TrueType fonts, nothing rasterised except images and dense scatter) |
| web, slides | SVG or 300 dpi PNG |
| Extended Data | 300 dpi RGB TIFF or JPEG |

- Rasterise dense layers (`rasterized=True` on scatter or heatmaps with more than about 10⁴ elements) so PDFs stay small, and keep axes and text as vectors.
- RGB colour mode; Nature accepts RGB.
- Each figure saves its data next to it (`.csv`) and a caption stub (`.txt`): what is plotted, n, what the error bands are, and the normalisation.
- A figure with no data is not saved. The GUI shows the message instead.

## 8. Checklist before a figure ships

1. Is it at final size (89/120/183 mm), with no rescaling?
2. Is every text item 5–7 pt in one sans-serif family, and are fonts embedded as Type 42?
3. Are lines ≥ 0.25 pt, with data lines about 0.75–1 pt?
4. Are condition colours the experiment's own, colour-blind safe, with the control in grey?
5. No rainbow; diverging maps centred at 0?
6. No overlapping text, nothing outside the canvas, no legend over data?
7. Are units on every axis, readable names, and time in hours?
8. Are n, the error-band definition and the normalisation stated?
9. Are compared panels on shared axes, with minimal whitespace and panel letters in reading order?

## 9. In FastTrack

- **Rules:** `fasttrack/plotting/style.py` holds the rules above. Every figure is drawn under `style.context(...)`: the "journal" preset for exported pages, and "screen" (9 pt) for quick figures at their own sizes.
- **Colours:** `style.condition_styles` gives the condition colours for every quick figure, fate view and dynamics view; `style.state_colors` gives the hidden states.
- **Comparison:** the comparison mode (overlay, reference, separate) is a setting of each time-course figure, because no one layout is best.
- **Pages:** `fasttrack/plotting/export.py` lays out a page, letters the panels and moves each panel's title to the caption. `check()` lists text below 5 pt and anything cut off at the page edge.
- **Fate and dynamics views:** `fasttrack/plotting/panels.py` lays them out for the space they get. Small multiples are as many as fit at about 25–30 mm each, with shared axes and tick labels only on the outer plots. There is one legend per panel, outside the plots, which moves underneath when the space on the right would cost plots. Whatever doesn't fit is left out and named in the caption. Heatmaps show as many feature rows as fit at 6 pt. On a page, a fate or dynamics panel can draw a subset of conditions, which keep their full-set colours.

## Sources

- Nature research figure guide, building and exporting panels: https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/
- Nature final submission: https://www.nature.com/nature/for-authors/final-submission
- Nature Extended Data formatting: https://research-figure-guide.nature.com/figures/extended-data-formatting-guidelines
- Paul Tol's colour schemes (summary via khroma): https://rdrr.io/cran/khroma/man/tol.html
- Okabe–Ito palette: https://rdrr.io/github/rstudio/thematic/man/okabe_ito.html
- Data visualization with flying colors (The Node): https://thenode.biologists.com/data-visualization-with-flying-colors/research/
- Crameri, Shephard & Heron 2020, *The misuse of colour in science communication*, Nat Commun: https://ideas.repec.org/a/nat/natcom/v11y2020i1d10.1038_s41467-020-19160-7.html
- Rougier, Droettboom & Bourne 2014, *Ten simple rules for better figures*: https://pmc.ncbi.nlm.nih.gov/articles/PMC4161295
- Weissgerber et al. 2015, *Beyond bar and line graphs*, summary: https://clauswilke.com/blog/2015/04/29/beyond-bar-and-line-graphs
- Bang Wong, *Points of View* index (Nature Methods): https://blogs.nature.com/blog/data-visualization-points-of-view/
- Matplotlib constrained layout guide: https://matplotlib.org/3.8.0/users/explain/axes/constrainedlayout_guide.html
