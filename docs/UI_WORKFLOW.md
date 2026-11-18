# How the app is laid out, and why

## The problem

FastTrack grew one tab at a time. By the end it had 10 top-level tabs and 6 more inside Analysis, all at the same
level, in roughly the order they were written. Nothing said which tabs are required and which are optional, what order
to use them in, or how far along an experiment is. Everything was there, but a new user had to already know the path.

## The workflow

Every experiment goes through the same five stages. Inside a stage you can go back and forth, but each stage needs
the one before it:

| Stage | Question it answers | Pages | Needs |
|---|---|---|---|
| **1 · Set up** | What is the data, and what is each well? | Experiment, Plate | nothing |
| **2 · Check & tune** | Are the images usable, and does tracking work on them? | Quality, Test run, Parameters, *Label centres* | folders and channels |
| **3 · Track** | Track every position, then look at the result | Run, Viewer | settings you trust |
| **4 · Analyse** | What do the cells do, per condition? | Figures, Readouts, Features, Fates, Dynamics, *Label events* | tracked positions |
| **5 · Publish** | A figure page for the paper | Export | figures or analyses |

*Italic* pages are optional. They improve a step (the learned seed finder, the event classifier) but nothing needs
them.

Stage 2 is a loop: try settings on one position (Test run), change them (Parameters), try again. Stage 4 is a
menu, not a sequence: Readouts and Features are the inputs that Dynamics and Fates use.

## How the layout follows from it

- **The sidebar is the workflow.** All 16 pages sit in one list, grouped under the five stages in order, so the
  path through the app reads top to bottom. Analysis no longer hides six pages behind a second row of tabs.
- **Every item shows where it stands,** in a short line under its name: "3 channels · 1 event",
  "10 conditions · 30 wells", "not checked", "2 flagged", "12 positions found", "4 figures", "2 fits". A dot shows
  whether the step is done (green), needs attention (amber) or hasn't been started (grey). Optional pages are marked
  *optional*, so it is clear what can be skipped.
- **Every page has the same header:** a title, one sentence on what the page is for, a **How to use** button, and
  **Next →**, which goes to the next step of the workflow.
- **How to use** (or F1) opens a guide for that page:
  - what the page is for and where it fits in the workflow;
  - the steps in order;
  - every control;
  - what is saved, and where;
  - tips and common mistakes.

  The guides cover what `GUIDE.md` covers, written for someone looking at the page.
- **Home** is the first page. It offers New, Open and recent files, with a short overview of the five stages. It
  replaces the old toolbar; File > New/Open/Save and their shortcuts still work.
- **One primary action per page.** The button that does the page's main job (Scan data folder, Check positions,
  Run, Fit selected, Export, …) is filled with the accent colour. Everything else is secondary.

## Visual style

- Calm, neutral surfaces: a light grey window, white cards for groups, thin borders and 8 px corners.
- One accent colour, indigo, used only for primary buttons, the selected item, focus rings and selected tabs. Data
  colours, from the experiment and the figure standard, never compete with it.
- A dark sidebar, so the navigation reads as separate from the work area.
- Inputs at least 28 px high with 6 px corners. Tables have light headers and quiet grid lines. Thin scrollbars.
- The theme lives in one place, `fasttrack/gui/theme.py`. Tabs only mark buttons as primary (`theme.primary`) and
  text as secondary (`theme.hint`); they set no colours of their own.

## Not done (yet)

- **Dark mode.** The palette is defined as tokens in `theme.py`, so a dark set is mostly new values. The figure
  canvases are white, though, and would need a dark style of their own.
- **Pages that are still dense.** Analysis → Figures and Dynamics still show every option at once. The next step
  would be to fold advanced options (plot settings, the parameter table, the HMM options) into collapsible sections.
