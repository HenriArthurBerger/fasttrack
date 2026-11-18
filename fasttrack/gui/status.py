"""Where each page stands, for the sidebar: a short line and a state (done, warn, todo, optional).

Reads only the experiment dict and what the tabs already hold. Never the disk, so it is cheap enough to run
after every edit.
"""
from __future__ import annotations

from ..experiment import pipeline_defaults


def _n(count: int, word: str, plural: str | None = None) -> str:
    return f"{count} {word if count == 1 else (plural or word + 's')}"


def _diff(a, b) -> int:
    if isinstance(a, dict) and isinstance(b, dict):
        return sum(_diff(a.get(k), b.get(k)) for k in set(a) | set(b))
    return int(a != b)


_DEFAULTS: dict | None = None


def page_status(exp: dict, window=None) -> dict[str, tuple[str, str]]:
    global _DEFAULTS
    out: dict[str, tuple[str, str]] = {}
    paths, channels = exp["paths"], exp["channels"]
    if not paths["raw"]:
        out["experiment"] = ("choose the raw data folder", "todo")
    elif not paths["results"]:
        out["experiment"] = ("choose a results folder", "warn")
    else:
        events = len(exp["time"]["events"])
        text = _n(len(channels), "channel") if channels else "channels: all, first nuclear"
        out["experiment"] = (text + (f" · {_n(events, 'event')}" if events else ""), "done")

    conditions = exp["plate"]["conditions"]
    wells = sum(len(c["wells"]) for c in conditions)
    if not conditions:
        out["plate"] = ("add conditions", "todo")
    elif not wells:
        out["plate"] = (f"{_n(len(conditions), 'condition')} · no wells yet", "warn")
    else:
        out["plate"] = (f"{_n(len(conditions), 'condition')} · {_n(wells, 'well')}", "done")

    q = exp["quality"]
    if not q["enabled"]:
        out["quality"] = ("check turned off", "warn")
    elif not q["checked"]:
        out["quality"] = ("not checked yet", "todo")
    else:
        flagged = [k for k in q["flagged"] if k not in q["keep"]]
        out["quality"] = (f"checked · {len(flagged)} flagged" if flagged else "checked · all usable",
                          "warn" if flagged else "done")

    out["inspect"] = ("try settings on one position", "todo")
    if window is not None and getattr(window.inspect, "tables", None) is not None:
        out["inspect"] = ("test run in memory", "done")

    if _DEFAULTS is None:
        _DEFAULTS = pipeline_defaults()
    changed = _diff(exp["pipeline"], _DEFAULTS)
    method = exp["pipeline"].get("centroids", {}).get("method", "")
    out["parameters"] = (("defaults" if not changed else f"{_n(changed, 'change')} from defaults")
                         + (" · learned seeds" if method == "cnn" else ""), "done")
    out["centres"] = ("in use for tracking" if method == "cnn" else "learned seed finder",
                      "done" if method == "cnn" else "optional")

    found, done = 0, 0
    if window is not None:
        table = window.run.table
        found = len(window.run.chosen)
        done = sum(1 for r in range(table.rowCount()) if table.item(r, 3) is not None and table.item(r, 3).text() == "Done")
    if not found:
        out["run"] = ("find the positions", "todo")
    elif done >= found:
        out["run"] = (f"all {found} positions tracked", "done")
    else:
        out["run"] = (f"{done} of {found} positions tracked", "warn" if done else "todo")
    out["viewer"] = ("look at the tracks", "done" if done else "todo")

    summ = exp["figures"]["summary"]
    out["summary"] = ("drawn after every run" if summ["enabled"] else "drawn here only, not after runs",
                      ("done" if done else "todo") if summ["enabled"] else "todo")
    figures = exp["figures"]["list"]
    ticked = sum(1 for f in figures if f.get("enabled", True))
    out["figures"] = ((_n(len(figures), "figure") + (f" · {ticked} ticked" if ticked != len(figures) else ""))
                      if figures else "no figures yet", "done" if figures else "todo")
    readouts = exp["analysis"]["readouts"]
    out["readouts"] = (_n(len(readouts), "readout") if readouts else "for the dynamics models", "done" if readouts else "todo")
    feats = exp["analysis"]["features"]["list"]
    on = sum(1 for f in feats if f.get("enabled", True))
    out["features"] = (f"{_n(on, 'feature')} ticked" if feats else "for fates and hidden states", "done" if feats else "todo")
    fates = exp["analysis"]["fates"]
    out["fates"] = (_n(len(fates), "analysis", "analyses") if fates else "commitment to a fate", "done" if fates else "todo")
    fits = exp["analysis"]["dynamics"]["fits"]
    out["dynamics"] = (_n(len(fits), "fit") if fits else "models of the dynamics", "done" if fits else "todo")
    pages = exp["figures"]["exports"]
    out["export"] = (_n(len(pages), "page") if pages else "figure pages for a paper", "done" if pages else "todo")
    return out
