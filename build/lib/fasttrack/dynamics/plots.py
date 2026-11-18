"""Figures of a fit: parameters and derived quantities over conditions and time, model checks,
example tracks with their hidden-state probabilities, and the recovery check."""
from __future__ import annotations

import math
import re

import numpy as np
from matplotlib.figure import Figure

from ..experiment import PALETTE

VIEWS = {"derived": "Dwell times and other derived quantities", "parameters": "Fitted parameters",
         "distributions": "Check: population distributions, observed vs model", "pit": "Check: PIT (calibration)",
         "acf": "Check: autocorrelation, observed vs simulated", "tracks": "Example tracks with hidden-state probabilities",
         "occupancy": "Share of cells in each hidden state over time", "recovery": "Recovery check (identifiability)",
         "loss": "Optimisation trace",
         "profiles": "States: profile of every state over the features", "graph": "States: transitions between states",
         "routes": "States: routes to the target state (committor and net flux)", "sweep": "States: how many (held-out positions)"}
STATE_VIEWS = {"profiles", "graph", "routes", "sweep"}


def state_colors(n: int) -> list:
    """Hidden states never share the condition palettes (plotting.style)."""
    from ..plotting.style import state_colors as colors
    return colors(n)


def _colors(exp: dict | None, names: list[str]) -> list[str]:
    """Condition colours under the figures palette (plotting.style), the reference first."""
    from ..plotting.style import condition_colors
    return condition_colors(exp, names)


def _time_axis(report: dict) -> tuple[np.ndarray, str]:
    """Window midpoints in hours when the interval is known, else frames."""
    mid = np.array([(a + b) / 2 for a, b in report["window_frames"]], float)
    if report["time_unit"] == "min":
        return mid * report["interval"] / 60, "Hours"
    return mid, "Frame"


LOG_NAMES = ("dwell", "cycle", "ktr_response", "k_", "rate", "up", "down", "switches")
ELSEWHERE = {"rate": "graph", "mean": "profiles", "sd": "profiles", "load": "profiles"}


def _groups(report: dict, table: dict) -> tuple[dict, list[str]]:
    """{prefix: [(member, name)]}: dwell_1 ... dwell_8 become one group 'dwell' with members '1' ... '8'.
    For the states model, rates and state means are left to the graph and profiles views."""
    groups, skipped = {}, []
    for name, e in table.items():
        if e.get("fixed"):
            continue
        m = re.match(r"(.+?)((?:_\d+)+)$", name)
        prefix, member = (m.group(1), m.group(2)[1:].replace("_", "→")) if m else (name, "")
        if report.get("model") == "states" and prefix in ELSEWHERE:
            if prefix not in skipped:
                skipped.append(prefix)
            continue
        groups.setdefault(prefix, []).append((member, name))
    return groups, skipped


def _is_log(name: str, entry: dict, vals: np.ndarray) -> bool:
    unit = entry.get("unit") or ""
    return (unit.startswith("per") or name.startswith(LOG_NAMES)) and bool(np.all(vals[np.isfinite(vals)] > 0))


def _values_panel(ax, report, table, prefix, members, idx, colors, x, xlabel, P) -> None:
    """One group of parameters. A single value per condition: dots with 95% bars (x = conditions).
    Several members (one per state): a heatmap, members down, conditions across. Over time windows: lines."""
    import matplotlib
    from matplotlib.colors import LogNorm
    entries = [table[n] for _, n in members]
    unit = entries[0].get("unit") or ""
    ax.set_title(prefix.replace("_", " ") + (f" ({unit})" if unit and unit != prefix else ""), loc="left", fontsize="small")
    vals = np.stack([np.asarray(e["values"], float) for e in entries])[:, idx]          # (members, C, W)
    log = _is_log(members[0][1], entries[0], vals)
    W = vals.shape[2]
    if len(members) == 1 or W > 1:
        e = entries[0]
        v = vals[0]
        ci = np.asarray(e["ci"], float)[:, idx] if e.get("ci") else None
        if W == 1:
            for p, c in enumerate(idx):
                if ci is not None and np.isfinite(ci[:, p, 0]).all():
                    ax.vlines(p, ci[0, p, 0], ci[1, p, 0], color=colors[c], lw=0.75)
                ax.plot(p, v[p, 0], "o", color=colors[c], ms=3.5)
                bound = e.get("at_bound")
                if bound and bound[c][0]:
                    ax.plot(p, v[p, 0], "x", color="k", ms=6, mew=0.75)
            ax.set_xlim(-0.6, len(idx) - 0.4)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        else:
            for p, c in enumerate(idx):
                ax.plot(x, v[p], "-o", ms=2.5, color=colors[c])
                if ci is not None:
                    ax.fill_between(x, ci[0, p], ci[1, p], color=colors[c], alpha=0.2, lw=0)
        if log:
            P.log_axis(ax)
        if len(members) > 1:
            P.note(ax.figure, f"{prefix}: first of {len(members)} (one per state) over time windows")
        return
    M = vals[:, :, 0]                                       # (members, C)
    shared = bool(np.allclose(M, M[:, :1], equal_nan=True)) and M.shape[1] > 1
    if shared:                                              # the same for every condition: one column
        M = M[:, :1]
    norm = LogNorm(np.nanmin(M), np.nanmax(M)) if log and np.nanmin(M) > 0 and np.nanmax(M) > np.nanmin(M) else None
    im = ax.imshow(M, aspect="auto", cmap="cividis", norm=norm, interpolation="nearest",
                   extent=(-0.5, M.shape[1] - 0.5, len(members) - 0.5, -0.5) if not shared else
                   (-0.6, len(idx) - 0.4, len(members) - 0.5, -0.5))
    ax.set_yticks(range(len(members)), [m for m, _ in members])
    if all(m.isdigit() for m, _ in members):
        ax.set_ylabel("State")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.06, pad=0.02, aspect=15)
    cb.ax.tick_params(labelsize="small")
    if norm is not None:
        P.log_colorbar(cb)
    if shared:
        ax.set_title(ax.get_title(loc="left") + ", shared", loc="left", fontsize="small")


def draw(fig: Figure, report: dict, view: str, exp: dict | None = None, states: dict | None = None, condition: int = 0,
         show: list[str] | None = None) -> Figure:
    """Draw one view. `show`: the conditions to draw (names; empty = all), colours staying those of the full set.
    The layout follows the figure's size, so the same view fits a window, a single-column or a double-column panel."""
    from ..plotting import panels as P
    with P.screen(fig):
        P.begin(fig)
        _draw(fig, report, view, exp, states, condition, show, P)
    return fig


def _draw(fig, report, view, exp, states, condition, show, P) -> None:
    import matplotlib
    from ..plotting.style import CONTEXT, REFERENCE
    conds = report["conditions"]
    colors = _colors(exp, conds)
    idx = P.pick(conds, show)
    names = [conds[c] for c in idx]
    keyed = P.line_handles(names, [colors[c] for c in idx])
    legend_w = 8 + 1.6 * max(len(n) for n in names)
    x, xlabel = _time_axis(report)
    tunit = report["time_unit"]
    boot = report.get("bootstrap")
    if view in ("parameters", "derived"):
        table = report["parameters"] if view == "parameters" else report["derived"]
        groups, skipped = _groups(report, table)
        if not groups:
            P.empty(fig, "Nothing to show")
            return
        axes = P.grid(fig, len(groups), sharex=True, sharey=False, min_w=max(20, 3.2 * len(idx) + 14), min_h=22,
                      reserve_h=4 + 1.4 * max(len(n) for n in names), what="groups")
        for ax, (prefix, members) in zip(axes, groups.items()):
            _values_panel(ax, report, table, prefix, members, idx, colors, x, xlabel, P)
        for ax in axes:
            if len(x) == 1:
                P.condition_ticks(ax, names)          # shared: only the bottom row shows them
            else:
                ax.set_xlabel(xlabel)
        if len(x) > 1:
            P.legend(fig, keyed)
        notes = [f"{report['model_text'].rstrip('.')}; rates per {tunit}"]
        if boot:
            notes.append(f"bars: 95% interval over {boot} position resamples")
        if view == "parameters" and any(e.get("at_bound") for e in table.values()):
            notes.append("× = pressed against a bound")
        if skipped:
            where = sorted({ELSEWHERE[s] for s in skipped})
            notes.append(f"{', '.join(skipped)} are in the {' and '.join(where)} view{'s' if len(where) > 1 else ''}")
        P.finish(fig, VIEWS[view], notes)
    elif view == "distributions":
        ch = report["_checks"]
        bins = ch["bins"]
        centres = (bins[1:] + bins[:-1]) / 2
        obs, pred = ch["hist_obs"], ch["hist_pred"]
        W = obs.shape[1]
        cells = [(c, w) for c in idx for w in range(W)]
        axes = P.grid(fig, len(cells), min_w=24, min_h=20, reserve_w=18, cols=W if W > 1 else None, what="plots")
        for ax, (c, w) in zip(axes, cells):
            o = obs[c, w] / max(obs[c, w].sum(), 1) / (bins[1] - bins[0])
            p = pred[c, w] / max(pred[c, w].sum(), 1e-9) / (bins[1] - bins[0])
            ax.bar(centres, o, width=bins[1] - bins[0], color=colors[c], alpha=0.5, lw=0)
            ax.plot(centres, p, "k-")
            ax.set_title(conds[c] + (f", {x[w]:.1f} {xlabel.lower()}" if W > 1 else ""), loc="left", fontsize="small")
        fig.supxlabel(report["readouts"][0]["name"])
        fig.supylabel("Density")
        from matplotlib.patches import Patch
        P.legend(fig, [Patch(color="0.6", alpha=0.6, lw=0, label="observed"),
                       *P.style_handles([("model", dict(ls="-"))])])
        P.finish(fig, VIEWS[view], ["Bars: observed; line: predicted by the model one step ahead from each cell's past"])
    elif view == "pit":
        pit = report["_checks"]["pit"]
        axes = P.grid(fig, len(idx), min_w=24, min_h=20, what="conditions")
        for ax, c in zip(axes, idx):
            h = pit[c] / max(pit[c].sum(), 1) * len(pit[c])
            ax.bar(np.arange(len(h)) / len(h) + 0.5 / len(h), h, width=1 / len(h), color=colors[c], alpha=0.7, lw=0)
            ax.axhline(1, color="k", lw=0.5)
            ax.set_ylim(0, max(2, float(np.nanmax(h)) * 1.1))
            ax.set_xlim(0, 1)
            ax.set_title(conds[c], loc="left", fontsize="small")
        fig.supxlabel("PIT value")
        fig.supylabel("Density")
        P.finish(fig, VIEWS[view], ["Where each observation fell in the model's prediction from the cell's past",
                                    "flat at 1 = calibrated; U = model too confident (or heavy tails); hump = too vague"])
    elif view == "acf":
        ch = report["_checks"]
        ax = fig.add_subplot()
        lags = np.arange(ch["acf_obs"].shape[1])
        lag_x = lags * report["interval"] if tunit == "min" else lags
        for c in idx:
            ax.plot(lag_x, ch["acf_obs"][c], "-", color=colors[c])
            ax.plot(lag_x, ch["acf_sim"][c], "--", color=colors[c], lw=0.6 * matplotlib.rcParams["lines.linewidth"])
        ax.axhline(0, color=CONTEXT, lw=0.5, zorder=0)
        ax.set_xlabel(f"Lag ({tunit})")
        ax.set_ylabel("Autocorrelation")
        P.legend(fig, keyed + P.style_handles([("observed", dict(ls="-")), ("model", dict(ls="--", lw=0.6))]))
        P.finish(fig, VIEWS[view], ["Solid: tracks; dashed: tracks simulated from the fit with the same frames and noise"])
    elif view in ("tracks", "occupancy"):
        if states is None:
            P.empty(fig, "No saved state probabilities (run the fit)")
            return
        modes_names = states["mode_names"]
        modes = states["modes"].astype(np.float32)
        T = modes.shape[1]
        t = np.arange(T) + states["first_frame"]
        if tunit == "min":
            t, tl = t * report["interval"] / 60, "Hours"
        else:
            tl = "Frame"
        mode_colors = state_colors(len(modes_names))
        from matplotlib.patches import Patch
        state_keys = [Patch(color=mode_colors[m], label=modes_names[m], lw=0) for m in range(len(modes_names))]
        if view == "occupancy":
            axes = P.grid(fig, len(idx), min_w=28, min_h=20, reserve_w=18, what="conditions")
            for ax, c in zip(axes, idx):
                sel = states["condition"] == c
                share = np.nanmean(modes[sel], 0)                    # (T, M)
                ok = np.isfinite(share).all(1)
                ax.stackplot(t[ok], share[ok].T, colors=mode_colors[:len(modes_names)], lw=0, rasterized=True)
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 0.5, 1])
                ax.set_xlim(t[ok][0] if ok.any() else t[0], t[-1])
                ax.set_title(conds[c], loc="left", fontsize="small")
            fig.supxlabel(tl)
            fig.supylabel("Share of cells")
            P.legend(fig, state_keys[::-1])                          # top to bottom, as stacked
            P.finish(fig, VIEWS[view], ["Posterior share of tracked cells in each hidden state"])
        else:
            condition = min(max(int(condition), 0), len(conds) - 1)
            sel = np.nonzero(states["condition"] == condition)[0]
            readout = states["readout"].astype(np.float32)
            counts = np.isfinite(readout[sel]).sum(1)
            _, h_mm = P.size_mm(fig)
            n = int(min(6, len(sel), max(1, (h_mm - 12) // 13)))
            pick = sel[np.argsort(-counts)[:n]]
            if not len(pick):
                P.empty(fig, f"No tracks in {conds[condition]}")
                return
            axes = list(fig.subplots(len(pick), 1, sharex=True, squeeze=False).ravel())
            for ax, i in zip(axes, pick):
                y = readout[i]
                ok = np.isfinite(y)
                lo, hi = np.nanmin(y), np.nanmax(y)
                pad = 0.05 * (hi - lo or 1)
                lo, hi = lo - pad, hi + pad
                seen = np.nonzero(ok)[0]
                live = np.zeros(T, bool)                 # shade only while the cell is tracked
                live[seen[0]:seen[-1] + 1] = True
                bottom = np.zeros(T)
                for m in range(len(modes_names)):
                    top = bottom + np.nan_to_num(modes[i, :, m])
                    ax.fill_between(t, lo + (hi - lo) * bottom, lo + (hi - lo) * top, where=live, color=mode_colors[m],
                                    alpha=0.45, lw=0, rasterized=True)
                    bottom = top
                ax.plot(t[ok], y[ok], ".", color="k", ms=1.5)
                if np.isfinite(states["latent"][i]).any():
                    ax.plot(t[live], states["latent"][i].astype(np.float32)[live], "-", color="k", lw=0.75)
                ax.set_ylim(lo, hi)
                ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
                ax.text(0.005, 0.97, f"{states['position'][i]} track {states['track'][i]}", transform=ax.transAxes,
                        va="top", fontsize="small", bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
                if ax is not axes[-1]:
                    ax.tick_params(labelbottom=False)
            axes[-1].set_xlabel(tl)
            fig.supylabel(report["readouts"][0]["name"] if report.get("readouts") else "Readout")
            P.legend(fig, state_keys + P.style_handles([("readout", dict(ls="none", marker=".", ms=3)),
                                                        ("latent", dict(ls="-", lw=0.75))]))
            P.finish(fig, f"{conds[condition]}: example tracks with hidden-state probabilities",
                     [f"The {len(pick)} longest tracks; dots: readout; line: posterior latent; "
                      "shading: probability of each state"])
    elif view == "recovery":
        rec = report.get("recovery")
        if not rec:
            P.empty(fig, "No recovery check in this fit (tick 'Recovery check' and fit again)")
            return
        ax = fig.add_subplot()
        rnames = list(rec)
        for i, n in enumerate(rnames):
            true = np.asarray(rec[n]["true"], float).ravel()
            got = np.asarray(rec[n]["recovered"], float).ravel()
            ok = np.asarray(rec[n]["ok"], bool).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = got / true if np.all(true > 0) else got - true + 1
            ax.plot(np.full(ok.sum(), i), ratio[ok], "o", color=REFERENCE, ms=3)
            ax.plot(np.full((~ok).sum(), i), ratio[~ok], "x", color="#BB5566", ms=4, mew=0.9)
        ax.axhline(1, color="k", lw=0.5)
        ax.axhspan(1 / 1.5, 1.5, color=CONTEXT, alpha=0.3, lw=0, zorder=0)
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ticks = [v for v in (0.1, 0.2, 0.5, 2 / 3, 1, 1.5, 2, 5, 10) if lo <= v <= hi]
        ax.set_yticks(ticks, [f"{v:.2g}" for v in ticks])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(-0.6, len(rnames) - 0.4)
        P.condition_ticks(ax, rnames)
        ax.set_ylabel("Recovered / true")
        P.legend(fig, P.style_handles([("recovered", dict(ls="none", marker="o", ms=3, color=REFERENCE)),
                                       ("not recovered", dict(ls="none", marker="x", ms=4, mew=0.9, color="#BB5566"))]))
        P.finish(fig, VIEWS[view], ["Data simulated from the fit and refitted from the default start",
                                    "grey band: within ×1.5 (levels: ±0.25); crosses: not identifiable from data like these"])
    elif view in STATE_VIEWS:
        _state_view(fig, report, view, colors, idx, P)
    elif view == "loss":
        ax = fig.add_subplot()
        ax.plot(report.get("loss") or [], color="k", lw=0.6 * matplotlib.rcParams["lines.linewidth"])
        ax.set_xlabel("Step")
        ax.set_ylabel("−log-likelihood per observation")
        P.finish(fig, VIEWS[view], ["Minibatch estimates"])
    else:
        P.empty(fig, f"Unknown view {view!r}")


def _state_means(report: dict) -> tuple[np.ndarray, list[str]]:
    """(K, F) state means (first condition and window: emissions are shared) and the feature names."""
    names = report.get("feature_names") or [r["name"] for r in report["readouts"]]
    K = report.get("states") or len(report["mode_names"])
    M = np.full((K, len(names)), np.nan)
    for k in range(K):
        for f in range(len(names)):
            e = report["parameters"].get(f"mean_{k + 1}_{f + 1}")
            if e:
                M[k, f] = np.asarray(e["values"], float)[0, 0]
    return M, names


def _graph(ax, rates: np.ndarray, sizes: np.ndarray, node_color, cmap=None, edge_scale=None, labels=None):
    """States on a circle (area = sizes), arrows i -> j with width ~ rates[i, j]. Nodes are circles in data units,
    so they never overlap, whatever the size of the plot."""
    import matplotlib
    from matplotlib.patches import Circle
    K = len(sizes)
    ang = np.pi / 2 - 2 * np.pi * np.arange(K) / K
    xy = np.stack([np.cos(ang), np.sin(ang)], 1)
    r_max = 0.85 * math.sin(math.pi / K) if K > 1 else 0.6
    rel = np.sqrt(np.clip(sizes / max(np.nanmax(sizes), 1e-9), 0, 1))
    radius = r_max * (0.3 + 0.7 * np.nan_to_num(rel))
    cm = matplotlib.colormaps[cmap] if cmap else None
    circles = []
    for k in range(K):
        fc = cm(float(node_color[k])) if cm is not None else node_color
        c = Circle(xy[k], radius[k], fc=fc, ec="k", lw=0.5, zorder=3)
        ax.add_patch(c)
        circles.append(c)
        light = (np.dot(matplotlib.colors.to_rgb(fc), [0.299, 0.587, 0.114]) > 0.55)
        ax.text(*xy[k], labels[k] if labels else str(k + 1), ha="center", va="center", fontsize="small", zorder=4,
                color="k" if light else "white")
    top = edge_scale or (np.nanmax(rates) if np.isfinite(rates).any() and np.nanmax(rates) > 0 else 1)
    lw0 = matplotlib.rcParams["lines.linewidth"]
    for i in range(K):
        for j in range(K):
            r = rates[i, j]
            if i == j or not np.isfinite(r) or r < 0.03 * top:
                continue
            d = xy[j] - xy[i]
            off = 0.04 * np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-9)   # the two directions side by side
            ax.annotate("", xy=xy[j] + off, xytext=xy[i] + off,
                        arrowprops=dict(arrowstyle="-|>", lw=0.25 + 2.5 * lw0 * r / top, color="0.25", alpha=0.8,
                                        patchA=circles[i], patchB=circles[j], shrinkA=1, shrinkB=1, mutation_scale=5))
    edge = 1 + r_max + 0.04
    ax.set_xlim(-edge, edge)
    ax.set_ylim(-edge, edge)
    ax.set_aspect("equal")
    ax.axis("off")


def _state_view(fig: Figure, report: dict, view: str, colors, idx, P) -> None:
    if report.get("model") != "states":
        P.empty(fig, "Only for the hidden Markov model (states)")
        return
    conds = report["conditions"]
    names = [conds[c] for c in idx]
    K = report.get("states") or len(report["mode_names"])
    W = len(report["windows"])
    tunit = report["time_unit"]
    rates = np.zeros((len(conds), W, K, K))
    for name, e in report["parameters"].items():
        if name.startswith("rate_"):
            _, i, j = name.split("_")
            rates[:, :, int(i) - 1, int(j) - 1] = np.asarray(e["values"], float)
    frac = np.stack([np.asarray(report["derived"][f"fraction_{k + 1}"]["values"], float) for k in range(K)], -1)   # (C, W, K)
    state_labels = [str(k + 1) for k in range(K)]
    if view == "profiles":
        from matplotlib.colors import LogNorm
        M, feats = _state_means(report)
        dwell = np.stack([np.asarray(report["derived"][f"dwell_{k + 1}"]["values"], float)[idx, -1] for k in range(K)], 0)  # (K, C)
        gs = fig.add_gridspec(1, 2, width_ratios=[max(3, len(feats)), max(2, len(idx))])
        ax = fig.add_subplot(gs[0, 0])
        lim = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(feats)), feats, rotation=90)
        ax.set_yticks(range(K), [f"state {k + 1}" for k in range(K)])
        w_mm, h_mm = P.size_mm(fig)
        cell_w = w_mm * 0.55 / max(len(feats), 1)
        cell_h = (h_mm - 30) / max(K, 1)
        if cell_w >= 5.5 and cell_h >= 2.8:
            for k in range(K):
                for f in range(len(feats)):
                    if np.isfinite(M[k, f]):
                        ax.text(f, k, f"{M[k, f]:.1f}", ha="center", va="center", fontsize="small",
                                color="white" if abs(M[k, f]) > 0.6 * lim else "k")
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.015, aspect=20)
        cb.set_label("Mean" + (" (robust spreads)" if report.get("inputs") == "features" else ""))
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
        ok = dwell[np.isfinite(dwell) & (dwell > 0)]
        norm = LogNorm(ok.min(), ok.max()) if ok.size and ok.max() > ok.min() else None
        im2 = ax2.imshow(dwell, cmap="cividis", norm=norm, aspect="auto", interpolation="nearest")
        P.condition_ticks(ax2, names)
        ax2.tick_params(labelleft=False, length=0)
        ax2.spines[:].set_visible(False)
        cb2 = fig.colorbar(im2, ax=ax2, fraction=0.06, pad=0.03, aspect=20)
        cb2.set_label(f"Mean dwell ({tunit})")
        if norm is not None:
            P.log_colorbar(cb2)
        notes = ["Left: each state's mean of every feature, shared by all conditions; right: how long cells stay in "
                 "each state, per condition (last time window)"]
        div = [report["parameters"].get(f"division_{k + 1}") for k in range(K)]
        if all(div):
            dv = [np.asarray(d["values"], float)[0, 0] for d in div]
            notes.append("division rate per state: " + ", ".join(f"{k + 1}: {v:.2g}/{tunit}" for k, v in enumerate(dv)))
        P.finish(fig, VIEWS[view], notes)
    elif view in ("graph", "routes"):
        fl = report.get("flux")
        if view == "routes" and not fl:
            P.empty(fig, "Choose a target in the fit's options (a feature, or start and target states), then fit again")
            return
        cbar_w = 14 if view == "routes" else 0
        axes = P.grid(fig, len(idx), sharex=False, sharey=False, min_w=24, min_h=24, reserve_w=cbar_w, reserve_h=6,
                      what="conditions")
        if view == "graph":
            top = float(np.nanmax(frac[idx, -1, :, None] * rates[idx, -1])) or 1.0
            for ax, c in zip(axes, idx):
                flow = frac[c, -1][:, None] * rates[c, -1]      # transitions per cell per time unit, at the long-run occupancy
                _graph(ax, flow, frac[c, -1], colors[c], edge_scale=top, labels=state_labels)
                ax.set_title(conds[c], fontsize="small")
            P.finish(fig, VIEWS[view], ["Circle area: long-run share of cells in the state",
                                        f"arrow width: jumps per cell per {tunit} (same scale in every plot); last time window"])
        else:
            com = np.asarray(fl["committor"], float)
            net = np.asarray(fl["net_flux"], float)
            tm = np.asarray(fl["time_to"], float)
            for ax, c in zip(axes, idx):
                through = net[c, -1].sum(0) + net[c, -1].sum(1)
                _graph(ax, net[c, -1], np.maximum(through, 1e-3), com[c, -1], cmap="viridis", labels=state_labels)
                ax.set_title(f"{conds[c]}\n{P.duration(tm[c, -1], tunit)}", fontsize="small", linespacing=1.1)
            import matplotlib
            sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap="viridis")
            fig.colorbar(sm, ax=axes, fraction=0.03, aspect=25, pad=0.02, shrink=0.8, label="Committor")
            target = f"states {fl['from']} → {fl['to']}" + (f" (lowest → highest {fl['feature']})" if fl.get("feature") else "")
            P.finish(fig, f"Routes {target}", ["Colour: committor, the chance to reach the target before going back",
                                              "arrows: net jumps on the way; circle area: traffic through the state",
                                              "under each name: mean time to the target",
                                              "an intermediate colour with heavy traffic marks a transition state"])
    elif view == "sweep":
        sw = report.get("sweep") or []
        if not sw:
            P.empty(fig, "No comparison of state numbers in this fit (list numbers under 'try states')")
            return
        ax = fig.add_subplot()
        k = [r["states"] for r in sw]
        ax.plot(k, [r["heldout_ll"] for r in sw], "o-", color="k", ms=3)
        ax.plot(k, [r["train_ll"] for r in sw], "o--", color="0.55", ms=3)
        best = max(sw, key=lambda r: r["heldout_ll"])["states"]
        ax.axvline(best, color="0.6", lw=0.5, ls="--", zorder=0)
        ax.text(best, 1, f" best: {best}", transform=ax.get_xaxis_transform(), va="top", fontsize="small")
        ax.set_xticks(k)
        ax.set_xlabel("Number of states")
        ax.set_ylabel("Log-likelihood per observation")
        P.legend(fig, P.style_handles([("held-out positions", dict(ls="-", marker="o", ms=3)),
                                       ("positions fitted", dict(ls="--", marker="o", ms=3, color="0.55"))]))
        P.finish(fig, VIEWS[view], ["More states always fit the fitted positions better; only real structure carries "
                                    "over to positions the fit has not seen"])


def render_fit(exp: dict, report: dict, states: dict | None, folder, formats=("png",), dpi: int = 150) -> list:
    """Every view of a fit as files in folder/figures."""
    from pathlib import Path
    out = []
    folder = Path(folder) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for view in VIEWS:
        if view == "recovery" and not report.get("recovery"):
            continue
        if view in STATE_VIEWS and (report.get("model") != "states" or (view == "sweep" and not report.get("sweep"))
                                    or (view == "routes" and not report.get("flux"))):
            continue
        fig = Figure(figsize=(7.2, 4.5))
        draw(fig, report, view, exp, states)
        for fmt in formats:
            path = folder / f"{view}.{fmt}"
            fig.savefig(path, dpi=dpi)
            out.append(path)
    return out
