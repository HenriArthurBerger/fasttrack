"""Run a model fit described in the experiment file, and keep everything it found.

    <results>/<dynamics.folder>/<fit name>/fit.json     settings, parameters per condition and window
                                                       (with bootstrap intervals), derived quantities
                                                       (dwell times ...), likelihood, AIC/BIC, recovery
    <results>/<dynamics.folder>/<fit name>/checks.npz   model checks: distributions, PIT, autocorrelation
    <results>/<dynamics.folder>/<fit name>/states.h5    per fitted track: probability of each mode at
                                                       every frame, and the posterior latent readout

Fits of different models on the same data (same readouts, conditions, frames, track sample) are
comparable by AIC/BIC; `compare(exp)` lists them.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from ..analysis.figures import ResultsCache, figure_conditions, safe_name
from ..experiment import FIT_DEFAULT, merge, normalise, time_unit  # noqa: F401 (time_unit re-exported)
from . import engine as E
from .data import TrackData, load_tracks
from .models import make_model


def windows(exp: dict, fit: dict, T: int, first: int, dt: float, unit: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """window_of_t (T,) and each window's [start, stop) in fitted columns."""
    cuts = {0, T}
    if fit["window_hours"] > 0 and unit == "min":
        step = max(1, int(round(fit["window_hours"] * 60 / dt)))
        cuts |= set(range(0, T, step))
    if fit["split_at_events"]:
        files = exp["files"]
        for ev in exp["time"]["events"]:
            col = int(round((ev["frame"] - files["first_frame"]) / max(1, files["stride"]))) - first
            if 0 < col < T:
                cuts.add(col)
    edges = sorted(cuts)
    spans = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    w = np.zeros(T, np.int64)
    for i, (a, b) in enumerate(spans):
        w[a:b] = i
    return w, spans


def fit_conditions(exp: dict, fit: dict) -> list[dict]:
    spec = {"conditions": fit["conditions"], "group": fit["group"]}
    return figure_conditions(exp, spec)


def fit_readouts(exp: dict, fit: dict) -> list[dict]:
    defined = {r["name"]: r for r in exp["analysis"]["readouts"]}
    if not defined:
        raise ValueError("Define a readout first (Analysis → Readouts): which channel holds the reporter, and its kind.")
    names = fit["readouts"] or [exp["analysis"]["readouts"][0]["name"]]
    missing = [n for n in names if n not in defined]
    if missing:
        raise KeyError(f"Fit {fit['name']} uses readouts {missing}, which are not defined; readouts are {list(defined)}.")
    if fit["model"] != "states":
        names = names[:1]  # SDE models describe one readout
    return [defined[n] for n in names]


def fit_folder(exp: dict, name: str) -> Path:
    return Path(exp["paths"]["results"]) / exp["analysis"]["dynamics"]["folder"] / safe_name(name or "fit")


def _data_key(exp, fit, readouts) -> str:
    """Fits with the same key saw the same tracks, so their likelihoods compare."""
    keep = {k: fit[k] for k in ("conditions", "group", "frames", "max_tracks", "seed", "features")}
    keep["division"] = bool(fit["options"].get("division"))
    return json.dumps({"fit": keep, "readouts": readouts, "filters": [exp["analysis"]["min_track_frames"], exp["analysis"]["drop_border"]]},
                      sort_keys=True)


def _standardise(data: TrackData, model_name: str, features: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """States models fit z-scored readouts (so one prior suits all); SDE models fit raw values. Features
    are already on one scale (the reference condition's median and spread) and stay as they are."""
    F = data.y.shape[2]
    if model_name != "states" or features:
        return np.zeros(F), np.ones(F)
    mu = np.array([np.nanmean(data.y[..., f]) for f in range(F)])
    sd = np.array([np.nanstd(data.y[..., f]) or 1.0 for f in range(F)])
    data.y = ((data.y - mu) / sd).astype(np.float32)
    data.se = (data.se / sd).astype(np.float32)
    return mu, sd


def routes(Q: np.ndarray, A: list[int], B: list[int], start: np.ndarray | None = None) -> dict:
    """Committor and routes between state sets A and B of a generator Q (S, S).

    committor[i]: probability that a cell in state i reaches B before A (0 in A, 1 in B).
    flux[i, j]: expected i -> j jumps made by a cell starting in A (spread by `start`, default evenly)
    before it first reaches B; net = flux minus the reverse, kept where positive.
    time: expected time from A to B. States with an intermediate committor that carry much of the
    net flux are the transition states on the way."""
    S = Q.shape[0]
    A, B = sorted(set(A)), sorted(set(B) - set(A))
    out = {"committor": np.full(S, np.nan), "flux": np.zeros((S, S)), "net": np.zeros((S, S)), "time": float("nan")}
    if not A or not B:
        return out
    q = np.zeros(S)
    q[B] = 1
    mid = [i for i in range(S) if i not in A and i not in B]
    if mid:
        try:
            q[mid] = np.linalg.solve(Q[np.ix_(mid, mid)], -Q[np.ix_(mid, B)].sum(1))
        except np.linalg.LinAlgError:
            q[mid] = np.nan
    out["committor"] = q
    T_ = [i for i in range(S) if i not in B]
    p0 = np.zeros(S)
    w = np.ones(len(A)) if start is None else np.asarray(start, float)[A] + 1e-12
    p0[A] = w / w.sum()
    try:
        tau = np.linalg.solve(-Q[np.ix_(T_, T_)].T, p0[T_])      # expected time in each state before reaching B
    except np.linalg.LinAlgError:
        return out
    tau_full = np.zeros(S)
    tau_full[T_] = np.maximum(tau, 0)
    flux = tau_full[:, None] * np.where(np.eye(S, dtype=bool), 0, Q)
    out["flux"], out["net"], out["time"] = flux, np.maximum(flux - flux.T, 0), float(tau_full.sum())
    return out


def _load(exp, fit, conditions, cache, on_event, stop):
    """TrackData, its readout descriptions, and event flags (None without lineage)."""
    if fit["model"] == "states" and fit["features"]:
        from ..analysis.features import load_features
        data = load_features([exp], names=list(fit["features"]), conditions=[c["name"] for c in conditions], frames=fit["frames"],
                             max_tracks=int(fit["max_tracks"]), seed=int(fit["seed"]), on_event=on_event, stop=stop)
        missing = [n for n in fit["features"] if n not in data.readouts]
        if missing:
            raise KeyError(f"features {missing} are not defined (Analysis → Features)")
        ev = None
        if fit["options"].get("division"):
            ev = E.track_events(data.y, data.rows["divides"], data.rows["born"], data.rows["last"])
            on_event({"type": "log", "message": f"Lineage: {int(data.rows['born'].sum())} tracks born in the movie, "
                                                  f"{int(((ev & E.DIVIDE) > 0).sum())} divisions observed"})
        return data, [{"name": n, "kind": "feature"} for n in data.readouts], ev
    readouts = fit_readouts(exp, fit)
    data = load_tracks(exp, readouts, conditions, fit["frames"], int(fit["max_tracks"]), int(fit["seed"]), cache, on_event, stop)
    if fit["options"].get("division"):
        on_event({"type": "log", "message": "Divisions need features (they carry lineage); fitting without them."})
    return data, readouts, None


def _table(values: dict[str, np.ndarray], C: int, W: int) -> dict:
    return {k: np.asarray(v, dtype=float).reshape(C, W).tolist() for k, v in values.items()}


def _unit_text(unit: str, tunit: str) -> str:
    return {"rate": f"per {tunit}", "time": tunit, "noise": "ln units" if tunit else "", "level": "readout",
            "fraction": "fraction", "logit": "log odds", "value": ""}.get(unit, unit)


def run_fit(exp: dict, fit: dict, on_event=lambda e: None, stop=None, cache: ResultsCache | None = None,
            save: bool = True) -> dict:
    """Fit one model from the experiment file. Returns the report (also written to fit.json)."""
    exp = normalise(exp)
    fit = merge(FIT_DEFAULT, fit)
    t_start = time.time()
    conditions = fit_conditions(exp, fit)
    if not conditions:
        raise ValueError("No conditions to fit: tick some, choose a group, or add conditions on the Plate tab.")
    dt, tunit = time_unit(exp)
    device = E.pick_device(exp["analysis"]["dynamics"]["device"])
    on_event({"type": "log", "message": f"Loading {len(conditions)} conditions"})
    data, readouts, ev = _load(exp, fit, conditions, cache, on_event, stop)
    use_features = fit["model"] == "states" and bool(fit["features"])
    mu, sd = _standardise(data, fit["model"], use_features)
    w_of_t, spans = windows(exp, fit, data.T, data.first_frame, dt, tunit)
    C, W = len(conditions), len(spans)
    flat = data.y[..., 0][np.isfinite(data.y[..., 0])]
    lo, hi = np.quantile(flat, [0.002, 0.998]) if len(flat) else (-3.0, 3.0)
    pad = 0.15 * (hi - lo)
    obs_flat = data.y[np.isfinite(data.y).any(-1)]

    def new_model(fit_):
        m = make_model(fit_, features=data.y.shape[2], feature_names=data.readouts)
        m.set_range(lo - pad, hi + pad)
        m.outlier = float(fit_["outlier"])
        return m

    # how many states: compare on positions held out from fitting
    sweep = []
    Ks = sorted({int(k) for k in (fit["options"].get("sweep") or []) if int(k) >= 2}) if fit["model"] == "states" else []
    if Ks:
        rng = np.random.default_rng(int(fit["seed"]) + 3)
        test_rows = []
        for c in range(C):
            pos = np.unique(data.position[data.cond == c])
            k = int(round(float(fit["heldout"]) * len(pos)))
            k = min(max(k, 1), len(pos) - 1) if len(pos) > 1 else 0
            for p_ in rng.choice(pos, k, replace=False) if k else []:
                test_rows.append(np.nonzero(data.position == p_)[0])
        test = np.concatenate(test_rows) if test_rows else np.zeros(0, int)
        train = np.setdiff1d(np.arange(len(data.y)), test)
        if not len(test):
            raise ValueError("Comparing numbers of states needs at least two positions per condition.")
        tb = E.Batches(data.y, data.se, data.cond, C, device, index=train, ev=ev)
        hb = E.Batches(data.y, data.se, data.cond, C, device, index=test, ev=ev)
        n_test = max(1, hb.n_obs())
        for K in Ks:
            if stop is not None and stop.is_set():
                break
            fk = copy.deepcopy(fit)
            fk["options"]["states"] = K
            mk = new_model(fk)
            lk = E.Layout(mk, {n: o for n, o in fit["parameters"].items() if n in mk.specs()}, C, W, minutes=tunit == "min")
            zk, _ = E.fit(mk, lk, tb, w_of_t, dt, int(fit["steps"]), int(fit["batch"]), float(fit["lr"]), int(fit["seed"]),
                          start=mk.start_values(torch.as_tensor(obs_flat)), stop=stop)
            held = E.log_likelihood(mk, lk, hb, w_of_t, dt, zk) / n_test
            trained = E.log_likelihood(mk, lk, tb, w_of_t, dt, zk) / max(1, tb.n_obs())
            sweep.append({"states": K, "heldout_ll": held, "train_ll": trained, "free_numbers": lk.size})
            on_event({"type": "log", "message": f"{K} states: held-out log-likelihood {held:.4f} per observation (train {trained:.4f})"})
        best = max(sweep, key=lambda r: r["heldout_ll"])["states"]
        fit = copy.deepcopy(fit)
        fit["options"]["states"] = best
        on_event({"type": "log", "message": f"Best on held-out positions: {best} states. Fitting it on every position."})

    model = new_model(fit)
    layout = E.Layout(model, fit["parameters"], C, W, minutes=tunit == "min")
    batches = E.Batches(data.y, data.se, data.cond, C, device, ev=ev)
    start = model.start_values(torch.as_tensor(obs_flat if fit["model"] == "states" else data.y[np.isfinite(data.y[..., 0])]))
    on_event({"type": "log", "message": f"Fitting {model.text.lower()}: {len(data.y)} tracks, {C} conditions × {W} windows, "
                                          f"{layout.size} free numbers, frame interval {dt:g} {tunit}, on {device}"})

    def progress(step, loss):
        if step % 10 == 0 or step == fit["steps"] - 1:
            on_event({"type": "progress", "stage": "fit", "done": step + 1, "total": fit["steps"], "loss": loss})

    z, history = E.fit(model, layout, batches, w_of_t, dt, int(fit["steps"]), int(fit["batch"]), float(fit["lr"]),
                       int(fit["seed"]), start=start, on_step=progress, stop=stop)
    ll = E.log_likelihood(model, layout, batches, w_of_t, dt, z)
    n_obs = batches.n_obs()
    k = layout.size

    def report_values(zv):
        p = layout.values(zv.to(device))
        vals = {n: v.detach().cpu().numpy() for n, v in p.items()}
        der = {n: v[0].detach().cpu().numpy() for n, v in model.derived(p).items()}
        if fit["model"] == "states":  # back to readout units
            for name in list(vals):
                parts = name.split("_")
                if parts[0] in ("mean", "sd") and len(parts) == 3:
                    f = int(parts[2]) - 1
                    vals[name] = vals[name] * sd[f] + (mu[f] if parts[0] == "mean" else 0)
                elif parts[0] == "load" and len(parts) == 4:
                    vals[name] = vals[name] * sd[int(parts[2]) - 1]
        return vals, der

    values, derived = report_values(z)
    specs = layout.specs
    der_info = {n: (u, txt) for n, (_, u, txt) in model.derived(layout.values(z.to(device))).items()}

    # bootstrap over positions
    boots_v, boots_d = [], []
    if fit["bootstrap"] > 0:
        rng = np.random.default_rng(int(fit["seed"]) + 1)
        for b in range(int(fit["bootstrap"])):
            if stop is not None and stop.is_set():
                break
            rows = []
            for c in range(C):
                pos = np.unique(data.position[data.cond == c])
                for p in rng.choice(pos, len(pos), replace=True):
                    rows.append(np.nonzero(data.position == p)[0])
            idx = np.concatenate(rows)
            bb = E.Batches(data.y, data.se, data.cond, C, device, index=idx, ev=ev)
            zb, _ = E.fit(model, layout, bb, w_of_t, dt, max(50, int(fit["steps"]) // 3), int(fit["batch"]), float(fit["lr"]) / 2,
                          int(fit["seed"]) + 10 + b, z0=z, stop=stop)
            v, d = report_values(zb)
            boots_v.append(v)
            boots_d.append(d)
            on_event({"type": "progress", "stage": "bootstrap", "done": b + 1, "total": int(fit["bootstrap"])})

    def interval(boots, name):
        if len(boots) < 2:
            return None
        stack = np.stack([np.asarray(b[name], float).reshape(C, W) for b in boots])
        return np.nanpercentile(stack, [2.5, 97.5], axis=0).tolist()

    bound = layout.at_bound(z)
    params = {}
    for name, spec in specs.items():
        flags = np.zeros((C, W), bool)
        if name in bound:
            flags = bound[name].reshape(C, W)
        params[name] = {"values": np.asarray(values[name], float).reshape(C, W).tolist(), "ci": interval(boots_v, name),
                        "fixed": layout.fixed[name], "share": layout.share[name], "unit": _unit_text(spec.unit, tunit),
                        "at_bound": flags.tolist(), "text": spec.text, "low": spec.low, "high": spec.high}
    der = {name: {"values": np.asarray(derived[name], float).reshape(C, W).tolist(), "ci": interval(boots_d, name),
                  "unit": _unit_text(der_info[name][0], tunit), "text": der_info[name][1]} for name in derived}

    # model checks
    on_event({"type": "log", "message": "Model checks: predicted distributions, PIT, autocorrelation"})
    bins = np.linspace(model.lo if fit["model"] != "states" else np.nanquantile(data.y[..., 0], 0.002),
                       model.hi if fit["model"] != "states" else np.nanquantile(data.y[..., 0], 0.998), 61)
    checks = E.predictive_checks(model, layout, batches, w_of_t, dt, z, bins, lags=min(30, data.T - 2), seed=int(fit["seed"]))
    if fit["model"] == "states":
        checks["bins"] = checks["bins"] * sd[0] + mu[0]
    modes, latent = E.state_posteriors(model, layout, batches, w_of_t, dt, z)

    # recovery check: refit data simulated from the fit, from the default start
    recovery = None
    if fit["recovery_check"] and not (stop is not None and stop.is_set()):
        on_event({"type": "log", "message": "Recovery check: refitting data simulated from the fit"})
        sim = E.simulate_discrete(model, layout, batches, w_of_t, dt, z, seed=int(fit["seed"]) + 99)
        sb = E.Batches(sim, data.se, data.cond, C, device, ev=ev)
        s_start = model.start_values(torch.as_tensor(sim[np.isfinite(sim).any(-1)] if fit["model"] == "states"
                                                     else sim[np.isfinite(sim[..., 0])]))
        zr, _ = E.fit(model, layout, sb, w_of_t, dt, int(fit["steps"]), int(fit["batch"]), float(fit["lr"]), int(fit["seed"]) + 5,
                      start=s_start, stop=stop)
        rv, rd = report_values(zr)
        recovery = {}
        for name, truth in [*values.items(), *derived.items()]:
            if name in values and layout.fixed.get(name):
                continue
            got = np.asarray(rv.get(name, rd.get(name)), float).reshape(C, W)
            true = np.asarray(truth, float).reshape(C, W)
            is_log = specs[name].log if name in specs else True
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.abs(np.log(got / true)) if is_log else np.abs(got - true)
            ok = err < (math.log(1.5) if is_log else 0.25)
            recovery[name] = {"true": true.tolist(), "recovered": got.tolist(), "ok": ok.tolist()}

    flux = None
    if fit["model"] == "states":
        flux = _routes_report(model, layout, z.to(device), fit, data.readouts, C, W, values)

    report = {
        "name": fit["name"], "model": fit["model"], "model_text": model.text, "settings": fit,
        "readouts": readouts, "conditions": data.conditions, "windows": spans,
        "window_frames": [[data.first_frame + a, data.first_frame + b] for a, b in spans],
        "interval": dt, "time_unit": tunit, "tracks": int(len(data.y)), "tracks_available": data.available,
        "tracks_per_condition": [int((data.cond == c).sum()) for c in range(C)], "observations": n_obs,
        "log_likelihood": ll, "free_numbers": k, "aic": 2 * k - 2 * ll, "bic": k * math.log(max(n_obs, 1)) - 2 * ll,
        "parameters": params, "derived": der, "recovery": recovery, "mode_names": model.mode_names,
        "bootstrap": len(boots_v), "loss": history, "data_key": _data_key(exp, fit, readouts),
        "range": [model.lo, model.hi], "device": str(device), "sweep": sweep, "flux": flux,
        "inputs": "features" if use_features else "readouts", "feature_names": data.readouts,
        "states": model.n_modes if fit["model"] == "states" else None, "seconds": round(time.time() - t_start, 1),
        "fitted": f"{_dt.datetime.now():%Y-%m-%d %H:%M}",
    }
    report["_checks"] = checks
    if save:
        folder = fit_folder(exp, fit["name"])
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "fit.json", "w") as f:
            json.dump({k: v for k, v in report.items() if not k.startswith("_")}, f, indent=1)
        np.savez_compressed(folder / "checks.npz", **{k: np.asarray(v) for k, v in checks.items()})
        with h5py.File(folder / "states.h5", "w") as f:
            f.attrs["mode_names"] = json.dumps(model.mode_names)
            f.attrs["first_frame"] = data.first_frame
            f.create_dataset("condition", data=data.cond)
            f.create_dataset("position", data=np.array(data.positions, dtype=h5py.string_dtype())[data.position])
            f.create_dataset("track", data=data.track)
            f.create_dataset("modes", data=modes.astype(np.float16), compression="lzf")
            f.create_dataset("latent", data=latent.astype(np.float16), compression="lzf")
            f.create_dataset("readout", data=(data.y[..., 0] * sd[0] + mu[0]).astype(np.float16), compression="lzf")
        report["folder"] = str(folder)
        from .plots import render_fit
        try:
            render_fit(exp, report, read_states(exp, fit["name"]), folder)
        except Exception as exc:  # figures are a convenience; the fit itself is saved
            on_event({"type": "log", "message": f"Could not draw the fit's figures: {exc}"})
    on_event({"type": "log", "message": f"Fit {fit['name']} done in {report['seconds']} s: log-likelihood {ll:.1f}, BIC {report['bic']:.1f}"})
    return report


def _routes_report(model, layout, z, fit, names, C, W, values) -> dict | None:
    """Committor and route flux per condition and window, between the states chosen in the options."""
    o = fit["options"]
    K = model.n_modes
    A = [int(i) - 1 for i in o.get("flux_from") or [] if 1 <= int(i) <= K]
    B = [int(i) - 1 for i in o.get("flux_to") or [] if 1 <= int(i) <= K]
    feat = o.get("flux_feature") or ""
    if (not A or not B) and feat in names:
        f = names.index(feat) + 1
        means = np.array([np.asarray(values[f"mean_{k + 1}_{f}"], float).ravel()[0] for k in range(K)])
        A, B = A or [int(np.argmin(means))], B or [int(np.argmax(means))]
    if not A or not B:
        return None
    with torch.no_grad():
        Q, _, _ = model.build(layout.values(z))
        Q = Q.detach().cpu().numpy().reshape(C, W, K, K)
    com = np.full((C, W, K), np.nan)
    net = np.zeros((C, W, K, K))
    tm = np.full((C, W), np.nan)
    for c in range(C):
        for w in range(W):
            r = routes(Q[c, w], A, B)
            com[c, w], net[c, w], tm[c, w] = r["committor"], r["net"], r["time"]
    return {"from": [a + 1 for a in A], "to": [b + 1 for b in B], "feature": feat, "committor": com.tolist(),
            "net_flux": net.tolist(), "time_to": tm.tolist()}


def read_fit(exp: dict, name: str) -> dict | None:
    folder = fit_folder(exp, name)
    path = folder / "fit.json"
    if not path.exists():
        return None
    with open(path) as f:
        report = json.load(f)
    if (folder / "checks.npz").exists():
        with np.load(folder / "checks.npz") as c:
            report["_checks"] = {k: c[k] for k in c.files}
    report["folder"] = str(folder)
    return report


def read_states(exp: dict, name: str) -> dict | None:
    path = fit_folder(exp, name) / "states.h5"
    if not path.exists():
        return None
    with h5py.File(path) as f:
        out = {k: f[k][()] for k in f}
        out["mode_names"] = json.loads(f.attrs["mode_names"])
        out["first_frame"] = int(f.attrs["first_frame"])
    out["position"] = np.array([p.decode() if isinstance(p, bytes) else p for p in out["position"]])
    return out


def compare(exp: dict) -> list[dict]:
    """Saved fits with their likelihood, AIC and BIC, grouped by the data they saw (lowest BIC first)."""
    rows = []
    for fit in normalise(exp)["analysis"]["dynamics"]["fits"]:
        rep = read_fit(exp, fit["name"])
        if rep:
            rows.append({k: rep[k] for k in ("name", "model", "log_likelihood", "free_numbers", "aic", "bic", "observations", "data_key")})
    rows.sort(key=lambda r: (r["data_key"], r["bic"]))
    return rows


def run_all(exp: dict, on_event=lambda e: None, stop=None) -> list[dict]:
    exp = normalise(exp)
    cache = ResultsCache(exp)
    out = []
    for fit in exp["analysis"]["dynamics"]["fits"]:
        if fit["enabled"]:
            out.append(run_fit(exp, copy.deepcopy(fit), on_event, stop, cache))
    return out
