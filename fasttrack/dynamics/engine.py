"""Fitting a model to many tracks at once: exact likelihood by the forward algorithm.

Every track is a sequence of noisy readouts at frames 0..T-1 (NaN where the cell was not seen).
The hidden state starts from the population's distribution at frame 0 and moves by P = expm(Q Δt)
each frame, so a cell that appears later starts from the population's state at that time and a
gap simply propagates the state without an observation. Each observation y has variance
var_state + se², where se is that observation's own measurement error (pixel noise), so dim,
noisy cells weigh less without being thrown away.

Parameters can be shared by all conditions and times, or differ per condition, per time window,
or both ("share"). Comparing fits with a parameter shared or free asks whether a drug changes it.
Optimisation is Adam on bounded, unconstrained parameters, with minibatches of tracks from every
condition. Uncertainty comes from refitting on positions resampled with replacement (positions
in a well are not independent cells), and a recovery check refits data simulated from the fit.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .models import Model, Spec, stationary

SHARES = ["all", "condition", "window", "condition_window"]
SHARE_TEXT = {"all": "one value", "condition": "per condition", "window": "per time window",
              "condition_window": "per condition and window"}
FRAME_MINUTES = 6.0   # starting values are per minute; with no frame interval they assume 6-minute frames


def pick_device(name: str = "auto") -> torch.device:
    """'auto': the CUDA device with the most free memory, else CPU (fasttrack/device.py)."""
    from ..device import resolve
    return torch.device(resolve(name))


# ---------------------------------------------------------------------- parameters
class Layout:
    """Maps a vector of free numbers to every parameter's value in every (condition, window) set.

    Values are bounded: v = low + (high - low) · sigmoid(z), on a log scale for positive
    parameters. Fixed parameters keep their value. Set s = condition · W + window."""

    def __init__(self, model: Model, overrides: dict, n_cond: int, n_win: int, minutes: bool):
        self.model, self.C, self.W = model, n_cond, n_win
        self.specs: dict[str, Spec] = {}
        self.fixed: dict[str, bool] = {}
        self.share: dict[str, str] = {}
        scale_time = 1.0 if minutes else FRAME_MINUTES
        for name, spec in model.specs().items():
            spec = Spec(**spec.__dict__)
            if spec.unit == "rate":
                spec.default, spec.low, spec.high = (v * scale_time for v in (spec.default, spec.low, spec.high))
            elif spec.unit == "noise" and name == "sigma":
                spec.default, spec.low, spec.high = (v * math.sqrt(scale_time) for v in (spec.default, spec.low, spec.high))
            o = (overrides or {}).get(name) or {}
            for key in ("low", "high"):
                if o.get(key) is not None:
                    setattr(spec, key, float(o[key]))
            if o.get("value") is not None:
                spec.default = float(o["value"])
            spec.default = min(max(spec.default, spec.low), spec.high)
            self.specs[name] = spec
            self.fixed[name] = bool(o.get("fixed", False))
            share = o.get("share") or spec.share
            if spec.unit == "logit":
                share = "condition"  # initial state odds are per condition by construction
            self.share[name] = share if share in SHARES else "all"
        self.offsets: dict[str, torch.Tensor] = {}
        n = 0
        sets = torch.arange(n_cond * n_win)
        c, w = sets // n_win, sets % n_win
        for name in self.specs:
            if self.fixed[name]:
                continue
            share = self.share[name]
            local = {"all": torch.zeros_like(sets), "condition": c, "window": w, "condition_window": sets}[share]
            self.offsets[name] = n + local
            n += int(local.max()) + 1
        self.size = n

    @property
    def free_names(self) -> list[str]:
        return list(self.offsets)

    def _bounds(self, spec: Spec) -> tuple[float, float]:
        return (math.log(spec.low), math.log(spec.high)) if spec.log else (spec.low, spec.high)

    def initial(self, start: dict[str, float] | None = None) -> torch.Tensor:
        z = torch.zeros(self.size, dtype=torch.float64)
        for name, off in self.offsets.items():
            spec = self.specs[name]
            v = (start or {}).get(name, spec.default)
            v = min(max(v, spec.low), spec.high)
            lo, hi = self._bounds(spec)
            frac = ((math.log(v) if spec.log else v) - lo) / (hi - lo)
            frac = min(max(frac, 1e-3), 1 - 1e-3)
            z[off] = math.log(frac / (1 - frac))
        return z

    def values(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        n_sets = self.C * self.W
        for name, spec in self.specs.items():
            if self.fixed[name]:
                out[name] = torch.full((n_sets,), spec.default, dtype=torch.float64, device=z.device)
                continue
            lo, hi = self._bounds(spec)
            v = lo + (hi - lo) * torch.sigmoid(z[self.offsets[name].to(z.device)])
            out[name] = torch.exp(v) if spec.log else v
        return out

    def at_bound(self, z: torch.Tensor) -> dict[str, np.ndarray]:
        """Per parameter and set: True where the fit pressed against a bound (not identified, or bound too tight)."""
        out = {}
        for name, off in self.offsets.items():
            s = torch.sigmoid(z[off.to(z.device)]).detach().cpu().numpy()
            out[name] = (s < 0.01) | (s > 0.99)
        return out


# ---------------------------------------------------------------------- the model as matrices
SURVIVE, DIVIDE, BORN = 1, 2, 4   # event flags per track and frame (see track_events)


def matrices(model: Model, layout: Layout, z: torch.Tensor, dt: float):
    """Transition matrices (C, W, S, S), initial distribution (C, S), emission parameters
    {mean, var (C, W, S, F), and optionally load (C, W, S, F, R), hazard (C, W, S), birth (C, S)}, and the values."""
    p = layout.values(z)
    Q, mean, var = model.build(p)
    P = torch.linalg.matrix_exp(Q * dt).clamp_min(0)
    P = P / P.sum(-1, keepdim=True)
    C, W = layout.C, layout.W
    S = Q.shape[-1]
    if model.initial == "free":
        pi0 = model.initial_probs(p).reshape(C, W, -1)[:, 0]
    else:
        pi0 = stationary(Q).reshape(C, W, S)[:, 0]
    F = mean.shape[-1]
    em = {"mean": mean.reshape(C, W, S, F).float(), "var": var.reshape(C, W, S, F).float(), "dt": float(dt)}
    ex = model.extras(p)
    if "load" in ex:
        em["load"] = ex["load"].reshape(C, W, S, F, -1).float()
    if "hazard" in ex:
        em["hazard"] = ex["hazard"].reshape(C, W, S).float()
    if "birth" in ex:
        em["birth"] = ex["birth"].reshape(C, W, S)[:, 0].float()
    return P.reshape(C, W, S, S).float(), pi0.float(), em, p


def robust(loge: torch.Tensor, n_obs: torch.Tensor, outlier: float, width: float) -> torch.Tensor:
    """Each observation is, with probability `outlier`, a wrong value spread evenly over the readout
    range (a tracking swap, a bad segmentation) instead of a draw from the state. Keeps one bad frame
    from dominating a track, and keeps the likelihood and its gradient finite."""
    if outlier <= 0:
        return loge
    floor = math.log(outlier) - n_obs[..., None] * math.log(width)
    return torch.logaddexp(loge + math.log1p(-outlier), floor.expand_as(loge))


def robust_args(model: Model) -> dict:
    return {"outlier": float(getattr(model, "outlier", 0.01)), "width": max(1e-6, model.hi - model.lo)}


def marginal_var(em: dict) -> torch.Tensor:
    """Each feature's total variance in each state (C, W, S, F): its own plus what the shared directions add."""
    v = em["var"]
    if "load" in em:
        v = v + (em["load"] ** 2).sum(-1)
    return v


def track_events(y: np.ndarray, divides: np.ndarray | None = None, born: np.ndarray | None = None,
                 last: np.ndarray | None = None) -> np.ndarray:
    """(N, T) int8 event flags: SURVIVE on every frame from a track's first to its last observation but the
    last (the cell went on without dividing), DIVIDE on the last if the track ends in a division inside
    the window, BORN on the first if the cell was born there."""
    N, T = y.shape[:2]
    seen = np.isfinite(y).any(-1)
    ev = np.zeros((N, T), np.int8)
    has = seen.any(1)
    first = np.argmax(seen, 1)
    lst = T - 1 - np.argmax(seen[:, ::-1], 1)
    cols = np.arange(T)
    ev[(cols >= first[:, None]) & (cols < lst[:, None]) & has[:, None]] |= SURVIVE
    if divides is not None:
        ends_here = divides & has & (lst == (last if last is not None else lst)) & (lst < T - 1)
        ev[np.nonzero(ends_here)[0], lst[ends_here]] |= DIVIDE
    if born is not None:
        b = born & has
        ev[np.nonzero(b)[0], first[b]] |= BORN
    return ev


def emission(em: dict, y: torch.Tensor, se2: torch.Tensor, wt: torch.Tensor, ev: torch.Tensor | None = None,
             outlier: float = 0.01, width: float = 8.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-likelihood of every frame's observation in every state, (C, B, T, S), and whether the frame
    carries information (C, B, T). Gaussian with covariance diag(var + se²) + L Lᵀ (low rank, Woodbury);
    features missing at a frame are left out exactly. Division events add the hazard's evidence."""
    seen = torch.isfinite(y)                                       # (C, B, T, F)
    obs = seen.float()
    mean = em["mean"][:, wt]                                       # (C, T, S, F)
    D = em["var"][:, wt][:, None] + se2[..., None, :]              # (C, B, T, S, F)
    d = (torch.nan_to_num(y)[..., None, :] - mean[:, None]) * obs[..., None, :]
    Dinv = obs[..., None, :] / D
    quad = (d * d * Dinv).sum(-1)
    logdet = (torch.log(D) * obs[..., None, :]).sum(-1)
    if "load" in em:
        L = em["load"][:, wt][:, None]                             # (C, 1, T, S, F, R)
        LD = L * Dinv[..., None]                                   # rows of missing features are zero
        A = torch.eye(L.shape[-1], device=y.device) + LD.transpose(-1, -2) @ L
        bvec = (LD * d[..., None]).sum(-2)                         # (C, B, T, S, R)
        chol = torch.linalg.cholesky(A)
        sol = torch.cholesky_solve(bvec[..., None], chol)[..., 0]
        quad = quad - (bvec * sol).sum(-1)
        logdet = logdet + 2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    n_obs = obs.sum(-1)                                            # (C, B, T)
    loge = -0.5 * (quad + logdet + n_obs[..., None] * math.log(2 * math.pi))
    loge = robust(loge, n_obs, outlier, width)
    loge = loge * (n_obs[..., None] > 0)
    info = n_obs > 0
    if ev is not None and "hazard" in em:
        hdt = (em["hazard"][:, wt] * em["dt"])[:, None]           # (C, 1, T, S)
        surv = ((ev & SURVIVE) > 0)[..., None]
        div = ((ev & DIVIDE) > 0)[..., None]
        loge = loge - hdt * surv + torch.log(-torch.expm1(-hdt.clamp_min(1e-12))) * div
        info = info | ((ev & (SURVIVE | DIVIDE)) > 0)
    return loge, info


def _step(alpha: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """alpha (C, B, S) times P (C, S, S). For few states a broadcast product beats a batched GEMM."""
    if P.shape[-1] <= 16:
        return (alpha[..., :, None] * P[:, None]).sum(-2)
    return torch.bmm(alpha, P)


def _prepare(P, pi0, em, y, se2, window_of_t, ev, outlier, width):
    """Everything the recursions need, computed for all frames at once: emission factors e (C, B, T, S)
    scaled per frame (1 where a frame carries no information), their log scales m (C, B, T), the
    transition matrix used to enter each frame, and the frames where cells are born."""
    wt = torch.as_tensor(window_of_t, device=y.device)
    loge, info = emission(em, y, se2, wt, ev, outlier, width)
    m = loge.max(-1).values * info
    e = torch.where(info[..., None], torch.exp(loge - m[..., None]), torch.ones_like(loge))
    wl = [int(w) for w in window_of_t]
    trans = [P[:, w] for w in range(P.shape[1])]
    enter = [None] + [trans[wl[t - 1]] for t in range(1, len(wl))]
    births = None
    if ev is not None and "birth" in em:
        born = (ev & BORN) > 0
        frames = born.any(0).any(0).nonzero().flatten().tolist()
        births = {t: born[:, :, t][..., None] for t in frames}
    return e, m, enter, births


TREE_STATES = 16   # up to this many hidden states, the likelihood is a balanced product of per-frame matrices


def _tree_loglik(P, pi0, em, e, m, window_of_t, ev) -> torch.Tensor:
    """log p(track) as pi0 diag(e_0) M_1 ... M_{T-1} 1 with M_t = P_t diag(e_t) (a born cell's M_t has every row
    equal to birth * e_t), multiplied pairwise in log2(T) levels with a scale kept per product. Same value as
    the forward recursion, far fewer sequential steps (for few states)."""
    C, B, T, S = e.shape
    wt = torch.as_tensor(window_of_t, device=e.device)
    Pt = P[:, wt[:-1]]                                             # (C, T-1, S, S): entering frames 1..T-1
    M = Pt[:, None] * e[:, :, 1:, None, :]                         # (C, B, T-1, S, S)
    v = pi0[:, None, :] * e[:, :, 0]                               # (C, B, S)
    if ev is not None and "birth" in em:
        born = (ev & BORN) > 0
        reset = (em["birth"][:, None, None, :] * e[:, :, 1:])[..., None, :].expand_as(M)
        M = torch.where(born[:, :, 1:, None, None], reset, M)
        v = torch.where(born[:, :, 0, None], em["birth"][:, None, :] * e[:, :, 0], v)
    scale = torch.zeros(M.shape[:3], device=e.device)
    while M.shape[2] > 1:
        if M.shape[2] % 2:
            eye = torch.eye(S, device=e.device).expand(C, B, 1, S, S)
            M = torch.cat([M, eye], 2)
            scale = torch.cat([scale, torch.zeros(C, B, 1, device=e.device)], 2)
        M = M[:, :, 0::2] @ M[:, :, 1::2]
        s = M.amax((-1, -2)).clamp_min(1e-30)
        M = M / s[..., None, None]
        scale = scale[:, :, 0::2] + scale[:, :, 1::2] + torch.log(s)
    total = v if M.shape[2] == 0 else (v[..., None, :] @ M[:, :, 0])[..., 0, :]
    sc = scale[:, :, 0] if M.shape[2] else 0
    return torch.log(total.sum(-1).clamp_min(1e-30)) + sc + m.sum(-1)


def forward(P, pi0, em, y, se2, window_of_t, ev=None, predictive: bool = False, outlier: float = 0.01, width: float = 8.0):
    """Log-likelihood per track (C, B) by the scaled forward algorithm.
    y, se2: (C, B, T, F), NaN in y where unobserved. window_of_t: (T,) long. ev: (C, B, T) event flags or None.
    predictive: also return the one-step-ahead predictive state distribution at every frame (C, B, T, S)."""
    C, B, T, F = y.shape
    S = P.shape[-1]
    if not predictive and S <= TREE_STATES and T > 1:
        wt = torch.as_tensor(window_of_t, device=y.device)
        loge, info = emission(em, y, se2, wt, ev, outlier, width)
        m = loge.max(-1).values * info
        e = torch.where(info[..., None], torch.exp(loge - m[..., None]), torch.ones_like(loge))
        return _tree_loglik(P, pi0, em, e, m, window_of_t, ev)
    e, m, enter, births = _prepare(P, pi0, em, y, se2, window_of_t, ev, outlier, width)
    alpha = pi0[:, None, :].expand(C, B, S)
    logc = []
    pred = [] if predictive else None
    for t in range(T):
        if t > 0:
            alpha = _step(alpha, enter[t])
        if births is not None and t in births:
            alpha = torch.where(births[t], em["birth"][:, None, :].expand_as(alpha), alpha)
        if predictive:
            pred.append(alpha)
        a = alpha * e[:, :, t]
        c = a.sum(-1, keepdim=True).clamp_min(1e-30)
        alpha = a / c
        logc.append(c)
    ll = torch.log(torch.cat(logc, -1)).sum(-1) + m.sum(-1)
    if predictive:
        return ll, torch.stack(pred, 2)
    return ll


def posterior(P, pi0, em, y, se2, window_of_t, ev=None, outlier: float = 0.01, width: float = 8.0):
    """Smoothed state probabilities (C, B, T, S) by forward-backward."""
    C, B, T, F = y.shape
    S = P.shape[-1]
    e, _, enter, births = _prepare(P, pi0, em, y, se2, window_of_t, ev, outlier, width)
    alphas = []
    alpha = pi0[:, None, :].expand(C, B, S)
    for t in range(T):
        if t > 0:
            alpha = _step(alpha, enter[t])
        if births is not None and t in births:
            alpha = torch.where(births[t], em["birth"][:, None, :].expand_as(alpha), alpha)
        a = alpha * e[:, :, t]
        alpha = a / a.sum(-1, keepdim=True).clamp_min(1e-30)
        alphas.append(alpha)
    beta = torch.ones(C, B, S, device=y.device)
    out = torch.empty(C, B, T, S, device=y.device)
    for t in range(T - 1, -1, -1):
        g = alphas[t] * beta
        out[:, :, t] = g / g.sum(-1, keepdim=True).clamp_min(1e-30)
        if t > 0:
            nb = _step(e[:, :, t] * beta, enter[t].transpose(1, 2))
            nb = nb / nb.sum(-1, keepdim=True).clamp_min(1e-30)
            if births is not None and t in births:   # a birth at t cuts the link to the past (earlier frames are not this cell's)
                nb = torch.where(births[t], torch.ones_like(nb), nb)
            beta = nb
    return out


# ---------------------------------------------------------------------- data on the device
class Batches:
    """Tracks grouped by condition, on the device, drawn as (C, B, T, F) minibatches (with event flags)."""

    def __init__(self, y: np.ndarray, se: np.ndarray, cond: np.ndarray, n_cond: int, device, index: np.ndarray | None = None,
                 ev: np.ndarray | None = None):
        self.device = device
        self.C = n_cond
        index = np.arange(len(y)) if index is None else np.asarray(index)
        self.rows = [index[cond[index] == c] for c in range(n_cond)]
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).to(device)
        self.se2 = torch.from_numpy(np.ascontiguousarray(np.nan_to_num(se, nan=0.0) ** 2, dtype=np.float32)).to(device)
        self.ev = None if ev is None else torch.from_numpy(np.ascontiguousarray(ev, dtype=np.int8)).to(device)
        self.T, self.F = y.shape[1], y.shape[2]

    def counts(self) -> np.ndarray:
        return np.array([len(r) for r in self.rows])

    def n_obs(self) -> int:
        """Observations: frames where a track has at least one feature, over the rows in use."""
        rows = np.concatenate([r for r in self.rows if len(r)]) if any(len(r) for r in self.rows) else np.zeros(0, int)
        if not len(rows):
            return 0
        return int(torch.isfinite(self.y[torch.as_tensor(rows, device=self.device)]).any(-1).sum())

    def draw(self, size: int, rng: np.random.Generator):
        """A minibatch from every condition: y, se2, ev (or None), and each condition's weight (tracks / drawn)."""
        picks, weights = [], []
        for r in self.rows:
            if len(r) == 0:
                picks.append(np.empty(0, int))
                weights.append(0.0)
                continue
            k = min(size, len(r))
            picks.append(rng.choice(r, k, replace=False))
            weights.append(len(r) / k)
        return (*self._stack(picks), torch.tensor(weights, device=self.device))

    def chunks(self, size: int):
        """Every track once: yields y, se2, ev blocks (C, B, ...) and the original row of each entry (-1 = padding)."""
        longest = max((len(r) for r in self.rows), default=0)
        for start in range(0, longest, size):
            picks = [r[start:start + size] for r in self.rows]
            y, se2, ev = self._stack(picks)
            rows = np.full((self.C, y.shape[1]), -1)
            for c, p in enumerate(picks):
                rows[c, :len(p)] = p
            yield y, se2, ev, rows

    def _stack(self, picks):
        B = max(1, max(len(p) for p in picks))
        y = torch.full((self.C, B, self.T, self.F), float("nan"), device=self.device)
        se2 = torch.zeros((self.C, B, self.T, self.F), device=self.device)
        ev = None if self.ev is None else torch.zeros((self.C, B, self.T), dtype=torch.int8, device=self.device)
        for c, p in enumerate(picks):
            if len(p):
                idx = torch.as_tensor(p, device=self.device)
                y[c, :len(p)] = self.y[idx]
                se2[c, :len(p)] = self.se2[idx]
                if ev is not None:
                    ev[c, :len(p)] = self.ev[idx]
        return y, se2, ev


# ---------------------------------------------------------------------- fitting
def fit(model: Model, layout: Layout, batches: Batches, window_of_t: np.ndarray, dt: float, steps: int = 300,
        batch: int = 256, lr: float = 0.05, seed: int = 0, z0: torch.Tensor | None = None, start: dict | None = None,
        on_step=None, stop=None) -> tuple[torch.Tensor, list[float]]:
    """Maximise the likelihood. Returns the free vector z and the loss history (-log-likelihood per observation)."""
    device = batches.device
    rng = np.random.default_rng(seed)
    z = (layout.initial(start) if z0 is None else z0.detach().clone()).to(device).requires_grad_(True)
    if layout.size == 0:
        return z.detach(), []
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.05)
    wt = torch.as_tensor(window_of_t, device=device)
    n_obs = max(1.0, float(batches.n_obs()))
    history = []
    for step in range(steps):
        if stop is not None and stop.is_set():
            break
        y, se2, ev, weight = batches.draw(batch, rng)
        P, pi0, em, _ = matrices(model, layout, z, dt)
        ll = forward(P, pi0, em, y, se2, wt, ev, **robust_args(model))
        loss = -(ll.sum(1) * weight).sum() / n_obs
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], 10.0)
        opt.step()
        sched.step()
        history.append(float(loss.detach()))
        if on_step is not None:
            on_step(step, history[-1])
    return z.detach(), history


@torch.no_grad()
def log_likelihood(model: Model, layout: Layout, batches: Batches, window_of_t, dt: float, z: torch.Tensor, chunk: int = 1024) -> float:
    P, pi0, em, _ = matrices(model, layout, z, dt)
    wt = torch.as_tensor(window_of_t, device=batches.device)
    total = 0.0
    for y, se2, ev, _ in batches.chunks(chunk):
        total += float(forward(P, pi0, em, y, se2, wt, ev, **robust_args(model)).sum())
    return total


@torch.no_grad()
def state_posteriors(model: Model, layout: Layout, batches: Batches, window_of_t, dt: float, z: torch.Tensor, chunk: int = 512):
    """Per track (in the order of the rows given to Batches): mode probabilities (N, T, M) and the
    posterior mean of the latent readout (N, T) when the model has one (NaN otherwise)."""
    P, pi0, em, _ = matrices(model, layout, z, dt)
    wt = torch.as_tensor(window_of_t, device=batches.device)
    mode = model.mode_of_state(batches.device)
    M = model.n_modes
    onehot = torch.nn.functional.one_hot(mode, M).float()
    latent = model.latent_of_state(batches.device)
    N = batches.y.shape[0]
    modes = np.full((N, batches.T, M), np.nan, np.float32)
    lat = np.full((N, batches.T), np.nan, np.float32)
    for y, se2, ev, rows in batches.chunks(chunk):
        post = posterior(P, pi0, em, y, se2, wt, ev, **robust_args(model))              # (C, B, T, S)
        pm = post @ onehot
        pl = (post @ latent.float()) if latent is not None else None
        for c in range(rows.shape[0]):
            ok = rows[c] >= 0
            modes[rows[c][ok]] = pm[c, ok].cpu().numpy()
            if pl is not None:
                lat[rows[c][ok]] = pl[c, ok].cpu().numpy()
    return modes, lat


# ---------------------------------------------------------------------- checks
def _acf(y: np.ndarray, lags: int) -> np.ndarray:
    """Autocorrelation of tracks (N, T), each track centred on its own mean, pooled over tracks."""
    yc = y - np.nanmean(y, 1, keepdims=True)
    out = np.full(lags + 1, np.nan)
    for L in range(lags + 1):
        a, b = yc[:, L:], yc[:, :yc.shape[1] - L]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10:
            out[L] = np.sum(a[ok] * b[ok]) / np.sqrt(np.sum(a[ok] ** 2) * np.sum(b[ok] ** 2))
    return out


@torch.no_grad()
def simulate_discrete(model: Model, layout: Layout, batches: Batches, window_of_t, dt: float, z: torch.Tensor,
                      seed: int = 0, chunk: int = 2048) -> np.ndarray:
    """Data drawn from the fitted model with exactly the observed tracks' frames and pixel noise: the
    hidden state moves by P each frame and each observation gets var_state + se² noise. Used for the
    model checks and the recovery check. Returns y (N, T, F) in the rows' order, NaN where unobserved."""
    P, pi0, em, _ = matrices(model, layout, z, dt)
    mean, var, load = em["mean"], em["var"], em.get("load")
    gen = torch.Generator(device=batches.device).manual_seed(seed)
    N = batches.y.shape[0]
    out = np.full((N, batches.T, batches.F), np.nan, np.float32)
    for y, se2, ev, rows in batches.chunks(chunk):
        C, B, T, F = y.shape
        state = torch.multinomial(pi0.repeat_interleave(B, 0), 1, generator=gen).reshape(C, B)
        sim = torch.empty_like(y)
        cidx = torch.arange(C, device=y.device)[:, None].expand(C, B)
        for t in range(T):
            w = int(window_of_t[t])
            if t > 0:
                probs = P[cidx, int(window_of_t[t - 1]), state]       # (C, B, S)
                state = torch.multinomial(probs.reshape(C * B, -1), 1, generator=gen).reshape(C, B)
            if ev is not None and "birth" in em:
                born = (ev[:, :, t] & BORN) > 0
                if born.any():
                    fresh = torch.multinomial(em["birth"].repeat_interleave(B, 0), 1, generator=gen).reshape(C, B)
                    state = torch.where(born, fresh, state)
            m = mean[cidx, w, state]                                   # (C, B, F)
            v = var[cidx, w, state] + se2[:, :, t]
            sim[:, :, t] = m + v.sqrt() * torch.randn(m.shape, generator=gen, device=y.device)
            if load is not None:
                Lw = load[cidx, w, state]                              # (C, B, F, R)
                sim[:, :, t] += (Lw @ torch.randn(*Lw.shape[:2], Lw.shape[-1], 1, generator=gen, device=y.device)).squeeze(-1)
        sim[~torch.isfinite(y)] = float("nan")
        for c in range(C):
            ok = rows[c] >= 0
            out[rows[c][ok]] = sim[c, ok].cpu().numpy()
    return out


@torch.no_grad()
def predictive_checks(model: Model, layout: Layout, batches: Batches, window_of_t, dt: float, z: torch.Tensor,
                      bins: np.ndarray, lags: int = 30, seed: int = 0, chunk: int = 512) -> dict:
    """What the fitted model predicts against what was seen, per condition (first readout):
      hist_obs / hist_pred (C, W, bins): the population distribution in each time window, observed and
        predicted (each observation's one-step-ahead predictive density, averaged);
      pit (C, 20): histogram of the probability integral transform of every observation given the
        cell's past. Flat if the model is right; U-shaped = the model is too sure; a hump = too vague;
      acf_obs / acf_sim (C, lags+1): autocorrelation of the tracks, and of data simulated from the fit
        with the same frames and noise."""
    P, pi0, em, _ = matrices(model, layout, z, dt)
    mean, var = em["mean"], marginal_var(em)
    wt = torch.as_tensor(window_of_t, device=batches.device)
    C, W = layout.C, layout.W
    centres = torch.as_tensor((bins[1:] + bins[:-1]) / 2, dtype=torch.float32, device=batches.device)
    width = float(bins[1] - bins[0])
    hist_obs = np.zeros((C, W, len(centres)))
    hist_pred = np.zeros((C, W, len(centres)))
    pit = np.zeros((C, 20))
    for y, se2, ev, rows in batches.chunks(chunk):
        _, pred = forward(P, pi0, em, y, se2, wt, ev, predictive=True, **robust_args(model))   # (C, B, T, S)
        for t in range(y.shape[2]):
            w = int(window_of_t[t])
            yt = y[:, :, t, 0]
            ok = torch.isfinite(yt)
            if not ok.any():
                continue
            v = var[:, w, :, 0][:, None] + se2[:, :, t, 0][:, :, None]       # (C, B, S)
            mu = mean[:, w, :, 0][:, None].expand_as(v)
            sd = v.sqrt()
            weights = pred[:, :, t]                                         # (C, B, S)
            dens = torch.exp(-0.5 * ((centres[None, None, None] - mu[..., None]) / sd[..., None]) ** 2) / (sd[..., None] * math.sqrt(2 * math.pi))
            mix = (weights[..., None] * dens).sum(2) * width                   # (C, B, bins)
            hist_pred[:, w] += (mix * ok[..., None]).sum(1).cpu().numpy()
            cdf = (weights * 0.5 * (1 + torch.erf((torch.nan_to_num(yt)[..., None] - mu) / (sd * math.sqrt(2))))).sum(-1)
            for c in range(C):
                vals = yt[c][ok[c]].cpu().numpy()
                hist_obs[c, w] += np.histogram(vals, bins)[0]
                pit[c] += np.histogram(cdf[c][ok[c]].cpu().numpy(), 20, (0, 1))[0]
    sim = simulate_discrete(model, layout, batches, window_of_t, dt, z, seed)
    acf_obs = np.full((C, lags + 1), np.nan)
    acf_sim = np.full((C, lags + 1), np.nan)
    y_all = batches.y[..., 0].cpu().numpy()
    for c, r in enumerate(batches.rows):
        if len(r):
            acf_obs[c] = _acf(y_all[r], lags)
            acf_sim[c] = _acf(sim[r, :, 0], lags)
    return {"bins": bins, "hist_obs": hist_obs, "hist_pred": hist_pred, "pit": pit, "acf_obs": acf_obs, "acf_sim": acf_sim}


def timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0
