"""Stochastic models of a reporter readout, written as continuous-time Markov generators.

Every model is a generator Q over a finite set of hidden states. Between two frames the hidden
state moves by P = expm(Q Δt), so switching much faster than the imaging interval is part of the
model, not lost between frames: many cells observed at Δt pin down rates above 1/Δt as long as
the reporter keeps a trace of them (its response time, the spread of values between frames).

Continuous latent variables (an SDE  du = a(u) dt + σ dW) are put on a grid of G cells with the
Scharfetter-Gummel scheme, a finite-volume Fokker-Planck discretisation that stays positive and
exact for constant drift. Discrete modes (ERK off / on) multiply the grid: S = modes × G states.

Each model describes its parameters (Spec), builds Q, the emission mean and sd of every state,
derived quantities (dwell times, response times), and simulates tracks in continuous time for
recovery checks. Rates are per time unit (minutes when the frame interval is known).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Spec:
    default: float
    low: float
    high: float
    log: bool = True          # positive and searched on a log scale
    unit: str = "rate"        # rate (per time unit), level (readout units), noise (readout sd), time, logit, value
    share: str = "condition"  # default sharing: all, condition, window, condition_window
    text: str = ""


# ---------------------------------------------------------------------- grid
def sg_bernoulli(z: torch.Tensor) -> torch.Tensor:
    """B(z) = z / (e^z - 1), computed without overflow (and with finite gradients) for any z."""
    small = z.abs() < 1e-4
    zn = torch.where(z < -1e-4, z, -torch.ones_like(z))   # negative side: z / expm1(z)
    zp = torch.where(z > 1e-4, z, torch.ones_like(z))     # positive side: z e^-z / (1 - e^-z)
    neg = zn / torch.expm1(zn)
    pos = zp * torch.exp(-zp) / (-torch.expm1(-zp))
    return torch.where(small, 1 - z / 2, torch.where(z < 0, neg, pos))


def fokker_planck(drift_face: torch.Tensor, D: torch.Tensor, h: float) -> torch.Tensor:
    """Generator (..., G, G) of du = a dt + sqrt(2D) dW on a uniform grid with reflecting ends.
    drift_face (..., G-1): drift at the faces between cells. D (...) or (..., G-1): diffusion."""
    D = D.unsqueeze(-1) if D.dim() == drift_face.dim() - 1 else D
    D = D.clamp_min(1e-12)
    z = drift_face * h / D
    right = D / h ** 2 * sg_bernoulli(-z)   # cell i -> i+1
    left = D / h ** 2 * sg_bernoulli(z)     # cell i+1 -> i
    G = drift_face.shape[-1] + 1
    Q = torch.zeros(*drift_face.shape[:-1], G, G, dtype=drift_face.dtype, device=drift_face.device)
    i = torch.arange(G - 1, device=drift_face.device)
    Q[..., i, i + 1] = right
    Q[..., i + 1, i] = left
    return Q - torch.diag_embed(Q.sum(-1))


def with_modes(blocks: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Full generator (P, M*G, M*G) from per-mode grid generators (P, M, G, G) and mode switching
    rates K (P, M, M) (off-diagonal; the diagonal is ignored). Switching keeps the grid position."""
    P, M, G, _ = blocks.shape
    K = K * (1 - torch.eye(M, device=K.device, dtype=K.dtype))
    K = K - torch.diag_embed(K.sum(-1))
    eye = torch.eye(G, device=K.device, dtype=K.dtype)
    switching = K[:, :, None, :, None] * eye[None, None, :, None, :]
    eye_m = torch.eye(M, device=K.device, dtype=K.dtype)
    within = eye_m[None, :, None, :, None] * blocks[:, :, :, None, :]
    return (switching + within).reshape(P, M * G, M * G)


def stationary(Q: torch.Tensor) -> torch.Tensor:
    """Stationary distribution (P, S) of generators (P, S, S): solves (Qᵀ + 1 1ᵀ) π = 1, which is
    well conditioned for an irreducible chain however stiff its rates."""
    P, S, _ = Q.shape
    A = Q.transpose(1, 2) + 1.0
    b = torch.ones(P, S, 1, dtype=Q.dtype, device=Q.device)
    pi = torch.linalg.solve(A, b).squeeze(-1).clamp_min(0)
    return pi / pi.sum(-1, keepdim=True).clamp_min(1e-300)


# ---------------------------------------------------------------------- base
class Model:
    """A model: parameter specs, generator, emissions, derived quantities, simulation."""
    name = ""
    text = ""
    features = 1          # readouts it fits at once
    initial = "stationary"  # or "free": initial state probabilities are parameters (init_k)

    def __init__(self, grid: int = 48, **options):
        self.G = int(grid)
        self.options = options
        self.lo, self.hi = -3.0, 3.0

    def set_range(self, lo: float, hi: float) -> None:
        self.lo, self.hi = float(lo), float(hi)

    # grid helpers
    @property
    def h(self) -> float:
        return (self.hi - self.lo) / self.G

    def centres(self, device) -> torch.Tensor:
        return self.lo + (torch.arange(self.G, device=device, dtype=torch.float64) + 0.5) * self.h

    def faces(self, device) -> torch.Tensor:
        return self.lo + torch.arange(1, self.G, device=device, dtype=torch.float64) * self.h

    # to implement
    def specs(self) -> dict[str, Spec]:
        raise NotImplementedError

    def build(self, p: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(Q (P,S,S), emission mean (P,S,F), emission variance (P,S,F)) for P parameter sets."""
        raise NotImplementedError

    def extras(self, p: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Optional emission extras: "load" (P,S,F,R) low-rank covariance loadings, "hazard" (P,S) division
        rate of every state, "birth" (P,S) where cells born in the movie start."""
        return {}

    def mode_of_state(self, device) -> torch.Tensor:
        """Mode index (S,) of every hidden state, for posteriors."""
        return torch.zeros(self.n_states, dtype=torch.long, device=device)

    @property
    def n_modes(self) -> int:
        return 1

    @property
    def mode_names(self) -> list[str]:
        return [f"state {k + 1}" for k in range(self.n_modes)]

    @property
    def n_states(self) -> int:
        return self.n_modes * self.G

    def latent_of_state(self, device) -> torch.Tensor | None:
        """Latent value (S,) of every state (posterior mean of the latent), or None."""
        return self.centres(device).repeat(self.n_modes)

    def derived(self, p: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, str, str]]:
        """{name: (values (P,), unit, text)}."""
        return {}

    def simulate(self, p: dict[str, float], n: int, T: int, dt: float, gen: torch.Generator, device="cpu",
                 substeps: int = 40) -> torch.Tensor:
        """(n, T, F) readouts in continuous time, without measurement noise (added by the caller)."""
        raise NotImplementedError

    def start_values(self, y: torch.Tensor) -> dict[str, float]:
        """Data-informed starting values (y: observed readouts, flat, finite)."""
        return {}


def _rates_matrix(M: int, p: dict, names_up: list[str], names_down: list[str]) -> torch.Tensor:
    """Birth-death chain between M ordered modes: up_i (i -> i+1), down_i (i+1 -> i)."""
    first = p[names_up[0]]
    K = torch.zeros(first.shape[0], M, M, dtype=first.dtype, device=first.device)
    for i in range(M - 1):
        K[:, i, i + 1] = p[names_up[i]]
        K[:, i + 1, i] = p[names_down[i]]
    return K


# ---------------------------------------------------------------------- ERK -> KTR
class ErkKtr(Model):
    """ERK switches between discrete activity modes (a birth-death chain; 2 modes = off / on) at
    rates that may be much faster than the imaging. The KTR follows ERK with its own kinetics:
    nuclear import k_i and ERK-driven export k_e,m = k_i · e^{L_m}, so the cytoplasmic fraction r obeys
        dr/dt = k_e,m (1 - r) - k_i r,
    and the observed ln(C/N) = u = logit(r) relaxes towards L_m with rate k_i (1 + e^{L_m}):
        du = [k_e,m (1 + e^{-u}) - k_i (1 + e^{u})] dt + σ dW.
    L_m is the ln(C/N) a cell would settle at if ERK stayed in mode m. The reporter is a low-pass
    filter: ERK switching much faster than the KTR response shows only as a spread of values, so
    the fit separates switching rates from their ratio only when the data hold that information
    (see the recovery check)."""
    name = "erk_ktr"
    text = "ERK modes switching in continuous time, read through KTR translocation kinetics"

    def __init__(self, grid: int = 48, modes: int = 2, **options):
        super().__init__(grid, **options)
        self.M = int(modes)

    @property
    def n_modes(self) -> int:
        return self.M

    @property
    def mode_names(self) -> list[str]:
        if self.M == 2:
            return ["ERK off", "ERK on"]
        return [f"ERK level {k + 1}" for k in range(self.M)]

    def _switch_names(self) -> tuple[list[str], list[str]]:
        if self.M == 2:
            return ["k_on"], ["k_off"]
        return [f"up_{i + 1}" for i in range(self.M - 1)], [f"down_{i + 1}" for i in range(self.M - 1)]

    def specs(self) -> dict[str, Spec]:
        up, down = self._switch_names()
        out = {}
        for i, (a, b) in enumerate(zip(up, down)):
            out[a] = Spec(0.05, 1e-4, 20.0, text=f"ERK switching rate from {self.mode_names[i]} to {self.mode_names[i + 1]}")
            out[b] = Spec(0.1, 1e-4, 20.0, text=f"ERK switching rate from {self.mode_names[i + 1]} to {self.mode_names[i]}")
        levels = torch.linspace(-1.5, 1.0, self.M).tolist()
        for m in range(self.M):
            out[f"level_{m + 1}"] = Spec(levels[m], -6.0, 6.0, log=False, unit="level", share="all",
                                          text=f"ln(C/N) the KTR settles at in {self.mode_names[m]}")
        out["k_import"] = Spec(0.5, 1e-3, 50.0, share="all", text="KTR nuclear import rate; sets the reporter's response time")
        out["sigma"] = Spec(0.05, 1e-4, 5.0, unit="noise", share="all",
                            text="intrinsic fluctuation of ln(C/N), per √time unit (fast processes not in the model)")
        out["noise"] = Spec(0.1, 1e-3, 3.0, unit="noise", share="all",
                            text="measurement sd of ln(C/N) on top of the pixel-noise estimate of each observation")
        return out

    def _drift(self, u: torch.Tensor, level: torch.Tensor, k_i: torch.Tensor) -> torch.Tensor:
        k_e = k_i * torch.exp(level)
        return k_e * (1 + torch.exp(-u)) - k_i * (1 + torch.exp(u))

    def build(self, p):
        dev = p["k_import"].device
        f = self.faces(dev)
        k_i = p["k_import"]
        levels = torch.stack([p[f"level_{m + 1}"] for m in range(self.M)], 1)             # (P, M)
        drift = self._drift(f[None, None, :], levels[:, :, None], k_i[:, None, None])      # (P, M, G-1)
        D = (p["sigma"] ** 2 / 2)[:, None].expand(-1, self.M)
        blocks = fokker_planck(drift, D, self.h)
        up, down = self._switch_names()
        Q = with_modes(blocks, _rates_matrix(self.M, p, up, down))
        P = Q.shape[0]
        mean = self.centres(dev).repeat(self.M)[None, :, None].expand(P, -1, 1)
        var = (p["noise"] ** 2 + self.h ** 2 / 12)[:, None, None].expand(P, self.n_states, 1)
        return Q, mean, var

    def mode_of_state(self, device):
        return torch.arange(self.M, device=device).repeat_interleave(self.G)

    def derived(self, p):
        up, down = self._switch_names()
        K = _rates_matrix(self.M, p, up, down)
        exit_rate = K.sum(-1)
        Kq = K - torch.diag_embed(exit_rate)
        frac = stationary(Kq)
        out = {}
        for m in range(self.M):
            out[f"dwell_{m + 1}"] = (1 / exit_rate[:, m], "time", f"mean time in {self.mode_names[m]} before switching")
            out[f"fraction_{m + 1}"] = (frac[:, m], "fraction", f"long-run share of time in {self.mode_names[m]}")
            level = p[f"level_{m + 1}"]
            out[f"ktr_response_{m + 1}"] = (1 / (p["k_import"] * (1 + torch.exp(level))), "time",
                                            f"KTR relaxation time near {self.mode_names[m]}'s level")
        if self.M == 2:
            out["cycle"] = (1 / p["k_on"] + 1 / p["k_off"], "time", "mean time for one off → on → off cycle")
            out["switches_per_time"] = (2 * p["k_on"] * p["k_off"] / (p["k_on"] + p["k_off"]), "rate",
                                        "ERK switches (either way) per time unit, at steady state")
        return out

    def start_values(self, y):
        qs = torch.quantile(y, torch.linspace(0.15, 0.85, self.M, dtype=y.dtype, device=y.device))
        return {f"level_{m + 1}": float(qs[m]) for m in range(self.M)}

    def simulate(self, p, n, T, dt, gen, device="cpu", substeps=40):
        up, down = self._switch_names()
        pt = {k: torch.tensor([float(v)], dtype=torch.float64, device=device) for k, v in p.items()}
        K = _rates_matrix(self.M, pt, up, down)[0]
        levels = torch.tensor([p[f"level_{m + 1}"] for m in range(self.M)], dtype=torch.float64, device=device)
        k_i, sigma = float(p["k_import"]), float(p["sigma"])
        # start at steady state of the mode chain, latent at its mode's level
        frac = stationary((K - torch.diag(K.sum(-1)))[None])[0]
        mode = torch.multinomial(frac.float(), n, replacement=True, generator=gen).to(device)
        u = levels[mode].clone()
        h = dt / substeps
        out = torch.empty(n, T, dtype=torch.float64, device=device)
        burn = int(5 * substeps)
        for step in range(-burn, T * substeps):
            if step >= 0 and step % substeps == 0:
                out[:, step // substeps] = u
            rate_out = K[mode].sum(-1)
            jump = torch.rand(n, generator=gen, dtype=torch.float64).to(device) < -torch.expm1(-rate_out * h)
            if jump.any():
                target = torch.multinomial(K[mode[jump]].float().clamp_min(1e-30), 1, generator=gen).squeeze(1).to(device)
                mode = mode.clone()
                mode[jump] = target
            u = u + self._drift(u, levels[mode], torch.tensor(k_i, dtype=torch.float64)) * h \
                + sigma * math.sqrt(h) * torch.randn(n, generator=gen, dtype=torch.float64).to(device)
            u = u.clamp(self.lo, self.hi)
        return out[:, :, None]


# ---------------------------------------------------------------------- OU
class OrnsteinUhlenbeck(Model):
    """The readout itself is a mean-reverting random walk, du = θ (μ - u) dt + σ dW. The simplest
    model with a single correlation time 1/θ; a baseline for anything with discrete states."""
    name = "ou"
    text = "Mean-reverting random walk (Ornstein-Uhlenbeck): one correlation time"

    def specs(self):
        return {"theta": Spec(0.05, 1e-4, 20.0, text="mean-reversion rate; 1/θ is the correlation time"),
                "mu": Spec(0.0, -10.0, 10.0, log=False, unit="level", text="mean readout"),
                "sigma": Spec(0.1, 1e-4, 5.0, unit="noise", share="all", text="noise intensity per √time unit"),
                "noise": Spec(0.1, 1e-3, 3.0, unit="noise", share="all", text="measurement sd on top of each observation's pixel-noise estimate")}

    def build(self, p):
        dev = p["theta"].device
        f = self.faces(dev)
        drift = p["theta"][:, None, None] * (p["mu"][:, None, None] - f[None, None, :])
        Q = fokker_planck(drift, (p["sigma"] ** 2 / 2)[:, None], self.h)[:, 0]
        P = Q.shape[0]
        mean = self.centres(dev)[None, :, None].expand(P, -1, 1)
        var = (p["noise"] ** 2 + self.h ** 2 / 12)[:, None, None].expand(P, self.G, 1)
        return Q, mean, var

    def derived(self, p):
        return {"correlation_time": (1 / p["theta"], "time", "time for a fluctuation to decay to 1/e"),
                "spread": (p["sigma"] / torch.sqrt(2 * p["theta"]), "level", "sd of the readout across cells at steady state")}

    def start_values(self, y):
        return {"mu": float(y.median())}

    def simulate(self, p, n, T, dt, gen, device="cpu", substeps=40):
        th, mu, s = float(p["theta"]), float(p["mu"]), float(p["sigma"])
        # exact OU transition
        a = math.exp(-th * dt)
        sd = s * math.sqrt((1 - a * a) / (2 * th))
        u = mu + s / math.sqrt(2 * th) * torch.randn(n, generator=gen, dtype=torch.float64)
        out = torch.empty(n, T, dtype=torch.float64)
        for t in range(T):
            out[:, t] = u
            u = mu + (u - mu) * a + sd * torch.randn(n, generator=gen, dtype=torch.float64)
        return out.clamp(self.lo, self.hi)[:, :, None].to(device)


# ---------------------------------------------------------------------- continuous-time HMM
class States(Model):
    """A continuous-time hidden Markov model: K discrete states, each with its own profile over every
    readout or feature, and transition rates between them. Rates (not per-frame probabilities) make
    dwell times comparable between experiments imaged at different intervals and allow dwell times
    near or below the interval.

    Emissions are Gaussian per state. With rank R > 0 each state's covariance is diagonal plus R shared
    directions (low rank), so correlated features (nuclear area and DNA content ...) are not counted as
    independent evidence. A feature missing for a cell at a frame is left out of that observation
    exactly. With divisions, each state has a division rate: every frame a tracked cell goes on without
    dividing is evidence against dividing in its state, and an observed division is evidence for it,
    so states with high division rates are the cycle's late phases. Cells born in the movie start from
    their own (post-mitotic) state distribution instead of the population's."""
    name = "states"
    text = "Hidden Markov model in continuous time: discrete states, one or more readouts or features"
    initial = "free"

    def __init__(self, grid: int = 1, states: int = 3, features: int = 1, topology: str = "full", rank: int = 0,
                 division: bool = False, **options):
        super().__init__(1, **options)
        self.K = int(states)
        self.features = int(features)
        self.topology = topology
        self.rank = max(0, min(int(rank), self.features - 1)) if self.features > 1 else 0
        self.division = bool(division)
        self.feature_names = options.get("feature_names") or [f"readout {f + 1}" for f in range(self.features)]

    @property
    def n_modes(self) -> int:
        return self.K

    @property
    def n_states(self) -> int:
        return self.K

    def latent_of_state(self, device):
        return None

    def pairs(self) -> list[tuple[int, int]]:
        if self.topology == "chain":
            return [(i, i + 1) for i in range(self.K - 1)] + [(i + 1, i) for i in range(self.K - 1)]
        return [(i, j) for i in range(self.K) for j in range(self.K) if i != j]

    def specs(self):
        out = {}
        for i, j in self.pairs():
            out[f"rate_{i + 1}_{j + 1}"] = Spec(0.02, 1e-5, 20.0, text=f"rate of switching from state {i + 1} to state {j + 1}")
        means = torch.linspace(-1.2, 1.2, self.K).tolist()
        rng = np.random.default_rng(7)
        spread = rng.normal(0, 0.4, (self.K, self.features))   # states start apart in every feature, not only the first
        for k in range(self.K):
            for f in range(self.features):
                start = means[k] if f == 0 else float(spread[k, f])
                out[f"mean_{k + 1}_{f + 1}"] = Spec(start, -6.0, 6.0, log=False, unit="level", share="all",
                                                    text=f"state {k + 1}: mean of {self.feature_names[f]}")
                out[f"sd_{k + 1}_{f + 1}"] = Spec(0.5, 0.02, 5.0, unit="noise", share="all",
                                                  text=f"state {k + 1}: sd of {self.feature_names[f]} not shared with other features")
                for r in range(self.rank):
                    out[f"load_{k + 1}_{f + 1}_{r + 1}"] = Spec(float(rng.normal(0, 0.2)), -4.0, 4.0, log=False, unit="value",
                                                                share="all", text=f"state {k + 1}: weight of {self.feature_names[f]} "
                                                                                  f"on shared direction {r + 1} (correlations)")
        for k in range(1, self.K):
            out[f"init_{k + 1}"] = Spec(0.0, -8.0, 8.0, log=False, unit="logit", share="condition",
                                        text=f"log odds of starting in state {k + 1} rather than state 1 (first frame)")
        if self.division:
            for k in range(self.K):
                out[f"division_{k + 1}"] = Spec(0.001, 1e-6, 1.0, share="all", text=f"rate of dividing while in state {k + 1}")
            for k in range(1, self.K):
                out[f"birth_{k + 1}"] = Spec(0.0, -8.0, 8.0, log=False, unit="logit", share="condition",
                                             text=f"log odds that a newborn cell starts in state {k + 1} rather than state 1")
        return out

    def build(self, p):
        first = next(iter(p.values()))
        P, dev = first.shape[0], first.device
        Q = torch.zeros(P, self.K, self.K, dtype=first.dtype, device=dev)
        for i, j in self.pairs():
            Q[:, i, j] = p[f"rate_{i + 1}_{j + 1}"]
        Q = Q - torch.diag_embed(Q.sum(-1))
        mean = torch.stack([torch.stack([p[f"mean_{k + 1}_{f + 1}"] for f in range(self.features)], -1) for k in range(self.K)], 1)
        var = torch.stack([torch.stack([p[f"sd_{k + 1}_{f + 1}"] ** 2 for f in range(self.features)], -1) for k in range(self.K)], 1)
        return Q, mean, var

    def extras(self, p):
        out = {}
        if self.rank:
            out["load"] = torch.stack([torch.stack([torch.stack([p[f"load_{k + 1}_{f + 1}_{r + 1}"] for r in range(self.rank)], -1)
                                                    for f in range(self.features)], 1) for k in range(self.K)], 1)   # (P,S,F,R)
        if self.division:
            out["hazard"] = torch.stack([p[f"division_{k + 1}"] for k in range(self.K)], -1)
            first = next(iter(p.values()))
            logits = torch.stack([torch.zeros_like(first)] + [p[f"birth_{k + 1}"] for k in range(1, self.K)], -1)
            out["birth"] = torch.softmax(logits, -1)
        return out

    def initial_probs(self, p):
        first = next(iter(p.values()))
        logits = torch.stack([torch.zeros_like(first)] + [p[f"init_{k + 1}"] for k in range(1, self.K)], -1)
        return torch.softmax(logits, -1)

    def mode_of_state(self, device):
        return torch.arange(self.K, device=device)

    def derived(self, p):
        Q, _, _ = self.build(p)
        frac = stationary(Q)
        out = {}
        for k in range(self.K):
            out[f"dwell_{k + 1}"] = (1 / (-Q[:, k, k]), "time", f"mean time in state {k + 1} before leaving")
            out[f"fraction_{k + 1}"] = (frac[:, k], "fraction", f"long-run share of time in state {k + 1}")
        return out

    def start_values(self, y):
        """Means from k-means on the observations (features with missing values filled by their median), so
        states start where the data are dense in every feature at once."""
        y = y if y.dim() > 1 else y[:, None]
        y = y.double().cpu()
        med = torch.nanmedian(y, 0).values if y.shape[0] else torch.zeros(y.shape[1], dtype=y.dtype)
        x = torch.where(torch.isfinite(y), y, med)
        if x.shape[0] > 50000:
            x = x[torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(0))[:50000]]
        if x.shape[0] < self.K:
            return {}
        order = torch.argsort(x[:, 0])
        cent = x[order[torch.linspace(0, len(x) - 1, self.K).long()]].clone()     # spread along the first feature
        for _ in range(25):
            lab = torch.cdist(x, cent).argmin(1)
            for k in range(self.K):
                if (lab == k).any():
                    cent[k] = x[lab == k].mean(0)
        cent = cent[torch.argsort(cent[:, 0])]
        out = {}
        for k in range(self.K):
            for f in range(min(self.features, x.shape[1])):
                out[f"mean_{k + 1}_{f + 1}"] = float(cent[k, f].clamp(-5.9, 5.9))
        return out

    def simulate(self, p, n, T, dt, gen, device="cpu", substeps=40):
        pt = {k: torch.tensor([float(v)], dtype=torch.float64) for k, v in p.items()}
        Q, mean, var = self.build(pt)
        load = self.extras(pt).get("load")
        P = torch.linalg.matrix_exp(Q[0] * dt)
        state = torch.multinomial(self.initial_probs(pt)[0].float(), n, replacement=True, generator=gen)
        out = torch.empty(n, T, self.features, dtype=torch.float64)
        for t in range(T):
            out[:, t] = mean[0, state] + var[0, state].sqrt() * torch.randn(n, self.features, generator=gen, dtype=torch.float64)
            if load is not None:
                out[:, t] += (load[0, state] @ torch.randn(n, self.rank, 1, generator=gen, dtype=torch.float64)).squeeze(-1)
            state = torch.multinomial(P[state].float().clamp_min(0), 1, generator=gen).squeeze(1)
        return out.to(device)


# ---------------------------------------------------------------------- your own SDE
_SAFE = {"exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt, "tanh": torch.tanh, "abs": torch.abs,
         "sigmoid": torch.sigmoid, "pi": math.pi, "minimum": torch.minimum, "maximum": torch.maximum}


class CustomSDE(Model):
    """Your own one-variable SDE, du = drift(u) dt + noise(u) dW, written as formulas in the
    experiment file (x is the readout; parameters by name). For example a bistable switch:
        drift: "a*(x - x**3/3) + b"    noise: "s"    parameters: {a: 0.1, b: 0.0, s: 0.2}
    Formulas use + - * / ** and exp, log, sqrt, tanh, abs, sigmoid, minimum, maximum."""
    name = "custom"
    text = "Your own SDE: drift and noise formulas in the readout x"

    def __init__(self, grid: int = 48, drift: str = "theta*(mu - x)", noise: str = "s", parameters: dict | None = None, **options):
        super().__init__(grid, **options)
        self.drift_expr, self.noise_expr = str(drift), str(noise)
        self.user = dict(parameters or {"theta": 0.05, "mu": 0.0, "s": 0.1})
        for expr in (self.drift_expr, self.noise_expr):
            compile(expr, "<formula>", "eval")  # syntax errors surface when the model is made

    def specs(self):
        out = {}
        for k, v in self.user.items():
            v = float(v)
            out[k] = Spec(v, -1e3, 1e3, log=False, unit="value", share="condition", text="your parameter")
        out["noise"] = Spec(0.1, 1e-3, 3.0, unit="noise", share="all", text="measurement sd on top of each observation's pixel-noise estimate")
        return out

    def _eval(self, expr: str, x: torch.Tensor, p: dict) -> torch.Tensor:
        env = {**_SAFE, **{k: p[k] for k in self.user}, "x": x}
        value = eval(expr, {"__builtins__": {}}, env)  # noqa: S307 - formulas come from the user's own experiment file
        return value * torch.ones_like(x) if not torch.is_tensor(value) else value.expand_as(x) if value.dim() < x.dim() else value

    def build(self, p):
        dev = p["noise"].device
        f = self.faces(dev)[None, :]
        q = {k: p[k][:, None] for k in self.user}
        drift = self._eval(self.drift_expr, f.expand(p["noise"].shape[0], -1), q)
        D = self._eval(self.noise_expr, f.expand(p["noise"].shape[0], -1), q) ** 2 / 2
        Q = fokker_planck(drift[:, None], D[:, None], self.h)[:, 0]
        P = Q.shape[0]
        mean = self.centres(dev)[None, :, None].expand(P, -1, 1)
        var = (p["noise"] ** 2 + self.h ** 2 / 12)[:, None, None].expand(P, self.G, 1)
        return Q, mean, var

    def simulate(self, p, n, T, dt, gen, device="cpu", substeps=40):
        q = {k: torch.tensor(float(p[k]), dtype=torch.float64) for k in self.user}
        u = torch.full((n,), (self.lo + self.hi) / 2, dtype=torch.float64)
        h = dt / substeps
        out = torch.empty(n, T, dtype=torch.float64)
        for step in range(-20 * substeps, T * substeps):
            if step >= 0 and step % substeps == 0:
                out[:, step // substeps] = u
            u = u + self._eval(self.drift_expr, u, q) * h + self._eval(self.noise_expr, u, q) * math.sqrt(h) * torch.randn(n, generator=gen, dtype=torch.float64)
            u = u.clamp(self.lo, self.hi)
        return out[:, :, None].to(device)


MODELS = {m.name: m for m in (ErkKtr, OrnsteinUhlenbeck, States, CustomSDE)}


def make_model(spec: dict, features: int = 1, feature_names: list[str] | None = None) -> Model:
    name = spec["model"]
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; models are {list(MODELS)}")
    opts = dict(spec.get("options") or {})
    if name == "states":
        return States(states=opts.get("states", 3), features=features, topology=opts.get("topology", "full"),
                      rank=opts.get("rank", 0), division=opts.get("division", False), feature_names=feature_names)
    if name == "erk_ktr":
        return ErkKtr(grid=opts.get("grid", 48), modes=opts.get("modes", 2))
    if name == "custom":
        return CustomSDE(grid=opts.get("grid", 48), drift=opts.get("drift", "theta*(mu - x)"), noise=opts.get("noise", "s"),
                         parameters=opts.get("parameters"))
    return OrnsteinUhlenbeck(grid=opts.get("grid", 48))
