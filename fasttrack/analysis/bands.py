"""What a line and its band show: the centre (median or mean) and the spread or uncertainty around it.

No single choice is right for every figure, so each figure picks one (`band` in experiment.FIGURE_DEFAULT):

| band     | centre | band                          | answers                                            |
|----------|--------|-------------------------------|----------------------------------------------------|
| iqr      | median | 25th to 75th percentile       | where the middle half of the values lie            |
| range90  | median | 10th to 90th percentile       | the spread, including most of the tails            |
| sd       | mean   | ± 1 standard deviation        | how much values vary (spread, not uncertainty)     |
| sem      | mean   | ± 1 standard error of the mean| how well the mean is known (about a 68% interval)  |
| ci95     | mean   | 95% confidence interval (t)   | how well the mean is known, at 95%                 |
| none     | median | no band                       |                                                    |
| auto     | iqr over pooled cells, sem over replicates (FastTrack's earlier behaviour)          |

Over replicates (wells or positions) the values are each replicate's summary, so SEM and CI describe how well the
condition's mean is known. Over pooled cells they are almost always tiny (cells of one well are not independent),
so a note says so.
"""
from __future__ import annotations

import warnings

import numpy as np

BANDS = {
    "auto": "automatic (quartiles of cells, s.e.m. of replicates)",
    "iqr": "median and quartiles (IQR)",
    "range90": "median and 10th-90th percentile",
    "sd": "mean ± SD",
    "sem": "mean ± s.e.m.",
    "ci95": "mean and 95% confidence interval",
    "none": "line only, no band",
}

_QUANTILES = {"iqr": (0.25, 0.75), "range90": (0.10, 0.90), "none": (None, None)}


def resolve(band: str, unit: str) -> str:
    band = band or "auto"
    if band == "auto":
        return "iqr" if unit == "cell" else "sem"
    if band not in BANDS:
        raise ValueError(f"unknown error band {band!r}; choose one of {', '.join(BANDS)}")
    return band


def uses_median(band: str) -> bool:
    return band in _QUANTILES


def quantiles(band: str) -> tuple[float, float, float]:
    lo, hi = _QUANTILES[band]
    return (lo if lo is not None else 0.5, 0.5, hi if hi is not None else 0.5)


def t_factor(n: np.ndarray) -> np.ndarray:
    """Two-sided 95% Student t quantile for n values (n - 1 degrees of freedom); NaN below 2."""
    from scipy.stats import t
    n = np.asarray(n, dtype=float)
    out = np.full(n.shape, np.nan)
    ok = n >= 2
    out[ok] = t.ppf(0.975, n[ok] - 1)
    return out


def from_moments(n: np.ndarray, mean: np.ndarray, sd: np.ndarray, band: str) -> tuple[np.ndarray, np.ndarray]:
    """lo, hi around a mean from the count and the (n - 1) standard deviation, per column."""
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        if band == "sd":
            half = sd
        elif band == "sem":
            half = sd / np.sqrt(n)
        elif band == "ci95":
            half = t_factor(n) * sd / np.sqrt(n)
        else:
            raise ValueError(band)
    half = np.where(n >= 2, half, np.nan)
    return mean - half, mean + half


def across(stack: np.ndarray, band: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Centre, lo, hi of each column of `stack` (rows are replicates, NaN where one has no value)."""
    stack = np.atleast_2d(np.asarray(stack, dtype=float))
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if uses_median(band):
            q_lo, q_mid, q_hi = quantiles(band)
            centre = np.nanquantile(stack, q_mid, axis=0)
            if band == "none":
                return centre, None, None
            return centre, np.nanquantile(stack, q_lo, axis=0), np.nanquantile(stack, q_hi, axis=0)
        n = np.sum(np.isfinite(stack), axis=0)
        mean = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0, ddof=1)
    lo, hi = from_moments(n, mean, sd, band)
    return mean, lo, hi


def proportion(k: np.ndarray, n: np.ndarray, band: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """A band for a percentage of pooled cells: the 95% Wilson interval for ci95, else none (a spread of 0/1
    values says nothing more than the percentage itself)."""
    if band != "ci95":
        return None, None
    k, n = np.asarray(k, float), np.asarray(n, float)
    z = 1.959964
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def describe(band: str, over: str) -> str:
    """'mean ± SD of well medians', 'median and quartiles of cells', ..."""
    what = {"iqr": "median and quartiles", "range90": "median and 10th-90th percentile", "sd": "mean ± SD",
            "sem": "mean ± s.e.m.", "ci95": "mean and 95% CI (t)", "none": "median"}[band]
    if band == "none" and over != "cells":
        what = "median"
    return f"{what} of {over}"


def table_columns(band: str) -> tuple[str, str, str]:
    """CSV column suffixes for the centre and the two band edges."""
    if uses_median(band):
        lo, _, hi = quantiles(band)
        return "median", f"q{round(100 * lo)}", f"q{round(100 * hi)}"
    return "mean", f"{band} low", f"{band} high"
