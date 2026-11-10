from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Interval:
    estimate: float
    lo: float
    hi: float

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi


def wilson(k: int, n: int, alpha: float = 0.05) -> Interval:
    if n == 0:
        return Interval(math.nan, math.nan, math.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Interval(p, centre - half, centre + half)


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> Interval:
    a, b = wilson(k1, n1, alpha), wilson(k2, n2, alpha)
    d = a.estimate - b.estimate
    lo = d - math.sqrt((a.estimate - a.lo) ** 2 + (b.hi - b.estimate) ** 2)
    hi = d + math.sqrt((a.hi - a.estimate) ** 2 + (b.estimate - b.lo) ** 2)
    return Interval(d, lo, hi)


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cohens_h_se(n1: int, n2: int) -> float:
    return math.sqrt(1 / n1 + 1 / n2)


def z_crit(alpha: float = 0.05) -> float:
    return float(stats.norm.ppf(1 - alpha / 2))


def wald(estimate: float, se: float, alpha: float = 0.05, bounds: bool = False) -> Interval:
    half = z_crit(alpha) * se
    lo, hi = estimate - half, estimate + half
    if bounds:
        lo, hi = max(0.0, lo), min(1.0, hi)
    return Interval(estimate, lo, hi)


def cluster_prop_se(groups: list[list[float]]) -> float:
    groups = [g for g in groups if g]
    m = len(groups)
    n = sum(len(g) for g in groups)
    if m < 2:
        return math.nan
    p = sum(sum(g) for g in groups) / n
    resid = sum((sum(g) - p * len(g)) ** 2 for g in groups)
    return math.sqrt(m / (m - 1) * resid) / n


def cohens_h_cell_se(p: float, p_se: float, n_units: int) -> float:
    if 0 < p < 1 and math.isfinite(p_se) and p_se > 0:
        return p_se / math.sqrt(p * (1 - p))
    return 1 / math.sqrt(n_units) if n_units > 0 else math.nan


def smd(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> tuple[float, float]:
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    g = j * (m1 - m2) / sp if sp > 0 else math.nan
    se = math.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))
    return g, se


@dataclass(frozen=True)
class Equivalence:
    delta: float
    se: float
    margin: float
    p: float
    ci90: Interval

    def equivalent(self, alpha: float = 0.05) -> bool:
        return self.p < alpha


def tost(delta: float, se: float, margin: float) -> Equivalence:
    if not (se > 0 and math.isfinite(se)):
        return Equivalence(delta, se, margin, math.nan, Interval(delta, math.nan, math.nan))
    p_lower = 1 - stats.norm.cdf((delta + margin) / se)
    p_upper = stats.norm.cdf((delta - margin) / se)
    z90 = stats.norm.ppf(0.95)
    return Equivalence(
        delta,
        se,
        margin,
        max(p_lower, p_upper),
        Interval(delta, delta - z90 * se, delta + z90 * se),
    )


def diff_prop_se(k1: int, n1: int, k2: int, n2: int) -> float:
    p1, p2 = k1 / n1, k2 / n2
    return math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)


def holm(pvalues: list[float]) -> list[float]:
    idx = [i for i, p in enumerate(pvalues) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvalues[i])
    m = len(order)
    adjusted = list(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def icc_binary_two_level(groups: list[list[int]]) -> float:
    groups = [g for g in groups if len(g) >= 1]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    if k < 2 or n_total <= k:
        return math.nan
    grand = np.mean([x for g in groups for x in g])
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ss_within = sum(((np.asarray(g) - np.mean(g)) ** 2).sum() for g in groups)
    ms_b = ss_between / (k - 1)
    ms_w = ss_within / (n_total - k)
    n0 = (n_total - sum(len(g) ** 2 for g in groups) / n_total) / (k - 1)
    denom = ms_b + (n0 - 1) * ms_w
    return float((ms_b - ms_w) / denom) if denom > 0 else math.nan


def bootstrap_ci(
    values: list[list[int]], fn: object, iterations: int, seed: int, alpha: float = 0.05
) -> Interval:
    assert callable(fn)
    rng = np.random.default_rng(seed)
    est = float(fn(values))
    k = len(values)
    reps = []
    for _ in range(iterations):
        sample = [values[i] for i in rng.integers(0, k, k)]
        r = float(fn(sample))
        if math.isfinite(r):
            reps.append(r)
    if not reps:
        return Interval(est, math.nan, math.nan)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return Interval(est, float(lo), float(hi))


def min_detectable_diff(
    n_model: int, n_human: int, p: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * math.sqrt(p * (1 - p) * (1 / n_model + 1 / n_human)))


def chi2_homogeneity(counts: list[tuple[int, int]]) -> tuple[float, float]:
    table = np.array([[k, n - k] for k, n in counts])
    if table.shape[0] < 2 or (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
        return math.nan, math.nan
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p)


def tost_power(se: float, margin: float, true_delta: float = 0.0, alpha: float = 0.05) -> float:
    z = stats.norm.ppf(1 - alpha)
    upper = stats.norm.cdf((margin - true_delta) / se - z)
    lower = stats.norm.cdf((-margin - true_delta) / se + z)
    return float(max(0.0, upper - lower))


def tost_n_for_power(
    human_se: float, margin: float, alpha: float = 0.05, power: float = 0.8
) -> int | None:
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - (1 - power) / 2)
    room = (margin / z) ** 2 - human_se**2
    return math.ceil(2 / room) if room > 0 else None
