"""Evaluation statistics (course BC notes, Sec. 5.2).

Wilson score interval for a success rate; Fisher's exact test for two
independent success counts; McNemar's exact test for paired outcomes (the
same test scenes run by two methods); Holm-Bonferroni for a family of tests.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def fisher(k1, n1, k2, n2, alternative="two-sided"):
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative=alternative).pvalue)


def mcnemar_exact(a: np.ndarray, b: np.ndarray):
    """Paired binary outcomes a, b (same scenes). Two-sided exact binomial test
    on the discordant pairs. Returns (p, n_a_only, n_b_only)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0, n10, n01
    return float(stats.binomtest(n10, n01 + n10, 0.5).pvalue), n10, n01


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
