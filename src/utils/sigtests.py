#!/usr/bin/env python3
from __future__ import annotations
from typing import Iterable, Tuple


def mcnemar(b01: int, b10: int) -> Tuple[float, float]:
    """
    McNemar's test for paired nominal data (with continuity correction).
    Returns (chi2, p_approx) using normal approximation.
    """
    import math

    n = b01 + b10
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    # Approximate p via chi2 with 1 df
    # p ~ exp(-chi2/2) * (1 + chi2/2)
    p = math.exp(-chi2 / 2) * (1 + chi2 / 2)
    return chi2, min(1.0, max(0.0, p))


def bootstrap_diff_ci(a: Iterable[float], b: Iterable[float], reps: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    import random

    a = list(a); b = list(b)
    if len(a) != len(b) or len(a) == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    base = sum(a_i - b_i for a_i, b_i in zip(a,b)) / n
    for _ in range(reps):
        idxs = [rng.randrange(n) for __ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idxs) / n)
    diffs.sort()
    return base, diffs[int(0.025*reps)], diffs[int(0.975*reps)]

