#!/usr/bin/env python3
from __future__ import annotations
import random
from typing import Iterable, Tuple, List


def mean_ci95(xs: Iterable[float], reps: int = 10000, rng_seed: int = 0) -> Tuple[float, float, float]:
    xs = list(xs)
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mu = sum(xs) / n
    rng = random.Random(rng_seed)
    boots: List[float] = []
    for _ in range(reps):
        samp = [xs[rng.randrange(n)] for __ in range(n)]
        boots.append(sum(samp) / n)
    boots.sort()
    lo = boots[int(0.025 * reps)]
    hi = boots[int(0.975 * reps)]
    return mu, lo, hi

