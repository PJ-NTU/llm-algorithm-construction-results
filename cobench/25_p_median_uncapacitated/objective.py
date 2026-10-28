# -*- coding: utf-8 -*-
"""Uncapacitated p-median location: objective function (smaller is better, Python standard library only).

objective(instance, solution) = sum_{i=0}^{n-1} min_{j ∈ medians} dist[i][j]

The problem itself minimises "the sum of distances from all vertices to their nearest selected median", so that total distance is returned directly, not negated;
downstream always uses it as "smaller is better", with no further negation.

Conventions:
- dist[i][j] is the distance from vertex i (origin) to vertex j (destination), always read in this direction; None (JSON null) means unreachable (+inf).
- medians is treated as a set (a repeated vertex counts once); whether there are exactly p and whether any repeat is decided by checker.check.
- If some vertex cannot reach any selected median (violating C3), or the instance/solution cannot be parsed per the contract
  (missing fields, medians not a list, elements not integers in 0..n-1, wrong dist shape), return float('inf'), i.e. the worst value.
"""

import math
import operator

_INF = float("inf")


def _as_int(x):
    """Integers (including integer types implementing __index__) return the corresponding int; bool, floats and other types return None."""
    if isinstance(x, bool):
        return None
    try:
        return operator.index(x)
    except TypeError:
        return None


def _finite_value(v):
    """Return the float value of a dist element when it is a finite number; return None for None, inf, nan or non-numeric values."""
    if v is None or isinstance(v, bool):
        return None
    try:
        if math.isfinite(v):
            return float(v)
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def objective(instance: dict, solution: dict) -> float:
    """Return the sum of distances from all vertices to their nearest selected median (smaller is better)."""
    if not isinstance(instance, dict) or not isinstance(solution, dict):
        return _INF
    n = _as_int(instance.get("n"))
    dist = instance.get("dist")
    if n is None or not isinstance(dist, (list, tuple)) or len(dist) != n:
        return _INF
    medians = solution.get("medians")
    if not isinstance(medians, (list, tuple)):
        return _INF

    selected = set()
    for x in medians:
        j = _as_int(x)
        if j is None or not 0 <= j < n:
            return _INF
        selected.add(j)
    selected = sorted(selected)

    nearest = []
    for i in range(n):
        row = dist[i]
        if not isinstance(row, (list, tuple)) or len(row) != n:
            return _INF
        best = _INF
        for j in selected:
            d = _finite_value(row[j])
            if d is not None and d < best:
                best = d
        if best == _INF:
            return _INF
        nearest.append(best)
    try:
        return float(math.fsum(nearest))
    except (OverflowError, ValueError):
        return _INF
