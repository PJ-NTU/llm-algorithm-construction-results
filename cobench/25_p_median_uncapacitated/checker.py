# -*- coding: utf-8 -*-
"""Uncapacitated p-median location: feasibility checker (Python standard library only).

check(instance, solution) returns a list of violation description strings; an empty list means the solution is feasible.

Input contract (input_schema.json)
    n    : int, number of vertices, n >= 1, vertices numbered 0..n-1
    p    : int, number of medians that must be selected exactly, 1 <= p <= n
    dist : list[list[float | None]], n×n; dist[i][j] is the shortest-path distance from vertex i (origin) to vertex j (destination);
           None (JSON null) means unreachable (+inf); symmetry is not required, always used in the dist[i][j] direction
Solution contract (solution_form.json)
    {"medians": list[int]}: exactly p distinct vertex indices in 0..n-1, in any order

Correspondence with the mathematical model (MODEL.md): y_j = number of occurrences of vertex j in medians
    C1  exactly p medians are selected                 sum_j y_j = p
    C2  the selected medians are distinct              y_j ∈ {0, 1}
    C3  every vertex has finite distance to its nearest selected median   min_{j: y_j = 1} dist[i][j] < +inf, for all i

Violation message prefixes:
    "Instance:"        the instance itself violates the input contract
    "Structure:"       the solution is missing fields, has the wrong shape, or contains elements that are not integers in 0..n-1
    "C1:" "C2:" "C3:"  the corresponding constraints in the model
"""

import math
import operator
from collections import Counter

_MAX_SHOW = 10


def _as_int(x):
    """Integers (including integer types implementing __index__) return the corresponding int; bool, floats and other types return None."""
    if isinstance(x, bool):
        return None
    try:
        return operator.index(x)
    except TypeError:
        return None


def _preview(items, limit=_MAX_SHOW):
    """Format entries as "[a, b, ...]", truncating beyond limit items."""
    items = list(items)
    text = ", ".join(str(v) for v in items[:limit])
    if len(items) > limit:
        text += ", ..."
    return "[" + text + "]"


def _short_repr(x, limit=40):
    s = repr(x)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _distance_is_finite(v):
    """Decide whether a dist element is a reachable finite distance.

    None (null) -> False; finite number -> True; inf / nan -> False;
    non-numeric (including bool and strings) -> None, meaning the element violates the input contract.
    """
    if v is None:
        return False
    if isinstance(v, bool):
        return None
    try:
        return math.isfinite(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_instance(instance):
    """Check the instance against the input contract, returning ((n, p, dist), errors)."""
    if not isinstance(instance, dict):
        return None, ["Instance: instance must be a dict, got %s" % type(instance).__name__]
    missing = [k for k in ("n", "p", "dist") if k not in instance]
    if missing:
        return None, ["Instance: missing field(s) %s" % ", ".join(repr(k) for k in missing)]

    errors = []
    n = _as_int(instance["n"])
    p = _as_int(instance["p"])
    dist = instance["dist"]
    if n is None:
        errors.append("Instance: n must be an integer, got %s" % _short_repr(instance["n"]))
    elif n < 1:
        errors.append("Instance: n must be >= 1, got %d" % n)
    if p is None:
        errors.append("Instance: p must be an integer, got %s" % _short_repr(instance["p"]))
    if not isinstance(dist, (list, tuple)):
        errors.append("Instance: dist must be an n×n two-dimensional list, got %s" % type(dist).__name__)
    if errors:
        return None, errors

    if not 1 <= p <= n:
        errors.append("Instance: p must satisfy 1 <= p <= n, got p=%d, n=%d" % (p, n))
    if len(dist) != n:
        errors.append("Instance: dist should have n=%d rows, got %d rows" % (n, len(dist)))
    else:
        bad_rows = [i for i, row in enumerate(dist)
                    if not isinstance(row, (list, tuple)) or len(row) != n]
        if bad_rows:
            errors.append("Instance: the following rows of dist are not lists of length n=%d, %d row(s) in total: %s"
                          % (n, len(bad_rows), _preview(bad_rows)))
    if errors:
        return None, errors
    return (n, p, dist), []


def check(instance: dict, solution: dict) -> list:
    """Return a list of violation strings; empty list = feasible."""
    parsed, errors = _parse_instance(instance)
    if errors:
        return errors
    n, p, dist = parsed
    violations = []

    # ---- Structural integrity: are all fields present, and are shape and element types correct ----
    if not isinstance(solution, dict):
        return ["Structure: solution must be a dict, got %s" % type(solution).__name__]
    if "medians" not in solution:
        return ["Structure: solution is missing field 'medians'"]
    medians = solution["medians"]
    if not isinstance(medians, (list, tuple)):
        return ["Structure: 'medians' must be a list of integers, got %s" % type(medians).__name__]

    valid = []          # vertex indices that are integers in 0..n-1, duplicates kept (for C2 counting)
    non_int = []        # non-integer elements "(position, value)"
    out_of_range = []   # out-of-range integers "(position, value)"
    for pos, x in enumerate(medians):
        j = _as_int(x)
        if j is None:
            non_int.append("(%d, %s)" % (pos, _short_repr(x)))
        elif 0 <= j < n:
            valid.append(j)
        else:
            out_of_range.append("(%d, %d)" % (pos, j))
    if non_int:
        violations.append("Structure: elements of 'medians' must be integer vertex indices; the following elements (position, value) are not, %d in total: %s"
                          % (len(non_int), _preview(non_int)))
    if out_of_range:
        violations.append("Structure: vertex indices in 'medians' must be within 0..%d; the following elements (position, value) are out of range, %d in total: %s"
                          % (n - 1, len(out_of_range), _preview(out_of_range)))

    # C1: exactly p medians are selected -- sum_j y_j = p; y_j is the number of occurrences of vertex j in medians, so the list length must equal p
    if len(medians) != p:
        violations.append("C1: exactly p=%d medians must be selected, got %d" % (p, len(medians)))

    # C2: the selected medians are distinct -- y_j ∈ {0, 1}, each vertex appears at most once in medians
    counts = Counter(valid)
    duplicated = sorted(j for j, c in counts.items() if c > 1)
    if duplicated:
        violations.append("C2: the selected medians must be distinct; the following vertices are selected more than once (vertex: count), %d in total: %s"
                          % (len(duplicated), _preview("%d: %d" % (j, counts[j]) for j in duplicated)))

    # C3: every vertex must have finite distance to its nearest selected median -- for each vertex i there is a selected median j with dist[i][j] non-null and finite
    selected = sorted(counts)   # set of selected medians after deduplication (only integer indices in 0..n-1)
    uncovered = []
    bad_entries = []
    for i in range(n):
        row = dist[i]
        covered = False
        for j in selected:
            finite = _distance_is_finite(row[j])
            if finite:
                covered = True
                break
            if finite is None:
                bad_entries.append("(%d, %d)" % (i, j))
        if not covered:
            uncovered.append(i)
    if bad_entries:
        violations.append("Instance: the following elements of dist (i, j) are neither numbers nor null, %d in total: %s"
                          % (len(bad_entries), _preview(bad_entries)))
    if uncovered:
        violations.append("C3: the following vertices have no finite distance to any selected median (null means unreachable) and are uncovered, %d in total: %s"
                          % (len(uncovered), _preview(uncovered)))

    return violations
