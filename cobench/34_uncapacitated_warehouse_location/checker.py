"""Feasibility check for the uncapacitated facility location problem (UFLP).

check(instance, solution) returns a list of violation description strings; an empty list means the solution is feasible.
Check order:
  1. Whether the instance meets the input contract (feasibility cannot be judged for an invalid instance, so the reason is returned directly);
  2. Structural integrity of the solution: whether keys y and x are present, whether shapes are M and M×N,
     and whether elements are integers 0 or 1 (split assignment not allowed);
  3. Semantic constraints C1 and C2, matching the mathematical model one by one.
Uses only the Python standard library.
"""

import math

_MAX_SHOWN = 10  # maximum number of indices listed in a single violation message


def _is_int(v):
    """Integer (excluding booleans)."""
    return isinstance(v, int) and not isinstance(v, bool)


def _is_finite_nonneg_real(v):
    """Finite non-negative real: int or float, excluding booleans, infinity and NaN."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return v >= 0
    if isinstance(v, float):
        return math.isfinite(v) and v >= 0
    return False


def _is_seq(v):
    return isinstance(v, (list, tuple))


def _non_binary(seq):
    """Return the list of indices whose value is not the integer 0 or 1."""
    return [k for k, v in enumerate(seq) if not (_is_int(v) and v in (0, 1))]


def _fmt(indices):
    shown = ", ".join(str(k) for k in indices[:_MAX_SHOWN])
    if len(indices) > _MAX_SHOWN:
        shown += ", ..."
    return "[" + shown + "] (" + str(len(indices)) + " total)"


def _short(v):
    text = repr(v)
    return text if len(text) <= 60 else text[:57] + "..."


def _instance_errors(instance):
    """Check the instance against the input contract: n and m_facilities are positive integers, costs are finite non-negative reals, shapes are M and M×N."""
    if not isinstance(instance, dict):
        return ["Invalid instance: instance must be a dict"]
    errors = []
    n = instance.get("n")
    m = instance.get("m_facilities")
    if not (_is_int(n) and n >= 1):
        errors.append("Invalid instance: n must be an integer ≥ 1, got " + _short(n))
    if not (_is_int(m) and m >= 1):
        errors.append("Invalid instance: m_facilities must be an integer ≥ 1, got " + _short(m))
    if errors:
        return errors
    fixed_cost = instance.get("fixed_cost")
    if not _is_seq(fixed_cost) or len(fixed_cost) != m:
        errors.append("Invalid instance: fixed_cost must be a list of length m_facilities=%d" % m)
    else:
        bad = [i for i, v in enumerate(fixed_cost) if not _is_finite_nonneg_real(v)]
        if bad:
            errors.append("Invalid instance: fixed_cost is not a finite non-negative real at the following facility indices " + _fmt(bad))
    alloc_cost = instance.get("alloc_cost")
    if not _is_seq(alloc_cost) or len(alloc_cost) != m:
        errors.append("Invalid instance: alloc_cost must be a 2-D list with m_facilities=%d rows (facility × customer)" % m)
    else:
        for i, row in enumerate(alloc_cost):
            if not _is_seq(row) or len(row) != n:
                errors.append("Invalid instance: alloc_cost[%d] must be a list of length n=%d" % (i, n))
                continue
            bad = [j for j, v in enumerate(row) if not _is_finite_nonneg_real(v)]
            if bad:
                errors.append("Invalid instance: alloc_cost[%d] is not a finite non-negative real at the following customer indices %s" % (i, _fmt(bad)))
    return errors


def check(instance: dict, solution: dict) -> list:
    errors = _instance_errors(instance)
    if errors:
        return errors
    M = instance["m_facilities"]
    N = instance["n"]
    violations = []

    # Structural integrity: the solution must be a dict containing keys y and x
    if not isinstance(solution, dict):
        return ["Structure error: solution must be a dict containing keys y and x"]

    # Structural integrity: y is a list of length M with integer elements 0 or 1
    y = solution.get("y")
    y_ok = False
    if "y" not in solution:
        violations.append("Structure error: key y is missing from the solution")
    elif not _is_seq(y):
        violations.append("Structure error: y must be a list, got type " + type(y).__name__)
    elif len(y) != M:
        violations.append("Structure error: length of y should be M=%d, got %d" % (M, len(y)))
    else:
        bad = _non_binary(y)
        if bad:
            violations.append("Value error: elements of y must be integers 0 or 1; invalid values at the following facility indices " + _fmt(bad))
        else:
            y_ok = True

    # Structural integrity: x is a 2-D list with M rows and N columns, integer elements 0 or 1 (split assignment not allowed)
    x = solution.get("x")
    x_ok = False
    if "x" not in solution:
        violations.append("Structure error: key x is missing from the solution")
    elif not _is_seq(x):
        violations.append("Structure error: x must be a 2-D list, got type " + type(x).__name__)
    elif len(x) != M:
        violations.append("Structure error: number of rows of x should be M=%d, got %d" % (M, len(x)))
    else:
        x_ok = True
        for i, row in enumerate(x):
            if not _is_seq(row) or len(row) != N:
                violations.append("Structure error: x[%d] must be a list of length N=%d" % (i, N))
                x_ok = False
                continue
            bad = _non_binary(row)
            if bad:
                violations.append(
                    "Value error: elements of x[%d] must be integers 0 or 1 (split assignment not allowed); invalid values at the following customer indices %s"
                    % (i, _fmt(bad))
                )
                x_ok = False

    # C1 each customer is assigned exactly once: for each customer j, Σ_i x[i][j] = 1
    if x_ok:
        for j, col in enumerate(zip(*x)):
            total = sum(col)
            if total != 1:
                if total == 0:
                    violations.append(
                        "C1 violation: customer %d is not served by any facility (Σ_i x[i][%d] = 0, should be 1)" % (j, j)
                    )
                else:
                    servers = [i for i, v in enumerate(col) if v == 1]
                    violations.append(
                        "C1 violation: customer %d is served by multiple facilities %s (Σ_i x[i][%d] = %d, should be 1)"
                        % (j, _fmt(servers), j, total)
                    )

    # C2 only open facilities may serve: for all facilities i and customers j, x[i][j] ≤ y[i]
    if x_ok and y_ok:
        for i in range(M):
            served = [j for j, v in enumerate(x[i]) if v > y[i]]
            if served:
                violations.append(
                    "C2 violation: facility %d is not open (y[%d] = %d) but serves customers %s (requires x[%d][j] ≤ y[%d])"
                    % (i, i, y[i], _fmt(served), i, i)
                )

    return violations
