"""Feasibility checker for the multidimensional knapsack solution format."""

import math
from fractions import Fraction


_REL = Fraction(1, 10**9)


def _is_number(value):
    """Return whether value is a finite JSON-style number (but not bool)."""
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _as_exact(value):
    """Represent an accepted int/float exactly, without finite-float overflow."""
    return Fraction(value)


def _leq_relative(a, b):
    """Compare computed continuous quantities at 1e-9 relative tolerance."""
    scale = max(Fraction(1), abs(a), abs(b))
    return a - b <= _REL * scale


def check_solution(instance: dict, solution: dict) -> list:
    """Return C1/format violations; an empty list means the solution is feasible."""
    violations = []

    try:
        if not isinstance(instance, dict):
            return ["FORMAT | instance must be a dict"]
        if not isinstance(solution, dict):
            return ["FORMAT | solution must be a dict"]

        n = instance.get("n")
        m = instance.get("m")
        if type(n) is not int:
            return ["FORMAT | instance.n must be an int"]
        if type(m) is not int:
            return ["FORMAT | instance.m must be an int"]

        if "selection_status" not in solution:
            return ["FORMAT | missing required field: selection_status"]
        selection = solution["selection_status"]
        if not isinstance(selection, list):
            return ["FORMAT | selection_status must be a list"]
        if len(selection) != n:
            return [
                "FORMAT | selection_status must contain exactly "
                f"n={n} entries; got {len(selection)}"
            ]

        for j, value in enumerate(selection):
            if type(value) is not int or value not in (0, 1):
                violations.append(
                    f"FORMAT | selection_status[{j}] must be integer 0 or 1"
                )
        if violations:
            return violations

        weights = instance.get("weights")
        capacity = instance.get("capacity")
        if not isinstance(weights, list):
            return ["FORMAT | instance.weights must be a list"]
        if not isinstance(capacity, list):
            return ["FORMAT | instance.capacity must be a list"]
        if len(weights) < m:
            return [
                f"FORMAT | weights has {len(weights)} resource rows but m={m}"
            ]
        if len(capacity) < m:
            return [
                f"FORMAT | capacity has {len(capacity)} entries but m={m}"
            ]

        for r in range(m):
            row = weights[r]
            if not isinstance(row, list):
                violations.append(f"FORMAT | weights[{r}] must be a list")
                continue
            if len(row) < n:
                violations.append(
                    f"FORMAT | weights[{r}] has {len(row)} entries but n={n}"
                )
                continue
            if not _is_number(capacity[r]):
                violations.append(f"FORMAT | capacity[{r}] must be a finite number")
                continue

            bad_weight = None
            for j in range(n):
                if not _is_number(row[j]):
                    bad_weight = j
                    break
            if bad_weight is not None:
                violations.append(
                    f"FORMAT | weights[{r}][{bad_weight}] must be a finite number"
                )
                continue

            # C1: sum_j weights[r][j] * x[j] <= capacity[r].
            # Because x is already checked as binary, summing selected weights is
            # exactly the formula above and ignores no nonzero term.
            selected_weights = [row[j] for j in range(n) if selection[j] == 1]
            usage = sum((_as_exact(value) for value in selected_weights), Fraction(0))
            limit = _as_exact(capacity[r])

            # Empty sums, one untouched instance value, and integer-only sums are
            # exactly decidable. Multiple terms containing a float are a continuous
            # accumulated quantity and receive the required linear relative tolerance.
            accumulated_float = len(selected_weights) > 1 and any(
                type(value) is float for value in selected_weights
            )
            within_capacity = (
                _leq_relative(usage, limit)
                if accumulated_float
                else usage <= limit
            )
            if not within_capacity:
                violations.append(
                    f"C1 | resource {r} usage exceeds capacity"
                )

        return violations
    except Exception as exc:
        return [f"FORMAT | checker could not read the supplied data: {type(exc).__name__}"]

