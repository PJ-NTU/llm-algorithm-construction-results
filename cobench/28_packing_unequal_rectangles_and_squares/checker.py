"""Check the circular rectangle-packing contract using only the standard library.

No feasibility tolerance is added. Fraction arithmetic compares integers and
supplied finite Python floats exactly, including corner and contact boundaries.
Invalid instance data or incomplete solutions also produce violation strings.
"""

from fractions import Fraction
from itertools import combinations
from math import isfinite


def _finite_number(value):
    """Accept JSON-style numbers, excluding booleans, NaN and infinities."""
    return type(value) is int or (type(value) is float and isfinite(value))


def check(instance: dict, solution: dict) -> list:
    """Return violation strings; an empty list means the solution is feasible."""
    violations = []
    if not isinstance(instance, dict):
        violations.append("instance: expected a dictionary.")
    if not isinstance(solution, dict):
        violations.append("solution: expected a dictionary.")
    if violations:
        return violations

    required_inputs = (
        "num_rectangles", "widths", "heights", "container_radius", "big_m"
    )
    required_decisions = ("alpha", "x", "y", "r")
    missing_inputs = [key for key in required_inputs if key not in instance]
    missing_decisions = [key for key in required_decisions if key not in solution]
    extra_decisions = [key for key in solution if key not in required_decisions]
    if missing_inputs:
        violations.append(f"instance: missing fields {missing_inputs!r}.")
    if missing_decisions:
        violations.append(f"solution: missing fields {missing_decisions!r}.")
    if extra_decisions:
        violations.append(f"solution: unexpected fields {extra_decisions!r}.")
    if violations:
        return violations

    n = instance["num_rectangles"]
    if type(n) is not int or n < 0:
        return ["instance.num_rectangles: expected a nonnegative integer, not bool."]

    for key in ("widths", "heights"):
        values = instance[key]
        if not isinstance(values, list) or len(values) != n:
            violations.append(f"instance.{key}: expected a list of length {n}.")
            continue
        for i, value in enumerate(values):
            if not _finite_number(value) or value <= 0:
                violations.append(
                    f"instance.{key}[{i}]: expected a strictly positive finite number."
                )

    radius = instance["container_radius"]
    if not _finite_number(radius) or radius <= 0:
        violations.append(
            "instance.container_radius: expected a strictly positive finite number."
        )
    if not _finite_number(instance["big_m"]):
        violations.append("instance.big_m: expected any finite number, not bool.")

    for key in required_decisions:
        values = solution[key]
        if not isinstance(values, list) or len(values) != n:
            violations.append(f"solution.{key}: expected a list of length {n}.")
            continue
        for i, value in enumerate(values):
            if key in ("alpha", "r"):
                if type(value) is not int or value not in (0, 1):
                    violations.append(
                        f"solution.{key}[{i}]: expected integer 0 or 1, not bool."
                    )
            elif not _finite_number(value):
                violations.append(
                    f"solution.{key}[{i}]: expected a finite number, not bool."
                )
    if violations:
        return violations

    alpha = solution["alpha"]
    rotation = solution["r"]
    x = [Fraction(value) for value in solution["x"]]
    y = [Fraction(value) for value in solution["y"]]
    widths = instance["widths"]
    heights = instance["heights"]
    half_width = [
        Fraction(heights[i] if rotation[i] else widths[i]) / 2
        for i in range(n)
    ]
    half_height = [
        Fraction(widths[i] if rotation[i] else heights[i]) / 2
        for i in range(n)
    ]
    radius_squared = Fraction(radius) ** 2
    selected = [i for i in range(n) if alpha[i] == 1]

    # C1
    # All four corners of every selected rectangle must be in the closed circle.
    # For an unselected rectangle, the original alpha-multiplied left side is zero.
    for i in selected:
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            corner_x = x[i] + sx * half_width[i]
            corner_y = y[i] + sy * half_height[i]
            if corner_x ** 2 + corner_y ** 2 > radius_squared:
                violations.append(
                    f"C1: rectangle {i}, corner signs ({sx:+d}, {sy:+d}), "
                    "lies outside the circular container."
                )

    # C2
    # The first OR branch holds whenever the pair is not both selected.
    # For two selected rectangles, M * (2 - alpha[i] - alpha[j]) is exactly zero.
    # Thus the full given disjunction is checked by separation on at least one axis.
    for i, j in combinations(selected, 2):
        separated_x = abs(x[i] - x[j]) >= half_width[i] + half_width[j]
        separated_y = abs(y[i] - y[j]) >= half_height[i] + half_height[j]
        if not (separated_x or separated_y):
            violations.append(
                f"C2: selected rectangles {i} and {j} overlap in their interiors."
            )

    # C3
    # Both dummy-position equations require exact zero coordinates when alpha=0.
    # The equations impose no additional restriction when alpha=1.
    for i in range(n):
        if alpha[i] == 0 and (x[i] != 0 or y[i] != 0):
            violations.append(
                f"C3: unselected rectangle {i} must have x=0 and y=0."
            )

    return violations
