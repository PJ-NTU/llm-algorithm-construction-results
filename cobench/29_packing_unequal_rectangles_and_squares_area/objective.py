"""Minimization score for the maximum-area packing problem.

The caller must use checker.check to establish input and solution validity.
This module negates the packed area exactly once; downstream must not negate
it again. Dimensions and products are accumulated exactly before conversion
to the required float return type. If the magnitude exceeds the float range,
the returned score is negative infinity.
"""

from fractions import Fraction


def objective(instance: dict, solution: dict) -> float:
    """Return minus the total selected area; smaller scores are better."""
    area = Fraction(0)
    for i in range(instance["n"]):
        if solution["alpha"][i] == 1:
            length, width = instance["items"][i]
            area += Fraction(length) * Fraction(width)
    try:
        return -float(area)
    except OverflowError:
        return float("-inf")
