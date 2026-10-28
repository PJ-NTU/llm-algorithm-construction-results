"""Minimization-oriented objective: negative number of packed rectangles.

Feasibility is checked separately by checker.check. This function validates only
N and alpha, the data needed to evaluate the objective, and raises ValueError
when those fields are malformed. No further sign change is needed downstream.
"""


def objective(instance: dict, solution: dict) -> float:
    """Return -sum(alpha); smaller is better. The empty solution scores 0.0."""
    if not isinstance(instance, dict):
        raise ValueError("instance must be a dictionary.")
    n = instance.get("num_rectangles")
    if type(n) is not int or n < 0:
        raise ValueError("num_rectangles must be a nonnegative integer, not bool.")
    if not isinstance(solution, dict):
        raise ValueError("solution must be a dictionary.")
    alpha = solution.get("alpha")
    if not isinstance(alpha, list) or len(alpha) != n:
        raise ValueError("alpha must be a list of length num_rectangles.")
    if any(type(value) is not int or value not in (0, 1) for value in alpha):
        raise ValueError("alpha entries must be integer 0 or 1, not bool.")
    return float(-sum(alpha))
