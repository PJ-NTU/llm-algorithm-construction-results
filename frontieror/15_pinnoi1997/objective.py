"""Objective evaluator for the fixed-station assembly-line model."""


def objective(instance: dict, solution: dict) -> float:
    """Return the minimization objective: reported worst-case station idle time."""
    if not isinstance(instance, dict):
        raise ValueError("instance must be a dict")
    if not isinstance(solution, dict):
        raise ValueError("solution must be a dict")
    if "worst_case_idle_time" not in solution:
        raise ValueError("solution is missing 'worst_case_idle_time'")

    value = solution["worst_case_idle_time"]
    if type(value) is not int or value < 0:
        raise ValueError("worst_case_idle_time must be a nonnegative integer")

    # The original model is already a minimization problem, so no sign change is used.
    return float(value)
