"""The only objective is the minimum number of vehicles; smaller is better, with no secondary objective.

Route feasibility is first verified by checker.check, then the values of this function are compared. This function does not repeat
all feasibility checks; it returns positive infinity on a structural mismatch or a non-feasible status. A valid
empty-task feasible solution returns 0.0. instance is kept as a contract interface parameter and is not used in the vehicle count.
"""

from math import inf


def objective(instance: dict, solution: dict) -> float:
    """Return the number of routes (one vehicle per route), without negation."""
    if not isinstance(solution, dict) or solution.get("status") != "feasible":
        return inf
    routes = solution.get("routes")
    if not isinstance(routes, list):
        return inf
    return float(len(routes))
