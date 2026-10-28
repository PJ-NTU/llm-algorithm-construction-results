"""Capacitated p-median location problem: objective function (smaller is better).

Objective: minimise the total assignment distance Σ_i distance_matrix[i][assignments[i]].
The problem itself is a minimisation, so the total distance is returned directly, without negation.
Returns float('inf') when the objective cannot be computed (status not feasible, missing structure or invalid indices).
"""


def objective(instance: dict, solution: dict) -> float:
    try:
        n = instance["n"]
        dist = instance["distance_matrix"]
        if solution.get("status") != "feasible":
            return float("inf")
        assignments = solution["assignments"]
        if not isinstance(assignments, list) or len(assignments) != n:
            return float("inf")
        total = 0
        for i, j in enumerate(assignments):
            if not isinstance(j, int) or isinstance(j, bool) or not (0 <= j < n):
                return float("inf")
            total += dist[i][j]
        return float(total)
    except (KeyError, TypeError, IndexError, AttributeError):
        return float("inf")
