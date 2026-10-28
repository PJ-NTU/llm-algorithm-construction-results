"""Set Partitioning objective: minimise the total cost of the selected schemes (lower is better)."""
import math


def objective(instance: dict, solution: dict) -> float:
    """Returns sum_j costs[j] * x[j], recomputed from the instance data using x; total_cost in the solution is not trusted.

    The problem is a minimisation, so the value is not negated. Returns +inf if the solution reports no feasible solution (feasible=false) or x is missing/malformed.
    """
    try:
        if not isinstance(solution, dict) or solution.get("feasible") is not True:
            return float("inf")
        x = solution.get("x")
        costs = instance["costs"]
        n = instance["n"]
        if not isinstance(x, list) or len(x) != n:
            return float("inf")
        total = []
        for j, v in enumerate(x):
            if isinstance(v, bool):
                v = int(v)
            if not isinstance(v, (int, float)) or v not in (0, 1):
                return float("inf")
            if v == 1:
                total.append(float(costs[j]))
        return float(math.fsum(total))
    except Exception:
        return float("inf")
