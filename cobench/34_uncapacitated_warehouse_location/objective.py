"""Objective function for the uncapacitated facility location problem (UFLP).

Total cost = Σ_i fixed_cost[i]·y[i] + Σ_i Σ_j alloc_cost[i][j]·x[i][j].
The problem is itself a minimisation, so the total cost is returned directly (not negated); smaller is better.
Feasibility is judged by checker.check; here the solution shape is only required to match the instance, otherwise ValueError is raised.
Uses only the Python standard library.
"""

import math


def objective(instance: dict, solution: dict) -> float:
    M = instance["m_facilities"]
    N = instance["n"]
    fixed_cost = instance["fixed_cost"]
    alloc_cost = instance["alloc_cost"]
    if not isinstance(solution, dict) or "y" not in solution or "x" not in solution:
        raise ValueError("solution must be a dict containing keys y and x")
    y = solution["y"]
    x = solution["x"]
    if not isinstance(y, (list, tuple)) or len(y) != M:
        raise ValueError("y must be a list of length m_facilities=%d" % M)
    if not isinstance(x, (list, tuple)) or len(x) != M:
        raise ValueError("x must be a 2-D list with m_facilities=%d rows" % M)
    for i, row in enumerate(x):
        if not isinstance(row, (list, tuple)) or len(row) != N:
            raise ValueError("x[%d] must be a list of length n=%d" % (i, N))

    def terms():
        # Fixed opening cost: Σ_i fixed_cost[i]·y[i]
        for i in range(M):
            if y[i]:
                yield fixed_cost[i] * y[i]
        # Assignment cost: Σ_i Σ_j alloc_cost[i][j]·x[i][j]
        for i in range(M):
            row_c = alloc_cost[i]
            for j, v in enumerate(x[i]):
                if v:
                    yield row_c[j] * v

    return float(math.fsum(terms()))
