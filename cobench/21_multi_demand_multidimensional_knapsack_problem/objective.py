"""Objective function for the multi-demand multidimensional knapsack problem (MDMKP).

The original problem maximises sum_j cost_vector[j]*x[j]. Here it is unified to "smaller is better" by returning its negation.
Downstream code need not negate it again.
Returns +inf when x is null (no feasible solution declared) or the structure is invalid.
"""


def objective(instance: dict, solution: dict) -> float:
    c = instance["cost_vector"]
    x = solution.get("x") if isinstance(solution, dict) else None
    if not isinstance(x, list) or len(x) != len(c):
        return float("inf")
    total = 0
    for cj, xj in zip(c, x):
        if xj == 1:
            total += cj
    return float(-total)
