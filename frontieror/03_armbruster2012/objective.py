from math import isfinite


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def objective(instance: dict, solution: dict) -> float:
    if not isinstance(instance, dict):
        raise TypeError("instance must be a dict")
    if not isinstance(solution, dict):
        raise TypeError("solution must be a dict")

    num_edges = instance.get("num_edges")
    if not _is_int(num_edges) or num_edges < 0:
        raise ValueError("instance['num_edges'] must be a nonnegative integer")

    edge_costs = instance.get("edge_costs")
    if not isinstance(edge_costs, list) or len(edge_costs) != num_edges:
        raise ValueError("instance['edge_costs'] must be a list of length num_edges")

    cut_edges = solution.get("cut_edges")
    if not isinstance(cut_edges, list) or len(cut_edges) != num_edges:
        raise ValueError("solution['cut_edges'] must be a list of length num_edges")

    total_cost = 0.0
    for e in range(num_edges):
        cost = edge_costs[e]
        cut = cut_edges[e]
        if not _is_real(cost):
            raise ValueError(f"instance['edge_costs'][{e}] must be a finite real number")
        if not _is_int(cut) or cut not in (0, 1):
            raise ValueError(f"solution['cut_edges'][{e}] must be integer 0 or 1")
        total_cost += float(cost) * cut

    return float(total_cost)
