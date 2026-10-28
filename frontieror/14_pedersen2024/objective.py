"""Objective evaluator for the wind-farm tree model; smaller is better."""

import math


_SOLUTION_FIELDS = {"node_selected", "edge_selected"}


def _is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _numeric_vector(instance, name, expected_length):
    values = instance.get(name)
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(f"{name} must be a list of length {expected_length}")
    if any(not _is_number(value) for value in values):
        raise ValueError(f"{name} must contain only finite numbers")
    return values


def _binary_vector(solution, name, expected_length):
    values = solution.get(name)
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(f"{name} must be a list of length {expected_length}")
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise ValueError(f"{name} must contain only integer 0/1 values")
    return values


def objective(instance: dict, solution: dict) -> float:
    """Return alpha*financial_cost + (1-alpha)*scenic_impact."""
    if not isinstance(instance, dict):
        raise ValueError("instance must be a dict")
    if not isinstance(solution, dict):
        raise ValueError("solution must be a dict")
    if set(solution) != _SOLUTION_FIELDS:
        missing = sorted(_SOLUTION_FIELDS - set(solution))
        unexpected = sorted(str(key) for key in set(solution) - _SOLUTION_FIELDS)
        raise ValueError(
            f"solution fields do not match the contract; missing={missing}, unexpected={unexpected}"
        )

    node_types = instance.get("node_types")
    edges = instance.get("edges")
    if not isinstance(node_types, list):
        raise ValueError("node_types must be a list")
    if not isinstance(edges, list):
        raise ValueError("edges must be a list")

    n = len(node_types)
    m = len(edges)
    node_selected = _binary_vector(solution, "node_selected", n)
    edge_selected = _binary_vector(solution, "edge_selected", m)

    installation_costs = _numeric_vector(instance, "installation_costs", n)
    node_scenic_impacts = _numeric_vector(instance, "node_scenic_impacts", n)
    edge_cable_costs = _numeric_vector(instance, "edge_cable_costs", m)
    edge_scenic_impacts = _numeric_vector(instance, "edge_scenic_impacts", m)

    alpha = instance.get("alpha")
    if not _is_number(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite number in [0, 1]")

    financial_cost = sum(
        edge_cable_costs[e] * edge_selected[e] for e in range(m)
    ) + sum(
        installation_costs[i] * node_selected[i]
        for i, kind in enumerate(node_types)
        if kind == "turbine"
    )

    scenic_impact = sum(
        edge_scenic_impacts[e] * edge_selected[e] for e in range(m)
    ) + sum(
        node_scenic_impacts[i] * node_selected[i]
        for i, kind in enumerate(node_types)
        if kind == "turbine"
    )

    value = alpha * financial_cost + (1.0 - alpha) * scenic_impact
    if not math.isfinite(value):
        raise ValueError("computed objective value is not finite")
    return float(value)
