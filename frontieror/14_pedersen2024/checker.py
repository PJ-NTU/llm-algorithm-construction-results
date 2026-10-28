"""Feasibility checker for the quota-constrained wind-farm Steiner tree model."""

import math


_INSTANCE_FIELDS = {
    "node_types",
    "node_coordinates",
    "installation_costs",
    "annual_energy_yields",
    "node_scenic_impacts",
    "edges",
    "edge_cable_costs",
    "edge_scenic_impacts",
    "minimum_energy_quota",
    "alpha",
}
_SOLUTION_FIELDS = {"node_selected", "edge_selected"}
_NODE_TYPES = {"substation", "turbine", "steiner"}


def _is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_instance(instance):
    errors = []
    if not isinstance(instance, dict):
        return None, ["Instance structure: instance must be a dict."]

    missing = sorted(_INSTANCE_FIELDS - set(instance))
    if missing:
        return None, [f"Instance structure: missing fields {missing}."]

    node_types = instance["node_types"]
    if not isinstance(node_types, list) or not node_types:
        errors.append("Instance structure: node_types must be a non-empty list[str].")
        return None, errors
    if any(type(value) is not str or value not in _NODE_TYPES for value in node_types):
        errors.append(
            "Instance structure: each entry of node_types must be substation, turbine, or steiner."
        )

    n = len(node_types)
    if "substation" not in node_types:
        errors.append("Instance structure: at least one substation is required to select the auxiliary-flow root node.")

    coordinates = instance["node_coordinates"]
    if not isinstance(coordinates, list) or len(coordinates) != n:
        errors.append(f"Instance structure: node_coordinates must have exactly {n} rows.")
    else:
        dimension = None
        for i, row in enumerate(coordinates):
            if not isinstance(row, list) or not row:
                errors.append(f"Instance structure: node_coordinates[{i}] must be a non-empty list.")
                continue
            if dimension is None:
                dimension = len(row)
            elif len(row) != dimension:
                errors.append("Instance structure: all node coordinates must have the same dimension.")
                break
            if any(not _is_number(value) for value in row):
                errors.append(f"Instance structure: node_coordinates[{i}] contains non-finite values.")

    for name in (
        "installation_costs",
        "annual_energy_yields",
        "node_scenic_impacts",
    ):
        values = instance[name]
        if not isinstance(values, list) or len(values) != n:
            errors.append(f"Instance structure: {name} must be a list of length {n}.")
        elif any(not _is_number(value) or value < 0 for value in values):
            errors.append(f"Instance structure: {name} must contain only non-negative finite values.")

    edges = instance["edges"]
    if not isinstance(edges, list):
        errors.append("Instance structure: edges must be a list[list[int]].")
        edges = []
    else:
        for e, endpoints in enumerate(edges):
            valid = (
                isinstance(endpoints, list)
                and len(endpoints) == 2
                and all(type(v) is int for v in endpoints)
                and all(0 <= v < n for v in endpoints)
                and endpoints[0] != endpoints[1]
            )
            if not valid:
                errors.append(
                    f"Instance structure: edges[{e}] must contain two distinct valid 0-based node indices."
                )

    m = len(edges)
    for name in ("edge_cable_costs", "edge_scenic_impacts"):
        values = instance[name]
        if not isinstance(values, list) or len(values) != m:
            errors.append(f"Instance structure: {name} must be a list of length {m}.")
        elif any(not _is_number(value) or value < 0 for value in values):
            errors.append(f"Instance structure: {name} must contain only non-negative finite values.")

    quota = instance["minimum_energy_quota"]
    if not _is_number(quota) or quota < 0:
        errors.append("Instance structure: minimum_energy_quota must be a non-negative finite value.")

    alpha = instance["alpha"]
    if not _is_number(alpha) or not 0 <= alpha <= 1:
        errors.append("Instance structure: alpha must be a finite value in [0,1].")

    if errors:
        return None, errors
    return (node_types, edges, instance["annual_energy_yields"], quota), []


def _read_binary_vector(solution, key, expected_length, errors):
    values = solution.get(key)
    if not isinstance(values, list):
        errors.append(f"Solution structure: {key} must be a list.")
        return None
    if len(values) != expected_length:
        errors.append(
            f"Solution structure: {key} should have length {expected_length}, but has length {len(values)}."
        )
        return None
    invalid = [i for i, value in enumerate(values) if type(value) is not int or value not in (0, 1)]
    if invalid:
        errors.append(f"Solution structure: {key} at index {invalid} is not the integer 0 or 1.")
        return None
    return values


def check(instance: dict, solution: dict) -> list:
    """Return all detected violations; an empty list means feasible."""
    data, instance_errors = _validate_instance(instance)
    if instance_errors:
        return instance_errors

    node_types, edges, energy_yields, quota = data
    n = len(node_types)
    m = len(edges)
    violations = []

    if not isinstance(solution, dict):
        return ["Solution structure: solution must be a dict."]

    missing = sorted(_SOLUTION_FIELDS - set(solution))
    unexpected = sorted(str(key) for key in set(solution) - _SOLUTION_FIELDS)
    if missing:
        violations.append(f"Solution structure: missing fields {missing}.")
    if unexpected:
        violations.append(f"Solution structure: unexpected fields {unexpected}.")

    node_selected = _read_binary_vector(solution, "node_selected", n, violations)
    edge_selected = _read_binary_vector(solution, "edge_selected", m, violations)
    if violations:
        return violations

    substations = [i for i, kind in enumerate(node_types) if kind == "substation"]
    root = substations[0]

    # C1
    missing_substations = [i for i in substations if node_selected[i] != 1]
    if missing_substations:
        violations.append(
            f"C1: substation nodes {missing_substations} are not included in the selected tree."
        )

    # C2
    bad_endpoint_edges = []
    for e, chosen in enumerate(edge_selected):
        if chosen:
            u, v = edges[e]
            if node_selected[u] != 1 or node_selected[v] != 1:
                bad_endpoint_edges.append(e)
    if bad_endpoint_edges:
        violations.append(
            f"C2: selected edges {bad_endpoint_edges} have at least one unselected endpoint."
        )

    # C3
    total_energy = sum(
        energy_yields[i] * node_selected[i]
        for i, kind in enumerate(node_types)
        if kind == "turbine"
    )
    if total_energy < quota:
        violations.append(
            f"C3: annual energy yield of selected turbines {total_energy} is below the minimum quota {quota}."
        )

    # The auxiliary flow is not part of the output contract. A feasible auxiliary flow for C4--C6 exists iff every selected non-root node with unit demand
    # is reachable from the root via selected edges; take a BFS tree and set each edge flow to the subtree demand count.
    adjacency = [[] for _ in range(n)]
    for e, chosen in enumerate(edge_selected):
        if chosen:
            u, v = edges[e]
            adjacency[u].append(v)
            adjacency[v].append(u)

    reached = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if v not in reached:
                reached.add(v)
                stack.append(v)

    unreachable = [
        i
        for i in range(n)
        if i != root and node_selected[i] == 1 and i not in reached
    ]

    # C4
    if unreachable:
        violations.append(
            f"C4: selected non-root nodes {unreachable} cannot receive the required one unit of net auxiliary flow from the root."
        )

    # C5
    if unreachable:
        violations.append(
            "C5: the root cannot supply auxiliary flow covering the total demand of all selected non-root nodes."
        )

    # C6
    if unreachable:
        violations.append(
            "C6: the required auxiliary flow cannot be routed along selected edges only within the |V|-1 capacity bound."
        )

    # C7
    selected_edge_count = sum(edge_selected)
    selected_node_count = sum(node_selected)
    required_edge_count = selected_node_count - 1
    if selected_edge_count != required_edge_count:
        violations.append(
            f"C7: number of selected edges {selected_edge_count} is not equal to the number of selected nodes minus one {required_edge_count}."
        )

    return violations
