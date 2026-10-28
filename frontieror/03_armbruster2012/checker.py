from math import isfinite


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def check(instance: dict, solution: dict) -> list:
    violations = []

    if not isinstance(instance, dict):
        return ["structure: instance must be a dict"]

    num_nodes_raw = instance.get("num_nodes")
    num_nodes_ok = _is_int(num_nodes_raw) and num_nodes_raw >= 0
    if not num_nodes_ok:
        violations.append("instance structure: num_nodes must be a nonnegative integer")
    num_nodes = num_nodes_raw if num_nodes_ok else None

    num_edges_raw = instance.get("num_edges")
    num_edges_ok = _is_int(num_edges_raw) and num_edges_raw >= 0
    if not num_edges_ok:
        violations.append("instance structure: num_edges must be a nonnegative integer")
    num_edges = num_edges_raw if num_edges_ok else None

    node_weights = instance.get("node_weights")
    weights_ok = isinstance(node_weights, list) and num_nodes is not None
    if not isinstance(node_weights, list):
        violations.append("instance structure: node_weights must be a list")
    elif num_nodes is not None and len(node_weights) != num_nodes:
        violations.append(
            f"instance structure: node_weights must have length {num_nodes}, got {len(node_weights)}"
        )
        weights_ok = False
    elif num_nodes is not None:
        for i, weight in enumerate(node_weights):
            if not _is_int(weight) or weight < 0:
                violations.append(
                    f"instance structure: node_weights[{i}] must be a nonnegative integer"
                )
                weights_ok = False

    edges = instance.get("edges")
    edges_ok = isinstance(edges, list) and num_edges is not None and num_nodes is not None
    if not isinstance(edges, list):
        violations.append("instance structure: edges must be a list")
    elif num_edges is not None and len(edges) != num_edges:
        violations.append(
            f"instance structure: edges must have length {num_edges}, got {len(edges)}"
        )
        edges_ok = False
    elif num_edges is not None and num_nodes is not None:
        for e, endpoints in enumerate(edges):
            if not isinstance(endpoints, list) or len(endpoints) != 2:
                violations.append(
                    f"instance structure: edges[{e}] must be a two-element list"
                )
                edges_ok = False
                continue
            for position, endpoint in enumerate(endpoints):
                if not _is_int(endpoint) or not 0 <= endpoint < num_nodes:
                    violations.append(
                        f"instance structure: edges[{e}][{position}] must be a node index in [0, {num_nodes - 1}]"
                    )
                    edges_ok = False

    edge_costs = instance.get("edge_costs")
    costs_ok = isinstance(edge_costs, list) and num_edges is not None
    if not isinstance(edge_costs, list):
        violations.append("instance structure: edge_costs must be a list")
    elif num_edges is not None and len(edge_costs) != num_edges:
        violations.append(
            f"instance structure: edge_costs must have length {num_edges}, got {len(edge_costs)}"
        )
        costs_ok = False
    elif num_edges is not None:
        for e, cost in enumerate(edge_costs):
            if not _is_real(cost):
                violations.append(
                    f"instance structure: edge_costs[{e}] must be a finite real number"
                )
                costs_ok = False

    capacity = instance.get("bisection_capacity")
    capacity_ok = _is_int(capacity)
    if not capacity_ok:
        violations.append("instance structure: bisection_capacity must be an integer")
    elif weights_ok:
        total_weight = sum(node_weights)
        lower_bound = (total_weight + 1) // 2
        if not lower_bound <= capacity <= total_weight:
            violations.append(
                "instance structure: bisection_capacity must be between "
                f"ceil(total node weight / 2)={lower_bound} and total node weight={total_weight}"
            )
            capacity_ok = False

    if not isinstance(solution, dict):
        violations.append("solution structure: solution must be a dict")
        return violations

    if "node_groups" not in solution:
        violations.append("solution structure: missing required field 'node_groups'")
        node_groups = None
    else:
        node_groups = solution["node_groups"]

    groups_ok = isinstance(node_groups, list) and num_nodes is not None
    if node_groups is not None and not isinstance(node_groups, list):
        violations.append("solution structure: node_groups must be a list")
    elif isinstance(node_groups, list) and num_nodes is not None:
        if len(node_groups) != num_nodes:
            violations.append(
                f"solution structure: node_groups must have length {num_nodes}, got {len(node_groups)}"
            )
            groups_ok = False
        else:
            for i, group in enumerate(node_groups):
                if not _is_int(group) or group not in (1, 2):
                    violations.append(
                        f"solution structure: node_groups[{i}] must be integer 1 or 2"
                    )
                    groups_ok = False

    if "cut_edges" not in solution:
        violations.append("solution structure: missing required field 'cut_edges'")
        cut_edges = None
    else:
        cut_edges = solution["cut_edges"]

    cuts_ok = isinstance(cut_edges, list) and num_edges is not None
    if cut_edges is not None and not isinstance(cut_edges, list):
        violations.append("solution structure: cut_edges must be a list")
    elif isinstance(cut_edges, list) and num_edges is not None:
        if len(cut_edges) != num_edges:
            violations.append(
                f"solution structure: cut_edges must have length {num_edges}, got {len(cut_edges)}"
            )
            cuts_ok = False
        else:
            for e, cut in enumerate(cut_edges):
                if not _is_int(cut) or cut not in (0, 1):
                    violations.append(
                        f"solution structure: cut_edges[{e}] must be integer 0 or 1"
                    )
                    cuts_ok = False

    # C1: The total weight assigned to group 1 must not exceed the capacity.
    if groups_ok and weights_ok and capacity_ok:
        group_1_weight = sum(
            node_weights[i] for i in range(num_nodes) if node_groups[i] == 1
        )
        if group_1_weight > capacity:
            violations.append(
                f"C1 violated: group 1 weight {group_1_weight} exceeds capacity {capacity}"
            )

    # C2: The total weight assigned to group 2 must not exceed the capacity.
    if groups_ok and weights_ok and capacity_ok:
        group_2_weight = sum(
            node_weights[i] for i in range(num_nodes) if node_groups[i] == 2
        )
        if group_2_weight > capacity:
            violations.append(
                f"C2 violated: group 2 weight {group_2_weight} exceeds capacity {capacity}"
            )

    # C3: Group 1 must contain at least one node.
    if groups_ok and not any(group == 1 for group in node_groups):
        violations.append("C3 violated: group 1 is empty")

    # C4: Group 2 must contain at least one node.
    if groups_ok and not any(group == 2 for group in node_groups):
        violations.append("C4 violated: group 2 is empty")

    # C5: y_e >= x_u - x_v for every edge e.
    if groups_ok and cuts_ok and edges_ok:
        for e, (u, v) in enumerate(edges):
            x_u = 1 if node_groups[u] == 1 else 0
            x_v = 1 if node_groups[v] == 1 else 0
            if cut_edges[e] < x_u - x_v:
                violations.append(
                    f"C5 violated at edge {e}: cut_edges[{e}]={cut_edges[e]} is less than x[{u}]-x[{v}]={x_u - x_v}"
                )

    # C6: y_e >= x_v - x_u for every edge e.
    if groups_ok and cuts_ok and edges_ok:
        for e, (u, v) in enumerate(edges):
            x_u = 1 if node_groups[u] == 1 else 0
            x_v = 1 if node_groups[v] == 1 else 0
            if cut_edges[e] < x_v - x_u:
                violations.append(
                    f"C6 violated at edge {e}: cut_edges[{e}]={cut_edges[e]} is less than x[{v}]-x[{u}]={x_v - x_u}"
                )

    # C7: y_e <= x_u + x_v for every edge e.
    if groups_ok and cuts_ok and edges_ok:
        for e, (u, v) in enumerate(edges):
            x_u = 1 if node_groups[u] == 1 else 0
            x_v = 1 if node_groups[v] == 1 else 0
            if cut_edges[e] > x_u + x_v:
                violations.append(
                    f"C7 violated at edge {e}: cut_edges[{e}]={cut_edges[e]} exceeds x[{u}]+x[{v}]={x_u + x_v}"
                )

    # C8: y_e <= 2 - x_u - x_v for every edge e.
    if groups_ok and cuts_ok and edges_ok:
        for e, (u, v) in enumerate(edges):
            x_u = 1 if node_groups[u] == 1 else 0
            x_v = 1 if node_groups[v] == 1 else 0
            upper_bound = 2 - x_u - x_v
            if cut_edges[e] > upper_bound:
                violations.append(
                    f"C8 violated at edge {e}: cut_edges[{e}]={cut_edges[e]} exceeds 2-x[{u}]-x[{v}]={upper_bound}"
                )

    return violations
