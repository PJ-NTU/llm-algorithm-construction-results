def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _dimensions(official_instance):
    instance = _as_dict(official_instance)
    raw_node_weights = _as_list(instance.get("node_weights"))
    raw_edges = _as_list(instance.get("edges"))
    raw_edge_weights = _as_list(instance.get("edge_weights"))

    inferred_nodes = len(raw_node_weights)
    for edge in raw_edges:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            u = _to_int(edge[0], -1)
            v = _to_int(edge[1], -1)
            if u >= 0:
                inferred_nodes = max(inferred_nodes, u + 1)
            if v >= 0:
                inferred_nodes = max(inferred_nodes, v + 1)

    declared_nodes = _to_int(instance.get("num_nodes"), -1)
    num_nodes = declared_nodes if declared_nodes >= 0 else inferred_nodes

    inferred_edges = max(len(raw_edges), len(raw_edge_weights))
    declared_edges = _to_int(instance.get("num_edges"), -1)
    num_edges = declared_edges if declared_edges >= 0 else inferred_edges

    return num_nodes, num_edges


def to_ours(official_instance: dict) -> dict:
    instance = _as_dict(official_instance)
    num_nodes, num_edges = _dimensions(instance)

    raw_node_weights = _as_list(instance.get("node_weights"))
    node_weights = [
        _to_int(raw_node_weights[i], 0) if i < len(raw_node_weights) else 0
        for i in range(num_nodes)
    ]

    raw_edges = _as_list(instance.get("edges"))
    edges = []
    for i in range(num_edges):
        edge = raw_edges[i] if i < len(raw_edges) else None
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            edges.append([_to_int(edge[0], 0), _to_int(edge[1], 0)])
        else:
            edges.append([0, 0])

    raw_edge_weights = _as_list(instance.get("edge_weights"))
    edge_costs = [
        _to_float(raw_edge_weights[i], 0.0)
        if i < len(raw_edge_weights)
        else 0.0
        for i in range(num_edges)
    ]

    default_capacity = (sum(node_weights) + 1) // 2
    bisection_capacity = _to_int(
        instance.get("bisection_capacity_F"), default_capacity
    )

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "node_weights": node_weights,
        "edges": edges,
        "edge_costs": edge_costs,
        "bisection_capacity": bisection_capacity,
    }


def _extract_official_partition(num_nodes, official_solution):
    solution = _as_dict(official_solution)
    raw_partition = _as_list(solution.get("partition"))

    if raw_partition:
        return [
            _to_int(raw_partition[i], 0) if i < len(raw_partition) else 0
            for i in range(num_nodes)
        ]

    partition = [0] * num_nodes
    partition_s = _as_list(solution.get("partition_S"))
    partition_complement = _as_list(solution.get("partition_complement"))

    for raw_node in partition_s:
        node = _to_int(raw_node, -1)
        if 0 <= node < num_nodes:
            partition[node] = 0

    for raw_node in partition_complement:
        node = _to_int(raw_node, -1)
        if 0 <= node < num_nodes:
            partition[node] = 1

    return partition


def to_official(official_instance: dict, our_solution: dict) -> dict:
    instance = _as_dict(official_instance)
    solution = _as_dict(our_solution)
    num_nodes, _ = _dimensions(instance)

    raw_groups = _as_list(solution.get("node_groups"))
    if "num_nodes" not in instance and len(raw_groups) > num_nodes:
        num_nodes = len(raw_groups)

    groups = []
    for i in range(num_nodes):
        raw_group = raw_groups[i] if i < len(raw_groups) else 1
        groups.append(2 if _to_int(raw_group, 1) == 2 else 1)

    center_group = groups[0] if groups else 1
    partition_s = [i for i, group in enumerate(groups) if group == center_group]
    partition_complement = [
        i for i, group in enumerate(groups) if group != center_group
    ]

    translated = {
        "partition_S": partition_s,
        "partition_complement": partition_complement,
    }
    objective_value = official_objective(instance, translated)

    return {
        "objective_value": objective_value,
        "partition_S": partition_s,
        "partition_complement": partition_complement,
        "num_nodes": num_nodes,
        "cut_value_original_edges": objective_value,
    }


def official_objective(official_instance: dict, official_solution: dict) -> float:
    ours = to_ours(official_instance)
    num_nodes = ours["num_nodes"]
    partition = _extract_official_partition(num_nodes, official_solution)

    objective = 0.0
    for edge, cost in zip(ours["edges"], ours["edge_costs"]):
        u, v = edge
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            if partition[u] != partition[v]:
                objective += float(cost)

    return float(objective)
