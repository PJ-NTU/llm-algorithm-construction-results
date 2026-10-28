def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, (list, tuple)) else []


def _as_float(value, default=0.0):
    try:
        if isinstance(value, (dict, list, tuple)) or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _as_nonnegative_int(value, default=0):
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if result >= 0 else default


def _is_selected(value):
    if isinstance(value, (dict, list, tuple)) or value is None:
        return False
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("", "0", "false", "no", "none"):
            return False
    try:
        return float(value) != 0.0
    except (TypeError, ValueError, OverflowError):
        return False


def _node_count(instance):
    params = _as_dict(instance.get("parameters"))
    nodes = _as_list(instance.get("nodes"))

    declared = _as_nonnegative_int(params.get("total_nodes"), 0)
    component_count = (
        _as_nonnegative_int(params.get("num_substations"), 0)
        + _as_nonnegative_int(params.get("num_potential_turbines"), 0)
        + _as_nonnegative_int(params.get("num_steiner_nodes"), 0)
    )

    highest_id = -1
    for position, raw_node in enumerate(nodes):
        node = _as_dict(raw_node)
        node_id = _as_nonnegative_int(node.get("id"), position)
        highest_id = max(highest_id, node_id)

    return max(declared, component_count, len(nodes), highest_id + 1)


def _edge_slots(instance):
    params = _as_dict(instance.get("parameters"))
    raw_edges = _as_list(instance.get("edges"))
    declared = _as_nonnegative_int(params.get("total_edges"), 0)

    highest_id = -1
    for position, raw_edge in enumerate(raw_edges):
        edge = _as_dict(raw_edge)
        edge_id = _as_nonnegative_int(edge.get("id"), position)
        highest_id = max(highest_id, edge_id)

    edge_count = max(declared, len(raw_edges), highest_id + 1)
    slots = [None] * edge_count
    for position, raw_edge in enumerate(raw_edges):
        edge = _as_dict(raw_edge)
        edge_id = _as_nonnegative_int(edge.get("id"), position)
        if 0 <= edge_id < edge_count:
            slots[edge_id] = edge
    return slots


def to_ours(official_instance: dict) -> dict:
    instance = _as_dict(official_instance)
    params = _as_dict(instance.get("parameters"))
    objective = _as_dict(instance.get("objective"))
    raw_nodes = _as_list(instance.get("nodes"))
    node_count = _node_count(instance)

    num_substations = min(
        _as_nonnegative_int(params.get("num_substations"), 0), node_count
    )
    num_turbines = min(
        _as_nonnegative_int(params.get("num_potential_turbines"), 0),
        max(node_count - num_substations, 0),
    )

    node_types = ["steiner"] * node_count
    for i in range(num_substations):
        node_types[i] = "substation"
    for i in range(num_substations, num_substations + num_turbines):
        node_types[i] = "turbine"

    node_coordinates = [[0.0, 0.0] for _ in range(node_count)]
    installation_costs = [0.0] * node_count
    annual_energy_yields = [0.0] * node_count
    node_scenic_impacts = [0.0] * node_count

    type_map = {
        "substation": "substation",
        "potential_turbine": "turbine",
        "steiner": "steiner",
    }

    for position, raw_node in enumerate(raw_nodes):
        node = _as_dict(raw_node)
        node_id = _as_nonnegative_int(node.get("id"), position)
        if not 0 <= node_id < node_count:
            continue

        official_type = node.get("type")
        if official_type in type_map:
            node_types[node_id] = type_map[official_type]

        node_coordinates[node_id] = [
            _as_float(node.get("x"), 0.0),
            _as_float(node.get("y"), 0.0),
        ]
        installation_costs[node_id] = _as_float(node.get("cost"), 0.0)
        annual_energy_yields[node_id] = _as_float(node.get("profit"), 0.0)
        node_scenic_impacts[node_id] = _as_float(
            node.get("scenic_impact"), 0.0
        )

    edge_slots = _edge_slots(instance)
    edges = []
    edge_cable_costs = []
    edge_scenic_impacts = []
    for raw_edge in edge_slots:
        edge = _as_dict(raw_edge)
        edges.append([
            _as_nonnegative_int(edge.get("from"), 0),
            _as_nonnegative_int(edge.get("to"), 0),
        ])
        edge_cable_costs.append(_as_float(edge.get("cost"), 0.0))
        edge_scenic_impacts.append(
            _as_float(edge.get("scenic_impact"), 0.0)
        )

    fallback_alpha = _as_float(objective.get("alpha"), 0.0)
    return {
        "node_types": node_types,
        "node_coordinates": node_coordinates,
        "installation_costs": installation_costs,
        "annual_energy_yields": annual_energy_yields,
        "node_scenic_impacts": node_scenic_impacts,
        "edges": edges,
        "edge_cable_costs": edge_cable_costs,
        "edge_scenic_impacts": edge_scenic_impacts,
        "minimum_energy_quota": _as_float(params.get("quota"), 0.0),
        "alpha": _as_float(params.get("alpha"), fallback_alpha),
    }


def to_official(official_instance: dict, our_solution: dict) -> dict:
    instance = _as_dict(official_instance)
    solution = _as_dict(our_solution)
    ours_instance = to_ours(instance)

    node_selected = _as_list(solution.get("node_selected"))
    selected_turbines = []
    for node_id, node_type in enumerate(ours_instance["node_types"]):
        flag = node_selected[node_id] if node_id < len(node_selected) else 0
        if node_type == "turbine" and _is_selected(flag):
            selected_turbines.append(node_id)

    edge_selected = _as_list(solution.get("edge_selected"))
    selected_edges = []
    for edge_id, raw_edge in enumerate(_edge_slots(instance)):
        flag = edge_selected[edge_id] if edge_id < len(edge_selected) else 0
        edge = _as_dict(raw_edge)
        if not _is_selected(flag) or "from" not in edge or "to" not in edge:
            continue
        selected_edges.append({
            "from": _as_nonnegative_int(edge.get("from"), 0),
            "to": _as_nonnegative_int(edge.get("to"), 0),
            "cost": _as_float(edge.get("cost"), 0.0),
        })

    official_solution = {
        "selected_turbines": selected_turbines,
        "selected_edges": selected_edges,
        "objective_value": 0.0,
    }
    official_solution["objective_value"] = official_objective(
        instance, official_solution
    )
    return official_solution


def official_objective(official_instance: dict, official_solution: dict) -> float:
    instance = _as_dict(official_instance)
    solution = _as_dict(official_solution)
    params = _as_dict(instance.get("parameters"))
    objective = _as_dict(instance.get("objective"))
    alpha = _as_float(
        params.get("alpha"), _as_float(objective.get("alpha"), 0.0)
    )

    node_by_id = {}
    for position, raw_node in enumerate(_as_list(instance.get("nodes"))):
        node = _as_dict(raw_node)
        node_id = _as_nonnegative_int(node.get("id"), position)
        node_by_id[node_id] = node

    edge_by_pair = {}
    for raw_edge in _as_list(instance.get("edges")):
        edge = _as_dict(raw_edge)
        if "from" not in edge or "to" not in edge:
            continue
        u = _as_nonnegative_int(edge.get("from"), 0)
        v = _as_nonnegative_int(edge.get("to"), 0)
        edge_by_pair[(u, v)] = edge
        edge_by_pair[(v, u)] = edge

    edge_value = 0.0
    for raw_selected_edge in _as_list(solution.get("selected_edges")):
        selected_edge = _as_dict(raw_selected_edge)
        if "from" not in selected_edge or "to" not in selected_edge:
            continue
        u = _as_nonnegative_int(selected_edge.get("from"), 0)
        v = _as_nonnegative_int(selected_edge.get("to"), 0)
        edge = edge_by_pair.get((u, v))
        if edge is None:
            continue
        edge_value += (
            alpha * _as_float(edge.get("cost"), 0.0)
            + (1.0 - alpha) * _as_float(edge.get("scenic_impact"), 0.0)
        )

    turbine_value = 0.0
    seen_turbines = set()
    for raw_turbine_id in _as_list(solution.get("selected_turbines")):
        turbine_id = _as_nonnegative_int(raw_turbine_id, -1)
        if turbine_id < 0 or turbine_id in seen_turbines:
            continue
        seen_turbines.add(turbine_id)
        node = node_by_id.get(turbine_id)
        if node is None or node.get("type") != "potential_turbine":
            continue
        turbine_value += (
            alpha * _as_float(node.get("cost"), 0.0)
            + (1.0 - alpha) * _as_float(node.get("scenic_impact"), 0.0)
        )

    return float(edge_value + turbine_value)
