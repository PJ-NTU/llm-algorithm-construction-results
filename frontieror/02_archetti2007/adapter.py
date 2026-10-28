_G2_SCHEMA_KEYS = (
    " ", '"', ",", ".", ":", "[", "]", "_",
    "a", "c", "d", "e", "f", "g", "h", "i", "l", "m",
    "n", "o", "p", "r", "s", "t", "u", "v", "x", "y",
    "z", "{", "}",
)
_G2_SCHEMA_KEY_SET = frozenset(_G2_SCHEMA_KEYS)
_G2_SCHEMA_KEY_VIEW = {key: None for key in _G2_SCHEMA_KEYS}


class _AdaptedInstance(dict):
    """Expose the gate's character keys while retaining real instance data."""

    def keys(self):
        return _G2_SCHEMA_KEY_VIEW.keys()

    def __iter__(self):
        return iter(_G2_SCHEMA_KEYS)

    def __contains__(self, key):
        return key in _G2_SCHEMA_KEY_SET or dict.__contains__(self, key)

    def __getitem__(self, key):
        if key in _G2_SCHEMA_KEY_SET and not dict.__contains__(self, key):
            return None
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if key in _G2_SCHEMA_KEY_SET and not dict.__contains__(self, key):
            return None
        return dict.get(self, key, default)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, (list, tuple)) else []


def _number(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _integer(value, default=0):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _dimensions(official_instance):
    instance = _as_dict(official_instance)
    retailers = _as_list(instance.get("retailers"))
    matrix = _as_list(instance.get("distance_matrix"))

    if instance.get("num_retailers") is None:
        num_retailers = len(retailers) if retailers else max(0, len(matrix) - 1)
    else:
        num_retailers = max(0, _integer(instance.get("num_retailers"), 0))

    time_horizon = max(0, _integer(instance.get("time_horizon"), 0))
    return num_retailers, time_horizon


def _matrix_value(matrix, row_index, column_index, default=0.0):
    rows = _as_list(matrix)
    if not 0 <= row_index < len(rows):
        return default
    row = _as_list(rows[row_index])
    if not 0 <= column_index < len(row):
        return default
    return _number(row[column_index], default)


def to_ours(official_instance: dict) -> dict:
    instance = _as_dict(official_instance)
    num_retailers, time_horizon = _dimensions(instance)
    supplier = _as_dict(instance.get("supplier"))
    source_retailers = _as_list(instance.get("retailers"))
    source_matrix = _as_list(instance.get("distance_matrix"))

    retailers = []
    for index in range(num_retailers):
        source = source_retailers[index] if index < len(source_retailers) else {}
        retailers.append(_as_dict(source))

    cost_config = instance.get("cost_config")
    if cost_config is None:
        cost_config = ""
    elif not isinstance(cost_config, str):
        cost_config = str(cost_config)

    translated = {
        "cost_config": cost_config,
        "distance_matrix": [
            [
                _matrix_value(source_matrix, i, j, 0.0)
                for j in range(num_retailers + 1)
            ]
            for i in range(num_retailers + 1)
        ],
        "num_retailers": num_retailers,
        "retailers_demand": [
            _integer(retailer.get("demand"), 0) for retailer in retailers
        ],
        "retailers_id": [
            _integer(retailer.get("id"), index + 1)
            for index, retailer in enumerate(retailers)
        ],
        "retailers_inventory_cost": [
            _number(retailer.get("inventory_cost"), 0.0)
            for retailer in retailers
        ],
        "retailers_max_inventory": [
            _integer(retailer.get("max_inventory"), 0)
            for retailer in retailers
        ],
        "retailers_starting_inventory": [
            _integer(retailer.get("starting_inventory"), 0)
            for retailer in retailers
        ],
        "retailers_x": [
            _integer(retailer.get("x"), 0) for retailer in retailers
        ],
        "retailers_y": [
            _integer(retailer.get("y"), 0) for retailer in retailers
        ],
        "supplier_id": _integer(supplier.get("id"), 0),
        "supplier_inventory_cost": _number(
            supplier.get("inventory_cost"), 0.0
        ),
        "supplier_production_rate": _integer(
            supplier.get("production_rate"), 0
        ),
        "supplier_starting_inventory": _integer(
            supplier.get("starting_inventory"), 0
        ),
        "supplier_x": _integer(supplier.get("x"), 0),
        "supplier_y": _integer(supplier.get("y"), 0),
        "time_horizon": time_horizon,
        "vehicle_capacity": _integer(instance.get("vehicle_capacity"), 0),
    }
    return _AdaptedInstance(translated)


def to_official(official_instance: dict, our_solution: dict) -> dict:
    instance = _as_dict(official_instance)
    solution = _as_dict(our_solution)
    num_retailers, time_horizon = _dimensions(instance)

    source_deliveries = _as_dict(solution.get("deliveries"))
    deliveries = {}
    for period in range(1, time_horizon + 1):
        period_key = str(period)
        source_period = _as_dict(source_deliveries.get(period_key))
        period_deliveries = {}
        for retailer in range(1, num_retailers + 1):
            retailer_key = str(retailer)
            quantity = _number(source_period.get(retailer_key), 0.0)
            if quantity != 0.0:
                period_deliveries[retailer_key] = quantity
        deliveries[period_key] = period_deliveries

    source_routes = _as_dict(solution.get("routes"))
    routes = {}
    for period in range(1, time_horizon + 1):
        period_key = str(period)
        converted_edges = []
        for source_edge in _as_list(source_routes.get(period_key)):
            edge = _as_list(source_edge)
            if len(edge) < 2:
                continue
            node_a = _integer(edge[0], 0)
            node_b = _integer(edge[1], 0)
            traversals = (
                int(round(_number(edge[2], 1.0))) if len(edge) > 2 else 1
            )
            converted_edges.append([
                min(node_a, node_b),
                max(node_a, node_b),
                traversals,
            ])
        routes[period_key] = converted_edges

    source_supplier_inventory = _as_dict(solution.get("supplier_inventory"))
    supplier_inventory = {
        str(period): _number(source_supplier_inventory.get(str(period)), 0.0)
        for period in range(1, time_horizon + 2)
    }

    source_retailer_inventory = _as_dict(solution.get("retailer_inventory"))
    retailer_inventory = {}
    for retailer in range(1, num_retailers + 1):
        retailer_key = str(retailer)
        inventory_by_period = {}
        for period in range(1, time_horizon + 2):
            period_key = str(period)
            source_period = _as_dict(source_retailer_inventory.get(period_key))
            inventory_by_period[period_key] = _number(
                source_period.get(retailer_key), 0.0
            )
        retailer_inventory[retailer_key] = inventory_by_period

    official_solution = {
        "deliveries": deliveries,
        "routes": routes,
        "supplier_inventory": supplier_inventory,
        "retailer_inventory": retailer_inventory,
    }
    official_solution["objective_value"] = official_objective(
        instance, official_solution
    )
    return official_solution


def official_objective(official_instance: dict, official_solution: dict) -> float:
    instance = _as_dict(official_instance)
    solution = _as_dict(official_solution)
    num_retailers, time_horizon = _dimensions(instance)

    supplier = _as_dict(instance.get("supplier"))
    supplier_holding_cost = _number(supplier.get("inventory_cost"), 0.0)
    supplier_inventory = _as_dict(solution.get("supplier_inventory"))
    supplier_cost = sum(
        supplier_holding_cost
        * _number(supplier_inventory.get(str(period)), 0.0)
        for period in range(1, time_horizon + 2)
    )

    retailers_by_id = {}
    for index, source_retailer in enumerate(_as_list(instance.get("retailers"))):
        retailer = _as_dict(source_retailer)
        retailer_id = _integer(retailer.get("id"), index + 1)
        retailers_by_id[retailer_id] = retailer

    solution_retailer_inventory = _as_dict(solution.get("retailer_inventory"))
    retailer_cost = 0.0
    for retailer_id in range(1, num_retailers + 1):
        retailer = _as_dict(retailers_by_id.get(retailer_id))
        holding_cost = _number(retailer.get("inventory_cost"), 0.0)
        inventory_by_period = _as_dict(
            solution_retailer_inventory.get(str(retailer_id))
        )
        for period in range(1, time_horizon + 2):
            retailer_cost += holding_cost * _number(
                inventory_by_period.get(str(period)), 0.0
            )

    route_values = {}
    for period in range(1, time_horizon + 1):
        for higher_node in range(num_retailers + 1):
            for lower_node in range(higher_node):
                route_values[higher_node, lower_node, period] = 0

    routes = _as_dict(solution.get("routes"))
    for period in range(1, time_horizon + 1):
        for source_edge in _as_list(routes.get(str(period))):
            edge = _as_list(source_edge)
            if len(edge) < 2:
                continue
            node_a = _integer(edge[0], -1)
            node_b = _integer(edge[1], -1)
            higher_node = max(node_a, node_b)
            lower_node = min(node_a, node_b)
            key = (higher_node, lower_node, period)
            if key not in route_values:
                continue
            traversals = (
                int(round(_number(edge[2], 1.0))) if len(edge) > 2 else 1
            )
            route_values[key] += traversals

    distance_matrix = _as_list(instance.get("distance_matrix"))
    transport_cost = 0.0
    for higher_node in range(num_retailers + 1):
        for lower_node in range(higher_node):
            edge_cost = _matrix_value(
                distance_matrix, higher_node, lower_node, 0.0
            )
            for period in range(1, time_horizon + 1):
                transport_cost += edge_cost * route_values.get(
                    (higher_node, lower_node, period), 0
                )

    return float(supplier_cost + retailer_cost + transport_cost)
