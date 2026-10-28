def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, (list, tuple)) else []


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _as_count(value):
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _at(values, index, default=None):
    values = _as_list(values)
    if 0 <= index < len(values):
        return values[index]
    return default


def _dimensions(instance):
    instance = _as_dict(instance)
    demands = _as_list(instance.get("customer_demands"))

    n = _as_count(instance.get("num_customers"))
    if n is None:
        n = max(
            len(demands),
            max(0, len(_as_list(instance.get("inventory_capacities"))) - 1),
            max(0, len(_as_list(instance.get("initial_inventories"))) - 1),
            max(0, len(_as_list(instance.get("holding_costs"))) - 1),
            max(0, len(_as_list(instance.get("distance_matrix"))) - 1),
        )

    periods_from_demands = max(
        [len(_as_list(row)) for row in demands] or [0]
    )
    T = _as_count(instance.get("num_periods"))
    if T is None:
        T = max(
            len(_as_list(instance.get("supplier_production"))),
            periods_from_demands,
        )

    m = _as_count(instance.get("num_vehicles"))
    if m is None:
        m = 0

    return n, T, m


def _vector(value, length, default=0.0):
    source = _as_list(value)
    return [
        _as_float(source[i], default) if i < len(source) else float(default)
        for i in range(length)
    ]


def _matrix(value, rows, columns, default=0.0):
    source = _as_list(value)
    result = []
    for r in range(rows):
        row = _as_list(source[r]) if r < len(source) else []
        result.append([
            _as_float(row[c], default) if c < len(row) else float(default)
            for c in range(columns)
        ])
    return result


def to_ours(official_instance: dict) -> dict:
    instance = _as_dict(official_instance)
    n, T, m = _dimensions(instance)

    return {
        "num_periods": T,
        "num_customers": n,
        "num_vehicles": m,
        "vehicle_capacity": _as_float(instance.get("vehicle_capacity"), 0.0),
        "travel_costs": _matrix(
            instance.get("distance_matrix"), n + 1, n + 1
        ),
        "production": _vector(instance.get("supplier_production"), T),
        "demands": _matrix(instance.get("customer_demands"), n, T),
        "inventory_capacities": _vector(
            instance.get("inventory_capacities"), n + 1
        ),
        "initial_inventories": _vector(
            instance.get("initial_inventories"), n + 1
        ),
        "holding_costs": _vector(instance.get("holding_costs"), n + 1),
    }


def _official_route_entries(official_solution, num_periods):
    solution = _as_dict(official_solution)
    raw_routes = _as_dict(solution.get("routes"))
    entries = []

    for period in range(1, num_periods + 1):
        period_data = raw_routes.get(str(period), {})
        if isinstance(period_data, list):
            route_objects = period_data
        elif isinstance(period_data, dict):
            route_objects = list(period_data.values())
        else:
            route_objects = []

        for route_object in route_objects:
            route_object = _as_dict(route_object)
            route = list(_as_list(route_object.get("route")))
            deliveries = []
            for customer, quantity in _as_dict(
                route_object.get("deliveries")
            ).items():
                try:
                    customer_id = int(customer)
                except (TypeError, ValueError, OverflowError):
                    continue
                deliveries.append((customer_id, _as_float(quantity, 0.0)))
            entries.append((period, route, deliveries))

    return entries


def _recomputed_inventories(ours, official_solution):
    n = ours["num_customers"]
    T = ours["num_periods"]
    entries = _official_route_entries(official_solution, T)

    shipped = [0.0 for _ in range(T)]
    delivered = [[0.0 for _ in range(n + 1)] for _ in range(T)]

    for period, route, route_deliveries in entries:
        t = period - 1
        for customer, quantity in route_deliveries:
            shipped[t] += quantity
            if 1 <= customer <= n:
                delivered[t][customer] += quantity

    previous = list(ours["initial_inventories"])
    ending = []
    for t in range(T):
        current = [0.0 for _ in range(n + 1)]
        current[0] = previous[0] + ours["production"][t] - shipped[t]
        for customer in range(1, n + 1):
            current[customer] = (
                previous[customer]
                + delivered[t][customer]
                - ours["demands"][customer - 1][t]
            )
        ending.append(current)
        previous = current

    return ending


def _objective_components(official_instance, official_solution):
    ours = to_ours(official_instance)
    n = ours["num_customers"]
    T = ours["num_periods"]
    valid_nodes = set(range(n + 1))
    entries = _official_route_entries(official_solution, T)

    travel_cost = 0.0
    for period, route, deliveries in entries:
        for index in range(len(route) - 1):
            start = route[index]
            end = route[index + 1]
            try:
                if start in valid_nodes and end in valid_nodes:
                    travel_cost += ours["travel_costs"][start][end]
            except (TypeError, IndexError):
                pass

    ending = _recomputed_inventories(ours, official_solution)
    holding_cost = 0.0
    for t in range(T):
        for node in range(n + 1):
            holding_cost += ours["holding_costs"][node] * ending[t][node]

    return travel_cost + holding_cost, travel_cost, holding_cost


def to_official(official_instance: dict, our_solution: dict) -> dict:
    ours = to_ours(official_instance)
    solution = _as_dict(our_solution)
    n = ours["num_customers"]
    T = ours["num_periods"]
    m = ours["num_vehicles"]

    raw_routes = _as_list(solution.get("routes"))
    raw_deliveries = _as_list(solution.get("deliveries"))
    official_routes = {}

    for t in range(T):
        period_routes = _as_list(_at(raw_routes, t, []))
        period_deliveries = _as_list(_at(raw_deliveries, t, []))
        official_period = {}

        for vehicle in range(m):
            customer_route = list(_as_list(_at(period_routes, vehicle, [])))
            delivery_row = _as_list(_at(period_deliveries, vehicle, []))
            quantities = [
                _as_float(_at(delivery_row, customer, 0.0), 0.0)
                for customer in range(n)
            ]

            if customer_route or any(quantity != 0.0 for quantity in quantities):
                official_period[str(vehicle)] = {
                    "route": [0] + customer_route + [0],
                    "deliveries": {
                        str(customer + 1): quantity
                        for customer, quantity in enumerate(quantities)
                        if quantity != 0.0
                    },
                }

        official_routes[str(t + 1)] = official_period

    provisional = {"routes": official_routes}
    computed_inventories = _recomputed_inventories(ours, provisional)
    raw_inventories = _as_list(solution.get("ending_inventories"))
    official_inventories = {}

    for node in range(n + 1):
        node_inventories = {}
        for t in range(T):
            supplied_row = _as_list(_at(raw_inventories, t, []))
            fallback = computed_inventories[t][node]
            value = (
                _as_float(supplied_row[node], fallback)
                if node < len(supplied_row)
                else fallback
            )
            node_inventories[str(t + 1)] = value
        official_inventories[str(node)] = node_inventories

    official_solution = {
        "routes": official_routes,
        "inventories": official_inventories,
    }
    objective, travel_cost, holding_cost = _objective_components(
        official_instance, official_solution
    )

    return {
        "objective_value": objective,
        "routes": official_routes,
        "inventories": official_inventories,
        "travel_cost": travel_cost,
        "holding_cost": holding_cost,
    }


def official_objective(official_instance: dict, official_solution: dict) -> float:
    objective, travel_cost, holding_cost = _objective_components(
        official_instance, official_solution
    )
    return float(objective)
