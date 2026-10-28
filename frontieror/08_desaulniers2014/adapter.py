def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, (list, tuple)) else []


def _to_float(value, default=0.0):
    try:
        result = float(value)
        return default if result != result else result
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _item(values, index, default=None):
    values = _as_list(values)
    if 0 <= index < len(values):
        return values[index]
    return default


def _dimensions(official_instance):
    data = _as_dict(official_instance)
    customers = _as_list(data.get("customers"))

    num_customers = _to_int(data.get("num_customers"), -1)
    if num_customers < 0:
        num_customers = len(customers)
        for position, raw_customer in enumerate(customers):
            customer = _as_dict(raw_customer)
            customer_id = _to_int(customer.get("id"), position + 1)
            num_customers = max(num_customers, customer_id)

    num_periods = max(0, _to_int(data.get("num_periods"), 0))
    num_vehicles = max(0, _to_int(data.get("num_vehicles"), 0))
    return max(0, num_customers), num_periods, num_vehicles


def to_ours(official_instance: dict) -> dict:
    data = _as_dict(official_instance)
    num_customers, num_periods, num_vehicles = _dimensions(data)
    depot = _as_dict(data.get("depot"))

    customers_by_id = {}
    for position, raw_customer in enumerate(_as_list(data.get("customers"))):
        customer = _as_dict(raw_customer)
        customer_id = _to_int(customer.get("id"), position + 1)
        if 1 <= customer_id <= num_customers:
            customers_by_id[customer_id] = customer

    production = _to_float(depot.get("production_per_period"), 0.0)
    supplier_production = [production for _ in range(num_periods)]

    customer_demand = []
    for customer_id in range(1, num_customers + 1):
        customer = customers_by_id.get(customer_id, {})
        demand = _to_float(customer.get("demand_per_period"), 0.0)
        customer_demand.append([demand for _ in range(num_periods)])

    inventory_capacity = [_to_float(depot.get("max_inventory"), 0.0)]
    initial_inventory = [_to_float(depot.get("initial_inventory"), 0.0)]
    holding_cost = [_to_float(depot.get("holding_cost"), 0.0)]
    for customer_id in range(1, num_customers + 1):
        customer = customers_by_id.get(customer_id, {})
        inventory_capacity.append(_to_float(customer.get("max_inventory"), 0.0))
        initial_inventory.append(_to_float(customer.get("initial_inventory"), 0.0))
        holding_cost.append(_to_float(customer.get("holding_cost"), 0.0))

    raw_distance = _as_list(data.get("distance_matrix"))
    travel_cost = []
    for origin in range(num_customers + 1):
        raw_row = _as_list(_item(raw_distance, origin, []))
        travel_cost.append([
            _to_float(_item(raw_row, destination, 0.0), 0.0)
            for destination in range(num_customers + 1)
        ])

    return {
        "num_periods": num_periods,
        "num_customers": num_customers,
        "num_vehicles": num_vehicles,
        "vehicle_capacity": _to_float(data.get("vehicle_capacity"), 0.0),
        "supplier_production": supplier_production,
        "customer_demand": customer_demand,
        "inventory_capacity": inventory_capacity,
        "initial_inventory": initial_inventory,
        "holding_cost": holding_cost,
        "travel_cost": travel_cost,
    }


def to_official(official_instance: dict, our_solution: dict) -> dict:
    instance = to_ours(official_instance)
    solution = _as_dict(our_solution)
    num_periods = instance["num_periods"]
    num_customers = instance["num_customers"]
    num_vehicles = instance["num_vehicles"]

    raw_routes = _as_list(solution.get("routes"))
    raw_deliveries = _as_list(solution.get("deliveries"))
    routes_by_period = []
    deliveries_by_period = []

    for period_index in range(num_periods):
        route_slots = _as_list(_item(raw_routes, period_index, []))
        delivery_slots = _as_list(_item(raw_deliveries, period_index, []))
        official_routes = []
        aggregate = [0.0 for _ in range(num_customers)]

        for vehicle_index in range(num_vehicles):
            raw_route = _as_list(_item(route_slots, vehicle_index, []))
            route = [_to_int(node, 0) for node in raw_route]
            if route:
                official_routes.append(route)

            vehicle_deliveries = _as_list(
                _item(delivery_slots, vehicle_index, [])
            )
            for customer_index in range(num_customers):
                aggregate[customer_index] += _to_float(
                    _item(vehicle_deliveries, customer_index, 0.0), 0.0
                )

        routes_by_period.append(official_routes)
        deliveries_by_period.append(aggregate)

    simulated_inventory = []
    previous = list(instance["initial_inventory"])
    for period_index in range(num_periods):
        current = [0.0 for _ in range(num_customers + 1)]
        aggregate = deliveries_by_period[period_index]
        current[0] = (
            previous[0]
            + instance["supplier_production"][period_index]
            - sum(aggregate)
        )
        for customer_index in range(num_customers):
            current[customer_index + 1] = (
                previous[customer_index + 1]
                + aggregate[customer_index]
                - instance["customer_demand"][customer_index][period_index]
            )
        simulated_inventory.append(current)
        previous = current

    raw_end_inventory = _as_list(solution.get("end_inventory"))
    periods = {}
    for period_index in range(num_periods):
        provided_row = _as_list(_item(raw_end_inventory, period_index, []))
        inventories = {}
        depot_default = simulated_inventory[period_index][0]
        inventories["depot"] = _to_float(
            _item(provided_row, 0, depot_default), depot_default
        )
        for customer_id in range(1, num_customers + 1):
            default_value = simulated_inventory[period_index][customer_id]
            inventories[str(customer_id)] = _to_float(
                _item(provided_row, customer_id, default_value), default_value
            )

        delivery_dict = {}
        for customer_id, quantity in enumerate(
            deliveries_by_period[period_index], start=1
        ):
            if quantity != 0.0:
                delivery_dict[str(customer_id)] = quantity

        periods[str(period_index + 1)] = {
            "routes": routes_by_period[period_index],
            "deliveries": delivery_dict,
            "inventories": inventories,
        }

    official_solution = {"solution_details": {"periods": periods}}
    official_solution["objective_value"] = official_objective(
        official_instance, official_solution
    )
    return official_solution


def official_objective(official_instance: dict, official_solution: dict) -> float:
    instance = to_ours(official_instance)
    solution = _as_dict(official_solution)
    details = _as_dict(solution.get("solution_details"))
    num_periods = instance["num_periods"]
    num_customers = instance["num_customers"]

    routes_by_period = [[] for _ in range(num_periods)]
    deliveries_by_period = [
        [0.0 for _ in range(num_customers)]
        for _ in range(num_periods)
    ]

    if "routes" in details:
        for raw_route_data in _as_list(details.get("routes")):
            route_data = _as_dict(raw_route_data)
            period = _to_int(route_data.get("period"), 0)
            if not 1 <= period <= num_periods:
                continue

            customers = [
                _to_int(customer_id, 0)
                for customer_id in _as_list(route_data.get("route"))
            ]
            routes_by_period[period - 1].append([0] + customers + [0])

            for raw_id, raw_quantity in _as_dict(
                route_data.get("deliveries")
            ).items():
                customer_id = _to_int(raw_id, 0)
                if not 1 <= customer_id <= num_customers:
                    continue
                if isinstance(raw_quantity, dict):
                    quantity = sum(
                        _to_float(value, 0.0)
                        for value in raw_quantity.values()
                    )
                else:
                    quantity = _to_float(raw_quantity, 0.0)
                deliveries_by_period[period - 1][customer_id - 1] += quantity

    elif "periods" in details:
        periods = _as_dict(details.get("periods"))
        for period in range(1, num_periods + 1):
            period_data = _as_dict(
                periods.get(str(period), periods.get(period, {}))
            )
            for raw_route in _as_list(period_data.get("routes")):
                routes_by_period[period - 1].append([
                    _to_int(node, 0) for node in _as_list(raw_route)
                ])

            for raw_id, raw_quantity in _as_dict(
                period_data.get("deliveries")
            ).items():
                customer_id = _to_int(raw_id, 0)
                if 1 <= customer_id <= num_customers:
                    deliveries_by_period[period - 1][customer_id - 1] = (
                        _to_float(raw_quantity, 0.0)
                    )

    travel_cost = 0.0
    matrix = instance["travel_cost"]
    for period_routes in routes_by_period:
        for route in period_routes:
            for arc_index in range(len(route) - 1):
                origin = route[arc_index]
                destination = route[arc_index + 1]
                if (
                    0 <= origin <= num_customers
                    and 0 <= destination <= num_customers
                ):
                    travel_cost += matrix[origin][destination]

    holding_cost = 0.0
    previous = list(instance["initial_inventory"])
    for period_index in range(num_periods):
        aggregate = deliveries_by_period[period_index]
        current = [0.0 for _ in range(num_customers + 1)]
        current[0] = (
            previous[0]
            + instance["supplier_production"][period_index]
            - sum(aggregate)
        )
        holding_cost += instance["holding_cost"][0] * current[0]

        for customer_index in range(num_customers):
            location = customer_index + 1
            current[location] = (
                previous[location]
                + aggregate[customer_index]
                - instance["customer_demand"][customer_index][period_index]
            )
            holding_cost += (
                instance["holding_cost"][location] * current[location]
            )
        previous = current

    return float(travel_cost + holding_cost)
