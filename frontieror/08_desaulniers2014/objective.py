from math import isfinite


def _number(value, label):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _integer(value, label, minimum):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def objective(instance: dict, solution: dict) -> float:
    if not isinstance(instance, dict):
        raise ValueError("instance must be a dictionary")
    if not isinstance(solution, dict):
        raise ValueError("solution must be a dictionary")

    required_instance = [
        "num_periods",
        "num_customers",
        "num_vehicles",
        "vehicle_capacity",
        "supplier_production",
        "customer_demand",
        "inventory_capacity",
        "initial_inventory",
        "holding_cost",
        "travel_cost",
    ]
    for name in required_instance:
        if name not in instance:
            raise ValueError(f"missing instance field {name}")
    for name in ("routes", "deliveries", "end_inventory"):
        if name not in solution:
            raise ValueError(f"missing solution field {name}")

    H = _integer(instance["num_periods"], "num_periods", 1)
    N = _integer(instance["num_customers"], "num_customers", 1)
    M = _integer(instance["num_vehicles"], "num_vehicles", 0)

    travel = instance["travel_cost"]
    if not isinstance(travel, list) or len(travel) != N + 1:
        raise ValueError(f"travel_cost must have {N + 1} rows")
    for i, row in enumerate(travel):
        if not isinstance(row, list) or len(row) != N + 1:
            raise ValueError(f"travel_cost[{i}] must have length {N + 1}")
        for j, value in enumerate(row):
            _number(value, f"travel_cost[{i}][{j}]")

    holding = instance["holding_cost"]
    if not isinstance(holding, list) or len(holding) != N + 1:
        raise ValueError(f"holding_cost must have length {N + 1}")
    holding = [
        _number(value, f"holding_cost[{location}]")
        for location, value in enumerate(holding)
    ]

    routes = solution["routes"]
    if not isinstance(routes, list) or len(routes) != H:
        raise ValueError(f"routes must have {H} period entries")
    for t, period_routes in enumerate(routes):
        if not isinstance(period_routes, list) or len(period_routes) != M:
            raise ValueError(f"routes[{t}] must have {M} vehicle entries")
        for k, route in enumerate(period_routes):
            if not isinstance(route, list):
                raise ValueError(f"routes[{t}][{k}] must be a list")
            for position, node in enumerate(route):
                if (
                    not isinstance(node, int)
                    or isinstance(node, bool)
                    or not 0 <= node <= N
                ):
                    raise ValueError(
                        f"routes[{t}][{k}][{position}] must be a location ID "
                        f"between 0 and {N}"
                    )

    deliveries = solution["deliveries"]
    if not isinstance(deliveries, list) or len(deliveries) != H:
        raise ValueError(f"deliveries must have {H} period entries")
    for t, period_deliveries in enumerate(deliveries):
        if not isinstance(period_deliveries, list) or len(period_deliveries) != M:
            raise ValueError(f"deliveries[{t}] must have {M} vehicle entries")
        for k, vehicle_deliveries in enumerate(period_deliveries):
            if not isinstance(vehicle_deliveries, list) or len(vehicle_deliveries) != N:
                raise ValueError(f"deliveries[{t}][{k}] must have length {N}")
            for i, value in enumerate(vehicle_deliveries):
                _number(value, f"deliveries[{t}][{k}][{i}]")

    inventories = solution["end_inventory"]
    if not isinstance(inventories, list) or len(inventories) != H:
        raise ValueError(f"end_inventory must have {H} period entries")
    for t, period_inventory in enumerate(inventories):
        if not isinstance(period_inventory, list) or len(period_inventory) != N + 1:
            raise ValueError(f"end_inventory[{t}] must have length {N + 1}")

    total = 0.0
    for t in range(H):
        for k in range(M):
            route = routes[t][k]
            for leg, (origin, destination) in enumerate(zip(route, route[1:])):
                total += _number(
                    travel[origin][destination],
                    f"travel_cost for routes[{t}][{k}] leg {leg}",
                )

        for location in range(N + 1):
            inventory = _number(
                inventories[t][location],
                f"end_inventory[{t}][{location}]",
            )
            total += holding[location] * inventory

    return float(total)
