from math import isfinite

TOL = 1e-7


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
    )


def _close(a, b):
    return abs(a - b) <= TOL * (1.0 + max(abs(a), abs(b)))


def _lt(a, b):
    return a < b - TOL * (1.0 + max(abs(a), abs(b)))


def _gt(a, b):
    return a > b + TOL * (1.0 + max(abs(a), abs(b)))


def _vector(instance, name, length, errors, nonnegative=False):
    value = instance[name]
    if not isinstance(value, list) or len(value) != length:
        errors.append(f"INSTANCE: {name} must be a list of length {length}")
        return None

    result = []
    valid = True
    for index, item in enumerate(value):
        if not _is_number(item):
            errors.append(f"INSTANCE: {name}[{index}] must be a finite number")
            valid = False
            continue
        number = float(item)
        if nonnegative and _lt(number, 0.0):
            errors.append(f"INSTANCE: {name}[{index}] must be nonnegative")
        result.append(number)
    return result if valid else None


def _matrix(instance, name, rows, columns, errors, nonnegative=False):
    value = instance[name]
    if not isinstance(value, list) or len(value) != rows:
        errors.append(f"INSTANCE: {name} must have {rows} rows")
        return None

    result = []
    valid = True
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            errors.append(
                f"INSTANCE: {name}[{row_index}] must be a list of length {columns}"
            )
            valid = False
            continue
        converted_row = []
        for column_index, item in enumerate(row):
            if not _is_number(item):
                errors.append(
                    f"INSTANCE: {name}[{row_index}][{column_index}] "
                    "must be a finite number"
                )
                valid = False
                continue
            number = float(item)
            if nonnegative and _lt(number, 0.0):
                errors.append(
                    f"INSTANCE: {name}[{row_index}][{column_index}] "
                    "must be nonnegative"
                )
            converted_row.append(number)
        result.append(converted_row)
    return result if valid else None


def _validate_instance(instance):
    errors = []
    required = [
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
    if not isinstance(instance, dict):
        return ["INSTANCE: instance must be a dictionary"], None

    missing = [name for name in required if name not in instance]
    if missing:
        return [f"INSTANCE: missing field {name}" for name in missing], None

    H = instance["num_periods"]
    N = instance["num_customers"]
    M = instance["num_vehicles"]
    if not _is_int(H) or H <= 0:
        errors.append("INSTANCE: num_periods must be a positive integer")
    if not _is_int(N) or N <= 0:
        errors.append("INSTANCE: num_customers must be a positive integer")
    if not _is_int(M) or M < 0:
        errors.append("INSTANCE: num_vehicles must be a nonnegative integer")
    if errors:
        return errors, None

    Q = instance["vehicle_capacity"]
    if not _is_number(Q) or not _gt(float(Q), 0.0):
        errors.append("INSTANCE: vehicle_capacity must be a positive finite number")

    production = _vector(
        instance, "supplier_production", H, errors, nonnegative=True
    )
    demand = _matrix(
        instance, "customer_demand", N, H, errors, nonnegative=True
    )
    capacity = _vector(
        instance, "inventory_capacity", N + 1, errors, nonnegative=True
    )
    initial = _vector(
        instance, "initial_inventory", N + 1, errors, nonnegative=True
    )
    holding = _vector(
        instance, "holding_cost", N + 1, errors, nonnegative=True
    )
    travel = _matrix(
        instance, "travel_cost", N + 1, N + 1, errors, nonnegative=True
    )

    if capacity is not None and initial is not None:
        for location in range(N + 1):
            if _gt(initial[location], capacity[location]):
                errors.append(
                    f"INSTANCE: initial_inventory[{location}] exceeds "
                    f"inventory_capacity[{location}]"
                )

    if travel is not None:
        for i in range(N + 1):
            if not _close(travel[i][i], 0.0):
                errors.append(f"INSTANCE: travel_cost[{i}][{i}] must be 0")
            for j in range(i + 1, N + 1):
                if not _close(travel[i][j], travel[j][i]):
                    errors.append(
                        f"INSTANCE: travel_cost must be symmetric at ({i},{j})"
                    )

    if errors:
        return errors, None

    return [], {
        "H": H,
        "N": N,
        "M": M,
        "Q": float(Q),
        "production": production,
        "demand": demand,
        "capacity": capacity,
        "initial": initial,
        "holding": holding,
        "travel": travel,
    }


def _validate_solution(data, solution):
    errors = []
    ready = True
    H, N, M = data["H"], data["N"], data["M"]

    if not isinstance(solution, dict):
        return ["STRUCTURE: solution must be a dictionary"], False

    required = ["routes", "deliveries", "end_inventory"]
    missing = [name for name in required if name not in solution]
    if missing:
        return [f"STRUCTURE: missing solution field {name}" for name in missing], False

    routes = solution["routes"]
    if not isinstance(routes, list) or len(routes) != H:
        errors.append(f"STRUCTURE: routes must have {H} period entries")
        ready = False
    else:
        for t, period_routes in enumerate(routes):
            if not isinstance(period_routes, list) or len(period_routes) != M:
                errors.append(
                    f"STRUCTURE: routes[{t}] must contain exactly {M} vehicle routes"
                )
                ready = False
                continue
            for k, route in enumerate(period_routes):
                if not isinstance(route, list):
                    errors.append(f"STRUCTURE: routes[{t}][{k}] must be a list")
                    ready = False
                    continue
                valid_nodes = True
                for position, node in enumerate(route):
                    if not _is_int(node) or not 0 <= node <= N:
                        errors.append(
                            f"STRUCTURE: routes[{t}][{k}][{position}] must be an "
                            f"integer location ID between 0 and {N}"
                        )
                        ready = False
                        valid_nodes = False
                if valid_nodes:
                    for position in range(len(route) - 1):
                        if route[position] == route[position + 1]:
                            errors.append(
                                f"STRUCTURE: routes[{t}][{k}] contains the forbidden "
                                f"self-loop {route[position]}->{route[position + 1]}"
                            )

    deliveries = solution["deliveries"]
    if not isinstance(deliveries, list) or len(deliveries) != H:
        errors.append(f"STRUCTURE: deliveries must have {H} period entries")
        ready = False
    else:
        for t, period_deliveries in enumerate(deliveries):
            if not isinstance(period_deliveries, list) or len(period_deliveries) != M:
                errors.append(
                    f"STRUCTURE: deliveries[{t}] must contain exactly {M} vehicle entries"
                )
                ready = False
                continue
            for k, vehicle_deliveries in enumerate(period_deliveries):
                if not isinstance(vehicle_deliveries, list) or len(vehicle_deliveries) != N:
                    errors.append(
                        f"STRUCTURE: deliveries[{t}][{k}] must contain exactly "
                        f"{N} customer quantities"
                    )
                    ready = False
                    continue
                for customer_index, quantity in enumerate(vehicle_deliveries):
                    if not _is_number(quantity):
                        errors.append(
                            f"STRUCTURE: deliveries[{t}][{k}][{customer_index}] "
                            "must be a finite number"
                        )
                        ready = False

    inventories = solution["end_inventory"]
    if not isinstance(inventories, list) or len(inventories) != H:
        errors.append(f"STRUCTURE: end_inventory must have {H} period entries")
        ready = False
    else:
        for t, period_inventory in enumerate(inventories):
            if not isinstance(period_inventory, list) or len(period_inventory) != N + 1:
                errors.append(
                    f"STRUCTURE: end_inventory[{t}] must contain exactly {N + 1} locations"
                )
                ready = False
                continue
            for location, quantity in enumerate(period_inventory):
                if not _is_number(quantity):
                    errors.append(
                        f"STRUCTURE: end_inventory[{t}][{location}] "
                        "must be a finite number"
                    )
                    ready = False

    return errors, ready


def _reachable(start, adjacency):
    reached = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    return reached


def check(instance: dict, solution: dict) -> list:
    violations, data = _validate_instance(instance)
    if data is None:
        return violations

    structure_errors, ready = _validate_solution(data, solution)
    violations.extend(structure_errors)
    if not ready:
        return violations

    H, N, M, Q = data["H"], data["N"], data["M"], data["Q"]
    routes = solution["routes"]
    deliveries = [
        [[float(q) for q in vehicle] for vehicle in period]
        for period in solution["deliveries"]
    ]
    inventories = [
        [float(q) for q in period] for period in solution["end_inventory"]
    ]

    # C1
    for t in range(H):
        for k in range(M):
            route = routes[t][k]
            legs = list(zip(route, route[1:]))
            for i in range(1, N + 1):
                visited = int(i in route)
                outgoing = sum(1 for origin, _ in legs if origin == i)
                if outgoing != visited:
                    violations.append(
                        f"C1: period {t + 1}, vehicle {k + 1}, customer {i} "
                        f"has {outgoing} outgoing arcs but visit indicator is {visited}"
                    )

    # C2
    for t in range(H):
        for k in range(M):
            route = routes[t][k]
            legs = list(zip(route, route[1:]))
            for i in range(1, N + 1):
                visited = int(i in route)
                incoming = sum(1 for _, destination in legs if destination == i)
                if incoming != visited:
                    violations.append(
                        f"C2: period {t + 1}, vehicle {k + 1}, customer {i} "
                        f"has {incoming} incoming arcs but visit indicator is {visited}"
                    )

    # C3
    for t in range(H):
        for k in range(M):
            route = routes[t][k]
            active = int(bool(route))
            departures = sum(1 for origin, _ in zip(route, route[1:]) if origin == 0)
            if departures != active:
                violations.append(
                    f"C3: period {t + 1}, vehicle {k + 1} has {departures} "
                    f"supplier departures but activity indicator is {active}"
                )

    # C4
    for t in range(H):
        for k in range(M):
            route = routes[t][k]
            active = int(bool(route))
            returns = sum(
                1 for _, destination in zip(route, route[1:]) if destination == 0
            )
            if returns != active:
                violations.append(
                    f"C4: period {t + 1}, vehicle {k + 1} has {returns} "
                    f"supplier returns but activity indicator is {active}"
                )

    # C5
    for t in range(H):
        for i in range(1, N + 1):
            visiting_vehicles = sum(1 for k in range(M) if i in routes[t][k])
            if visiting_vehicles > 1:
                violations.append(
                    f"C5: period {t + 1}, customer {i} is visited by "
                    f"{visiting_vehicles} vehicles"
                )

    # C6
    for t in range(H):
        for k in range(M):
            for i in range(1, N + 1):
                quantity = deliveries[t][k][i - 1]
                visited = int(i in routes[t][k])
                if _lt(quantity, 0.0) or _gt(quantity, Q * visited):
                    violations.append(
                        f"C6: period {t + 1}, vehicle {k + 1}, customer {i} "
                        f"has delivery {quantity}, outside [0, {Q * visited}]"
                    )

    # C7
    for t in range(H):
        for k in range(M):
            active = int(bool(routes[t][k]))
            load = sum(deliveries[t][k])
            if _gt(load, Q * active):
                violations.append(
                    f"C7: period {t + 1}, vehicle {k + 1} has load {load}, "
                    f"exceeding allowed load {Q * active}"
                )

    # C8
    for t in range(H):
        route_count = sum(1 for route in routes[t] if route)
        if route_count > M:
            violations.append(
                f"C8: period {t + 1} uses {route_count} routes but only {M} vehicles exist"
            )

    # C9
    for t in range(H):
        for k in range(M):
            route = routes[t][k]
            if not route:
                continue
            outgoing = {node: set() for node in range(N + 1)}
            incoming = {node: set() for node in range(N + 1)}
            for origin, destination in zip(route, route[1:]):
                outgoing[origin].add(destination)
                incoming[destination].add(origin)
            visited_customers = {node for node in route if node != 0}
            reachable_from_supplier = _reachable(0, outgoing)
            able_to_return = _reachable(0, incoming)
            disconnected = visited_customers - reachable_from_supplier
            cannot_return = visited_customers - able_to_return
            if disconnected or cannot_return:
                violations.append(
                    f"C9: period {t + 1}, vehicle {k + 1} has customers not "
                    f"connected to a supplier tour; unreachable={sorted(disconnected)}, "
                    f"cannot_return={sorted(cannot_return)}"
                )

    # C10
    for t in range(H):
        for i in range(1, N + 1):
            previous = data["initial"][i] if t == 0 else inventories[t - 1][i]
            delivered = sum(deliveries[t][k][i - 1] for k in range(M))
            available_before_demand = previous + delivered
            demand = data["demand"][i - 1][t]
            capacity = data["capacity"][i]
            if _lt(available_before_demand, demand) or _gt(
                available_before_demand, capacity
            ):
                violations.append(
                    f"C10: period {t + 1}, customer {i} has "
                    f"{available_before_demand} before demand, outside "
                    f"[{demand}, {capacity}]"
                )

    # C11
    for t in range(H):
        for i in range(1, N + 1):
            previous = data["initial"][i] if t == 0 else inventories[t - 1][i]
            delivered = sum(deliveries[t][k][i - 1] for k in range(M))
            expected = previous + delivered - data["demand"][i - 1][t]
            actual = inventories[t][i]
            if not _close(actual, expected):
                violations.append(
                    f"C11: period {t + 1}, customer {i} end inventory is "
                    f"{actual}, expected {expected}"
                )

    # C12
    for t in range(H):
        previous = data["initial"][0] if t == 0 else inventories[t - 1][0]
        loaded = sum(
            deliveries[t][k][customer_index]
            for k in range(M)
            for customer_index in range(N)
        )
        expected = previous + data["production"][t] - loaded
        actual = inventories[t][0]
        if not _close(actual, expected):
            violations.append(
                f"C12: period {t + 1} supplier end inventory is {actual}, "
                f"expected {expected}"
            )

    # C13
    for t in range(H):
        inventory = inventories[t][0]
        capacity = data["capacity"][0]
        if _lt(inventory, 0.0) or _gt(inventory, capacity):
            violations.append(
                f"C13: period {t + 1} supplier end inventory {inventory} "
                f"is outside [0, {capacity}]"
            )

    return violations
