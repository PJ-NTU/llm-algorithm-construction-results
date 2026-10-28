import math

TOL = 1e-6


def _is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _close(left, right):
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return abs(float(left) - float(right)) <= TOL * scale


def _leq(left, right):
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return float(left) <= float(right) + TOL * scale


def _check_array(value, shape, path, errors, nonnegative=True):
    if not shape:
        if not _is_number(value):
            errors.append(f"STRUCTURE: {path} must be a finite number")
            return False
        if nonnegative and float(value) < -TOL:
            errors.append(f"STRUCTURE: {path} must be nonnegative")
            return False
        return True

    if not isinstance(value, list):
        errors.append(f"STRUCTURE: {path} must be a list with shape {shape}")
        return False
    if len(value) != shape[0]:
        errors.append(
            f"STRUCTURE: {path} has length {len(value)}, expected {shape[0]}"
        )
        return False

    valid = True
    for index, item in enumerate(value):
        if not _check_array(
            item, shape[1:], f"{path}[{index}]", errors, nonnegative
        ):
            valid = False
    return valid


def _check_scalar(instance, name, errors, strictly_positive=False):
    value = instance[name]
    if not _is_number(value):
        errors.append(f"STRUCTURE: instance.{name} must be a finite number")
        return False
    if strictly_positive:
        if float(value) <= 0.0:
            errors.append(f"STRUCTURE: instance.{name} must be positive")
            return False
    elif float(value) < 0.0:
        errors.append(f"STRUCTURE: instance.{name} must be nonnegative")
        return False
    return True


def _validate_structure(instance, solution):
    errors = []
    if not isinstance(instance, dict):
        return ["STRUCTURE: instance must be a dict"]
    if not isinstance(solution, dict):
        return ["STRUCTURE: solution must be a dict"]

    dimension_names = (
        "num_periods",
        "num_customers",
        "num_vehicles",
        "num_scenarios",
    )
    for name in dimension_names:
        if name not in instance:
            errors.append(f"STRUCTURE: missing instance field {name}")
        elif type(instance[name]) is not int or instance[name] < 1:
            errors.append(f"STRUCTURE: instance.{name} must be a positive int")
    if errors:
        return errors

    h = instance["num_periods"]
    n = instance["num_customers"]
    m = instance["num_vehicles"]
    s_count = instance["num_scenarios"]

    required_instance = (
        "scenario_probabilities",
        "demand_scenarios",
        "initial_inventories",
        "inventory_capacities",
        "production_capacity",
        "vehicle_capacity",
        "setup_cost",
        "unit_production_cost",
        "holding_costs",
        "unmet_demand_penalties",
        "transportation_costs",
    )
    required_solution = (
        "production_setup",
        "vehicle_routes",
        "production_quantities",
        "delivery_quantities",
        "end_inventories",
        "unmet_demand_quantities",
    )
    for name in required_instance:
        if name not in instance:
            errors.append(f"STRUCTURE: missing instance field {name}")
    for name in required_solution:
        if name not in solution:
            errors.append(f"STRUCTURE: missing solution field {name}")
    if errors:
        return errors

    probabilities_ok = _check_array(
        instance["scenario_probabilities"],
        (s_count,),
        "instance.scenario_probabilities",
        errors,
    )
    if probabilities_ok:
        for scenario, probability in enumerate(instance["scenario_probabilities"]):
            if float(probability) <= 0.0:
                errors.append(
                    "STRUCTURE: scenario_probabilities"
                    f"[{scenario}] must be positive"
                )
        if not _close(sum(instance["scenario_probabilities"]), 1.0):
            errors.append("STRUCTURE: scenario probabilities must sum to 1")

    _check_array(
        instance["demand_scenarios"],
        (s_count, h, n),
        "instance.demand_scenarios",
        errors,
    )
    initial_ok = _check_array(
        instance["initial_inventories"],
        (n + 1,),
        "instance.initial_inventories",
        errors,
    )
    capacities_ok = _check_array(
        instance["inventory_capacities"],
        (n + 1,),
        "instance.inventory_capacities",
        errors,
    )
    if initial_ok and capacities_ok:
        for location in range(n + 1):
            if not _leq(
                instance["initial_inventories"][location],
                instance["inventory_capacities"][location],
            ):
                errors.append(
                    "STRUCTURE: initial inventory exceeds capacity at location "
                    f"{location}"
                )

    _check_scalar(instance, "production_capacity", errors)
    _check_scalar(instance, "vehicle_capacity", errors, strictly_positive=True)
    _check_scalar(instance, "setup_cost", errors)
    _check_scalar(instance, "unit_production_cost", errors)
    _check_array(
        instance["holding_costs"],
        (n + 1,),
        "instance.holding_costs",
        errors,
    )
    _check_array(
        instance["unmet_demand_penalties"],
        (n,),
        "instance.unmet_demand_penalties",
        errors,
    )
    transportation_ok = _check_array(
        instance["transportation_costs"],
        (n + 1, n + 1),
        "instance.transportation_costs",
        errors,
    )
    if transportation_ok:
        costs = instance["transportation_costs"]
        for i in range(n + 1):
            if not _close(costs[i][i], 0.0):
                errors.append(
                    f"STRUCTURE: transportation_costs[{i}][{i}] must be zero"
                )
            for j in range(i + 1, n + 1):
                if not _close(costs[i][j], costs[j][i]):
                    errors.append(
                        "STRUCTURE: transportation costs must be symmetric at "
                        f"locations {i} and {j}"
                    )

    setup = solution["production_setup"]
    if not isinstance(setup, list) or len(setup) != h:
        errors.append(f"STRUCTURE: production_setup must have shape [{h}]")
    else:
        for period, value in enumerate(setup):
            if type(value) is not int or value not in (0, 1):
                errors.append(
                    f"STRUCTURE: production_setup[{period}] must be 0 or 1"
                )

    routes = solution["vehicle_routes"]
    if not isinstance(routes, list) or len(routes) != h:
        errors.append(f"STRUCTURE: vehicle_routes must have outer length {h}")
    else:
        for period, period_routes in enumerate(routes):
            if not isinstance(period_routes, list) or len(period_routes) != m:
                errors.append(
                    "STRUCTURE: vehicle_routes"
                    f"[{period}] must contain exactly {m} vehicle routes"
                )
                continue
            for vehicle, route in enumerate(period_routes):
                path = f"vehicle_routes[{period}][{vehicle}]"
                if not isinstance(route, list) or len(route) == 0:
                    errors.append(f"STRUCTURE: {path} must be a nonempty list")
                    continue
                for position, node in enumerate(route):
                    if type(node) is not int or not 0 <= node <= n:
                        errors.append(
                            f"STRUCTURE: {path}[{position}] must be an integer "
                            f"node in 0..{n}"
                        )

    _check_array(
        solution["production_quantities"],
        (s_count, h),
        "solution.production_quantities",
        errors,
    )
    _check_array(
        solution["delivery_quantities"],
        (s_count, h, m, n),
        "solution.delivery_quantities",
        errors,
    )
    _check_array(
        solution["end_inventories"],
        (s_count, h, n + 1),
        "solution.end_inventories",
        errors,
    )
    _check_array(
        solution["unmet_demand_quantities"],
        (s_count, h, n),
        "solution.unmet_demand_quantities",
        errors,
    )
    return errors


def _same_history(demands, first, second, period, num_customers):
    for earlier_period in range(period + 1):
        for customer in range(num_customers):
            if (
                demands[first][earlier_period][customer]
                != demands[second][earlier_period][customer]
            ):
                return False
    return True


def check(instance: dict, solution: dict) -> list:
    violations = _validate_structure(instance, solution)
    if violations:
        return violations

    h = instance["num_periods"]
    n = instance["num_customers"]
    m = instance["num_vehicles"]
    s_count = instance["num_scenarios"]
    demands = instance["demand_scenarios"]
    initial = instance["initial_inventories"]
    capacities = instance["inventory_capacities"]
    production_capacity = instance["production_capacity"]
    vehicle_capacity = instance["vehicle_capacity"]

    setup = solution["production_setup"]
    routes = solution["vehicle_routes"]
    production = solution["production_quantities"]
    deliveries = solution["delivery_quantities"]
    inventories = solution["end_inventories"]
    unmet = solution["unmet_demand_quantities"]

    # C1
    for scenario in range(s_count):
        for period in range(h):
            upper = production_capacity * setup[period]
            if not _leq(production[scenario][period], upper):
                violations.append(
                    f"C1: scenario {scenario}, period {period}: production "
                    f"{production[scenario][period]} exceeds setup-linked "
                    f"capacity {upper}"
                )

    # C2
    for scenario in range(s_count):
        for period in range(h):
            previous = (
                initial[0]
                if period == 0
                else inventories[scenario][period - 1][0]
            )
            shipped = sum(
                deliveries[scenario][period][vehicle][customer]
                for vehicle in range(m)
                for customer in range(n)
            )
            left = previous + production[scenario][period]
            right = shipped + inventories[scenario][period][0]
            if not _close(left, right):
                violations.append(
                    f"C2: scenario {scenario}, period {period}: plant balance "
                    f"has {left} available but {right} shipped plus retained"
                )

    # C3
    for scenario in range(s_count):
        for period in range(h):
            value = inventories[scenario][period][0]
            if not _leq(value, capacities[0]):
                violations.append(
                    f"C3: scenario {scenario}, period {period}: plant ending "
                    f"inventory {value} exceeds capacity {capacities[0]}"
                )

    # C4
    for scenario in range(s_count):
        for period in range(h):
            for customer in range(n):
                location = customer + 1
                previous = (
                    initial[location]
                    if period == 0
                    else inventories[scenario][period - 1][location]
                )
                delivered = sum(
                    deliveries[scenario][period][vehicle][customer]
                    for vehicle in range(m)
                )
                left = previous + delivered + unmet[scenario][period][customer]
                right = (
                    demands[scenario][period][customer]
                    + inventories[scenario][period][location]
                )
                if not _close(left, right):
                    violations.append(
                        f"C4: scenario {scenario}, period {period}, customer "
                        f"{location}: balance left side {left} differs from "
                        f"demand plus ending inventory {right}"
                    )

    # C5
    for scenario in range(s_count):
        for period in range(h):
            for customer in range(n):
                value = unmet[scenario][period][customer]
                demand = demands[scenario][period][customer]
                if not _leq(value, demand):
                    violations.append(
                        f"C5: scenario {scenario}, period {period}, customer "
                        f"{customer + 1}: unmet demand {value} exceeds demand "
                        f"{demand}"
                    )

    # C6
    for scenario in range(s_count):
        for period in range(h):
            for customer in range(n):
                location = customer + 1
                used = (
                    inventories[scenario][period][location]
                    + demands[scenario][period][customer]
                )
                if not _leq(used, capacities[location]):
                    violations.append(
                        f"C6: scenario {scenario}, period {period}, customer "
                        f"{location}: ending inventory plus demand {used} "
                        f"exceeds capacity {capacities[location]}"
                    )

    # C7
    for scenario in range(s_count):
        for period in range(h):
            for vehicle in range(m):
                load = sum(deliveries[scenario][period][vehicle])
                dispatched = 0 if routes[period][vehicle] == [0] else 1
                upper = vehicle_capacity * dispatched
                if not _leq(load, upper):
                    violations.append(
                        f"C7: scenario {scenario}, period {period}, vehicle "
                        f"{vehicle}: load {load} exceeds dispatched capacity "
                        f"{upper}"
                    )

    # C8
    for scenario in range(s_count):
        for period in range(h):
            for vehicle in range(m):
                route = routes[period][vehicle]
                for customer in range(n):
                    customer_node = customer + 1
                    visited = 1 if customer_node in route else 0
                    quantity = deliveries[scenario][period][vehicle][customer]
                    upper = vehicle_capacity * visited
                    if not _leq(quantity, upper):
                        violations.append(
                            f"C8: scenario {scenario}, period {period}, vehicle "
                            f"{vehicle}, customer {customer_node}: delivery "
                            f"{quantity} is positive without a route visit"
                        )

    # C9
    for scenario in range(s_count):
        for period in range(h):
            for customer in range(n):
                location = customer + 1
                previous = (
                    initial[location]
                    if period == 0
                    else inventories[scenario][period - 1][location]
                )
                delivered = sum(
                    deliveries[scenario][period][vehicle][customer]
                    for vehicle in range(m)
                )
                remaining_space = capacities[location] - previous
                if not _leq(delivered, remaining_space):
                    violations.append(
                        f"C9: scenario {scenario}, period {period}, customer "
                        f"{location}: delivery {delivered} exceeds remaining "
                        f"storage space {remaining_space}"
                    )

    # C10
    for scenario in range(s_count):
        for period in range(h):
            for customer in range(n):
                delivered = sum(
                    deliveries[scenario][period][vehicle][customer]
                    for vehicle in range(m)
                )
                future_demand = sum(
                    demands[scenario][future_period][customer]
                    for future_period in range(period, h)
                )
                if not _leq(delivered, future_demand):
                    violations.append(
                        f"C10: scenario {scenario}, period {period}, customer "
                        f"{customer + 1}: delivery {delivered} exceeds remaining "
                        f"horizon demand {future_demand}"
                    )

    # C11
    for period in range(h):
        for customer_node in range(1, n + 1):
            visiting_vehicles = sum(
                1
                for vehicle in range(m)
                if customer_node in routes[period][vehicle]
            )
            if visiting_vehicles > 1:
                violations.append(
                    f"C11: period {period}, customer {customer_node}: visited "
                    f"by {visiting_vehicles} vehicles"
                )

    # C12
    for period in range(h):
        for vehicle in range(m):
            route = routes[period][vehicle]
            arcs = list(zip(route, route[1:]))
            for customer_node in range(1, n + 1):
                visited = 1 if customer_node in route else 0
                incoming = sum(1 for _, end in arcs if end == customer_node)
                outgoing = sum(1 for start, _ in arcs if start == customer_node)
                if incoming != visited or outgoing != visited:
                    violations.append(
                        f"C12: period {period}, vehicle {vehicle}, customer "
                        f"{customer_node}: visit={visited}, incoming={incoming}, "
                        f"outgoing={outgoing}"
                    )

    # C13
    for period in range(h):
        for vehicle in range(m):
            route = routes[period][vehicle]
            if route == [0]:
                continue
            arcs = list(zip(route, route[1:]))
            departures = sum(
                1 for start, end in arcs if start == 0 and end != 0
            )
            returns = sum(1 for start, end in arcs if start != 0 and end == 0)
            canonical = (
                len(route) >= 3
                and route[0] == 0
                and route[-1] == 0
                and all(node != 0 for node in route[1:-1])
            )
            if not canonical or departures != 1 or returns != 1:
                violations.append(
                    f"C13: period {period}, vehicle {vehicle}: dispatched route "
                    "must start once at depot, end once at depot, and contain no "
                    f"intermediate depot; got {route}"
                )

    # C14
    for period in range(h):
        for vehicle in range(m):
            route = routes[period][vehicle]
            if route == [0]:
                continue
            customers = [node for node in route if node != 0]
            repeated_customer = len(customers) != len(set(customers))
            self_arc = any(
                start == end for start, end in zip(route, route[1:])
            )
            if repeated_customer or self_arc:
                violations.append(
                    f"C14: period {period}, vehicle {vehicle}: route {route} "
                    "is not a simple single tour and cannot satisfy the MTZ "
                    "ordering constraints"
                )

    # C15
    for first in range(s_count):
        for second in range(first + 1, s_count):
            for period in range(h):
                if not _same_history(demands, first, second, period, n):
                    continue
                if not _close(
                    production[first][period], production[second][period]
                ):
                    violations.append(
                        f"C15: scenarios {first} and {second}, period {period}: "
                        "production differs despite identical observed history"
                    )
                if any(
                    not _close(
                        deliveries[first][period][vehicle][customer],
                        deliveries[second][period][vehicle][customer],
                    )
                    for vehicle in range(m)
                    for customer in range(n)
                ):
                    violations.append(
                        f"C15: scenarios {first} and {second}, period {period}: "
                        "deliveries differ despite identical observed history"
                    )
                if any(
                    not _close(
                        inventories[first][period][location],
                        inventories[second][period][location],
                    )
                    for location in range(n + 1)
                ):
                    violations.append(
                        f"C15: scenarios {first} and {second}, period {period}: "
                        "ending inventories differ despite identical observed "
                        "history"
                    )
                if any(
                    not _close(
                        unmet[first][period][customer],
                        unmet[second][period][customer],
                    )
                    for customer in range(n)
                ):
                    violations.append(
                        f"C15: scenarios {first} and {second}, period {period}: "
                        "unmet demand differs despite identical observed history"
                    )

    return violations
