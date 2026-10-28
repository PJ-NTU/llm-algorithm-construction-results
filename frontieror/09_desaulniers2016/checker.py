import math

EPS = 1e-7


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _tolerance(*values):
    return EPS * max([1.0] + [abs(float(value)) for value in values])


def _close(left, right):
    return abs(float(left) - float(right)) <= _tolerance(left, right)


def _leq(left, right):
    return float(left) <= float(right) + _tolerance(left, right)


def _route_edges(route):
    if not route:
        return []
    return [(0, route[0])] + list(zip(route, route[1:])) + [(route[-1], 0)]


def check(instance: dict, solution: dict) -> list:
    violations = []

    if not isinstance(instance, dict):
        return ['instance: instance must be a dict']

    required_instance_fields = (
        'num_periods',
        'num_customers',
        'num_vehicles',
        'vehicle_capacity',
        'travel_costs',
        'production',
        'demands',
        'inventory_capacities',
        'initial_inventories',
        'holding_costs',
    )
    missing_instance_fields = [key for key in required_instance_fields if key not in instance]
    if missing_instance_fields:
        return [f'instance: missing field {key}' for key in missing_instance_fields]

    T = instance['num_periods']
    n = instance['num_customers']
    m = instance['num_vehicles']
    Q = instance['vehicle_capacity']

    if not isinstance(T, int) or isinstance(T, bool) or T < 0:
        return ['instance: num_periods must be a nonnegative int']
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        return ['instance: num_customers must be a nonnegative int']
    if not isinstance(m, int) or isinstance(m, bool) or m < 0:
        return ['instance: num_vehicles must be a nonnegative int']
    if not _is_number(Q) or Q < 0:
        return ['instance: vehicle_capacity must be a finite nonnegative number']

    if not isinstance(solution, dict):
        return ['structure: solution must be a dict']

    required_solution_fields = ('routes', 'deliveries', 'ending_inventories')
    missing_solution_fields = [key for key in required_solution_fields if key not in solution]
    if missing_solution_fields:
        return [f'structure: missing solution field {key}' for key in missing_solution_fields]

    routes = solution['routes']
    deliveries = solution['deliveries']
    ending = solution['ending_inventories']
    fatal_structure_error = False

    if not isinstance(routes, list):
        violations.append('structure: routes must be a list')
        fatal_structure_error = True
    else:
        if len(routes) != T:
            violations.append(f'structure: routes must have {T} period entries')
            fatal_structure_error = True
        for t, period_routes in enumerate(routes):
            if not isinstance(period_routes, list):
                violations.append(f'structure: routes[{t}] must be a list')
                fatal_structure_error = True
                continue
            if len(period_routes) != m:
                violations.append(f'structure: routes[{t}] must have {m} vehicle entries')
                fatal_structure_error = True
            for k, route in enumerate(period_routes):
                if not isinstance(route, list):
                    violations.append(f'structure: routes[{t}][{k}] must be a list')
                    fatal_structure_error = True
                    continue
                for position, customer in enumerate(route):
                    if not isinstance(customer, int) or isinstance(customer, bool) or not 1 <= customer <= n:
                        violations.append(
                            f'structure: routes[{t}][{k}][{position}] must be a customer id from 1 to {n}'
                        )
                        fatal_structure_error = True

    if not isinstance(deliveries, list):
        violations.append('structure: deliveries must be a list')
        fatal_structure_error = True
    else:
        if len(deliveries) != T:
            violations.append(f'structure: deliveries must have {T} period entries')
            fatal_structure_error = True
        for t, period_deliveries in enumerate(deliveries):
            if not isinstance(period_deliveries, list):
                violations.append(f'structure: deliveries[{t}] must be a list')
                fatal_structure_error = True
                continue
            if len(period_deliveries) != m:
                violations.append(f'structure: deliveries[{t}] must have {m} vehicle entries')
                fatal_structure_error = True
            for k, vehicle_deliveries in enumerate(period_deliveries):
                if not isinstance(vehicle_deliveries, list):
                    violations.append(f'structure: deliveries[{t}][{k}] must be a list')
                    fatal_structure_error = True
                    continue
                if len(vehicle_deliveries) != n:
                    violations.append(f'structure: deliveries[{t}][{k}] must have {n} customer entries')
                    fatal_structure_error = True
                for customer_index, quantity in enumerate(vehicle_deliveries):
                    if not _is_number(quantity):
                        violations.append(
                            f'structure: deliveries[{t}][{k}][{customer_index}] must be a finite number'
                        )
                        fatal_structure_error = True
                    elif quantity < -_tolerance(quantity):
                        violations.append(
                            f'domain: delivery to customer {customer_index + 1} by vehicle {k + 1} '
                            f'in period {t + 1} is negative ({quantity})'
                        )

    if not isinstance(ending, list):
        violations.append('structure: ending_inventories must be a list')
        fatal_structure_error = True
    else:
        if len(ending) != T:
            violations.append(f'structure: ending_inventories must have {T} period entries')
            fatal_structure_error = True
        for t, period_inventory in enumerate(ending):
            if not isinstance(period_inventory, list):
                violations.append(f'structure: ending_inventories[{t}] must be a list')
                fatal_structure_error = True
                continue
            if len(period_inventory) != n + 1:
                violations.append(
                    f'structure: ending_inventories[{t}] must have {n + 1} location entries'
                )
                fatal_structure_error = True
            for location, inventory in enumerate(period_inventory):
                if not _is_number(inventory):
                    violations.append(
                        f'structure: ending_inventories[{t}][{location}] must be a finite number'
                    )
                    fatal_structure_error = True

    if fatal_structure_error:
        return violations

    # C1
    for t in range(T):
        for k in range(m):
            route = routes[t][k]
            departure_count = sum(1 for origin, _ in _route_edges(route) if origin == 0)
            expected = 1 if route else 0
            if departure_count != expected:
                violations.append(
                    f'C1: vehicle {k + 1} in period {t + 1} has {departure_count} depot departures; '
                    f'expected {expected}'
                )

    # C2
    for t in range(T):
        for k in range(m):
            route = routes[t][k]
            return_count = sum(1 for _, destination in _route_edges(route) if destination == 0)
            expected = 1 if route else 0
            if return_count != expected:
                violations.append(
                    f'C2: vehicle {k + 1} in period {t + 1} has {return_count} depot returns; '
                    f'expected {expected}'
                )

    # C3
    for t in range(T):
        for k in range(m):
            route = routes[t][k]
            edges = _route_edges(route)
            for customer in range(1, n + 1):
                outgoing = sum(1 for origin, _ in edges if origin == customer)
                visited = 1 if customer in route else 0
                if outgoing != visited:
                    violations.append(
                        f'C3: customer {customer}, vehicle {k + 1}, period {t + 1} has '
                        f'{outgoing} outgoing route arcs but visit indicator is {visited}'
                    )

    # C4
    for t in range(T):
        for k in range(m):
            route = routes[t][k]
            edges = _route_edges(route)
            for customer in range(1, n + 1):
                incoming = sum(1 for _, destination in edges if destination == customer)
                visited = 1 if customer in route else 0
                if incoming != visited:
                    violations.append(
                        f'C4: customer {customer}, vehicle {k + 1}, period {t + 1} has '
                        f'{incoming} incoming route arcs but visit indicator is {visited}'
                    )

    # C5
    for t in range(T):
        for customer in range(1, n + 1):
            visiting_vehicles = sum(1 for k in range(m) if customer in routes[t][k])
            if visiting_vehicles > 1:
                violations.append(
                    f'C5: customer {customer} is visited by {visiting_vehicles} vehicles in period {t + 1}'
                )

    # C6
    for t in range(T):
        active_routes = sum(1 for k in range(m) if routes[t][k])
        if active_routes > m:
            violations.append(
                f'C6: period {t + 1} operates {active_routes} routes but only {m} vehicles are available'
            )

    # C7
    for t in range(T):
        for k in range(m):
            route = routes[t][k]
            if len(route) != len(set(route)):
                violations.append(
                    f'C7: vehicle {k + 1} in period {t + 1} repeats a customer, creating a '
                    f'non-simple route/customer-only cycle: {route}'
                )

    # C8
    for t in range(T):
        for k in range(m):
            route = routes[t][k]
            for customer in range(1, n + 1):
                quantity = deliveries[t][k][customer - 1]
                visit_indicator = 1 if customer in route else 0
                limit = Q * visit_indicator
                if not _leq(quantity, limit):
                    violations.append(
                        f'C8: vehicle {k + 1} delivers {quantity} to customer {customer} in '
                        f'period {t + 1}, exceeding visit-linked limit {limit}'
                    )

    # C9
    for t in range(T):
        for k in range(m):
            load = sum(deliveries[t][k])
            active = 1 if routes[t][k] else 0
            limit = Q * active
            if not _leq(load, limit):
                violations.append(
                    f'C9: vehicle {k + 1} carries {load} in period {t + 1}, exceeding limit {limit}'
                )

    # C10
    for t in range(T):
        for customer in range(1, n + 1):
            previous = (
                instance['initial_inventories'][customer]
                if t == 0
                else ending[t - 1][customer]
            )
            delivered = sum(deliveries[t][k][customer - 1] for k in range(m))
            expected = previous + delivered - instance['demands'][customer - 1][t]
            actual = ending[t][customer]
            if not _close(actual, expected):
                violations.append(
                    f'C10: customer {customer} ending inventory in period {t + 1} is {actual}; '
                    f'inventory balance requires {expected}'
                )

    # C11
    for t in range(T):
        previous = instance['initial_inventories'][0] if t == 0 else ending[t - 1][0]
        shipped = sum(deliveries[t][k][customer] for k in range(m) for customer in range(n))
        expected = previous + instance['production'][t] - shipped
        actual = ending[t][0]
        if not _close(actual, expected):
            violations.append(
                f'C11: depot ending inventory in period {t + 1} is {actual}; '
                f'inventory balance requires {expected}'
            )

    # C12
    for t in range(T):
        for customer in range(1, n + 1):
            previous = (
                instance['initial_inventories'][customer]
                if t == 0
                else ending[t - 1][customer]
            )
            delivered = sum(deliveries[t][k][customer - 1] for k in range(m))
            after_receipt = previous + delivered
            capacity = instance['inventory_capacities'][customer]
            if not _leq(after_receipt, capacity):
                violations.append(
                    f'C12: customer {customer} has {after_receipt} units after receipt in '
                    f'period {t + 1}, exceeding capacity {capacity}'
                )

    # C13
    for t in range(T):
        for customer in range(1, n + 1):
            inventory = ending[t][customer]
            if inventory < -_tolerance(inventory):
                violations.append(
                    f'C13: customer {customer} has negative ending inventory {inventory} '
                    f'in period {t + 1}'
                )

    # C14
    depot_capacity = instance['inventory_capacities'][0]
    for t in range(T):
        inventory = ending[t][0]
        if inventory < -_tolerance(inventory):
            violations.append(
                f'C14: depot has negative ending inventory {inventory} in period {t + 1}'
            )
        if not _leq(inventory, depot_capacity):
            violations.append(
                f'C14: depot ending inventory {inventory} in period {t + 1} exceeds '
                f'capacity {depot_capacity}'
            )

    return violations
