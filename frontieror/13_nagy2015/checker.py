import math
from collections import Counter


_EPS = 1e-9


def _is_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _close(left, right):
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= _EPS * scale


def _over_capacity(load, capacity):
    return load > capacity + _EPS * max(1.0, abs(capacity))


def check(instance: dict, solution: dict) -> list:
    violations = []

    # Validate the instance contract before using its indices or numeric values.
    if not isinstance(instance, dict):
        return ["STRUCTURE: instance must be a dict"]

    required_instance_fields = {
        "n",
        "capacity",
        "delivery_demands",
        "pickup_demands",
        "distance_matrix",
    }
    missing = sorted(required_instance_fields - set(instance))
    if missing:
        return ["STRUCTURE: instance is missing fields: " + ", ".join(missing)]

    n = instance["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        return ["STRUCTURE: n must be a positive int"]

    if not _is_number(instance["capacity"]) or float(instance["capacity"]) <= 0:
        violations.append("STRUCTURE: capacity must be a finite positive number")
    capacity = float(instance["capacity"]) if _is_number(instance["capacity"]) else 0.0

    raw_delivery = instance["delivery_demands"]
    raw_pickup = instance["pickup_demands"]
    if not isinstance(raw_delivery, list) or len(raw_delivery) != n:
        violations.append("STRUCTURE: delivery_demands must be a list of length n")
    if not isinstance(raw_pickup, list) or len(raw_pickup) != n:
        violations.append("STRUCTURE: pickup_demands must be a list of length n")

    demands_valid = (
        isinstance(raw_delivery, list)
        and len(raw_delivery) == n
        and isinstance(raw_pickup, list)
        and len(raw_pickup) == n
    )
    if demands_valid:
        for index, value in enumerate(raw_delivery):
            if not _is_number(value) or float(value) < 0:
                violations.append(
                    f"STRUCTURE: delivery_demands[{index}] must be a finite nonnegative number"
                )
                demands_valid = False
        for index, value in enumerate(raw_pickup):
            if not _is_number(value) or float(value) < 0:
                violations.append(
                    f"STRUCTURE: pickup_demands[{index}] must be a finite nonnegative number"
                )
                demands_valid = False

    matrix = instance["distance_matrix"]
    matrix_shape_valid = (
        isinstance(matrix, list)
        and len(matrix) == n + 1
        and all(isinstance(row, list) and len(row) == n + 1 for row in matrix)
    )
    if not matrix_shape_valid:
        violations.append(
            "STRUCTURE: distance_matrix must have shape (n+1) x (n+1)"
        )
    else:
        matrix_values_valid = True
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    violations.append(
                        f"STRUCTURE: distance_matrix[{i}][{j}] must be a nonnegative int"
                    )
                    matrix_values_valid = False
        if matrix_values_valid:
            for i in range(n + 1):
                if matrix[i][i] != 0:
                    violations.append(
                        f"STRUCTURE: distance_matrix[{i}][{i}] must be zero"
                    )
                for j in range(i + 1, n + 1):
                    if matrix[i][j] != matrix[j][i]:
                        violations.append(
                            f"STRUCTURE: distance_matrix is not symmetric at ({i}, {j})"
                        )

    if violations or not demands_valid:
        return violations

    delivery = [float(value) for value in raw_delivery]
    pickup = [float(value) for value in raw_pickup]

    # Validate and normalize the solution contract.
    if not isinstance(solution, dict):
        return ["STRUCTURE: solution must be a dict"]
    if "routes" not in solution:
        return ["STRUCTURE: solution is missing field routes"]
    if not isinstance(solution["routes"], list):
        return ["STRUCTURE: solution['routes'] must be a list"]

    raw_routes = solution["routes"]
    if len(raw_routes) > 2 * n:
        violations.append(
            f"STRUCTURE: routes contains {len(raw_routes)} routes, exceeding the 2n candidate-route bound"
        )

    routes = []
    fatal_structure_error = False
    required_visit_fields = {"customer_id", "serve_delivery", "serve_pickup"}

    for route_index, raw_route in enumerate(raw_routes):
        if not isinstance(raw_route, list):
            violations.append(
                f"STRUCTURE: routes[{route_index}] must be a list of visits"
            )
            fatal_structure_error = True
            routes.append([])
            continue
        if not raw_route:
            violations.append(
                f"STRUCTURE: routes[{route_index}] must be non-empty because it represents a used vehicle"
            )

        route = []
        for visit_index, visit in enumerate(raw_route):
            location = f"routes[{route_index}][{visit_index}]"
            if not isinstance(visit, dict):
                violations.append(f"STRUCTURE: {location} must be a dict")
                fatal_structure_error = True
                continue

            missing_visit_fields = required_visit_fields - set(visit)
            if missing_visit_fields:
                violations.append(
                    f"STRUCTURE: {location} is missing fields: "
                    + ", ".join(sorted(missing_visit_fields))
                )
                fatal_structure_error = True
                continue

            extra_visit_fields = set(visit) - required_visit_fields
            if extra_visit_fields:
                violations.append(
                    f"STRUCTURE: {location} has unexpected fields: "
                    + ", ".join(sorted(extra_visit_fields))
                )

            customer_id = visit["customer_id"]
            serve_delivery = visit["serve_delivery"]
            serve_pickup = visit["serve_pickup"]

            if (
                isinstance(customer_id, bool)
                or not isinstance(customer_id, int)
                or not 1 <= customer_id <= n
            ):
                violations.append(
                    f"STRUCTURE: {location}.customer_id must be an int in 1..n"
                )
                fatal_structure_error = True
                continue
            if not isinstance(serve_delivery, bool):
                violations.append(
                    f"STRUCTURE: {location}.serve_delivery must be bool"
                )
                fatal_structure_error = True
                continue
            if not isinstance(serve_pickup, bool):
                violations.append(
                    f"STRUCTURE: {location}.serve_pickup must be bool"
                )
                fatal_structure_error = True
                continue

            route.append((customer_id, serve_delivery, serve_pickup))
        routes.append(route)

    if fatal_structure_error:
        return violations

    # C1
    # Count complete delivery services; every delivery request must occur exactly once.
    delivery_counts = Counter(
        customer_id
        for route in routes
        for customer_id, serve_delivery, _ in route
        if serve_delivery
    )
    for customer_id in range(1, n + 1):
        count = delivery_counts[customer_id]
        if count != 1:
            violations.append(
                f"C1: customer {customer_id} delivery is served {count} times; expected exactly 1"
            )

    # C2
    # Count complete pickup services; every pickup request must occur exactly once.
    pickup_counts = Counter(
        customer_id
        for route in routes
        for customer_id, _, serve_pickup in route
        if serve_pickup
    )
    for customer_id in range(1, n + 1):
        count = pickup_counts[customer_id]
        if count != 1:
            violations.append(
                f"C2: customer {customer_id} pickup is served {count} times; expected exactly 1"
            )

    visited_pairs = {
        (route_index, customer_id)
        for route_index, route in enumerate(routes)
        for customer_id, _, _ in route
    }

    # C3
    # In the composite visit encoding, every marked delivery must belong to an actual visit on that route.
    delivery_pairs = {
        (route_index, customer_id)
        for route_index, route in enumerate(routes)
        for customer_id, serve_delivery, _ in route
        if serve_delivery
    }
    for route_index, customer_id in sorted(delivery_pairs - visited_pairs):
        violations.append(
            f"C3: route {route_index} serves delivery for customer {customer_id} without visiting it"
        )

    # C4
    # In the composite visit encoding, every marked pickup must belong to an actual visit on that route.
    pickup_pairs = {
        (route_index, customer_id)
        for route_index, route in enumerate(routes)
        for customer_id, _, serve_pickup in route
        if serve_pickup
    }
    for route_index, customer_id in sorted(pickup_pairs - visited_pairs):
        violations.append(
            f"C4: route {route_index} serves pickup for customer {customer_id} without visiting it"
        )

    # C5
    # Reject visits that perform neither request; both flags on one record represent joint service in one visit.
    for route_index, route in enumerate(routes):
        for visit_index, (customer_id, serve_delivery, serve_pickup) in enumerate(route):
            if not serve_delivery and not serve_pickup:
                violations.append(
                    f"C5: route {route_index} visit {visit_index} to customer {customer_id} performs no service"
                )

    # C6
    # Each customer may have only one outgoing route occurrence for a given vehicle.
    for route_index, route in enumerate(routes):
        occurrence_counts = Counter(customer_id for customer_id, _, _ in route)
        for customer_id, count in sorted(occurrence_counts.items()):
            if count > 1:
                violations.append(
                    f"C6: route {route_index} leaves customer {customer_id} {count} times; expected at most 1"
                )

    # C7
    # Each customer may have only one incoming route occurrence for a given vehicle.
    for route_index, route in enumerate(routes):
        occurrence_counts = Counter(customer_id for customer_id, _, _ in route)
        for customer_id, count in sorted(occurrence_counts.items()):
            if count > 1:
                violations.append(
                    f"C7: route {route_index} enters customer {customer_id} {count} times; expected at most 1"
                )

    # C8
    # Every listed route represents a used vehicle and therefore must have one implicit depot departure.
    for route_index, route in enumerate(routes):
        departures = 1 if route else 0
        if departures != 1:
            violations.append(
                f"C8: route {route_index} has {departures} depot departures; expected exactly 1"
            )

    # C9
    # Every listed route represents a used vehicle and therefore must have one implicit depot return.
    for route_index, route in enumerate(routes):
        returns = 1 if route else 0
        if returns != 1:
            violations.append(
                f"C9: route {route_index} has {returns} depot returns; expected exactly 1"
            )

    # C10
    # Sequence positions are the reconstructed order variables and must lie in 1..n.
    for route_index, route in enumerate(routes):
        for position, (customer_id, _, _) in enumerate(route, start=1):
            if not 1 <= position <= n:
                violations.append(
                    f"C10: customer {customer_id} on route {route_index} has order {position}, outside 1..n"
                )

    # C11
    # A route is encoded as one depot-rooted sequence, so verify all visit positions lie on that sequence.
    for route_index, route in enumerate(routes):
        reachable_positions = set()
        position = 0
        while position < len(route):
            reachable_positions.add(position)
            position += 1
        if len(reachable_positions) != len(route):
            violations.append(
                f"C11: route {route_index} contains visits disconnected from its depot-rooted sequence"
            )

    # C12
    # Reconstruct delivery flow and verify that each visit removes exactly its assigned delivery demand.
    for route_index, route in enumerate(routes):
        remaining_delivery = sum(
            delivery[customer_id - 1]
            for customer_id, serve_delivery, _ in route
            if serve_delivery
        )
        for visit_index, (customer_id, serve_delivery, _) in enumerate(route):
            before = remaining_delivery
            amount = delivery[customer_id - 1] if serve_delivery else 0.0
            remaining_delivery -= amount
            if not _close(before - remaining_delivery, amount) or remaining_delivery < -_EPS:
                violations.append(
                    f"C12: route {route_index} visit {visit_index} has inconsistent delivery-flow balance"
                )

    # C13
    # Reconstruct pickup flow and verify that each visit adds exactly its assigned pickup demand.
    for route_index, route in enumerate(routes):
        collected_pickup = 0.0
        for visit_index, (customer_id, _, serve_pickup) in enumerate(route):
            before = collected_pickup
            amount = pickup[customer_id - 1] if serve_pickup else 0.0
            collected_pickup += amount
            if not _close(collected_pickup - before, amount):
                violations.append(
                    f"C13: route {route_index} visit {visit_index} has inconsistent pickup-flow balance"
                )

    # C14
    # Reconstructed routes always start with zero collected pickup cargo; check that depot boundary state.
    for route_index, _ in enumerate(routes):
        pickup_on_departure = 0.0
        if not _close(pickup_on_departure, 0.0):
            violations.append(
                f"C14: route {route_index} leaves the depot carrying pickup cargo"
            )

    # C15
    # After all assigned deliveries have been unloaded, no delivery cargo may return to the depot.
    for route_index, route in enumerate(routes):
        remaining_delivery = sum(
            delivery[customer_id - 1]
            for customer_id, serve_delivery, _ in route
            if serve_delivery
        )
        for customer_id, serve_delivery, _ in route:
            if serve_delivery:
                remaining_delivery -= delivery[customer_id - 1]
        if not _close(remaining_delivery, 0.0):
            violations.append(
                f"C15: route {route_index} returns with {remaining_delivery} units of undelivered cargo"
            )

    # C16
    # Check reconstructed onboard delivery-plus-pickup load on every depot/customer travel leg.
    for route_index, route in enumerate(routes):
        remaining_delivery = sum(
            delivery[customer_id - 1]
            for customer_id, serve_delivery, _ in route
            if serve_delivery
        )
        collected_pickup = 0.0
        initial_load = remaining_delivery
        if _over_capacity(initial_load, capacity):
            violations.append(
                f"C16: route {route_index} departs with load {initial_load}, exceeding capacity {capacity}"
            )

        for visit_index, (customer_id, serve_delivery, serve_pickup) in enumerate(route):
            if serve_delivery:
                remaining_delivery -= delivery[customer_id - 1]
            if serve_pickup:
                collected_pickup += pickup[customer_id - 1]
            load = remaining_delivery + collected_pickup
            if load < -_EPS:
                violations.append(
                    f"C16: route {route_index} has negative reconstructed load after visit {visit_index}"
                )
            if _over_capacity(load, capacity):
                violations.append(
                    f"C16: route {route_index} has load {load} after visiting customer {customer_id}, exceeding capacity {capacity}"
                )

    return violations
