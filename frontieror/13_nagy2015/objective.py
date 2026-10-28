import math


def _is_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def objective(instance: dict, solution: dict) -> float:
    if not isinstance(instance, dict):
        raise ValueError("instance must be a dict")
    if "n" not in instance or "distance_matrix" not in instance:
        raise ValueError("instance must contain n and distance_matrix")

    n = instance["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive int")

    matrix = instance["distance_matrix"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != n + 1
        or any(not isinstance(row, list) or len(row) != n + 1 for row in matrix)
    ):
        raise ValueError("distance_matrix must have shape (n+1) x (n+1)")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if not _is_number(value) or float(value) < 0:
                raise ValueError(
                    f"distance_matrix[{i}][{j}] must be a finite nonnegative number"
                )

    if not isinstance(solution, dict) or "routes" not in solution:
        raise ValueError("solution must be a dict containing routes")
    routes = solution["routes"]
    if not isinstance(routes, list):
        raise ValueError("solution['routes'] must be a list")

    total_distance = 0.0
    required_visit_fields = {"customer_id", "serve_delivery", "serve_pickup"}

    for route_index, route in enumerate(routes):
        if not isinstance(route, list):
            raise ValueError(f"routes[{route_index}] must be a list")

        previous = 0
        for visit_index, visit in enumerate(route):
            location = f"routes[{route_index}][{visit_index}]"
            if not isinstance(visit, dict) or not required_visit_fields.issubset(visit):
                raise ValueError(
                    f"{location} must contain customer_id, serve_delivery, and serve_pickup"
                )

            customer_id = visit["customer_id"]
            if (
                isinstance(customer_id, bool)
                or not isinstance(customer_id, int)
                or not 1 <= customer_id <= n
            ):
                raise ValueError(f"{location}.customer_id must be an int in 1..n")
            if not isinstance(visit["serve_delivery"], bool):
                raise ValueError(f"{location}.serve_delivery must be bool")
            if not isinstance(visit["serve_pickup"], bool):
                raise ValueError(f"{location}.serve_pickup must be bool")

            total_distance += float(matrix[previous][customer_id])
            previous = customer_id

        if route:
            total_distance += float(matrix[previous][0])

    return float(total_distance)
