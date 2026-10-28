def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def _nonnegative_int(value):
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _number_or_zero(value):
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _instance_size(official_instance):
    instance = _as_dict(official_instance)
    declared = _nonnegative_int(instance.get("num_customers"))
    if declared is not None:
        return declared

    customers = _as_list(instance.get("customers"))
    distance_matrix = _as_list(instance.get("distance_matrix"))
    largest_id = 0
    for customer in customers:
        customer_id = _nonnegative_int(_as_dict(customer).get("id"))
        if customer_id is not None:
            largest_id = max(largest_id, customer_id)

    return max(len(customers), max(0, len(distance_matrix) - 1), largest_id)


def to_ours(official_instance: dict) -> dict:
    instance = _as_dict(official_instance)
    n = _instance_size(instance)
    customers = _as_list(instance.get("customers"))

    customers_by_id = {}
    for position, raw_customer in enumerate(customers, start=1):
        customer = _as_dict(raw_customer)
        customer_id = _nonnegative_int(customer.get("id"))
        if customer_id is None or customer_id == 0:
            customer_id = position
        if 1 <= customer_id <= n:
            customers_by_id[customer_id] = customer

    delivery_demands = []
    pickup_demands = []
    for customer_id in range(1, n + 1):
        customer = customers_by_id.get(customer_id, {})
        delivery_demands.append(_number_or_zero(customer.get("delivery_demand")))
        pickup_demands.append(_number_or_zero(customer.get("pickup_demand")))

    raw_matrix = _as_list(instance.get("distance_matrix"))
    distance_matrix = []
    for i in range(n + 1):
        raw_row = _as_list(raw_matrix[i]) if i < len(raw_matrix) else []
        row = []
        for j in range(n + 1):
            value = raw_row[j] if j < len(raw_row) else 0
            row.append(_number_or_zero(value))
        distance_matrix.append(row)

    return {
        "n": n,
        "capacity": _number_or_zero(instance.get("vehicle_capacity")),
        "delivery_demands": delivery_demands,
        "pickup_demands": pickup_demands,
        "distance_matrix": distance_matrix,
    }


def to_official(official_instance: dict, our_solution: dict) -> dict:
    n = _instance_size(official_instance)
    solution = _as_dict(our_solution)
    our_routes = _as_list(solution.get("routes"))
    official_routes = []

    for raw_route in our_routes:
        route = _as_list(raw_route)
        nodes = [0]

        for raw_visit in route:
            visit = _as_dict(raw_visit)
            customer_id = _nonnegative_int(visit.get("customer_id"))
            if customer_id is None or not 1 <= customer_id <= n:
                continue

            serve_delivery = visit.get("serve_delivery") is True
            serve_pickup = visit.get("serve_pickup") is True

            if serve_delivery:
                nodes.append(customer_id)
            if serve_pickup:
                nodes.append(n + customer_id)

        if len(nodes) > 1:
            nodes.append(0)
            official_routes.append(nodes)

    official_solution = {
        "routes": official_routes,
        "num_routes": len(official_routes),
    }
    official_solution["objective_value"] = official_objective(
        official_instance, official_solution
    )
    return official_solution


def official_objective(official_instance: dict, official_solution: dict) -> float:
    ours = to_ours(official_instance)
    n = ours["n"]
    distance_matrix = ours["distance_matrix"]
    solution = _as_dict(official_solution)
    routes = _as_list(solution.get("routes"))

    def original_node(raw_node):
        node = _nonnegative_int(raw_node)
        if node is None:
            return None
        if node == 0:
            return 0
        if 1 <= node <= n:
            return node
        if n + 1 <= node <= 2 * n:
            return node - n
        return None

    total = 0.0
    for raw_route in routes:
        route = _as_list(raw_route)
        for index in range(len(route) - 1):
            start = original_node(route[index])
            end = original_node(route[index + 1])
            if start is not None and end is not None:
                total += _number_or_zero(distance_matrix[start][end])

    return float(total)
