"""Check a complete route solution against the fixed input/output contract, using only the Python standard library.

check returns an empty list if and only if the submission is a complete feasible route plan; it does not check optimality.
A correct infeasible report is not a feasible route plan, so it also returns a non-empty explanation. An incorrect infeasibility
claim is pointed out separately: with no fleet-size limit and effective times satisfying the triangle inequality, an instance
is feasible if and only if every customer can be served by its own single-vehicle round trip.

Finite JSON numbers are converted to Fraction from their decimal representation and all comparisons are exact; no
epsilon is allowed for overloading or crossing time bounds. That input distances are Euclidean is an agreed data
guarantee; customer coordinates are not provided, so they are not reconstructed. This module checks shape, value domains, the precision and
symmetry of base times, and the directly verifiable triangle inequality of effective travel times.
"""

from collections import Counter
from fractions import Fraction
import math


_INPUT_KEYS = (
    "customer_demands", "vehicle_capacity", "customer_service_times",
    "customer_time_windows", "depot_time_window", "depot_coordinates",
    "base_travel_times",
)
_SOLUTION_KEYS = (
    "status", "routes", "route_departure_times", "customer_departure_times",
    "route_return_times", "customer_loads",
)


def _number(value, label, errors):
    if type(value) not in (int, float):
        errors.append(f"Structure: {label} must be a JSON number, not a boolean or string.")
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"Structure: {label} must be a finite number.")
            return None
        return Fraction(str(value))
    return Fraction(value)


def _vector(value, length, label, errors):
    if not isinstance(value, list):
        errors.append(f"Structure: {label} must be a list.")
        return None
    if length is not None and len(value) != length:
        errors.append(f"Structure: {label} must have length {length}, got {len(value)}.")
        return None
    result = [_number(v, f"{label}[{k}]", errors) for k, v in enumerate(value)]
    return None if any(v is None for v in result) else result


def _matrix(value, rows, columns, label, errors):
    if not isinstance(value, list) or len(value) != rows:
        errors.append(f"Structure: {label} must be a list with {rows} rows.")
        return None
    result = [_vector(row, columns, f"{label}[{k}]", errors)
              for k, row in enumerate(value)]
    return None if any(row is None for row in result) else result


def _prepare_instance(instance, errors):
    if not isinstance(instance, dict):
        errors.append("Structure: instance must be a dict.")
        return None
    missing = [key for key in _INPUT_KEYS if key not in instance]
    if missing:
        errors.append("Structure: instance is missing fields " + ", ".join(missing) + ".")
        return None
    demands = _vector(instance["customer_demands"], None, "customer_demands", errors)
    if demands is None:
        return None
    n = len(demands)
    capacity = _number(instance["vehicle_capacity"], "vehicle_capacity", errors)
    services = _vector(instance["customer_service_times"], n,
                       "customer_service_times", errors)
    windows = _matrix(instance["customer_time_windows"], n, 2,
                      "customer_time_windows", errors)
    horizon = _vector(instance["depot_time_window"], 2, "depot_time_window", errors)
    coordinates = _vector(instance["depot_coordinates"], None,
                          "depot_coordinates", errors)
    base = _matrix(instance["base_travel_times"], n + 1, n + 1,
                   "base_travel_times", errors)
    if errors:
        return None
    if any(q <= 0 for q in demands):
        errors.append("Input: every entry of customer_demands must be strictly positive.")
    if capacity <= 0:
        errors.append("Input: vehicle_capacity must be strictly positive.")
    if any(s < 0 for s in services):
        errors.append("Input: every entry of customer_service_times must be non-negative.")
    for k, (ready, due) in enumerate(windows):
        if ready < 0 or due < ready:
            errors.append(f"Input: customer_time_windows[{k}] must satisfy 0<=ready<=due.")
    if horizon[0] > horizon[1]:
        errors.append("Input: the start of depot_time_window must not be later than its end.")
    if not coordinates:
        errors.append("Input: depot_coordinates must not be empty.")
    for i, row in enumerate(base):
        for j, value in enumerate(row):
            if value < 0 or (value * 10).denominator != 1:
                errors.append(f"Input: base_travel_times[{i}][{j}] must be a non-negative integer multiple of 0.1.")
            if i == j and value != 0:
                errors.append(f"Input: base_travel_times[{i}][{i}] must be zero.")
            if i < j and value != base[j][i]:
                errors.append(f"Input: base travel times between locations {i} and {j} are not symmetric.")
    if errors:
        return None

    ready = [horizon[0]] + [window[0] for window in windows]
    due = [horizon[1]] + [window[1] for window in windows]
    effective = [row[:] for row in base]
    for i in range(1, n + 1):
        for j in range(n + 1):
            if i != j:
                effective[i][j] = max(base[i][j] + services[i - 1], ready[j] - due[i])
    # Arcs leaving the depot keep the base time; the diagonal is only a zero placeholder, not an arc.
    for i in range(n + 1):
        for h in range(n + 1):
            if h == i:
                continue
            for j in range(n + 1):
                if j != i and j != h and effective[i][j] > effective[i][h] + effective[h][j]:
                    errors.append(f"Input: effective travel times violate the triangle inequality {i}->{j} via {h}.")
                    return None
    return n, [Fraction(0)] + demands, capacity, ready, due, effective


def _check_infeasible_report(data):
    """Verify an infeasibility claim, without treating the infeasibility report as a feasible route plan."""
    n, demand, capacity, ready, due, travel = data
    for i in range(1, n + 1):
        earliest = max(ready[i], ready[0] + travel[0][i])
        if (demand[i] > capacity or earliest > due[i]
                or earliest + travel[i][0] > due[0]):
            return [
                f"STATUS: the infeasibility report holds; customer {i} cannot be served by a single round trip; "
                "under the triangle inequality and unlimited fleet of this problem no complete feasible route plan exists. "
                "The report is not a feasible route plan, so check does not return an empty list."
            ]
    return ["STATUS: incorrect infeasibility claim; serving every customer with its own single-vehicle round trip forms a complete feasible plan."]


def check(instance: dict, solution: dict) -> list:
    """Return structural or constraint violation messages; only a complete feasible route plan returns []."""
    errors = []
    data = _prepare_instance(instance, errors)
    if data is None:
        return errors
    n, demand, capacity, ready, due, travel = data
    if not isinstance(solution, dict):
        return ["Structure: solution must be a dict."]
    missing = [key for key in _SOLUTION_KEYS if key not in solution]
    if missing:
        return ["Structure: solution is missing fields " + ", ".join(missing) + "."]
    status = solution["status"]
    if status not in ("feasible", "infeasible"):
        return ["Structure: status must be feasible or infeasible."]
    if status == "infeasible":
        for key in _SOLUTION_KEYS[1:]:
            if not isinstance(solution[key], list) or solution[key] != []:
                errors.append(f"Structure: {key} in an infeasible report must be [].")
        return errors if errors else _check_infeasible_report(data)

    routes = solution["routes"]
    if not isinstance(routes, list):
        return ["Structure: routes must be a list of routes."]
    for r, route in enumerate(routes):
        if not isinstance(route, list) or len(route) < 3:
            errors.append(f"Structure: routes[{r}] must be [0, at least one customer, 0].")
            continue
        if any(type(node) is not int for node in route):
            errors.append(f"Structure: location ids in routes[{r}] must be integers, not booleans.")
            continue
        if route[0] != 0 or route[-1] != 0:
            errors.append(f"Structure: routes[{r}] must start from depot 0 and return to depot 0.")
        if any(node < 1 or node > n for node in route[1:-1]):
            errors.append(f"Structure: the interior of routes[{r}] may only contain customer ids 1 to {n}, not the depot.")
        if any(i == j for i, j in zip(route, route[1:])):
            errors.append(f"Structure: routes[{r}] must not contain self-loops.")
    departure = _vector(solution["route_departure_times"], len(routes),
                        "route_departure_times", errors)
    times = _vector(solution["customer_departure_times"], n,
                    "customer_departure_times", errors)
    returns = _vector(solution["route_return_times"], len(routes),
                      "route_return_times", errors)
    loads = _vector(solution["customer_loads"], n, "customer_loads", errors)
    if errors:
        return errors
    # Insert a depot placeholder so customer arrays can be indexed directly by location id; the depot load only means an empty departure.
    times = [None] + times
    loads = [Fraction(0)] + loads
    edges = [(r, i, j) for r, route in enumerate(routes)
             for i, j in zip(route, route[1:])]
    customer_edges = [(r, i, j) for r, i, j in edges if i != 0 and j != 0]

    # C1
    incoming = Counter(j for _, _, j in edges if j != 0)
    for i in range(1, n + 1):
        if incoming[i] != 1:
            errors.append(f"C1: customer {i} is entered {incoming[i]} times; it must be exactly once.")

    # C2
    outgoing = Counter(i for _, i, _ in edges if i != 0)
    for i in range(1, n + 1):
        if outgoing[i] != 1:
            errors.append(f"C2: customer {i} is left {outgoing[i]} times; it must be exactly once.")

    # C3
    for r, i, j in customer_edges:
        if ready[i] + travel[i][j] > due[j] or demand[i] + demand[j] > capacity:
            errors.append(f"C3: route {r} uses customer arc {i}->{j}, which is incompatible in time windows or capacity.")

    # C4
    for i in range(1, n + 1):
        if not ready[i] <= times[i] <= due[i]:
            errors.append(f"C4: the departure time of customer {i} is not within its closed time window.")

    # C5
    for r, value in enumerate(departure):
        if not ready[0] <= value <= due[0]:
            errors.append(f"C5: the depot departure time of route {r} is not within the depot horizon.")

    # C6
    for r, i, j in customer_edges:
        if times[j] < times[i] + travel[i][j]:
            errors.append(f"C6: route {r}: customer arc {i}->{j} does not satisfy the effective travel time recursion.")

    # C7
    for r, route in enumerate(routes):
        first = route[1]
        if times[first] < departure[r] + travel[0][first]:
            errors.append(f"C7: route {r}: first customer {first} does not allow enough travel time from the depot.")

    # C8
    for r, route in enumerate(routes):
        last = route[-2]
        if returns[r] < times[last] + travel[last][0]:
            errors.append(f"C8: route {r}: the depot return from last customer {last} does not satisfy the effective travel time.")

    # C9
    for r, value in enumerate(returns):
        if not ready[0] <= value <= due[0]:
            errors.append(f"C9: the depot return time of route {r} is not within the depot horizon.")

    # C10
    for i in range(1, n + 1):
        if loads[i] < demand[i]:
            errors.append(f"C10: the departure load of customer {i} is less than its own demand.")

    # C11
    for i in range(1, n + 1):
        if loads[i] > capacity:
            errors.append(f"C11: the departure load of customer {i} exceeds the vehicle capacity.")

    # C12
    for r, i, j in edges:
        if j != 0 and loads[j] < loads[i] + demand[j]:
            errors.append(f"C12: route {r}: arc {i}->{j} does not satisfy the load lower-bound recursion.")
    return errors
