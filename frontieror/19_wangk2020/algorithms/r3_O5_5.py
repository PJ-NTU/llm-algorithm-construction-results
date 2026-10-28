"""Capacity-conserving assignment with batched joint retiming
Combines airport event assignment, cross-scenario batch min-cut and cumulative-indicator joint integer neighbourhoods, improving continually within the wall-clock budget and keeping the best verified solution.
"""
import math
import random
import time
from bisect import bisect_left
from collections import deque

try:
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint, linear_sum_assignment
    from scipy.sparse import coo_matrix
except Exception:
    milp = None
    linear_sum_assignment = None


def _timed_cut(weights, implications, soft_edges, stop_time):
    """Time-limited min-cut with floating-point capacities; an interruption only abandons this neighbourhood and leaves the kept solution unchanged."""
    clock = time.monotonic
    n = len(weights)
    if not n:
        return []
    source, sink = n, n + 1
    graph = [[] for _ in range(n + 2)]
    infinity = 1.0 + sum(abs(x) for x in weights) + sum(c for _, _, c in soft_edges)

    def add(u, v, cap):
        if u != v and cap > 0:
            graph[u].append([v, len(graph[v]), cap])
            graph[v].append([u, len(graph[u]) - 1, 0.0])

    for i, weight in enumerate(weights):
        if weight < 0:
            add(source, i, -weight)
        elif weight > 0:
            add(i, sink, weight)
        if i & 255 == 0 and clock() >= stop_time:
            return None
    for k, (u, v) in enumerate(implications):
        add(u, v, infinity)
        if k & 255 == 0 and clock() >= stop_time:
            return None
    for u, v, cap in soft_edges:
        add(u, v, cap)
    ticks = 0
    while clock() < stop_time:
        level = [-1] * len(graph)
        level[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v, _, cap in graph[u]:
                ticks += 1
                if ticks & 511 == 0 and clock() >= stop_time:
                    return None
                if cap > 1e-11 and level[v] < 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        if level[sink] < 0:
            return [x >= 0 for x in level[:n]]
        pointer = [0] * len(graph)
        # An explicit stack avoids hitting the recursion depth limit on long connection chains.
        while clock() < stop_time:
            vertices, path, limits = [source], [], [infinity]
            pushed = False
            while vertices:
                ticks += 1
                if ticks & 255 == 0 and clock() >= stop_time:
                    return None
                u = vertices[-1]
                if u == sink:
                    amount = limits[-1]
                    for p, index in path:
                        edge = graph[p][index]
                        edge[2] -= amount
                        graph[edge[0]][edge[1]][2] += amount
                    pushed = True
                    break
                while pointer[u] < len(graph[u]):
                    edge = graph[u][pointer[u]]
                    ticks += 1
                    if ticks & 511 == 0 and clock() >= stop_time:
                        return None
                    if edge[2] > 1e-11 and level[edge[0]] == level[u] + 1:
                        break
                    pointer[u] += 1
                if pointer[u] < len(graph[u]):
                    edge = graph[u][pointer[u]]
                    path.append((u, pointer[u]))
                    vertices.append(edge[0])
                    limits.append(min(limits[-1], edge[2]))
                else:
                    level[u] = -1
                    vertices.pop()
                    limits.pop()
                    if path:
                        parent, _ = path.pop()
                        pointer[parent] += 1
            if not pushed:
                break
    return None


def _clip(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def _fallback(instance):
    try:
        data = instance if isinstance(instance, dict) else {}
        n = max(0, int(data.get("n_flights", 0)))
        scenarios = max(0, int(data.get("n_scenarios", 0)))
        horizon = max(1, int(data.get("n_periods", 1)))
        requested = data.get("requested_departure_period", [])
        durations = data.get("scheduled_enroute_duration", [])
        minimum = data.get("min_enroute_duration", [])
        maximum = data.get("max_enroute_duration", [])

        plan = []
        scheduled_arrival = []
        base_departure = []
        base_arrival = []
        for f in range(n):
            request = int(requested[f]) if f < len(requested) else 1
            duration = int(durations[f]) if f < len(durations) else 0
            departure = _clip(request, 1, max(1, horizon - duration))
            arrival = _clip(departure + duration, 1, horizon)
            plan.append(departure)
            scheduled_arrival.append(arrival)

            low_duration = int(minimum[f]) if f < len(minimum) else duration
            high_duration = int(maximum[f]) if f < len(maximum) else duration
            actual_duration = _clip(duration, low_duration, high_duration)
            actual_departure = departure
            actual_arrival = actual_departure + actual_duration
            if actual_arrival > horizon:
                actual_arrival = horizon
                actual_departure = max(departure, horizon - actual_duration)
            base_departure.append(_clip(actual_departure, 1, horizon))
            base_arrival.append(_clip(actual_arrival, 1, horizon))

        operated_departure = [base_departure[:] for _ in range(scenarios)]
        operated_arrival = [base_arrival[:] for _ in range(scenarios)]
        departure_delay = [
            [float(operated_departure[w][f] - plan[f]) for f in range(n)]
            for w in range(scenarios)
        ]
        arrival_delay = [
            [
                float(max(0, operated_arrival[w][f] - scheduled_arrival[f]))
                for f in range(n)
            ]
            for w in range(scenarios)
        ]
        return {
            "scheduled_departure_period": plan,
            "scheduled_arrival_period": scheduled_arrival,
            "operated_departure_period": operated_departure,
            "operated_arrival_period": operated_arrival,
            "departure_delay": departure_delay,
            "arrival_delay": arrival_delay,
        }
    except BaseException:
        return {
            "scheduled_departure_period": [],
            "scheduled_arrival_period": [],
            "operated_departure_period": [],
            "operated_arrival_period": [],
            "departure_delay": [],
            "arrival_delay": [],
        }


def _solve_core(instance, time_limit_s, started, retained):
    clock = time.monotonic
    try:
        budget = float(time_limit_s)
    except BaseException:
        budget = 0.0
    if not math.isfinite(budget) or budget < 0.0:
        budget = 0.0
    reserve = min(0.25, max(0.001, budget * 0.025))
    deadline = started + max(0.0, budget - reserve)

    n = int(instance["n_flights"])
    airport_count = int(instance["n_airports"])
    horizon = int(instance["n_periods"])
    scenario_count = int(instance["n_scenarios"])

    origin = [int(x) for x in instance["flight_departure_airport"]]
    destination = [int(x) for x in instance["flight_arrival_airport"]]
    requested = [int(x) for x in instance["requested_departure_period"]]
    duration = [int(x) for x in instance["scheduled_enroute_duration"]]
    min_duration = [int(x) for x in instance["min_enroute_duration"]]
    max_duration = [int(x) for x in instance["max_enroute_duration"]]
    max_dep_delay = [int(x) for x in instance["max_departure_delay"]]
    max_arr_delay = [int(x) for x in instance["max_arrival_delay"]]
    dep_cost = [float(x) for x in instance["departure_delay_cost"]]
    original_arr_cost = [float(x) for x in instance["arrival_delay_cost"]]
    arr_cost = [max(0.0, x) for x in original_arr_cost]
    displacement = int(instance["max_schedule_displacement"])
    probabilities = [float(x) for x in instance["scenario_probabilities"]]
    rho = float(instance["rho"])

    weather = instance["weather_condition"]
    groups, scenario_mapping, unique_weather, merged_probabilities = {}, [], [], []
    for w in range(scenario_count):
        key = tuple(tuple(int(v) for v in row) for row in weather[w])
        index = groups.get(key)
        if index is None:
            index = len(unique_weather)
            groups[key] = index
            unique_weather.append(weather[w])
            merged_probabilities.append(0.0)
        scenario_mapping.append(index)
        merged_probabilities[index] += probabilities[w]
    weather = unique_weather
    probabilities = merged_probabilities
    scenario_count = len(unique_weather)

    def topological(successors, indegree):
        queue = [f for f in range(n) if indegree[f] == 0]
        order = []
        head = 0
        while head < len(queue):
            f = queue[head]
            head += 1
            order.append(f)
            for g, _ in successors[f]:
                indegree[g] -= 1
                if indegree[g] == 0:
                    queue.append(g)
        return order

    schedule_successors = [[] for _ in range(n)]
    schedule_predecessors = [[] for _ in range(n)]
    schedule_indegree = [0] * n
    schedule_pairs = instance["schedule_connection_pairs"]
    schedule_times = instance["schedule_min_connection_time"]
    for i, pair in enumerate(schedule_pairs):
        f = int(pair[0])
        g = int(pair[1])
        lag = duration[f] + int(schedule_times[i])
        schedule_successors[f].append((g, lag))
        schedule_predecessors[g].append((f, lag))
        schedule_indegree[g] += 1

    aircraft_successors = [[] for _ in range(n)]
    aircraft_predecessors = [[] for _ in range(n)]
    aircraft_indegree = [0] * n
    aircraft_pairs = instance["aircraft_connection_pairs"]
    aircraft_times = instance["aircraft_min_connection_time"]
    for i, pair in enumerate(aircraft_pairs):
        f = int(pair[0])
        g = int(pair[1])
        connection = int(aircraft_times[i])
        aircraft_successors[f].append((g, connection))
        aircraft_predecessors[g].append((f, connection))
        aircraft_indegree[g] += 1

    schedule_order = topological(schedule_successors, schedule_indegree[:])
    aircraft_order = topological(aircraft_successors, aircraft_indegree[:])
    if len(schedule_order) != n or len(aircraft_order) != n:
        return _fallback(instance)

    aircraft_position = [0] * n
    for position, f in enumerate(aircraft_order):
        aircraft_position[f] = position

    lower = [max(1, requested[f] - displacement) for f in range(n)]
    upper = [
        min(horizon - duration[f], requested[f] + displacement)
        for f in range(n)
    ]
    if any(lower[f] > upper[f] for f in range(n)):
        return _fallback(instance)

    earliest = lower[:]
    for f in schedule_order:
        if earliest[f] > upper[f]:
            return _fallback(instance)
        for g, lag in schedule_successors[f]:
            earliest[g] = max(earliest[g], earliest[f] + lag)
    latest = upper[:]
    for f in reversed(schedule_order):
        for g, lag in schedule_successors[f]:
            latest[f] = min(latest[f], latest[g] - lag)
    if any(earliest[f] > latest[f] for f in range(n)):
        return _fallback(instance)

    base_plan = [0] * n
    for f in schedule_order:
        low = lower[f]
        for p, lag in schedule_predecessors[f]:
            low = max(low, base_plan[p] + lag)
        if low > latest[f]:
            return _fallback(instance)
        base_plan[f] = _clip(requested[f], low, latest[f])
    for f in reversed(schedule_order):
        low = lower[f]
        high = upper[f]
        for p, lag in schedule_predecessors[f]:
            low = max(low, base_plan[p] + lag)
        for g, lag in schedule_successors[f]:
            high = min(high, base_plan[g] - lag)
        if low <= high:
            base_plan[f] = _clip(requested[f], low, high)

    seed = 1469598103934665603
    for value in (n, airport_count, horizon, scenario_count, displacement):
        seed = ((seed ^ int(value)) * 1099511628211) & ((1 << 64) - 1)
    for value in requested + duration:
        seed = ((seed ^ int(value)) * 1099511628211) & ((1 << 64) - 1)
    rng = random.Random(seed)

    criticality = [0] * n
    for f in reversed(aircraft_order):
        if aircraft_successors[f]:
            criticality[f] = 1 + max(
                criticality[g] for g, _ in aircraft_successors[f]
            )

    capacity_dep = instance["capacity_departure_coefficients"]
    capacity_arr = instance["capacity_arrival_coefficients"]
    capacity_limit = instance["capacity_limits"]
    hull_envelopes = []
    origin_count = [0] * airport_count
    destination_count = [0] * airport_count
    for f in range(n):
        origin_count[origin[f]] += 1
        destination_count[destination[f]] += 1
    for condition in range(len(capacity_limit)):
        airport_hulls = []
        for airport in range(airport_count):
            lines = [(float(cd), float(ca), float(cap)) for cd, ca, cap in zip(
                capacity_dep[condition][airport], capacity_arr[condition][airport],
                capacity_limit[condition][airport])]
            hull = []
            usable = all(cd >= 0 and ca >= 0 and cap >= 0 for cd, ca, cap in lines)
            if usable:
                for d in range(origin_count[airport] + 1):
                    if clock() >= deadline:
                        usable = False
                        break
                    a = destination_count[airport]
                    for cd, ca, cap in lines:
                        slack = cap - cd * d
                        if slack < -1e-9 * (1 + abs(cap)):
                            a = -1
                            break
                        if ca > 0:
                            a = min(a, int(math.floor((slack + 1e-9 * (1 + abs(cap))) / ca)))
                    if a < 0:
                        break
                    point = (d, a)
                    while len(hull) >= 2:
                        x, y = hull[-2]
                        u, v = hull[-1]
                        if (u - x) * (a - v) - (v - y) * (d - u) < 0:
                            break
                        hull.pop()
                    hull.append(point)
            if usable and hull:
                tightened = [(1.0, 0.0, float(hull[-1][0])),
                             (0.0, 1.0, float(hull[0][1]))]
                for (x, y), (u, v) in zip(hull, hull[1:]):
                    cd, ca = y - v, u - x
                    divisor = math.gcd(cd, ca)
                    cd, ca = cd // divisor, ca // divisor
                    tightened.append((float(cd), float(ca), float(cd * x + ca * y)))
                # Keep the original lines to avoid loosening at floating-point boundaries.
                airport_hulls.append(tuple(dict.fromkeys(lines + tightened)))
            else:
                airport_hulls.append(tuple(lines))
        hull_envelopes.append(airport_hulls)
    capacities = []
    model_capacities = []
    scenario_difficulty = []
    for w in range(scenario_count):
        rows = [[()] * (horizon + 1) for _ in range(airport_count)]
        cut_rows = [[()] * (horizon + 1) for _ in range(airport_count)]
        difficulty = 0.0
        for airport in range(airport_count):
            for period in range(1, horizon + 1):
                condition = int(weather[w][airport][period - 1])
                difficulty += condition
                cut_rows[airport][period] = hull_envelopes[condition][airport]
                rows[airport][period] = tuple(
                    (
                        float(capacity_dep[condition][airport][line]),
                        float(capacity_arr[condition][airport][line]),
                        float(capacity_limit[condition][airport][line]),
                    )
                    for line in range(len(capacity_limit[condition][airport]))
                )
        capacities.append(rows)
        model_capacities.append(cut_rows)
        scenario_difficulty.append(difficulty)
    scenario_order = list(range(scenario_count))
    scenario_order.sort(key=lambda w: (-scenario_difficulty[w], w))

    def empty_counts():
        return (
            [[0] * (horizon + 1) for _ in range(airport_count)],
            [[0] * (horizon + 1) for _ in range(airport_count)],
        )

    def slot_allows(dep_counts, arr_counts, w, airport, period, extra_dep, extra_arr):
        departures = dep_counts[airport][period] + extra_dep
        arrivals = arr_counts[airport][period] + extra_arr
        for cd, ca, limit in capacities[w][airport][period]:
            if cd * departures + ca * arrivals > limit + 1e-9 * (1.0 + abs(limit)):
                return False
        return True

    def slot_pressure(dep_counts, arr_counts, w, airport, period, extra_dep, extra_arr):
        departures = dep_counts[airport][period] + extra_dep
        arrivals = arr_counts[airport][period] + extra_arr
        value = 0.0
        for cd, ca, limit in capacities[w][airport][period]:
            value += (cd * departures + ca * arrivals) / max(1.0, abs(limit))
        return value

    def pair_allows(dep_counts, arr_counts, w, f, departure, arrival):
        if origin[f] == destination[f] and departure == arrival:
            return slot_allows(
                dep_counts, arr_counts, w, origin[f], departure, 1, 1
            )
        return (
            slot_allows(dep_counts, arr_counts, w, origin[f], departure, 1, 0)
            and slot_allows(dep_counts, arr_counts, w, destination[f], arrival, 0, 1)
        )

    def pair_pressure(dep_counts, arr_counts, w, f, departure, arrival):
        if origin[f] == destination[f] and departure == arrival:
            return slot_pressure(
                dep_counts, arr_counts, w, origin[f], departure, 1, 1
            )
        return (
            slot_pressure(dep_counts, arr_counts, w, origin[f], departure, 1, 0)
            + slot_pressure(dep_counts, arr_counts, w, destination[f], arrival, 0, 1)
        )

    def change_counts(dep_counts, arr_counts, f, departure, arrival, amount):
        dep_counts[origin[f]][departure] += amount
        arr_counts[destination[f]][arrival] += amount

    def individual_valid(plan, f, departure, arrival):
        if departure < plan[f] or departure > min(horizon, plan[f] + max_dep_delay[f]):
            return False
        if arrival < 1 or arrival > horizon:
            return False
        actual_duration = arrival - departure
        if actual_duration < min_duration[f] or actual_duration > max_duration[f]:
            return False
        if arrival - plan[f] - duration[f] > max_arr_delay[f]:
            return False
        return True

    def row_cost(plan, departures, arrivals):
        value = 0.0
        for f in range(n):
            value += dep_cost[f] * (departures[f] - plan[f])
            value += arr_cost[f] * max(
                0, arrivals[f] - plan[f] - duration[f]
            )
        return value

    def objective(plan, operations):
        value = rho * sum(abs(plan[f] - requested[f]) for f in range(n))
        if scenario_count and 1.0 - rho != 0.0:
            expected = 0.0
            for w in range(scenario_count):
                expected += probabilities[w] * row_cost(
                    plan, operations[w][0], operations[w][1]
                )
            value += (1.0 - rho) * expected
        return value

    def copy_operations(operations):
        return [[item[0][:], item[1][:]] for item in operations]

    def row_valid(plan, w, operation):
        if operation is None or len(operation) != 2:
            return False
        departures, arrivals = operation
        if len(departures) != n or len(arrivals) != n:
            return False
        dep_counts, arr_counts = empty_counts()
        for f in range(n):
            departure = int(departures[f])
            arrival = int(arrivals[f])
            if not individual_valid(plan, f, departure, arrival):
                return False
            change_counts(dep_counts, arr_counts, f, departure, arrival, 1)
        for f in range(n):
            for g, connection in aircraft_successors[f]:
                if departures[g] - arrivals[f] < connection:
                    return False
        for airport in range(airport_count):
            for period in range(1, horizon + 1):
                if not slot_allows(
                    dep_counts, arr_counts, w, airport, period, 0, 0
                ):
                    return False
        return True

    def plan_valid_with_operations(plan, operations):
        if len(plan) != n or len(operations) != scenario_count:
            return False
        for f in range(n):
            if plan[f] < lower[f] or plan[f] > upper[f]:
                return False
            for g, lag in schedule_successors[f]:
                if plan[g] < plan[f] + lag:
                    return False
        for w in range(scenario_count):
            departures, arrivals = operations[w]
            for f in range(n):
                if not individual_valid(plan, f, departures[f], arrivals[f]):
                    return False
        return True

    def all_valid(plan, operations):
        if not plan_valid_with_operations(plan, operations):
            return False
        for w in range(scenario_count):
            if not row_valid(plan, w, operations[w]):
                return False
        return True

    def random_plan(anchor, elapsed, broad=False):
        result = [0] * n
        level = math.log1p(max(0.0, elapsed))
        exploration = min(0.95, 0.10 + 0.055 * level)
        if broad:
            exploration = max(0.58, exploration)
        for f in schedule_order:
            low = lower[f]
            for p, lag in schedule_predecessors[f]:
                low = max(low, result[p] + lag)
            high = latest[f]
            if low > high:
                return None
            if anchor is None:
                target = requested[f]
            elif rng.random() < 0.56:
                target = anchor[f]
            else:
                target = requested[f]
            target = _clip(target, low, high)
            if low == high:
                result[f] = low
            elif rng.random() < exploration:
                result[f] = rng.randint(low, high)
            else:
                result[f] = _clip(
                    int(round(rng.triangular(low, high, target))), low, high
                )
        return result

    def construct_row(plan, w, stop_time, noise):
        if n == 0:
            return [[], []]
        scheduled_arrival = [plan[f] + duration[f] for f in range(n)]
        arrival_limit = [0] * n
        departure_limit = [0] * n
        for f in reversed(aircraft_order):
            limit = min(horizon, scheduled_arrival[f] + max_arr_delay[f])
            for g, connection in aircraft_successors[f]:
                limit = min(limit, departure_limit[g] - connection)
            arrival_limit[f] = limit
            departure_limit[f] = min(
                horizon,
                plan[f] + max_dep_delay[f],
                limit - min_duration[f],
            )
            if departure_limit[f] < plan[f]:
                return None

        dep_counts, arr_counts = empty_counts()
        departures = [0] * n
        arrivals = [0] * n
        assigned = [False] * n
        assigned_count = 0
        frames = []

        while clock() < stop_time:
            if assigned_count == n:
                return [departures[:], arrivals[:]]

            chosen = -1
            chosen_low = 0
            chosen_key = None
            for f in range(n):
                if assigned[f]:
                    continue
                ready = True
                low = plan[f]
                for p, connection in aircraft_predecessors[f]:
                    if not assigned[p]:
                        ready = False
                        break
                    low = max(low, arrivals[p] + connection)
                if not ready:
                    continue
                width = departure_limit[f] - low + 1
                key = (
                    max(0, width) * max(1, max_duration[f] - min_duration[f] + 1),
                    arrival_limit[f] - low - min_duration[f],
                    -criticality[f],
                    rng.random() * noise,
                )
                if chosen_key is None or key < chosen_key:
                    chosen_key = key
                    chosen = f
                    chosen_low = low

            candidates = []
            if chosen >= 0 and chosen_low <= departure_limit[chosen]:
                scale = (
                    1.0
                    + dep_cost[chosen] * max_dep_delay[chosen]
                    + arr_cost[chosen] * max_arr_delay[chosen]
                )
                inspected = 0
                for departure in range(chosen_low, departure_limit[chosen] + 1):
                    low_arrival = max(1, departure + min_duration[chosen])
                    high_arrival = min(
                        departure + max_duration[chosen], arrival_limit[chosen]
                    )
                    for arrival in range(low_arrival, high_arrival + 1):
                        inspected += 1
                        if inspected & 63 == 0 and clock() >= stop_time:
                            return None
                        if not pair_allows(
                            dep_counts, arr_counts, w, chosen, departure, arrival
                        ):
                            continue
                        cost = (
                            dep_cost[chosen] * (departure - plan[chosen])
                            + arr_cost[chosen]
                            * max(0, arrival - scheduled_arrival[chosen])
                        )
                        pressure = pair_pressure(
                            dep_counts, arr_counts, w, chosen, departure, arrival
                        )
                        score = (
                            cost
                            + 0.025 * scale * pressure
                            + noise * scale * rng.random()
                        )
                        candidates.append(
                            (score, cost, rng.random(), departure, arrival)
                        )
                candidates.sort()

            if candidates:
                frames.append([chosen, candidates, 0, None])

            placed = False
            while frames and not placed:
                frame = frames[-1]
                f = frame[0]
                current = frame[3]
                if current is not None:
                    change_counts(
                        dep_counts, arr_counts, f, current[0], current[1], -1
                    )
                    assigned[f] = False
                    departures[f] = 0
                    arrivals[f] = 0
                    assigned_count -= 1
                    frame[3] = None

                while frame[2] < len(frame[1]):
                    item = frame[1][frame[2]]
                    frame[2] += 1
                    departure = item[3]
                    arrival = item[4]
                    if not pair_allows(
                        dep_counts, arr_counts, w, f, departure, arrival
                    ):
                        continue
                    change_counts(
                        dep_counts, arr_counts, f, departure, arrival, 1
                    )
                    assigned[f] = True
                    departures[f] = departure
                    arrivals[f] = arrival
                    assigned_count += 1
                    frame[3] = (departure, arrival)
                    placed = True
                    break
                if not placed:
                    frames.pop()

            if not placed:
                return None
        return None

    row_cache = {}

    def build_all_rows(plan, stop_time, noise):
        key = tuple(plan)
        operations = row_cache.get(key)
        if operations is None:
            operations = [None] * scenario_count
            row_cache[key] = operations
        pending = [w for w in scenario_order if operations[w] is None]
        for position, w in enumerate(pending):
            now = clock()
            if now >= stop_time:
                return None
            left = len(pending) - position
            local_end = now + max(0.0, stop_time - now) / max(1, left)
            operation = construct_row(plan, w, local_end, noise)
            if operation is None:
                return None
            operations[w] = operation
        return copy_operations(operations)

    def related_subset(plan, departures, arrivals, target, mandatory=None):
        if n == 0:
            return []
        target = min(n, max(1, target, len(mandatory or [])))
        selected = set(mandatory or [])
        events = {}
        for f in range(n):
            events.setdefault((origin[f], departures[f]), []).append(f)
            events.setdefault((destination[f], arrivals[f]), []).append(f)
        if not selected:
            sample_size = min(n, max(1, int(math.sqrt(n) + math.log1p(budget))))
            candidates = rng.sample(range(n), sample_size)
            selected.add(max(candidates, key=lambda f:
                dep_cost[f] * (departures[f] - plan[f]) + arr_cost[f] *
                max(0, arrivals[f] - plan[f] - duration[f]) + rng.random()))
        frontier = list(selected)
        pool = {}
        head = 0
        radius = max(1, int(math.sqrt(target) / 2))
        while len(selected) < target and clock() < deadline:
            if head < len(frontier):
                f = frontier[head]
                head += 1
                for g, lag in aircraft_predecessors[f] + aircraft_successors[f]:
                    if g not in selected:
                        pool[g] = pool.get(g, 0.0) + 5.0
                for g, lag in schedule_predecessors[f] + schedule_successors[f]:
                    if g not in selected:
                        pool[g] = pool.get(g, 0.0) + 3.0
                intervals = (
                    (origin[f], max(1, plan[f] - radius), departures[f] + radius),
                    (destination[f], max(1, plan[f] + min_duration[f] - radius),
                     arrivals[f] + radius))
                for airport, first, last in intervals:
                    for t in range(first, min(horizon, last) + 1):
                        for g in events.get((airport, t), ()):
                            if g not in selected:
                                pool[g] = pool.get(g, 0.0) + 1.0
            if pool:
                choices = list(pool)
                weights = [pool[g] for g in choices]
                g = rng.choices(choices, weights=weights, k=1)[0]
                del pool[g]
            else:
                remaining = [f for f in range(n) if f not in selected]
                if not remaining:
                    break
                g = rng.choice(remaining)
            selected.add(g)
            frontier.append(g)
        return sorted(selected)

    def optimize_subset(
        plan,
        w,
        old_departures,
        old_arrivals,
        subset,
        stop_time,
        incumbent_feasible=True,
        first_feasible=False,
        noise=0.0,
    ):
        if clock() >= stop_time:
            return None
        subset = list(dict.fromkeys(
            int(f) for f in subset if 0 <= int(f) < n
        ))
        if not subset:
            return None
        inside = set(subset)
        dep_counts, arr_counts = empty_counts()
        for f in range(n):
            if f not in inside:
                change_counts(
                    dep_counts,
                    arr_counts,
                    f,
                    old_departures[f],
                    old_arrivals[f],
                    1,
                )

        departures = old_departures[:]
        arrivals = old_arrivals[:]
        for f in subset:
            departures[f] = 0
            arrivals[f] = 0

        domains = {}
        minimum_cost = {}
        minimum_arrival = {}
        maximum_departure = {}
        for f in subset:
            low_departure = plan[f]
            for p, connection in aircraft_predecessors[f]:
                if p not in inside:
                    low_departure = max(
                        low_departure, old_arrivals[p] + connection
                    )
            high_arrival = min(
                horizon, plan[f] + duration[f] + max_arr_delay[f]
            )
            for g, connection in aircraft_successors[f]:
                if g not in inside:
                    high_arrival = min(
                        high_arrival, old_departures[g] - connection
                    )
            high_departure = min(
                horizon,
                plan[f] + max_dep_delay[f],
                high_arrival - min_duration[f],
            )
            if low_departure > high_departure:
                return None

            values = []
            scale = (
                1.0
                + dep_cost[f] * max_dep_delay[f]
                + arr_cost[f] * max_arr_delay[f]
            )
            inspected = 0
            for departure in range(low_departure, high_departure + 1):
                low_arrival = max(1, departure + min_duration[f])
                last_arrival = min(
                    departure + max_duration[f], high_arrival
                )
                for arrival in range(low_arrival, last_arrival + 1):
                    inspected += 1
                    if inspected & 127 == 0 and clock() >= stop_time:
                        return None
                    if not pair_allows(
                        dep_counts, arr_counts, w, f, departure, arrival
                    ):
                        continue
                    cost = (
                        dep_cost[f] * (departure - plan[f])
                        + arr_cost[f]
                        * max(0, arrival - plan[f] - duration[f])
                    )
                    pressure = pair_pressure(
                        dep_counts, arr_counts, w, f, departure, arrival
                    )
                    rank = (
                        cost
                        + 0.002 * scale * pressure
                        + noise * scale * rng.random()
                    )
                    values.append((rank, cost, departure, arrival))
            if not values:
                return None
            values.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            domains[f] = values
            minimum_cost[f] = min(item[1] for item in values)
            minimum_arrival[f] = min(item[3] for item in values)
            maximum_departure[f] = max(item[2] for item in values)

        old_subset_cost = 0.0
        best_cost = float("inf")
        best_values = None
        if incumbent_feasible:
            old_ok = True
            for f in subset:
                if not individual_valid(
                    plan, f, old_departures[f], old_arrivals[f]
                ):
                    old_ok = False
                    break
                old_subset_cost += (
                    dep_cost[f] * (old_departures[f] - plan[f])
                    + arr_cost[f]
                    * max(0, old_arrivals[f] - plan[f] - duration[f])
                )
            if old_ok:
                best_cost = old_subset_cost
                best_values = {
                    f: (old_departures[f], old_arrivals[f]) for f in subset
                }
            else:
                incumbent_feasible = False

        unassigned = set(subset)
        stopped = [False]
        calls = [0]

        def links_allow(f, departure, arrival):
            for p, connection in aircraft_predecessors[f]:
                if p in unassigned:
                    if departure < minimum_arrival[p] + connection:
                        return False
                elif departures[p] <= 0 or departure < arrivals[p] + connection:
                    return False
            for g, connection in aircraft_successors[f]:
                if g in unassigned:
                    if maximum_departure[g] < arrival + connection:
                        return False
                elif departures[g] <= 0 or departures[g] < arrival + connection:
                    return False
            return True

        def search(partial_cost, remaining_bound):
            nonlocal best_cost, best_values
            if stopped[0]:
                return
            calls[0] += 1
            if calls[0] & 63 == 0 and clock() >= stop_time:
                return
            if best_values is not None:
                tolerance = 1e-10 * (1.0 + abs(best_cost))
                if partial_cost + remaining_bound >= best_cost - tolerance:
                    return
            if not unassigned:
                best_cost = partial_cost
                best_values = {f: (departures[f], arrivals[f]) for f in subset}
                if first_feasible:
                    stopped[0] = True
                return

            chosen = None
            viable_chosen = None
            chosen_key = None
            best_size = 10 ** 30
            for f in list(unassigned):
                viable = []
                checked = 0
                for item in domains[f]:
                    checked += 1
                    if checked & 127 == 0 and clock() >= stop_time:
                        return
                    departure = item[2]
                    arrival = item[3]
                    if not links_allow(f, departure, arrival):
                        continue
                    if pair_allows(
                        dep_counts, arr_counts, w, f, departure, arrival
                    ):
                        viable.append(item)
                        if len(viable) > best_size:
                            break
                if not viable:
                    return
                key = (
                    len(viable),
                    -(len(aircraft_predecessors[f]) + len(aircraft_successors[f])),
                    -criticality[f],
                )
                if chosen_key is None or key < chosen_key:
                    chosen = f
                    viable_chosen = viable
                    chosen_key = key
                    best_size = len(viable)

            if chosen is None:
                return
            unassigned.remove(chosen)
            new_bound = remaining_bound - minimum_cost[chosen]
            for item in viable_chosen:
                if stopped[0] or clock() >= stop_time:
                    break
                cost = item[1]
                departure = item[2]
                arrival = item[3]
                if best_values is not None:
                    tolerance = 1e-10 * (1.0 + abs(best_cost))
                    if partial_cost + cost + new_bound >= best_cost - tolerance:
                        continue
                departures[chosen] = departure
                arrivals[chosen] = arrival
                change_counts(
                    dep_counts, arr_counts, chosen, departure, arrival, 1
                )
                search(partial_cost + cost, new_bound)
                change_counts(
                    dep_counts, arr_counts, chosen, departure, arrival, -1
                )
                departures[chosen] = 0
                arrivals[chosen] = 0
            unassigned.add(chosen)

        try:
            search(0.0, sum(minimum_cost.values()))
        except BaseException:
            pass
        if best_values is None:
            return None
        if incumbent_feasible:
            tolerance = 1e-10 * (1.0 + abs(old_subset_cost))
            if best_cost >= old_subset_cost - tolerance:
                return None
        result_departures = old_departures[:]
        result_arrivals = old_arrivals[:]
        for f, pair in best_values.items():
            result_departures[f], result_arrivals[f] = pair
        return [result_departures, result_arrivals]

    def optimize_plan(plan, operations, stop_time):
        if clock() >= stop_time:
            return plan[:]
        try:
            low = lower[:]
            high = upper[:]
            for f in range(n):
                for w in range(scenario_count):
                    departure = operations[w][0][f]
                    arrival = operations[w][1][f]
                    low[f] = max(
                        low[f],
                        departure - max_dep_delay[f],
                        arrival - duration[f] - max_arr_delay[f],
                    )
                    high[f] = min(high[f], departure)
            for f in schedule_order:
                for g, lag in schedule_successors[f]:
                    low[g] = max(low[g], low[f] + lag)
            for f in reversed(schedule_order):
                for g, lag in schedule_successors[f]:
                    high[f] = min(high[f], high[g] - lag)
            if any(low[f] > high[f] for f in range(n)):
                return plan[:]

            def unary(f, value):
                result = rho * abs(value - requested[f])
                factor = 1.0 - rho
                if factor != 0.0:
                    for w in range(scenario_count):
                        departure = operations[w][0][f]
                        arrival = operations[w][1][f]
                        runtime = dep_cost[f] * (departure - value)
                        runtime += arr_cost[f] * max(
                            0, arrival - value - duration[f]
                        )
                        result += factor * probabilities[w] * runtime
                return result

            node_id = [{} for _ in range(n)]
            weights = []
            for f in range(n):
                previous = unary(f, low[f])
                for value in range(low[f] + 1, high[f] + 1):
                    if len(weights) & 1023 == 0 and clock() >= stop_time:
                        return plan[:]
                    current = unary(f, value)
                    node_id[f][value] = len(weights)
                    weights.append(current - previous)
                    previous = current

            node_count = len(weights)
            if node_count == 0:
                candidate = low[:]
                if plan_valid_with_operations(candidate, operations):
                    return candidate
                return plan[:]

            source = node_count
            sink = node_count + 1
            graph = [[] for _ in range(node_count + 2)]

            def add_edge(left, right, capacity):
                graph[left].append([right, len(graph[right]), capacity])
                graph[right].append([left, len(graph[left]) - 1, 0.0])

            infinity = sum(abs(weight) for weight in weights) + 1.0
            for node, weight in enumerate(weights):
                if weight < 0.0:
                    add_edge(source, node, -weight)
                elif weight > 0.0:
                    add_edge(node, sink, weight)
            for f in range(n):
                for value in range(low[f] + 2, high[f] + 1):
                    add_edge(
                        node_id[f][value], node_id[f][value - 1], infinity
                    )
                for g, lag in schedule_successors[f]:
                    for value in range(low[f] + 1, high[f] + 1):
                        target = value + lag
                        if target <= low[g]:
                            continue
                        if target > high[g]:
                            add_edge(node_id[f][value], sink, infinity)
                        else:
                            add_edge(
                                node_id[f][value], node_id[g][target], infinity
                            )
                if clock() >= stop_time:
                    return plan[:]

            level = [-1] * len(graph)
            pointer = [0] * len(graph)
            expired = [False]
            ticks = [0]

            def bfs():
                for i in range(len(level)):
                    level[i] = -1
                level[source] = 0
                queue = [source]
                head = 0
                while head < len(queue):
                    vertex = queue[head]
                    head += 1
                    ticks[0] += 1
                    if ticks[0] & 511 == 0 and clock() >= stop_time:
                        expired[0] = True
                        return False
                    for edge in graph[vertex]:
                        if edge[2] > 1e-12 and level[edge[0]] < 0:
                            level[edge[0]] = level[vertex] + 1
                            queue.append(edge[0])
                return level[sink] >= 0

            def send(vertex, pushed):
                if vertex == sink:
                    return pushed
                ticks[0] += 1
                if ticks[0] & 511 == 0 and clock() >= stop_time:
                    expired[0] = True
                    return 0.0
                while pointer[vertex] < len(graph[vertex]):
                    edge = graph[vertex][pointer[vertex]]
                    if edge[2] > 1e-12 and level[edge[0]] == level[vertex] + 1:
                        amount = send(edge[0], min(pushed, edge[2]))
                        if amount > 1e-12:
                            edge[2] -= amount
                            graph[edge[0]][edge[1]][2] += amount
                            return amount
                        if expired[0]:
                            return 0.0
                    pointer[vertex] += 1
                return 0.0

            while clock() < stop_time and bfs():
                if expired[0]:
                    return plan[:]
                for i in range(len(pointer)):
                    pointer[i] = 0
                while clock() < stop_time:
                    pushed = send(source, infinity)
                    if expired[0]:
                        return plan[:]
                    if pushed <= 1e-12:
                        break
            if expired[0] or clock() >= stop_time:
                return plan[:]

            reachable = [False] * len(graph)
            reachable[source] = True
            stack = [source]
            while stack:
                vertex = stack.pop()
                for edge in graph[vertex]:
                    if edge[2] > 1e-10 and not reachable[edge[0]]:
                        reachable[edge[0]] = True
                        stack.append(edge[0])

            candidate = low[:]
            for f in range(n):
                for value in range(low[f] + 1, high[f] + 1):
                    if reachable[node_id[f][value]]:
                        candidate[f] = value
            if plan_valid_with_operations(candidate, operations):
                return candidate
            return plan[:]
        except BaseException:
            return plan[:]

    def plan_bounds(plan, f):
        low = lower[f]
        high = upper[f]
        for p, lag in schedule_predecessors[f]:
            low = max(low, plan[p] + lag)
        for g, lag in schedule_successors[f]:
            high = min(high, plan[g] - lag)
        return low, high

    def flight_contribution(plan, operations, f):
        value = rho * abs(plan[f] - requested[f])
        for w in range(scenario_count):
            value += (1.0 - rho) * probabilities[w] * (
                dep_cost[f] * (operations[w][0][f] - plan[f])
                + arr_cost[f]
                * max(0, operations[w][1][f] - plan[f] - duration[f])
            )
        return value

    def choose_flight(plan, operations):
        if n <= 1:
            return 0
        sample = [rng.randrange(n) for _ in range(min(11, n))]
        if rng.random() < 0.18:
            return rng.choice(sample)
        return max(sample, key=lambda f: flight_contribution(plan, operations, f))

    def joint_flight_move(plan, operations, f, stop_time):
        if clock() >= stop_time:
            return False
        low, high = plan_bounds(plan, f)
        if low > high:
            return False
        alternatives = []
        for w in range(scenario_count):
            dc, ac = empty_counts()
            for g in range(n):
                if g != f:
                    change_counts(dc, ac, g, operations[w][0][g], operations[w][1][g], 1)
            first = low
            for g, lag in aircraft_predecessors[f]:
                first = max(first, operations[w][1][g] + lag)
            last_arrival = min(horizon, high + duration[f] + max_arr_delay[f])
            for g, lag in aircraft_successors[f]:
                last_arrival = min(last_arrival, operations[w][0][g] - lag)
            last = min(horizon, high + max_dep_delay[f], last_arrival - min_duration[f])
            choices = []
            for d in range(first, last + 1):
                if clock() >= stop_time:
                    return False
                for a in range(max(1, d + min_duration[f]),
                               min(last_arrival, d + max_duration[f]) + 1):
                    if pair_allows(dc, ac, w, f, d, a):
                        choices.append((d, a))
                        break
            if not choices:
                return False
            alternatives.append(choices)

        old_value = flight_contribution(plan, operations, f)
        best_value, best_schedule, best_pairs = old_value, plan[f], None
        old_times = sum(operations[w][0][f] + operations[w][1][f] for w in range(scenario_count))
        best_times = old_times
        for value in range(low, high + 1):
            if clock() >= stop_time:
                break
            total = rho * abs(value - requested[f])
            pairs = []
            complete = True
            for w, choices in enumerate(alternatives):
                best_runtime = math.inf
                best_pair = None
                for d, a in choices:
                    if d < value or d > value + max_dep_delay[f] or a > value + duration[f] + max_arr_delay[f]:
                        continue
                    cost = dep_cost[f] * (d - value) + arr_cost[f] * max(0, a - value - duration[f])
                    if cost < best_runtime - 1e-12:
                        best_runtime = cost
                        best_pair = d, a
                if best_pair is None:
                    complete = False
                    break
                pairs.append(best_pair)
                total += (1.0 - rho) * probabilities[w] * best_runtime
            if complete:
                times = sum(d + a for d, a in pairs)
                tolerance = 1e-10 * (1.0 + abs(best_value))
                if total < best_value - tolerance or (abs(total - best_value) <= tolerance and times < best_times):
                    best_value, best_schedule, best_pairs, best_times = total, value, pairs, times
        if best_pairs is None:
            return False
        plan[f] = best_schedule
        for w in range(scenario_count):
            operations[w][0][f], operations[w][1][f] = best_pairs[w]
        return True

    # Event indices: scheduled departure, plus actual departure and actual arrival for each scenario.
    event_count = n * (1 + 2 * scenario_count)
    temporal_links = []
    for f in range(n):
        for g, lag in schedule_successors[f]:
            temporal_links.append((f, g, lag))
    for w in range(scenario_count):
        doff = n * (1 + 2 * w)
        aoff = doff + n
        for f in range(n):
            temporal_links.extend((
                (f, doff + f, 0),
                (doff + f, f, -max_dep_delay[f]),
                (doff + f, aoff + f, min_duration[f]),
                (aoff + f, doff + f, -max_duration[f]),
                (aoff + f, f, -duration[f] - max_arr_delay[f]),
            ))
            for g, lag in aircraft_successors[f]:
                temporal_links.append((aoff + f, doff + g, lag))
    incident_links = [[] for _ in range(event_count)]
    for k, (u, v, lag) in enumerate(temporal_links):
        incident_links[u].append(k)
        incident_links[v].append(k)

    def joint_milp(plan, operations, subset, stop_time, radius=None,
                   recourse_scenario=None, stride=None):
        """Fix events outside the neighbourhood and jointly optimise the schedule and all scenarios inside it; without an initial solution, solve for a global feasible solution."""
        if milp is None or clock() >= stop_time:
            return None
        try:
            inside = set(subset)
            if not inside:
                return None
            free_events = set()
            for f in inside:
                if recourse_scenario is None:
                    free_events.add(f)
                for w in range(scenario_count):
                    if recourse_scenario is None or recourse_scenario == w:
                        free_events.add(n * (1 + 2 * w) + f)
                        free_events.add(n * (2 + 2 * w) + f)
            lower_time = [1] * event_count
            upper_time = [horizon] * event_count
            anchor = [0] * event_count
            fixed_counts = [empty_counts() for _ in range(scenario_count)]
            for f in range(n):
                lower_time[f] = earliest[f]
                upper_time[f] = latest[f]
                if plan is not None:
                    anchor[f] = plan[f]
                    if f not in free_events:
                        lower_time[f] = upper_time[f] = plan[f]
                    elif radius is not None:
                        lower_time[f] = max(lower_time[f], plan[f] - radius)
                        upper_time[f] = min(upper_time[f], plan[f] + radius)
                for w in range(scenario_count):
                    d = n * (1 + 2 * w) + f
                    a = d + n
                    lower_time[d] = lower_time[f]
                    upper_time[d] = min(horizon, upper_time[f] + max_dep_delay[f])
                    lower_time[a] = max(1, lower_time[d] + min_duration[f])
                    upper_time[a] = min(horizon, upper_time[f] + duration[f] + max_arr_delay[f])
                    if operations is not None:
                        old_d, old_a = operations[w][0][f], operations[w][1][f]
                        anchor[d], anchor[a] = old_d, old_a
                        if d not in free_events:
                            lower_time[d] = upper_time[d] = old_d
                            lower_time[a] = upper_time[a] = old_a
                            change_counts(*fixed_counts[w], f, old_d, old_a, 1)
                        elif radius is not None:
                            lower_time[d] = max(lower_time[d], old_d - radius)
                            upper_time[d] = min(upper_time[d], old_d + radius)
                            lower_time[a] = max(lower_time[a], old_a - radius)
                            upper_time[a] = min(upper_time[a], old_a + radius)

            def propagate(allowed=None):
                queue = deque(range(event_count))
                queued = [True] * event_count
                while queue:
                    if clock() >= stop_time:
                        return False
                    e = queue.popleft()
                    queued[e] = False
                    if lower_time[e] > upper_time[e]:
                        return False
                    for k in incident_links[e]:
                        u, v, lag = temporal_links[k]
                        low = max(lower_time[v], lower_time[u] + lag)
                        high = min(upper_time[u], upper_time[v] - lag)
                        if allowed is not None:
                            j = bisect_left(allowed[v], low)
                            if j == len(allowed[v]):
                                return False
                            low = allowed[v][j]
                            j = bisect_left(allowed[u], high + 1) - 1
                            if j < 0:
                                return False
                            high = allowed[u][j]
                        if low > upper_time[v] or high < lower_time[u]:
                            return False
                        if low > lower_time[v]:
                            lower_time[v] = low
                            if not queued[v]:
                                queued[v] = True
                                queue.append(v)
                        if high < upper_time[u]:
                            upper_time[u] = high
                            if not queued[u]:
                                queued[u] = True
                                queue.append(u)
                return True

            if not propagate():
                return None
            supports = [None] * event_count
            for e in range(event_count):
                if clock() >= stop_time:
                    return None
                f = e % n
                if stride is not None and e in free_events and plan is not None:
                    values = sorted({anchor[e], anchor[e] - 1, anchor[e] + 1,
                                     anchor[e] - stride, anchor[e] + stride})
                    values = [t for t in values if lower_time[e] <= t <= upper_time[e]]
                else:
                    values = range(lower_time[e], upper_time[e] + 1)
                if e < n or e not in free_events:
                    supports[e] = list(values)
                else:
                    block = e // n - 1
                    w, arrival_event = block // 2, block % 2
                    airport = destination[f] if arrival_event else origin[f]
                    dc, ac = fixed_counts[w]
                    supports[e] = [t for t in values if slot_allows(
                        dc, ac, w, airport, t, 1 - arrival_event, arrival_event)]
                if not supports[e]:
                    return None
                lower_time[e] = supports[e][0]
                upper_time[e] = supports[e][-1]
            if not propagate(supports):
                return None
            for e in range(event_count):
                lo = bisect_left(supports[e], lower_time[e])
                hi = bisect_left(supports[e], upper_time[e] + 1)
                supports[e] = supports[e][lo:hi]

            costs, integrality, start_values = [], [], []
            row_indices, col_indices, coefficients = [], [], []
            row_lower, row_upper = [], []
            variables = [[] for _ in range(event_count)]

            def variable(cost, integer, start):
                index = len(costs)
                costs.append(float(cost))
                integrality.append(integer)
                start_values.append(float(start))
                return index

            def constraint(terms, high, low=-math.inf):
                row = len(row_lower)
                for col, coefficient in terms.items():
                    if coefficient:
                        row_indices.append(row)
                        col_indices.append(col)
                        coefficients.append(float(coefficient))
                row_lower.append(float(low))
                row_upper.append(float(high))

            for e, values in enumerate(supports):
                f = e % n
                ids = variables[e]
                for j in range(1, len(values)):
                    step = values[j] - values[j - 1]
                    cost = 0.0
                    if e < n:
                        cost = rho * (abs(values[j] - requested[f]) - abs(values[j - 1] - requested[f]))
                        cost -= (1.0 - rho) * sum(probabilities) * dep_cost[f] * step
                    elif (e // n - 1) % 2 == 0:
                        w = (e // n - 1) // 2
                        cost = (1.0 - rho) * probabilities[w] * dep_cost[f] * step
                    ids.append(variable(cost, 1, int(anchor[e] >= values[j])))
                    if len(ids) > 1:
                        constraint({ids[-1]: 1.0, ids[-2]: -1.0}, 0.0)
                if clock() >= stop_time:
                    return None

            def indicator(e, threshold):
                j = bisect_left(supports[e], threshold)
                if j == 0:
                    return -1, 1.0
                if j == len(supports[e]):
                    return -1, 0.0
                return variables[e][j - 1], 0.0

            # Cumulative-indicator implications avoid the weak relaxation obtained by linking only through average event times.
            for k, (u, v, lag) in enumerate(temporal_links):
                if k % 128 == 0 and clock() >= stop_time:
                    return None
                for t in supports[u]:
                    iu, cu = indicator(u, t)
                    iv, cv = indicator(v, t + lag)
                    if iu < 0 and cu == 0.0 or iv < 0 and cv == 1.0:
                        continue
                    terms = {}
                    if iu >= 0:
                        terms[iu] = 1.0
                    if iv >= 0:
                        terms[iv] = terms.get(iv, 0.0) - 1.0
                    if terms:
                        constraint(terms, cv - cu)
                    elif cu > cv:
                        return None

            # Arrival delay keeps a non-negative lower bound; early arrival is fully allowed.
            # Expand the positive part into the positive part of a cumulative-indicator difference to tighten the continuous relaxation.
            for f in inside:
                for w in range(scenario_count):
                    a = n * (2 + 2 * w) + f
                    first = supports[f][0] + duration[f] + 1
                    last = supports[a][-1]
                    if first > last:
                        continue
                    points = {first, last + 1}
                    points.update(t + 1 for t in supports[a] if first < t + 1 <= last)
                    points.update(t + duration[f] + 1 for t in supports[f]
                                  if first < t + duration[f] + 1 <= last)
                    points = sorted(points)
                    delay_terms = {}
                    delay_constant = 0.0
                    weight = (1.0 - rho) * probabilities[w] * arr_cost[f]
                    for left, right in zip(points, points[1:]):
                        length = right - left
                        ia, ca = indicator(a, left)
                        ip, cp = indicator(f, left - duration[f])
                        if ia < 0 and ca == 0.0 or ip < 0 and cp == 1.0:
                            continue
                        if ia < 0 and ip < 0:
                            delay_constant += length * max(0.0, ca - cp)
                        elif ip < 0:
                            costs[ia] += weight * length
                            delay_terms[ia] = delay_terms.get(ia, 0.0) + length
                        elif ia < 0:
                            costs[ip] -= weight * length
                            delay_terms[ip] = delay_terms.get(ip, 0.0) - length
                            delay_constant += length
                        else:
                            value = max(0.0, start_values[ia] - start_values[ip])
                            y = variable(weight * length, 0, value)
                            constraint({ia: 1.0, ip: -1.0, y: -1.0}, 0.0)
                            delay_terms[y] = float(length)
                    if delay_terms:
                        constraint(delay_terms, max_arr_delay[f] - delay_constant)
                    elif delay_constant > max_arr_delay[f]:
                        return None
                    if clock() >= stop_time:
                        return None

            # Each capacity line is applied to the total number of departure and arrival events at the same airport and period.
            for w in range(scenario_count):
                dc, ac = fixed_counts[w]
                slots = {}
                for arrival_event in (0, 1):
                    for f in inside:
                        airport = destination[f] if arrival_event else origin[f]
                        e = n * (1 + 2 * w + arrival_event) + f
                        if e not in free_events:
                            continue
                        values, ids = supports[e], variables[e]
                        for j, period in enumerate(values):
                            key = airport, period
                            if key not in slots:
                                slots[key] = [{}, {}, float(dc[airport][period]), float(ac[airport][period])]
                            entry = slots[key]
                            terms = entry[arrival_event]
                            if j == 0:
                                entry[2 + arrival_event] += 1.0
                            else:
                                z = ids[j - 1]
                                terms[z] = terms.get(z, 0.0) + 1.0
                            if j < len(ids):
                                z = ids[j]
                                terms[z] = terms.get(z, 0.0) - 1.0
                        if clock() >= stop_time:
                            return None
                for (airport, period), (dt, at, d0, a0) in slots.items():
                    for cd, ca, cap in model_capacities[w][airport][period]:
                        terms = {v: cd * c for v, c in dt.items() if cd * c}
                        for v, c in at.items():
                            if ca * c:
                                terms[v] = terms.get(v, 0.0) + ca * c
                        rhs = cap - cd * d0 - ca * a0
                        if terms:
                            constraint(terms, rhs)
                        elif rhs < -1e-8:
                            return None
                    if clock() >= stop_time:
                        return None

            if not costs:
                candidate_plan = [supports[f][0] for f in range(n)]
                candidate_operations = [
                    [[supports[n * (1 + 2 * w) + f][0] for f in range(n)],
                     [supports[n * (2 + 2 * w) + f][0] for f in range(n)]]
                    for w in range(scenario_count)
                ]
                if all_valid(candidate_plan, candidate_operations):
                    return candidate_plan, candidate_operations, True
                return None
            if plan is not None:
                upper_objective = sum(c * x for c, x in zip(costs, start_values))
                constraint({j: c for j, c in enumerate(costs) if c},
                           upper_objective + 1e-8 * (1.0 + abs(upper_objective)))
            now = clock()
            if now >= stop_time:
                return None
            matrix = coo_matrix((np.asarray(coefficients, dtype=float),
                                 (np.asarray(row_indices, dtype=np.int32),
                                  np.asarray(col_indices, dtype=np.int32))),
                                shape=(len(row_lower), len(costs))).tocsc()
            available = stop_time - clock()
            if available <= 0.0:
                return None
            result = milp(
                np.asarray(costs, dtype=float),
                integrality=np.asarray(integrality, dtype=np.int32),
                bounds=Bounds(np.zeros(len(costs)), np.ones(len(costs))),
                constraints=LinearConstraint(matrix, np.asarray(row_lower), np.asarray(row_upper)),
                options={"time_limit": available * 0.86, "mip_rel_gap": 0.0, "disp": False},
            )
            if result.x is None:
                return None
            solution = result.x
            times = []
            for e, values in enumerate(supports):
                value = values[0]
                for j, z in enumerate(variables[e]):
                    if abs(solution[z] - round(solution[z])) > 1e-5:
                        return None
                    if solution[z] > 0.5:
                        value += values[j + 1] - values[j]
                times.append(int(value))
            candidate_plan = times[:n]
            candidate_operations = [
                [times[n * (1 + 2 * w):n * (2 + 2 * w)],
                 times[n * (2 + 2 * w):n * (3 + 2 * w)]]
                for w in range(scenario_count)
            ]
            if all_valid(candidate_plan, candidate_operations):
                return candidate_plan, candidate_operations, bool(result.success)
        except Exception:
            pass
        return None
    def batch_retime(plan, operations, stop_time):
        """Fix the composition and order of airport batches and jointly move both stages' times within a single min-cut."""
        if clock() >= stop_time:
            return None
        try:
            mapping = list(range(n)) + [-1] * (event_count - n)
            anchors = plan[:]
            metadata = [None] * n
            lower_batch = earliest[:]
            upper_batch = latest[:]
            slopes = [-(1.0 - rho) * sum(probabilities) * dep_cost[f]
                      for f in range(n)]
            resource_edges = []
            for w in range(scenario_count):
                slots = {}
                for kind in (0, 1):
                    off = n * (1 + 2 * w + kind)
                    for f in range(n):
                        t = operations[w][kind][f]
                        airport = destination[f] if kind else origin[f]
                        slots.setdefault((airport, t), []).append((off + f, kind, f))
                previous_airport, previous_group = -1, -1
                for (airport, t), members in sorted(slots.items()):
                    g = len(anchors)
                    anchors.append(t)
                    nd = sum(1 - kind for _, kind, _ in members)
                    na = len(members) - nd
                    metadata.append((w, airport, nd, na))
                    lo, hi, slope = 1, horizon, 0.0
                    for e, kind, f in members:
                        mapping[e] = g
                        if kind:
                            lo = max(lo, earliest[f] + min_duration[f])
                            hi = min(hi, latest[f] + duration[f] + max_arr_delay[f],
                                     latest[f] + max_dep_delay[f] + max_duration[f])
                        else:
                            lo = max(lo, earliest[f])
                            hi = min(hi, latest[f] + max_dep_delay[f])
                            slope += (1.0 - rho) * probabilities[w] * dep_cost[f]
                    lower_batch.append(lo)
                    upper_batch.append(hi)
                    slopes.append(slope)
                    if airport == previous_airport:
                        resource_edges.append((previous_group, g, 1))
                    previous_airport, previous_group = airport, g
                if clock() >= stop_time:
                    return None
            m = len(anchors)
            # Control the size of this graph; a longer total budget and longer time slices allow a wider retiming range.
            remaining = max(0.0, stop_time - clock())
            work = 650.0 * (1.0 + math.log1p(budget)) * math.sqrt(1.0 + budget)
            work *= min(1.5, math.sqrt(remaining / max(1e-9, budget * 0.035)))
            radius = max(1, min(horizon, int(work / max(1, 2 * m))))
            supports = []
            for g in range(m):
                lo = max(lower_batch[g], anchors[g] - radius)
                hi = min(upper_batch[g], anchors[g] + radius)
                if g < n:
                    values = list(range(lo, hi + 1))
                else:
                    w, airport, nd, na = metadata[g]
                    values = [t for t in range(lo, hi + 1) if all(
                        cd * nd + ca * na <= cap + 1e-9 * (1.0 + abs(cap))
                        for cd, ca, cap in capacities[w][airport][t])]
                if not values or anchors[g] not in values:
                    return None
                supports.append(values)
                if clock() >= stop_time:
                    return None

            merged_edges = {}
            for u, v, lag in temporal_links:
                a, b = mapping[u], mapping[v]
                if a != b:
                    merged_edges[a, b] = max(lag, merged_edges.get((a, b), -math.inf))
                elif lag > 0:
                    return None
            for u, v, lag in resource_edges:
                merged_edges[u, v] = max(lag, merged_edges.get((u, v), -math.inf))
            edges = [(u, v, lag) for (u, v), lag in merged_edges.items()]
            incident = [[] for _ in range(m)]
            for k, (u, v, _) in enumerate(edges):
                incident[u].append(k)
                incident[v].append(k)
            lows = [s[0] for s in supports]
            highs = [s[-1] for s in supports]
            queue = deque(range(m))
            queued = [True] * m
            while queue:
                if clock() >= stop_time:
                    return None
                e = queue.popleft()
                queued[e] = False
                for k in incident[e]:
                    u, v, lag = edges[k]
                    lo = max(lows[v], lows[u] + lag)
                    hi = min(highs[u], highs[v] - lag)
                    j = bisect_left(supports[v], lo)
                    z = bisect_left(supports[u], hi + 1) - 1
                    if j == len(supports[v]) or z < 0:
                        return None
                    lo, hi = supports[v][j], supports[u][z]
                    if lo > highs[v] or hi < lows[u]:
                        return None
                    if lo > lows[v]:
                        lows[v] = lo
                        if not queued[v]:
                            queue.append(v)
                            queued[v] = True
                    if hi < highs[u]:
                        highs[u] = hi
                        if not queued[u]:
                            queue.append(u)
                            queued[u] = True
            weights, ids = [], []
            implications, soft_edges = [], []
            for g in range(m):
                values = supports[g]
                values = values[bisect_left(values, lows[g]):bisect_left(values, highs[g] + 1)]
                supports[g] = values
                cols = []
                for left, right in zip(values, values[1:]):
                    col = len(weights)
                    cols.append(col)
                    weight = slopes[g] * (right - left)
                    if g < n:
                        weight += rho * (abs(right - requested[g]) - abs(left - requested[g]))
                    weights.append(weight)
                    if len(cols) > 1:
                        implications.append((col, cols[-2]))
                ids.append(cols)

            def threshold(g, t):
                j = bisect_left(supports[g], t)
                if j == 0:
                    return -2  # Constant 1.
                if j == len(supports[g]):
                    return -1  # Constant 0.
                return ids[g][j - 1]

            for u, v, lag in edges:
                for t, col in zip(supports[u][1:], ids[u]):
                    other = threshold(v, t + lag)
                    if other >= 0:
                        implications.append((col, other))
                    elif other == -1:
                        return None  # Should not occur after upper-bound propagation.
                if clock() >= stop_time:
                    return None
            for w in range(scenario_count):
                for f in range(n):
                    weight = (1.0 - rho) * probabilities[w] * arr_cost[f]
                    if weight <= 0.0:
                        continue
                    g = mapping[n * (2 + 2 * w) + f]
                    first = supports[f][0] + duration[f] + 1
                    last = supports[g][-1]
                    if first > last:
                        continue
                    points = {first, last + 1}
                    points.update(t + 1 for t in supports[g] if first < t + 1 <= last)
                    points.update(t + duration[f] + 1 for t in supports[f]
                                  if first < t + duration[f] + 1 <= last)
                    points = sorted(points)
                    for left, right in zip(points, points[1:]):
                        ia = threshold(g, left)
                        ip = threshold(f, left - duration[f])
                        cost = weight * (right - left)
                        if ia == -1 or ip == -2:
                            continue
                        if ia == -2:
                            if ip >= 0:
                                weights[ip] -= cost
                        elif ip == -1:
                            weights[ia] += cost
                        elif ip >= 0:
                            soft_edges.append((ia, ip, cost))
                    if clock() >= stop_time:
                        return None
            selected = _timed_cut(weights, implications, soft_edges, stop_time)
            if selected is None:
                return None
            values = []
            for g in range(m):
                value = supports[g][0]
                for j, col in enumerate(ids[g]):
                    if selected[col]:
                        value = supports[g][j + 1]
                values.append(value)
            candidate_plan = values[:n]
            candidate_operations = [
                [[values[mapping[n * (1 + 2 * w) + f]] for f in range(n)],
                 [values[mapping[n * (2 + 2 * w) + f]] for f in range(n)]]
                for w in range(scenario_count)]
            if all_valid(candidate_plan, candidate_operations):
                if objective(candidate_plan, candidate_operations) <= objective(plan, operations) + 1e-9:
                    return candidate_plan, candidate_operations
        except Exception:
            pass
        return None

    assignment_turn = [0]

    def assignment_move(plan, operations, w, stop_time):
        """Exactly assign events at the same airport or operating pairs on the same route, preserving capacity usage in all periods."""
        if linear_sum_assignment is None or clock() >= stop_time:
            return None
        try:
            kind = assignment_turn[0] % 3
            assignment_turn[0] += 1
            departures, arrivals = operations[w]
            buckets = {}
            for f in range(n):
                key = origin[f] if kind == 0 else (destination[f] if kind == 1 else (origin[f], destination[f]))
                buckets.setdefault(key, []).append(f)
            groups = [g for g in buckets.values() if len(g) > 1]
            if not groups:
                return None
            group = rng.choices(groups, weights=[1.0 + sum(
                dep_cost[f] * (departures[f] - plan[f]) + arr_cost[f] *
                max(0, arrivals[f] - plan[f] - duration[f]) for f in g)
                for g in groups], k=1)[0]
            rng.shuffle(group)
            if kind == 2:
                independent, excluded = [], set()
                for f in group:
                    if f not in excluded:
                        independent.append(f)
                        excluded.update(g for g, _ in aircraft_predecessors[f])
                        excluded.update(g for g, _ in aircraft_successors[f])
                group = independent
            target = max(2, int(32.0 * (1.0 + math.log1p(budget)) *
                                    math.sqrt(max(1e-9, stop_time - clock()) /
                                              max(1e-9, budget * 0.006))))
            if len(group) > target:
                seed_f = max(group, key=lambda f: dep_cost[f] * (departures[f] - plan[f]) +
                             arr_cost[f] * max(0, arrivals[f] - plan[f] - duration[f]))
                group.sort(key=lambda f: abs(departures[f] - departures[seed_f]) +
                           abs(arrivals[f] - arrivals[seed_f]) + rng.random())
                group = group[:target]
            count = len(group)
            if count < 2:
                return None
            columns = group[:]
            rng.shuffle(columns)
            costs = np.full((count, count), math.inf)
            for i, f in enumerate(group):
                first_d = plan[f]
                last_a = min(horizon, plan[f] + duration[f] + max_arr_delay[f])
                for p, lag in aircraft_predecessors[f]:
                    first_d = max(first_d, arrivals[p] + lag)
                for g, lag in aircraft_successors[f]:
                    last_a = min(last_a, departures[g] - lag)
                for j, g in enumerate(columns):
                    d = departures[g] if kind != 1 else departures[f]
                    a = arrivals[g] if kind != 0 else arrivals[f]
                    if (first_d <= d <= plan[f] + max_dep_delay[f] and a <= last_a and
                            min_duration[f] <= a - d <= max_duration[f]):
                        costs[i, j] = dep_cost[f] * (d - plan[f]) + arr_cost[f] * max(0, a - plan[f] - duration[f])
                if clock() >= stop_time:
                    return None
            # Each row's original slot is always in the column set, so the current feasible solution is also feasible for the assignment model.
            rows, cols = linear_sum_assignment(costs)
            result = [departures[:], arrivals[:]]
            for i, j in zip(rows, cols):
                f, g = group[int(i)], columns[int(j)]
                if kind != 1:
                    result[0][f] = departures[g]
                if kind != 0:
                    result[1][f] = arrivals[g]
            if result != operations[w] and row_cost(plan, *result) <= row_cost(plan, departures, arrivals) + 1e-9:
                if row_valid(plan, w, result):
                    return result
        except Exception:
            pass
        return None

    def choose_scenario(plan, operations):
        if scenario_count <= 1:
            return 0
        weights = [probabilities[w] * (1.0 + row_cost(plan, *operations[w]))
                   for w in range(scenario_count)]
        if rng.random() < 0.2 or sum(weights) <= 0.0:
            return rng.randrange(scenario_count)
        return rng.choices(range(scenario_count), weights=weights, k=1)[0]

    def pack(plan, operations):
        sa = [plan[f] + duration[f] for f in range(n)]
        od = [operations[w][0][:] for w in scenario_mapping]
        oa = [operations[w][1][:] for w in scenario_mapping]
        return {
            "scheduled_departure_period": plan[:],
            "scheduled_arrival_period": sa,
            "operated_departure_period": od,
            "operated_arrival_period": oa,
            "departure_delay": [[float(od[w][f] - plan[f]) for f in range(n)]
                                for w in range(len(scenario_mapping))],
            "arrival_delay": [[float(max_arr_delay[f] if original_arr_cost[f] < 0.0
                                     else max(0, oa[w][f] - sa[f])) for f in range(n)]
                              for w in range(len(scenario_mapping))],
        }

    if n == 0:
        return pack([], [[[], []] for _ in range(scenario_count)])

    # First keep a solution verified against all constraints. If construction fails, the joint model may change the schedule.
    best_plan = None
    best_operations = None
    attempt = 0
    next_exact = started + budget * 0.12
    while best_plan is None and clock() < deadline:
        now = clock()
        if milp is not None and now >= next_exact:
            result = joint_milp(None, None, range(n),
                                now + 0.68 * (deadline - now))
            next_exact = clock() + budget * 0.08
            if result is not None:
                best_plan, best_operations = result[0], result[1]
                break
            continue
        mode = attempt % 7
        if mode == 0:
            candidate_plan = base_plan[:]
        elif mode == 1:
            candidate_plan = earliest[:]
        elif mode == 2:
            candidate_plan = latest[:]
        else:
            candidate_plan = random_plan(base_plan, now - started, mode >= 5)
            if candidate_plan is None:
                candidate_plan = base_plan[:]
        attempt += 1
        quantum = budget / (6.0 + math.sqrt(n + scenario_count + 1.0))
        stop = min(deadline, now + quantum)
        if milp is not None:
            stop = min(stop, next_exact)
        noise = 0.0 if attempt == 1 else min(0.6, 0.04 * math.log1p(attempt))
        candidate_operations = build_all_rows(candidate_plan, stop, noise)
        if candidate_operations is not None and all_valid(candidate_plan, candidate_operations):
            best_plan, best_operations = candidate_plan, candidate_operations

    if best_plan is None:
        return _fallback(instance)

    safe_plan, safe_operations = best_plan[:], copy_operations(best_operations)
    retained[:] = [pack(best_plan, best_operations)]
    row_cache.clear()
    best_plan = optimize_plan(best_plan, best_operations,
                              min(deadline, clock() + budget * 0.025))
    best_value = objective(best_plan, best_operations)
    retained[:] = [pack(best_plan, best_operations)]
    current_plan = best_plan[:]
    current_operations = copy_operations(best_operations)
    last_improvement = clock()
    effort = math.log1p(budget)
    variable_budget = 140.0 * (1.0 + effort) * math.sqrt(1.0 + budget)
    width_estimate = [
        max(1, latest[f] - earliest[f]) + scenario_count * (
            max(1, latest[f] - earliest[f] + max_dep_delay[f]) +
            max(1, latest[f] - earliest[f] + duration[f] + max_arr_delay[f] - min_duration[f]))
        for f in range(n)
    ]
    mean_width = sum(width_estimate) / max(1, n)
    local_order = sorted(range(n), key=lambda f: -flight_contribution(best_plan, best_operations, f))
    local_position = 0
    next_model = clock() + budget / (20.0 + math.sqrt(n * max(1, scenario_count)))
    model_turn = 0
    adaptation = [1.0, 1.0, 1.0]
    first_full = True
    next_retime = clock()

    try:
        while clock() < deadline:
            now = clock()
            stagnation = max(0.0, now - last_improvement)
            changed = False
            if now >= next_retime:
                quantum = budget / (12.0 + math.sqrt(max(1, event_count)) / 6.0)
                result = batch_retime(current_plan, current_operations,
                                      min(deadline, now + quantum))
                if result is not None:
                    current_plan, current_operations = result
                    changed = True
                next_retime = clock() + budget / (4.0 + effort)
            elif milp is not None and now >= next_model:
                kind = model_turn % 3
                model_turn += 1
                recourse, stride, radius = None, None, None
                if first_full and sum(width_estimate) <= variable_budget * 1.8:
                    subset = list(range(n))
                    kind = 1
                elif kind == 0:
                    # All flights enter a narrow time window simultaneously, rather than solving only a few flights over their full time domain.
                    scale = 1.0 + math.sqrt(stagnation / max(budget, 1e-9))
                    radius = max(1, int(scale * (1.0 + effort / 3.0)))
                    if model_turn % 2:
                        radius = 1
                    elif stagnation > budget / (2.0 + effort):
                        stride = max(2, int(math.sqrt(1.0 + budget + stagnation)))
                        radius = stride
                    width = (4 if stride is not None else 2 * radius) * (1 + 2 * scenario_count)
                    target = min(n, max(2, int(variable_budget * adaptation[kind] / max(1, width))))
                    if target >= n:
                        subset = list(range(n))
                    elif scenario_count:
                        w = choose_scenario(current_plan, current_operations)
                        subset = related_subset(current_plan, *current_operations[w], target)
                    else:
                        subset = rng.sample(range(n), target)
                elif kind == 2 and scenario_count:
                    # Release only one scenario, but allow more flights to use their full operating time domain.
                    recourse = choose_scenario(current_plan, current_operations)
                    recourse_width = sum(max_dep_delay[f] + duration[f] + max_arr_delay[f] - min_duration[f] + 2 for f in range(n)) / max(1, n)
                    target = min(n, max(2, int(variable_budget * adaptation[kind] / max(1.0, recourse_width))))
                    subset = (list(range(n)) if target >= n else related_subset(
                        current_plan, *current_operations[recourse], target))
                else:
                    kind = 1
                    target = min(n, max(2, int(adaptation[kind] * variable_budget / max(1, mean_width) *
                                             (1.0 + math.sqrt(stagnation / max(budget, 1e-9))))))
                    if target >= n:
                        subset = list(range(n))
                    elif scenario_count:
                        w = choose_scenario(current_plan, current_operations)
                        f = choose_flight(current_plan, current_operations)
                        subset = related_subset(current_plan, *current_operations[w], target, [f])
                    else:
                        subset = rng.sample(range(n), target)
                first_full = False
                quantum = budget / (6.0 + math.sqrt(1.0 + n / max(1, len(subset))))
                if kind == 0:
                    quantum /= 1.0 + 0.15 * effort
                result = joint_milp(current_plan, current_operations, subset,
                                    min(deadline, now + quantum), radius, recourse, stride)
                if result is not None:
                    candidate_plan, candidate_operations = result[0], result[1]
                    candidate_plan = optimize_plan(candidate_plan, candidate_operations,
                                                   min(deadline, clock() + budget * 0.012))
                    candidate_value = objective(candidate_plan, candidate_operations)
                    current_value = objective(current_plan, current_operations)
                    if candidate_value <= current_value + 1e-9:
                        current_plan, current_operations = candidate_plan, candidate_operations
                        changed = True
                    adaptation[kind] *= 1.12 if result[2] else 0.98
                else:
                    adaptation[kind] *= 0.78
                adaptation[kind] = _clip(adaptation[kind], 0.25, 1.0 + effort)
                next_model = clock() + quantum / (8.0 + effort)
            else:
                if local_position >= n:
                    rng.shuffle(local_order)
                    local_position = 0
                f = local_order[local_position]
                local_position += 1
                quantum = budget / (100.0 + math.sqrt(n * max(1, scenario_count)))
                changed = joint_flight_move(current_plan, current_operations, f,
                                             min(deadline, now + quantum))
                if not changed and scenario_count and clock() < deadline:
                    w = choose_scenario(current_plan, current_operations)
                    result = assignment_move(current_plan, current_operations, w,
                                             min(deadline, clock() + quantum))
                    if result is None and rng.random() < 1.0 / (2.0 + effort):
                        target = min(n, max(2, int(1.0 + effort + math.sqrt(stagnation))))
                        subset = related_subset(current_plan, *current_operations[w], target, [f])
                        result = optimize_subset(current_plan, w, *current_operations[w], subset,
                                                 min(deadline, clock() + quantum), noise=0.0)
                    if result is not None:
                        current_operations[w] = result
                        changed = True
            if changed:
                if clock() < deadline:
                    current_plan = optimize_plan(current_plan, current_operations,
                                                 min(deadline, clock() + budget * 0.01))
                value = objective(current_plan, current_operations)
                if value < best_value - 1e-10 * (1.0 + abs(best_value)):
                    # Fully verify before updating the saved point; an exception never loses the existing feasible solution.
                    if all_valid(current_plan, current_operations):
                        best_plan = current_plan[:]
                        best_operations = copy_operations(current_operations)
                        best_value = value
                        retained[:] = [pack(best_plan, best_operations)]
                        last_improvement = clock()
    except Exception:
        pass

    if not all_valid(best_plan, best_operations):
        best_plan, best_operations = safe_plan, safe_operations
    return pack(best_plan, best_operations)


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.monotonic()
    retained = []
    try:
        return _solve_core(instance, time_limit_s, started, retained)
    except BaseException:
        return retained[0] if retained else _fallback(instance)
