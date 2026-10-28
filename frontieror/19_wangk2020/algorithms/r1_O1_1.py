"""Cumulative-period joint-neighbourhood hybrid search
Constructs a feasible solution by backtracking, then improves it continually with min-cut schedule optimisation and cross-scenario mixed-integer joint neighbourhoods.
"""
import math
import random
import time
from bisect import bisect_left
from collections import deque

try:
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
except Exception:
    milp = None


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


def _solve_core(instance, time_limit_s, started):
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
    arr_cost = [float(x) for x in instance["arrival_delay_cost"]]
    displacement = int(instance["max_schedule_displacement"])
    probabilities = [float(x) for x in instance["scenario_probabilities"]]
    rho = float(instance["rho"])

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

    weather = instance["weather_condition"]
    capacity_dep = instance["capacity_departure_coefficients"]
    capacity_arr = instance["capacity_arrival_coefficients"]
    capacity_limit = instance["capacity_limits"]
    capacities = []
    scenario_difficulty = []
    for w in range(scenario_count):
        rows = [[()] * (horizon + 1) for _ in range(airport_count)]
        difficulty = 0.0
        for airport in range(airport_count):
            for period in range(1, horizon + 1):
                condition = int(weather[w][airport][period - 1])
                difficulty += condition
                rows[airport][period] = tuple(
                    (
                        float(capacity_dep[condition][airport][line]),
                        float(capacity_arr[condition][airport][line]),
                        float(capacity_limit[condition][airport][line]),
                    )
                    for line in range(len(capacity_limit[condition][airport]))
                )
        capacities.append(rows)
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

    def build_all_rows(plan, stop_time, noise):
        operations = [None] * scenario_count
        for position, w in enumerate(scenario_order):
            now = clock()
            if now >= stop_time:
                return None
            left = scenario_count - position
            local_end = now + max(0.0, stop_time - now) / max(1, left)
            operation = construct_row(plan, w, local_end, noise)
            if operation is None:
                return None
            operations[w] = operation
        return operations

    def related_subset(plan, departures, arrivals, target, mandatory=None):
        if n == 0:
            return []
        selected = set(mandatory or [])
        target = min(n, max(1, target, len(selected)))
        events = {}
        for f in range(n):
            events.setdefault((origin[f], departures[f]), []).append(f)
            events.setdefault((destination[f], arrivals[f]), []).append(f)

        if not selected:
            candidates = [rng.randrange(n) for _ in range(min(9, n))]
            if rng.random() < 0.18:
                selected.add(rng.choice(candidates))
            else:
                selected.add(max(
                    candidates,
                    key=lambda f: (
                        dep_cost[f] * (departures[f] - plan[f])
                        + arr_cost[f]
                        * max(0, arrivals[f] - plan[f] - duration[f])
                        + 0.02
                        * (
                            len(events.get((origin[f], departures[f]), ()))
                            + len(events.get((destination[f], arrivals[f]), ()))
                        )
                    ),
                ))

        frontier = list(selected)
        head = 0
        radius = max(1, int(math.sqrt(target) / 2.0))
        while len(selected) < target:
            pool = set()
            if head < len(frontier):
                f = frontier[head]
                head += 1
                for g, _ in aircraft_predecessors[f] + schedule_predecessors[f]:
                    if g not in selected:
                        pool.add(g)
                for g, _ in aircraft_successors[f] + schedule_successors[f]:
                    if g not in selected:
                        pool.add(g)
                for airport, period in (
                    (origin[f], departures[f]),
                    (destination[f], arrivals[f]),
                ):
                    for t in range(
                        max(1, period - radius), min(horizon, period + radius) + 1
                    ):
                        for g in events.get((airport, t), ()):
                            if g not in selected:
                                pool.add(g)
            if pool:
                choices = list(pool)
                g = choices[rng.randrange(len(choices))]
            else:
                remaining = [f for f in range(n) if f not in selected]
                if not remaining:
                    break
                g = remaining[rng.randrange(len(remaining))]
            selected.add(g)
            frontier.append(g)
        return list(selected)

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

        base_counts = []
        for w in range(scenario_count):
            dep_counts, arr_counts = empty_counts()
            for g in range(n):
                if g != f:
                    change_counts(
                        dep_counts,
                        arr_counts,
                        g,
                        operations[w][0][g],
                        operations[w][1][g],
                        1,
                    )
            base_counts.append((dep_counts, arr_counts))

        old_value = flight_contribution(plan, operations, f)
        best_value = old_value
        best_schedule = plan[f]
        best_pairs = [
            (operations[w][0][f], operations[w][1][f])
            for w in range(scenario_count)
        ]
        values = list(range(low, high + 1))
        values.sort(key=lambda value: (
            abs(value - plan[f]), abs(value - requested[f]), rng.random()
        ))

        for scheduled_departure in values:
            if clock() >= stop_time:
                break
            scheduled_arrival = scheduled_departure + duration[f]
            candidate_value = rho * abs(scheduled_departure - requested[f])
            candidate_pairs = []
            complete = True
            for w in range(scenario_count):
                dep_counts, arr_counts = base_counts[w]
                best_runtime = float("inf")
                best_pair = None
                high_departure = min(
                    horizon, scheduled_departure + max_dep_delay[f]
                )
                for departure in range(scheduled_departure, high_departure + 1):
                    low_arrival = max(1, departure + min_duration[f])
                    high_arrival = min(
                        departure + max_duration[f],
                        horizon,
                        scheduled_arrival + max_arr_delay[f],
                    )
                    for arrival in range(low_arrival, high_arrival + 1):
                        valid = True
                        for p, connection in aircraft_predecessors[f]:
                            if departure < operations[w][1][p] + connection:
                                valid = False
                                break
                        if not valid:
                            continue
                        for g, connection in aircraft_successors[f]:
                            if operations[w][0][g] < arrival + connection:
                                valid = False
                                break
                        if not valid or not pair_allows(
                            dep_counts, arr_counts, w, f, departure, arrival
                        ):
                            continue
                        runtime = (
                            dep_cost[f] * (departure - scheduled_departure)
                            + arr_cost[f] * max(0, arrival - scheduled_arrival)
                        )
                        if runtime < best_runtime - 1e-12:
                            best_runtime = runtime
                            best_pair = (departure, arrival)
                        elif abs(runtime - best_runtime) <= 1e-12 and rng.random() < 0.08:
                            best_pair = (departure, arrival)
                    if clock() >= stop_time:
                        complete = False
                        break
                if not complete or best_pair is None:
                    complete = False
                    break
                candidate_pairs.append(best_pair)
                candidate_value += (
                    (1.0 - rho) * probabilities[w] * best_runtime
                )
            if complete and candidate_value < best_value - 1e-10 * (1.0 + abs(best_value)):
                best_value = candidate_value
                best_schedule = scheduled_departure
                best_pairs = candidate_pairs

        if best_value >= old_value - 1e-10 * (1.0 + abs(old_value)):
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

    def joint_milp(plan, operations, subset, stop_time, radius=None):
        """Fix events outside the neighbourhood and jointly optimise the schedule and all scenarios inside it; without an initial solution, solve for a global feasible solution."""
        if milp is None or clock() >= stop_time:
            return None
        try:
            inside = set(subset)
            if not inside:
                return None
            lower_time = [1] * event_count
            upper_time = [horizon] * event_count
            anchor = [0] * event_count
            fixed_counts = [empty_counts() for _ in range(scenario_count)]
            for f in range(n):
                lower_time[f] = earliest[f]
                upper_time[f] = latest[f]
                if plan is not None:
                    anchor[f] = plan[f]
                    if f not in inside:
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
                        if f not in inside:
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
                if e < n or f not in inside:
                    supports[e] = list(range(lower_time[e], upper_time[e] + 1))
                else:
                    block = e // n - 1
                    w, arrival_event = block // 2, block % 2
                    airport = destination[f] if arrival_event else origin[f]
                    dc, ac = fixed_counts[w]
                    supports[e] = [
                        t for t in range(lower_time[e], upper_time[e] + 1)
                        if slot_allows(dc, ac, w, airport, t,
                                       1 - arrival_event, arrival_event)
                    ]
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
                    for cd, ca, cap in capacities[w][airport][period]:
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
                options={"time_limit": available * 0.92, "mip_rel_gap": 0.0, "disp": False},
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
        od = [row[0][:] for row in operations]
        oa = [row[1][:] for row in operations]
        return {
            "scheduled_departure_period": plan[:],
            "scheduled_arrival_period": sa,
            "operated_departure_period": od,
            "operated_arrival_period": oa,
            "departure_delay": [[float(od[w][f] - plan[f]) for f in range(n)]
                                for w in range(scenario_count)],
            "arrival_delay": [[float(max(0, oa[w][f] - sa[f])) for f in range(n)]
                              for w in range(scenario_count)],
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
    best_plan = optimize_plan(best_plan, best_operations,
                              min(deadline, clock() + budget * 0.025))
    best_value = objective(best_plan, best_operations)
    current_plan = best_plan[:]
    current_operations = copy_operations(best_operations)
    last_improvement = clock()
    next_joint = clock()
    effort = math.log1p(budget)
    variable_budget = 140.0 * (1.0 + effort) * math.sqrt(1.0 + budget)
    width_estimate = [
        max(1, latest[f] - earliest[f]) + scenario_count * (
            max(1, latest[f] - earliest[f] + max_dep_delay[f])
            + max(1, latest[f] - earliest[f] + duration[f]
                  + max_arr_delay[f] - min_duration[f]))
        for f in range(n)
    ]
    mean_width = sum(width_estimate) / max(1, n)
    first_joint = True
    adaptation = 1.0
    local_order = list(range(n))
    rng.shuffle(local_order)
    local_position = 0

    try:
        while clock() < deadline:
            now = clock()
            remaining = deadline - now
            stagnation = max(0.0, now - last_improvement)
            progress = max(0.0, now - started) / max(budget, 1e-9)
            changed = False
            if milp is not None and now >= next_joint:
                size = min(n, max(2, int(
                    adaptation * variable_budget / max(1.0, mean_width)
                    * (0.75 + rng.random() * 0.5)
                    * (1.0 + math.sqrt(stagnation / max(budget, 1e-9))))))
                radius = max(1, int(math.sqrt(1.0 + budget)
                                    * (1.0 + progress)
                                    + math.sqrt(stagnation)))
                global_move = first_joint and sum(width_estimate) <= variable_budget * 2.5
                if global_move or size >= n:
                    subset = list(range(n))
                    radius = None
                else:
                    w = choose_scenario(current_plan, current_operations)
                    f = choose_flight(current_plan, current_operations)
                    if scenario_count:
                        subset = related_subset(current_plan, *current_operations[w], size, [f])
                    else:
                        subset = rng.sample(range(n), size)
                first_joint = False
                quantum = budget / (5.0 + math.sqrt(1.0 + n / max(1, len(subset))))
                quantum *= 1.0 + 0.6 * math.sqrt(stagnation / max(budget, 1e-9))
                result = joint_milp(current_plan, current_operations, subset,
                                    min(deadline, now + quantum), radius)
                if result is not None:
                    candidate_plan, candidate_operations = result[0], result[1]
                    candidate_plan = optimize_plan(
                        candidate_plan, candidate_operations,
                        min(deadline, clock() + budget * 0.018))
                    candidate_value = objective(candidate_plan, candidate_operations)
                    current_value = objective(current_plan, current_operations)
                    if candidate_value <= current_value + 1e-9:
                        current_plan, current_operations = candidate_plan, candidate_operations
                        changed = True
                    adaptation = min(2.0, adaptation * (1.12 if result[2] else 1.025))
                else:
                    adaptation = max(0.35, adaptation * 0.8)
                next_joint = clock() + quantum * 0.18
            elif scenario_count and rng.random() < 0.62:
                w = choose_scenario(current_plan, current_operations)
                size = min(n, max(2, int(2.0 + effort + math.sqrt(stagnation))))
                subset = related_subset(current_plan, *current_operations[w], size)
                quantum = budget / (90.0 + math.sqrt(n * scenario_count + 1.0))
                result = optimize_subset(
                    current_plan, w, *current_operations[w], subset,
                    min(deadline, now + quantum),
                    incumbent_feasible=True, first_feasible=False,
                    noise=0.002 * effort,
                )
                if result is not None:
                    current_operations[w] = result
                    changed = True
            else:
                if local_position >= n:
                    rng.shuffle(local_order)
                    local_position = 0
                f = local_order[local_position]
                local_position += 1
                quantum = budget / (90.0 + math.sqrt(n * max(1, scenario_count) + 1.0))
                changed = joint_flight_move(current_plan, current_operations, f,
                                             min(deadline, now + quantum))

            if changed:
                if clock() < deadline:
                    current_plan = optimize_plan(
                        current_plan, current_operations,
                        min(deadline, clock() + budget * 0.012))
                value = objective(current_plan, current_operations)
                if value < best_value - 1e-10 * (1.0 + abs(best_value)):
                    best_plan, best_operations = current_plan[:], copy_operations(current_operations)
                    best_value = value
                    last_improvement = clock()
    except Exception:
        pass

    if not all_valid(best_plan, best_operations):
        best_plan, best_operations = safe_plan, safe_operations
    return pack(best_plan, best_operations)


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.monotonic()
    try:
        return _solve_core(instance, time_limit_s, started)
    except Exception:
        return _fallback(instance)
