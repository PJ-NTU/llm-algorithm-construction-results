import bisect
import math
import random
import time


def _num(value, default=0.0):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except BaseException:
        pass
    return default


def _pos_int(value, default=1):
    try:
        value = int(value)
        if value >= 1:
            return value
    except BaseException:
        pass
    return default


def _nested(value, indices, default=0.0):
    try:
        for index in indices:
            value = value[index]
        return _num(value, default)
    except BaseException:
        return default


def _fallback(instance):
    try:
        source = instance if isinstance(instance, dict) else {}
        H = _pos_int(source.get("num_periods", 1), 1)
        N = _pos_int(source.get("num_customers", 1), 1)
        M = _pos_int(source.get("num_vehicles", 1), 1)
        S = _pos_int(source.get("num_scenarios", 1), 1)
        raw_initial = source.get("initial_inventories", [])
        raw_demand = source.get("demand_scenarios", [])
        initial = [
            max(0.0, _nested(raw_initial, (v,), 0.0))
            for v in range(N + 1)
        ]

        production = [[0.0 for _ in range(H)] for _ in range(S)]
        delivery = [
            [
                [[0.0 for _ in range(N)] for _ in range(M)]
                for _ in range(H)
            ]
            for _ in range(S)
        ]
        inventories = [
            [[0.0 for _ in range(N + 1)] for _ in range(H)]
            for _ in range(S)
        ]
        unmet = [
            [[0.0 for _ in range(N)] for _ in range(H)]
            for _ in range(S)
        ]

        for s in range(S):
            previous = initial[:]
            for t in range(H):
                ending = [0.0 for _ in range(N + 1)]
                ending[0] = previous[0]
                for c in range(N):
                    d = max(0.0, _nested(raw_demand, (s, t, c), 0.0))
                    served = min(previous[c + 1], d)
                    ending[c + 1] = max(0.0, previous[c + 1] - served)
                    unmet[s][t][c] = max(0.0, d - served)
                inventories[s][t] = ending
                previous = ending

        return {
            "problem_type": "multi_period_stochastic_production_routing_with_lost_sales",
            "production_setup": [0 for _ in range(H)],
            "vehicle_routes": [[[0] for _ in range(M)] for _ in range(H)],
            "production_quantities": production,
            "delivery_quantities": delivery,
            "end_inventories": inventories,
            "unmet_demand_quantities": unmet,
        }
    except BaseException:
        return {
            "problem_type": "multi_period_stochastic_production_routing_with_lost_sales",
            "production_setup": [0],
            "vehicle_routes": [[[0]]],
            "production_quantities": [[0.0]],
            "delivery_quantities": [[[[0.0]]]],
            "end_inventories": [[[0.0, 0.0]]],
            "unmet_demand_quantities": [[[0.0]]],
        }


def _solve(instance, time_limit_s, started, fallback_solution, fallback_elapsed):
    budget = _num(time_limit_s, 0.0)
    if budget < 0.0:
        budget = 0.0
    hard_deadline = started + budget
    reserve = max(0.007, 2.3 * fallback_elapsed + 0.004)
    deadline = hard_deadline - reserve
    if time.perf_counter() >= deadline:
        return fallback_solution

    H = int(instance["num_periods"])
    N = int(instance["num_customers"])
    M = int(instance["num_vehicles"])
    S = int(instance["num_scenarios"])
    if H < 1 or N < 1 or M < 1 or S < 1:
        return fallback_solution

    demand = [
        [
            [max(0.0, float(instance["demand_scenarios"][s][t][c]))
             for c in range(N)]
            for t in range(H)
        ]
        for s in range(S)
    ]
    probabilities = [
        max(0.0, float(instance["scenario_probabilities"][s]))
        for s in range(S)
    ]
    probability_sum = sum(probabilities)
    if probability_sum <= 0.0:
        return fallback_solution
    probabilities = [p / probability_sum for p in probabilities]

    initial = [max(0.0, float(x)) for x in instance["initial_inventories"]]
    capacities = [
        max(0.0, float(x)) for x in instance["inventory_capacities"]
    ]
    holding = [max(0.0, float(x)) for x in instance["holding_costs"]]
    penalty = [
        max(0.0, float(x)) for x in instance["unmet_demand_penalties"]
    ]
    transport = [
        [max(0.0, float(instance["transportation_costs"][i][j]))
         for j in range(N + 1)]
        for i in range(N + 1)
    ]
    production_capacity = max(0.0, float(instance["production_capacity"]))
    vehicle_capacity = max(0.0, float(instance["vehicle_capacity"]))
    setup_cost = max(0.0, float(instance["setup_cost"]))
    production_cost = max(0.0, float(instance["unit_production_cost"]))

    if (len(initial) != N + 1 or len(capacities) != N + 1 or
            len(holding) != N + 1 or len(penalty) != N or
            vehicle_capacity <= 0.0):
        return fallback_solution
    for v in range(N + 1):
        if initial[v] > capacities[v] + 1e-8:
            return fallback_solution

    eps = 1e-10
    suffix = [
        [[0.0 for _ in range(N)] for _ in range(H + 1)]
        for _ in range(S)
    ]
    for s in range(S):
        for t in range(H - 1, -1, -1):
            for c in range(N):
                suffix[s][t][c] = demand[s][t][c] + suffix[s][t + 1][c]
        if time.perf_counter() >= deadline:
            return fallback_solution

    expected_demand = [[0.0 for _ in range(N)] for _ in range(H)]
    for s in range(S):
        p = probabilities[s]
        for t in range(H):
            for c in range(N):
                expected_demand[t][c] += p * demand[s][t][c]

    expected_suffix = [[0.0 for _ in range(N)] for _ in range(H + 1)]
    for t in range(H - 1, -1, -1):
        for c in range(N):
            expected_suffix[t][c] = (
                expected_demand[t][c] + expected_suffix[t + 1][c]
            )

    node_ids = [[-1 for _ in range(H)] for _ in range(S)]
    groups = []
    minimum_suffix = []
    for t in range(H):
        index = {}
        level = []
        for s in range(S):
            parent = node_ids[s][t - 1] if t > 0 else -1
            key = (parent, tuple(demand[s][t]))
            nid = index.get(key)
            if nid is None:
                nid = len(level)
                index[key] = nid
                level.append({
                    "parent": parent,
                    "members": [],
                    "mass": 0.0,
                    "demand": demand[s][t][:],
                })
            level[nid]["members"].append(s)
            level[nid]["mass"] += probabilities[s]
            node_ids[s][t] = nid
        groups.append(level)
        minimum_suffix.append([
            [min(suffix[s][t][c] for s in group["members"])
             for c in range(N)]
            for group in level
        ])
        if time.perf_counter() >= deadline:
            return fallback_solution

    seed_value = 2166136261
    for value in (H, N, M, S):
        seed_value ^= value & 0xffffffff
        seed_value = (seed_value * 16777619) & 0xffffffff
    for s in range(S):
        for t in range(H):
            for value in demand[s][t]:
                seed_value ^= int(round(value * 1000.0)) & 0xffffffff
                seed_value = (seed_value * 16777619) & 0xffffffff
    rng = random.Random(seed_value)

    route_cache = {}
    curve_cache = {}

    def route_cost(order):
        if not order:
            return 0.0
        value = transport[0][order[0]] + transport[order[-1]][0]
        for i in range(len(order) - 1):
            value += transport[order[i]][order[i + 1]]
        return value

    def two_opt(order, local_deadline):
        order = order[:]
        n = len(order)
        while n >= 3 and time.perf_counter() < local_deadline:
            best_delta = -eps
            best_i = -1
            best_j = -1
            checks = 0
            for i in range(n - 1):
                a = 0 if i == 0 else order[i - 1]
                b = order[i]
                for j in range(i + 1, n):
                    c = order[j]
                    d = 0 if j == n - 1 else order[j + 1]
                    delta = (
                        transport[a][c] + transport[b][d]
                        - transport[a][b] - transport[c][d]
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                    checks += 1
                    if (checks & 127) == 0 and time.perf_counter() >= local_deadline:
                        return order
            if best_i < 0:
                break
            order[best_i:best_j + 1] = reversed(order[best_i:best_j + 1])
        return order

    def relocate(order, local_deadline):
        order = order[:]
        n = len(order)
        while n >= 3 and time.perf_counter() < local_deadline:
            best = None
            checks = 0
            for i in range(n):
                node = order[i]
                left = 0 if i == 0 else order[i - 1]
                right = 0 if i == n - 1 else order[i + 1]
                removal = (
                    transport[left][right]
                    - transport[left][node] - transport[node][right]
                )
                reduced = order[:i] + order[i + 1:]
                for pos in range(n):
                    before = 0 if pos == 0 else reduced[pos - 1]
                    after = 0 if pos == n - 1 else reduced[pos]
                    delta = (
                        removal + transport[before][node]
                        + transport[node][after] - transport[before][after]
                    )
                    if delta < -eps and (best is None or delta < best[0]):
                        best = (delta, i, pos)
                    checks += 1
                    if (checks & 127) == 0 and time.perf_counter() >= local_deadline:
                        return order
            if best is None:
                break
            node = order.pop(best[1])
            order.insert(best[2], node)
        return order

    def initial_route(key, local_deadline):
        remaining = list(key)
        nearest = []
        current = 0
        while remaining:
            nxt = min(remaining, key=lambda v: (transport[current][v], v))
            nearest.append(nxt)
            remaining.remove(nxt)
            current = nxt
            if time.perf_counter() >= local_deadline:
                nearest.extend(remaining)
                remaining.clear()
        nearest = two_opt(nearest, local_deadline)
        best = nearest
        best_value = route_cost(best)

        if time.perf_counter() < local_deadline:
            remaining = list(key)
            insertion = []
            while remaining:
                choice = None
                for node in remaining:
                    for pos in range(len(insertion) + 1):
                        left = 0 if pos == 0 else insertion[pos - 1]
                        right = 0 if pos == len(insertion) else insertion[pos]
                        delta = (
                            transport[left][node] + transport[node][right]
                            - transport[left][right]
                        )
                        item = (delta, node, pos)
                        if choice is None or item < choice:
                            choice = item
                insertion.insert(choice[2], choice[1])
                remaining.remove(choice[1])
                if time.perf_counter() >= local_deadline:
                    insertion.extend(remaining)
                    remaining.clear()
            insertion = two_opt(insertion, local_deadline)
            value = route_cost(insertion)
            if value + eps < best_value:
                best = insertion
                best_value = value
        return best, best_value

    def exact_route(key, local_deadline):
        n = len(key)
        if n <= 1:
            return list(key), route_cost(key)
        size = 1 << n
        inf = float("inf")
        dp = [[inf for _ in range(n)] for _ in range(size)]
        parent = [[-1 for _ in range(n)] for _ in range(size)]
        for j in range(n):
            dp[1 << j][j] = transport[0][key[j]]
        for mask in range(1, size):
            if (mask & 127) == 0 and time.perf_counter() >= local_deadline:
                return None
            bits = mask
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                previous_mask = mask ^ bit
                if previous_mask:
                    previous_bits = previous_mask
                    best_value = inf
                    best_i = -1
                    while previous_bits:
                        previous_bit = previous_bits & -previous_bits
                        i = previous_bit.bit_length() - 1
                        value = dp[previous_mask][i] + transport[key[i]][key[j]]
                        if value < best_value:
                            best_value = value
                            best_i = i
                        previous_bits ^= previous_bit
                    dp[mask][j] = best_value
                    parent[mask][j] = best_i
                bits ^= bit
        full = size - 1
        last = min(
            range(n),
            key=lambda j: dp[full][j] + transport[key[j]][0],
        )
        value = dp[full][last] + transport[key[last]][0]
        reversed_order = []
        mask = full
        while last >= 0:
            reversed_order.append(key[last])
            previous = parent[mask][last]
            mask ^= 1 << last
            last = previous
        reversed_order.reverse()
        return reversed_order, value

    def get_route(customers, local_deadline, explore, elapsed):
        key = tuple(sorted(customers))
        cached = route_cache.get(key)
        if cached is None:
            order, value = initial_route(key, local_deadline)
            cached = (tuple(order), value, False)
            route_cache[key] = cached

        if explore and time.perf_counter() < local_deadline:
            exact_limit = min(11, 6 + int(math.log1p(elapsed + 1.0)))
            exact_probability = min(0.45, 0.025 + elapsed / (90.0 + elapsed))
            if (not cached[2] and len(key) <= exact_limit and
                    rng.random() < exact_probability):
                exact = exact_route(key, local_deadline)
                if exact is not None:
                    cached = (tuple(exact[0]), exact[1], True)
                    route_cache[key] = cached
            elif not cached[2] and len(key) >= 3:
                trial = list(cached[0])
                mode = rng.randrange(5)
                if mode == 0:
                    i = rng.randrange(len(trial))
                    j = rng.randrange(len(trial))
                    if i > j:
                        i, j = j, i
                    if i != j:
                        trial[i:j + 1] = reversed(trial[i:j + 1])
                elif mode == 1:
                    i, j = rng.sample(range(len(trial)), 2)
                    trial[i], trial[j] = trial[j], trial[i]
                elif mode == 2:
                    node = trial.pop(rng.randrange(len(trial)))
                    trial.insert(rng.randrange(len(trial) + 1), node)
                elif mode == 3 and len(trial) >= 6:
                    cuts = sorted(rng.sample(range(1, len(trial)), 3))
                    a, b, c = cuts
                    trial = trial[:a] + trial[b:c] + trial[a:b] + trial[c:]
                else:
                    rng.shuffle(trial)
                trial = two_opt(trial, local_deadline)
                if time.perf_counter() < local_deadline:
                    trial = relocate(trial, local_deadline)
                value = route_cost(trial)
                if value + eps < cached[1]:
                    cached = (tuple(trial), value, False)
                    route_cache[key] = cached
        return list(cached[0]), cached[1]

    def weighted_quantile(values, weights, level):
        if not values:
            return 0.0
        pairs = sorted(zip(values, weights), key=lambda pair: pair[0])
        total = sum(weight for _, weight in pairs)
        if total <= eps:
            return pairs[-1][0]
        target = min(1.0, max(0.0, level)) * total
        accumulated = 0.0
        for value, weight in pairs:
            accumulated += weight
            if accumulated + eps >= target:
                return value
        return pairs[-1][0]

    def future_curve(t, nid, c, end_t, local_deadline):
        if end_t <= t:
            return []
        key = (t, nid, c, end_t)
        cached = curve_cache.get(key)
        if cached is not None:
            return cached
        group = groups[t][nid]
        members = group["members"]
        mass = group["mass"]
        if mass <= eps:
            curve_cache[key] = []
            return []

        cumulative = {}
        points = [0.0]
        for s in members:
            values = []
            amount = 0.0
            for tau in range(t + 1, end_t + 1):
                amount += demand[s][tau][c]
                values.append(amount)
                points.append(amount)
            cumulative[s] = values
            if time.perf_counter() >= local_deadline:
                return None

        points = sorted(set(points))
        if len(points) > 129:
            last = len(points) - 1
            points = sorted(set(points[(j * last) // 128] for j in range(129)))

        curve = []
        for j in range(len(points) - 1):
            lo = points[j]
            hi = points[j + 1]
            if hi <= lo + eps:
                continue
            midpoint = 0.5 * (lo + hi)
            consumed = 0.0
            held_steps = 0.0
            for s in members:
                values = cumulative[s]
                index = bisect.bisect_left(values, midpoint)
                if index < len(values):
                    consumed += probabilities[s]
                    steps = index + 1
                else:
                    steps = end_t - t + 1
                held_steps += probabilities[s] * steps
            curve.append((
                lo,
                hi,
                consumed / mass,
                held_steps / mass,
            ))
            if (j & 31) == 0 and time.perf_counter() >= local_deadline:
                return None
        curve_cache[key] = curve
        return curve

    def clamp_params(values):
        values[0] = min(2.3, max(0.0, values[0]))
        values[1] = min(1.0, max(0.0, values[1]))
        values[2] = min(2.2, max(0.0, values[2]))
        values[3] = min(1.8, max(0.5, values[3]))
        values[4] = min(1.55, max(0.25, values[4]))
        values[5] = min(1.8, max(0.0, values[5]))
        values[6] = min(2.5, max(0.0, values[6]))
        values[7] = min(1.0, max(0.0, values[7]))
        values[8] = min(1.5, max(0.0, values[8]))
        return tuple(values)

    def evaluate(setup, assignment, params, local_deadline, explore_routes):
        if time.perf_counter() >= local_deadline:
            return None
        cover_factor = params[0]
        mean_mix = params[1]
        buffer_factor = params[2]
        production_threshold = params[3]
        future_scale = params[4]
        stock_guard = params[5]
        lookahead_factor = params[6]
        curve_mix = params[7]
        factory_credit = params[8]

        next_visit = [[-1 for _ in range(N)] for _ in range(H)]
        last_visit = [-1 for _ in range(N)]
        for t in range(H - 1, -1, -1):
            for c in range(N):
                next_visit[t][c] = last_visit[c]
                if 0 <= assignment[t][c] < M:
                    last_visit[c] = t

        next_setup = [-1 for _ in range(H)]
        last_setup = -1
        for t in range(H - 1, -1, -1):
            next_setup[t] = last_setup
            if setup[t]:
                last_setup = t

        gap_ends = [H - 1 for _ in range(H)]
        future_shadows = [0.0 for _ in range(H)]
        for t in range(H):
            nxt = next_setup[t]
            gap_end = H - 1 if nxt < 0 else nxt - 1
            gap_ends[t] = gap_end
            shadow = 0.0
            for tau in range(t + 1, gap_end + 1):
                delay = holding[0] * (tau - t)
                for c in range(N):
                    if 0 <= assignment[tau][c] < M:
                        shadow = max(shadow, penalty[c] - delay)
            future_shadows[t] = max(0.0, shadow)

        records = []
        active_assignment = [[-1 for _ in range(N)] for _ in range(H)]
        used_setup = [0 for _ in range(H)]
        expected_operating_cost = 0.0
        expected_lost = [[0.0 for _ in range(N)] for _ in range(H)]
        expected_delivered = [[0.0 for _ in range(N)] for _ in range(H)]
        expected_production = [0.0 for _ in range(H)]
        tick = 0

        for t in range(H):
            level_records = [None for _ in groups[t]]
            for nid, group in enumerate(groups[t]):
                if time.perf_counter() >= local_deadline:
                    return None
                previous = initial[:] if t == 0 else list(
                    records[t - 1][group["parent"]][2]
                )
                members = group["members"]
                mass = group["mass"]
                current_demand = group["demand"]
                factory_stock = max(0.0, previous[0])

                reserve_target = 0.0
                gap_end = gap_ends[t]
                if (setup[t] and buffer_factor > eps and gap_end > t and
                        mass > eps):
                    branch_values = []
                    branch_weights = []
                    weighted_mean = 0.0
                    minimum = None
                    for s in members:
                        branch_need = 0.0
                        for tau in range(t + 1, gap_end + 1):
                            vehicle_need = [0.0 for _ in range(M)]
                            delay = holding[0] * (tau - t)
                            for c in range(N):
                                k = assignment[tau][c]
                                if (0 <= k < M and penalty[c] >
                                        production_cost * production_threshold + delay):
                                    vehicle_need[k] += demand[s][tau][c]
                            branch_need += sum(
                                min(vehicle_capacity, value)
                                for value in vehicle_need
                            )
                        branch_values.append(branch_need)
                        branch_weights.append(probabilities[s])
                        weighted_mean += probabilities[s] * branch_need
                        minimum = branch_need if minimum is None else min(
                            minimum, branch_need
                        )
                        tick += 1
                        if (tick & 31) == 0 and time.perf_counter() >= local_deadline:
                            return None
                    weighted_mean /= mass
                    minimum = 0.0 if minimum is None else minimum
                    blended = mean_mix * weighted_mean + (1.0 - mean_mix) * minimum
                    quantile = weighted_quantile(
                        branch_values,
                        branch_weights,
                        0.50 + 0.47 * mean_mix,
                    )
                    reserve_base = (
                        (1.0 - curve_mix) * blended + curve_mix * quantile
                    )
                    reserve_target = min(
                        capacities[0], max(0.0, buffer_factor * reserve_base)
                    )

                candidates = []
                customer_limit = [0.0 for _ in range(N)]
                extra_steps = int(round(
                    lookahead_factor * max(1.0, math.sqrt(float(H)))
                ))

                for c in range(N):
                    k = assignment[t][c]
                    if k < 0 or k >= M:
                        continue
                    prior = max(0.0, previous[c + 1])
                    remaining_space = max(0.0, capacities[c + 1] - prior)
                    maximum_delivery = min(
                        remaining_space,
                        max(0.0, minimum_suffix[t][nid][c]),
                    )
                    if maximum_delivery <= eps:
                        continue

                    shortage = max(0.0, current_demand[c] - prior)
                    current_quantity = min(shortage, maximum_delivery)
                    if current_quantity > eps and penalty[c] > eps:
                        old_score = penalty[c] + factory_credit * holding[0]
                        candidates.append((
                            old_score, penalty[c], c, k, current_quantity
                        ))

                    nxt = next_visit[t][c]
                    natural_horizon = H - 1 if nxt < 0 else nxt - 1
                    horizon = max(
                        natural_horizon,
                        min(H - 1, t + extra_steps),
                    )
                    natural_end = max(0.0, prior - current_demand[c])
                    maximum_end = natural_end + max(
                        0.0, maximum_delivery - current_quantity
                    )

                    if horizon > t and maximum_end > natural_end + eps and mass > eps:
                        future_values = []
                        weighted_future = 0.0
                        minimum_future = None
                        for s in members:
                            value = (
                                suffix[s][t + 1][c]
                                - suffix[s][horizon + 1][c]
                            )
                            future_values.append(value)
                            weighted_future += probabilities[s] * value
                            minimum_future = value if minimum_future is None else min(
                                minimum_future, value
                            )
                        weighted_future /= mass
                        minimum_future = 0.0 if minimum_future is None else minimum_future
                        target_base = (
                            mean_mix * weighted_future
                            + (1.0 - mean_mix) * minimum_future
                        )
                        maximum_rule_end = max(
                            0.0, capacities[c + 1] - current_demand[c]
                        )
                        minimum_total_end = max(
                            0.0,
                            minimum_suffix[t][nid][c] - current_demand[c],
                        )
                        target_end = min(
                            maximum_end,
                            maximum_rule_end,
                            minimum_total_end,
                            max(0.0, cover_factor * target_base),
                        )
                        average_delay = max(
                            1.0, 0.5 * (horizon - t + 1)
                        )
                        heuristic_score = (
                            penalty[c] * future_scale
                            - holding[c + 1] * average_delay
                        )

                        if curve_mix <= 1e-8:
                            amount = max(0.0, target_end - natural_end)
                            if amount > eps and heuristic_score > eps:
                                candidates.append((
                                    heuristic_score + factory_credit * holding[0] * average_delay,
                                    heuristic_score,
                                    c,
                                    k,
                                    amount,
                                ))
                        else:
                            allowed_end = min(
                                maximum_end,
                                target_end + curve_mix * max(
                                    0.0, maximum_end - target_end
                                ),
                            )
                            curve = future_curve(
                                t, nid, c, horizon, local_deadline
                            )
                            if curve is None:
                                return None
                            for lo, hi, consume_probability, held_steps in curve:
                                clipped_lo = max(lo, natural_end)
                                clipped_hi = min(hi, allowed_end)
                                if clipped_hi <= clipped_lo + eps:
                                    continue
                                midpoint = 0.5 * (clipped_lo + clipped_hi)
                                if midpoint <= target_end + eps:
                                    heuristic_part = heuristic_score
                                else:
                                    heuristic_part = -holding[c + 1] * average_delay
                                marginal_part = (
                                    future_scale * penalty[c] * consume_probability
                                    - holding[c + 1] * held_steps
                                )
                                new_score = (
                                    (1.0 - curve_mix) * heuristic_part
                                    + curve_mix * marginal_part
                                )
                                old_score = (
                                    new_score
                                    + factory_credit * holding[0] * held_steps
                                )
                                if old_score > eps or new_score > eps:
                                    candidates.append((
                                        old_score,
                                        new_score,
                                        c,
                                        k,
                                        clipped_hi - clipped_lo,
                                    ))
                    customer_limit[c] = maximum_delivery
                    tick += 1
                    if (tick & 63) == 0 and time.perf_counter() >= local_deadline:
                        return None

                candidates.sort(
                    key=lambda item: (
                        max(
                            item[0],
                            item[1] - production_cost * production_threshold,
                        ),
                        item[1],
                        -item[2],
                    ),
                    reverse=True,
                )
                remaining_vehicle = [vehicle_capacity for _ in range(M)]
                remaining_customer = customer_limit[:]
                delivered = [0.0 for _ in range(N)]

                protected = min(factory_stock, reserve_target)
                free_stock = max(0.0, factory_stock - protected)
                protected_stock = protected
                production_left = production_capacity if setup[t] else 0.0
                stock_used = 0.0
                production_used = 0.0
                guard_value = stock_guard * future_shadows[t]

                for old_score, new_score, c, k, amount in candidates:
                    possible = min(
                        amount,
                        remaining_vehicle[k],
                        remaining_customer[c],
                    )
                    if possible <= eps:
                        continue
                    taken = 0.0
                    if old_score > eps and free_stock > eps:
                        use = min(possible, free_stock)
                        free_stock -= use
                        stock_used += use
                        taken += use
                        possible -= use
                    if (possible > eps and protected_stock > eps and
                            old_score + eps >= guard_value and old_score > eps):
                        use = min(possible, protected_stock)
                        protected_stock -= use
                        stock_used += use
                        taken += use
                        possible -= use
                    if (possible > eps and production_left > eps and
                            new_score > production_cost * production_threshold + eps):
                        use = min(possible, production_left)
                        production_left -= use
                        production_used += use
                        taken += use
                        possible -= use
                    if taken > eps:
                        delivered[c] += taken
                        remaining_vehicle[k] -= taken
                        remaining_customer[c] -= taken

                factory_after_shipments = max(0.0, factory_stock - stock_used)
                extra_production = 0.0
                if setup[t] and production_left > eps:
                    extra_production = min(
                        max(0.0, reserve_target - factory_after_shipments),
                        production_left,
                        max(0.0, capacities[0] - factory_after_shipments),
                    )
                production_quantity = production_used + extra_production
                if production_quantity <= eps:
                    production_quantity = 0.0
                else:
                    used_setup[t] = 1
                factory_end = factory_after_shipments + extra_production
                if factory_end < eps:
                    factory_end = 0.0
                if factory_end > capacities[0] and factory_end-capacities[0] < 1e-7:
                    factory_end = capacities[0]

                sparse_delivery = []
                end_inventory = [factory_end]
                lost_sales = []
                period_cost = (
                    production_cost * production_quantity
                    + holding[0] * factory_end
                )
                for c in range(N):
                    quantity = delivered[c]
                    if quantity > eps:
                        k = assignment[t][c]
                        sparse_delivery.append((k, c, quantity))
                        active_assignment[t][c] = k
                    available = max(0.0, previous[c + 1]) + quantity
                    d = current_demand[c]
                    if available + eps >= d:
                        ending = max(0.0, available - d)
                        lost = 0.0
                    else:
                        ending = 0.0
                        lost = d - available
                    if ending < eps:
                        ending = 0.0
                    if lost < eps:
                        lost = 0.0
                    end_inventory.append(ending)
                    lost_sales.append(lost)
                    period_cost += holding[c + 1] * ending + penalty[c] * lost
                    expected_lost[t][c] += mass * lost
                    expected_delivered[t][c] += mass * quantity
                expected_production[t] += mass * production_quantity
                expected_operating_cost += mass * period_cost
                level_records[nid] = (
                    production_quantity,
                    tuple(sparse_delivery),
                    tuple(end_inventory),
                    tuple(lost_sales),
                )
            records.append(level_records)

        elapsed = max(0.0, time.perf_counter() - started)
        routes = []
        transportation_total = 0.0
        for t in range(H):
            period_routes = []
            for k in range(M):
                customers = [
                    c + 1 for c in range(N)
                    if active_assignment[t][c] == k
                ]
                if not customers:
                    period_routes.append([0])
                else:
                    order, value = get_route(
                        customers, local_deadline, explore_routes, elapsed
                    )
                    if time.perf_counter() >= local_deadline:
                        return None
                    period_routes.append([0] + order + [0])
                    transportation_total += value
            routes.append(period_routes)

        return {
            "cost": (
                expected_operating_cost
                + transportation_total
                + setup_cost * sum(used_setup)
            ),
            "setup": used_setup,
            "assignment": active_assignment,
            "params": tuple(params),
            "routes": routes,
            "records": records,
            "lost": expected_lost,
            "delivered": expected_delivered,
            "production": expected_production,
        }

    def materialize(result):
        production_output = []
        delivery_output = []
        inventory_output = []
        unmet_output = []
        records = result["records"]
        for s in range(S):
            scenario_production = []
            scenario_delivery = []
            scenario_inventory = []
            scenario_unmet = []
            for t in range(H):
                record = records[t][node_ids[s][t]]
                scenario_production.append(float(record[0]))
                dense = [[0.0 for _ in range(N)] for _ in range(M)]
                for k, c, quantity in record[1]:
                    dense[k][c] = float(quantity)
                scenario_delivery.append(dense)
                scenario_inventory.append([float(x) for x in record[2]])
                scenario_unmet.append([float(x) for x in record[3]])
            production_output.append(scenario_production)
            delivery_output.append(scenario_delivery)
            inventory_output.append(scenario_inventory)
            unmet_output.append(scenario_unmet)
        return {
            "problem_type": "multi_period_stochastic_production_routing_with_lost_sales",
            "production_setup": [int(x) for x in result["setup"]],
            "vehicle_routes": [
                [route[:] for route in period]
                for period in result["routes"]
            ],
            "production_quantities": production_output,
            "delivery_quantities": delivery_output,
            "end_inventories": inventory_output,
            "unmet_demand_quantities": unmet_output,
        }

    zero_setup = [0 for _ in range(H)]
    zero_assignment = [[-1 for _ in range(N)] for _ in range(H)]
    zero_params = (0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    zero_result = evaluate(
        zero_setup, zero_assignment, zero_params, deadline, False
    )
    if zero_result is None:
        return fallback_solution
    best_result = zero_result
    best_cost = zero_result["cost"]

    average_depot_cost = sum(
        transport[0][c + 1] + transport[c + 1][0]
        for c in range(N)
    ) / max(1, N)
    route_scale = max(1.0, average_depot_cost)

    def insertion_delta(order, node):
        if not order:
            return transport[0][node] + transport[node][0]
        best = float("inf")
        for pos in range(len(order) + 1):
            left = 0 if pos == 0 else order[pos - 1]
            right = 0 if pos == len(order) else order[pos]
            value = (
                transport[left][node] + transport[node][right]
                - transport[left][right]
            )
            if value < best:
                best = value
        return best

    def pack_selected(selected, jitter):
        assignment = [[-1 for _ in range(N)] for _ in range(H)]
        next_selected = [[H for _ in range(N)] for _ in range(H)]
        last = [H for _ in range(N)]
        for t in range(H - 1, -1, -1):
            for c in range(N):
                next_selected[t][c] = last[c]
                if selected[t][c]:
                    last[c] = t
        for t in range(H):
            customers = [c for c in range(N) if selected[t][c]]
            quantities = {}
            for c in customers:
                nxt = next_selected[t][c]
                quantities[c] = min(
                    vehicle_capacity,
                    capacities[c + 1],
                    max(0.0, expected_suffix[t][c] - expected_suffix[nxt][c]),
                )
            customers.sort(
                key=lambda c: -quantities[c] * (penalty[c] + eps)
                * ((0.85 + 0.30 * rng.random()) if jitter else 1.0)
            )
            orders = [[] for _ in range(M)]
            loads = [0.0 for _ in range(M)]
            for c in customers:
                node = c + 1
                best = None
                for k in range(M):
                    delta = insertion_delta(orders[k], node)
                    overflow = max(
                        0.0, loads[k] + quantities[c] - vehicle_capacity
                    )
                    score = (
                        delta + 4.5 * route_scale
                        * overflow / max(vehicle_capacity, eps)
                        + 0.12 * route_scale * loads[k]
                        / max(vehicle_capacity, eps)
                    )
                    if jitter:
                        score *= 0.92 + 0.16 * rng.random()
                    if best is None or score < best[0]:
                        best = (score, k)
                k = best[1]
                assignment[t][c] = k
                loads[k] += quantities[c]
                best_pos = 0
                best_delta = float("inf")
                for pos in range(len(orders[k]) + 1):
                    left = 0 if pos == 0 else orders[k][pos - 1]
                    right = 0 if pos == len(orders[k]) else orders[k][pos]
                    delta = (
                        transport[left][node] + transport[node][right]
                        - transport[left][right]
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_pos = pos
                orders[k].insert(best_pos, node)
            if time.perf_counter() >= deadline:
                break
        return assignment

    def best_vehicle(row, customer, t, forbidden=-1):
        best_k = 0
        best_value = float("inf")
        for k in range(M):
            if k == forbidden and M > 1:
                continue
            members = [c for c in range(N) if row[c] == k and c != customer]
            load = sum(expected_demand[t][c] for c in members)
            if members:
                travel = min(
                    2.0 * transport[customer + 1][c + 1]
                    for c in members
                )
            else:
                travel = (
                    transport[0][customer + 1]
                    + transport[customer + 1][0]
                )
            pressure = (
                load + expected_demand[t][customer]
            ) / max(vehicle_capacity, eps)
            value = pressure * pressure + 0.30 * travel / route_scale
            value *= 0.97 + 0.06 * rng.random()
            if value < best_value:
                best_value = value
                best_k = k
        return best_k

    def marginal_trip(row, customer):
        k = row[customer]
        if k < 0:
            return transport[0][customer + 1] + transport[customer + 1][0]
        others = [c for c in range(N) if c != customer and row[c] == k]
        if not others:
            return transport[0][customer + 1] + transport[customer + 1][0]
        return min(
            2.0 * transport[customer + 1][c + 1]
            for c in others
        )

    def make_seed(kind):
        selected = [[False for _ in range(N)] for _ in range(H)]
        setup = [1 for _ in range(H)]
        if kind == 0:
            for t in range(H):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (0.15, 1.0, 0.15, 1.0, 0.90, 0.0, 0.0, 0.0, 0.0)
        elif kind == 1:
            for t in range(H):
                for c in range(N):
                    gross = penalty[c] * expected_demand[t][c]
                    trip = transport[0][c + 1] + transport[c + 1][0]
                    selected[t][c] = gross > 0.18 * trip
            params = (0.68, 1.0, 0.28, 0.95, 1.0, 0.0, 0.0, 0.0, 0.0)
        elif kind == 2:
            setup = [1 if t % 2 == 0 else 0 for t in range(H)]
            for t in range(0, H, 2):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (1.0, 0.72, 0.85, 0.98, 1.05, 0.35, 0.25, 0.0, 0.0)
        elif kind == 3:
            gap = max(1, int(round(math.sqrt(float(H)))))
            setup = [1 if t % gap == 0 else 0 for t in range(H)]
            for t in range(0, H, gap):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (1.08, 0.62, 0.98, 0.90, 1.10, 0.55, 0.40, 0.0, 0.0)
        elif kind == 4:
            mean_total = sum(sum(row) for row in expected_demand) / max(1, H)
            setup_gap = max(1, int(round(math.sqrt(float(H)))))
            if mean_total > eps:
                setup_gap = min(
                    H,
                    setup_gap,
                    max(1, int(production_capacity / mean_total)),
                )
            setup = [1 if t % setup_gap == 0 else 0 for t in range(H)]
            for c in range(N):
                mean = expected_suffix[0][c] / max(1, H)
                if mean <= eps:
                    continue
                fixed = transport[0][c + 1] + transport[c + 1][0]
                if holding[c + 1] * mean > eps:
                    gap = int(round(math.sqrt(
                        2.0 * max(fixed, eps)
                        / (holding[c + 1] * mean)
                    )))
                else:
                    gap = H
                storage_gap = max(
                    1, int(min(capacities[c + 1], vehicle_capacity) / mean)
                )
                gap = min(H, max(1, gap), storage_gap)
                for t in range(0, H, gap):
                    selected[t][c] = True
            params = (0.90, 0.78, 0.72, 0.92, 1.03, 0.35, 0.65, 0.42, 0.45)
        else:
            for c in range(N):
                accumulated = 0.0
                for t in range(H):
                    accumulated += expected_demand[t][c]
                    benefit = max(0.0, penalty[c] - production_cost) * accumulated
                    trip = transport[0][c + 1] + transport[c + 1][0]
                    if benefit >= 0.52 * trip or t == H - 1:
                        if accumulated > eps and benefit > 0.15 * trip:
                            selected[t][c] = True
                        accumulated = 0.0
            setup = [
                1 if production_capacity > eps and any(selected[t]) else 0
                for t in range(H)
            ]
            params = (1.12, 0.58, 1.02, 0.91, 1.08, 0.62, 0.82, 0.78, 0.70)
        if production_capacity <= eps:
            setup = [0 for _ in range(H)]
        return setup, pack_selected(selected, kind >= 4), params

    def random_candidate(elapsed):
        spread = min(1.0, math.log1p(elapsed + 1.0) / 6.0)
        setup_density = 0.06 + 0.91 * rng.random()
        visit_density = 0.04 + 0.94 * rng.random()
        setup = [
            1 if production_capacity > eps and rng.random() < setup_density else 0
            for _ in range(H)
        ]
        selected = [[False for _ in range(N)] for _ in range(H)]
        for t in range(H):
            for c in range(N):
                gross = (
                    max(0.0, penalty[c] - production_cost)
                    * expected_demand[t][c]
                )
                trip = transport[0][c + 1] + transport[c + 1][0]
                attraction = gross / max(eps, gross + trip)
                probability = visit_density * (0.25 + 1.45 * attraction)
                probability += 0.08 * spread * rng.random()
                selected[t][c] = rng.random() < min(0.99, probability)
        params = clamp_params([
            2.2 * rng.random(),
            rng.random(),
            2.1 * rng.random(),
            0.55 + 1.20 * rng.random(),
            0.30 + 1.20 * rng.random(),
            1.7 * rng.random(),
            2.4 * rng.random(),
            rng.random(),
            1.4 * rng.random(),
        ])
        return setup, pack_selected(selected, True), params

    def entry_from_result(result):
        return {
            "cost": result["cost"],
            "setup": result["setup"][:],
            "assignment": [row[:] for row in result["assignment"]],
            "params": tuple(result["params"]),
            "lost": [row[:] for row in result["lost"]],
            "delivered": [row[:] for row in result["delivered"]],
        }

    def signature(entry):
        return (
            tuple(entry["setup"]),
            tuple(tuple(row) for row in entry["assignment"]),
            tuple(round(x, 4) for x in entry["params"]),
        )

    move_weights = [1.0 for _ in range(11)]

    def choose_mode():
        total = sum(move_weights)
        draw = rng.random() * total
        for i, weight in enumerate(move_weights):
            draw -= weight
            if draw <= 0.0:
                return i
        return len(move_weights) - 1

    def mutate(parent, elapsed, stagnation):
        setup = parent["setup"][:]
        assignment = [row[:] for row in parent["assignment"]]
        params = list(parent["params"])
        depth = 1 + int(0.50 * math.log1p(elapsed + 1.0))
        if stagnation > 1.0 and rng.random() < min(0.7, stagnation / (8.0 + stagnation)):
            depth += 1 + int(math.sqrt(math.log1p(stagnation)))
        primary = choose_mode()

        for step in range(max(1, depth)):
            mode = primary if step == 0 else choose_mode()
            t = rng.randrange(H)
            c = rng.randrange(N)
            if mode == 0:
                choices = []
                for tt in range(H):
                    for cc in range(N):
                        if assignment[tt][cc] < 0:
                            benefit = (
                                parent["lost"][tt][cc] * penalty[cc]
                                + 0.16 * expected_demand[tt][cc] * penalty[cc]
                            )
                            trip = transport[0][cc + 1] + transport[cc + 1][0]
                            choices.append((
                                benefit - trip * (0.15 + 0.45 * rng.random()),
                                tt,
                                cc,
                            ))
                if choices:
                    _, t, c = max(choices)
                    assignment[t][c] = best_vehicle(assignment[t], c, t)
                    if production_capacity > eps and rng.random() < 0.72:
                        setup[t] = 1
            elif mode == 1:
                choices = []
                for tt in range(H):
                    for cc in range(N):
                        if assignment[tt][cc] >= 0:
                            benefit = parent["delivered"][tt][cc] * penalty[cc]
                            choices.append((
                                benefit - marginal_trip(assignment[tt], cc)
                                * (0.48 + 0.65 * rng.random()),
                                tt,
                                cc,
                            ))
                if choices:
                    _, t, c = min(choices)
                    assignment[t][c] = -1
            elif mode == 2:
                if rng.random() < 0.62:
                    choices = []
                    for tt in range(H):
                        value = sum(
                            parent["lost"][tt][cc] * penalty[cc]
                            for cc in range(N)
                        )
                        if not setup[tt]:
                            choices.append((value, tt))
                    if choices:
                        setup[max(choices)[1]] = 1
                    else:
                        setup[t] = 1 - int(bool(setup[t]))
                else:
                    setup[t] = 1 - int(bool(setup[t]))
            elif mode == 3:
                active = [
                    (tt, cc) for tt in range(H) for cc in range(N)
                    if assignment[tt][cc] >= 0
                ]
                if active and M > 1:
                    t, c = rng.choice(active)
                    old = assignment[t][c]
                    assignment[t][c] = -1
                    assignment[t][c] = best_vehicle(assignment[t], c, t, old)
            elif mode == 4:
                visits = [tt for tt in range(H) if assignment[tt][c] >= 0]
                if visits:
                    old_t = rng.choice(visits)
                    radius = 1 + int(min(
                        H - 1,
                        math.sqrt(elapsed + stagnation + 1.0),
                    ) * rng.random())
                    new_t = min(H - 1, max(0, old_t + rng.randint(-radius, radius)))
                    if new_t != old_t:
                        assignment[old_t][c] = -1
                        assignment[new_t][c] = best_vehicle(
                            assignment[new_t], c, new_t
                        )
                        if production_capacity > eps and rng.random() < 0.6:
                            setup[new_t] = 1
                else:
                    assignment[t][c] = best_vehicle(assignment[t], c, t)
            elif mode == 5:
                p = rng.randrange(9)
                sigma = 0.035 + 0.30 / (1.0 + 0.04 * math.sqrt(elapsed + 1.0))
                params[p] += rng.gauss(0.0, sigma)
            elif mode == 6:
                ranked = sorted(
                    range(N),
                    key=lambda cc: (
                        parent["lost"][t][cc] * penalty[cc]
                        + expected_demand[t][cc] * penalty[cc]
                        - 0.22 * (
                            transport[0][cc + 1] + transport[cc + 1][0]
                        )
                    ),
                    reverse=True,
                )
                take = 1 + rng.randrange(N)
                assignment[t] = [-1 for _ in range(N)]
                for cc in ranked[:take]:
                    assignment[t][cc] = best_vehicle(assignment[t], cc, t)
                if production_capacity > eps:
                    setup[t] = 1
            elif mode == 7:
                gap = 1 + rng.randrange(H)
                phase = rng.randrange(gap)
                add = rng.random() < 0.65
                for tt in range(H):
                    if tt % gap == phase and add:
                        assignment[tt][c] = best_vehicle(assignment[tt], c, tt)
                    elif tt % gap != phase and rng.random() < 0.58:
                        assignment[tt][c] = -1
            elif mode == 8:
                span = 1 + rng.randrange(max(1, min(H, 2 + int(math.sqrt(H)))))
                begin = rng.randrange(H)
                add = rng.random() < 0.5
                for tt in range(begin, min(H, begin + span)):
                    for cc in range(N):
                        if rng.random() < min(0.8, 0.15 + 0.08 * depth):
                            assignment[tt][cc] = (
                                best_vehicle(assignment[tt], cc, tt) if add else -1
                            )
                    if add and production_capacity > eps:
                        setup[tt] = 1
            elif mode == 9:
                setup[t] = 1 if production_capacity > eps else 0
                ranked = sorted(
                    range(N),
                    key=lambda cc: (
                        parent["lost"][t][cc] * penalty[cc]
                        + expected_demand[t][cc]
                        * max(0.0, penalty[cc] - production_cost)
                    ),
                    reverse=True,
                )
                additions = 1 + rng.randrange(max(1, min(N, M + 3)))
                for cc in ranked[:additions]:
                    assignment[t][cc] = best_vehicle(assignment[t], cc, t)
            else:
                if rng.random() < 0.5:
                    value = rng.randrange(2)
                    span = 1 + rng.randrange(max(1, min(H, depth + 2)))
                    for tt in range(t, min(H, t + span)):
                        setup[tt] = value
                else:
                    if assignment[t][c] < 0:
                        assignment[t][c] = best_vehicle(assignment[t], c, t)
                    else:
                        assignment[t][c] = -1

        if rng.random() < 0.35:
            p = rng.randrange(9)
            params[p] += rng.gauss(0.0, 0.04 + 0.15 * rng.random())
        params = clamp_params(params)
        if production_capacity <= eps:
            setup = [0 for _ in range(H)]
        return (setup, assignment, params), primary

    def crossover(a, b, elapsed):
        setup = [0 for _ in range(H)]
        assignment = [[-1 for _ in range(N)] for _ in range(H)]
        if rng.random() < 0.5:
            for t in range(H):
                source = a if rng.random() < 0.5 else b
                setup[t] = source["setup"][t]
                assignment[t] = source["assignment"][t][:]
        else:
            for t in range(H):
                setup[t] = a["setup"][t] if rng.random() < 0.5 else b["setup"][t]
            for c in range(N):
                source = a if rng.random() < 0.5 else b
                for t in range(H):
                    assignment[t][c] = source["assignment"][t][c]
        blend = rng.random()
        sigma = 0.025 + 0.11 / (1.0 + math.sqrt(elapsed + 1.0))
        params = clamp_params([
            blend * a["params"][p] + (1.0 - blend) * b["params"][p]
            + rng.gauss(0.0, sigma)
            for p in range(9)
        ])
        if production_capacity <= eps:
            setup = [0 for _ in range(H)]
        return setup, assignment, params

    zero_entry = entry_from_result(zero_result)
    best_entry = zero_entry
    current = zero_entry
    population = [zero_entry]
    seed_kind = 0
    last_improvement = time.perf_counter()
    vnd_cursor = 0
    vnd_round = 0
    vnd_step = 0.16

    def vnd_candidate():
        nonlocal vnd_cursor, vnd_round, vnd_step
        base = best_entry
        setup_span = H
        visit_span = H * N
        vehicle_span = H * N * max(0, M - 1)
        shift_span = 2 * H * N
        parameter_span = 18
        total = setup_span + visit_span + vehicle_span + shift_span + parameter_span
        attempts = 0
        while attempts <= total + 1:
            if vnd_cursor >= total:
                vnd_cursor = 0
                vnd_round += 1
                vnd_step *= 0.72
                if vnd_step < 0.008:
                    vnd_step = 0.16 * (0.85 + 0.30 * rng.random())
            index = vnd_cursor
            vnd_cursor += 1
            attempts += 1
            setup = base["setup"][:]
            assignment = [row[:] for row in base["assignment"]]
            params = list(base["params"])

            if index < setup_span:
                t = index
                if production_capacity <= eps:
                    continue
                setup[t] = 1 - int(bool(setup[t]))
            else:
                index -= setup_span
                if index < visit_span:
                    t = index // N
                    c = index % N
                    if assignment[t][c] < 0:
                        assignment[t][c] = best_vehicle(assignment[t], c, t)
                        if production_capacity > eps:
                            setup[t] = 1
                    else:
                        assignment[t][c] = -1
                else:
                    index -= visit_span
                    if index < vehicle_span:
                        if M <= 1:
                            continue
                        block = M - 1
                        pair = index // block
                        alternative = index % block
                        t = pair // N
                        c = pair % N
                        old = assignment[t][c]
                        if old < 0:
                            continue
                        if alternative >= old:
                            alternative += 1
                        assignment[t][c] = alternative
                    else:
                        index -= vehicle_span
                        if index < shift_span:
                            pair = index // 2
                            direction = -1 if index % 2 == 0 else 1
                            t = pair // N
                            c = pair % N
                            if assignment[t][c] < 0:
                                continue
                            new_t = t + direction
                            if new_t < 0 or new_t >= H:
                                continue
                            assignment[t][c] = -1
                            assignment[new_t][c] = best_vehicle(
                                assignment[new_t], c, new_t
                            )
                            if production_capacity > eps:
                                setup[new_t] = 1
                        else:
                            index -= shift_span
                            p = index // 2
                            direction = -1.0 if index % 2 == 0 else 1.0
                            params[p] += direction * vnd_step
                            params = list(clamp_params(params))
            if production_capacity <= eps:
                setup = [0 for _ in range(H)]
            return setup, assignment, tuple(params)
        return random_candidate(max(0.0, time.perf_counter() - started))

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        elapsed = max(0.0, now - started)
        stagnation = max(0.0, now - last_improvement)
        parent = None
        credited_mode = None

        if seed_kind < 6:
            candidate = make_seed(seed_kind)
            seed_kind += 1
        else:
            vnd_probability = min(
                0.58,
                0.30 + 0.20 * (1.0 - math.exp(-elapsed / 75.0))
                + min(0.08, stagnation / 150.0),
            )
            if rng.random() < 0.035:
                candidate = (
                    best_entry["setup"][:],
                    [row[:] for row in best_entry["assignment"]],
                    tuple(best_entry["params"]),
                )
            elif rng.random() < vnd_probability:
                candidate = vnd_candidate()
            else:
                restart_probability = 0.06 + min(
                    0.40, stagnation / (16.0 + elapsed + stagnation)
                )
                crossover_probability = min(
                    0.34, 0.07 + math.log1p(elapsed + 1.0) / 32.0
                )
                if rng.random() < restart_probability:
                    candidate = random_candidate(elapsed)
                elif len(population) >= 2 and rng.random() < crossover_probability:
                    tournament = min(
                        len(population), 2 + int(math.log1p(elapsed + 1.0) / 2.8)
                    )
                    a = min(
                        (rng.choice(population) for _ in range(tournament)),
                        key=lambda entry: entry["cost"],
                    )
                    b = rng.choice(population)
                    candidate = crossover(a, b, elapsed)
                else:
                    tournament = min(
                        len(population), 2 + int(math.log1p(elapsed + 1.0) / 2.4)
                    )
                    parent = current if rng.random() < 0.27 else min(
                        (rng.choice(population) for _ in range(tournament)),
                        key=lambda entry: entry["cost"],
                    )
                    candidate, credited_mode = mutate(
                        parent, elapsed, stagnation
                    )

        if time.perf_counter() >= deadline:
            break
        route_probability = min(
            0.72,
            0.15 + 0.52 * (1.0 - math.exp(-elapsed / 45.0)),
        )
        result = evaluate(
            candidate[0],
            candidate[1],
            candidate[2],
            deadline,
            rng.random() < route_probability,
        )
        if result is None:
            break
        entry = entry_from_result(result)
        cost = entry["cost"]

        if credited_mode is not None and parent is not None:
            if cost + 1e-9 < parent["cost"]:
                move_weights[credited_mode] *= 1.08
            else:
                move_weights[credited_mode] = max(
                    0.18, move_weights[credited_mode] * 0.996
                )
            if max(move_weights) > 50.0:
                for i in range(len(move_weights)):
                    move_weights[i] = 0.5 + 0.5 * move_weights[i]

        tolerance = 1e-9 * (1.0 + abs(best_cost))
        if cost < best_cost - tolerance:
            best_cost = cost
            best_result = result
            best_entry = entry
            last_improvement = time.perf_counter()
            vnd_cursor = 0

        temperature = 0.026 * (1.0 + abs(best_cost))
        temperature /= 1.0 + math.log1p(elapsed + 1.0)
        temperature *= 1.0 + min(3.0, stagnation / 11.0)
        accept = cost <= current["cost"]
        if not accept and temperature > eps:
            scaled = (cost - current["cost"]) / temperature
            if scaled < 45.0 and rng.random() < math.exp(-scaled):
                accept = True
        if accept:
            current = entry

        target_population = 7 + int(2.7 * math.sqrt(elapsed + 1.0))
        new_signature = signature(entry)
        duplicate = -1
        for i, old in enumerate(population):
            if signature(old) == new_signature:
                duplicate = i
                break
        if duplicate >= 0:
            if cost < population[duplicate]["cost"]:
                population[duplicate] = entry
        else:
            population.append(entry)
            if len(population) > target_population:
                worst = max(
                    range(len(population)),
                    key=lambda i: population[i]["cost"],
                )
                population.pop(worst)

        route_target = 260 + int(65.0 * math.sqrt(elapsed + 1.0))
        while len(route_cache) > route_target:
            route_cache.pop(next(iter(route_cache)))
        curve_target = 900 + int(110.0 * math.sqrt(elapsed + 1.0))
        while len(curve_cache) > curve_target:
            curve_cache.pop(next(iter(curve_cache)))

    try:
        return materialize(best_result)
    except BaseException:
        return fallback_solution


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    try:
        fallback_solution = _fallback(instance if isinstance(instance, dict) else {})
        fallback_elapsed = time.perf_counter() - started
        if not isinstance(instance, dict):
            return fallback_solution
        return _solve(
            instance,
            time_limit_s,
            started,
            fallback_solution,
            fallback_elapsed,
        )
    except BaseException:
        try:
            return _fallback(instance if isinstance(instance, dict) else {})
        except BaseException:
            return {
                "problem_type": "multi_period_stochastic_production_routing_with_lost_sales",
                "production_setup": [0],
                "vehicle_routes": [[[0]]],
                "production_quantities": [[0.0]],
                "delivery_quantities": [[[[0.0]]]],
                "end_inventories": [[[0.0, 0.0]]],
                "unmet_demand_quantities": [[[0.0]]],
            }
