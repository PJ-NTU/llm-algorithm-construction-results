import math
import random
import time


def _number(value, default=0.0):
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except BaseException:
        pass
    return default


def _positive_int(value, default=1):
    try:
        result = int(value)
        if result >= 1:
            return result
    except BaseException:
        pass
    return default


def _nested(value, indices, default=0.0):
    try:
        for index in indices:
            value = value[index]
        return _number(value, default)
    except BaseException:
        return default


def _fallback(instance):
    try:
        if not isinstance(instance, dict):
            instance = {}
        H = _positive_int(instance.get("num_periods", 1), 1)
        N = _positive_int(instance.get("num_customers", 1), 1)
        M = _positive_int(instance.get("num_vehicles", 1), 1)
        S = _positive_int(instance.get("num_scenarios", 1), 1)
        raw_demand = instance.get("demand_scenarios", [])
        raw_initial = instance.get("initial_inventories", [])
        initial = [
            max(0.0, _nested(raw_initial, (v,), 0.0))
            for v in range(N + 1)
        ]

        production = [[0.0 for _ in range(H)] for _ in range(S)]
        deliveries = [
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
                    available = max(0.0, previous[c + 1])
                    served = min(available, d)
                    ending[c + 1] = max(0.0, available - served)
                    unmet[s][t][c] = max(0.0, d - served)
                inventories[s][t] = ending
                previous = ending

        return {
            "problem_type": "multi_period_stochastic_production_routing_with_lost_sales",
            "production_setup": [0 for _ in range(H)],
            "vehicle_routes": [[[0] for _ in range(M)] for _ in range(H)],
            "production_quantities": production,
            "delivery_quantities": deliveries,
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


def _solve(instance, time_limit_s, started):
    budget = max(0.0, _number(time_limit_s, 0.0))
    hard_deadline = started + budget

    fallback_started = time.perf_counter()
    fallback_solution = _fallback(instance)
    fallback_time = time.perf_counter() - fallback_started
    reserve = max(0.012, 3.0 * fallback_time + 0.006)
    deadline = hard_deadline - reserve
    if time.perf_counter() >= deadline or not isinstance(instance, dict):
        return fallback_solution

    H = int(instance["num_periods"])
    N = int(instance["num_customers"])
    M = int(instance["num_vehicles"])
    S = int(instance["num_scenarios"])
    if H < 1 or N < 1 or M < 1 or S < 1:
        return fallback_solution

    demand = [
        [
            [
                max(0.0, float(instance["demand_scenarios"][s][t][c]))
                for c in range(N)
            ]
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

    initial = [float(instance["initial_inventories"][v]) for v in range(N + 1)]
    capacities = [
        max(0.0, float(instance["inventory_capacities"][v]))
        for v in range(N + 1)
    ]
    holding = [
        max(0.0, float(instance["holding_costs"][v]))
        for v in range(N + 1)
    ]
    penalty = [
        max(0.0, float(instance["unmet_demand_penalties"][c]))
        for c in range(N)
    ]
    transport = [
        [
            max(0.0, float(instance["transportation_costs"][i][j]))
            for j in range(N + 1)
        ]
        for i in range(N + 1)
    ]
    production_capacity = max(0.0, float(instance["production_capacity"]))
    vehicle_capacity = max(0.0, float(instance["vehicle_capacity"]))
    setup_cost = max(0.0, float(instance["setup_cost"]))
    production_cost = max(0.0, float(instance["unit_production_cost"]))

    if (
        len(initial) != N + 1
        or len(capacities) != N + 1
        or len(holding) != N + 1
        or len(penalty) != N
    ):
        return fallback_solution
    for v in range(N + 1):
        if initial[v] < 0.0 or initial[v] > capacities[v] + 1e-8:
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
                level.append(
                    {
                        "parent": parent,
                        "members": [],
                        "probability": 0.0,
                        "demand": demand[s][t][:],
                    }
                )
            level[nid]["members"].append(s)
            level[nid]["probability"] += probabilities[s]
            node_ids[s][t] = nid
        groups.append(level)
        level_minimum = []
        for group in level:
            members = group["members"]
            level_minimum.append(
                [min(suffix[s][t][c] for s in members) for c in range(N)]
            )
        minimum_suffix.append(level_minimum)
        if time.perf_counter() >= deadline:
            return fallback_solution

    seed_value = 146959810 + 1009 * H + 9176 * N + 65537 * M + 131071 * S
    seed_value ^= int(sum(initial) * 10007.0) & 0x7FFFFFFF
    seed_value ^= int(
        sum(sum(row) for row in expected_demand) * 1009.0
    ) & 0x7FFFFFFF
    rng = random.Random(seed_value)
    route_cache = {}

    def route_cost(order):
        if not order:
            return 0.0
        value = transport[0][order[0]] + transport[order[-1]][0]
        for i in range(len(order) - 1):
            value += transport[order[i]][order[i + 1]]
        return value

    def two_opt(order, local_deadline):
        result = order[:]
        n = len(result)
        while n >= 3:
            best_delta = -eps
            best_i = -1
            best_j = -1
            checks = 0
            for i in range(n - 1):
                a = 0 if i == 0 else result[i - 1]
                b = result[i]
                for j in range(i + 1, n):
                    c = result[j]
                    d = 0 if j == n - 1 else result[j + 1]
                    delta = (
                        transport[a][c]
                        + transport[b][d]
                        - transport[a][b]
                        - transport[c][d]
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                    checks += 1
                    if (
                        (checks & 127) == 0
                        and time.perf_counter() >= local_deadline
                    ):
                        return result
            if best_i < 0:
                break
            result[best_i : best_j + 1] = reversed(
                result[best_i : best_j + 1]
            )
            if time.perf_counter() >= local_deadline:
                break
        return result

    def relocate_route(order, local_deadline):
        result = order[:]
        n = len(result)
        while n >= 3:
            best_delta = -eps
            best_i = -1
            best_pos = -1
            checks = 0
            for i in range(n):
                node = result[i]
                left = 0 if i == 0 else result[i - 1]
                right = 0 if i == n - 1 else result[i + 1]
                removal = (
                    transport[left][right]
                    - transport[left][node]
                    - transport[node][right]
                )
                reduced = result[:i] + result[i + 1 :]
                for pos in range(len(reduced) + 1):
                    before = 0 if pos == 0 else reduced[pos - 1]
                    after = 0 if pos == len(reduced) else reduced[pos]
                    delta = (
                        removal
                        + transport[before][node]
                        + transport[node][after]
                        - transport[before][after]
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_pos = pos
                    checks += 1
                    if (
                        (checks & 127) == 0
                        and time.perf_counter() >= local_deadline
                    ):
                        return result
            if best_i < 0:
                break
            node = result.pop(best_i)
            result.insert(best_pos, node)
            if time.perf_counter() >= local_deadline:
                break
        return result

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
        best_cost = route_cost(best)

        if time.perf_counter() < local_deadline:
            remaining = list(key)
            insertion = []
            if remaining:
                first = min(
                    remaining,
                    key=lambda v: (transport[0][v] + transport[v][0], v),
                )
                insertion.append(first)
                remaining.remove(first)
            while remaining:
                choice = None
                for node in remaining:
                    for pos in range(len(insertion) + 1):
                        before = 0 if pos == 0 else insertion[pos - 1]
                        after = 0 if pos == len(insertion) else insertion[pos]
                        delta = (
                            transport[before][node]
                            + transport[node][after]
                            - transport[before][after]
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
            if value + eps < best_cost:
                best = insertion
                best_cost = value
        return best, best_cost

    def exact_route(key, local_deadline):
        n = len(key)
        if n <= 1:
            return list(key), route_cost(list(key))
        size = 1 << n
        dp = [{} for _ in range(size)]
        for j in range(n):
            dp[1 << j][j] = (transport[0][key[j]], -1)
        for mask in range(1, size):
            if (mask & 63) == 0 and time.perf_counter() >= local_deadline:
                return None
            if mask & (mask - 1) == 0:
                continue
            bits = mask
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                previous_mask = mask ^ bit
                best_value = float("inf")
                best_previous = -1
                previous_bits = previous_mask
                while previous_bits:
                    previous_bit = previous_bits & -previous_bits
                    i = previous_bit.bit_length() - 1
                    old = dp[previous_mask].get(i)
                    if old is not None:
                        value = old[0] + transport[key[i]][key[j]]
                        if value < best_value:
                            best_value = value
                            best_previous = i
                    previous_bits ^= previous_bit
                if best_previous >= 0:
                    dp[mask][j] = (best_value, best_previous)
                bits ^= bit
        if time.perf_counter() >= local_deadline:
            return None
        full = size - 1
        last = min(
            dp[full],
            key=lambda j: dp[full][j][0] + transport[key[j]][0],
        )
        best_cost = dp[full][last][0] + transport[key[last]][0]
        reversed_order = []
        mask = full
        current = last
        while current >= 0:
            reversed_order.append(key[current])
            previous = dp[mask][current][1]
            mask ^= 1 << current
            current = previous
        reversed_order.reverse()
        return reversed_order, best_cost

    def get_route(customers, local_deadline, polish):
        key = tuple(sorted(customers))
        if not key:
            return [], 0.0
        cached = route_cache.get(key)
        if cached is None:
            order, cost = initial_route(key, local_deadline)
            cached = (tuple(order), cost, False)
            route_cache[key] = cached

        if (
            polish
            and not cached[2]
            and len(key) <= 10
            and time.perf_counter() < local_deadline
        ):
            exact = exact_route(key, local_deadline)
            if exact is not None and exact[1] <= cached[1] + eps:
                cached = (tuple(exact[0]), exact[1], True)
                route_cache[key] = cached
        elif polish and not cached[2] and len(key) >= 3:
            trial = list(cached[0])
            mode = rng.randrange(5)
            if mode == 0:
                i = rng.randrange(len(trial))
                j = rng.randrange(len(trial))
                if i > j:
                    i, j = j, i
                trial[i : j + 1] = reversed(trial[i : j + 1])
            elif mode == 1:
                i = rng.randrange(len(trial))
                j = rng.randrange(len(trial))
                trial[i], trial[j] = trial[j], trial[i]
            elif mode == 2:
                node = trial.pop(rng.randrange(len(trial)))
                trial.insert(rng.randrange(len(trial) + 1), node)
            elif mode == 3:
                rng.shuffle(trial)
            else:
                cut = rng.randrange(1, len(trial))
                trial = trial[cut:] + trial[:cut]
            trial = two_opt(trial, local_deadline)
            if time.perf_counter() < local_deadline:
                trial = relocate_route(trial, local_deadline)
            value = route_cost(trial)
            if value + eps < cached[1]:
                cached = (tuple(trial), value, False)
                route_cache[key] = cached
        return list(cached[0]), cached[1]

    def weighted_quantile(values, mass, quantile):
        if not values or mass <= eps:
            return 0.0
        ordered = sorted(values, key=lambda item: item[0])
        target = min(1.0, max(0.0, quantile)) * mass
        accumulated = 0.0
        for value, weight in ordered:
            accumulated += weight
            if accumulated + eps >= target:
                return value
        return ordered[-1][0]

    def evaluate(setup, assignment, params, local_deadline, route_effort):
        if time.perf_counter() >= local_deadline:
            return None

        cover_factor = params[0]
        reserve_quantile = params[1]
        reserve_scale = params[2]
        production_threshold = params[3]
        future_scale = params[4]
        stock_guard = params[5]
        horizon_scale = params[6]
        reserve_mean_mix = params[7]

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

        records = []
        active_assignment = [[-1 for _ in range(N)] for _ in range(H)]
        used_setup = [0 for _ in range(H)]
        expected_operating_cost = 0.0
        expected_lost = [[0.0 for _ in range(N)] for _ in range(H)]
        expected_delivered = [[0.0 for _ in range(N)] for _ in range(H)]
        expected_production = [0.0 for _ in range(H)]
        elapsed_now = max(0.0, time.perf_counter() - started)
        segment_limit = 10 + int(6.0 * math.sqrt(math.log1p(elapsed_now)))
        tick = 0

        for t in range(H):
            level_records = [None for _ in groups[t]]
            for nid, group in enumerate(groups[t]):
                if time.perf_counter() >= local_deadline:
                    return None
                if t == 0:
                    previous = initial[:]
                else:
                    previous = list(records[t - 1][group["parent"]][2])

                members = group["members"]
                mass = group["probability"]
                current_demand = group["demand"]
                factory_stock = max(0.0, previous[0])

                next_prod = next_setup[t]
                reserve_end = H - 1 if next_prod < 0 else next_prod - 1
                reserve_target = 0.0
                future_shadow = 0.0
                if reserve_end > t and mass > eps:
                    branch_needs = []
                    weighted_mean = 0.0
                    for s in members:
                        branch_need = 0.0
                        for tau in range(t + 1, reserve_end + 1):
                            vehicle_need = [0.0 for _ in range(M)]
                            delay = tau - t
                            for c in range(N):
                                k = assignment[tau][c]
                                marginal_value = (
                                    penalty[c]
                                    - holding[c + 1] * delay
                                    - holding[0] * delay
                                )
                                if 0 <= k < M and marginal_value > production_cost:
                                    vehicle_need[k] += demand[s][tau][c]
                                    if marginal_value > future_shadow:
                                        future_shadow = marginal_value
                            for k in range(M):
                                branch_need += min(vehicle_capacity, vehicle_need[k])
                        branch_needs.append((branch_need, probabilities[s]))
                        weighted_mean += probabilities[s] * branch_need
                        tick += 1
                        if (
                            (tick & 31) == 0
                            and time.perf_counter() >= local_deadline
                        ):
                            return None
                    weighted_mean /= mass
                    quantile_need = weighted_quantile(
                        branch_needs, mass, reserve_quantile
                    )
                    blended = (
                        reserve_mean_mix * weighted_mean
                        + (1.0 - reserve_mean_mix) * quantile_need
                    )
                    reserve_target = min(
                        capacities[0], max(0.0, reserve_scale * blended)
                    )

                segments = []
                customer_limit = [0.0 for _ in range(N)]
                for c in range(N):
                    k = assignment[t][c]
                    if k < 0 or k >= M:
                        continue
                    prior_stock = max(0.0, previous[c + 1])
                    remaining_space = max(0.0, capacities[c + 1] - prior_stock)
                    maximum_delivery = min(
                        remaining_space,
                        max(0.0, minimum_suffix[t][nid][c]),
                    )
                    if maximum_delivery <= eps:
                        continue
                    customer_limit[c] = maximum_delivery

                    shortage = max(0.0, current_demand[c] - prior_stock)
                    current_quantity = min(shortage, maximum_delivery)
                    if current_quantity > eps and penalty[c] > eps:
                        segments.append(
                            (penalty[c] + 1e-12, c, k, current_quantity)
                        )

                    next_customer_visit = next_visit[t][c]
                    natural_end = (
                        H - 1
                        if next_customer_visit < 0
                        else next_customer_visit - 1
                    )
                    natural_span = max(0, natural_end - t)
                    if horizon_scale <= 1.0:
                        span = int(math.floor(natural_span * horizon_scale + eps))
                    else:
                        span = natural_span + int(
                            round(
                                (horizon_scale - 1.0)
                                * max(1.0, math.sqrt(float(H)))
                            )
                        )
                    horizon = min(H - 1, t + max(0, span))
                    base_inventory = max(0.0, prior_stock - current_demand[c])

                    if horizon > t and mass > eps:
                        conditional_mean = 0.0
                        for s in members:
                            amount = (
                                suffix[s][t + 1][c]
                                - suffix[s][horizon + 1][c]
                            )
                            conditional_mean += probabilities[s] * amount
                        conditional_mean /= mass
                        minimum_future_total = min(
                            suffix[s][t + 1][c] for s in members
                        )
                        future_space = max(
                            0.0, maximum_delivery - current_quantity
                        )
                        target_extra = min(
                            future_space,
                            max(0.0, minimum_future_total - base_inventory),
                            max(
                                0.0,
                                cover_factor * conditional_mean - base_inventory,
                            ),
                        )

                        if target_extra > eps:
                            points = [0.0, target_extra]
                            upper_level = base_inventory + target_extra
                            for s in members:
                                cumulative = 0.0
                                for tau in range(t + 1, H):
                                    cumulative += demand[s][tau][c]
                                    if base_inventory + eps < cumulative < upper_level - eps:
                                        points.append(cumulative - base_inventory)
                            points = sorted(set(points))
                            if len(points) > segment_limit + 1:
                                old_points = points
                                last_index = len(old_points) - 1
                                points = [
                                    old_points[
                                        int(round(i * last_index / segment_limit))
                                    ]
                                    for i in range(segment_limit + 1)
                                ]
                                points[0] = 0.0
                                points[-1] = target_extra
                                points = sorted(set(points))

                            for point_index in range(len(points) - 1):
                                left = points[point_index]
                                right = points[point_index + 1]
                                amount = right - left
                                if amount <= eps:
                                    continue
                                level = base_inventory + 0.5 * (left + right)
                                expected_delay = 0.0
                                for s in members:
                                    cumulative = 0.0
                                    consume_time = H - 1
                                    for tau in range(t + 1, H):
                                        cumulative += demand[s][tau][c]
                                        if cumulative + eps >= level:
                                            consume_time = tau
                                            break
                                    expected_delay += (
                                        probabilities[s] * (consume_time - t)
                                    )
                                expected_delay /= mass
                                marginal_value = (
                                    penalty[c] * future_scale
                                    - holding[c + 1] * expected_delay
                                )
                                if marginal_value > eps:
                                    segments.append(
                                        (marginal_value, c, k, amount)
                                    )

                    tick += 1
                    if (
                        (tick & 31) == 0
                        and time.perf_counter() >= local_deadline
                    ):
                        return None

                segments.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                remaining_vehicle = [vehicle_capacity for _ in range(M)]
                remaining_customer = customer_limit[:]
                delivered = [0.0 for _ in range(N)]

                protected_stock = min(factory_stock, reserve_target)
                free_stock = max(0.0, factory_stock - protected_stock)
                production_left = production_capacity if setup[t] else 0.0
                stock_used = 0.0
                production_used = 0.0
                guard_value = max(0.0, stock_guard * future_shadow)

                for value, c, k, requested in segments:
                    possible = min(
                        requested,
                        remaining_vehicle[k],
                        remaining_customer[c],
                    )
                    if possible <= eps:
                        continue
                    taken = 0.0

                    use = min(possible, free_stock)
                    if use > eps:
                        free_stock -= use
                        stock_used += use
                        taken += use
                        possible -= use

                    if (
                        possible > eps
                        and protected_stock > eps
                        and value + eps >= guard_value
                    ):
                        use = min(possible, protected_stock)
                        protected_stock -= use
                        stock_used += use
                        taken += use
                        possible -= use

                    if (
                        possible > eps
                        and production_left > eps
                        and value > production_cost * production_threshold + eps
                    ):
                        use = min(possible, production_left)
                        production_left -= use
                        production_used += use
                        taken += use
                        possible -= use

                    if taken > eps:
                        delivered[c] += taken
                        remaining_customer[c] -= taken
                        remaining_vehicle[k] -= taken

                factory_after_shipping = max(0.0, factory_stock - stock_used)
                extra_production = 0.0
                if (
                    setup[t]
                    and production_left > eps
                    and future_shadow
                    > production_cost * production_threshold + holding[0] + eps
                ):
                    extra_production = min(
                        max(0.0, reserve_target - factory_after_shipping),
                        production_left,
                        max(0.0, capacities[0] - factory_after_shipping),
                    )
                    if extra_production <= eps:
                        extra_production = 0.0

                production_quantity = production_used + extra_production
                if production_quantity > eps:
                    used_setup[t] = 1
                else:
                    production_quantity = 0.0
                factory_end = factory_after_shipping + extra_production
                if factory_end < eps:
                    factory_end = 0.0

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

        def set_route_cost(customer_set, polish=False):
            if not customer_set:
                return 0.0
            return get_route(
                [c + 1 for c in customer_set], local_deadline, polish
            )[1]

        if route_effort:
            for t in range(H):
                if time.perf_counter() >= local_deadline:
                    return None
                active = [c for c in range(N) if active_assignment[t][c] >= 0]
                if len(active) <= 1:
                    continue
                group_count = len(groups[t])
                vectors = [[0.0 for _ in range(group_count)] for _ in range(N)]
                for nid, record in enumerate(records[t]):
                    for _, c, quantity in record[1]:
                        vectors[c][nid] = quantity

                current_sets = [set() for _ in range(M)]
                for c in active:
                    current_sets[active_assignment[t][c]].add(c)
                current_loads = [
                    [
                        sum(vectors[c][nid] for c in current_sets[k])
                        for nid in range(group_count)
                    ]
                    for k in range(M)
                ]

                def partition_cost(sets):
                    total = 0.0
                    for customer_set in sets:
                        total += set_route_cost(customer_set, False)
                    return total

                best_sets = [set(x) for x in current_sets]
                best_loads = [row[:] for row in current_loads]
                best_partition_cost = partition_cost(best_sets)

                orderings = [
                    sorted(
                        active,
                        key=lambda c: (
                            max(vectors[c]),
                            sum(vectors[c]),
                            transport[0][c + 1],
                        ),
                        reverse=True,
                    ),
                    sorted(
                        active,
                        key=lambda c: (
                            transport[0][c + 1],
                            max(vectors[c]),
                        ),
                        reverse=True,
                    ),
                ]
                for ordering in orderings:
                    trial_sets = [set() for _ in range(M)]
                    trial_loads = [
                        [0.0 for _ in range(group_count)] for _ in range(M)
                    ]
                    feasible_partition = True
                    for c in ordering:
                        choices = []
                        for k in range(M):
                            feasible = True
                            for nid in range(group_count):
                                if (
                                    trial_loads[k][nid] + vectors[c][nid]
                                    > vehicle_capacity + 1e-8
                                ):
                                    feasible = False
                                    break
                            if feasible:
                                old_cost = set_route_cost(trial_sets[k], False)
                                new_set = set(trial_sets[k])
                                new_set.add(c)
                                delta = set_route_cost(new_set, False) - old_cost
                                pressure = max(
                                    (
                                        trial_loads[k][nid] + vectors[c][nid]
                                    )
                                    / max(vehicle_capacity, eps)
                                    for nid in range(group_count)
                                )
                                choices.append((delta + 1e-7 * pressure, k))
                        if not choices:
                            feasible_partition = False
                            break
                        k = min(choices)[1]
                        trial_sets[k].add(c)
                        for nid in range(group_count):
                            trial_loads[k][nid] += vectors[c][nid]
                    if feasible_partition:
                        value = partition_cost(trial_sets)
                        if value + eps < best_partition_cost:
                            best_partition_cost = value
                            best_sets = trial_sets
                            best_loads = trial_loads
                    if time.perf_counter() >= local_deadline:
                        return None

                improving = True
                while improving and time.perf_counter() < local_deadline:
                    improving = False
                    best_move = None
                    best_delta = -eps
                    for a in range(M):
                        for c in tuple(best_sets[a]):
                            for b in range(M):
                                if a == b:
                                    continue
                                feasible = True
                                for nid in range(group_count):
                                    if (
                                        best_loads[b][nid] + vectors[c][nid]
                                        > vehicle_capacity + 1e-8
                                    ):
                                        feasible = False
                                        break
                                if not feasible:
                                    continue
                                old_value = set_route_cost(best_sets[a], False)
                                old_value += set_route_cost(best_sets[b], False)
                                new_a = set(best_sets[a])
                                new_b = set(best_sets[b])
                                new_a.remove(c)
                                new_b.add(c)
                                new_value = set_route_cost(new_a, False)
                                new_value += set_route_cost(new_b, False)
                                delta = new_value - old_value
                                if delta < best_delta:
                                    best_delta = delta
                                    best_move = (a, b, c)
                    if best_move is not None:
                        a, b, c = best_move
                        best_sets[a].remove(c)
                        best_sets[b].add(c)
                        for nid in range(group_count):
                            best_loads[a][nid] -= vectors[c][nid]
                            best_loads[b][nid] += vectors[c][nid]
                        improving = True
                    if time.perf_counter() >= local_deadline:
                        return None

                best_swap = None
                best_delta = -eps
                for a in range(M):
                    for b in range(a + 1, M):
                        for c in tuple(best_sets[a]):
                            for d in tuple(best_sets[b]):
                                feasible = True
                                for nid in range(group_count):
                                    load_a = (
                                        best_loads[a][nid]
                                        - vectors[c][nid]
                                        + vectors[d][nid]
                                    )
                                    load_b = (
                                        best_loads[b][nid]
                                        - vectors[d][nid]
                                        + vectors[c][nid]
                                    )
                                    if (
                                        load_a > vehicle_capacity + 1e-8
                                        or load_b > vehicle_capacity + 1e-8
                                    ):
                                        feasible = False
                                        break
                                if not feasible:
                                    continue
                                old_value = set_route_cost(best_sets[a], False)
                                old_value += set_route_cost(best_sets[b], False)
                                new_a = set(best_sets[a])
                                new_b = set(best_sets[b])
                                new_a.remove(c)
                                new_b.remove(d)
                                new_a.add(d)
                                new_b.add(c)
                                delta = (
                                    set_route_cost(new_a, False)
                                    + set_route_cost(new_b, False)
                                    - old_value
                                )
                                if delta < best_delta:
                                    best_delta = delta
                                    best_swap = (a, b, c, d)
                                tick += 1
                                if (
                                    (tick & 127) == 0
                                    and time.perf_counter() >= local_deadline
                                ):
                                    return None
                if best_swap is not None:
                    a, b, c, d = best_swap
                    best_sets[a].remove(c)
                    best_sets[b].remove(d)
                    best_sets[a].add(d)
                    best_sets[b].add(c)

                active_assignment[t] = [-1 for _ in range(N)]
                for k in range(M):
                    for c in best_sets[k]:
                        active_assignment[t][c] = k
                for nid, record in enumerate(records[t]):
                    remapped = tuple(
                        (active_assignment[t][c], c, quantity)
                        for _, c, quantity in record[1]
                    )
                    records[t][nid] = (
                        record[0],
                        remapped,
                        record[2],
                        record[3],
                    )

        routes = []
        transportation_total = 0.0
        for t in range(H):
            period_routes = []
            for k in range(M):
                customers = [
                    c + 1
                    for c in range(N)
                    if active_assignment[t][c] == k
                ]
                if not customers:
                    period_routes.append([0])
                else:
                    order, value = get_route(
                        customers, local_deadline, route_effort
                    )
                    if time.perf_counter() >= local_deadline:
                        return None
                    period_routes.append([0] + order + [0])
                    transportation_total += value
            routes.append(period_routes)

        total_cost = (
            expected_operating_cost
            + transportation_total
            + setup_cost * sum(used_setup)
        )
        return {
            "cost": total_cost,
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
        for s in range(S):
            scenario_production = []
            scenario_delivery = []
            scenario_inventory = []
            scenario_unmet = []
            for t in range(H):
                record = result["records"][t][node_ids[s][t]]
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
                [route[:] for route in period] for period in result["routes"]
            ],
            "production_quantities": production_output,
            "delivery_quantities": delivery_output,
            "end_inventories": inventory_output,
            "unmet_demand_quantities": unmet_output,
        }

    zero_setup = [0 for _ in range(H)]
    zero_assignment = [[-1 for _ in range(N)] for _ in range(H)]
    zero_params = (0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0)
    zero_result = evaluate(
        zero_setup, zero_assignment, zero_params, deadline, False
    )
    if zero_result is None:
        return fallback_solution

    best_result = zero_result
    best_cost = zero_result["cost"]
    average_depot_cost = sum(
        transport[0][c + 1] + transport[c + 1][0] for c in range(N)
    ) / max(1, N)
    route_scale = max(1.0, average_depot_cost)

    def approximate_insertion(row, customer, k):
        members = [c for c in range(N) if row[c] == k and c != customer]
        node = customer + 1
        if not members:
            return transport[0][node] + transport[node][0]
        member_nodes = [c + 1 for c in members]
        value = min(
            transport[0][node]
            + transport[node][v]
            - transport[0][v]
            for v in member_nodes
        )
        value = min(
            value,
            min(
                transport[v][node]
                + transport[node][0]
                - transport[v][0]
                for v in member_nodes
            ),
        )
        for a in member_nodes:
            for b in member_nodes:
                if a != b:
                    value = min(
                        value,
                        transport[a][node]
                        + transport[node][b]
                        - transport[a][b],
                    )
        return max(0.0, value)

    def best_vehicle(row, customer, t, forbidden=-1, planned_load=None):
        best_k = 0
        best_value = float("inf")
        own_load = expected_demand[t][customer]
        for k in range(M):
            if k == forbidden and M > 1:
                continue
            if planned_load is None:
                load = sum(
                    expected_demand[t][c]
                    for c in range(N)
                    if row[c] == k and c != customer
                )
            else:
                load = planned_load[k]
            pressure = (load + own_load) / max(vehicle_capacity, eps)
            travel = approximate_insertion(row, customer, k)
            value = pressure * pressure + 0.34 * travel / route_scale
            value *= 0.985 + 0.03 * rng.random()
            if value < best_value:
                best_value = value
                best_k = k
        return best_k

    def balanced_assignment(selected, jitter):
        result = [[-1 for _ in range(N)] for _ in range(H)]
        planned = [[0.0 for _ in range(N)] for _ in range(H)]
        for c in range(N):
            times = [t for t in range(H) if selected[t][c]]
            for index, t in enumerate(times):
                end = H - 1 if index + 1 == len(times) else times[index + 1] - 1
                planned[t][c] = sum(
                    expected_demand[tau][c] for tau in range(t, end + 1)
                )
        for t in range(H):
            loads = [0.0 for _ in range(M)]
            customers = [c for c in range(N) if selected[t][c]]
            customers.sort(
                key=lambda c: -(
                    planned[t][c]
                    * (penalty[c] + eps)
                    * (0.82 + 0.36 * rng.random() if jitter else 1.0)
                )
            )
            for c in customers:
                best_k = 0
                best_value = float("inf")
                for k in range(M):
                    pressure = (
                        loads[k] + min(vehicle_capacity, planned[t][c])
                    ) / max(vehicle_capacity, eps)
                    travel = approximate_insertion(result[t], c, k)
                    value = pressure * pressure + 0.34 * travel / route_scale
                    if jitter:
                        value *= 0.97 + 0.06 * rng.random()
                    if value < best_value:
                        best_value = value
                        best_k = k
                result[t][c] = best_k
                loads[best_k] += min(vehicle_capacity, planned[t][c])
        return result

    def economic_setup_schedule(phase_shift=0, scale=1.0):
        if production_capacity <= eps:
            return [0 for _ in range(H)]
        rate = sum(sum(row) for row in expected_demand) / max(1, H)
        if rate <= eps:
            return [0 for _ in range(H)]
        carrying = max(holding[0], 1e-6)
        gap = int(round(scale * math.sqrt(max(eps, 2.0 * setup_cost / (carrying * rate)))))
        gap = max(1, min(H, gap))
        capacity_gap = max(1, int(production_capacity / max(rate, eps)))
        gap = min(gap, capacity_gap)
        phase = phase_shift % gap
        return [1 if t % gap == phase else 0 for t in range(H)]

    def make_seed(serial):
        kind = serial % 8
        variant = serial // 8
        selected = [[False for _ in range(N)] for _ in range(H)]
        setup = [1 for _ in range(H)]

        if kind == 0:
            for t in range(H):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (0.10, 0.25, 0.10, 1.00, 1.00, 0.0, 1.0, 1.0)
        elif kind == 1:
            for t in range(H):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            setup = economic_setup_schedule(variant, 0.75)
            params = (0.65, 0.65, 0.55, 0.95, 0.98, 0.25, 1.0, 0.65)
        elif kind == 2:
            gap = 2
            phase = variant % gap
            for t in range(phase, H, gap):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            setup = [1 if t % gap == phase else 0 for t in range(H)]
            params = (1.0, 0.55, 0.9, 0.95, 1.0, 0.35, 1.0, 0.55)
        elif kind == 3:
            gap = max(1, int(round(math.sqrt(float(H)))))
            phase = variant % gap
            for t in range(phase, H, gap):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            setup = economic_setup_schedule(phase, 1.1)
            params = (1.10, 0.70, 1.0, 0.90, 0.98, 0.55, 1.05, 0.45)
        elif kind == 4:
            for c in range(N):
                rate = sum(expected_demand[t][c] for t in range(H)) / max(1, H)
                trip = transport[0][c + 1] + transport[c + 1][0]
                best_gap = 1
                best_value = float("inf")
                for gap in range(1, H + 1):
                    routing_term = 0.55 * trip / gap
                    inventory_term = holding[c + 1] * rate * (gap - 1) * 0.5
                    capacity_loss = penalty[c] * max(
                        0.0, rate - vehicle_capacity / gap
                    )
                    value = routing_term + inventory_term + capacity_loss
                    if value < best_value:
                        best_value = value
                        best_gap = gap
                phase = variant % best_gap
                for t in range(phase, H, best_gap):
                    selected[t][c] = expected_suffix[t][c] > eps
            setup = economic_setup_schedule(variant, 1.0)
            params = (1.0, 0.72, 0.82, 0.92, 1.0, 0.48, 1.1, 0.52)
        elif kind == 5:
            for c in range(N):
                accumulated = 0.0
                trip = transport[0][c + 1] + transport[c + 1][0]
                for t in range(H):
                    accumulated += expected_demand[t][c]
                    benefit = accumulated * max(0.0, penalty[c] - production_cost)
                    if benefit >= (0.35 + 0.08 * (variant % 4)) * trip:
                        selected[t][c] = True
                        accumulated = 0.0
            setup = economic_setup_schedule(variant, 0.8)
            params = (0.82, 0.62, 0.68, 1.05, 0.92, 0.32, 1.25, 0.7)
        elif kind == 6:
            gap = max(1, 1 + (variant % max(1, H)))
            for c in range(N):
                phase = (variant + 3 * c) % gap
                for t in range(phase, H, gap):
                    gross = expected_suffix[t][c] * max(
                        0.0, penalty[c] - production_cost
                    )
                    trip = transport[0][c + 1] + transport[c + 1][0]
                    selected[t][c] = gross > 0.15 * trip
            setup = economic_setup_schedule(variant, 1.25)
            params = (1.18, 0.82, 1.15, 0.88, 0.96, 0.72, 1.2, 0.35)
        else:
            density = 0.25 + 0.65 * rng.random()
            for t in range(H):
                for c in range(N):
                    gross = expected_demand[t][c] * max(
                        0.0, penalty[c] - production_cost
                    )
                    trip = transport[0][c + 1] + transport[c + 1][0]
                    attraction = gross / max(eps, gross + 0.2 * trip)
                    selected[t][c] = rng.random() < min(
                        0.98, density * (0.35 + attraction)
                    )
            setup = economic_setup_schedule(variant, 0.65 + 0.9 * rng.random())
            params = (
                0.35 + 1.15 * rng.random(),
                rng.random(),
                0.15 + 1.25 * rng.random(),
                0.72 + 0.55 * rng.random(),
                0.65 + 0.35 * rng.random(),
                1.1 * rng.random(),
                0.55 + 1.45 * rng.random(),
                rng.random(),
            )

        if production_capacity <= eps:
            setup = [0 for _ in range(H)]
        return setup, balanced_assignment(selected, variant > 0), params

    def clamp_params(params):
        params[0] = min(1.9, max(0.0, params[0]))
        params[1] = min(1.0, max(0.0, params[1]))
        params[2] = min(1.9, max(0.0, params[2]))
        params[3] = min(1.55, max(0.55, params[3]))
        params[4] = min(1.0, max(0.35, params[4]))
        params[5] = min(1.7, max(0.0, params[5]))
        params[6] = min(2.6, max(0.15, params[6]))
        params[7] = min(1.0, max(0.0, params[7]))

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

    def marginal_trip(row, customer):
        k = row[customer]
        if k < 0:
            return transport[0][customer + 1] + transport[customer + 1][0]
        members = [c for c in range(N) if c != customer and row[c] == k]
        if not members:
            return transport[0][customer + 1] + transport[customer + 1][0]
        return max(0.0, approximate_insertion(row, customer, k))

    move_weights = [1.0 for _ in range(13)]

    def choose_mode():
        total = sum(move_weights)
        value = rng.random() * total
        for i, weight in enumerate(move_weights):
            value -= weight
            if value <= 0.0:
                return i
        return len(move_weights) - 1

    def mutate(parent, elapsed, stagnation):
        setup = parent["setup"][:]
        assignment = [row[:] for row in parent["assignment"]]
        params = list(parent["params"])
        depth = 1 + int(0.7 * math.log1p(elapsed))
        if stagnation > 1.0:
            depth += int(math.sqrt(math.log1p(stagnation)))
        primary_mode = choose_mode()

        for step in range(max(1, depth)):
            mode = primary_mode if step == 0 else choose_mode()
            t = rng.randrange(H)
            c = rng.randrange(N)

            if mode == 0:
                candidates = []
                for tt in range(H):
                    for cc in range(N):
                        if assignment[tt][cc] < 0:
                            benefit = (
                                parent["lost"][tt][cc] * penalty[cc]
                                + 0.25
                                * expected_demand[tt][cc]
                                * max(0.0, penalty[cc] - production_cost)
                            )
                            trip = transport[0][cc + 1] + transport[cc + 1][0]
                            candidates.append(
                                (
                                    benefit
                                    - trip * (0.10 + 0.45 * rng.random()),
                                    tt,
                                    cc,
                                )
                            )
                if candidates:
                    _, t, c = max(candidates)
                    assignment[t][c] = best_vehicle(assignment[t], c, t)
                    if production_capacity > eps and rng.random() < 0.75:
                        setup[t] = 1

            elif mode == 1:
                candidates = []
                for tt in range(H):
                    for cc in range(N):
                        if assignment[tt][cc] >= 0:
                            benefit = parent["delivered"][tt][cc] * penalty[cc]
                            trip = marginal_trip(assignment[tt], cc)
                            candidates.append(
                                (benefit - trip * (0.55 + 0.65 * rng.random()), tt, cc)
                            )
                if candidates:
                    _, t, c = min(candidates)
                    assignment[t][c] = -1

            elif mode == 2:
                if rng.random() < 0.65:
                    choices = []
                    for tt in range(H):
                        value = sum(
                            parent["lost"][tt][cc] * penalty[cc]
                            for cc in range(N)
                        )
                        if not setup[tt]:
                            choices.append((value * (0.8 + 0.4 * rng.random()), tt))
                    if choices:
                        setup[max(choices)[1]] = 1
                    else:
                        setup[t] = 1 - int(bool(setup[t]))
                else:
                    setup[t] = 1 - int(bool(setup[t]))

            elif mode == 3:
                active = [
                    (tt, cc)
                    for tt in range(H)
                    for cc in range(N)
                    if assignment[tt][cc] >= 0
                ]
                if active and M > 1:
                    t, c = rng.choice(active)
                    old = assignment[t][c]
                    assignment[t][c] = -1
                    assignment[t][c] = best_vehicle(
                        assignment[t], c, t, old
                    )

            elif mode == 4:
                active_times = [tt for tt in range(H) if assignment[tt][c] >= 0]
                if active_times:
                    old_t = rng.choice(active_times)
                    radius = min(
                        H - 1,
                        1 + int(math.sqrt(max(1.0, elapsed + stagnation))),
                    )
                    new_t = min(
                        H - 1,
                        max(0, old_t + rng.randint(-radius, radius)),
                    )
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
                p = rng.randrange(8)
                sigma = 0.035 + 0.28 / (1.0 + 0.04 * math.sqrt(elapsed))
                params[p] += rng.gauss(0.0, sigma)

            elif mode == 6:
                ranked = sorted(
                    range(N),
                    key=lambda cc: (
                        parent["lost"][t][cc] * penalty[cc]
                        + expected_demand[t][cc]
                        * max(0.0, penalty[cc] - production_cost)
                        - 0.18
                        * (transport[0][cc + 1] + transport[cc + 1][0])
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
                for tt in range(H):
                    if tt % gap == phase:
                        assignment[tt][c] = best_vehicle(assignment[tt], c, tt)
                    elif rng.random() < 0.7:
                        assignment[tt][c] = -1

            elif mode == 8:
                times = [tt for tt in range(H) if assignment[tt][c] >= 0]
                if len(times) >= 2:
                    index = rng.randrange(1, len(times))
                    assignment[times[index]][c] = -1
                    params[0] += 0.12 + 0.30 * rng.random()
                    params[6] += 0.10 + 0.35 * rng.random()
                else:
                    assignment[t][c] = best_vehicle(assignment[t], c, t)

            elif mode == 9:
                window = min(
                    H,
                    1 + int(math.log1p(elapsed + stagnation)) + rng.randrange(2),
                )
                first = rng.randrange(H)
                for offset in range(window):
                    tt = (first + offset) % H
                    if rng.random() < 0.45:
                        assignment[tt] = [-1 for _ in range(N)]
                    if rng.random() < 0.35:
                        setup[tt] = 0
                ranked = sorted(
                    (
                        (parent["lost"][tt][cc] * penalty[cc], tt, cc)
                        for tt in range(H)
                        for cc in range(N)
                    ),
                    reverse=True,
                )
                repair = 1 + int(math.sqrt(max(1.0, elapsed)))
                for _, tt, cc in ranked[: min(len(ranked), repair)]:
                    if rng.random() < 0.7:
                        assignment[tt][cc] = best_vehicle(
                            assignment[tt], cc, tt
                        )
                        if production_capacity > eps:
                            setup[tt] = 1

            elif mode == 10:
                if assignment[t][c] < 0:
                    assignment[t][c] = best_vehicle(assignment[t], c, t)
                else:
                    assignment[t][c] = -1
                if rng.random() < 0.5:
                    setup[t] = 1 - int(bool(setup[t]))

            elif mode == 11:
                gap = 1 + rng.randrange(H)
                phase = rng.randrange(gap)
                setup = [
                    1
                    if production_capacity > eps and tt % gap == phase
                    else 0
                    for tt in range(H)
                ]
                params[2] += 0.15 + 0.45 * rng.random()
                params[1] = min(1.0, params[1] + 0.15 * rng.random())

            else:
                ranked = sorted(
                    range(N),
                    key=lambda cc: (
                        parent["lost"][t][cc] * penalty[cc]
                        + expected_demand[t][cc] * penalty[cc]
                    ),
                    reverse=True,
                )
                additions = 1 + rng.randrange(max(1, min(N, M + 4)))
                setup[t] = 1 if production_capacity > eps else 0
                for cc in ranked[:additions]:
                    assignment[t][cc] = best_vehicle(assignment[t], cc, t)

        if rng.random() < 0.3:
            p = rng.randrange(8)
            params[p] += rng.gauss(0.0, 0.04 + 0.14 * rng.random())
        clamp_params(params)
        if production_capacity <= eps:
            setup = [0 for _ in range(H)]
        return (setup, assignment, tuple(params)), primary_mode

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
        sigma = 0.025 + 0.12 / (1.0 + math.sqrt(elapsed))
        params = [
            blend * a["params"][p]
            + (1.0 - blend) * b["params"][p]
            + rng.gauss(0.0, sigma)
            for p in range(8)
        ]
        clamp_params(params)
        if production_capacity <= eps:
            setup = [0 for _ in range(H)]
        return setup, assignment, tuple(params)

    guided_counter = 0

    def guided_candidate(parent, elapsed):
        nonlocal guided_counter
        slot = guided_counter
        guided_counter += 1
        mode = slot % 6
        cycle = slot // 6
        setup = parent["setup"][:]
        assignment = [row[:] for row in parent["assignment"]]
        params = list(parent["params"])

        if mode == 0:
            index = cycle % (H * N)
            t = index // N
            c = index % N
            if assignment[t][c] < 0:
                assignment[t][c] = best_vehicle(assignment[t], c, t)
                if production_capacity > eps:
                    setup[t] = 1
            else:
                assignment[t][c] = -1
        elif mode == 1:
            index = cycle % (H * N)
            t = index // N
            c = index % N
            if assignment[t][c] >= 0 and M > 1:
                assignment[t][c] = (assignment[t][c] + 1 + cycle // (H * N)) % M
            else:
                assignment[t][c] = best_vehicle(assignment[t], c, t)
        elif mode == 2:
            t = cycle % H
            setup[t] = 1 - int(bool(setup[t]))
        elif mode == 3:
            index = cycle % (H * N)
            t = index // N
            c = index % N
            if assignment[t][c] >= 0:
                direction = -1 if (cycle // (H * N)) % 2 == 0 else 1
                new_t = min(H - 1, max(0, t + direction))
                if new_t != t:
                    assignment[t][c] = -1
                    assignment[new_t][c] = best_vehicle(
                        assignment[new_t], c, new_t
                    )
            else:
                assignment[t][c] = best_vehicle(assignment[t], c, t)
        elif mode == 4:
            p = cycle % 8
            direction = -1.0 if (cycle // 8) % 2 == 0 else 1.0
            scale = 0.22 / (1.0 + 0.08 * math.sqrt(cycle // 16 + 1.0))
            scale += 0.025 / (1.0 + math.log1p(elapsed))
            params[p] += direction * scale
        else:
            c = cycle % N
            times = [t for t in range(H) if assignment[t][c] >= 0]
            if len(times) >= 2:
                remove_index = 1 + (cycle // max(1, N)) % (len(times) - 1)
                assignment[times[remove_index]][c] = -1
                params[0] += 0.18
                params[6] += 0.15
            else:
                t = (cycle // max(1, N)) % H
                assignment[t][c] = best_vehicle(assignment[t], c, t)

        clamp_params(params)
        if production_capacity <= eps:
            setup = [0 for _ in range(H)]
        return setup, assignment, tuple(params)

    zero_entry = entry_from_result(zero_result)
    population = [zero_entry]
    current = zero_entry
    seed_serial = 0
    last_improvement = time.perf_counter()

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        elapsed = max(0.0, now - started)
        stagnation = max(0.0, now - last_improvement)
        parent = None
        credited_mode = None

        if seed_serial < 8:
            candidate = make_seed(seed_serial)
            seed_serial += 1
        else:
            restart_probability = 0.055 + min(
                0.34, stagnation / (20.0 + elapsed + stagnation)
            )
            crossover_probability = min(
                0.34, 0.08 + math.log1p(elapsed) / 30.0
            )
            guided_probability = min(
                0.42, 0.16 + math.log1p(elapsed) / 28.0
            )
            draw = rng.random()
            if draw < 0.025 + min(0.08, stagnation / 100.0):
                candidate = (
                    best_result["setup"][:],
                    [row[:] for row in best_result["assignment"]],
                    tuple(best_result["params"]),
                )
            elif draw < restart_probability:
                candidate = make_seed(seed_serial)
                seed_serial += 1
            elif draw < restart_probability + guided_probability:
                candidate = guided_candidate(
                    entry_from_result(best_result), elapsed
                )
            elif (
                len(population) >= 2
                and draw
                < restart_probability + guided_probability + crossover_probability
            ):
                tournament = min(
                    len(population), 2 + int(math.log1p(elapsed) / 2.5)
                )
                a = min(
                    (rng.choice(population) for _ in range(tournament)),
                    key=lambda entry: entry["cost"],
                )
                b = rng.choice(population)
                candidate = crossover(a, b, elapsed)
            else:
                tournament = min(
                    len(population), 2 + int(math.log1p(elapsed) / 2.0)
                )
                if rng.random() < 0.3:
                    parent = current
                else:
                    parent = min(
                        (rng.choice(population) for _ in range(tournament)),
                        key=lambda entry: entry["cost"],
                    )
                candidate, credited_mode = mutate(parent, elapsed, stagnation)

        if time.perf_counter() >= deadline:
            break
        route_probability = min(
            0.72,
            0.12 + 0.52 * (1.0 - math.exp(-elapsed / 45.0)),
        )
        route_effort = rng.random() < route_probability
        result = evaluate(
            candidate[0], candidate[1], candidate[2], deadline, route_effort
        )
        if result is None:
            break
        entry = entry_from_result(result)
        cost = entry["cost"]

        if credited_mode is not None and parent is not None:
            if cost + 1e-9 < parent["cost"]:
                move_weights[credited_mode] *= 1.09
            else:
                move_weights[credited_mode] = max(
                    0.2, move_weights[credited_mode] * 0.995
                )
            if max(move_weights) > 50.0:
                move_weights = [0.5 + 0.5 * weight for weight in move_weights]

        tolerance = 1e-9 * (1.0 + abs(best_cost))
        if cost < best_cost - tolerance:
            best_cost = cost
            best_result = result
            last_improvement = time.perf_counter()

        temperature = 0.024 * (1.0 + abs(best_cost))
        temperature /= 1.0 + math.log1p(elapsed)
        temperature *= 1.0 + min(3.0, stagnation / 10.0)
        accept = cost <= current["cost"]
        if not accept and temperature > eps:
            scaled = (cost - current["cost"]) / temperature
            if scaled < 45.0 and rng.random() < math.exp(-scaled):
                accept = True
        if accept:
            current = entry

        target_population = 7 + int(2.8 * math.sqrt(elapsed + 1.0))
        new_signature = signature(entry)
        duplicate = -1
        for index, old in enumerate(population):
            if signature(old) == new_signature:
                duplicate = index
                break
        if duplicate >= 0:
            if cost < population[duplicate]["cost"]:
                population[duplicate] = entry
        else:
            population.append(entry)
            if len(population) > target_population:
                population.sort(key=lambda item: item["cost"])
                elite = max(2, target_population // 3)
                if len(population) > target_population:
                    removable = rng.randrange(elite, len(population))
                    population.pop(removable)

        cache_target = 300 + int(70.0 * math.sqrt(elapsed + 1.0))
        while len(route_cache) > cache_target:
            route_cache.pop(next(iter(route_cache)))

    try:
        return materialize(best_result)
    except BaseException:
        return fallback_solution


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    try:
        return _solve(instance, time_limit_s, started)
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
