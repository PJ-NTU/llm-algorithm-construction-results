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
        if not isinstance(instance, dict):
            instance = {}
        H = _pos_int(instance.get("num_periods", 1), 1)
        N = _pos_int(instance.get("num_customers", 1), 1)
        M = _pos_int(instance.get("num_vehicles", 1), 1)
        S = _pos_int(instance.get("num_scenarios", 1), 1)
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
    budget = max(0.0, _num(time_limit_s, 0.0))
    hard_deadline = started + budget

    H = int(instance["num_periods"])
    N = int(instance["num_customers"])
    M = int(instance["num_vehicles"])
    S = int(instance["num_scenarios"])
    if H < 1 or N < 1 or M < 1 or S < 1:
        return _fallback(instance)

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
        return _fallback(instance)
    probabilities = [p / probability_sum for p in probabilities]

    initial = [max(0.0, float(x)) for x in instance["initial_inventories"]]
    capacities = [max(0.0, float(x)) for x in instance["inventory_capacities"]]
    holding = [max(0.0, float(x)) for x in instance["holding_costs"]]
    penalty = [max(0.0, float(x)) for x in instance["unmet_demand_penalties"]]
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
        return _fallback(instance)

    eps = 1e-10
    baseline_started = time.perf_counter()
    baseline_production = [[0.0 for _ in range(H)] for _ in range(S)]
    baseline_delivery = [
        [
            [[0.0 for _ in range(N)] for _ in range(M)]
            for _ in range(H)
        ]
        for _ in range(S)
    ]
    baseline_inventory = [
        [[0.0 for _ in range(N + 1)] for _ in range(H)]
        for _ in range(S)
    ]
    baseline_unmet = [
        [[0.0 for _ in range(N)] for _ in range(H)]
        for _ in range(S)
    ]

    for s in range(S):
        previous = initial[:]
        for t in range(H):
            ending = [0.0 for _ in range(N + 1)]
            ending[0] = previous[0]
            for c in range(N):
                d = demand[s][t][c]
                served = min(previous[c + 1], d)
                ending[c + 1] = max(0.0, previous[c + 1] - served)
                baseline_unmet[s][t][c] = max(0.0, d - served)
            baseline_inventory[s][t] = ending
            previous = ending

    baseline_solution = {
        "problem_type": "multi_period_stochastic_production_routing_with_lost_sales",
        "production_setup": [0 for _ in range(H)],
        "vehicle_routes": [[[0] for _ in range(M)] for _ in range(H)],
        "production_quantities": baseline_production,
        "delivery_quantities": baseline_delivery,
        "end_inventories": baseline_inventory,
        "unmet_demand_quantities": baseline_unmet,
    }
    baseline_time = time.perf_counter() - baseline_started
    reserve_time = max(0.006, 2.4 * baseline_time + 0.004)
    search_deadline = hard_deadline - reserve_time
    if time.perf_counter() >= search_deadline:
        return baseline_solution

    suffix = [
        [[0.0 for _ in range(N)] for _ in range(H + 1)]
        for _ in range(S)
    ]
    for s in range(S):
        for t in range(H - 1, -1, -1):
            for c in range(N):
                suffix[s][t][c] = demand[s][t][c] + suffix[s][t + 1][c]
        if time.perf_counter() >= search_deadline:
            return baseline_solution

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

    node_id = [[-1 for _ in range(H)] for _ in range(S)]
    levels = []
    for t in range(H):
        index = {}
        level = []
        for s in range(S):
            parent = node_id[s][t - 1] if t > 0 else -1
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
            node_id[s][t] = nid
        for node in level:
            members = node["members"]
            node["minimum_suffix"] = [
                min(suffix[s][t][c] for s in members)
                for c in range(N)
            ]
        levels.append(level)
        if time.perf_counter() >= search_deadline:
            return baseline_solution

    seed_value = 146959810 + H * 1000003 + N * 9176 + M * 65537 + S * 131071
    seed_value ^= int(sum(initial) * 1009.0) & 0x7fffffff
    seed_value ^= int(sum(sum(row) for row in expected_demand) * 101.0) & 0x7fffffff
    rng = random.Random(seed_value)
    route_cache = {}

    def route_cost(order):
        if not order:
            return 0.0
        total = transport[0][order[0]] + transport[order[-1]][0]
        for a, b in zip(order, order[1:]):
            total += transport[a][b]
        return total

    def improve_order(order, local_deadline, relocate=True):
        order = order[:]
        n = len(order)
        while n >= 3:
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
                    if ((checks & 127) == 0 and
                            time.perf_counter() >= local_deadline):
                        return order
            if best_i < 0:
                break
            order[best_i:best_j + 1] = reversed(order[best_i:best_j + 1])
            if time.perf_counter() >= local_deadline:
                return order

        if not relocate or n < 3:
            return order
        while True:
            best_delta = -eps
            best_i = -1
            best_pos = -1
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
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_pos = pos
                    checks += 1
                    if ((checks & 127) == 0 and
                            time.perf_counter() >= local_deadline):
                        return order
            if best_i < 0:
                break
            node = order.pop(best_i)
            order.insert(best_pos, node)
            if time.perf_counter() >= local_deadline:
                break
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
        nearest = improve_order(nearest, local_deadline, False)
        best = nearest
        best_cost = route_cost(best)

        if time.perf_counter() < local_deadline:
            remaining = list(key)
            insertion = []
            while remaining:
                choice = None
                for node in remaining:
                    for pos in range(len(insertion) + 1):
                        before = 0 if pos == 0 else insertion[pos - 1]
                        after = 0 if pos == len(insertion) else insertion[pos]
                        delta = (
                            transport[before][node] + transport[node][after]
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
            insertion = improve_order(insertion, local_deadline, False)
            value = route_cost(insertion)
            if value + eps < best_cost:
                best = insertion
                best_cost = value
        return best, best_cost

    def exact_route(key, local_deadline):
        n = len(key)
        if n == 0:
            return [], 0.0
        dp = {}
        for j in range(n):
            dp[(1 << j, j)] = (transport[0][key[j]], -1)
        checks = 0
        for size in range(2, n + 1):
            for mask in range(1, 1 << n):
                if mask.bit_count() != size:
                    continue
                for j in range(n):
                    bit = 1 << j
                    if not mask & bit:
                        continue
                    prior_mask = mask ^ bit
                    best_value = float("inf")
                    best_prior = -1
                    for k in range(n):
                        if prior_mask & (1 << k):
                            old = dp.get((prior_mask, k))
                            if old is not None:
                                value = old[0] + transport[key[k]][key[j]]
                                if value < best_value:
                                    best_value = value
                                    best_prior = k
                    if best_prior >= 0:
                        dp[(mask, j)] = (best_value, best_prior)
                    checks += 1
                    if ((checks & 511) == 0 and
                            time.perf_counter() >= local_deadline):
                        return None
        full = (1 << n) - 1
        end = min(
            range(n),
            key=lambda j: dp[(full, j)][0] + transport[key[j]][0],
        )
        cost = dp[(full, end)][0] + transport[key[end]][0]
        order = []
        mask = full
        j = end
        while j >= 0:
            order.append(key[j])
            prior = dp[(mask, j)][1]
            mask ^= 1 << j
            j = prior
        order.reverse()
        return order, cost

    def get_route(customers, local_deadline, effort):
        key = tuple(sorted(customers))
        cached = route_cache.get(key)
        if cached is None:
            order, cost = initial_route(key, local_deadline)
            cached = (tuple(order), cost, False)
            route_cache[key] = cached

        if effort >= 1 and len(key) >= 3 and not cached[2] and time.perf_counter() < local_deadline:
            trial = list(cached[0])
            mode = rng.randrange(4)
            if mode == 0:
                a = rng.randrange(len(trial))
                b = rng.randrange(len(trial))
                if a > b:
                    a, b = b, a
                if a != b:
                    trial[a:b + 1] = reversed(trial[a:b + 1])
            elif mode == 1:
                a = rng.randrange(len(trial))
                b = rng.randrange(len(trial))
                trial[a], trial[b] = trial[b], trial[a]
            elif mode == 2:
                a = rng.randrange(len(trial))
                node = trial.pop(a)
                trial.insert(rng.randrange(len(trial) + 1), node)
            else:
                rng.shuffle(trial)
            trial = improve_order(trial, local_deadline, True)
            value = route_cost(trial)
            if value + eps < cached[1]:
                cached = (tuple(trial), value, False)
                route_cache[key] = cached

        if (effort >= 2 and 2 <= len(key) <= 11 and not cached[2] and
                time.perf_counter() < local_deadline):
            exact = exact_route(key, local_deadline)
            if exact is not None:
                cached = (tuple(exact[0]), exact[1], True)
                route_cache[key] = cached
        return list(cached[0]), cached[1]

    def evaluate(policy_setup, policy_assignment, params, local_deadline, route_effort):
        if time.perf_counter() >= local_deadline:
            return None

        cover_factor, mean_mix, reserve_factor, production_threshold, future_scale, stock_guard, lookahead_factor = params

        next_visit = [[-1 for _ in range(N)] for _ in range(H)]
        last_visit = [-1 for _ in range(N)]
        for t in range(H - 1, -1, -1):
            next_visit[t] = last_visit[:]
            for c in range(N):
                if 0 <= policy_assignment[t][c] < M:
                    last_visit[c] = t

        next_setup = [-1 for _ in range(H)]
        last_setup = -1
        for t in range(H - 1, -1, -1):
            next_setup[t] = last_setup
            if policy_setup[t]:
                last_setup = t

        gap_end = [H - 1 for _ in range(H)]
        future_shadow = [0.0 for _ in range(H)]
        for t in range(H):
            nxt = next_setup[t]
            gap_end[t] = H - 1 if nxt < 0 else nxt - 1
            shadow = 0.0
            for tau in range(t + 1, gap_end[t] + 1):
                delay = holding[0] * (tau - t)
                for c in range(N):
                    if policy_assignment[tau][c] >= 0:
                        shadow = max(shadow, penalty[c] - delay)
            future_shadow[t] = max(0.0, shadow)

        records = []
        active_assignment = [[-1 for _ in range(N)] for _ in range(H)]
        used_setup = [0 for _ in range(H)]
        expected_cost = 0.0
        expected_lost = [[0.0 for _ in range(N)] for _ in range(H)]
        expected_delivered = [[0.0 for _ in range(N)] for _ in range(H)]
        expected_production = [0.0 for _ in range(H)]
        ticks = 0

        for t in range(H):
            stage_records = [None for _ in levels[t]]
            for nid, node in enumerate(levels[t]):
                if time.perf_counter() >= local_deadline:
                    return None
                if t == 0:
                    previous = initial[:]
                else:
                    previous = list(records[t - 1][node["parent"]][2])

                members = node["members"]
                mass = node["mass"]
                current_demand = node["demand"]
                minimum_suffix = node["minimum_suffix"]
                factory_stock = max(0.0, previous[0])

                reserve_target = 0.0
                if (policy_setup[t] and reserve_factor > eps and
                        gap_end[t] > t and mass > eps):
                    weighted_need = 0.0
                    minimum_need = None
                    for s in members:
                        branch_need = 0.0
                        for tau in range(t + 1, gap_end[t] + 1):
                            vehicle_need = [0.0 for _ in range(M)]
                            delay = holding[0] * (tau - t)
                            for c in range(N):
                                k = policy_assignment[tau][c]
                                if (0 <= k < M and penalty[c] >
                                        production_cost * production_threshold + delay):
                                    vehicle_need[k] += demand[s][tau][c]
                            branch_need += sum(
                                min(vehicle_capacity, amount)
                                for amount in vehicle_need
                            )
                        weighted_need += probabilities[s] * branch_need
                        if minimum_need is None or branch_need < minimum_need:
                            minimum_need = branch_need
                        ticks += 1
                        if ((ticks & 31) == 0 and
                                time.perf_counter() >= local_deadline):
                            return None
                    weighted_need /= mass
                    if minimum_need is None:
                        minimum_need = 0.0
                    target = mean_mix * weighted_need + (1.0 - mean_mix) * minimum_need
                    reserve_target = min(capacities[0], max(0.0, reserve_factor * target))

                segments = []
                customer_limit = [0.0 for _ in range(N)]
                extra_steps = int(round(lookahead_factor * max(1.0, math.sqrt(H))))

                for c in range(N):
                    k = policy_assignment[t][c]
                    if k < 0 or k >= M:
                        continue
                    prior_stock = max(0.0, previous[c + 1])
                    space = max(0.0, capacities[c + 1] - prior_stock)
                    maximum_delivery = min(space, max(0.0, minimum_suffix[c]))
                    if maximum_delivery <= eps:
                        continue

                    shortage = max(0.0, current_demand[c] - prior_stock)
                    immediate = min(shortage, maximum_delivery)
                    if immediate > eps and penalty[c] > eps:
                        segments.append((penalty[c], c, k, immediate))

                    nxt = next_visit[t][c]
                    natural_horizon = H - 1 if nxt < 0 else nxt - 1
                    horizon = max(natural_horizon, min(H - 1, t + extra_steps))
                    if horizon > t and mass > eps:
                        weighted = 0.0
                        minimum = None
                        for s in members:
                            amount = suffix[s][t + 1][c] - suffix[s][horizon + 1][c]
                            weighted += probabilities[s] * amount
                            if minimum is None or amount < minimum:
                                minimum = amount
                        weighted /= mass
                        if minimum is None:
                            minimum = 0.0
                        target_base = mean_mix * weighted + (1.0 - mean_mix) * minimum
                        maximum_ending = max(0.0, capacities[c + 1] - current_demand[c])
                        minimum_future = max(0.0, minimum_suffix[c] - current_demand[c])
                        target_ending = min(
                            maximum_ending,
                            minimum_future,
                            max(0.0, cover_factor * target_base),
                        )
                        natural_ending = max(0.0, prior_stock - current_demand[c])
                        advance = max(0.0, target_ending - natural_ending)
                        advance = min(
                            advance,
                            max(0.0, maximum_delivery - immediate),
                        )
                        average_delay = max(1.0, 0.5 * (horizon - t + 1))
                        value = (
                            penalty[c] * future_scale
                            - holding[c + 1] * average_delay
                        )
                        if advance > eps and value > eps:
                            segments.append((value, c, k, advance))
                    customer_limit[c] = maximum_delivery
                    ticks += 1
                    if ((ticks & 63) == 0 and
                            time.perf_counter() >= local_deadline):
                        return None

                segments.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                remaining_vehicle = [vehicle_capacity for _ in range(M)]
                remaining_customer = customer_limit[:]
                delivered = [0.0 for _ in range(N)]

                protected = min(factory_stock, reserve_target)
                free_stock = max(0.0, factory_stock - protected)
                protected_stock = protected
                production_left = production_capacity if policy_setup[t] else 0.0
                stock_used = 0.0
                production_used = 0.0
                guard_value = stock_guard * future_shadow[t]

                for value, c, k, amount in segments:
                    possible = min(amount, remaining_vehicle[k], remaining_customer[c])
                    if possible <= eps:
                        continue
                    taken = 0.0

                    use = min(possible, free_stock)
                    if use > eps:
                        free_stock -= use
                        stock_used += use
                        possible -= use
                        taken += use

                    if (possible > eps and protected_stock > eps and
                            value + eps >= guard_value):
                        use = min(possible, protected_stock)
                        protected_stock -= use
                        stock_used += use
                        possible -= use
                        taken += use

                    if (possible > eps and production_left > eps and
                            value > production_cost * production_threshold + eps):
                        use = min(possible, production_left)
                        production_left -= use
                        production_used += use
                        possible -= use
                        taken += use

                    if taken > eps:
                        delivered[c] += taken
                        remaining_customer[c] -= taken
                        remaining_vehicle[k] -= taken

                factory_after_shipping = max(0.0, factory_stock - stock_used)
                extra_production = 0.0
                if policy_setup[t] and production_left > eps:
                    extra_production = min(
                        max(0.0, reserve_target - factory_after_shipping),
                        production_left,
                        max(0.0, capacities[0] - factory_after_shipping),
                    )
                production_quantity = production_used + extra_production
                if production_quantity > eps:
                    used_setup[t] = 1
                else:
                    production_quantity = 0.0
                factory_ending = factory_after_shipping + extra_production
                if factory_ending < eps:
                    factory_ending = 0.0

                sparse_delivery = []
                end_inventory = [factory_ending]
                lost_sales = []
                period_cost = production_cost * production_quantity
                period_cost += holding[0] * factory_ending

                for c in range(N):
                    quantity = delivered[c]
                    if quantity > eps:
                        k = policy_assignment[t][c]
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
                expected_cost += mass * period_cost
                stage_records[nid] = (
                    production_quantity,
                    tuple(sparse_delivery),
                    tuple(end_inventory),
                    tuple(lost_sales),
                )
            records.append(stage_records)

        routes = []
        transportation_total = 0.0
        for t in range(H):
            stage_routes = []
            for k in range(M):
                customers = [
                    c + 1 for c in range(N)
                    if active_assignment[t][c] == k
                ]
                if not customers:
                    stage_routes.append([0])
                else:
                    order, cost = get_route(customers, local_deadline, route_effort)
                    if time.perf_counter() >= local_deadline:
                        return None
                    stage_routes.append([0] + order + [0])
                    transportation_total += cost
            routes.append(stage_routes)

        return {
            "cost": expected_cost + transportation_total + setup_cost * sum(used_setup),
            "setup": used_setup,
            "routes": routes,
            "records": records,
            "active_assignment": active_assignment,
            "policy_setup": [int(bool(x)) for x in policy_setup],
            "policy_assignment": [row[:] for row in policy_assignment],
            "params": tuple(params),
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
                record = records[t][node_id[s][t]]
                scenario_production.append(float(record[0]))
                dense = [[0.0 for _ in range(N)] for _ in range(M)]
                for k, c, amount in record[1]:
                    dense[k][c] = float(amount)
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
                [route[:] for route in stage]
                for stage in result["routes"]
            ],
            "production_quantities": production_output,
            "delivery_quantities": delivery_output,
            "end_inventories": inventory_output,
            "unmet_demand_quantities": unmet_output,
        }

    zero_setup = [0 for _ in range(H)]
    zero_assignment = [[-1 for _ in range(N)] for _ in range(H)]
    zero_params = (0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    zero_result = evaluate(
        zero_setup, zero_assignment, zero_params, search_deadline, 0
    )
    if zero_result is None:
        return baseline_solution

    best_result = zero_result
    best_cost = zero_result["cost"]

    average_trip = sum(
        transport[0][c + 1] + transport[c + 1][0]
        for c in range(N)
    ) / max(1, N)
    route_scale = max(1.0, average_trip)

    def insertion_proxy(row, customer, vehicle):
        members = [c for c in range(N) if row[c] == vehicle and c != customer]
        node = customer + 1
        if not members:
            return transport[0][node] + transport[node][0]
        return min(
            transport[node][c + 1] + transport[c + 1][node]
            for c in members
        )

    def best_vehicle(row, customer, t, forbidden=-1, jitter=True):
        best_k = 0
        best_value = float("inf")
        for k in range(M):
            if M > 1 and k == forbidden:
                continue
            members = [c for c in range(N) if row[c] == k and c != customer]
            load = sum(expected_demand[t][c] for c in members)
            projected = load + expected_demand[t][customer]
            pressure = projected / max(vehicle_capacity, eps)
            overflow = max(0.0, pressure - 1.0)
            value = insertion_proxy(row, customer, k) / route_scale
            value += 0.16 * pressure * pressure + 1.4 * overflow * overflow
            if jitter:
                value *= 0.965 + 0.07 * rng.random()
            if value < best_value:
                best_value = value
                best_k = k
        return best_k

    def build_assignment(selected, jitter):
        assignment = [[-1 for _ in range(N)] for _ in range(H)]
        for t in range(H):
            customers = [c for c in range(N) if selected[t][c]]
            customers.sort(
                key=lambda c: -(
                    expected_demand[t][c] * (penalty[c] + eps)
                    * ((0.84 + 0.32 * rng.random()) if jitter else 1.0)
                )
            )
            for c in customers:
                assignment[t][c] = best_vehicle(
                    assignment[t], c, t, -1, jitter
                )
        return assignment

    def make_seed(kind):
        selected = [[False for _ in range(N)] for _ in range(H)]
        setup = [1 for _ in range(H)]
        if kind == 0:
            for t in range(H):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (0.12, 1.0, 0.10, 1.0, 0.92, 0.0, 0.0)
        elif kind == 1:
            for t in range(H):
                for c in range(N):
                    selected[t][c] = expected_demand[t][c] > eps
            params = (0.62, 0.92, 0.28, 0.96, 1.0, 0.05, 0.15)
        elif kind == 2:
            setup = [1 if t % 2 == 0 else 0 for t in range(H)]
            for t in range(0, H, 2):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (1.02, 0.72, 0.88, 0.98, 1.05, 0.32, 0.30)
        elif kind == 3:
            gap = max(1, int(round(math.sqrt(H))))
            setup = [1 if t % gap == 0 else 0 for t in range(H)]
            for t in range(0, H, gap):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (1.12, 0.60, 1.02, 0.91, 1.10, 0.56, 0.48)
        elif kind == 4:
            setup = [1 if t % 2 == 0 else 0 for t in range(H)]
            for t in range(H):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (0.40, 0.88, 1.08, 0.86, 0.98, 0.46, 1.12)
        elif kind == 5:
            for c in range(N):
                accumulated = 0.0
                for t in range(H):
                    accumulated += expected_demand[t][c]
                    value = accumulated * max(0.0, penalty[c] - production_cost)
                    trip = transport[0][c + 1] + transport[c + 1][0]
                    if t == 0 or value >= 0.48 * trip:
                        selected[t][c] = expected_suffix[t][c] > eps
                        accumulated = 0.0
            params = (0.86, 0.79, 0.62, 1.04, 0.92, 0.28, 0.66)
        elif kind == 6:
            for c in range(N):
                total = sum(expected_demand[t][c] for t in range(H))
                average = total / H
                trip = transport[0][c + 1] + transport[c + 1][0]
                gain = average * max(penalty[c] - production_cost, eps)
                gap = max(1, min(H, int(round(math.sqrt(max(1.0, trip) / max(gain, eps))))))
                for t in range(0, H, gap):
                    selected[t][c] = expected_suffix[t][c] > eps
            setup_gap = max(1, int(round(math.sqrt(H))))
            setup = [1 if t % setup_gap == 0 else 0 for t in range(H)]
            params = (1.18, 0.52, 1.12, 0.89, 1.12, 0.72, 0.82)
        elif kind == 7:
            setup = [1 if t == 0 else 0 for t in range(H)]
            for c in range(N):
                selected[0][c] = expected_suffix[0][c] > eps
            params = (1.55, 0.45, 1.42, 0.84, 1.17, 0.88, 1.65)
        else:
            setup = [1 for _ in range(H)]
            for t in range(H):
                for c in range(N):
                    gross = expected_demand[t][c] * max(0.0, penalty[c] - production_cost)
                    trip = transport[0][c + 1] + transport[c + 1][0]
                    selected[t][c] = gross > 0.12 * trip
            params = (0.72, 0.66, 0.42, 1.08, 0.90, 0.18, 0.42)
        if production_capacity <= eps:
            setup = [0 for _ in range(H)]
        return setup, build_assignment(selected, kind >= 5), params

    def random_candidate(elapsed, stagnation):
        spread = min(1.0, math.log1p(elapsed + stagnation) / 6.0)
        setup_density = 0.05 + 0.92 * rng.random()
        visit_density = 0.03 + 0.94 * rng.random()
        setup = [
            1 if production_capacity > eps and rng.random() < setup_density else 0
            for _ in range(H)
        ]
        selected = [[False for _ in range(N)] for _ in range(H)]
        for t in range(H):
            for c in range(N):
                gross = max(0.0, penalty[c] - production_cost) * expected_demand[t][c]
                trip = transport[0][c + 1] + transport[c + 1][0]
                attractiveness = gross / max(eps, gross + trip)
                probability = visit_density * (0.22 + 1.45 * attractiveness)
                probability += 0.10 * spread * rng.random()
                selected[t][c] = rng.random() < min(0.99, probability)
        params = (
            2.3 * rng.random(),
            rng.random(),
            2.3 * rng.random(),
            0.5 + 1.3 * rng.random(),
            0.25 + 1.25 * rng.random(),
            1.7 * rng.random(),
            2.4 * rng.random(),
        )
        return setup, build_assignment(selected, True), params

    def entry_from_result(result):
        return {
            "cost": result["cost"],
            "setup": result["policy_setup"][:],
            "assignment": [row[:] for row in result["policy_assignment"]],
            "params": tuple(result["params"]),
            "lost": [row[:] for row in result["lost"]],
            "delivered": [row[:] for row in result["delivered"]],
            "production": result["production"][:],
        }

    def signature(entry):
        return (
            tuple(entry["setup"]),
            tuple(tuple(row) for row in entry["assignment"]),
            tuple(round(x, 3) for x in entry["params"]),
        )

    def clamp_params(params):
        params[0] = min(2.6, max(0.0, params[0]))
        params[1] = min(1.0, max(0.0, params[1]))
        params[2] = min(2.6, max(0.0, params[2]))
        params[3] = min(1.9, max(0.45, params[3]))
        params[4] = min(1.55, max(0.20, params[4]))
        params[5] = min(1.8, max(0.0, params[5]))
        params[6] = min(2.6, max(0.0, params[6]))

    move_weights = [1.0 for _ in range(12)]

    def choose_mode():
        total = sum(move_weights)
        value = rng.random() * total
        for i, weight in enumerate(move_weights):
            value -= weight
            if value <= 0.0:
                return i
        return 11

    def marginal_trip(row, customer):
        k = row[customer]
        if k < 0:
            return transport[0][customer + 1] + transport[customer + 1][0]
        others = [c for c in range(N) if c != customer and row[c] == k]
        if not others:
            return transport[0][customer + 1] + transport[customer + 1][0]
        return min(
            transport[customer + 1][c + 1] + transport[c + 1][customer + 1]
            for c in others
        )

    def mutate(parent, elapsed, stagnation):
        setup = parent["setup"][:]
        assignment = [row[:] for row in parent["assignment"]]
        params = list(parent["params"])
        depth = 1 + int(0.55 * math.log1p(elapsed))
        if stagnation > 0.5 and rng.random() < min(0.78, stagnation / (5.0 + stagnation)):
            depth += 1 + int(math.sqrt(math.log1p(stagnation) * 2.0))
        primary = choose_mode()

        for step in range(depth):
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
                                + 0.22 * expected_demand[tt][cc] * penalty[cc]
                            )
                            trip = transport[0][cc + 1] + transport[cc + 1][0]
                            choices.append((benefit - trip * (0.12 + 0.50 * rng.random()), tt, cc))
                if choices:
                    _, t, c = max(choices)
                    assignment[t][c] = best_vehicle(assignment[t], c, t)
                    if production_capacity > eps and rng.random() < 0.74:
                        setup[t] = 1

            elif mode == 1:
                choices = []
                for tt in range(H):
                    for cc in range(N):
                        if assignment[tt][cc] >= 0:
                            value = parent["delivered"][tt][cc] * penalty[cc]
                            choices.append((value - marginal_trip(assignment[tt], cc) * (0.48 + 0.70 * rng.random()), tt, cc))
                if choices:
                    _, t, c = min(choices)
                    assignment[t][c] = -1

            elif mode == 2:
                if rng.random() < 0.62:
                    choices = []
                    for tt in range(H):
                        lost_value = sum(parent["lost"][tt][cc] * penalty[cc] for cc in range(N))
                        if not setup[tt]:
                            choices.append((lost_value * (0.8 + 0.4 * rng.random()), tt))
                    if choices:
                        setup[max(choices)[1]] = 1
                    else:
                        setup[t] = 1 - int(bool(setup[t]))
                else:
                    setup[t] = 1 - int(bool(setup[t]))

            elif mode == 3:
                active = [(tt, cc) for tt in range(H) for cc in range(N) if assignment[tt][cc] >= 0]
                if active:
                    t, c = rng.choice(active)
                    old = assignment[t][c]
                    assignment[t][c] = -1
                    assignment[t][c] = best_vehicle(assignment[t], c, t, old)

            elif mode == 4:
                active_times = [tt for tt in range(H) if assignment[tt][c] >= 0]
                if active_times:
                    old_t = rng.choice(active_times)
                    radius = 1 + int(min(H - 1, math.sqrt(1.0 + elapsed + stagnation)) * rng.random())
                    new_t = min(H - 1, max(0, old_t + rng.randint(-radius, radius)))
                    if new_t != old_t:
                        assignment[old_t][c] = -1
                        assignment[new_t][c] = best_vehicle(assignment[new_t], c, new_t)
                        if production_capacity > eps and rng.random() < 0.65:
                            setup[new_t] = 1
                else:
                    assignment[t][c] = best_vehicle(assignment[t], c, t)

            elif mode == 5:
                p = rng.randrange(7)
                sigma = 0.035 + 0.32 / (1.0 + 0.025 * math.sqrt(elapsed))
                params[p] += rng.gauss(0.0, sigma)

            elif mode == 6:
                ranked = sorted(
                    range(N),
                    key=lambda cc: (
                        parent["lost"][t][cc] * penalty[cc]
                        + expected_demand[t][cc] * penalty[cc]
                        - 0.20 * (transport[0][cc + 1] + transport[cc + 1][0])
                    ),
                    reverse=True,
                )
                ruin = min(N, 1 + int((1.0 + math.sqrt(stagnation + elapsed * 0.02)) * rng.random()))
                if rng.random() < 0.28:
                    assignment[t] = [-1 for _ in range(N)]
                else:
                    for cc in rng.sample(range(N), min(N, ruin)):
                        assignment[t][cc] = -1
                take = min(N, 1 + rng.randrange(N))
                for cc in ranked[:take]:
                    if assignment[t][cc] < 0:
                        assignment[t][cc] = best_vehicle(assignment[t], cc, t)
                if production_capacity > eps:
                    setup[t] = 1

            elif mode == 7:
                gap = 1 + rng.randrange(H)
                phase = rng.randrange(gap)
                for tt in range(H):
                    if tt % gap == phase:
                        assignment[tt][c] = best_vehicle(assignment[tt], c, tt)
                    elif rng.random() < 0.62:
                        assignment[tt][c] = -1

            elif mode == 8:
                assignment[t][c] = -1 if assignment[t][c] >= 0 else best_vehicle(assignment[t], c, t)
                if rng.random() < 0.42:
                    setup[t] = 1 - int(bool(setup[t]))
                if H > 1 and rng.random() < 0.5:
                    t2 = min(H - 1, max(0, t + rng.choice((-1, 1))))
                    if assignment[t2][c] < 0:
                        assignment[t2][c] = best_vehicle(assignment[t2], c, t2)

            elif mode == 9:
                ranked = sorted(
                    range(N),
                    key=lambda cc: (
                        parent["lost"][t][cc] * penalty[cc]
                        + expected_demand[t][cc] * max(0.0, penalty[cc] - production_cost)
                    ),
                    reverse=True,
                )
                additions = 1 + rng.randrange(max(1, min(N, M + 4)))
                for cc in ranked[:additions]:
                    assignment[t][cc] = best_vehicle(assignment[t], cc, t)
                setup[t] = 1 if production_capacity > eps else 0

            elif mode == 10:
                width = 1 + int(min(H, 1.0 + math.sqrt(stagnation + 1.0)) * rng.random())
                start_t = rng.randrange(H)
                action = rng.random()
                for offset in range(width):
                    tt = (start_t + offset) % H
                    if action < 0.40:
                        setup[tt] = 0
                    elif action < 0.78:
                        setup[tt] = 1 if production_capacity > eps else 0
                    else:
                        setup[tt] = 1 - int(bool(setup[tt]))

            else:
                chosen = [cc for cc in range(N) if assignment[t][cc] >= 0]
                rng.shuffle(chosen)
                assignment[t] = [-1 for _ in range(N)]
                chosen.sort(
                    key=lambda cc: -(
                        expected_demand[t][cc] * (penalty[cc] + eps)
                        * (0.9 + 0.2 * rng.random())
                    )
                )
                for cc in chosen:
                    assignment[t][cc] = best_vehicle(assignment[t], cc, t)

        if rng.random() < 0.34:
            p = rng.randrange(7)
            params[p] += rng.gauss(0.0, 0.04 + 0.18 * rng.random())
        clamp_params(params)
        if production_capacity <= eps:
            setup = [0 for _ in range(H)]
        return (setup, assignment, tuple(params)), primary

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
        sigma = 0.025 + 0.14 / (1.0 + math.sqrt(elapsed + 1.0))
        params = [
            blend * a["params"][p] + (1.0 - blend) * b["params"][p]
            + rng.gauss(0.0, sigma)
            for p in range(7)
        ]
        clamp_params(params)
        if production_capacity <= eps:
            setup = [0 for _ in range(H)]
        return setup, assignment, tuple(params)

    zero_entry = entry_from_result(zero_result)
    population = [zero_entry]
    current = zero_entry
    best_entry = zero_entry
    seed_kind = 0
    last_improvement = time.perf_counter()

    while time.perf_counter() < search_deadline:
        now = time.perf_counter()
        elapsed = max(0.0, now - started)
        stagnation = max(0.0, now - last_improvement)
        parent = None
        credited_mode = None
        force_polish = False

        if seed_kind < 9:
            candidate = make_seed(seed_kind)
            seed_kind += 1
        else:
            polish_probability = min(0.16, 0.025 + stagnation / (80.0 + 8.0 * stagnation))
            restart_probability = 0.055 + min(0.42, stagnation / (15.0 + elapsed + stagnation))
            crossover_probability = min(0.34, 0.07 + math.log1p(elapsed) / 32.0)

            if rng.random() < polish_probability:
                candidate = (
                    best_entry["setup"][:],
                    [row[:] for row in best_entry["assignment"]],
                    tuple(best_entry["params"]),
                )
                force_polish = True
            elif rng.random() < restart_probability:
                candidate = random_candidate(elapsed, stagnation)
            elif len(population) >= 2 and rng.random() < crossover_probability:
                tournament = min(len(population), 2 + int(math.log1p(elapsed) / 2.6))
                a = min(
                    (rng.choice(population) for _ in range(tournament)),
                    key=lambda entry: entry["cost"],
                )
                b = rng.choice(population)
                candidate = crossover(a, b, elapsed)
            else:
                tournament = min(len(population), 2 + int(math.log1p(elapsed) / 2.2))
                choice = rng.random()
                if choice < 0.22:
                    parent = current
                elif choice < 0.52:
                    parent = best_entry
                else:
                    parent = min(
                        (rng.choice(population) for _ in range(tournament)),
                        key=lambda entry: entry["cost"],
                    )
                candidate, credited_mode = mutate(parent, elapsed, stagnation)

        if time.perf_counter() >= search_deadline:
            break
        route_effort = 0
        if force_polish:
            route_effort = 2
        else:
            improve_probability = min(0.48, 0.07 + 0.28 * (1.0 - math.exp(-elapsed / 32.0)))
            exact_probability = min(0.24, math.log1p(elapsed) / 24.0)
            if rng.random() < improve_probability:
                route_effort = 1
            if elapsed > 0.5 and rng.random() < exact_probability:
                route_effort = 2

        result = evaluate(
            candidate[0], candidate[1], candidate[2],
            search_deadline, route_effort,
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
                    0.18, move_weights[credited_mode] * 0.995
                )
            if max(move_weights) > 60.0:
                move_weights = [0.5 + 0.5 * weight for weight in move_weights]

        tolerance = 1e-9 * (1.0 + abs(best_cost))
        if cost < best_cost - tolerance:
            best_cost = cost
            best_result = result
            best_entry = entry
            last_improvement = time.perf_counter()

        temperature = 0.026 * (1.0 + abs(best_cost))
        temperature /= 1.0 + math.log1p(elapsed)
        temperature *= 1.0 + min(3.0, stagnation / 10.0)
        accept = cost <= current["cost"]
        if not accept and temperature > eps:
            scaled = (cost - current["cost"]) / temperature
            if scaled < 45.0 and rng.random() < math.exp(-scaled):
                accept = True
        if accept:
            current = entry

        target_population = 7 + int(3.0 * math.sqrt(elapsed + 1.0))
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
                population.sort(key=lambda item: item["cost"])
                lower = max(1, len(population) // 2)
                remove_index = rng.randrange(lower, len(population))
                population.pop(remove_index)

        cache_target = 260 + int(70.0 * math.sqrt(elapsed + 1.0))
        while len(route_cache) > cache_target:
            route_cache.pop(next(iter(route_cache)))

    try:
        return materialize(best_result)
    except BaseException:
        return baseline_solution


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
