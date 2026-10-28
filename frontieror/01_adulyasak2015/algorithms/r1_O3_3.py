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
    budget = _number(time_limit_s, 0.0)
    if budget < 0.0:
        budget = 0.0
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

    initial = [float(x) for x in instance["initial_inventories"]]
    capacities = [max(0.0, float(x)) for x in instance["inventory_capacities"]]
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
            len(holding) != N + 1 or len(penalty) != N):
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
    baseline_cost = 0.0

    for s in range(S):
        previous = initial[:]
        scenario_cost = 0.0
        for t in range(H):
            ending = [0.0 for _ in range(N + 1)]
            ending[0] = previous[0]
            scenario_cost += holding[0] * ending[0]
            for c in range(N):
                d = demand[s][t][c]
                served = min(max(0.0, previous[c + 1]), d)
                ending[c + 1] = max(0.0, previous[c + 1] - served)
                lost = max(0.0, d - served)
                baseline_unmet[s][t][c] = lost
                scenario_cost += holding[c + 1] * ending[c + 1]
                scenario_cost += penalty[c] * lost
            baseline_inventory[s][t] = ending
            previous = ending
        baseline_cost += probabilities[s] * scenario_cost

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
    reserve = max(0.004, 1.8 * baseline_time + 0.003)
    deadline = hard_deadline - reserve
    if time.perf_counter() >= deadline:
        return baseline_solution

    suffix = [
        [[0.0 for _ in range(N)] for _ in range(H + 1)]
        for _ in range(S)
    ]
    for s in range(S):
        for t in range(H - 1, -1, -1):
            for c in range(N):
                suffix[s][t][c] = demand[s][t][c] + suffix[s][t + 1][c]
        if time.perf_counter() >= deadline:
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
                    "demand": demand[s][t][:],
                    "probability": 0.0,
                })
            level[nid]["members"].append(s)
            level[nid]["probability"] += probabilities[s]
            node_ids[s][t] = nid
        groups.append(level)
        level_minimum = []
        for group in level:
            members = group["members"]
            level_minimum.append([
                min(suffix[s][t][c] for s in members)
                for c in range(N)
            ])
        minimum_suffix.append(level_minimum)
        if time.perf_counter() >= deadline:
            return baseline_solution

    seed_value = 91815541 + 1009 * H + 9176 * N + 65537 * M + 131071 * S
    seed_value ^= int(sum(initial) * 1009.0) & 0x7fffffff
    seed_value ^= int(sum(sum(row) for row in expected_demand) * 101.0) & 0x7fffffff
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
                    if ((checks & 127) == 0 and local_deadline is not None and
                            time.perf_counter() >= local_deadline):
                        return order
            if best_i < 0:
                break
            order[best_i:best_j + 1] = reversed(order[best_i:best_j + 1])
            if local_deadline is not None and time.perf_counter() >= local_deadline:
                break
        return order

    def relocate_improve(order, local_deadline):
        order = order[:]
        n = len(order)
        while n >= 3:
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
                for pos in range(len(reduced) + 1):
                    before = 0 if pos == 0 else reduced[pos - 1]
                    after = 0 if pos == len(reduced) else reduced[pos]
                    delta = (
                        removal + transport[before][node]
                        + transport[node][after] - transport[before][after]
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_pos = pos
                    checks += 1
                    if ((checks & 127) == 0 and local_deadline is not None and
                            time.perf_counter() >= local_deadline):
                        return order
            if best_i < 0:
                break
            node = order.pop(best_i)
            order.insert(best_pos, node)
            if local_deadline is not None and time.perf_counter() >= local_deadline:
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
            if local_deadline is not None and time.perf_counter() >= local_deadline:
                nearest.extend(remaining)
                remaining.clear()
        nearest = two_opt(nearest, local_deadline)
        best = nearest
        best_cost = route_cost(best)

        if local_deadline is None or time.perf_counter() < local_deadline:
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
                            transport[before][node] + transport[node][after]
                            - transport[before][after]
                        )
                        item = (delta, node, pos)
                        if choice is None or item < choice:
                            choice = item
                insertion.insert(choice[2], choice[1])
                remaining.remove(choice[1])
                if local_deadline is not None and time.perf_counter() >= local_deadline:
                    insertion.extend(remaining)
                    remaining.clear()
            insertion = two_opt(insertion, local_deadline)
            insertion_cost = route_cost(insertion)
            if insertion_cost + eps < best_cost:
                best = insertion
                best_cost = insertion_cost
        return best, best_cost

    def get_route(customers, local_deadline, perturb):
        key = tuple(sorted(customers))
        cached = route_cache.get(key)
        if cached is None:
            order, cost = initial_route(key, local_deadline)
            cached = (tuple(order), cost)
            route_cache[key] = cached
        elif perturb and len(key) >= 3:
            trial = list(cached[0])
            mode = rng.randrange(4)
            if mode == 0:
                i = rng.randrange(len(trial))
                j = rng.randrange(len(trial))
                if i > j:
                    i, j = j, i
                if i != j:
                    trial[i:j + 1] = reversed(trial[i:j + 1])
            elif mode == 1:
                i = rng.randrange(len(trial))
                j = rng.randrange(len(trial))
                trial[i], trial[j] = trial[j], trial[i]
            elif mode == 2:
                i = rng.randrange(len(trial))
                node = trial.pop(i)
                trial.insert(rng.randrange(len(trial) + 1), node)
            else:
                rng.shuffle(trial)
            trial = two_opt(trial, local_deadline)
            if local_deadline is None or time.perf_counter() < local_deadline:
                trial = relocate_improve(trial, local_deadline)
            trial_cost = route_cost(trial)
            if trial_cost + eps < cached[1]:
                cached = (tuple(trial), trial_cost)
                route_cache[key] = cached
        return list(cached[0]), cached[1]

    def evaluate(setup, assignment, params, local_deadline, route_effort):
        if local_deadline is not None and time.perf_counter() >= local_deadline:
            return None

        cover_factor = params[0]
        mean_mix = params[1]
        buffer_factor = params[2]
        production_threshold = params[3]
        future_scale = params[4]
        stock_guard = params[5]
        lookahead_factor = params[6]

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
                delay_cost = holding[0] * (tau - t)
                for c in range(N):
                    if (0 <= assignment[tau][c] < M and
                            expected_demand[tau][c] > eps):
                        shadow = max(shadow, penalty[c] - delay_cost)
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
                if local_deadline is not None and time.perf_counter() >= local_deadline:
                    return None
                if t == 0:
                    previous = initial[:]
                else:
                    previous = list(records[t - 1][group["parent"]][2])

                members = group["members"]
                mass = group["probability"]
                current_demand = group["demand"]
                factory_stock = max(0.0, previous[0])

                reserve_target = 0.0
                gap_end = gap_ends[t]
                if (buffer_factor > eps and gap_end > t and mass > eps and
                        setup[t]):
                    mean_need = 0.0
                    minimum_need = None
                    for s in members:
                        branch_need = 0.0
                        for tau in range(t + 1, gap_end + 1):
                            vehicle_needs = [0.0 for _ in range(M)]
                            delay_cost = holding[0] * (tau - t)
                            for c in range(N):
                                k = assignment[tau][c]
                                if (0 <= k < M and
                                        penalty[c] > production_cost * production_threshold + delay_cost):
                                    vehicle_needs[k] += demand[s][tau][c]
                            for k in range(M):
                                branch_need += min(vehicle_capacity, vehicle_needs[k])
                        mean_need += probabilities[s] * branch_need
                        if minimum_need is None or branch_need < minimum_need:
                            minimum_need = branch_need
                        tick += 1
                        if ((tick & 31) == 0 and local_deadline is not None and
                                time.perf_counter() >= local_deadline):
                            return None
                    mean_need /= mass
                    if minimum_need is None:
                        minimum_need = 0.0
                    blended_need = (
                        mean_mix * mean_need + (1.0 - mean_mix) * minimum_need
                    )
                    reserve_target = min(
                        capacities[0], max(0.0, buffer_factor * blended_need)
                    )

                segments = []
                customer_limit = [0.0 for _ in range(N)]
                extra_steps = int(round(
                    lookahead_factor * max(1.0, math.sqrt(float(H)))
                ))

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

                    shortage = max(0.0, current_demand[c] - prior_stock)
                    current_quantity = min(shortage, maximum_delivery)
                    if current_quantity > eps and penalty[c] > eps:
                        segments.append((penalty[c], c, k, current_quantity))

                    nxt = next_visit[t][c]
                    natural_horizon = H - 1 if nxt < 0 else nxt - 1
                    horizon = max(natural_horizon, min(H - 1, t + extra_steps))
                    future_quantity = 0.0
                    if horizon > t and mass > eps:
                        conditional_mean = 0.0
                        conditional_minimum = None
                        for s in members:
                            amount = (
                                suffix[s][t + 1][c]
                                - suffix[s][horizon + 1][c]
                            )
                            conditional_mean += probabilities[s] * amount
                            if conditional_minimum is None or amount < conditional_minimum:
                                conditional_minimum = amount
                        conditional_mean /= mass
                        if conditional_minimum is None:
                            conditional_minimum = 0.0
                        target_base = (
                            mean_mix * conditional_mean
                            + (1.0 - mean_mix) * conditional_minimum
                        )
                        maximum_ending = max(
                            0.0, capacities[c + 1] - current_demand[c]
                        )
                        minimum_future_total = max(
                            0.0,
                            minimum_suffix[t][nid][c] - current_demand[c],
                        )
                        target_ending = min(
                            maximum_ending,
                            minimum_future_total,
                            max(0.0, cover_factor * target_base),
                        )
                        natural_remainder = max(
                            0.0, prior_stock - current_demand[c]
                        )
                        future_quantity = max(
                            0.0, target_ending - natural_remainder
                        )
                        future_quantity = min(
                            future_quantity,
                            max(0.0, maximum_delivery - current_quantity),
                        )
                        average_delay = max(1.0, 0.5 * (horizon - t + 1))
                        future_value = (
                            penalty[c] * future_scale
                            - holding[c + 1] * average_delay
                        )
                        if future_quantity > eps and future_value > eps:
                            segments.append((future_value, c, k, future_quantity))
                    customer_limit[c] = maximum_delivery

                    tick += 1
                    if ((tick & 63) == 0 and local_deadline is not None and
                            time.perf_counter() >= local_deadline):
                        return None

                segments.sort(key=lambda item: (item[0], -item[1]), reverse=True)
                remaining_vehicle = [vehicle_capacity for _ in range(M)]
                remaining_customer = customer_limit[:]
                delivered = [0.0 for _ in range(N)]

                protected_stock = min(factory_stock, reserve_target)
                free_stock_left = max(0.0, factory_stock - protected_stock)
                protected_stock_left = protected_stock
                production_left = production_capacity if setup[t] else 0.0
                stock_used = 0.0
                production_used = 0.0
                guard_value = stock_guard * future_shadows[t]

                for value, c, k, amount in segments:
                    possible = min(
                        amount,
                        remaining_vehicle[k],
                        remaining_customer[c],
                    )
                    if possible <= eps:
                        continue
                    taken = 0.0

                    use = min(possible, free_stock_left)
                    if use > eps:
                        free_stock_left -= use
                        stock_used += use
                        taken += use
                        possible -= use

                    if (possible > eps and protected_stock_left > eps and
                            value + eps >= guard_value):
                        use = min(possible, protected_stock_left)
                        protected_stock_left -= use
                        stock_used += use
                        taken += use
                        possible -= use

                    if (possible > eps and production_left > eps and
                            value > production_cost * production_threshold + eps):
                        use = min(possible, production_left)
                        production_left -= use
                        production_used += use
                        taken += use
                        possible -= use

                    if taken > eps:
                        delivered[c] += taken
                        remaining_customer[c] -= taken
                        remaining_vehicle[k] -= taken

                factory_after_shipments = max(0.0, factory_stock - stock_used)
                extra_production = 0.0
                if setup[t] and production_left > eps:
                    extra_production = min(
                        max(0.0, reserve_target - factory_after_shipments),
                        production_left,
                        max(0.0, capacities[0] - factory_after_shipments),
                    )
                    if extra_production <= eps:
                        extra_production = 0.0

                production_quantity = production_used + extra_production
                if production_quantity <= eps:
                    production_quantity = 0.0
                else:
                    used_setup[t] = 1
                factory_end = factory_after_shipments + extra_production
                if factory_end < eps:
                    factory_end = 0.0
                if factory_end > capacities[0] and factory_end - capacities[0] < 1e-7:
                    factory_end = capacities[0]

                sparse_delivery = []
                end_inventory = [factory_end]
                lost_sales = []
                period_cost = production_cost * production_quantity
                period_cost += holding[0] * factory_end

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
                    period_cost += holding[c + 1] * ending
                    period_cost += penalty[c] * lost
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
                    order, cost = get_route(
                        customers, local_deadline, route_effort
                    )
                    if (local_deadline is not None and
                            time.perf_counter() >= local_deadline):
                        return None
                    period_routes.append([0] + order + [0])
                    transportation_total += cost
            routes.append(period_routes)

        total_cost = (
            expected_operating_cost + transportation_total
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
    zero_params = (0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    zero_result = evaluate(
        zero_setup, zero_assignment, zero_params, deadline, False
    )
    if zero_result is None:
        return baseline_solution

    best_result = zero_result
    best_cost = zero_result["cost"]

    average_depot_cost = sum(
        transport[0][c + 1] + transport[c + 1][0]
        for c in range(N)
    ) / max(1, N)
    route_scale = max(1.0, average_depot_cost)

    def best_vehicle(row, customer, t, forbidden=-1):
        best_k = 0
        best_value = float("inf")
        for k in range(M):
            if k == forbidden and M > 1:
                continue
            members = [c for c in range(N) if row[c] == k and c != customer]
            load = sum(expected_demand[t][c] for c in members)
            if members:
                proximity = min(
                    transport[customer + 1][c + 1]
                    for c in members
                )
                travel_term = 2.0 * proximity
            else:
                travel_term = (
                    transport[0][customer + 1]
                    + transport[customer + 1][0]
                )
            pressure = load / max(vehicle_capacity, eps)
            value = pressure * pressure + 0.24 * travel_term / route_scale
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
            transport[customer + 1][c + 1]
            + transport[c + 1][customer + 1]
            for c in others
        )

    def balanced_assignment(selected, jitter):
        assignment = [[-1 for _ in range(N)] for _ in range(H)]
        for t in range(H):
            customers = [c for c in range(N) if selected[t][c]]
            customers.sort(
                key=lambda c: -(
                    expected_demand[t][c] * (penalty[c] + eps)
                    * (0.85 + 0.30 * rng.random() if jitter else 1.0)
                )
            )
            for c in customers:
                assignment[t][c] = best_vehicle(assignment[t], c, t)
        return assignment

    def make_seed(kind):
        selected = [[False for _ in range(N)] for _ in range(H)]
        setup = [1 for _ in range(H)]
        if kind == 0:
            for t in range(H):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (0.15, 1.0, 0.15, 1.0, 0.90, 0.0, 0.0)
        elif kind == 1:
            for t in range(H):
                for c in range(N):
                    gross = penalty[c] * expected_demand[t][c]
                    trip = transport[0][c + 1] + transport[c + 1][0]
                    selected[t][c] = gross > 0.20 * trip or gross > eps
            params = (0.65, 1.0, 0.25, 0.95, 1.0, 0.0, 0.0)
        elif kind == 2:
            setup = [1 if t % 2 == 0 else 0 for t in range(H)]
            for t in range(0, H, 2):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (1.0, 0.72, 0.85, 0.98, 1.05, 0.35, 0.25)
        elif kind == 3:
            gap = max(1, int(round(math.sqrt(float(H)))))
            setup = [1 if t % gap == 0 else 0 for t in range(H)]
            for t in range(0, H, gap):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (1.08, 0.62, 0.95, 0.90, 1.10, 0.55, 0.40)
        elif kind == 4:
            setup = [1 if t % 2 == 0 else 0 for t in range(H)]
            for t in range(H):
                for c in range(N):
                    selected[t][c] = expected_suffix[t][c] > eps
            params = (0.35, 0.85, 1.05, 0.85, 0.98, 0.45, 1.15)
        elif kind == 5:
            for c in range(N):
                accumulated = 0.0
                for t in range(H):
                    accumulated += expected_demand[t][c]
                    benefit = max(0.0, penalty[c] - production_cost) * accumulated
                    trip = transport[0][c + 1] + transport[c + 1][0]
                    if t == 0 or benefit >= 0.55 * trip:
                        selected[t][c] = expected_suffix[t][c] > eps
                        accumulated = 0.0
            params = (0.82, 0.78, 0.55, 1.08, 0.88, 0.30, 0.65)
        else:
            for c in range(N):
                total = sum(expected_demand[t][c] for t in range(H))
                average = total / max(1, H)
                trip = transport[0][c + 1] + transport[c + 1][0]
                value = average * max(penalty[c] - production_cost, eps)
                gap = max(1, int(round(math.sqrt(
                    max(1.0, trip) / max(value, eps)
                ))))
                gap = min(H, gap)
                phase = 0
                for t in range(phase, H, gap):
                    selected[t][c] = expected_suffix[t][c] > eps
            setup_gap = max(1, int(round(math.sqrt(float(H)))))
            setup = [1 if t % setup_gap == 0 else 0 for t in range(H)]
            params = (1.15, 0.50, 1.10, 0.88, 1.12, 0.75, 0.80)
        if production_capacity <= eps:
            setup = [0 for _ in range(H)]
        return setup, balanced_assignment(selected, kind in (5, 6)), params

    def random_candidate(elapsed):
        spread = min(1.0, math.log1p(max(0.0, elapsed)) / 6.0)
        setup_density = 0.08 + 0.88 * rng.random()
        visit_density = 0.05 + 0.93 * rng.random()
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
                attractiveness = gross / max(eps, gross + trip)
                probability = visit_density * (0.28 + 1.35 * attractiveness)
                probability += 0.08 * spread * rng.random()
                selected[t][c] = rng.random() < min(0.985, probability)
        params = (
            1.8 * rng.random(),
            rng.random(),
            1.8 * rng.random(),
            0.55 + 1.15 * rng.random(),
            0.35 + 1.05 * rng.random(),
            1.4 * rng.random(),
            1.8 * rng.random(),
        )
        return setup, balanced_assignment(selected, True), params

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
            tuple(round(x, 3) for x in entry["params"]),
        )

    move_weights = [1.0 for _ in range(10)]

    def choose_mode():
        total = sum(move_weights)
        value = rng.random() * total
        for i, weight in enumerate(move_weights):
            value -= weight
            if value <= 0.0:
                return i
        return len(move_weights) - 1

    def clamp_params(params):
        params[0] = min(2.2, max(0.0, params[0]))
        params[1] = min(1.0, max(0.0, params[1]))
        params[2] = min(2.2, max(0.0, params[2]))
        params[3] = min(1.8, max(0.5, params[3]))
        params[4] = min(1.45, max(0.25, params[4]))
        params[5] = min(1.6, max(0.0, params[5]))
        params[6] = min(2.2, max(0.0, params[6]))

    def mutate(parent, elapsed, stagnation):
        setup = parent["setup"][:]
        assignment = [row[:] for row in parent["assignment"]]
        params = list(parent["params"])
        base_depth = 1 + int(0.52 * math.log1p(max(0.0, elapsed)))
        if stagnation > 1.0 and rng.random() < min(0.65, stagnation / (8.0 + stagnation)):
            base_depth += 1 + int(math.sqrt(math.log1p(stagnation)))
        depth = max(1, base_depth)
        primary_mode = choose_mode()

        for step in range(depth):
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
                                + 0.18 * expected_demand[tt][cc] * penalty[cc]
                            )
                            trip = (
                                transport[0][cc + 1]
                                + transport[cc + 1][0]
                            )
                            key = benefit - trip * (0.18 + 0.45 * rng.random())
                            candidates.append((key, tt, cc))
                if candidates:
                    _, t, c = max(candidates)
                    assignment[t][c] = best_vehicle(assignment[t], c, t)
                    if rng.random() < 0.72 and production_capacity > eps:
                        setup[t] = 1

            elif mode == 1:
                candidates = []
                for tt in range(H):
                    for cc in range(N):
                        if assignment[tt][cc] >= 0:
                            benefit = parent["delivered"][tt][cc] * penalty[cc]
                            trip = marginal_trip(assignment[tt], cc)
                            key = benefit - trip * (0.55 + 0.55 * rng.random())
                            candidates.append((key, tt, cc))
                if candidates:
                    _, t, c = min(candidates)
                    assignment[t][c] = -1

            elif mode == 2:
                if rng.random() < 0.58:
                    choices = []
                    for tt in range(H):
                        lost_value = sum(
                            parent["lost"][tt][cc] * penalty[cc]
                            for cc in range(N)
                        )
                        if not setup[tt]:
                            choices.append((lost_value * (0.8 + 0.4 * rng.random()), tt))
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
                if active:
                    t, c = rng.choice(active)
                    old = assignment[t][c]
                    if M > 1:
                        assignment[t][c] = -1
                        assignment[t][c] = best_vehicle(
                            assignment[t], c, t, old
                        )

            elif mode == 4:
                active_times = [tt for tt in range(H) if assignment[tt][c] >= 0]
                if active_times:
                    old_t = rng.choice(active_times)
                    radius = 1 + int(
                        min(H - 1, math.sqrt(max(1.0, elapsed + stagnation)))
                        * rng.random()
                    )
                    shift = rng.randint(-radius, radius)
                    new_t = min(H - 1, max(0, old_t + shift))
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
                p = rng.randrange(7)
                sigma = 0.04 + 0.30 / (
                    1.0 + 0.035 * math.sqrt(max(0.0, elapsed))
                )
                params[p] += rng.gauss(0.0, sigma)

            elif mode == 6:
                t = rng.randrange(H)
                if rng.random() < 0.28:
                    assignment[t] = [-1 for _ in range(N)]
                else:
                    ranked = sorted(
                        range(N),
                        key=lambda cc: (
                            parent["lost"][t][cc] * penalty[cc]
                            + expected_demand[t][cc] * penalty[cc]
                            - 0.25 * (
                                transport[0][cc + 1]
                                + transport[cc + 1][0]
                            )
                        ),
                        reverse=True,
                    )
                    take = 1 + rng.randrange(N)
                    assignment[t] = [-1 for _ in range(N)]
                    for cc in ranked[:take]:
                        assignment[t][cc] = best_vehicle(
                            assignment[t], cc, t
                        )
                    if production_capacity > eps:
                        setup[t] = 1

            elif mode == 7:
                gap = 1 + rng.randrange(H)
                phase = rng.randrange(gap)
                add_schedule = rng.random() < 0.65
                for tt in range(H):
                    if tt % gap == phase and add_schedule:
                        assignment[tt][c] = best_vehicle(
                            assignment[tt], c, tt
                        )
                    elif tt % gap != phase and rng.random() < 0.55:
                        assignment[tt][c] = -1

            elif mode == 8:
                if assignment[t][c] < 0:
                    assignment[t][c] = best_vehicle(assignment[t], c, t)
                else:
                    assignment[t][c] = -1
                if rng.random() < 0.35:
                    setup[t] = 1 - int(bool(setup[t]))

            else:
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
                    assignment[t][cc] = best_vehicle(
                        assignment[t], cc, t
                    )

        if rng.random() < 0.32:
            p = rng.randrange(7)
            params[p] += rng.gauss(0.0, 0.05 + 0.16 * rng.random())
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
                setup[t] = (
                    a["setup"][t] if rng.random() < 0.5 else b["setup"][t]
                )
            for c in range(N):
                source = a if rng.random() < 0.5 else b
                for t in range(H):
                    assignment[t][c] = source["assignment"][t][c]
        blend = rng.random()
        sigma = 0.03 + 0.12 / (1.0 + math.sqrt(max(0.0, elapsed)))
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
    seed_kind = 0
    last_improvement = time.perf_counter()

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        elapsed = max(0.0, now - started)
        stagnation = max(0.0, now - last_improvement)
        parent = None
        credited_mode = None

        if seed_kind < 7:
            candidate = make_seed(seed_kind)
            seed_kind += 1
        else:
            restart_probability = 0.07 + min(
                0.38, stagnation / (18.0 + elapsed + stagnation)
            )
            crossover_probability = min(
                0.30, 0.08 + math.log1p(elapsed) / 35.0
            )
            if rng.random() < restart_probability:
                candidate = random_candidate(elapsed)
            elif len(population) >= 2 and rng.random() < crossover_probability:
                tournament = min(len(population), 2 + int(math.log1p(elapsed) / 3.0))
                a = min(
                    (rng.choice(population) for _ in range(tournament)),
                    key=lambda e: e["cost"],
                )
                b = rng.choice(population)
                candidate = crossover(a, b, elapsed)
            else:
                tournament = min(len(population), 2 + int(math.log1p(elapsed) / 2.5))
                if rng.random() < 0.28:
                    parent = current
                else:
                    parent = min(
                        (rng.choice(population) for _ in range(tournament)),
                        key=lambda e: e["cost"],
                    )
                candidate, credited_mode = mutate(
                    parent, elapsed, stagnation
                )

        if time.perf_counter() >= deadline:
            break
        route_probability = min(
            0.38,
            0.08 + 0.24 * (1.0 - math.exp(-elapsed / 35.0)),
        )
        result = evaluate(
            candidate[0], candidate[1], candidate[2], deadline,
            rng.random() < route_probability,
        )
        if result is None:
            break
        entry = entry_from_result(result)
        cost = entry["cost"]

        if credited_mode is not None and parent is not None:
            if cost + 1e-9 < parent["cost"]:
                move_weights[credited_mode] *= 1.075
            else:
                move_weights[credited_mode] = max(
                    0.22, move_weights[credited_mode] * 0.996
                )
            if max(move_weights) > 40.0:
                move_weights = [0.5 + w * 0.5 for w in move_weights]

        tolerance = 1e-9 * (1.0 + abs(best_cost))
        if cost < best_cost - tolerance:
            best_cost = cost
            best_result = result
            last_improvement = time.perf_counter()

        temperature = 0.028 * (1.0 + abs(best_cost))
        temperature /= 1.0 + math.log1p(elapsed)
        temperature *= 1.0 + min(2.5, stagnation / 12.0)
        accept = cost <= current["cost"]
        if not accept and temperature > eps:
            scaled = (cost - current["cost"]) / temperature
            if scaled < 45.0 and rng.random() < math.exp(-scaled):
                accept = True
        if accept:
            current = entry

        target_population = 6 + int(2.4 * math.sqrt(elapsed + 1.0))
        new_signature = signature(entry)
        duplicate_index = -1
        for i, old in enumerate(population):
            if signature(old) == new_signature:
                duplicate_index = i
                break
        if duplicate_index >= 0:
            if cost < population[duplicate_index]["cost"]:
                population[duplicate_index] = entry
        else:
            population.append(entry)
            if len(population) > target_population:
                worst = max(
                    range(len(population)),
                    key=lambda i: population[i]["cost"],
                )
                population.pop(worst)

        cache_target = 240 + int(55.0 * math.sqrt(elapsed + 1.0))
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
