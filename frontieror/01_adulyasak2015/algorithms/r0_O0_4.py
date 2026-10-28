import math
import random
import time


def _fallback(instance):
    try:
        H = max(0, int(instance.get("num_periods", 0)))
        N = max(0, int(instance.get("num_customers", 0)))
        M = max(0, int(instance.get("num_vehicles", 0)))
        S = max(0, int(instance.get("num_scenarios", 0)))
        raw_demand = instance.get("demand_scenarios", [])
        raw_initial = instance.get("initial_inventories", [])

        initial = []
        for i in range(N + 1):
            try:
                initial.append(max(0.0, float(raw_initial[i])))
            except BaseException:
                initial.append(0.0)

        production = [[0.0 for _ in range(H)] for _ in range(S)]
        deliveries = [
            [
                [[0.0 for _ in range(N)] for _ in range(M)]
                for _ in range(H)
            ]
            for _ in range(S)
        ]
        inventories = []
        unmet = []

        for s in range(S):
            customer_stock = initial[1:]
            scenario_inv = []
            scenario_unmet = []
            for t in range(H):
                period_unmet = []
                next_stock = []
                for c in range(N):
                    try:
                        d = max(0.0, float(raw_demand[s][t][c]))
                    except BaseException:
                        d = 0.0
                    served = min(customer_stock[c], d)
                    next_stock.append(max(0.0, customer_stock[c] - served))
                    period_unmet.append(max(0.0, d - served))
                customer_stock = next_stock
                scenario_inv.append([initial[0]] + next_stock[:])
                scenario_unmet.append(period_unmet)
            inventories.append(scenario_inv)
            unmet.append(scenario_unmet)

        return {
            "production_setup": [0 for _ in range(H)],
            "vehicle_routes": [
                [[0] for _ in range(M)] for _ in range(H)
            ],
            "production_quantities": production,
            "delivery_quantities": deliveries,
            "end_inventories": inventories,
            "unmet_demand_quantities": unmet,
        }
    except BaseException:
        return {
            "production_setup": [],
            "vehicle_routes": [],
            "production_quantities": [],
            "delivery_quantities": [],
            "end_inventories": [],
            "unmet_demand_quantities": [],
        }


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    try:
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget < 0.0:
            budget = 0.0

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
        probabilities = [float(x) for x in instance["scenario_probabilities"]]
        probability_sum = sum(probabilities)
        if probability_sum <= 0.0:
            return _fallback(instance)
        probabilities = [max(0.0, p) / probability_sum for p in probabilities]

        initial = [float(x) for x in instance["initial_inventories"]]
        capacities = [float(x) for x in instance["inventory_capacities"]]
        production_capacity = max(0.0, float(instance["production_capacity"]))
        vehicle_capacity = max(0.0, float(instance["vehicle_capacity"]))
        setup_cost = max(0.0, float(instance["setup_cost"]))
        production_cost = max(0.0, float(instance["unit_production_cost"]))
        holding = [max(0.0, float(x)) for x in instance["holding_costs"]]
        penalty = [max(0.0, float(x)) for x in instance["unmet_demand_penalties"]]
        transport = [
            [max(0.0, float(instance["transportation_costs"][i][j]))
             for j in range(N + 1)]
            for i in range(N + 1)
        ]

        if (len(initial) != N + 1 or len(capacities) != N + 1 or
                len(holding) != N + 1 or len(penalty) != N):
            return _fallback(instance)

        eps = 1e-10
        suffix = [
            [[0.0 for _ in range(N)] for _ in range(H + 1)]
            for _ in range(S)
        ]
        for s in range(S):
            for t in range(H - 1, -1, -1):
                for c in range(N):
                    suffix[s][t][c] = demand[s][t][c] + suffix[s][t + 1][c]

        node_ids = [[-1 for _ in range(H)] for _ in range(S)]
        groups = []
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

        minimum_suffix = []
        for t in range(H):
            level_minimum = []
            for group in groups[t]:
                members = group["members"]
                level_minimum.append([
                    min(suffix[s][t][c] for s in members)
                    for c in range(N)
                ])
            minimum_suffix.append(level_minimum)

        expected_demand = [
            [
                sum(probabilities[s] * demand[s][t][c] for s in range(S))
                for c in range(N)
            ]
            for t in range(H)
        ]

        route_cache = {}
        rng = random.Random(
            91815541 + 1009 * H + 9176 * N + 65537 * M + 131071 * S
        )

        def route_cost(order):
            if not order:
                return 0.0
            value = transport[0][order[0]] + transport[order[-1]][0]
            for i in range(len(order) - 1):
                value += transport[order[i]][order[i + 1]]
            return value

        def two_opt(order, deadline):
            order = order[:]
            n = len(order)
            while n >= 3:
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                best_delta = -eps
                best_i = -1
                best_j = -1
                for i in range(n - 1):
                    if deadline is not None and time.perf_counter() >= deadline:
                        return order
                    a = 0 if i == 0 else order[i - 1]
                    b = order[i]
                    for j in range(i + 1, n):
                        c = order[j]
                        d = 0 if j == n - 1 else order[j + 1]
                        delta = (transport[a][c] + transport[b][d] -
                                 transport[a][b] - transport[c][d])
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_j = j
                if best_i < 0:
                    break
                order[best_i:best_j + 1] = reversed(order[best_i:best_j + 1])
            return order

        def get_route(customers, deadline, perturb):
            key = tuple(sorted(customers))
            cached = route_cache.get(key)
            if cached is None:
                remaining = list(key)
                order = []
                current = 0
                while remaining:
                    nxt = min(remaining, key=lambda v: (transport[current][v], v))
                    order.append(nxt)
                    remaining.remove(nxt)
                    current = nxt
                    if deadline is not None and time.perf_counter() >= deadline:
                        order.extend(sorted(remaining))
                        remaining.clear()
                order = two_opt(order, deadline)
                cached = (tuple(order), route_cost(order))
                route_cache[key] = cached
            elif perturb and len(key) >= 4:
                trial = list(cached[0])
                i = rng.randrange(len(trial))
                j = rng.randrange(len(trial))
                if i > j:
                    i, j = j, i
                if i != j:
                    trial[i:j + 1] = reversed(trial[i:j + 1])
                    trial = two_opt(trial, deadline)
                    trial_cost = route_cost(trial)
                    if trial_cost + eps < cached[1]:
                        cached = (tuple(trial), trial_cost)
                        route_cache[key] = cached
            return list(cached[0]), cached[1]

        def evaluate(setup, assignment, params, deadline, perturb_routes):
            stock_factor, buffer_factor, production_threshold, future_scale = params

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
            used_setup = [0 for _ in range(H)]
            active_assignment = [[-1 for _ in range(N)] for _ in range(H)]
            expected_operating_cost = 0.0
            clock_tick = 0

            for t in range(H):
                level_records = [None for _ in groups[t]]
                for nid, group in enumerate(groups[t]):
                    if deadline is not None and time.perf_counter() >= deadline:
                        return None

                    if t == 0:
                        previous = initial[:]
                    else:
                        previous = list(records[t - 1][group["parent"]][2])

                    factory_stock = max(0.0, previous[0])
                    current_demand = group["demand"]
                    group_probability = group["probability"]
                    segments = []
                    customer_limit = [0.0 for _ in range(N)]

                    for c in range(N):
                        k = assignment[t][c]
                        if k < 0 or k >= M:
                            continue
                        prior_stock = max(0.0, previous[c + 1])
                        space = max(0.0, capacities[c + 1] - prior_stock)
                        upper = max(0.0, minimum_suffix[t][nid][c])
                        max_delivery = min(space, upper)
                        if max_delivery <= eps:
                            continue

                        shortage = max(0.0, current_demand[c] - prior_stock)
                        current_quantity = min(shortage, max_delivery)
                        if current_quantity > eps:
                            segments.append((penalty[c], c, k, current_quantity))

                        future_quantity = 0.0
                        nxt = next_visit[t][c]
                        horizon = H - 1 if nxt < 0 else nxt - 1
                        if horizon > t and group_probability > eps:
                            mean_future = 0.0
                            for s in group["members"]:
                                amount = (suffix[s][t + 1][c] -
                                          suffix[s][horizon + 1][c])
                                mean_future += probabilities[s] * amount
                            mean_future /= group_probability
                            target_after_demand = stock_factor * mean_future
                            natural_remainder = max(
                                0.0, prior_stock - current_demand[c]
                            )
                            future_quantity = max(
                                0.0, target_after_demand - natural_remainder
                            )
                            future_quantity = min(
                                future_quantity,
                                max_delivery - current_quantity,
                            )
                            average_delay = 0.5 * (horizon - t + 1)
                            future_score = (
                                penalty[c] * future_scale -
                                holding[c + 1] * average_delay
                            )
                            if future_quantity > eps and future_score > eps:
                                segments.append(
                                    (future_score, c, k, future_quantity)
                                )
                        customer_limit[c] = max_delivery

                        clock_tick += 1
                        if (clock_tick & 31) == 0 and deadline is not None:
                            if time.perf_counter() >= deadline:
                                return None

                    segments.sort(key=lambda x: (x[0], -x[1]), reverse=True)
                    remaining_vehicle = [vehicle_capacity for _ in range(M)]
                    remaining_customer = customer_limit[:]
                    delivered = [0.0 for _ in range(N)]
                    total_shipment = 0.0

                    for score, c, k, amount in segments:
                        if score <= eps:
                            continue
                        possible = min(
                            amount,
                            remaining_vehicle[k],
                            remaining_customer[c],
                        )
                        if possible <= eps:
                            continue

                        free_part = min(
                            possible,
                            max(0.0, factory_stock - total_shipment),
                        )
                        take = free_part
                        remainder = possible - free_part
                        if (remainder > eps and setup[t] and
                                score > production_cost * production_threshold + eps):
                            production_already_needed = max(
                                0.0, total_shipment + take - factory_stock
                            )
                            producible = max(
                                0.0,
                                production_capacity - production_already_needed,
                            )
                            take += min(remainder, producible)

                        if take > eps:
                            delivered[c] += take
                            remaining_customer[c] -= take
                            remaining_vehicle[k] -= take
                            total_shipment += take

                    production_needed = max(0.0, total_shipment - factory_stock)
                    base_factory_end = max(
                        0.0,
                        factory_stock + production_needed - total_shipment,
                    )
                    extra_production = 0.0

                    if setup[t] and production_capacity - production_needed > eps:
                        nxt_setup = next_setup[t]
                        gap_end = H - 1 if nxt_setup < 0 else nxt_setup - 1
                        expected_gap_need = 0.0
                        if gap_end > t and group_probability > eps:
                            for s in group["members"]:
                                branch_need = 0.0
                                for tau in range(t + 1, gap_end + 1):
                                    period_need = 0.0
                                    delay_cost = holding[0] * (tau - t)
                                    for c in range(N):
                                        if (0 <= assignment[tau][c] < M and
                                                penalty[c] >
                                                production_cost * production_threshold +
                                                delay_cost):
                                            period_need += demand[s][tau][c]
                                    branch_need += min(
                                        period_need,
                                        vehicle_capacity * M,
                                    )
                                expected_gap_need += probabilities[s] * branch_need
                            expected_gap_need /= group_probability

                        target_factory_stock = buffer_factor * expected_gap_need
                        extra_production = min(
                            max(0.0, target_factory_stock - base_factory_end),
                            max(0.0, production_capacity - production_needed),
                            max(0.0, capacities[0] - base_factory_end),
                        )

                    production = production_needed + extra_production
                    if production > eps:
                        used_setup[t] = 1
                    factory_end = max(
                        0.0,
                        factory_stock + production - total_shipment,
                    )
                    if factory_end > capacities[0] and factory_end - capacities[0] < 1e-8:
                        factory_end = capacities[0]

                    sparse_delivery = []
                    end_inventory = [factory_end]
                    lost_sales = []
                    for c in range(N):
                        quantity = delivered[c]
                        if quantity > eps:
                            k = assignment[t][c]
                            sparse_delivery.append((k, c, quantity))
                            active_assignment[t][c] = k
                        available = max(0.0, previous[c + 1]) + quantity
                        d = current_demand[c]
                        if available >= d:
                            end_stock = available - d
                            lost = 0.0
                        else:
                            end_stock = 0.0
                            lost = d - available
                        if end_stock < eps:
                            end_stock = 0.0
                        if lost < eps:
                            lost = 0.0
                        end_inventory.append(end_stock)
                        lost_sales.append(lost)

                    period_cost = production_cost * production
                    for v in range(N + 1):
                        period_cost += holding[v] * end_inventory[v]
                    for c in range(N):
                        period_cost += penalty[c] * lost_sales[c]
                    expected_operating_cost += group_probability * period_cost

                    level_records[nid] = (
                        production,
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
                            customers, deadline, perturb_routes
                        )
                        if deadline is not None and time.perf_counter() >= deadline:
                            return None
                        period_routes.append([0] + order + [0])
                        transportation_total += cost
                routes.append(period_routes)

            total_cost = (
                expected_operating_cost + transportation_total +
                setup_cost * sum(used_setup)
            )
            return (
                total_cost,
                used_setup,
                active_assignment,
                params,
                routes,
                records,
            )

        def materialize(setup, routes, records):
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
                    record = records[t][node_ids[s][t]]
                    scenario_production.append(float(record[0]))
                    dense_delivery = [
                        [0.0 for _ in range(N)] for _ in range(M)
                    ]
                    for k, c, quantity in record[1]:
                        dense_delivery[k][c] = float(quantity)
                    scenario_delivery.append(dense_delivery)
                    scenario_inventory.append([float(x) for x in record[2]])
                    scenario_unmet.append([float(x) for x in record[3]])
                production_output.append(scenario_production)
                delivery_output.append(scenario_delivery)
                inventory_output.append(scenario_inventory)
                unmet_output.append(scenario_unmet)

            return {
                "production_setup": [int(x) for x in setup],
                "vehicle_routes": [
                    [route[:] for route in period] for period in routes
                ],
                "production_quantities": production_output,
                "delivery_quantities": delivery_output,
                "end_inventories": inventory_output,
                "unmet_demand_quantities": unmet_output,
            }

        zero_setup = [0 for _ in range(H)]
        zero_assignment = [[-1 for _ in range(N)] for _ in range(H)]
        baseline_result = evaluate(
            zero_setup,
            zero_assignment,
            (0.0, 0.0, 1.0, 1.0),
            None,
            False,
        )
        if baseline_result is None:
            return _fallback(instance)

        baseline_materialize_started = time.perf_counter()
        baseline_solution = materialize(
            baseline_result[1], baseline_result[4], baseline_result[5]
        )
        materialize_time = time.perf_counter() - baseline_materialize_started

        best_cost = baseline_result[0]
        best_setup = baseline_result[1]
        best_assignment = baseline_result[2]
        best_params = baseline_result[3]
        best_routes = baseline_result[4]
        best_records = baseline_result[5]

        hard_deadline = started + budget
        reserve = max(0.003, 1.6 * materialize_time + 0.002)
        deadline = hard_deadline - reserve
        if time.perf_counter() >= deadline:
            return baseline_solution

        average_depot_cost = (
            sum(transport[0][c + 1] for c in range(N)) / max(1, N)
        )

        def balanced_assignment(selected, jitter):
            result = [[-1 for _ in range(N)] for _ in range(H)]
            for t in range(H):
                customers = [c for c in range(N) if selected[t][c]]
                customers.sort(
                    key=lambda c: expected_demand[t][c] * penalty[c],
                    reverse=True,
                )
                if jitter:
                    rng.shuffle(customers)
                loads = [0.0 for _ in range(M)]
                members = [[] for _ in range(M)]
                for c in customers:
                    best_k = 0
                    best_value = float("inf")
                    for k in range(M):
                        if members[k]:
                            proximity = min(
                                transport[c + 1][other + 1]
                                for other in members[k]
                            )
                        else:
                            proximity = 2.0 * transport[0][c + 1]
                        value = (
                            loads[k] / max(vehicle_capacity, eps) +
                            0.22 * proximity / max(1.0, average_depot_cost)
                        )
                        if jitter:
                            value *= 0.92 + 0.16 * rng.random()
                        if value < best_value:
                            best_value = value
                            best_k = k
                    result[t][c] = best_k
                    members[best_k].append(c)
                    loads[best_k] += min(
                        vehicle_capacity,
                        expected_demand[t][c],
                    )
            return result

        def make_seed(kind):
            selected = [[False for _ in range(N)] for _ in range(H)]
            setup = [1 for _ in range(H)]
            if kind == 0:
                for t in range(H):
                    for c in range(N):
                        selected[t][c] = True
                params = (0.15, 0.15, 1.0, 0.9)
            elif kind == 1:
                for t in range(H):
                    for c in range(N):
                        gross = penalty[c] * expected_demand[t][c]
                        trip = 2.0 * transport[0][c + 1]
                        selected[t][c] = gross > 0.20 * trip or gross > eps
                params = (0.65, 0.25, 0.95, 1.0)
            elif kind == 2:
                setup = [1 if t % 2 == 0 else 0 for t in range(H)]
                for t in range(0, H, 2):
                    for c in range(N):
                        selected[t][c] = True
                params = (0.95, 0.85, 1.0, 1.05)
            elif kind == 3:
                gap = max(1, int(math.sqrt(H)))
                setup = [1 if t % gap == 0 else 0 for t in range(H)]
                for t in range(0, H, gap):
                    for c in range(N):
                        selected[t][c] = True
                params = (1.05, 0.95, 0.9, 1.1)
            elif kind == 4:
                setup = [1 if t % 2 == 0 else 0 for t in range(H)]
                for t in range(H):
                    for c in range(N):
                        selected[t][c] = True
                params = (0.25, 1.05, 0.85, 0.95)
            else:
                for c in range(N):
                    accumulated = 0.0
                    for t in range(H):
                        accumulated += expected_demand[t][c]
                        benefit = max(0.0, penalty[c] - production_cost) * accumulated
                        threshold = 0.6 * 2.0 * transport[0][c + 1]
                        if benefit >= threshold or t == 0:
                            selected[t][c] = benefit > eps
                            accumulated = 0.0
                params = (0.8, 0.55, 1.1, 0.85)
            return setup, balanced_assignment(selected, kind == 5), params

        def random_candidate():
            setup_density = 0.12 + 0.83 * rng.random()
            visit_density = 0.08 + 0.90 * rng.random()
            setup = [1 if rng.random() < setup_density else 0 for _ in range(H)]
            selected = [[False for _ in range(N)] for _ in range(H)]
            for t in range(H):
                for c in range(N):
                    gross = max(0.0, penalty[c] - production_cost) * expected_demand[t][c]
                    trip = 2.0 * transport[0][c + 1]
                    attractiveness = gross / max(eps, gross + trip)
                    probability = visit_density * (0.35 + 1.15 * attractiveness)
                    selected[t][c] = rng.random() < min(0.98, probability)
            params = (
                2.0 * rng.random(),
                2.0 * rng.random(),
                0.55 + 1.20 * rng.random(),
                0.35 + 1.00 * rng.random(),
            )
            return setup, balanced_assignment(selected, True), params

        def best_vehicle(row, customer, t):
            best_k = 0
            best_value = float("inf")
            for k in range(M):
                members = [c for c in range(N) if row[c] == k]
                load = sum(expected_demand[t][c] for c in members)
                if members:
                    proximity = min(
                        transport[customer + 1][c + 1] for c in members
                    )
                else:
                    proximity = 2.0 * transport[0][customer + 1]
                value = (
                    load / max(vehicle_capacity, eps) +
                    0.2 * proximity / max(1.0, average_depot_cost)
                )
                if value < best_value:
                    best_value = value
                    best_k = k
            return best_k

        def mutate(parent, elapsed):
            setup = parent["setup"][:]
            assignment = [row[:] for row in parent["assignment"]]
            params = list(parent["params"])
            depth = 1 + int(
                math.log1p(max(0.0, elapsed)) * (0.45 + 0.65 * rng.random())
            )

            for _ in range(depth):
                mode = rng.randrange(9)
                t = rng.randrange(H)
                c = rng.randrange(N)
                if mode == 0:
                    if assignment[t][c] < 0:
                        assignment[t][c] = best_vehicle(assignment[t], c, t)
                        if rng.random() < 0.65:
                            setup[t] = 1
                    else:
                        assignment[t][c] = -1
                elif mode == 1:
                    setup[t] = 1 - int(bool(setup[t]))
                elif mode == 2:
                    if assignment[t][c] < 0:
                        assignment[t][c] = rng.randrange(M)
                    elif M > 1:
                        old = assignment[t][c]
                        new = rng.randrange(M - 1)
                        if new >= old:
                            new += 1
                        assignment[t][c] = new
                elif mode == 3:
                    active = [tt for tt in range(H) if assignment[tt][c] >= 0]
                    if active:
                        old_t = rng.choice(active)
                        direction = -1 if rng.random() < 0.5 else 1
                        new_t = min(H - 1, max(0, old_t + direction))
                        if new_t != old_t:
                            vehicle = assignment[old_t][c]
                            assignment[old_t][c] = -1
                            assignment[new_t][c] = vehicle
                    else:
                        assignment[t][c] = best_vehicle(assignment[t], c, t)
                elif mode == 4:
                    span_scale = 1 + int(math.sqrt(H) * (0.2 + rng.random()))
                    length = 1 + rng.randrange(max(1, min(H, span_scale)))
                    end = min(H, t + length)
                    add = rng.random() < 0.5
                    for tt in range(t, end):
                        if add:
                            assignment[tt][c] = best_vehicle(
                                assignment[tt], c, tt
                            )
                        else:
                            assignment[tt][c] = -1
                elif mode == 5:
                    add = rng.random() < 0.5
                    for cc in range(N):
                        if rng.random() < 0.25 + 0.5 * rng.random():
                            assignment[t][cc] = (
                                best_vehicle(assignment[t], cc, t) if add else -1
                            )
                    if add:
                        setup[t] = 1
                elif mode == 6:
                    sigma = 0.04 + 0.28 / (
                        1.0 + 0.04 * math.sqrt(max(0.0, elapsed))
                    )
                    p = rng.randrange(4)
                    params[p] += rng.gauss(0.0, sigma)
                elif mode == 7:
                    setup[t] = 1
                    for cc in range(N):
                        gross = penalty[cc] * expected_demand[t][cc]
                        if rng.random() < gross / max(
                                eps, gross + 2.0 * transport[0][cc + 1]):
                            assignment[t][cc] = best_vehicle(
                                assignment[t], cc, t
                            )
                else:
                    gap = 1 + rng.randrange(H)
                    phase = rng.randrange(gap)
                    for tt in range(H):
                        if tt % gap == phase:
                            assignment[tt][c] = best_vehicle(
                                assignment[tt], c, tt
                            )
                        elif rng.random() < 0.5:
                            assignment[tt][c] = -1

            if rng.random() < 0.35:
                sigma = 0.05 + 0.22 / (
                    1.0 + 0.03 * math.sqrt(max(0.0, elapsed))
                )
                p = rng.randrange(4)
                params[p] += rng.gauss(0.0, sigma)

            params[0] = min(2.0, max(0.0, params[0]))
            params[1] = min(2.0, max(0.0, params[1]))
            params[2] = min(1.8, max(0.5, params[2]))
            params[3] = min(1.4, max(0.3, params[3]))
            return setup, assignment, tuple(params)

        population = [{
            "cost": best_cost,
            "setup": best_setup[:],
            "assignment": [row[:] for row in best_assignment],
            "params": best_params,
        }]
        seed_kind = 0

        while time.perf_counter() < deadline:
            elapsed = max(0.0, time.perf_counter() - started)
            if seed_kind < 6:
                candidate = make_seed(seed_kind)
                seed_kind += 1
            else:
                restart_probability = 0.10 + 0.07 * (
                    0.5 + 0.5 * math.sin(math.log1p(elapsed) * 2.7)
                )
                if rng.random() < restart_probability:
                    candidate = random_candidate()
                else:
                    tournament_size = min(
                        len(population),
                        2 + int(math.log1p(elapsed) > 2.0),
                    )
                    parent = min(
                        (rng.choice(population) for _ in range(tournament_size)),
                        key=lambda entry: entry["cost"],
                    )
                    candidate = mutate(parent, elapsed)

            if time.perf_counter() >= deadline:
                break
            result = evaluate(
                candidate[0], candidate[1], candidate[2], deadline, True
            )
            if result is None:
                break

            cost, used_setup, active_assignment, params, routes, records = result
            entry = {
                "cost": cost,
                "setup": used_setup[:],
                "assignment": [row[:] for row in active_assignment],
                "params": params,
            }

            target_population = 4 + int(
                2.2 * math.sqrt(max(0.0, elapsed) + 1.0)
            )
            if len(population) < target_population:
                population.append(entry)
            else:
                worst_index = max(
                    range(len(population)),
                    key=lambda i: population[i]["cost"],
                )
                if cost < population[worst_index]["cost"]:
                    population[worst_index] = entry

            if cost + 1e-9 < best_cost:
                best_cost = cost
                best_setup = used_setup
                best_assignment = active_assignment
                best_params = params
                best_routes = routes
                best_records = records

            cache_target = 200 + int(
                45.0 * math.sqrt(max(0.0, elapsed) + 1.0)
            )
            while len(route_cache) > cache_target:
                route_cache.pop(next(iter(route_cache)))

        try:
            return materialize(best_setup, best_routes, best_records)
        except BaseException:
            return baseline_solution

    except BaseException:
        return _fallback(instance)