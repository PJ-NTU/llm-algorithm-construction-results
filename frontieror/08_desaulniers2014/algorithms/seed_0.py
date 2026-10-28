import math
import random
import time


_EPS = 1e-10
_INF = float("inf")


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    try:
        try:
            budget = float(time_limit_s)
        except BaseException:
            budget = 0.0
        if not math.isfinite(budget) or budget < 0.0:
            budget = 0.0

        data = _prepare(instance)
        rng = random.Random(_seed(data))
        reserve = min(0.06, max(0.002, 0.0015 * budget))
        search_deadline = started + max(0.0, budget - reserve)

        baseline_policy = {
            "kind": "buffer",
            "aggression": 1.0,
            "span": 1.0,
            "route_weight": 0.35,
            "noise": 0.0,
        }
        raw = _construct(data, rng, baseline_policy, None)

        emergency_deadline = started + max(0.6, min(3.0, 0.18 * max(1.0, budget)))
        while raw is None and time.perf_counter() < emergency_deadline:
            raw = _construct(
                data, rng, _random_policy(data, rng, time.perf_counter() - started),
                emergency_deadline
            )

        if raw is None:
            return _fallback(instance)

        init_deadline = min(
            search_deadline,
            started + max(0.15, min(1.8, 0.12 * max(1.0, budget)))
        )
        baseline = _make_state(data, raw, rng, init_deadline, 0.0, False)
        if baseline is None or not _valid_state(data, baseline):
            baseline = _make_state(data, raw, rng, None, 0.0, False)
        if baseline is None or not _valid_state(data, baseline):
            return _fallback(instance)

        best = baseline
        current = baseline
        last_improvement = time.perf_counter()
        bootstrap = [
            {
                "kind": "jit", "aggression": 0.0, "span": 0.7,
                "route_weight": 0.55, "noise": 0.0,
            },
            {
                "kind": "economic", "aggression": 0.42, "span": 0.85,
                "route_weight": 0.50, "noise": 0.0,
            },
            {
                "kind": "economic", "aggression": 0.78, "span": 1.25,
                "route_weight": 0.25, "noise": 0.03,
            },
        ]
        bootstrap_index = 0

        while time.perf_counter() < search_deadline:
            now = time.perf_counter()
            elapsed = max(0.0, now - started)
            remaining = search_deadline - now
            if remaining <= 0.0:
                break

            maturity = math.log1p(elapsed)
            slice_seconds = (
                0.010
                + min(0.32, 0.000012 * data["N"] * data["N"])
                + min(0.12, 0.012 * maturity)
            )
            local_deadline = min(search_deadline, now + slice_seconds)
            candidate = None

            try:
                if bootstrap_index < len(bootstrap):
                    policy = bootstrap[bootstrap_index]
                    bootstrap_index += 1
                    rebuilt = _construct(data, rng, policy, local_deadline)
                    if rebuilt is not None:
                        candidate = _make_state(
                            data, rebuilt, rng, local_deadline, elapsed, True
                        )
                else:
                    stagnation = max(0.0, now - last_improvement)
                    restart_probability = (
                        0.045
                        + 0.10 / (1.0 + elapsed / 30.0)
                        + 0.10 * stagnation / (stagnation + 25.0)
                    )
                    draw = rng.random()
                    if draw < restart_probability:
                        rebuilt = _construct(
                            data, rng, _random_policy(data, rng, elapsed),
                            local_deadline
                        )
                        if rebuilt is not None:
                            candidate = _make_state(
                                data, rebuilt, rng, local_deadline, elapsed, True
                            )
                    elif draw < restart_probability + 0.16:
                        base = best if rng.random() < 0.72 else current
                        candidate = _route_elimination_neighbor(
                            data, base, rng, local_deadline, elapsed
                        )
                    elif draw < restart_probability + 0.58:
                        base = best if rng.random() < 0.62 else current
                        candidate = _temporal_neighbor(
                            data, base, rng, local_deadline, elapsed
                        )
                    else:
                        base = best if rng.random() < 0.70 else current
                        candidate = _routing_neighbor(
                            data, base, rng, local_deadline, elapsed
                        )
            except BaseException:
                candidate = None

            if candidate is None or not _valid_state(data, candidate):
                continue

            if candidate["cost"] < best["cost"] - data["tol"]:
                best = candidate
                last_improvement = time.perf_counter()

            delta = candidate["cost"] - current["cost"]
            wave = math.sin(1.9 * math.log1p(elapsed) + elapsed / 41.0)
            temperature = data["avg_c"] * (
                0.012
                + 0.24 / (1.0 + elapsed / 55.0)
                + 0.055 * wave * wave
            )
            if delta <= data["tol"]:
                current = candidate
            elif rng.random() < math.exp(-delta / max(_EPS, temperature)):
                current = candidate
            elif rng.random() < 0.015 + 0.04 * elapsed / (elapsed + 180.0):
                current = best

        if not _valid_state(data, best):
            best = baseline
        return _emit(data, best)
    except BaseException:
        try:
            return _fallback(instance)
        except BaseException:
            return {"routes": [], "deliveries": [], "end_inventory": []}


def _prepare(instance):
    H = int(instance["num_periods"])
    N = int(instance["num_customers"])
    M = int(instance["num_vehicles"])
    Q = float(instance["vehicle_capacity"])
    if H <= 0 or N <= 0 or M <= 0 or not math.isfinite(Q) or Q <= 0.0:
        raise ValueError("invalid dimensions")

    p = [float(instance["supplier_production"][t]) for t in range(H)]
    d = [
        [float(instance["customer_demand"][i][t]) for t in range(H)]
        for i in range(N)
    ]
    U = [float(instance["inventory_capacity"][i]) for i in range(N + 1)]
    I0 = [float(instance["initial_inventory"][i]) for i in range(N + 1)]
    h = [float(instance["holding_cost"][i]) for i in range(N + 1)]
    c = [
        [float(instance["travel_cost"][i][j]) for j in range(N + 1)]
        for i in range(N + 1)
    ]

    prefix = [[0.0] * (H + 1) for _ in range(N)]
    suffix = [[0.0] * (H + 1) for _ in range(N)]
    anchor = [[0.0] * H for _ in range(N)]
    total_demand = [0.0] * H
    for i in range(N):
        for t in range(H):
            prefix[i][t + 1] = prefix[i][t] + d[i][t]
            total_demand[t] += d[i][t]
        for t in range(H - 1, -1, -1):
            suffix[i][t] = suffix[i][t + 1] + d[i][t]
            nxt = anchor[i][t + 1] if t + 1 < H else 0.0
            anchor[i][t] = d[i][t] + max(0.0, nxt - Q)

    aggregate_anchor = [0.0] * H
    supplier_anchor = [0.0] * H
    fleet = M * Q
    for t in range(H - 1, -1, -1):
        nxt_a = aggregate_anchor[t + 1] if t + 1 < H else 0.0
        nxt_s = supplier_anchor[t + 1] if t + 1 < H else 0.0
        aggregate_anchor[t] = total_demand[t] + max(0.0, nxt_a - fleet)
        supplier_anchor[t] = p[t] + max(0.0, nxt_s - fleet)

    positive = []
    for i in range(N + 1):
        for j in range(i + 1, N + 1):
            if math.isfinite(c[i][j]) and c[i][j] > 0.0:
                positive.append(c[i][j])
    avg_c = sum(positive) / len(positive) if positive else 1.0
    magnitude = max(
        [1.0, abs(Q)]
        + [abs(x) for x in p]
        + [abs(x) for x in U]
        + [abs(x) for row in d for x in row]
    )
    return {
        "H": H, "N": N, "M": M, "Q": Q,
        "p": p, "d": d, "U": U, "I0": I0, "h": h, "c": c,
        "prefix": prefix, "suffix": suffix, "anchor": anchor,
        "aggregate_anchor": aggregate_anchor,
        "supplier_anchor": supplier_anchor,
        "avg_c": avg_c, "tol": 1e-7 * magnitude,
        "exact_cache": {},
    }


def _seed(data):
    value = 1469598103934665603
    mask = (1 << 64) - 1
    values = [data["H"], data["N"], data["M"], data["Q"]]
    values.extend(data["p"])
    values.extend(data["I0"])
    for row in data["d"]:
        values.extend(row)
    for row in data["c"]:
        values.extend(row)
    for x in values:
        try:
            z = int(round(float(x) * 1000003.0)) & mask
        except BaseException:
            z = 0
        value ^= z
        value = (value * 1099511628211) & mask
    return value


def _random_policy(data, rng, elapsed):
    draw = rng.random()
    if draw < 0.12:
        kind = "buffer"
    elif draw < 0.38:
        kind = "jit"
    elif draw < 0.84:
        kind = "economic"
    else:
        kind = "random"
    mean_customer_h = sum(data["h"][1:]) / max(1, data["N"])
    scale = 1.0 + abs(data["h"][0]) + abs(mean_customer_h)
    bias = 0.5 + 0.35 * math.tanh(
        (data["h"][0] - mean_customer_h) / scale
    )
    aggression = 0.10 + 0.72 * rng.random() + 0.18 * bias
    if kind == "jit":
        aggression *= 0.18 + 0.25 * rng.random()
    elif kind == "buffer":
        aggression = 0.82 + 0.18 * rng.random()
    return {
        "kind": kind,
        "aggression": max(0.0, min(1.0, aggression)),
        "span": math.exp(0.55 * (2.0 * rng.random() - 1.0)),
        "route_weight": rng.random() * (
            0.55 + 0.15 * elapsed / (elapsed + 80.0)
        ),
        "noise": rng.random() * (
            0.04 + 0.20 * elapsed / (elapsed + 90.0)
        ),
    }


def _construct(data, rng, policy, deadline):
    H, N, M, Q = data["H"], data["N"], data["M"], data["Q"]
    supplier = data["I0"][0]
    customer_inv = list(data["I0"][1:])
    q_all = [[0.0] * N for _ in range(H)]
    inv_all = [[0.0] * (N + 1) for _ in range(H)]
    group_all = []

    for t in range(H):
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        available = supplier + data["p"][t]
        lower = [0.0] * N
        maximum = [0.0] * N
        desired = [0.0] * N

        for i in range(N):
            lower[i] = max(0.0, data["anchor"][i][t] - customer_inv[i])
            maximum[i] = max(
                0.0, min(Q, data["U"][i + 1] - customer_inv[i])
            )
            if lower[i] > maximum[i] + data["tol"]:
                return None

            kind = policy["kind"]
            if kind == "buffer":
                desired[i] = maximum[i]
            elif kind == "jit":
                desired[i] = lower[i]
            else:
                remaining = H - t
                future = data["suffix"][i][t]
                mean_demand = future / max(1, remaining)
                route_proxy = max(
                    data["avg_c"], 2.0 * data["c"][0][i + 1]
                )
                differential = data["h"][i + 1] - data["h"][0]
                if kind == "economic":
                    if differential <= _EPS:
                        raw_span = remaining * (
                            0.35 + 0.65 * policy["aggression"]
                        )
                    else:
                        lot = math.sqrt(max(
                            0.0,
                            2.0 * route_proxy * max(mean_demand, _EPS)
                            / max(differential, _EPS)
                        ))
                        raw_span = lot / max(mean_demand, _EPS)
                        raw_span *= 0.55 + 1.05 * policy["aggression"]
                else:
                    raw_span = 0.6 + policy["aggression"] * math.sqrt(remaining)
                    raw_span *= -math.log(max(1e-12, rng.random()))
                raw_span *= policy["span"]
                raw_span *= math.exp(
                    policy["noise"] * (2.0 * rng.random() - 1.0)
                )
                span = max(1, min(remaining, int(1.0 + raw_span)))
                need = (
                    data["prefix"][i][min(H, t + span)]
                    - data["prefix"][i][t]
                    - customer_inv[i]
                )
                useful = max(0.0, future - customer_inv[i])
                desired[i] = max(
                    lower[i], min(maximum[i], max(0.0, min(need, useful)))
                )

        lower_sum = sum(lower)
        aggregate_required = max(
            0.0, data["aggregate_anchor"][t] - sum(customer_inv)
        )
        supplier_required = max(
            0.0, supplier + data["supplier_anchor"][t] - data["U"][0]
        )
        required = max(lower_sum, aggregate_required, supplier_required)
        upper = min(available, M * Q, sum(maximum))
        if required > upper + data["tol"] or lower_sum > upper + data["tol"]:
            return None

        groups = _pack_amounts(
            data, lower, rng, policy["route_weight"], policy["noise"], deadline
        )
        if groups is None:
            return None
        q = list(lower)
        loads = [sum(q[i] for i in group) for group in groups]
        owner = [-1] * N
        for k, group in enumerate(groups):
            for i in group:
                owner[i] = k

        order = _urgency_order(data, t, customer_inv, q, rng, policy["noise"])
        total = sum(q)
        if total < required - data["tol"]:
            total = _allocate_to(
                required, desired, order, q, groups, loads, owner,
                data, rng, policy
            )
        if total < required - data["tol"]:
            total = _allocate_to(
                required, maximum, order, q, groups, loads, owner,
                data, rng, policy
            )
        if total < required - data["tol"]:
            return None

        desired_cap = min(upper, sum(desired))
        if policy["kind"] == "buffer":
            goal = upper
        else:
            goal = required + policy["aggression"] * max(
                0.0, desired_cap - required
            )
            goal = min(upper, max(required, goal))
        if total < goal - data["tol"]:
            total = _allocate_to(
                goal, desired, order, q, groups, loads, owner,
                data, rng, policy
            )
        if policy["kind"] == "buffer" and total < goal - data["tol"]:
            total = _allocate_to(
                goal, maximum, order, q, groups, loads, owner,
                data, rng, policy
            )

        shipped = sum(q)
        if shipped > available + data["tol"] or shipped > M * Q + data["tol"]:
            return None
        supplier = available - shipped
        if abs(supplier) <= data["tol"]:
            supplier = 0.0
        if supplier < -data["tol"] or supplier > data["U"][0] + data["tol"]:
            return None
        inv_all[t][0] = supplier

        for i in range(N):
            before = customer_inv[i] + q[i]
            if before + data["tol"] < data["d"][i][t]:
                return None
            if before > data["U"][i + 1] + data["tol"]:
                return None
            customer_inv[i] = before - data["d"][i][t]
            if abs(customer_inv[i]) <= data["tol"]:
                customer_inv[i] = 0.0
            if customer_inv[i] < -data["tol"]:
                return None
            inv_all[t][i + 1] = customer_inv[i]

        q_all[t] = q
        group_all.append(groups)

    return {"q": q_all, "inv": inv_all, "groups": group_all}


def _urgency_order(data, t, inventory, q, rng, noise):
    ranked = []
    holding_scale = 1.0 + max(abs(x) for x in data["h"])
    for i in range(data["N"]):
        stock = inventory[i] + q[i]
        cumulative = 0.0
        shortage = data["H"] + 1
        for u in range(t, data["H"]):
            cumulative += data["d"][i][u]
            if cumulative > stock + data["tol"]:
                shortage = u
                break
        holding_bias = 0.12 * (
            data["h"][i + 1] - data["h"][0]
        ) / holding_scale
        route_bias = -0.035 * data["c"][0][i + 1] / max(data["avg_c"], _EPS)
        ranked.append((
            shortage + holding_bias + route_bias + noise * rng.random(), i
        ))
    ranked.sort()
    return [i for _, i in ranked]


def _allocate_to(target, limits, order, q, groups, loads, owner,
                 data, rng, policy):
    total = sum(q)
    Q = data["Q"]
    for i in order:
        if total >= target - data["tol"]:
            break
        gap = max(0.0, limits[i] - q[i])
        if gap <= _EPS:
            continue
        k = owner[i]
        if k < 0:
            choices = []
            for b in range(data["M"]):
                room = Q - loads[b]
                if room <= _EPS:
                    continue
                cluster = _cluster_distance(data, groups[b], i)
                residual = max(0.0, room - min(gap, target - total))
                score = (
                    0.38 * residual / max(Q, _EPS)
                    + policy["route_weight"] * cluster / max(data["avg_c"], _EPS)
                    + policy["noise"] * rng.random()
                )
                choices.append((score, b))
            if not choices:
                continue
            k = min(choices)[1]
            owner[i] = k
            groups[k].append(i)
        addition = min(gap, target - total, max(0.0, Q - loads[k]))
        if addition <= _EPS:
            continue
        q[i] += addition
        loads[k] += addition
        total += addition
    return total


def _cluster_distance(data, group, customer):
    node = customer + 1
    if not group:
        return 2.0 * data["c"][0][node]
    return min(data["c"][node][j + 1] for j in group)


def _pack_amounts(data, amounts, rng, route_weight, noise, deadline):
    items = [i for i, x in enumerate(amounts) if x > _EPS]
    M, Q = data["M"], data["Q"]
    if not items:
        return [[] for _ in range(M)]
    if any(amounts[i] > Q + data["tol"] for i in items):
        return None

    for attempt in range(4):
        if attempt == 0:
            order = sorted(items, key=lambda i: (-amounts[i], i))
        else:
            order = sorted(
                items,
                key=lambda i: -(
                    amounts[i] + Q * (0.015 + 0.035 * attempt) * rng.random()
                )
            )
        groups = [[] for _ in range(M)]
        loads = [0.0] * M
        feasible = True
        for i in order:
            choices = []
            for k in range(M):
                if loads[k] + amounts[i] <= Q + data["tol"]:
                    residual = Q - loads[k] - amounts[i]
                    cluster = _cluster_distance(data, groups[k], i)
                    if attempt == 0:
                        score = residual
                    else:
                        score = (
                            residual
                            + route_weight * Q * cluster / max(data["avg_c"], _EPS)
                            + noise * Q * rng.random()
                        )
                    choices.append((score, k))
            if not choices:
                feasible = False
                break
            k = min(choices)[1]
            groups[k].append(i)
            loads[k] += amounts[i]
        if feasible:
            return groups
        if deadline is not None and time.perf_counter() >= deadline:
            return None

    dfs_deadline = deadline
    if dfs_deadline is None:
        dfs_deadline = time.perf_counter() + 0.40
    return _binpack_dfs(data, items, amounts, dfs_deadline)


def _binpack_dfs(data, items, amounts, deadline):
    ordered = sorted(items, key=lambda i: (-amounts[i], i))
    M, Q = data["M"], data["Q"]
    groups = [[] for _ in range(M)]
    loads = [0.0] * M
    remaining = [0.0] * (len(ordered) + 1)
    for pos in range(len(ordered) - 1, -1, -1):
        remaining[pos] = remaining[pos + 1] + amounts[ordered[pos]]
    answer = [None]

    def visit(pos):
        if time.perf_counter() >= deadline:
            return False
        if pos == len(ordered):
            answer[0] = [list(g) for g in groups]
            return True
        if remaining[pos] > sum(Q - x for x in loads) + data["tol"]:
            return False
        i = ordered[pos]
        choices = sorted(
            range(M), key=lambda k: (Q - loads[k] - amounts[i], k)
        )
        seen_loads = []
        for k in choices:
            if loads[k] + amounts[i] > Q + data["tol"]:
                continue
            duplicate = False
            for value in seen_loads:
                if abs(value - loads[k]) <= data["tol"]:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen_loads.append(loads[k])
            groups[k].append(i)
            loads[k] += amounts[i]
            if visit(pos + 1):
                return True
            loads[k] -= amounts[i]
            groups[k].pop()
        return False

    visit(0)
    return answer[0]


def _make_state(data, raw, rng, deadline, elapsed, randomize):
    inv = _inventories(data, raw["q"])
    if inv is None:
        return None
    groups_all = []
    routes_all = []
    travel_t = []
    for t in range(data["H"]):
        if deadline is not None:
            now = time.perf_counter()
            left = data["H"] - t
            sub_deadline = min(deadline, now + max(0.0, deadline - now) / max(1, left))
        else:
            sub_deadline = None
        result = _period_best(
            data, raw["q"][t], raw["groups"][t], None,
            rng, sub_deadline, elapsed, randomize
        )
        if result is None:
            return None
        groups, routes, cost = result
        groups_all.append(groups)
        routes_all.append(routes)
        travel_t.append(cost)
    holding = _holding_cost(data, inv)
    return {
        "q": [list(row) for row in raw["q"]],
        "inv": inv,
        "groups": groups_all,
        "routes": routes_all,
        "travel_t": travel_t,
        "holding": holding,
        "cost": holding + sum(travel_t),
    }


def _period_best(data, amounts, seed_groups, seed_routes, rng,
                 deadline, elapsed, randomize):
    base = _quick_period(data, amounts, seed_groups, seed_routes, rng, deadline)
    if base is None:
        return None
    best_groups, best_routes, best_cost = base
    best_seq = [[node - 1 for node in route[1:-1]] for route in best_routes]
    seen = set()

    def consider(seqs):
        nonlocal best_groups, best_routes, best_cost, best_seq
        normalized = _normalize_sequences(data, amounts, seqs)
        if normalized is None:
            return
        key = tuple(sorted(tuple(sorted(r)) for r in normalized))
        if key in seen:
            return
        seen.add(key)
        improved = _partition_descent(
            data, amounts, normalized, rng, deadline, elapsed
        )
        routes = [_route_from_seq(r) for r in improved]
        cost = sum(_route_cost(route, data["c"]) for route in routes)
        if cost < best_cost - data["tol"]:
            best_seq = [list(r) for r in improved]
            best_groups = [list(r) for r in improved]
            best_routes = routes
            best_cost = cost

    consider(best_seq)
    if deadline is None or time.perf_counter() < deadline:
        consider(_savings_sequences(data, amounts, rng, False, deadline))
    if deadline is None or time.perf_counter() < deadline:
        consider(_giant_sequences(data, amounts, rng, False, deadline))

    if randomize and deadline is not None:
        while time.perf_counter() < deadline:
            draw = rng.random()
            if draw < 0.48:
                candidate = _ruin_sequences(
                    data, amounts, best_seq, rng, deadline, elapsed
                )
            elif draw < 0.70:
                candidate = _savings_sequences(
                    data, amounts, rng, True, deadline
                )
            elif draw < 0.90:
                candidate = _giant_sequences(
                    data, amounts, rng, True, deadline
                )
            else:
                packed = _pack_amounts(
                    data, amounts, rng, 0.7, 0.20, deadline
                )
                candidate = None if packed is None else [
                    _cheapest_sequence(data, group) for group in packed
                ]
            consider(candidate)

    return best_groups, best_routes, best_cost


def _quick_period(data, amounts, seed_groups, seed_routes, rng, deadline):
    seqs = _seed_sequences(data, amounts, seed_groups, seed_routes)
    if seqs is None:
        groups = _pack_amounts(data, amounts, rng, 0.45, 0.0, deadline)
        if groups is None:
            return None
        seqs = []
        for group in groups:
            cheapest = _cheapest_sequence(data, group)
            nearest = _nearest_sequence(data, group, rng, False)
            if _sequence_cost(data, nearest) < _sequence_cost(data, cheapest):
                cheapest = nearest
            seqs.append(cheapest)
    result = []
    for seq in seqs:
        result.append(_route_descent(data, seq, deadline))
    routes = [_route_from_seq(seq) for seq in result]
    return result, routes, sum(_route_cost(r, data["c"]) for r in routes)


def _seed_sequences(data, amounts, groups, routes):
    if groups is None or len(groups) != data["M"]:
        return None
    seqs = []
    seen = set()
    for k in range(data["M"]):
        group = [
            i for i in groups[k]
            if 0 <= i < data["N"] and amounts[i] > _EPS
        ]
        if routes is not None and k < len(routes):
            route = routes[k]
            order = [node - 1 for node in route[1:-1]] if route else []
            if len(order) == len(group) and set(order) == set(group):
                group = order
        if sum(amounts[i] for i in group) > data["Q"] + data["tol"]:
            return None
        if any(i in seen for i in group):
            return None
        seen.update(group)
        seqs.append(group)
    needed = {i for i, x in enumerate(amounts) if x > _EPS}
    return seqs if seen == needed else None


def _normalize_sequences(data, amounts, seqs):
    if seqs is None:
        return None
    result = [[] for _ in range(data["M"])]
    seen = set()
    for k in range(min(data["M"], len(seqs))):
        for i in seqs[k]:
            if i in seen or i < 0 or i >= data["N"] or amounts[i] <= _EPS:
                return None
            result[k].append(i)
            seen.add(i)
        if sum(amounts[i] for i in result[k]) > data["Q"] + data["tol"]:
            return None
    needed = {i for i, x in enumerate(amounts) if x > _EPS}
    return result if seen == needed else None


def _route_from_seq(seq):
    if not seq:
        return []
    return [0] + [i + 1 for i in seq] + [0]


def _sequence_cost(data, seq):
    if not seq:
        return 0.0
    total = data["c"][0][seq[0] + 1]
    for a, b in zip(seq, seq[1:]):
        total += data["c"][a + 1][b + 1]
    return total + data["c"][seq[-1] + 1][0]


def _cheapest_sequence(data, group):
    remaining = list(group)
    if not remaining:
        return []
    first = min(remaining, key=lambda i: (2.0 * data["c"][0][i + 1], i))
    seq = [first]
    remaining.remove(first)
    while remaining:
        best = None
        for i in remaining:
            node = i + 1
            for pos in range(len(seq) + 1):
                a = 0 if pos == 0 else seq[pos - 1] + 1
                b = 0 if pos == len(seq) else seq[pos] + 1
                delta = data["c"][a][node] + data["c"][node][b] - data["c"][a][b]
                value = (delta, i, pos)
                if best is None or value < best:
                    best = value
        _, i, pos = best
        seq.insert(pos, i)
        remaining.remove(i)
    return seq


def _nearest_sequence(data, group, rng, randomized):
    remaining = list(group)
    seq = []
    current = 0
    while remaining:
        ranked = sorted(remaining, key=lambda i: data["c"][current][i + 1])
        if randomized and len(ranked) > 1:
            width = min(len(ranked), 1 + int(math.sqrt(len(ranked)) * rng.random()))
            chosen = ranked[rng.randrange(width)]
        else:
            chosen = ranked[0]
        seq.append(chosen)
        remaining.remove(chosen)
        current = chosen + 1
    return seq


def _route_descent(data, sequence, deadline):
    seq = list(sequence)
    if len(seq) <= 2:
        return seq
    while deadline is None or time.perf_counter() < deadline:
        best_delta = -data["tol"]
        best_move = None
        n = len(seq)
        for left in range(n - 1):
            a = 0 if left == 0 else seq[left - 1] + 1
            b = seq[left] + 1
            for right in range(left + 1, n):
                c = seq[right] + 1
                d = 0 if right + 1 == n else seq[right + 1] + 1
                delta = (
                    data["c"][a][c] + data["c"][b][d]
                    - data["c"][a][b] - data["c"][c][d]
                )
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("two", left, right, False)
            if deadline is not None and time.perf_counter() >= deadline:
                return seq
        if best_move is not None:
            _, left, right, _ = best_move
            seq[left:right + 1] = reversed(seq[left:right + 1])
            continue

        best_delta = -data["tol"]
        best_move = None
        for length in (1, 2, 3):
            if length >= n:
                continue
            for start in range(n - length + 1):
                end = start + length
                block = seq[start:end]
                before = 0 if start == 0 else seq[start - 1] + 1
                first = block[0] + 1
                last = block[-1] + 1
                after = 0 if end == n else seq[end] + 1
                removal = (
                    data["c"][before][after]
                    - data["c"][before][first]
                    - data["c"][last][after]
                )
                rest = seq[:start] + seq[end:]
                for edge in range(len(rest) + 1):
                    u = 0 if edge == 0 else rest[edge - 1] + 1
                    v = 0 if edge == len(rest) else rest[edge] + 1
                    forward = removal + (
                        data["c"][u][first] + data["c"][last][v]
                        - data["c"][u][v]
                    )
                    if forward < best_delta:
                        best_delta = forward
                        best_move = ("or", start, end, edge, False)
                    reverse = removal + (
                        data["c"][u][last] + data["c"][first][v]
                        - data["c"][u][v]
                    )
                    if reverse < best_delta:
                        best_delta = reverse
                        best_move = ("or", start, end, edge, True)
                if deadline is not None and time.perf_counter() >= deadline:
                    return seq
        if best_move is None:
            break
        _, start, end, edge, reverse = best_move
        block = seq[start:end]
        rest = seq[:start] + seq[end:]
        if reverse:
            block.reverse()
        seq = rest[:edge] + block + rest[edge:]
    return seq


def _partition_descent(data, amounts, sequences, rng, deadline, elapsed):
    seqs = [list(r) for r in sequences]
    for k in range(len(seqs)):
        seqs[k] = _route_descent(data, seqs[k], deadline)
    loads = [sum(amounts[i] for i in r) for r in seqs]

    while deadline is None or time.perf_counter() < deadline:
        best = None
        for a, route_a in enumerate(seqs):
            for pos, customer in enumerate(route_a):
                node = customer + 1
                prev_node = 0 if pos == 0 else route_a[pos - 1] + 1
                next_node = 0 if pos + 1 == len(route_a) else route_a[pos + 1] + 1
                removal = (
                    data["c"][prev_node][next_node]
                    - data["c"][prev_node][node]
                    - data["c"][node][next_node]
                )
                for b, route_b in enumerate(seqs):
                    if a == b or loads[b] + amounts[customer] > data["Q"] + data["tol"]:
                        continue
                    for edge in range(len(route_b) + 1):
                        u = 0 if edge == 0 else route_b[edge - 1] + 1
                        v = 0 if edge == len(route_b) else route_b[edge] + 1
                        delta = removal + data["c"][u][node] + data["c"][node][v] - data["c"][u][v]
                        if best is None or delta < best[0]:
                            best = (delta, a, b, pos, edge, customer)
                if deadline is not None and time.perf_counter() >= deadline:
                    return seqs
        if best is not None and best[0] < -data["tol"]:
            _, a, b, pos, edge, customer = best
            seqs[a].pop(pos)
            seqs[b].insert(edge, customer)
            loads[a] -= amounts[customer]
            loads[b] += amounts[customer]
            seqs[a] = _route_descent(data, seqs[a], deadline)
            seqs[b] = _route_descent(data, seqs[b], deadline)
            continue

        best = None
        for a in range(len(seqs)):
            for b in range(a + 1, len(seqs)):
                for pa, ia in enumerate(seqs[a]):
                    na = ia + 1
                    ap = 0 if pa == 0 else seqs[a][pa - 1] + 1
                    an = 0 if pa + 1 == len(seqs[a]) else seqs[a][pa + 1] + 1
                    for pb, ib in enumerate(seqs[b]):
                        new_a = loads[a] - amounts[ia] + amounts[ib]
                        new_b = loads[b] - amounts[ib] + amounts[ia]
                        if new_a > data["Q"] + data["tol"] or new_b > data["Q"] + data["tol"]:
                            continue
                        nb = ib + 1
                        bp = 0 if pb == 0 else seqs[b][pb - 1] + 1
                        bn = 0 if pb + 1 == len(seqs[b]) else seqs[b][pb + 1] + 1
                        delta = (
                            data["c"][ap][nb] + data["c"][nb][an]
                            - data["c"][ap][na] - data["c"][na][an]
                            + data["c"][bp][na] + data["c"][na][bn]
                            - data["c"][bp][nb] - data["c"][nb][bn]
                        )
                        if best is None or delta < best[0]:
                            best = (delta, a, b, pa, pb, ia, ib, new_a, new_b)
                if deadline is not None and time.perf_counter() >= deadline:
                    return seqs
        if best is not None and best[0] < -data["tol"]:
            _, a, b, pa, pb, ia, ib, la, lb = best
            seqs[a][pa], seqs[b][pb] = ib, ia
            loads[a], loads[b] = la, lb
            seqs[a] = _route_descent(data, seqs[a], deadline)
            seqs[b] = _route_descent(data, seqs[b], deadline)
            continue

        best = None
        for a in range(len(seqs)):
            prefix_a = [0.0]
            for i in seqs[a]:
                prefix_a.append(prefix_a[-1] + amounts[i])
            for b in range(a + 1, len(seqs)):
                prefix_b = [0.0]
                for i in seqs[b]:
                    prefix_b.append(prefix_b[-1] + amounts[i])
                for ca in range(len(seqs[a]) + 1):
                    left_a = 0 if ca == 0 else seqs[a][ca - 1] + 1
                    right_a = 0 if ca == len(seqs[a]) else seqs[a][ca] + 1
                    tail_a = loads[a] - prefix_a[ca]
                    for cb in range(len(seqs[b]) + 1):
                        if ca == len(seqs[a]) and cb == len(seqs[b]):
                            continue
                        new_a = prefix_a[ca] + loads[b] - prefix_b[cb]
                        new_b = prefix_b[cb] + tail_a
                        if new_a > data["Q"] + data["tol"] or new_b > data["Q"] + data["tol"]:
                            continue
                        left_b = 0 if cb == 0 else seqs[b][cb - 1] + 1
                        right_b = 0 if cb == len(seqs[b]) else seqs[b][cb] + 1
                        delta = (
                            data["c"][left_a][right_b]
                            + data["c"][left_b][right_a]
                            - data["c"][left_a][right_a]
                            - data["c"][left_b][right_b]
                        )
                        if best is None or delta < best[0]:
                            best = (delta, a, b, ca, cb, new_a, new_b)
                if deadline is not None and time.perf_counter() >= deadline:
                    return seqs
        if best is not None and best[0] < -data["tol"]:
            _, a, b, ca, cb, la, lb = best
            old_a = seqs[a]
            old_b = seqs[b]
            seqs[a] = old_a[:ca] + old_b[cb:]
            seqs[b] = old_b[:cb] + old_a[ca:]
            loads[a], loads[b] = la, lb
            seqs[a] = _route_descent(data, seqs[a], deadline)
            seqs[b] = _route_descent(data, seqs[b], deadline)
            continue
        break

    threshold = min(13, 9 + int(math.log1p(max(0.0, elapsed)) / 1.75))
    for k, seq in enumerate(seqs):
        if len(seq) <= threshold and (deadline is None or time.perf_counter() < deadline):
            exact = _exact_sequence(data, seq, deadline)
            if exact is not None and _sequence_cost(data, exact) < _sequence_cost(data, seq) - data["tol"]:
                seqs[k] = exact
    return seqs


def _exact_sequence(data, group, deadline):
    key = tuple(sorted(group))
    if not key:
        return []
    cached = data["exact_cache"].get(key)
    if cached is not None:
        return list(cached)
    n = len(key)
    if n == 1:
        data["exact_cache"][key] = key
        return list(key)
    full = (1 << n) - 1
    dp = {}
    parent = {}
    for j, customer in enumerate(key):
        dp[(1 << j, j)] = data["c"][0][customer + 1]
        parent[(1 << j, j)] = -1
    scans = 0
    for mask in range(1, full + 1):
        for last in range(n):
            state = (mask, last)
            if state not in dp:
                continue
            base = dp[state]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                new_state = (mask | (1 << nxt), nxt)
                value = base + data["c"][key[last] + 1][key[nxt] + 1]
                if value < dp.get(new_state, _INF):
                    dp[new_state] = value
                    parent[new_state] = last
                scans += 1
                if scans >= 1024:
                    scans = 0
                    if deadline is not None and time.perf_counter() >= deadline:
                        return None
    last = min(
        range(n),
        key=lambda j: dp.get((full, j), _INF) + data["c"][key[j] + 1][0]
    )
    if (full, last) not in dp:
        return None
    sequence = []
    mask = full
    while last >= 0:
        sequence.append(key[last])
        previous = parent[(mask, last)]
        mask ^= 1 << last
        last = previous
    sequence.reverse()
    if len(data["exact_cache"]) < 4096:
        data["exact_cache"][key] = tuple(sequence)
    return sequence


def _savings_sequences(data, amounts, rng, randomized, deadline):
    items = [i for i, x in enumerate(amounts) if x > _EPS]
    if not items:
        return [[] for _ in range(data["M"])]
    if len(items) > 230:
        return None
    routes = {i: [i] for i in items}
    loads = {i: amounts[i] for i in items}
    owner = {i: i for i in items}
    pairs = []
    for a_pos in range(len(items)):
        i = items[a_pos]
        for b_pos in range(a_pos + 1, len(items)):
            j = items[b_pos]
            saving = (
                data["c"][0][i + 1] + data["c"][0][j + 1]
                - data["c"][i + 1][j + 1]
            )
            score = saving
            if randomized:
                score += data["avg_c"] * 0.22 * (2.0 * rng.random() - 1.0)
            pairs.append((score, saving, i, j))
        if deadline is not None and time.perf_counter() >= deadline:
            return None
    pairs.sort(reverse=True)
    for _, saving, i, j in pairs:
        ri, rj = owner[i], owner[j]
        if ri == rj or ri not in routes or rj not in routes:
            continue
        if len(routes) <= data["M"] and saving <= data["tol"]:
            continue
        first, second = routes[ri], routes[rj]
        if i != first[0] and i != first[-1]:
            continue
        if j != second[0] and j != second[-1]:
            continue
        if loads[ri] + loads[rj] > data["Q"] + data["tol"]:
            continue
        if first[0] == i:
            first = list(reversed(first))
        else:
            first = list(first)
        if second[-1] == j:
            second = list(reversed(second))
        else:
            second = list(second)
        merged = first + second
        routes[ri] = merged
        loads[ri] += loads[rj]
        del routes[rj]
        del loads[rj]
        for x in merged:
            owner[x] = ri
        if deadline is not None and time.perf_counter() >= deadline:
            break
    if len(routes) > data["M"]:
        return None
    result = list(routes.values())
    result.sort(key=lambda r: (-sum(amounts[i] for i in r), r[0]))
    result.extend([[] for _ in range(data["M"] - len(result))])
    return result


def _giant_sequences(data, amounts, rng, randomized, deadline):
    items = [i for i, x in enumerate(amounts) if x > _EPS]
    if not items:
        return [[] for _ in range(data["M"])]
    if len(items) > 320:
        return None
    remaining = list(items)
    order = []
    current = 0
    while remaining:
        ranked = sorted(remaining, key=lambda i: data["c"][current][i + 1])
        if randomized and len(ranked) > 1:
            width = min(
                len(ranked),
                1 + int((1.0 + math.sqrt(len(ranked))) * rng.random())
            )
            chosen = ranked[rng.randrange(width)]
        else:
            chosen = ranked[0]
        order.append(chosen)
        remaining.remove(chosen)
        current = chosen + 1
        if deadline is not None and time.perf_counter() >= deadline:
            return None

    L = len(order)
    R = min(data["M"], L)
    dp = [[_INF] * (L + 1) for _ in range(R + 1)]
    parent = [[-1] * (L + 1) for _ in range(R + 1)]
    dp[0][0] = 0.0
    for r in range(1, R + 1):
        for start in range(L):
            if not math.isfinite(dp[r - 1][start]):
                continue
            load = 0.0
            segment_cost = 0.0
            last = 0
            for end in range(start, L):
                customer = order[end]
                node = customer + 1
                load += amounts[customer]
                if load > data["Q"] + data["tol"]:
                    break
                if end == start:
                    segment_cost = data["c"][0][node] + data["c"][node][0]
                else:
                    segment_cost += (
                        data["c"][last][node] + data["c"][node][0]
                        - data["c"][last][0]
                    )
                last = node
                value = dp[r - 1][start] + segment_cost
                if value < dp[r][end + 1]:
                    dp[r][end + 1] = value
                    parent[r][end + 1] = start
            if deadline is not None and time.perf_counter() >= deadline:
                return None
    best_r = min(range(1, R + 1), key=lambda r: dp[r][L])
    if not math.isfinite(dp[best_r][L]):
        return None
    groups = []
    pos, r = L, best_r
    while pos > 0:
        start = parent[r][pos]
        if start < 0:
            return None
        groups.append(order[start:pos])
        pos = start
        r -= 1
    groups.reverse()
    groups.extend([[] for _ in range(data["M"] - len(groups))])
    return groups


def _ruin_sequences(data, amounts, sequences, rng, deadline, elapsed):
    items = [i for route in sequences for i in route]
    if len(items) <= 2:
        return None
    fraction = 0.10 + 0.28 * elapsed / (elapsed + 100.0) + 0.12 * rng.random()
    remove_count = max(2, min(len(items) - 1, int(1.0 + fraction * len(items))))
    seed = rng.choice(items)
    owner = {}
    for k, route in enumerate(sequences):
        for i in route:
            owner[i] = k
    related = sorted(
        items,
        key=lambda i: (
            data["c"][seed + 1][i + 1]
            - (0.30 * data["avg_c"] if owner.get(i) == owner.get(seed) else 0.0)
            + 0.18 * data["avg_c"] * rng.random()
        )
    )
    removed = set(related[:remove_count])
    routes = [[i for i in route if i not in removed] for route in sequences]
    loads = [sum(amounts[i] for i in route) for route in routes]
    pending = list(removed)

    while pending:
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        chosen = None
        chosen_options = None
        chosen_regret = -_INF
        for customer in pending:
            node = customer + 1
            options = []
            for k, route in enumerate(routes):
                if loads[k] + amounts[customer] > data["Q"] + data["tol"]:
                    continue
                for edge in range(len(route) + 1):
                    a = 0 if edge == 0 else route[edge - 1] + 1
                    b = 0 if edge == len(route) else route[edge] + 1
                    delta = data["c"][a][node] + data["c"][node][b] - data["c"][a][b]
                    noisy = delta + 0.06 * data["avg_c"] * rng.random()
                    options.append((noisy, delta, k, edge))
            if not options:
                return None
            options.sort()
            second = options[min(2, len(options) - 1)][0]
            regret = second - options[0][0]
            if regret > chosen_regret:
                chosen_regret = regret
                chosen = customer
                chosen_options = options
        width = min(len(chosen_options), 1 + int(2.0 * rng.random()))
        option = chosen_options[rng.randrange(width)]
        _, _, k, edge = option
        routes[k].insert(edge, chosen)
        loads[k] += amounts[chosen]
        pending.remove(chosen)
    return routes


def _inventories(data, q):
    if len(q) != data["H"]:
        return None
    result = [[0.0] * (data["N"] + 1) for _ in range(data["H"])]
    supplier = data["I0"][0]
    customers = list(data["I0"][1:])
    for t in range(data["H"]):
        if len(q[t]) != data["N"]:
            return None
        shipped = 0.0
        for i in range(data["N"]):
            value = q[t][i]
            if not math.isfinite(value) or value < -data["tol"] or value > data["Q"] + data["tol"]:
                return None
            shipped += max(0.0, value)
        if shipped > data["M"] * data["Q"] + data["tol"]:
            return None
        supplier += data["p"][t] - shipped
        if supplier < -data["tol"] or supplier > data["U"][0] + data["tol"]:
            return None
        if abs(supplier) <= data["tol"]:
            supplier = 0.0
        result[t][0] = supplier
        next_customers = [0.0] * data["N"]
        for i in range(data["N"]):
            before = customers[i] + max(0.0, q[t][i])
            if before + data["tol"] < data["d"][i][t]:
                return None
            if before > data["U"][i + 1] + data["tol"]:
                return None
            value = before - data["d"][i][t]
            if abs(value) <= data["tol"]:
                value = 0.0
            if value < -data["tol"]:
                return None
            next_customers[i] = value
            result[t][i + 1] = value
        customers = next_customers
    return result


def _holding_cost(data, inv):
    total = 0.0
    for t in range(data["H"]):
        for loc in range(data["N"] + 1):
            total += data["h"][loc] * inv[t][loc]
    return total


def _transfer_limit(data, q, inv, customer, source, target):
    H = data["H"]
    if source == target or source < 0 or source >= H:
        return 0.0
    limit = q[source][customer]
    if target < H:
        limit = min(limit, data["Q"] - q[target][customer])
        limit = min(limit, data["M"] * data["Q"] - sum(q[target]))
    if source < target:
        for t in range(source, target):
            limit = min(limit, inv[t][customer + 1])
            limit = min(limit, data["U"][0] - inv[t][0])
    else:
        for t in range(target, source):
            customer_room = (
                data["U"][customer + 1]
                - data["d"][customer][t]
                - inv[t][customer + 1]
            )
            limit = min(limit, customer_room)
            limit = min(limit, inv[t][0])
    if not math.isfinite(limit):
        return 0.0
    return max(0.0, limit)


def _route_change_estimate(data, state, customer, source, target, delta):
    node = customer + 1
    estimate = 0.0
    if state["q"][source][customer] - delta <= data["tol"]:
        for route in state["routes"][source]:
            if node in route:
                pos = route.index(node)
                estimate += (
                    data["c"][route[pos - 1]][route[pos + 1]]
                    - data["c"][route[pos - 1]][node]
                    - data["c"][node][route[pos + 1]]
                )
                break
    if target < data["H"] and state["q"][target][customer] <= _EPS:
        best = _INF
        for k, route in enumerate(state["routes"][target]):
            load = sum(state["q"][target][i] for i in state["groups"][target][k])
            if load + delta > data["Q"] + data["tol"]:
                continue
            if not route:
                insertion = 2.0 * data["c"][0][node]
            else:
                insertion = min(
                    data["c"][route[pos - 1]][node]
                    + data["c"][node][route[pos]]
                    - data["c"][route[pos - 1]][route[pos]]
                    for pos in range(1, len(route))
                )
            best = min(best, insertion)
        estimate += best if math.isfinite(best) else 2.0 * data["avg_c"]
    return estimate


def _temporal_neighbor(data, state, rng, deadline, elapsed):
    visits = [
        (t, i)
        for t in range(data["H"])
        for i in range(data["N"])
        if state["q"][t][i] > _EPS
    ]
    if not visits:
        return None
    now = time.perf_counter()
    generation_deadline = now + 0.23 * max(0.0, deadline - now)
    proposals = []
    first = True
    while first or time.perf_counter() < generation_deadline:
        first = False
        source, customer = rng.choice(visits)
        existing = [
            t for t in range(data["H"])
            if t != source and state["q"][t][customer] > _EPS
        ]
        draw = rng.random()
        if draw < 0.15:
            target = data["H"]
        elif existing and draw < 0.58:
            target = rng.choice(existing)
        else:
            span = min(
                data["H"] - 1,
                1 + int(math.sqrt(1.0 + elapsed) * (0.5 + rng.random()))
            )
            nearby = [
                t for t in range(data["H"])
                if t != source and abs(t - source) <= span
            ]
            target = rng.choice(nearby) if nearby else data["H"]
        limit = _transfer_limit(
            data, state["q"], state["inv"], customer, source, target
        )
        if limit > data["tol"]:
            holding_delta = (
                (source - target)
                * (data["h"][customer + 1] - data["h"][0])
                * limit
            )
            route_delta = _route_change_estimate(
                data, state, customer, source, target, limit
            )
            score = holding_delta + route_delta
            score += 0.02 * data["avg_c"] * rng.random()
            proposals.append((score, customer, source, target, limit))
            if len(proposals) > 180:
                proposals.sort(key=lambda x: x[0])
                del proposals[120:]
        if time.perf_counter() >= deadline:
            break
    if not proposals:
        return None
    proposals.sort(key=lambda x: x[0])
    best_candidate = None
    best_cost = _INF
    cursor = 0
    while cursor < len(proposals) and time.perf_counter() < deadline:
        _, customer, source, target, delta = proposals[cursor]
        cursor += 1
        q = [list(row) for row in state["q"]]
        q[source][customer] -= delta
        if q[source][customer] <= data["tol"]:
            q[source][customer] = 0.0
        affected = {source}
        if target < data["H"]:
            q[target][customer] += delta
            affected.add(target)
        now = time.perf_counter()
        candidate_deadline = min(
            deadline, now + max(0.003, 0.32 * max(0.0, deadline - now))
        )
        candidate = _state_with_q(
            data, state, q, affected, rng, candidate_deadline, elapsed, True
        )
        if candidate is not None and candidate["cost"] < best_cost:
            best_candidate = candidate
            best_cost = candidate["cost"]
            if best_cost < state["cost"] - 0.05 * data["avg_c"]:
                break
    return best_candidate


def _route_elimination_neighbor(data, state, rng, deadline, elapsed):
    choices = []
    for t in range(data["H"]):
        for k, group in enumerate(state["groups"][t]):
            if group:
                cost = _route_cost(state["routes"][t][k], data["c"])
                choices.append((max(_EPS, cost), t, k))
    if not choices:
        return None
    total_weight = sum(x[0] for x in choices)
    draw = rng.random() * total_weight
    source = choices[-1][1]
    route_index = choices[-1][2]
    for weight, t, k in choices:
        draw -= weight
        if draw <= 0.0:
            source, route_index = t, k
            break

    q = [list(row) for row in state["q"]]
    inv = [list(row) for row in state["inv"]]
    affected = {source}
    moved = False
    customers = list(state["groups"][source][route_index])
    customers.sort(key=lambda i: q[source][i])

    for customer in customers:
        while q[source][customer] > data["tol"] and time.perf_counter() < deadline:
            targets = []
            for target in range(data["H"] + 1):
                if target == source:
                    continue
                limit = _transfer_limit(data, q, inv, customer, source, target)
                if limit <= data["tol"]:
                    continue
                holding = (
                    (source - target)
                    * (data["h"][customer + 1] - data["h"][0])
                    * limit
                )
                route_bias = 0.0
                if target < data["H"] and q[target][customer] <= _EPS:
                    route_bias = 0.8 * data["c"][0][customer + 1]
                elif target < data["H"]:
                    route_bias = -0.35 * data["avg_c"]
                score = holding + route_bias + 0.01 * abs(target - source) * data["avg_c"]
                targets.append((score, target, limit))
            if not targets:
                break
            targets.sort(key=lambda x: x[0])
            _, target, delta = targets[0]
            q[source][customer] -= delta
            if q[source][customer] <= data["tol"]:
                q[source][customer] = 0.0
            if target < data["H"]:
                q[target][customer] += delta
                affected.add(target)
            inv_new = _inventories(data, q)
            if inv_new is None:
                return None
            inv = inv_new
            moved = True
    if not moved:
        return None
    return _state_with_q(
        data, state, q, affected, rng, deadline, elapsed, True
    )


def _state_with_q(data, state, q, affected, rng, deadline, elapsed, randomize):
    inv = _inventories(data, q)
    if inv is None:
        return None
    groups = list(state["groups"])
    routes = list(state["routes"])
    travel_t = list(state["travel_t"])
    periods = sorted(affected)
    for pos, t in enumerate(periods):
        now = time.perf_counter()
        if deadline is None:
            sub_deadline = None
        else:
            left = len(periods) - pos
            sub_deadline = min(
                deadline, now + max(0.0, deadline - now) / max(1, left)
            )
        result = _period_best(
            data, q[t], state["groups"][t], state["routes"][t],
            rng, sub_deadline, elapsed, randomize
        )
        if result is None:
            return None
        period_groups, period_routes, period_cost = result
        groups[t] = period_groups
        routes[t] = period_routes
        travel_t[t] = period_cost
    holding = _holding_cost(data, inv)
    return {
        "q": q, "inv": inv, "groups": groups, "routes": routes,
        "travel_t": travel_t, "holding": holding,
        "cost": holding + sum(travel_t),
    }


def _routing_neighbor(data, state, rng, deadline, elapsed):
    weights = [max(_EPS, x + 0.01 * data["avg_c"]) for x in state["travel_t"]]
    draw = rng.random() * sum(weights)
    t = len(weights) - 1
    for idx, weight in enumerate(weights):
        draw -= weight
        if draw <= 0.0:
            t = idx
            break
    result = _period_best(
        data, state["q"][t], state["groups"][t], state["routes"][t],
        rng, deadline, elapsed, True
    )
    if result is None:
        return None
    groups_t, routes_t, cost_t = result
    groups = list(state["groups"])
    routes = list(state["routes"])
    travel_t = list(state["travel_t"])
    groups[t] = groups_t
    routes[t] = routes_t
    travel_t[t] = cost_t
    return {
        "q": state["q"], "inv": state["inv"],
        "groups": groups, "routes": routes, "travel_t": travel_t,
        "holding": state["holding"],
        "cost": state["holding"] + sum(travel_t),
    }


def _route_cost(route, travel):
    if not route:
        return 0.0
    return sum(travel[route[i]][route[i + 1]] for i in range(len(route) - 1))


def _valid_state(data, state):
    try:
        inv = _inventories(data, state["q"])
        if inv is None:
            return False
        if len(state["groups"]) != data["H"] or len(state["routes"]) != data["H"]:
            return False
        for t in range(data["H"]):
            if len(state["groups"][t]) != data["M"] or len(state["routes"][t]) != data["M"]:
                return False
            seen = set()
            for k in range(data["M"]):
                group = state["groups"][t][k]
                route = state["routes"][t][k]
                if any(i in seen for i in group):
                    return False
                seen.update(group)
                if sum(state["q"][t][i] for i in group) > data["Q"] + data["tol"]:
                    return False
                if group:
                    if not route or route[0] != 0 or route[-1] != 0:
                        return False
                    if len(route[1:-1]) != len(group):
                        return False
                    if set(route[1:-1]) != {i + 1 for i in group}:
                        return False
                elif route:
                    return False
            required = {i for i, x in enumerate(state["q"][t]) if x > _EPS}
            if seen != required:
                return False
        return True
    except BaseException:
        return False


def _emit(data, state):
    routes = [[[] for _ in range(data["M"])] for _ in range(data["H"])]
    deliveries = [
        [[0.0] * data["N"] for _ in range(data["M"])]
        for _ in range(data["H"])
    ]
    for t in range(data["H"]):
        for k in range(data["M"]):
            routes[t][k] = list(state["routes"][t][k])
            for i in state["groups"][t][k]:
                deliveries[t][k][i] = float(max(0.0, state["q"][t][i]))
    end_inventory = [
        [0.0 if abs(x) <= data["tol"] else float(x) for x in row]
        for row in state["inv"]
    ]
    return {
        "routes": routes,
        "deliveries": deliveries,
        "end_inventory": end_inventory,
    }


def _fallback(instance):
    try:
        H = max(0, int(instance.get("num_periods", 0)))
    except BaseException:
        H = 0
    try:
        N = max(0, int(instance.get("num_customers", 0)))
    except BaseException:
        N = 0
    try:
        M = max(0, int(instance.get("num_vehicles", 0)))
    except BaseException:
        M = 0
    try:
        Q = max(0.0, float(instance.get("vehicle_capacity", 0.0)))
    except BaseException:
        Q = 0.0

    routes = [[[] for _ in range(M)] for _ in range(H)]
    deliveries = [[[0.0] * N for _ in range(M)] for _ in range(H)]
    raw_initial = instance.get("initial_inventory", [0.0] * (N + 1))
    inventory = []
    for i in range(N + 1):
        try:
            inventory.append(float(raw_initial[i]))
        except BaseException:
            inventory.append(0.0)
    production = instance.get("supplier_production", [0.0] * H)
    demand = instance.get("customer_demand", [[0.0] * H for _ in range(N)])
    end_inventory = []

    for t in range(H):
        try:
            inventory[0] += float(production[t])
        except BaseException:
            pass
        loads = [0.0] * M
        clients = [[] for _ in range(M)]
        for i in range(N):
            try:
                dem = float(demand[i][t])
            except BaseException:
                dem = 0.0
            need = max(0.0, dem - inventory[i + 1])
            chosen = None
            for k in range(M):
                if loads[k] + need <= Q + 1e-8 and need <= inventory[0] + 1e-8:
                    chosen = k
                    break
            delivered = 0.0
            if chosen is not None and need > 0.0:
                delivered = need
                deliveries[t][chosen][i] = delivered
                loads[chosen] += delivered
                clients[chosen].append(i)
                inventory[0] -= delivered
            inventory[i + 1] = max(0.0, inventory[i + 1] + delivered - dem)
        for k in range(M):
            if clients[k]:
                routes[t][k] = [0] + [i + 1 for i in clients[k]] + [0]
        end_inventory.append(list(inventory))
    return {
        "routes": routes,
        "deliveries": deliveries,
        "end_inventory": end_inventory,
    }
