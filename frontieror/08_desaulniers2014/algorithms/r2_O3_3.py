"""Joint search of cross-period replenishment patterns and inventory routing
Drives replenishment dynamic programming with route marginal costs, repairs replenishment patterns by full-horizon inventory-flow projection and joint mixed-integer neighbourhoods, and improves feasible solutions with route destruction-reconstruction.
"""
import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import csc_matrix
    _HAS_OPT = True
except Exception:
    _HAS_OPT = False

_EPS = 1e-10
_INF = float("inf")


def _prepare(instance):
    H = int(instance["num_periods"])
    N = int(instance["num_customers"])
    M = int(instance["num_vehicles"])
    Q = float(instance["vehicle_capacity"])
    if H <= 0 or N <= 0 or M <= 0 or not math.isfinite(Q) or Q <= 0.0:
        raise ValueError("invalid input dimensions")

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
        "avg_c": avg_c, "tol": 1e-7 + 1e-10 * magnitude,
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
            data, lower, rng, policy["route_weight"], policy["noise"],
            (None if deadline is None else min(deadline, time.perf_counter()
             + max(0.001, (deadline - time.perf_counter()) / max(1, H - t))))
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
        if abs(supplier) <= 1e-10:
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
            if abs(customer_inv[i]) <= 1e-10:
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
    if sum(amounts[i] for i in items) > M * Q + data["tol"]:
        return None
    now = time.perf_counter()
    if deadline is None:
        deadline = now + max(0.001, data.get("budget", 1.0) ** 0.5 * 0.02)
    random_deadline = now + 0.20 * max(0.0, deadline - now)
    first = True
    while first or time.perf_counter() < random_deadline:
        if first:
            order = sorted(items, key=lambda i: (-amounts[i], i))
        else:
            strength = 0.03 + 0.20 * rng.random()
            order = sorted(items, key=lambda i: -(amounts[i] + Q * strength * rng.random()))
        groups = [[] for _ in range(M)]
        loads = [0.0] * M
        feasible = True
        for i in order:
            choices = []
            for k in range(M):
                if loads[k] + amounts[i] <= Q + data["tol"]:
                    residual = Q - loads[k] - amounts[i]
                    score = residual
                    if not first:
                        cluster = _cluster_distance(data, groups[k], i)
                        score += (route_weight * Q * cluster / max(data["avg_c"], _EPS)
                                  + noise * Q * rng.random())
                    choices.append((score, k))
            if not choices:
                feasible = False
                break
            k = min(choices)[1]
            groups[k].append(i)
            loads[k] += amounts[i]
        if feasible:
            return groups
        first = False
    if time.perf_counter() >= deadline:
        return None
    return _binpack_dfs(data, items, amounts, deadline)


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
                    _cheapest_sequence(data, group, deadline) for group in packed
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
            cheapest = _cheapest_sequence(data, group, deadline)
            nearest = _nearest_sequence(data, group, rng, False, deadline)
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


def _cheapest_sequence(data, group, deadline=None):
    remaining = list(group)
    if not remaining:
        return []
    first = min(remaining, key=lambda i: (2.0 * data["c"][0][i + 1], i))
    seq = [first]
    remaining.remove(first)
    while remaining:
        if deadline is not None and time.perf_counter() >= deadline:
            return seq + remaining
        best = None
        for i in remaining:
            if deadline is not None and time.perf_counter() >= deadline:
                return seq + remaining
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


def _nearest_sequence(data, group, rng, randomized, deadline=None):
    remaining = list(group)
    seq = []
    current = 0
    while remaining:
        if deadline is not None and time.perf_counter() >= deadline:
            return seq + remaining
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

    threshold = min(16, 7 + int(math.log1p(data.get("budget", 1.0)) + math.log1p(max(0.0, elapsed)) / 2.0))
    for k, seq in enumerate(seqs):
        if len(seq) <= threshold and (deadline is None or time.perf_counter() < deadline):
            exact = _exact_sequence(data, seq, deadline)
            if exact is not None and _sequence_cost(data, exact) < _sequence_cost(data, seq) - data["tol"]:
                seqs[k] = exact
    return seqs


def _exact_sequence(data, group, deadline):
    nodes = tuple(sorted(group))
    cached = data["exact_cache"].get(nodes)
    if cached is not None:
        return list(cached)
    n = len(nodes)
    if n <= 1:
        return list(nodes)
    # The subset dynamic programme records only shortest-path values and predecessors, and checks the wall-clock deadline.
    size = 1 << n
    costs = [[_INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    c = data["c"]
    for j, node in enumerate(nodes):
        costs[1 << j][j] = c[0][node + 1]
    for mask in range(1, size):
        if deadline is not None and (mask & 31) == 0 and time.perf_counter() > deadline:
            return None
        bits = mask
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            previous = mask ^ bit
            if previous:
                best = _INF
                best_k = -1
                scan = previous
                while scan:
                    pbit = scan & -scan
                    k = pbit.bit_length() - 1
                    value = costs[previous][k] + c[nodes[k] + 1][nodes[j] + 1]
                    if value < best:
                        best, best_k = value, k
                    scan ^= pbit
                costs[mask][j] = best
                parent[mask][j] = best_k
            bits ^= bit
    full = size - 1
    last = min(range(n), key=lambda j: costs[full][j] + c[nodes[j] + 1][0])
    order = []
    mask = full
    while last >= 0:
        order.append(nodes[last])
        previous_last = parent[mask][last]
        mask ^= 1 << last
        last = previous_last
    order.reverse()
    if len(data["exact_cache"]) < data.get("cache_limit", 4096):
        data["exact_cache"][nodes] = tuple(order)
    return order


def _savings_sequences(data, amounts, rng, randomized, deadline):
    items = [i for i, x in enumerate(amounts) if x > _EPS]
    if not items:
        return [[] for _ in range(data["M"])]
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
    fraction = 0.10 + 0.28 * elapsed / (elapsed + max(0.01, data.get("budget", 1.0))) + 0.12 * rng.random()
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
        if abs(supplier) <= 1e-10:
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
            if abs(value) <= 1e-10:
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
        [0.0 if abs(x) <= 1e-10 else float(x) for x in row]
        for row in state["inv"]
    ]
    return {
        "routes": routes,
        "deliveries": deliveries,
        "end_inventory": end_inventory,
    }




def _assemble(data, q, sequences):
    """Accept only actual delivery quantities and connected routes, never the solver's approximate inventory values."""
    inv = _inventories(data, q)
    if inv is None:
        return None
    groups = [
        [[i for i in r if q[t][i] > _EPS] for r in period]
        for t, period in enumerate(sequences)
    ]
    routes = [[_route_from_seq(r) for r in period] for period in groups]
    travel = [sum(_route_cost(r, data["c"]) for r in period) for period in routes]
    holding = _holding_cost(data, inv)
    state = {"q": q, "inv": inv, "groups": groups, "routes": routes,
             "travel_t": travel, "holding": holding,
             "cost": holding + sum(travel)}
    return state if _valid_state(data, state) else None


def _insert_cost(data, seq, customer):
    node = customer + 1
    c = data["c"]
    if not seq:
        return c[0][node] + c[node][0], 0
    best = (_INF, 0)
    last = 0
    for pos in range(len(seq) + 1):
        nxt = seq[pos] + 1 if pos < len(seq) else 0
        value = c[last][node] + c[node][nxt] - c[last][nxt]
        if value < best[0]:
            best = value, pos
        last = nxt
    return best


def _quantity_neighborhood(data, state, selected, rng, deadline,
                           forbidden=None, perturb=0.0, quantity_prices=None, guide=None):
    """Full-horizon inventory-flow model; visits of selected customers and vehicle assignments are also variables."""
    if not _HAS_OPT or time.perf_counter() >= deadline:
        return None
    H, N, M, Q = data["H"], data["N"], data["M"], data["Q"]
    selected = set(selected)
    if state is None:
        selected = set(range(N))
        skeleton = [[[] for _ in range(M)] for _ in range(H)]
        old_owner = [[-1] * N for _ in range(H)]
    else:
        skeleton = [
            [[i for i in r if i not in selected] for r in period]
            for period in state["groups"]
        ]
        old_owner = [[-1] * N for _ in range(H)]
        for t in range(H):
            for k, r in enumerate(state["groups"][t]):
                for i in r:
                    old_owner[t][i] = k

    # The first H*(N+1) variables are inventories, and the rows with the same indices are the corresponding inventory balances.
    count_inv = H * (N + 1)
    obj, lower, upper, integer = [], [], [], []
    rr, cc, vv = [], [], []
    row_lower, row_upper = [], []
    for t in range(H):
        for loc in range(N + 1):
            v = t * (N + 1) + loc
            cap = data["U"][loc]
            if loc:
                cap -= data["d"][loc - 1][t]
            if cap < -data["tol"]:
                return None
            obj.append(0.0 if state is None else data["h"][loc])
            lower.append(0.0)
            upper.append(max(0.0, cap))
            integer.append(0)
            rr.append(v); cc.append(v); vv.append(1.0)
            rhs = data["p"][t] if loc == 0 else -data["d"][loc - 1][t]
            if t:
                rr.append(v); cc.append(v - N - 1); vv.append(-1.0)
            else:
                rhs += data["I0"][loc]
            row_lower.append(rhs)
            row_upper.append(rhs)
    # Capacity row per vehicle per period.
    for t in range(H):
        for k in range(M):
            row_lower.append(-_INF)
            row_upper.append(Q)

    def new_row(lo, hi):
        idx = len(row_lower)
        row_lower.append(lo)
        row_upper.append(hi)
        return idx

    def add_var(cost, hi, is_integer):
        idx = len(obj)
        obj.append(cost)
        lower.append(0.0)
        upper.append(hi)
        integer.append(is_integer)
        return idx

    records = []
    activation = {}
    option_width = max(1, int(1.0 + math.log1p(data["budget"])))
    for t in range(H):
        if time.perf_counter() >= deadline:
            return None
        for i in range(N):
            unavoidable = max(0.0, data["I0"][i + 1] - data["prefix"][i][t])
            bound = max(0.0, min(Q, data["U"][i + 1] - unavoidable))
            if bound <= _EPS:
                continue
            if i in selected:
                options = []
                empty_seen = False
                for k in range(M):
                    if forbidden == (t, k):
                        continue
                    if not skeleton[t][k]:
                        if state is not None and empty_seen and old_owner[t][i] != k:
                            continue
                        empty_seen = True
                    cost, _ = _insert_cost(data, skeleton[t][k], i)
                    options.append((cost, k))
                options.sort()
                if state is not None and len(options) > option_width:
                    keep = options[:option_width]
                    for entry in options[option_width:]:
                        if entry[1] == old_owner[t][i]:
                            keep.append(entry)
                    options = keep
                visit_row = new_row(-_INF, 1.0)
            else:
                k = old_owner[t][i]
                options = [] if k < 0 or forbidden == (t, k) else [(0.0, k)]
                visit_row = -1
            for cost, k in options:
                qidx = add_var(0.0 if quantity_prices is None else quantity_prices[t][i], bound, 0)
                records.append((qidx, t, k, i))
                rr.extend((t * (N + 1), t * (N + 1) + i + 1,
                           count_inv + t * M + k))
                cc.extend((qidx, qidx, qidx))
                vv.extend((1.0, -1.0, 1.0))
                if visit_row >= 0:
                    # The first feasibility pass ignores routing; afterwards the true increment of inserted edges serves as a proxy.
                    if state is None:
                        price = 0.0
                    else:
                        price = cost * math.exp(perturb * (2.0 * rng.random() - 1.0))
                        if guide is not None:
                            price *= 0.90 if guide[t][i] > _EPS else 1.08
                    # Empty routes share one activation cost, so jointly served customers do not each pay a round trip.
                    key = (t, k)
                    if state is not None and not skeleton[t][k]:
                        if key not in activation:
                            anchor = min(selected, key=lambda j: data["c"][0][j + 1])
                            charge = max(0.0, 2.0 * data["c"][0][anchor + 1])
                            activation[key] = add_var(charge, 1.0, 1), charge
                        zidx, charge = activation[key]
                        price = max(0.0, price - charge)
                    else:
                        zidx = -1
                    yidx = add_var(price, 1.0, 1)
                    if zidx >= 0:
                        link_active = new_row(-_INF, 0.0)
                        rr.extend((link_active, link_active))
                        cc.extend((yidx, zidx)); vv.extend((1.0, -1.0))
                    rr.append(visit_row); cc.append(yidx); vv.append(1.0)
                    link = new_row(-_INF, 0.0)
                    rr.extend((link, link)); cc.extend((qidx, yidx))
                    vv.extend((1.0, -bound))

    if state is None:
        # Break the symmetry of identical vehicles by decreasing load, without changing the actual plans in the feasible region.
        for t in range(H):
            rows = [new_row(0.0, _INF) for _ in range(max(0, M - 1))]
            for qidx, tt, k, i in records:
                if tt != t:
                    continue
                if k < M - 1:
                    rr.append(rows[k]); cc.append(qidx); vv.append(1.0)
                if k > 0:
                    rr.append(rows[k - 1]); cc.append(qidx); vv.append(-1.0)
    left = deadline - time.perf_counter()
    if left <= 0.002:
        return None
    matrix = csc_matrix((np.asarray(vv), (np.asarray(rr, dtype=np.int32),
                         np.asarray(cc, dtype=np.int32))),
                        shape=(len(row_lower), len(obj)))
    opts = {"time_limit": max(0.001, 0.93 * (deadline - time.perf_counter())),
            "presolve": True, "mip_rel_gap": 0.001 / math.sqrt(1.0 + data["budget"])}
    result = milp(np.asarray(obj), integrality=np.asarray(integer, dtype=np.uint8),
                  bounds=Bounds(np.asarray(lower), np.asarray(upper)),
                  constraints=LinearConstraint(matrix, np.asarray(row_lower),
                                               np.asarray(row_upper)),
                  options=opts)
    if result.x is None:
        return None
    q = [[0.0] * N for _ in range(H)]
    owners = [[-1] * N for _ in range(H)]
    for qidx, t, k, i in records:
        amount = float(result.x[qidx])
        if not math.isfinite(amount) or amount < -data["tol"]:
            return None
        if amount > _EPS:
            if owners[t][i] >= 0 and owners[t][i] != k:
                return None
            owners[t][i] = k
            q[t][i] += amount
    sequences = [
        [[i for i in r if q[t][i] > _EPS] for r in period]
        for t, period in enumerate(skeleton)
    ]
    for t in range(H):
        pending = [i for i in selected if owners[t][i] >= 0]
        pending.sort(key=lambda i: -data["c"][0][i + 1])
        for i in pending:
            k = owners[t][i]
            _, position = _insert_cost(data, sequences[t][k], i)
            sequences[t][k].insert(position, i)
        for k in range(M):
            if time.perf_counter() < deadline:
                sequences[t][k] = _route_descent(data, sequences[t][k], deadline)
    return _assemble(data, q, sequences)


def _joint_selection(data, state, rng, stagnation):
    """Fine optimisation of single customers, joint optimisation of related customers and whole-route elimination share one model."""
    N, H = data["N"], data["H"]
    value = [max(_EPS, 0.02 * data["avg_c"]) for _ in range(N)]
    route_choices = []
    for t in range(H):
        for k, route in enumerate(state["routes"][t]):
            if not route:
                continue
            route_choices.append((t, k))
            for pos in range(1, len(route) - 1):
                i = route[pos] - 1
                saving = (data["c"][route[pos - 1]][i + 1]
                          + data["c"][i + 1][route[pos + 1]]
                          - data["c"][route[pos - 1]][route[pos + 1]])
                value[i] += max(0.0, saving)
    draw = rng.random() * sum(value)
    seed = N - 1
    for i, weight in enumerate(value):
        draw -= weight
        if draw <= 0.0:
            seed = i
            break
    scale = math.sqrt(max(0.0, data["budget"]))
    width = max(1, min(N, int(1.0 + scale * math.log1p(N) / (1.0 + math.log1p(H)))))
    draw = rng.random()
    if draw < 0.27 and route_choices:
        t, k = rng.choice(route_choices)
        group = list(state["groups"][t][k])
        if len(group) <= 2 * width:
            return group, (t, k) if rng.random() < 0.65 else None
    if draw < 0.66:
        return [seed], None
    count = min(N, 1 + int(width * rng.random() ** (1.0 / (1.0 + stagnation))))
    related = sorted(range(N), key=lambda i: (
        data["c"][seed + 1][i + 1]
        + data["avg_c"] * 0.30 * rng.random()))
    if seed in related:
        related.remove(seed)
    return [seed] + related[:max(0, count - 1)], None


def _polish(data, state, rng, deadline, elapsed):
    best = state
    periods = sorted(range(data["H"]), key=lambda t: -state["travel_t"][t])
    groups = list(state["groups"])
    routes = list(state["routes"])
    travel = list(state["travel_t"])
    for pos, t in enumerate(periods):
        now = time.perf_counter()
        if now >= deadline:
            break
        sub = min(deadline, now + (deadline - now) / max(1, len(periods) - pos))
        result = _period_best(data, state["q"][t], groups[t], routes[t],
                              rng, sub, elapsed, False)
        if result is not None and result[2] < travel[t] - data["tol"]:
            groups[t], routes[t], travel[t] = result
    candidate = dict(state)
    candidate.update(groups=groups, routes=routes, travel_t=travel,
                     cost=state["holding"] + sum(travel))
    if _valid_state(data, candidate) and candidate["cost"] < best["cost"]:
        best = candidate
    return best


def _empty_solution(instance):
    """The degenerate no-transport case still generates every slot from the original balances."""
    H = max(0, int(instance.get("num_periods", 0)))
    N = max(0, int(instance.get("num_customers", 0)))
    M = max(0, int(instance.get("num_vehicles", 0)))
    inventories = []
    inv = [float(x) for x in instance.get("initial_inventory", [0.0] * (N + 1))]
    for t in range(H):
        inv[0] += float(instance["supplier_production"][t])
        for i in range(N):
            inv[i + 1] -= float(instance["customer_demand"][i][t])
        inventories.append(list(inv))
    return {"routes": [[[] for _ in range(M)] for _ in range(H)],
            "deliveries": [[[0.0] * N for _ in range(M)] for _ in range(H)],
            "end_inventory": inventories}



def _init_patterns(data):
    """Build prefix sums of spatial phase and holding cost over replenishment intervals."""
    N, H, M = data["N"], data["H"], data["M"]
    c = data["c"]
    nearest = [min((c[i + 1][j + 1] for j in range(N) if i != j),
                   default=c[0][i + 1]) for i in range(N)]
    unused = set(range(N))
    order = []
    node = 0
    while unused:
        i = min(unused, key=lambda j: c[node][j + 1])
        order.append(i)
        unused.remove(i)
        node = i + 1
    block = max(1, int(round(N / max(2.0, 2.0 * M))))
    color = [0] * N
    for pos, i in enumerate(order):
        color[i] = (pos // block) & 1
    data["color"] = color
    data["visit_proxy"] = [max(_EPS, 1.15 * nearest[i]
        + 2.0 * c[0][i + 1] / max(3.0, N / max(1.0, 2.0 * M)))
        for i in range(N)]
    data["prefix_sum"] = []
    for pref in data["prefix"]:
        acc = [0.0]
        for t in range(H):
            acc.append(acc[-1] + pref[t + 1])
        data["prefix_sum"].append(acc)
    width = 3 + 2 * int(math.log1p(data["budget"]) / 2.0)
    scales = [0.30 * math.exp(3.0 * (j / (width - 1) - 0.5))
              for j in range(width)]
    scales.sort(key=lambda x: abs(math.log(x / 0.30)))
    data["price_banks"] = [{"scale": x, "prices": [0.0] * H, "age": 0}
                            for x in scales]
    data["pattern_turn"] = 0
    data["cache_limit"] = max(64, int(256 * (1.0 + math.sqrt(data["budget"]))
                                    * math.sqrt(max(1, N))))


def _lot_plan(data, i, fixed, prices, deadline):
    """Dynamic programming over zero-inventory regeneration intervals; the initial surplus is handled separately."""
    H, Q = data["H"], data["Q"]
    pref, acc = data["prefix"][i], data["prefix_sum"][i]
    initial, capacity = data["I0"][i + 1], data["U"][i + 1]
    dh = data["h"][i + 1] - data["h"][0]
    dp, nxt = [_INF] * (H + 1), [-1] * H
    dp[H] = 0.0
    for a in range(H - 1, -1, -1):
        if time.perf_counter() >= deadline:
            return None
        if data["d"][i][a] <= _EPS:
            dp[a], nxt[a] = dp[a + 1], a + 1
        for b in range(a + 1, H + 1):
            amount = pref[b] - pref[a]
            if amount > min(Q, capacity) + data["tol"]:
                break
            if amount <= _EPS:
                continue
            holding = (b - a) * pref[b] - acc[b] + acc[a]
            value = fixed[a] + prices[a] * amount + dh * holding + dp[b]
            if value < dp[a]:
                dp[a], nxt[a] = value, b
    first = next((t for t in range(H) if pref[t + 1] > initial + data["tol"]), H)
    if first == H:
        return [0.0] * H
    best, start = _INF, None
    for a in range(first + 1):
        if time.perf_counter() >= deadline:
            return None
        prior_holding = a * initial - acc[a]
        for b in range(a + 1, H + 1):
            amount = pref[b] - initial
            pre_level = pref[b] - pref[a]
            if pre_level > capacity + data["tol"] or amount > Q + data["tol"]:
                break
            if amount <= data["tol"]:
                continue
            holding = prior_holding + (b - a) * pref[b] - acc[b] + acc[a]
            value = fixed[a] + prices[a] * amount + dh * holding + dp[b]
            if value < best:
                best, start = value, (a, b, amount)
    if start is None:
        # When single-period demand exceeds one vehicle's capacity, keep the originally feasible visits in the joint inventory-flow model.
        return None
    q = [0.0] * H
    a, b, amount = start
    q[a] = amount
    a = b
    while a < H:
        b = nxt[a]
        if b <= a:
            return None
        q[a] = max(0.0, pref[b] - pref[a])
        a = b
    return q


def _pattern_guide(data, state, selected, rng, deadline):
    """Route marginal costs enter the dynamic programme, and resource violations update cross-period prices in return."""
    H, N, Q = data["H"], data["N"], data["Q"]
    banks = data["price_banks"]
    turn = data["pattern_turn"]
    bank = banks[turn % len(banks)] if turn < len(banks) else rng.choice(banks)
    data["pattern_turn"] += 1
    scale = bank["scale"] * math.exp(rng.uniform(-0.12, 0.12))
    phase = min(0.40, 0.07 + 0.09 * math.sqrt(scale))
    flip = rng.randrange(2)
    route_feedback = 0.50 * rng.random()
    guide = [list(row) for row in state["q"]]
    charges = [[0.0] * N for _ in range(H)]
    c = data["c"]
    for i in selected:
        if time.perf_counter() >= deadline:
            return None, None
        proxy = data["visit_proxy"][i]
        fixed = []
        node = i + 1
        jitter = math.exp(rng.uniform(-0.05, 0.05))
        for t in range(H):
            marginal = _INF
            for route in state["routes"][t]:
                if node in route:
                    pos = route.index(node)
                    marginal = (c[route[pos - 1]][node] + c[node][route[pos + 1]]
                                - c[route[pos - 1]][route[pos + 1]])
                    break
                if not route:
                    marginal = min(marginal, 2.0 * c[0][node])
                else:
                    marginal = min(marginal, min(
                        c[a][node] + c[node][b] - c[a][b]
                        for a, b in zip(route, route[1:])))
            marginal = max(0.0, marginal) if math.isfinite(marginal) else proxy
            value = ((1.0 - route_feedback) * proxy + route_feedback * marginal)
            value *= scale * jitter
            value *= 1.0 - phase if (t & 1) == (data["color"][i] ^ flip) else 1.0 + 0.55 * phase
            fixed.append(value)
            charges[t][i] = max(0.025 * proxy * scale, value)
        plan = _lot_plan(data, i, fixed, bank["prices"], deadline)
        if plan is not None:
            for t in range(H):
                guide[t][i] = plan[t]

    loads = [sum(row) for row in guide]
    unit = max(_EPS, data["avg_c"] / max(Q, _EPS))
    step = 0.20 * unit / math.sqrt(1.0 + 0.10 * bank["age"])
    bank["age"] += 1
    prices = bank["prices"]
    supplier = data["I0"][0]
    for t in range(H):
        overload = max(0.0, loads[t] - data["M"] * Q) / Q
        prices[t] = 0.97 * prices[t] + 3.2 * step * overload
        supplier += data["p"][t] - loads[t]
        imbalance = (max(0.0, -supplier) - max(0.0, supplier - data["U"][0])) / Q
        if abs(imbalance) > _EPS:
            change = step * max(-2.0, min(2.0, imbalance)) / (t + 1)
            for u in range(t + 1):
                prices[u] += change
    for t in range(H):
        prices[t] = max(-5.0 * unit, min(5.0 * unit, prices[t]))
    return guide, charges


def _pattern_fusion(data, state, rng, deadline, elapsed):
    """Project batched replenishment patterns onto the feasible region of shared inventory, vehicle capacity and visit assignment."""
    now = time.perf_counter()
    guide, charges = _pattern_guide(data, state, range(data["N"]), rng,
                                   now + 0.25 * max(0.0, deadline - now))
    if guide is None:
        return None
    H, N, M, Q = data["H"], data["N"], data["M"], data["Q"]
    candidate = None
    # Keep the visit support proposed by the dynamic programme first; the inventory-flow model may adjust each replenishment quantity.
    groups = []
    strict_end = time.perf_counter() + 0.40 * max(0.0, deadline - time.perf_counter())
    for t in range(H):
        if time.perf_counter() >= strict_end or sum(guide[t]) > M * Q + data["tol"]:
            break
        period = _savings_sequences(data, guide[t], rng, False, strict_end)
        if period is None:
            now = time.perf_counter()
            period = _pack_amounts(data, guide[t], rng, 0.45, 0.05,
                min(strict_end, now + (strict_end - now) / max(1, H - t)))
        if period is None:
            break
        groups.append(period)
    if len(groups) == H:
        candidate = _assemble(data, guide, groups)
        if _HAS_OPT and time.perf_counter() < strict_end:
            refined = _quantity_neighborhood(data, {"groups": groups}, [], rng, strict_end)
            if refined is not None and (candidate is None or refined["cost"] < candidate["cost"]):
                candidate = refined
        if candidate is not None and candidate["cost"] < state["cost"] - data["tol"]:
            return _polish(data, candidate, rng, deadline, elapsed)
    if not _HAS_OPT or time.perf_counter() >= deadline:
        return candidate

    # Add back all visits of the incumbent so the projected linear model still contains the current feasible solution.
    # New time points are inserted into existing routes; actual loads are reassigned by the inventory-flow model rather than greedily truncated.
    groups = [[list(r) for r in period] for period in state["groups"]]
    prices = [[0.0] * N for _ in range(H)]
    for t in range(H):
        if time.perf_counter() >= deadline:
            return candidate
        owner = {i: k for k, r in enumerate(groups[t]) for i in r}
        predicted = [sum(guide[t][i] for i in r) for r in groups[t]]
        pending = [i for i in range(N) if guide[t][i] > _EPS and i not in owner]
        pending.sort(key=lambda i: (-guide[t][i], -data["c"][0][i + 1]))
        for i in pending:
            options = []
            for k in range(M):
                cost, pos = _insert_cost(data, groups[t][k], i)
                congestion = max(0.0, predicted[k] + guide[t][i] - Q) / Q
                score = cost + 4.0 * data["avg_c"] * congestion * congestion
                options.append((score, k, pos))
            _, k, pos = min(options)
            groups[t][k].insert(pos, i)
            predicted[k] += guide[t][i]
        for i in range(N):
            typical = max(state["q"][t][i], data["suffix"][i][0] / max(1, H))
            denominator = max(guide[t][i], 0.15 * typical, Q * 1e-8)
            prices[t][i] = charges[t][i] / denominator
    now = time.perf_counter()
    projected = _quantity_neighborhood(data, {"groups": groups}, [], rng,
                now + 0.72 * max(0.0, deadline - now), quantity_prices=prices)
    if projected is not None:
        now = time.perf_counter()
        # Solve once more without proxy prices to remove unnecessary end-of-horizon inventory caused by pattern guidance.
        cleaned = _quantity_neighborhood(data, projected, [], rng,
                   now + 0.50 * max(0.0, deadline - now))
        if cleaned is not None and cleaned["cost"] < projected["cost"]:
            projected = cleaned
        projected = _polish(data, projected, rng, deadline, elapsed)
        if candidate is None or projected["cost"] < candidate["cost"]:
            candidate = projected
    return candidate


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    best, data = None, None
    try:
        budget = float(time_limit_s)
        budget = max(0.0, budget) if math.isfinite(budget) else 0.0
        if (int(instance["num_periods"]) == 0 or int(instance["num_customers"]) == 0
                or int(instance["num_vehicles"]) == 0
                or float(instance["vehicle_capacity"]) <= 0.0):
            return _empty_solution(instance)
        data = _prepare(instance)
        data["budget"] = budget
        rng = random.Random(_seed(data))
        reserve = min(0.10 * budget, 0.012 + 0.006 * budget)
        deadline = started + max(0.0, budget - reserve)
        policies = [
            {"kind": "jit", "aggression": 0.0, "span": 1.0,
             "route_weight": 0.40, "noise": 0.0},
            {"kind": "buffer", "aggression": 1.0, "span": 1.0,
             "route_weight": 0.35, "noise": 0.0},
            {"kind": "economic", "aggression": 0.72, "span": 1.0,
             "route_weight": 0.40, "noise": 0.0},
        ]
        construct_end = min(deadline, started + 0.09 * budget)
        cursor = 0
        first = True
        while first or (best is None and time.perf_counter() < construct_end):
            first = False
            policy = policies[cursor] if cursor < len(policies) else _random_policy(data, rng, 0.0)
            cursor += 1
            raw = _construct(data, rng, policy, construct_end if budget > 0 else None)
            if raw is not None:
                best = _assemble(data, raw["q"], raw["groups"])
        if best is None and _HAS_OPT and time.perf_counter() < deadline:
            best = _quantity_neighborhood(data, None, range(data["N"]), rng, deadline)
        while best is None and time.perf_counter() < deadline:
            raw = _construct(data, rng, _random_policy(data, rng, 0.0), deadline)
            if raw is not None:
                best = _assemble(data, raw["q"], raw["groups"])
        if best is None:
            return _empty_solution(instance)
        if time.perf_counter() >= deadline:
            return _emit(data, best)
        _init_patterns(data)
        now = time.perf_counter()
        try:
            candidate = _quantity_neighborhood(data, best, [], rng,
                min(deadline, now + 0.05 * budget))
            if candidate is not None and candidate["cost"] < best["cost"]:
                best = candidate
        except Exception:
            pass
        now = time.perf_counter()
        best = _polish(data, best, rng, min(deadline, now + 0.075 * budget), now - started)
        current = best
        last_improvement = last_lp = time.perf_counter()
        credit = [1.0] * 6
        complexity = max(1, data["N"] * data["H"])
        # All search stops at the wall clock; more time enlarges joint customer blocks, perturbations and exact path depth together.
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            elapsed, remaining = now - started, deadline - now
            stagnation = (now - last_improvement) / max(0.001, budget)
            base = best if rng.random() < 0.78 else current
            weights = [0.44 * credit[0], 0.27 * credit[1], 0.15 * credit[2],
                       0.08 * credit[3], 0.03 * credit[4], 0.03 * credit[5]]
            if not _HAS_OPT:
                weights[0] = 0.0
            if data["pattern_turn"] < len(data["price_banks"]) and elapsed < 0.32 * budget:
                operator = 1
            else:
                operator = rng.choices(range(6), weights=weights)[0]
            candidate = None
            local = min(deadline, now + max(0.001, math.sqrt(budget) * 0.018
                                           + complexity * 0.000003))
            try:
                if operator == 0:
                    selected, forbidden = _joint_selection(data, base, rng, stagnation)
                    duration = budget ** 0.55 * (0.037 + 0.008 * math.sqrt(len(selected)))
                    duration += complexity * 0.000004
                    local = min(deadline, now + min(0.72 * remaining, duration))
                    guide = None
                    if len(selected) > 1:
                        guide, _ = _pattern_guide(data, base, selected, rng,
                            now + 0.10 * max(0.0, local - now))
                    candidate = _quantity_neighborhood(data, base, selected, rng, local,
                        forbidden, 0.10 * rng.random() if rng.random() < 0.20 else 0.0,
                        guide=guide)
                elif operator == 1:
                    local = min(deadline, now + min(0.65 * remaining,
                        budget ** 0.55 * 0.12 + complexity * 0.000006))
                    candidate = _pattern_fusion(data, base, rng, local, elapsed)
                elif operator == 2:
                    candidate = _routing_neighbor(data, base, rng, local, elapsed)
                elif operator == 3:
                    candidate = _temporal_neighbor(data, base, rng, local, elapsed)
                elif operator == 4:
                    raw = _construct(data, rng, _random_policy(data, rng, elapsed), local)
                    if raw is not None:
                        candidate = _assemble(data, raw["q"], raw["groups"])
                        if candidate is not None:
                            candidate = _polish(data, candidate, rng, local, elapsed)
                else:
                    candidate = _route_elimination_neighbor(data, base, rng, local, elapsed)
            except Exception:
                candidate = None
            if candidate is None or not _valid_state(data, candidate):
                credit[operator] = 0.94 * credit[operator] + 0.06
                continue
            if (operator not in (0, 1) and _HAS_OPT
                    and time.perf_counter() - last_lp > 0.065 * budget
                    and candidate["cost"] < best["cost"] + 0.10 * abs(best["cost"])):
                last_lp = time.perf_counter()
                try:
                    refined = _quantity_neighborhood(data, candidate, [], rng,
                        min(deadline, last_lp + math.sqrt(budget) * 0.025))
                    if refined is not None and refined["cost"] < candidate["cost"]:
                        candidate = refined
                except Exception:
                    pass
            gain = max(0.0, best["cost"] - candidate["cost"])
            if gain > data["tol"]:
                best = candidate
                last_improvement = time.perf_counter()
                best = _polish(data, best, rng,
                    min(deadline, last_improvement + math.sqrt(budget) * 0.010), elapsed)
                candidate = best
            credit[operator] = 0.92 * credit[operator] + 0.08 * (
                1.0 + min(5.0, gain / max(_EPS, 0.20 * data["avg_c"])))
            delta = candidate["cost"] - current["cost"]
            phase = min(1.0, (time.perf_counter() - started) / max(0.001, budget))
            temperature = data["avg_c"] * (0.01 + 0.11 * (1.0 - phase) ** 2
                                          + 0.035 * min(1.0, stagnation))
            if delta <= data["tol"] or rng.random() < math.exp(-max(0.0, delta) / max(_EPS, temperature)):
                current = candidate
            elif rng.random() < 0.08:
                current = best
        return _emit(data, best)
    except BaseException:
        if best is not None and data is not None:
            return _emit(data, best)
        try:
            return _empty_solution(instance)
        except BaseException:
            return {"routes": [], "deliveries": [], "end_inventory": []}
