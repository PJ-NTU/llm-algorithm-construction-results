import math
import random
import time


_EPS = 1e-9
_INF = float("inf")


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime heuristic for the finite-horizon production inventory routing problem."""
    t0 = time.time()
    try:
        try:
            budget = float(time_limit_s)
        except BaseException:
            budget = 0.0
        if not math.isfinite(budget) or budget < 0.0:
            budget = 0.0

        # The examination budget is 10 seconds.  Keeping the fraction here also
        # makes the function polite when it is called with a smaller test budget.
        deadline = t0 + 0.85 * budget
        data = _prepare(instance)
        rng = random.Random(_seed(data))

        q0 = _baseline_schedule(data)
        baseline = _make_state(data, q0, rng, None, False) if q0 is not None else None
        if baseline is None:
            return _last_resort(data)

        best = baseline
        current = baseline
        # High fixed-charge regimes are tried first; gentler batching follows
        # and is especially useful on cheap, dense travel matrices.
        scale_regimes = [0.30, 0.55, 0.15, 0.90, 0.07]
        phase_regimes = [0.14, 0.22, 0.08, 0.30, 0.04]
        price_banks = [[0.0] * data["H"] for _ in scale_regimes]
        best_prices = [0.0] * data["H"]
        iteration = 0
        stagnation = 0

        # There is deliberately no iteration cap: all remaining wall-clock time
        # is converted into temporal restarts or routing improvements.
        while time.time() - t0 < 0.85 * budget:
            if time.time() > deadline:
                break
            now = time.time()
            local_deadline = min(deadline, now + 0.035 + 0.000012 * data["N"] ** 2)
            candidate = None

            try:
                if iteration % 5 == 4:
                    base = best if rng.random() < 0.82 else current
                    candidate = _reroute_state(data, base, rng, local_deadline)
                elif iteration % 3 == 2:
                    base = best if rng.random() < 0.75 else current
                    candidate = _temporal_neighbor(data, base, rng, local_deadline)
                else:
                    regime = (iteration // 30) % len(scale_regimes)
                    prices = price_banks[regime]
                    if stagnation > 28 and rng.random() < 0.22:
                        for t in range(data["H"]):
                            prices[t] = 0.65 * prices[t] + rng.uniform(-0.05, 0.05)
                        stagnation = 0
                    fixed_scale = scale_regimes[regime] * math.exp(rng.uniform(-0.035, 0.035))
                    phase_strength = phase_regimes[regime]
                    q = _pattern_schedule(
                        data,
                        prices,
                        rng,
                        fixed_scale,
                        phase_strength,
                        iteration,
                        local_deadline,
                    )
                    if q is not None:
                        loads = [sum(row) for row in q]
                        _update_prices(data, prices, loads, q, iteration)
                        q = _repair_schedule(data, q, local_deadline)
                        if q is not None and time.time() <= local_deadline:
                            candidate = _make_state(
                                data, q, rng, local_deadline, iteration > 1
                            )
            except BaseException:
                candidate = None

            iteration += 1
            if candidate is None:
                stagnation += 1
                continue

            if candidate["cost"] < best["cost"] - data["tol"]:
                best = candidate
                best_prices = list(prices)
                stagnation = 0
            else:
                stagnation += 1

            delta = candidate["cost"] - current["cost"]
            elapsed = max(0.0, time.time() - t0)
            temperature = data["avg_c"] * (0.08 + 0.38 / (1.0 + elapsed))
            if delta <= data["tol"]:
                current = candidate
            elif rng.random() < math.exp(-delta / max(_EPS, temperature)):
                current = candidate
            elif rng.random() < 0.025:
                current = best

        if not _valid_state(data, best):
            best = baseline
        return _emit(data, best)
    except BaseException:
        try:
            data = _prepare(instance)
            q = _baseline_schedule(data)
            state = _make_state(data, q, random.Random(0), None, False)
            if state is not None:
                return _emit(data, state)
            return _last_resort(data)
        except BaseException:
            return _structural_fallback(instance)


def _prepare(instance):
    H = int(instance["num_periods"])
    N = int(instance["num_customers"])
    M = int(instance["num_vehicles"])
    Q = float(instance["vehicle_capacity"])
    if H <= 0 or N <= 0 or M < 0 or Q <= 0.0:
        raise ValueError("invalid dimensions")

    p = [float(instance["supplier_production"][t]) for t in range(H)]
    d = [
        [float(instance["customer_demand"][i][t]) for t in range(H)]
        for i in range(N)
    ]
    U = [float(x) for x in instance["inventory_capacity"]]
    I0 = [float(x) for x in instance["initial_inventory"]]
    h = [float(x) for x in instance["holding_cost"]]
    c = [
        [float(instance["travel_cost"][i][j]) for j in range(N + 1)]
        for i in range(N + 1)
    ]

    prefix = [[0.0] * (H + 1) for _ in range(N)]
    for i in range(N):
        for t in range(H):
            prefix[i][t + 1] = prefix[i][t] + d[i][t]

    positive = [
        c[i][j]
        for i in range(N + 1)
        for j in range(i + 1, N + 1)
        if c[i][j] > 0.0
    ]
    avg_c = sum(positive) / len(positive) if positive else 1.0
    nearest = []
    for i in range(1, N + 1):
        values = [c[i][j] for j in range(1, N + 1) if j != i]
        nearest.append(min(values) if values else c[0][i])

    # A nearest-neighbour ordering gives stable geographical blocks without
    # assuming that coordinates are part of the input.
    spatial_order = []
    unused = set(range(N))
    node = min(unused, key=lambda i: c[0][i + 1])
    while unused:
        if node not in unused:
            node = min(unused)
        spatial_order.append(node)
        unused.remove(node)
        if unused:
            node = min(unused, key=lambda j: c[spatial_order[-1] + 1][j + 1])
    position = [0] * N
    for pos, i in enumerate(spatial_order):
        position[i] = pos
    block = max(2, int(round(N / max(2.0, 2.0 * max(1, M)))))
    color = [(position[i] // block) & 1 for i in range(N)]

    magnitude = max(
        [1.0, Q] + p + U + I0 + [x for row in d for x in row]
    )
    return {
        "H": H,
        "N": N,
        "M": M,
        "Q": Q,
        "p": p,
        "d": d,
        "U": U,
        "I0": I0,
        "h": h,
        "c": c,
        "prefix": prefix,
        "nearest": nearest,
        "color": color,
        "avg_c": avg_c,
        "tol": 1e-7 * magnitude,
        "exact_cache": {},
    }


def _seed(data):
    value = 0xCBF29CE484222325
    mask = (1 << 64) - 1
    values = [data["H"], data["N"], data["M"], int(round(data["Q"] * 1000))]
    values.extend(int(round(x * 100)) for x in data["p"])
    values.extend(int(round(data["c"][0][i] * 100)) for i in range(1, data["N"] + 1))
    for item in values:
        value ^= item & mask
        value = (value * 0x100000001B3) & mask
    return value


def _baseline_schedule(data):
    H, N, M, Q = data["H"], data["N"], data["M"], data["Q"]
    q = [[0.0] * N for _ in range(H)]
    customer_inventory = list(data["I0"][1:])
    supplier_inventory = data["I0"][0]

    for t in range(H):
        needs = [max(0.0, data["d"][i][t] - customer_inventory[i]) for i in range(N)]
        if any(x > Q + data["tol"] for x in needs):
            return None
        groups = _pack_items(needs, M, Q, None, False)
        if groups is None:
            return None
        assigned = [-1] * N
        loads = [0.0] * M
        for k, group in enumerate(groups):
            for i in group:
                assigned[i] = k
                loads[k] += needs[i]
        q[t] = list(needs)

        available = supplier_inventory + data["p"][t]
        must_ship = max(sum(needs), available - data["U"][0])
        extra = must_ship - sum(needs)
        if extra > data["tol"]:
            active = [i for i in range(N) if needs[i] > _EPS]
            inactive = [i for i in range(N) if needs[i] <= _EPS]
            active.sort(key=lambda i: (data["h"][i + 1], -data["nearest"][i]))
            inactive.sort(key=lambda i: (data["h"][i + 1], data["c"][0][i + 1]))
            for i in active + inactive:
                if extra <= data["tol"]:
                    break
                room = min(
                    data["U"][i + 1] - customer_inventory[i] - q[t][i],
                    Q - q[t][i],
                )
                if room <= data["tol"]:
                    continue
                k = assigned[i]
                if k < 0:
                    choices = [v for v in range(M) if loads[v] < Q - data["tol"]]
                    if not choices:
                        continue
                    k = max(choices, key=lambda v: Q - loads[v])
                    assigned[i] = k
                    groups[k].append(i)
                amount = min(extra, room, Q - loads[k])
                if amount > data["tol"]:
                    q[t][i] += amount
                    loads[k] += amount
                    extra -= amount
            if extra > 5.0 * data["tol"]:
                return None

        shipped = sum(q[t])
        if shipped > available + data["tol"]:
            return None
        supplier_inventory = available - shipped
        if supplier_inventory < -data["tol"] or supplier_inventory > data["U"][0] + data["tol"]:
            return None
        for i in range(N):
            pre = customer_inventory[i] + q[t][i]
            if pre > data["U"][i + 1] + data["tol"] or pre < data["d"][i][t] - data["tol"]:
                return None
            customer_inventory[i] = pre - data["d"][i][t]
    return q


def _pattern_schedule(data, prices, rng, fixed_scale, phase_strength, iteration, deadline):
    H, N = data["H"], data["N"]
    q = [[0.0] * N for _ in range(H)]
    order = list(range(N))
    rng.shuffle(order)
    phase_flip = (iteration // 90) & 1
    jitter = [rng.uniform(0.97, 1.03) for _ in range(N)]

    for count, i in enumerate(order):
        if time.time() > deadline:
            return None
        plan = _customer_plan(
            data,
            i,
            prices,
            fixed_scale * jitter[i],
            phase_strength,
            phase_flip,
            iteration,
            deadline,
        )
        if plan is None:
            return None
        for t, amount in plan:
            q[t][i] = amount
        if (count & 7) == 7 and time.time() > deadline:
            return None
    return q


def _customer_plan(data, i, prices, scale, phase_strength, phase_flip, iteration, deadline):
    H = data["H"]
    pref = data["prefix"][i]
    initial = data["I0"][i + 1]
    capacity = data["U"][i + 1]
    Q = data["Q"]
    effective_h = data["h"][i + 1] - data["h"][0]
    route_charge = (
        1.15 * data["nearest"][i]
        + 2.0 * data["c"][0][i + 1] / max(3.0, data["N"] / max(1.0, 2.0 * max(1, data["M"])))
    ) * scale
    preferred = data["color"][i] ^ phase_flip

    def visit_cost(t, amount):
        parity_factor = 1.0 - phase_strength if (t & 1) == preferred else 1.0 + 0.55 * phase_strength
        return route_charge * parity_factor + prices[t] * amount

    # dp[a] is the best tail when the customer is empty at the start of a.
    dp = [_INF] * (H + 1)
    nxt = [-1] * H
    dp[H] = 0.0
    for a in range(H - 1, -1, -1):
        if time.time() > deadline:
            return None
        best_value = _INF
        best_b = -1
        for b in range(a + 1, H + 1):
            amount = pref[b] - pref[a]
            if amount > Q + data["tol"] or amount > capacity + data["tol"]:
                break
            holding = 0.0
            for u in range(a, b):
                holding += pref[b] - pref[u + 1]
            value = visit_cost(a, amount) + effective_h * holding + dp[b]
            if value < best_value:
                best_value = value
                best_b = b
        dp[a] = best_value
        nxt[a] = best_b

    first_shortage = H
    for t in range(H):
        if pref[t + 1] > initial + data["tol"]:
            first_shortage = t
            break
    if first_shortage == H:
        return []

    best_value = _INF
    best_first = None
    for f in range(first_shortage + 1):
        remaining = initial - pref[f]
        if remaining < -data["tol"]:
            continue
        prior_holding = 0.0
        for u in range(f):
            prior_holding += max(0.0, initial - pref[u + 1])
        for b in range(f + 1, H + 1):
            pre_level = pref[b] - pref[f]
            amount = pre_level - remaining
            if pre_level > capacity + data["tol"] or amount > Q + data["tol"]:
                break
            if amount <= data["tol"]:
                continue
            holding = prior_holding
            for u in range(f, b):
                holding += pref[b] - pref[u + 1]
            value = visit_cost(f, amount) + effective_h * holding + dp[b]
            if value < best_value:
                best_value = value
                best_first = (f, b, amount)

    if best_first is None:
        return None
    f, b, amount = best_first
    result = [(f, max(0.0, amount))]
    a = b
    while a < H:
        b = nxt[a]
        if b <= a:
            return None
        result.append((a, pref[b] - pref[a]))
        a = b
    return result


def _update_prices(data, prices, loads, q, iteration):
    H = data["H"]
    step = 0.22 / math.sqrt(1.0 + 0.08 * iteration)
    for t in range(H):
        capacity_excess = max(0.0, loads[t] - data["M"] * data["Q"]) / max(data["Q"], 1.0)
        # Penalise only an actual fleet violation.  Pulling every period toward
        # the arithmetic mean needlessly moves stock from the cheap supplier to
        # customers and was especially harmful on tight, dense instances.
        prices[t] = 0.96 * prices[t] + 3.2 * step * capacity_excess
        prices[t] = max(-1.5, min(2.5, prices[t]))

    inv = _inventories(data, q)
    if inv is not None:
        for t in range(H):
            overflow = inv[t][0] - data["U"][0]
            shortage = -inv[t][0]
            if overflow > data["tol"]:
                change = step * min(0.8, overflow / max(data["Q"], 1.0))
                for r in range(t + 1):
                    prices[r] = max(-1.5, prices[r] - change / (t + 1))
            elif shortage > data["tol"]:
                change = step * min(0.8, shortage / max(data["Q"], 1.0))
                for r in range(t + 1):
                    prices[r] = min(1.5, prices[r] + change / (t + 1))


def _repair_schedule(data, q, deadline):
    H, N = data["H"], data["N"]
    q = [list(row) for row in q]
    for t in range(H):
        if sum(q[t]) > data["M"] * data["Q"] + data["tol"]:
            return None

    inv = _inventories(data, q, allow_supplier=True)
    if inv is None:
        return None

    # Supplier overflow is repaired first by advancing already planned loads.
    for t in range(H):
        if time.time() > deadline:
            return None
        guard = 0
        while inv[t][0] > data["U"][0] + data["tol"]:
            if time.time() > deadline:
                return None
            guard += 1
            if guard > 2 * N + 4:
                return None
            need = inv[t][0] - data["U"][0]
            moved = False
            period_room = data["M"] * data["Q"] - sum(q[t])
            if period_room > data["tol"]:
                future = []
                for s in range(t + 1, H):
                    for i in range(N):
                        if q[s][i] > data["tol"]:
                            room = min(
                                data["U"][i + 1] - data["d"][i][u] - inv[u][i + 1]
                                for u in range(t, s)
                            )
                            limit = min(
                                q[s][i],
                                room,
                                data["Q"] - q[t][i],
                                period_room,
                            )
                            if limit > data["tol"]:
                                bonus = 0 if q[t][i] > _EPS else data["nearest"][i]
                                future.append((bonus + 0.01 * (s - t), s, i, limit))
                future.sort()
                for _, s, i, limit in future:
                    amount = min(need, limit)
                    if amount <= data["tol"]:
                        continue
                    q[s][i] -= amount
                    q[t][i] += amount
                    moved = True
                    break
            if not moved:
                # Keeping additional stock at a customer is sometimes necessary
                # because total system inventory exceeds supplier capacity.
                candidates = []
                for i in range(N):
                    room = min(
                        data["U"][i + 1] - data["d"][i][u] - inv[u][i + 1]
                        for u in range(t, H)
                    )
                    limit = min(
                        room,
                        data["Q"] - q[t][i],
                        data["M"] * data["Q"] - sum(q[t]),
                    )
                    if limit > data["tol"]:
                        insertion = 0.0 if q[t][i] > _EPS else data["nearest"][i]
                        candidates.append((data["h"][i + 1] + insertion, i, limit))
                if not candidates:
                    return None
                candidates.sort()
                _, i, limit = candidates[0]
                amount = min(need, limit)
                q[t][i] += amount
                moved = True
            inv = _inventories(data, q, allow_supplier=True)
            if inv is None or not moved:
                return None

    # A negative supplier inventory means that a load was sent too early.  Move
    # such load later while retaining enough customer stock before the move.
    for t in range(H):
        if inv[t][0] >= -data["tol"]:
            continue
        need = -inv[t][0]
        moved = False
        for s in range(t, -1, -1):
            for i in range(N):
                if q[s][i] <= data["tol"]:
                    continue
                for r in range(t + 1, H):
                    room_later = data["M"] * data["Q"] - sum(q[r])
                    if room_later <= data["tol"] or q[r][i] >= data["Q"] - data["tol"]:
                        continue
                    stock = min(inv[u][i + 1] for u in range(s, r))
                    amount = min(need, q[s][i], stock, room_later, data["Q"] - q[r][i])
                    if amount > data["tol"]:
                        q[s][i] -= amount
                        q[r][i] += amount
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break
        if not moved:
            return None
        inv = _inventories(data, q, allow_supplier=True)
        if inv is None:
            return None

    inv = _inventories(data, q)
    if inv is None:
        return None
    for t in range(H):
        if _pack_items(q[t], data["M"], data["Q"], None, False) is None:
            return None
    return q


def _inventories(data, q, allow_supplier=False):
    H, N = data["H"], data["N"]
    if q is None or len(q) != H:
        return None
    previous = list(data["I0"])
    result = []
    for t in range(H):
        if len(q[t]) != N:
            return None
        shipped = 0.0
        row = [0.0] * (N + 1)
        for i in range(N):
            amount = q[t][i]
            if amount < -data["tol"] or amount > data["Q"] + data["tol"]:
                return None
            pre = previous[i + 1] + amount
            if pre < data["d"][i][t] - data["tol"]:
                return None
            if pre > data["U"][i + 1] + data["tol"]:
                return None
            row[i + 1] = pre - data["d"][i][t]
            if abs(row[i + 1]) <= data["tol"]:
                row[i + 1] = 0.0
            shipped += amount
        row[0] = previous[0] + data["p"][t] - shipped
        if not allow_supplier and (
            row[0] < -data["tol"] or row[0] > data["U"][0] + data["tol"]
        ):
            return None
        previous = row
        result.append(row)
    return result


def _make_state(data, q, rng, deadline, randomize):
    if q is None:
        return None
    inv = _inventories(data, q)
    if inv is None:
        return None
    groups = []
    routes = []
    travel_t = []
    for t in range(data["H"]):
        if deadline is not None and time.time() > deadline:
            return None
        result = _route_period(data, q[t], rng, deadline, randomize)
        if result is None:
            return None
        period_groups, period_routes, cost = result
        groups.append(period_groups)
        routes.append(period_routes)
        travel_t.append(cost)
    holding = sum(
        data["h"][loc] * inv[t][loc]
        for t in range(data["H"])
        for loc in range(data["N"] + 1)
    )
    return {
        "q": [list(row) for row in q],
        "inv": inv,
        "groups": groups,
        "routes": routes,
        "travel_t": travel_t,
        "holding": holding,
        "cost": holding + sum(travel_t),
    }


def _pack_items(amounts, M, Q, rng, randomize):
    active = [i for i, amount in enumerate(amounts) if amount > _EPS]
    if not active:
        return [[] for _ in range(M)]
    if M <= 0 or any(amounts[i] > Q + 1e-7 for i in active):
        return None
    attempts = 5 if randomize and rng is not None else 1
    best = None
    best_spread = _INF
    for attempt in range(attempts):
        items = list(active)
        if attempt == 0:
            items.sort(key=lambda i: amounts[i], reverse=True)
        else:
            items.sort(key=lambda i: amounts[i] * (0.9 + 0.2 * rng.random()), reverse=True)
        groups = [[] for _ in range(M)]
        loads = [0.0] * M
        feasible = True
        for i in items:
            choices = [k for k in range(M) if loads[k] + amounts[i] <= Q + 1e-7]
            if not choices:
                feasible = False
                break
            if attempt & 1:
                k = min(choices, key=lambda x: loads[x])
            else:
                k = max(choices, key=lambda x: loads[x])
            groups[k].append(i)
            loads[k] += amounts[i]
        if feasible:
            spread = max(loads) - min(loads)
            if spread < best_spread:
                best, best_spread = groups, spread
    return best


def _route_period(data, amounts, rng, deadline, randomize):
    active = [i for i, amount in enumerate(amounts) if amount > _EPS]
    M, Q = data["M"], data["Q"]
    if not active:
        return ([[] for _ in range(M)], [[] for _ in range(M)], 0.0)
    if M <= 0:
        return None

    route_map = {i: i for i in active}
    sequences = {i: [i] for i in active}
    loads = {i: amounts[i] for i in active}
    savings = []
    noise = 0.035 * data["avg_c"] if randomize else 0.0
    for pos, i in enumerate(active):
        for j in active[pos + 1:]:
            value = data["c"][0][i + 1] + data["c"][0][j + 1] - data["c"][i + 1][j + 1]
            if noise:
                value += rng.uniform(-noise, noise)
            savings.append((value, i, j))
    savings.sort(reverse=True)

    for _, i, j in savings:
        if deadline is not None and time.time() > deadline:
            break
        ri, rj = route_map[i], route_map[j]
        if ri == rj or loads[ri] + loads[rj] > Q + data["tol"]:
            continue
        a, b = sequences[ri], sequences[rj]
        if i != a[0] and i != a[-1]:
            continue
        if j != b[0] and j != b[-1]:
            continue
        if a[-1] != i:
            a = list(reversed(a))
        if b[0] != j:
            b = list(reversed(b))
        merged = a + b
        sequences[ri] = merged
        loads[ri] += loads[rj]
        for node in b:
            route_map[node] = ri
        del sequences[rj]
        del loads[rj]

    group_list = list(sequences.values())
    if len(group_list) > M:
        packed = _pack_items(amounts, M, Q, rng, randomize)
        if packed is None:
            return None
        group_list = [group for group in packed if group]
    while len(group_list) < M:
        group_list.append([])
    if len(group_list) > M:
        return None

    routes = []
    for group in group_list:
        if not group:
            routes.append([])
            continue
        order = _best_tour_order(data, group, rng, randomize, deadline)
        route = [0] + [i + 1 for i in order] + [0]
        _two_opt(data, route, deadline)
        routes.append(route)

    group_list, routes = _cross_relocate(data, amounts, group_list, routes, deadline)
    _exactify_routes(data, routes, deadline)
    cost = sum(_route_cost(route, data["c"]) for route in routes)
    return group_list, routes, cost


def _best_tour_order(data, group, rng, randomize, deadline):
    if len(group) <= 2:
        return list(group)
    threshold = 10
    if len(group) <= threshold:
        exact = _exact_tour_order(data, group, deadline)
        if exact is not None:
            return exact
    starts = list(group)
    if randomize:
        rng.shuffle(starts)
        starts = starts[:min(len(starts), 4)]
    else:
        starts = [min(group, key=lambda i: data["c"][0][i + 1])]
    best_order = None
    best_cost = _INF
    for start in starts:
        unused = set(group)
        unused.remove(start)
        order = [start]
        while unused:
            last = order[-1]
            nxt = min(
                unused,
                key=lambda j: data["c"][last + 1][j + 1]
                * (1.0 + (rng.uniform(-0.025, 0.025) if randomize else 0.0)),
            )
            order.append(nxt)
            unused.remove(nxt)
        route = [0] + [i + 1 for i in order] + [0]
        value = _route_cost(route, data["c"])
        if value < best_cost:
            best_cost, best_order = value, order
    return best_order


def _exact_tour_order(data, group, deadline):
    nodes = tuple(sorted(group))
    cached = data["exact_cache"].get(nodes)
    if cached is not None:
        return list(cached)
    n = len(nodes)
    if n <= 1:
        return list(nodes)
    # (mask,last) -> best depot-to-last path cost.  Keeping predecessor tables
    # separate avoids storing whole paths in the exponential inner loop.
    size = 1 << n
    costs = [[_INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    c = data["c"]
    for j, node in enumerate(nodes):
        costs[1 << j][j] = c[0][node + 1]
    for mask in range(1, size):
        if deadline is not None and (mask & 31) == 0 and time.time() > deadline:
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
    data["exact_cache"][nodes] = tuple(order)
    return order


def _exactify_routes(data, routes, deadline):
    for k, route in enumerate(routes):
        if not route or len(route) - 2 > 10:
            continue
        if deadline is not None and time.time() > deadline:
            return
        group = [node - 1 for node in route[1:-1]]
        order = _exact_tour_order(data, group, deadline)
        if order is not None:
            routes[k] = [0] + [i + 1 for i in order] + [0]


def _two_opt(data, route, deadline):
    if len(route) <= 4:
        return
    c = data["c"]
    while True:
        if deadline is not None and time.time() > deadline:
            return
        best_delta = -data["tol"]
        best_pair = None
        n = len(route)
        for a in range(1, n - 2):
            if deadline is not None and time.time() > deadline:
                return
            x, y = route[a - 1], route[a]
            for b in range(a + 1, n - 1):
                u, v = route[b], route[b + 1]
                delta = c[x][u] + c[y][v] - c[x][y] - c[u][v]
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (a, b)
        if best_pair is None:
            return
        a, b = best_pair
        route[a:b + 1] = reversed(route[a:b + 1])


def _cross_relocate(data, amounts, groups, routes, deadline):
    groups = [list(g) for g in groups]
    routes = [list(r) for r in routes]
    loads = [sum(amounts[i] for i in group) for group in groups]
    c = data["c"]
    while True:
        if deadline is not None and time.time() > deadline:
            break
        best = None
        best_delta = -data["tol"]
        # One-customer relocation.
        for a, route_a in enumerate(routes):
            if not route_a:
                continue
            for pos in range(1, len(route_a) - 1):
                node = route_a[pos]
                i = node - 1
                removal = c[route_a[pos - 1]][route_a[pos + 1]] - c[route_a[pos - 1]][node] - c[node][route_a[pos + 1]]
                for b, route_b in enumerate(routes):
                    if b == a or loads[b] + amounts[i] > data["Q"] + data["tol"]:
                        continue
                    if not route_b:
                        delta = removal + 2.0 * c[0][node]
                        insert_pos = 1
                    else:
                        delta, insert_pos = min(
                            (
                                removal + c[route_b[k - 1]][node] + c[node][route_b[k]] - c[route_b[k - 1]][route_b[k]],
                                k,
                            )
                            for k in range(1, len(route_b))
                        )
                    if delta < best_delta:
                        best_delta = delta
                        best = (a, pos, b, insert_pos, i)
            if deadline is not None and time.time() > deadline:
                break
        if best is not None:
            a, pos, b, insert_pos, i = best
            node = routes[a].pop(pos)
            if len(routes[a]) == 2:
                routes[a] = []
            routes[b] = [0, 0] if not routes[b] else routes[b]
            routes[b].insert(insert_pos, node)
            groups[a].remove(i)
            groups[b].append(i)
            loads[a] -= amounts[i]
            loads[b] += amounts[i]
            if routes[a]:
                _two_opt(data, routes[a], deadline)
            _two_opt(data, routes[b], deadline)
            continue

        # Capacity-preserving swaps are particularly valuable when every route
        # is almost full and relocation alone is frozen.
        swap = None
        swap_delta = -data["tol"]
        for a in range(len(routes)):
            ra = routes[a]
            if not ra:
                continue
            for b in range(a + 1, len(routes)):
                rb = routes[b]
                if not rb:
                    continue
                for pa in range(1, len(ra) - 1):
                    x = ra[pa]
                    ix = x - 1
                    for pb in range(1, len(rb) - 1):
                        y = rb[pb]
                        iy = y - 1
                        if (
                            loads[a] - amounts[ix] + amounts[iy] > data["Q"] + data["tol"]
                            or loads[b] - amounts[iy] + amounts[ix] > data["Q"] + data["tol"]
                        ):
                            continue
                        delta = (
                            c[ra[pa - 1]][y] + c[y][ra[pa + 1]]
                            - c[ra[pa - 1]][x] - c[x][ra[pa + 1]]
                            + c[rb[pb - 1]][x] + c[x][rb[pb + 1]]
                            - c[rb[pb - 1]][y] - c[y][rb[pb + 1]]
                        )
                        if delta < swap_delta:
                            swap_delta = delta
                            swap = (a, pa, b, pb, ix, iy)
                if deadline is not None and time.time() > deadline:
                    break
            if deadline is not None and time.time() > deadline:
                break
        if swap is not None:
            a, pa, b, pb, ix, iy = swap
            routes[a][pa], routes[b][pb] = routes[b][pb], routes[a][pa]
            groups[a].remove(ix)
            groups[b].remove(iy)
            groups[a].append(iy)
            groups[b].append(ix)
            loads[a] += amounts[iy] - amounts[ix]
            loads[b] += amounts[ix] - amounts[iy]
            _two_opt(data, routes[a], deadline)
            _two_opt(data, routes[b], deadline)
            continue

        # 2-opt* exchanges complete route tails, jointly changing assignment and
        # order.  This is a much larger move but has an O(n^2) exact delta.
        star = None
        star_delta = -data["tol"]
        for a in range(len(routes)):
            ra = routes[a]
            if not ra:
                continue
            prefix_a = [0.0]
            for node in ra[1:-1]:
                prefix_a.append(prefix_a[-1] + amounts[node - 1])
            for b in range(a + 1, len(routes)):
                rb = routes[b]
                if not rb:
                    continue
                prefix_b = [0.0]
                for node in rb[1:-1]:
                    prefix_b.append(prefix_b[-1] + amounts[node - 1])
                for ca in range(len(ra) - 1):
                    kept_a = prefix_a[ca]
                    tail_a = loads[a] - kept_a
                    for cb in range(len(rb) - 1):
                        kept_b = prefix_b[cb]
                        tail_b = loads[b] - kept_b
                        if (
                            kept_a + tail_b > data["Q"] + data["tol"]
                            or kept_b + tail_a > data["Q"] + data["tol"]
                        ):
                            continue
                        delta = (
                            c[ra[ca]][rb[cb + 1]] + c[rb[cb]][ra[ca + 1]]
                            - c[ra[ca]][ra[ca + 1]] - c[rb[cb]][rb[cb + 1]]
                        )
                        if delta < star_delta:
                            star_delta = delta
                            star = (a, ca, b, cb)
                if deadline is not None and time.time() > deadline:
                    break
            if deadline is not None and time.time() > deadline:
                break
        if star is None:
            break
        a, ca, b, cb = star
        ra, rb = routes[a], routes[b]
        new_a = ra[:ca + 1] + rb[cb + 1:-1] + [0]
        new_b = rb[:cb + 1] + ra[ca + 1:-1] + [0]
        routes[a] = new_a if len(new_a) > 2 else []
        routes[b] = new_b if len(new_b) > 2 else []
        groups[a] = [node - 1 for node in routes[a][1:-1]] if routes[a] else []
        groups[b] = [node - 1 for node in routes[b][1:-1]] if routes[b] else []
        loads[a] = sum(amounts[i] for i in groups[a])
        loads[b] = sum(amounts[i] for i in groups[b])
        if routes[a]:
            _two_opt(data, routes[a], deadline)
        if routes[b]:
            _two_opt(data, routes[b], deadline)
    return groups, routes


def _reroute_state(data, state, rng, deadline):
    weights = [x + 0.05 * data["avg_c"] for x in state["travel_t"]]
    draw = rng.random() * sum(weights)
    t = 0
    for k, weight in enumerate(weights):
        draw -= weight
        if draw <= 0.0:
            t = k
            break

    best_result = (
        [list(g) for g in state["groups"][t]],
        [list(r) for r in state["routes"][t]],
        state["travel_t"][t],
    )
    # Repeated perturb-and-descend uses the complete local time slice.  The
    # incumbent period is retained, so an unlucky destroy step cannot hurt best.
    while time.time() < deadline:
        if rng.random() < 0.22:
            result = _route_period(data, state["q"][t], rng, deadline, True)
        else:
            result = _perturb_period(
                data,
                state["q"][t],
                state["groups"][t],
                state["routes"][t],
                rng,
                deadline,
            )
        if result is not None and result[2] < best_result[2] - data["tol"]:
            best_result = result

    groups = list(state["groups"])
    routes = list(state["routes"])
    travel_t = list(state["travel_t"])
    groups[t], routes[t], travel_t[t] = best_result
    return {
        "q": state["q"],
        "inv": state["inv"],
        "groups": groups,
        "routes": routes,
        "travel_t": travel_t,
        "holding": state["holding"],
        "cost": state["holding"] + sum(travel_t),
    }


def _perturb_period(data, amounts, old_groups, old_routes, rng, deadline):
    groups = [list(g) for g in old_groups]
    routes = [list(r) for r in old_routes]
    loads = [sum(amounts[i] for i in g) for g in groups]
    intensity = 1 + rng.randrange(max(2, min(7, 2 + len([g for g in groups if g]))))

    for _ in range(intensity):
        if time.time() > deadline:
            return None
        nonempty = [k for k, g in enumerate(groups) if g]
        if len(nonempty) < 2:
            break
        a, b = rng.sample(nonempty, 2)
        ia = rng.choice(groups[a])
        ib = rng.choice(groups[b])
        new_a = loads[a] - amounts[ia] + amounts[ib]
        new_b = loads[b] - amounts[ib] + amounts[ia]
        if new_a <= data["Q"] + data["tol"] and new_b <= data["Q"] + data["tol"]:
            pa = routes[a].index(ia + 1)
            pb = routes[b].index(ib + 1)
            routes[a][pa], routes[b][pb] = routes[b][pb], routes[a][pa]
            groups[a].remove(ia)
            groups[b].remove(ib)
            groups[a].append(ib)
            groups[b].append(ia)
            loads[a], loads[b] = new_a, new_b
            continue

        # When a swap is capacity-infeasible, a relocation into genuine slack
        # still provides a useful kick.
        if loads[b] + amounts[ia] <= data["Q"] + data["tol"]:
            routes[a].remove(ia + 1)
            if len(routes[a]) == 2:
                routes[a] = []
            if not routes[b]:
                routes[b] = [0, ia + 1, 0]
            else:
                pos = rng.randrange(1, len(routes[b]))
                routes[b].insert(pos, ia + 1)
            groups[a].remove(ia)
            groups[b].append(ia)
            loads[a] -= amounts[ia]
            loads[b] += amounts[ia]

    for route in routes:
        if route:
            _two_opt(data, route, deadline)
    groups, routes = _cross_relocate(data, amounts, groups, routes, deadline)
    _exactify_routes(data, routes, deadline)
    cost = sum(_route_cost(route, data["c"]) for route in routes)
    return groups, routes, cost


def _temporal_neighbor(data, state, rng, deadline):
    H, N = data["H"], data["N"]
    proposals = []
    for i in range(N):
        active = [t for t in range(H) if state["q"][t][i] > data["tol"]]
        for source in active:
            amount = state["q"][source][i]
            for target in range(H):
                if target == source or state["q"][target][i] + amount > data["Q"] + data["tol"]:
                    continue
                if sum(state["q"][target]) + amount > data["M"] * data["Q"] + data["tol"]:
                    continue
                if target < source:
                    room = min(
                        data["U"][i + 1] - data["d"][i][u] - state["inv"][u][i + 1]
                        for u in range(target, source)
                    )
                    supplier_room = min(state["inv"][u][0] for u in range(target, source))
                    if amount > min(room, supplier_room) + data["tol"]:
                        continue
                else:
                    stock = min(state["inv"][u][i + 1] for u in range(source, target))
                    supplier_room = min(data["U"][0] - state["inv"][u][0] for u in range(source, target))
                    if amount > min(stock, supplier_room) + data["tol"]:
                        continue
                hold_delta = (source - target) * (data["h"][i + 1] - data["h"][0]) * amount
                route_delta = _move_route_estimate(data, state, i, source, target)
                proposals.append((hold_delta + route_delta, i, source, target))
        if time.time() > deadline:
            return None
    if not proposals:
        return None
    proposals.sort(key=lambda x: x[0])
    width = min(len(proposals), 1 + int(math.sqrt(len(proposals))))
    _, i, source, target = proposals[rng.randrange(width)]
    q = [list(row) for row in state["q"]]
    q[target][i] += q[source][i]
    q[source][i] = 0.0
    q = _repair_schedule(data, q, deadline)
    if q is None:
        return None
    return _state_with_affected_q(data, state, q, {source, target}, rng, deadline)


def _state_with_affected_q(data, state, q, affected, rng, deadline):
    inv = _inventories(data, q)
    if inv is None:
        return None
    groups = list(state["groups"])
    routes = list(state["routes"])
    travel_t = list(state["travel_t"])
    for t in affected:
        if time.time() > deadline:
            return None
        result = _route_period(data, q[t], rng, deadline, True)
        if result is None:
            return None
        groups[t], routes[t], travel_t[t] = result
    holding = sum(
        data["h"][loc] * inv[t][loc]
        for t in range(data["H"])
        for loc in range(data["N"] + 1)
    )
    return {
        "q": q,
        "inv": inv,
        "groups": groups,
        "routes": routes,
        "travel_t": travel_t,
        "holding": holding,
        "cost": holding + sum(travel_t),
    }


def _move_route_estimate(data, state, i, source, target):
    node = i + 1
    delta = 0.0
    for route in state["routes"][source]:
        if node in route:
            pos = route.index(node)
            delta += data["c"][route[pos - 1]][route[pos + 1]] - data["c"][route[pos - 1]][node] - data["c"][node][route[pos + 1]]
            break
    if state["q"][target][i] <= _EPS:
        insertion = _INF
        for k, route in enumerate(state["routes"][target]):
            load = sum(state["q"][target][j] for j in state["groups"][target][k])
            if load + state["q"][source][i] > data["Q"] + data["tol"]:
                continue
            if not route:
                insertion = min(insertion, 2.0 * data["c"][0][node])
            else:
                insertion = min(
                    insertion,
                    min(
                        data["c"][route[p - 1]][node] + data["c"][node][route[p]] - data["c"][route[p - 1]][route[p]]
                        for p in range(1, len(route))
                    ),
                )
        delta += insertion if math.isfinite(insertion) else 2.0 * data["avg_c"]
    return delta


def _route_cost(route, c):
    if not route:
        return 0.0
    return sum(c[route[k]][route[k + 1]] for k in range(len(route) - 1))


def _valid_state(data, state):
    try:
        inv = _inventories(data, state["q"])
        if inv is None:
            return False
        for t in range(data["H"]):
            if len(state["groups"][t]) != data["M"] or len(state["routes"][t]) != data["M"]:
                return False
            seen = set()
            for k in range(data["M"]):
                group = state["groups"][t][k]
                route = state["routes"][t][k]
                if sum(state["q"][t][i] for i in group) > data["Q"] + data["tol"]:
                    return False
                if any(i in seen for i in group):
                    return False
                seen.update(group)
                if group:
                    if route[0] != 0 or route[-1] != 0:
                        return False
                    if len(route[1:-1]) != len(group):
                        return False
                    if set(route[1:-1]) != {i + 1 for i in group}:
                        return False
                elif route:
                    return False
            if seen != {i for i in range(data["N"]) if state["q"][t][i] > _EPS}:
                return False
        return True
    except BaseException:
        return False


def _emit(data, state):
    deliveries = [
        [[0.0] * data["N"] for _ in range(data["M"])]
        for _ in range(data["H"])
    ]
    routes = [[[] for _ in range(data["M"])] for _ in range(data["H"])]
    for t in range(data["H"]):
        for k in range(data["M"]):
            routes[t][k] = list(state["routes"][t][k])
            for i in state["groups"][t][k]:
                deliveries[t][k][i] = float(max(0.0, state["q"][t][i]))
    inventories = [
        [0.0 if abs(x) <= data["tol"] else float(x) for x in row]
        for row in state["inv"]
    ]
    return {"routes": routes, "deliveries": deliveries, "end_inventory": inventories}


def _last_resort(data):
    q = _baseline_schedule(data)
    if q is not None:
        state = _make_state(data, q, random.Random(0), None, False)
        if state is not None:
            return _emit(data, state)
    return _structural_fallback(
        {
            "num_periods": data["H"],
            "num_customers": data["N"],
            "num_vehicles": data["M"],
            "initial_inventory": data["I0"],
            "supplier_production": data["p"],
            "customer_demand": data["d"],
        }
    )


def _structural_fallback(instance):
    try:
        H = max(0, int(instance.get("num_periods", 0)))
        N = max(0, int(instance.get("num_customers", 0)))
        M = max(0, int(instance.get("num_vehicles", 0)))
    except BaseException:
        H = N = M = 0
    routes = [[[] for _ in range(M)] for _ in range(H)]
    deliveries = [[[0.0] * N for _ in range(M)] for _ in range(H)]
    try:
        previous = [float(x) for x in instance.get("initial_inventory", [0.0] * (N + 1))]
    except BaseException:
        previous = [0.0] * (N + 1)
    inventories = []
    production = instance.get("supplier_production", [0.0] * H)
    demand = instance.get("customer_demand", [[0.0] * H for _ in range(N)])
    for t in range(H):
        row = [0.0] * (N + 1)
        row[0] = previous[0] + float(production[t])
        for i in range(N):
            row[i + 1] = previous[i + 1] - float(demand[i][t])
        inventories.append(row)
        previous = row
    return {"routes": routes, "deliveries": deliveries, "end_inventory": inventories}
