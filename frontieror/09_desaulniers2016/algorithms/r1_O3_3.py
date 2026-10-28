"""Reverse-demand-guided joint inventory-flow and route search
Guides a time-space network flow by reverse replenishment grouping and improves continuously with visit-removal flow re-optimisation, cross-period shipment moves, slack adjustment and route neighbourhoods.
"""
import time
import math
import random
import heapq
from collections import deque

_ACTIVE = 1e-9
_TOL = 1e-9


def _blank_solution(instance):
    try:
        T = max(0, int(instance.get("num_periods", 0)))
    except BaseException:
        T = 0
    try:
        n = max(0, int(instance.get("num_customers", 0)))
    except BaseException:
        n = 0
    try:
        m = max(0, int(instance.get("num_vehicles", 0)))
    except BaseException:
        m = 0
    return {
        "routes": [[[] for _ in range(m)] for _ in range(T)],
        "deliveries": [[[0.0 for _ in range(n)] for _ in range(m)] for _ in range(T)],
        "ending_inventories": [[0.0 for _ in range(n + 1)] for _ in range(T)],
    }

def _add_edge(graph, u, v, capacity, cost=0.0):
    capacity = max(0.0, float(capacity))
    forward = [v, len(graph[v]), capacity, float(cost), capacity]
    reverse = [u, len(graph[u]), 0.0, -float(cost), 0.0]
    graph[u].append(forward)
    graph[v].append(reverse)
    return forward

def _build_flow_graph(data, assignment, delivery_bias, upper=None):
    T, n, m, Q = data["T"], data["n"], data["m"], data["Q"]
    counter = [0]

    def node():
        value = counter[0]
        counter[0] += 1
        return value

    depot = [node() for _ in range(T)]
    if assignment is None:
        gates = [[node()] for _ in range(T)]
    else:
        gates = [[node() for _ in range(m)] for _ in range(T)]
    cin = [[node() for _ in range(T)] for _ in range(n)]
    cout = [[node() for _ in range(T)] for _ in range(n)]
    end_node = node()
    source = node()
    sink = node()
    graph = [[] for _ in range(counter[0])]

    production = data["production"]
    initial = data["initial"]
    caps = data["caps"]
    demands = data["demands"]
    holding = data["holding"]

    total_supply = initial[0] + sum(production)
    total_supply += sum(initial[1:])
    total_demand = sum(sum(row) for row in demands)
    excess = total_supply - total_demand
    if excess < -_TOL:
        return None
    if excess < 0.0:
        excess = 0.0

    _add_edge(graph, source, depot[0], initial[0])
    for t in range(T):
        _add_edge(graph, source, depot[t], production[t])
    for i in range(n):
        _add_edge(graph, source, cin[i][0], initial[i + 1])

    delivery_refs = [[None for _ in range(n)] for _ in range(T)]
    for t in range(T):
        if assignment is None:
            _add_edge(graph, depot[t], gates[t][0], m * Q)
            for i in range(n):
                bias = 0.0 if delivery_bias is None else delivery_bias[t][i]
                delivery_refs[t][i] = _add_edge(
                    graph, gates[t][0], cin[i][t], Q if upper is None else upper[t][i], bias
                )
        else:
            for k in range(m):
                _add_edge(graph, depot[t], gates[t][k], Q)
            for i in range(n):
                k = assignment[t][i]
                bias = 0.0 if delivery_bias is None else delivery_bias[t][i]
                delivery_refs[t][i] = _add_edge(
                    graph, gates[t][k], cin[i][t], Q if upper is None else upper[t][i], bias
                )

        if t + 1 < T:
            _add_edge(graph, depot[t], depot[t + 1], caps[0], holding[0])
        else:
            _add_edge(graph, depot[t], end_node, caps[0], holding[0])

    for i in range(n):
        cap = caps[i + 1]
        h = holding[i + 1]
        for t in range(T):
            _add_edge(graph, cin[i][t], cout[i][t], cap)
            _add_edge(graph, cout[i][t], sink, demands[i][t])
            if t + 1 < T:
                _add_edge(graph, cout[i][t], cin[i][t + 1], cap, h)
            else:
                _add_edge(graph, cout[i][t], end_node, cap, h)

    _add_edge(graph, end_node, sink, excess)
    return graph, source, sink, total_supply, delivery_refs

def _flow_schedule(data, assignment, use_cost, deadline, delivery_bias=None, upper=None):
    built = _build_flow_graph(data, assignment, delivery_bias, upper)
    if built is None:
        return None
    graph, source, sink, target, refs = built
    if target <= _TOL:
        return [[0.0 for _ in range(data["n"])] for _ in range(data["T"])]
    if use_cost:
        flow = _min_cost_flow(graph, source, sink, target, deadline)
    else:
        flow = _dinic(graph, source, sink, target, deadline)
    if target - flow > max(1e-7, 1e-11 * target):
        return None
    q = [[0.0 for _ in range(data["n"])] for _ in range(data["T"])]
    for t in range(data["T"]):
        for i in range(data["n"]):
            edge = refs[t][i]
            value = edge[4] - edge[2]
            if value < _ACTIVE:
                value = 0.0
            q[t][i] = value
    return q

def _make_assignment(data, rng, mode):
    T, n, m = data["T"], data["n"], data["m"]
    demands = data["demands"]
    if m <= 0:
        return None

    def assign_weights(weights, randomized):
        loads = [0.0] * m
        result = [0] * n
        order = list(range(n))
        if randomized:
            order.sort(key=lambda i: -(weights[i] * (0.7 + 0.6 * rng.random())))
        else:
            order.sort(key=lambda i: (-weights[i], i))
        for i in order:
            if randomized:
                k = min(
                    range(m),
                    key=lambda x: loads[x] + weights[i] + rng.random() * (1.0 + weights[i]) * 0.12,
                )
            else:
                k = min(range(m), key=lambda x: (loads[x], x))
            result[i] = k
            loads[k] += weights[i]
        return result

    if mode == "static":
        weights = [sum(demands[i]) for i in range(n)]
        row = assign_weights(weights, False)
        return [row[:] for _ in range(T)]

    assignment = []
    for t in range(T):
        weights = []
        for i in range(n):
            value = demands[i][t]
            if t + 1 < T:
                value += 0.45 * demands[i][t + 1]
            if t + 2 < T:
                value += 0.15 * demands[i][t + 2]
            weights.append(value)
        assignment.append(assign_weights(weights, mode == "random"))
    return assignment

def _route_cost(route, costs):
    if not route:
        return 0.0
    value = costs[0][route[0]] + costs[route[-1]][0]
    for a, b in zip(route, route[1:]):
        value += costs[a][b]
    return value

def _two_opt(route, costs, deadline):
    route = route[:]
    length = len(route)
    if length < 3:
        return route
    while time.monotonic() < deadline:
        best_delta = -1e-10
        best_pair = None
        for left in range(length - 1):
            a = 0 if left == 0 else route[left - 1]
            b = route[left]
            for right in range(left + 1, length):
                if left == 0 and right == length - 1:
                    continue
                c = route[right]
                d = 0 if right + 1 == length else route[right + 1]
                delta = costs[a][c] + costs[b][d] - costs[a][b] - costs[c][d]
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (left, right)
            if time.monotonic() >= deadline:
                break
        if best_pair is None:
            break
        left, right = best_pair
        route[left:right + 1] = reversed(route[left:right + 1])
    return route

def _valid_groups(groups, active, amounts, m, Q):
    if groups is None or len(groups) > m:
        return False
    seen = set()
    for group in groups:
        load = 0.0
        for customer in group:
            if customer not in active or customer in seen:
                return False
            seen.add(customer)
            load += amounts[customer - 1]
        if load > Q + 1e-6:
            return False
    return seen == active

def _hint_groups(hint, active, amounts, m, Q, costs):
    if hint is None:
        return None
    groups = [[] for _ in range(m)]
    seen = set()
    loads = [0.0] * m
    for k in range(min(m, len(hint))):
        for customer in hint[k]:
            if customer in active and customer not in seen:
                groups[k].append(customer)
                seen.add(customer)
                loads[k] += amounts[customer - 1]
    if any(load > Q + 1e-6 for load in loads):
        return None
    for customer in active - seen:
        best = None
        amount = amounts[customer - 1]
        for k in range(m):
            if loads[k] + amount > Q + 1e-6:
                continue
            route = groups[k]
            for pos in range(len(route) + 1):
                a = 0 if pos == 0 else route[pos - 1]
                b = 0 if pos == len(route) else route[pos]
                delta = costs[a][customer] + costs[customer][b] - costs[a][b]
                candidate = (delta, k, pos)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return None
        _, k, pos = best
        groups[k].insert(pos, customer)
        loads[k] += amount
    return groups

def _ffd_groups(active, amounts, m, Q):
    groups = [[] for _ in range(m)]
    loads = [0.0] * m
    order = sorted(active, key=lambda x: (-amounts[x - 1], x))
    for customer in order:
        amount = amounts[customer - 1]
        choices = [k for k in range(m) if loads[k] + amount <= Q + 1e-7]
        if not choices:
            return None
        k = max(choices, key=lambda x: (loads[x], -x))
        groups[k].append(customer)
        loads[k] += amount
    return groups

def _exact_groups(active, amounts, m, Q, deadline):
    items = sorted(active, key=lambda x: (-amounts[x - 1], x))
    groups = [[] for _ in range(m)]
    loads = [0.0] * m

    def search(pos):
        if time.monotonic() >= deadline:
            return False
        if pos == len(items):
            return True
        customer = items[pos]
        amount = amounts[customer - 1]
        order = sorted(range(m), key=lambda k: -loads[k])
        tried = []
        for k in order:
            if loads[k] + amount > Q + 1e-7:
                continue
            if any(abs(loads[k] - old) <= 1e-10 for old in tried):
                continue
            tried.append(loads[k])
            groups[k].append(customer)
            loads[k] += amount
            if search(pos + 1):
                return True
            loads[k] -= amount
            groups[k].pop()
        return False

    try:
        return groups if search(0) else None
    except RecursionError:
        return None

def _cw_groups(active, amounts, m, Q, costs, deadline):
    if not active:
        return [[] for _ in range(m)]
    routes = {customer: [customer] for customer in active}
    loads = {customer: amounts[customer - 1] for customer in active}
    where = {customer: customer for customer in active}
    savings = []
    ordered = sorted(active)
    for p, i in enumerate(ordered):
        if time.monotonic() >= deadline:
            return None
        for j in ordered[p + 1:]:
            savings.append((-(costs[0][i] + costs[0][j] - costs[i][j]), i, j))
    savings.sort()
    for _, i, j in savings:
        if time.monotonic() >= deadline:
            break
        ri, rj = where[i], where[j]
        if ri == rj:
            continue
        left, right = routes[ri], routes[rj]
        if i not in (left[0], left[-1]) or j not in (right[0], right[-1]):
            continue
        if loads[ri] + loads[rj] > Q + 1e-7:
            continue
        if left[-1] != i:
            left = list(reversed(left))
        if right[0] != j:
            right = list(reversed(right))
        merged = left + right
        routes[ri] = merged
        loads[ri] += loads[rj]
        del routes[rj]
        del loads[rj]
        for customer in merged:
            where[customer] = ri
    groups = list(routes.values())
    if len(groups) > m:
        return None
    groups.extend([[] for _ in range(m - len(groups))])
    return groups

def _routes_for_amounts(amounts, data, deadline, hint=None):
    m, Q, costs = data["m"], data["Q"], data["costs"]
    active = {i + 1 for i, value in enumerate(amounts) if value > _ACTIVE}
    if not active:
        return ([[] for _ in range(m)], 0.0)
    if m <= 0 or any(amounts[i - 1] > Q + 1e-6 for i in active):
        return None

    candidates = []
    hinted = _hint_groups(hint, active, amounts, m, Q, costs)
    if _valid_groups(hinted, active, amounts, m, Q):
        candidates.append(hinted)
    ffd = _ffd_groups(active, amounts, m, Q)
    if _valid_groups(ffd, active, amounts, m, Q):
        candidates.append(ffd)
    if time.monotonic() < deadline:
        cw = _cw_groups(active, amounts, m, Q, costs, deadline)
        if _valid_groups(cw, active, amounts, m, Q):
            candidates.append(cw)
    if not candidates:
        exact = _exact_groups(active, amounts, m, Q, deadline)
        if _valid_groups(exact, active, amounts, m, Q):
            candidates.append(exact)
    if not candidates:
        return None

    best_routes = None
    best_cost = float("inf")
    for groups in candidates:
        routes = []
        for group in groups[:m]:
            if time.monotonic() < deadline:
                routes.append(_order_group(group, costs, deadline))
            else:
                routes.append(list(group))
        routes.extend([[] for _ in range(m - len(routes))])
        value = sum(_route_cost(route, costs) for route in routes)
        if value < best_cost:
            best_cost = value
            best_routes = routes
    return best_routes, best_cost

def _inventories(q, data):
    T, n, m, Q = data["T"], data["n"], data["m"], data["Q"]
    caps, initial = data["caps"], data["initial"]
    production, demands = data["production"], data["demands"]
    depot = initial[0]
    customers = initial[1:]
    result = []
    for t in range(T):
        total = 0.0
        for i in range(n):
            value = q[t][i]
            if value < -1e-7 or value > Q + 1e-6:
                return None
            total += value
        if total > m * Q + 1e-6:
            return None
        depot += production[t] - total
        if depot < -1e-6 or depot > caps[0] + 1e-6:
            return None
        if abs(depot) < 1e-9:
            depot = 0.0
        if abs(depot - caps[0]) < 1e-9:
            depot = caps[0]
        row = [depot]
        for i in range(n):
            before_demand = customers[i] + q[t][i]
            if before_demand > caps[i + 1] + 1e-6:
                return None
            customers[i] = before_demand - demands[i][t]
            if customers[i] < -1e-6:
                return None
            if abs(customers[i]) < 1e-9:
                customers[i] = 0.0
            row.append(customers[i])
        result.append(row)
    return result

def _holding_cost(inventories, data):
    holding = data["holding"]
    return sum(
        sum(row[i] * holding[i] for i in range(len(row)))
        for row in inventories
    )

def _state_from_schedule(q, data, deadline, assignment=None, route_hints=None):
    inventories = _inventories(q, data)
    if inventories is None:
        return None
    routes = []
    period_costs = []
    for t in range(data["T"]):
        hint = None
        if route_hints is not None:
            hint = route_hints[t]
        elif assignment is not None:
            hint = [[] for _ in range(data["m"])]
            for i, value in enumerate(q[t]):
                if value > _ACTIVE:
                    hint[assignment[t][i]].append(i + 1)
        built = _routes_for_amounts(q[t], data, deadline, hint)
        if built is None:
            return None
        period_routes, period_cost = built
        routes.append(period_routes)
        period_costs.append(period_cost)
    objective = sum(period_costs) + _holding_cost(inventories, data)
    return {
        "q": [row[:] for row in q],
        "inv": inventories,
        "routes": routes,
        "pcost": period_costs,
        "obj": objective,
    }

def _max_transfer(state, source_t, target_t, customer, data):
    if source_t == target_t:
        return 0.0
    q = state["q"]
    inv = state["inv"]
    Q, m = data["Q"], data["m"]
    caps, demands = data["caps"], data["demands"]
    value = min(q[source_t][customer], Q - q[target_t][customer])
    value = min(value, m * Q - sum(q[target_t]))
    if target_t < source_t:
        for t in range(target_t, source_t):
            customer_pre = inv[t][customer + 1] + demands[customer][t]
            value = min(value, caps[customer + 1] - customer_pre)
            value = min(value, inv[t][0])
    else:
        for t in range(source_t, target_t):
            value = min(value, inv[t][customer + 1])
            value = min(value, caps[0] - inv[t][0])
    return max(0.0, value)

def _removal_saving(routes, customer, costs):
    for route in routes:
        if customer in route:
            pos = route.index(customer)
            a = 0 if pos == 0 else route[pos - 1]
            b = 0 if pos + 1 == len(route) else route[pos + 1]
            return costs[a][customer] + costs[customer][b] - costs[a][b]
    return 0.0

def _one_route_move(routes, amounts, data, rng):
    m, Q = data["m"], data["Q"]
    result = [route[:] for route in routes]
    nonempty = [k for k in range(m) if result[k]]
    if not nonempty:
        return None
    loads = [sum(amounts[c - 1] for c in route) for route in result]
    operation = rng.randrange(4)

    if operation == 0:
        choices = [k for k in nonempty if len(result[k]) >= 3]
        if not choices:
            return None
        k = rng.choice(choices)
        left = rng.randrange(len(result[k]) - 1)
        right = rng.randrange(left + 1, len(result[k]))
        result[k][left:right + 1] = reversed(result[k][left:right + 1])
        return result

    if operation == 1:
        a = rng.choice(nonempty)
        b = rng.randrange(m)
        pos = rng.randrange(len(result[a]))
        customer = result[a][pos]
        if a != b and loads[b] + amounts[customer - 1] > Q + 1e-7:
            return None
        result[a].pop(pos)
        insert_at = rng.randrange(len(result[b]) + 1)
        result[b].insert(insert_at, customer)
        if result == routes:
            return None
        return result

    if operation == 2:
        a = rng.choice(nonempty)
        b = rng.choice(nonempty)
        if a == b and len(result[a]) < 2:
            return None
        pa = rng.randrange(len(result[a]))
        pb = rng.randrange(len(result[b]))
        ca, cb = result[a][pa], result[b][pb]
        if a != b:
            new_a = loads[a] - amounts[ca - 1] + amounts[cb - 1]
            new_b = loads[b] - amounts[cb - 1] + amounts[ca - 1]
            if new_a > Q + 1e-7 or new_b > Q + 1e-7:
                return None
        result[a][pa], result[b][pb] = cb, ca
        if result == routes:
            return None
        return result

    if len(nonempty) < 2:
        return None
    a, b = rng.sample(nonempty, 2)
    cut_a = rng.randrange(len(result[a]) + 1)
    cut_b = rng.randrange(len(result[b]) + 1)
    first = result[a][:cut_a] + result[b][cut_b:]
    second = result[b][:cut_b] + result[a][cut_a:]
    load_first = sum(amounts[c - 1] for c in first)
    load_second = sum(amounts[c - 1] for c in second)
    if load_first > Q + 1e-7 or load_second > Q + 1e-7:
        return None
    result[a], result[b] = first, second
    if result == routes:
        return None
    return result

def _seed_from_data(data):
    value = 1469598103934665603
    values = [data["T"], data["n"], data["m"], int(data["Q"] * 1000.0)]
    values.extend(int(x * 100.0) for x in data["production"])
    for row in data["demands"]:
        values.extend(int(x * 100.0) for x in row)
    for item in values:
        value ^= item & 0xFFFFFFFFFFFFFFFF
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value

def _to_output(state, data):
    T, n, m = data["T"], data["n"], data["m"]
    clean_q = []
    for row in state["q"]:
        clean_q.append([0.0 if abs(x) <= _ACTIVE else float(x) for x in row])
    inventories = _inventories(clean_q, data)
    if inventories is None:
        inventories = state["inv"]
    routes = [[list(route) for route in period] for period in state["routes"]]
    deliveries = [[[0.0 for _ in range(n)] for _ in range(m)] for _ in range(T)]
    for t in range(T):
        assigned = set()
        for k in range(m):
            filtered = []
            for customer in routes[t][k]:
                if customer in assigned:
                    continue
                value = clean_q[t][customer - 1]
                if value > _ACTIVE:
                    filtered.append(customer)
                    assigned.add(customer)
                    deliveries[t][k][customer - 1] = value
            routes[t][k] = filtered
    return {
        "routes": routes,
        "deliveries": deliveries,
        "ending_inventories": [[float(x) for x in row] for row in inventories],
    }

def _dinic(graph, source, sink, target, deadline):
    """Non-recursive blocking flow, avoiding call-stack exhaustion for long planning horizons."""
    size = len(graph)
    flow = 0.0
    while flow + _TOL < target and time.monotonic() < deadline:
        level = [-1] * size
        level[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for e in graph[u]:
                if e[2] > _TOL and level[e[0]] < 0:
                    level[e[0]] = level[u] + 1
                    queue.append(e[0])
        if level[sink] < 0:
            break
        work = [0] * size
        while time.monotonic() < deadline:
            vertices = [source]
            path = []
            available = [target - flow]
            while vertices and vertices[-1] != sink:
                u = vertices[-1]
                while work[u] < len(graph[u]):
                    e = graph[u][work[u]]
                    if e[2] > _TOL and level[e[0]] == level[u] + 1:
                        break
                    work[u] += 1
                if work[u] == len(graph[u]):
                    level[u] = -1
                    vertices.pop()
                    available.pop()
                    if path:
                        path.pop()
                else:
                    e = graph[u][work[u]]
                    path.append((u, work[u]))
                    vertices.append(e[0])
                    available.append(min(available[-1], e[2]))
            if not vertices:
                break
            amount = available[-1]
            for u, j in path:
                e = graph[u][j]
                e[2] -= amount
                graph[e[0]][e[1]][2] += amount
            flow += amount
            if flow + _TOL >= target:
                break
    return flow


def _min_cost_flow(graph, source, sink, target, deadline):
    """Shortest augmenting paths with potentials; when costs can be negative, compute the initial potentials first."""
    size = len(graph)
    potential = [0.0] * size
    if any(e[2] > _TOL and e[3] < 0.0 for row in graph for e in row):
        dist = [float('inf')] * size
        dist[source] = 0.0
        queue = deque([source])
        queued = [False] * size
        queued[source] = True
        while queue:
            if time.monotonic() >= deadline:
                return 0.0
            u = queue.popleft()
            queued[u] = False
            for e in graph[u]:
                if e[2] > _TOL and dist[e[0]] > dist[u] + e[3] + 1e-11:
                    dist[e[0]] = dist[u] + e[3]
                    if not queued[e[0]]:
                        queued[e[0]] = True
                        queue.append(e[0])
        potential = [v if math.isfinite(v) else 0.0 for v in dist]
    flow = 0.0
    while flow + _TOL < target and time.monotonic() < deadline:
        dist = [float('inf')] * size
        parent = [None] * size
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            value, u = heapq.heappop(heap)
            if value > dist[u] + 1e-11:
                continue
            if time.monotonic() >= deadline:
                return flow
            for j, e in enumerate(graph[u]):
                if e[2] <= _TOL:
                    continue
                v = e[0]
                reduced = e[3] + potential[u] - potential[v]
                new_value = value + max(0.0, reduced)
                if new_value + 1e-11 < dist[v]:
                    dist[v] = new_value
                    parent[v] = (u, j)
                    heapq.heappush(heap, (new_value, v))
        if parent[sink] is None:
            break
        for u in range(size):
            if math.isfinite(dist[u]):
                potential[u] += dist[u]
        amount = target - flow
        v = sink
        while v != source:
            u, j = parent[v]
            amount = min(amount, graph[u][j][2])
            v = u
        v = sink
        while v != source:
            u, j = parent[v]
            e = graph[u][j]
            e[2] -= amount
            graph[v][e[1]][2] += amount
            v = u
        flow += amount
    return flow


def _order_group(group, costs, deadline):
    group = list(group)
    if len(group) < 3 or time.monotonic() >= deadline:
        return group
    remaining = set(group)
    nearest = []
    current = 0
    while remaining and time.monotonic() < deadline:
        current = min(remaining, key=lambda x: (costs[current][x], x))
        remaining.remove(current)
        nearest.append(current)
    nearest.extend(sorted(remaining))
    choices = [group, nearest]
    if time.monotonic() < deadline:
        insertion = [max(group, key=lambda x: costs[0][x])]
        remaining = set(group) - set(insertion)
        while remaining and time.monotonic() < deadline:
            best = None
            for c in remaining:
                delta, pos = _insertion(insertion, c, costs)
                item = (delta, c, pos)
                if best is None or item < best:
                    best = item
                if time.monotonic() >= deadline:
                    break
            _, c, pos = best
            insertion.insert(pos, c)
            remaining.remove(c)
        insertion.extend(sorted(remaining))
        choices.append(insertion)
    route = min(choices, key=lambda r: _route_cost(r, costs))
    return _two_opt(route, costs, deadline)


def _insertion(route, customer, costs):
    best = (float('inf'), 0)
    left = 0
    for pos in range(len(route) + 1):
        right = route[pos] if pos < len(route) else 0
        value = costs[left][customer] + costs[customer][right] - costs[left][right]
        if value < best[0]:
            best = (value, pos)
        left = right
    return best


def _backward_assignment(data, rng, randomized, deadline):
    """Back-calculate the inventory that must be stocked in advance, passing remainders below a full truckload to the previous period."""
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    demands, caps, costs = data['demands'], data['caps'], data['costs']
    floors = [[0.0] * n for _ in range(T)]
    assignment = [[-1] * n for _ in range(T)]
    for t in range(T - 1, -1, -1):
        if time.monotonic() >= deadline:
            return None
        need = [max(0.0, demands[i][t] + floors[t][i] -
                    (data['initial'][i + 1] if t == 0 else 0.0)) for i in range(n)]
        order = list(range(n))
        order.sort(key=lambda i: -min(Q, need[i]) *
                   (0.65 + 0.7 * rng.random() if randomized else 1.0))
        residual = [Q] * m
        groups = [[] for _ in range(m)]
        for i in order:
            if time.monotonic() >= deadline:
                return None
            wanted = min(Q, need[i])
            fitting = [k for k in range(m) if residual[k] + 1e-8 >= wanted]
            if fitting:
                if randomized:
                    k = min(fitting, key=lambda z:
                            (_insertion(groups[z], i + 1, costs)[0] +
                             costs[0][i + 1] * (residual[z] - wanted) / max(Q, 1e-9)) *
                            (0.8 + 0.4 * rng.random()))
                else:
                    k = min(fitting, key=lambda z: residual[z] - wanted)
            else:
                k = max(range(m), key=lambda z: residual[z])
            quantity = min(wanted, residual[k])
            assignment[t][i] = k
            residual[k] -= quantity
            if quantity > _ACTIVE:
                groups[k].append(i + 1)
            if t > 0:
                floors[t - 1][i] = max(0.0, need[i] - quantity)
                if floors[t - 1][i] + demands[i][t - 1] > caps[i + 1] + 1e-7:
                    return None
    return assignment, floors


def _state_with_routes(state, t, routes, data):
    value = sum(_route_cost(r, data['costs']) for r in routes)
    all_routes = list(state['routes'])
    all_routes[t] = routes
    pcost = state['pcost'][:]
    delta = value - pcost[t]
    pcost[t] = value
    return {'q': state['q'], 'inv': state['inv'], 'routes': all_routes,
            'pcost': pcost, 'obj': state['obj'] + delta}


def _route_neighbor(state, data, rng, deadline):
    T, m, Q, costs = data['T'], data['m'], data['Q'], data['costs']
    t = rng.randrange(T)
    routes = state['routes'][t]
    choices = [k for k in range(m) if routes[k]]
    if not choices:
        return None
    a = rng.choice(choices)
    ra = routes[a]
    values = state['q'][t]
    loads = [sum(values[c - 1] for c in r) for r in routes]
    mode = rng.random()
    if mode < 0.25:
        improved = _two_opt(ra, costs, deadline)
        if improved == ra:
            return None
        result = list(routes)
        result[a] = improved
        return _state_with_routes(state, t, result, data)
    pa = rng.randrange(len(ra))
    ca = ra[pa]
    qa = values[ca - 1]
    left = ra[pa - 1] if pa else 0
    right = ra[pa + 1] if pa + 1 < len(ra) else 0
    saving = costs[left][ca] + costs[ca][right] - costs[left][right]
    result = None
    best_delta = float('inf')
    if mode < 0.70:
        reduced = ra[:pa] + ra[pa + 1:]
        for b in range(m):
            if loads[b] + (qa if b != a else 0.0) > Q + 1e-7:
                continue
            rb = reduced if a == b else routes[b]
            delta, pos = _insertion(rb, ca, costs)
            delta -= saving
            if delta < best_delta:
                candidate = list(routes)
                candidate[a] = reduced
                candidate[b] = rb[:pos] + [ca] + rb[pos:]
                if candidate != routes:
                    result, best_delta = candidate, delta
    elif mode < 0.90:
        for b in choices:
            if b == a:
                continue
            rb = routes[b]
            for pb, cb in enumerate(rb):
                qb = values[cb - 1]
                if max(loads[a] - qa + qb, loads[b] - qb + qa) > Q + 1e-7:
                    continue
                lb = rb[pb - 1] if pb else 0
                rb_next = rb[pb + 1] if pb + 1 < len(rb) else 0
                delta = (costs[left][cb] + costs[cb][right] - costs[left][ca] - costs[ca][right]
                         + costs[lb][ca] + costs[ca][rb_next] - costs[lb][cb] - costs[cb][rb_next])
                if delta < best_delta:
                    result = list(routes)
                    result[a], result[b] = ra[:], rb[:]
                    result[a][pa], result[b][pb] = cb, ca
                    best_delta = delta
            if time.monotonic() >= deadline:
                break
    elif m > 1:
        b = rng.randrange(m - 1)
        if b >= a:
            b += 1
        rb = routes[b]
        cut_a = rng.randrange(len(ra) + 1)
        prefix_a = sum(values[c - 1] for c in ra[:cut_a])
        prefix_b = 0.0
        for cut_b in range(len(rb) + 1):
            la = prefix_a + loads[b] - prefix_b
            lb = prefix_b + loads[a] - prefix_a
            if max(la, lb) <= Q + 1e-7:
                na = ra[:cut_a] + rb[cut_b:]
                nb = rb[:cut_b] + ra[cut_a:]
                delta = _route_cost(na, costs) + _route_cost(nb, costs) - state['pcost'][t]
                delta += sum(_route_cost(routes[k], costs) for k in range(m) if k != a and k != b)
                if delta < best_delta and (na != ra or nb != rb):
                    result = list(routes)
                    result[a], result[b] = na, nb
                    best_delta = delta
            if cut_b < len(rb):
                prefix_b += values[rb[cut_b] - 1]
    return _state_with_routes(state, t, result, data) if result is not None else None


def _target_options(state, t, i, data):
    routes = state['routes'][t]
    amounts = state['q'][t]
    c = i + 1
    existing = next((k for k, route in enumerate(routes) if c in route), -1)
    choices = []
    for k, route in enumerate(routes):
        if existing >= 0 and k != existing:
            continue
        room = data['Q'] - sum(amounts[j - 1] for j in route)
        if room <= _ACTIVE:
            continue
        delta, pos = (0.0, 0) if existing >= 0 else _insertion(route, c, data['costs'])
        choices.append((delta, k, pos, max(0.0, room)))
    return choices


def _quantity_state(state, changes, inserts, data):
    """Update deliveries, routes and inventories as one transaction, never using an unverified state as the incumbent."""
    q = list(state['q'])
    routes = list(state['routes'])
    pcost = state['pcost'][:]
    periods = {t for t, i, amount in changes}
    for t in periods:
        q[t] = q[t][:]
        routes[t] = [r[:] for r in routes[t]]
    for t, i, amount in changes:
        q[t][i] += amount
        if abs(q[t][i]) < _ACTIVE:
            q[t][i] = 0.0
        if q[t][i] <= _ACTIVE:
            for route in routes[t]:
                if i + 1 in route:
                    route.remove(i + 1)
        elif not any(i + 1 in r for r in routes[t]):
            k, pos = inserts[t, i]
            routes[t][k].insert(pos, i + 1)
    inv = _inventories(q, data)
    if inv is None:
        return None
    for t in periods:
        active = {i + 1 for i, amount in enumerate(q[t]) if amount > _ACTIVE}
        if not _valid_groups(routes[t], active, q[t], data['m'], data['Q']):
            return None
        pcost[t] = sum(_route_cost(r, data['costs']) for r in routes[t])
    return {'q': q, 'inv': inv, 'routes': routes, 'pcost': pcost,
            'obj': sum(pcost) + _holding_cost(inv, data)}


def _schedule_neighbor(state, data, rng, span, deadline):
    T, n = data['T'], data['n']
    if T < 2 or n == 0:
        return None
    i = rng.randrange(n)
    sources = [t for t in range(T) if state['q'][t][i] > _ACTIVE]
    if not sources:
        return None
    rng.shuffle(sources)
    choices = {}
    best = None
    hdiff = data['holding'][i + 1] - data['holding'][0]
    for a in sources:
        saving = _removal_saving(state['routes'][a], i + 1, data['costs'])
        for b in range(max(0, a - span), min(T, a + span + 1)):
            if a == b:
                continue
            maximum = _max_transfer(state, a, b, i, data)
            if maximum <= _ACTIVE:
                continue
            if b not in choices:
                choices[b] = _target_options(state, b, i, data)
            for addition, k, pos, room in choices[b]:
                amount = min(maximum, room)
                if amount <= _ACTIVE:
                    continue
                deleted = amount + _ACTIVE >= state['q'][a][i]
                delta = addition - (saving if deleted else 0.0) + hdiff * amount * (a - b)
                rank = delta + rng.random() * 1e-8
                if best is None or rank < best[0]:
                    best = (rank, a, b, amount, k, pos)
            if time.monotonic() >= deadline:
                break
        if time.monotonic() >= deadline:
            break
    if best is None:
        return None
    _, a, b, amount, k, pos = best
    return _quantity_state(state, [(a, i, -amount), (b, i, amount)], {(b, i): (k, pos)}, data)


def _terminal_neighbor(state, data, rng, deadline):
    T, n = data['T'], data['n']
    if n == 0:
        return None
    i = rng.randrange(n)
    t = rng.randrange(T)
    hdiff = data['holding'][i + 1] - data['holding'][0]
    q = state['q'][t][i]
    best = None
    if q > _ACTIVE:
        amount = q
        for u in range(t, T):
            amount = min(amount, state['inv'][u][i + 1], data['caps'][0] - state['inv'][u][0])
        if amount > _ACTIVE:
            saving = _removal_saving(state['routes'][t], i + 1, data['costs']) if amount + _ACTIVE >= q else 0.0
            best = (-saving - amount * hdiff * (T - t), -amount, None)
    if hdiff < 0.0:
        amount = data['Q'] - q
        for u in range(t, T):
            before = state['inv'][u][i + 1] + data['demands'][i][u]
            amount = min(amount, state['inv'][u][0], data['caps'][i + 1] - before)
        if amount > _ACTIVE:
            for extra, k, pos, room in _target_options(state, t, i, data):
                delta = min(amount, room)
                rank = extra + delta * hdiff * (T - t)
                if delta > _ACTIVE and (best is None or rank < best[0]):
                    best = (rank, delta, (k, pos))
    if best is None:
        return None
    _, amount, insertion = best
    return _quantity_state(state, [(t, i, amount)], {(t, i): insertion}, data)
def _flow_neighbor(state, data, rng, strength, deadline, polish=False):
    """Route removal and replenishment extension jointly define the network-flow neighbourhood, rather than picking two separate algorithms."""
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    costs = data['costs']
    if n == 0 or time.monotonic() >= deadline:
        return None
    selected = set()
    banned = set()
    back = None
    if not polish:
        removable = []
        for t in range(T):
            for i in range(n):
                amount = state['q'][t][i]
                if amount <= _ACTIVE:
                    continue
                cover = (data['initial'][i + 1] if t == 0 else
                         data['caps'][i + 1] - data['demands'][i][t - 1])
                if cover + 1e-7 < data['demands'][i][t]:
                    continue
                saving = _removal_saving(state['routes'][t], i + 1, costs)
                key = (saving + 1.0) / max(amount, 1e-6) ** 0.35
                removable.append((key * rng.random() ** 0.35, t, i))
        if removable and rng.random() < 0.72:
            _, t, center = max(removable)
            banned.add((t, center))
        else:
            center = rng.randrange(n)
        count = min(n, 1 + int(rng.random() * strength))
        near = sorted(range(n), key=lambda i: costs[center + 1][i + 1] * (0.6 + rng.random()))
        selected.update(near[:count])
        selected.add(center)
        if rng.random() < 0.25:
            back = _backward_assignment(data, rng, True, deadline)
        if banned and count > 1 and rng.random() < 0.2:
            for _, t, i in sorted(removable, reverse=True):
                if i in selected and i != center:
                    banned.add((t, i))
                    break
    assignment = [[0] * n for _ in range(T)]
    upper = [[0.0] * n for _ in range(T)]
    bias = [[0.0] * n for _ in range(T)]
    route_hints = []
    weight = 0.0 if polish else 0.35 + 1.3 * rng.random()
    for t in range(T):
        if time.monotonic() >= deadline:
            return None
        owner = [-1] * n
        period = state['routes'][t]
        loads = [sum(state['q'][t][c - 1] for c in r) for r in period]
        for k, route in enumerate(period):
            for c in route:
                owner[c - 1] = k
        for i in range(n):
            amount = state['q'][t][i]
            k = owner[i]
            if k >= 0:
                assignment[t][i] = k
                upper[t][i] = Q
                if not polish:
                    saving = max(0.0, _removal_saving(period, i + 1, costs))
                    bias[t][i] = weight * saving / max(amount, 1e-8)
            elif i in selected:
                options = []
                for k in range(m):
                    extra, pos = _insertion(period[k], i + 1, costs)
                    preference = extra * (0.85 + 0.3 * rng.random())
                    preference += costs[0][i + 1] * 0.08 * loads[k] / max(Q, 1e-9)
                    if back is not None and k == back[0][t][i]:
                        preference *= 0.65
                    options.append((preference, extra, k))
                _, extra, k = min(options)
                assignment[t][i] = k
                upper[t][i] = Q
                horizon = min(T, t + 1 + max(1, int(strength)))
                desired = min(Q, data['caps'][i + 1], sum(data['demands'][i][t:horizon]))
                if back is not None:
                    desired = min(Q, data['caps'][i + 1], max(desired, back[1][t][i] + data['demands'][i][t]))
                bias[t][i] = weight * max(0.0, extra) / max(desired, Q * 0.03, 1e-8)
            if (t, i) in banned:
                upper[t][i] = 0.0
        route_hints.append([r[:] for r in period])
    q = _flow_schedule(data, assignment, True, deadline, bias, upper)
    if q is None:
        return None
    for t in range(T):
        route_hints[t] = [[c for c in route if q[t][c - 1] > _ACTIVE]
                          for route in route_hints[t]]
        placed = {c for route in route_hints[t] for c in route}
        for i, amount in enumerate(q[t]):
            if amount > _ACTIVE and i + 1 not in placed:
                route_hints[t][assignment[t][i]].append(i + 1)
    return _state_from_schedule(q, data, deadline, route_hints=route_hints)


def _greedy_seed(data, backward, deadline):
    """Combine reverse safety stock with forward forced dispatch, preferring to form a complete solution that can be kept."""
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    floors = backward[1] if backward is not None else [[0.0] * n for _ in range(T)]
    stocks = data['initial'][1:]
    depot = data['initial'][0]
    q = []
    hints = []
    for t in range(T):
        if time.monotonic() >= deadline:
            return None
        amounts = [max(0.0, data['demands'][i][t] + floors[t][i] - stocks[i]) for i in range(n)]
        active = {i + 1 for i in range(n) if amounts[i] > _ACTIVE}
        groups = _ffd_groups(active, amounts, m, Q)
        if groups is None and backward is not None:
            groups = [[] for _ in range(m)]
            for c in active:
                groups[backward[0][t][c - 1]].append(c)
            if not _valid_groups(groups, active, amounts, m, Q):
                groups = None
        if groups is None:
            return None
        available = depot + data['production'][t]
        outbound = sum(amounts)
        if outbound > available + 1e-7:
            return None
        forced = max(0.0, available - outbound - data['caps'][0])
        loads = [sum(amounts[c - 1] for c in route) for route in groups]
        owner = [-1] * n
        for k, route in enumerate(groups):
            for c in route:
                owner[c - 1] = k
        ids = sorted(range(n), key=lambda i: (owner[i] < 0, data['holding'][i + 1], -sum(data['demands'][i][t:])))
        for i in ids:
            if forced <= _ACTIVE:
                break
            room = min(Q - amounts[i], data['caps'][i + 1] - stocks[i] - amounts[i])
            k = owner[i]
            if k < 0:
                k = min(range(m), key=lambda z: loads[z])
            addition = min(forced, room, Q - loads[k])
            if addition > _ACTIVE:
                if owner[i] < 0:
                    groups[k].append(i + 1)
                    owner[i] = k
                amounts[i] += addition
                loads[k] += addition
                outbound += addition
                forced -= addition
        if forced > 1e-7:
            return None
        depot = available - outbound
        stocks = [stocks[i] + amounts[i] - data['demands'][i][t] for i in range(n)]
        q.append(amounts)
        hints.append(groups)
    return _state_from_schedule(q, data, deadline, route_hints=hints)


def _integer_feasibility(data, deadline):
    """On construction failure, repair with a delivery-vehicle assignment mixed-integer model, without introducing travel-arc variables."""
    try:
        import numpy as np
        from scipy.optimize import milp, Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
        if time.monotonic() >= deadline:
            return None
        T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
        size_q = T * m * n
        offset_inv = 2 * size_q
        count = offset_inv + T * (n + 1)
        lower = np.zeros(count)
        upper = np.full(count, np.inf)
        upper[:size_q] = Q
        upper[size_q:offset_inv] = 1.0
        objective = np.zeros(count)
        integrality = np.zeros(count, dtype=np.uint8)
        integrality[size_q:offset_inv] = 1
        rows, cols, vals, low, high = [], [], [], [], []

        def constraint(terms, a, b):
            row = len(low)
            low.append(a)
            high.append(b)
            for col, value in terms:
                rows.append(row)
                cols.append(col)
                vals.append(value)

        def quantity(t, k, i):
            return (t * m + k) * n + i

        def inventory(t, i):
            return offset_inv + t * (n + 1) + i

        for t in range(T):
            if time.monotonic() >= deadline:
                return None
            for i in range(n + 1):
                index = inventory(t, i)
                upper[index] = data['caps'][i] - (data['demands'][i - 1][t] if i else 0.0)
                objective[index] = data['holding'][i]
                terms = [(index, 1.0)]
                if t:
                    terms.append((inventory(t - 1, i), -1.0))
                if i:
                    terms.extend((quantity(t, k, i - 1), -1.0) for k in range(m))
                    rhs = -data['demands'][i - 1][t]
                else:
                    terms.extend((quantity(t, k, j), 1.0) for k in range(m) for j in range(n))
                    rhs = data['production'][t]
                if not t:
                    rhs += data['initial'][i]
                constraint(terms, rhs, rhs)
            for k in range(m):
                constraint([(quantity(t, k, i), 1.0) for i in range(n)], -np.inf, Q)
                if k:
                    terms = [(quantity(t, k, i), 1.0) for i in range(n)]
                    terms.extend((quantity(t, k - 1, i), -1.0) for i in range(n))
                    constraint(terms, -np.inf, 0.0)
                for i in range(n):
                    x = quantity(t, k, i)
                    constraint([(x, 1.0), (size_q + x, -min(Q, data['caps'][i + 1]))], -np.inf, 0.0)
                    objective[size_q + x] = data['costs'][0][i + 1] / max(1, n)
            for i in range(n):
                constraint([(size_q + quantity(t, k, i), 1.0) for k in range(m)], -np.inf, 1.0)
        if time.monotonic() >= deadline:
            return None
        matrix = coo_matrix((vals, (rows, cols)), shape=(len(low), count)).tocsc()
        result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                      constraints=LinearConstraint(matrix, np.array(low), np.array(high)),
                      options={'time_limit': max(0.001, (deadline - time.monotonic()) * 0.87),
                               'mip_rel_gap': 1.0, 'presolve': True})
        if result.x is None:
            return None
        assignment = [[0] * n for _ in range(T)]
        support = [[0.0] * n for _ in range(T)]
        for t in range(T):
            for i in range(n):
                k = max(range(m), key=lambda z: result.x[size_q + quantity(t, z, i)])
                assignment[t][i] = k
                if result.x[size_q + quantity(t, k, i)] > 0.5:
                    support[t][i] = Q
        # Then use a continuous network flow to remove the integer solver's numerical error.
        q = _flow_schedule(data, assignment, False, deadline, upper=support)
        if q is None:
            q = [[sum(max(0.0, float(result.x[quantity(t, k, i)])) for k in range(m))
                  for i in range(n)] for t in range(T)]
        return _state_from_schedule(q, data, deadline, assignment)
    except BaseException:
        return None


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.monotonic()
    safe = _blank_solution(instance)
    best = None
    data = None
    try:
        budget = float(time_limit_s)
        if not math.isfinite(budget):
            budget = 0.0
        budget = max(0.0, budget)
        T, n, m = (int(instance[key]) for key in ('num_periods', 'num_customers', 'num_vehicles'))
        if T <= 0:
            return safe
        Q = float(instance['vehicle_capacity'])
        data = {'T': T, 'n': n, 'm': m, 'Q': Q,
                'costs': [[float(x) for x in row] for row in instance['travel_costs']],
                'production': [float(x) for x in instance['production']],
                'demands': [[float(x) for x in row] for row in instance['demands']],
                'caps': [float(x) for x in instance['inventory_capacities']],
                'initial': [float(x) for x in instance['initial_inventories']],
                'holding': [float(x) for x in instance['holding_costs']]}
        reserve = max(0.001, budget * 0.012)
        deadline = start + max(0.0, budget - reserve)
        rng = random.Random(_seed_from_data(data))
        zero = [[0.0] * n for _ in range(T)]
        inv = _inventories(zero, data)
        if inv is not None:
            best = {'q': zero, 'inv': inv, 'routes': [[[] for _ in range(m)] for _ in range(T)],
                    'pcost': [0.0] * T, 'obj': _holding_cost(inv, data)}
        if n == 0 or m <= 0 or Q <= 0.0:
            return _to_output(best, data) if best is not None else safe
        strength = max(1.0, math.log1p(budget) * math.sqrt(1.0 + n / max(1, T)))
        cycle = budget / max(4.0, 3.0 + 2.0 * math.sqrt(1.0 + budget) + math.log1p(n * T))
        init_deadline = min(deadline, start + budget * 0.18)
        back = None
        attempt = 0
        initial_assignment = None
        while best is None and time.monotonic() < init_deadline:
            now = time.monotonic()
            until = min(init_deadline, now + max(cycle * 0.7, (init_deadline - now) * 0.2))
            mode = attempt % 4
            candidate = None
            if mode == 0:
                assignment = _make_assignment(data, rng, 'dynamic' if attempt == 0 else 'random')
            elif mode == 1:
                back = _backward_assignment(data, rng, attempt > 4, until)
                assignment = back[0] if back is not None else _make_assignment(data, rng, 'static')
                candidate = _greedy_seed(data, back, until)
            elif mode == 2:
                assignment = None
            else:
                assignment = _make_assignment(data, rng, 'random')
            if candidate is None and time.monotonic() < until:
                q = _flow_schedule(data, assignment, False, until)
                if q is not None:
                    candidate = _state_from_schedule(q, data, until, assignment)
            if candidate is not None:
                best = candidate
                initial_assignment = assignment
            attempt += 1
        if best is None and time.monotonic() < deadline:
            best = _integer_feasibility(data, min(deadline, start + budget * 0.73))
        if best is not None:
            safe = _to_output(best, data)
        current = best
        if best is not None and time.monotonic() < deadline:
            until = min(deadline, time.monotonic() + max(cycle, budget * 0.09))
            if initial_assignment is not None:
                q = _flow_schedule(data, initial_assignment, True, until)
                candidate = _state_from_schedule(q, data, until, initial_assignment) if q is not None else None
            else:
                candidate = _flow_neighbor(best, data, rng, strength, until, True)
            if candidate is not None and candidate['obj'] < best['obj']:
                best = candidate
                current = candidate
        next_flow = time.monotonic()
        next_shake = time.monotonic() + cycle
        flow_duration = cycle * 0.3
        while time.monotonic() < deadline:
            try:
                now = time.monotonic()
                elapsed = now - start
                remaining = deadline - now
                fraction = min(1.0, elapsed / max(budget, 1e-9))
                if best is None:
                    back = _backward_assignment(data, rng, True, min(deadline, now + cycle))
                    assignment = back[0] if back is not None else _make_assignment(data, rng, 'random')
                    q = _flow_schedule(data, assignment, False, min(deadline, now + cycle))
                    candidate = _state_from_schedule(q, data, deadline, assignment) if q is not None else None
                    if candidate is not None:
                        best = current = candidate
                        safe = _to_output(best, data)
                    continue
                candidate = None
                if now >= next_flow:
                    duration = max(cycle * 0.7, flow_duration * 1.8)
                    until = min(deadline, now + min(remaining * 0.45, duration))
                    if rng.random() < 0.14:
                        back = _backward_assignment(data, rng, True, until)
                        if back is not None:
                            bias = [[2.0 * data['costs'][0][i + 1] / max(Q, 1e-8) *
                                     (0.2 + 1.2 * rng.random()) for i in range(n)] for t in range(T)]
                            q = _flow_schedule(data, back[0], True, until, bias)
                            if q is not None:
                                candidate = _state_from_schedule(q, data, until, back[0])
                    else:
                        candidate = _flow_neighbor(current, data, rng, strength * (0.5 + fraction),
                                                   until, rng.random() < 0.17)
                    flow_duration = 0.7 * flow_duration + 0.3 * (time.monotonic() - now)
                    next_flow = time.monotonic() + cycle * (0.25 + 0.5 * rng.random())
                else:
                    until = min(deadline, now + max(cycle * 0.2, budget / max(1, n * T) * 0.03))
                    draw = rng.random()
                    if draw < 0.45:
                        span = min(T, 1 + int(math.sqrt(1.0 + budget) * (0.7 + fraction)))
                        candidate = _schedule_neighbor(current, data, rng, span, until)
                    elif draw < 0.55:
                        candidate = _terminal_neighbor(current, data, rng, until)
                    else:
                        candidate = _route_neighbor(current, data, rng, until)
                if candidate is not None:
                    delta = candidate['obj'] - current['obj']
                    visit_count = sum(len(r) for period in best['routes'] for r in period)
                    travel_scale = max(1.0, sum(best['pcost']) / max(1, visit_count))
                    phase = ((now - start) / max(cycle * 2.5, 1e-9)) % 1.0
                    temperature = travel_scale * (0.005 + 0.07 * (1.0 - phase)) * (1.0 - 0.65 * fraction)
                    if delta <= 1e-9 or rng.random() < math.exp(-min(700.0, max(0.0, delta) / max(temperature, 1e-10))):
                        current = candidate
                    if candidate['obj'] + 1e-8 < best['obj']:
                        best = candidate
                now = time.monotonic()
                if now >= next_shake and now < deadline:
                    current = best
                    depth = min(max(1, n), 1 + int(math.log1p(budget) * (0.5 + fraction)))
                    shake_until = min(deadline, now + cycle * 0.08)
                    for _ in range(depth):
                        if time.monotonic() >= shake_until:
                            break
                        t = rng.randrange(T)
                        routes = _one_route_move(current['routes'][t], current['q'][t], data, rng)
                        if routes is not None:
                            current = _state_with_routes(current, t, routes, data)
                    next_shake = time.monotonic() + cycle * (2.0 + rng.random())
            except BaseException:
                current = best
        if best is not None:
            return _to_output(best, data)
        return safe
    except BaseException:
        if best is not None and data is not None:
            try:
                return _to_output(best, data)
            except BaseException:
                pass
        return safe
