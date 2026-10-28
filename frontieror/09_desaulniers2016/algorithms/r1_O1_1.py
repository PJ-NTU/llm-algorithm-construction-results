"""Full-horizon inventory re-optimisation and route variable neighbourhood search
Builds a feasible solution by network flow, jointly adjusts deliveries with small-scale visit integer neighbourhoods and global continuous replenishment re-optimisation, and improves travel cost with route relocation, swap and tail exchange.
"""
import time
import math
import random
from collections import deque

try:
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    _HAS_MIP = True
except Exception:
    _HAS_MIP = False

_ACTIVE = 1e-9
_TOL = 1e-8

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

def _build_flow_graph(data, assignment, delivery_bias):
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
                    graph, gates[t][0], cin[i][t], Q, bias
                )
        else:
            for k in range(m):
                _add_edge(graph, depot[t], gates[t][k], Q)
            for i in range(n):
                k = assignment[t][i]
                if k < 0:
                    continue
                bias = 0.0 if delivery_bias is None else delivery_bias[t][i]
                delivery_refs[t][i] = _add_edge(
                    graph, gates[t][k], cin[i][t], Q, bias
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

def _dinic(graph, source, sink, target, deadline):
    count = len(graph)
    flow = 0.0
    while flow + _TOL < target and time.monotonic() < deadline:
        level = [-1] * count
        level[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for edge in graph[u]:
                if edge[2] > _TOL and level[edge[0]] < 0:
                    level[edge[0]] = level[u] + 1
                    queue.append(edge[0])
        if level[sink] < 0:
            break
        work = [0] * count

        def send(u, amount):
            if u == sink:
                return amount
            if time.monotonic() >= deadline:
                return 0.0
            while work[u] < len(graph[u]):
                edge = graph[u][work[u]]
                if edge[2] > _TOL and level[edge[0]] == level[u] + 1:
                    pushed = send(edge[0], min(amount, edge[2]))
                    if pushed > _TOL:
                        edge[2] -= pushed
                        graph[edge[0]][edge[1]][2] += pushed
                        return pushed
                work[u] += 1
            return 0.0

        while flow + _TOL < target and time.monotonic() < deadline:
            pushed = send(source, target - flow)
            if pushed <= _TOL:
                break
            flow += pushed
    return flow

def _min_cost_flow(graph, source, sink, target, deadline):
    count = len(graph)
    flow = 0.0
    while flow + _TOL < target and time.monotonic() < deadline:
        dist = [float("inf")] * count
        prev_node = [-1] * count
        prev_edge = [-1] * count
        in_queue = [False] * count
        queue = deque([source])
        dist[source] = 0.0
        in_queue[source] = True

        while queue and time.monotonic() < deadline:
            u = queue.popleft()
            in_queue[u] = False
            base = dist[u]
            for ei, edge in enumerate(graph[u]):
                if edge[2] <= _TOL:
                    continue
                v = edge[0]
                nd = base + edge[3]
                if nd + 1e-12 < dist[v]:
                    dist[v] = nd
                    prev_node[v] = u
                    prev_edge[v] = ei
                    if not in_queue[v]:
                        in_queue[v] = True
                        queue.append(v)

        if prev_node[sink] < 0 or time.monotonic() >= deadline:
            break
        amount = target - flow
        v = sink
        while v != source:
            u = prev_node[v]
            edge = graph[u][prev_edge[v]]
            amount = min(amount, edge[2])
            v = u
        if amount <= _TOL:
            break
        v = sink
        while v != source:
            u = prev_node[v]
            edge = graph[u][prev_edge[v]]
            edge[2] -= amount
            graph[v][edge[1]][2] += amount
            v = u
        flow += amount
    return flow

def _flow_schedule(data, assignment, use_cost, deadline, delivery_bias=None):
    built = _build_flow_graph(data, assignment, delivery_bias)
    if built is None:
        return None
    graph, source, sink, target, refs = built
    if target <= _TOL:
        return [[0.0 for _ in range(data["n"])] for _ in range(data["T"])]
    if use_cost:
        flow = _min_cost_flow(graph, source, sink, target, deadline)
    else:
        flow = _dinic(graph, source, sink, target, deadline)
    if target - flow > 1e-7:
        return None
    q = [[0.0 for _ in range(data["n"])] for _ in range(data["T"])]
    for t in range(data["T"]):
        for i in range(data["n"]):
            edge = refs[t][i]
            value = 0.0 if edge is None else edge[4] - edge[2]
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

    return groups if search(0) else None

def _cw_groups(active, amounts, m, Q, costs, deadline):
    if not active:
        return [[] for _ in range(m)]
    routes = {customer: [customer] for customer in active}
    loads = {customer: amounts[customer - 1] for customer in active}
    where = {customer: customer for customer in active}
    savings = []
    ordered = sorted(active)
    for p, i in enumerate(ordered):
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

def _order_group(group, costs, deadline):
    if len(group) < 3:
        return list(group)
    remaining = set(group)
    route = []
    last = 0
    while remaining and time.monotonic() < deadline:
        customer = min(remaining, key=lambda c: costs[last][c])
        route.append(customer)
        remaining.remove(customer)
        last = customer
    route.extend(sorted(remaining))
    return _two_opt(route, costs, deadline)


def _routes_for_amounts(amounts, data, deadline, hint=None):
    m, Q, costs = data['m'], data['Q'], data['costs']
    active = {i + 1 for i, q in enumerate(amounts) if q > _ACTIVE}
    if not active:
        return [[] for _ in range(m)], 0.0
    if m == 0 or any(amounts[c - 1] > Q + 1e-7 for c in active):
        return None
    candidates = []
    for groups in (_hint_groups(hint, active, amounts, m, Q, costs),
                   _ffd_groups(active, amounts, m, Q)):
        if _valid_groups(groups, active, amounts, m, Q):
            candidates.append(groups)
    if time.monotonic() < deadline:
        groups = _cw_groups(active, amounts, m, Q, costs, deadline)
        if _valid_groups(groups, active, amounts, m, Q):
            candidates.append(groups)
    if not candidates and time.monotonic() < deadline:
        groups = _exact_groups(active, amounts, m, Q, deadline)
        if _valid_groups(groups, active, amounts, m, Q):
            candidates.append(groups)
    if not candidates:
        return None
    best_routes, best_cost = None, float('inf')
    for groups in candidates:
        routes = []
        for group in groups:
            route = list(group)
            if time.monotonic() < deadline:
                ordered = _order_group(group, costs, deadline)
                if _route_cost(ordered, costs) < _route_cost(route, costs):
                    route = ordered
            routes.append(route)
        value = sum(_route_cost(r, costs) for r in routes)
        if value < best_cost:
            best_routes, best_cost = routes, value
    return best_routes, best_cost


def _state(q, routes, data):
    inv = _inventories(q, data)
    if inv is None:
        return None
    cleaned = []
    pcost = []
    for t, period in enumerate(routes):
        if len(period) != data['m']:
            return None
        seen = set()
        out = []
        for route in period:
            r = []
            load = 0.0
            for c in route:
                if c < 1 or c > data['n'] or c in seen:
                    return None
                seen.add(c)
                if q[t][c - 1] > _ACTIVE:
                    load += q[t][c - 1]
                    r.append(c)
            if load > data['Q'] + 1e-7:
                return None
            out.append(r)
        if any(q[t][i] > _ACTIVE and i + 1 not in seen for i in range(data['n'])):
            return None
        cleaned.append(out)
        pcost.append(sum(_route_cost(r, data['costs']) for r in out))
    return {'q': q, 'inv': inv, 'routes': cleaned, 'pcost': pcost,
            'obj': sum(pcost) + _holding_cost(inv, data)}


def _state_from_schedule(q, data, deadline, assignment=None):
    if _inventories(q, data) is None:
        return None
    routes = []
    for t in range(data['T']):
        hint = None
        if assignment is not None:
            hint = [[] for _ in range(data['m'])]
            for i in range(data['n']):
                if assignment[t][i] >= 0 and q[t][i] > _ACTIVE:
                    hint[assignment[t][i]].append(i + 1)
        built = _routes_for_amounts(q[t], data, deadline, hint)
        if built is None:
            return None
        routes.append(built[0])
    return _state(q, routes, data)


def _insertion(route, customer, costs):
    best, where = float('inf'), 0
    a = 0
    for pos in range(len(route) + 1):
        b = 0 if pos == len(route) else route[pos]
        value = costs[a][customer] + costs[customer][b] - costs[a][b]
        if value < best:
            best, where = value, pos
        a = b
    return best, where


def _route_descent(routes, amounts, data, deadline):
    routes = [r[:] for r in routes]
    costs, Q, m = data['costs'], data['Q'], data['m']
    loads = [sum(amounts[c - 1] for c in r) for r in routes]
    while time.monotonic() < deadline:
        changed = False
        for k in range(m):
            route = _two_opt(routes[k], costs, deadline)
            if _route_cost(route, costs) + 1e-8 < _route_cost(routes[k], costs):
                routes[k] = route
                changed = True
        # The full relocation neighbourhood also allows merging one route into other routes.
        move, gain = None, -1e-8
        for a, route in enumerate(routes):
            if time.monotonic() >= deadline:
                break
            for p, customer in enumerate(route):
                if time.monotonic() >= deadline:
                    break
                prev = 0 if p == 0 else route[p - 1]
                nxt = 0 if p + 1 == len(route) else route[p + 1]
                removal = costs[prev][nxt] - costs[prev][customer] - costs[customer][nxt]
                for b in range(m):
                    if a != b and loads[b] + amounts[customer - 1] > Q + 1e-8:
                        continue
                    target = route[:p] + route[p + 1:] if a == b else routes[b]
                    insertion, pos = _insertion(target, customer, costs)
                    delta = removal + insertion
                    if delta < gain:
                        move, gain = (a, p, b, pos, customer), delta
        if move is not None:
            a, p, b, pos, customer = move
            routes[a].pop(p)
            routes[b].insert(pos, customer)
            loads[a] -= amounts[customer - 1]
            loads[b] += amounts[customer - 1]
            continue
        # Customer swaps between different vehicles.
        move, gain = None, -1e-8
        for a in range(m):
            if time.monotonic() >= deadline:
                break
            for b in range(a + 1, m):
                ra, rb = routes[a], routes[b]
                for p, ca in enumerate(ra):
                    if time.monotonic() >= deadline:
                        break
                    ap = 0 if p == 0 else ra[p - 1]
                    an = 0 if p + 1 == len(ra) else ra[p + 1]
                    for v, cb in enumerate(rb):
                        weight = amounts[cb - 1] - amounts[ca - 1]
                        if loads[a] + weight > Q + 1e-8 or loads[b] - weight > Q + 1e-8:
                            continue
                        bp = 0 if v == 0 else rb[v - 1]
                        bn = 0 if v + 1 == len(rb) else rb[v + 1]
                        delta = (costs[ap][cb] + costs[cb][an] - costs[ap][ca] - costs[ca][an]
                                 + costs[bp][ca] + costs[ca][bn] - costs[bp][cb] - costs[cb][bn])
                        if delta < gain:
                            move, gain = (a, p, b, v, weight), delta
        if move is not None:
            a, p, b, v, weight = move
            routes[a][p], routes[b][v] = routes[b][v], routes[a][p]
            loads[a] += weight
            loads[b] -= weight
            continue
        # Exchange the tails of two routes to reassign several customers in one move.
        move, gain = None, -1e-8
        prefixes = []
        for r in routes:
            cumulative = [0.0]
            for c in r:
                cumulative.append(cumulative[-1] + amounts[c - 1])
            prefixes.append(cumulative)
        for a in range(m):
            if time.monotonic() >= deadline:
                break
            for b in range(a + 1, m):
                ra, rb = routes[a], routes[b]
                for p in range(len(ra) + 1):
                    if time.monotonic() >= deadline:
                        break
                    ap = 0 if p == 0 else ra[p - 1]
                    an = 0 if p == len(ra) else ra[p]
                    for v in range(len(rb) + 1):
                        la = prefixes[a][p] + loads[b] - prefixes[b][v]
                        lb = prefixes[b][v] + loads[a] - prefixes[a][p]
                        if la > Q + 1e-8 or lb > Q + 1e-8:
                            continue
                        bp = 0 if v == 0 else rb[v - 1]
                        bn = 0 if v == len(rb) else rb[v]
                        delta = costs[ap][bn] + costs[bp][an] - costs[ap][an] - costs[bp][bn]
                        if delta < gain:
                            move, gain = (a, p, b, v, la, lb), delta
        if move is not None:
            a, p, b, v, la, lb = move
            ra, rb = routes[a], routes[b]
            routes[a], routes[b] = ra[:p] + rb[v:], rb[:v] + ra[p:]
            loads[a], loads[b] = la, lb
            continue
        if not changed:
            break
    return routes


def _polish(state, data, deadline, rng, shake=False):
    routes = list(state['routes'])
    period_costs = state['pcost'][:]
    periods = list(range(data['T']))
    rng.shuffle(periods)
    for ix, t in enumerate(periods):
        now = time.monotonic()
        if now >= deadline:
            break
        stop = now + (deadline - now) / max(1, len(periods) - ix)
        base = routes[t]
        if shake:
            moved = _one_route_move(base, state['q'][t], data, rng)
            if moved is not None:
                base = moved
        proposed = _route_descent(base, state['q'][t], data, stop)
        value = sum(_route_cost(r, data['costs']) for r in proposed)
        if value < period_costs[t] - 1e-8 or shake:
            routes[t], period_costs[t] = proposed, value
    return {'q': state['q'], 'inv': state['inv'], 'routes': routes,
            'pcost': period_costs,
            'obj': state['obj'] + sum(period_costs) - sum(state['pcost'])}


def _select_customers(state, data, rng, strength, step):
    n, T, costs = data['n'], data['T'], data['costs']
    scores = [0.0] * n
    for period in state['routes']:
        for r in period:
            for p, c in enumerate(r):
                a = 0 if p == 0 else r[p - 1]
                b = 0 if p + 1 == len(r) else r[p + 1]
                scores[c - 1] += max(0.0, costs[a][c] + costs[c][b] - costs[a][b])
    ranking = sorted(range(n), key=lambda i: -scores[i])
    seed = ranking[step % n] if rng.random() < 0.7 else rng.randrange(n)
    if rng.random() < 0.6:
        return [seed]
    count = rng.randint(2, max(2, strength))
    if rng.random() < 0.35:
        t = rng.randrange(T)
        group = next((r for r in state['routes'][t] if seed + 1 in r), [])
        selected = [c - 1 for c in group]
        if selected:
            rng.shuffle(selected)
            return selected[:count]
    neighbors = sorted((i for i in range(n) if i != seed),
                       key=lambda i: costs[i + 1][seed + 1] * (0.8 + 0.4 * rng.random()))
    return [seed] + neighbors[:count - 1]


def _joint_neighborhood(state, selected, data, deadline, rng, effort):
    """Create visit binary variables only for the selected customers, and re-optimise inventories and existing deliveries of all customers together."""
    if not _HAS_MIP or time.monotonic() >= deadline:
        return None
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    selected = set(selected)
    costs, caps, demands = data['costs'], data['caps'], data['demands']
    old_routes = ([[[] for _ in range(m)] for _ in range(T)]
                  if state is None else state['routes'])
    base = [[[c for c in r if c - 1 not in selected] for r in p] for p in old_routes]
    owners = [[-1] * n for _ in range(T)]
    for t, period in enumerate(old_routes):
        for k, route in enumerate(period):
            for c in route:
                owners[t][c - 1] = k

    row_lo, row_hi = [], []
    for t in range(T):
        for i in range(n + 1):
            rhs = data['production'][t] if i == 0 else -demands[i - 1][t]
            if t == 0:
                rhs += data['initial'][i]
            row_lo.append(rhs)
            row_hi.append(rhs)
    route_offset = len(row_lo)
    row_lo.extend([-float('inf')] * (T * m))
    row_hi.extend([Q] * (T * m))
    obj, var_hi, integrality = [], [], []
    row_index, col_index, coeffs = [], [], []

    def variable(cost, upper, integer=0):
        index = len(obj)
        obj.append(float(cost))
        var_hi.append(max(0.0, float(upper)))
        integrality.append(integer)
        return index

    def coefficient(row, column, value):
        row_index.append(row)
        col_index.append(column)
        coeffs.append(float(value))

    def constraint(lower, upper):
        index = len(row_lo)
        row_lo.append(lower)
        row_hi.append(upper)
        return index

    # Inventory variables enter the per-period holding cost directly; the customer upper bound is an equivalent form of the pre-receipt capacity constraint.
    for t in range(T):
        for i in range(n + 1):
            upper = caps[i] if i == 0 else caps[i] - demands[i - 1][t]
            if upper < -1e-7:
                return None
            col = variable(data['holding'][i], upper)
            coefficient(t * (n + 1) + i, col, 1.0)
            if t + 1 < T:
                coefficient((t + 1) * (n + 1) + i, col, -1.0)

    delivery_columns = []
    chosen = sorted(selected)
    # Empty routes use an anchor activation cost, avoiding misestimating a jointly added route as several separate round trips.
    anchors = []
    if len(chosen) > 1:
        anchors = [max(chosen, key=lambda i: costs[0][i + 1])]
        while len(anchors) < min(m, len(chosen)):
            other = max((i for i in chosen if i not in anchors),
                        key=lambda i: min(costs[i + 1][a + 1] for a in anchors))
            anchors.append(other)
            if time.monotonic() >= deadline:
                return None
    activation = {}
    unique_rows = {}
    alternative_count = min(m, max(2, int(2 + math.log1p(effort))))

    for t in range(T):
        if time.monotonic() >= deadline:
            return None
        for i in range(n):
            if i in selected:
                unique_rows[t, i] = constraint(-float('inf'), 1.0)
                prices = []
                for k in range(m):
                    if not base[t][k] and len(chosen) > 1:
                        old = [c - 1 for c in old_routes[t][k] if c - 1 in selected]
                        anchor = max(old, key=lambda j: costs[0][j + 1]) if old else anchors[k % len(anchors)]
                        a = anchor + 1
                        price = max(0.0, costs[0][i + 1] + costs[i + 1][a] - costs[0][a])
                        if (t, k) not in activation:
                            z = variable(2.0 * costs[0][a], 1.0, 1)
                            coefficient(route_offset + t * m + k, z, -Q)
                            row_hi[route_offset + t * m + k] = 0.0
                            activation[t, k] = z
                        ranking_price = price + 2.0 * costs[0][a] / max(1, len(chosen) / max(1, m))
                    else:
                        price = max(0.0, _insertion(base[t][k], i + 1, costs)[0])
                        ranking_price = price
                    prices.append((ranking_price, k, price))
                prices.sort()
                available = prices if state is None else prices[:alternative_count]
                owner = owners[t][i]
                if owner >= 0 and not any(p[1] == owner for p in available):
                    available = available + [next(p for p in prices if p[1] == owner)]
            else:
                k = owners[t][i]
                available = [] if k < 0 else [(0.0, k, 0.0)]
            for _, k, price in available:
                upper = min(Q, caps[i + 1])
                col = variable(0.0, upper)
                coefficient(t * (n + 1), col, 1.0)
                coefficient(t * (n + 1) + i + 1, col, -1.0)
                coefficient(route_offset + t * m + k, col, 1.0)
                delivery_columns.append((col, t, k, i))
                if i in selected:
                    y = variable(price, 1.0, 1)
                    coefficient(unique_rows[t, i], y, 1.0)
                    link = constraint(-float('inf'), 0.0)
                    coefficient(link, col, 1.0)
                    coefficient(link, y, -upper)

    remaining = deadline - time.monotonic()
    if remaining < 0.002:
        return None
    try:
        matrix = coo_matrix((np.asarray(coeffs, dtype=float),
                             (np.asarray(row_index, dtype=np.int32),
                              np.asarray(col_index, dtype=np.int32))),
                            shape=(len(row_lo), len(obj))).tocsc()
        remaining = deadline - time.monotonic()
        if remaining <= 0.001:
            return None
        result = milp(np.asarray(obj), integrality=np.asarray(integrality, dtype=np.uint8),
                      bounds=Bounds(np.zeros(len(obj)), np.asarray(var_hi)),
                      constraints=LinearConstraint(matrix, np.asarray(row_lo), np.asarray(row_hi)),
                      options={'time_limit': max(0.001, remaining * 0.92),
                               'mip_rel_gap': 0.0001 if len(selected) <= 1 else 0.002,
                               'presolve': True})
        if result.x is None or not np.all(np.isfinite(result.x)):
            return None
        values = result.x
    except Exception:
        return None

    q = [[0.0] * n for _ in range(T)]
    new_owner = [[-1] * n for _ in range(T)]
    for col, t, k, i in delivery_columns:
        amount = max(0.0, float(values[col]))
        if amount <= _ACTIVE:
            continue
        if new_owner[t][i] not in (-1, k):
            return None
        new_owner[t][i] = k
        q[t][i] += amount
    inv = _inventories(q, data)
    if inv is None:
        return None
    routes = []
    for t in range(T):
        period = []
        for k in range(m):
            route = [c for c in base[t][k] if q[t][c - 1] > _ACTIVE]
            pending = [i + 1 for i in chosen if new_owner[t][i] == k]
            # Insert the most remote first, then insert the other removed customers by true minimum increase.
            pending.sort(key=lambda c: -costs[0][c])
            for c in pending:
                _, pos = _insertion(route, c, costs)
                route.insert(pos, c)
            period.append(route)
        routes.append(period)
    return _state(q, routes, data)


def _fast_transfer(state, data, rng, deadline):
    """Cheap cross-period whole-shipment merge neighbourhood that does not depend on an integer solver."""
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    if T < 2 or n == 0:
        return None
    i = rng.randrange(n)
    active = [t for t in range(T) if state['q'][t][i] > _ACTIVE]
    if not active:
        return None
    source = rng.choice(active)
    target = rng.randrange(T - 1)
    if target >= source:
        target += 1
    q, inv = state['q'], state['inv']
    amount = min(q[source][i], Q - q[target][i], m * Q - sum(q[target]))
    if target < source:
        for t in range(target, source):
            amount = min(amount, inv[t][0], data['caps'][i + 1] - inv[t][i + 1] - data['demands'][i][t])
    else:
        for t in range(source, target):
            amount = min(amount, inv[t][i + 1], data['caps'][0] - inv[t][0])
    if amount <= _ACTIVE:
        return None
    amounts = [r[:] for r in q]
    amounts[source][i] -= amount
    amounts[target][i] += amount
    if abs(amounts[source][i]) <= _ACTIVE:
        amounts[source][i] = 0.0
    routes = list(state['routes'])
    for t in (source, target):
        built = _routes_for_amounts(amounts[t], data, deadline, routes[t])
        if built is None:
            return None
        routes[t] = built[0]
    return _state(amounts, routes, data)


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.monotonic()
    safe = _blank_solution(instance)
    try:
        budget = max(0.0, float(time_limit_s))
        T = int(instance['num_periods'])
        n = int(instance['num_customers'])
        m = int(instance['num_vehicles'])
        Q = float(instance['vehicle_capacity'])
        data = {
            'T': T, 'n': n, 'm': m, 'Q': Q,
            'costs': [[float(x) for x in row] for row in instance['travel_costs']],
            'production': [float(x) for x in instance['production']],
            'demands': [[float(x) for x in row] for row in instance['demands']],
            'caps': [float(x) for x in instance['inventory_capacities']],
            'initial': [float(x) for x in instance['initial_inventories']],
            'holding': [float(x) for x in instance['holding_costs']],
        }
        if T <= 0:
            return safe
        reserve = min(budget * 0.04, max(0.006, budget * 0.018,
                                       0.000002 * T * max(1, n) * max(1, m)))
        deadline = start + budget - reserve
        rng = random.Random(_seed_from_data(data))
        zero_q = [[0.0] * n for _ in range(T)]
        zero_routes = [[[] for _ in range(m)] for _ in range(T)]
        best = _state(zero_q, zero_routes, data)
        if best is not None:
            safe = _to_output(best, data)
            if n == 0 or m == 0 or Q <= 0.0:
                return safe
        if m <= 0 or Q <= 0.0 or n < 0:
            return safe

        # Obtain a complete feasible solution first; never treat an unfinished flow or integer subproblem as feasible.
        initial_deadline = min(deadline, start + budget * 0.24)
        attempt = 0
        while best is None and time.monotonic() < initial_deadline:
            now = time.monotonic()
            stop = min(initial_deadline, now + max(0.002, budget * 0.07))
            if attempt == 2:
                assignment = None
            else:
                mode = 'dynamic' if attempt == 0 else ('static' if attempt == 1 else 'random')
                assignment = _make_assignment(data, rng, mode)
            q = _flow_schedule(data, assignment, False, stop)
            if q is not None:
                candidate = _state_from_schedule(q, data, stop, assignment)
                if candidate is not None:
                    best = candidate
                    safe = _to_output(best, data)
            attempt += 1
        if best is None and _HAS_MIP and time.monotonic() < deadline:
            now = time.monotonic()
            candidate = _joint_neighborhood(None, list(range(n)), data,
                                            now + (deadline - now) * 0.88, rng, budget)
            if candidate is not None:
                best = candidate
                safe = _to_output(best, data)
        while best is None and time.monotonic() < deadline:
            assignment = _make_assignment(data, rng, 'random')
            q = _flow_schedule(data, assignment, False, deadline)
            if q is not None:
                candidate = _state_from_schedule(q, data, deadline, assignment)
                if candidate is not None:
                    best = candidate
                    safe = _to_output(best, data)
        if best is None:
            return safe

        current = best
        candidate = _polish(best, data, min(deadline, time.monotonic() + budget * 0.06), rng)
        if candidate['obj'] < best['obj'] - 1e-8:
            best = current = candidate
            safe = _to_output(best, data)
        if _HAS_MIP and time.monotonic() < deadline:
            candidate = _joint_neighborhood(best, [], data,
                                            min(deadline, time.monotonic() + budget * 0.08), rng, budget)
            if candidate is not None and candidate['obj'] < best['obj'] - 1e-8:
                best = current = candidate
                safe = _to_output(best, data)

        complexity = math.sqrt(max(1, n * T))
        strength = min(n, max(2, int(2 + math.sqrt(budget) * max(1, n) ** 0.25
                                     / (1 + math.sqrt(T)))))
        slice_time = budget / (18.0 + 3.0 * complexity)
        route_time = budget / (90.0 + 7.0 * complexity)
        cycle_time = budget / (5.0 + complexity)
        last_improvement = time.monotonic()
        next_shake = last_improvement + cycle_time
        step = 0
        while time.monotonic() < deadline:
            now = time.monotonic()
            remaining = deadline - now
            progress = min(1.0, max(0.0, (now - start) / max(budget, 1e-9)))
            candidate = None
            if now >= next_shake:
                # Perturbation strength grows with the time budget and stagnation time; an independent best solution is always kept.
                current = best
                shake_stop = min(deadline, now + route_time * (1.0 + progress))
                candidate = _polish(current, data, shake_stop, rng, True)
                current = candidate
                next_shake = time.monotonic() + cycle_time
            elif _HAS_MIP and remaining > max(0.006, reserve) and rng.random() < 0.77:
                selected = _select_customers(current, data, rng, strength, step)
                step += 1
                stagnation = (now - last_improvement) / max(cycle_time, 1e-6)
                if stagnation > 1.0 and len(selected) > 1:
                    enlarged = min(n, max(len(selected), int(strength * math.sqrt(stagnation))))
                    center = selected[0]
                    extras = sorted((i for i in range(n) if i not in selected),
                                    key=lambda i: data['costs'][center + 1][i + 1])
                    selected.extend(extras[:enlarged - len(selected)])
                span = slice_time * (1.0 + math.sqrt(len(selected)))
                stop = min(deadline - reserve * 0.3, now + min(remaining * 0.45, span))
                candidate = _joint_neighborhood(current, selected, data, stop, rng, budget)
                if candidate is not None and time.monotonic() < deadline:
                    candidate = _polish(candidate, data,
                                        min(deadline, time.monotonic() + route_time), rng)
            elif rng.random() < 0.64:
                stop = min(deadline, now + route_time)
                candidate = _fast_transfer(current, data, rng, stop)
            else:
                candidate = _polish(current, data, min(deadline, now + route_time), rng,
                                    shake=rng.random() < 0.3)

            if candidate is not None:
                if candidate['obj'] < best['obj'] - 1e-8:
                    best = candidate
                    safe = _to_output(best, data)
                    last_improvement = time.monotonic()
                delta = candidate['obj'] - current['obj']
                visits = sum(len(r) for p in best['routes'] for r in p)
                scale = max(1.0, sum(best['pcost']) / max(1, visits))
                temperature = scale * (0.005 + 0.06 * (1.0 - progress) ** 2)
                if delta <= 1e-9 or (delta < 40.0 * temperature
                                     and rng.random() < math.exp(-delta / temperature)):
                    current = candidate
        return safe
    except BaseException:
        return safe
