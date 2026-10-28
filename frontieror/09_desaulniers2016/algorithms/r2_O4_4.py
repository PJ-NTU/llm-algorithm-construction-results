"""Global inventory-coupled implicit route-column search
Builds deliveries by time-expanded flow and jointly optimises visits and full-horizon inventory with exact-insertion route columns, an implicit subsequence integer model and capacity-feasible route search.
"""
import time
import math
import random
from collections import deque

try:
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    _OPT_AVAILABLE = True
except Exception:
    _OPT_AVAILABLE = False

_ACTIVE = 1e-9
_TOL = 1e-7

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
    if target - flow > 1e-6 * (1.0 + target):
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
    if len(active) ** 2 < 30000.0 * max(0.0, deadline - time.monotonic()):
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
        "ending_inventories": [[max(0.0, min(data["caps"][i], float(x)))
                                for i, x in enumerate(row)] for row in inventories],
    }


def _insert(route, customer, costs):
    best, pos = float('inf'), 0
    for p in range(len(route) + 1):
        a = 0 if p == 0 else route[p - 1]
        b = 0 if p == len(route) else route[p]
        delta = costs[a][customer] + costs[customer][b] - costs[a][b]
        if delta < best:
            best, pos = delta, p
    return best, pos


def _order_group(group, costs, deadline):
    if len(group) <= 2:
        return list(group)
    pending = list(group)
    route = []
    at = 0
    while pending and time.monotonic() < deadline:
        c = min(pending, key=lambda x: costs[at][x])
        route.append(c)
        pending.remove(c)
        at = c
    route.extend(pending)
    if time.monotonic() < deadline:
        other = []
        for c in sorted(group, key=lambda x: -costs[0][x]):
            _, p = _insert(other, c, costs)
            other.insert(p, c)
            if time.monotonic() >= deadline:
                other = route
                break
        if _route_cost(other, costs) < _route_cost(route, costs):
            route = other
    return _two_opt(route, costs, deadline)


def _checked(q, routes, data):
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    if len(q) != T or len(routes) != T:
        return None
    clean = []
    rr = []
    for t in range(T):
        if len(q[t]) != n or len(routes[t]) != m:
            return None
        row = [float(x) for x in q[t]]
        if any(not math.isfinite(x) or x < -1e-7 for x in row):
            return None
        row = [0.0 if x < _ACTIVE else x for x in row]
        seen, period = set(), []
        for route in routes[t]:
            r = []
            for c in route:
                if c < 1 or c > n or c in seen:
                    return None
                seen.add(c)
                if row[c - 1] > _ACTIVE:
                    r.append(c)
            if sum(row[c - 1] for c in r) > Q + 1e-7:
                return None
            period.append(r)
        if any(x > _ACTIVE and i + 1 not in seen for i, x in enumerate(row)):
            return None
        clean.append(row)
        rr.append(period)
    inv = _inventories(clean, data)
    if inv is None:
        return None
    pc = [sum(_route_cost(r, data['costs']) for r in p) for p in rr]
    return {'q': clean, 'routes': rr, 'inv': inv, 'pcost': pc,
            'obj': sum(pc) + _holding_cost(inv, data)}


class _Model:
    def __init__(self):
        self.c, self.lb, self.ub, self.integer = [], [], [], []
        self.ri, self.ci, self.a = [], [], []
        self.lo, self.hi = [], []

    def var(self, cost=0.0, upper=float('inf'), integer=0):
        j = len(self.c)
        self.c.append(float(cost))
        self.lb.append(0.0)
        self.ub.append(max(0.0, float(upper)))
        self.integer.append(integer)
        return j

    def row(self, terms, low=-float('inf'), high=float('inf')):
        r = len(self.lo)
        for j, a in terms:
            if a:
                self.ri.append(r)
                self.ci.append(j)
                self.a.append(float(a))
        self.lo.append(float(low))
        self.hi.append(float(high))

    def run(self, until, gap=0.0):
        if not _OPT_AVAILABLE or time.monotonic() >= until:
            return None
        try:
            a = coo_matrix((np.asarray(self.a),
                            (np.asarray(self.ri, dtype=np.int32),
                             np.asarray(self.ci, dtype=np.int32))),
                           shape=(len(self.lo), len(self.c))).tocsc()
            seconds = until - time.monotonic()
            if seconds <= 0.001:
                return None
            result = milp(
                np.asarray(self.c), integrality=np.asarray(self.integer),
                bounds=Bounds(np.asarray(self.lb), np.asarray(self.ub)),
                constraints=LinearConstraint(a, np.asarray(self.lo), np.asarray(self.hi)),
                options={'time_limit': seconds * 0.94, 'mip_rel_gap': gap, 'presolve': True},
            )
            if result.x is None or not np.all(np.isfinite(result.x)):
                return None
            x = result.x
            # A solver time-limit status does not mean a feasible integer solution exists; it must be checked independently.
            if np.any(x < np.asarray(self.lb) - 1e-7):
                return None
            if np.any(x > np.asarray(self.ub) + 1e-7):
                return None
            mask = np.asarray(self.integer, dtype=bool)
            if np.any(np.abs(x[mask] - np.rint(x[mask])) > 1e-6):
                return None
            ax = a @ x
            if np.any(ax < np.asarray(self.lo) - 5e-7):
                return None
            if np.any(ax > np.asarray(self.hi) + 5e-7):
                return None
            return x
        except Exception:
            return None


def _stock_rows(model, data, customers, incoming, depot_out, fixed=None, optimize=True):
    T = data['T']
    caps, initial = data['caps'], data['initial']
    demand, holding = data['demands'], data['holding']
    for j, i in enumerate(customers):
        previous = None
        for t in range(T):
            bound = caps[i + 1] - demand[i][t]
            if bound < -1e-7:
                return False
            inv = model.var(holding[i + 1] if optimize else 0.0, bound)
            terms = [(inv, 1.0)] + [(v, -1.0) for v in incoming[t][j]]
            rhs = -demand[i][t]
            if previous is None:
                rhs += initial[i + 1]
            else:
                terms.append((previous, -1.0))
            model.row(terms, rhs, rhs)
            previous = inv
    previous = None
    for t in range(T):
        inv = model.var(holding[0] if optimize else 0.0, caps[0])
        terms = [(inv, 1.0)] + [(v, 1.0) for v in depot_out[t]]
        rhs = data['production'][t] - (fixed[t] if fixed is not None else 0.0)
        if previous is None:
            rhs += initial[0]
        else:
            terms.append((previous, -1.0))
        model.row(terms, rhs, rhs)
        previous = inv
    return True


def _support_lp(state, data, forbidden, until):
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    model = _Model()
    incoming = [[[] for _ in range(n)] for _ in range(T)]
    outbound = [[] for _ in range(T)]
    indices = []
    for t in range(T):
        if time.monotonic() >= until:
            return None
        for k, route in enumerate(state['routes'][t]):
            vehicle = []
            for c in route:
                i = c - 1
                upper = 0.0 if (t, i) in forbidden else min(Q, data['caps'][c])
                v = model.var(0.0, upper)
                incoming[t][i].append(v)
                outbound[t].append(v)
                vehicle.append((v, 1.0))
                indices.append((t, i, v))
            if vehicle:
                model.row(vehicle, high=Q)
    if not _stock_rows(model, data, list(range(n)), incoming, outbound):
        return None
    x = model.run(until)
    if x is None:
        return None
    q = [[0.0] * n for _ in range(T)]
    for t, i, v in indices:
        q[t][i] = max(0.0, float(x[v]))
    return _checked(q, state['routes'], data)


def _feasibility_mip(data, until):
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    used_m = min(m, n)
    model = _Model()
    incoming = [[[] for _ in range(n)] for _ in range(T)]
    outbound = [[] for _ in range(T)]
    indices = []
    for t in range(T):
        if time.monotonic() >= until:
            return None
        loads = [[] for _ in range(used_m)]
        for i in range(n):
            visits = []
            for k in range(used_m):
                upper = min(Q, data['caps'][i + 1])
                y = model.var(1e-4, 1.0, 1)
                v = model.var(0.0, upper)
                model.row([(v, 1.0), (y, -upper)], high=0.0)
                visits.append((y, 1.0))
                loads[k].append((v, 1.0))
                incoming[t][i].append(v)
                outbound[t].append(v)
                indices.append((t, k, i, v))
            model.row(visits, high=1.0)
        for k in range(used_m):
            model.row(loads[k], high=Q)
            if k:
                model.row(loads[k - 1] + [(v, -1.0) for v, _ in loads[k]], low=0.0)
    if not _stock_rows(model, data, list(range(n)), incoming, outbound, optimize=False):
        return None
    x = model.run(until)
    if x is None:
        return None
    q = [[0.0] * n for _ in range(T)]
    routes = [[[] for _ in range(m)] for _ in range(T)]
    for t, k, i, v in indices:
        amount = max(0.0, float(x[v]))
        if amount > _ACTIVE:
            q[t][i] += amount
            routes[t][k].append(i + 1)
    return _checked(q, routes, data)


def _ordered_subsets(base, chosen, costs, until):
    """Exactly compute the shortest interleaved route of each customer subset with the fixed skeleton."""
    r, length = len(chosen), len(base)
    full, width = 1 << r, r + 1
    size = (length + 1) * full * width
    values = [float('inf')] * size
    parent = [-1] * size
    start = r
    values[start] = 0.0
    for j in range(length + 1):
        if time.monotonic() >= until:
            return None
        base_last = base[j - 1] if j else 0
        for mask in range(full):
            offset = (j * full + mask) * width
            for last in range(width):
                idx = offset + last
                value = values[idx]
                if not math.isfinite(value):
                    continue
                c = base_last if last == r else chosen[last] + 1
                if j < length:
                    nxt = ((j + 1) * full + mask) * width + r
                    new_value = value + costs[c][base[j]]
                    if new_value < values[nxt]:
                        values[nxt], parent[nxt] = new_value, idx
                bits = (full - 1) ^ mask
                while bits:
                    bit = bits & -bits
                    bits -= bit
                    k = bit.bit_length() - 1
                    nxt = (j * full + (mask | bit)) * width + k
                    new_value = value + costs[c][chosen[k] + 1]
                    if new_value < values[nxt]:
                        values[nxt], parent[nxt] = new_value, idx
    paths, route_values = [], []
    for mask in range(full):
        offset = (length * full + mask) * width
        best, end = float('inf'), -1
        for last in range(width):
            c = (base[-1] if base else 0) if last == r else chosen[last] + 1
            value = values[offset + last] + costs[c][0]
            if value < best:
                best, end = value, offset + last
        route = []
        idx = end
        while idx != start:
            last = idx % width
            j = (idx // width) // full
            route.append(base[j - 1] if last == r else chosen[last] + 1)
            idx = parent[idx]
        route.reverse()
        paths.append(route)
        route_values.append(best)
    return paths, route_values


def _column_neighbor(state, chosen, data, until):
    """Integer route columns enumerate only the selected customers; deliveries of all customers can be linked across periods."""
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    C, caps = data['costs'], data['caps']
    r, selected = len(chosen), {i + 1 for i in chosen}
    full = 1 << r
    model = _Model()
    incoming = [[[] for _ in range(n)] for _ in range(T)]
    outbound = [[] for _ in range(T)]
    q_indices, columns = [], []
    path_cache = {}
    for t in range(T):
        if time.monotonic() >= until:
            return None
        visits = [[] for _ in chosen]
        empty_used, previous_empty = 0, None
        for k, original in enumerate(state['routes'][t]):
            base = [c for c in original if c not in selected]
            if not base:
                empty_used += 1
                if empty_used > r:
                    continue
            key = tuple(base)
            if key not in path_cache:
                built = _ordered_subsets(base, chosen, C, until)
                if built is None:
                    return None
                path_cache[key] = built
            paths, values = path_cache[key]
            original_mask = sum(1 << j for j, i in enumerate(chosen) if i + 1 in original)
            choices, includes = [], [[] for _ in chosen]
            for mask in range(full):
                if not paths[mask]:
                    continue
                route, value = paths[mask], values[mask]
                if mask == original_mask and _route_cost(original, C) < value - 1e-8:
                    route, value = original, _route_cost(original, C)
                v = model.var(value, 1.0, 1)
                choices.append((v, 1.0))
                columns.append((t, k, v, route))
                for j in range(r):
                    if mask & (1 << j):
                        includes[j].append(v)
                        visits[j].append((v, 1.0))
            model.row(choices, high=1.0)
            if not base:
                if previous_empty is not None:
                    model.row(previous_empty + [(v, -a) for v, a in choices], low=0.0)
                previous_empty = choices
            load = []
            for c in base:
                upper = min(Q, caps[c])
                v = model.var(0.0, upper)
                model.row([(v, 1.0)] + [(y, -upper) for y, _ in choices], high=0.0)
                incoming[t][c - 1].append(v)
                outbound[t].append(v)
                load.append((v, 1.0))
                q_indices.append((t, c - 1, v))
            for j, i in enumerate(chosen):
                upper = min(Q, caps[i + 1])
                v = model.var(0.0, upper)
                model.row([(v, 1.0)] + [(y, -upper) for y in includes[j]], high=0.0)
                incoming[t][i].append(v)
                outbound[t].append(v)
                load.append((v, 1.0))
                q_indices.append((t, i, v))
            model.row(load + [(y, -Q) for y, _ in choices], high=0.0)
        for row in visits:
            model.row(row, high=1.0)
    if not _stock_rows(model, data, list(range(n)), incoming, outbound):
        return None
    # The current solution always belongs to the neighbourhood; only complete route and inventory combinations no worse than it are searched.
    model.row([(j, c) for j, c in enumerate(model.c) if c], high=state['obj'] + 1e-6)
    x = model.run(until, gap=0.0001)
    if x is None:
        return None
    q = [[0.0] * n for _ in range(T)]
    routes = [[[] for _ in range(m)] for _ in range(T)]
    for t, i, v in q_indices:
        q[t][i] += max(0.0, float(x[v]))
    for t, k, v, route in columns:
        if x[v] > 0.5:
            routes[t][k] = list(route)
    return _checked(q, routes, data)


def _subsequence_neighbor(state, chosen, data, until):
    """Select all customer visits and continuous delivery amounts simultaneously on an ordered acyclic path graph."""
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    C, caps = data['costs'], data['caps']
    model = _Model()
    incoming = [[[] for _ in range(n)] for _ in range(T)]
    outbound = [[] for _ in range(T)]
    records, periods = [], []
    for t in range(T):
        if time.monotonic() >= until:
            return None
        visits = [[] for _ in range(n)]
        period, empty_used = [], 0
        for k, old in enumerate(state['routes'][t]):
            route = old[:]
            if not route:
                empty_used += 1
                if empty_used > len(chosen):
                    period.append([])
                    continue
            # Insertion does not change the relative order of the old route, so the original feasible route can still be selected directly.
            for i in chosen:
                if i + 1 not in route:
                    _, pos = _insert(route, i + 1, C)
                    route.insert(pos, i + 1)
            period.append(route)
            if not route:
                continue
            length = len(route)
            enter, leave = [[] for _ in range(length + 2)], [[] for _ in range(length + 2)]
            nodes = [0] + route + [0]
            for a in range(length + 1):
                if time.monotonic() >= until:
                    return None
                for b in range(a + 1, length + 2):
                    if a == 0 and b == length + 1:
                        continue
                    v = model.var(C[nodes[a]][nodes[b]], 1.0)
                    leave[a].append((v, 1.0))
                    enter[b].append((v, 1.0))
            # Visit binary variables and ordered unit path flows together ensure that path arcs are also integer choices.
            used = model.var(0.0, 1.0)
            model.row(leave[0] + [(used, -1.0)], 0.0, 0.0)
            model.row(enter[-1] + [(used, -1.0)], 0.0, 0.0)
            load = [(used, -Q)]
            for p, c in enumerate(route, 1):
                y = model.var(0.0, 1.0, 1)
                model.row(enter[p] + [(y, -1.0)], 0.0, 0.0)
                model.row(leave[p] + [(y, -1.0)], 0.0, 0.0)
                visits[c - 1].append((y, 1.0))
                upper = min(Q, caps[c])
                v = model.var(0.0, upper)
                model.row([(v, 1.0), (y, -upper)], high=0.0)
                incoming[t][c - 1].append(v)
                outbound[t].append(v)
                load.append((v, 1.0))
                records.append((t, k, c - 1, v, y))
            model.row(load, high=0.0)
        periods.append(period)
        for row in visits:
            if row:
                model.row(row, high=1.0)
    if not _stock_rows(model, data, list(range(n)), incoming, outbound):
        return None
    model.row([(j, c) for j, c in enumerate(model.c) if c], high=state['obj'] + 1e-6)
    x = model.run(until, gap=0.0005)
    if x is None:
        return None
    q = [[0.0] * n for _ in range(T)]
    owners = [[-1] * n for _ in range(T)]
    for t, k, i, v, y in records:
        if x[y] > 0.5:
            q[t][i] = max(0.0, float(x[v]))
            owners[t][i] = k
    routes = [[[c for c in route if owners[t][c - 1] == k]
               for k, route in enumerate(period)] for t, period in enumerate(periods)]
    return _checked(q, routes, data)


def _descent_period(routes, amounts, data, until):
    routes = [r[:] for r in routes]
    C, Q = data['costs'], data['Q']
    for k in range(len(routes)):
        if time.monotonic() >= until:
            break
        routes[k] = _two_opt(routes[k], C, until)
    while time.monotonic() < until:
        loads = [sum(amounts[c - 1] for c in r) for r in routes]
        best_delta, move = -1e-8, None
        for a, ra in enumerate(routes):
            for pa, c in enumerate(ra):
                if time.monotonic() >= until:
                    return routes
                ap = 0 if pa == 0 else ra[pa - 1]
                an = 0 if pa + 1 == len(ra) else ra[pa + 1]
                remove = C[ap][an] - C[ap][c] - C[c][an]
                for b, rb in enumerate(routes):
                    if a == b or loads[b] + amounts[c - 1] > Q + 1e-8:
                        continue
                    ins, pb = _insert(rb, c, C)
                    if remove + ins < best_delta:
                        best_delta, move = remove + ins, ('relocate', a, pa, b, pb)
                # Single-point reinsertion within the same route complements the 2-opt neighbourhood.
                reduced = ra[:pa] + ra[pa + 1:]
                ins, pb = _insert(reduced, c, C)
                if remove + ins < best_delta:
                    best_delta, move = remove + ins, ('relocate', a, pa, a, pb)
            for b in range(a + 1, len(routes)):
                rb = routes[b]
                for pa, ca in enumerate(ra):
                    if time.monotonic() >= until:
                        return routes
                    ap = 0 if pa == 0 else ra[pa - 1]
                    an = 0 if pa + 1 == len(ra) else ra[pa + 1]
                    for pb, cb in enumerate(rb):
                        if (loads[a] - amounts[ca - 1] + amounts[cb - 1] > Q + 1e-8
                                or loads[b] - amounts[cb - 1] + amounts[ca - 1] > Q + 1e-8):
                            continue
                        bp = 0 if pb == 0 else rb[pb - 1]
                        bn = 0 if pb + 1 == len(rb) else rb[pb + 1]
                        delta = (C[ap][cb] + C[cb][an] - C[ap][ca] - C[ca][an]
                                 + C[bp][ca] + C[ca][bn] - C[bp][cb] - C[cb][bn])
                        if delta < best_delta:
                            best_delta, move = delta, ('swap', a, pa, b, pb)
                prefix_a, prefix_b = [0.0], [0.0]
                for c in ra:
                    prefix_a.append(prefix_a[-1] + amounts[c - 1])
                for c in rb:
                    prefix_b.append(prefix_b[-1] + amounts[c - 1])
                for pa in range(len(ra) + 1):
                    if time.monotonic() >= until:
                        return routes
                    ap = 0 if pa == 0 else ra[pa - 1]
                    an = 0 if pa == len(ra) else ra[pa]
                    for pb in range(len(rb) + 1):
                        if (prefix_a[pa] + loads[b] - prefix_b[pb] > Q + 1e-8
                                or prefix_b[pb] + loads[a] - prefix_a[pa] > Q + 1e-8):
                            continue
                        bp = 0 if pb == 0 else rb[pb - 1]
                        bn = 0 if pb == len(rb) else rb[pb]
                        delta = C[ap][bn] + C[bp][an] - C[ap][an] - C[bp][bn]
                        if delta < best_delta:
                            best_delta, move = delta, ('tails', a, pa, b, pb)
        if move is None:
            break
        kind, a, pa, b, pb = move
        if kind == 'relocate':
            c = routes[a].pop(pa)
            routes[b].insert(pb, c)
        elif kind == 'swap':
            routes[a][pa], routes[b][pb] = routes[b][pb], routes[a][pa]
        else:
            ra, rb = routes[a], routes[b]
            routes[a], routes[b] = ra[:pa] + rb[pb:], rb[:pb] + ra[pa:]
        for k in {a, b}:
            if time.monotonic() < until:
                routes[k] = _two_opt(routes[k], C, until)
    return routes


def _route_neighbor(state, data, rng, until, strength, perturb):
    T, m, Q, C = data['T'], data['m'], data['Q'], data['costs']
    periods = [t for t in range(T) if sum(len(r) for r in state['routes'][t]) > 1]
    if not periods:
        return None
    t = rng.choice(periods)
    routes = [r[:] for r in state['routes'][t]]
    q = state['q'][t]
    items = [c for route in routes for c in route]
    if perturb and time.monotonic() < until:
        seed = rng.choice(items)
        count = min(len(items), max(2, int(strength * math.sqrt(len(items)))))
        if rng.random() < 0.65:
            removed = sorted(items, key=lambda c: C[seed][c] * rng.uniform(0.8, 1.2))[:count]
        else:
            removed = rng.sample(items, count)
        selected = set(removed)
        routes = [[c for c in r if c not in selected] for r in routes]
        loads = [sum(q[c - 1] for c in r) for r in routes]
        while removed and time.monotonic() < until:
            best = None
            for c in removed:
                options = []
                for k in range(m):
                    if loads[k] + q[c - 1] <= Q + 1e-8:
                        delta, p = _insert(routes[k], c, C)
                        options.append((delta, k, p))
                if not options:
                    return None
                options.sort()
                regret = options[1][0] - options[0][0] if len(options) > 1 else 1e12
                key = (regret * rng.uniform(0.8, 1.2), q[c - 1])
                if best is None or key > best[0]:
                    best = (key, c, options[0])
            _, c, (_, k, p) = best
            routes[k].insert(p, c)
            loads[k] += q[c - 1]
            removed.remove(c)
        if removed:
            return None
    routes = _descent_period(routes, q, data, until)
    schedule = list(state['routes'])
    schedule[t] = routes
    return _checked(state['q'], schedule, data)


def _fast_transfer(state, data, rng, until):
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    if T < 2 or n == 0 or m == 0:
        return None
    i = rng.randrange(n)
    c = i + 1
    C, inv, q = data['costs'], state['inv'], state['q']
    owners = [-1] * T
    loads = [[sum(q[t][x - 1] for x in r) for r in state['routes'][t]] for t in range(T)]
    for t in range(T):
        for k, route in enumerate(state['routes'][t]):
            if c in route:
                owners[t] = k
    best_delta, best = -1e-8, None
    for a in range(T):
        if owners[a] < 0:
            continue
        ka = owners[a]
        ra = state['routes'][a][ka]
        pa = ra.index(c)
        x = 0 if pa == 0 else ra[pa - 1]
        y = 0 if pa + 1 == len(ra) else ra[pa + 1]
        remove = C[x][y] - C[x][c] - C[c][y]
        for b in range(T):
            if time.monotonic() >= until:
                break
            if a == b:
                continue
            upper = min(q[a][i], Q - q[b][i])
            if b < a:
                for s in range(b, a):
                    upper = min(upper, inv[s][0],
                                data['caps'][c] - inv[s][c] - data['demands'][i][s])
            else:
                for s in range(a, b):
                    upper = min(upper, inv[s][c], data['caps'][0] - inv[s][0])
            if upper <= _ACTIVE:
                continue
            hdelta = (a - b) * (data['holding'][c] - data['holding'][0])
            for kb in ([owners[b]] if owners[b] >= 0 else range(m)):
                amount = min(upper, Q - loads[b][kb])
                if amount <= _ACTIVE:
                    continue
                insertion, pb = (0.0, 0) if owners[b] >= 0 else _insert(state['routes'][b][kb], c, C)
                full = amount >= q[a][i] - 1e-8
                if full:
                    amount = q[a][i]
                delta = insertion + (remove if full else 0.0) + amount * hdelta
                if delta < best_delta:
                    best_delta, best = delta, (a, b, ka, pa, kb, pb, amount, full)
    if best is None:
        return None
    a, b, ka, pa, kb, pb, amount, full = best
    qq = [row[:] for row in q]
    rr = [[r[:] for r in period] for period in state['routes']]
    qq[a][i] -= amount
    qq[b][i] += amount
    if full:
        qq[a][i] = 0.0
        rr[a][ka].pop(pa)
    if owners[b] < 0:
        rr[b][kb].insert(pb, c)
    return _checked(qq, rr, data)


def _burdens(state, data):
    if 'burdens' in state:
        return state['burdens']
    C = data['costs']
    customer = [0.0] * data['n']
    visits = []
    for t, period in enumerate(state['routes']):
        for route in period:
            share = _route_cost(route, C) / max(1, len(route))
            for p, c in enumerate(route):
                a = 0 if p == 0 else route[p - 1]
                b = 0 if p + 1 == len(route) else route[p + 1]
                saving = max(0.0, C[a][c] + C[c][b] - C[a][b])
                score = saving + 0.12 * share
                visits.append((score, t, c - 1))
                customer[c - 1] += score
    state['burdens'] = customer, visits
    return customer, visits


def _choose_customers(state, data, rng, ages, now, quantum, strength, count):
    burden, _ = _burdens(state, data)
    if rng.random() < 0.8:
        seed = max(range(data['n']), key=lambda i: (1.0 + burden[i])
                   * (1.0 + min(strength, (now - ages[i]) / max(quantum, 1e-5)))
                   * rng.uniform(0.7, 1.3))
    else:
        seed = min(range(data['n']), key=lambda i: ages[i])
    others = [i for i in range(data['n']) if i != seed]
    if rng.random() < 0.85:
        others.sort(key=lambda i: data['costs'][seed + 1][i + 1]
                    * rng.uniform(0.75, 1.25)
                    / (1.0 + 0.08 * burden[i] / max(1.0, burden[seed])))
    else:
        rng.shuffle(others)
    chosen = [seed] + others[:max(0, count - 1)]
    for i in chosen:
        ages[i] = now
    return chosen


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.monotonic()
    safe = _blank_solution(instance)
    best, data = None, None
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        reserve = max(0.002, 0.025 * budget)
        deadline = started + max(0.0, budget - reserve)
        T, n, m = int(instance['num_periods']), int(instance['num_customers']), int(instance['num_vehicles'])
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
        if T == 0:
            return safe
        rng = random.Random(_seed_from_data(data))
        complexity = max(1, T * (n + m + 1))
        quantum = budget / (16.0 + math.sqrt(complexity))
        strength = math.log2(2.0 + budget)
        # Keep the per-period just-in-time construction first; never treat an empty dict with merely the correct shape as feasible.
        q = [[0.0] * n for _ in range(T)]
        stock = data['initial'][1:]
        for t in range(T):
            for i in range(n):
                q[t][i] = max(0.0, data['demands'][i][t] - stock[i])
                stock[i] += q[t][i] - data['demands'][i][t]
        if _inventories(q, data) is not None:
            rr = []
            for t in range(T):
                active = {i + 1 for i, x in enumerate(q[t]) if x > _ACTIVE}
                routes = _ffd_groups(active, q[t], m, Q)
                if routes is None:
                    rr = None
                    break
                rr.append(routes)
            if rr is not None:
                best = _checked(q, rr, data)
        if best is not None:
            safe = _to_output(best, data)
            if n == 0 or m == 0 or Q <= 0.0:
                return safe
        base_end = min(deadline, started + budget * 0.14)
        attempt = 0
        while best is None and time.monotonic() < base_end:
            mode = 'dynamic' if attempt == 0 else ('static' if attempt == 1 else 'random')
            assignment = _make_assignment(data, rng, mode) if m > 0 else None
            local = min(base_end, time.monotonic() + max(quantum, budget * 0.03))
            q = _flow_schedule(data, assignment, False, local)
            if q is not None:
                rr = [[[] for _ in range(m)] for _ in range(T)]
                if assignment is not None:
                    for t in range(T):
                        for i in range(n):
                            if q[t][i] > _ACTIVE:
                                rr[t][assignment[t][i]].append(i + 1)
                best = _checked(q, rr, data)
            if attempt == 1 and best is None:
                aggregate = _flow_schedule(data, None, False, base_end)
                if aggregate is not None:
                    best = _state_from_schedule(aggregate, data, base_end)
            attempt += 1
        if best is None and _OPT_AVAILABLE and time.monotonic() < deadline:
            best = _feasibility_mip(data, min(deadline, started + budget * 0.82))
        while best is None and time.monotonic() < deadline:
            assignment = _make_assignment(data, rng, 'random') if m > 0 else None
            q = _flow_schedule(data, assignment, False, deadline)
            if q is not None:
                best = _state_from_schedule(q, data, deadline, assignment)
        if best is None:
            return safe
        safe = _to_output(best, data)
        # Keep the existing routes of each period, avoiding sorted rebuilds damaging the current best route of any period.
        setup_end = min(deadline, time.monotonic() + budget * 0.055)
        rr = []
        for t in range(T):
            now = time.monotonic()
            until = min(setup_end, now + max(0.0, setup_end - now) / max(1, T - t))
            built = _routes_for_amounts(best['q'][t], data, until, best['routes'][t])
            if built is not None and built[1] < best['pcost'][t]:
                rr.append(built[0])
            else:
                rr.append(best['routes'][t])
        candidate = _checked(best['q'], rr, data)
        if candidate is not None and candidate['obj'] < best['obj']:
            best = candidate
        if _OPT_AVAILABLE and time.monotonic() < deadline:
            now = time.monotonic()
            span = min((deadline - now) * 0.3, budget * 0.13)
            arc_work = sum((len(r) + 1) * (len(r) + 2) // 2
                           for p in best['routes'] for r in p if r)
            if arc_work < span * 18000.0:
                candidate = _subsequence_neighbor(best, [], data, min(deadline, now + span))
            else:
                candidate = _support_lp(best, data, set(), min(deadline, now + quantum))
            if candidate is not None and candidate['obj'] < best['obj']:
                best = candidate
        current = best
        safe = _to_output(best, data)
        last_best = time.monotonic()
        last_reset = last_best
        ages = [started] * n
        tested = set()
        model_ema = max(0.001, quantum * 0.2)
        group_scale = int(math.log2(2.0 + budget * 2500.0 / max(1, T * m)))
        max_group = max(1, min(n, 2 + int(strength), group_scale))
        while time.monotonic() < deadline:
            now = time.monotonic()
            remaining = deadline - now
            stagnant = now - last_best
            candidate = None
            try:
                draw = rng.random()
                if not _OPT_AVAILABLE or remaining < max(0.01, 1.6 * model_ema):
                    draw = 0.83 if rng.random() < 0.7 else 0.98
                if draw < 0.54 and n:
                    depth = min(max_group, 2 + int(strength * stagnant / max(0.01, budget * 0.15)))
                    count = min(n, max_group, 1 + int((rng.random() ** 0.7) * max(1, depth)))
                    chosen = _choose_customers(current, data, rng, ages, now, quantum, strength, count)
                    span = max(quantum * (0.65 + 0.35 * count), model_ema * 1.8)
                    until = min(deadline, now + min(remaining * 0.55, span))
                    candidate = _column_neighbor(current, chosen, data, until)
                    duration = time.monotonic() - now
                    model_ema = 0.85 * model_ema + 0.15 * duration
                elif draw < 0.68 and n and _OPT_AVAILABLE:
                    count = min(n, max_group, 1 + int(rng.random() * strength))
                    chosen = _choose_customers(current, data, rng, ages, now, quantum, strength, count)
                    span = min(remaining * 0.35, max(quantum * 2.5,
                               budget / (6.0 + 0.2 * math.sqrt(complexity))))
                    arc_work = sum((len(r) + count + 1) * (len(r) + count + 2) // 2
                                   for p in current['routes'] for r in p)
                    until = min(deadline, now + span)
                    if arc_work < span * 18000.0:
                        candidate = _subsequence_neighbor(current, chosen, data, until)
                    else:
                        candidate = _column_neighbor(current, chosen[:1], data, until)
                elif draw < 0.78 and _OPT_AVAILABLE:
                    _, visits = _burdens(current, data)
                    available = [v for v in visits if (v[1], v[2]) not in tested]
                    if available:
                        _, t, i = max(available, key=lambda z: z[0] * rng.uniform(0.7, 1.3))
                        tested.add((t, i))
                        until = min(deadline, now + min(remaining * 0.4, max(quantum * 0.5, model_ema)))
                        candidate = _support_lp(current, data, {(t, i)}, until)
                    else:
                        candidate = _fast_transfer(current, data, rng, deadline)
                elif draw < 0.94:
                    span = max(budget / (65.0 + 4.0 * math.sqrt(complexity)), quantum * 0.18)
                    until = min(deadline, now + span)
                    amplitude = (0.25 + 0.14 * strength) * (1.0 + min(1.0, stagnant / max(budget * 0.12, 1e-5)))
                    candidate = _route_neighbor(current, data, rng, until, amplitude, rng.random() < 0.65)
                else:
                    candidate = _fast_transfer(current, data, rng, deadline)
                if candidate is not None:
                    if candidate['obj'] + 1e-7 < best['obj']:
                        best = candidate
                        safe = _to_output(best, data)
                        last_best = time.monotonic()
                    delta = candidate['obj'] - current['obj']
                    if delta < -1e-8:
                        current = candidate
                        tested.clear()
                    elif draw >= 0.78 and delta > 1e-8:
                        visits_count = sum(len(r) for p in best['routes'] for r in p)
                        scale = sum(best['pcost']) / max(1, visits_count)
                        phase = ((now - started) / max(quantum * (2.0 + strength), 0.001)) % 1.0
                        temperature = scale * (0.012 + 0.06 * (1.0 - phase))
                        if temperature > 0.0 and delta / temperature < 30.0 and rng.random() < math.exp(-delta / temperature):
                            current = candidate
                            tested.clear()
                now = time.monotonic()
                if now - last_reset > max(quantum, budget / (3.0 + strength)):
                    current = best
                    tested.clear()
                    last_reset = now
            except Exception:
                # Failures in neighbourhood construction, solving or numerical validation never overwrite the verified best solution.
                continue
        return _to_output(best, data)
    except BaseException:
        if best is not None and data is not None:
            try:
                return _to_output(best, data)
            except BaseException:
                pass
        return safe
