"""Flow-relaxation and elastic route-column joint search
Guides customer-group selection with reweighted minimum-cost flow, jointly recomputes all inventory deliveries in a true route-column integer neighbourhood, and is supplemented by visit removal, cross-period transfer and capacity-feasible route search.
"""
import time
import math
import random
import heapq

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
        if abs(depot) < 1e-7 or -1e-7 < depot < 0.0:
            depot = 0.0
        if abs(depot - caps[0]) < 1e-7:
            depot = caps[0]
        row = [depot]
        for i in range(n):
            before_demand = customers[i] + q[t][i]
            if before_demand > caps[i + 1] + 1e-6:
                return None
            customers[i] = before_demand - demands[i][t]
            if customers[i] < -1e-6:
                return None
            if abs(customers[i]) < 1e-7:
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
        "ending_inventories": [[float(x) for x in row] for row in inventories],
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
            seconds = 0.90 * (until - time.monotonic())
            if seconds <= 0.001:
                return None
            result = milp(
                np.asarray(self.c), integrality=np.asarray(self.integer),
                bounds=Bounds(np.asarray(self.lb), np.asarray(self.ub)),
                constraints=LinearConstraint(a, np.asarray(self.lo), np.asarray(self.hi)),
                options={'time_limit': seconds, 'mip_rel_gap': gap, 'presolve': True},
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
                if time.monotonic() >= until:
                    return None
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

def _circulation(data, assignment, use_cost, until, prices=None, allowed=None):
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    production, initial = data['production'], data['initial']
    capacities, demands, holding = data['caps'], data['demands'], data['holding']
    EPS, TOL = 1e-9, 1e-7
    if time.monotonic() >= until:
        return None

    root = 0
    nxt = 1
    depot = list(range(nxt, nxt + T))
    nxt += T
    customer = []
    for _ in range(n):
        customer.append(list(range(nxt, nxt + T)))
        nxt += T

    if assignment is None:
        hubs = list(range(nxt, nxt + T))
        nxt += T
        vehicles = None
    else:
        hubs = None
        vehicles = []
        for _ in range(T):
            vehicles.append(list(range(nxt, nxt + m)))
            nxt += m

    ss = nxt
    tt = nxt + 1
    graph = [[] for _ in range(nxt + 2)]
    balance = [0.0 for _ in range(nxt)]
    valid = [True]

    def add_edge(u, v, cap, cost):
        c = max(0.0, float(cap))
        forward = [v, len(graph[v]), c, float(cost), c]
        backward = [u, len(graph[u]), 0.0, -float(cost), 0.0]
        graph[u].append(forward)
        graph[v].append(backward)
        return u, len(graph[u]) - 1

    def lower_edge(u, v, low, high, cost=0.0):
        lo = float(low)
        hi = float(high)
        if hi + TOL < lo:
            valid[0] = False
            hi = lo
        handle = add_edge(u, v, max(0.0, hi - lo), cost)
        balance[u] -= lo
        balance[v] += lo
        return handle

    for t in range(T):
        lower_edge(root, depot[t], production[t], production[t])
    lower_edge(root, depot[0], initial[0], initial[0])

    for i in range(n):
        lower_edge(root, customer[i][0], initial[i + 1], initial[i + 1])
        for t in range(T):
            lower_edge(customer[i][t], root, demands[i][t], demands[i][t])

    handles = [[None for _ in range(n)] for _ in range(T)]
    if assignment is None:
        for t in range(T):
            lower_edge(depot[t], hubs[t], 0.0, m * Q)
            for i in range(n):
                c = 0.0 if prices is None else max(0.0, prices[t][i])
                handles[t][i] = lower_edge(hubs[t], customer[i][t], 0.0, (Q if allowed is None or allowed[t][i] else 0.0), c)
    else:
        for t in range(T):
            for k in range(m):
                lower_edge(depot[t], vehicles[t][k], 0.0, Q)
            for i in range(n):
                c = 0.0 if prices is None else max(0.0, prices[t][i])
                if m <= 0:
                    handles[t][i] = lower_edge(depot[t], customer[i][t], 0.0, 0.0, c)
                else:
                    k = assignment[t][i]
                    if not isinstance(k, int) or k < 0 or k >= m:
                        k = 0
                    handles[t][i] = lower_edge(vehicles[t][k], customer[i][t], 0.0, (Q if allowed is None or allowed[t][i] else 0.0), c)

    for t in range(T - 1):
        lower_edge(depot[t], depot[t + 1], 0.0, capacities[0], holding[0])
    lower_edge(depot[T - 1], root, 0.0, capacities[0], holding[0])

    for i in range(n):
        for t in range(T):
            cap = capacities[i + 1] - demands[i][t]
            if cap < -TOL:
                valid[0] = False
            target = customer[i][t + 1] if t + 1 < T else root
            lower_edge(customer[i][t], target, 0.0, max(0.0, cap), holding[i + 1])

    if not valid[0]:
        return None

    required = 0.0
    for v in range(nxt):
        if balance[v] > EPS:
            add_edge(ss, v, balance[v], 0.0)
            required += balance[v]
        elif balance[v] < -EPS:
            add_edge(v, tt, -balance[v], 0.0)

    sent = 0.0
    node_count = len(graph)

    if use_cost:
        potential = [0.0 for _ in range(node_count)]
        while sent + EPS < required:
            if time.monotonic() >= until:
                return None
            inf = float("inf")
            dist = [inf for _ in range(node_count)]
            pv = [-1 for _ in range(node_count)]
            pe = [-1 for _ in range(node_count)]
            dist[ss] = 0.0
            heap = [(0.0, ss)]
            scans = 0
            while heap:
                du, u = heapq.heappop(heap)
                if du > dist[u] + 1e-11:
                    continue
                scans += 1
                if (scans & 127) == 0 and time.monotonic() >= until:
                    return None
                for ei, edge in enumerate(graph[u]):
                    if edge[2] <= EPS:
                        continue
                    rc = edge[3] + potential[u] - potential[edge[0]]
                    if rc < 0.0:
                        rc = 0.0
                    nd = du + rc
                    if nd + 1e-11 < dist[edge[0]]:
                        dist[edge[0]] = nd
                        pv[edge[0]] = u
                        pe[edge[0]] = ei
                        heapq.heappush(heap, (nd, edge[0]))
            if pv[tt] < 0:
                return None
            for v in range(node_count):
                if dist[v] < inf:
                    potential[v] += dist[v]
            amount = required - sent
            v = tt
            while v != ss:
                u = pv[v]
                amount = min(amount, graph[u][pe[v]][2])
                v = u
            if amount <= EPS:
                return None
            v = tt
            while v != ss:
                u = pv[v]
                ei = pe[v]
                edge = graph[u][ei]
                edge[2] -= amount
                graph[v][edge[1]][2] += amount
                v = u
            sent += amount
    else:
        level = [-1 for _ in range(node_count)]
        work = [0 for _ in range(node_count)]
        timed_out = [False]

        def bfs():
            for v in range(node_count):
                level[v] = -1
            level[ss] = 0
            queue = [ss]
            head = 0
            while head < len(queue):
                if (head & 127) == 0 and time.monotonic() >= until:
                    timed_out[0] = True
                    return False
                u = queue[head]
                head += 1
                for edge in graph[u]:
                    if edge[2] > EPS and level[edge[0]] < 0:
                        level[edge[0]] = level[u] + 1
                        queue.append(edge[0])
            return level[tt] >= 0

        def dfs(u, amount):
            if u == tt:
                return amount
            if time.monotonic() >= until:
                timed_out[0] = True
                return 0.0
            while work[u] < len(graph[u]):
                ei = work[u]
                edge = graph[u][ei]
                if edge[2] > EPS and level[edge[0]] == level[u] + 1:
                    pushed = dfs(edge[0], min(amount, edge[2]))
                    if pushed > EPS:
                        edge[2] -= pushed
                        graph[edge[0]][edge[1]][2] += pushed
                        return pushed
                work[u] += 1
            return 0.0

        while sent + EPS < required and bfs():
            for v in range(node_count):
                work[v] = 0
            while sent + EPS < required:
                pushed = dfs(ss, required - sent)
                if pushed <= EPS:
                    break
                sent += pushed
            if timed_out[0]:
                return None

    if sent + TOL < required:
        return None

    q = [[0.0 for _ in range(n)] for _ in range(T)]
    for t in range(T):
        for i in range(n):
            u, ei = handles[t][i]
            edge = graph[u][ei]
            x = edge[4] - edge[2]
            if x < EPS:
                x = 0.0
            if x > Q and x < Q + TOL:
                x = Q
            q[t][i] = max(0.0, x)
    return q

def _support_flow(state, data, forbidden, until):
    """Inventory optimisation on a fixed visit support is a minimum-cost flow and needs no integer solve."""
    T, n = data['T'], data['n']
    owners = [[0] * n for _ in range(T)]
    allowed = [[False] * n for _ in range(T)]
    for t, period in enumerate(state['routes']):
        for k, route in enumerate(period):
            for c in route:
                owners[t][c - 1] = k
                allowed[t][c - 1] = (t, c - 1) not in forbidden
    q = _circulation(data, owners, True, until, allowed=allowed)
    return None if q is None else _checked(q, state['routes'], data)


def _elastic_columns(state, chosen, data, until):
    """Selected customers change their visits; the other customers keep their visit support but release all deliveries."""
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    C, caps = data['costs'], data['caps']
    r = len(chosen)
    selected = {i + 1 for i in chosen}
    bit_of = {i + 1: 1 << j for j, i in enumerate(chosen)}
    full = 1 << r
    model = _Model()
    incoming = [[[] for _ in range(n)] for _ in range(T)]
    outgoing = [[] for _ in range(T)]
    q_indices, columns, skeleton = [], [], []
    base_cost = 0.0
    for t in range(T):
        if time.monotonic() >= until:
            return None
        period = [[c for c in route if c not in selected] for route in state['routes'][t]]
        skeleton.append(period)
        visits = [[] for _ in chosen]
        # Keep the existing empty skeleton vehicles, then add at most r equivalent empty vehicle slots.
        empties = [k for k in range(m) if not period[k]]
        empties.sort(key=lambda k: not bool(state['routes'][t][k]))
        eligible = set(empties[:r])
        previous_empty_choice = None
        for k, base in enumerate(period):
            if time.monotonic() >= until:
                return None
            cost0 = _route_cost(base, C)
            base_cost += cost0
            capacity = []
            for c in base:
                v = model.var(0.0, min(Q, caps[c]))
                incoming[t][c - 1].append(v)
                outgoing[t].append(v)
                capacity.append((v, 1.0))
                q_indices.append((t, c - 1, v))
            if not base and k not in eligible:
                continue
            paths = [None] * full
            values = [float('inf')] * full
            paths[0], values[0] = base, cost0
            old_route = state['routes'][t][k]
            original_mask = 0
            for c in old_route:
                original_mask |= bit_of.get(c, 0)
            old_cost = _route_cost(old_route, C)
            vcols = [[] for _ in chosen]
            choice = []
            for mask in range(1, full):
                if time.monotonic() >= until:
                    return None
                bits = mask
                while bits:
                    bit = bits & -bits
                    bits -= bit
                    j = bit.bit_length() - 1
                    old = paths[mask ^ bit]
                    delta, p = _insert(old, chosen[j] + 1, C)
                    value = values[mask ^ bit] + delta
                    if value < values[mask]:
                        paths[mask] = old[:p] + [chosen[j] + 1] + old[p:]
                        values[mask] = value
                if mask == original_mask and old_cost <= values[mask] + 1e-9:
                    paths[mask], values[mask] = old_route[:], old_cost
                y = model.var(values[mask] - cost0, 1.0, 1)
                choice.append((y, 1.0))
                columns.append((t, k, y, paths[mask]))
                for j in range(r):
                    if mask & (1 << j):
                        vcols[j].append(y)
                        visits[j].append((y, 1.0))
            model.row(choice, high=1.0)
            # Empty vehicles have only permutation symmetry; impose an order only on originally unused slots.
            if not base and not old_route:
                if previous_empty_choice is not None:
                    model.row(previous_empty_choice + [(y, -a) for y, a in choice], low=0.0)
                previous_empty_choice = choice
            added_load = []
            for j, i in enumerate(chosen):
                upper = min(Q, caps[i + 1])
                v = model.var(0.0, upper)
                model.row([(v, 1.0)] + [(y, -upper) for y in vcols[j]], high=0.0)
                incoming[t][i].append(v)
                outgoing[t].append(v)
                capacity.append((v, 1.0))
                added_load.append((v, 1.0))
                q_indices.append((t, i, v))
            model.row(capacity, high=Q)
            # Route-column activation is linked to added load, tightening the capacity relaxation of fractional columns.
            model.row(added_load + [(y, -Q) for y, _ in choice], high=0.0)
        for terms in visits:
            model.row(terms, high=1.0)
    if not _stock_rows(model, data, list(range(n)), incoming, outgoing):
        return None
    # The columns and flows of the current solution are all kept, so this bound never excludes the current solution.
    model.row([(v, cost) for v, cost in enumerate(model.c) if cost],
              high=state['obj'] - base_cost + 1e-5)
    x = model.run(until, gap=1e-5)
    if x is None:
        return None
    q = [[0.0] * n for _ in range(T)]
    for t, i, v in q_indices:
        q[t][i] += max(0.0, float(x[v]))
    for t, k, y, route in columns:
        if x[y] > 0.5:
            skeleton[t][k] = route
    return _checked(q, skeleton, data)


def _flow_neighbor(state, data, rng, until, strength, stagnation):
    """Reweight by true route marginal costs and feed visit changes back to the integer neighbourhood."""
    T, n, m, Q = data['T'], data['n'], data['m'], data['Q']
    if not m or not n:
        return None, None
    C, demand = data['costs'], data['demands']
    now = time.monotonic()
    flow_end = now + 0.78 * max(0.0, until - now)
    owners = [[-1] * n for _ in range(T)]
    prices = [[0.0] * n for _ in range(T)]
    scale = math.exp(rng.uniform(-1.35, 1.05 + 0.10 * strength))
    noise = 0.14 + 0.04 * strength + 0.12 * min(1.0, stagnation)
    perturb = min(0.24, 0.012 * strength + 0.07 * stagnation)
    if rng.random() < 0.58:
        perturb = 0.0
    for t, period in enumerate(state['routes']):
        if time.monotonic() >= flow_end:
            return None, None
        charges = [0.0] * n
        loads = [sum(state['q'][t][c - 1] for c in route) for route in period]
        for k, route in enumerate(period):
            share = _route_cost(route, C) / max(1, len(route))
            for p, c in enumerate(route):
                owners[t][c - 1] = k
                a = 0 if p == 0 else route[p - 1]
                b = 0 if p + 1 == len(route) else route[p + 1]
                charges[c - 1] = max(0.0, C[a][c] + C[c][b] - C[a][b]) + 0.10 * share
        for i in range(n):
            k = owners[t][i]
            if k < 0 or (m > 1 and rng.random() < perturb):
                options = []
                for kk, route in enumerate(period):
                    if i + 1 in route:
                        delta = charges[i]
                    else:
                        delta, _ = _insert(route, i + 1, C)
                    shortage = max(0.0, loads[kk] + state['q'][t][i] - Q)
                    options.append((max(0.0, delta) * rng.uniform(0.86, 1.14)
                                    + shortage * max(1.0, C[0][i + 1]) / max(1.0, Q), kk, delta))
                _, newk, charge = min(options)
                if k >= 0 and newk != k:
                    loads[k] -= state['q'][t][i]
                    loads[newk] += state['q'][t][i]
                owners[t][i] = newk
                charges[i] = max(0.0, charge)
            avg = sum(demand[i]) / max(1, T)
            theta = max(1e-6, 0.015 * Q + 0.18 * avg)
            denominator = state['q'][t][i] + theta
            prices[t][i] = scale * charges[i] * math.exp(rng.uniform(-noise, noise)) / denominator
    q = _circulation(data, owners, True, flow_end, prices)
    if q is None:
        return None, None
    burden, _ = _burdens(state, data)
    signal = []
    for i in range(n):
        changed = sum((q[t][i] > _ACTIVE) != (state['q'][t][i] > _ACTIVE) for t in range(T))
        moved = sum(abs(q[t][i] - state['q'][t][i]) for t in range(T))
        signal.append((1.0 + burden[i]) * (changed + moved / max(1.0, Q)))
    routes = []
    for t in range(T):
        period = [[] for _ in range(m)]
        for k, old in enumerate(state['routes'][t]):
            period[k] = [c for c in old if owners[t][c - 1] == k and q[t][c - 1] > _ACTIVE]
        seen = {c for route in period for c in route}
        pending = [i + 1 for i in range(n) if q[t][i] > _ACTIVE and i + 1 not in seen]
        pending.sort(key=lambda c: -C[0][c])
        for c in pending:
            k = owners[t][c - 1]
            if time.monotonic() < until:
                _, p = _insert(period[k], c, C)
                period[k].insert(p, c)
            else:
                period[k].append(c)
        routes.append(period)
    candidate = _checked(q, routes, data)
    if candidate is not None:
        for t in range(T):
            now = time.monotonic()
            if now >= until:
                break
            end = now + (until - now) / max(1, T - t)
            for k in range(m):
                routes[t][k] = _two_opt(routes[t][k], C, end)
        candidate = _checked(q, routes, data)
    return candidate, signal


def _choose_group(state, data, rng, ages, signal, count, strength, quantum, now):
    n, C = data['n'], data['costs']
    burden, _ = _burdens(state, data)
    max_signal = max(signal, default=0.0) if signal is not None else 0.0
    def weight(i):
        age = min(strength, (now - ages[i]) / max(quantum, 1e-6))
        hint = 0.0 if not max_signal else signal[i] / max_signal
        return (1.0 + burden[i]) * (1.0 + age) * (1.0 + strength * hint)
    if rng.random() < 0.18:
        seed = min(range(n), key=lambda i: ages[i])
    else:
        seed = max(range(n), key=lambda i: weight(i) * rng.uniform(0.65, 1.35))
    chosen = [seed]
    if count > 1:
        same_route = [0] * n
        for period in state['routes']:
            for route in period:
                if seed + 1 in route:
                    for c in route:
                        same_route[c - 1] += 1
        others = [i for i in range(n) if i != seed]
        if rng.random() < 0.85:
            others.sort(key=lambda i: (1.0 + C[seed + 1][i + 1]) * rng.uniform(0.7, 1.3)
                        / (1.0 + 0.25 * same_route[i] + 0.35 * (signal[i] / max_signal if max_signal else 0.0)))
        else:
            rng.shuffle(others)
        chosen.extend(others[:count - 1])
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
        reserve = min(0.12 * budget, max(0.025 * budget, 2e-6 * T * (m * n + n + 1)))
        deadline = started + max(0.0, budget - reserve)
        if T == 0:
            return safe
        rng = random.Random(_seed_from_data(data))
        complexity = max(1, T * (n + m + 1))
        quantum = budget / (20.0 + math.sqrt(complexity))
        strength = math.log2(2.0 + budget)
        # First cache a structurally complete feasible solution that passes the balance check; no later neighbourhood overwrites it.
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
                period = _ffd_groups(active, q[t], m, Q)
                if period is None:
                    rr = None
                    break
                rr.append(period)
            if rr is not None:
                best = _checked(q, rr, data)
        if best is not None:
            safe = _to_output(best, data)
        base_end = min(deadline, started + 0.15 * budget)
        attempt = 0
        while best is None and time.monotonic() < base_end:
            until = min(base_end, time.monotonic() + max(quantum, 0.03 * budget))
            if attempt == 1:
                q = _circulation(data, None, False, until)
                if q is not None:
                    best = _state_from_schedule(q, data, until)
            else:
                mode = 'dynamic' if attempt == 0 else ('static' if attempt == 2 else 'random')
                owners = _make_assignment(data, rng, mode) if m else None
                q = _circulation(data, owners, False, until)
                if q is not None:
                    rr = [[[] for _ in range(m)] for _ in range(T)]
                    if owners is not None:
                        for t in range(T):
                            for i in range(n):
                                if q[t][i] > _ACTIVE:
                                    rr[t][owners[t][i]].append(i + 1)
                    best = _checked(q, rr, data)
            attempt += 1
        if best is None and _OPT_AVAILABLE and time.monotonic() < deadline:
            best = _feasibility_mip(data, min(deadline, started + 0.82 * budget))
        while best is None and time.monotonic() < deadline:
            owners = _make_assignment(data, rng, 'random') if m else None
            q = _circulation(data, owners, False, deadline)
            if q is not None:
                best = _state_from_schedule(q, data, deadline, owners)
        if best is None:
            return safe
        safe = _to_output(best, data)
        setup_end = min(deadline, time.monotonic() + 0.045 * budget)
        rr = []
        for t in range(T):
            now = time.monotonic()
            end = min(setup_end, now + max(0.0, setup_end - now) / max(1, T - t))
            built = _routes_for_amounts(best['q'][t], data, end, best['routes'][t])
            rr.append(built[0] if built is not None else best['routes'][t])
        candidate = _checked(best['q'], rr, data)
        if candidate is not None and candidate['obj'] < best['obj']:
            best = candidate
            safe = _to_output(best, data)
        fs = time.monotonic()
        candidate = _support_flow(best, data, set(), min(deadline, fs + max(quantum, 0.035 * budget)))
        flow_ema = max(1e-5, time.monotonic() - fs)
        if candidate is not None and candidate['obj'] < best['obj']:
            best = candidate
            safe = _to_output(best, data)
        current = best
        last_best = time.monotonic()
        last_reset = last_best
        last_flow = last_best - quantum * (5.0 + strength)
        model_ema = max(1e-5, quantum * 0.2)
        ages = [started] * n
        signal = None
        tested = set()
        max_group = max(1, min(n, 1 + int(strength),
                              int(math.log2(2.0 + 1200.0 * budget / complexity))))
        while time.monotonic() < deadline:
            now = time.monotonic()
            remaining = deadline - now
            stagnant = now - last_best
            elapsed = now - started
            candidate = None
            operator = 'transfer'
            try:
                draw = rng.random()
                model_ok = _OPT_AVAILABLE and remaining > max(0.006, 1.6 * model_ema)
                if now - last_flow > quantum * (5.0 + strength) and remaining > 2.0 * flow_ema:
                    draw = 0.72
                if not model_ok and draw < 0.54:
                    draw = 0.57 if rng.random() < 0.35 else rng.uniform(0.65, 1.0)
                if draw < 0.54 and n:
                    operator = 'columns'
                    depth = max(1, min(max_group,
                                      1 + int(strength * stagnant / max(1e-5, 0.12 * budget))))
                    count = min(n, max_group, 1 + int(rng.random() ** 0.7 * depth))
                    chosen = _choose_group(current, data, rng, ages, signal, count, strength, quantum, now)
                    slice_s = max(quantum * (0.60 + 0.30 * count), 2.2 * model_ema)
                    until = min(deadline, now + min(0.6 * remaining, slice_s))
                    candidate = _elastic_columns(current, chosen, data, until)
                    model_ema = 0.82 * model_ema + 0.18 * (time.monotonic() - now)
                    if signal is not None:
                        for i in chosen:
                            signal[i] *= 0.25
                elif draw < 0.66:
                    operator = 'support'
                    _, visits = _burdens(current, data)
                    available = [v for v in visits if (v[1], v[2]) not in tested]
                    if available:
                        _, t, i = max(available, key=lambda z: z[0] * rng.uniform(0.65, 1.35))
                        tested.add((t, i))
                        until = min(deadline, now + min(remaining * 0.5, max(0.6 * quantum, 2.2 * flow_ema)))
                        candidate = _support_flow(current, data, {(t, i)}, until)
                        flow_ema = 0.85 * flow_ema + 0.15 * (time.monotonic() - now)
                    else:
                        candidate = _fast_transfer(current, data, rng, deadline)
                elif draw < 0.83:
                    operator = 'flow'
                    until = min(deadline, now + min(0.55 * remaining, max(quantum, 2.8 * flow_ema)))
                    candidate, guidance = _flow_neighbor(current, data, rng, until, strength,
                                                        stagnant / max(quantum, budget * 0.15))
                    if guidance is not None:
                        signal = guidance
                    flow_ema = 0.82 * flow_ema + 0.18 * (time.monotonic() - now)
                    last_flow = time.monotonic()
                    # Feasible plans with large-scale flow changes are sent directly into the same local route procedure.
                    if candidate is not None and time.monotonic() < until:
                        polished = _route_neighbor(candidate, data, rng, until, strength, False)
                        if polished is not None and polished['obj'] < candidate['obj']:
                            candidate = polished
                elif draw < 0.95:
                    operator = 'routes'
                    slice_s = max(budget / (70.0 + 4.0 * math.sqrt(complexity)), 0.14 * quantum)
                    until = min(deadline, now + slice_s)
                    amplitude = (0.30 + 0.12 * strength) * (1.0 + min(1.0, stagnant / max(1e-5, 0.12 * budget)))
                    candidate = _route_neighbor(current, data, rng, until, amplitude, rng.random() < 0.72)
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
                    elif operator in ('flow', 'routes') and delta > 1e-8:
                        visits_count = sum(len(r) for p in best['routes'] for r in p)
                        scale = sum(best['pcost']) / max(1, visits_count)
                        phase = (elapsed / max(quantum * (2.0 + strength), 1e-6)) % 1.0
                        temperature = scale * (0.015 + 0.07 * (1.0 - phase))
                        if temperature > 0.0 and delta / temperature < 30.0 and rng.random() < math.exp(-delta / temperature):
                            current = candidate
                            tested.clear()
                now = time.monotonic()
                if now - last_reset > max(quantum, budget / (3.0 + strength)):
                    current = best
                    tested.clear()
                    last_reset = now
            except Exception:
                # A neighbourhood failure never changes the verified best feasible solution.
                continue
        return safe
    except BaseException:
        if best is not None and data is not None:
            try:
                return _to_output(best, data)
            except BaseException:
                pass
        return safe
