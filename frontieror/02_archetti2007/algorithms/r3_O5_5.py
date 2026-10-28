"""Joint search over a replenishment network and multi-segment reconnection
Combines replenishment-network integer programming, two-retailer pattern-array search and exact reconnection of route segments; optimises continuously under the wall clock and saves the best constraint-checked plan.
"""

import math
import random
import time
from array import array

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    _BACKEND = (np, Bounds, LinearConstraint, milp, coo_matrix)
except Exception:
    _BACKEND = None


class _Search:
    def __init__(self, data, seconds):
        self.start = time.perf_counter()
        self.budget = max(0.001, float(seconds))
        self.end = self.start + 0.97 * self.budget
        self.search_end = self.start + 0.93 * self.budget
        self.data = data
        self.n = int(data['num_retailers'])
        self.h = int(data['time_horizon'])
        self.d = list(map(float, data['retailers_demand']))
        self.u = list(map(float, data['retailers_max_inventory']))
        self.a = list(map(float, data['retailers_starting_inventory']))
        self.b = float(data['supplier_starting_inventory'])
        self.r = float(data['supplier_production_rate'])
        self.cap = float(data['vehicle_capacity'])
        h0 = float(data['supplier_inventory_cost'])
        self.hc = [float(v) - h0 for v in data['retailers_inventory_cost']]
        raw = data['distance_matrix']
        self.dist = [[float(raw[max(i, j)][min(i, j)]) if i != j else 0.0
                      for j in range(self.n + 1)] for i in range(self.n + 1)]
        self.rng = random.Random(31847 + 1009 * self.n + 9176 * self.h
                                 + int(data.get('seed', 0)))
        self.scale = max(1.0, sum(self.dist[0]) / max(1, self.n))
        self.cache = {0: (0.0, (0, 0))}
        self.exact_masks = {0}
        self.exact_cost = None
        self.exact_parent = None
        self.exact_last = None
        self.backend = _BACKEND
        self.backend_failed = _BACKEND is None
        self.networks = {}
        self.global_exact = False
        self.best = None
        self.best_value = math.inf
        self.plan_cache = [{} for _ in range(self.n)]
        self.patterns = [None] * self.n
        self.walks = {}
        self.dp_rate = 800000.0
        self.seed = [self.latest(s) for s in range(self.n)]
        self.near = []
        self.group_max = min(self.n, max(2, 4 + int(math.log2(1.0 + self.budget))))
        self.quantum = self.budget / (8.0 + math.sqrt(max(1, self.n * self.h)))
        self.route_slice = self.budget / (180.0 + 24.0 * max(1, self.n * self.h))

    def plan(self, s, mask):
        cached = self.plan_cache[s].get(mask)
        if cached is not None:
            return cached
        stock = self.a[s]
        if stock < -1e-7 or stock > self.u[s] + 1e-7:
            return None
        inv = [stock]
        qty = []
        for t in range(self.h):
            q = self.u[s] - stock if mask & (1 << t) else 0.0
            if q < -1e-7 or q > self.cap + 1e-7:
                return None
            stock += q - self.d[s]
            if stock < -1e-7 or stock > self.u[s] + 1e-7:
                return None
            qty.append(max(0.0, q))
            inv.append(max(0.0, stock))
        # Total inventory is determined only by production and consumption; this differential cost differs from the original objective by a constant.
        result = (mask, tuple(qty), tuple(inv), self.hc[s] * sum(inv[1:]))
        self.plan_cache[s][mask] = result
        return result

    def latest(self, s):
        stock = self.a[s]
        mask = 0
        for t in range(self.h):
            needs_later = stock < (self.h - t) * self.d[s] - 1e-7
            # A single order-up-to quantity may exceed vehicle capacity, so the retailer must be visited early while it can still be filled up.
            last_chance = stock - self.d[s] < self.u[s] - self.cap - 1e-7
            if stock < self.d[s] - 1e-7 or (needs_later and last_chance):
                mask |= 1 << t
                stock = self.u[s]
            stock -= self.d[s]
        return self.plan(s, mask)

    def state(self, plans):
        masks = [0] * self.h
        loads = [0.0] * self.h
        for s, p in enumerate(plans):
            if p is None:
                continue
            bits = p[0]
            while bits:
                bit = bits & -bits
                t = bit.bit_length() - 1
                masks[t] |= 1 << s
                loads[t] += p[1][t]
                bits ^= bit
        return masks, loads

    def load_ok(self, loads):
        cum = 0.0
        for t, q in enumerate(loads):
            cum += q
            if q > self.cap + 1e-6 or q < -1e-6 or cum > self.b + t * self.r + 1e-6:
                return False
        return True

    def valid(self, plans):
        return (len(plans) == self.n and all(p is not None for p in plans)
                and self.load_ok(self.state(plans)[1]))

    def nodes(self, mask):
        result = []
        while mask:
            bit = mask & -mask
            result.append(bit.bit_length())
            mask ^= bit
        return result

    def path_cost(self, path):
        d = self.dist
        return sum(d[a][b] for a, b in zip(path, path[1:]))

    def improve_route(self, path, cutoff, relocate=False):
        d = self.dist
        path = list(path)
        changed = True
        while changed and time.perf_counter() < cutoff:
            changed = False
            for i in range(1, len(path) - 2):
                if time.perf_counter() >= cutoff:
                    return path
                a, b = path[i - 1], path[i]
                for j in range(i + 1, len(path) - 1):
                    c, e = path[j], path[j + 1]
                    if d[a][c] + d[b][e] < d[a][b] + d[c][e] - 1e-9:
                        path[i:j + 1] = reversed(path[i:j + 1])
                        changed = True
                        break
                if changed:
                    break
            if changed or not relocate:
                continue
            # Contiguous segment relocation includes single-node relocation and can also move two or three nodes together.
            chain_width = min(len(path) - 2, max(1, int(math.log2(2.0 + self.budget))))
            for length in range(1, chain_width + 1):
                for i in range(1, len(path) - length):
                    if time.perf_counter() >= cutoff:
                        return path
                    j = i + length
                    first, last = path[i], path[j - 1]
                    gain = d[path[i - 1]][first] + d[last][path[j]] - d[path[i - 1]][path[j]]
                    shortened = path[:i] + path[j:]
                    best_delta, best_pos = -1e-9, -1
                    for pos in range(1, len(shortened)):
                        delta = d[shortened[pos - 1]][first] + d[last][shortened[pos]] - d[shortened[pos - 1]][shortened[pos]] - gain
                        if delta < best_delta:
                            best_delta, best_pos = delta, pos
                    if best_pos >= 0:
                        path = shortened[:best_pos] + path[i:j] + shortened[best_pos:]
                        changed = True
                        break
                if changed:
                    break
        return path

    def tour_dp(self, nodes, cutoff):
        n = len(nodes)
        if not n or n > 18:
            return None
        work = n * max(1, n - 1) * (1 << max(0, n - 2))
        if work > self.dp_rate * max(0.0, cutoff - time.perf_counter()):
            return None
        started = time.perf_counter()
        size = 1 << n
        dp = array('d', [math.inf]) * (size * n)
        parent = array('b', [-1]) * (size * n)
        costs = array('d', [0.0]) * size
        ends = array('b', [-1]) * size
        d = self.dist
        for mask in range(1, size):
            if (mask & 15) == 0 and time.perf_counter() >= cutoff:
                return None
            bits = mask
            tour_best = math.inf
            last = -1
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                prev = mask ^ bit
                value = d[0][nodes[j]] if not prev else math.inf
                pk = -1
                remaining = prev
                base = prev * n
                while remaining:
                    kb = remaining & -remaining
                    k = kb.bit_length() - 1
                    v = dp[base + k] + d[nodes[k]][nodes[j]]
                    if v < value:
                        value, pk = v, k
                    remaining ^= kb
                idx = mask * n + j
                dp[idx] = value
                parent[idx] = pk
                v = value + d[nodes[j]][0]
                if v < tour_best:
                    tour_best, last = v, j
                bits ^= bit
            costs[mask], ends[mask] = tour_best, last
        elapsed = time.perf_counter() - started
        self.dp_rate = 0.65 * self.dp_rate + 0.35 * work / max(1e-6, elapsed)
        return costs, parent, ends

    def all_small_tours(self, cutoff):
        result = self.tour_dp(list(range(1, self.n + 1)), cutoff)
        if result is not None:
            self.exact_cost, self.exact_parent, self.exact_last = result
            self.cache = {0: (0.0, (0, 0))}

    def exact_route(self, mask, cutoff):
        if mask in self.exact_masks or self.exact_cost is not None:
            return
        nodes = self.nodes(mask)
        n = len(nodes)
        result = self.tour_dp(nodes, cutoff)
        if result is None:
            return
        costs, parent, ends = result
        full = (1 << n) - 1
        subset, last = full, ends[full]
        path = [0]
        while subset:
            path.append(nodes[last])
            nxt = parent[subset * n + last]
            subset ^= 1 << last
            last = nxt
        path.append(0)
        self.cache[mask] = (float(costs[full]), tuple(path))
        self.exact_masks.add(mask)

    def route(self, mask, cutoff=None, hint=None):
        saved = self.cache.get(mask)
        if saved is not None:
            return saved
        if self.exact_cost is not None:
            subset = mask
            last = self.exact_last[mask]
            rev = [0]
            while subset:
                rev.append(last + 1)
                nxt = self.exact_parent[subset * self.n + last]
                subset ^= 1 << last
                last = nxt
            rev.append(0)
            result = (float(self.exact_cost[mask]), tuple(rev))
            self.cache[mask] = result
            return result
        nodes = self.nodes(mask)
        if len(nodes) <= 2:
            path = [0] + nodes + [0]
            result = (self.path_cost(path), tuple(path))
            self.cache[mask] = result
            self.exact_masks.add(mask)
            return result
        now = time.perf_counter()
        local_end = min(self.search_end, now + self.route_slice)
        if cutoff is not None:
            local_end = min(local_end, cutoff)
        d = self.dist
        if hint is None:
            bits = mask
            while bits:
                bit = bits & -bits
                old = self.cache.get(mask ^ bit)
                if old is not None:
                    hint = old[1]
                    break
                bits ^= bit
        if hint is not None:
            wanted = set(nodes)
            path = [0] + [v for v in hint[1:-1] if v in wanted] + [0]
            remaining = wanted.difference(path)
        else:
            far = max(nodes, key=lambda v: d[0][v])
            path = [0, far, 0]
            remaining = set(nodes)
            remaining.remove(far)
        pending = sorted(remaining, key=lambda v: -d[0][v])
        construction_end = self.end if cutoff is None else min(self.end, cutoff)
        for at, v in enumerate(pending):
            if time.perf_counter() >= construction_end:
                path[-1:-1] = pending[at:]
                break
            pos = min(range(1, len(path)), key=lambda j: d[path[j - 1]][v] + d[v][path[j]] - d[path[j - 1]][path[j]])
            path.insert(pos, v)
        if time.perf_counter() < local_end:
            path = self.improve_route(path, local_end)
        result = (self.path_cost(path), tuple(path))
        self.cache[mask] = result
        return result

    def value(self, plans, masks=None):
        if masks is None:
            masks = self.state(plans)[0]
        return sum(p[3] for p in plans) + sum(self.route(m)[0] for m in masks)

    def remember(self, plans):
        if not self.valid(plans):
            return False
        value = self.value(plans)
        if self.best is not None:
            self.best_value = self.value(self.best)
        if self.best is None or value < self.best_value - 1e-8:
            self.best, self.best_value = list(plans), value
            return True
        return False

    def single(self, s, masks, loads, noise, cutoff, reserve=None, relaxed=False):
        h = self.h
        route_cost = []
        for t in range(h):
            if time.perf_counter() >= cutoff:
                return None
            route_cost.append(noise[t] * 1e-6 if relaxed else
                              self.route(masks[t] | (1 << s), cutoff)[0] - self.route(masks[t], cutoff)[0] + noise[t])
        cumulative = []
        used = 0.0
        for q in loads:
            used += q
            cumulative.append(used)
        states = {self.a[s]: (0.0, 0)}
        for t in range(h):
            if time.perf_counter() >= cutoff:
                return None
            following_states = {}
            for stock, (cost, mask) in states.items():
                for z in (0, 1):
                    q = self.u[s] - stock if z else 0.0
                    following = stock + q - self.d[s]
                    if q < -1e-7 or q > self.cap + 1e-7 or following < -1e-7:
                        continue
                    own_cumulative = following - self.a[s] + (t + 1) * self.d[s]
                    total_cumulative = cumulative[t] + own_cumulative
                    reserved = 0.0 if reserve is None else reserve[t]
                    excess1 = max(0.0, loads[t] + q - self.cap)
                    excess2 = max(0.0, total_cumulative + reserved - self.b - t * self.r)
                    if not relaxed and (excess1 > 1e-6 or excess2 > 1e-6):
                        continue
                    extra = excess1 * excess1 + excess2 * excess2 if relaxed else 0.0
                    nc = cost + (0.0 if relaxed else self.hc[s] * following) + extra
                    if z:
                        nc += route_cost[t]
                    old = following_states.get(following)
                    if old is None or nc < old[0] - 1e-10:
                        following_states[following] = (nc, mask | ((1 << t) if z else 0))
            states = following_states
            if not states:
                return None
        saved = min(states.values(), key=lambda item: item[0])
        return self.plan(s, saved[1])

    def replace(self, plans, masks, loads, s, p):
        old = plans[s]
        bit = 1 << s
        for t in range(self.h):
            loads[t] += p[1][t] - (old[1][t] if old is not None else 0.0)
            masks[t] &= ~bit
            if p[0] & (1 << t):
                masks[t] |= bit
        plans[s] = p

    def descend(self, plans, cutoff):
        plans = list(plans)
        masks, loads = self.state(plans)
        changed = True
        while changed and time.perf_counter() < cutoff:
            changed = False
            order = list(range(self.n))
            self.rng.shuffle(order)
            for s in order:
                if time.perf_counter() >= cutoff:
                    break
                old = plans[s]
                outside_masks = [m & ~(1 << s) for m in masks]
                outside_loads = [loads[t] - old[1][t] for t in range(self.h)]
                p = self.single(s, outside_masks, outside_loads, [0.0] * self.h, cutoff)
                if p is None or p[0] == old[0]:
                    continue
                delta = p[3] - old[3]
                for t in range(self.h):
                    m = outside_masks[t] | ((1 << s) if p[0] & (1 << t) else 0)
                    delta += self.route(m)[0] - self.route(masks[t])[0]
                if delta < -1e-8:
                    self.replace(plans, masks, loads, s, p)
                    changed = True
        return plans

    def pairs(self, plans, cutoff):
        plans = list(plans)
        seen = set()
        width = min(self.n - 1, max(1, int(math.log2(2 + self.budget))))
        for s in range(self.n):
            for r in self.near[s][:width]:
                seen.add((min(s, r), max(s, r)))
        pairs = list(seen)
        changed = True
        while changed and time.perf_counter() < cutoff:
            changed = False
            self.rng.shuffle(pairs)
            for group in pairs:
                if time.perf_counter() >= cutoff:
                    break
                candidate = self.joint_dp(plans, group, min(cutoff, time.perf_counter() + self.quantum))
                if candidate is not None and self.value(candidate) < self.value(plans) - 1e-8:
                    plans = candidate
                    self.remember(plans)
                    changed = True
        return plans

    def outside(self, plans, group):
        masks, loads = self.state(plans)
        bits = sum(1 << s for s in group)
        for t in range(self.h):
            masks[t] &= ~bits
            loads[t] -= sum(plans[s][1][t] for s in group)
        return masks, loads

    def route_table(self, group, outside_masks, cutoff):
        count = 1 << len(group)
        bits = [0] * count
        for c in range(1, count):
            bit = c & -c
            bits[c] = bits[c ^ bit] | (1 << group[bit.bit_length() - 1])
        table = []
        for base in outside_masks:
            row = []
            for choice in range(count):
                if time.perf_counter() >= cutoff:
                    return None
                row.append(self.route(base | bits[choice], cutoff)[0])
            table.append(row)
        return table

    def joint_dp(self, plans, group, cutoff):
        if len(group) == 2 and all(self.patterns[s] is not None for s in group):
            result = self.pair_array(plans, group, cutoff)
            if result is not None or time.perf_counter() >= cutoff:
                return result
        outside_masks, loads = self.outside(plans, group)
        table = self.route_table(group, outside_masks, cutoff)
        if table is None:
            return None
        k = len(group)
        states = {tuple(self.a[s] for s in group): (0.0, (0,) * k)}
        cum = 0.0
        total_d = sum(self.d[s] for s in group)
        total_a = sum(self.a[s] for s in group)
        for t in range(self.h):
            cum += loads[t]
            future = {}
            for stocks, (cost, old_masks) in states.items():
                if time.perf_counter() >= cutoff:
                    return None
                mandatory = 0
                q = [self.u[s] - stocks[j] for j, s in enumerate(group)]
                base = [stocks[j] - self.d[s] for j, s in enumerate(group)]
                for j in range(k):
                    if base[j] < -1e-7:
                        mandatory |= 1 << j
                optional = ((1 << k) - 1) ^ mandatory
                subset = optional
                while True:
                    choice = subset | mandatory
                    shipped = sum(q[j] for j in range(k) if choice & (1 << j))
                    if loads[t] + shipped <= self.cap + 1e-6:
                        after = tuple(base[j] + (q[j] if choice & (1 << j) else 0.0) for j in range(k))
                        supply = cum + sum(after) - total_a + (t + 1) * total_d
                        if min(after) >= -1e-7 and supply <= self.b + t * self.r + 1e-6:
                            nc = cost + table[t][choice] + sum(self.hc[s] * after[j] for j, s in enumerate(group))
                            saved = future.get(after)
                            if saved is None or nc < saved[0] - 1e-9:
                                nm = tuple(old_masks[j] | ((1 << t) if choice & (1 << j) else 0) for j in range(k))
                                future[after] = (nc, nm)
                    if not subset:
                        break
                    subset = (subset - 1) & optional
            if not future:
                return None
            states = future
        saved = min(states.values(), key=lambda v: v[0])
        result = list(plans)
        for j, s in enumerate(group):
            result[s] = self.plan(s, saved[1][j])
        return result if self.valid(result) else None

    def get_backend(self):
        return None if self.backend_failed else self.backend

    def network(self, s, cutoff):
        saved = self.networks.get(s)
        if saved is not None:
            return saved
        h, demand, maximum, initial = self.h, self.d[s], self.u[s], self.a[s]
        arcs = []
        reachable = {-1}
        for p in range(-1, h):
            if time.perf_counter() >= cutoff:
                return None
            if p not in reachable:
                continue
            for t in range(p + 1, h + 1):
                if time.perf_counter() >= cutoff:
                    return None
                gap = t if p < 0 else t - p
                base = initial if p < 0 else maximum
                before = base - gap * demand
                if before < -1e-7:
                    break
                quantity = maximum - before if t < h else 0.0
                if quantity < -1e-7 or quantity > self.cap + 1e-7:
                    continue
                if t < h and maximum < demand - 1e-7:
                    continue
                holding = self.hc[s] * (gap * base - demand * gap * (gap + 1) / 2.0)
                arcs.append((p, t, max(0.0, quantity), holding))
                reachable.add(t)
        # Remove arcs that cannot reach the end; every remaining path fully satisfies the single-retailer inventory constraints.
        good = {h}
        filtered = []
        for arc in reversed(arcs):
            if arc[1] in good:
                filtered.append(arc)
                good.add(arc[0])
        filtered.reverse()
        if -1 not in good:
            return None
        self.networks[s] = filtered
        return filtered

    def master_columns(self, plans, cutoff):
        h, n = self.h, self.n
        current = self.state(plans)[0]
        available = max(0.0, cutoff - time.perf_counter())
        # Small instances use all visit subsets; larger ones use priced routes and subsets near the current routes.
        full = (self.exact_cost is not None and
                h * (n + 1) * (1 << n) <= 90000.0 * available)
        if full:
            rows = [list(range(1 << n)) for _ in range(h)]
            return rows, True
        limit = max(2 * n + h, int(2400.0 * available / max(1, h)))
        universe = list(self.cache)
        if len(universe) > limit * 3:
            universe = self.rng.sample(universe, limit * 3)
        common = set(universe)
        common.update(current)
        common.add(0)
        rows = []
        for t, mask in enumerate(current):
            if time.perf_counter() >= cutoff:
                return None, False
            required = set(current)
            required.add(0)
            required.add(mask)
            for s in range(n):
                required.add(mask ^ (1 << s))
            # Whole-group swaps/merges between two periods also enter the pool, allowing per-retailer capacity blocks to be broken.
            for other in current:
                required.add(mask | other)
                required.add(mask & other)
            candidates = list(common.difference(required))
            self.rng.shuffle(candidates)
            candidates.sort(key=lambda m: (m ^ mask).bit_count())
            take = max(0, limit - len(required))
            close = candidates[:(take + 1) // 2]
            distant = candidates[(take + 1) // 2:]
            if len(distant) > take - len(close):
                distant = self.rng.sample(distant, max(0, take - len(close)))
            required.update(close)
            required.update(distant)
            rows.append(sorted(required))
        return rows, False

    def joint_mip(self, plans, group, cutoff, feasibility=False, global_pool=False):
        if time.perf_counter() >= cutoff:
            return None
        backend = self.get_backend()
        if backend is None or time.perf_counter() >= cutoff:
            return None
        np, Bounds, LinearConstraint, milp, coo_matrix = backend
        h, k = self.h, len(group)
        nets = []
        for s in group:
            net = self.network(s, cutoff)
            if net is None:
                return None
            nets.append(net)
        if feasibility:
            outside_masks, loads = [0] * h, [0.0] * h
        else:
            outside_masks, loads = self.outside(plans, group)
        column_masks, full = None, False
        if not feasibility:
            if global_pool:
                column_masks, full = self.master_columns(plans, cutoff)
                if column_masks is None:
                    return None
            else:
                bits = [0] * (1 << k)
                for choice in range(1, 1 << k):
                    bit = choice & -choice
                    bits[choice] = bits[choice ^ bit] | (1 << group[bit.bit_length() - 1])
                column_masks = [[base | bit for bit in bits] for base in outside_masks]
        costs, integer = [], []
        incoming = [[[] for _ in range(h + 1)] for _ in group]
        outgoing = [[[] for _ in range(h + 1)] for _ in group]
        shipments = [[] for _ in range(h)]
        arcs_by_shop = []
        for j, net in enumerate(nets):
            indexed = []
            for p, t, quantity, holding in net:
                idx = len(costs)
                costs.append(0.0 if feasibility else holding)
                integer.append(1)
                incoming[j][t].append(idx)
                outgoing[j][p + 1].append(idx)
                if t < h and quantity:
                    shipments[t].append((idx, quantity))
                indexed.append((idx, p, t))
            arcs_by_shop.append(indexed)
            if time.perf_counter() >= cutoff:
                return None
        rr, cc, vv, lows, ups = [], [], [], [], []

        def add(items, low=-math.inf, high=math.inf):
            row = len(lows)
            for col, val in items:
                if val:
                    rr.append(row)
                    cc.append(col)
                    vv.append(val)
            lows.append(low)
            ups.append(high)

        for j in range(k):
            add(((idx, 1.0) for idx in outgoing[j][0]), 1.0, 1.0)
            for t in range(h):
                items = [(idx, 1.0) for idx in incoming[j][t]]
                items.extend((idx, -1.0) for idx in outgoing[j][t + 1])
                if items:
                    add(items, 0.0, 0.0)
        cumulative, prefix = 0.0, []
        for t in range(h):
            cumulative += loads[t]
            add(shipments[t], high=self.cap - loads[t])
            prefix.extend(shipments[t])
            add(prefix, high=self.b + t * self.r - cumulative)
            if time.perf_counter() >= cutoff:
                return None
            if feasibility:
                continue
            cols = []
            member_cols = [[] for _ in group]
            # Once binary replenishment paths fix the visit sets, continuous route columns may only take the same sets.
            allowed = outside_masks[t]
            for j, s in enumerate(group):
                if incoming[j][t]:
                    allowed |= 1 << s
            for mask in column_masks[t]:
                if mask & ~allowed:
                    continue
                if time.perf_counter() >= cutoff:
                    return None
                idx = len(costs)
                costs.append(self.route(mask, cutoff)[0])
                integer.append(0)
                cols.append((idx, 1.0))
                for j, s in enumerate(group):
                    if mask & (1 << s):
                        member_cols[j].append((idx, 1.0))
            if not cols:
                return None
            add(cols, 1.0, 1.0)
            for j in range(k):
                items = member_cols[j]
                items.extend((idx, -1.0) for idx in incoming[j][t])
                if items:
                    add(items, 0.0, 0.0)
        if not feasibility:
            # Keep the current feasible solution and prevent the solver from spending time on worse integer solutions.
            bound = sum(plans[s][3] for s in group)
            bound += sum(self.route(mask, cutoff)[0] for mask in self.state(plans)[0])
            add(enumerate(costs), high=bound + 1e-6)
        remaining = cutoff - time.perf_counter()
        if remaining <= self.budget * 0.001:
            return None
        try:
            size = len(costs)
            matrix = coo_matrix((vv, (rr, cc)), shape=(len(lows), size)).tocsc()
            remaining = cutoff - time.perf_counter()
            if remaining <= self.budget * 0.001:
                return None
            result = milp(np.asarray(costs), integrality=np.asarray(integer, dtype=np.uint8),
                          bounds=Bounds(np.zeros(size), np.ones(size)),
                          constraints=LinearConstraint(matrix, np.asarray(lows), np.asarray(ups)),
                          options={'time_limit': 0.80 * remaining, 'mip_rel_gap': 0.0,
                                   'presolve': True})
            if result.x is None:
                return None
            answer = [None] * self.n if feasibility else list(plans)
            for j, s in enumerate(group):
                mask = 0
                for idx, p, t in arcs_by_shop[j]:
                    v = float(result.x[idx])
                    if abs(v - round(v)) > 1e-5:
                        return None
                    if v > 0.5 and t < h:
                        mask |= 1 << t
                answer[s] = self.plan(s, mask)
            if not self.valid(answer):
                return None
            if global_pool and full and result.status == 0:
                self.global_exact = True
            return answer
        except Exception:
            self.backend_failed = True
            return None

    def reconstruct(self, plans, group, cutoff, amplitude):
        answer = list(plans)
        for s in group:
            answer[s] = None
        masks, loads = self.state(answer)
        order = list(group)
        self.rng.shuffle(order)
        if self.rng.random() < 0.6:
            order.sort(key=lambda s: (self.u[s] / max(1e-8, self.d[s]), -self.d[s]))
        reserve = [sum(max(0.0, (t + 1) * self.d[s] - self.a[s]) for s in group) for t in range(self.h)]
        common = [self.rng.uniform(-amplitude, amplitude) for _ in range(self.h)]
        for s in order:
            if time.perf_counter() >= cutoff:
                return None
            for t in range(self.h):
                reserve[t] -= max(0.0, (t + 1) * self.d[s] - self.a[s])
            noise = [0.7 * common[t] + 0.3 * self.rng.uniform(-amplitude, amplitude) for t in range(self.h)]
            p = self.single(s, masks, loads, noise, cutoff, reserve)
            if p is None:
                return None
            self.replace(answer, masks, loads, s, p)
        return answer if self.valid(answer) else None

    def violation(self, plans):
        if any(p is None for p in plans):
            return math.inf
        loads = self.state(plans)[1]
        total = 0.0
        violation = 0.0
        for t, q in enumerate(loads):
            total += q
            violation += max(0.0, q - self.cap) ** 2 + max(0.0, total - self.b - t * self.r) ** 2
        return violation

    def initial(self):
        if self.valid(self.seed):
            self.remember(self.seed)
            return
        early = min(self.end, self.start + 0.10 * self.budget)
        repair = list(self.seed)
        score = self.violation(repair)
        while self.best is None and time.perf_counter() < early:
            if not math.isfinite(score) or self.rng.random() < 0.25:
                result = self.reconstruct([None] * self.n, list(range(self.n)), early, self.scale)
                if result is not None:
                    self.remember(result)
                continue
            masks, loads = self.state(repair)
            order = list(range(self.n))
            self.rng.shuffle(order)
            for s in order:
                if time.perf_counter() >= early:
                    break
                outside_masks = [m & ~(1 << s) for m in masks]
                outside_loads = [loads[t] - repair[s][1][t] for t in range(self.h)]
                p = self.single(s, outside_masks, outside_loads,
                                [self.rng.uniform(-self.scale, self.scale) for _ in range(self.h)],
                                early, relaxed=True)
                if p is not None:
                    self.replace(repair, masks, loads, s, p)
                    if self.valid(repair):
                        self.remember(repair)
                        return
            score = self.violation(repair)
        if self.best is not None:
            return
        mip_end = min(self.end, self.start + 0.78 * self.budget)
        result = self.joint_mip(None, list(range(self.n)), mip_end, True)
        if result is not None:
            self.remember(result)
            return
        # Until a feasible solution is found, all remaining budget keeps going to feasibility repair.
        while self.best is None and time.perf_counter() < self.end:
            cutoff = min(self.end, time.perf_counter() + self.quantum)
            if not math.isfinite(score) or self.rng.random() < 0.25:
                result = self.reconstruct([None] * self.n, list(range(self.n)), cutoff, self.scale)
                if result is not None:
                    self.remember(result)
                continue
            masks, loads = self.state(repair)
            s = self.rng.randrange(self.n)
            p = self.single(s, [m & ~(1 << s) for m in masks],
                            [loads[t] - repair[s][1][t] for t in range(self.h)],
                            [self.rng.uniform(-self.scale, self.scale) for _ in range(self.h)],
                            cutoff, relaxed=True)
            if p is not None:
                trial = list(repair)
                trial[s] = p
                val = self.violation(trial)
                if val <= score + 1e-6 or self.rng.random() < 0.04:
                    repair, score = trial, val
                if self.valid(trial):
                    self.remember(trial)

    def select_group(self, plans, count):
        count = min(self.n, max(1, count))
        anchor = self.rng.randrange(self.n)
        if count == self.n:
            result = list(range(self.n))
            self.rng.shuffle(result)
            return result
        mode = self.rng.random()
        if mode < 0.45:
            nearby = self.near[anchor]
            width = min(len(nearby), max(count - 1, int(count * (1.0 + self.rng.random()))))
            return [anchor] + self.rng.sample(nearby[:width], count - 1)
        if mode < 0.78:
            target_mask = plans[anchor][0]
            ranked = sorted((s for s in range(self.n) if s != anchor),
                            key=lambda s: (-bin(target_mask & plans[s][0]).count('1'),
                                           self.dist[anchor + 1][s + 1] * self.rng.uniform(0.7, 1.3)))
            return [anchor] + ranked[:count - 1]
        return self.rng.sample(range(self.n), count)

    def make_patterns(self, cutoff):
        if self.backend is None or not self.h:
            return
        available = max(0.0, cutoff - time.perf_counter())
        if self.h > math.log2(max(2.0, 120000.0 * available / max(1, self.n))):
            return
        np = self.backend[0]
        order = list(range(self.n))
        self.rng.shuffle(order)
        for s in order:
            pool = []
            for mask in range(1 << self.h):
                if time.perf_counter() >= cutoff:
                    return
                p = self.plan(s, mask)
                if p is not None:
                    pool.append(p)
            if not pool:
                continue
            z = np.asarray([[bool(p[0] & (1 << t)) for t in range(self.h)] for p in pool], dtype=float)
            q = np.asarray([p[1] for p in pool], dtype=float)
            holding = np.asarray([p[3] for p in pool], dtype=float)
            self.patterns[s] = (pool, z, q, np.cumsum(q, axis=1), holding)

    def pair_array(self, plans, group, cutoff):
        np = self.backend[0]
        a, b = group
        pa, za, xa, ca, ha = self.patterns[a]
        pb, zb, xb, cb, hb = self.patterns[b]
        entries = len(pa) * len(pb)
        # Limit both this call's workload and temporary array size; must not eat into the time reserved for returning.
        available = max(0.0, cutoff - time.perf_counter())
        if entries * self.h > 5000000.0 * available or entries > 2000000:
            return None
        outside, loads = self.outside(plans, group)
        values = ha[:, None] + hb[None, :]
        valid = np.ones((len(pa), len(pb)), dtype=bool)
        used = 0.0
        abit, bbit = 1 << a, 1 << b
        for t, base in enumerate(outside):
            if time.perf_counter() >= cutoff:
                return None
            used += loads[t]
            valid &= xa[:, t, None] + xb[None, :, t] <= self.cap - loads[t] + 1e-6
            valid &= ca[:, t, None] + cb[None, :, t] <= self.b + t * self.r - used + 1e-6
            c0 = self.route(base, cutoff)[0]
            c1 = self.route(base | abit, cutoff)[0]
            c2 = self.route(base | bbit, cutoff)[0]
            c3 = self.route(base | abit | bbit, cutoff)[0]
            az, bz = za[:, t, None], zb[None, :, t]
            values += (c1 - c0) * az + (c2 - c0) * bz + (c3 - c1 - c2 + c0) * az * bz
        values[~valid] = np.inf
        index = int(np.argmin(values))
        i, j = divmod(index, len(pb))
        if not math.isfinite(float(values[i, j])):
            return None
        result = list(plans)
        result[a], result[b] = pa[i], pb[j]
        return result if self.valid(result) else None

    def reconnect(self, path, cutoff):
        nodes = path[1:-1]
        m = len(nodes)
        available = max(0.0, cutoff - time.perf_counter())
        if m < 3 or available <= 0.0:
            return path
        width = min(m, max(2, int(math.log2(2.0 + 12000.0 * available))))
        # The number of segments grows with the actual remaining time; the array-size bound is only a memory guard.
        while width > 2 and (4 * width * width * (1 << width) > self.dp_rate * available
                              or 2 * width * (1 << width) > 2000000):
            width -= 1
        if width < 3:
            return path
        if self.rng.random() < 0.5:
            cuts = sorted(self.rng.sample(range(1, m), width - 1))
        else:
            d = self.dist
            ranked = sorted(range(1, m), key=lambda j:
                            d[nodes[j - 1]][nodes[j]] * self.rng.uniform(0.2, 1.6), reverse=True)
            cuts = sorted(ranked[:width - 1])
        bounds = [0] + cuts + [m]
        fragments = [nodes[bounds[j]:bounds[j + 1]] for j in range(width)]
        starts, ends = [], []
        for fragment in fragments:
            starts.extend((fragment[0], fragment[-1]))
            ends.extend((fragment[-1], fragment[0]))
        dim, size = 2 * width, 1 << width
        d = self.dist
        links = [[d[u][v] for v in starts] for u in ends]
        dp = array('d', [math.inf]) * (size * dim)
        parents = array('h', [-1]) * (size * dim)
        for mask in range(1, size):
            if (mask & 7) == 0 and time.perf_counter() >= cutoff:
                return path
            bits = mask
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                prev = mask ^ bit
                base = prev * dim
                for orientation in (0, 1):
                    last = 2 * j + orientation
                    value, source = (d[0][starts[last]], -1) if not prev else (math.inf, -1)
                    left = prev
                    while left:
                        ibit = left & -left
                        i = 2 * (ibit.bit_length() - 1)
                        v = dp[base + i] + links[i][last]
                        if v < value:
                            value, source = v, i
                        v = dp[base + i + 1] + links[i + 1][last]
                        if v < value:
                            value, source = v, i + 1
                        left ^= ibit
                    dp[mask * dim + last] = value
                    parents[mask * dim + last] = source
                bits ^= bit
        mask = size - 1
        last = min(range(dim), key=lambda j: dp[mask * dim + j] + d[ends[j]][0])
        order = []
        while mask:
            order.append(last)
            prev = parents[mask * dim + last]
            mask ^= 1 << (last // 2)
            last = prev
        result = [0]
        for j in reversed(order):
            fragment = fragments[j // 2]
            result.extend(reversed(fragment) if j & 1 else fragment)
        result.append(0)
        return result if self.path_cost(result) < self.path_cost(path) - 1e-9 else path

    def polish(self, cutoff):
        if self.best is None or self.exact_cost is not None:
            return
        masks = [m for m in set(self.state(self.best)[0]) if m not in self.exact_masks]
        if not masks:
            return
        mask = max(masks, key=lambda m: self.route(m)[0] * self.rng.uniform(0.4, 1.4))
        saved = self.route(mask)
        self.exact_route(mask, min(cutoff, time.perf_counter() + (cutoff - time.perf_counter()) * 0.35))
        if mask in self.exact_masks or time.perf_counter() >= cutoff:
            return
        path = list(saved[1])
        if mask in self.walks and self.rng.random() < 0.35:
            path = list(self.walks[mask])
        if len(path) > 5 and self.rng.random() < 0.45:
            if len(path) >= 10:
                a, b, c = sorted(self.rng.sample(range(1, len(path) - 1), 3))
                path = path[:a] + path[b:c] + path[a:b] + path[c:]
            else:
                a, b = sorted(self.rng.sample(range(1, len(path) - 1), 2))
                path[a:b + 1] = reversed(path[a:b + 1])
        local_end = min(cutoff, time.perf_counter() + (cutoff - time.perf_counter()) * 0.30)
        path = self.improve_route(path, local_end, True)
        # Every reconnection includes the original segment order, so solving to completion never worsens the current route.
        while time.perf_counter() < cutoff:
            before = self.path_cost(path)
            path = self.reconnect(path, cutoff)
            if self.path_cost(path) >= before - 1e-9:
                break
            path = self.improve_route(path, min(cutoff, time.perf_counter() + self.route_slice), True)
        self.walks[mask] = tuple(path)
        value = self.path_cost(path)
        if value < saved[0] - 1e-9:
            self.cache[mask] = (value, tuple(path))

    def run(self):
        if self.n == 0 or self.h == 0:
            self.best = self.seed
            return self.output(self.best)
        self.initial()
        if self.best is None:
            return self.output(self.seed)
        pre_end = min(self.search_end, time.perf_counter() + 0.065 * self.budget)
        self.all_small_tours(pre_end)
        self.make_patterns(min(self.search_end, time.perf_counter() + 0.025 * self.budget))
        self.best_value = self.value(self.best)
        self.near = [sorted((r for r in range(self.n) if r != s),
                            key=lambda r: self.dist[s + 1][r + 1]) for s in range(self.n)]
        current = self.descend(self.best, min(self.search_end, self.start + 0.18 * self.budget))
        self.remember(current)
        current = self.pairs(current, min(self.search_end, self.start + 0.28 * self.budget))
        self.remember(current)
        self.polish(min(self.search_end, time.perf_counter() + 0.035 * self.budget))
        first_end = min(self.search_end, max(time.perf_counter(), self.start + 0.49 * self.budget))
        candidate = self.joint_mip(self.best, list(range(self.n)), first_end, global_pool=True)
        if candidate is not None:
            self.remember(candidate)
            current = list(self.best)
        last_improvement = time.perf_counter()
        # The main loop is controlled only by the wall clock; group size, destroy strength, column pool and subproblem time all scale with the budget.
        while time.perf_counter() < self.search_end:
            now = time.perf_counter()
            progress = min(1.0, (now - self.start) / self.budget)
            stalled = (now - last_improvement) / max(self.quantum, 1e-6)
            mode = self.rng.random()
            candidate = None
            if mode < 0.12:
                cutoff = min(self.search_end, now + self.quantum)
                candidate = self.descend(current, cutoff)
            elif mode < 0.35:
                count = self.rng.randint(2, max(2, min(self.group_max, 2 + int(math.log2(2 + self.budget)))))
                if self.rng.random() < 0.55:
                    count = min(2, self.n)
                group = self.select_group(current, count)
                cutoff = min(self.search_end, now + self.quantum * (1.0 + 0.25 * len(group)))
                candidate = self.joint_dp(current, group, cutoff)
            elif mode < 0.52 and not self.backend_failed:
                count = self.rng.randint(min(4, self.group_max), self.group_max)
                group = self.select_group(current, count)
                cutoff = min(self.search_end, now + self.quantum * (2.0 + math.log1p(self.budget)))
                candidate = self.joint_mip(current, group, cutoff)
            elif mode < 0.70 and not self.backend_failed and not self.global_exact:
                cutoff = min(self.search_end, now + self.quantum * (3.0 + math.log1p(self.budget)))
                candidate = self.joint_mip(self.best, list(range(self.n)), cutoff, global_pool=True)
            elif mode < 0.84:
                strength = min(self.n, max(2, int(math.sqrt(self.n) * math.log2(2 + self.budget))))
                group = self.select_group(current, self.rng.randint(1, max(1, strength)))
                cutoff = min(self.search_end, now + self.quantum * (1.0 + math.sqrt(len(group))))
                amplitude = self.scale * (0.08 + self.rng.random() * (0.25 + 0.08 * math.log1p(stalled)))
                candidate = self.reconstruct(current, group, cutoff, amplitude)
                if candidate is not None and time.perf_counter() < cutoff:
                    candidate = self.descend(candidate, cutoff)
            else:
                self.polish(min(self.search_end, now + self.quantum))
            if candidate is not None and self.valid(candidate):
                cv, old = self.value(candidate), self.value(current)
                if self.remember(candidate):
                    last_improvement = time.perf_counter()
                temperature = self.scale * (0.005 + 0.16 * (1.0 - progress) ** 2)
                if cv <= old + 1e-9 or self.rng.random() < math.exp(-min(700.0, (cv - old) / temperature)):
                    current = candidate
            if stalled > 2.0 + math.log1p(self.budget) and self.rng.random() < 0.25:
                current = list(self.best)
        while time.perf_counter() < self.end:
            if self.exact_cost is not None or all(m in self.exact_masks for m in self.state(self.best)[0]):
                group = self.select_group(self.best, min(self.n, 2))
                candidate = self.joint_dp(self.best, group, min(self.end, time.perf_counter() + self.quantum))
                if candidate is not None:
                    self.remember(candidate)
            else:
                self.polish(min(self.end, time.perf_counter() + self.quantum))
        return self.output(self.best)

    def output(self, plans):
        routes, visits, deliveries, inventories = {}, {}, {}, {}
        supplier = {'1': float(self.b)}
        stocks = list(self.a)
        b = self.b
        inventories['1'] = {str(s + 1): float(stocks[s]) for s in range(self.n)}
        for t in range(self.h):
            key = str(t + 1)
            flags = [int(p is not None and bool(p[0] & (1 << t))) for p in plans]
            quantities = [self.u[s] - stocks[s] if flags[s] else 0.0 for s in range(self.n)]
            mask = sum(1 << s for s, flag in enumerate(flags) if flag)
            saved = self.cache.get(mask)
            path = saved[1] if saved is not None else tuple([0] + self.nodes(mask) + [0])
            counts = {}
            if mask:
                for a, c in zip(path, path[1:]):
                    edge = (max(a, c), min(a, c))
                    counts[edge] = counts.get(edge, 0) + 1
            routes[key] = [[a, c, number] for (a, c), number in sorted(counts.items())]
            visits[key] = flags
            deliveries[key] = {str(s + 1): float(quantities[s]) for s in range(self.n)}
            for s in range(self.n):
                stocks[s] += quantities[s] - self.d[s]
            b += self.r - sum(quantities)
            inventories[str(t + 2)] = {str(s + 1): float(stocks[s]) for s in range(self.n)}
            supplier[str(t + 2)] = float(b)
        return {'routes': routes, 'retailer_visit_status': visits,
                'deliveries': deliveries, 'retailer_inventory': inventories,
                'supplier_inventory': supplier}


def solve(instance: dict, time_limit_s: float) -> dict:
    search = _Search(instance, time_limit_s)
    try:
        return search.run()
    except Exception:
        # An optimiser failure does not overwrite the complete feasible solution saved earlier.
        return search.output(search.best if search.best is not None else search.seed)
