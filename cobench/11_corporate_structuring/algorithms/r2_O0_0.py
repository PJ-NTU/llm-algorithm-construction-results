"""Subset dynamic programming and iterated neighbourhood search for dividend trees
Combines maximum-retention construction, incremental subtree migration, exact local regrouping and perturbation, maintaining the best feasible solution by wall clock.
"""

import math
import random
import time


# The input material does not define the internal order of the list; here we explicitly adopt the yet-to-be-confirmed
# [tax code, domestic tax rate, foreign income tax rate, profit], with tax rates in 0 to 1.
_CODE, _DOMESTIC, _FOREIGN, _PROFIT = 0, 1, 2, 3


class _Tree:
    def __init__(self, data, parent):
        self.data = data
        self.n, self.root, self.p, self.b, self.c, self.a, self.q = data
        self.parent = parent[:]
        self.children = [[] for _ in parent]
        for u, v in enumerate(parent):
            if v >= 0:
                self.children[v].append(u)
        order = [self.root]
        for u in order:
            order.extend(self.children[u])
        self.f = [0.0] * self.n
        self.s = [0.0] * self.n
        self.h = [0.0] * self.n
        for u in reversed(order):
            self.s[u] = self.p[u] + math.fsum(self.s[v] for v in self.children[u])
            if self.c[u] == 3:
                self.h[u] = math.fsum(min(self.q[v][u] * self.f[v],
                                         self.a[u] * self.s[v])
                                      for v in self.children[u])
            else:
                self.h[u] = math.fsum(self.q[v][u] * self.f[v]
                                      for v in self.children[u])
            self.f[u] = self.out(u, self.h[u], self.s[u])

    def out(self, u, incoming, principal):
        code = self.c[u]
        if code == 2:
            return self.b[u] + self.a[u] * incoming
        if code == 4:
            return self.b[u] + min(incoming, self.a[u] * (principal - self.p[u]))
        return self.b[u] + incoming

    def arc(self, u, v, flow, principal):
        amount = self.q[u][v] * flow
        if self.c[v] == 3:
            return min(amount, self.a[v] * principal)
        return amount

    def movable(self, u, v):
        if u == self.root or u == v or self.parent[u] == v:
            return False
        while v >= 0:
            if v == u:
                return False
            v = self.parent[v]
        return True

    def move(self, u, v, commit=False):
        old = self.parent[u]
        flow, principal = self.f[u], self.s[u]
        log = []
        for start, sign in ((old, -1.0), (v, 1.0)):
            dh = sign * self.arc(u, start, flow, principal)
            ds = sign * principal
            w = start
            while w >= 0:
                of, oh, os = self.f[w], self.h[w], self.s[w]
                log.append((w, of, oh, os))
                nh, ns = oh + dh, os + ds
                nf = self.out(w, nh, ns)
                self.h[w], self.s[w], self.f[w] = nh, ns, nf
                z = self.parent[w]
                if z < 0:
                    break
                dh = self.arc(w, z, nf, ns) - self.arc(w, z, of, os)
                if dh == 0.0 and ds == 0.0:
                    break
                w = z
        value = self.f[self.root]
        if commit:
            self.children[old].remove(u)
            self.children[v].append(u)
            self.parent[u] = v
        else:
            for w, of, oh, os in reversed(log):
                self.f[w], self.h[w], self.s[w] = of, oh, os
        return value

    def solution(self):
        # Recompute with the original tax formula, avoiding writing incremental rounding error from the search into the result.
        order = [self.root]
        for u in order:
            order.extend(self.children[u])
        flows = [0.0] * self.n
        sums = [0.0] * self.n
        taxes = [0.0] * self.n
        for u in reversed(order):
            children = self.children[u]
            sums[u] = self.p[u] + math.fsum(sums[v] for v in children)
            incoming = math.fsum(flows[v] * self.q[v][u] for v in children)
            if self.c[u] == 2:
                taxes[u] = (1.0 - self.a[u]) * incoming
            elif self.c[u] == 3:
                taxes[u] = math.fsum(max(0.0, flows[v] * self.q[v][u]
                                          - self.a[u] * sums[v]) for v in children)
            elif self.c[u] == 4:
                taxes[u] = max(0.0, incoming - self.a[u]
                               * math.fsum(sums[v] for v in children))
            flows[u] = max(0.0, self.b[u] + incoming - taxes[u])
        return {
            'structure': {str(i + 1): int(self.parent[i] + 1) for i in range(self.n)},
            'net_dividends': {str(i + 1): float(flows[i]) for i in range(self.n)},
            'foreign_tax_paid': {str(i + 1): float(taxes[i]) for i in range(self.n)},
        }


def _retention_tree(data, mode, rng, deadline):
    n, root, p, b, c, a, q = data
    dist = [0.0] * n
    dist[root] = 1.0
    parent = [root] * n
    parent[root] = -1
    seen = [False] * n
    for _ in range(n):
        if time.perf_counter() >= deadline:
            return parent, None
        j = -1
        value = -1.0
        for i in range(n):
            if not seen[i] and dist[i] > value:
                value, j = dist[i], i
        seen[j] = True
        for i in range(n):
            if seen[i]:
                continue
            retain = q[i][j]
            if c[j] == 2:
                retain *= a[j]
            elif mode and c[j] in (3, 4):
                ratio = b[i] / p[i] if p[i] > 0.0 else 1.0
                if mode == 2:
                    ratio = (ratio + 0.5) * 0.5
                if ratio > 0.0:
                    retain = min(retain, a[j] / ratio)
            if mode >= 3:
                retain *= 0.85 + 0.15 * rng.random()
            candidate = dist[j] * retain
            if candidate > dist[i]:
                dist[i], parent[i] = candidate, j
    return parent, dist


def _dp_work(m):
    return (m + 2) * (3.0 ** max(0, m - 2)) * 0.5


def _block_count(seconds, rate, n):
    m = 1
    while m < n and _dp_work(m + 1) <= max(0.0, seconds) * rate:
        # Bound the dynamic-programming storage; the number of outer search rounds is not limited.
        if (m + 1) * (1 << m) > 1500000:
            break
        m += 1
    return m


def _exact_blocks(tree, selected, deadline):
    """Exactly regroup an ancestor-closed connected skeleton; unselected branches stay unchanged."""
    root = selected[-1]
    k = len(selected) - 1
    if k < 1:
        return None, True, 0.0, 0.0
    begun = time.perf_counter()
    chosen = set(selected)
    base_p, base_h = [], []
    for u in selected:
        fixed = [v for v in tree.children[u] if v not in chosen]
        base_p.append(tree.p[u] + math.fsum(tree.s[v] for v in fixed))
        base_h.append(math.fsum(tree.arc(v, u, tree.f[v], tree.s[v]) for v in fixed))
    size = 1 << k
    h = [[0.0] * size for _ in selected]
    g = [[0.0] * size for _ in selected]
    partition = [[0] * size for _ in selected]
    child_root = [[0] * size for _ in selected]
    mass = [0.0] * size
    single_sum = [0.0] * size
    single = []
    for i, u in enumerate(selected[:-1]):
        fi = tree.out(u, base_h[i], base_p[i])
        single.append(tree.arc(u, root, fi, base_p[i]))
    total_single = math.fsum(single)
    total_mass = math.fsum(base_p)
    best_value = tree.f[root]
    best_mask = -1
    operations = 0
    done = False
    all_bits = size - 1
    local_q = [[tree.q[u][v] for v in selected] for u in selected]
    local_c = [tree.c[u] for u in selected]
    local_a = [tree.a[u] for u in selected]
    local_b = [tree.b[u] for u in selected]
    own_p = [tree.p[u] for u in selected]

    def emit():
        if best_mask < 0:
            return None
        result = tree.parent[:]
        for u in selected[:-1]:
            result[u] = root
        stack = [(k, best_mask)]
        while stack:
            j, mask = stack.pop()
            while mask:
                part = partition[j][mask]
                child = child_root[j][part]
                result[selected[child]] = selected[j]
                rest = part ^ (1 << child)
                if rest:
                    stack.append((child, rest))
                mask ^= part
        return result

    for mask in range(1, size):
        if mask % 16 == 1 and time.perf_counter() >= deadline:
            break
        bit = mask & -mask
        idx = bit.bit_length() - 1
        rest = mask ^ bit
        mass[mask] = mass[rest] + base_p[idx]
        single_sum[mask] = single_sum[rest] + single[idx]
        roots = []
        bits = mask
        while bits:
            low = bits & -bits
            i = low.bit_length() - 1
            incoming = base_h[i] + h[i][mask ^ low]
            if local_c[i] == 2:
                fi = local_b[i] + local_a[i] * incoming
            elif local_c[i] == 4:
                fi = local_b[i] + min(incoming, local_a[i] * (mass[mask] - own_p[i]))
            else:
                fi = local_b[i] + incoming
            roots.append((i, fi))
            bits ^= low
        possible = []
        bits = all_bits ^ mask
        while bits:
            low = bits & -bits
            possible.append(low.bit_length() - 1)
            bits ^= low
        possible.append(k)
        interrupted = False
        for j in possible:
            cap = local_a[j] * mass[mask]
            gj, hj = g[j], h[j]
            best, who = -1.0, 0
            for i, fi in roots:
                value = local_q[i][j] * fi
                if local_c[j] == 3:
                    value = min(value, cap)
                if value > best:
                    best, who = value, i
            gj[mask] = best
            child_root[j][mask] = who
            best_part = mask
            tail = mask ^ bit
            sub = tail
            while True:
                part = bit | sub
                value = gj[part] + hj[mask ^ part]
                if value > best:
                    best, best_part = value, part
                operations += 1
                if operations % 2048 == 0 and time.perf_counter() >= deadline:
                    interrupted = True
                    break
                if sub == 0:
                    break
                sub = (sub - 1) & tail
            if interrupted:
                break
            hj[mask] = best
            partition[j][mask] = best_part
        if interrupted:
            break
        incoming = base_h[k] + h[k][mask] + total_single - single_sum[mask]
        value = tree.out(root, incoming, total_mass)
        if value > best_value + 1e-11 * max(1.0, abs(best_value)):
            best_value, best_mask = value, mask
        if mask == all_bits:
            done = True
    return emit(), done, float(operations), time.perf_counter() - begun


def _skeleton(tree, root, count, rng):
    selected = [root]
    frontier = tree.children[root][:]
    while frontier and len(selected) < count:
        if rng.random() < 0.5:
            pos = rng.randrange(len(frontier))
        else:
            pos = max(range(len(frontier)),
                      key=lambda j: (tree.s[frontier[j]] + 1.0) * rng.random())
        u = frontier[pos]
        frontier[pos] = frontier[-1]
        frontier.pop()
        selected.append(u)
        frontier.extend(tree.children[u])
    return selected[1:] + [root]


def _alternate(tree, u, v, kind):
    parent = tree.parent[:]
    if kind == 0:
        # Swap the positions of two countries; ancestor-descendant relations also remain acyclic.
        def exchange(w):
            return v if w == u else u if w == v else w
        parent[u] = exchange(tree.parent[v])
        parent[v] = exchange(tree.parent[u])
        for w in tree.children[u]:
            if w != v:
                parent[w] = v
        for w in tree.children[v]:
            if w != u:
                parent[w] = u
    else:
        old = parent[u]
        for w in tree.children[u]:
            parent[w] = old
        if kind == 1:
            parent[u] = v
        else:
            parent[u], parent[v] = parent[v], u
    return _Tree(tree.data, parent)


def _search(data, limit, started, fallback):
    n, root, p, b, c, a, q = data
    reserve = min(limit * 0.08, max(0.002, n * 0.000004))
    deadline = started + max(0.0, limit - reserve)
    tree = _Tree(data, [root if i != root else -1 for i in range(n)])
    best_parent = tree.parent[:]
    best_value = tree.f[root]
    if n <= 1 or time.perf_counter() >= deadline:
        return tree.solution()
    rng = random.Random(718243)
    tolerance = 2e-11 * max(1.0, math.fsum(abs(x) for x in b))
    seeds = []
    upper = math.fsum(b)
    if c[root] in (3, 4):
        upper = min(upper, b[root] + a[root] * (math.fsum(p) - p[root]))
    elif c[root] == 2:
        upper = min(upper, b[root] + a[root] * (math.fsum(b) - b[root]))

    def remember(candidate):
        nonlocal best_parent, best_value
        value = candidate.f[root]
        if value > best_value + tolerance:
            # Store only the recomputed complete tree; temporary incremental states are never used as the final solution.
            fresh = _Tree(data, candidate.parent)
            if fresh.f[root] > best_value + tolerance:
                best_value = fresh.f[root]
                best_parent = fresh.parent[:]

    for mode in (0, 1, 2):
        if time.perf_counter() >= deadline:
            return _Tree(data, best_parent).solution()
        parent, dist = _retention_tree(data, mode, rng, deadline)
        candidate = _Tree(data, parent)
        seeds.append(candidate.parent[:])
        remember(candidate)
        if mode == 0 and dist is not None:
            upper = min(upper, math.fsum(b[i] * dist[i] for i in range(n)))
        if best_value >= upper - tolerance:
            return _Tree(data, best_parent).solution()
    tree = _Tree(data, best_parent)
    rate = 500000.0
    remaining = deadline - time.perf_counter()
    if n <= _block_count(remaining * 0.7, rate, n):
        selected = [i for i in range(n) if i != root] + [root]
        proposal, complete, operations, elapsed = _exact_blocks(
            tree, selected, time.perf_counter() + remaining * 0.75)
        if elapsed > 0.0:
            rate = max(10000.0, operations / elapsed)
        if proposal is not None:
            tree = _Tree(data, proposal)
            remember(tree)
        if complete:
            return _Tree(data, best_parent).solution()

    strength = math.log1p(max(0.0, limit))
    width = min(n, max(4, int(4.0 + 3.0 * strength
                             + math.sqrt(max(0.0, limit) * 1800.0 / n))))
    nearest = [[] for _ in range(n)]
    for i in range(n):
        if time.perf_counter() >= deadline:
            return _Tree(data, best_parent).solution()
        ratio = b[i] / p[i] if p[i] > 0.0 else 1.0
        def edge_key(j):
            value = q[i][j]
            if c[j] == 2:
                value *= a[j]
            elif c[j] in (3, 4) and ratio > 0.0:
                value = min(value, a[j] / ratio)
            return value
        nearest[i] = sorted((j for j in range(n) if j != i),
                            key=edge_key, reverse=True)[:width]
    nodes = [i for i in range(n) if i != root]
    stagnation = 0
    seed_index = 0
    while time.perf_counter() < deadline:
        if best_value >= upper - tolerance:
            break
        improved = False
        rng.shuffle(nodes)
        for u in nodes:
            if time.perf_counter() >= deadline:
                break
            if width >= n - 1:
                candidates = nearest[u][:]
            else:
                candidates = nearest[u][:] + [rng.randrange(n) for _ in range(width)]
                candidates.append(root)
            rng.shuffle(candidates)
            value = tree.f[root]
            destination = -1
            for v in candidates:
                if time.perf_counter() >= deadline:
                    break
                if tree.movable(u, v):
                    score = tree.move(u, v)
                    if score > value + tolerance:
                        value, destination = score, v
            if destination >= 0:
                tree.move(u, destination, True)
                remember(tree)
                improved = True
        if improved:
            stagnation = 0
            # Rebuilding removes numerical error accumulated over many migrations.
            tree = _Tree(data, tree.parent)
            continue
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            break
        # Exactly regroup the skeleton of the locally optimal tree; the more time, the larger the skeleton that can be handled.
        slice_s = min(remaining * 0.3, max(0.001, limit ** 0.65 * 0.08))
        count = _block_count(slice_s * 0.75, rate, n)
        local_root = root
        if rng.random() < 0.35:
            internals = [u for u in range(n) if tree.children[u]]
            if internals:
                local_root = rng.choice(internals)
        selected = _skeleton(tree, local_root, count, rng)
        local_changed = False
        if len(selected) >= 3:
            proposal, _, operations, elapsed = _exact_blocks(
                tree, selected, min(deadline, time.perf_counter() + slice_s))
            if operations > 0 and elapsed > 0:
                rate = 0.65 * rate + 0.35 * operations / elapsed
            if proposal is not None:
                candidate = _Tree(data, proposal)
                if candidate.f[root] >= tree.f[root] - tolerance:
                    tree = candidate
                    remember(tree)
                    local_changed = True
        if local_changed:
            continue
        # Position swaps, single-country extraction and insertion fill in neighbourhoods that plain subtree migration cannot reach.
        neighborhood_end = min(deadline, time.perf_counter()
                               + min(remaining * 0.15, limit ** 0.6 * 0.025))
        while time.perf_counter() < neighborhood_end and len(nodes) > 1:
            u, v = rng.sample(nodes, 2)
            candidate = _alternate(tree, u, v, rng.randrange(3))
            if candidate.f[root] > tree.f[root] + tolerance:
                tree = candidate
                remember(tree)
                local_changed = True
                break
        if local_changed:
            stagnation = 0
            continue
        stagnation += 1
        if seed_index < len(seeds):
            tree = _Tree(data, seeds[seed_index])
            seed_index += 1
            continue
        if rng.random() < 1.0 / (2.0 + strength):
            parent, _ = _retention_tree(data, 3, rng, deadline)
            tree = _Tree(data, parent)
        else:
            tree = _Tree(data, best_parent)
            magnitude = min(n - 1, max(1, int(1.0 + strength
                                             + math.sqrt(stagnation * (1.0 + strength)))))
            attempts = 0
            changed = 0
            perturb_end = min(deadline, time.perf_counter()
                              + min(remaining * 0.1, limit ** 0.5 * 0.02))
            while changed < magnitude and time.perf_counter() < perturb_end:
                u = rng.choice(nodes)
                v = rng.choice(nearest[u]) if rng.random() < 0.7 else rng.randrange(n)
                attempts += 1
                if tree.movable(u, v):
                    tree.move(u, v, True)
                    changed += 1
                elif attempts % max(1, n) == 0:
                    # When legal migrations are rare, switch to single-country extraction instead of waiting for a fixed number of trials to run out.
                    if v != u and v != root:
                        tree = _alternate(tree, u, v, 1)
                        changed += 1
            tree = _Tree(data, tree.parent)
        remember(tree)
    return _Tree(data, best_parent).solution()


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    fallback = {'structure': {}, 'net_dividends': {}, 'foreign_tax_paid': {}}
    try:
        n = int(instance['N'])
        root = int(instance['target']) - 1
        countries = instance['countries']
        records = [countries[str(i + 1)] if str(i + 1) in countries
                   else countries[i + 1] for i in range(n)]
        p = [float(row[_PROFIT]) for row in records]
        c = [int(row[_CODE]) for row in records]
        b = [p[i] * (1.0 - float(records[i][_DOMESTIC])) for i in range(n)]
        a = [1.0 - float(row[_FOREIGN]) for row in records]
        withholding = instance['withholding']
        q = []
        for i in range(n):
            row = withholding[str(i + 1)] if str(i + 1) in withholding else withholding[i + 1]
            q.append([1.0 - float(row[str(j + 1)] if str(j + 1) in row else row[j + 1])
                      for j in range(n)])
        data = (n, root, p, b, c, a, q)
        parent = [root if i != root else -1 for i in range(n)]
        fallback = _Tree(data, parent).solution()
        limit = float(time_limit_s)
        if not math.isfinite(limit) or limit <= 0.0:
            return fallback
        return _search(data, limit, started, fallback)
    except Exception:
        # Even if the search fails, keep the complete star-shaped feasible solution already built.
        return fallback
