"""Subtree dynamic programming and multi-neighbourhood iterated search
Computes the four tax types exactly layer by layer and combines path construction, incremental re-parenting, exact connected-subtree reconstruction and perturbation, keeping the best solution within the wall-clock limit.
"""

import math
import random
import time


# The input list follows an explicit interface convention: [tax code, domestic tax rate, foreign income tax rate, profit].
# Tax rates are in 0 to 1; the provided material does not define the list internals, so this convention still needs upstream confirmation.
_CODE, _DOMESTIC, _FOREIGN, _PROFIT = 0, 1, 2, 3


class _Tree:
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent[:]
        p, base, code, alpha, w, root = data
        n = len(p)
        children = [[] for _ in p]
        for i, j in enumerate(parent):
            if j >= 0:
                children[j].append(i)
        order = [root]
        for i in order:
            order.extend(children[i])
        self.g = p[:]
        self.f = [0.0] * n
        self.a = [0.0] * n
        self.h = [0.0] * n
        for i in reversed(order):
            self.f[i] = self.value(i)
            j = parent[i]
            if j >= 0:
                z = self.f[i] * w[i][j]
                self.g[j] += self.g[i]
                self.a[j] += z
                if code[j] == 3:
                    self.h[j] += min(z, alpha[j] * self.g[i])

    def value(self, i):
        p, base, code, alpha, w, root = self.data
        c = code[i]
        if c == 1:
            return base[i] + self.a[i]
        if c == 2:
            return base[i] + alpha[i] * self.a[i]
        if c == 3:
            return base[i] + self.h[i]
        return base[i] + min(self.a[i], alpha[i] * max(0.0, self.g[i] - p[i]))

    def children(self):
        result = [[] for _ in self.parent]
        for i, j in enumerate(self.parent):
            if j >= 0:
                result[j].append(i)
        return result

    def legal(self, i, j):
        while j >= 0:
            if j == i:
                return False
            j = self.parent[j]
        return True

    def move(self, u, v):
        p, base, code, alpha, w, root = self.data
        old_parent = self.parent[u]
        undo = []
        fu, gu = self.f[u], self.g[u]
        for start, old_f, old_g, new_f, new_g in (
                (old_parent, fu, gu, 0.0, 0.0),
                (v, 0.0, 0.0, fu, gu)):
            child, j = u, start
            while j >= 0:
                gf, ff, aa, hh = self.g[j], self.f[j], self.a[j], self.h[j]
                undo.append((j, gf, ff, aa, hh))
                rate = w[child][j]
                self.g[j] += new_g - old_g
                self.a[j] += (new_f - old_f) * rate
                if code[j] == 3:
                    self.h[j] += (min(new_f * rate, alpha[j] * new_g)
                                  - min(old_f * rate, alpha[j] * old_g))
                self.f[j] = self.value(j)
                old_f, old_g = ff, gf
                new_f, new_g = self.f[j], self.g[j]
                child, j = j, self.parent[j]
        self.parent[u] = v
        return old_parent, undo

    def restore(self, u, record):
        self.parent[u] = record[0]
        for j, g, f, a, h in reversed(record[1]):
            self.g[j], self.f[j], self.a[j], self.h[j] = g, f, a, h

    def output(self):
        p, base, code, alpha, w, root = self.data
        children = self.children()
        tax = [0.0] * len(p)
        for i in range(len(p)):
            if code[i] == 2:
                tax[i] = (1.0 - alpha[i]) * self.a[i]
            elif code[i] == 3:
                tax[i] = sum(max(0.0, self.f[k] * w[k][i] - alpha[i] * self.g[k])
                             for k in children[i])
            elif code[i] == 4:
                mass = sum(self.g[k] for k in children[i])
                tax[i] = max(0.0, self.a[i] - alpha[i] * mass)
        return {
            "structure": {str(i + 1): int(j + 1) for i, j in enumerate(self.parent)},
            "net_dividends": {str(i + 1): float(max(0.0, v)) for i, v in enumerate(self.f)},
            "foreign_tax_paid": {str(i + 1): float(max(0.0, v)) for i, v in enumerate(tax)},
        }


def _path_tree(data, density, deadline):
    p, base, code, alpha, w, root = data
    n = len(p)
    parent = [root] * n
    parent[root] = -1
    distance = [0.0] * n
    distance[root] = 1.0
    done = [False] * n
    for _ in range(n):
        if time.perf_counter() >= deadline:
            break
        v = max((i for i in range(n) if not done[i]), key=distance.__getitem__)
        done[v] = True
        for u in range(n):
            if done[u]:
                continue
            factor = w[u][v]
            if code[v] == 2:
                factor *= alpha[v]
            elif density is not None and code[v] in (3, 4) and density[u] > 0.0:
                factor = min(factor, alpha[v] / density[u])
            z = factor * distance[v]
            if z > distance[u] + 1e-15:
                distance[u] = z
                parent[u] = v
    return parent


def _exact_region(tree, nodes, deadline):
    """Subtrees outside the region stay fixed; exactly optimise all rooted trees inside the region."""
    p, base, code, alpha, w, root = tree.data
    k = len(nodes)
    if k < 2:
        return None, False
    m = k - 1
    size = 1 << m
    chosen = set(nodes)
    children = tree.children()
    own_g, ext_a, ext_h, ext_g = [], [], [], []
    for u in nodes:
        outside = [v for v in children[u] if v not in chosen]
        eg = sum(tree.g[v] for v in outside)
        ea = sum(tree.f[v] * w[v][u] for v in outside)
        eh = sum(min(tree.f[v] * w[v][u], alpha[u] * tree.g[v]) for v in outside)
        own_g.append(p[u] + eg)
        ext_g.append(eg)
        ext_a.append(ea)
        ext_h.append(eh)
    if time.perf_counter() >= deadline:
        return None, False
    mass = [0.0] * size
    dp = [[0.0] * size for _ in nodes]
    group = [[0.0] * size for _ in nodes]
    forest = [[0.0] * size for _ in nodes]
    split = [[0] * size for _ in nodes]

    def value(j, received, gross):
        u = nodes[j]
        if code[u] == 1:
            return base[u] + ext_a[j] + received
        if code[u] == 2:
            return base[u] + alpha[u] * (ext_a[j] + received)
        if code[u] == 3:
            return base[u] + ext_h[j] + received
        return base[u] + min(ext_a[j] + received, alpha[u] * (ext_g[j] + gross))

    for j in range(1, k):
        dp[j][1 << (j - 1)] = value(j, 0.0, 0.0)
    operations = 0
    for mask in range(1, size):
        if time.perf_counter() >= deadline:
            return None, False
        low = mask & -mask
        mass[mask] = mass[mask ^ low] + own_g[low.bit_length()]
        members = []
        bits = mask
        while bits:
            bit = bits & -bits
            members.append(bit.bit_length())
            bits ^= bit
        for j, u in enumerate(nodes):
            if j and mask & (1 << (j - 1)):
                continue
            edge_best = max(dp[i][mask] * w[nodes[i]][u] for i in members)
            if code[u] == 3:
                edge_best = min(edge_best, alpha[u] * mass[mask])
            gj, fj, sj = group[j], forest[j], split[j]
            gj[mask] = edge_best
            best, best_part = edge_best, mask
            free = mask ^ low
            if free:
                sub = (free - 1) & free
                while True:
                    part = sub | low
                    z = gj[part] + fj[mask ^ part]
                    if z > best:
                        best, best_part = z, part
                    operations += 1
                    if operations & 2047 == 0 and time.perf_counter() >= deadline:
                        return None, False
                    if sub == 0:
                        break
                    sub = (sub - 1) & free
            fj[mask], sj[mask] = best, best_part
            if j:
                dp[j][mask | (1 << (j - 1))] = value(j, best, mass[mask])
    full = size - 1
    optimum = value(0, forest[0][full], mass[full])
    if optimum <= tree.f[nodes[0]] + 1e-11 * max(1.0, abs(optimum)):
        return None, True
    parent = tree.parent[:]
    pending = [(0, full)]
    while pending:
        j, mask = pending.pop()
        while mask:
            part = split[j][mask]
            bits, child, val = part, -1, -1.0
            while bits:
                bit = bits & -bits
                i = bit.bit_length()
                z = dp[i][part] * w[nodes[i]][nodes[j]]
                if z > val:
                    child, val = i, z
                bits ^= bit
            parent[nodes[child]] = nodes[j]
            pending.append((child, part ^ (1 << (child - 1))))
            mask ^= part
    return parent, True


def _region(tree, count, rng):
    children = tree.children()
    root = tree.data[-1]
    possible = [i for i, ch in enumerate(children) if ch]
    if not possible:
        return [root]
    top = root if rng.random() < 0.55 else rng.choice(possible)
    result = [top]
    frontier = children[top][:]
    while frontier and len(result) < count:
        if rng.random() < 0.5:
            q = max(range(len(frontier)), key=lambda t: tree.f[frontier[t]])
        else:
            q = rng.randrange(len(frontier))
        u = frontier[q]
        frontier[q] = frontier[-1]
        frontier.pop()
        result.append(u)
        frontier.extend(children[u])
    return result


def _compound(tree, rng):
    parent = tree.parent[:]
    n = len(parent)
    root = tree.data[-1]
    u, v = rng.randrange(n), rng.randrange(n)
    if u == root or u == v:
        return None
    kind = rng.randrange(4)
    if kind == 0:
        if v == root:
            return None
        def exchange(x):
            return v if x == u else u if x == v else x
        parent = [0] * n
        for i, j in enumerate(tree.parent):
            parent[exchange(i)] = exchange(j)
    elif kind == 1:
        if v == root or not tree.legal(u, v) or not tree.legal(v, u):
            return None
        parent[u], parent[v] = parent[v], parent[u]
    elif kind == 2:
        v = parent[u]
        if v == root:
            return None
        parent[u], parent[v] = parent[v], u
    else:
        old = parent[u]
        for i in range(n):
            if parent[i] == u:
                parent[i] = old
        parent[u] = v
        for i in range(n):
            if i != u and parent[i] == v and rng.random() < 0.5:
                parent[i] = u
    return _Tree(tree.data, parent)


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    fallback = {"structure": {}, "net_dividends": {}, "foreign_tax_paid": {}}
    best_parent = None
    data = None
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        deadline = started + budget * 0.96
        n = int(instance["N"])
        root = int(instance["target"]) - 1
        parent = [root] * n
        parent[root] = -1
        fallback = {
            "structure": {str(i + 1): int(j + 1) for i, j in enumerate(parent)},
            "net_dividends": {str(i + 1): 0.0 for i in range(n)},
            "foreign_tax_paid": {str(i + 1): 0.0 for i in range(n)},
        }
        rows = instance["countries"]
        entries = [rows[str(i + 1)] if str(i + 1) in rows else rows[i + 1] for i in range(n)]
        p = [float(row[_PROFIT]) for row in entries]
        base = [p[i] * (1.0 - float(entries[i][_DOMESTIC])) for i in range(n)]
        code = [int(row[_CODE]) for row in entries]
        alpha = [1.0 - float(row[_FOREIGN]) for row in entries]
        withholding = instance["withholding"]
        w = []
        for i in range(n):
            row = withholding[str(i + 1)] if str(i + 1) in withholding else withholding[i + 1]
            w.append([1.0 - float(row[str(j + 1)] if str(j + 1) in row else row[j + 1])
                      for j in range(n)])
        data = (p, base, code, alpha, w, root)
        state = _Tree(data, parent)
        best_parent, best_score = parent[:], state.f[root]
        fallback = state.output()
        if n < 2 or time.perf_counter() >= deadline:
            return fallback
        rng = random.Random(739391)
        strength = math.log1p(budget)
        elite_limit = max(2, int(2.0 + strength + math.sqrt(n) / 4.0))
        elite = [(best_score, tuple(best_parent))]
        tolerance = 1e-11 * max(1.0, sum(abs(x) for x in p))

        def keep(candidate):
            nonlocal best_parent, best_score
            score = candidate.f[root]
            if score > best_score + tolerance:
                best_parent, best_score = candidate.parent[:], score

        # Keep the star tree first, then try the credit-free upper-bound path and the single-branch credit path.
        variants = [None, [base[i] / p[i] if p[i] > 0.0 else 0.0 for i in range(n)]]
        for density in variants:
            if time.perf_counter() >= started + budget * 0.12:
                break
            candidate = _Tree(data, _path_tree(data, density, min(deadline, started + budget * 0.12)))
            keep(candidate)
        state = _Tree(data, best_parent)

        # At small scale, directly exhaust the subset recursion over all rooted trees; on timeout, keep the existing solution.
        remaining = deadline - time.perf_counter()
        log_work = math.log(max(1, n + 2)) + max(0, n - 2) * math.log(3.0)
        if log_work < math.log(max(1.0, remaining * 4000000.0)):
            exact_end = min(deadline, time.perf_counter() + remaining * 0.42)
            result, finished = _exact_region(state, [root] + [i for i in range(n) if i != root], exact_end)
            if result is not None:
                state = _Tree(data, result)
                keep(state)
            if finished:
                return _Tree(data, best_parent).output()

        width = min(n, max(2, int(math.sqrt(n) * (1.0 + strength) + 4.0 * math.sqrt(budget))))
        region_base = min(n, max(2, int(4.0 + math.log2(1.0 + budget))))
        stagnant = 0
        sweep = 0
        while time.perf_counter() < deadline:
            before = state.f[root]
            order = [i for i in range(n) if i != root]
            if sweep == 0:
                order.sort(key=lambda i: p[i], reverse=True)
            else:
                rng.shuffle(order)
            for u in order:
                if time.perf_counter() >= deadline:
                    break
                current = state.f[root]
                selected, score = -1, current
                if width >= n:
                    destinations = list(range(n))
                    rng.shuffle(destinations)
                else:
                    fu, gu = state.f[u], state.g[u]
                    def rank(j):
                        incoming = fu * w[u][j]
                        if code[j] == 2:
                            return incoming * alpha[j]
                        if code[j] == 3:
                            return min(incoming, alpha[j] * gu)
                        if code[j] == 4:
                            old = min(state.a[j], alpha[j] * max(0.0, state.g[j] - p[j]))
                            return min(state.a[j] + incoming,
                                       alpha[j] * max(0.0, state.g[j] - p[j] + gu)) - old
                        return incoming
                    destinations = sorted(range(n), key=rank, reverse=True)[:width]
                    destinations.extend(rng.sample(range(n), min(n, max(1, width // 3))))
                    destinations.append(root)
                    ancestor = state.parent[u]
                    if ancestor >= 0 and state.parent[ancestor] >= 0:
                        destinations.append(state.parent[ancestor])
                    destinations = list(dict.fromkeys(destinations))
                for j in destinations:
                    if time.perf_counter() >= deadline:
                        break
                    if j == state.parent[u] or not state.legal(u, j):
                        continue
                    record = state.move(u, j)
                    z = state.f[root]
                    if z > score + tolerance:
                        selected, score = j, z
                    state.restore(u, record)
                if selected >= 0:
                    state.move(u, selected)
                    keep(state)
            # Recomputation removes floating-point error accumulated by incremental updates.
            state = _Tree(data, state.parent)
            keep(state)
            if time.perf_counter() >= deadline:
                break
            remaining = deadline - time.perf_counter()
            count = min(n, region_base + int(math.log2(1.0 + stagnant)))
            local_budget = min(remaining * 0.22, budget / (3.0 + math.sqrt(n)))
            while count > 2 and (count + 2) * 3.0 ** (count - 2) > max(1.0, local_budget * 3000000.0):
                count -= 1
            nodes = _region(state, count, rng)
            result, finished = _exact_region(state, nodes, min(deadline, time.perf_counter() + local_budget))
            if result is not None:
                state = _Tree(data, result)
                keep(state)
            if len(nodes) == n and finished:
                return _Tree(data, best_parent).output()
            improved = state.f[root] > before + tolerance
            stagnant = 0 if improved else stagnant + 1
            sweep += 1
            if improved:
                continue

            item = (state.f[root], tuple(state.parent))
            if all(item[1] != old[1] for old in elite):
                elite.append(item)
                elite.sort(reverse=True)
                del elite[elite_limit:]
            # Atomic position swaps, subtree exchanges, edge rotations and intermediate holding-node insertion.
            now = time.perf_counter()
            compound_end = min(deadline, now + budget / ((2.0 + strength) * (2.0 + math.sqrt(n))))
            fraction = max(0.0, (deadline - now) / max(budget, 1e-12))
            temperature = max(tolerance, (sum(base) - best_score) * fraction * fraction /
                              (max(1, n) * (3.0 + strength)))
            while time.perf_counter() < compound_end:
                candidate = _compound(state, rng)
                if candidate is None:
                    continue
                delta = candidate.f[root] - state.f[root]
                if delta >= 0.0 or rng.random() < math.exp(max(-700.0, delta / temperature)):
                    state = candidate
                    keep(state)
            if state.f[root] > before + tolerance:
                continue
            if rng.random() < 0.65:
                state = _Tree(data, best_parent)
            else:
                state = _Tree(data, list(rng.choice(elite)[1]))
            amplitude = max(2, int(math.sqrt(n) * (1.0 + strength) *
                                   (0.15 + stagnant / (stagnant + 5.0))))
            kicks = 1 + rng.randrange(amplitude)
            for _ in range(kicks):
                if time.perf_counter() >= deadline:
                    break
                u, j = rng.randrange(n), rng.randrange(n)
                if u != root and j != state.parent[u] and state.legal(u, j):
                    state.move(u, j)
                    keep(state)
        return _Tree(data, best_parent).output()
    except Exception:
        if data is not None and best_parent is not None:
            try:
                return _Tree(data, best_parent).output()
            except Exception:
                pass
        return fallback
