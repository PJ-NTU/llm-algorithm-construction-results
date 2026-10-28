"""Incremental tree search and exact regional reconstruction
Keeps several maximum-product path initial solutions, searches re-parenting, swaps and rotations with incremental tax computation, and jointly reconstructs local hierarchies by subset dynamic programming.
"""

import heapq
import math
import random
import time


def _read(instance):
    n = int(instance["N"])
    root = int(instance["target"])
    code = [1] * (n + 1)
    rate = [0.0] * (n + 1)
    profit = [0.0] * (n + 1)
    own = [0.0] * (n + 1)
    countries = instance["countries"]
    for i in range(1, n + 1):
        rec = countries.get(str(i), countries.get(i))
        code[i] = int(round(float(rec[0])))
        rate[i] = 1.0 - float(rec[2])
        profit[i] = float(rec[3])
        own[i] = profit[i] * (1.0 - float(rec[1]))
    keep = [[1.0] * (n + 1) for _ in range(n + 1)]
    withholding = instance["withholding"]
    for i in range(1, n + 1):
        row = withholding.get(str(i), withholding.get(i, {}))
        for j in range(1, n + 1):
            keep[i][j] = 1.0 - float(row.get(str(j), row.get(j, 0.0)))
    return n, root, code, rate, profit, own, keep


def _answer(data, par):
    n, root, code, rate, profit, own, keep = data
    children = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        if i != root:
            children[par[i]].append(i)
    order = [root]
    for i in order:
        order.extend(children[i])
    flow = [0.0] * (n + 1)
    gross = [0.0] * (n + 1)
    tax = [0.0] * (n + 1)
    for i in reversed(order):
        gross[i] = profit[i] + math.fsum(gross[k] for k in children[i])
        total = math.fsum(flow[k] * keep[k][i] for k in children[i])
        if code[i] == 2:
            tax[i] = (1.0 - rate[i]) * total
        elif code[i] == 3:
            tax[i] = math.fsum(max(0.0, flow[k] * keep[k][i] - rate[i] * gross[k])
                               for k in children[i])
        elif code[i] == 4:
            tax[i] = max(0.0, total - rate[i] * (gross[i] - profit[i]))
        flow[i] = own[i] + total - tax[i]
    return {
        "structure": {str(i): (0 if i == root else int(par[i])) for i in range(1, n + 1)},
        "net_dividends": {str(i): float(flow[i]) for i in range(1, n + 1)},
        "foreign_tax_paid": {str(i): float(tax[i]) for i in range(1, n + 1)},
    }


def _paths(data, mult, deadline):
    n, root, _, _, _, _, keep = data
    reach = [0.0] * (n + 1)
    par = [root] * (n + 1)
    par[root] = 0
    reach[root] = 1.0
    done = [False] * (n + 1)
    queue = [(-1.0, root)]
    while queue and time.perf_counter() < deadline:
        negative, u = heapq.heappop(queue)
        if done[u]:
            continue
        done[u] = True
        weight = -negative * mult[u]
        for v in range(1, n + 1):
            if not done[v]:
                candidate = weight * keep[v][u]
                if candidate > reach[v]:
                    reach[v] = candidate
                    par[v] = u
                    heapq.heappush(queue, (-candidate, v))
    return reach, par


class _Tree:
    def __init__(self, data, par):
        self.n, self.root, self.code, self.rate, self.profit, self.own, self.keep = data
        self.sub = [0.0] * (self.n + 1)
        self.flow = [0.0] * (self.n + 1)
        self.raw = [0.0] * (self.n + 1)
        self.clip = [0.0] * (self.n + 1)
        self.install(par)

    def children(self):
        children = [[] for _ in range(self.n + 1)]
        for v in range(1, self.n + 1):
            if v != self.root:
                children[self.par[v]].append(v)
        return children

    def install(self, par):
        self.par = list(par)
        children = self.children()
        order = [self.root]
        for v in order:
            order.extend(children[v])
        for v in reversed(order):
            self.sub[v] = self.profit[v] + math.fsum(self.sub[k] for k in children[v])
            self.raw[v] = math.fsum(self.flow[k] * self.keep[k][v] for k in children[v])
            self.clip[v] = math.fsum(min(self.flow[k] * self.keep[k][v],
                                         self.rate[v] * self.sub[k]) for k in children[v])
            self.flow[v] = self.value(v)
        return self.flow[self.root]

    def value(self, v):
        c = self.code[v]
        if c == 3:
            incoming = self.clip[v]
        elif c == 2:
            incoming = self.rate[v] * self.raw[v]
        elif c == 4:
            incoming = min(self.raw[v], self.rate[v] * (self.sub[v] - self.profit[v]))
        else:
            incoming = self.raw[v]
        return self.own[v] + incoming

    def under(self, a, v):
        while a:
            if a == v:
                return True
            a = self.par[a]
        return False

    def change(self, v, k, old_f, old_s, new_f, new_s, log=None):
        # Propagate only the changed flows and pre-tax profits; per-country credits are also updated one child branch at a time.
        par, sub, flow, raw, clip = self.par, self.sub, self.flow, self.raw, self.clip
        keep, code, rate, own, profit = self.keep, self.code, self.rate, self.own, self.profit
        while v:
            sf, ss, sr, sc = flow[v], sub[v], raw[v], clip[v]
            if log is not None:
                log.append((v, sf, ss, sr, sc))
            w = keep[k][v]
            sub[v] = ss + (new_s - old_s)
            raw[v] = sr + (new_f - old_f) * w
            c = code[v]
            if c == 3:
                clip[v] = sc + (min(new_f * w, rate[v] * new_s)
                                 - min(old_f * w, rate[v] * old_s))
                incoming = clip[v]
            elif c == 2:
                incoming = rate[v] * raw[v]
            elif c == 4:
                incoming = min(raw[v], rate[v] * (sub[v] - profit[v]))
            else:
                incoming = raw[v]
            flow[v] = own[v] + incoming
            if flow[v] == sf and sub[v] == ss:
                break
            old_f, old_s, new_f, new_s = sf, ss, flow[v], sub[v]
            k, v = v, par[v]

    def restore(self, log):
        for v, f, s, raw, clip in reversed(log):
            self.flow[v], self.sub[v], self.raw[v], self.clip[v] = f, s, raw, clip

    def move(self, v, new, log=None, parents=None):
        old = self.par[v]
        if old == new:
            return
        f, s = self.flow[v], self.sub[v]
        self.change(old, v, f, s, 0.0, 0.0, log)
        if parents is not None:
            parents.append((v, old))
        self.par[v] = new
        self.change(new, v, 0.0, 0.0, f, s, log)

    def gradients(self):
        children = self.children()
        order = [self.root]
        a = [0.0] * (self.n + 1)
        b = [0.0] * (self.n + 1)
        a[self.root] = 1.0
        for v in order:
            for k in children[v]:
                order.append(k)
                c, r, w = self.code[v], self.rate[v], self.keep[k][v]
                capped = (c == 3 and self.flow[k] * w > r * self.sub[k]) or (
                    c == 4 and self.raw[v] > r * (self.sub[v] - self.profit[v]))
                if capped:
                    a[k], b[k] = 0.0, b[v] + a[v] * r
                else:
                    a[k] = a[v] * w * (r if c == 2 else 1.0)
                    b[k] = b[v]
        return a, b, children


def _region_dp(tree, selected, deadline, eps):
    """Fix the edges outside the region and exactly enumerate subtree partitions inside it; on interruption, keep the complete tree already restored."""
    started = time.perf_counter()
    anchor = selected[0]
    ids = selected[1:] + [anchor]
    m = len(ids) - 1
    if m < 2:
        return False, 0, time.perf_counter() - started
    size = 1 << m
    all_mask = size - 1
    chosen_nodes = set(ids)
    children = tree.children()
    codes = [tree.code[v] for v in ids]
    rates = [tree.rate[v] for v in ids]
    own = [tree.own[v] for v in ids]
    base_g, base_raw, base_clip, base_cap = [], [], [], []
    for v in ids:
        fixed = [k for k in children[v] if k not in chosen_nodes]
        child_g = math.fsum(tree.sub[k] for k in fixed)
        base_g.append(tree.profit[v] + child_g)
        base_raw.append(math.fsum(tree.flow[k] * tree.keep[k][v] for k in fixed))
        base_clip.append(math.fsum(min(tree.flow[k] * tree.keep[k][v],
                                      tree.rate[v] * tree.sub[k]) for k in fixed))
        base_cap.append(tree.rate[v] * child_g)
    weights = [[tree.keep[u][v] for v in ids] for u in ids]
    gross = [0.0] * size
    for mask in range(1, size):
        bit = mask & -mask
        gross[mask] = gross[mask ^ bit] + base_g[bit.bit_length() - 1]

    def node_value(p, extra_g, incoming):
        c = codes[p]
        if c == 1:
            return own[p] + base_raw[p] + incoming
        if c == 2:
            return own[p] + rates[p] * (base_raw[p] + incoming)
        if c == 3:
            return own[p] + base_clip[p] + incoming
        return own[p] + min(base_raw[p] + incoming, base_cap[p] + rates[p] * extra_g)

    values = [[0.0] * size for _ in ids]
    branch = [[0.0] * size for _ in ids]
    forest = [[0.0] * size for _ in ids]
    branch_root = [[0] * size for _ in ids]
    partition = [[0] * size for _ in ids]
    singleton = []
    for k in range(m):
        amount = node_value(k, 0.0, 0.0) * weights[k][m]
        if codes[m] == 3:
            amount = min(amount, rates[m] * base_g[k])
        singleton.append(amount)
    single_sum = [0.0] * size
    for mask in range(1, size):
        bit = mask & -mask
        single_sum[mask] = single_sum[mask ^ bit] + singleton[bit.bit_length() - 1]
    star = single_sum[all_mask]
    best_value = tree.flow[anchor]
    best_par = None
    operations = 0

    def reconstruct(mask):
        par = tree.par[:]
        for v in ids[:-1]:
            par[v] = anchor
        pending = [(m, mask)]
        while pending:
            parent, remaining = pending.pop()
            while remaining:
                part = partition[parent][remaining]
                child = branch_root[parent][part]
                par[ids[child]] = ids[parent]
                below = part ^ (1 << child)
                if below:
                    pending.append((child, below))
                remaining ^= part
        return par

    interrupted = False
    for mask in range(1, size):
        if time.perf_counter() >= deadline:
            break
        bits = []
        rem = mask
        while rem:
            bit = rem & -rem
            k = bit.bit_length() - 1
            bits.append(k)
            below = mask ^ bit
            values[k][mask] = node_value(k, gross[below], forest[k][below])
            rem ^= bit
        pivot = mask & -mask
        other = mask ^ pivot
        splits = []
        if other:
            part = (other - 1) & other
            while True:
                splits.append((part | pivot, other ^ part))
                if not part:
                    break
                part = (part - 1) & other
        available = [m] + [p for p in range(m) if not (mask >> p) & 1]
        for p in available:
            if time.perf_counter() >= deadline:
                interrupted = True
                break
            bv, bk = -1.0, bits[0]
            cap = rates[p] * gross[mask]
            for k in bits:
                value = values[k][mask] * weights[k][p]
                if codes[p] == 3:
                    value = min(value, cap)
                if value > bv:
                    bv, bk = value, k
            branch[p][mask], branch_root[p][mask] = bv, bk
            best, chosen = bv, mask
            bp, fp = branch[p], forest[p]
            for pos, (part, remainder) in enumerate(splits):
                value = bp[part] + fp[remainder]
                if value > best:
                    best, chosen = value, part
                if (pos & 255) == 255 and time.perf_counter() >= deadline:
                    interrupted = True
                    break
            if interrupted:
                break
            fp[mask], partition[p][mask] = best, chosen
            operations += len(splits) + len(bits) + 1
            if p == m:
                value = node_value(m, gross[all_mask], best + star - single_sum[mask])
                if value > best_value + eps:
                    best_value = value
                    best_par = reconstruct(mask)
        if interrupted:
            break
    if best_par is not None:
        tree.install(best_par)
    return best_par is not None, operations, time.perf_counter() - started


class _Search:
    def __init__(self, data, star, limit, deadline):
        self.data = data
        self.n, self.root, self.code, self.rate, self.profit, self.own, self.keep = data
        self.limit, self.deadline = limit, deadline
        self.rng = random.Random(0xA71D ^ self.n ^ (self.root << 9))
        self.tree = _Tree(data, star)
        self.nodes = [v for v in range(1, self.n + 1) if v != self.root]
        self.best = self.tree.flow[self.root]
        self.best_par = star[:]
        self.eps = 1e-10 * max(1.0, math.fsum(self.own) / max(1, self.n))
        self.width = min(self.n - 1, max(4, int(6 + 2 * math.sqrt(limit) * self.n ** 0.25)))
        self.candidates = [(self.root,)] * (self.n + 1)
        self.elites = []
        self.elite_size = max(2, int(2 + math.log1p(limit) * math.log1p(self.n) ** 0.5))
        self.dp_rate = 700000.0
        self.stall = 0

    def record(self):
        value = self.tree.flow[self.root]
        if value > self.best + self.eps:
            self.best, self.best_par = value, self.tree.par[:]
            self.stall = 0
            return True
        return False

    def archive(self):
        par = tuple(self.tree.par)
        if all(par != p for _, p in self.elites):
            self.elites.append((self.tree.flow[self.root], par))
            self.elites.sort(key=lambda entry: entry[0], reverse=True)
            del self.elites[self.elite_size:]

    def initialize(self):
        mult = [self.rate[v] if self.code[v] == 2 else 1.0 for v in range(self.n + 1)]
        reach, par = _paths(self.data, mult, self.deadline)
        self.tree.install(par)
        self.record()
        self.archive()
        for capped in ((3, 4), (3,), (4,)):
            if time.perf_counter() >= self.deadline:
                break
            factors = [self.rate[v] if self.code[v] == 2 or self.code[v] in capped else 1.0
                       for v in range(self.n + 1)]
            _, par = _paths(self.data, factors, self.deadline)
            self.tree.install(par)
            self.record()
            self.archive()
        self.tree.install(self.best_par)
        for v in self.nodes:
            if time.perf_counter() >= self.deadline:
                break
            others = [j for j in range(1, self.n + 1) if j != v]
            if len(others) <= 2 * self.width:
                self.candidates[v] = tuple(others)
            else:
                ranked = heapq.nlargest(self.width, others,
                                        key=lambda j: self.keep[v][j] * mult[j] * reach[j])
                ranked += heapq.nlargest(max(2, self.width // 3), others,
                                         key=lambda j: self.keep[v][j])
                self.candidates[v] = tuple(dict.fromkeys(ranked + [self.root]))

    def parent_pool(self, v, a, b, children):
        t = self.tree
        pool = list(self.candidates[v])
        if len(pool) < self.n - 1:
            f, s = t.flow[v], t.sub[v]

            def estimate(j):
                amount = f * self.keep[v][j]
                c, r = self.code[j], self.rate[j]
                if c == 2:
                    amount *= r
                elif c == 3:
                    amount = min(amount, r * s)
                elif c == 4:
                    amount = min(t.raw[j] + amount, r * (t.sub[j] - self.profit[j] + s))
                    amount -= t.flow[j] - self.own[j]
                return a[j] * amount + b[j] * s

            pool += heapq.nlargest(max(2, self.width // 2),
                                  (j for j in range(1, self.n + 1) if j != v), key=estimate)
            pool += self.rng.sample(range(1, self.n + 1), min(self.n, max(1, self.width // 4)))
        old = t.par[v]
        pool += children[old]
        if t.par[old]:
            pool.append(t.par[old])
        return pool

    def reparent(self, v, pool):
        t = self.tree
        old = t.par[v]
        f, s = t.flow[v], t.sub[v]
        best_value, best_parent = t.flow[self.root], old
        # Detach the same subtree only once, then test the attachment points one by one.
        t.change(old, v, f, s, 0.0, 0.0)
        t.par[v] = 0
        tried = {v, old}
        for new in pool:
            if time.perf_counter() >= self.deadline:
                break
            if new in tried:
                continue
            tried.add(new)
            if t.under(new, v):
                continue
            log = []
            t.change(new, v, 0.0, 0.0, f, s, log)
            value = t.flow[self.root]
            t.restore(log)
            if value > best_value + self.eps:
                best_value, best_parent = value, new
        t.par[v] = best_parent
        t.change(best_parent, v, 0.0, 0.0, f, s)
        return best_parent != old

    def descend(self):
        while time.perf_counter() < self.deadline:
            gained = False
            a, b, children = self.tree.gradients()
            self.rng.shuffle(self.nodes)
            for v in self.nodes:
                if time.perf_counter() >= self.deadline:
                    break
                if self.reparent(v, self.parent_pool(v, a, b, children)):
                    gained = True
                    self.record()
            # Periodically re-aggregate from the tree edges to remove floating-point error accumulated by successive incremental updates.
            self.tree.install(self.tree.par)
            self.record()
            if not gained:
                break

    def compound(self):
        t = self.tree
        best_value = t.flow[self.root]
        best_ops = None

        def trial(ops):
            nonlocal best_value, best_ops
            log, parents = [], []
            for v, new in ops:
                t.move(v, new, log, parents)
            value = t.flow[self.root]
            for v, old in reversed(parents):
                t.par[v] = old
            t.restore(log)
            if value > best_value + self.eps:
                best_value, best_ops = value, ops

        for v in self.nodes:
            if time.perf_counter() >= self.deadline:
                break
            old = t.par[v]
            if old != self.root:
                trial(((v, t.par[old]), (old, v)))
            pool = list(self.candidates[v])
            pool += self.rng.sample(self.nodes, min(len(self.nodes), max(1, self.width // 2)))
            seen = set()
            for w in pool:
                if time.perf_counter() >= self.deadline:
                    break
                if w in seen or w == self.root or w == v or t.par[w] == old:
                    continue
                seen.add(w)
                if not t.under(w, v) and not t.under(v, w):
                    trial(((v, t.par[w]), (w, old)))
        if best_ops is not None:
            for v, new in best_ops:
                t.move(v, new)
            self.tree.install(self.tree.par)
            self.record()
            return True
        return False

    def region(self):
        remaining = self.deadline - time.perf_counter()
        if remaining <= 0.0 or self.n < 3:
            return False
        budget = min(remaining * 0.32, max(remaining / math.sqrt(self.n), self.limit * 0.045))
        capacity = budget * self.dp_rate
        m = 1
        while m < self.n - 1:
            next_m = m + 1
            work = (next_m + 3) * 3 ** next_m / 6 + (next_m * next_m + next_m) * 2 ** (next_m - 2)
            entries = 5 * (next_m + 1) * (1 << next_m)
            if work > capacity or entries > 2000000:
                break
            m = next_m
        if m < 2:
            return False
        t = self.tree
        children = t.children()
        if m >= self.n - 1 or self.rng.random() < 0.65:
            anchor = self.root
        else:
            internal = [v for v in range(1, self.n + 1) if children[v]]
            anchor = self.rng.choice(internal) if internal else self.root
        selected = [anchor]
        frontier = children[anchor][:]
        random_mode = self.rng.random() < 0.5
        while frontier and len(selected) < m + 1:
            if random_mode:
                index = self.rng.randrange(len(frontier))
            else:
                index = max(range(len(frontier)),
                            key=lambda j: t.sub[frontier[j]] * (0.25 + self.rng.random()))
            v = frontier.pop(index)
            selected.append(v)
            frontier.extend(children[v])
        changed, work, elapsed = _region_dp(t, selected,
                                            min(self.deadline, time.perf_counter() + budget), self.eps)
        if work and elapsed > 0.0:
            self.dp_rate = 0.5 * self.dp_rate + 0.5 * work / elapsed
        if changed:
            self.record()
        return changed

    def diversify(self):
        t, rng = self.tree, self.rng
        if rng.random() < self.stall / (self.stall + 5.0 + math.log1p(self.limit)):
            factors = [1.0] * (self.n + 1)
            for i in range(1, self.n + 1):
                if self.code[i] == 2:
                    factors[i] = self.rate[i]
                elif self.code[i] in (3, 4):
                    factors[i] = 1.0 - (1.0 - self.rate[i]) * rng.random()
            _, par = _paths(self.data, factors, self.deadline)
            t.install(par)
        else:
            if self.elites and rng.random() < 0.35:
                t.install(rng.choice(self.elites)[1])
            else:
                t.install(self.best_par)
            cap = max(1, min(len(self.nodes), int(1 + math.sqrt(self.n) *
                            math.log1p(self.limit) / (1.0 + math.log1p(self.limit)))))
            strength = 1 + rng.randrange(max(1, min(cap, 1 + int(math.sqrt(self.stall + 1)))))
            for _ in range(strength):
                if time.perf_counter() >= self.deadline:
                    break
                v = rng.choice(self.nodes)
                old = t.par[v]
                if old != self.root and rng.random() < 0.3:
                    t.move(v, t.par[old])
                    t.move(old, v)
                else:
                    new = rng.choice(self.candidates[v])
                    if new != old and new != v and not t.under(new, v):
                        t.move(v, new)
                self.record()
        self.record()

    def run(self):
        if not self.nodes:
            return
        self.initialize()
        while time.perf_counter() < self.deadline:
            self.descend()
            self.archive()
            if time.perf_counter() >= self.deadline:
                break
            if self.compound():
                continue
            if time.perf_counter() >= self.deadline:
                break
            if self.region():
                continue
            self.stall += 1
            self.diversify()


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    fallback = {"structure": {}, "net_dividends": {}, "foreign_tax_paid": {}}
    search = None
    data = None
    star = None
    try:
        limit = max(0.0, float(time_limit_s))
        if not math.isfinite(limit):
            limit = 0.0
        data = _read(instance)
        n, root = data[0], data[1]
        star = [root] * (n + 1)
        star[root] = 0
        fallback = _answer(data, star)
        deadline = started + limit * 0.965
        if n > 1 and time.perf_counter() < deadline:
            search = _Search(data, star, limit, deadline)
            search.run()
    except Exception:
        # A search exception on known input must not lose the previously saved complete feasible solution.
        pass
    if search is not None and data is not None:
        try:
            return _answer(data, search.best_par)
        except Exception:
            pass
    return fallback
