"""Adaptive hierarchy search with shared root responses
Accelerates exact move-and-attach with path initial solutions and a piecewise-linear root-response cache, improving continuously with neutral perturbation, rotation swaps and multi-start.
"""

import heapq
import math
import random
import time


def _read(instance):
    n, root = int(instance["N"]), int(instance["target"])
    codes = [1] * (n + 1)
    rates = [0.0] * (n + 1)
    profit = [0.0] * (n + 1)
    own = [0.0] * (n + 1)
    countries = instance["countries"]
    # The positional definition follows the reference implementation provided in this round: tax regime, domestic tax rate, foreign tax rate, profit.
    for i in range(1, n + 1):
        rec = countries[str(i)] if str(i) in countries else countries[i]
        codes[i] = int(rec[0])
        rates[i] = float(rec[2])
        profit[i] = float(rec[3])
        own[i] = profit[i] * (1.0 - float(rec[1]))
    keep = [[1.0] * (n + 1) for _ in range(n + 1)]
    raw = instance["withholding"]
    for i in range(1, n + 1):
        row = raw[str(i)] if str(i) in raw else raw[i]
        for j, w in row.items():
            keep[i][int(j)] = 1.0 - float(w)
    return n, root, codes, rates, profit, own, keep


def _answer(data, par):
    n, root, codes, rates, profit, own, keep = data
    children = [[] for _ in range(n + 1)]
    for v in range(1, n + 1):
        if v != root:
            children[par[v]].append(v)
    order = [root]
    for v in order:
        order.extend(children[v])
    mass = [0.0] * (n + 1)
    flow = [0.0] * (n + 1)
    tax = [0.0] * (n + 1)
    for v in reversed(order):
        mass[v] = profit[v] + math.fsum(mass[k] for k in children[v])
        received = math.fsum(flow[k] * keep[k][v] for k in children[v])
        if codes[v] == 2:
            tax[v] = rates[v] * received
        elif codes[v] == 3:
            tax[v] = math.fsum(max(0.0, flow[k] * keep[k][v] -
                                       (1.0 - rates[v]) * mass[k]) for k in children[v])
        elif codes[v] == 4:
            tax[v] = max(0.0, received - (1.0 - rates[v]) *
                         math.fsum(mass[k] for k in children[v]))
        flow[v] = own[v] + received - tax[v]
    return {
        "structure": {str(i): int(par[i]) for i in range(1, n + 1)},
        "net_dividends": {str(i): float(flow[i]) for i in range(1, n + 1)},
        "foreign_tax_paid": {str(i): float(tax[i]) for i in range(1, n + 1)},
    }


class _Search:
    def __init__(self, data, star, budget, deadline):
        self.data = data
        self.n, self.root, self.code, rate, self.profit, self.own, self.keep = data
        self.rest = [1.0 - r for r in rate]
        self.ids = list(range(1, self.n + 1))
        self.nodes = [v for v in self.ids if v != self.root]
        self.budget, self.deadline = budget, deadline
        self.clock = time.perf_counter
        self.rng = random.Random(0x5EED ^ self.n ^ (self.root << 8))
        self.effort = math.log1p(budget)
        self.width = min(self.n - 1, max(2, int(5 + math.sqrt(self.n) *
                                               (0.5 + 0.7 * self.effort))))
        self.population = max(2, int(2 + self.effort * math.sqrt(math.log1p(self.n))))
        self.patience = max(2.0, math.sqrt(self.n) + self.effort)
        self.eps = max(1e-10, math.fsum(abs(x) for x in self.own) * 2e-12)
        self.par = star[:]
        self.kids = [set() for _ in range(self.n + 1)]
        self.mass = [0.0] * (self.n + 1)
        self.flow = [0.0] * (self.n + 1)
        self.acc = [0.0] * (self.n + 1)
        self.a = [0.0] * (self.n + 1)
        self.b = [0.0] * (self.n + 1)
        self.h = [0.0] * (self.n + 1)
        self.mark = [0] * (self.n + 1)
        self.epoch = 0
        self.candidates = [self.ids for _ in range(self.n + 1)]
        self.near = []
        self.elites = []
        self.best = self.install(star)
        self.best_par = star[:]

    def value(self, v, a, s):
        c = self.code[v]
        if c == 2:
            a *= self.rest[v]
        elif c == 4:
            a = min(a, self.rest[v] * (s - self.profit[v]))
        return self.own[v] + a

    def install(self, pa):
        self.par[:] = pa
        for ch in self.kids:
            ch.clear()
        for v in self.nodes:
            self.kids[self.par[v]].add(v)
        order = [self.root]
        for u in order:
            order.extend(self.kids[u])
        for u in reversed(order):
            s, a = self.profit[u], 0.0
            for v in self.kids[u]:
                s += self.mass[v]
                x = self.flow[v] * self.keep[v][u]
                if self.code[u] == 3:
                    x = min(x, self.rest[u] * self.mass[v])
                a += x
            self.mass[u], self.acc[u] = s, a
            self.flow[u] = self.value(u, a, s)
        return self.flow[self.root]

    def record(self):
        if self.flow[self.root] > self.best + self.eps:
            self.best = self.flow[self.root]
            self.best_par = self.par[:]
            return True
        return False

    def bubble(self, u, child, oldf, olds, newf, news, log):
        flow, mass, acc = self.flow, self.mass, self.acc
        code, rest, keep, par = self.code, self.rest, self.keep, self.par
        while u:
            f, s, a = flow[u], mass[u], acc[u]
            log.append((u, f, s, a))
            w = keep[child][u]
            if code[u] == 3:
                a += min(newf * w, rest[u] * news) - min(oldf * w, rest[u] * olds)
            else:
                a += (newf - oldf) * w
            ns = s + (news - olds)
            nf = self.value(u, a, ns)
            flow[u], mass[u], acc[u] = nf, ns, a
            if nf == f and ns == s:
                break
            child, oldf, olds, newf, news = u, f, s, nf, ns
            u = par[u]

    def restore(self, log):
        for u, f, s, a in reversed(log):
            self.flow[u], self.mass[u], self.acc[u] = f, s, a

    def under(self, u, v):
        while u:
            if u == v:
                return True
            u = self.par[u]
        return False

    def move(self, v, p, log, edits):
        old = self.par[v]
        if not p or v == self.root or p == old or self.under(p, v):
            return False
        f, s = self.flow[v], self.mass[v]
        self.bubble(old, v, f, s, 0.0, 0.0, log)
        self.kids[old].remove(v)
        self.par[v] = p
        self.kids[p].add(v)
        edits.append((v, old, p))
        self.bubble(p, v, 0.0, 0.0, f, s, log)
        return True

    def undo(self, log, edits):
        for v, old, new in reversed(edits):
            self.kids[new].remove(v)
            self.kids[old].add(v)
            self.par[v] = old
        self.restore(log)

    def choose(self, v, pool, stop):
        """Remove once first; each common ancestor edge composes the exact root response only once."""
        par, flow, mass, acc = self.par, self.flow, self.mass, self.acc
        code, rest, keep, profit = self.code, self.rest, self.keep, self.profit
        a, b, h, mark = self.a, self.b, self.h, self.mark
        root, eps, clock = self.root, self.eps, self.clock
        old, original = par[v], flow[root]
        fv, sv = flow[v], mass[v]
        removed = []
        self.bubble(old, v, fv, sv, 0.0, 0.0, removed)
        par[v] = 0
        base = flow[root]
        self.epoch += 1
        stamp = self.epoch
        mark[root], a[root], b[root], h[root] = stamp, 1.0, 0.0, float("inf")
        bestp, bestval = old, original
        altp, altval, ties = old, -float("inf"), 0
        try:
            for p in pool:
                if clock() >= stop:
                    break
                if p == v or p == old or not p or mark[p] == -stamp:
                    continue
                if mark[p] != stamp:
                    path = []
                    u = p
                    while u and mark[u] not in (stamp, -stamp):
                        path.append(u)
                        u = par[u]
                    if not u or mark[u] == -stamp:
                        for x in path:
                            mark[x] = -stamp
                        continue
                    for x in reversed(path):
                        q = par[x]
                        w, c, r = keep[x][q], code[q], rest[q]
                        aq, bq, hq = a[q], b[q], h[q]
                        ax, bx, hx = aq * w, bq, hq
                        if c == 2:
                            ax *= r
                        elif c == 3 or c == 4:
                            if c == 3:
                                raw, cap = flow[x] * w, r * mass[x]
                            else:
                                raw, cap = acc[q], r * (mass[q] - profit[q])
                            # The single-edge response is min(w*d + excess, r*S + slack).
                            extra = max(0.0, raw - cap)
                            slack = max(0.0, cap - raw)
                            bx = bq + aq * extra
                            hx = min(hq, bq + aq * (r * sv + slack))
                        a[x], b[x], h[x], mark[x] = ax, bx, hx, stamp
                incoming = fv * keep[v][p]
                c, r = code[p], rest[p]
                if c == 2:
                    incoming *= r
                elif c == 3:
                    incoming = min(incoming, r * sv)
                elif c == 4:
                    cap = r * (mass[p] - profit[p])
                    incoming = min(acc[p] + incoming, cap + r * sv) - min(acc[p], cap)
                val = base + min(a[p] * incoming + b[p], h[p])
                if val > bestval + eps:
                    bestp, bestval = p, val
                if val > altval + eps:
                    altp, altval, ties = p, val, 1
                elif abs(val - altval) <= eps:
                    ties += 1
                    if self.rng.randrange(ties) == 0:
                        altp, altval = p, val
        finally:
            par[v] = old
            self.restore(removed)
        return bestp, bestval, altp, altval

    def paths(self, mult, noise=0.0):
        pa = [self.root] * (self.n + 1)
        pa[0] = pa[self.root] = 0
        dist = [0.0] * (self.n + 1)
        dist[self.root] = 1.0
        done = [False] * (self.n + 1)
        heap = [(-1.0, self.root)]
        while heap and self.clock() < self.deadline:
            neg, u = heapq.heappop(heap)
            if done[u]:
                continue
            done[u] = True
            onward = -neg * mult[u]
            for v in self.ids:
                if not done[v]:
                    z = onward * self.keep[v][u]
                    if noise:
                        z *= 1.0 - noise * self.rng.random()
                    if z > dist[v]:
                        dist[v], pa[v] = z, u
                        heapq.heappush(heap, (-z, v))
        return pa, dist

    def archive(self):
        pa = tuple(self.par)
        if all(p != pa for _, p in self.elites):
            # On ties keep the new structure, so neutral migrations on a plateau are not discarded immediately.
            self.elites.insert(0, (self.flow[self.root], pa))
            self.elites.sort(key=lambda z: z[0], reverse=True)
            del self.elites[self.population:]

    def initialize(self):
        optimistic = [self.rest[v] if self.code[v] == 2 else 1.0
                      for v in range(self.n + 1)]
        conservative = [self.rest[v] if self.code[v] != 1 else 1.0
                        for v in range(self.n + 1)]
        first, reach = self.paths(optimistic)
        self.install(first)
        self.record()
        self.archive()
        lowreach = reach
        for selected in ((3, 4), (3,), (4,)):
            if self.clock() >= self.deadline:
                return
            mult = [self.rest[v] if self.code[v] == 2 or self.code[v] in selected
                    else 1.0 for v in range(self.n + 1)]
            pa, dr = self.paths(mult)
            if len(selected) == 2:
                lowreach = dr
            self.install(pa)
            self.record()
            self.archive()
        for v in self.nodes:
            if self.clock() >= self.deadline:
                return
            if self.n - 1 <= 2 * self.width:
                self.candidates[v] = self.ids
                continue
            row = self.keep[v]
            pool = (p for p in self.ids if p != v)
            top = heapq.nlargest(self.width, pool,
                                key=lambda p: row[p] * optimistic[p] * reach[p])
            width2 = max(1, self.width // 2)
            low = heapq.nlargest(width2, self.ids,
                                key=lambda p: row[p] * conservative[p] * lowreach[p])
            plain = heapq.nlargest(width2, self.ids, key=lambda p: row[p])
            self.candidates[v] = list(dict.fromkeys(top + low + plain + [self.root]))
        self.install(self.best_par)

    def descend(self, stop, full=False):
        while self.clock() < stop:
            began = self.clock()
            gained = False
            near = []
            order = self.nodes[:]
            self.rng.shuffle(order)
            candidate_count = 0
            completed = True
            for v in order:
                if self.clock() >= stop:
                    completed = False
                    break
                pool = self.ids if full else self.candidates[v]
                if len(pool) < self.n:
                    pool = pool + [self.par[self.par[v]], self.rng.choice(self.ids)]
                candidate_count += len(pool)
                base = self.flow[self.root]
                p, val, alt, av = self.choose(v, pool, stop)
                if alt != self.par[v]:
                    near.append((av - base, v, alt))
                if p != self.par[v] and val > base + self.eps:
                    self.move(v, p, [], [])
                    self.record()
                    gained = True
            self.install(self.par[:])
            self.record()
            if not gained:
                near.sort(reverse=True)
                self.near = near
                # Whether a full neighbourhood is worth adding is decided by the measured throughput of this run, not by a hard size cutoff.
                elapsed = self.clock() - began
                forecast = elapsed * self.n * len(self.nodes) / max(1, candidate_count)
                if (completed and not full and candidate_count < self.n * len(self.nodes)
                        and forecast * (2.0 + self.effort) < stop - self.clock()):
                    full = True
                    continue
                return

    def compound(self, stop):
        order = self.nodes[:]
        self.rng.shuffle(order)
        for v in order:
            if self.clock() >= stop:
                break
            old = self.par[v]
            pairs = []
            if old != self.root:
                pairs.append(((v, self.par[old]), (old, v)))
            w = self.rng.choice(self.nodes)
            if (v != w and self.par[w] != old and not self.under(w, v)
                    and not self.under(v, w)):
                pairs.append(((v, self.par[w]), (w, old)))
            for ops in pairs:
                if self.clock() >= stop:
                    break
                baseline = self.flow[self.root]
                log, edits = [], []
                for u, p in ops:
                    self.move(u, p, log, edits)
                if self.flow[self.root] > baseline + self.eps:
                    self.record()
                    return True
                self.undo(log, edits)
        return False

    def perturb(self, stall):
        rng = self.rng
        strength = min(len(self.nodes), max(1, int(1 + self.effort / 2.0 +
                       math.sqrt(self.n) * math.sqrt(stall / self.patience))))
        amount = rng.randint(1, strength)
        near_width = min(len(self.near), max(1, int((1.0 + self.effort) * math.sqrt(self.n))))
        for _ in range(amount):
            if self.clock() >= self.deadline:
                break
            if near_width and rng.random() < 0.65:
                _, v, p = self.near[int(near_width * rng.random() ** 2)]
            else:
                v = rng.choice(self.nodes)
                pool = self.candidates[v]
                p = rng.choice(pool) if rng.random() < 0.85 else rng.choice(self.ids)
            if not self.move(v, p, [], []):
                old = self.par[v]
                if old != self.root and rng.random() < 0.5:
                    self.move(v, self.par[old], [], [])
                    self.move(old, v, [], [])
            self.record()

    def run(self):
        self.initialize()
        stall = 0
        while self.clock() < self.deadline:
            before = self.best
            self.descend(self.deadline)
            self.archive()
            if self.clock() >= self.deadline:
                break
            stall = 0 if self.best > before + self.eps else stall + 1
            # A small fraction of time goes to low-cost joint edge changes, without letting exponential neighbourhoods crowd out the main search.
            if self.rng.random() < 1.0 / (2.0 + self.effort):
                span = min((self.deadline - self.clock()) * 0.08,
                           self.budget / ((4.0 + self.effort) * (1.0 + math.sqrt(self.n))))
                if self.compound(min(self.deadline, self.clock() + span)):
                    continue
            if self.clock() >= self.deadline:
                break
            restart_prob = 0.35 * stall / (self.patience + stall)
            if self.rng.random() < restart_prob:
                mult = [1.0] * (self.n + 1)
                for v in self.ids:
                    if self.code[v] == 2:
                        mult[v] = self.rest[v]
                    elif self.code[v] in (3, 4):
                        mult[v] = 1.0 - (1.0 - self.rest[v]) * self.rng.random()
                pa, _ = self.paths(mult, self.rng.random() / (2.0 + self.effort))
                self.install(pa)
                self.record()
            else:
                q = self.rng.random()
                if q < 0.45:
                    self.install(self.best_par)
                elif q < 0.65 and self.elites:
                    idx = int(len(self.elites) * self.rng.random() ** 2)
                    self.install(self.elites[idx][1])
                self.perturb(stall)


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    fallback = {"structure": {}, "net_dividends": {}, "foreign_tax_paid": {}}
    search = None
    data = None
    try:
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget < 0.0:
            budget = 0.0
        data = _read(instance)
        n, root = data[:2]
        if n <= 0:
            return fallback
        star = [root] * (n + 1)
        star[0] = star[root] = 0
        fallback = _answer(data, star)
        elapsed = time.perf_counter() - started
        reserve = min(budget * 0.08, max(budget * 0.025, elapsed * 0.3))
        deadline = started + budget - reserve
        if n > 1 and time.perf_counter() < deadline:
            search = _Search(data, star, budget, deadline)
            search.run()
    except Exception:
        # Any search interruption keeps the previous complete feasible tree; invalid input never enters the search.
        pass
    if search is not None:
        try:
            return _answer(data, search.best_par)
        except Exception:
            pass
    return fallback
