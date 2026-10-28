"""Schedule-linked adaptive large neighbourhood search
Combines cross-day customer removal, regret insertion, capacity backtracking and route exchange search, keeping the complete feasible solution with minimum total distance.
"""

import itertools
import math
import random
import time


class _Solver:
    def __init__(self, data, seconds, start):
        self.n = data['num_customers']
        self.t = data['planning_period']
        self.kout = data['num_vehicles']
        self.k = min(self.kout, max(1, self.n))
        self.Q = float(data['vehicle_capacity'])
        self.tol = 1.e-10 * max(1.0, self.Q)
        self.q = [0.0] + list(map(float, data['demands']))
        self.d = data['distance_matrix']
        self.candidates = data['candidate_schedules']
        self.patterns = [[]]
        vectors = data['schedule_vectors']
        for cs in self.candidates:
            self.patterns.append([tuple(t for t, v in enumerate(vectors[s]) if v) for s in cs])
        self.start = start
        self.seconds = max(0.0, float(seconds))
        self.end = start + self.seconds - min(self.seconds * .025, .12)
        self.rng = random.Random(731829)
        self.scale = max(1.e-12, sum(self.d[0][1:]) / max(1, self.n))
        self.lam = 8.0 * self.scale / max(1.e-12, sum(self.q) / max(1, self.n))
        self.eps = 1.e-10 * max(1.0, self.scale)
        self.width = min(self.n, max(8, int(math.sqrt(self.n) * (1.5 + .5 * math.log1p(self.seconds)))))
        self.near = [[] for _ in range(self.n + 1)]
        self.best = None
        self.bestcost = float('inf')
        self.bestsel = None
        self.y = [[[[0 for _ in range(self.t)] for _ in range(self.kout)]
                   for _ in range(self.n + 1)] for _ in range(self.n + 1)]
        self.u = [[[self.q[i] for _ in range(self.t)] for _ in range(self.kout)]
                  for i in range(1, self.n + 1)]

    def empty(self):
        return [[[] for _ in range(self.k)] for _ in range(self.t)]

    def loads(self, routes):
        return [[sum(self.q[i] for i in r) for r in day] for day in routes]

    def cost(self, routes):
        d = self.d
        total = 0.0
        for day in routes:
            for r in day:
                prev = 0
                for i in r:
                    total += d[prev][i]
                    prev = i
                total += d[prev][0]
        return total

    def excess(self, loads):
        return sum(max(0.0, v - self.Q - self.tol) for day in loads for v in day)

    def save(self, routes, sel, loads=None):
        if loads is None:
            loads = self.loads(routes)
        if any(v > self.Q + self.tol for day in loads for v in day):
            return False
        value = self.cost(routes)
        if value < self.bestcost:
            self.bestcost = value
            self.best = [[r[:] for r in day] for day in routes]
            self.bestsel = sel[:]
            return True
        return False

    def repair(self, routes, sel, missing, hard, noisy, deadline):
        """The insertion-cost cache is invalidated only after the corresponding route changes."""
        loads = self.loads(routes)
        versions = [[0] * self.k for _ in range(self.t)]
        cache = {}
        remaining = list(missing)
        rng, d, q, Q, lam = self.rng, self.d, self.q, self.Q, self.lam
        # A customer's random preference stays fixed within a single repair.
        jitter = {i: rng.uniform(.85, 1.15) if noisy else 1.0 for i in remaining}
        inf = float('inf')
        while remaining:
            if time.perf_counter() >= deadline:
                return False
            selected = None
            priority = -inf
            for i in remaining:
                qi = q[i]
                day_options = {}
                for days in self.patterns[i]:
                    for t in days:
                        if t in day_options:
                            continue
                        first = (inf, -1, -1)
                        second = inf
                        empty_seen = False
                        for h, route in enumerate(routes[t]):
                            load = loads[t][h]
                            if hard and load + qi > Q + self.tol:
                                continue
                            if not route:
                                if empty_seen:
                                    # Equivalent empty vehicles remain candidate insertion positions but are not rescanned.
                                    second = min(second, 2.0 * d[0][i])
                                    continue
                                empty_seen = True
                            key = (i, t, h)
                            entry = cache.get(key)
                            version = versions[t][h]
                            if entry is not None and entry[0] == version:
                                delta, pos = entry[1], entry[2]
                            else:
                                delta, pos = inf, 0
                                a = 0
                                for p in range(len(route) + 1):
                                    b = route[p] if p < len(route) else 0
                                    v = d[a][i] + d[i][b] - d[a][b]
                                    if v < delta:
                                        delta, pos = v, p
                                    a = b
                                cache[key] = (version, delta, pos)
                            v = delta + lam * (max(0.0, load + qi - Q) - max(0.0, load - Q))
                            if v < first[0]:
                                second = first[0]
                                first = (v, h, pos)
                            elif v < second:
                                second = v
                        day_options[t] = (first, second)
                best = inf
                second = inf
                best_pattern = -1
                best_plan = None
                for r, days in enumerate(self.patterns[i]):
                    value = 0.0
                    alternative = inf
                    plan = []
                    for t in days:
                        first, nxt = day_options[t]
                        if first[1] < 0:
                            value = inf
                            break
                        value += first[0]
                        alternative = min(alternative, nxt - first[0])
                        plan.append((t, first[1], first[2]))
                    if value < best:
                        second = min(best, value + alternative)
                        best, best_pattern, best_plan = value, r, plan
                    else:
                        second = min(second, value)
                if best_pattern < 0:
                    return False
                regret = second - best if second < inf else 1000.0 * self.scale
                score = (regret + .08 * best + .015 * qi * self.scale / Q) * jitter[i]
                if score > priority:
                    priority = score
                    selected = (i, best_pattern, best_plan)
            i, r, plan = selected
            sel[i] = r
            for t, h, pos in plan:
                routes[t][h].insert(pos, i)
                loads[t][h] += q[i]
                versions[t][h] += 1
            remaining.remove(i)
        return True

    def packing(self):
        """Complete capacity backtracking: infeasibility is declared only after exhausting the state space.

        Finding a feasible solution within the time limit and proving infeasibility cannot both be guaranteed on general inputs.
        This fallback prioritises truthful status and never disguises an unknown result as infeasible.
        """
        n, T, K, q, Q = self.n, self.t, self.k, self.q, self.Q
        order = sorted(range(1, n + 1), key=lambda i: (
            len(self.patterns[i]), -q[i], -min(map(len, self.patterns[i]))))
        mandatory = []
        for i in order:
            days = set(self.patterns[i][0])
            for p in self.patterns[i][1:]:
                days.intersection_update(p)
            mandatory.append(days)
        suffix = [[0.0] * T for _ in range(n + 1)]
        frequency = [0.0] * (n + 1)
        for a in range(n - 1, -1, -1):
            suffix[a] = suffix[a + 1][:]
            for t in mandatory[a]:
                suffix[a][t] += q[order[a]]
            frequency[a] = frequency[a + 1] + q[order[a]] * min(map(len, self.patterns[order[a]]))
        loads = [[0.0] * K for _ in range(T)]
        routes = self.empty()
        sel = [-1] * (n + 1)
        dead = set()

        def key_at(a):
            return (a, tuple(tuple(sorted(day)) for day in loads))

        def options(a):
            i = order[a]
            qi = q[i]
            if any(suffix[a][t] > K * Q - sum(loads[t]) + 1.e-10 * Q for t in range(T)):
                return
            if frequency[a] > K * T * Q - sum(map(sum, loads)) + 1.e-10 * Q:
                return
            pats = list(enumerate(self.patterns[i]))
            pats.sort(key=lambda p: (len(p[1]), sum(sum(loads[t]) for t in p[1])))
            for r, days in pats:
                choices = []
                for t in days:
                    seen = set()
                    hs = []
                    for h in sorted(range(K), key=lambda h: -loads[t][h]):
                        v = loads[t][h]
                        if v + qi <= Q + self.tol and v not in seen:
                            hs.append(h)
                            seen.add(v)
                    if not hs:
                        break
                    choices.append(hs)
                else:
                    for hs in itertools.product(*choices):
                        yield r, tuple(zip(days, hs))

        depth = 0
        stack = [iter(options(0))]
        chosen = []
        keys = [key_at(0)]
        while depth >= 0:
            try:
                r, visits = next(stack[depth])
            except StopIteration:
                dead.add(keys.pop())
                stack.pop()
                depth -= 1
                if depth < 0:
                    return None
                old = chosen.pop()
                i = order[depth]
                for t, h in old:
                    routes[t][h].pop()
                    loads[t][h] = sum(q[j] for j in routes[t][h])
                continue
            i = order[depth]
            for t, h in visits:
                routes[t][h].append(i)
                loads[t][h] += q[i]
            sel[i] = r
            chosen.append(visits)
            depth += 1
            if depth == n:
                return routes, sel
            key = key_at(depth)
            if key in dead:
                depth -= 1
                chosen.pop()
                for t, h in visits:
                    routes[t][h].pop()
                    loads[t][h] = sum(q[j] for j in routes[t][h])
            else:
                stack.append(iter(options(depth)))
                keys.append(key)
        return None

    def two_opt(self, route, deadline):
        d, eps = self.d, self.eps
        changed = True
        while changed and time.perf_counter() < deadline:
            changed = False
            m = len(route)
            for a in range(m - 1):
                if time.perf_counter() >= deadline:
                    return
                prev = route[a - 1] if a else 0
                i = route[a]
                for b in range(a + 1, m):
                    j = route[b]
                    nxt = route[b + 1] if b + 1 < m else 0
                    if d[prev][j] + d[i][nxt] < d[prev][i] + d[j][nxt] - eps:
                        route[a:b + 1] = reversed(route[a:b + 1])
                        changed = True
                        break
                if changed:
                    break

    def local(self, routes, hard, deadline, days=None):
        """Apply relocate, swap, two-customer relocate and two-route tail exchange using distance deltas."""
        d, q, Q, lam, eps = self.d, self.q, self.Q, self.lam, self.eps
        if days is None:
            days = list(range(self.t))
        else:
            days = list(days)
        self.rng.shuffle(days)
        loads = self.loads(routes)

        def penalty(a, b, x, y):
            if hard and (x > Q + self.tol or y > Q + self.tol):
                return float('inf')
            return lam * (max(0.0, x - Q) + max(0.0, y - Q)
                          - max(0.0, a - Q) - max(0.0, b - Q))

        for t in days:
            if time.perf_counter() >= deadline:
                break
            day = routes[t]
            for route in day:
                self.two_opt(route, deadline)
            changed = True
            while changed and time.perf_counter() < deadline:
                changed = False
                owner = [-1] * (self.n + 1)
                position = [-1] * (self.n + 1)
                prefix = []
                nodes = []
                for h, route in enumerate(day):
                    sums = [0.0]
                    for p, i in enumerate(route):
                        owner[i], position[i] = h, p
                        nodes.append(i)
                        sums.append(sums[-1] + q[i])
                    prefix.append(sums)
                    loads[t][h] = sums[-1]
                self.rng.shuffle(nodes)
                for i in nodes:
                    if time.perf_counter() >= deadline:
                        return
                    ha, pa = owner[i], position[i]
                    ra = day[ha]
                    ia = ra[pa - 1] if pa else 0
                    ib = ra[pa + 1] if pa + 1 < len(ra) else 0
                    la = loads[t][ha]
                    removal = d[ia][ib] - d[ia][i] - d[i][ib]
                    targets = []
                    for j in self.near[i]:
                        hb = owner[j]
                        if hb >= 0:
                            targets.append((hb, position[j]))
                            targets.append((hb, position[j] + 1))
                    # Depot-adjacent positions and empty routes must also enter the neighbourhood.
                    for hb, rb in enumerate(day):
                        targets.append((hb, 0))
                        if rb:
                            targets.append((hb, len(rb)))
                    seen = set()
                    for hb, pb in targets:
                        if (hb, pb) in seen:
                            continue
                        seen.add((hb, pb))
                        rb = day[hb]
                        ja = rb[pb - 1] if pb else 0
                        jb = rb[pb] if pb < len(rb) else 0
                        lb = loads[t][hb]
                        if hb == ha:
                            if pb == pa or pb == pa + 1:
                                continue
                            delta = removal + d[ja][i] + d[i][jb] - d[ja][jb]
                            if delta < -eps:
                                ra.pop(pa)
                                ra.insert(pb - int(pb > pa), i)
                                changed = True
                                break
                            continue
                        p1 = penalty(la, lb, la - q[i], lb + q[i])
                        delta = removal + d[ja][i] + d[i][jb] - d[ja][jb] + p1
                        if delta < -eps:
                            ra.pop(pa)
                            rb.insert(pb, i)
                            changed = True
                            break
                        # Relocate two consecutive customers together, keeping or reversing their order.
                        if ib:
                            ic = ra[pa + 2] if pa + 2 < len(ra) else 0
                            weight = q[i] + q[ib]
                            pp = penalty(la, lb, la - weight, lb + weight)
                            cut = d[ia][ic] - d[ia][i] - d[ib][ic] - d[ja][jb]
                            forward = d[ja][i] + d[ib][jb]
                            reverse = d[ja][ib] + d[i][jb]
                            if cut + min(forward, reverse) + pp < -eps:
                                block = ra[pa:pa + 2]
                                del ra[pa:pa + 2]
                                if reverse < forward:
                                    block.reverse()
                                rb[pb:pb] = block
                                changed = True
                                break
                        if jb:
                            j = jb
                            jc = rb[pb + 1] if pb + 1 < len(rb) else 0
                            pp = penalty(la, lb, la - q[i] + q[j], lb - q[j] + q[i])
                            delta = (d[ia][j] + d[j][ib] + d[ja][i] + d[i][jc]
                                     - d[ia][i] - d[i][ib] - d[ja][j] - d[j][jc] + pp)
                            if delta < -eps:
                                ra[pa], rb[pb] = j, i
                                changed = True
                                break
                        # Cut points are after i and before pb of rb; compare both reconnection directions.
                        a_pre = prefix[ha][pa + 1]
                        b_pre = prefix[hb][pb]
                        pp = penalty(la, lb, a_pre + lb - b_pre, b_pre + la - a_pre)
                        delta = d[i][jb] + d[ja][ib] - d[i][ib] - d[ja][jb] + pp
                        if delta < -eps:
                            day[ha] = ra[:pa + 1] + rb[pb:]
                            day[hb] = rb[:pb] + ra[pa + 1:]
                            changed = True
                            break
                        pp = penalty(la, lb, a_pre + b_pre, la - a_pre + lb - b_pre)
                        delta = d[i][ja] + d[ib][jb] - d[i][ib] - d[ja][jb] + pp
                        if delta < -eps:
                            day[ha] = ra[:pa + 1] + rb[:pb][::-1]
                            day[hb] = ra[pa + 1:][::-1] + rb[pb:]
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    for route in day:
                        self.two_opt(route, deadline)

    def destroy(self, routes, sel, count, mode):
        rng, d = self.rng, self.d
        n = self.n
        seed = rng.randrange(1, n + 1)
        if mode == 0:
            removed = set(rng.sample(range(1, n + 1), count))
        elif mode == 1:
            pd = set(self.patterns[seed][sel[seed]])
            ranking = sorted(range(1, n + 1), key=lambda i:
                             d[seed][i] / self.scale
                             + .7 * len(pd.symmetric_difference(self.patterns[i][sel[i]]))
                             + rng.random() * .8)
            removed = set(ranking[:count])
        elif mode == 2:
            savings = [0.0] * (n + 1)
            for day in routes:
                for r in day:
                    for p, i in enumerate(r):
                        a = r[p - 1] if p else 0
                        b = r[p + 1] if p + 1 < len(r) else 0
                        savings[i] += d[a][i] + d[i][b] - d[a][b]
            ranking = sorted(range(1, n + 1), key=lambda i: -savings[i] * rng.uniform(.7, 1.3))
            removed = set(ranking[:count])
        else:
            nonempty = [r for day in routes for r in day if r]
            if not nonempty:
                removed = {seed}
            else:
                r = rng.choice(nonempty)
                anchor = rng.choice(r)
                ranking = sorted(r, key=lambda i: d[anchor][i] + rng.random() * self.scale * .05)
                removed = set(ranking[:count])
                if len(removed) < count:
                    rest = [i for i in range(1, n + 1) if i not in removed]
                    rest.sort(key=lambda i: d[anchor][i] * rng.uniform(.8, 1.2))
                    removed.update(rest[:count - len(removed)])
        affected = set()
        for t, day in enumerate(routes):
            for h, r in enumerate(day):
                if any(i in removed for i in r):
                    routes[t][h] = [i for i in r if i not in removed]
                    affected.add(t)
        for i in removed:
            sel[i] = -1
        return removed, affected

    def run(self):
        if not self.n:
            self.best = self.empty()
            self.bestsel = [-1]
            self.bestcost = 0.0
            return self.output()
        Q = self.Q
        if any(v > Q for v in self.q[1:]):
            return self.infeasible()
        if sum(self.q[i] * min(map(len, self.patterns[i])) for i in range(1, self.n + 1)) > self.t * self.k * Q + self.tol:
            return self.infeasible()
        for t in range(self.t):
            forced = [self.q[i] for i in range(1, self.n + 1)
                      if all(t in p for p in self.patterns[i])]
            if sum(forced) > self.k * Q + self.tol or sum(v > Q / 2 + self.tol for v in forced) > self.k:
                return self.infeasible()
        routes, sel = self.empty(), [-1] * (self.n + 1)
        initial_end = min(self.end, self.start + self.seconds * .16)
        ok = self.repair(routes, sel, range(1, self.n + 1), True, False, initial_end)
        if not ok:
            # Backtracking only finds schedules and capacity groupings; the travel order of each group is optimised afterwards.
            packed = self.packing()
            if packed is None:
                return self.infeasible()
            routes, sel = packed
        self.save(routes, sel)
        for i in range(1, self.n + 1):
            if time.perf_counter() >= self.end:
                return self.output()
            self.near[i] = sorted((j for j in range(1, self.n + 1) if j != i),
                                  key=self.d[i].__getitem__)[:self.width]
        self.local(routes, True, min(self.end, time.perf_counter() + self.seconds * .08))
        self.save(routes, sel)
        loads = self.loads(routes)
        current_cost = self.cost(routes)
        current_excess = self.excess(loads)
        weights = [1.0] * 4
        rng = self.rng
        last_best = time.perf_counter()
        last_restart = last_best
        strength = 1.0 + math.log1p(self.seconds)
        max_remove = min(self.n, max(2, int(math.sqrt(self.n) * strength)))
        base_temp = max(self.scale * .012, self.bestcost * .025 / math.sqrt(self.n))
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            remaining_time = self.end - now
            progress = min(1.0, (now - self.start) / max(.000001, self.seconds))
            stagnant = (now - last_best) / max(.001, self.seconds)
            if now - last_restart > self.seconds / (2.0 + math.log1p(self.seconds)):
                routes = [[r[:] for r in day] for day in self.best]
                sel = self.bestsel[:]
                current_cost, current_excess = self.bestcost, 0.0
                last_restart = now
            bound = min(self.n, max(2, int(max_remove * (1.0 + stagnant))))
            count = 1 if rng.random() < .13 else rng.randint(2, max(2, bound))
            count = min(self.n, count)
            mode = rng.choices(range(4), weights=weights)[0]
            trial = [[r[:] for r in day] for day in routes]
            trial_sel = sel[:]
            removed, affected = self.destroy(trial, trial_sel, count, mode)
            hard = current_excess == 0.0 and rng.random() < .78
            # On repair failure discard the whole candidate; never use a partial-service plan as the final result.
            if not self.repair(trial, trial_sel, removed, hard, True, self.end):
                weights[mode] = .98 * weights[mode] + .02 * .3
                continue
            affected.update(t for i in removed for t in self.patterns[i][trial_sel[i]])
            local_budget = self.seconds / (55.0 + 8.0 * math.sqrt(self.n))
            local_budget *= 1.0 + count / max(1.0, math.sqrt(self.n))
            self.local(trial, hard, min(self.end, time.perf_counter() + min(remaining_time * .2, local_budget)), affected)
            trial_loads = self.loads(trial)
            trial_excess = self.excess(trial_loads)
            trial_cost = self.cost(trial)
            improved = self.save(trial, trial_sel, trial_loads)
            delta = trial_cost - current_cost + self.lam * (trial_excess - current_excess)
            temperature = base_temp * (.04 + .96 * (1.0 - progress) ** 2)
            temperature *= 1.0 + min(3.0, stagnant * 8.0)
            accept = delta <= self.eps or rng.random() < math.exp(-min(700.0, max(0.0, delta) / max(1.e-14, temperature)))
            reward = .5
            if improved:
                reward = 7.0
                last_best = time.perf_counter()
            elif delta < -self.eps:
                reward = 2.5
            elif accept:
                reward = 1.0
            weights[mode] = .95 * weights[mode] + .05 * reward
            if accept:
                routes, sel = trial, trial_sel
                current_cost, current_excess = trial_cost, trial_excess
            # The capacity penalty is used only for search; the final record is strictly limited to solutions feasible under the original objective.
            self.lam *= 1.025 if current_excess > 0 else .995
            self.lam = max(self.scale / Q, min(self.lam, self.scale / Q * (100.0 + 20.0 * self.n)))
        return self.output()

    @staticmethod
    def infeasible():
        return {'status': 'infeasible', 'x': None, 'y': None, 'u': None}

    def output(self):
        x = [[int(r == self.bestsel[i]) for r in range(len(cs))]
             for i, cs in enumerate(self.candidates, 1)]
        for t, day in enumerate(self.best):
            for h, route in enumerate(day):
                previous, load = 0, 0.0
                for i in route:
                    self.y[previous][i][h][t] = 1
                    load += self.q[i]
                    self.u[i - 1][h][t] = load
                    previous = i
                if route:
                    self.y[previous][0][h][t] = 1
        return {'status': 'feasible', 'x': x, 'y': self.y, 'u': self.u}


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    solver = _Solver(instance, time_limit_s, start)
    return solver.run()
