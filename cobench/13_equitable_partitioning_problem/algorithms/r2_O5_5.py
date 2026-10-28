"""Complementary-group integer neighbourhoods and incremental tabu equitable grouping
Merges duplicate attribute columns and, with exact integer costs, combines moves, swaps, chains, time-limited integer reassignment and destroy-rebuild, continuously keeping the best feasible solution.
"""

import math
import time
import numpy as np


class _Search:
    def __init__(self, matrix, weights, groups, limit, deadline):
        self.a = matrix
        self.w = weights
        self.n, self.m = matrix.shape
        self.k = groups
        self.limit = limit
        self.deadline = deadline
        self.rng = np.random.default_rng(562057)
        self.strength = math.log1p(limit)
        self.total = matrix.sum(axis=0, dtype=np.int64)
        remainder = self.total % groups
        self.lower = int((2 * remainder * (groups - remainder)) @ weights)
        self.degree = matrix @ weights
        self.x = np.arange(self.n, dtype=np.int64) % groups
        self.count = np.zeros((groups, self.m), dtype=np.int64)
        np.add.at(self.count, self.x, matrix)
        self.value = self.measure(self.count)
        self.square = self.quadratic(self.count)
        self.best = self.x.copy()
        self.best_value, self.best_square = self.value, self.square
        self.last_progress = time.perf_counter()
        self.phase = limit / (3.0 + math.log1p(self.n * groups))
        self.step_time = self.phase / max(1, self.n + groups)
        self.pool = max(6, int(25 * (1 + self.strength)
                              / (1 + self.m / 64.0) ** 0.25))
        # Both the cache size and the arithmetic work of one rebuild are bounded; people are rotated afterwards.
        cells = int(140000 * (1 + math.sqrt(limit)))
        work = int(24000000 * math.sqrt(1 + self.strength))
        self.width = min(self.n, max(2, min(
            cells // groups, work // max(1, groups * self.m),
            int(260000 * (1 + math.sqrt(limit)) / max(1, self.m)))))
        self.ids = np.arange(self.n, dtype=np.int64)
        self.qscale = float(4 * self.n * int(weights.sum()) + 1)
        self.solver = None
        self.solver_tried = False
        self.proven = False
        self.cache_ready = False

    def alive(self, stop=None):
        return time.perf_counter() < (self.deadline if stop is None else stop)

    def measure(self, counts):
        # This integer equals exactly the problem objective multiplied by K²; the weights of duplicate columns must not be lost.
        return int(np.abs(self.k * counts - self.total).sum(axis=0) @ self.w)

    def quadratic(self, counts):
        return int((counts * counts).sum(axis=0) @ self.w)

    def save(self):
        if (self.value, self.square) < (self.best_value, self.best_square):
            if self.value < self.best_value:
                self.last_progress = time.perf_counter()
            self.best_value, self.best_square = self.value, self.square
            self.best = self.x.copy()

    def stalled(self):
        return max(0.0, time.perf_counter() - self.last_progress) / max(1e-9, self.phase)

    def recount(self):
        self.count.fill(0)
        np.add.at(self.count, self.x, self.a)
        self.value = self.measure(self.count)
        self.square = self.quadratic(self.count)
        self.save()
        self.cache_ready = False

    def cache(self, resample=False):
        self.cache_ready = False
        if resample or len(self.ids) != self.width:
            self.ids = (np.arange(self.n, dtype=np.int64) if self.width == self.n
                        else self.rng.choice(self.n, self.width, replace=False))
        self.b = self.a[self.ids]
        self.rows = np.arange(len(self.ids))
        residual = self.k * self.count - self.total
        self.up = (np.abs(residual + self.k) - np.abs(residual)) * self.w
        self.down = (np.abs(residual - self.k) - np.abs(residual)) * self.w
        shape = (len(self.ids), self.k)
        self.plus = np.empty(shape, dtype=np.int64)
        self.minus = np.empty(shape, dtype=np.int64)
        self.products = np.empty(shape, dtype=np.int64)
        block = max(1, min(self.k, int(1800000 / max(1, len(self.ids) * self.m))))
        for g in range(0, self.k, block):
            if not self.alive():
                return False
            h = min(self.k, g + block)
            self.plus[:, g:h] = self.b @ self.up[g:h].T
            self.minus[:, g:h] = self.b @ self.down[g:h].T
            self.products[:, g:h] = self.b @ (self.count[g:h] * self.w).T
        self.cache_ready = True
        return True

    def dot_change(self, vector):
        active = np.flatnonzero(vector)
        if not active.size:
            return 0
        if active.size * 3 < self.m:
            return self.b[:, active] @ vector[active]
        return self.b @ vector

    def refresh(self, groups, old_counts):
        for pos, g in enumerate(groups):
            residual = self.k * self.count[g] - self.total
            up = (np.abs(residual + self.k) - np.abs(residual)) * self.w
            down = (np.abs(residual - self.k) - np.abs(residual)) * self.w
            self.plus[:, g] += self.dot_change(up - self.up[g])
            self.minus[:, g] += self.dot_change(down - self.down[g])
            self.products[:, g] += self.dot_change((self.count[g] - old_counts[pos]) * self.w)
            self.up[g], self.down[g] = up, down

    def apply(self, changes, primary, secondary):
        touched = sorted({g for i, target in changes for g in (int(self.x[i]), int(target))})
        old = self.count[touched].copy()
        for i, target in changes:
            source = int(self.x[i])
            self.count[source] -= self.a[i]
            self.count[target] += self.a[i]
            self.x[i] = target
        self.value += int(primary)
        self.square += int(secondary)
        self.save()
        if self.alive():
            self.refresh(touched, old)

    def scores(self):
        source = self.x[self.ids]
        primary = self.plus + self.minus[self.rows, source, None]
        secondary = (2 * (self.products - self.products[self.rows, source, None])
                     + 2 * self.degree[self.ids, None])
        primary[self.rows, source] = np.iinfo(np.int64).max // 8
        return primary, secondary

    def pick_move(self, primary, secondary, strict=True, allowed=None):
        scores = primary if allowed is None else np.where(
            allowed, primary, np.iinfo(np.int64).max // 8)
        p = int(scores.min())
        if p >= np.iinfo(np.int64).max // 16 or (strict and p > 0):
            return None
        positions = np.argwhere(scores == p)
        ties = secondary[positions[:, 0], positions[:, 1]]
        if not strict and p >= 0 and self.rng.random() < 1.0 / (2.0 + self.strength):
            chosen = int(self.rng.integers(len(positions)))
        else:
            good = np.flatnonzero(ties == ties.min())
            chosen = int(self.rng.choice(good))
        q = int(ties[chosen])
        if strict and p == 0 and q >= 0:
            return None
        i, target = positions[chosen]
        return int(i), int(target), p, q

    def candidates(self, members, target, primary, secondary):
        if members.size <= self.pool:
            return members
        third = max(1, self.pool // 3)
        directed = primary[members, target] + secondary[members, target] / self.qscale
        general = primary[members].min(axis=1)
        one = members[np.argpartition(directed, third - 1)[:third]]
        two = members[np.argpartition(general, third - 1)[:third]]
        three = self.rng.choice(members, self.pool - 2 * third, replace=False)
        return np.unique(np.concatenate((one, two, three)))

    def pair_choice(self, delta, left, right, qleft, qright, targets_left,
                    targets_right, correction):
        p = int(delta.min())
        if p > 0:
            return None
        positions = np.argwhere(delta == p)
        u, v = positions[:, 0], positions[:, 1]
        intersections = np.einsum('ij,ij,j->i', self.b[left[u]], self.b[right[v]],
                                  self.w, dtype=np.int64)
        ties = qleft[u] + qright[v] - correction * intersections
        choice = int(np.argmin(ties))
        q = int(ties[choice])
        if p == 0 and q >= 0:
            return None
        u, v = int(u[choice]), int(v[choice])
        changes = [(int(self.ids[left[u]]), int(targets_left[u])),
                   (int(self.ids[right[v]]), int(targets_right[v]))]
        return changes, p, q

    def pair_search(self, primary, secondary, stop):
        source = self.x[self.ids]
        present = np.unique(source)
        width = min(len(present), max(2, int(
            (2 + self.strength) * (1 + math.sqrt(len(self.ids) / self.k)))))
        if len(present) > width:
            present = self.rng.choice(present, width, replace=False)
        self.rng.shuffle(present)
        members = {int(g): np.flatnonzero(source == g) for g in present}
        for pos, g in enumerate(present):
            for h in present[pos + 1:]:
                if not self.alive(stop):
                    return None
                left = self.candidates(members[int(g)], h, primary, secondary)
                right = self.candidates(members[int(h)], g, primary, secondary)
                a, b = self.b[left], self.b[right]
                cg = (a * (self.up[g] + self.down[g])) @ b.T
                ch = (a * (self.up[h] + self.down[h])) @ b.T
                delta = primary[left, h, None] + primary[right, g][None, :] - cg - ch
                answer = self.pair_choice(
                    delta, left, right, secondary[left, h], secondary[right, g],
                    np.full(len(left), h), np.full(len(right), g), 4)
                if answer is not None:
                    return answer
                if self.k <= 2 or not self.alive(stop):
                    continue
                # The interaction between i:g→h and j:h→t appears only in the middle group h.
                scores = primary[right].astype(float) + secondary[right] / self.qscale
                scores[:, [g, h]] = np.inf
                targets = np.argmin(scores, axis=1)
                delta = primary[left, h, None] + primary[right, targets][None, :] - ch
                answer = self.pair_choice(
                    delta, left, right, secondary[left, h], secondary[right, targets],
                    np.full(len(left), h), targets, 2)
                if answer is not None:
                    return answer
                scores = primary[left].astype(float) + secondary[left] / self.qscale
                scores[:, [g, h]] = np.inf
                targets = np.argmin(scores, axis=1)
                delta = primary[left, targets, None] + primary[right, g][None, :] - cg
                answer = self.pair_choice(
                    delta, left, right, secondary[left, targets], secondary[right, g],
                    targets, np.full(len(right), g), 2)
                if answer is not None:
                    return answer
        return None

    def local(self, stop):
        while self.alive(stop) and self.best_value > self.lower:
            p, q = self.scores()
            move = self.pick_move(p, q)
            if move is not None:
                i, target, dp, dq = move
                self.apply([(int(self.ids[i]), target)], dp, dq)
                continue
            pair = self.pair_search(p, q, stop)
            if pair is None:
                return
            self.apply(*pair)

    def tabu(self, stop):
        expiry = np.zeros((len(self.ids), self.k), dtype=float)
        while self.alive(stop) and self.best_value > self.lower:
            started = time.perf_counter()
            primary, secondary = self.scores()
            admissible = (expiry <= started) | (primary + self.value < self.best_value)
            move = self.pick_move(primary, secondary, False, admissible)
            if move is None:
                expiry.fill(0.0)
                continue
            i, target, p, q = move
            person = int(self.ids[i])
            source = int(self.x[person])
            tenure = self.step_time * math.sqrt(len(self.ids)) * (1 + 0.18 * self.strength)
            tenure = min(max(self.step_time, tenure), max(self.step_time, (stop - started) * 0.3))
            expiry[i, source] = started + tenure * self.rng.uniform(0.7, 1.3)
            self.apply([(person, target)], p, q)
            self.step_time = 0.8 * self.step_time + 0.2 * max(1e-6, time.perf_counter() - started)

    def greedy(self, stop):
        rarity = self.a @ (self.w / np.maximum(1, self.total))
        priority = rarity + self.degree / max(1.0, 4.0 * float(self.w.sum()))
        order = np.argsort(-priority * self.rng.uniform(0.75, 1.25, self.n))
        trial = self.x.copy()
        counts = np.zeros_like(self.count)
        sizes = np.zeros(self.k, dtype=np.int64)
        for i in order:
            if not self.alive(stop):
                return False
            costs = counts @ (self.a[i] * self.w)
            choices = np.flatnonzero(costs == costs.min())
            choices = choices[sizes[choices] == sizes[choices].min()]
            g = int(self.rng.choice(choices))
            trial[i] = g
            counts[g] += self.a[i]
            sizes[g] += 1
        self.x, self.count = trial, counts
        self.value, self.square = self.measure(counts), self.quadratic(counts)
        self.save()
        self.cache_ready = False
        return True

    def perturb(self):
        self.x = self.best.copy()
        self.recount()
        stalled = self.stalled()
        amount = min(self.n, max(2, int(math.sqrt(self.n) *
                     (0.35 + 0.1 * self.strength) * (1 + math.sqrt(stalled)))))
        selected = self.rng.choice(self.n, amount, replace=False)
        np.add.at(self.count, self.x[selected], -self.a[selected])
        selected = selected[np.argsort(-self.degree[selected] *
                                      self.rng.uniform(0.6, 1.4, amount))]
        noise = self.k * math.sqrt(float(self.w.sum())) * (0.02 + 0.02 * self.strength)
        noise *= min(3.0, math.sqrt(1 + stalled))
        for i in selected:
            if not self.alive():
                return False
            row = self.a[i]
            residual = self.k * self.count - self.total
            primary = (np.abs(residual + self.k * row) - np.abs(residual)) @ self.w
            secondary = 2 * (self.count @ (row * self.w)) + self.degree[i]
            costs = primary + secondary / self.qscale + self.rng.gumbel(0, noise, self.k)
            group = int(np.argmin(costs))
            self.x[i] = group
            self.count[group] += row
        self.value, self.square = self.measure(self.count), self.quadratic(self.count)
        self.save()
        return True

    def complementary_groups(self):
        cost = np.abs(self.k * self.count - self.total) @ self.w
        probabilities = (cost.astype(float) + self.k)
        probabilities /= probabilities.sum()
        g = int(self.rng.choice(self.k, p=probabilities))
        joint = self.count + self.count[g]
        half = joint // 2
        lower = (np.abs(self.k * half - self.total) +
                 np.abs(self.k * (joint - half) - self.total)) @ self.w
        potential = np.maximum(0, cost + cost[g] - lower).astype(float)
        potential[g] = 0
        if potential.sum() > 0:
            h = int(self.rng.choice(self.k, p=potential / potential.sum()))
        else:
            h = int(self.rng.integers(self.k - 1))
            h += int(h >= g)
        groups = [g, h]
        if self.k > 2 and self.rng.random() < 0.42:
            depth = min(self.k, max(3, int(2 + self.strength + math.sqrt(self.stalled()))))
            others = np.array([i for i in range(self.k) if i not in groups], dtype=np.int64)
            groups.extend(self.rng.choice(others, depth - 2, replace=False).tolist())
        return np.asarray(groups, dtype=np.int64)

    def load_solver(self):
        if not self.solver_tried:
            self.solver_tried = True
            try:
                from scipy.optimize import milp, Bounds, LinearConstraint
                from scipy.sparse import csr_matrix, eye, kron, hstack, vstack
                self.solver = (milp, Bounds, LinearConstraint, csr_matrix, eye, kron, hstack, vstack)
            except Exception:
                self.solver = None
        return self.solver is not None

    def integer_neighborhood(self):
        remaining = self.deadline - time.perf_counter()
        if remaining < max(0.04, self.limit * 0.018) or not self.load_solver():
            return False
        if not self.alive():
            return False
        milp, Bounds, LinearConstraint, csr, eye, kron, hstack, vstack = self.solver
        groups = self.complementary_groups()
        depth = len(groups)
        selected = np.flatnonzero(np.isin(self.x, groups))
        cap = max(4, int(140 * (1 + self.strength) /
                         (depth * (1 + self.m / 64.0) ** 0.4)))
        if selected.size > cap:
            p, q = self.scores()
            local = np.flatnonzero(np.isin(self.x[self.ids], groups))
            useful = min(len(local), cap // 2)
            if useful:
                costs = (p[local][:, groups] + q[local][:, groups] / self.qscale).min(axis=1)
                ranked = self.ids[local[np.argpartition(costs, useful - 1)[:useful]]]
            else:
                ranked = np.empty(0, dtype=np.int64)
            rest = np.setdiff1d(selected, ranked, assume_unique=True)
            selected = np.concatenate((ranked, self.rng.choice(rest, cap - useful, replace=False)))
        if not selected.size or not self.alive():
            return False
        free = self.a[selected]
        original = self.x[selected]
        fixed = self.count[groups].copy()
        for pos, g in enumerate(groups):
            fixed[pos] -= free[original == g].sum(axis=0, dtype=np.int64)
        full_scope = selected.size == self.n and depth == self.k
        stalled = self.stalled()
        radius = min(len(selected), max(3, int((3 + 2 * self.strength) * math.sqrt(1 + stalled))))
        restrict = len(selected) > radius and self.rng.random() < 0.6
        complete_model = full_scope and not restrict
        if depth == 2:
            g, h = groups
            joint = self.count[g] + self.count[h]
            lo = np.minimum(self.total, self.k * joint - self.total)
            hi = np.maximum(self.total, self.k * joint - self.total)
            offset = self.k * fixed[0]
            initial = np.maximum(np.maximum(self.k * self.count[g] - hi,
                                             lo - self.k * self.count[g]), 0)
            cutoff = int(initial @ self.w) - 1
            if cutoff < 0:
                return False
            binary_count, slack_count = len(selected), self.m
            costs = self.w.astype(float)
            block = csr(self.k * free.T, dtype=float)
            ident = eye(self.m, format='csr')
            rows = [hstack((block, -ident)), hstack((-block, -ident)),
                    hstack((csr((1, binary_count)), csr(costs[None, :])))]
            upper = np.concatenate((hi - offset, offset - lo, [cutoff])).astype(float)
            lower = np.full(len(upper), -np.inf)
            if restrict:
                initial_binary = (original == g).astype(float)
                rows.append(hstack((csr((1 - 2 * initial_binary)[None, :]), csr((1, slack_count)))))
                upper = np.append(upper, radius - initial_binary.sum())
                lower = np.append(lower, -np.inf)
        else:
            binary_count, slack_count = len(selected) * depth, self.m * depth
            costs = np.repeat(self.w, depth).astype(float)
            block = kron(csr(self.k * free.T, dtype=float), eye(depth, format='csr'), format='csr')
            ident = eye(slack_count, format='csr')
            assignment_rows = kron(eye(len(selected), format='csr'),
                                   csr(np.ones((1, depth))), format='csr')
            residual = (self.k * fixed.T - self.total[:, None]).ravel()
            cutoff = int((np.abs(self.k * self.count[groups] - self.total) @ self.w).sum()) - 1
            if cutoff < 0:
                return False
            rows = [hstack((assignment_rows, csr((len(selected), slack_count)))),
                    hstack((block, -ident)), hstack((-block, -ident)),
                    hstack((csr((1, binary_count)), csr(costs[None, :])))]
            upper = np.concatenate((np.ones(len(selected)), -residual, residual, [cutoff]))
            lower = np.full(len(upper), -np.inf)
            lower[:len(selected)] = 1
            if restrict:
                same = (original[:, None] == groups[None, :]).astype(float).ravel()
                rows.append(hstack((csr(-same[None, :]), csr((1, slack_count)))))
                upper = np.append(upper, radius - len(selected))
                lower = np.append(lower, -np.inf)
        variables = binary_count + slack_count
        objective = np.concatenate((np.zeros(binary_count), costs))
        integrality = np.zeros(variables, dtype=np.uint8)
        integrality[:binary_count] = 1
        lb = np.zeros(variables)
        ub = np.full(variables, np.inf)
        ub[:binary_count] = 1
        # Break group-label symmetry only when all people and all groups are free, without restricting attainable objective values.
        if full_scope:
            if depth == 2:
                lb[0] = ub[0] = float(original[0] == groups[0])
            else:
                ub[:depth] = 0
                first = int(np.flatnonzero(groups == original[0])[0])
                lb[first] = ub[first] = 1
        matrix = vstack(rows, format='csr')
        remaining = self.deadline - time.perf_counter()
        budget = min(remaining * 0.3, self.limit *
                     (0.019 + 0.012 * math.log1p(stalled)) * math.sqrt(1 + self.strength))
        if budget < 0.015 or not self.alive():
            return False
        answer = milp(objective, integrality=integrality, bounds=Bounds(lb, ub),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={'time_limit': budget, 'mip_rel_gap': 0.0,
                               'presolve': True, 'disp': False})
        if answer.x is None:
            if answer.status == 2 and complete_model:
                self.proven = True
            return False
        if not np.all(np.isfinite(answer.x[:binary_count])):
            return False
        if depth == 2:
            proposed = np.where(answer.x[:binary_count] >= 0.5, groups[0], groups[1])
        else:
            proposed = groups[np.argmax(answer.x[:binary_count].reshape(-1, depth), axis=1)]
        candidate = self.count.copy()
        np.add.at(candidate, original, -free)
        np.add.at(candidate, proposed, free)
        p, q = self.measure(candidate), self.quadratic(candidate)
        if (p, q) < (self.value, self.square):
            self.x[selected] = proposed
            self.count, self.value, self.square = candidate, p, q
            self.save()
            self.cache_ready = False
            return True
        return False

    def run(self):
        if self.best_value == self.lower or not self.alive():
            return
        self.greedy(min(self.deadline, time.perf_counter() + self.limit * 0.16))
        if self.best_value == self.lower or not self.alive():
            return
        if not self.cache(True):
            return
        while self.alive() and self.best_value > self.lower and not self.proven:
            self.local(min(self.deadline, time.perf_counter() + self.phase * 0.5))
            if not self.alive() or self.best_value == self.lower:
                break
            if self.value != self.best_value or self.square != self.best_square:
                self.x = self.best.copy()
                self.recount()
                if not self.cache():
                    break
            improved = False
            try:
                improved = self.integer_neighborhood()
            except Exception:
                # If the optional solver or modelling fails, continue the native search without discarding the existing solution.
                self.solver = None
                self.solver_tried = True
            if not self.alive() or self.proven or self.best_value == self.lower:
                break
            if improved:
                if not self.cache():
                    break
                continue
            self.tabu(min(self.deadline, time.perf_counter() + self.phase))
            if not self.alive() or self.best_value == self.lower:
                break
            if self.rng.random() < min(0.28, 0.04 * self.stalled()):
                if not self.greedy(min(self.deadline, time.perf_counter() + self.phase * 0.5)):
                    if not self.perturb():
                        break
            elif not self.perturb():
                break
            if self.best_value == self.lower or not self.cache(True):
                break


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    result = {'group_assignment': {}}
    search = None
    active_rows = None
    n, k = 0, 1
    try:
        n = max(0, int(instance.get('num_individuals', 0)))
        k = max(1, int(instance.get('num_groups', 1)))
        result['group_assignment'] = {f'person_{i}': int(i % k) for i in range(n)}
        limit = max(0.0, float(time_limit_s))
        if not math.isfinite(limit):
            limit = 0.0
        m = max(0, int(instance.get('num_attributes', 0)))
        if n == 0 or m == 0 or k == 1 or k >= n or limit == 0:
            return result
        reserve = min(limit * 0.2, max(0.004, limit * 0.04, n * 0.0000015))
        deadline = started + limit - reserve
        if time.perf_counter() >= deadline:
            return result
        matrix = np.asarray(instance['attribute_matrix'], dtype=np.int64).reshape(n, m)
        active_rows = np.flatnonzero(matrix.any(axis=1))
        if not active_rows.size:
            return result
        if k >= len(active_rows):
            for group, person in enumerate(active_rows):
                result['group_assignment'][f'person_{int(person)}'] = int(group)
            return result
        matrix = matrix[active_rows]
        matrix = matrix[:, np.flatnonzero(matrix.any(axis=0))]
        weights = np.ones(matrix.shape[1], dtype=np.int64)
        # Objective terms of duplicate attributes are identical, so weighted merging drops no original objective term.
        if matrix.size < 1500000 * (1 + math.log1p(limit)) and time.perf_counter() < deadline:
            packed = np.packbits(matrix, axis=0).T
            _, columns, multiplicity = np.unique(packed, axis=0, return_index=True, return_counts=True)
            matrix = np.ascontiguousarray(matrix[:, columns])
            weights = multiplicity.astype(np.int64)
        if time.perf_counter() >= deadline:
            return result
        search = _Search(matrix, weights, k, limit, deadline)
        search.run()
    except Exception:
        # A complete per-person assignment is built before any optimisation work; exceptions do not change the interface or feasibility.
        pass
    if search is not None and active_rows is not None:
        try:
            for person, group in zip(active_rows, search.best.tolist()):
                result['group_assignment'][f'person_{int(person)}'] = int(group)
        except Exception:
            pass
    return result
