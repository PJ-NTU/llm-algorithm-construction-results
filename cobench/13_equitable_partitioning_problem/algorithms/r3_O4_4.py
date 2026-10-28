"""Integer convex hull and cross-group coupled equitable grouping
Merges duplicate attributes with weights and keeps improving feasible solutions via incremental tabu search, count integer convex hulls, complementary-group reassignment and all-group sparse coupled moves.
"""

import math
import time
try:
    import numpy as np
except Exception:
    np = None


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
        self.floor = self.total // self.k
        self.rem = self.total % self.k

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
        # First merge groups with complementary counts, then expand by the integer lower bound that can decrease after merging.
        cost = np.abs(self.k * self.count - self.total) @ self.w
        excess = (np.maximum(self.floor - self.count, 0) * (self.k - self.rem)
                  + np.maximum(self.count - self.floor - 1, 0) * self.rem) @ self.w
        probability = excess.astype(float) + max(1, self.k)
        probability /= probability.sum()
        g = int(self.rng.choice(self.k, p=probability))
        groups = [g]
        depth = 2
        max_depth = min(self.k, max(2, int(2 + math.sqrt(
            self.strength * (1 + self.stalled())))))
        if max_depth > 2 and self.rng.random() < 0.4:
            depth = int(self.rng.integers(3, max_depth + 1))
        affordable = 85 * (1 + self.strength) / (1 + self.m / 64.0) ** 0.4
        if self.n * self.k <= affordable and self.rng.random() < 0.5:
            depth = self.k
        joint = self.count[g].copy()
        current_cost = int(cost[g])
        while len(groups) < depth and self.alive():
            merged = joint + self.count
            divisor = len(groups) + 1
            low = merged // divisor
            remainder = merged % divisor
            bound = ((divisor - remainder) * np.abs(self.k * low - self.total)
                     + remainder * np.abs(self.k * (low + 1) - self.total)) @ self.w
            gain = np.maximum(0, current_cost + cost - bound).astype(float)
            gain += max(1.0, float(gain.max()) * 0.025)
            gain[groups] = 0
            h = int(self.rng.choice(self.k, p=gain / gain.sum()))
            groups.append(h)
            joint += self.count[h]
            current_cost += int(cost[h])
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

    def accept_integer(self, selected, proposed):
        original = self.x[selected]
        free = self.a[selected]
        candidate = self.count.copy()
        np.add.at(candidate, original, -free)
        np.add.at(candidate, proposed, free)
        p, q = self.measure(candidate), self.quadratic(candidate)
        if (p, q) >= (self.value, self.square):
            return False
        self.x[selected] = proposed
        self.count, self.value, self.square = candidate, p, q
        self.save()
        self.cache_ready = False
        return True

    def optimize_integer(self, rows, lower, upper, binary_count, costs,
                         anchor=None, complete=False):
        milp, Bounds, LinearConstraint, csr, eye, kron, hstack, vstack = self.solver
        matrix = vstack(rows, format='csr')
        remaining = self.deadline - time.perf_counter()
        # Time slices grow with the total budget and stagnation time, always leaving slack for decoding and native search.
        budget = min(remaining * 0.28, self.limit *
                     (0.023 + 0.014 * math.log1p(self.stalled())) *
                     math.sqrt(1 + self.strength))
        if budget < 0.012 or not self.alive():
            return None
        variables = binary_count + len(costs)
        objective = np.concatenate((np.zeros(binary_count), costs))
        integrality = np.zeros(variables, dtype=np.uint8)
        integrality[:binary_count] = 1
        lb = np.zeros(variables)
        ub = np.full(variables, np.inf)
        ub[:binary_count] = 1
        if anchor is not None:
            indices, values = anchor
            lb[indices] = values
            ub[indices] = values
        answer = milp(objective, integrality=integrality, bounds=Bounds(lb, ub),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={'time_limit': budget, 'mip_rel_gap': 0.0,
                               'presolve': True, 'disp': False})
        if answer.x is None:
            if answer.status == 2 and complete:
                self.proven = True
            return None
        values = answer.x[:binary_count]
        if not np.all(np.isfinite(values)) or np.max(np.abs(values - np.rint(values))) > 1e-5:
            return None
        return np.rint(values).astype(np.int64)

    def count_envelope(self, block, fixed, groups):
        # T=Kq+r. The original cost at integer c equals:
        # r+(K-2r)(c-q)+2*max((K-r)(q-c), r(c-q-1), 0).
        # Within the neighbourhood the total of each attribute is unchanged and the linear term is constant; only the last term is minimised.
        # Compared with directly linearising the absolute value, this formulation gives the convex-hull lower bound for integer counts.
        milp, Bounds, LinearConstraint, csr, eye, kron, hstack, vstack = self.solver
        depth = len(groups)
        q = np.repeat(self.floor, depth)
        r = np.repeat(self.rem, depth)
        fixed = fixed.T.ravel()
        ident = eye(len(q), format='csr')
        rows = [hstack((block.multiply(-(self.k - r)[:, None]), -ident)),
                hstack((block.multiply(r[:, None]), -ident))]
        upper = np.concatenate(((self.k - r) * (fixed - q), r * (q + 1 - fixed)))
        lower = np.full(len(upper), -np.inf)
        counts = self.count[groups].T.ravel()
        costs = np.repeat(self.w, depth).astype(float)
        initial = np.maximum(q - counts, 0) * (self.k - r)
        initial += np.maximum(counts - q - 1, 0) * r
        cutoff = int(initial @ np.repeat(self.w, depth)) - 1
        return rows, lower, upper, costs, cutoff

    def integer_neighborhood(self, directed=False):
        remaining = self.deadline - time.perf_counter()
        if remaining < max(0.04, self.limit * 0.022) or not self.load_solver():
            return False
        if not self.alive():
            return False
        if directed:
            return self.directed_neighborhood()
        milp, Bounds, LinearConstraint, csr, eye, kron, hstack, vstack = self.solver
        groups = self.complementary_groups()
        depth = len(groups)
        if depth < 2:
            return False
        selected = np.flatnonzero(np.isin(self.x, groups))
        all_members = len(selected)
        cap = max(4, int(175 * (1 + self.strength) /
                         (depth * (1 + self.m / 64.0) ** 0.4)))
        if selected.size > cap:
            p, q = self.scores()
            local = np.flatnonzero(np.isin(self.x[self.ids], groups))
            useful = min(len(local), cap // 2)
            if useful:
                scores = (p[local][:, groups] + q[local][:, groups] / self.qscale).min(axis=1)
                ranked = self.ids[local[np.argpartition(scores, useful - 1)[:useful]]]
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
        radius = min(len(selected), max(3, int((4 + 3 * self.strength) *
                     math.sqrt(1 + self.stalled()))))
        restrict = len(selected) > radius and self.rng.random() < 0.55
        complete = full_scope and not restrict
        anchor = None
        if depth == 2:
            # Take the integer convex hull after combining the two groups; keep only one auxiliary epigraph variable per attribute.
            g, h = groups
            joint = self.count[g] + self.count[h]
            middle = joint // 2
            minimum = (np.abs(self.k * middle - self.total)
                       + np.abs(self.k * (joint - middle) - self.total))
            initial = (np.abs(self.k * self.count[g] - self.total)
                       + np.abs(self.k * self.count[h] - self.total))
            cutoff = int(((initial - minimum) // 2) @ self.w) - 1
            if cutoff < 0:
                return False
            binary_count, slack_count = len(selected), self.m
            costs = self.w.astype(float)
            base = csr(free.T, dtype=float)
            ident = eye(self.m, format='csr')
            q = self.floor
            knots = (np.minimum(q - 1, joint - q - 2), q, q + 1,
                     joint - q - 1, joint - q)
            rows, rhs = [], []
            for point in knots:
                f = np.abs(self.k * point - self.total) + np.abs(self.k * (joint - point) - self.total)
                f_next = (np.abs(self.k * (point + 1) - self.total)
                          + np.abs(self.k * (joint - point - 1) - self.total))
                slope = (f_next - f) // 2
                intercept = (f - minimum) // 2 - slope * point
                rows.append(hstack((base.multiply(slope[:, None]), -ident)))
                rhs.append(-intercept - slope * fixed[0])
            upper = np.concatenate(rhs).astype(float)
            lower = np.full(len(upper), -np.inf)
            if restrict:
                initial_binary = (original == g).astype(float)
                rows.append(hstack((csr((1 - 2 * initial_binary)[None, :]), csr((1, slack_count)))))
                upper = np.append(upper, radius - initial_binary.sum())
                lower = np.append(lower, -np.inf)
            if len(selected) == all_members and not restrict:
                anchor = (np.array([0]), np.array([float(original[0] == g)]))
        else:
            binary_count, slack_count = len(selected) * depth, self.m * depth
            block = kron(csr(free.T, dtype=float), eye(depth, format='csr'), format='csr')
            rows, lower, upper, costs, cutoff = self.count_envelope(block, fixed, groups)
            if cutoff < 0:
                return False
            assignment_rows = kron(eye(len(selected), format='csr'),
                                   csr(np.ones((1, depth))), format='csr')
            rows.append(hstack((assignment_rows, csr((len(selected), slack_count)))))
            upper = np.concatenate((upper, np.ones(len(selected))))
            lower = np.concatenate((lower, np.ones(len(selected))))
            if restrict:
                same = (original[:, None] == groups[None, :]).astype(float).ravel()
                rows.append(hstack((csr(-same[None, :]), csr((1, slack_count)))))
                upper = np.append(upper, radius - len(selected))
                lower = np.append(lower, -np.inf)
            if len(selected) == all_members and not restrict:
                anchor = (np.arange(depth), (groups == original[0]).astype(float))
        rows.append(hstack((csr((1, binary_count)), csr(costs[None, :]))))
        upper = np.append(upper, cutoff)
        lower = np.append(lower, -np.inf)
        answer = self.optimize_integer(rows, lower, upper, binary_count, costs, anchor, complete)
        if answer is None:
            return False
        if depth == 2:
            proposed = np.where(answer >= 1, groups[0], groups[1])
        else:
            onehot = answer.reshape(-1, depth)
            if not np.all(onehot.sum(axis=1) == 1):
                return False
            proposed = groups[np.argmax(onehot, axis=1)]
        return self.accept_integer(selected, proposed)

    def directed_neighborhood(self):
        # Each person frees only the current group and one candidate target group; all other x[i][j] are fixed to zero.
        # One binary variable per person suffices to repair multiple groups, long cycles and multiple move chains simultaneously.
        milp, Bounds, LinearConstraint, csr, eye, kron, hstack, vstack = self.solver
        p, q = self.scores()
        noise = self.k * math.sqrt(float(self.w.sum())) * (0.025 + 0.018 * self.strength)
        noise *= math.sqrt(1 + min(5.0, self.stalled()))
        ranking = p.astype(float) + q / self.qscale
        ranking += self.rng.gumbel(0.0, noise, ranking.shape)
        targets = np.argmin(ranking, axis=1)
        costs = ranking[self.rows, targets]
        cap = min(len(self.ids), max(4, int(min(
            230 * (1 + self.strength) / (1 + self.m / 64.0) ** 0.4,
            130000 * (1 + self.strength) / max(1, self.m)))))
        if len(self.ids) > cap:
            useful = max(1, cap // 2)
            chosen = np.argpartition(costs, useful - 1)[:useful]
            rest = np.setdiff1d(self.rows, chosen, assume_unique=True)
            chosen = np.concatenate((chosen, self.rng.choice(rest, cap - useful, replace=False)))
        else:
            chosen = self.rows
        selected = self.ids[chosen]
        targets = targets[chosen]
        original = self.x[selected]
        groups = np.unique(np.concatenate((original, targets)))
        depth = len(groups)
        if depth < 2 or not self.alive():
            return False
        free = self.a[selected]
        people, attributes = np.nonzero(free)
        src = np.searchsorted(groups, original)
        dst = np.searchsorted(groups, targets)
        row_indices = np.concatenate((attributes * depth + src[people], attributes * depth + dst[people]))
        columns = np.concatenate((people, people))
        coefficients = np.concatenate((-np.ones(len(people)), np.ones(len(people))))
        binary_count = len(selected)
        block = csr((coefficients, (row_indices, columns)), shape=(self.m * depth, binary_count))
        rows, lower, upper, objective, cutoff = self.count_envelope(block, self.count[groups], groups)
        if cutoff < 0:
            return False
        slack_count = len(objective)
        rows.append(hstack((csr((1, binary_count)), csr(objective[None, :]))))
        upper = np.append(upper, cutoff)
        lower = np.append(lower, -np.inf)
        radius = min(binary_count, max(3, int((5 + 4 * self.strength) *
                     math.sqrt(1 + self.stalled()))))
        if radius < binary_count and self.rng.random() < 0.7:
            rows.append(hstack((csr(np.ones((1, binary_count))), csr((1, slack_count)))))
            upper = np.append(upper, radius)
            lower = np.append(lower, -np.inf)
        answer = self.optimize_integer(rows, lower, upper, binary_count, objective)
        if answer is None:
            return False
        return self.accept_integer(selected, np.where(answer >= 1, targets, original))

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
                directed_first = self.k > 2 and self.rng.random() < 0.45
                improved = self.integer_neighborhood(directed_first)
                if not improved and not self.proven and self.k > 2 and self.alive():
                    improved = self.integer_neighborhood(not directed_first)
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
            self.tabu(min(self.deadline, time.perf_counter() + self.phase / (1 + 0.22 * self.strength)))
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
        if n == 0 or m == 0 or k == 1 or k >= n or limit == 0 or np is None:
            return result
        reserve = min(limit * 0.2, max(0.004, limit * 0.055, n * 0.0000015))
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
