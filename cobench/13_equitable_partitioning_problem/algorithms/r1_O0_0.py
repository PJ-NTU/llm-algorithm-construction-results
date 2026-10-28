"""Integer-deviation-driven adaptive neighbourhood equitable grouping
Combines greedy construction, incremental moves and swaps, time-limited integer-programming reassignment and perturbation restarts, continuously storing the best feasible solution under the original objective.
"""

import time
import math
import numpy as np


class _Search:
    def __init__(self, matrix, groups, limit, deadline):
        self.a_matrix = matrix
        self.n, self.m = matrix.shape
        self.k = groups
        self.limit = limit
        self.deadline = deadline
        self.rng = np.random.default_rng(562019)
        self.rows = np.arange(self.n)
        self.weight = matrix.sum(axis=1, dtype=np.int64)
        self.total = matrix.sum(axis=0, dtype=np.int64)
        remainder = self.total % groups
        self.lower = int((2 * remainder * (groups - remainder)).sum())
        self.assignment = self.rows % groups
        self.count = np.zeros((groups, self.m), dtype=np.int64)
        np.add.at(self.count, self.assignment, matrix)
        self.value = self.measure(self.count)
        self.square = int((self.count * self.count).sum())
        self.best_value = self.value
        self.best_square = self.square
        self.best = self.assignment.copy()
        self.stagnation = 0
        self.strength = math.log1p(limit)
        self.pool = max(8, int(30 * (1 + self.strength)
                              / max(1.0, self.m / 64.0) ** 0.25))
        self.scipy_tools = None
        self.scipy_attempted = False

    def alive(self):
        return time.perf_counter() < self.deadline

    def measure(self, count):
        # This integer value equals exactly the problem objective multiplied by K².
        return int(np.abs(self.k * count - self.total).sum())

    def save(self):
        if (self.value < self.best_value or
                (self.value == self.best_value and self.square < self.best_square)):
            if self.value < self.best_value:
                self.stagnation = 0
            self.best_value = self.value
            self.best_square = self.square
            self.best = self.assignment.copy()

    def rebuild(self):
        self.count.fill(0)
        np.add.at(self.count, self.assignment, self.a_matrix)
        self.value = self.measure(self.count)
        self.square = int((self.count * self.count).sum())
        self.save()
        residual = self.k * self.count - self.total
        self.plus_weight = np.abs(residual + self.k) - np.abs(residual)
        self.minus_weight = np.abs(residual - self.k) - np.abs(residual)
        self.plus = self.a_matrix @ self.plus_weight.T
        self.minus = self.a_matrix @ self.minus_weight.T
        self.products = self.a_matrix @ self.count.T

    def refresh(self, groups):
        groups = np.asarray(groups, dtype=np.int64)
        residual = self.k * self.count[groups] - self.total
        self.plus_weight[groups] = np.abs(residual + self.k) - np.abs(residual)
        self.minus_weight[groups] = np.abs(residual - self.k) - np.abs(residual)
        self.plus[:, groups] = self.a_matrix @ self.plus_weight[groups].T
        self.minus[:, groups] = self.a_matrix @ self.minus_weight[groups].T
        self.products[:, groups] = self.a_matrix @ self.count[groups].T

    def greedy(self):
        order = np.argsort(-self.weight * self.rng.uniform(0.7, 1.3, self.n))
        count = np.zeros_like(self.count)
        sizes = np.zeros(self.k, dtype=np.int64)
        assignment = self.assignment.copy()
        for i in order:
            if not self.alive():
                return False
            score = count @ self.a_matrix[i]
            choices = np.flatnonzero(score == score.min())
            smallest = sizes[choices].min()
            choices = choices[sizes[choices] == smallest]
            group = int(self.rng.choice(choices))
            assignment[i] = group
            count[group] += self.a_matrix[i]
            sizes[group] += 1
        self.assignment = assignment
        self.rebuild()
        return True

    def move_scores(self):
        primary = self.plus + self.minus[self.rows, self.assignment, None]
        secondary = (2 * (self.products - self.products[self.rows, self.assignment, None])
                     + 2 * self.weight[:, None])
        primary[self.rows, self.assignment] = np.iinfo(np.int64).max // 4
        return primary, secondary

    @staticmethod
    def better(p, q, best_p=0, best_q=0):
        return p < best_p or (p == best_p and q < best_q)

    def transfer(self, i, group, p, q):
        previous = int(self.assignment[i])
        self.count[previous] -= self.a_matrix[i]
        self.count[group] += self.a_matrix[i]
        self.assignment[i] = group
        self.value += p
        self.square += q
        self.save()
        if self.alive():
            self.refresh([previous, group])

    def exchange(self, i, j, p, q):
        g, h = int(self.assignment[i]), int(self.assignment[j])
        difference = self.a_matrix[j] - self.a_matrix[i]
        self.count[g] += difference
        self.count[h] -= difference
        self.assignment[i], self.assignment[j] = h, g
        self.value += p
        self.square += q
        self.save()
        if self.alive():
            self.refresh([g, h])

    def candidates(self, members, target, primary, secondary):
        if members.size <= self.pool:
            return members
        ranked_count = max(1, self.pool * 2 // 3)
        scores = primary[members, target].astype(float)
        scores += secondary[members, target] / (4.0 * self.n * self.m + 1.0)
        good = members[np.argpartition(scores, ranked_count - 1)[:ranked_count]]
        random = self.rng.choice(members, self.pool - ranked_count, replace=False)
        return np.unique(np.concatenate((good, random)))

    def swap_search(self, primary, secondary):
        members = [np.flatnonzero(self.assignment == g) for g in range(self.k)]
        pairs = [(g, h) for g in range(self.k) for h in range(g + 1, self.k)
                 if members[g].size and members[h].size]
        self.rng.shuffle(pairs)
        for g, h in pairs:
            if not self.alive():
                return False
            left = self.candidates(members[g], h, primary, secondary)
            right = self.candidates(members[h], g, primary, secondary)
            a = self.a_matrix[left]
            b = self.a_matrix[right]
            curvature = (self.plus_weight[g] + self.minus_weight[g]
                         + self.plus_weight[h] + self.minus_weight[h])
            correction = (a * curvature) @ b.T
            delta = primary[left, h, None] + primary[right, g][None, :] - correction
            p = int(delta.min())
            if p > 0:
                continue
            positions = np.argwhere(delta == p)
            # Compute squared deviation only for the best original-objective candidates, used as a strict tie-breaker among equal values.
            intersections = np.einsum('ij,ij->i', a[positions[:, 0]],
                                      b[positions[:, 1]], dtype=np.int64)
            ties = (secondary[left[positions[:, 0]], h]
                    + secondary[right[positions[:, 1]], g] - 4 * intersections)
            chosen = int(np.argmin(ties))
            q = int(ties[chosen])
            if self.better(p, q):
                u, v = positions[chosen]
                self.exchange(int(left[u]), int(right[v]), p, q)
                return True
        return False

    def local_search(self):
        while self.alive() and self.best_value > self.lower:
            primary, secondary = self.move_scores()
            p = int(primary.min())
            if p <= 0:
                positions = np.argwhere(primary == p)
                ties = secondary[positions[:, 0], positions[:, 1]]
                chosen = int(np.argmin(ties))
                q = int(ties[chosen])
                if self.better(p, q):
                    i, group = positions[chosen]
                    self.transfer(int(i), int(group), p, q)
                    continue
            if not self.swap_search(primary, secondary):
                break

    def load_milp(self):
        if not self.scipy_attempted:
            self.scipy_attempted = True
            try:
                from scipy.optimize import milp, Bounds, LinearConstraint
                from scipy.sparse import csr_matrix, eye, kron, hstack, vstack
                self.scipy_tools = (milp, Bounds, LinearConstraint, csr_matrix,
                                    eye, kron, hstack, vstack)
            except Exception:
                self.scipy_tools = None
        return self.scipy_tools is not None

    def choose_neighborhood(self):
        depth = min(self.k, 2 + int(math.log1p(self.stagnation)
                                    * self.strength / 3.5))
        # Each time keep a chance of choosing a smaller neighbourhood, so deep neighbourhoods do not monopolise the remaining budget.
        if self.rng.random() < 0.35:
            depth = min(2, self.k)
        errors = np.abs(self.k * self.count - self.total).sum(axis=1).astype(float)
        probabilities = (errors + self.k) / (errors.sum() + self.k * self.k)
        groups = self.rng.choice(self.k, depth, replace=False, p=probabilities)
        selected = np.flatnonzero(np.isin(self.assignment, groups))
        cap = max(12, int(85 * (1 + self.strength) / math.sqrt(depth)
                          / max(1.0, self.m / 64.0) ** 0.25))
        if selected.size > cap:
            removals = self.minus[selected, self.assignment[selected]]
            useful = cap // 2
            ranked = selected[np.argpartition(removals, useful - 1)[:useful]]
            rest = np.setdiff1d(selected, ranked, assume_unique=True)
            sampled = self.rng.choice(rest, cap - useful, replace=False)
            selected = np.concatenate((ranked, sampled))
        return groups, selected

    def integer_neighborhood(self):
        remaining = self.deadline - time.perf_counter()
        if remaining < max(0.08, self.limit * 0.018):
            return False
        if not self.load_milp() or not self.alive():
            return False
        milp, Bounds, LinearConstraint, csr, eye, kron, hstack, vstack = self.scipy_tools
        groups, selected = self.choose_neighborhood()
        if selected.size == 0:
            return False
        free = self.a_matrix[selected]
        fixed = self.count.copy()
        np.add.at(fixed, self.assignment[selected], -free)
        group_count = len(groups)
        current = self.count[groups]
        if group_count == 2:
            # The absolute deviation of two groups can be rewritten exactly as the distance from a count to an interval.
            g, h = groups
            joint = self.count[g] + self.count[h]
            lo = np.minimum(self.total, self.k * joint - self.total)
            hi = np.maximum(self.total, self.k * joint - self.total)
            offset = self.k * fixed[g]
            initial = np.maximum(np.maximum(self.k * self.count[g] - hi,
                                             lo - self.k * self.count[g]), 0)
            upper_cost = int(initial.sum()) - 1
            if upper_cost < 0:
                return False
            binary_count = selected.size
            attributes = self.m
            block = csr(self.k * free.T, dtype=float)
            identity = eye(attributes, format='csr')
            matrix = vstack((hstack((block, -identity)),
                             hstack((-block, -identity)),
                             hstack((csr((1, binary_count)),
                                     csr(np.ones((1, attributes)))))), format='csr')
            upper = np.concatenate((hi - offset, offset - lo, [upper_cost])).astype(float)
            lower = np.full(upper.size, -np.inf)
        else:
            binary_count = selected.size * group_count
            attributes = self.m * group_count
            block = kron(csr(self.k * free.T, dtype=float),
                         eye(group_count, format='csr'), format='csr')
            identity = eye(attributes, format='csr')
            assignment_rows = kron(eye(selected.size, format='csr'),
                                   csr(np.ones((1, group_count))), format='csr')
            residual = (self.k * fixed[groups].T - self.total[:, None]).ravel()
            upper_cost = int(np.abs(self.k * current - self.total).sum()) - 1
            if upper_cost < 0:
                return False
            matrix = vstack((hstack((assignment_rows, csr((selected.size, attributes)))),
                             hstack((block, -identity)),
                             hstack((-block, -identity)),
                             hstack((csr((1, binary_count)),
                                     csr(np.ones((1, attributes)))))), format='csr')
            upper = np.concatenate((np.ones(selected.size), -residual,
                                    residual, [upper_cost])).astype(float)
            lower = np.full(upper.size, -np.inf)
            lower[:selected.size] = 1.0
        variables = binary_count + attributes
        objective = np.zeros(variables)
        objective[binary_count:] = 1.0
        integrality = np.zeros(variables, dtype=np.uint8)
        integrality[:binary_count] = 1
        variable_upper = np.full(variables, np.inf)
        variable_upper[:binary_count] = 1.0
        remaining = self.deadline - time.perf_counter()
        portion = self.limit * (0.025 + 0.015 * math.log1p(self.stagnation))
        budget = min(remaining * 0.32, portion)
        if budget < 0.015:
            return False
        answer = milp(objective, integrality=integrality,
                      bounds=Bounds(np.zeros(variables), variable_upper),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={'time_limit': budget, 'mip_rel_gap': 0.0,
                               'presolve': True, 'disp': False})
        if answer.x is None or not np.all(np.isfinite(answer.x[:binary_count])):
            return False
        if group_count == 2:
            proposed = np.where(answer.x[:binary_count] >= 0.5, groups[0], groups[1])
        else:
            proposed = groups[np.argmax(answer.x[:binary_count].reshape(-1, group_count), axis=1)]
        candidate_count = fixed
        np.add.at(candidate_count, proposed, free)
        p = self.measure(candidate_count)
        q = int((candidate_count * candidate_count).sum())
        if self.better(p, q, self.value, self.square):
            self.assignment[selected] = proposed
            self.count = candidate_count
            self.value, self.square = p, q
            self.save()
            if self.alive():
                self.rebuild()
            return True
        return False

    def perturb(self):
        self.assignment = self.best.copy()
        self.count.fill(0)
        np.add.at(self.count, self.assignment, self.a_matrix)
        fraction = min(0.38, (0.008 + 0.004 * self.strength)
                       * (1.0 + math.sqrt(self.stagnation + 1)))
        amount = min(self.n, max(2, int(self.n * fraction)))
        selected = self.rng.choice(self.n, amount, replace=False)
        np.add.at(self.count, self.assignment[selected], -self.a_matrix[selected])
        order = selected[np.argsort(-self.weight[selected]
                                    * self.rng.uniform(0.5, 1.5, amount))]
        noise = self.k * min(1.5, 0.04 * (1 + self.strength)
                            * math.sqrt(self.stagnation + 1))
        for i in order:
            if not self.alive():
                return
            row = self.a_matrix[i]
            residual = self.k * self.count - self.total
            primary = (np.abs(residual + self.k * row) - np.abs(residual)).sum(axis=1)
            secondary = 2 * (self.count @ row) + self.weight[i]
            score = primary.astype(float) + secondary / (4.0 * self.n * self.m + 1)
            score += self.rng.gumbel(0.0, noise, self.k)
            group = int(np.argmin(score))
            self.assignment[i] = group
            self.count[group] += row
        # A single random move lets even fully deterministic repair leave the original local solution.
        i = int(self.rng.integers(self.n))
        self.assignment[i] = int(self.rng.integers(self.k))
        self.rebuild()

    def run(self):
        if self.best_value == self.lower or not self.alive():
            return
        if not self.greedy():
            return
        while self.alive() and self.best_value > self.lower:
            before = self.best_value
            self.local_search()
            if not self.alive() or self.best_value == self.lower:
                break
            improved = False
            try:
                improved = self.integer_neighborhood()
            except Exception:
                # Exceptions from the optional solver do not affect the existing feasible solution or the subsequent native neighbourhood search.
                self.scipy_tools = None
                self.scipy_attempted = True
            if improved:
                continue
            if self.best_value >= before:
                self.stagnation += 1
            if not self.alive():
                break
            if self.rng.random() < min(0.18, 0.012 * self.stagnation):
                if not self.greedy():
                    break
            else:
                self.perturb()


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    result = {'group_assignment': {}}
    search = None
    active_rows = None
    try:
        n = max(0, int(instance['num_individuals']))
        k = max(1, int(instance['num_groups']))
        assignment = {f'person_{i}': int(i % k) for i in range(n)}
        result['group_assignment'] = assignment
        limit = max(0.0, float(time_limit_s))
        if n == 0 or k == 1 or k >= n or limit == 0:
            return result
        # Reserve time for returning the encoding and for the native sub-solver to wrap up.
        reserve = min(limit * 0.2, max(0.003, limit * 0.035, n * 0.0000015))
        deadline = start + limit - reserve
        if time.perf_counter() >= deadline:
            return result
        matrix = np.asarray(instance['attribute_matrix'], dtype=np.int64)
        m = int(instance['num_attributes'])
        if matrix.size == 0 or m == 0:
            return result
        matrix = matrix.reshape(n, m)
        columns = np.flatnonzero(matrix.sum(axis=0))
        if not columns.size:
            return result
        matrix = matrix[:, columns]
        active_rows = np.flatnonzero(matrix.any(axis=1))
        if not active_rows.size:
            return result
        if k >= active_rows.size:
            for group, person in enumerate(active_rows):
                assignment[f'person_{int(person)}'] = int(group)
            return result
        if time.perf_counter() >= deadline:
            return result
        search = _Search(matrix[active_rows], k, limit, deadline)
        search.run()
    except Exception:
        pass
    if search is not None and active_rows is not None:
        try:
            values = search.best.tolist()
            for person, group in zip(active_rows, values):
                result['group_assignment'][f'person_{int(person)}'] = int(group)
        except Exception:
            pass
    return result
