import itertools
import math
import random
import time


class _SearchTimeout(Exception):
    pass


class _Simplex:
    """Maximize c*x subject to A*x <= b and x >= 0."""

    EPS = 1e-10

    def __init__(self, A, b, c, deadline):
        self.m = len(b)
        self.n = len(c)
        self.deadline = deadline
        self.B = [self.n + i for i in range(self.m)]
        self.N = list(range(self.n)) + [-1]
        self.D = [[0.0] * (self.n + 2) for _ in range(self.m + 2)]

        for i in range(self.m):
            row = self.D[i]
            source = A[i]
            for j in range(self.n):
                row[j] = float(source[j])
            row[self.n] = -1.0
            row[self.n + 1] = float(b[i])

        for j in range(self.n):
            self.D[self.m][j] = -float(c[j])
        self.D[self.m + 1][self.n] = 1.0

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise _SearchTimeout()

    def _pivot(self, r, s):
        self._check_time()
        D = self.D
        inv = 1.0 / D[r][s]
        rows = self.m + 2
        cols = self.n + 2

        for i in range(rows):
            if i == r:
                continue
            factor = D[i][s] * inv
            if factor != 0.0:
                ri = D[i]
                rr = D[r]
                for j in range(cols):
                    if j != s:
                        ri[j] -= rr[j] * factor
            D[i][s] *= -inv
            if (i & 31) == 0:
                self._check_time()

        rr = D[r]
        for j in range(cols):
            if j != s:
                rr[j] *= inv
        rr[s] = inv
        self.B[r], self.N[s] = self.N[s], self.B[r]

    def _simplex(self, phase):
        objective_row = self.m + 1 if phase == 1 else self.m
        while True:
            self._check_time()
            s = -1
            for j in range(self.n + 1):
                if phase == 2 and self.N[j] == -1:
                    continue
                if s == -1:
                    s = j
                else:
                    a = self.D[objective_row][j]
                    q = self.D[objective_row][s]
                    if a < q - self.EPS or (
                        abs(a - q) <= self.EPS and self.N[j] < self.N[s]
                    ):
                        s = j

            if s == -1 or self.D[objective_row][s] >= -self.EPS:
                return True

            r = -1
            for i in range(self.m):
                coefficient = self.D[i][s]
                if coefficient <= self.EPS:
                    continue
                if r == -1:
                    r = i
                    continue
                lhs = self.D[i][self.n + 1] / coefficient
                rhs = self.D[r][self.n + 1] / self.D[r][s]
                if lhs < rhs - self.EPS or (
                    abs(lhs - rhs) <= self.EPS and self.B[i] < self.B[r]
                ):
                    r = i

            if r == -1:
                return False
            self._pivot(r, s)

    def solve(self):
        self._check_time()
        r = 0
        for i in range(1, self.m):
            if self.D[i][self.n + 1] < self.D[r][self.n + 1]:
                r = i

        if self.m and self.D[r][self.n + 1] < -self.EPS:
            self._pivot(r, self.n)
            if not self._simplex(1):
                return None
            phase_value = self.D[self.m + 1][self.n + 1]
            if phase_value < -self.EPS or abs(phase_value) > 1e-7:
                return None

            for i in range(self.m):
                if self.B[i] != -1:
                    continue
                s = 0
                for j in range(1, self.n + 1):
                    a = self.D[i][j]
                    q = self.D[i][s]
                    if a < q - self.EPS or (
                        abs(a - q) <= self.EPS and self.N[j] < self.N[s]
                    ):
                        s = j
                if abs(self.D[i][s]) > self.EPS:
                    self._pivot(i, s)
                break

        if not self._simplex(2):
            return None

        result = [0.0] * self.n
        for i in range(self.m):
            if 0 <= self.B[i] < self.n:
                result[self.B[i]] = self.D[i][self.n + 1]
        return result


def _structural_fallback(instance):
    try:
        n = max(1, int(instance.get("num_securities", 1)))
        k = min(n, max(1, int(instance.get("max_securities", 1))))

        raw_min = instance.get("minimum_weights", [])
        raw_max = instance.get("maximum_weights", [])
        minimum = (
            [max(0.0, float(v)) for v in raw_min]
            if isinstance(raw_min, list) and len(raw_min) == n
            else [0.0] * n
        )
        maximum = (
            [max(0.0, float(v)) for v in raw_max]
            if isinstance(raw_max, list) and len(raw_max) == n
            else [1.0] * n
        )

        order = sorted(range(n), key=lambda i: maximum[i], reverse=True)
        support = order[:k]
        weights = [0.0] * n
        base = sum(minimum[i] for i in support)

        if base <= 1.0 and sum(maximum[i] for i in support) >= 1.0:
            for i in support:
                weights[i] = minimum[i]
            remainder = 1.0 - base
            for i in support:
                addition = min(remainder, max(0.0, maximum[i] - weights[i]))
                weights[i] += addition
                remainder -= addition
                if remainder <= 1e-15:
                    break
        else:
            weights[order[0]] = 1.0

        return {
            "weights": weights,
            "selected": [1 if value > 0.0 else 0 for value in weights],
        }
    except BaseException:
        return {"weights": [1.0], "selected": [1]}


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    try:
        limit = max(0.0, float(time_limit_s))
        reserve = min(0.05, max(0.001, 0.005 * limit))
        deadline = started + max(0.0, limit - reserve)

        n = int(instance["num_securities"])
        if n <= 0:
            return _structural_fallback(instance)

        mu = [float(v) for v in instance["expected_returns"]]
        if len(mu) != n:
            return _structural_fallback(instance)

        return_weight = max(0.0, float(instance["return_weight"]))
        theta = float(instance["concentration_parameter"])
        if not math.isfinite(theta) or theta <= 0.0:
            return _structural_fallback(instance)
        inverse_theta = 1.0 / theta

        k_limit = min(n, max(1, int(instance["max_securities"])))

        raw_minimum = instance.get("minimum_weights", [])
        raw_maximum = instance.get("maximum_weights", [])
        if len(raw_minimum) == n and len(raw_maximum) == n:
            minimum = [max(0.0, float(v)) for v in raw_minimum]
            maximum = [max(0.0, float(v)) for v in raw_maximum]
        else:
            minimum = [0.0] * n
            maximum = [1.0] * n

        rules = [
            [float(value) for value in row]
            for row in instance.get("linear_rule_coefficients", [])
        ]
        lower_rules = [
            float(v) for v in instance.get("linear_rule_lower_bounds", [])
        ]
        upper_rules = [
            float(v) for v in instance.get("linear_rule_upper_bounds", [])
        ]
        rule_count = min(len(rules), len(lower_rules), len(upper_rules))
        rules = rules[:rule_count]
        lower_rules = lower_rules[:rule_count]
        upper_rules = upper_rules[:rule_count]
        if any(len(row) != n for row in rules):
            return _structural_fallback(instance)

        covariance_format = instance.get("covariance_format", "direct")
        direct = covariance_format == "direct"
        if direct:
            covariance = [
                [float(value) for value in row]
                for row in instance.get("covariance_matrix", [])
            ]
            if len(covariance) != n or any(len(row) != n for row in covariance):
                return _structural_fallback(instance)
            factors = []
            idiosyncratic = [0.0] * n
        else:
            covariance = []
            factors = [
                [float(value) for value in row]
                for row in instance.get("factor_loadings", [])
            ]
            if any(len(row) != n for row in factors):
                return _structural_fallback(instance)
            idiosyncratic = [
                float(value)
                for value in instance.get("idiosyncratic_variances", [])
            ]
            if len(idiosyncratic) != n:
                return _structural_fallback(instance)

        if time.perf_counter() >= deadline:
            return _structural_fallback(instance)

        diagonal = [0.0] * n
        if direct:
            for i in range(n):
                diagonal[i] = covariance[i][i]
        else:
            for i in range(n):
                value = idiosyncratic[i]
                for row in factors:
                    value += row[i] * row[i]
                diagonal[i] = value

        raw_score = [
            return_weight * mu[i] - 0.5 * (diagonal[i] + inverse_theta)
            for i in range(n)
        ]
        score_scale = max(1.0, max(abs(v) for v in raw_score))
        base_score = [v / score_scale for v in raw_score]

        seed = 1469598103934665603
        seed ^= n + 0x9E3779B97F4A7C15
        seed ^= k_limit << 17
        seed ^= rule_count << 33
        for value in mu[: min(n, 64)]:
            if math.isfinite(value):
                seed ^= int(abs(value) * 1000000007.0) & ((1 << 63) - 1)
                seed = (seed * 1099511628211) & ((1 << 64) - 1)
        rng = random.Random(seed)

        def check_time():
            if time.perf_counter() >= deadline:
                raise _SearchTimeout()

        def active_indices(x):
            return [i for i, value in enumerate(x) if value > 0.0]

        def feasible_x(x, tolerance=2e-7):
            if not isinstance(x, list) or len(x) != n:
                return False
            total = 0.0
            count = 0
            for i, value in enumerate(x):
                if not math.isfinite(value) or value < -tolerance:
                    return False
                value = max(0.0, value)
                total += value
                if value > 0.0:
                    count += 1
                    scale = 1.0 + abs(minimum[i]) + abs(maximum[i])
                    if value < minimum[i] - tolerance * scale:
                        return False
                    if value > maximum[i] + tolerance * scale:
                        return False
            if count > k_limit or abs(total - 1.0) > tolerance:
                return False

            for r in range(rule_count):
                value = 0.0
                row = rules[r]
                for i in range(n):
                    value += row[i] * max(0.0, x[i])
                scale = 1.0 + abs(lower_rules[r]) + abs(upper_rules[r])
                margin = tolerance * scale
                if value < lower_rules[r] - margin:
                    return False
                if value > upper_rules[r] + margin:
                    return False
            return True

        def sanitize(x):
            result = [0.0] * n
            for i, value in enumerate(x):
                if math.isfinite(value) and value > 0.0:
                    result[i] = float(value)
            return result

        def rule_values(x, support=None):
            if support is None:
                support = active_indices(x)
            result = [0.0] * rule_count
            for r in range(rule_count):
                row = rules[r]
                total = 0.0
                for i in support:
                    total += row[i] * x[i]
                result[r] = total
            return result

        def objective(x, support=None):
            check_time()
            if support is None:
                support = active_indices(x)

            linear = 0.0
            concentration = 0.0
            for i in support:
                linear += mu[i] * x[i]
                concentration += x[i] * x[i]

            if direct:
                risk = 0.0
                for position, i in enumerate(support):
                    xi = x[i]
                    if xi != 0.0:
                        row = covariance[i]
                        subtotal = 0.0
                        for j in support:
                            subtotal += row[j] * x[j]
                        risk += xi * subtotal
                    if (position & 31) == 0:
                        check_time()
            else:
                risk = 0.0
                for row_index, row in enumerate(factors):
                    exposure = 0.0
                    for i in support:
                        exposure += row[i] * x[i]
                    risk += exposure * exposure
                    if (row_index & 31) == 0:
                        check_time()
                for i in support:
                    risk += idiosyncratic[i] * x[i] * x[i]

            return (
                0.5 * risk
                + 0.5 * inverse_theta * concentration
                - return_weight * linear
            )

        def gradient(x, targets, source):
            check_time()
            targets = list(targets)
            result = [0.0] * len(targets)

            if direct:
                for position, i in enumerate(targets):
                    row = covariance[i]
                    value = 0.0
                    for j in source:
                        value += row[j] * x[j]
                    result[position] = (
                        value + inverse_theta * x[i] - return_weight * mu[i]
                    )
                    if (position & 31) == 0:
                        check_time()
            else:
                exposures = [0.0] * len(factors)
                for f, row in enumerate(factors):
                    value = 0.0
                    for j in source:
                        value += row[j] * x[j]
                    exposures[f] = value
                    if (f & 31) == 0:
                        check_time()
                for position, i in enumerate(targets):
                    value = idiosyncratic[i] * x[i]
                    for f, row in enumerate(factors):
                        value += row[i] * exposures[f]
                    result[position] = (
                        value + inverse_theta * x[i] - return_weight * mu[i]
                    )
                    if (position & 31) == 0:
                        check_time()
            return result

        def direction_curvature(direction, support):
            check_time()
            concentration = 0.0
            for i in support:
                concentration += direction[i] * direction[i]

            if direct:
                value = 0.0
                for position, i in enumerate(support):
                    di = direction[i]
                    if di != 0.0:
                        row = covariance[i]
                        subtotal = 0.0
                        for j in support:
                            subtotal += row[j] * direction[j]
                        value += di * subtotal
                    if (position & 31) == 0:
                        check_time()
            else:
                value = 0.0
                for f, row in enumerate(factors):
                    exposure = 0.0
                    for i in support:
                        exposure += row[i] * direction[i]
                    value += exposure * exposure
                    if (f & 31) == 0:
                        check_time()
                for i in support:
                    value += idiosyncratic[i] * direction[i] * direction[i]
            return max(0.0, value + inverse_theta * concentration)

        def polytope_lp(support, lows, linear_objective):
            check_time()
            support = tuple(sorted(set(support)))
            size = len(support)
            if size == 0:
                return None

            local_low = [float(lows[j]) for j in range(size)]
            local_high = [maximum[i] for i in support]
            target = 1.0 - sum(local_low)
            capacity = [local_high[j] - local_low[j] for j in range(size)]
            if target < -1e-9 or target > sum(capacity) + 1e-9:
                return None
            if any(value < -1e-9 for value in capacity):
                return None

            A = []
            b = []

            for j in range(size):
                row = [0.0] * size
                row[j] = 1.0
                A.append(row)
                b.append(max(0.0, capacity[j]))

            A.append([1.0] * size)
            b.append(target)
            A.append([-1.0] * size)
            b.append(-target)

            for r in range(rule_count):
                source = rules[r]
                base = 0.0
                row = [0.0] * size
                for j, i in enumerate(support):
                    coefficient = source[i]
                    row[j] = coefficient
                    base += coefficient * local_low[j]
                A.append(row)
                b.append(upper_rules[r] - base)
                A.append([-value for value in row])
                b.append(-(lower_rules[r] - base))

            solver = _Simplex(A, b, linear_objective, deadline)
            local_solution = solver.solve()
            if local_solution is None:
                return None

            x = [0.0] * n
            for j, i in enumerate(support):
                value = local_low[j] + max(0.0, local_solution[j])
                x[i] = value
            x = sanitize(x)
            if not feasible_x(x):
                return None
            return x

        def fixed_support_lp(support, linear_objective=None):
            support = tuple(sorted(set(support)))
            if not support or len(support) > k_limit:
                return None
            if linear_objective is None:
                linear_objective = [base_score[i] for i in support]
            lows = [minimum[i] for i in support]
            return polytope_lp(support, lows, linear_objective)

        best_x = None
        best_value = math.inf
        pool = []
        pool_capacity = min(
            128,
            max(6, 5 + int(1.6 * math.sqrt(max(1.0, limit)))),
        )

        def record(x, known_value=None):
            nonlocal best_x, best_value
            x = sanitize(x)
            if not feasible_x(x):
                return False

            if best_x is None:
                best_x = x[:]

            value = known_value
            if value is None:
                value = objective(x)
            if value < best_value - 1e-13:
                best_value = value
                best_x = x[:]
                return True
            return False

        def install_state(support, x):
            support = tuple(sorted(set(support)))
            if not support or len(support) > k_limit:
                return False
            x = sanitize(x)
            if not feasible_x(x):
                return False

            record(x)
            value = objective(x, support)
            state = {
                "support": support,
                "x": x,
                "value": value,
                "rules": rule_values(x, support),
                "stalls": 0,
            }

            for position, old in enumerate(pool):
                if old["support"] == support:
                    if value < old["value"]:
                        pool[position] = state
                    return True

            if len(pool) < pool_capacity:
                pool.append(state)
            else:
                worst = max(range(len(pool)), key=lambda p: pool[p]["value"])
                if value < pool[worst]["value"] or rng.random() < 0.06:
                    pool[worst] = state
            return True

        def add_support(support, linear_objective=None):
            support = tuple(sorted(set(support)))
            if not support or len(support) > k_limit:
                return False
            x = fixed_support_lp(support, linear_objective)
            if x is None:
                return False
            return install_state(support, x)

        def greedy_weights(support):
            support = tuple(sorted(set(support)))
            if not support or len(support) > k_limit:
                return None
            x = [0.0] * n
            base = 0.0
            for i in support:
                x[i] = minimum[i]
                base += minimum[i]
            if base > 1.0 + 1e-12:
                return None
            remainder = 1.0 - base
            order = sorted(support, key=lambda i: base_score[i], reverse=True)
            for i in order:
                addition = min(remainder, max(0.0, maximum[i] - x[i]))
                x[i] += addition
                remainder -= addition
                if remainder <= 1e-14:
                    break
            if remainder > 1e-9:
                return None
            x = sanitize(x)
            return x if feasible_x(x) else None

        def full_relaxation():
            check_time()
            support = tuple(range(n))
            lows = [0.0] * n
            objective_vector = [
                base_score[i] + 0.03 * rng.uniform(-1.0, 1.0) for i in range(n)
            ]
            x = polytope_lp(support, lows, objective_vector)
            if x is None:
                return False

            positive = active_indices(x)
            if len(positive) <= k_limit and feasible_x(x):
                install_state(tuple(positive), x)
                return True

            ordered = sorted(
                range(n),
                key=lambda i: (x[i], objective_vector[i]),
                reverse=True,
            )
            candidate = tuple(sorted(ordered[:k_limit]))
            return add_support(candidate)

        def pair_step(state):
            support = state["support"]
            if len(support) < 2:
                state["stalls"] += 1
                return False

            i, j = rng.sample(support, 2)
            x = state["x"]
            gradients = gradient(x, (i, j), support)
            linear_term = gradients[0] - gradients[1]

            low = max(minimum[i] - x[i], x[j] - maximum[j])
            high = min(maximum[i] - x[i], x[j] - minimum[j])

            for r in range(rule_count):
                coefficient = rules[r][i] - rules[r][j]
                if abs(coefficient) <= 1e-18:
                    continue
                value = state["rules"][r]
                a = (lower_rules[r] - value) / coefficient
                b = (upper_rules[r] - value) / coefficient
                low = max(low, min(a, b))
                high = min(high, max(a, b))
                if low > high + 1e-12:
                    state["stalls"] += 1
                    return False

            if low > high:
                midpoint = 0.5 * (low + high)
                low = midpoint
                high = midpoint

            if direct:
                curvature = (
                    covariance[i][i]
                    + covariance[j][j]
                    - covariance[i][j]
                    - covariance[j][i]
                    + 2.0 * inverse_theta
                )
            else:
                curvature = idiosyncratic[i] + idiosyncratic[j] + 2.0 * inverse_theta
                for row in factors:
                    difference = row[i] - row[j]
                    curvature += difference * difference
            curvature = max(curvature, 1e-18)

            delta = max(low, min(high, -linear_term / curvature))
            if abs(delta) <= 1e-14:
                state["stalls"] += 1
                return False

            predicted_change = linear_term * delta + 0.5 * curvature * delta * delta
            if predicted_change >= -1e-14:
                state["stalls"] += 1
                return False

            candidate = x[:]
            candidate[i] += delta
            candidate[j] -= delta
            candidate = sanitize(candidate)
            if not feasible_x(candidate):
                state["stalls"] += 1
                return False

            value = objective(candidate, support)
            if value >= state["value"] - 1e-13:
                state["stalls"] += 1
                return False

            state["x"] = candidate
            state["value"] = value
            for r in range(rule_count):
                state["rules"][r] += (rules[r][i] - rules[r][j]) * delta
            state["stalls"] = 0
            record(candidate, value)
            return True

        def frank_wolfe_step(state):
            support = state["support"]
            x = state["x"]
            gradients = gradient(x, support, support)
            oracle_objective = [-value for value in gradients]
            vertex = fixed_support_lp(support, oracle_objective)
            if vertex is None:
                state["stalls"] += 1
                return False

            direction = [0.0] * n
            directional_derivative = 0.0
            for position, i in enumerate(support):
                direction[i] = vertex[i] - x[i]
                directional_derivative += gradients[position] * direction[i]

            if directional_derivative >= -1e-13:
                state["stalls"] += 1
                return False

            curvature = direction_curvature(direction, support)
            if curvature <= 1e-18:
                alpha = 1.0
            else:
                alpha = max(0.0, min(1.0, -directional_derivative / curvature))
            if alpha <= 1e-14:
                state["stalls"] += 1
                return False

            candidate = x[:]
            for i in support:
                candidate[i] = (1.0 - alpha) * x[i] + alpha * vertex[i]
            candidate = sanitize(candidate)
            if not feasible_x(candidate):
                state["stalls"] += 1
                return False

            value = objective(candidate, support)
            if value >= state["value"] - 1e-13:
                state["stalls"] += 1
                return False

            state["x"] = candidate
            state["value"] = value
            new_rule_values = rule_values(vertex, support)
            for r in range(rule_count):
                state["rules"][r] = (
                    (1.0 - alpha) * state["rules"][r]
                    + alpha * new_rule_values[r]
                )
            state["stalls"] = 0
            record(candidate, value)
            return True

        order_by_score = sorted(range(n), key=lambda i: base_score[i], reverse=True)
        order_by_risk = sorted(range(n), key=lambda i: diagonal[i])
        order_by_return = sorted(range(n), key=lambda i: mu[i], reverse=True)
        order_by_capacity = sorted(range(n), key=lambda i: maximum[i], reverse=True)

        heuristic_supports = []
        seen_heuristics = set()
        candidate_sizes = sorted(
            set((k_limit, max(1, k_limit // 2), 1)), reverse=True
        )
        for ordering in (
            order_by_score,
            order_by_risk,
            order_by_return,
            order_by_capacity,
        ):
            for size in candidate_sizes:
                support = tuple(sorted(ordering[:size]))
                if support and support not in seen_heuristics:
                    seen_heuristics.add(support)
                    heuristic_supports.append(support)

        for support in heuristic_supports:
            check_time()
            candidate = greedy_weights(support)
            if candidate is not None:
                install_state(support, candidate)

        # Exact support enumeration advances one support per wall-clock-driven pass.
        combination_order = list(range(n))
        combination_size = k_limit
        combination_iterator = itertools.combinations(range(n), combination_size)

        def next_systematic_support():
            nonlocal combination_size, combination_iterator, combination_order
            while True:
                check_time()
                try:
                    positions = next(combination_iterator)
                    return tuple(sorted(combination_order[p] for p in positions))
                except StopIteration:
                    combination_size -= 1
                    if combination_size < 1:
                        rng.shuffle(combination_order)
                        combination_size = k_limit
                    combination_iterator = itertools.combinations(
                        range(n), combination_size
                    )

        def random_support():
            check_time()
            if k_limit == 1:
                return (rng.randrange(n),)
            size = k_limit - int((rng.random() ** 2) * k_limit)
            size = max(1, min(k_limit, size))
            if size == n:
                return tuple(range(n))
            support = set(rng.sample(range(n), size))
            if sum(maximum[i] for i in support) < 1.0:
                support = set(order_by_capacity[:size])
            return tuple(sorted(support))

        def neighboring_support(elapsed):
            if not pool:
                return random_support()
            ranked = sorted(pool, key=lambda state: state["value"])
            rank = min(len(ranked) - 1, int((rng.random() ** 2) * len(ranked)))
            state = ranked[rank]
            current = set(state["support"])
            x = state["x"]

            if rng.random() < 0.45:
                targets = list(range(n))
                gradients = gradient(x, targets, state["support"])
                outsiders = [i for i in range(n) if i not in current]
                if outsiders:
                    outsiders.sort(key=lambda i: gradients[i])
                    width = min(len(outsiders), 1 + int(math.sqrt(elapsed + 1.0)))
                    entering = outsiders[rng.randrange(width)]
                    if len(current) >= k_limit:
                        leaving = min(
                            current,
                            key=lambda i: (x[i], -gradients[i], rng.random()),
                        )
                        current.remove(leaving)
                    current.add(entering)
            else:
                amplitude = min(k_limit, 1 + int(math.log1p(elapsed + 1.0)))
                changes = 1 + rng.randrange(max(1, amplitude))
                for _ in range(changes):
                    if current and (len(current) >= k_limit or rng.random() < 0.7):
                        current.remove(rng.choice(tuple(current)))
                    outsiders = [i for i in range(n) if i not in current]
                    if outsiders and len(current) < k_limit:
                        current.add(rng.choice(outsiders))

            if not current:
                current.add(rng.randrange(n))
            return tuple(sorted(current))

        # LP relaxation of the binary linking formulation, used as an anytime
        # feasibility branch-and-bound when its tableau is reasonably sized.
        estimated_mip_cells = (3 * n + 2 * rule_count + 3) * (2 * n + 2)
        mip_enabled = estimated_mip_cells <= 2000000
        mip_nodes = [(frozenset(), frozenset())] if mip_enabled else []

        def mip_relaxation(fixed_zero, fixed_one):
            check_time()
            variable_count = 2 * n
            A = []
            b = []

            for i in range(n):
                row = [0.0] * variable_count
                row[i] = 1.0
                row[n + i] = -maximum[i]
                A.append(row)
                b.append(0.0)

                row = [0.0] * variable_count
                row[i] = -1.0
                row[n + i] = minimum[i]
                A.append(row)
                b.append(0.0)

                row = [0.0] * variable_count
                row[n + i] = 1.0
                A.append(row)
                b.append(1.0)

            row = [0.0] * variable_count
            for i in range(n):
                row[n + i] = 1.0
            A.append(row)
            b.append(float(k_limit))

            row = [0.0] * variable_count
            for i in range(n):
                row[i] = 1.0
            A.append(row)
            b.append(1.0)
            A.append([-value for value in row])
            b.append(-1.0)

            for r in range(rule_count):
                row = [0.0] * variable_count
                for i in range(n):
                    row[i] = rules[r][i]
                A.append(row)
                b.append(upper_rules[r])
                A.append([-value for value in row])
                b.append(-lower_rules[r])

            for i in fixed_zero:
                row = [0.0] * variable_count
                row[n + i] = 1.0
                A.append(row)
                b.append(0.0)
            for i in fixed_one:
                row = [0.0] * variable_count
                row[n + i] = -1.0
                A.append(row)
                b.append(-1.0)

            objective_vector = [0.0] * variable_count
            for i in range(n):
                objective_vector[i] = base_score[i] + 0.01 * rng.uniform(-1.0, 1.0)
                objective_vector[n + i] = -0.001 * (1.0 + rng.random())

            solution = _Simplex(A, b, objective_vector, deadline).solve()
            if solution is None:
                return None
            return sanitize(solution[:n]), solution[n:]

        def process_mip_node():
            if not mip_nodes:
                return False
            fixed_zero, fixed_one = mip_nodes.pop()
            relaxation = mip_relaxation(fixed_zero, fixed_one)
            if relaxation is None:
                return False

            x, z = relaxation
            found = False
            positive = active_indices(x)
            if len(positive) <= k_limit and feasible_x(x):
                found = install_state(tuple(positive), x) or found

            ordering = sorted(
                range(n), key=lambda i: (z[i], x[i], base_score[i]), reverse=True
            )
            rounded_support = tuple(sorted(ordering[:k_limit]))
            found = add_support(rounded_support) or found

            fractional = [
                i
                for i in range(n)
                if i not in fixed_zero
                and i not in fixed_one
                and z[i] > 1e-7
                and z[i] < 1.0 - 1e-7
            ]
            if fractional:
                branch = min(fractional, key=lambda i: abs(z[i] - 0.5))
                zero_child = (fixed_zero | frozenset((branch,)), fixed_one)
                one_child = (fixed_zero, fixed_one | frozenset((branch,)))
                if x[branch] > 1e-9 or z[branch] >= 0.5:
                    mip_nodes.append(zero_child)
                    mip_nodes.append(one_child)
                else:
                    mip_nodes.append(one_child)
                    mip_nodes.append(zero_child)
            return found

        continuous_relaxation_cells = (n + 2 * rule_count + 2) * (n + 2)
        full_lp_enabled = continuous_relaxation_cells <= 2000000
        full_lp_pending = full_lp_enabled
        heuristic_position = 0
        iteration = 0

        while time.perf_counter() < deadline:
            try:
                now = time.perf_counter()
                elapsed = max(0.0, now - started)
                progress = min(1.0, elapsed / max(limit, 1e-9))
                iteration += 1

                if best_x is None:
                    if full_lp_pending:
                        full_lp_pending = False
                        full_relaxation()
                        continue
                    if mip_nodes and (iteration & 1) == 0:
                        process_mip_node()
                        continue
                    if heuristic_position < len(heuristic_supports):
                        support = heuristic_supports[heuristic_position]
                        heuristic_position += 1
                        add_support(support)
                        continue
                    if mip_nodes:
                        process_mip_node()
                        continue
                    if full_lp_enabled and rng.random() < 0.25:
                        full_relaxation()
                    elif rng.random() < 0.55:
                        add_support(next_systematic_support())
                    else:
                        add_support(random_support())
                    continue

                exploration_probability = max(0.22, 0.72 * (1.0 - progress))
                if not pool or rng.random() < exploration_probability:
                    choice = rng.random()
                    if mip_nodes and choice < 0.08:
                        process_mip_node()
                    elif full_lp_enabled and choice < 0.20:
                        full_relaxation()
                    elif pool and choice < 0.63:
                        add_support(neighboring_support(elapsed))
                    elif choice < 0.84:
                        add_support(next_systematic_support())
                    else:
                        add_support(random_support())
                else:
                    ranked = sorted(pool, key=lambda state: state["value"])
                    rank = min(
                        len(ranked) - 1,
                        int((rng.random() ** 2) * len(ranked)),
                    )
                    state = ranked[rank]
                    fw_probability = 0.22 + 0.33 * progress
                    if state["stalls"] > 2 or rng.random() < fw_probability:
                        frank_wolfe_step(state)
                    else:
                        pair_step(state)
            except _SearchTimeout:
                break
            except Exception:
                continue

        if best_x is None:
            return _structural_fallback(instance)

        weights = sanitize(best_x)
        residual = 1.0 - sum(weights)
        if abs(residual) > 0.0 and abs(residual) <= 1e-9:
            for i in active_indices(weights):
                candidate = weights[:]
                candidate[i] += residual
                if candidate[i] >= 0.0 and feasible_x(candidate, 5e-8):
                    weights = candidate
                    break

        return {
            "weights": weights,
            "selected": [1 if value > 0.0 else 0 for value in weights],
        }
    except BaseException:
        return _structural_fallback(instance)