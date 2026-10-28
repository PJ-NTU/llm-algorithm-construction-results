import math
import random
import time
import itertools


def _project_box_sum(v, lower, upper):
    s = len(v)
    if s == 0:
        return None
    if sum(lower) > 1.0 + 1e-12 or sum(upper) < 1.0 - 1e-12:
        return None

    left = min(v[j] - upper[j] for j in range(s))
    right = max(v[j] - lower[j] for j in range(s))
    for _ in range(70):
        tau = 0.5 * (left + right)
        total = 0.0
        for j in range(s):
            total += min(upper[j], max(lower[j], v[j] - tau))
        if total > 1.0:
            left = tau
        else:
            right = tau

    tau = 0.5 * (left + right)
    x = [min(upper[j], max(lower[j], v[j] - tau)) for j in range(s)]
    residual = 1.0 - sum(x)

    if residual > 0.0:
        for j in range(s):
            add = min(residual, upper[j] - x[j])
            if add > 0.0:
                x[j] += add
                residual -= add
            if residual <= 1e-14:
                break
    elif residual < 0.0:
        residual = -residual
        for j in range(s):
            take = min(residual, x[j] - lower[j])
            if take > 0.0:
                x[j] -= take
                residual -= take
            if residual <= 1e-14:
                break
    return x


def _rule_violation_local(x, coefficients, lower_bounds, upper_bounds):
    worst = 0.0
    for r in range(len(coefficients)):
        value = sum(coefficients[r][j] * x[j] for j in range(len(x)))
        if value < lower_bounds[r]:
            worst = max(worst, lower_bounds[r] - value)
        elif value > upper_bounds[r]:
            worst = max(worst, value - upper_bounds[r])
    return worst


def _project_feasible(support, seed, lower, upper, rules, rule_lower,
                      rule_upper, end_time):
    s = len(support)
    if s == 0:
        return None

    lo = [lower[i] for i in support]
    hi = [upper[i] for i in support]
    if sum(lo) > 1.0 + 1e-12 or sum(hi) < 1.0 - 1e-12:
        return None

    local_rules = [[row[i] for i in support] for row in rules]
    if seed is None:
        v = [(lo[j] + hi[j]) * 0.5 for j in range(s)]
    else:
        v = [seed[i] for i in support]

    x = _project_box_sum(v, lo, hi)
    if x is None:
        return None

    scale = 1.0
    for r in range(len(local_rules)):
        scale = max(scale, abs(rule_lower[r]), abs(rule_upper[r]))
    target = 2e-9 * scale + 2e-10

    violation = _rule_violation_local(
        x, local_rules, rule_lower, rule_upper
    )
    if violation <= target:
        answer = [0.0] * len(lower)
        for j, i in enumerate(support):
            answer[i] = x[j]
        return answer

    correction_box = [0.0] * s
    correction_rules = [[0.0] * s for _ in local_rules]
    best_box = x[:]
    best_violation = violation

    while time.perf_counter() < end_time:
        y = [x[j] + correction_box[j] for j in range(s)]
        new_x = _project_box_sum(y, lo, hi)
        if new_x is None:
            return None
        correction_box = [y[j] - new_x[j] for j in range(s)]
        x = new_x

        violation = _rule_violation_local(
            x, local_rules, rule_lower, rule_upper
        )
        if violation < best_violation:
            best_violation = violation
            best_box = x[:]
        if violation <= target:
            answer = [0.0] * len(lower)
            for j, i in enumerate(support):
                answer[i] = x[j]
            return answer

        interrupted = False
        for r, a in enumerate(local_rules):
            if (r & 7) == 0 and time.perf_counter() >= end_time:
                interrupted = True
                break
            corr = correction_rules[r]
            y = [x[j] + corr[j] for j in range(s)]
            value = sum(a[j] * y[j] for j in range(s))
            norm2 = sum(value_a * value_a for value_a in a)

            if value < rule_lower[r]:
                if norm2 <= 1e-30:
                    return None
                alpha = (rule_lower[r] - value) / norm2
                new_x = [y[j] + alpha * a[j] for j in range(s)]
            elif value > rule_upper[r]:
                if norm2 <= 1e-30:
                    return None
                alpha = (rule_upper[r] - value) / norm2
                new_x = [y[j] + alpha * a[j] for j in range(s)]
            else:
                new_x = y

            correction_rules[r] = [y[j] - new_x[j] for j in range(s)]
            x = new_x
        if interrupted:
            break

    if best_violation <= target:
        answer = [0.0] * len(lower)
        for j, i in enumerate(support):
            answer[i] = best_box[j]
        return answer
    return None


def _objective(x, risk_kind, covariance, loadings, idiosyncratic,
               inverse_theta, return_weight, expected_returns):
    n = len(x)
    linear = -return_weight * sum(expected_returns[i] * x[i] for i in range(n))
    concentration = 0.5 * inverse_theta * sum(value * value for value in x)

    if risk_kind == "direct":
        risk = 0.0
        active = [i for i in range(n) if x[i] != 0.0]
        for i in active:
            risk += x[i] * sum(covariance[i][j] * x[j] for j in active)
        return 0.5 * risk + concentration + linear

    exposures = []
    for row in loadings:
        exposures.append(sum(row[i] * x[i] for i in range(n)))
    risk = sum(value * value for value in exposures)
    risk += sum(idiosyncratic[i] * x[i] * x[i] for i in range(n))
    return 0.5 * risk + concentration + linear


def _gradient(x, risk_kind, covariance, loadings, idiosyncratic,
              inverse_theta, return_weight, expected_returns):
    n = len(x)
    if risk_kind == "direct":
        active = [i for i in range(n) if x[i] != 0.0]
        return [
            sum(covariance[i][j] * x[j] for j in active)
            + inverse_theta * x[i]
            - return_weight * expected_returns[i]
            for i in range(n)
        ]

    exposures = [sum(row[i] * x[i] for i in range(n)) for row in loadings]
    return [
        sum(loadings[f][i] * exposures[f] for f in range(len(loadings)))
        + (idiosyncratic[i] + inverse_theta) * x[i]
        - return_weight * expected_returns[i]
        for i in range(n)
    ]


def _orthonormal_equalities(support, rules, rule_lower, rule_upper):
    s = len(support)
    candidates = [[1.0] * s]
    for r, row in enumerate(rules):
        scale = 1.0 + abs(rule_lower[r]) + abs(rule_upper[r])
        if abs(rule_upper[r] - rule_lower[r]) <= 1e-11 * scale:
            candidates.append([row[i] for i in support])

    basis = []
    for vector in candidates:
        v = vector[:]
        for b in basis:
            dot = sum(v[j] * b[j] for j in range(s))
            for j in range(s):
                v[j] -= dot * b[j]
        norm = math.sqrt(sum(value * value for value in v))
        if norm > 1e-11:
            basis.append([value / norm for value in v])
    return basis


def _local_optimize(x, support, lower, upper, rules, rule_lower,
                    rule_upper, risk_kind, covariance, loadings,
                    idiosyncratic, inverse_theta, return_weight,
                    expected_returns, rng, end_time, strength):
    if len(support) < 2:
        return x

    n = len(x)
    support = list(support)
    position = {asset: j for j, asset in enumerate(support)}
    equality_basis = _orthonormal_equalities(
        support, rules, rule_lower, rule_upper
    )
    has_extra_equalities = len(equality_basis) > 1
    rule_values = [sum(row[i] * x[i] for i in support) for row in rules]

    if risk_kind == "direct":
        gradient = _gradient(
            x, risk_kind, covariance, loadings, idiosyncratic,
            inverse_theta, return_weight, expected_returns
        )
        exposures = None
    else:
        gradient = None
        exposures = [sum(row[i] * x[i] for i in support) for row in loadings]

    while time.perf_counter() < end_time:
        use_multi = has_extra_equalities or rng.random() < min(
            0.45, 0.10 + 0.035 * math.log1p(strength)
        )

        if use_multi:
            q = min(
                len(support),
                3 + int(math.sqrt(max(0.0, math.log1p(strength))))
            )
            chosen = rng.sample(support, q)
            raw = [rng.gauss(0.0, 1.0) for _ in range(q)]
            mean = sum(raw) / q
            raw = [value - mean for value in raw]
            dense = [0.0] * len(support)
            for asset, value in zip(chosen, raw):
                dense[position[asset]] = value
        else:
            i, j = rng.sample(support, 2)
            dense = [0.0] * len(support)
            dense[position[i]] = 1.0
            dense[position[j]] = -1.0

        for b in equality_basis:
            dot = sum(dense[j] * b[j] for j in range(len(support)))
            for j in range(len(support)):
                dense[j] -= dot * b[j]

        magnitude = max(abs(value) for value in dense)
        if magnitude <= 1e-13:
            continue
        dense = [value / magnitude for value in dense]
        indices = []
        direction = []
        for j, value in enumerate(dense):
            if abs(value) > 1e-13:
                indices.append(support[j])
                direction.append(value)

        t_low = -float("inf")
        t_high = float("inf")
        for asset, d_value in zip(indices, direction):
            a = (lower[asset] - x[asset]) / d_value
            b = (upper[asset] - x[asset]) / d_value
            if a > b:
                a, b = b, a
            t_low = max(t_low, a)
            t_high = min(t_high, b)

        rule_deltas = []
        for r, row in enumerate(rules):
            delta = sum(row[asset] * d_value
                        for asset, d_value in zip(indices, direction))
            rule_deltas.append(delta)
            if abs(delta) <= 1e-13:
                continue
            a = (rule_lower[r] - rule_values[r]) / delta
            b = (rule_upper[r] - rule_values[r]) / delta
            if a > b:
                a, b = b, a
            t_low = max(t_low, a)
            t_high = min(t_high, b)

        if t_low > t_high + 1e-13:
            continue

        if risk_kind == "direct":
            directional_gradient = sum(
                gradient[asset] * d_value
                for asset, d_value in zip(indices, direction)
            )
            curvature = inverse_theta * sum(value * value for value in direction)
            for a_pos, asset_a in enumerate(indices):
                da = direction[a_pos]
                for b_pos, asset_b in enumerate(indices):
                    curvature += da * covariance[asset_a][asset_b] * direction[b_pos]
            loading_direction = None
        else:
            local_gradient = []
            for asset in indices:
                value = sum(loadings[f][asset] * exposures[f]
                            for f in range(len(loadings)))
                value += (idiosyncratic[asset] + inverse_theta) * x[asset]
                value -= return_weight * expected_returns[asset]
                local_gradient.append(value)
            directional_gradient = sum(
                local_gradient[j] * direction[j] for j in range(len(indices))
            )
            loading_direction = [
                sum(loadings[f][asset] * d_value
                    for asset, d_value in zip(indices, direction))
                for f in range(len(loadings))
            ]
            curvature = sum(value * value for value in loading_direction)
            curvature += sum(
                (idiosyncratic[asset] + inverse_theta) * d_value * d_value
                for asset, d_value in zip(indices, direction)
            )

        if curvature <= 1e-18:
            trial = t_low if directional_gradient > 0.0 else t_high
        else:
            trial = -directional_gradient / curvature
            trial = min(t_high, max(t_low, trial))

        predicted_change = directional_gradient * trial + 0.5 * curvature * trial * trial
        if not math.isfinite(trial) or predicted_change >= -1e-15:
            continue

        for asset, d_value in zip(indices, direction):
            x[asset] += trial * d_value
        for r in range(len(rules)):
            rule_values[r] += trial * rule_deltas[r]

        if risk_kind == "direct":
            for k in range(n):
                change = 0.0
                for asset, d_value in zip(indices, direction):
                    change += covariance[k][asset] * d_value
                gradient[k] += trial * change
            for asset, d_value in zip(indices, direction):
                gradient[asset] += trial * inverse_theta * d_value
        else:
            for f in range(len(exposures)):
                exposures[f] += trial * loading_direction[f]

    return x


def _valid(x, lower, upper, max_securities, rules, rule_lower, rule_upper):
    if any(not math.isfinite(value) or value < -2e-8 for value in x):
        return False
    if abs(sum(x) - 1.0) > 2e-7:
        return False

    count = 0
    for i, value in enumerate(x):
        if value > 0.0:
            count += 1
            tolerance = 2e-7 * (1.0 + abs(value) + abs(lower[i]) + abs(upper[i]))
            if value < lower[i] - tolerance or value > upper[i] + tolerance:
                return False
    if count > max_securities:
        return False

    for r, row in enumerate(rules):
        value = sum(row[i] * x[i] for i in range(len(x)))
        tolerance = 2e-7 * (
            1.0 + abs(value) + abs(rule_lower[r]) + abs(rule_upper[r])
        )
        if value < rule_lower[r] - tolerance:
            return False
        if value > rule_upper[r] + tolerance:
            return False
    return True


def _null_vector(matrix, rng):
    if not matrix or not matrix[0]:
        return None
    columns = len(matrix[0])
    permutation = list(range(columns))
    rng.shuffle(permutation)
    a = [[row[j] for j in permutation] for row in matrix]
    rows = len(a)
    pivot_columns = []
    pivot_row = 0

    for column in range(columns):
        best = max(range(pivot_row, rows), key=lambda r: abs(a[r][column]), default=-1)
        if best < 0 or abs(a[best][column]) <= 1e-11:
            continue
        a[pivot_row], a[best] = a[best], a[pivot_row]
        pivot = a[pivot_row][column]
        a[pivot_row] = [value / pivot for value in a[pivot_row]]
        for r in range(rows):
            if r == pivot_row:
                continue
            factor = a[r][column]
            if abs(factor) > 1e-13:
                a[r] = [a[r][j] - factor * a[pivot_row][j]
                        for j in range(columns)]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break

    pivot_set = set(pivot_columns)
    free = next((j for j in range(columns - 1, -1, -1)
                 if j not in pivot_set), None)
    if free is None:
        return None

    vector = [0.0] * columns
    vector[free] = 1.0
    for r in range(len(pivot_columns) - 1, -1, -1):
        column = pivot_columns[r]
        vector[column] = -sum(a[r][j] * vector[j]
                              for j in range(column + 1, columns))

    unpermuted = [0.0] * columns
    for j, original in enumerate(permutation):
        unpermuted[original] = vector[j]
    return unpermuted


def _sparsify_relaxed(x, upper, rules, max_securities, rng, end_time):
    x = x[:]
    while time.perf_counter() < end_time:
        active = [i for i, value in enumerate(x) if value > 1e-10]
        if len(active) <= max_securities:
            break
        matrix = [[1.0] * len(active)]
        matrix.extend([[row[i] for i in active] for row in rules])
        direction = _null_vector(matrix, rng)
        if direction is None:
            break

        low = -float("inf")
        high = float("inf")
        for j, asset in enumerate(active):
            d_value = direction[j]
            if abs(d_value) <= 1e-14:
                continue
            a = -x[asset] / d_value
            b = (upper[asset] - x[asset]) / d_value
            if a > b:
                a, b = b, a
            low = max(low, a)
            high = min(high, b)

        choices = [value for value in (low, high) if math.isfinite(value)]
        if not choices:
            break
        choices.sort(
            key=lambda t: sum(1 for j, asset in enumerate(active)
                              if x[asset] + t * direction[j] <= 1e-9),
            reverse=True
        )
        step = choices[0]
        if abs(step) <= 1e-13:
            continue
        for j, asset in enumerate(active):
            value = x[asset] + step * direction[j]
            if abs(value) <= 1e-9:
                value = 0.0
            elif abs(value - upper[asset]) <= 1e-9:
                value = upper[asset]
            x[asset] = value
    return x


def _weighted_support(weights, size, rng):
    keys = []
    for i, weight in enumerate(weights):
        adjusted = max(1e-12, weight)
        key = math.log(max(1e-300, rng.random())) / adjusted
        keys.append((key, i))
    keys.sort(reverse=True)
    return tuple(sorted(i for _, i in keys[:size]))


def _emergency(instance):
    try:
        n = max(0, int(instance.get("num_securities", 0)))
        if n == 0:
            return {"weights": [], "selected": []}
        k = max(1, min(n, int(instance.get("max_securities", n))))
        raw_min = instance.get("minimum_weights", [])
        raw_max = instance.get("maximum_weights", [])
        lower = [float(raw_min[i]) for i in range(n)] if len(raw_min) == n else [0.0] * n
        upper = [float(raw_max[i]) for i in range(n)] if len(raw_max) == n else [1.0] * n
        order = sorted(range(n), key=lambda i: upper[i], reverse=True)
        support = []
        lower_sum = 0.0
        upper_sum = 0.0
        for i in order:
            if len(support) >= k:
                break
            if lower_sum + lower[i] <= 1.0 + 1e-12:
                support.append(i)
                lower_sum += lower[i]
                upper_sum += upper[i]
                if upper_sum >= 1.0 - 1e-12:
                    break
        if not support or upper_sum < 1.0 - 1e-12:
            support = order[:k]
        lo = [lower[i] for i in support]
        hi = [upper[i] for i in support]
        local = _project_box_sum([(lo[j] + hi[j]) * 0.5 for j in range(len(support))], lo, hi)
        if local is None:
            weights = [0.0] * n
            weights[order[0]] = 1.0
        else:
            weights = [0.0] * n
            for j, i in enumerate(support):
                weights[i] = local[j]
        return {
            "weights": weights,
            "selected": [1 if value > 0.0 else 0 for value in weights],
        }
    except BaseException:
        return {"weights": [], "selected": []}


def _solve_impl(instance, time_limit_s):
    start = time.perf_counter()
    budget = float(time_limit_s)
    if not math.isfinite(budget) or budget < 0.0:
        budget = 0.0
    deadline = start + budget
    if budget > 0.0:
        deadline -= min(0.005, budget * 0.001)

    n = int(instance["num_securities"])
    expected_returns = [float(v) for v in instance["expected_returns"]]
    return_weight = max(0.0, float(instance["return_weight"]))
    theta = float(instance["concentration_parameter"])
    inverse_theta = 1.0 / theta
    max_securities = max(1, min(n, int(instance["max_securities"])))

    raw_lower = instance.get("minimum_weights", [])
    raw_upper = instance.get("maximum_weights", [])
    if len(raw_lower) == n and len(raw_upper) == n:
        lower = [max(0.0, float(v)) for v in raw_lower]
        upper = [min(1.0, float(v)) for v in raw_upper]
    else:
        lower = [0.0] * n
        upper = [1.0] * n

    rules = [[float(v) for v in row]
             for row in instance.get("linear_rule_coefficients", [])]
    rule_lower = [float(v) for v in instance.get("linear_rule_lower_bounds", [])]
    rule_upper = [float(v) for v in instance.get("linear_rule_upper_bounds", [])]

    risk_kind = instance.get("covariance_format", "direct")
    if risk_kind == "factor":
        covariance = None
        loadings = [[float(v) for v in row]
                    for row in instance.get("factor_loadings", [])]
        idiosyncratic = [float(v) for v in instance.get("idiosyncratic_variances", [])]
    else:
        risk_kind = "direct"
        covariance = [[float(v) for v in row]
                      for row in instance.get("covariance_matrix", [])]
        loadings = []
        idiosyncratic = []

    seed_number = n * 1000003 + max_securities * 9176 + len(rules) * 131
    for i, value in enumerate(expected_returns):
        seed_number ^= int(abs(value) * 1000000007.0 + i * 97) & 0xFFFFFFFF
    rng = random.Random(seed_number)

    diagonal = []
    for i in range(n):
        if risk_kind == "direct":
            variance = covariance[i][i]
        else:
            variance = idiosyncratic[i] + sum(row[i] * row[i] for row in loadings)
        diagonal.append(variance + inverse_theta)
    standalone_score = [
        0.5 * diagonal[i] - return_weight * expected_returns[i]
        for i in range(n)
    ]
    ranked = sorted(range(n), key=lambda i: standalone_score[i])

    best_x = None
    best_value = float("inf")
    best_update_time = start
    elites = []
    relaxed = None

    if time.perf_counter() < deadline:
        relaxed_end = min(deadline, time.perf_counter() + 0.12)
        relaxed = _project_feasible(
            tuple(range(n)), None, [0.0] * n, upper,
            rules, rule_lower, rule_upper, relaxed_end
        )
        if relaxed is not None:
            actual = tuple(i for i, value in enumerate(relaxed) if value > 0.0)
            if _valid(relaxed, lower, upper, max_securities,
                      rules, rule_lower, rule_upper):
                best_x = relaxed[:]
                best_value = _objective(
                    best_x, risk_kind, covariance, loadings, idiosyncratic,
                    inverse_theta, return_weight, expected_returns
                )
                elites.append((best_value, best_x[:], actual))
            if len(actual) > max_securities and time.perf_counter() < deadline:
                sparse_end = min(deadline, time.perf_counter() + 0.08)
                sparse = _sparsify_relaxed(
                    relaxed, upper, rules, max_securities, rng, sparse_end
                )
                sparse_support = tuple(i for i, value in enumerate(sparse) if value > 1e-10)
                if len(sparse_support) <= max_securities:
                    repaired = _project_feasible(
                        sparse_support, sparse, lower, upper,
                        rules, rule_lower, rule_upper,
                        min(deadline, time.perf_counter() + 0.05)
                    )
                    if repaired is not None and _valid(
                        repaired, lower, upper, max_securities,
                        rules, rule_lower, rule_upper
                    ):
                        value = _objective(
                            repaired, risk_kind, covariance, loadings,
                            idiosyncratic, inverse_theta, return_weight,
                            expected_returns
                        )
                        best_x, best_value = repaired, value
                        elites.append((value, repaired[:], sparse_support))

    initial_supports = []
    support_keys = set()

    def add_initial(values):
        key = tuple(sorted(set(int(i) for i in values if 0 <= int(i) < n)))
        if 0 < len(key) <= max_securities and key not in support_keys:
            support_keys.add(key)
            initial_supports.append(key)

    add_initial(ranked[:max_securities])
    add_initial(sorted(range(n), key=lambda i: upper[i], reverse=True)[:max_securities])
    add_initial(sorted(range(n), key=lambda i: expected_returns[i], reverse=True)[:max_securities])
    if n <= max_securities:
        add_initial(range(n))
    if relaxed is not None:
        add_initial(sorted(range(n), key=lambda i: relaxed[i], reverse=True)[:max_securities])

    for i in ranked:
        if upper[i] >= 1.0 - 1e-12 and lower[i] <= 1.0 + 1e-12:
            add_initial((i,))
        if time.perf_counter() >= deadline:
            break

    build_end = min(deadline, time.perf_counter() + 0.03)
    for row in rules:
        if time.perf_counter() >= build_end:
            break
        add_initial(sorted(range(n), key=lambda i: row[i])[:max_securities])
        add_initial(sorted(range(n), key=lambda i: row[i], reverse=True)[:max_securities])

    enumeration = itertools.chain.from_iterable(
        itertools.combinations(range(n), size)
        for size in range(max_securities, 0, -1)
    )
    initial_index = 0
    attempts = 0

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        elapsed = max(0.0, now - start)
        attempts += 1
        seed_x = None

        if initial_index < len(initial_supports):
            support = initial_supports[initial_index]
            initial_index += 1
            if relaxed is not None:
                seed_x = relaxed
        elif enumeration is not None and attempts % max(
            3, 13 - int(min(9.0, math.log1p(elapsed) * 2.0))
        ) == 0:
            try:
                support = next(enumeration)
            except StopIteration:
                enumeration = None
                continue
        elif elites and rng.random() > min(
            0.65, 0.18 + 0.055 * math.log1p(elapsed)
        ):
            parent_index = min(
                len(elites) - 1,
                int((rng.random() ** 2) * len(elites))
            )
            _, parent_x, parent_support = elites[parent_index]
            seed_x = parent_x
            chosen = set(parent_support)
            gradient = _gradient(
                parent_x, risk_kind, covariance, loadings, idiosyncratic,
                inverse_theta, return_weight, expected_returns
            )
            stagnation = max(0.0, now - best_update_time)
            radius = 1 + int(
                min(
                    max(0, max_securities - 1),
                    rng.random() * (
                        math.log1p(elapsed) + math.sqrt(math.log1p(stagnation))
                    )
                )
            )

            if len(chosen) < max_securities and rng.random() < 0.45:
                remove_count = 0
                add_count = min(radius, max_securities - len(chosen))
            else:
                remove_count = min(radius, max(0, len(chosen) - 1))
                add_count = remove_count
                if len(chosen) < max_securities and rng.random() < 0.35:
                    add_count = min(add_count + 1, max_securities - len(chosen) + remove_count)

            for _ in range(remove_count):
                if len(chosen) <= 1:
                    break
                sample_size = min(
                    len(chosen), 3 + int(math.sqrt(attempts))
                )
                candidates = rng.sample(list(chosen), sample_size)
                outgoing = min(
                    candidates,
                    key=lambda i: parent_x[i] - 0.05 * gradient[i]
                )
                chosen.remove(outgoing)

            for _ in range(add_count):
                outside = [i for i in range(n) if i not in chosen]
                if not outside or len(chosen) >= max_securities:
                    break
                pool_size = min(
                    len(outside),
                    4 + int(math.sqrt(attempts)) + int(3.0 * math.log1p(elapsed))
                )
                pool = rng.sample(outside, pool_size)
                incoming = min(pool, key=lambda i: gradient[i])
                chosen.add(incoming)
            support = tuple(sorted(chosen))
        else:
            size = max_securities
            if max_securities > 1 and rng.random() < 0.18:
                size = 1 + rng.randrange(max_securities)
            if relaxed is not None and rng.random() < 0.65:
                support = _weighted_support(relaxed, size, rng)
                seed_x = relaxed
            else:
                expanding_pool = min(
                    n, max(size, size + int(math.sqrt(attempts)))
                )
                if expanding_pool < n and rng.random() < 0.75:
                    pool = ranked[:expanding_pool]
                    support = tuple(sorted(rng.sample(pool, size)))
                else:
                    support = tuple(sorted(rng.sample(range(n), size)))

        if not support:
            continue

        feasibility_slice = (
            0.012 + 0.006 * math.log1p(elapsed)
            + (0.08 if best_x is None else 0.0)
        )
        feasible = _project_feasible(
            support, seed_x, lower, upper, rules, rule_lower, rule_upper,
            min(deadline, time.perf_counter() + feasibility_slice)
        )
        if feasible is None:
            continue

        before_local = feasible[:]
        before_value = _objective(
            before_local, risk_kind, covariance, loadings, idiosyncratic,
            inverse_theta, return_weight, expected_returns
        )
        promising = best_x is None or before_value < best_value + 0.05 * (1.0 + abs(best_value))
        optimization_slice = (
            0.005 + 0.003 * math.log1p(elapsed)
            + (0.012 if promising else 0.0)
        )
        feasible = _local_optimize(
            feasible, support, lower, upper, rules, rule_lower, rule_upper,
            risk_kind, covariance, loadings, idiosyncratic,
            inverse_theta, return_weight, expected_returns, rng,
            min(deadline, time.perf_counter() + optimization_slice),
            elapsed + 1.0
        )

        if not _valid(feasible, lower, upper, max_securities,
                      rules, rule_lower, rule_upper):
            feasible = before_local
        if not _valid(feasible, lower, upper, max_securities,
                      rules, rule_lower, rule_upper):
            continue

        value = _objective(
            feasible, risk_kind, covariance, loadings, idiosyncratic,
            inverse_theta, return_weight, expected_returns
        )
        actual_support = tuple(i for i, weight in enumerate(feasible) if weight > 0.0)

        replaced = False
        for index, entry in enumerate(elites):
            if entry[2] == actual_support:
                if value < entry[0]:
                    elites[index] = (value, feasible[:], actual_support)
                replaced = True
                break
        if not replaced:
            elites.append((value, feasible[:], actual_support))
        elites.sort(key=lambda entry: entry[0])
        elite_capacity = 8 + int(4.0 * math.sqrt(elapsed + 1.0))
        if len(elites) > elite_capacity:
            del elites[elite_capacity:]

        if value < best_value - 1e-14 * (1.0 + abs(best_value)):
            best_value = value
            best_x = feasible[:]
            best_update_time = time.perf_counter()

    if best_x is None:
        return _emergency(instance)

    weights = []
    for value in best_x:
        if value < 0.0 and value > -1e-10:
            value = 0.0
        weights.append(float(value))
    residual = 1.0 - sum(weights)
    if abs(residual) <= 1e-8:
        positive = [i for i, value in enumerate(weights) if value > 0.0]
        if positive:
            weights[max(positive, key=lambda i: weights[i])] += residual

    return {
        "weights": weights,
        "selected": [1 if value > 0.0 else 0 for value in weights],
    }


def solve(instance: dict, time_limit_s: float) -> dict:
    try:
        result = _solve_impl(instance, time_limit_s)
        if result is None:
            return _emergency(instance)
        return result
    except BaseException:
        return _emergency(instance)
