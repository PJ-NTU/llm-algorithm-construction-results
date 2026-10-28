import heapq
import itertools
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    try:
        answer = _solve(instance, time_limit_s)
        if answer is not None:
            return answer
    except BaseException:
        pass
    return _fallback(instance)


def _fallback(instance):
    try:
        n = max(1, int(instance.get("num_securities", 1)))
        k_limit = max(1, min(n, int(instance.get("max_securities", n))))
        raw_lower = instance.get("minimum_weights") or []
        raw_upper = instance.get("maximum_weights") or []
        lower = ([max(0.0, float(v)) for v in raw_lower]
                 if len(raw_lower) == n else [0.0] * n)
        upper = ([max(0.0, float(v)) for v in raw_upper]
                 if len(raw_upper) == n else [1.0] * n)
        rules = instance.get("linear_rule_coefficients") or []
        rule_lower = instance.get("linear_rule_lower_bounds") or []
        rule_upper = instance.get("linear_rule_upper_bounds") or []
        returns = instance.get("expected_returns") or []

        def construct(support):
            if not support or len(support) > k_limit:
                return None
            values = [lower[i] for i in support]
            remaining = 1.0 - sum(values)
            if remaining < -1e-9:
                return None
            while remaining > 1e-13:
                free = [j for j, asset in enumerate(support)
                        if upper[asset] - values[j] > 1e-15]
                if not free:
                    return None
                share = remaining / len(free)
                change = 0.0
                for j in free:
                    addition = min(share, upper[support[j]] - values[j])
                    values[j] += addition
                    change += addition
                if change <= 1e-16:
                    return None
                remaining -= change
            if abs(sum(values) - 1.0) > 2e-7:
                return None
            for r, row in enumerate(rules):
                value = sum(float(row[asset]) * values[j]
                            for j, asset in enumerate(support))
                if r < len(rule_lower) and value < float(rule_lower[r]) - 2e-7:
                    return None
                if r < len(rule_upper) and value > float(rule_upper[r]) + 2e-7:
                    return None
            weights = [0.0] * n
            for j, asset in enumerate(support):
                weights[asset] = max(0.0, values[j])
            return {
                "weights": weights,
                "selected": [1 if value > 0.0 else 0 for value in weights],
            }

        upper_order = sorted(range(n), key=lambda i: upper[i], reverse=True)
        orders = [upper_order]
        if len(returns) == n:
            orders.append(sorted(range(n),
                                 key=lambda i: float(returns[i]), reverse=True))
        for asset in range(n):
            result = construct((asset,))
            if result is not None:
                return result
        for order in orders:
            for size in range(2, k_limit + 1):
                result = construct(tuple(sorted(order[:size])))
                if result is not None:
                    return result
        weights = [0.0] * n
        weights[upper_order[0]] = 1.0
        return {
            "weights": weights,
            "selected": [1 if value > 0.0 else 0 for value in weights],
        }
    except BaseException:
        return {"weights": [1.0], "selected": [1]}


def _solve(instance, time_limit_s):
    clock = time.monotonic
    started = clock()
    try:
        limit = float(time_limit_s)
    except BaseException:
        limit = 0.0
    if not math.isfinite(limit) or limit < 0.0:
        limit = 0.0
    deadline = started + limit
    reserve = min(0.05, max(0.003, limit * 0.001)) if limit > 0.0 else 0.0
    search_end = max(started, deadline - reserve)

    n = int(instance["num_securities"])
    k_limit = max(1, min(n, int(instance["max_securities"])))
    expected_returns = [float(v) for v in instance["expected_returns"]]
    return_weight = max(0.0, float(instance["return_weight"]))
    theta = float(instance["concentration_parameter"])
    inverse_theta = 1.0 / theta

    rules = [[float(v) for v in row]
             for row in (instance.get("linear_rule_coefficients") or [])]
    rule_lower = [float(v)
                  for v in (instance.get("linear_rule_lower_bounds") or [])]
    rule_upper = [float(v)
                  for v in (instance.get("linear_rule_upper_bounds") or [])]
    rule_count = len(rules)

    raw_lower = instance.get("minimum_weights") or []
    raw_upper = instance.get("maximum_weights") or []
    minimum = ([max(0.0, float(v)) for v in raw_lower]
               if len(raw_lower) == n else [0.0] * n)
    maximum = ([max(0.0, float(v)) for v in raw_upper]
               if len(raw_upper) == n else [1.0] * n)

    risk_kind = instance.get("covariance_format", "direct")
    if risk_kind == "factor":
        covariance = None
        loadings = [[float(v) for v in row]
                    for row in (instance.get("factor_loadings") or [])]
        idiosyncratic = [float(v)
                         for v in instance["idiosyncratic_variances"]]
        diagonal = [
            idiosyncratic[i] + sum(row[i] * row[i] for row in loadings)
            for i in range(n)
        ]
    else:
        risk_kind = "direct"
        covariance = [[float(v) for v in row]
                      for row in instance["covariance_matrix"]]
        loadings = []
        idiosyncratic = []
        diagonal = [covariance[i][i] for i in range(n)]

    base_score = [
        0.5 * (diagonal[i] + inverse_theta)
        - return_weight * expected_returns[i]
        for i in range(n)
    ]
    base_order = sorted(range(n), key=lambda i: (base_score[i], i))
    return_order = sorted(
        range(n), key=lambda i: (-expected_returns[i], base_score[i], i)
    )
    risk_order = sorted(
        range(n), key=lambda i: (diagonal[i], -expected_returns[i], i)
    )
    upper_order = sorted(
        range(n), key=lambda i: (-maximum[i], base_score[i], i)
    )
    base_rank = {asset: rank for rank, asset in enumerate(base_order)}

    seed = n * 1000003 + k_limit * 9176 + rule_count * 131
    for i, value in enumerate(expected_returns[:256]):
        if math.isfinite(value):
            seed ^= ((int(abs(value) * 1000000007.0) + i * 104729)
                     & 0xFFFFFFFFFFFFFFFF)
    rng = random.Random(seed & 0xFFFFFFFFFFFFFFFF)
    feasibility_tolerance = 2e-7

    def residual(support, values, lows=None, highs=None):
        if len(support) != len(values):
            return float("inf")
        if lows is None:
            lows = [minimum[i] for i in support]
        if highs is None:
            highs = [maximum[i] for i in support]
        violation = abs(sum(values) - 1.0)
        for j, value in enumerate(values):
            if not math.isfinite(value):
                return float("inf")
            violation = max(
                violation, -value, lows[j] - value, value - highs[j]
            )
        for r in range(rule_count):
            value = sum(rules[r][asset] * values[j]
                        for j, asset in enumerate(support))
            if math.isfinite(rule_lower[r]):
                violation = max(violation, rule_lower[r] - value)
            if math.isfinite(rule_upper[r]):
                violation = max(violation, value - rule_upper[r])
        return max(0.0, violation)

    def project_box_sum(vector, lows, highs):
        size = len(vector)
        if size == 0:
            return None
        if sum(lows) > 1.0 + 1e-11 or sum(highs) < 1.0 - 1e-11:
            return None
        left = min(vector[j] - highs[j] for j in range(size))
        right = max(vector[j] - lows[j] for j in range(size))
        for _ in range(64):
            tau = 0.5 * (left + right)
            total = sum(min(highs[j], max(lows[j], vector[j] - tau))
                        for j in range(size))
            if total > 1.0:
                left = tau
            else:
                right = tau
        tau = 0.5 * (left + right)
        result = [min(highs[j], max(lows[j], vector[j] - tau))
                  for j in range(size)]
        difference = 1.0 - sum(result)
        if difference > 0.0:
            for j in range(size):
                addition = min(difference, highs[j] - result[j])
                if addition > 0.0:
                    result[j] += addition
                    difference -= addition
                if difference <= 1e-14:
                    break
        elif difference < 0.0:
            difference = -difference
            for j in range(size):
                removal = min(difference, result[j] - lows[j])
                if removal > 0.0:
                    result[j] -= removal
                    difference -= removal
                if difference <= 1e-14:
                    break
        return result

    def projected_feasible(support, lows, highs, seed_values, until):
        size = len(support)
        if size == 0 or clock() >= until:
            return None
        if sum(lows) > 1.0 + 1e-10 or sum(highs) < 1.0 - 1e-10:
            return None
        local_rules = [[row[asset] for asset in support] for row in rules]
        if seed_values is None or len(seed_values) != size:
            vector = [(lows[j] + highs[j]) * 0.5 for j in range(size)]
        else:
            vector = list(seed_values)
        values = project_box_sum(vector, lows, highs)
        if values is None:
            return None
        if rule_count == 0:
            return values

        scale = 1.0
        for r in range(rule_count):
            scale = max(scale, abs(rule_lower[r]), abs(rule_upper[r]))
        target = 2e-9 * scale + 2e-10
        box_correction = [0.0] * size
        rule_corrections = [0.0] * rule_count

        while clock() < until:
            y = [values[j] + box_correction[j] for j in range(size)]
            new_values = project_box_sum(y, lows, highs)
            if new_values is None:
                return None
            box_correction = [y[j] - new_values[j] for j in range(size)]
            values = new_values
            if residual(support, values, lows, highs) <= target:
                return values

            interrupted = False
            for r, normal in enumerate(local_rules):
                if (r & 7) == 0 and clock() >= until:
                    interrupted = True
                    break
                correction = rule_corrections[r]
                y = [values[j] + correction * normal[j]
                     for j in range(size)]
                value = sum(normal[j] * y[j] for j in range(size))
                norm_squared = sum(coefficient * coefficient
                                   for coefficient in normal)
                alpha = 0.0
                if value < rule_lower[r]:
                    if norm_squared <= 1e-30:
                        return None
                    alpha = (rule_lower[r] - value) / norm_squared
                elif value > rule_upper[r]:
                    if norm_squared <= 1e-30:
                        return None
                    alpha = (rule_upper[r] - value) / norm_squared
                values = [y[j] + alpha * normal[j] for j in range(size)]
                rule_corrections[r] = -alpha
            if interrupted:
                break
        if residual(support, values, lows, highs) <= target:
            return values
        return None

    def support_possible(support):
        if not support or len(support) > k_limit:
            return False
        lows = [minimum[i] for i in support]
        capacities = [maximum[i] - minimum[i] for i in support]
        if any(capacity < -1e-11 for capacity in capacities):
            return False
        remaining = 1.0 - sum(lows)
        if remaining < -1e-9 or remaining > sum(capacities) + 1e-9:
            return False
        for r in range(rule_count):
            row = rules[r]
            base = sum(row[asset] * lows[j]
                       for j, asset in enumerate(support))
            pairs = sorted((row[asset], max(0.0, capacities[j]))
                           for j, asset in enumerate(support))
            left = max(0.0, remaining)
            minimum_value = base
            for coefficient, capacity in pairs:
                used = min(left, capacity)
                minimum_value += coefficient * used
                left -= used
                if left <= 1e-14:
                    break
            left = max(0.0, remaining)
            maximum_value = base
            for coefficient, capacity in reversed(pairs):
                used = min(left, capacity)
                maximum_value += coefficient * used
                left -= used
                if left <= 1e-14:
                    break
            if (math.isfinite(rule_lower[r])
                    and maximum_value < rule_lower[r] - 1e-8):
                return False
            if (math.isfinite(rule_upper[r])
                    and minimum_value > rule_upper[r] + 1e-8):
                return False
        return True

    def box_point(lows, highs):
        values = list(lows)
        remaining = 1.0 - sum(values)
        capacity = sum(highs[j] - lows[j] for j in range(len(lows)))
        if remaining < -1e-10 or remaining > capacity + 1e-10:
            return None
        while remaining > 1e-14:
            free = [j for j in range(len(values))
                    if highs[j] - values[j] > 1e-15]
            if not free:
                return None
            share = remaining / len(free)
            change = 0.0
            for j in free:
                addition = min(share, highs[j] - values[j])
                values[j] += addition
                change += addition
            if change <= 1e-16:
                return None
            remaining -= change
        difference = 1.0 - sum(values)
        if difference != 0.0:
            for j in range(len(values)):
                if lows[j] - 1e-12 <= values[j] + difference <= highs[j] + 1e-12:
                    values[j] += difference
                    break
        return values

    def lp_feasible(support, lows, highs, until):
        size = len(support)
        if size == 0 or clock() >= until:
            return None
        if rule_count == 0:
            return box_point(lows, highs)
        remaining = 1.0 - sum(lows)
        capacities = [highs[j] - lows[j] for j in range(size)]
        if (remaining < -1e-9
                or remaining > sum(capacities) + 1e-9
                or any(value < -1e-10 for value in capacities)):
            return None

        rows = []
        right_hand_sides = []
        for j in range(size):
            row = [0.0] * size
            row[j] = 1.0
            rows.append(row)
            right_hand_sides.append(capacities[j])
        rows.append([1.0] * size)
        right_hand_sides.append(remaining)
        rows.append([-1.0] * size)
        right_hand_sides.append(-remaining)
        for r in range(rule_count):
            coefficients = [rules[r][asset] for asset in support]
            base = sum(coefficients[j] * lows[j] for j in range(size))
            if math.isfinite(rule_upper[r]):
                rows.append(coefficients)
                right_hand_sides.append(rule_upper[r] - base)
            if math.isfinite(rule_lower[r]):
                rows.append([-value for value in coefficients])
                right_hand_sides.append(-rule_lower[r] + base)

        row_count = len(rows)
        variable_count = size
        epsilon = 1e-10
        tableau = [[0.0] * (variable_count + 2)
                   for _ in range(row_count + 2)]
        basic = [0] * row_count
        nonbasic = [0] * (variable_count + 1)
        for i in range(row_count):
            if clock() >= until:
                return None
            for j in range(variable_count):
                tableau[i][j] = rows[i][j]
            basic[i] = variable_count + i
            tableau[i][variable_count] = -1.0
            tableau[i][variable_count + 1] = right_hand_sides[i]
        for j in range(variable_count):
            nonbasic[j] = j
        nonbasic[variable_count] = -1
        tableau[row_count + 1][variable_count] = 1.0

        def pivot(leaving, entering):
            inverse = 1.0 / tableau[leaving][entering]
            for i in range(row_count + 2):
                if i == leaving:
                    continue
                factor = tableau[i][entering] * inverse
                if factor == 0.0:
                    continue
                for j in range(variable_count + 2):
                    if j != entering:
                        tableau[i][j] -= tableau[leaving][j] * factor
            for j in range(variable_count + 2):
                if j != entering:
                    tableau[leaving][j] *= inverse
            for i in range(row_count + 2):
                if i != leaving:
                    tableau[i][entering] *= -inverse
            tableau[leaving][entering] = inverse
            basic[leaving], nonbasic[entering] = (
                nonbasic[entering], basic[leaving]
            )

        def simplex(phase):
            objective_row = row_count + 1 if phase == 1 else row_count
            while clock() < until:
                entering = -1
                for j in range(variable_count + 1):
                    if phase == 2 and nonbasic[j] == -1:
                        continue
                    if (entering == -1
                            or tableau[objective_row][j]
                            < tableau[objective_row][entering] - epsilon
                            or (abs(tableau[objective_row][j]
                                    - tableau[objective_row][entering]) <= epsilon
                                and nonbasic[j] < nonbasic[entering])):
                        entering = j
                if (entering == -1
                        or tableau[objective_row][entering] >= -epsilon):
                    return True
                leaving = -1
                for i in range(row_count):
                    if tableau[i][entering] <= epsilon:
                        continue
                    if leaving == -1:
                        leaving = i
                    else:
                        first = (tableau[i][variable_count + 1]
                                 / tableau[i][entering])
                        second = (tableau[leaving][variable_count + 1]
                                  / tableau[leaving][entering])
                        if (first < second - epsilon
                                or (abs(first - second) <= epsilon
                                    and basic[i] < basic[leaving])):
                            leaving = i
                if leaving == -1:
                    return False
                pivot(leaving, entering)
            return None

        leaving = min(range(row_count),
                      key=lambda i: tableau[i][variable_count + 1])
        if tableau[leaving][variable_count + 1] < -epsilon:
            pivot(leaving, variable_count)
            phase_result = simplex(1)
            if (phase_result is not True
                    or tableau[row_count + 1][variable_count + 1] < -epsilon
                    or abs(tableau[row_count + 1][variable_count + 1]) > 2e-8):
                return None
            for i in range(row_count):
                if basic[i] == -1:
                    entering = -1
                    for j in range(variable_count + 1):
                        if (abs(tableau[i][j]) > epsilon
                                and (entering == -1
                                     or abs(tableau[i][j])
                                     > abs(tableau[i][entering]) + epsilon
                                     or (abs(abs(tableau[i][j])
                                             - abs(tableau[i][entering])) <= epsilon
                                         and nonbasic[j] < nonbasic[entering]))):
                            entering = j
                    if entering != -1:
                        pivot(i, entering)
                    break
        if simplex(2) is not True:
            return None
        increments = [0.0] * variable_count
        for i in range(row_count):
            if 0 <= basic[i] < variable_count:
                increments[basic[i]] = tableau[i][variable_count + 1]
        values = [lows[j] + max(0.0, increments[j])
                  for j in range(variable_count)]
        if residual(support, values, lows, highs) <= feasibility_tolerance:
            return values
        return None

    def build_hessian(support, until):
        size = len(support)
        hessian = [[0.0] * size for _ in range(size)]
        if risk_kind == "direct":
            for i, asset_i in enumerate(support):
                if clock() >= until:
                    return None
                source = covariance[asset_i]
                for j, asset_j in enumerate(support):
                    hessian[i][j] = source[asset_j]
                hessian[i][i] += inverse_theta
        else:
            for i, asset in enumerate(support):
                hessian[i][i] = idiosyncratic[asset] + inverse_theta
            for factor, row in enumerate(loadings):
                if clock() >= until:
                    return None
                local = [row[asset] for asset in support]
                for i in range(size):
                    value_i = local[i]
                    for j in range(i + 1):
                        addition = value_i * local[j]
                        hessian[i][j] += addition
                        if i != j:
                            hessian[j][i] += addition
        return hessian

    def objective(support, values, hessian):
        total = 0.0
        for i in range(len(support)):
            hessian_value = sum(hessian[i][j] * values[j]
                                  for j in range(len(support)))
            total += (0.5 * values[i] * hessian_value
                      - return_weight * expected_returns[support[i]] * values[i])
        return total

    def cholesky(hessian, until):
        size = len(hessian)
        result = [[0.0] * size for _ in range(size)]
        scale = max(1.0, max(abs(hessian[i][i]) for i in range(size)))
        floor = 1e-14 * scale
        for i in range(size):
            if clock() >= until:
                return None
            for j in range(i + 1):
                value = (0.5 * (hessian[i][j] + hessian[j][i])
                         - sum(result[i][p] * result[j][p]
                               for p in range(j)))
                if i == j:
                    result[i][j] = math.sqrt(max(floor, value))
                else:
                    if result[j][j] <= 0.0:
                        return None
                    result[i][j] = value / result[j][j]
        return result

    def cholesky_solve(factor, right_hand_side):
        size = len(right_hand_side)
        intermediate = [0.0] * size
        for i in range(size):
            intermediate[i] = (
                right_hand_side[i]
                - sum(factor[i][j] * intermediate[j] for j in range(i))
            ) / factor[i][i]
        result = [0.0] * size
        for i in range(size - 1, -1, -1):
            result[i] = (
                intermediate[i]
                - sum(factor[j][i] * result[j]
                      for j in range(i + 1, size))
            ) / factor[i][i]
        return result

    def linear_solve(matrix, right_hand_side, until):
        size = len(right_hand_side)
        if size == 0:
            return []
        augmented = [list(matrix[i]) + [right_hand_side[i]]
                     for i in range(size)]
        scale = max(1.0, max(abs(value)
                             for row in matrix for value in row))
        for column in range(size):
            if clock() >= until:
                return None
            pivot_row = max(range(column, size),
                            key=lambda row: abs(augmented[row][column]))
            if abs(augmented[pivot_row][column]) <= 1e-13 * scale:
                return None
            if pivot_row != column:
                augmented[column], augmented[pivot_row] = (
                    augmented[pivot_row], augmented[column]
                )
            inverse = 1.0 / augmented[column][column]
            for j in range(column, size + 1):
                augmented[column][j] *= inverse
            for row in range(size):
                if row == column:
                    continue
                multiplier = augmented[row][column]
                if multiplier != 0.0:
                    for j in range(column, size + 1):
                        augmented[row][j] -= multiplier * augmented[column][j]
        return [augmented[i][size] for i in range(size)]

    def equality_rows(support, lows=None, highs=None):
        size = len(support)
        candidates = [[1.0] * size]
        if lows is not None and highs is not None:
            for j in range(size):
                scale = 1.0 + abs(lows[j]) + abs(highs[j])
                if abs(highs[j] - lows[j]) <= 1e-12 * scale:
                    row = [0.0] * size
                    row[j] = 1.0
                    candidates.append(row)
        exact_rules = set()
        for r in range(rule_count):
            scale = 1.0 + abs(rule_lower[r]) + abs(rule_upper[r])
            if (math.isfinite(rule_lower[r])
                    and math.isfinite(rule_upper[r])
                    and abs(rule_upper[r] - rule_lower[r]) <= 1e-11 * scale):
                candidates.append([rules[r][asset] for asset in support])
                exact_rules.add(r)
        basis = []
        for candidate in candidates:
            vector = list(candidate)
            for existing in basis:
                coefficient = sum(vector[j] * existing[j]
                                  for j in range(size))
                for j in range(size):
                    vector[j] -= coefficient * existing[j]
            norm = math.sqrt(sum(value * value for value in vector))
            if norm > 1e-11:
                basis.append([value / norm for value in vector])
        return basis, exact_rules

    def active_optimize(support, initial, hessian, lows, highs, until):
        size = len(support)
        if size == 0 or clock() >= until:
            return list(initial)
        factor = cholesky(hessian, until)
        if factor is None:
            return list(initial)

        equalities, exact_rules = equality_rows(support, lows, highs)
        inequalities = []

        def add_inequality(normal, right_hand_side):
            norm = math.sqrt(sum(value * value for value in normal))
            if norm > 1e-15:
                inequalities.append(
                    ([value / norm for value in normal], right_hand_side / norm)
                )

        for j in range(size):
            scale = 1.0 + abs(lows[j]) + abs(highs[j])
            if abs(highs[j] - lows[j]) <= 1e-12 * scale:
                continue
            lower_normal = [0.0] * size
            lower_normal[j] = -1.0
            add_inequality(lower_normal, -lows[j])
            upper_normal = [0.0] * size
            upper_normal[j] = 1.0
            add_inequality(upper_normal, highs[j])
        for r in range(rule_count):
            if r in exact_rules:
                continue
            normal = [rules[r][asset] for asset in support]
            if math.isfinite(rule_upper[r]):
                add_inequality(normal, rule_upper[r])
            if math.isfinite(rule_lower[r]):
                add_inequality([-value for value in normal], -rule_lower[r])

        values = list(initial)
        best = list(values)
        best_value = objective(support, values, hessian)
        active = []

        def independent(candidate, active_ids):
            vector = list(inequalities[candidate][0])
            basis = [list(row) for row in equalities]
            for index in active_ids:
                trial = list(inequalities[index][0])
                for row in basis:
                    coefficient = sum(trial[j] * row[j] for j in range(size))
                    for j in range(size):
                        trial[j] -= coefficient * row[j]
                norm = math.sqrt(sum(value * value for value in trial))
                if norm > 1e-9:
                    basis.append([value / norm for value in trial])
            for row in basis:
                coefficient = sum(vector[j] * row[j] for j in range(size))
                for j in range(size):
                    vector[j] -= coefficient * row[j]
            return sum(value * value for value in vector) > 1e-18

        tight_order = sorted(
            range(len(inequalities)),
            key=lambda index: (
                inequalities[index][1]
                - sum(inequalities[index][0][j] * values[j]
                      for j in range(size))
            ),
        )
        for index in tight_order:
            slack = (inequalities[index][1]
                     - sum(inequalities[index][0][j] * values[j]
                           for j in range(size)))
            if slack > 3e-8:
                break
            if independent(index, active):
                active.append(index)

        while clock() < until:
            gradient = [
                sum(hessian[i][j] * values[j] for j in range(size))
                - return_weight * expected_returns[support[i]]
                for i in range(size)
            ]
            unconstrained = cholesky_solve(factor, [-value for value in gradient])
            normals = equalities + [inequalities[index][0] for index in active]
            inverse_normals = [cholesky_solve(factor, normal)
                               for normal in normals]
            gram = [
                [sum(normals[i][j] * inverse_normals[h][j]
                     for j in range(size))
                 for h in range(len(normals))]
                for i in range(len(normals))
            ]
            right_hand_side = [
                sum(normal[j] * unconstrained[j] for j in range(size))
                for normal in normals
            ]
            multipliers = linear_solve(gram, right_hand_side, until)
            if multipliers is None:
                if clock() >= until:
                    break
                if active:
                    active.pop()
                    continue
                break
            direction = [
                unconstrained[j]
                - sum(inverse_normals[h][j] * multipliers[h]
                      for h in range(len(normals)))
                for j in range(size)
            ]
            direction_norm = math.sqrt(sum(value * value for value in direction))
            if direction_norm <= 2e-11 * (
                    1.0 + math.sqrt(sum(value * value for value in values))):
                if not active:
                    break
                offset = len(equalities)
                worst = min(range(len(active)),
                            key=lambda position: multipliers[offset + position])
                if multipliers[offset + worst] >= -2e-10:
                    break
                active.pop(worst)
                continue

            directional_derivative = sum(gradient[j] * direction[j]
                                         for j in range(size))
            if directional_derivative >= -1e-13 * (1.0 + abs(best_value)):
                break
            step = 1.0
            blocker = None
            active_set = set(active)
            for index, (normal, right_hand_side_value) in enumerate(inequalities):
                if index in active_set:
                    continue
                movement = sum(normal[j] * direction[j] for j in range(size))
                if movement > 1e-13:
                    slack = (right_hand_side_value
                             - sum(normal[j] * values[j] for j in range(size)))
                    candidate_step = max(0.0, slack) / movement
                    if (candidate_step < step - 1e-13
                            or (abs(candidate_step - step) <= 1e-13
                                and (blocker is None or index < blocker))):
                        step = candidate_step
                        blocker = index
            candidate = [values[j] + step * direction[j] for j in range(size)]
            difference = 1.0 - sum(candidate)
            anchor = max(
                range(size),
                key=lambda j: min(candidate[j] - lows[j],
                                  highs[j] - candidate[j]),
            )
            candidate[anchor] += difference
            if residual(support, candidate, lows, highs) > 5e-7:
                break
            candidate_value = objective(support, candidate, hessian)
            if candidate_value > best_value + 1e-10 * (1.0 + abs(best_value)):
                break
            values = candidate
            if candidate_value < best_value - 1e-14:
                best = list(candidate)
                best_value = candidate_value
            if blocker is not None and step < 1.0 - 1e-12:
                if independent(blocker, active):
                    active.append(blocker)
                elif step <= 1e-13:
                    break
        return best

    def null_vector(matrix):
        if not matrix or not matrix[0]:
            return None
        column_count = len(matrix[0])
        permutation = list(range(column_count))
        rng.shuffle(permutation)
        work = [[row[j] for j in permutation] for row in matrix]
        row_count = len(work)
        pivot_columns = []
        pivot_row = 0
        for column in range(column_count):
            best_row = -1
            best_value = 0.0
            for row in range(pivot_row, row_count):
                value = abs(work[row][column])
                if value > best_value:
                    best_value = value
                    best_row = row
            if best_row < 0 or best_value <= 1e-11:
                continue
            work[pivot_row], work[best_row] = work[best_row], work[pivot_row]
            pivot = work[pivot_row][column]
            work[pivot_row] = [value / pivot for value in work[pivot_row]]
            for row in range(row_count):
                if row == pivot_row:
                    continue
                multiplier = work[row][column]
                if abs(multiplier) > 1e-13:
                    work[row] = [
                        work[row][j] - multiplier * work[pivot_row][j]
                        for j in range(column_count)
                    ]
            pivot_columns.append(column)
            pivot_row += 1
            if pivot_row == row_count:
                break
        pivot_set = set(pivot_columns)
        free_column = None
        for column in range(column_count - 1, -1, -1):
            if column not in pivot_set:
                free_column = column
                break
        if free_column is None:
            return None
        vector = [0.0] * column_count
        vector[free_column] = 1.0
        for row in range(len(pivot_columns) - 1, -1, -1):
            column = pivot_columns[row]
            vector[column] = -sum(
                work[row][j] * vector[j]
                for j in range(column + 1, column_count)
            )
        result = [0.0] * column_count
        for j, original in enumerate(permutation):
            result[original] = vector[j]
        return result

    def directional_polish(support, initial, hessian, lows, highs,
                           until, strength):
        size = len(support)
        if size < 2 or clock() >= until:
            return list(initial)
        values = list(initial)
        gradient = []
        for i in range(size):
            if clock() >= until:
                return values
            gradient.append(
                sum(hessian[i][j] * values[j] for j in range(size))
                - return_weight * expected_returns[support[i]]
            )
        rule_values = [
            sum(rules[r][support[j]] * values[j] for j in range(size))
            for r in range(rule_count)
        ]
        exact_rows = [[1.0] * size]
        for r in range(rule_count):
            scale = 1.0 + abs(rule_lower[r]) + abs(rule_upper[r])
            if (math.isfinite(rule_lower[r])
                    and math.isfinite(rule_upper[r])
                    and abs(rule_upper[r] - rule_lower[r]) <= 1e-11 * scale):
                exact_rows.append([rules[r][asset] for asset in support])
        movable = [j for j in range(size)
                   if highs[j] - lows[j] > 1e-13]
        if len(movable) < 2:
            return values

        while clock() < until:
            use_general = (len(exact_rows) > 1
                           or rng.random() < min(
                               0.55, 0.12 + 0.05 * math.log1p(strength)
                           ))
            if use_general:
                requested = (len(exact_rows) + 1
                             + int(1.5 * math.sqrt(math.log1p(strength))))
                count = min(len(movable), max(3, requested))
                chosen_positions = rng.sample(movable, count)
                matrix = [[row[j] for j in chosen_positions]
                          for row in exact_rows]
                local_direction = null_vector(matrix)
                if local_direction is None:
                    continue
            else:
                chosen_positions = rng.sample(movable, 2)
                local_direction = [1.0, -1.0]

            magnitude = max(abs(value) for value in local_direction)
            if magnitude <= 1e-14:
                continue
            local_direction = [value / magnitude for value in local_direction]
            step_low = -float("inf")
            step_high = float("inf")
            for j, direction_value in zip(chosen_positions, local_direction):
                if abs(direction_value) <= 1e-14:
                    continue
                first = (lows[j] - values[j]) / direction_value
                second = (highs[j] - values[j]) / direction_value
                if first > second:
                    first, second = second, first
                step_low = max(step_low, first)
                step_high = min(step_high, second)

            rule_deltas = []
            for r in range(rule_count):
                delta = sum(rules[r][support[j]] * direction_value
                            for j, direction_value
                            in zip(chosen_positions, local_direction))
                rule_deltas.append(delta)
                if abs(delta) <= 1e-14:
                    continue
                first = (rule_lower[r] - rule_values[r]) / delta
                second = (rule_upper[r] - rule_values[r]) / delta
                if first > second:
                    first, second = second, first
                step_low = max(step_low, first)
                step_high = min(step_high, second)
            if step_low > step_high + 1e-13:
                continue

            directional_gradient = sum(
                gradient[j] * direction_value
                for j, direction_value in zip(chosen_positions, local_direction)
            )
            curvature = 0.0
            for a, position_a in enumerate(chosen_positions):
                for b, position_b in enumerate(chosen_positions):
                    curvature += (local_direction[a]
                                  * hessian[position_a][position_b]
                                  * local_direction[b])
            if curvature <= 1e-18:
                trial = step_low if directional_gradient > 0.0 else step_high
            else:
                trial = min(step_high, max(step_low,
                                           -directional_gradient / curvature))
            predicted = (directional_gradient * trial
                         + 0.5 * curvature * trial * trial)
            if not math.isfinite(trial) or predicted >= -1e-15:
                continue

            hessian_direction = []
            interrupted = False
            for i in range(size):
                if (i & 15) == 0 and clock() >= until:
                    interrupted = True
                    break
                hessian_direction.append(sum(
                    hessian[i][j] * direction_value
                    for j, direction_value
                    in zip(chosen_positions, local_direction)
                ))
            if interrupted:
                break
            for j, direction_value in zip(chosen_positions, local_direction):
                values[j] += trial * direction_value
            for i in range(size):
                gradient[i] += trial * hessian_direction[i]
            for r in range(rule_count):
                rule_values[r] += trial * rule_deltas[r]
        return values

    def relaxed_polish(initial, until, strength):
        values = list(initial)
        if n < 2 or clock() >= until:
            return values
        rule_values = [sum(rules[r][i] * values[i] for i in range(n))
                       for r in range(rule_count)]
        exact_rows = [[1.0] * n]
        for r in range(rule_count):
            scale = 1.0 + abs(rule_lower[r]) + abs(rule_upper[r])
            if (math.isfinite(rule_lower[r])
                    and math.isfinite(rule_upper[r])
                    and abs(rule_upper[r] - rule_lower[r]) <= 1e-11 * scale):
                exact_rows.append(list(rules[r]))
        movable = [i for i in range(n) if maximum[i] > 1e-13]
        if len(movable) < 2:
            return values

        if risk_kind == "direct":
            active = [i for i, value in enumerate(values) if value != 0.0]
            gradient = []
            for i in range(n):
                if (i & 7) == 0 and clock() >= until:
                    return values
                gradient.append(
                    sum(covariance[i][j] * values[j] for j in active)
                    + inverse_theta * values[i]
                    - return_weight * expected_returns[i]
                )
            exposures = None
        else:
            gradient = None
            exposures = [sum(row[i] * values[i] for i in range(n))
                         for row in loadings]

        while clock() < until:
            use_general = (len(exact_rows) > 1
                           or rng.random() < min(
                               0.55, 0.10 + 0.045 * math.log1p(strength)
                           ))
            if use_general:
                requested = (len(exact_rows) + 1
                             + int(1.5 * math.sqrt(math.log1p(strength))))
                count = min(len(movable), max(3, requested))
                assets = rng.sample(movable, count)
                matrix = [[row[asset] for asset in assets]
                          for row in exact_rows]
                direction = null_vector(matrix)
                if direction is None:
                    continue
            else:
                assets = rng.sample(movable, 2)
                direction = [1.0, -1.0]
            magnitude = max(abs(value) for value in direction)
            if magnitude <= 1e-14:
                continue
            direction = [value / magnitude for value in direction]

            step_low = -float("inf")
            step_high = float("inf")
            for asset, direction_value in zip(assets, direction):
                if abs(direction_value) <= 1e-14:
                    continue
                first = -values[asset] / direction_value
                second = (maximum[asset] - values[asset]) / direction_value
                if first > second:
                    first, second = second, first
                step_low = max(step_low, first)
                step_high = min(step_high, second)

            rule_deltas = []
            for r in range(rule_count):
                delta = sum(rules[r][asset] * direction_value
                            for asset, direction_value in zip(assets, direction))
                rule_deltas.append(delta)
                if abs(delta) <= 1e-14:
                    continue
                first = (rule_lower[r] - rule_values[r]) / delta
                second = (rule_upper[r] - rule_values[r]) / delta
                if first > second:
                    first, second = second, first
                step_low = max(step_low, first)
                step_high = min(step_high, second)
            if step_low > step_high + 1e-13:
                continue

            if risk_kind == "direct":
                directional_gradient = sum(
                    gradient[asset] * direction_value
                    for asset, direction_value in zip(assets, direction)
                )
                curvature = inverse_theta * sum(value * value
                                                  for value in direction)
                for a, asset_a in enumerate(assets):
                    for b, asset_b in enumerate(assets):
                        curvature += (direction[a] * covariance[asset_a][asset_b]
                                      * direction[b])
                loading_direction = None
            else:
                local_gradient = []
                for asset in assets:
                    value = sum(loadings[f][asset] * exposures[f]
                                for f in range(len(loadings)))
                    value += ((idiosyncratic[asset] + inverse_theta)
                              * values[asset])
                    value -= return_weight * expected_returns[asset]
                    local_gradient.append(value)
                directional_gradient = sum(
                    local_gradient[j] * direction[j]
                    for j in range(len(assets))
                )
                loading_direction = [
                    sum(loadings[f][asset] * direction_value
                        for asset, direction_value in zip(assets, direction))
                    for f in range(len(loadings))
                ]
                curvature = sum(value * value for value in loading_direction)
                curvature += sum(
                    (idiosyncratic[asset] + inverse_theta)
                    * direction_value * direction_value
                    for asset, direction_value in zip(assets, direction)
                )

            if curvature <= 1e-18:
                trial = step_low if directional_gradient > 0.0 else step_high
            else:
                trial = min(step_high, max(step_low,
                                           -directional_gradient / curvature))
            predicted = (directional_gradient * trial
                         + 0.5 * curvature * trial * trial)
            if not math.isfinite(trial) or predicted >= -1e-15:
                continue

            if risk_kind == "direct":
                changes = []
                for i in range(n):
                    changes.append(sum(
                        covariance[i][asset] * direction_value
                        for asset, direction_value in zip(assets, direction)
                    ))
                direction_map = {asset: direction_value
                                 for asset, direction_value in zip(assets, direction)}
                for i in range(n):
                    gradient[i] += trial * (
                        changes[i] + inverse_theta * direction_map.get(i, 0.0)
                    )
            else:
                for f in range(len(exposures)):
                    exposures[f] += trial * loading_direction[f]
            for asset, direction_value in zip(assets, direction):
                values[asset] += trial * direction_value
            for r in range(rule_count):
                rule_values[r] += trial * rule_deltas[r]
        return values

    def sparsify_relaxed(relaxed, until, strength):
        values = list(relaxed)
        all_rows = [[1.0] * n] + [list(row) for row in rules]
        while clock() < until:
            active = [i for i, value in enumerate(values) if value > 1e-10]
            if len(active) <= k_limit:
                return tuple(sorted(active))
            requested = (len(all_rows) + 1
                         + int(math.sqrt(math.log1p(strength))))
            count = min(len(active), max(3, requested))
            chosen = rng.sample(active, count)
            matrix = [[row[asset] for asset in chosen] for row in all_rows]
            direction = null_vector(matrix)
            if direction is None:
                break
            magnitude = max(abs(value) for value in direction)
            if magnitude <= 1e-14:
                continue
            direction = [value / magnitude for value in direction]
            low = -float("inf")
            high = float("inf")
            for asset, direction_value in zip(chosen, direction):
                if abs(direction_value) <= 1e-14:
                    continue
                first = -values[asset] / direction_value
                second = (maximum[asset] - values[asset]) / direction_value
                if first > second:
                    first, second = second, first
                low = max(low, first)
                high = min(high, second)
            choices = [step for step in (low, high) if math.isfinite(step)]
            if not choices:
                break
            choices.sort(
                key=lambda step: (
                    sum(1 for asset, direction_value in zip(chosen, direction)
                        if values[asset] + step * direction_value <= 1e-9),
                    -sum(base_score[asset] * step * direction_value
                         for asset, direction_value in zip(chosen, direction)),
                ),
                reverse=True,
            )
            step = choices[0]
            if abs(step) <= 1e-14:
                continue
            for asset, direction_value in zip(chosen, direction):
                value = values[asset] + step * direction_value
                if abs(value) <= 1e-9:
                    value = 0.0
                elif abs(value - maximum[asset]) <= 1e-9:
                    value = maximum[asset]
                values[asset] = value
        active = [i for i, value in enumerate(values) if value > 1e-10]
        active.sort(key=lambda i: (-values[i], base_score[i], i))
        return tuple(sorted(active[:k_limit]))

    def repair_support(items):
        support = []
        present = set()
        for asset in items:
            asset = int(asset)
            if 0 <= asset < n and asset not in present:
                support.append(asset)
                present.add(asset)
                if len(support) == k_limit:
                    break
        if not support:
            support = [upper_order[0]]
            present = set(support)
        while (len(support) > 1
               and sum(minimum[i] for i in support) > 1.0 + 1e-10):
            removed = max(support,
                          key=lambda i: (minimum[i], base_rank[i]))
            support.remove(removed)
            present.remove(removed)
        while sum(maximum[i] for i in support) < 1.0 - 1e-10:
            outside = [i for i in upper_order if i not in present]
            if not outside:
                break
            addition = outside[0]
            if len(support) < k_limit:
                support.append(addition)
                present.add(addition)
            else:
                removed = min(support,
                              key=lambda i: (maximum[i], -base_rank[i]))
                if maximum[addition] <= maximum[removed] + 1e-15:
                    break
                support.remove(removed)
                present.remove(removed)
                support.append(addition)
                present.add(addition)
        return tuple(sorted(support))

    def weighted_support(weights, size, exploration):
        keys = []
        baseline = max(1e-15, exploration / max(1, n))
        for i, weight in enumerate(weights):
            adjusted = max(1e-15, weight) + baseline
            key = math.log(max(1e-300, rng.random())) / adjusted
            keys.append((key, i))
        keys.sort(reverse=True)
        return repair_support([asset for _, asset in keys[:size]])

    def all_gradient(support, values, until):
        held = {asset: values[j] for j, asset in enumerate(support)}
        result = [0.0] * n
        if risk_kind == "direct":
            for i in range(n):
                if (i & 7) == 0 and clock() >= until:
                    return None
                result[i] = sum(covariance[i][asset] * values[j]
                                for j, asset in enumerate(support))
                result[i] += inverse_theta * held.get(i, 0.0)
                result[i] -= return_weight * expected_returns[i]
        else:
            exposures = [sum(row[asset] * values[j]
                             for j, asset in enumerate(support))
                         for row in loadings]
            for i in range(n):
                if (i & 15) == 0 and clock() >= until:
                    return None
                result[i] = sum(loadings[f][i] * exposures[f]
                                for f in range(len(loadings)))
                result[i] += ((idiosyncratic[i] + inverse_theta)
                              * held.get(i, 0.0))
                result[i] -= return_weight * expected_returns[i]
        return result

    state = {
        "best_support": None,
        "best_values": None,
        "best_hessian": None,
        "best_objective": float("inf"),
        "last_feasible": None,
        "elite": {},
        "last_improvement": started,
        "neighbor_heap": [],
        "outside_order": None,
        "neighbor_width": 0,
        "needs_neighbors": False,
        "relaxed": None,
        "pending": [],
    }
    seen = set()

    def consider(support, values, hessian):
        if residual(support, values) > feasibility_tolerance:
            return False
        value = objective(support, values, hessian)
        if not math.isfinite(value):
            return False
        key = tuple(support)
        old = state["elite"].get(key)
        if old is None or value < old[0] - 1e-14:
            state["elite"][key] = (value, list(values))
        elapsed = max(0.0, clock() - started)
        capacity = 8 + int(4.0 * math.sqrt(elapsed + 1.0))
        if len(state["elite"]) > capacity:
            state["elite"] = dict(sorted(
                state["elite"].items(), key=lambda item: item[1][0]
            )[:capacity])
        if value < state["best_objective"] - 1e-13 * (1.0 + abs(value)):
            state["best_support"] = key
            state["best_values"] = list(values)
            state["best_hessian"] = hessian
            state["best_objective"] = value
            state["last_improvement"] = clock()
            state["needs_neighbors"] = True
            return True
        return False

    def rebuild_neighbors(until):
        support = state["best_support"]
        values = state["best_values"]
        if support is None or clock() >= until:
            return
        gradient = all_gradient(support, values, until)
        if gradient is None:
            return
        elapsed = max(0.0, clock() - started)
        desired_width = min(
            n,
            max(k_limit + 8,
                int((k_limit + 1) * (3.0 + math.sqrt(elapsed + 1.0)))),
        )
        members = list(support)
        member_set = set(members)
        outside = [i for i in range(n) if i not in member_set]
        outside.sort(key=lambda i: (gradient[i], base_score[i], i))
        state["outside_order"] = outside
        outside = outside[:desired_width]
        heap = []
        unique = set()
        if len(members) < k_limit:
            for addition in outside:
                if clock() >= until:
                    break
                candidate = tuple(sorted(members + [addition]))
                if candidate not in unique:
                    unique.add(candidate)
                    heapq.heappush(heap, (gradient[addition], candidate))
        if len(members) > 1:
            for removal in members:
                candidate = tuple(asset for asset in support if asset != removal)
                if candidate not in unique:
                    unique.add(candidate)
                    heapq.heappush(heap, (-gradient[removal], candidate))
        interrupted = False
        for removal in members:
            if interrupted:
                break
            for addition in outside:
                if clock() >= until:
                    interrupted = True
                    break
                candidate = tuple(sorted(
                    [asset for asset in members if asset != removal] + [addition]
                ))
                if candidate not in unique:
                    unique.add(candidate)
                    score = gradient[addition] - gradient[removal]
                    heapq.heappush(heap, (score, candidate))
        state["neighbor_heap"] = heap
        state["neighbor_width"] = desired_width
        state["needs_neighbors"] = False

    def process_support(candidate):
        support = tuple(sorted(set(candidate)))
        if (not support or len(support) > k_limit
                or support in seen or clock() >= search_end):
            return
        seen.add(support)
        if not support_possible(support):
            return
        lows = [minimum[i] for i in support]
        highs = [maximum[i] for i in support]
        seed_values = None
        if state["relaxed"] is not None:
            seed_values = [state["relaxed"][i] for i in support]

        elapsed = max(0.0, clock() - started)
        feasibility_slice = (0.012 + 0.006 * math.log1p(elapsed)
                             + (0.12 if state["best_support"] is None else 0.0))
        feasibility_end = min(search_end, clock() + feasibility_slice)
        projection_end = min(
            feasibility_end,
            clock() + max(0.003, 0.35 * feasibility_slice),
        )
        values = projected_feasible(
            support, lows, highs, seed_values, projection_end
        )
        if values is None and clock() < feasibility_end:
            values = lp_feasible(support, lows, highs, feasibility_end)
        if values is None:
            return
        state["last_feasible"] = (support, list(values))

        promising = state["best_support"] is None
        hessian_slice = (0.012 + 0.004 * math.log1p(elapsed)
                         + (0.08 if promising else 0.0))
        hessian_end = min(search_end, clock() + hessian_slice)
        hessian = build_hessian(support, hessian_end)
        if hessian is None:
            return
        consider(support, values, hessian)

        raw_value = objective(support, values, hessian)
        promising = (state["best_support"] is None
                     or raw_value <= state["best_objective"]
                     + 0.05 * (1.0 + abs(state["best_objective"])))
        optimization_slice = (0.008 + 0.004 * math.log1p(elapsed)
                              + (0.025 if promising else 0.0))
        optimization_end = min(search_end, clock() + optimization_slice)
        first_polish_end = min(
            optimization_end, clock() + 0.25 * optimization_slice
        )
        polished = directional_polish(
            support, values, hessian, lows, highs,
            first_polish_end, elapsed + 1.0,
        )
        if residual(support, polished, lows, highs) <= 5e-7:
            values = polished
            consider(support, values, hessian)
        if clock() < optimization_end:
            optimized = active_optimize(
                support, values, hessian, lows, highs, optimization_end
            )
            if residual(support, optimized, lows, highs) <= 5e-7:
                values = optimized
                consider(support, values, hessian)

    seed_supports = [
        repair_support(base_order[:k_limit]),
        repair_support(return_order[:k_limit]),
        repair_support(risk_order[:k_limit]),
        repair_support(upper_order[:k_limit]),
    ]
    if n <= k_limit:
        seed_supports.insert(0, tuple(range(n)))
    for asset in base_order:
        if maximum[asset] >= 1.0 - 1e-12 and minimum[asset] <= 1.0 + 1e-12:
            seed_supports.append((asset,))
            if len(seed_supports) >= 8:
                break

    enum_sizes = list(range(k_limit, 0, -1))
    enum_iterators = {}
    enum_finished = set()
    enum_position = 0

    def next_enumerated():
        nonlocal enum_position
        if len(enum_finished) == len(enum_sizes):
            return None
        checked = 0
        while checked < len(enum_sizes):
            size = enum_sizes[enum_position % len(enum_sizes)]
            enum_position += 1
            checked += 1
            if size in enum_finished:
                continue
            iterator = enum_iterators.get(size)
            if iterator is None:
                iterator = itertools.combinations(tuple(base_order), size)
                enum_iterators[size] = iterator
            try:
                return tuple(sorted(next(iterator)))
            except StopIteration:
                enum_finished.add(size)
        return None

    rule_seed_position = 0

    def next_rule_seed():
        nonlocal rule_seed_position
        if rule_count == 0 or rule_seed_position >= 4 * rule_count:
            return None
        rule = rule_seed_position // 4
        variant = rule_seed_position % 4
        rule_seed_position += 1
        high = sorted(range(n),
                      key=lambda i: (rules[rule][i], -base_score[i]),
                      reverse=True)
        low = list(reversed(high))
        mix = max(1, k_limit // 2)
        if variant == 0:
            return repair_support(high[:k_limit])
        if variant == 1:
            return repair_support(low[:k_limit])
        if variant == 2:
            return repair_support(high[:mix] + base_order)
        return repair_support(low[:mix] + base_order)

    def refine_relaxation(elapsed):
        if clock() >= search_end:
            return None
        if state["relaxed"] is None:
            all_assets = tuple(range(n))
            zero_lows = [0.0] * n
            slice_length = 0.03 + 0.015 * math.log1p(elapsed + 1.0)
            if state["best_support"] is None:
                slice_length += 0.12
            until = min(search_end, clock() + slice_length)
            projection_end = min(until, clock() + 0.45 * slice_length)
            dense = projected_feasible(
                all_assets, zero_lows, maximum, None, projection_end
            )
            if dense is None and clock() < until:
                dense = lp_feasible(all_assets, zero_lows, maximum, until)
            if dense is None:
                return None
            state["relaxed"] = dense
        polish_slice = 0.008 + 0.006 * math.log1p(elapsed + 1.0)
        polish_end = min(search_end, clock() + polish_slice)
        polished = relaxed_polish(
            state["relaxed"], polish_end, elapsed + 1.0
        )
        if residual(tuple(range(n)), polished, [0.0] * n, maximum) <= 5e-7:
            state["relaxed"] = polished
        top = sorted(range(n),
                     key=lambda i: (-state["relaxed"][i], base_score[i], i))
        return repair_support(top[:k_limit] + base_order)

    def random_candidate(elapsed):
        elites = list(state["elite"].items())
        if (len(elites) >= 2
                and rng.random() < min(0.38,
                                       0.05 + math.log1p(elapsed) / 24.0)):
            elites.sort(key=lambda item: item[1][0])
            bound = min(len(elites), 2 + int(math.sqrt(elapsed + 1.0)))
            first = elites[rng.randrange(bound)][0]
            second = elites[rng.randrange(bound)][0]
            common = list(set(first).intersection(second))
            pool = list(set(first).union(second))
            rng.shuffle(pool)
            return repair_support(common + pool + base_order)

        if elites:
            elites.sort(key=lambda item: item[1][0])
            bound = min(len(elites), 2 + int(math.sqrt(elapsed + 1.0)))
            support, data = elites[rng.randrange(max(1, bound))]
            old_values = data[1]
            members = list(support)
            stagnation = max(0.0, clock() - state["last_improvement"])
            radius_limit = min(
                len(members),
                1 + int(math.log1p(elapsed + stagnation * stagnation + 1.0)),
            )
            radius = rng.randint(1, max(1, radius_limit))
            removal_positions = sorted(
                range(len(members)),
                key=lambda j: (old_values[j], rng.random()),
            )[:radius]
            remove_set = set(removal_positions)
            retained = [asset for j, asset in enumerate(members)
                        if j not in remove_set]
            present = set(retained)
            preferred = state["outside_order"] or base_order
            exploration = min(0.82, 0.08 + math.log1p(elapsed + 1.0) / 11.0)
            target_size = min(k_limit, max(1, len(members)))
            while len(retained) < target_size:
                if rng.random() < exploration:
                    choices = [i for i in range(n) if i not in present]
                else:
                    width = min(
                        n,
                        max(k_limit + radius,
                            int((k_limit + 1)
                                * (1.0 + math.sqrt(elapsed + 1.0)))),
                    )
                    choices = [i for i in preferred[:width] if i not in present]
                    if not choices:
                        choices = [i for i in range(n) if i not in present]
                if not choices:
                    break
                addition = choices[rng.randrange(len(choices))]
                retained.append(addition)
                present.add(addition)
            return repair_support(retained)

        if state["relaxed"] is not None and rng.random() < 0.7:
            if k_limit == 1:
                size = 1
            else:
                spread = min(k_limit - 1,
                             int(math.log1p(elapsed + 1.0) * rng.random()))
                size = max(1, k_limit - spread)
            return weighted_support(
                state["relaxed"], size,
                min(1.0, 0.02 + math.log1p(elapsed + 1.0) / 20.0),
            )

        exploration = min(0.85, math.log1p(elapsed + 1.0) / 9.0)
        if k_limit == 1:
            size = 1
        elif rng.random() < exploration:
            size = rng.randint(1, k_limit)
        else:
            size = max(1, k_limit - int((rng.random() ** 3) * k_limit))
        chosen = rng.sample(range(n), min(size, n))
        if rule_count and rng.random() < 0.65:
            rule = rng.randrange(rule_count)
            reverse = rng.random() < 0.5
            extreme = sorted(range(n), key=lambda i: rules[rule][i],
                             reverse=reverse)
            chosen = extreme[:max(1, size // 3)] + chosen
        return repair_support(chosen)

    seed_index = 0
    attempt = 0
    while clock() < search_end:
        elapsed = max(0.0, clock() - started)
        candidate = None

        desired_width = min(
            n,
            max(k_limit + 8,
                int((k_limit + 1) * (3.0 + math.sqrt(elapsed + 1.0)))),
        )
        if (state["best_support"] is not None
                and desired_width > max(1, int(state["neighbor_width"] * 1.25))):
            state["needs_neighbors"] = True
        if state["needs_neighbors"]:
            neighbor_slice = 0.01 + 0.006 * math.log1p(elapsed + 1.0)
            rebuild_neighbors(min(search_end, clock() + neighbor_slice))

        if state["pending"]:
            candidate = state["pending"].pop()
        elif seed_index < len(seed_supports):
            candidate = seed_supports[seed_index]
            seed_index += 1
        else:
            mode = attempt % 16
            if mode in (0, 8):
                candidate = next_enumerated()
            elif mode in (1, 6, 10) and state["neighbor_heap"]:
                candidate = heapq.heappop(state["neighbor_heap"])[1]
            elif mode == 2:
                candidate = next_rule_seed()
            elif mode == 3 and state["relaxed"] is not None:
                if k_limit == 1:
                    size = 1
                else:
                    depth = min(k_limit - 1,
                                int(math.sqrt(math.log1p(elapsed + 1.0))
                                    * rng.random()))
                    size = max(1, k_limit - depth)
                candidate = weighted_support(
                    state["relaxed"], size,
                    min(1.0, 0.03 + math.log1p(elapsed + 1.0) / 18.0),
                )
            elif mode == 4 and state["relaxed"] is not None:
                sparse_slice = 0.006 + 0.004 * math.log1p(elapsed + 1.0)
                candidate = sparsify_relaxed(
                    state["relaxed"],
                    min(search_end, clock() + sparse_slice),
                    elapsed + 1.0,
                )
                candidate = repair_support(candidate)
            elif mode in (5, 13):
                candidate = refine_relaxation(elapsed)
            elif mode == 11 and state["best_support"] is not None:
                support = state["best_support"]
                lows = [minimum[i] for i in support]
                highs = [maximum[i] for i in support]
                slice_length = 0.006 + 0.004 * math.log1p(elapsed + 1.0)
                until = min(search_end, clock() + slice_length)
                polished = directional_polish(
                    support, state["best_values"], state["best_hessian"],
                    lows, highs, until, elapsed + 1.0,
                )
                if residual(support, polished, lows, highs) <= 5e-7:
                    consider(support, polished, state["best_hessian"])
            else:
                candidate = random_candidate(elapsed)

        attempt += 1
        if candidate is not None:
            process_support(candidate)

    def clean(support, values):
        cleaned = [0.0 if -1e-10 < value < 0.0 else float(value)
                   for value in values]
        if any(not math.isfinite(value) or value < 0.0 for value in cleaned):
            return None
        difference = 1.0 - sum(cleaned)
        if difference != 0.0:
            order = sorted(
                range(len(support)),
                key=lambda j: min(cleaned[j] - minimum[support[j]],
                                  maximum[support[j]] - cleaned[j]),
                reverse=True,
            )
            repaired = None
            for j in order:
                trial = list(cleaned)
                trial[j] += difference
                if residual(support, trial) <= 8e-7:
                    repaired = trial
                    break
            if repaired is not None:
                cleaned = repaired
        if residual(support, cleaned) > 8e-7:
            return None
        weights = [0.0] * n
        for j, asset in enumerate(support):
            weights[asset] = max(0.0, cleaned[j])
        selected = [1 if value > 0.0 else 0 for value in weights]
        if sum(selected) > k_limit:
            return None
        return {"weights": weights, "selected": selected}

    candidates = []
    if state["best_support"] is not None:
        candidates.append((state["best_objective"],
                           state["best_support"], state["best_values"]))
    for support, data in state["elite"].items():
        candidates.append((data[0], support, data[1]))
    candidates.sort(key=lambda item: item[0])
    checked = set()
    for _, support, values in candidates:
        if support in checked:
            continue
        checked.add(support)
        answer = clean(support, values)
        if answer is not None:
            return answer
    if state["last_feasible"] is not None:
        support, values = state["last_feasible"]
        answer = clean(support, values)
        if answer is not None:
            return answer
    return _fallback(instance)
