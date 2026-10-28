import heapq
import itertools
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    try:
        return _solve(instance, time_limit_s)
    except BaseException:
        return _emergency(instance)


def _emergency(instance):
    try:
        n = max(1, int(instance.get("num_securities", 1)))
        K = max(1, min(n, int(instance.get("max_securities", n))))
        raw_lo = instance.get("minimum_weights") or []
        raw_hi = instance.get("maximum_weights") or []
        lo = [max(0.0, float(v)) for v in raw_lo] if len(raw_lo) == n else [0.0] * n
        hi = [max(0.0, float(v)) for v in raw_hi] if len(raw_hi) == n else [1.0] * n
        A = instance.get("linear_rule_coefficients") or []
        lb = instance.get("linear_rule_lower_bounds") or []
        ub = instance.get("linear_rule_upper_bounds") or []
        mu = instance.get("expected_returns") or []

        def construct(support):
            support = tuple(dict.fromkeys(i for i in support if 0 <= i < n))
            if not support or len(support) > K:
                return None
            values = [lo[i] for i in support]
            remaining = 1.0 - sum(values)
            capacity = sum(max(0.0, hi[i] - values[j]) for j, i in enumerate(support))
            if remaining < -1e-9 or remaining > capacity + 1e-9:
                return None
            while remaining > 1e-13:
                free = [j for j, i in enumerate(support) if hi[i] - values[j] > 1e-14]
                if not free:
                    return None
                share = remaining / len(free)
                moved = 0.0
                for j in free:
                    addition = min(share, hi[support[j]] - values[j])
                    values[j] += addition
                    moved += addition
                if moved <= 1e-16:
                    return None
                remaining -= moved
            if values:
                values[-1] += 1.0 - sum(values)
            for r, row in enumerate(A):
                value = sum(float(row[i]) * values[j] for j, i in enumerate(support))
                if r < len(lb) and value < float(lb[r]) - 1e-7:
                    return None
                if r < len(ub) and value > float(ub[r]) + 1e-7:
                    return None
            weights = [0.0] * n
            for j, i in enumerate(support):
                weights[i] = max(0.0, values[j])
            return {
                "weights": weights,
                "selected": [1 if value > 0.0 else 0 for value in weights],
            }

        orders = [sorted(range(n), key=lambda i: (-hi[i], i))]
        if len(mu) == n:
            orders.append(sorted(range(n), key=lambda i: (-float(mu[i]), -hi[i], i)))
        for i in range(n):
            answer = construct((i,))
            if answer is not None:
                return answer
        for order in orders:
            for size in range(2, K + 1):
                answer = construct(order[:size])
                if answer is not None:
                    return answer
        weights = [0.0] * n
        asset = max(range(n), key=lambda i: hi[i])
        weights[asset] = 1.0
        return {"weights": weights, "selected": [1 if v > 0.0 else 0 for v in weights]}
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
    reserve = min(0.05, max(0.001, 0.002 * limit)) if limit > 0.0 else 0.0
    search_end = max(started, deadline - reserve)

    n = max(1, int(instance["num_securities"]))
    K = max(1, min(n, int(instance["max_securities"])))
    mu = [float(v) for v in instance["expected_returns"]]
    lam = max(0.0, float(instance["return_weight"]))
    theta = float(instance["concentration_parameter"])
    inv_theta = 1.0 / theta

    A = [[float(v) for v in row] for row in (instance.get("linear_rule_coefficients") or [])]
    nrules = len(A)
    raw_rule_lo = [float(v) for v in (instance.get("linear_rule_lower_bounds") or [])]
    raw_rule_hi = [float(v) for v in (instance.get("linear_rule_upper_bounds") or [])]
    rule_lo = [raw_rule_lo[r] if r < len(raw_rule_lo) else -float("inf") for r in range(nrules)]
    rule_hi = [raw_rule_hi[r] if r < len(raw_rule_hi) else float("inf") for r in range(nrules)]

    raw_lo = instance.get("minimum_weights") or []
    raw_hi = instance.get("maximum_weights") or []
    minimum = [max(0.0, float(v)) for v in raw_lo] if len(raw_lo) == n else [0.0] * n
    maximum = [max(0.0, float(v)) for v in raw_hi] if len(raw_hi) == n else [1.0] * n

    if instance.get("covariance_format", "direct") == "factor":
        B = [[float(v) for v in row] for row in (instance.get("factor_loadings") or [])]
        raw_d = [float(v) for v in (instance.get("idiosyncratic_variances") or [])]
        d = [raw_d[i] if i < len(raw_d) else 0.0 for i in range(n)]
        Q = None
        diagonal = [d[i] + sum(row[i] * row[i] for row in B) for i in range(n)]
    else:
        Q = [[float(v) for v in row] for row in instance["covariance_matrix"]]
        B = None
        d = None
        diagonal = [Q[i][i] for i in range(n)]

    single_score = [0.5 * (diagonal[i] + inv_theta) - lam * mu[i] for i in range(n)]
    base_order = sorted(range(n), key=lambda i: (single_score[i], i))
    return_order = sorted(range(n), key=lambda i: (-mu[i], single_score[i], i))
    risk_order = sorted(range(n), key=lambda i: (diagonal[i], -mu[i], i))
    capacity_order = sorted(range(n), key=lambda i: (-maximum[i], single_score[i], i))
    base_rank = {asset: rank for rank, asset in enumerate(base_order)}

    seed = (n * 1000003 + K * 9176 + nrules * 131) & 0xFFFFFFFFFFFFFFFF
    for i, value in enumerate(mu[:256]):
        if math.isfinite(value):
            seed ^= (int(abs(value) * 1000000007.0) + i * 104729) & 0xFFFFFFFFFFFFFFFF
    rng = random.Random(seed)
    feasibility_tol = 2e-7

    def covariance_entry(i, j):
        if Q is not None:
            return Q[i][j]
        value = sum(row[i] * row[j] for row in B)
        if i == j:
            value += d[i]
        return value

    def hessian_entry(i, j):
        value = covariance_entry(i, j)
        if i == j:
            value += inv_theta
        return value

    def residual(S, x, lows=None, highs=None):
        if len(S) != len(x):
            return float("inf")
        if lows is None:
            lows = [minimum[i] for i in S]
        if highs is None:
            highs = [maximum[i] for i in S]
        result = abs(sum(x) - 1.0)
        for j, value in enumerate(x):
            if not math.isfinite(value):
                return float("inf")
            result = max(result, -value, lows[j] - value, value - highs[j])
        for r in range(nrules):
            value = sum(A[r][asset] * x[j] for j, asset in enumerate(S))
            if math.isfinite(rule_lo[r]):
                result = max(result, rule_lo[r] - value)
            if math.isfinite(rule_hi[r]):
                result = max(result, value - rule_hi[r])
        return max(0.0, result)

    def box_point(lows, highs):
        values = list(lows)
        remaining = 1.0 - sum(values)
        capacity = sum(highs[j] - lows[j] for j in range(len(values)))
        if remaining < -1e-10 or remaining > capacity + 1e-10:
            return None
        while remaining > 1e-14:
            free = [j for j in range(len(values)) if highs[j] - values[j] > 1e-15]
            if not free:
                return None
            share = remaining / len(free)
            moved = 0.0
            for j in free:
                addition = min(share, highs[j] - values[j])
                if addition > 0.0:
                    values[j] += addition
                    moved += addition
            if moved <= 1e-16:
                return None
            remaining -= moved
        delta = 1.0 - sum(values)
        for j in range(len(values)):
            if lows[j] - 1e-12 <= values[j] + delta <= highs[j] + 1e-12:
                values[j] += delta
                break
        return values

    def support_possible(S, lows=None, highs=None):
        if not S or len(S) > K and lows is None:
            return False
        if lows is None:
            lows = [minimum[i] for i in S]
        if highs is None:
            highs = [maximum[i] for i in S]
        remaining = 1.0 - sum(lows)
        caps = [max(0.0, highs[j] - lows[j]) for j in range(len(S))]
        if remaining < -1e-9 or remaining > sum(caps) + 1e-9:
            return False
        for r in range(nrules):
            base = sum(A[r][asset] * lows[j] for j, asset in enumerate(S))
            pairs = sorted((A[r][asset], caps[j]) for j, asset in enumerate(S))
            left = max(0.0, remaining)
            low_value = base
            for coefficient, cap in pairs:
                used = min(left, cap)
                low_value += coefficient * used
                left -= used
                if left <= 1e-14:
                    break
            left = max(0.0, remaining)
            high_value = base
            for coefficient, cap in reversed(pairs):
                used = min(left, cap)
                high_value += coefficient * used
                left -= used
                if left <= 1e-14:
                    break
            if math.isfinite(rule_lo[r]) and high_value < rule_lo[r] - 1e-8:
                return False
            if math.isfinite(rule_hi[r]) and low_value > rule_hi[r] + 1e-8:
                return False
        return True

    def lp_feasible(S, lows, highs, until):
        k = len(S)
        if k == 0 or clock() >= until:
            return None
        if nrules == 0:
            return box_point(lows, highs)
        remaining = 1.0 - sum(lows)
        caps = [highs[j] - lows[j] for j in range(k)]
        if remaining < -1e-9 or remaining > sum(caps) + 1e-9:
            return None

        rows = []
        rhs = []
        for j in range(k):
            row = [0.0] * k
            row[j] = 1.0
            rows.append(row)
            rhs.append(caps[j])
        rows.append([1.0] * k)
        rhs.append(remaining)
        rows.append([-1.0] * k)
        rhs.append(-remaining)
        for r in range(nrules):
            coefficients = [A[r][asset] for asset in S]
            base = sum(coefficients[j] * lows[j] for j in range(k))
            if math.isfinite(rule_hi[r]):
                rows.append(coefficients)
                rhs.append(rule_hi[r] - base)
            if math.isfinite(rule_lo[r]):
                rows.append([-v for v in coefficients])
                rhs.append(-rule_lo[r] + base)

        m = len(rows)
        nv = k
        eps = 1e-10
        D = [[0.0] * (nv + 2) for _ in range(m + 2)]
        basic = [0] * m
        nonbasic = [0] * (nv + 1)
        for i in range(m):
            if clock() >= until:
                return None
            for j in range(nv):
                D[i][j] = rows[i][j]
            basic[i] = nv + i
            D[i][nv] = -1.0
            D[i][nv + 1] = rhs[i]
        for j in range(nv):
            nonbasic[j] = j
        nonbasic[nv] = -1
        D[m + 1][nv] = 1.0

        def pivot(r, s):
            inverse = 1.0 / D[r][s]
            for i in range(m + 2):
                if i == r:
                    continue
                factor = D[i][s] * inverse
                if factor == 0.0:
                    continue
                for j in range(nv + 2):
                    if j != s:
                        D[i][j] -= D[r][j] * factor
            for j in range(nv + 2):
                if j != s:
                    D[r][j] *= inverse
            for i in range(m + 2):
                if i != r:
                    D[i][s] *= -inverse
            D[r][s] = inverse
            basic[r], nonbasic[s] = nonbasic[s], basic[r]

        def simplex(phase):
            objective_row = m + 1 if phase == 1 else m
            while clock() < until:
                entering = -1
                for j in range(nv + 1):
                    if phase == 2 and nonbasic[j] == -1:
                        continue
                    if entering == -1 or D[objective_row][j] < D[objective_row][entering] - eps or (
                        abs(D[objective_row][j] - D[objective_row][entering]) <= eps
                        and nonbasic[j] < nonbasic[entering]
                    ):
                        entering = j
                if entering == -1 or D[objective_row][entering] >= -eps:
                    return True
                leaving = -1
                for i in range(m):
                    if D[i][entering] <= eps:
                        continue
                    if leaving == -1:
                        leaving = i
                    else:
                        left_ratio = D[i][nv + 1] / D[i][entering]
                        right_ratio = D[leaving][nv + 1] / D[leaving][entering]
                        if left_ratio < right_ratio - eps or (
                            abs(left_ratio - right_ratio) <= eps and basic[i] < basic[leaving]
                        ):
                            leaving = i
                if leaving == -1:
                    return False
                pivot(leaving, entering)
            return None

        leaving = min(range(m), key=lambda i: D[i][nv + 1])
        if D[leaving][nv + 1] < -eps:
            pivot(leaving, nv)
            phase = simplex(1)
            if phase is not True or D[m + 1][nv + 1] < -eps:
                return None
            if abs(D[m + 1][nv + 1]) > 2e-8:
                return None
            for i in range(m):
                if basic[i] == -1:
                    entering = -1
                    for j in range(nv + 1):
                        if abs(D[i][j]) > eps and (
                            entering == -1
                            or abs(D[i][j]) > abs(D[i][entering]) + eps
                            or (
                                abs(abs(D[i][j]) - abs(D[i][entering])) <= eps
                                and nonbasic[j] < nonbasic[entering]
                            )
                        ):
                            entering = j
                    if entering != -1:
                        pivot(i, entering)
                    break
        if simplex(2) is not True:
            return None
        values = [0.0] * nv
        for i in range(m):
            if 0 <= basic[i] < nv:
                values[basic[i]] = D[i][nv + 1]
        x = [lows[j] + max(0.0, values[j]) for j in range(k)]
        return x if residual(S, x, lows, highs) <= feasibility_tol else None

    def build_hessian(S, until):
        k = len(S)
        H = [[0.0] * k for _ in range(k)]
        if Q is not None:
            for i, ai in enumerate(S):
                if clock() >= until:
                    return None
                for j, aj in enumerate(S):
                    H[i][j] = Q[ai][aj]
                H[i][i] += inv_theta
        else:
            for i, asset in enumerate(S):
                H[i][i] = d[asset] + inv_theta
            for row in B:
                if clock() >= until:
                    return None
                values = [row[asset] for asset in S]
                for i in range(k):
                    for j in range(i + 1):
                        addition = values[i] * values[j]
                        H[i][j] += addition
                        if i != j:
                            H[j][i] += addition
        return H

    def objective(S, x, H):
        value = 0.0
        for i in range(len(S)):
            hx = sum(H[i][j] * x[j] for j in range(len(S)))
            value += 0.5 * x[i] * hx - lam * mu[S[i]] * x[i]
        return value

    def cholesky(H):
        k = len(H)
        L = [[0.0] * k for _ in range(k)]
        scale = max(1.0, max(abs(H[i][i]) for i in range(k)))
        floor = 1e-14 * scale
        for i in range(k):
            for j in range(i + 1):
                value = 0.5 * (H[i][j] + H[j][i])
                value -= sum(L[i][p] * L[j][p] for p in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(value, floor))
                else:
                    if L[j][j] <= 0.0:
                        return None
                    L[i][j] = value / L[j][j]
        return L

    def chol_solve(L, b):
        k = len(b)
        y = [0.0] * k
        for i in range(k):
            y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
        x = [0.0] * k
        for i in range(k - 1, -1, -1):
            x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, k))) / L[i][i]
        return x

    def linear_solve(matrix, rhs):
        size = len(rhs)
        if size == 0:
            return []
        augmented = [list(matrix[i]) + [rhs[i]] for i in range(size)]
        scale = max(1.0, max(abs(v) for row in matrix for v in row))
        for column in range(size):
            pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
            if abs(augmented[pivot][column]) <= 1e-13 * scale:
                return None
            if pivot != column:
                augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            inverse = 1.0 / augmented[column][column]
            for j in range(column, size + 1):
                augmented[column][j] *= inverse
            for r in range(size):
                if r == column:
                    continue
                factor = augmented[r][column]
                if factor != 0.0:
                    for j in range(column, size + 1):
                        augmented[r][j] -= factor * augmented[column][j]
        return [augmented[i][size] for i in range(size)]

    def active_optimize(S, initial, H, lows, highs, until):
        k = len(S)
        if k == 0 or clock() >= until:
            return list(initial)
        L = cholesky(H)
        if L is None:
            return list(initial)
        inequalities = []

        def add_inequality(normal, rhs):
            norm = math.sqrt(sum(v * v for v in normal))
            if norm > 1e-15:
                inequalities.append(([v / norm for v in normal], rhs / norm))

        for j in range(k):
            normal = [0.0] * k
            normal[j] = -1.0
            add_inequality(normal, -lows[j])
            normal = [0.0] * k
            normal[j] = 1.0
            add_inequality(normal, highs[j])
        for r in range(nrules):
            normal = [A[r][asset] for asset in S]
            if math.isfinite(rule_hi[r]):
                add_inequality(normal, rule_hi[r])
            if math.isfinite(rule_lo[r]):
                add_inequality([-v for v in normal], -rule_lo[r])

        equality = [1.0 / math.sqrt(k)] * k
        x = list(initial)
        best = list(x)
        best_value = objective(S, x, H)
        active = []

        def independent(candidate, active_ids):
            vector = list(inequalities[candidate][0])
            basis = [list(equality)]
            for idx in active_ids:
                v = list(inequalities[idx][0])
                for q in basis:
                    coefficient = sum(v[j] * q[j] for j in range(k))
                    for j in range(k):
                        v[j] -= coefficient * q[j]
                norm = math.sqrt(sum(t * t for t in v))
                if norm > 1e-9:
                    basis.append([t / norm for t in v])
            for q in basis:
                coefficient = sum(vector[j] * q[j] for j in range(k))
                for j in range(k):
                    vector[j] -= coefficient * q[j]
            return sum(t * t for t in vector) > 1e-18

        tight = sorted(
            range(len(inequalities)),
            key=lambda idx: inequalities[idx][1]
            - sum(inequalities[idx][0][j] * x[j] for j in range(k)),
        )
        for idx in tight:
            slack = inequalities[idx][1] - sum(inequalities[idx][0][j] * x[j] for j in range(k))
            if slack > 2e-8:
                break
            if independent(idx, active):
                active.append(idx)

        while clock() < until:
            gradient = [
                sum(H[i][j] * x[j] for j in range(k)) - lam * mu[S[i]]
                for i in range(k)
            ]
            p0 = chol_solve(L, [-v for v in gradient])
            normals = [equality] + [inequalities[idx][0] for idx in active]
            hinv_normals = [chol_solve(L, normal) for normal in normals]
            gram = [
                [
                    sum(normals[i][j] * hinv_normals[h][j] for j in range(k))
                    for h in range(len(normals))
                ]
                for i in range(len(normals))
            ]
            rhs = [sum(normal[j] * p0[j] for j in range(k)) for normal in normals]
            multipliers = linear_solve(gram, rhs)
            if multipliers is None:
                if active:
                    active.pop()
                    continue
                break
            direction = [
                p0[j] - sum(hinv_normals[h][j] * multipliers[h] for h in range(len(normals)))
                for j in range(k)
            ]
            norm = math.sqrt(sum(v * v for v in direction))
            if norm <= 2e-11 * (1.0 + math.sqrt(sum(v * v for v in x))):
                if not active:
                    break
                position = min(range(len(active)), key=lambda pos: multipliers[pos + 1])
                if multipliers[position + 1] >= -2e-10:
                    break
                active.pop(position)
                continue

            directional = sum(gradient[j] * direction[j] for j in range(k))
            if directional >= -1e-13 * (1.0 + abs(best_value)):
                break
            alpha = 1.0
            blocker = None
            active_set = set(active)
            for idx, (normal, bound) in enumerate(inequalities):
                if idx in active_set:
                    continue
                movement = sum(normal[j] * direction[j] for j in range(k))
                if movement > 1e-13:
                    slack = bound - sum(normal[j] * x[j] for j in range(k))
                    candidate_alpha = max(0.0, slack) / movement
                    if candidate_alpha < alpha - 1e-13 or (
                        abs(candidate_alpha - alpha) <= 1e-13
                        and (blocker is None or idx < blocker)
                    ):
                        alpha = candidate_alpha
                        blocker = idx
            candidate = [x[j] + alpha * direction[j] for j in range(k)]
            delta = 1.0 - sum(candidate)
            anchor = max(
                range(k),
                key=lambda j: min(candidate[j] - lows[j], highs[j] - candidate[j]),
            )
            candidate[anchor] += delta
            if residual(S, candidate, lows, highs) > 5e-7:
                break
            candidate_value = objective(S, candidate, H)
            if candidate_value > best_value + 1e-10 * (1.0 + abs(best_value)):
                break
            x = candidate
            if candidate_value < best_value - 1e-14:
                best = list(candidate)
                best_value = candidate_value
            if blocker is not None and alpha < 1.0 - 1e-12:
                if independent(blocker, active):
                    active.append(blocker)
                elif alpha <= 1e-13:
                    break
        return best

    def repair_support(items):
        support = []
        present = set()
        for asset in items:
            if 0 <= asset < n and asset not in present and maximum[asset] > 0.0:
                support.append(asset)
                present.add(asset)
                if len(support) == K:
                    break
        if not support:
            support = [max(range(n), key=lambda i: maximum[i])]
            present = set(support)
        while len(support) > 1 and sum(minimum[i] for i in support) > 1.0 + 1e-10:
            removed = max(support, key=lambda i: (minimum[i], base_rank[i]))
            support.remove(removed)
            present.remove(removed)
        while sum(maximum[i] for i in support) < 1.0 - 1e-10:
            outside = [i for i in capacity_order if i not in present]
            if not outside:
                break
            addition = outside[0]
            if len(support) < K:
                support.append(addition)
                present.add(addition)
            else:
                removed = min(support, key=lambda i: (maximum[i], -base_rank[i]))
                if maximum[addition] <= maximum[removed] + 1e-15:
                    break
                support.remove(removed)
                present.remove(removed)
                support.append(addition)
                present.add(addition)
        return tuple(sorted(support))

    def all_gradient(S, x):
        held = {asset: x[j] for j, asset in enumerate(S)}
        result = [0.0] * n
        if Q is not None:
            for i in range(n):
                result[i] = sum(Q[i][asset] * x[j] for j, asset in enumerate(S))
                result[i] += inv_theta * held.get(i, 0.0) - lam * mu[i]
        else:
            factors = [sum(row[asset] * x[j] for j, asset in enumerate(S)) for row in B]
            for i in range(n):
                result[i] = sum(B[f][i] * factors[f] for f in range(len(B)))
                result[i] += (d[i] + inv_theta) * held.get(i, 0.0) - lam * mu[i]
        return result

    state = {
        "best_S": None,
        "best_x": None,
        "best_H": None,
        "best_obj": float("inf"),
        "feasible": None,
        "elite": {},
        "last_improvement": started,
        "neighbor_heap": [],
        "outside_order": list(base_order),
        "relaxed": [0.0] * n,
        "relax_size": 0,
    }
    seen = set()

    def rebuild_neighbors():
        S = state["best_S"]
        x = state["best_x"]
        if S is None:
            state["neighbor_heap"] = []
            return
        gradients = all_gradient(S, x)
        members = list(S)
        member_set = set(members)
        outside = [i for i in range(n) if i not in member_set and maximum[i] > 0.0]
        outside.sort(
            key=lambda i: (
                gradients[i] - 0.15 * state["relaxed"][i],
                single_score[i],
                i,
            )
        )
        state["outside_order"] = outside
        heap = []
        unique = set()
        if len(members) < K:
            donor_gradient = sum(gradients[i] * x[j] for j, i in enumerate(S))
            for addition in outside:
                candidate = tuple(sorted(members + [addition]))
                if candidate not in unique:
                    unique.add(candidate)
                    heapq.heappush(heap, (gradients[addition] - donor_gradient, candidate))
                if clock() >= search_end:
                    break
        if len(members) > 1 and clock() < search_end:
            for j, removal in enumerate(members):
                candidate = tuple(i for i in S if i != removal)
                if candidate not in unique:
                    unique.add(candidate)
                    heapq.heappush(heap, (-gradients[removal] * max(x[j], 1e-12), candidate))
        if clock() < search_end:
            for j, removal in enumerate(members):
                amount = max(x[j], minimum[removal], 1e-10)
                for addition in outside:
                    curvature = (
                        hessian_entry(addition, addition)
                        + hessian_entry(removal, removal)
                        - 2.0 * hessian_entry(addition, removal)
                    )
                    estimate = amount * (gradients[addition] - gradients[removal])
                    estimate += 0.5 * amount * amount * max(0.0, curvature)
                    candidate = tuple(sorted([i for i in members if i != removal] + [addition]))
                    if candidate not in unique:
                        unique.add(candidate)
                        heapq.heappush(heap, (estimate, candidate))
                    if clock() >= search_end:
                        break
                if clock() >= search_end:
                    break
        state["neighbor_heap"] = heap

    def consider(S, x, H):
        if residual(S, x) > feasibility_tol:
            return False
        value = objective(S, x, H)
        if not math.isfinite(value):
            return False
        key = tuple(S)
        previous = state["elite"].get(key)
        if previous is None or value < previous[0] - 1e-14:
            state["elite"][key] = (value, list(x), H)
        elapsed = max(0.0, clock() - started)
        elite_cap = 10 + int(5.0 * math.sqrt(elapsed + 1.0))
        if len(state["elite"]) > elite_cap:
            kept = sorted(state["elite"].items(), key=lambda item: item[1][0])[:elite_cap]
            state["elite"] = dict(kept)
        if value < state["best_obj"] - 1e-13 * (1.0 + abs(value)):
            state["best_S"] = key
            state["best_x"] = list(x)
            state["best_H"] = H
            state["best_obj"] = value
            state["last_improvement"] = clock()
            rebuild_neighbors()
            return True
        return False

    def process_support(candidate):
        S = tuple(sorted(set(candidate)))
        if not S or len(S) > K or S in seen or clock() >= search_end:
            return
        seen.add(S)
        if not support_possible(S):
            return
        lows = [minimum[i] for i in S]
        highs = [maximum[i] for i in S]
        x = lp_feasible(S, lows, highs, search_end)
        if x is None:
            return
        state["feasible"] = (S, list(x))
        H = build_hessian(S, search_end)
        if H is None:
            return
        consider(S, x, H)
        if clock() < search_end:
            optimized = active_optimize(S, x, H, lows, highs, search_end)
            consider(S, optimized, H)

    seed_supports = []

    def add_seed(items):
        candidate = repair_support(items)
        if candidate and candidate not in seed_supports:
            seed_supports.append(candidate)

    for order in (base_order, return_order, risk_order, capacity_order):
        add_seed(order[:K])
        add_seed(order)
    mix = max(1, K // 2)
    for r in range(nrules):
        high = sorted(range(n), key=lambda i: (A[r][i], -single_score[i]), reverse=True)
        low = list(reversed(high))
        add_seed(high[:K])
        add_seed(low[:K])
        add_seed(high[:mix] + base_order)
        add_seed(low[:mix] + base_order)
        add_seed(high[:mix] + low[:mix] + base_order)

    enum_sizes = list(range(K, 0, -1))
    enum_iterators = {size: itertools.combinations(tuple(base_order), size) for size in enum_sizes}
    enum_finished = set()
    enum_position = 0

    def next_enumerated():
        nonlocal enum_position
        if len(enum_finished) == len(enum_sizes):
            return None
        visited_sizes = 0
        while visited_sizes < len(enum_sizes):
            size = enum_sizes[enum_position % len(enum_sizes)]
            enum_position += 1
            visited_sizes += 1
            if size in enum_finished:
                continue
            try:
                return tuple(sorted(next(enum_iterators[size])))
            except StopIteration:
                enum_finished.add(size)
        return None

    def progressive_relaxation(elapsed):
        if clock() >= search_end:
            return None
        target = min(
            n,
            max(K, 2 * K + int(18.0 * ((elapsed + 0.1) ** (1.0 / 3.0)))),
        )
        last = state["relax_size"]
        if target <= last + max(1, last // 5):
            return None

        orders = [base_order, return_order, risk_order, capacity_order]
        for r in range(nrules):
            orders.append(sorted(range(n), key=lambda i: A[r][i]))
            orders.append(sorted(range(n), key=lambda i: A[r][i], reverse=True))
        pool = []
        present = set()
        depth = 0
        while len(pool) < target and depth < n:
            for order in orders:
                if depth < len(order):
                    asset = order[depth]
                    if asset not in present and maximum[asset] > 0.0:
                        present.add(asset)
                        pool.append(asset)
                        if len(pool) == target:
                            break
            depth += 1
        pool = tuple(sorted(pool))
        state["relax_size"] = target
        if not pool:
            return None
        lows = [0.0] * len(pool)
        highs = [maximum[i] for i in pool]
        if not support_possible(pool, lows, highs):
            return None
        x = lp_feasible(pool, lows, highs, search_end)
        if x is None or clock() >= search_end:
            return None
        H = build_hessian(pool, search_end)
        if H is None:
            return None
        optimized = active_optimize(pool, x, H, lows, highs, search_end)
        for i in range(n):
            state["relaxed"][i] *= 0.35
        for j, asset in enumerate(pool):
            state["relaxed"][asset] += optimized[j]

        positive = [
            asset
            for j, asset in sorted(
                enumerate(pool),
                key=lambda pair: (optimized[pair[0]], -single_score[pair[1]]),
                reverse=True,
            )
            if optimized[j] > 1e-12
        ]
        if positive:
            exact_support = tuple(sorted(positive))
            if len(exact_support) <= K and all(
                optimized[pool.index(asset)] >= minimum[asset] - 1e-8 for asset in exact_support
            ):
                process_support(exact_support)
        return repair_support(positive + base_order)

    def random_candidate(elapsed):
        elites = sorted(state["elite"].items(), key=lambda item: item[1][0])
        stagnation = max(0.0, clock() - state["last_improvement"])
        exploration = min(0.92, 0.06 + 1.0 - math.exp(-elapsed / max(2.0, K + 1.0)))

        if len(elites) >= 2 and rng.random() < min(0.45, 0.08 + elapsed / (elapsed + 30.0)):
            first = elites[rng.randrange(min(len(elites), 2 + int(math.sqrt(elapsed + 1.0))))][0]
            second = elites[rng.randrange(len(elites))][0]
            common = list(set(first).intersection(second))
            symmetric = list(set(first).symmetric_difference(second))
            rng.shuffle(symmetric)
            guide = sorted(
                range(n),
                key=lambda i: (state["relaxed"][i], -single_score[i]),
                reverse=True,
            )
            return repair_support(common + symmetric + guide + base_order)

        if elites:
            elite_width = min(len(elites), 2 + int(math.sqrt(elapsed + 1.0)))
            S, data = elites[rng.randrange(max(1, elite_width))]
            old_x = data[1]
            members = list(S)
            radius_limit = min(
                len(members),
                1 + int(math.sqrt(elapsed + 2.0 * stagnation + 1.0)),
            )
            radius = 1 + int((rng.random() ** 1.7) * max(1, radius_limit))
            radius = min(len(members), radius)
            removal_positions = sorted(
                range(len(members)),
                key=lambda j: (old_x[j] / max(minimum[members[j]] + 1e-12, 1e-12), rng.random()),
            )[:radius]
            remove_set = set(removal_positions)
            retained = [asset for j, asset in enumerate(members) if j not in remove_set]
            excluded = set(retained)
            desired = len(members)
            if desired < K and rng.random() < 0.35 + 0.5 * exploration:
                desired += rng.randint(1, K - desired)
            preferred = state["outside_order"] or base_order
            relaxed_order = sorted(
                range(n),
                key=lambda i: (state["relaxed"][i], -single_score[i]),
                reverse=True,
            )
            while len(retained) < desired:
                draw = rng.random()
                if draw < 0.30 and state["relax_size"] > 0:
                    choices = [i for i in relaxed_order if i not in excluded]
                    width = min(len(choices), max(K, int((K + 1) * math.sqrt(elapsed + 1.0))))
                    choices = choices[:width]
                elif draw < exploration:
                    choices = [i for i in range(n) if i not in excluded and maximum[i] > 0.0]
                else:
                    width = min(n, max(K + radius, int((K + 1) * math.sqrt(elapsed + 1.0))))
                    choices = [i for i in preferred[:width] if i not in excluded]
                if not choices:
                    break
                addition = choices[rng.randrange(len(choices))]
                retained.append(addition)
                excluded.add(addition)
            if nrules and rng.random() < 0.30:
                r = rng.randrange(nrules)
                direction = rng.random() < 0.5
                extreme = sorted(range(n), key=lambda i: A[r][i], reverse=direction)
                retained = retained[:max(0, desired - max(1, radius // 2))] + extreme
            return repair_support(retained)

        if K == 1:
            size = 1
        elif rng.random() < exploration:
            size = rng.randint(1, K)
        else:
            size = max(1, K - int((rng.random() ** 3) * K))
        chosen = rng.sample(range(n), min(size, n))
        if state["relax_size"] > 0:
            relaxed_order = sorted(range(n), key=lambda i: state["relaxed"][i], reverse=True)
            chosen = relaxed_order[:max(1, size // 2)] + chosen
        if nrules and rng.random() < 0.65:
            r = rng.randrange(nrules)
            extreme = sorted(range(n), key=lambda i: A[r][i], reverse=rng.random() < 0.5)
            chosen = extreme[:max(1, size // 3)] + chosen
        return repair_support(chosen)

    seed_index = 0
    attempt = 0
    pending = None
    while clock() < search_end:
        try:
            elapsed = max(0.0, clock() - started)
            candidate = None
            if pending is not None:
                candidate = pending
                pending = None
            elif seed_index < len(seed_supports):
                candidate = seed_supports[seed_index]
                seed_index += 1
            else:
                mode = attempt % 17
                if mode in (0, 9):
                    candidate = next_enumerated()
                elif mode == 5:
                    candidate = progressive_relaxation(elapsed)
                elif mode == 13 and state["best_S"] is not None:
                    S = state["best_S"]
                    lows = [minimum[i] for i in S]
                    highs = [maximum[i] for i in S]
                    improved = active_optimize(
                        S,
                        state["best_x"],
                        state["best_H"],
                        lows,
                        highs,
                        search_end,
                    )
                    consider(S, improved, state["best_H"])
                elif state["neighbor_heap"] and mode not in (4, 8, 14):
                    candidate = heapq.heappop(state["neighbor_heap"])[1]
                else:
                    candidate = random_candidate(elapsed)
            attempt += 1
            if candidate is not None:
                process_support(candidate)
        except BaseException:
            attempt += 1
            continue

    def clean(S, x):
        values = [0.0 if -1e-9 < value < 0.0 else value for value in x]
        if any(value < 0.0 or not math.isfinite(value) for value in values):
            return None
        delta = 1.0 - sum(values)
        if abs(delta) > 0.0:
            order = sorted(
                range(len(S)),
                key=lambda j: min(
                    values[j] - minimum[S[j]],
                    maximum[S[j]] - values[j],
                ),
                reverse=True,
            )
            for j in order:
                trial = list(values)
                trial[j] += delta
                if residual(S, trial) <= 8e-7:
                    values = trial
                    break
        if residual(S, values) > 8e-7:
            return None
        weights = [0.0] * n
        for j, asset in enumerate(S):
            weights[asset] = max(0.0, values[j])
        return {
            "weights": weights,
            "selected": [1 if value > 0.0 else 0 for value in weights],
        }

    candidates = []
    if state["best_S"] is not None:
        candidates.append((state["best_obj"], state["best_S"], state["best_x"]))
    for S, data in state["elite"].items():
        candidates.append((data[0], S, data[1]))
    candidates.sort(key=lambda item: item[0])
    checked = set()
    for _, S, x in candidates:
        if S in checked:
            continue
        checked.add(S)
        answer = clean(S, x)
        if answer is not None:
            return answer
    if state["feasible"] is not None:
        answer = clean(state["feasible"][0], state["feasible"][1])
        if answer is not None:
            return answer
    return _emergency(instance)
