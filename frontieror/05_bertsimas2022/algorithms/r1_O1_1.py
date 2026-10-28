import heapq
import itertools
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    try:
        return _solve(instance, time_limit_s)
    except BaseException:
        return _fallback(instance)


def _fallback(instance):
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

        def make(S):
            if not S or len(S) > K:
                return None
            x = [lo[i] for i in S]
            remaining = 1.0 - sum(x)
            if remaining < -1e-9:
                return None
            while remaining > 1e-12:
                free = [j for j, i in enumerate(S) if hi[i] - x[j] > 1e-14]
                if not free:
                    return None
                share = remaining / len(free)
                changed = 0.0
                for j in free:
                    addition = min(share, hi[S[j]] - x[j])
                    x[j] += addition
                    changed += addition
                if changed <= 1e-16:
                    return None
                remaining -= changed
            if abs(sum(x) - 1.0) > 1e-7:
                return None
            for r, row in enumerate(A):
                value = sum(float(row[i]) * x[j] for j, i in enumerate(S))
                if r < len(lb) and value < float(lb[r]) - 1e-7:
                    return None
                if r < len(ub) and value > float(ub[r]) + 1e-7:
                    return None
            w = [0.0] * n
            for j, i in enumerate(S):
                w[i] = max(0.0, x[j])
            return {"weights": w, "selected": [1 if v > 0.0 else 0 for v in w]}

        for i in range(n):
            answer = make((i,))
            if answer is not None:
                return answer
        orders = [sorted(range(n), key=lambda i: hi[i], reverse=True)]
        mu = instance.get("expected_returns") or []
        if len(mu) == n:
            orders.append(sorted(range(n), key=lambda i: float(mu[i]), reverse=True))
        for order in orders:
            for q in range(2, K + 1):
                answer = make(tuple(sorted(order[:q])))
                if answer is not None:
                    return answer
        S = tuple(sorted(orders[0][:K]))
        answer = make(S)
        if answer is not None:
            return answer
        w = [0.0] * n
        w[orders[0][0]] = 1.0
        return {"weights": w, "selected": [1 if v > 0.0 else 0 for v in w]}
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
    reserve = min(0.02, max(0.001, limit * 0.001)) if limit > 0.0 else 0.0
    search_end = max(started, deadline - reserve)

    n = int(instance["num_securities"])
    K = max(1, min(n, int(instance["max_securities"])))
    mu = [float(v) for v in instance["expected_returns"]]
    lam = max(0.0, float(instance["return_weight"]))
    theta = float(instance["concentration_parameter"])
    inv_theta = 1.0 / theta

    A = [[float(v) for v in row] for row in (instance.get("linear_rule_coefficients") or [])]
    rule_lo = [float(v) for v in (instance.get("linear_rule_lower_bounds") or [])]
    rule_hi = [float(v) for v in (instance.get("linear_rule_upper_bounds") or [])]
    nrules = len(A)

    raw_lo = instance.get("minimum_weights") or []
    raw_hi = instance.get("maximum_weights") or []
    minimum = [max(0.0, float(v)) for v in raw_lo] if len(raw_lo) == n else [0.0] * n
    maximum = [max(0.0, float(v)) for v in raw_hi] if len(raw_hi) == n else [1.0] * n

    if instance.get("covariance_format", "direct") == "factor":
        B = [[float(v) for v in row] for row in (instance.get("factor_loadings") or [])]
        d = [float(v) for v in instance.get("idiosyncratic_variances", [])]
        Q = None
        diagonal = [d[i] + sum(row[i] * row[i] for row in B) for i in range(n)]
    else:
        Q = [[float(v) for v in row] for row in instance["covariance_matrix"]]
        B = None
        d = None
        diagonal = [Q[i][i] for i in range(n)]

    base_score = [0.5 * (diagonal[i] + inv_theta) - lam * mu[i] for i in range(n)]
    base_order = sorted(range(n), key=lambda i: (base_score[i], i))
    return_order = sorted(range(n), key=lambda i: (-mu[i], base_score[i], i))
    risk_order = sorted(range(n), key=lambda i: (diagonal[i], -mu[i], i))
    base_rank = {asset: rank for rank, asset in enumerate(base_order)}

    seed = n * 1000003 + K * 9176 + nrules * 131
    for i, value in enumerate(mu[:128]):
        if math.isfinite(value):
            seed ^= (int(abs(value) * 1000000007.0) + i * 104729) & 0xFFFFFFFFFFFFFFFF
    rng = random.Random(seed & 0xFFFFFFFFFFFFFFFF)
    feasibility_tol = 2e-7

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
        k = len(lows)
        x = list(lows)
        remaining = 1.0 - sum(x)
        if remaining < -1e-10 or remaining > sum(highs[j] - lows[j] for j in range(k)) + 1e-10:
            return None
        while remaining > 1e-14:
            free = [j for j in range(k) if highs[j] - x[j] > 1e-15]
            if not free:
                return None
            share = remaining / len(free)
            changed = 0.0
            for j in free:
                addition = min(share, highs[j] - x[j])
                if addition > 0.0:
                    x[j] += addition
                    changed += addition
            if changed <= 1e-16:
                return None
            remaining -= changed
        delta = 1.0 - sum(x)
        if abs(delta) > 0.0:
            for j in range(k):
                if lows[j] - 1e-12 <= x[j] + delta <= highs[j] + 1e-12:
                    x[j] += delta
                    break
        return x

    def support_possible(S):
        if not S or len(S) > K:
            return False
        lows = [minimum[i] for i in S]
        caps = [maximum[i] - minimum[i] for i in S]
        remaining = 1.0 - sum(lows)
        if remaining < -1e-9 or remaining > sum(caps) + 1e-9:
            return False
        for r in range(nrules):
            row = A[r]
            base = sum(row[asset] * lows[j] for j, asset in enumerate(S))
            pairs = sorted((row[asset], max(0.0, caps[j])) for j, asset in enumerate(S))
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
                        a = D[i][nv + 1] / D[i][entering]
                        b = D[leaving][nv + 1] / D[leaving][entering]
                        if a < b - eps or (abs(a - b) <= eps and basic[i] < basic[leaving]):
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
                            or (abs(abs(D[i][j]) - abs(D[i][entering])) <= eps
                                and nonbasic[j] < nonbasic[entering])
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

    def hessian(S, until):
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
                value = 0.5 * (H[i][j] + H[j][i]) - sum(L[i][p] * L[j][p] for p in range(j))
                if i == j:
                    if value < floor:
                        value = floor
                    L[i][j] = math.sqrt(value)
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

    def linear_solve(M, b):
        nrow = len(b)
        if nrow == 0:
            return []
        aug = [list(M[i]) + [b[i]] for i in range(nrow)]
        scale = max(1.0, max(abs(v) for row in M for v in row))
        for col in range(nrow):
            pivot = max(range(col, nrow), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot][col]) <= 1e-13 * scale:
                return None
            if pivot != col:
                aug[col], aug[pivot] = aug[pivot], aug[col]
            inverse = 1.0 / aug[col][col]
            for j in range(col, nrow + 1):
                aug[col][j] *= inverse
            for r in range(nrow):
                if r == col:
                    continue
                factor = aug[r][col]
                if factor != 0.0:
                    for j in range(col, nrow + 1):
                        aug[r][j] -= factor * aug[col][j]
        return [aug[i][nrow] for i in range(nrow)]

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
            lower_normal = [0.0] * k
            lower_normal[j] = -1.0
            add_inequality(lower_normal, -lows[j])
            upper_normal = [0.0] * k
            upper_normal[j] = 1.0
            add_inequality(upper_normal, highs[j])
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

        tight = sorted(range(len(inequalities)),
                       key=lambda idx: inequalities[idx][1] - sum(inequalities[idx][0][j] * x[j] for j in range(k)))
        for idx in tight:
            slack = inequalities[idx][1] - sum(inequalities[idx][0][j] * x[j] for j in range(k))
            if slack > 2e-8:
                break
            if independent(idx, active):
                active.append(idx)

        while clock() < until:
            gradient = [sum(H[i][j] * x[j] for j in range(k)) - lam * mu[S[i]] for i in range(k)]
            p0 = chol_solve(L, [-v for v in gradient])
            normals = [equality] + [inequalities[idx][0] for idx in active]
            hinv_normals = [chol_solve(L, normal) for normal in normals]
            gram = [[sum(normals[i][j] * hinv_normals[h][j] for j in range(k))
                     for h in range(len(normals))] for i in range(len(normals))]
            rhs = [sum(normal[j] * p0[j] for j in range(k)) for normal in normals]
            multipliers = linear_solve(gram, rhs)
            if multipliers is None:
                if active:
                    active.pop()
                    continue
                break
            p = [p0[j] - sum(hinv_normals[h][j] * multipliers[h] for h in range(len(normals)))
                 for j in range(k)]
            pnorm = math.sqrt(sum(v * v for v in p))
            if pnorm <= 2e-11 * (1.0 + math.sqrt(sum(v * v for v in x))):
                if not active:
                    break
                worst_position = min(range(len(active)), key=lambda pos: multipliers[pos + 1])
                if multipliers[worst_position + 1] >= -2e-10:
                    break
                active.pop(worst_position)
                continue

            directional = sum(gradient[j] * p[j] for j in range(k))
            if directional >= -1e-13 * (1.0 + abs(best_value)):
                break
            alpha = 1.0
            blocker = None
            active_set = set(active)
            for idx, (normal, rhs_value) in enumerate(inequalities):
                if idx in active_set:
                    continue
                movement = sum(normal[j] * p[j] for j in range(k))
                if movement > 1e-13:
                    slack = rhs_value - sum(normal[j] * x[j] for j in range(k))
                    candidate_alpha = max(0.0, slack) / movement
                    if candidate_alpha < alpha - 1e-13 or (
                        abs(candidate_alpha - alpha) <= 1e-13 and (blocker is None or idx < blocker)
                    ):
                        alpha = candidate_alpha
                        blocker = idx
            candidate = [x[j] + alpha * p[j] for j in range(k)]
            delta = 1.0 - sum(candidate)
            anchor = max(range(k), key=lambda j: min(candidate[j] - lows[j], highs[j] - candidate[j]))
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
            if 0 <= asset < n and asset not in present:
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
            outside = [i for i in base_order if i not in present]
            if not outside:
                break
            addition = max(outside, key=lambda i: (maximum[i], -base_rank[i]))
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
        "outside_order": None,
        "relaxed": None,
        "pending": None,
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
        outside = [i for i in range(n) if i not in member_set]
        state["outside_order"] = sorted(outside, key=lambda i: (gradients[i], base_score[i], i))
        heap = []
        unique = set()
        if len(members) < K:
            for addition in outside:
                candidate = tuple(sorted(members + [addition]))
                if candidate not in unique:
                    unique.add(candidate)
                    heapq.heappush(heap, (gradients[addition], candidate))
        if len(members) > 1:
            for removal in members:
                candidate = tuple(i for i in S if i != removal)
                if candidate not in unique:
                    unique.add(candidate)
                    heapq.heappush(heap, (-gradients[removal], candidate))
        for removal in members:
            for addition in outside:
                candidate = tuple(sorted([i for i in members if i != removal] + [addition]))
                if candidate not in unique:
                    unique.add(candidate)
                    score = gradients[addition] - gradients[removal]
                    heapq.heappush(heap, (score, candidate))
        state["neighbor_heap"] = heap

    def consider(S, x, H):
        if residual(S, x) > feasibility_tol:
            return False
        value = objective(S, x, H)
        if not math.isfinite(value):
            return False
        key = tuple(S)
        old = state["elite"].get(key)
        if old is None or value < old[0] - 1e-14:
            state["elite"][key] = (value, list(x))
        elapsed = max(0.0, clock() - started)
        elite_cap = 8 + int(4.0 * math.sqrt(elapsed + 1.0))
        if len(state["elite"]) > elite_cap:
            state["elite"] = dict(sorted(state["elite"].items(), key=lambda item: item[1][0])[:elite_cap])
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
        H = hessian(S, search_end)
        if H is None:
            return
        consider(S, x, H)
        if clock() < search_end:
            improved = active_optimize(S, x, H, lows, highs, search_end)
            consider(S, improved, H)

    seed_supports = [
        repair_support(base_order[:K]),
        repair_support(return_order[:K]),
        repair_support(risk_order[:K]),
    ]
    mix = max(1, K // 2)
    for r in range(nrules):
        high = sorted(range(n), key=lambda i: (A[r][i], -base_score[i]), reverse=True)
        low = list(reversed(high))
        seed_supports.append(repair_support(high[:K]))
        seed_supports.append(repair_support(low[:K]))
        seed_supports.append(repair_support(high[:mix] + base_order))
        seed_supports.append(repair_support(low[:mix] + base_order))

    enum_sizes = list(range(K, 0, -1))
    enum_iterators = {q: itertools.combinations(tuple(base_order), q) for q in enum_sizes}
    enum_finished = set()
    enum_position = 0

    def next_enumerated():
        nonlocal enum_position
        if len(enum_finished) == len(enum_sizes):
            return None
        checked = 0
        while checked < len(enum_sizes):
            q = enum_sizes[enum_position % len(enum_sizes)]
            enum_position += 1
            checked += 1
            if q in enum_finished:
                continue
            try:
                return tuple(sorted(next(enum_iterators[q])))
            except StopIteration:
                enum_finished.add(q)
        return None

    def random_candidate(elapsed):
        elites = list(state["elite"].items())
        if elites and len(elites) >= 2 and rng.random() < min(0.3, 0.04 + math.log1p(elapsed) / 30.0):
            first = elites[rng.randrange(len(elites))][0]
            second = elites[rng.randrange(len(elites))][0]
            pool = list(set(first).union(second))
            rng.shuffle(pool)
            common = list(set(first).intersection(second))
            return repair_support(common + pool + base_order)

        if elites:
            elites.sort(key=lambda item: item[1][0])
            bound = min(len(elites), 2 + int(math.sqrt(elapsed + 1.0)))
            S, data = elites[rng.randrange(max(1, bound))]
            old_x = data[1]
            members = list(S)
            stagnation = max(0.0, clock() - state["last_improvement"])
            radius_limit = min(len(members), 1 + int(math.log1p(elapsed + stagnation * stagnation + 1.0)))
            radius = rng.randint(1, max(1, radius_limit))
            removal_positions = sorted(range(len(members)), key=lambda j: (old_x[j], rng.random()))[:radius]
            remove_set = set(removal_positions)
            retained = [asset for j, asset in enumerate(members) if j not in remove_set]
            excluded = set(retained)
            preferred = state["outside_order"] or base_order
            exploration = min(0.75, 0.08 + math.log1p(elapsed) / 12.0)
            while len(retained) < min(K, len(members)):
                if rng.random() < exploration:
                    choices = [i for i in range(n) if i not in excluded]
                else:
                    width = min(n, max(K + radius, int((K + 1) * (1.0 + math.sqrt(elapsed + 1.0)))))
                    choices = [i for i in preferred[:width] if i not in excluded]
                    if not choices:
                        choices = [i for i in range(n) if i not in excluded]
                if not choices:
                    break
                addition = choices[rng.randrange(len(choices))]
                retained.append(addition)
                excluded.add(addition)
            return repair_support(retained)

        exploration = min(0.8, math.log1p(elapsed + 1.0) / 10.0)
        if K == 1:
            q = 1
        elif rng.random() < exploration:
            q = rng.randint(1, K)
        else:
            q = max(1, K - int((rng.random() ** 3) * K))
        chosen = rng.sample(range(n), min(q, n))
        if state["relaxed"] is not None:
            top = sorted(range(n), key=lambda i: state["relaxed"][i], reverse=True)
            chosen = top[:max(1, q // 2)] + chosen
        if nrules and rng.random() < 0.65:
            r = rng.randrange(nrules)
            extreme = sorted(range(n), key=lambda i: A[r][i], reverse=rng.random() < 0.5)
            chosen = extreme[:max(1, q // 3)] + chosen
        return repair_support(chosen)

    seed_index = 0
    dense_attempted = False
    attempt = 0
    while clock() < search_end:
        elapsed = max(0.0, clock() - started)
        candidate = None

        if state["pending"] is not None:
            candidate = state["pending"]
            state["pending"] = None
        elif seed_index < len(seed_supports):
            candidate = seed_supports[seed_index]
            seed_index += 1
        elif not dense_attempted and nrules > 0:
            dense_attempted = True
            all_assets = tuple(range(n))
            dense = lp_feasible(all_assets, [0.0] * n, list(maximum), search_end)
            if dense is not None:
                state["relaxed"] = dense
                top = sorted(range(n), key=lambda i: (dense[i], -base_score[i]), reverse=True)
                state["pending"] = repair_support(top[:K] + base_order)
            continue
        else:
            mode = attempt % 12
            if mode in (0, 7):
                candidate = next_enumerated()
            elif mode == 11 and state["best_S"] is not None:
                S = state["best_S"]
                lows = [minimum[i] for i in S]
                highs = [maximum[i] for i in S]
                improved = active_optimize(S, state["best_x"], state["best_H"], lows, highs, search_end)
                consider(S, improved, state["best_H"])
            elif state["neighbor_heap"] and mode not in (4, 9):
                candidate = heapq.heappop(state["neighbor_heap"])[1]
            else:
                candidate = random_candidate(elapsed)

        attempt += 1
        if candidate is not None:
            process_support(candidate)

    def clean(S, x):
        values = [0.0 if -1e-10 < value < 0.0 else value for value in x]
        if any(value < 0.0 or not math.isfinite(value) for value in values):
            return None
        delta = 1.0 - sum(values)
        if abs(delta) > 0.0:
            order = sorted(range(len(S)),
                           key=lambda j: min(values[j] - minimum[S[j]], maximum[S[j]] - values[j]),
                           reverse=True)
            repaired = None
            for j in order:
                trial = list(values)
                trial[j] += delta
                if residual(S, trial) <= 8e-7:
                    repaired = trial
                    break
            if repaired is not None:
                values = repaired
        if residual(S, values) > 8e-7:
            return None
        weights = [0.0] * n
        for j, asset in enumerate(S):
            weights[asset] = max(0.0, values[j])
        return {"weights": weights, "selected": [1 if value > 0.0 else 0 for value in weights]}

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
    return _fallback(instance)
