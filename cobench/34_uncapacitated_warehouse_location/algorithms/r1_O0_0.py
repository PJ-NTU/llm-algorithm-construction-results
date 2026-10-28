"""Adaptive hybrid facility-location search
Combines open, close and swap local search, time-limited mixed-integer optimisation and perturbation restarts, always keeping the best feasible assignment.
"""

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {"y": [], "x": []}
    best_assignment = None
    try:
        m = instance["m_facilities"]
        n = instance["n"]
        if type(m) is not int or type(n) is not int or m < 1 or n < 1:
            return {"y": [], "x": [], "error": "invalid input"}
        fixed = instance["fixed_cost"]
        costs = instance["alloc_cost"]
        if len(fixed) != m or len(costs) != m or any(len(row) != n for row in costs):
            return {"y": [], "x": [], "error": "invalid input"}

        # Build a complete fallback solution first; failures of the search or the optional solver do not affect feasibility.
        answer = {"y": [1] + [0] * (m - 1), "x": [[0] * n for _ in range(m)]}
        answer["x"][0] = [1] * n
        best_assignment = [0] * n
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        deadline = started + budget

        import numpy as np

        f = np.asarray(fixed, dtype=float)
        c = np.asarray(costs, dtype=float)
        if (f.shape != (m,) or c.shape != (m, n) or not np.isfinite(f).all()
                or not np.isfinite(c).all() or (f < 0).any() or (c < 0).any()):
            return {"y": [], "x": [], "error": "invalid input"}
        scale = max(float(f.max()), float(c.max()))
        if scale == 0:
            return answer
        f = f / scale
        c = c / scale
        # Remove the constant each customer must pay, reducing cancellation in difference computations.
        c -= c.min(axis=0)
        setup_time = time.perf_counter() - started
        reserve = min(budget * 0.18, 0.008 + 0.5 * setup_time)
        stop = deadline - reserve
        columns = np.arange(n)
        rng = random.Random(917231)
        best_value = math.inf
        best_ids = np.array([0], dtype=int)
        block_rows = max(1, 262144 // n)
        strength = math.log1p(budget)

        def state(ids):
            ids = np.unique(np.asarray(ids, dtype=int))
            if len(ids) == 0:
                ids = best_ids.copy()
            k = len(ids)
            if k * n <= 1048576:
                rows = c[ids]
                owner = rows.argmin(axis=0)
                used = np.unique(owner)
                if len(used) < k:
                    return state(ids[used])
                first = rows[owner, columns].copy()
                if k > 1:
                    rows.partition(1, axis=0)
                    second = rows[1].copy()
                else:
                    second = np.full(n, np.inf)
            else:
                first = np.full(n, np.inf)
                second = np.full(n, np.inf)
                owner = np.zeros(n, dtype=int)
                for p, i in enumerate(ids):
                    row = c[i]
                    better = row < first
                    second = np.where(better, first, np.minimum(second, row))
                    first = np.minimum(first, row)
                    owner[better] = p
                used = np.unique(owner)
                if len(used) < k:
                    return state(ids[used])
            value = float(f[ids].sum() + first.sum())
            return ids, owner, first, second, value

        def retain(st):
            nonlocal best_value, best_ids, best_assignment
            ids, owner, first, second, value = st
            if value < best_value:
                best_value = value
                best_ids = ids.copy()
                best_assignment = ids[owner].tolist()

        def local_search(ids, until):
            while time.perf_counter() < until:
                st = state(ids)
                ids, owner, first, second, value = st
                retain(st)
                k = len(ids)
                tolerance = 2e-12 * max(value, 1e-280)
                change = -tolerance
                remove = -1
                add = -1
                if k > 1:
                    loss = np.bincount(owner, weights=second - first, minlength=k) - f[ids]
                    p = int(loss.argmin())
                    if loss[p] < change:
                        change, remove = float(loss[p]), int(ids[p])
                opened = np.zeros(m, dtype=bool)
                opened[ids] = True
                closed = np.flatnonzero(~opened)
                largest_fixed = float(f[ids].max())
                complete = True
                for a in range(0, len(closed), block_rows):
                    if time.perf_counter() >= until:
                        complete = False
                        break
                    candidates = closed[a:a + block_rows]
                    savings = np.maximum(first - c[candidates], 0.0).sum(axis=1)
                    increments = f[candidates] - savings
                    p = int(increments.argmin())
                    if increments[p] < change:
                        change = float(increments[p])
                        add, remove = int(candidates[p]), -1
                    # Swap in a, then close b: aggregate the exact reassignment loss by original serving facility.
                    eligible = np.flatnonzero(increments - largest_fixed < change)
                    for p in eligible:
                        if time.perf_counter() >= until:
                            complete = False
                            break
                        i = int(candidates[p])
                        correction = np.maximum(np.minimum(c[i], second) - first, 0.0)
                        swaps = (float(increments[p]) - f[ids]
                                 + np.bincount(owner, weights=correction, minlength=k))
                        q = int(swaps.argmin())
                        if swaps[q] < change:
                            change = float(swaps[q])
                            add, remove = i, int(ids[q])
                    if not complete:
                        break
                if add < 0 and remove < 0:
                    return ids
                if not complete or time.perf_counter() >= until:
                    return ids
                if remove >= 0:
                    ids = ids[ids != remove]
                if add >= 0:
                    ids = np.append(ids, add)
            # The last accepted move must also enter the fallback record.
            if len(ids):
                retain(state(ids))
            return ids

        singles = f + c.sum(axis=1)
        single = int(singles.argmin())
        retain(state([single]))
        if m == 1 or n == 1:
            stop = started
        elif not np.any(f):
            retain(state(np.unique(c.argmin(axis=0))))
            stop = started
        elif time.perf_counter() < stop:
            initial = np.union1d([single], np.flatnonzero(f == 0))
            until = time.perf_counter() + 0.18 * max(0.0, stop - time.perf_counter())
            local_search(initial, until)

        milp_api = None
        solver_disabled = False

        def integer_search(variable_ids, fixed_ids, until):
            nonlocal milp_api, solver_disabled
            if solver_disabled or time.perf_counter() >= until:
                return False
            try:
                if milp_api is None:
                    from scipy.optimize import milp, Bounds, LinearConstraint
                    from scipy.sparse import coo_matrix
                    milp_api = milp, Bounds, LinearConstraint, coo_matrix
                milp, Bounds, LinearConstraint, coo_matrix = milp_api
                left = until - time.perf_counter()
                if left <= max(0.015, budget * 0.002):
                    return False
                variable_ids = np.unique(np.asarray(variable_ids, dtype=int))
                fixed_ids = np.asarray(fixed_ids, dtype=int)
                real_count = len(variable_ids)
                if real_count == m:
                    matrix = c
                else:
                    matrix = c[variable_ids]
                opening = f[variable_ids]
                if len(fixed_ids):
                    baseline = np.full(n, np.inf)
                    for a in range(0, len(fixed_ids), block_rows):
                        baseline = np.minimum(baseline, c[fixed_ids[a:a + block_rows]].min(axis=0))
                    matrix = np.vstack((matrix, baseline))
                    opening = np.append(opening, 0.0)
                v = len(opening)
                # Control model size by the remaining wall-clock budget; large models keep candidate arcs from both orderings.
                capacity = max(n, int(7500 * math.sqrt(max(left, 0.001))))
                full = v * n <= capacity
                width = min(v, max(1, capacity // max(2 * n, 1)))
                pieces = []
                chunk = max(1, 262144 // max(v, 1))
                share = max(1.0, n / max(1, len(best_ids)))
                for a in range(0, n, chunk):
                    if time.perf_counter() >= until:
                        return False
                    b = min(n, a + chunk)
                    customers = np.arange(a, b)
                    if full:
                        facilities = np.repeat(np.arange(v), b - a)
                        clients = np.tile(customers, v)
                        pieces.append(facilities * n + clients)
                    else:
                        block = matrix[:, a:b]
                        nearest = np.argpartition(block, width - 1, axis=0)[:width]
                        adjusted = block + opening[:, None] / share
                        shared = np.argpartition(adjusted, width - 1, axis=0)[:width]
                        pieces.append((nearest * n + customers).ravel())
                        pieces.append((shared * n + customers).ravel())
                # However sparse the model, keep all assignment arcs of the current best solution.
                mapping = np.full(m, -1, dtype=int)
                mapping[variable_ids] = np.arange(real_count)
                incumbent = mapping[np.asarray(best_assignment, dtype=int)]
                if len(fixed_ids):
                    incumbent[incumbent < 0] = real_count
                valid = incumbent >= 0
                pieces.append(incumbent[valid] * n + columns[valid])
                encoded = np.unique(np.concatenate(pieces))
                facility = encoded // n
                client = encoded % n
                e = len(encoded)
                index = np.arange(e)
                rows = np.concatenate((client, n + index, n + index))
                cols = np.concatenate((v + index, v + index, facility))
                data = np.concatenate((np.ones(2 * e), -np.ones(e)))
                constraints = coo_matrix((data, (rows, cols)), shape=(n + e, v + e)).tocsc()
                lower = np.concatenate((np.ones(n), np.full(e, -np.inf)))
                upper = np.concatenate((np.ones(n), np.zeros(e)))
                lb = np.zeros(v + e)
                ub = np.ones(v + e)
                if len(fixed_ids):
                    lb[real_count] = 1.0
                objective = np.concatenate((opening, matrix[facility, client]))
                unit = max(best_value, 1e-280)
                objective = np.minimum(objective, 2.0 * unit) / unit
                integral = np.zeros(v + e, dtype=np.uint8)
                integral[:v] = 1
                remaining = until - time.perf_counter() - min(0.04, 0.03 * budget)
                if remaining <= max(0.01, budget * 0.001):
                    return False
                result = milp(
                    objective, integrality=integral, bounds=Bounds(lb, ub),
                    constraints=LinearConstraint(constraints, lower, upper),
                    options={"disp": False, "time_limit": remaining, "mip_rel_gap": 0.0},
                )
                if result.x is not None:
                    chosen = variable_ids[np.flatnonzero(result.x[:real_count] > 0.5)]
                    chosen = np.union1d(chosen, fixed_ids)
                    if len(chosen):
                        retain(state(chosen))
                return bool(full and not len(fixed_ids) and real_count == m
                            and result.status == 0 and result.x is not None)
            except Exception:
                solver_disabled = True
                return False

        proved = best_value <= float(f.min())
        now = time.perf_counter()
        if not proved and now < stop:
            proved = integer_search(np.arange(m), [], now + 0.55 * (stop - now))
        current = best_ids.copy()
        stale = 0
        next_integer = time.perf_counter() + 0.4 * max(0.0, stop - time.perf_counter())

        # No restart count or iteration cap; apart from proven optimality, only the wall clock ends the run.
        while not proved and time.perf_counter() < stop:
            previous_best = best_value
            now = time.perf_counter()
            if (not solver_disabled and now >= next_integer
                    and stop - now > max(0.06, budget * 0.015)):
                # Fix some good facilities and jointly optimise the released facilities and their promising substitutes.
                release_count = min(len(best_ids), max(1, int(math.sqrt((1 + strength) * (1 + stale)))))
                released = np.asarray(rng.sample(best_ids.tolist(), release_count), dtype=int)
                fixed_ids = np.setdiff1d(best_ids, released)
                if len(fixed_ids):
                    baseline = np.full(n, np.inf)
                    for a in range(0, len(fixed_ids), block_rows):
                        baseline = np.minimum(baseline, c[fixed_ids[a:a + block_rows]].min(axis=0))
                    scores = np.empty(m)
                    for a in range(0, m, block_rows):
                        if time.perf_counter() >= stop:
                            break
                        b = min(m, a + block_rows)
                        scores[a:b] = f[a:b] - np.maximum(baseline - c[a:b], 0).sum(axis=1)
                    if time.perf_counter() >= stop:
                        break
                else:
                    scores = singles.copy()
                scores[best_ids] = np.inf
                count = min(m - len(best_ids), max(1, int((2 + strength) * math.sqrt(release_count + 1))))
                alternatives = np.argsort(scores)[:count]
                variables = np.union1d(released, alternatives)
                integer_search(variables, fixed_ids, now + 0.4 * (stop - now))
                current = best_ids.copy()
                next_integer = time.perf_counter() + 0.5 * max(0.0, stop - time.perf_counter())
            else:
                if rng.random() < 1.0 / (2.0 + strength):
                    current = best_ids.copy()
                chosen = set(int(i) for i in current)
                depth = min(m, max(1, int(1 + rng.random() * (strength + math.sqrt(stale)))))
                remove_count = min(len(chosen), max(1, depth // 2))
                chosen.difference_update(rng.sample(sorted(chosen), remove_count))
                for _ in range(depth):
                    if time.perf_counter() >= stop:
                        break
                    if rng.random() < 0.8:
                        j = rng.randrange(n)
                        score = c[:, j] + f * max(1, len(best_ids)) / n
                        width = min(m, max(1, int(1 + strength)))
                        candidates = np.argpartition(score, width - 1)[:width]
                        chosen.add(int(candidates[rng.randrange(width)]))
                    else:
                        chosen.add(rng.randrange(m))
                if not chosen:
                    chosen.add(single)
                current = np.asarray(sorted(chosen), dtype=int)
            current = local_search(current, stop)
            if best_value < previous_best:
                stale = 0
            else:
                stale += 1
            proved = best_value <= float(f.min())
    except Exception:
        # Do not let optional modules, numerical computation or input-parsing exceptions break the existing complete fallback solution.
        pass

    if best_assignment is not None and answer["x"]:
        try:
            y = [0] * len(answer["y"])
            for j, i in enumerate(best_assignment):
                i = int(i)
                answer["x"][0][j] = 0
                answer["x"][i][j] = 1
                y[i] = 1
            answer["y"] = y
        except Exception:
            # The search stores only valid indices; this branch preserves the initial solution in exceptional cases.
            answer = {"y": [1] + [0] * (m - 1), "x": [[0] * n for _ in range(m)]}
            answer["x"][0] = [1] * n
    return answer
