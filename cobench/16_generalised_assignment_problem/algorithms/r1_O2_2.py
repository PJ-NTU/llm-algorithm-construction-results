"""Linear-relaxation-guided adaptive integer-programming search
Combines capacity-price construction, tabu feasibility repair and reduced-cost screening, keeping the best feasible solution in integer-programming neighbourhoods that scale with the time limit.
"""

import math
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    budget = max(0.0, float(time_limit_s))
    deadline = start + budget - min(budget * 0.09, 0.035 * (1.0 + math.sqrt(budget)))
    m = int(instance["num_machines"])
    n = int(instance["num_jobs"])
    if n == 0:
        return {"assignments": {}}

    original_c = instance["cost_matrix"]
    original_r = instance["resource_matrix"]
    original_b = instance["capacities"]
    c = np.asarray(original_c, dtype=float)
    r = np.asarray(original_r, dtype=float)
    b = np.asarray(original_b, dtype=float)
    jobs = np.arange(n)
    scale = np.maximum(1.0, np.abs(b))
    nonnegative = bool(np.all(r >= 0))
    allowed = r <= b[:, None] if nonnegative else np.ones((m, n), dtype=bool)
    shifted = c - np.min(c, axis=0)
    span = max(1.0, float(np.mean(np.max(c, axis=0) - np.min(c, axis=0))))
    rng = np.random.default_rng((n * 1000003 + m * 9176) % (2 ** 32))
    best = None
    best_cost = math.inf
    best_load = None
    emergency = np.argmin(r / scale[:, None], axis=0)
    emergency_bad = math.inf
    lp_x = None
    reduced = np.zeros((m, n))
    lower_bound = -math.inf
    prices = np.zeros(m)
    strength = 1.0 + math.log1p(budget)

    def remaining():
        return deadline - time.perf_counter()

    def record(a):
        nonlocal best, best_cost, best_load, emergency, emergency_bad
        if a is None or len(a) != n:
            return False
        a = np.asarray(a, dtype=int)
        if np.any(a < 0) or np.any(a >= m):
            return False
        # Final acceptance verifies each entry on the original integers, without relying on solver floating-point tolerance.
        loads = [0] * m
        value = 0
        for j in range(n):
            i = int(a[j])
            loads[i] += original_r[i][j]
            value += original_c[i][j]
        bad = sum(max(0, loads[i] - original_b[i]) for i in range(m))
        if bad < emergency_bad:
            emergency_bad = bad
            emergency = a.copy()
        if bad or value >= best_cost:
            return False
        best = a.copy()
        best_load = np.asarray(loads, dtype=float)
        best_cost = value
        return True

    def construct(scores, stop):
        """One capacity-price-ordered construction; an unfinished construction is not used as a solution."""
        masked = np.where(allowed, scores, np.inf)
        if m > 1:
            two = np.partition(masked, 1, axis=0)[:2]
            regret = two[1] - two[0]
        else:
            regret = np.zeros(n)
        difficulty = np.min(np.where(allowed, r / scale[:, None], np.inf), axis=0)
        priority = regret + span * difficulty
        order = np.argsort(-priority, kind="stable")
        a = np.full(n, -1, dtype=int)
        loads = np.zeros(m)
        for j in order:
            if time.perf_counter() >= stop:
                return None
            fits = allowed[:, j] & (loads + r[:, j] <= b)
            if not np.any(fits):
                return None
            pressure = np.maximum(0.0, r[:, j]) / np.maximum(1.0, b - loads)
            i = int(np.argmin(np.where(fits, scores[:, j] + span * pressure * 0.12, np.inf)))
            a[j] = i
            loads[i] += r[i, j]
        return a

    def repair(seed, stop):
        """Tabu search on capacity violation, allowing temporary worsening to cross swap barriers."""
        a = np.asarray(seed, dtype=int).copy()
        loads = np.bincount(a, weights=r[a, jobs], minlength=m)
        tabu = np.zeros((m, n), dtype=np.int64)
        tick = 0
        stale = 0
        minimum = float(np.sum(np.maximum(0.0, loads - b) / scale))
        tenure = max(1, int(math.sqrt(n) * strength / 2))
        noise = span / max(1.0, n * strength)
        while time.perf_counter() < stop:
            if np.all(loads <= b):
                record(a)
                return a
            tick += 1
            excess = np.maximum(0.0, loads - b) / scale
            source_after = np.maximum(0.0, loads[a] - r[a, jobs] - b[a]) / scale[a]
            target_after = np.maximum(0.0, loads[:, None] + r - b[:, None]) / scale[:, None]
            delta = target_after - excess[:, None] + source_after[None, :] - excess[a][None, :]
            merit = delta + (c - c[a, jobs]) * (1e-5 / span)
            merit += rng.random((m, n)) * (1e-7 / strength)
            merit[~allowed] = np.inf
            merit[a, jobs] = np.inf
            blocked = (tabu > tick) & (minimum <= float(np.sum(excess)) + delta + 1e-10)
            merit[blocked] = np.inf
            flat = int(np.argmin(merit))
            target, j = divmod(flat, n)
            if not math.isfinite(float(merit[target, j])):
                tabu.fill(0)
                continue
            source = int(a[j])
            loads[source] -= r[source, j]
            loads[target] += r[target, j]
            a[j] = target
            tabu[source, j] = tick + tenure + int(rng.integers(1, tenure + 2))
            violation = float(np.sum(np.maximum(0.0, loads - b) / scale))
            if violation + 1e-12 < minimum:
                minimum = violation
                stale = 0
                record(a)
            else:
                stale += 1
            if stale > max(n, int(n * strength)):
                if lp_x is not None:
                    cumulative = np.cumsum(lp_x, axis=0)
                    a = np.minimum(m - 1, np.sum(cumulative < rng.random(n), axis=0))
                else:
                    scores = shifted + span * r / scale[:, None]
                    scores += rng.exponential(noise, size=(m, n))
                    a = np.argmin(np.where(allowed, scores, np.inf), axis=0)
                loads = np.bincount(a, weights=r[a, jobs], minlength=m)
                tabu.fill(0)
                stale = 0
        record(a)
        return None

    def integer_search(free, seconds, cutoff=True):
        """Fix the remaining tasks and exactly model all residual capacity constraints for the released tasks."""
        if seconds <= 0 or remaining() <= 0:
            return False, False
        free = np.asarray(free, dtype=int)
        k = len(free)
        if not k:
            return False, False
        if best is None:
            base = None
            residual = b.copy()
            fixed_cost = 0
        else:
            base = best.copy()
            residual = b - best_load
            np.add.at(residual, base[free], r[base[free], free])
            fixed_cost = best_cost - sum(original_c[int(base[j])][int(j)] for j in free)
        active = allowed[:, free].copy()
        if nonnegative:
            active &= r[:, free] <= residual[:, None] + 1e-8
        # LP duals give globally valid reduced-cost bounds; only variables that cannot strictly improve are removed.
        if cutoff and best is not None and math.isfinite(lower_bound):
            margin = best_cost - lower_bound - 1.0
            tolerance = 1e-6 * (1.0 + abs(lower_bound))
            active &= reduced[:, free] <= margin + tolerance
        if np.any(~np.any(active, axis=0)):
            return False, bool(k == n and cutoff and best is not None)
        machines, local_jobs = np.nonzero(active)
        global_jobs = free[local_jobs]
        nv = len(machines)
        cols = np.arange(nv, dtype=np.int32)
        values = c[machines, global_jobs]
        rows = [local_jobs, k + machines]
        column = [cols, cols]
        data = [np.ones(nv), r[machines, global_jobs]]
        lb = [np.ones(k), np.full(m, -np.inf)]
        ub = [np.ones(k), residual]
        if cutoff and best is not None:
            # Input costs are integers, so a strict improvement is equivalent to reducing cost by at least 1.
            rows.append(np.full(nv, k + m, dtype=np.int32))
            column.append(cols)
            data.append(values)
            lb.append(np.array([-np.inf]))
            ub.append(np.array([best_cost - fixed_cost - 1]))
        nr = k + m + int(cutoff and best is not None)
        matrix = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(column))),
                            shape=(nr, nv)).tocsc()
        seconds = min(seconds, remaining() * 0.92)
        if seconds <= 0:
            return False, False
        try:
            result = milp(
                values, integrality=np.ones(nv, dtype=np.uint8),
                bounds=Bounds(np.zeros(nv), np.ones(nv)),
                constraints=LinearConstraint(matrix, np.concatenate(lb), np.concatenate(ub)),
                options={"time_limit": float(seconds), "mip_rel_gap": 0.0, "presolve": True},
            )
        except Exception:
            return False, False
        improved = False
        if result.x is not None:
            selected = np.flatnonzero(result.x > 0.5)
            if len(selected) == k and len(np.unique(local_jobs[selected])) == k:
                candidate = np.empty(n, dtype=int) if base is None else base
                candidate[global_jobs[selected]] = machines[selected]
                improved = record(candidate)
        proven = bool(k == n and ((result.status == 0 and improved) or
                                 (result.status == 2 and cutoff and best is not None)))
        return improved, proven

    record(emergency)
    record(np.argmin(np.where(allowed, c, np.inf), axis=0))
    record(construct(shifted + span * r / scale[:, None], min(deadline, start + budget * 0.035)))
    if m == 1:
        return {"assignments": {str(j): 0 for j in range(n)}}

    # The relaxation and the integer model use the same per-task equalities and per-machine capacity inequalities.
    if remaining() > 0:
        nv = m * n
        cols = np.arange(nv, dtype=np.int32)
        row_job = np.tile(jobs, m)
        row_machine = np.repeat(np.arange(m), n)
        eq = coo_matrix((np.ones(nv), (row_job, cols)), shape=(n, nv)).tocsc()
        cap = coo_matrix((r.ravel(), (row_machine, cols)), shape=(m, nv)).tocsc()
        try:
            relaxation = linprog(
                c.ravel(), A_ub=cap, b_ub=b, A_eq=eq, b_eq=np.ones(n),
                bounds=(0, None), method="highs",
                options={"time_limit": max(1e-6, min(remaining() * 0.22, budget * 0.14))},
            )
            if relaxation.status == 0 and relaxation.x is not None:
                lp_x = np.maximum(0.0, relaxation.x.reshape(m, n))
                lp_x /= np.maximum(1e-15, np.sum(lp_x, axis=0))
                lower_bound = float(relaxation.fun)
                prices = -np.asarray(relaxation.ineqlin.marginals)
                reduced = np.maximum(0.0, c + prices[:, None] * r - relaxation.eqlin.marginals[None, :])
                record(np.argmax(lp_x, axis=0))
                record(construct(c + prices[:, None] * r - span * lp_x,
                                 min(deadline, time.perf_counter() + budget * 0.03)))
        except Exception:
            pass

    if best is not None and best_cost - lower_bound < 1.0 - 1e-6:
        return {"assignments": {str(j): int(best[j]) for j in range(n)}}

    # If there is no feasible solution yet, first repair the LP rounding; if that fails, rescue with the full integer model.
    if best is None and remaining() > 0:
        seed = np.argmax(lp_x, axis=0) if lp_x is not None else emergency
        repair(seed, min(deadline, time.perf_counter() + budget * 0.08))
    if remaining() > 0:
        portion = 0.50 if best is None else 0.32
        _, proven = integer_search(jobs, remaining() * portion)
        if proven:
            return {"assignments": {str(j): int(best[j]) for j in range(n)}}

    width = min(n, max(1, int(math.ceil(math.sqrt(n) * strength))))
    base_width = width
    failures = 0
    last_global = time.perf_counter()
    global_spacing = budget / max(2.0, strength)
    while remaining() > 0:
        if best is None:
            # Do not fix most variables, to avoid treating an infeasible LP rounding as a feasible starting point.
            _, proven = integer_search(jobs, remaining() * 0.66, cutoff=False)
            if best is None and remaining() > 0:
                repair(emergency, min(deadline, time.perf_counter() + budget / (8.0 * strength)))
            if proven:
                break
            continue
        if best_cost - lower_bound < 1.0 - 1e-6:
            break
        now = time.perf_counter()
        global_turn = width >= n or now - last_global >= global_spacing
        if global_turn:
            free = jobs
            last_global = now
        else:
            # Release tasks inconsistent with the LP and tasks sharing tight machines; swap-chain length is not limited.
            current = c[best, jobs]
            potential = current - np.min(np.where(allowed, c, np.inf), axis=0)
            uncertainty = np.zeros(n) if lp_x is None else 1.0 - lp_x[best, jobs]
            noise_scale = max(1.0, float(np.mean(potential))) / strength
            priority = potential + span * uncertainty + rng.exponential(noise_scale, n)
            pivot = int(np.argmax(priority))
            destination_score = reduced[:, pivot] + rng.random(m) * noise_scale
            target = int(np.argmin(np.where(allowed[:, pivot], destination_score, np.inf)))
            priority += (best == target) * span / strength
            priority += (best == best[pivot]) * span / (2.0 * strength)
            free = np.argpartition(priority, n - width)[n - width:]
        before = best_cost
        seconds = min(remaining(), budget * (0.06 + 0.025 * math.log1p(failures + strength)))
        if global_turn:
            seconds = min(remaining(), max(seconds, remaining() / strength))
        changed, proven = integer_search(free, seconds)
        if proven:
            break
        if changed and best_cost < before:
            failures = 0
            width = max(base_width, int(width / 1.3))
        else:
            failures += 1
            width = min(n, max(width + 1, int(width * (1.0 + 1.0 / strength))))

    # The saved incumbent is never overwritten by a candidate that fails the integer capacity check.
    answer = best if best is not None else emergency
    return {"assignments": {str(j): int(answer[j]) for j in range(n)}}
