"""Relaxation-guided adaptive assignment search
Combines linear relaxation, integer repair, reduced-cost screening and adaptive large neighbourhood search, improving feasible solutions with migrations and swaps.
"""

import math
import random
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    budget = max(0.0, float(time_limit_s))
    reserve = min(0.15, max(0.003, budget * 0.025))
    deadline = started + max(0.0, budget - reserve)
    n = int(instance["num_jobs"])
    m = int(instance["num_machines"])
    answer = {"assignments": {str(j): 0 for j in range(n)}}
    if n == 0:
        return answer
    if m <= 0:
        return answer

    try:
        # Data are organised by job internally; the two input matrices are still interpreted as machine x job.
        raw_c = instance["cost_matrix"]
        raw_r = instance["resource_matrix"]
        raw_b = instance["capacities"]
        c = np.asarray(raw_c, dtype=np.float64).T.copy()
        r = np.asarray(raw_r, dtype=np.float64).T.copy()
        capacity = np.asarray(raw_b, dtype=np.float64)
        jobs = np.arange(n)
        rng = np.random.default_rng(731)
        prng = random.Random(731)
        nonnegative = bool(np.all(r >= 0))
        eligible = r <= capacity[None, :] if nonnegative else np.ones((n, m), bool)
        masked_c = np.where(eligible, c, np.inf)
        base = np.argmin(masked_c, axis=1)
        best = None
        best_cost = math.inf
        lower_bound = -math.inf
        reduced = None
        lp_x = None
        lp_base = None
        proven = False
        strength = math.log1p(budget)
        cost_scale = max(1.0, float(np.mean(np.ptp(c, axis=1))))
        resource_scale = np.maximum(1.0, np.abs(capacity))

        def remaining():
            return deadline - time.perf_counter()

        def loads(a):
            return np.bincount(a, weights=r[jobs, a], minlength=m)

        def accept(a):
            nonlocal best, best_cost, answer, proven
            if a is None or len(a) != n:
                return False
            # Final feasibility and objective are both re-checked on the input integers, without relying on solver tolerance.
            use = [0] * m
            value = 0
            for j in range(n):
                i = int(a[j])
                if i < 0 or i >= m:
                    return False
                use[i] += int(raw_r[i][j])
                value += int(raw_c[i][j])
            if any(use[i] > int(raw_b[i]) for i in range(m)):
                return False
            if value < best_cost:
                best = np.asarray(a, dtype=np.int64).copy()
                best_cost = value
                answer = {"assignments": {str(j): int(best[j]) for j in range(n)}}
                if math.isfinite(lower_bound) and value <= math.ceil(lower_bound):
                    proven = True
                return True
            return False

        # If the cheapest assignment without capacity limits is feasible, it is already globally optimal.
        for j in range(n):
            answer["assignments"][str(j)] = int(base[j])
        if accept(base):
            return answer

        def greedy(stop, prices=None):
            attempt = 0
            while time.perf_counter() < stop:
                if prices is not None and attempt % 3 == 0:
                    score = c + r * prices[None, :]
                else:
                    weight = 0.0 if attempt == 0 else math.exp(prng.uniform(-2.0, 2.5))
                    score = c + weight * cost_scale * (n / max(1, m)) * r / resource_scale
                score = np.where(eligible, score, np.inf)
                ordered = np.sort(score, axis=1)
                regret = ordered[:, 1] - ordered[:, 0] if m > 1 else np.ones(n)
                regret = np.nan_to_num(regret, nan=0.0, posinf=cost_scale * (n + 1))
                priority = regret / cost_scale + 1.0 / np.maximum(1, eligible.sum(axis=1))
                if attempt:
                    priority *= rng.uniform(0.3, 1.7, size=n)
                order = np.argsort(-priority)
                a = np.full(n, -1, dtype=np.int64)
                free = capacity.copy()
                complete = True
                for pos, j in enumerate(order):
                    if pos % 64 == 0 and time.perf_counter() >= stop:
                        complete = False
                        break
                    choices = eligible[j] & (r[j] <= free)
                    if not np.any(choices):
                        complete = False
                        break
                    marginal = score[j]
                    if attempt % 3 == 2:
                        marginal = marginal + cost_scale * r[j] / np.maximum(1.0, free)
                    i = int(np.argmin(np.where(choices, marginal, np.inf)))
                    a[j] = i
                    free[i] -= r[j, i]
                if complete:
                    accept(a)
                    return
                attempt += 1

        greedy(min(deadline, started + budget * 0.025))

        # LP capacity dual prices are used to screen alternative machines and choose jobs to release.
        if remaining() > 0.005:
            edge_j, edge_i = np.nonzero(eligible)
            edge_count = len(edge_j)
            columns = np.arange(edge_count)
            assignment_matrix = coo_matrix(
                (np.ones(edge_count), (edge_j, columns)), shape=(n, edge_count)
            ).tocsr()
            resource_matrix = coo_matrix(
                (r[edge_j, edge_i], (edge_i, columns)), shape=(m, edge_count)
            ).tocsr()
            offsets = np.min(masked_c, axis=1)
            shifted = c[edge_j, edge_i] - offsets[edge_j]
            lp_time = min(remaining() * 0.3, budget * 0.13)
            try:
                relaxation = linprog(
                    shifted,
                    A_ub=resource_matrix,
                    b_ub=capacity,
                    A_eq=assignment_matrix,
                    b_eq=np.ones(n),
                    bounds=(0.0, None),
                    method="highs-ds",
                    options={"time_limit": max(0.001, lp_time), "presolve": True},
                )
                if relaxation.success and relaxation.x is not None:
                    lp_x = np.zeros((n, m))
                    lp_x[edge_j, edge_i] = relaxation.x
                    lp_base = np.argmax(lp_x, axis=1)
                    value = float(relaxation.fun + np.sum(offsets))
                    lower_bound = value - 1e-7 * max(1.0, abs(value))
                    prices = -np.asarray(relaxation.ineqlin.marginals)
                    reduced = c - offsets[:, None] + r * prices[None, :]
                    reduced -= np.asarray(relaxation.eqlin.marginals)[:, None]
                    reduced = np.where(eligible, np.maximum(0.0, reduced), np.inf)
                    accept(lp_base)
                    if best is not None and best_cost <= math.ceil(lower_bound):
                        proven = True
                    if best is None:
                        greedy(min(deadline, time.perf_counter() + budget * 0.02), prices)
            except Exception:
                pass

        if proven:
            return answer

        def exact_neighborhood(anchor, free_jobs, seconds, restricted=False):
            nonlocal proven
            end = min(deadline, time.perf_counter() + seconds)
            free_jobs = np.unique(np.asarray(free_jobs, dtype=np.int64))
            q = len(free_jobs)
            if q == 0 or time.perf_counter() + 0.002 >= end:
                return False, False
            residual = loads(anchor)
            np.add.at(residual, anchor[free_jobs], -r[free_jobs, anchor[free_jobs]])
            residual = capacity - residual
            rcost = c[free_jobs]
            res = r[free_jobs]
            possible = eligible[free_jobs].copy()
            if nonnegative:
                possible &= res <= residual[None, :] + 1e-8
            if reduced is not None and best is not None:
                # Safe reduced-cost edge removal for the full model: never removes solutions that could improve the upper bound.
                gap = max(0.0, best_cost - lower_bound)
                possible &= reduced[free_jobs] <= gap + 1e-5
            if restricted and m > 1:
                ranking = reduced[free_jobs] if reduced is not None else rcost
                width = min(m, max(2, int(math.ceil(1.0 + strength))))
                choices = np.argpartition(ranking, width - 1, axis=1)[:, :width]
                limited = np.zeros((q, m), dtype=bool)
                limited[np.arange(q)[:, None], choices] = True
                limited[np.arange(q), anchor[free_jobs]] = True
                if lp_x is not None:
                    limited |= lp_x[free_jobs] > 1e-8
                possible &= limited
            if not np.all(np.any(possible, axis=1)):
                return False, True
            ej, ei = np.nonzero(possible)
            k = len(ej)
            col = np.arange(k)
            matrix = coo_matrix(
                (
                    np.concatenate((np.ones(k), res[ej, ei])),
                    (np.concatenate((ej, q + ei)), np.concatenate((col, col))),
                ),
                shape=(q + m, k),
            ).tocsc()
            low = np.concatenate((np.ones(q), np.full(m, -np.inf)))
            high = np.concatenate((np.ones(q), residual))
            offset = np.min(np.where(possible, rcost, np.inf), axis=1)
            objective = rcost[ej, ei] - offset[ej]
            available = end - time.perf_counter() - 0.001
            if available <= 0.001:
                return False, False
            try:
                result = milp(
                    objective,
                    integrality=np.ones(k, dtype=np.uint8),
                    bounds=Bounds(np.zeros(k), np.ones(k)),
                    constraints=LinearConstraint(matrix, low, high),
                    options={
                        "time_limit": available,
                        "mip_rel_gap": 0.0,
                        "presolve": True,
                    },
                )
                improved = False
                if result.x is not None:
                    selected = np.flatnonzero(result.x > 0.5)
                    if len(selected) == q and len(np.unique(ej[selected])) == q:
                        candidate = anchor.copy()
                        candidate[free_jobs[ej[selected]]] = ei[selected]
                        improved = accept(candidate)
                        if result.status == 0 and q == n and not restricted and best is not None:
                            # The global optimality proof is adopted only if the returned solution passes the integer check.
                            exact_cost = sum(int(raw_c[int(candidate[j])][j]) for j in range(n))
                            if exact_cost == best_cost:
                                proven = True
                return improved, result.status in (0, 2)
            except Exception:
                return False, False

        def descent(anchor, stop):
            a = anchor.copy()
            slack = capacity - loads(a)
            while time.perf_counter() < stop:
                delta = c - c[jobs, a][:, None]
                possible = r <= slack[None, :] + 1e-8
                possible[jobs, a] = False
                delta = np.where(possible, delta, np.inf)
                p = int(np.argmin(delta))
                j, i = divmod(p, m)
                if delta[j, i] < -1e-8:
                    old = int(a[j])
                    slack[old] += r[j, old]
                    slack[i] -= r[j, i]
                    a[j] = i
                    continue
                changed = False
                machines = list(range(m))
                prng.shuffle(machines)
                groups = [np.flatnonzero(a == i) for i in range(m)]
                # Chunking only controls temporary matrix memory; it does not limit search iterations.
                block = max(1, int(math.sqrt(max(1, n) * (1.0 + strength))))
                for ai, i in enumerate(machines):
                    if changed or time.perf_counter() >= stop:
                        break
                    left = groups[i]
                    if not len(left):
                        continue
                    for h in machines[ai + 1:]:
                        if changed or time.perf_counter() >= stop:
                            break
                        right = groups[h]
                        if not len(right):
                            continue
                        for start in range(0, len(left), block):
                            if time.perf_counter() >= stop:
                                break
                            jj = left[start:start + block]
                            gain = (c[jj, h] - c[jj, i])[:, None]
                            gain = gain + (c[right, i] - c[right, h])[None, :]
                            feasible = r[right, i][None, :] <= slack[i] + r[jj, i][:, None] + 1e-8
                            feasible &= r[jj, h][:, None] <= slack[h] + r[right, h][None, :] + 1e-8
                            gain = np.where(feasible, gain, np.inf)
                            flat = int(np.argmin(gain))
                            u, v = divmod(flat, len(right))
                            if gain[u, v] < -1e-8:
                                j, k = int(jj[u]), int(right[v])
                                slack[i] += r[j, i] - r[k, i]
                                slack[h] += r[k, h] - r[j, h]
                                a[j], a[k] = h, i
                                changed = True
                                break
                if not changed:
                    break
            return accept(a)

        # First release the LP's fractional jobs, then add jobs with low substitution cost that can relieve capacity.
        min_size = min(n, max(1, int(math.ceil(math.sqrt(n) * (1.0 + strength) / 2.0))))
        neighborhood_size = min(
            n, max(min_size, int(math.ceil(math.sqrt(n) * (2.0 + strength) / math.sqrt(math.log1p(m) + 1.0))))
        )
        if lp_x is not None and remaining() > 0.004:
            fractional = np.flatnonzero(lp_x[jobs, lp_base] < 1.0 - 1e-7)
            if len(fractional):
                alternatives = reduced.copy()
                alternatives[jobs, lp_base] = np.inf
                flexibility = np.min(alternatives, axis=1)
                active_machines = np.any(lp_x[fractional] > 1e-7, axis=0)
                flexibility *= np.where(active_machines[lp_base], 0.25, 1.0)
                flexibility[fractional] = -np.inf
                q = min(n, max(neighborhood_size, len(fractional)))
                free_jobs = np.argsort(flexibility)[:q]
                exact_neighborhood(lp_base, free_jobs, min(remaining() * 0.25, budget * 0.09))

        if best is not None and not proven and remaining() > 0.004:
            descent(best, min(deadline, time.perf_counter() + budget * 0.025))

        # The full integer model is responsible for crossing local neighbourhoods and may directly give a proof of optimality.
        if not proven and remaining() > 0.004:
            anchor = best if best is not None else (lp_base if lp_base is not None else base)
            fraction = 0.40 if best is not None else 0.52
            exact_neighborhood(anchor, jobs, remaining() * fraction)

        def repair(anchor, stop):
            a = anchor.copy()
            used = loads(a)
            tabu = np.zeros((n, m), dtype=np.int64)
            tick = 0
            tenure = max(1, int(math.sqrt(n) * (1.0 + strength) / 2.0))
            least = math.inf
            while time.perf_counter() < stop:
                violation = np.maximum(0.0, used - capacity) / resource_scale
                total = float(np.sum(violation))
                if total < 1e-12 and accept(a):
                    return
                if total < 1e-12:
                    return
                least = min(least, total)
                release = violation[a] - np.maximum(
                    0.0, used[a] - r[jobs, a] - capacity[a]
                ) / resource_scale[a]
                added = (np.maximum(0.0, used[None, :] + r - capacity[None, :]) / resource_scale)
                change = added - violation[None, :] - release[:, None]
                allowed = eligible & ((tabu <= tick) | (total + change < least - 1e-10))
                allowed[jobs, a] = False
                tie = (c - c[jobs, a][:, None]) / cost_scale
                score = change + tie * (1e-5 / max(1, n))
                score += rng.uniform(0.0, 1e-8, size=(n, m))
                score = np.where(allowed, score, np.inf)
                flat = int(np.argmin(score))
                j, i = divmod(flat, m)
                if not math.isfinite(float(score[j, i])):
                    tabu.fill(0)
                    continue
                old = int(a[j])
                used[old] -= r[j, old]
                used[i] += r[j, i]
                a[j] = i
                tabu[j, old] = tick + tenure + prng.randrange(tenure + 1)
                tick += 1

        def select_jobs(size, mode):
            if size >= n:
                return jobs
            a = best
            if reduced is not None:
                current_rc = reduced[jobs, a]
                alt = reduced.copy()
                alt[jobs, a] = np.inf
                flexibility = np.min(alt, axis=1)
                flexibility = np.nan_to_num(flexibility, posinf=cost_scale * (n + 1))
            else:
                current_rc = c[jobs, a] - np.min(masked_c, axis=1)
                flexibility = np.zeros(n)
            positive = current_rc[current_rc > 1e-8]
            unit = max(1.0, float(np.mean(positive)) if len(positive) else cost_scale)
            weights = 0.15 + current_rc / unit + 0.8 / (1.0 + flexibility / unit)
            if mode == 0 and lp_x is not None:
                weights += (1.0 - lp_x[jobs, a]) * (2.0 + strength)
            elif mode == 1:
                machine_weight = np.bincount(a, weights=weights, minlength=m) + 1e-8
                seed = int(rng.choice(m, p=machine_weight / machine_weight.sum()))
                group = a == seed
                weights *= np.where(group, 5.0 + strength, 1.0)
                # Low-cost migrations between machines connect the neighbourhood.
                pricing = reduced if reduced is not None else c - np.min(c, axis=1)[:, None]
                weights += (2.0 + strength) / (1.0 + np.maximum(0.0, pricing[:, seed]) / unit)
                if np.any(group):
                    alternatives = pricing[group].copy()
                    alternatives[:, seed] = np.inf
                    target = np.argmin(alternatives, axis=1)
                    connections = np.bincount(target, minlength=m).astype(float)
                    if np.sum(connections):
                        other = int(rng.choice(m, p=connections / connections.sum()))
                        weights[a == other] *= 3.0 + strength / 2.0
            elif mode == 2:
                # Uniformly random neighbourhoods avoid always locking the same set of LP-integral variables.
                weights = np.ones(n)
            else:
                slack = capacity - loads(a)
                target = np.argmin(np.where(eligible, c, np.inf), axis=1)
                weights += np.maximum(0.0, c[jobs, a] - c[jobs, target]) / cost_scale
                weights *= 1.0 + r[jobs, a] / np.maximum(1.0, slack[a] + r[jobs, a])
            keys = -np.log(np.maximum(rng.random(n), 1e-300)) / np.maximum(weights, 1e-12)
            return np.argpartition(keys, size - 1)[:size]

        nominal_slice = budget / (5.0 + math.log1p(n * max(1, m)))
        turn = 0
        while not proven and remaining() > 0.004:
            left = remaining()
            if best is None:
                anchor = lp_base if lp_base is not None else base
                repair(anchor, min(deadline, time.perf_counter() + min(left * 0.13, budget * 0.04)))
                if best is not None:
                    continue
                greedy(min(deadline, time.perf_counter() + min(remaining() * 0.06, budget * 0.02)))
                if best is not None:
                    continue
                exact_neighborhood(
                    anchor, jobs, min(remaining() * 0.65, nominal_slice * (1.0 + math.sqrt(turn + 1))),
                    restricted=(turn % 2 == 0),
                )
            else:
                size = max(1, min(n, int(neighborhood_size)))
                subset = select_jobs(size, turn % 4)
                allotment = nominal_slice * math.sqrt(size / max(1, min_size))
                allotment = min(remaining() * 0.85, allotment)
                before = time.perf_counter()
                improved, exhausted = exact_neighborhood(best, subset, allotment)
                elapsed = time.perf_counter() - before
                if improved and not proven:
                    descent(best, min(deadline, time.perf_counter() + min(budget * 0.02, remaining() * 0.12)))
                    neighborhood_size = max(min_size, int(size * 0.9))
                elif exhausted:
                    neighborhood_size = min(n, max(size + 1, int(size * (1.15 + 0.03 * strength))))
                elif elapsed >= allotment * 0.85:
                    neighborhood_size = max(min_size, int(size * 0.8))
            turn += 1
        return answer
    except Exception:
        # Keep the previously fully verified solution; the incumbent is not lost if later improvement fails.
        return answer
