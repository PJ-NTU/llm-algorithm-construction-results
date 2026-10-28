"""Bidirectional parallel iterated greedy scheduling
Combines multi-start NEH, accelerated insertion descent and adaptive destruction-reconstruction, searching by wall clock and keeping the best feasible permutation.
"""

import math
import time

import numpy as np


class _TimeUp(Exception):
    pass


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    result = {"perm": []}
    try:
        n = int(instance["n"])
        result["perm"] = list(range(n))
        budget = max(0.0, float(time_limit_s))
        if n < 2 or budget <= 0.0:
            return result
        deadline = started + budget - min(0.05, 0.02 * budget)
        m = int(instance["n_machines"])
        if m <= 1 or time.perf_counter() >= deadline:
            return result
        p = np.asarray(instance["proc"], dtype=np.float64)
        if m == 2:
            # For two machines, use the optimal Johnson order directly.
            first = [j for j in range(n) if p[j, 0] <= p[j, 1]]
            last = [j for j in range(n) if p[j, 0] > p[j, 1]]
            first.sort(key=lambda j: p[j, 0])
            last.sort(key=lambda j: p[j, 1], reverse=True)
            result["perm"] = first + last
            return result

        def check():
            if time.perf_counter() >= deadline:
                raise _TimeUp

        check()
        # Limit batch memory rather than the number of search rounds; more time allows more trajectories.
        width = max(1, min(n, int(1 + 2 * math.log1p(budget)),
                           max(1, 131072 // max(1, n * m))))
        rows = np.arange(width)
        reversed_lane = (rows % 2 == 1)
        lane_p = np.broadcast_to(p, (width, n, m)).copy()
        lane_p[reversed_lane] = p[:, ::-1]
        rng = np.random.default_rng(104729 + 131 * n + 17 * m)
        totals = p.sum(axis=1)
        mean_p = float(p.mean())
        scale = max(1.0, float(np.abs(p).max()) * (n + m))
        tolerance = np.finfo(np.float64).eps * scale * 16.0
        best_cost = float("inf")
        last_best_time = started

        def remember(seq, costs):
            nonlocal best_cost, last_best_time
            r = int(np.argmin(costs))
            value = float(costs[r])
            if value < best_cost:
                best_cost = value
                chosen = seq[r, ::-1] if reversed_lane[r] else seq[r]
                result["perm"] = chosen.tolist()
                last_best_time = time.perf_counter()

        def evaluate(seq):
            """Exact objective by the earliest-feasible-completion recursion, with no surrogate objective."""
            check()
            length = seq.shape[1]
            if length == 0:
                return np.zeros(width)
            a = lane_p[rows[:, None], seq].transpose(2, 0, 1)
            sums = np.cumsum(a, axis=2)
            finish = sums[0].copy()
            for machine in range(1, m):
                if machine % 8 == 0:
                    check()
                finish[:, 1:] -= sums[machine, :, :-1]
                np.maximum.accumulate(finish, axis=1, out=finish)
                finish += sums[machine]
            return finish[:, -1].copy()

        def insertion_costs(seq, jobs):
            """Prefix, reversed-suffix and Taillard insertion recursions evaluate all positions in a batch."""
            check()
            length = seq.shape[1]
            if length == 0:
                return lane_p[rows, jobs].sum(axis=1)[:, None]
            a = lane_p[rows[:, None], seq].transpose(2, 0, 1)
            both = np.concatenate((a, a[::-1, :, ::-1]), axis=1)
            sums = np.cumsum(both, axis=2)
            heads = np.zeros((m, 2 * width, length + 1))
            heads[0, :, 1:] = sums[0]
            for machine in range(1, m):
                if machine % 8 == 0:
                    check()
                v = heads[machine, :, 1:]
                v[:] = heads[machine - 1, :, 1:]
                v[:, 1:] -= sums[machine, :, :-1]
                np.maximum.accumulate(v, axis=1, out=v)
                v += sums[machine]
            tails = heads[::-1, width:, ::-1]
            times = lane_p[rows, jobs]
            finish = np.zeros((width, length + 1))
            costs = np.zeros_like(finish)
            for machine in range(m):
                if machine % 8 == 0:
                    check()
                np.maximum(finish, heads[machine, :width], out=finish)
                finish += times[:, machine, None]
                np.maximum(costs, finish + tails[machine], out=costs)
            return costs

        def best_positions(costs):
            minimum = costs.min(axis=1)
            # Break ties randomly only when objectives are identical, avoiding bias from fixed positions.
            tie = rng.random(costs.shape)
            tie[costs != minimum[:, None]] = -1.0
            return np.argmax(tie, axis=1), minimum

        def insert(seq, jobs, positions):
            length = seq.shape[1]
            if length == 0:
                return jobs[:, None].copy()
            col = np.arange(length + 1)[None, :]
            idx = col - (col > positions[:, None])
            out = np.take_along_axis(seq, np.minimum(idx, length - 1), axis=1)
            out[rows, positions] = jobs
            return out

        def descend(seq, costs, phase_end):
            """Insertion descent over a random job order; the neighbourhood is reopened after a change."""
            length = seq.shape[1]
            if length < 2:
                return seq, costs
            untested = np.ones((width, n), dtype=bool)
            untested[:] = False
            untested[rows[:, None], seq] = True
            columns = np.arange(length - 1)[None, :]
            while time.perf_counter() < phase_end:
                active = untested.any(axis=1)
                if not active.any():
                    break
                priority = rng.random((width, n))
                priority[~untested] = -1.0
                jobs = np.argmax(priority, axis=1)
                # Converged trajectories hold an existing job as a placeholder but no longer move.
                jobs[~active] = seq[~active, 0]
                old_pos = np.argmax(seq == jobs[:, None], axis=1)
                idx = columns + (columns >= old_pos[:, None])
                shortened = np.take_along_axis(seq, idx, axis=1)
                candidates = insertion_costs(shortened, jobs)
                positions, values = best_positions(candidates)
                original = candidates[rows, old_pos]
                improving = active & (values < original - tolerance)
                untested[rows, jobs] = False
                if improving.any():
                    moved = insert(shortened, jobs, positions)
                    seq[improving] = moved[improving]
                    costs[improving] = values[improving]
                    untested[improving] = False
                    rr = rows[improving]
                    untested[rr[:, None], seq[improving]] = True
                    untested[rr, jobs[improving]] = False
                    if length == n:
                        remember(seq, costs)
            return seq, costs

        # A permutation is equivalent to x[j][k] = 1 if and only if perm[k] = j;
        # insertion, removal and reconstruction all keep a one-to-one assignment, and the completion recursion satisfies all precedence constraints.
        order = np.empty((width, n), dtype=np.int64)
        ramp = np.linspace(-1.0, 1.0, m)
        for r in range(width):
            if r < 2:
                priority = totals
            elif r % 3 == 2:
                priority = (lane_p[r] * (1.0 + 0.35 * ramp)).sum(axis=1)
            elif r % 3 == 0:
                priority = totals + 0.5 * p.std(axis=1)
            else:
                priority = totals * (0.94 + 0.12 * rng.random(n))
            order[r] = np.argsort(-priority, kind="stable")
        current = order.copy()
        current_cost = evaluate(current)
        remember(current, current_cost)

        partial = np.empty((width, 0), dtype=np.int64)
        construction_end = min(deadline, started + 0.25 * budget)
        k = 0
        while k < n and time.perf_counter() < construction_end:
            jobs = order[:, k]
            positions, _ = best_positions(insertion_costs(partial, jobs))
            partial = insert(partial, jobs, positions)
            k += 1
        constructed = np.concatenate((partial, order[:, k:]), axis=1)
        constructed_cost = evaluate(constructed)
        better = constructed_cost <= current_cost
        current[better] = constructed[better]
        current_cost[better] = constructed_cost[better]
        remember(current, current_cost)
        current, current_cost = descend(
            current, current_cost,
            min(deadline, time.perf_counter() + 0.3 * budget))
        personal = current.copy()
        personal_cost = current_cost.copy()
        personal_time = np.full(width, time.perf_counter())
        base_d = max(2, int(2 + math.log1p(n) / 3 + math.log1p(budget) / 4))
        base_d = min(n - 1, base_d)
        thermal = max(tolerance, mean_p * (0.02 + 0.015 * math.log1p(budget)))
        temperature_factors = np.linspace(0.6, 1.5, width)
        round_time = max(0.001, (time.perf_counter() - started) / max(1, n))

        # The main loop has no round cap; apart from the time limit, the only early exit is the degenerate case with a known optimum.
        while time.perf_counter() < deadline:
            round_start = time.perf_counter()
            stale = max(0.0, round_start - last_best_time)
            growth = math.log1p(stale / max(round_time, 0.001))
            extra = int((1 + math.log1p(budget)) * growth / 3)
            d = min(n - 1, base_d + int(rng.integers(extra + 1)))
            # Mix small adjacent perturbations with scattered destruction to balance local intensification and escaping local optima.
            if rng.random() < 1.0 / (1.0 + math.sqrt(base_d)):
                d = min(n - 1, max(2, base_d // 2))
            selected = np.argpartition(rng.random((width, n)), d - 1, axis=1)[:, :d]
            for r in range(width):
                if rng.random() < 1.0 / (1.0 + base_d):
                    start = int(rng.integers(n - d + 1))
                    selected[r] = np.arange(start, start + d)
            removed = np.take_along_axis(current, selected, axis=1)
            keep = np.ones((width, n), dtype=bool)
            keep[rows[:, None], selected] = False
            candidate = current[keep].reshape(width, n - d).copy()
            # Some trajectories rebuild in order of total processing time; the others keep a random rebuild order.
            ranked = np.argsort(-totals[removed], axis=1)
            ranked_removed = np.take_along_axis(removed, ranked, axis=1)
            removed[::2] = ranked_removed[::2]
            for j in range(d):
                positions, candidate_cost = best_positions(
                    insertion_costs(candidate, removed[:, j]))
                candidate = insert(candidate, removed[:, j], positions)
            remember(candidate, candidate_cost)
            remaining = max(0.0, deadline - time.perf_counter())
            candidate, candidate_cost = descend(
                candidate, candidate_cost,
                min(deadline, time.perf_counter() + max(round_time * 2,
                                                        remaining / math.sqrt(n))))
            improved = candidate_cost < personal_cost - tolerance
            personal[improved] = candidate[improved]
            personal_cost[improved] = candidate_cost[improved]
            personal_time[improved] = time.perf_counter()
            fraction = max(0.0, (deadline - time.perf_counter()) / budget)
            temperatures = (thermal * temperature_factors *
                            (0.35 + 0.65 * fraction) * (1.0 + growth / 4))
            delta = np.maximum(0.0, candidate_cost - current_cost)
            acceptance = np.exp(-delta / np.maximum(temperatures, tolerance))
            accept = (candidate_cost <= current_cost) | (rng.random(width) < acceptance)
            current[accept] = candidate[accept]
            current_cost[accept] = candidate_cost[accept]
            now = time.perf_counter()
            round_time = 0.8 * round_time + 0.2 * max(0.0001, now - round_start)
            restart_after = math.sqrt(max(0.000001, budget * round_time))
            for r in range(width):
                if now - personal_time[r] > restart_after:
                    if rng.random() < 0.5:
                        chosen = result["perm"]
                        current[r] = chosen[::-1] if reversed_lane[r] else chosen
                        current_cost[r] = best_cost
                    else:
                        current[r] = personal[r]
                        current_cost[r] = personal_cost[r]
                    personal_time[r] = now
    except Exception:
        # On timeout or numerical/resource errors, return the complete feasible permutation saved earlier.
        pass
    return result
