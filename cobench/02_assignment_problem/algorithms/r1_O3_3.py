"""Shortest-augmenting-path assignment warm-started with auction prices
Generates prices and a complete candidate solution by auction with decreasing epsilon, reuses the prices to build dual labels and refines with shortest augmenting paths, keeping the best permutation within the wall-clock budget.
"""

import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    result = {"assignment": []}
    try:
        n = int(instance["n"])
        result["assignment"] = list(range(n))
        budget = max(0.0, float(time_limit_s))
        deadline = start + 0.97 * budget
        if n < 2 or time.perf_counter() >= deadline:
            return result
        costs = instance["cost"]
        if np is None:
            _swap_search(costs, n, deadline, result)
        else:
            _hybrid(costs, n, start, budget, deadline, result)
    except Exception:
        # Interruption, numerical problems or resource shortage never overwrite the complete permutation already saved.
        pass
    return result


def _swap_search(costs, n, deadline, result):
    """When the general numerical module is unavailable, still maintain a complete feasible solution and improve it on time."""
    rng = random.Random(0)
    assignment = result["assignment"]
    while time.perf_counter() < deadline:
        i = rng.randrange(n)
        k = rng.randrange(n - 1)
        if k >= i:
            k += 1
        j, q = assignment[i], assignment[k]
        delta = (costs[i][q] - costs[i][j]) + (costs[k][j] - costs[k][q])
        if delta < 0:
            assignment[i], assignment[k] = q, j


def _hybrid(costs, n, start, budget, deadline, result):
    clock = time.perf_counter
    matrix = np.empty((n, n), dtype=float)
    magnitude = 0.0
    for i in range(n):
        if clock() >= deadline:
            return
        matrix[i] = costs[i]
        lo, hi = float(matrix[i].min()), float(matrix[i].max())
        if not math.isfinite(lo) or not math.isfinite(hi):
            return
        magnitude = max(magnitude, abs(lo), abs(hi))

    # Row shifts do not change the cost difference between any two complete assignments; extreme magnitudes are rescaled uniformly first.
    scale = magnitude if magnitude > 1e100 or 0 < magnitude < 1e-100 else 1.0
    spread = 0.0
    for i in range(n):
        if clock() >= deadline:
            return
        matrix[i] /= scale
        matrix[i] -= matrix[i].min()
        spread = max(spread, float(matrix[i].max()))

    work = list(result["assignment"])
    inverse = list(range(n))
    best_value = math.fsum(float(matrix[i, work[i]]) for i in range(n))
    work_value = best_value

    def project(i, j):
        # Embed the partial matching into a complete permutation via one swap, so a solution with missing entries is never published.
        nonlocal work_value
        old = work[i]
        if old == j:
            return
        k = inverse[j]
        work_value += float((matrix[i, j] - matrix[i, old]) +
                            (matrix[k, old] - matrix[k, j]))
        work[i], work[k] = j, old
        inverse[j], inverse[old] = i, k

    def publish(force=False):
        nonlocal work_value, best_value
        if force or work_value < best_value:
            work_value = math.fsum(float(matrix[i, work[i]]) for i in range(n))
            if work_value < best_value:
                best_value = work_value
                result["assignment"] = list(work)

    prices = np.zeros(n)
    worker_task = [-1] * n
    task_worker = [-1] * n
    auction_deadline = min(deadline, start + (deadline - start) / (2.0 + math.log1p(n)))
    epsilon = spread / (n + 1.0)
    epsilon_floor = max(math.ulp(spread), math.ulp(epsilon))
    reduction = 2.0 + math.log1p(budget)
    pending = list(range(n - 1, -1, -1))

    while epsilon > 0 and clock() < auction_deadline:
        while pending and clock() < auction_deadline:
            i = pending.pop()
            values = matrix[i] + prices
            j = int(values.argmin())
            first = float(values[j])
            values[j] = np.inf
            second = float(values.min())
            raised = float(prices[j]) + (second - first) + epsilon
            if raised <= prices[j]:
                raised = float(np.nextafter(prices[j], np.inf))
            if not math.isfinite(raised):
                return
            prices[j] = raised
            previous = task_worker[j]
            if previous >= 0:
                worker_task[previous] = -1
                pending.append(previous)
            worker_task[i] = j
            task_worker[j] = i
            project(i, j)
            publish()

        publish(True)
        if pending or epsilon <= epsilon_floor or clock() >= auction_deadline:
            break
        epsilon = max(epsilon_floor, epsilon / reduction)
        prices -= prices.min()
        # At the new epsilon release only edges violating the epsilon condition, keeping prices and the rest of the matching.
        for i in range(n):
            if clock() >= auction_deadline:
                break
            j = worker_task[i]
            if j >= 0:
                values = matrix[i] + prices
                if float(values[j] - values.min()) > epsilon:
                    worker_task[i] = -1
                    task_worker[j] = -1
                    pending.append(i)

    if clock() >= deadline:
        return

    # True primal-dual handover: v=-price, u=min_j(c-v), keeping only tight edges.
    v = -prices
    u = np.empty(n)
    match_worker = [-1] * n
    match_task = np.full(n, -1, dtype=int)
    work = list(result["assignment"])
    inverse = [0] * n
    for i, j in enumerate(work):
        inverse[j] = i
    work_value = best_value
    unmatched = []
    for i in range(n):
        if clock() >= deadline:
            publish(True)
            return
        values = matrix[i] - v
        minimum = float(values.min())
        u[i] = minimum
        j = worker_task[i]
        if j < 0 or match_task[j] >= 0 or values[j] != minimum:
            candidates = np.flatnonzero((values == minimum) & (match_task < 0))
            j = int(candidates[0]) if candidates.size else -1
        if j < 0:
            unmatched.append(i)
        else:
            match_worker[i] = j
            match_task[j] = i
            project(i, j)
    publish(True)

    rng = random.Random(0)
    cycle_width = min(n, max(2, 1 + int(math.log1p(budget) * math.log2(n + 1))))

    # Unmatched rows are refined by shortest augmenting paths; afterwards, keep checking cyclic swaps that reduce cost.
    # No cap on restarts or iterations; all search phases are controlled by the wall clock.
    while clock() < deadline:
        if unmatched:
            root = unmatched.pop()
            distances = matrix[root] - u[root] - v
            parent = np.full(n, root, dtype=int)
            used = np.zeros(n, dtype=bool)
            settled = np.zeros(n)
            endpoint = -1
            while clock() < deadline:
                j = int(distances.argmin())
                distance = float(distances[j])
                if not math.isfinite(distance):
                    return
                used[j] = True
                settled[j] = distance
                distances[j] = np.inf
                i = int(match_task[j])
                if i < 0:
                    endpoint = j
                    break
                candidate = distance + matrix[i] - u[i] - v
                improve = (~used) & (candidate < distances)
                distances[improve] = candidate[improve]
                parent[improve] = i
            if endpoint < 0 or clock() >= deadline:
                return

            visited = np.flatnonzero(used)
            changes = distance - settled[visited]
            v[visited] -= changes
            owners = match_task[visited]
            occupied = owners >= 0
            u[owners[occupied]] += changes[occupied]
            u[root] += distance

            j = endpoint
            while j >= 0:
                if clock() >= deadline:
                    publish(True)
                    return
                i = int(parent[j])
                previous = match_worker[i]
                match_worker[i] = j
                match_task[j] = i
                project(i, j)
                j = previous
            publish(True)
        else:
            # A complete tight-edge matching is done; accept only verifiable improvements, never corrupting the best solution.
            assignment = result["assignment"]
            width = 2 if rng.getrandbits(1) else rng.randint(2, cycle_width)
            workers = rng.sample(range(n), width)
            tasks = [assignment[i] for i in workers]
            delta = math.fsum(float(matrix[workers[k], tasks[(k + 1) % width]] -
                                    matrix[workers[k], tasks[k]]) for k in range(width))
            if delta < 0 and clock() < deadline:
                candidate = list(assignment)
                for k, i in enumerate(workers):
                    candidate[i] = tasks[(k + 1) % width]
                value = math.fsum(float(matrix[i, candidate[i]]) for i in range(n))
                if value < best_value:
                    best_value = value
                    result["assignment"] = candidate
