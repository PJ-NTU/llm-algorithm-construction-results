"""Interruptible primal-dual assignment algorithm
Optimises the assignment with dense shortest augmenting paths, maintaining a complete permutation at every augmentation, keeping the better solution within the wall-clock budget and reusing dual potentials for further refinement.
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
    best = []
    try:
        n = int(instance["n"])
        best = list(range(n))
        budget = float(time_limit_s)
        if n < 2 or budget <= 0.0 or not math.isfinite(budget):
            return {"assignment": best}
        costs = instance["cost"]
        deadline = start + 0.97 * budget
        clock = time.perf_counter
        if clock() >= deadline:
            return {"assignment": best}

        def value(assignment):
            return math.fsum(costs[i][assignment[i]] for i in range(n))

        best_value = value(best)
        candidate = best.copy()
        inverse = best.copy()
        cached_rows = [None] * n
        # Pure-Python scans also check the wall clock in segments; check granularity adapts to scale and budget.
        stride = max(1, min(n, int(math.sqrt(n * (1.0 + budget)))))

        def row(i):
            if np is None:
                return costs[i]
            if cached_rows[i] is None:
                cached_rows[i] = np.asarray(costs[i], dtype=np.float64)
            return cached_rows[i]

        def swap_to(i, j):
            old_j = candidate[i]
            other = inverse[j]
            candidate[i], candidate[other] = j, old_j
            inverse[j], inverse[old_j] = i, other

        # The greedy phase uses only a small part of the budget; every intermediate state is still a complete permutation.
        seed_deadline = min(deadline, start + 0.06 * budget)
        available = [True] * n if np is None else np.ones(n, dtype=bool)
        i = 0
        while clock() < seed_deadline and i < n:
            c = row(i)
            if np is not None:
                choices = np.where(available, c, math.inf)
                j = int(np.argmin(choices))
            else:
                j, smallest = -1, math.inf
                for k in range(n):
                    if k % stride == 0 and clock() >= seed_deadline:
                        break
                    if available[k] and c[k] < smallest:
                        j, smallest = k, c[k]
                if j < 0:
                    break
            swap_to(i, j)
            available[j] = False
            i += 1
        candidate_value = value(candidate)
        if candidate_value < best_value:
            best_value, best = candidate_value, candidate.copy()

        # Dual potentials do not depend on the worker processing order, so later rounds can reuse task potentials directly.
        u = [0.0] * n
        v = [0.0] * n if np is None else np.zeros(n)
        order = list(range(n))
        rng = random.Random(0)
        position = n
        row_match = []
        col_match = []

        # No fixed number of restarts; all augmentation and later refinement is driven by the wall clock.
        while clock() < deadline:
            if position == n:
                if row_match:
                    rng.shuffle(order)
                candidate = best.copy()
                inverse = [0] * n
                for i, j in enumerate(candidate):
                    inverse[j] = i
                row_match = [-1] * n
                col_match = [-1] * n if np is None else np.full(n, -1, dtype=np.int64)
                position = 0

            root = order[position]
            c = row(root)
            if np is not None:
                dist = c - v
                u[root] = float(np.min(dist))
                dist -= u[root]
                parent = np.full(n, root, dtype=np.int64)
                used = np.zeros(n, dtype=bool)
            else:
                dist = [c[j] - v[j] for j in range(n)]
                u[root] = min(dist)
                dist = [d - u[root] for d in dist]
                parent = [root] * n
                used = [False] * n
            closed = []
            endpoint = -1

            while clock() < deadline:
                if np is not None:
                    j = int(np.argmin(dist))
                    d = float(dist[j])
                    # At equal distance prefer free tasks, avoiding needless expansion from equivalent costs.
                    if col_match[j] >= 0:
                        tied_free = np.flatnonzero((dist == d) & (col_match < 0))
                        if tied_free.size:
                            j = int(tied_free[0])
                else:
                    j, d = -1, math.inf
                    for k in range(n):
                        if k % stride == 0 and clock() >= deadline:
                            return {"assignment": best}
                        if not used[k] and (dist[k] < d or (
                            dist[k] == d and col_match[k] < 0
                        )):
                            j, d = k, dist[k]
                if j < 0 or not math.isfinite(d):
                    return {"assignment": best}
                if col_match[j] < 0:
                    endpoint = j
                    break

                used[j] = True
                dist[j] = math.inf
                closed.append((j, d))
                worker = int(col_match[j])
                c = row(worker)
                if np is not None:
                    proposed = (c - u[worker] - v) + d
                    improve = (~used) & (proposed < dist)
                    dist[improve] = proposed[improve]
                    parent[improve] = worker
                else:
                    for k in range(n):
                        if k % stride == 0 and clock() >= deadline:
                            return {"assignment": best}
                        if not used[k]:
                            proposed = c[k] - u[worker] - v[k] + d
                            if proposed < dist[k]:
                                dist[k] = proposed
                                parent[k] = worker

            if endpoint < 0 or clock() >= deadline:
                break

            # Update potentials only after a complete augmenting path is found, never accessing the dummy worker of a free task.
            u[root] += d
            for j, settled_distance in closed:
                if clock() >= deadline:
                    return {"assignment": best}
                change = d - settled_distance
                v[j] -= change
                u[int(col_match[j])] += change

            j = endpoint
            while j >= 0:
                if clock() >= deadline:
                    return {"assignment": best}
                worker = int(parent[j])
                previous = row_match[worker]
                # Implement the augmentation projection with swaps, so unprocessed workers also always hold a unique task.
                swap_to(worker, j)
                row_match[worker] = j
                col_match[j] = worker
                j = previous
            position += 1
            candidate_value = value(candidate)
            if candidate_value < best_value:
                best_value, best = candidate_value, candidate.copy()
    except Exception:
        # Numerical or optional-acceleration failures on valid input also do not break the saved complete permutation.
        pass
    return {"assignment": best}
