"""Vectorised augmenting assignment algorithm that keeps a feasible solution
Uses time-limited greedy construction and primal-dual shortest augmenting paths, saving the best solution as a complete permutation and re-augmenting in a loop with the remaining time.
"""

import math
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
        if n <= 1 or budget <= 0 or not math.isfinite(budget) or np is None:
            return {"assignment": best}

        # Leave a return margin that scales with the budget; any unfinished augmentation does not affect the best solution.
        deadline = start + 0.96 * budget
        if time.perf_counter() >= deadline:
            return {"assignment": best}
        costs = instance["cost"]
        dtype = np.longdouble
        rows = [None] * n

        def row(i):
            if rows[i] is None:
                rows[i] = np.asarray(costs[i], dtype=dtype)
            return rows[i]

        current = best.copy()
        owner = list(range(n))
        chosen = np.asarray([costs[i][i] for i in range(n)], dtype=dtype)
        best_value = np.sum(chosen, dtype=dtype)

        def put(i, j):
            # A swap always keeps a complete permutation; the selected costs are updated in sync.
            old = current[i]
            if old != j:
                other = owner[j]
                current[i], current[other] = j, old
                owner[j], owner[old] = i, other
                chosen[i] = costs[i][j]
                chosen[other] = costs[other][old]

        # Greedy uses only a small part of the budget, rather than unconditionally completing a quadratic initialisation.
        available = np.ones(n, dtype=bool)
        greedy_deadline = min(deadline, start + 0.12 * budget)
        i = 0
        while time.perf_counter() < greedy_deadline and i < n:
            j = int(np.argmin(np.where(available, row(i), np.inf)))
            put(i, j)
            available[j] = False
            value = np.sum(chosen, dtype=dtype)
            if value < best_value:
                best, best_value = current.copy(), value
            i += 1

        if time.perf_counter() >= deadline:
            return {"assignment": best}
        current = best.copy()
        for i, j in enumerate(current):
            owner[j] = i
            chosen[i] = costs[i][j]

        # Dual constraints u[i] + v[j] <= cost[i][j]; matched edges stay tight.
        u = np.zeros(n, dtype=dtype)
        v = np.zeros(n, dtype=dtype)
        match_row = np.full(n, -1, dtype=np.intp)
        match_col = np.full(n, -1, dtype=np.intp)
        root = 0

        # Not capped by rounds. The first round builds the globally optimal matching step by step, then releases one row at a time to re-augment.
        while time.perf_counter() < deadline:
            old = int(match_row[root])
            if old >= 0:
                match_col[old] = -1
                match_row[root] = -1

            reduced = row(root) - v
            u[root] = np.min(reduced)
            distance = reduced - u[root]
            parent = np.full(n, root, dtype=np.intp)
            visited = np.zeros(n, dtype=bool)
            settled_cols = []
            settled_distances = []
            free = match_col < 0

            while time.perf_counter() < deadline:
                j = int(np.argmin(distance))
                d = distance[j]
                if not np.isfinite(d):
                    return {"assignment": best}

                # At equal distance prefer ending at a free column, avoiding fruitless expansion caused by equal costs.
                if not free[j]:
                    tied_free = free & (distance == d)
                    if np.any(tied_free):
                        j = int(np.argmax(tied_free))

                if free[j]:
                    end = j
                    break

                visited[j] = True
                settled_cols.append(j)
                settled_distances.append(d)
                worker = int(match_col[j])
                tentative = d + (row(worker) - u[worker] - v)
                improve = (~visited) & (tentative < distance)
                np.copyto(distance, tentative, where=improve)
                parent[improve] = worker
                distance[j] = np.inf
            else:
                return {"assignment": best}

            if time.perf_counter() >= deadline:
                return {"assignment": best}

            # Update only settled occupied columns; free columns never index the fictitious worker -1.
            u[root] += d
            if settled_cols:
                columns = np.asarray(settled_cols, dtype=np.intp)
                increments = d - np.asarray(settled_distances, dtype=dtype)
                v[columns] -= increments
                u[match_col[columns]] += increments

            # Extend the partial exact matching to a complete permutation, keeping feasible placements for not-yet-matched workers.
            j = end
            while j >= 0:
                if time.perf_counter() >= deadline:
                    return {"assignment": best}
                worker = int(parent[j])
                previous = int(match_row[worker])
                match_row[worker] = j
                match_col[j] = worker
                put(worker, j)
                j = previous

            # Recompute the selected cost directly each time, avoiding drift from accumulating swap deltas over a long time.
            value = np.sum(chosen, dtype=dtype)
            if value < best_value:
                best, best_value = current.copy(), value
            root = (root + 1) % n

    except Exception:
        # Even if numerical or resource operations fail, return only the previously saved complete feasible permutation.
        pass
    return {"assignment": best}
