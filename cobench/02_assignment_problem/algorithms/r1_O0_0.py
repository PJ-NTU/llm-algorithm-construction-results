"""Time-limited primal-dual assignment algorithm
First improves a complete permutation with reduced-cost swaps, then solves exactly with vectorised shortest augmenting paths, returning the saved best complete assignment at the deadline.
"""

import math
import time

import numpy as np


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    best = []
    try:
        n = int(instance["n"])
        best = list(range(n))
        budget = max(0.0, float(time_limit_s))
        if n <= 1 or not math.isfinite(budget) or budget <= 0.0:
            return {"assignment": best}

        # Reserve output time; all search phases share the same wall-clock deadline.
        deadline = started + 0.97 * budget
        clock = time.perf_counter
        if clock() >= deadline:
            return {"assignment": best}
        source = instance["cost"]
        cost = np.empty((n, n), dtype=np.float64)
        u = np.empty(n, dtype=np.float64)
        v = np.full(n, np.inf, dtype=np.float64)
        for i in range(n):
            if clock() >= deadline:
                return {"assignment": best}
            cost[i] = source[i]
            u[i] = np.min(cost[i])
            np.minimum(v, cost[i] - u[i], out=v)

        columns = np.arange(n, dtype=np.intp)
        candidate = columns.copy()
        owner = columns.copy()
        column_cost = cost[columns, columns].copy()

        # The swap-phase budget scales with the total time limit; there is no fixed number of sweeps.
        exchange_deadline = min(deadline, started + 0.12 * budget)
        while clock() < exchange_deadline:
            improved = False
            for i in range(n):
                if clock() >= exchange_deadline:
                    break
                old = int(candidate[i])
                delta = cost[i] - cost[i, old] + cost[owner, old] - column_cost
                j = int(np.argmin(delta))
                k = int(owner[j])
                gain = math.fsum((float(cost[i, j]), float(cost[k, old]),
                                  -float(cost[i, old]), -float(cost[k, j])))
                if gain < 0.0:
                    candidate[i], candidate[k] = j, old
                    owner[j], owner[old] = i, k
                    column_cost[j] = cost[i, j]
                    column_cost[old] = cost[k, old]
                    best[i], best[k] = j, old
                    improved = True
            if not improved:
                break

        if clock() >= deadline:
            return {"assignment": best}
        best_value = np.sum(cost[columns, candidate], dtype=np.longdouble)
        row_match = np.full(n, -1, dtype=np.intp)
        col_match = np.full(n, -1, dtype=np.intp)
        root = 0

        # The partial matching is responsible for optimality; candidate is always a complete permutation extension of it.
        # Augmentation can be abandoned immediately if unfinished, returning only the independently saved best.
        while clock() < deadline:
            if root == n:
                # A perfect matching with feasible duals satisfies complementary slackness, so the exact solve is complete.
                return {"assignment": best}

            distance = np.full(n, np.inf, dtype=np.float64)
            predecessor = np.full(n, -1, dtype=np.intp)
            seen = np.zeros(n, dtype=np.bool_)
            visited_rows = []
            row = root
            level = 0.0

            while clock() < deadline:
                visited_rows.append(row)
                reduced = (cost[row] - u[row] - v) + level
                relax = (~seen) & (reduced < distance)
                distance[relax] = reduced[relax]
                predecessor[relax] = row
                j = int(np.argmin(np.where(seen, np.inf, distance)))
                level = float(distance[j])
                if not math.isfinite(level):
                    return {"assignment": best}

                # On ties prefer reaching an unmatched task, avoiding needless expansion along zero-cost edges.
                preferred = int(candidate[root])
                if (not seen[preferred] and col_match[preferred] < 0
                        and distance[preferred] == level):
                    j = preferred
                elif col_match[j] >= 0:
                    free = np.flatnonzero((~seen) & (col_match < 0)
                                          & (distance == level))
                    if free.size:
                        j = int(free[0])
                seen[j] = True
                if col_match[j] < 0:
                    sink = j
                    break
                row = int(col_match[j])
            else:
                return {"assignment": best}

            # Recover the full path first; a timeout during recovery does not break the saved feasible solution.
            path = []
            j = sink
            while clock() < deadline:
                row = int(predecessor[j])
                if row < 0:
                    return {"assignment": best}
                path.append((row, j))
                if row == root:
                    break
                j = int(row_match[row])
            else:
                return {"assignment": best}

            u[root] += level
            if len(visited_rows) > 1:
                rows = np.asarray(visited_rows[1:], dtype=np.intp)
                u[rows] += level - distance[row_match[rows]]
            v[seen] -= level - distance[seen]

            # The augmenting chain plus one unprocessed worker forms a permutation cycle, keeping the assignment complete.
            displaced = int(owner[sink])
            old = int(candidate[root])
            for row, j in path:
                if clock() >= deadline:
                    return {"assignment": best}
                row_match[row] = j
                col_match[j] = row
                candidate[row] = j
                owner[j] = row
            if displaced != root:
                candidate[displaced] = old
                owner[old] = displaced

            value = np.sum(cost[columns, candidate], dtype=np.longdouble)
            if value < best_value:
                best = candidate.tolist()
                best_value = value
            root += 1

        return {"assignment": best}
    except Exception:
        # On exceptions or resource shortage, keep the complete permutation already built.
        return {"assignment": best}
