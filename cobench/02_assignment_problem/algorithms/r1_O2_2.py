"""Adaptive exact block assignment
Always keeps a complete feasible permutation, re-optimises task blocks with an interruptible primal-dual algorithm, and adjusts the coupling range by wall-clock feedback and time budget.
"""

import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None


def _optimize_block(cost, workers, tasks, stop):
    """Optimise fixed workers and their current tasks; on interruption, complete the partial matching into a full permutation."""
    clock = time.perf_counter
    size = len(workers)
    if np is None:
        matrix = []
    else:
        matrix = np.empty((size, size), dtype=float)

    # Subtracting a constant from each row does not change the ranking of assignments, and narrows the numeric range of the potentials.
    for local_i, worker in enumerate(workers):
        if clock() >= stop:
            return [], False
        row = cost[worker]
        values = [float(row[task]) for task in tasks]
        offset = min(values)
        values = [value - offset for value in values]
        if np is None:
            matrix.append(values)
        else:
            matrix[local_i] = values

    if np is None:
        u = [0.0] * (size + 1)
        v = [0.0] * (size + 1)
        p = [0] * (size + 1)
        way = [0] * (size + 1)
    else:
        u = np.zeros(size + 1)
        v = np.zeros(size + 1)
        p = np.zeros(size + 1, dtype=np.intp)
        way = np.zeros(size + 1, dtype=np.intp)

    completed = 0
    for root in range(1, size + 1):
        if clock() >= stop:
            break
        p[0] = root
        j0 = 0
        interrupted = False
        if np is None:
            slack = [math.inf] * (size + 1)
            used = [False] * (size + 1)
        else:
            slack = np.full(size + 1, math.inf)
            used = np.zeros(size + 1, dtype=bool)

        while True:
            if clock() >= stop:
                interrupted = True
                break
            used[j0] = True
            i0 = int(p[j0])
            if np is None:
                delta = math.inf
                j1 = 0
                row = matrix[i0 - 1]
                for j in range(1, size + 1):
                    # This is the granularity of deadline checks, not a cap on the number of searches.
                    if j % 64 == 0 and clock() >= stop:
                        interrupted = True
                        break
                    if used[j]:
                        continue
                    reduced = row[j - 1] - u[i0] - v[j]
                    if reduced < slack[j]:
                        slack[j] = reduced
                        way[j] = j0
                    if slack[j] < delta or (
                        slack[j] == delta and p[j] == 0 and p[j1] != 0
                    ):
                        delta = slack[j]
                        j1 = j
                if interrupted or not math.isfinite(delta):
                    interrupted = True
                    break
                for j in range(size + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        slack[j] -= delta
            else:
                active = ~used[1:]
                reduced = matrix[i0 - 1] - u[i0] - v[1:]
                improved = active & (reduced < slack[1:])
                slack[1:][improved] = reduced[improved]
                way[1:][improved] = j0
                available = np.where(active, slack[1:], math.inf)
                j1 = int(np.argmin(available)) + 1
                delta = float(available[j1 - 1])
                if not math.isfinite(delta):
                    interrupted = True
                    break
                # For equal-length paths prefer ending at a free task, avoiding heavy degeneracy from equal costs.
                if p[j1] != 0:
                    free = np.flatnonzero((available == delta) & (p[1:] == 0))
                    if free.size:
                        j1 = int(free[0]) + 1
                visited = np.flatnonzero(used)
                u[p[visited]] += delta
                v[visited] -= delta
                slack[1:][active] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        if interrupted:
            break
        # Complete a whole augmenting path at once, so interruption cannot break the existing partial matching.
        while j0:
            previous = int(way[j0])
            p[j0] = p[previous]
            j0 = previous
        completed += 1

    result = [-1] * size
    free_tasks = set(range(size))
    for j in range(1, size + 1):
        worker = int(p[j])
        if worker:
            result[worker - 1] = j - 1
            free_tasks.remove(j - 1)
    # Rows not augmented keep their original assignment where possible; remaining rows are paired one-to-one with remaining tasks.
    pending = []
    for i in range(size):
        if result[i] != -1:
            continue
        if i in free_tasks:
            result[i] = i
            free_tasks.remove(i)
        else:
            pending.append(i)
    for i, j in zip(pending, sorted(free_tasks)):
        result[i] = j
    return result, completed == size


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    assignment = []
    try:
        n = max(0, int(instance["n"]))
        assignment = list(range(n))
        budget = float(time_limit_s)
        if n < 2 or not math.isfinite(budget) or budget <= 0:
            return {"assignment": assignment}
        cost = instance["cost"]
        # Reserve time for completion, comparison and output; the safety margin scales with the call budget.
        deadline = start + budget * 0.97
        rng = random.Random(0)
        width = min(n, max(2, int(16 * math.sqrt(budget))))
        inverse = list(range(n))

        # No fixed cap on restarts or neighbourhoods; all searches are terminated by the wall clock.
        while time.perf_counter() < deadline:
            begin = time.perf_counter()
            remaining = deadline - begin
            rows = rng.sample(range(n), width)
            if width < n:
                # Add the current holders of the tasks most wanted by randomly seeded workers to the coupled block.
                # These links are expanded only if time allows; leaving them unfinished does not affect permutation feasibility.
                selected = set(rows[:max(1, width // 2)])
                seeds = list(selected)
                link_stop = begin + remaining * 0.08
                for worker in seeds:
                    if len(selected) >= width or time.perf_counter() >= link_stop:
                        break
                    row = cost[worker]
                    task = assignment[worker]
                    value = row[task]
                    for j in range(n):
                        if j % 64 == 0 and time.perf_counter() >= link_stop:
                            break
                        if row[j] < value:
                            task, value = j, row[j]
                    selected.add(inverse[task])
                for worker in rows:
                    if len(selected) >= width:
                        break
                    selected.add(worker)
                rows = sorted(selected)
                rng.shuffle(rows)

            tasks = [assignment[i] for i in rows]
            block_stop = deadline if width == n else begin + remaining * 0.45
            local, complete = _optimize_block(cost, rows, tasks, block_stop)
            if local:
                # Compare original costs directly, not objective values with potentials or shifts applied.
                difference = math.fsum(
                    value
                    for k, worker in enumerate(rows)
                    for value in (cost[worker][tasks[local[k]]], -cost[worker][tasks[k]])
                )
                if difference < 0:
                    candidate = assignment.copy()
                    for k, worker in enumerate(rows):
                        candidate[worker] = tasks[local[k]]
                    assignment = candidate
                    for worker in rows:
                        inverse[assignment[worker]] = worker

            elapsed = max(time.perf_counter() - begin, 1e-9)
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if complete:
                # Extrapolate conservatively with cubic complexity, enlarging the exact neighbourhood when the budget is more ample.
                target = min(remaining * 0.4, budget * 0.2)
                factor = min(2.0, max(1.0, (target / elapsed) ** (1.0 / 3.0)))
                width = min(n, max(width, int(width * factor)))
            else:
                # An unfinished large block has still been completed and any improvement kept; then shrink the block to get more feedback.
                width = max(2, int(width * 0.7))
    except Exception:
        # No unfinished computation overwrites the saved complete permutation.
        pass
    return {"assignment": assignment}
