"""Adaptive greedy and integer-programming neighbourhood search
Constructs a feasible solution by greedy covering and redundancy removal, then improves it with destroy-repair guided by linear relaxation and time-limited integer programming.
"""

import heapq
import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix
except Exception:
    np = None


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.monotonic()
    n = int(instance.get("n", 0))
    answer = {"selection_status": [1] * n}
    try:
        _solve(instance, float(time_limit_s), started, answer)
    except Exception:
        # On valid feasible instances, selecting all is the fallback solution; improved solutions are published only after full verification.
        pass
    return answer


def _solve(instance, budget, started, answer):
    now = time.monotonic
    n, m = int(instance["n"]), int(instance["m"])
    original_cost = instance["costs"]
    covers = instance["covers"]
    budget = max(0.0, budget)
    reserve = min(budget * 0.12, 0.04 + budget * 0.035)
    deadline = started + budget - reserve
    strength = math.log1p(budget)
    rng = random.Random(721347)
    fixed = [int(original_cost[j] <= 0) for j in range(n)]
    for i, row in enumerate(covers):
        if len(row) == 1:
            fixed[row[0]] = 1
        if i % 256 == 0 and now() >= deadline:
            return
    if m == 0:
        answer["selection_status"] = fixed
        return

    # First get an initial solution from the sparse incidence lists, so bitset preprocessing does not eat into the time for reaching feasibility.
    members = [[] for _ in range(n)]
    rows = []
    for i, raw in enumerate(covers):
        if not any(fixed[j] for j in raw):
            row = list(dict.fromkeys(raw))
            if not row:
                return
            index = len(rows)
            rows.append(row)
            for j in row:
                members[j].append(index)
        if i % 256 == 0 and now() >= deadline:
            return
    r = len(rows)
    if not r:
        answer["selection_status"] = fixed
        return
    gains = [len(col) for col in members]
    heap = [(original_cost[j] / g, j) for j, g in enumerate(gains) if g]
    heapq.heapify(heap)
    covered = bytearray(r)
    left = r
    initial = set()
    while left and heap:
        if now() >= deadline:
            return
        old, j = heapq.heappop(heap)
        gain = gains[j]
        if not gain:
            continue
        score = original_cost[j] / gain
        if score > old + 1e-12:
            heapq.heappush(heap, (score, j))
            continue
        initial.add(j)
        for i in members[j]:
            if not covered[i]:
                covered[i] = 1
                left -= 1
                for k in rows[i]:
                    gains[k] -= 1
    if left:
        return
    counts = [0] * r
    for j in initial:
        for i in members[j]:
            counts[i] += 1
    for j in sorted(initial, key=lambda j: original_cost[j], reverse=True):
        if now() >= deadline:
            break
        if all(counts[i] > 1 for i in members[j]):
            initial.remove(j)
            for i in members[j]:
                counts[i] -= 1
    status = fixed.copy()
    for j in initial:
        status[j] = 1
    answer["selection_status"] = status
    if now() >= deadline:
        return

    # Limit preprocessing by actual bitset memory and remaining budget; large sparse instances can still continue with sparse search.
    estimated_bytes = sum((col[-1] // 8 + 1) for col in members if col)
    use_bits = estimated_bytes <= (1.0 + strength) * 33554432
    if not use_bits:
        _sparse_search(original_cost, members, rows, fixed, initial,
                       counts, answer, deadline, strength, rng)
        return
    representative = {}
    for j, col in enumerate(members):
        if not col:
            continue
        mask = 0
        for i in col:
            mask |= 1 << i
        previous = representative.get(mask)
        if previous is None or original_cost[j] < original_cost[previous]:
            representative[mask] = j
        if j % 128 == 0 and now() >= deadline:
            return
    masks = list(representative)
    originals = list(representative.values())
    cost = [original_cost[j] for j in originals]
    cols = [members[j] for j in originals]
    index_of_mask = {mask: j for j, mask in enumerate(masks)}
    selected = set()
    for original in initial:
        mask = 0
        for i in members[original]:
            mask |= 1 << i
        selected.add(index_of_mask[mask])
    q = len(masks)
    all_rows = (1 << r) - 1
    best = selected.copy()
    best_cost = sum(cost[j] for j in best)
    fractions = [0.0] * q

    def count_solution(solution):
        count = [0] * r
        for j in solution:
            for i in cols[j]:
                count[i] += 1
        return count

    def prune(solution, count):
        for j in sorted(solution, key=lambda j: cost[j], reverse=True):
            if now() >= deadline:
                break
            if all(count[i] > 1 for i in cols[j]):
                solution.remove(j)
                for i in cols[j]:
                    count[i] -= 1
        return sum(cost[j] for j in solution)

    def publish(solution, value):
        nonlocal best, best_cost
        if value >= best_cost:
            return False
        union = 0
        for j in solution:
            union |= masks[j]
        if union != all_rows:
            return False
        result = fixed.copy()
        for j in solution:
            result[originals[j]] = 1
        answer["selection_status"] = result
        best, best_cost = solution.copy(), value
        return True

    # Publish the solution after same-coverage column replacement, keeping consistency even without strict improvement.
    status = fixed.copy()
    for j in best:
        status[originals[j]] = 1
    answer["selection_status"] = status
    count = count_solution(best)
    value = prune(selected, count)
    publish(selected, value)
    current, current_count = best.copy(), count_solution(best)
    current_cost = best_cost

    def repair(solution, count, uncovered, noise, bias):
        queue = []
        adjusted = {}
        for j, mask in enumerate(masks):
            gain = (mask & uncovered).bit_count()
            if gain:
                price = cost[j] / (1.0 + bias * fractions[j])
                if noise:
                    price *= math.exp(noise * (rng.random() - 0.5))
                adjusted[j] = price
                queue.append((price / gain, j))
            if j % 256 == 0 and now() >= deadline:
                return False
        heapq.heapify(queue)
        while uncovered and queue:
            if now() >= deadline:
                return False
            old, j = heapq.heappop(queue)
            gain = (masks[j] & uncovered).bit_count()
            if not gain:
                continue
            score = adjusted[j] / gain
            if score > old + 1e-12:
                heapq.heappush(queue, (score, j))
                continue
            solution.add(j)
            for i in cols[j]:
                count[i] += 1
            uncovered &= ~masks[j]
        return not uncovered

    def make_model(uncovered, allowance):
        # Merge identical columns on uncovered rows, keeping only the cheapest representative; the neighbourhood remains exact.
        patterns = {}
        for j, mask in enumerate(masks):
            part = mask & uncovered
            if part and cost[j] <= allowance:
                old = patterns.get(part)
                if old is None or cost[j] < cost[old]:
                    patterns[part] = j
            if j % 256 == 0 and now() >= deadline:
                return None
        if not patterns:
            return None
        row_map = {}
        bits = uncovered
        while bits:
            bit = bits & -bits
            row_map[bit.bit_length() - 1] = len(row_map)
            bits -= bit
        nr = len(row_map)
        indices, values, pointers, variables = [], [], [0], []
        for part, j in patterns.items():
            variables.append(j)
            for i in cols[j]:
                target = row_map.get(i)
                if target is not None:
                    indices.append(target)
                    values.append(1.0)
            indices.append(nr)
            values.append(float(cost[j]))
            pointers.append(len(indices))
            if len(variables) % 128 == 0 and now() >= deadline:
                return None
        matrix = csc_matrix((np.asarray(values), np.asarray(indices, dtype=np.int32),
                             np.asarray(pointers, dtype=np.int32)),
                            shape=(nr + 1, len(variables)))
        lower = np.ones(nr + 1)
        lower[-1] = -np.inf
        upper = np.full(nr + 1, np.inf)
        upper[-1] = allowance
        return variables, matrix, lower, upper

    def optimize(model, allowance, seconds, relaxed=False):
        if model is None or np is None:
            return None
        remaining = deadline - now()
        seconds = min(seconds, remaining * 0.55)
        if seconds <= max(0.005, reserve * 0.15):
            return None
        variables, matrix, lower, upper = model
        upper[-1] = allowance
        try:
            result = milp(
                c=np.asarray([cost[j] for j in variables], dtype=float),
                integrality=np.zeros(len(variables)) if relaxed else np.ones(len(variables)),
                bounds=Bounds(0.0, 1.0),
                constraints=LinearConstraint(matrix, lower, upper),
                options={"time_limit": seconds, "mip_rel_gap": 0.0, "presolve": True},
            )
            return result
        except Exception:
            return None

    lower_bound = 0.0
    if np is not None and deadline - now() > budget * 0.45:
        model = make_model(all_rows, best_cost - 0.5)
        result = optimize(model, best_cost - 0.5, budget * 0.10, True)
        if result is not None:
            if result.status == 2:
                return
            if result.status == 0 and result.x is not None:
                lower_bound = float(result.fun)
                for j, x in zip(model[0], result.x):
                    fractions[j] = max(0.0, min(1.0, float(x)))
                rounded, rc = set(), [0] * r
                if repair(rounded, rc, all_rows, 0.0, 1.0 + strength):
                    value = prune(rounded, rc)
                    publish(rounded, value)
                if best_cost <= math.ceil(lower_bound - 1e-7):
                    return
        result = optimize(model, best_cost - 0.5, budget * 0.18)
        if result is not None:
            if result.x is not None:
                candidate = {j for j, x in zip(model[0], result.x) if x > 0.5}
                publish(candidate, sum(cost[j] for j in candidate))
            if result.status in (0, 2):
                return
    current, current_count = best.copy(), count_solution(best)
    current_cost = best_cost
    last_improvement = now()
    next_exact = now() + budget / (2.0 + strength)

    while now() < deadline:
        clock = now()
        stagnation = (clock - last_improvement) / max(budget, 1e-9)
        exact = (np is not None and clock >= next_exact and
                 deadline - clock > reserve + budget * 0.08)
        source = best if exact or rng.random() < min(0.8, 0.2 + stagnation) else current
        trial = source.copy()
        tc = count_solution(source) if source is best else current_count.copy()
        k = len(trial)
        if not k:
            return
        # Time budget, current solution size and stagnation time jointly determine neighbourhood depth.
        depth = min(k, max(1, int((1.0 + strength) * math.sqrt(k) *
                                 (0.25 + stagnation))))
        if exact:
            depth = min(k, max(depth, int(math.sqrt(k) * (1.0 + strength))))
        else:
            depth = rng.randint(1, depth)
        pivot = rng.choice(tuple(trial))
        pivot_mask = masks[pivot]
        priorities = []
        for j in trial:
            related = (masks[j] & pivot_mask).bit_count() / max(1, masks[j].bit_count())
            priorities.append((related + rng.random() + cost[j] / max(1, current_cost), j))
        priorities.sort(reverse=True)
        missing = 0
        for _, j in priorities[:depth]:
            trial.remove(j)
            for i in cols[j]:
                tc[i] -= 1
                if tc[i] == 0:
                    missing |= 1 << i
        base = trial.copy()
        base_value = sum(cost[j] for j in base)
        if exact and missing and base_value < best_cost:
            model = make_model(missing, best_cost - base_value - 0.5)
            portion = budget / ((2.0 + strength) * (2.0 + math.sqrt(max(1, k / depth))))
            result = optimize(model, best_cost - base_value - 0.5, portion)
            if result is not None and result.x is not None:
                candidate = base | {j for j, x in zip(model[0], result.x) if x > 0.5}
                candidate_value = sum(cost[j] for j in candidate)
                if publish(candidate, candidate_value):
                    last_improvement = now()
            if not base and result is not None and result.status in (0, 2):
                return
            next_exact = now() + budget / (2.0 + strength)
        noise = rng.random() * (0.25 + strength * (0.2 + stagnation))
        bias = rng.random() * (1.0 + strength)
        if not repair(trial, tc, missing, noise, bias):
            continue
        value = prune(trial, tc)
        if publish(trial, value):
            last_improvement = now()
        if best_cost <= math.ceil(lower_bound - 1e-7):
            return
        temperature = max(1e-9, best_cost / max(1, len(best)) *
                          (0.1 + stagnation) / (1.0 + strength))
        if value <= current_cost or rng.random() < math.exp(
                max(-700.0, (current_cost - value) / temperature)):
            current, current_count, current_cost = trial, tc, value


def _sparse_search(cost, members, rows, fixed, initial, counts, answer,
                   deadline, strength, rng):
    """When bitset memory is too large, run the same feasible destroy-repair on the sparse incidence lists."""
    best, best_counts = initial.copy(), counts.copy()
    best_cost = sum(cost[j] for j in best)
    while time.monotonic() < deadline:
        selected, count = best.copy(), best_counts.copy()
        depth = min(len(best), max(1, int(math.sqrt(len(best)) * (1 + strength))))
        if not depth:
            return
        removed = rng.sample(tuple(best), rng.randint(1, depth))
        missing = set()
        for j in removed:
            selected.remove(j)
            for i in members[j]:
                count[i] -= 1
                if not count[i]:
                    missing.add(i)
        gain = {}
        for i in missing:
            for j in rows[i]:
                gain[j] = gain.get(j, 0) + 1
            if time.monotonic() >= deadline:
                return
        adjusted = {j: cost[j] * math.exp(rng.uniform(-0.5, 0.5) * strength)
                    for j in gain}
        queue = [(adjusted[j] / g, j) for j, g in gain.items()]
        heapq.heapify(queue)
        while missing and queue:
            if time.monotonic() >= deadline:
                return
            old, j = heapq.heappop(queue)
            g = gain[j]
            if not g:
                continue
            score = adjusted[j] / g
            if score > old + 1e-12:
                heapq.heappush(queue, (score, j))
                continue
            selected.add(j)
            for i in members[j]:
                count[i] += 1
                if i in missing:
                    missing.remove(i)
                    for p in rows[i]:
                        gain[p] -= 1
        if missing:
            continue
        for j in sorted(selected, key=lambda j: cost[j], reverse=True):
            if time.monotonic() >= deadline:
                break
            if all(count[i] > 1 for i in members[j]):
                selected.remove(j)
                for i in members[j]:
                    count[i] -= 1
        value = sum(cost[j] for j in selected)
        if value < best_cost:
            result = fixed.copy()
            for j in selected:
                result[j] = 1
            answer["selection_status"] = result
            best, best_counts, best_cost = selected, count, value
