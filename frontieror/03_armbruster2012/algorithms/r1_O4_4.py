"""Capacity-pruned joint neighbourhood search
Alternates incremental tabu search, weight-indexed compound swaps and time-limited branch and bound on induced subgraphs, keeping the best feasible partition.
"""
import bisect
import heapq
import math
import random
import time


def _valid(bits, weights, lower, upper):
    return 0 < sum(bits) < len(bits) and lower <= sum(
        w for w, b in zip(weights, bits) if b
    ) <= upper


def _initial(weights, lower, upper, deadline, rng):
    n = len(weights)
    fallback = [1] + [0] * (n - 1)
    if n < 2 or lower <= 0:
        return fallback
    order = sorted(range(n), key=lambda i: (-weights[i], i))
    loads = [0, 0]
    counts = [0, 0]
    bits = [0] * n
    for i in order:
        side = 0 if (loads[0], counts[0]) <= (loads[1], counts[1]) else 1
        loads[side] += weights[i]
        counts[side] += 1
        bits[i] = side
    if _valid(bits, weights, lower, upper):
        return bits
    best_violation = float('inf')
    for seq in (order, order[::-1], range(n)):
        bits = [0] * n
        load = 0
        for i in seq:
            if load >= lower:
                break
            if load + weights[i] <= upper:
                load += weights[i]
                bits[i] = 1
        if _valid(bits, weights, lower, upper):
            return bits
        violation = max(0, lower - load, load - upper)
        if 0 < sum(bits) < n and violation < best_violation:
            fallback, best_violation = bits, violation

    # Complementary symmetry allows searching only for subsets weighing at most half of the total.
    half = sum(weights) // 2
    positive = [i for i in order if 0 < weights[i] <= half]
    divisor = 0
    for i in positive:
        divisor = math.gcd(divisor, weights[i])
    divisor = max(1, divisor)
    lo = (lower + divisor - 1) // divisor
    hi = min(upper, half) // divisor
    if hi >= lo and hi <= 8_000_000 and time.perf_counter() < deadline:
        # Store bitsets in blocks and rebuild only the current block when backtracking, avoiding storing large bitsets per node.
        block = max(1, math.isqrt(len(positive)))
        checkpoints = [1]
        reachable = 1
        mask = (1 << (hi + 1)) - 1
        for p, i in enumerate(positive):
            reachable = (reachable | (reachable << (weights[i] // divisor))) & mask
            available = reachable >> lo
            if available:
                target = lo + (available & -available).bit_length() - 1
                bits = [0] * n
                end = p + 1
                while end > 0:
                    start = ((end - 1) // block) * block
                    states = [checkpoints[start // block]]
                    for q in range(start, end):
                        v = states[-1]
                        states.append((v | (v << (weights[positive[q]] // divisor))) & mask)
                    for q in range(end - 1, start - 1, -1):
                        if not ((states[q - start] >> target) & 1):
                            bits[positive[q]] = 1
                            target -= weights[positive[q]] // divisor
                    end = start
                if _valid(bits, weights, lower, upper):
                    return bits
            if (p + 1) % block == 0:
                checkpoints.append(reachable)
            if time.perf_counter() >= deadline:
                break

    # Large integer weights use sparse branching search; suffix weights and the greatest common divisor both allow pruning.
    items = positive
    suffix = [0] * (len(items) + 1)
    gcds = [0] * (len(items) + 1)
    for p in range(len(items) - 1, -1, -1):
        suffix[p] = suffix[p + 1] + weights[items[p]]
        gcds[p] = math.gcd(gcds[p + 1], weights[items[p]])
    stack = [(0, 0, 0)]
    seen = set()
    while stack and time.perf_counter() < deadline:
        p, load, chosen = stack.pop()
        if lower <= load <= min(upper, half):
            bits = [0] * n
            for q, i in enumerate(items):
                bits[i] = (chosen >> q) & 1
            if _valid(bits, weights, lower, upper):
                return bits
        if p == len(items) or load > half or load + suffix[p] < lower:
            continue
        g = gcds[p]
        if g and (max(0, lower - load) + g - 1) // g > (half - load) // g:
            continue
        key = (p, load)
        if key in seen:
            continue
        if len(seen) < 200_000:
            seen.add(key)
        stack.append((p + 1, load, chosen))
        if load + weights[items[p]] <= half:
            stack.append((p + 1, load + weights[items[p]], chosen | (1 << p)))
    return fallback


class _State:
    __slots__ = ('bits', 'load', 'count', 'value', 'gain')

    def __init__(self, bits, weights, adj):
        self.bits = bits[:]
        self.load = sum(w for w, b in zip(weights, bits) if b)
        self.count = sum(bits)
        self.gain = [0.0] * len(bits)
        value = 0.0
        for i, neighbors in enumerate(adj):
            d = 0.0
            for j, c in neighbors:
                if bits[i] == bits[j]:
                    d += c
                else:
                    d -= c
                    if i < j:
                        value += c
            self.gain[i] = d
        self.value = value

    def flip(self, i, weights, adj):
        old = self.bits[i]
        d = self.gain[i]
        self.value += d
        for j, c in adj[i]:
            self.gain[j] += -2.0 * c if old == self.bits[j] else 2.0 * c
        self.gain[i] = -d
        self.bits[i] = 1 - old
        sign = 1 - 2 * old
        self.load += sign * weights[i]
        self.count += sign


def _optimize(weights, edges, costs, lower, upper, initial, deadline, budget, rng, save):
    n = len(weights)
    maps = [{} for _ in weights]
    for (u, v), c in zip(edges, costs):
        if u != v:
            maps[u][v] = maps[u].get(v, 0.0) + c
            maps[v][u] = maps[v].get(u, 0.0) + c
    adj = [[(j, c) for j, c in row.items() if c] for row in maps]
    current = _State(initial, weights, adj)
    best = initial[:]
    best_value = current.value
    started = time.perf_counter()
    strength = math.log2(2.0 + budget)
    eps = 1e-11 * max(1.0, sum(abs(c) for c in costs))
    elite_cap = max(2, int(2 + 2 * strength))
    elite = [(best_value, best[:])]

    def record(state):
        nonlocal best, best_value
        if lower <= state.load <= upper and 0 < state.count < n and state.value < best_value - eps:
            actual = math.fsum(c for (u, v), c in zip(edges, costs) if state.bits[u] != state.bits[v])
            state.value = actual
            if actual < best_value - eps:
                best_value = actual
                best = state.bits[:]
                save(best)
                elite.append((actual, best[:]))
                return True
        return False

    def pool(group, state, width):
        if len(group) <= width:
            return group[:]
        q = max(1, width // 4)
        chosen = heapq.nsmallest(width, group, key=state.gain.__getitem__)
        chosen += rng.sample(group, min(q, len(group)))
        chosen += heapq.nsmallest(q, group, key=weights.__getitem__)
        chosen += heapq.nlargest(q, group, key=weights.__getitem__)
        return list(dict.fromkeys(chosen))

    def compound(state, until, width):
        # Same-side single and double node combinations are joined by total weight, covering three- and four-node swaps uniformly.
        while time.perf_counter() < until:
            sides = [[i for i in range(n) if state.bits[i] == b] for b in (1, 0)]
            selections = [pool(side, state, width) for side in sides]
            records = []
            for selected in selections:
                row = [(weights[i], state.gain[i], (i,)) for i in selected]
                for p, a in enumerate(selected):
                    for b in selected[p + 1:]:
                        row.append((weights[a] + weights[b], state.gain[a] + state.gain[b]
                                    - 2.0 * maps[a].get(b, 0.0), (a, b)))
                records.append(row)
            records[1].sort(key=lambda r: r[0])
            ws = [r[0] for r in records[1]]
            chosen = None
            best_delta = -eps
            # Random rotation keeps time slices from always being taken by the first candidates.
            rng.shuffle(records[0])
            for wa, da, aa in records[0]:
                if time.perf_counter() >= until:
                    break
                left = bisect.bisect_left(ws, lower - state.load + wa)
                right = bisect.bisect_right(ws, upper - state.load + wa)
                for p in range(left, right):
                    wb, db, bb = records[1][p]
                    count = state.count - len(aa) + len(bb)
                    if 0 < count < n:
                        delta = da + db
                        for a in aa:
                            for b in bb:
                                delta += 2.0 * maps[a].get(b, 0.0)
                        if delta < best_delta:
                            best_delta, chosen = delta, aa + bb
                    if (p & 127) == 0 and time.perf_counter() >= until:
                        break
            if chosen is None:
                break
            for i in chosen:
                state.flip(i, weights, adj)
            record(state)
        return state

    def select_block(state, size, serial):
        if size >= n - 1:
            # The whole-graph neighbourhood fixes one node, removing the complementary symmetry of the groups.
            return list(range(1, n))
        one = [i for i in range(n) if state.bits[i]]
        zero = [i for i in range(n) if not state.bits[i]]
        if serial % 3 == 0:
            a = min(len(one), size // 2)
            b = min(len(zero), size - a)
            a = min(len(one), size - b)
            spread = max(eps, sum(abs(v) for v in state.gain) / max(1, n))
            jitter = spread * (0.05 + 0.15 * rng.random())
            return heapq.nsmallest(a, one, key=lambda i: state.gain[i] + jitter * rng.random()) + heapq.nsmallest(b, zero, key=lambda i: state.gain[i] + jitter * rng.random())
        weak = heapq.nsmallest(min(n, max(2, size)), range(n), key=state.gain.__getitem__)
        root = weak[rng.randrange(len(weak))]
        selected = [root]
        chosen = {root}
        frontier = []
        for j, c in adj[root]:
            heapq.heappush(frontier, (-abs(c) * (0.7 + rng.random()), j))
        quota = [min(len(zero), (size + 1) // 2), min(len(one), (size + 1) // 2)]
        taken = [0, 0]
        taken[state.bits[root]] += 1
        while frontier and len(selected) < size and time.perf_counter() < deadline:
            _, i = heapq.heappop(frontier)
            if i in chosen or taken[state.bits[i]] >= quota[state.bits[i]]:
                continue
            selected.append(i)
            chosen.add(i)
            taken[state.bits[i]] += 1
            for j, c in adj[i]:
                if j not in chosen:
                    heapq.heappush(frontier, (-abs(c) * (0.7 + rng.random()), j))
        candidates = heapq.nsmallest(min(n, 2 * size), range(n), key=state.gain.__getitem__)
        for i in candidates:
            if len(selected) >= size:
                break
            if i not in chosen:
                selected.append(i)
                chosen.add(i)
        return selected

    def exact_block(state, selected, until):
        # Boundary edges are computed once; branching updates only the two costs of unassigned nodes inside the block.
        k = len(selected)
        if k == 0 or time.perf_counter() >= until:
            return state, False
        inside = set(selected)
        selected.sort(key=lambda i: -(sum(abs(c) for j, c in adj[i] if j in inside)
                                      + abs(state.gain[i]) * 0.2))
        pos = {i: p for p, i in enumerate(selected)}
        h0 = [0.0] * k
        h1 = [0.0] * k
        forward = [[] for _ in selected]
        for p, i in enumerate(selected):
            for j, c in adj[i]:
                q = pos.get(j)
                if q is None:
                    if state.bits[j]:
                        h0[p] += c
                    else:
                        h1[p] += c
                elif q > p:
                    forward[p].append((q, c))
        original = [state.bits[i] for i in selected]
        ws = [weights[i] for i in selected]
        fixed_load = state.load - sum(w for w, b in zip(ws, original) if b)
        fixed_count = state.count - sum(original)
        low = lower - fixed_load
        high = upper - fixed_load
        low_count = max(0, 1 - fixed_count)
        high_count = min(k, n - 1 - fixed_count)
        suffix = [0] * (k + 1)
        gcds = [0] * (k + 1)
        negative = [0.0] * (k + 1)
        for p in range(k - 1, -1, -1):
            suffix[p] = suffix[p + 1] + ws[p]
            gcds[p] = math.gcd(gcds[p + 1], ws[p])
            negative[p] = negative[p + 1] + sum(min(0.0, c) for _, c in forward[p])
        base_energy = sum(h1[p] if original[p] else h0[p] for p in range(k))
        base_energy += sum(c for p in range(k) for q, c in forward[p] if original[p] != original[q])
        local_best = base_energy
        assignment = original[:]
        best_assignment = original[:]
        # A one-dimensional Lagrange multiplier gives an extra capacity lower bound, independent of the sign of edge costs.
        preferred_load = sum(ws[p] for p in range(k) if h1[p] < h0[p])
        multiplier = 0.0
        target = high if preferred_load > high else low
        if preferred_load < low or preferred_load > high:
            thresholds = sorted(((h0[p] - h1[p]) / ws[p], ws[p]) for p in range(k) if ws[p])
            remaining = suffix[0]
            for ratio, w in thresholds:
                remaining -= w
                if remaining <= target:
                    multiplier = ratio
                    break
        dual0 = [multiplier * w for w in ws]
        timed_out = False
        visits = 0

        def dfs(p, load, count, value):
            nonlocal local_best, best_assignment, timed_out, visits
            visits += 1
            if (visits & 31) == 1 and time.perf_counter() >= until:
                timed_out = True
                return
            if load > high or load + suffix[p] < low or count > high_count or count + k - p < low_count:
                return
            g = gcds[p]
            if g and (max(0, low - load) + g - 1) // g > (high - load) // g:
                return
            if p == k:
                if low <= load <= high and low_count <= count <= high_count and value < local_best - eps:
                    local_best = value
                    best_assignment = assignment[:]
                return
            bound = value + negative[p]
            dual = bound - multiplier * ((high if multiplier > 0 else low) - load)
            for q in range(p, k):
                bound += min(h0[q], h1[q])
                if multiplier:
                    dual += min(h0[q], h1[q] + dual0[q])
            if bound >= local_best - eps or (multiplier and dual >= local_best - eps):
                return
            prefer = original[p] if abs(h0[p] - h1[p]) < eps else int(h1[p] + dual0[p] < h0[p])
            for bit in (prefer, 1 - prefer):
                assignment[p] = bit
                next_value = value + (h1[p] if bit else h0[p])
                field = h0 if bit else h1
                for q, c in forward[p]:
                    field[q] += c
                dfs(p + 1, load + bit * ws[p], count + bit, next_value)
                for q, c in forward[p]:
                    field[q] -= c
                if timed_out:
                    return
        dfs(0, 0, 0, 0.0)
        if local_best < base_energy - eps:
            for i, bit in zip(selected, best_assignment):
                if state.bits[i] != bit:
                    state.flip(i, weights, adj)
            record(state)
        return state, not timed_out

    tabu = [0] * n
    step = 0
    stagnant = 0
    restarts = 0
    deep_serial = 0
    deep_spent = 0.0
    next_deep = started
    block_size = min(n - 1, max(4, int(13 + strength)))
    block_cap = min(n - 1, max(4, int(19 + 3 * strength)))
    sqrt_n = math.sqrt(n + 1)
    last_improvement = started

    # The overall search is driven by wall clock; if time remains after all rounds, the search continues.
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        step += 1
        scale = strength + math.log2(2.0 + now - started) * 0.5
        width = max(4, int(5 + 2 * scale + math.log1p(n) * 0.5))
        one = []
        zero = []
        chosen = None
        change = float('inf')
        for i in range(n):
            b = current.bits[i]
            (one if b else zero).append(i)
            sign = 1 - 2 * b
            load = current.load + sign * weights[i]
            count = current.count + sign
            d = current.gain[i]
            if lower <= load <= upper and 0 < count < n and (tabu[i] <= step or current.value + d < best_value - eps):
                if d < change:
                    chosen, change = (i,), d
            if (i & 255) == 255 and time.perf_counter() >= deadline:
                break
        if time.perf_counter() >= deadline:
            break
        if len(one) * len(zero) <= n * width * 2:
            pa, pb = one, zero
        else:
            pa, pb = pool(one, current, width), pool(zero, current, width)
        # Under tight capacity, add weight-matched candidates from the whole groups instead of pairing only two small pools.
        narrow = upper - lower < sum(weights) / max(1, n)
        indexed_zero = sorted(zero, key=weights.__getitem__) if narrow else []
        zero_weights = [weights[i] for i in indexed_zero]
        for a in pa:
            candidates = pb
            if narrow:
                left = bisect.bisect_left(zero_weights, lower - current.load + weights[a])
                right = bisect.bisect_right(zero_weights, upper - current.load + weights[a])
                matched = indexed_zero[left:right]
                candidates = matched if len(matched) <= width else heapq.nsmallest(width, matched, key=current.gain.__getitem__)
            for b in candidates:
                load = current.load - weights[a] + weights[b]
                if lower <= load <= upper:
                    d = current.gain[a] + current.gain[b] + 2.0 * maps[a].get(b, 0.0)
                    if (tabu[a] <= step and tabu[b] <= step) or current.value + d < best_value - eps:
                        if d < change:
                            chosen, change = (a, b), d
            if time.perf_counter() >= deadline:
                break
        if time.perf_counter() >= deadline:
            break

        # Use the joint neighbourhood from the first stagnation on; the share of expensive steps is controlled by actual time spent.
        if (chosen is None or change >= -eps) and now >= next_deep and deep_spent < 0.42 * (now - started) + 0.025 * budget:
            deep_start = time.perf_counter()
            base = best if deep_serial % 3 != 2 else current.bits
            deep = _State(base, weights, adj)
            before = best_value
            compound_end = min(deadline, deep_start + budget / (65.0 + 5.0 * strength))
            deep = compound(deep, compound_end, max(4, int(6 + 2 * strength)))
            block_end = min(deadline, time.perf_counter() + budget / (28.0 + 2.0 * strength))
            selected = select_block(deep, block_size, deep_serial)
            deep, complete = exact_block(deep, selected, block_end)
            block_size = min(block_cap, block_size + 1) if complete else max(4, block_size - 1)
            deep_serial += 1
            deep_spent += time.perf_counter() - deep_start
            next_deep = time.perf_counter() + budget / (70.0 + 8.0 * strength)
            if best_value < before - eps:
                current = _State(best, weights, adj)
                tabu = [0] * n
                stagnant = 0
                last_improvement = time.perf_counter()
                continue
            if time.perf_counter() >= deadline:
                break

        if chosen is not None:
            for i in chosen:
                current.flip(i, weights, adj)
            tenure = max(2, int(sqrt_n * 0.6 + scale + rng.random() * (scale + 1)))
            for i in chosen:
                tabu[i] = step + tenure
            if record(current):
                stagnant = 0
                last_improvement = time.perf_counter()
            else:
                stagnant += 1
        else:
            stagnant += max(1, int(sqrt_n))
        if len(elite) > elite_cap:
            elite.sort(key=lambda item: item[0])
            unique = []
            signatures = set()
            for value, bits in elite:
                signature = bytes(bits if not bits[0] else [1 - b for b in bits])
                if signature not in signatures:
                    unique.append((value, bits))
                    signatures.add(signature)
                if len(unique) >= elite_cap:
                    break
            elite[:] = unique

        stagnation_limit = max(8, int((3 + scale) * sqrt_n))
        if chosen is None or stagnant >= stagnation_limit:
            restarts += 1
            base = elite[rng.randrange(len(elite))][1] if len(elite) > 1 and rng.random() < 0.35 else best
            current = _State(base, weights, adj)
            tabu = [0] * n
            stagnant = 0
            # Perturbation magnitude grows with budget, size and actual stagnation time; attempts are bounded only by time slices.
            stale_fraction = (time.perf_counter() - last_improvement) / max(budget, 1e-9)
            target = min(n, max(2, int(2 + scale + n * min(0.3, stale_fraction) * rng.random())))
            kick_end = min(deadline, time.perf_counter() + budget / (150.0 + 10.0 * strength))
            accepted = 0
            while accepted < target and time.perf_counter() < kick_end:
                i = rng.randrange(n)
                sign = 1 - 2 * current.bits[i]
                load = current.load + sign * weights[i]
                count = current.count + sign
                if rng.random() < 0.4 and lower <= load <= upper and 0 < count < n:
                    current.flip(i, weights, adj)
                    accepted += 1
                    record(current)
                    continue
                j = rng.randrange(n)
                if current.bits[i] != current.bits[j]:
                    load = current.load + sign * (weights[i] - weights[j])
                    if lower <= load <= upper:
                        current.flip(i, weights, adj)
                        current.flip(j, weights, adj)
                        accepted += 1
                        record(current)
    save(best)


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    n = int(instance['num_nodes'])
    m = int(instance['num_edges'])
    weights = [int(v) for v in instance['node_weights']]
    edges = [(int(e[0]), int(e[1])) for e in instance['edges']]
    costs = [float(c) for c in instance['edge_costs']]
    upper = int(instance['bisection_capacity'])
    lower = sum(weights) - upper
    budget = max(0.0, float(time_limit_s))
    if not math.isfinite(budget):
        budget = 0.0
    reserve = min(budget * 0.1, 0.005 + 0.000002 * (n + m))
    deadline = started + max(0.0, budget - reserve)
    rng = random.Random(0xC0FFEE ^ (n * 1000003) ^ (m * 9176) ^ sum(weights))
    best = [1] + [0] * max(0, n - 1)

    def save(bits):
        nonlocal best
        best = bits[:]

    try:
        best = _initial(weights, lower, upper, deadline, rng)
        if n >= 2 and _valid(best, weights, lower, upper) and time.perf_counter() < deadline:
            _optimize(weights, edges, costs, lower, upper, best, deadline, budget, rng, save)
    except BaseException:
        # The search state is separate from the output snapshot; an interrupted optimisation never corrupts the saved feasible solution.
        pass
    return {'node_groups': [1 if b else 2 for b in best[:n]],
            'cut_edges': [int(best[u] != best[v]) for u, v in edges[:m]]}
