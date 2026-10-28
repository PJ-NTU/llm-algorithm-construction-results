"""Capacity-aware fused variable-neighbourhood tabu search
Drives tabu search with incremental flip costs, fusing weight-matched compound swaps, bounded exact subneighbourhoods and elite-guided perturbation, and keeps the best feasible partition.
"""
import bisect
import heapq
import math
import random
import time


def _valid(bits, weights, lower, upper):
    return (0 < sum(bits) < len(bits)
            and lower <= sum(w for w, b in zip(weights, bits) if b) <= upper)


def _initial(weights, lower, upper, deadline, rng):
    n = len(weights)
    fallback = [1] + [0] * (n - 1)
    if lower <= 0 or n < 2:
        return fallback
    order = sorted(range(n), key=lambda i: -weights[i])
    bits = [0] * n
    loads, counts = [0, 0], [0, 0]
    for i in order:
        side = min((0, 1), key=lambda s: (loads[s], counts[s]))
        loads[side] += weights[i]
        counts[side] += 1
        bits[i] = side
    if _valid(bits, weights, lower, upper):
        return bits
    fallback = bits
    violation = max(0, lower - loads[1], loads[1] - upper)
    for seq in (order, order[::-1], list(range(n))):
        bits = [0] * n
        load = 0
        for i in seq:
            if load >= lower:
                break
            if load + weights[i] <= upper:
                bits[i] = 1
                load += weights[i]
        if _valid(bits, weights, lower, upper):
            return bits
        v = max(0, lower - load, load - upper)
        if 0 < sum(bits) < n and v < violation:
            fallback, violation = bits, v

    # Using group symmetry, only subsets weighing at most half of the total need to be searched.
    divisor = 0
    for w in weights:
        divisor = math.gcd(divisor, w)
    divisor = max(1, divisor)
    low = (lower + divisor - 1) // divisor
    high = sum(weights) // (2 * divisor)
    items = [(i, weights[i] // divisor) for i in order
             if 0 < weights[i] // divisor <= high]
    block = max(1, math.isqrt(len(items)))
    # Store bitsets in blocks and recompute only the containing block on reconstruction, avoiding storing large bitsets per item.
    memory_bits = 192_000_000
    if (high + 1) * (block + len(items) // block + 2) <= memory_bits:
        mask = (1 << (high + 1)) - 1
        reachable = 1
        checkpoints = [1]
        target = None
        used = 0
        for p, (_, w) in enumerate(items):
            reachable = (reachable | (reachable << w)) & mask
            used = p + 1
            available = reachable >> low
            if available:
                target = low + (available & -available).bit_length() - 1
                break
            if used % block == 0:
                checkpoints.append(reachable)
            if time.perf_counter() >= deadline:
                break
        if target is not None:
            bits = [0] * n
            while used:
                start = ((used - 1) // block) * block
                state = checkpoints[start // block]
                history = [state]
                for p in range(start, used):
                    state = (state | (state << items[p][1])) & mask
                    history.append(state)
                for p in range(used - 1, start - 1, -1):
                    if not ((history[p - start] >> target) & 1):
                        i, w = items[p]
                        bits[i] = 1
                        target -= w
                used = start
            if _valid(bits, weights, lower, upper):
                return bits

    # Large integer weights use a subset search bounded by remaining weight and the greatest common divisor.
    length = len(items)
    suffix = [0] * (length + 1)
    gcds = [0] * (length + 1)
    for p in range(length - 1, -1, -1):
        suffix[p] = suffix[p + 1] + items[p][1]
        gcds[p] = math.gcd(gcds[p + 1], items[p][1])
    stack = [(0, 0, 0)]
    seen = set()
    while stack and time.perf_counter() < deadline:
        p, load, chosen = stack.pop()
        if low <= load <= high:
            bits = [0] * n
            while chosen:
                bit = chosen & -chosen
                bits[items[bit.bit_length() - 1][0]] = 1
                chosen -= bit
            return bits
        if p == length or load > high or load + suffix[p] < low:
            continue
        g = gcds[p]
        if g and (low - load + g - 1) // g > (high - load) // g:
            continue
        key = (p, load)
        if key in seen:
            continue
        # The cache limit only bounds memory and does not stop the search.
        if len(seen) < memory_bits // 512:
            seen.add(key)
        stack.append((p + 1, load, chosen))
        if load + items[p][1] <= high:
            stack.append((p + 1, load + items[p][1], chosen | (1 << p)))
    return fallback


class _State:
    __slots__ = ('bits', 'load', 'count', 'value', 'gain')

    def __init__(self, bits, weights, adj):
        self.bits = bits[:]
        self.load = sum(w for w, b in zip(weights, bits) if b)
        self.count = sum(bits)
        self.gain = [sum(c if bits[i] == bits[j] else -c for j, c in row)
                     for i, row in enumerate(adj)]
        self.value = math.fsum(c for i, row in enumerate(adj)
                              for j, c in row if i < j and bits[i] != bits[j])

    def flip(self, i, weights, adj):
        old = self.bits[i]
        self.value += self.gain[i]
        for j, c in adj[i]:
            self.gain[j] += -2.0 * c if old == self.bits[j] else 2.0 * c
        self.gain[i] = -self.gain[i]
        self.bits[i] = 1 - old
        direction = 1 - 2 * old
        self.load += direction * weights[i]
        self.count += direction


def _pool(group, size, state, weights, rng):
    if len(group) <= size:
        return group[:]
    q = max(1, size // 4)
    result = heapq.nsmallest(size, group, key=state.gain.__getitem__)
    result += rng.sample(group, min(q, len(group)))
    result += heapq.nsmallest(q, group, key=weights.__getitem__)
    result += heapq.nlargest(q, group, key=weights.__getitem__)
    return list(dict.fromkeys(result))


def _compound(state, pools, weights, maps, lower, upper, tabu, step,
              best_value, end, incumbent_move, incumbent_delta):
    gain = state.gain
    move, delta = incumbent_move, incumbent_delta
    eps = 1e-12 * (1.0 + abs(best_value))
    pair_lists = []
    for pool in pools:
        pairs = [(weights[a] + weights[b],
                  gain[a] + gain[b] - 2.0 * maps[a].get(b, 0.0), a, b)
                 for p, a in enumerate(pool) for b in pool[p + 1:]]
        pairs.sort()
        pair_lists.append(pairs)
    # Nodes in pools[0] move out of group 1, and nodes in pools[1] move in.
    checks = 0
    for side in (0, 1):
        other = sorted(pools[1 - side], key=weights.__getitem__)
        ws = [weights[i] for i in other]
        direction = -1 if side == 0 else 1
        count = state.count + direction
        if not 0 < count < len(weights):
            continue
        for weight, base, a, b in pair_lists[side]:
            if time.perf_counter() >= end:
                return move, delta
            if side == 0:
                low = lower - state.load + weight
                high = upper - state.load + weight
            else:
                low = state.load + weight - upper
                high = state.load + weight - lower
            left, right = bisect.bisect_left(ws, low), bisect.bisect_right(ws, high)
            for index in range(left, right):
                c = other[index]
                change = base + gain[c] + 2.0 * (maps[a].get(c, 0.0) + maps[b].get(c, 0.0))
                if change < delta and (max(tabu[a], tabu[b], tabu[c]) <= step
                                        or state.value + change < best_value - eps):
                    move, delta = (a, b, c), change
                checks += 1
                if checks & 127 == 0 and time.perf_counter() >= end:
                    return move, delta
    # Node pairs within the two groups are matched by weight, skipping every four-node swap that must violate capacity.
    other = pair_lists[1]
    ws = [p[0] for p in other]
    for weight, base, a, b in pair_lists[0]:
        if time.perf_counter() >= end:
            break
        left = bisect.bisect_left(ws, lower - state.load + weight)
        right = bisect.bisect_right(ws, upper - state.load + weight)
        for index in range(left, right):
            _, other_base, c, d = other[index]
            change = base + other_base + 2.0 * (
                maps[a].get(c, 0.0) + maps[a].get(d, 0.0)
                + maps[b].get(c, 0.0) + maps[b].get(d, 0.0))
            if change < delta and (max(tabu[a], tabu[b], tabu[c], tabu[d]) <= step
                                    or state.value + change < best_value - eps):
                move, delta = (a, b, c, d), change
            checks += 1
            if checks & 127 == 0 and time.perf_counter() >= end:
                return move, delta
    return move, delta


def _block(state, selected, weights, adj, lower, upper, end, bias=None):
    """Fix nodes outside the block and search assignments inside it with capacity bounds, signed-edge lower bounds and fractional knapsack bounds."""
    if len(selected) < 2 or time.perf_counter() >= end:
        return (), False
    chosen = set(selected)
    selected.sort(key=lambda i: -sum(abs(c) for j, c in adj[i] if j in chosen))
    k = len(selected)
    pos = {i: p for p, i in enumerate(selected)}
    a0, a1 = [0.0] * k, [0.0] * k
    links = [[] for _ in selected]
    negative = [0.0] * (k + 1)
    ws = [weights[i] for i in selected]
    suffix = [0] * (k + 1)
    gcds = [0] * (k + 1)
    old_mask = 0
    outside_load, outside_count = state.load, state.count
    for p, i in enumerate(selected):
        if state.bits[i]:
            old_mask |= 1 << p
            outside_load -= weights[i]
            outside_count -= 1
        for j, c in adj[i]:
            q = pos.get(j)
            if q is None:
                if state.bits[j]:
                    a0[p] += c
                else:
                    a1[p] += c
            elif q > p:
                links[p].append((q, c))
        if bias is not None:
            a1[p] += bias.get(i, 0.0)
    old_cost = math.fsum(a1[p] if (old_mask >> p) & 1 else a0[p] for p in range(k))
    old_cost += math.fsum(c for p in range(k) for q, c in links[p]
                          if ((old_mask >> p) ^ (old_mask >> q)) & 1)
    best_cost, best_mask = old_cost, old_mask
    for p in range(k - 1, -1, -1):
        suffix[p] = suffix[p + 1] + ws[p]
        gcds[p] = math.gcd(gcds[p + 1], ws[p])
        negative[p] = negative[p + 1] + sum(min(0.0, c) for _, c in links[p])
    lo, hi = lower - outside_load, upper - outside_load
    cl, ch = 1 - outside_count, len(weights) - 1 - outside_count
    minimum = sum(min(a, b) for a, b in zip(a0, a1))
    eps = 1e-12 * (1.0 + abs(old_cost) + sum(abs(c) for row in links for _, c in row))
    visits = 0
    expired = False

    def visit(p, load, count, fixed, minimum, mask):
        nonlocal best_cost, best_mask, visits, expired
        visits += 1
        if visits & 63 == 0 and time.perf_counter() >= end:
            expired = True
        if expired or load > hi or load + suffix[p] < lo or count > ch or count + k - p < cl:
            return
        g = gcds[p]
        if g and (lo - load + g - 1) // g > (hi - load) // g:
            return
        bound = fixed + minimum + negative[p]
        if bound >= best_cost - eps:
            return
        if p == k:
            if fixed < best_cost - eps:
                best_cost, best_mask = fixed, mask
            return
        # In the upper levels of the search tree, strengthen the lower bound with the minimum fractional cost needed to meet the remaining weight interval.
        if k - p >= 5 and p % 3 == 0:
            preferred = sum(ws[q] for q in range(p, k) if a1[q] < a0[q])
            need_up = max(0, lo - load - preferred)
            need_down = max(0, preferred - (hi - load))
            need = need_up or need_down
            if need:
                adjustments = []
                for q in range(p, k):
                    if ws[q] and ((need_up and a1[q] >= a0[q])
                                  or (need_down and a1[q] < a0[q])):
                        adjustments.append((abs(a1[q] - a0[q]) / ws[q], ws[q]))
                adjustments.sort()
                extra = 0.0
                for ratio, w in adjustments:
                    take = min(need, w)
                    extra += ratio * take
                    need -= take
                    if need <= 0:
                        break
                if bound + extra >= best_cost - eps:
                    return
        first = 1 if a1[p] < a0[p] else 0
        for bit in (first, 1 - first):
            new_load = load + bit * ws[p]
            new_count = count + bit
            if (new_load > hi or new_load + suffix[p + 1] < lo
                    or new_count > ch or new_count + k - p - 1 < cl):
                continue
            new_minimum = minimum - min(a0[p], a1[p])
            saved = []
            for q, c in links[p]:
                before = min(a0[q], a1[q])
                saved.append((q, a0[q], a1[q]))
                if bit:
                    a0[q] += c
                else:
                    a1[q] += c
                new_minimum += min(a0[q], a1[q]) - before
            visit(p + 1, new_load, new_count, fixed + (a1[p] if bit else a0[p]),
                  new_minimum, mask | (bit << p))
            for q, v0, v1 in saved:
                a0[q], a1[q] = v0, v1
            if expired:
                return

    visit(0, 0, 0, 0.0, minimum, 0)
    changed = tuple(i for p, i in enumerate(selected) if ((best_mask ^ old_mask) >> p) & 1)
    return changed, not expired


def _select(state, weights, adj, size, serial, elite, rng, end):
    n = len(weights)
    if n <= size + 1:
        # Fixing one node removes the symmetry of swapping the two groups as a whole.
        return list(range(1, n))
    selected, used = [], set()

    def add(i):
        if i not in used and len(selected) < size:
            used.add(i)
            selected.append(i)

    mode = serial % 4
    if mode == 2 and elite:
        donor = elite[rng.randrange(len(elite))][1]
        invert = sum(a != b for a, b in zip(donor, state.bits)) > n // 2
        difference = [i for i in range(n) if (donor[i] ^ invert) != state.bits[i]]
        rng.shuffle(difference)
        for i in difference[:size]:
            add(i)
    if mode in (1, 3):
        roots = heapq.nsmallest(max(2, size // 2), range(n), key=state.gain.__getitem__)
        root = roots[rng.randrange(len(roots))] if mode == 1 else rng.randrange(n)
        add(root)
        frontier = [(-abs(c), j) for j, c in adj[root]]
        heapq.heapify(frontier)
        goal = max(2, 2 * size // 3)
        while frontier and len(selected) < goal and time.perf_counter() < end:
            _, i = heapq.heappop(frontier)
            if i in used:
                continue
            add(i)
            for j, c in adj[i]:
                if j not in used:
                    heapq.heappush(frontier, (-abs(c), j))
    one = [i for i in range(n) if state.bits[i] and i not in used]
    zero = [i for i in range(n) if not state.bits[i] and i not in used]
    target_one = max(0, size // 2 - sum(state.bits[i] for i in selected))
    fill_one = heapq.nsmallest(target_one, one, key=state.gain.__getitem__)
    for i in fill_one:
        add(i)
    for i in heapq.nsmallest(size - len(selected), zero, key=state.gain.__getitem__):
        add(i)
    if len(selected) < size:
        for i in heapq.nsmallest(size, one, key=state.gain.__getitem__):
            add(i)
    # Randomly replace a few nodes so repeated low-increment candidate blocks still yield new combinations.
    if mode == 0:
        available = [i for i in range(n) if i not in used]
        for p, i in enumerate(rng.sample(available, min(len(available), max(1, size // 5)))):
            selected[-1 - p] = i
    return selected


def _search(initial, weights, edges, costs, lower, upper, start, end, budget, rng, saved):
    n = len(weights)
    maps = [{} for _ in range(n)]
    for (u, v), c in zip(edges, costs):
        if u != v:
            maps[u][v] = maps[u].get(v, 0.0) + c
            maps[v][u] = maps[v].get(u, 0.0) + c
    adj = [[(j, c) for j, c in row.items() if c != 0.0] for row in maps]
    current = _State(initial, weights, adj)
    best_value = current.value
    strength = math.log2(2.0 + budget)
    elite_cap = max(2, int(2 + 2 * strength))
    elite = []

    def remember(state):
        nonlocal best_value
        if not (lower <= state.load <= upper and 0 < state.count < n):
            return False
        if state.value < best_value - 1e-12 * (1.0 + abs(best_value)):
            value = math.fsum(c for (u, v), c in zip(edges, costs)
                              if state.bits[u] != state.bits[v])
            state.value = value
            if value < best_value:
                best_value, saved[0] = value, state.bits[:]
                return True
        return False

    def archive(state):
        bits = state.bits[:] if state.bits[0] else [1 - b for b in state.bits]
        gap = max(1, n // max(8, int(12 + 2 * strength)))
        for p, (value, old) in enumerate(elite):
            distance = sum(a != b for a, b in zip(bits, old))
            if min(distance, n - distance) < gap:
                if state.value < value:
                    elite[p] = (state.value, bits)
                    elite.sort(key=lambda item: item[0])
                return
        elite.append((state.value, bits))
        elite.sort(key=lambda item: item[0])
        del elite[elite_cap:]

    archive(current)
    tabu = [0] * n
    step = stagnant = restarts = serial = 0
    deep_spent = 0.0
    last_deep = start
    global_done = False
    avg_degree = sum(len(row) for row in adj) / max(1, n)
    local_size = max(6, int(12 + 2 * strength - 0.25 * math.log2(1 + avg_degree)))
    size_bonus = 0
    scale_cost = sum(abs(c) for c in costs) / max(1, n)

    while time.perf_counter() < end:
        step += 1
        one, zero = [], []
        move, delta = None, float('inf')
        aspiration = best_value - current.value - 1e-12 * (1 + abs(best_value))
        for i, bit in enumerate(current.bits):
            (one if bit else zero).append(i)
            direction = 1 - 2 * bit
            load = current.load + direction * weights[i]
            count = current.count + direction
            change = current.gain[i]
            if (lower <= load <= upper and 0 < count < n
                    and (tabu[i] <= step or change < aspiration)
                    and change < delta):
                move, delta = (i,), change
            if i & 255 == 255 and time.perf_counter() >= end:
                return
        width = max(4, int(4 + 2 * strength + math.log1p(step) / 2))
        if len(one) * len(zero) <= 200 * (1 + strength) ** 2:
            pool_one, pool_zero = one, zero
        else:
            pool_one = _pool(one, width, current, weights, rng)
            pool_zero = _pool(zero, width, current, weights, rng)
        sorted_zero = sorted(pool_zero, key=weights.__getitem__)
        zero_weights = [weights[i] for i in sorted_zero]
        checks = 0
        for a in pool_one:
            lo = lower - current.load + weights[a]
            hi = upper - current.load + weights[a]
            left, right = bisect.bisect_left(zero_weights, lo), bisect.bisect_right(zero_weights, hi)
            for index in range(left, right):
                b = sorted_zero[index]
                change = current.gain[a] + current.gain[b] + 2.0 * maps[a].get(b, 0.0)
                if change < delta and (max(tabu[a], tabu[b]) <= step or change < aspiration):
                    move, delta = (a, b), change
                checks += 1
                if checks & 255 == 0 and time.perf_counter() >= end:
                    return
            if time.perf_counter() >= end:
                return

        now = time.perf_counter()
        stall_limit = max(8, int((3 + strength) * math.sqrt(n)))
        # Compound neighbourhoods and exact blocks share the current state; their cost is accounted by elapsed wall clock.
        allowance = 0.48 * (now - start) + 0.015 * budget
        deep_due = (delta >= -1e-12 * (1 + abs(current.value))
                    and now - last_deep >= budget / (35 + 5 * strength)
                    and deep_spent < allowance)
        if deep_due and not global_done:
            deep_start = now
            slice_time = min(end - now, budget / (30 + 2 * strength))
            deep_end = min(end, now + slice_time)
            deep_width = max(5, int(5 + 3 * strength))
            pools = (_pool(one, deep_width, current, weights, rng),
                     _pool(zero, deep_width, current, weights, rng))
            move, delta = _compound(current, pools, weights, maps, lower, upper,
                                    tabu, step, best_value, now + slice_time * 0.3, move, delta)
            # Accept the compound swap first, then search a larger exact block on its updated incremental structure.
            if move is not None and delta < -1e-12 * (1 + abs(current.value)):
                for i in move:
                    current.flip(i, weights, adj)
                    tabu[i] = step + max(2, int(math.sqrt(n) / 2 + strength))
                improved = remember(current)
                if improved:
                    archive(current)
                stagnant = 0 if improved else stagnant + 1
                move, delta = None, float('inf')
            selected = _select(current, weights, adj, min(n, local_size + size_bonus),
                               serial, elite, rng, deep_end)
            changed, complete = _block(current, selected, weights, adj, lower, upper, deep_end)
            serial += 1
            if complete:
                size_bonus = min(int(2 + strength), size_bonus + 1)
                if len(selected) == n - 1:
                    global_done = True
            else:
                size_bonus = max(0, size_bonus - 1)
            if changed:
                for i in changed:
                    current.flip(i, weights, adj)
                    tabu[i] = step + max(2, int(math.sqrt(n) / 2 + strength))
                improved = remember(current)
                if improved:
                    archive(current)
                stagnant = 0 if improved else stagnant + 1
                move = None
            deep_spent += time.perf_counter() - deep_start
            last_deep = time.perf_counter()
            if changed or move is None:
                if changed:
                    continue
                # Once a compound swap is applied, candidate costs from the old state are no longer used.
                if delta == float('inf'):
                    continue

        if move is not None:
            tenure = max(2, int(math.sqrt(n) + strength + rng.random() * (2 + strength)))
            for i in move:
                current.flip(i, weights, adj)
                tabu[i] = step + tenure
            if remember(current):
                stagnant = 0
                archive(current)
            else:
                stagnant += 1
        else:
            stagnant += max(1, int(math.sqrt(n)))

        if stagnant >= stall_limit or move is None:
            archive(current)
            restarts += 1
            base = saved[0] if rng.random() < 0.6 else elite[rng.randrange(len(elite))][1]
            current = _State(base, weights, adj)
            tabu = [0] * n
            # Perturbation also goes through the same exact capacity subproblem, allowing several nodes of different weights to move at once.
            now = time.perf_counter()
            if (not global_done and deep_spent < 0.6 * (now - start) + 0.02 * budget
                    and now < end):
                slice_end = min(end, now + budget / (45 + 3 * strength))
                selected = _select(current, weights, adj, min(n, local_size + size_bonus),
                                   serial, elite, rng, slice_end)
                serial += 1
                amplitude = max(1e-12, scale_cost) * (0.3 + rng.random()) * (1 + strength / 6)
                donor = elite[rng.randrange(len(elite))][1]
                invert = sum(a != b for a, b in zip(donor, current.bits)) > n // 2
                bias = {i: amplitude * ((1 - 2 * (donor[i] ^ invert)) * 0.4
                                        + rng.uniform(-1.0, 1.0)) for i in selected}
                changed, _ = _block(current, selected, weights, adj, lower, upper, slice_end, bias)
                for i in changed:
                    current.flip(i, weights, adj)
                remember(current)
                deep_spent += time.perf_counter() - now
            kick_end = min(end, time.perf_counter() + budget / (150 + 10 * strength))
            goal = max(2, int((1 + strength) * (1 + math.sqrt(restarts % max(2, int(strength + 2))))))
            if restarts % max(2, int(2 + strength)) == 0:
                goal = max(goal, int(math.sqrt(n) * (1 + strength)))
            accepted = 0
            while accepted < goal and time.perf_counter() < kick_end:
                a = rng.randrange(n)
                direction = 1 - 2 * current.bits[a]
                if rng.random() < 0.35 and (lower <= current.load + direction * weights[a] <= upper
                                            and 0 < current.count + direction < n):
                    current.flip(a, weights, adj)
                    accepted += 1
                else:
                    b = rng.randrange(n)
                    if current.bits[a] == current.bits[b]:
                        continue
                    load = current.load + direction * (weights[a] - weights[b])
                    if lower <= load <= upper:
                        current.flip(a, weights, adj)
                        current.flip(b, weights, adj)
                        accepted += 1
                remember(current)
            stagnant = 0


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    n = int(instance['num_nodes'])
    m = int(instance['num_edges'])
    weights = [int(w) for w in instance['node_weights']]
    edges = [(int(e[0]), int(e[1])) for e in instance['edges']]
    costs = [float(c) for c in instance['edge_costs']]
    upper = int(instance['bisection_capacity'])
    lower = sum(weights) - upper
    saved = [[1] + [0] * max(0, n - 1)]
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        # Time is reserved for output and one incremental update; no fixed-seconds search cap is used.
        reserve = min(budget * 0.1, max(budget * 0.012, (n + m + 1) * 8e-7))
        end = start + budget - reserve
        seed = 0xC0FFEE ^ (n * 1000003) ^ (m * 9176)
        for i, w in enumerate(weights):
            seed = (seed * 6364136223846793005 + w + i + 1) & ((1 << 64) - 1)
        rng = random.Random(seed)
        initial = _initial(weights, lower, upper, end, rng)
        saved[0] = initial[:]
        if n >= 2 and _valid(initial, weights, lower, upper) and time.perf_counter() < end:
            _search(initial, weights, edges, costs, lower, upper, start, end, budget, rng, saved)
    except BaseException:
        # An unexpected abort during optimisation never overwrites the complete feasible solution already obtained.
        pass
    bits = saved[0]
    return {'node_groups': [1 if b else 2 for b in bits],
            'cut_edges': [int(bits[u] != bits[v]) for u, v in edges]}
