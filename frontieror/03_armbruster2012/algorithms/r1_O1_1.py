"""Capacity-aware tabu search with exact block re-optimisation
Combines incremental swap search, elite-solution perturbation and capacity-constrained multi-node block optimisation, always keeping the best verified feasible partition.
"""

import bisect
import heapq
import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix, csr_matrix
except Exception:
    np = None
    milp = None


def _valid(bits, weights, lower, upper):
    return (0 < sum(bits) < len(bits)
            and lower <= sum(w for w, b in zip(weights, bits) if b) <= upper)


def _initial(weights, lower, upper, deadline):
    n = len(weights)
    fallback = [1] + [0] * (n - 1)
    for i, w in enumerate(weights):
        if lower <= w <= upper:
            bits = [0] * n
            bits[i] = 1
            return bits
    order = sorted(range(n), key=lambda i: (-weights[i], i))
    loads = [0, 0]
    counts = [0, 0]
    bits = [0] * n
    for i in order:
        side = 0 if (loads[0], counts[0]) <= (loads[1], counts[1]) else 1
        if loads[side] + weights[i] > upper:
            break
        loads[side] += weights[i]
        counts[side] += 1
        bits[i] = side
    else:
        if _valid(bits, weights, lower, upper):
            return bits

    best_violation = float('inf')
    for seq in (order, order[::-1], list(range(n))):
        bits = [0] * n
        load = 0
        for i in seq:
            if load < lower and load + weights[i] <= upper:
                bits[i] = 1
                load += weights[i]
        if _valid(bits, weights, lower, upper):
            return bits
        violation = max(lower - load, load - upper, 0)
        if not 0 < sum(bits) < n:
            violation += sum(weights) + 1
        if violation < best_violation:
            best_violation, fallback = violation, bits

    # Symmetry allows searching only for subsets weighing at most half of the total; blockwise checkpoints bound reconstruction memory.
    cap = min(upper, sum(weights) // 2)
    positive = [i for i in order if 0 < weights[i] <= cap]
    block = max(1, math.isqrt(len(positive)))
    bit_budget = 256_000_000 // max(4, 2 * block + 2)
    if cap <= bit_budget and time.perf_counter() < deadline:
        mask = (1 << (cap + 1)) - 1
        reach = 1
        checkpoints = [reach]
        target = None
        used = 0
        for i in positive:
            if time.perf_counter() >= deadline:
                break
            reach = (reach | (reach << weights[i])) & mask
            used += 1
            available = reach >> lower
            if available:
                target = lower + (available & -available).bit_length() - 1
                break
            if used % block == 0:
                checkpoints.append(reach)
        if target is not None:
            bits = [0] * n
            stop = used
            while stop:
                start = ((stop - 1) // block) * block
                reach = checkpoints[start // block]
                history = [reach]
                for p in range(start, stop):
                    reach = (reach | (reach << weights[positive[p]])) & mask
                    history.append(reach)
                for p in range(stop - start, 0, -1):
                    if not ((history[p - 1] >> target) & 1):
                        i = positive[start + p - 1]
                        bits[i] = 1
                        target -= weights[i]
                stop = start
            if _valid(bits, weights, lower, upper):
                return bits

    # Large integer weights use a meet-in-the-middle subset sum, without allocating arrays that grow with the weight.
    left_n = n // 2
    room = max(0.0, deadline - time.perf_counter())
    if (1 << ((n + 1) // 2)) <= min(500_000, max(0, int(room * 120_000))):
        halves = []
        for start, stop in ((0, left_n), (left_n, n)):
            sums = [(0, 0)]
            for i in range(start, stop):
                if time.perf_counter() >= deadline:
                    break
                w = weights[i]
                sums += [(v + w, b | (1 << i)) for v, b in sums if v + w <= cap]
            halves.append(sums)
        halves[1].sort()
        values = [v for v, b in halves[1]]
        for value, chosen in halves[0]:
            p = bisect.bisect_left(values, lower - value)
            if p < len(values) and value + values[p] <= cap:
                chosen |= halves[1][p][1]
                bits = [(chosen >> i) & 1 for i in range(n)]
                if _valid(bits, weights, lower, upper):
                    return bits

    if milp is not None and time.perf_counter() < deadline:
        try:
            scale = max(1.0, max(weights) / 1_000_000.0)
            matrix = np.array([[w / scale for w in weights], [1.0] * n])
            result = milp(
                np.zeros(n), integrality=np.ones(n), bounds=Bounds(0, 1),
                constraints=LinearConstraint(matrix, [lower / scale, 1],
                                             [upper / scale, n - 1]),
                options={'time_limit': max(0.001, deadline - time.perf_counter()),
                         'mip_rel_gap': 0.0, 'presolve': True})
            if result.x is not None:
                bits = [int(v >= 0.5) for v in result.x]
                if _valid(bits, weights, lower, upper):
                    return bits
        except Exception:
            pass

    # Sparse reachable sums are the last resort for restoring capacity; every expansion is still bounded by wall clock.
    parents = {0: None}
    for i in positive:
        if time.perf_counter() >= deadline:
            break
        additions = []
        for old in parents:
            new = old + weights[i]
            if new <= cap and new not in parents:
                additions.append((new, old))
            if len(additions) % 1024 == 0 and time.perf_counter() >= deadline:
                break
        for new, old in additions:
            parents[new] = (old, i)
            if lower <= new <= cap:
                bits = [0] * n
                while new:
                    new, node = parents[new]
                    bits[node] = 1
                if _valid(bits, weights, lower, upper):
                    return bits
        if len(parents) > 500_000:
            break
    return fallback


class _State:
    __slots__ = ('bits', 'load', 'count', 'value', 'gain')

    def __init__(self, bits, weights, adjacency):
        self.bits = bits[:]
        self.load = sum(w for w, b in zip(weights, bits) if b)
        self.count = sum(bits)
        self.gain = [0.0] * len(bits)
        value = 0.0
        for i, neighbors in enumerate(adjacency):
            delta = 0.0
            for j, c in neighbors:
                if bits[i] == bits[j]:
                    delta += c
                else:
                    delta -= c
                    if i < j:
                        value += c
            self.gain[i] = delta
        self.value = value

    def flip(self, i, weights, adjacency):
        old = self.bits[i]
        delta = self.gain[i]
        self.value += delta
        for j, c in adjacency[i]:
            self.gain[j] += -2.0 * c if old == self.bits[j] else 2.0 * c
        self.gain[i] = -delta
        self.bits[i] = 1 - old
        self.load += -weights[i] if old else weights[i]
        self.count += -1 if old else 1


def _search(weights, lower, upper, edges, costs, initial, deadline, budget, keep):
    n = len(weights)
    now = time.perf_counter
    maps = [{} for _ in range(n)]
    for (u, v), c in zip(edges, costs):
        if u != v:
            maps[u][v] = maps[u].get(v, 0.0) + c
            maps[v][u] = maps[v].get(u, 0.0) + c
    adjacency = [[(j, c) for j, c in row.items() if c != 0.0] for row in maps]
    current = _State(initial, weights, adjacency)
    best_bits, best_value = initial[:], current.value
    keep[0] = best_bits
    tolerance = 1e-10
    strength = math.log2(2.0 + budget)
    rng = random.Random(0xC0FFEE ^ (n * 1000003) ^ len(edges))
    deep_rng = random.Random(0x94D049BB ^ n ^ (len(edges) * 9176))
    elite = []
    elite_cap = max(3, int(2 + 2 * strength))
    sparse_graph = None
    dense_graph = None
    vectorized = np is not None and sum(weights) < (1 << 60)
    if vectorized and now() < deadline:
        wn = np.asarray(weights, dtype=np.int64)
        dense_bytes = 8 * n * n
        if dense_bytes <= min(64_000_000, 4_000_000 * (1 + math.sqrt(budget))):
            dense_graph = np.zeros((n, n), dtype=float)
            for i, row in enumerate(adjacency):
                for j, c in row:
                    dense_graph[i, j] = c
        else:
            rows, cols, vals = [], [], []
            for i, row in enumerate(adjacency):
                for j, c in row:
                    rows.append(i)
                    cols.append(j)
                    vals.append(c)
            sparse_graph = csr_matrix((vals, (rows, cols)), shape=(n, n))

    def objective(bits):
        return math.fsum(c for (u, v), c in zip(edges, costs) if bits[u] != bits[v])

    # Recompute from the original edges when archiving, so incremental floating-point errors or merged parallel edges do not affect comparisons.
    best_value = objective(initial)

    def record(state):
        nonlocal best_value, best_bits
        if (lower <= state.load <= upper and 0 < state.count < n
                and state.value < best_value - tolerance * (1 + abs(best_value))):
            value = objective(state.bits)
            state.value = value
            if value < best_value - tolerance * (1 + abs(best_value)):
                best_value, best_bits = value, state.bits[:]
                keep[0] = best_bits
                return True
        return False

    def add_elite(state):
        if not lower <= state.load <= upper or not 0 < state.count < n:
            return
        bits = state.bits[:]
        signature = bytes((b ^ bits[0]) for b in bits)
        for value, saved, code in elite:
            if code == signature:
                return
        elite.append((objective(bits), bits, signature))
        elite.sort(key=lambda e: e[0])
        if len(elite) > elite_cap:
            # Never delete the best member; prefer deleting worse members that are close to better ones.
            remove = len(elite) - 1
            closest = n + 1
            for i in range(1, len(elite)):
                distance = min(sum(a != b for a, b in zip(elite[i][2], elite[j][2]))
                               for j in range(i))
                if distance <= closest:
                    closest, remove = distance, i
            elite.pop(remove)

    add_elite(current)
    density = sum(len(a) for a in adjacency) / max(1, n * (n - 1))
    cluster_bonus = math.sqrt(n) / (1 + density * math.sqrt(n))
    block_size = min(n, max(4, int(12 + 5 * strength + cluster_bonus)))
    max_block = min(n, max(block_size, int(20 + 10 * strength + cluster_bonus)))
    block_floor = min(n, max(4, int(8 + 2 * strength)))
    slice_budget = budget / (8 + 2 * strength)
    lns_spent = 0.0
    serial = 0

    def select_block(state, size, mode):
        if size >= n:
            # Whole-graph re-optimisation fixes one node, removing the symmetry of swapping the two groups.
            return list(range(1, n))
        ones = [i for i in range(n) if state.bits[i]]
        zeros = [i for i in range(n) if not state.bits[i]]
        take_one = min(len(ones), size // 2)
        take_zero = min(len(zeros), size - take_one)
        take_one = min(len(ones), size - take_zero)
        targets = [take_zero, take_one]
        chosen, selected = [], set()
        counts = [0, 0]

        def add(i):
            side = state.bits[i]
            if i not in selected and counts[side] < targets[side]:
                selected.add(i)
                chosen.append(i)
                counts[side] += 1
                return True
            return False

        if mode == 2 and len(elite) > 1:
            other = elite[deep_rng.randrange(len(elite))][1]
            distance = sum(a != b for a, b in zip(state.bits, other))
            invert = distance > n // 2
            different = [i for i in range(n) if state.bits[i] != (other[i] ^ invert)]
            deep_rng.shuffle(different)
            different.sort(key=lambda i: state.gain[i])
            for i in different:
                add(i)
        elif mode == 1 or mode == 3:
            frontier = []
            for group in (ones, zeros):
                if group:
                    weak = heapq.nsmallest(max(1, int(strength)), group,
                                          key=lambda i: state.gain[i])
                    root = (deep_rng.choice(group) if mode == 3
                            else deep_rng.choice(weak))
                    heapq.heappush(frontier, (-float('inf'), root))
            while frontier and len(chosen) < size and now() < deadline:
                _, i = heapq.heappop(frontier)
                if not add(i):
                    continue
                for j, c in adjacency[i]:
                    if j not in selected:
                        synergy = c if state.bits[i] == state.bits[j] else -c
                        heapq.heappush(frontier, (-synergy, j))
        else:
            for group, take in ((ones, take_one), (zeros, take_zero)):
                weak = heapq.nsmallest(max(0, take - max(1, take // 4)), group,
                                      key=lambda i: state.gain[i])
                for i in weak:
                    add(i)
                for i in deep_rng.sample(group, min(len(group), take)):
                    add(i)
        for group in (ones, zeros):
            for i in sorted(group, key=lambda i: state.gain[i]):
                add(i)
                if len(chosen) == size:
                    break
        return chosen

    def block_optimize(state, selected, stop):
        """Flip-variable model: linear terms are single-node increments, and quadratic terms subtract internal edges counted twice."""
        k = len(selected)
        if not k or np is None or now() >= stop:
            return state, False
        location = {v: i for i, v in enumerate(selected)}
        internal = []
        for p, i in enumerate(selected):
            for j, c in adjacency[i]:
                q = location.get(j, -1)
                if p < q:
                    correction = -2.0 * c if state.bits[i] == state.bits[j] else 2.0 * c
                    internal.append((p, q, correction))
        gains = [state.gain[i] for i in selected]
        changes = [-weights[i] if state.bits[i] else weights[i] for i in selected]
        counts = [-1 if state.bits[i] else 1 for i in selected]
        room = stop - now()
        enum_size = min(21, max(1, int(math.log2(max(2, room * 2_000_000)))))
        if vectorized and k <= enum_size:
            # Each extension updates only local arrays, without repeatedly visiting outside neighbours as whole-graph Gray-code enumeration would.
            total = 1 << k
            delta = np.empty(total, dtype=float)
            loads = np.empty(total, dtype=np.int64)
            nums = np.empty(total, dtype=np.int32)
            delta[0] = 0.0
            loads[0] = state.load
            nums[0] = state.count
            interactions = [[] for _ in range(k)]
            for p, q, c in internal:
                interactions[q].append((p, c))
            chosen_mask, best_delta = 0, 0.0
            finished = True
            for p in range(k):
                if now() >= stop:
                    finished = False
                    break
                size = 1 << p
                target = delta[size:2 * size]
                target[:] = delta[:size] + gains[p]
                for q, c in interactions[p]:
                    width = 1 << q
                    target.reshape(-1, 2 * width)[:, width:] += c
                loads[size:2 * size] = loads[:size] + changes[p]
                nums[size:2 * size] = nums[:size] + counts[p]
                good = ((loads[size:2 * size] >= lower)
                        & (loads[size:2 * size] <= upper)
                        & (nums[size:2 * size] > 0)
                        & (nums[size:2 * size] < n))
                scores = np.where(good, target, np.inf)
                arg = int(np.argmin(scores))
                value = float(scores[arg])
                if value < best_delta:
                    best_delta, chosen_mask = value, size + arg
            if chosen_mask:
                bits = state.bits[:]
                for p, i in enumerate(selected):
                    if (chosen_mask >> p) & 1:
                        bits[i] ^= 1
                candidate = _State(bits, weights, adjacency)
                if _valid(bits, weights, lower, upper) and candidate.value < state.value:
                    record(candidate)
                    return candidate, finished
            return state, finished
        if milp is None or now() >= stop:
            return state, False
        try:
            variables = k + len(internal)
            coefficients = np.array(gains + [c for p, q, c in internal], dtype=float)
            norm = max(1.0, float(np.max(np.abs(coefficients))))
            coefficients /= norm
            row, col, data = [], [], []
            low, high = [], []
            weight_scale = max(1.0, max(abs(w) for w in changes) / 1_000_000.0)
            for p in range(k):
                row.extend((0, 1))
                col.extend((p, p))
                data.extend((changes[p] / weight_scale, counts[p]))
            low.extend(((lower - state.load) / weight_scale, 1 - state.count))
            high.extend(((upper - state.load) / weight_scale, n - 1 - state.count))
            for e, (p, q, c) in enumerate(internal):
                y = k + e
                r = len(low)
                # The continuous auxiliary variable equals the product of two binary flip variables.
                row.extend((r, r, r + 1, r + 1, r + 2, r + 2, r + 2))
                col.extend((y, p, y, q, p, q, y))
                data.extend((1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0))
                low.extend((-np.inf, -np.inf, -np.inf))
                high.extend((0.0, 0.0, 1.0))
            # Zero flips is a feasible start; only objective regions no worse than the start are kept.
            r = len(low)
            for p, c in enumerate(coefficients):
                if c:
                    row.append(r)
                    col.append(p)
                    data.append(float(c))
            low.append(-np.inf)
            high.append(1e-9)
            matrix = coo_matrix((data, (row, col)), shape=(len(low), variables)).tocsc()
            remaining = stop - now()
            if remaining <= 0:
                return state, False
            integer = np.zeros(variables, dtype=np.uint8)
            integer[:k] = 1
            result = milp(coefficients, integrality=integer, bounds=Bounds(0, 1),
                          constraints=LinearConstraint(matrix, low, high),
                          options={'time_limit': remaining * 0.86,
                                   'mip_rel_gap': 0.0, 'presolve': True})
            optimal = result.status == 0
            if result.x is not None:
                bits = state.bits[:]
                for p, i in enumerate(selected):
                    if result.x[p] >= 0.5:
                        bits[i] ^= 1
                if _valid(bits, weights, lower, upper):
                    candidate = _State(bits, weights, adjacency)
                    if candidate.value < state.value:
                        record(candidate)
                        return candidate, optimal
            return state, optimal
        except Exception:
            return state, False

    tabu = [0] * n
    step = 0
    stagnant = 0
    restarts = 0
    start = now()
    last_deep = start
    root_n = math.sqrt(n + 1)
    pair_allowance = max(256, int(60_000 * math.sqrt(1 + budget)))
    pool_width = max(5, int(5 + 3 * strength + root_n * strength / 8))
    stagnation_limit = max(8, int((3 + strength) * root_n))

    def pool(group):
        if len(group) <= 2 * pool_width:
            return group
        q = max(1, pool_width // 4)
        chosen = heapq.nsmallest(pool_width, group, key=lambda i: current.gain[i])
        chosen += rng.sample(group, min(q, len(group)))
        chosen += heapq.nsmallest(q, group, key=lambda i: weights[i])
        chosen += heapq.nlargest(q, group, key=lambda i: weights[i])
        return list(dict.fromkeys(chosen))

    while now() < deadline:
        step += 1
        one, zero = [], []
        move = None
        change = float('inf')
        aspiration = best_value - current.value - tolerance * (1 + abs(best_value))
        for i in range(n):
            if current.bits[i]:
                one.append(i)
                load, count = current.load - weights[i], current.count - 1
            else:
                zero.append(i)
                load, count = current.load + weights[i], current.count + 1
            g = current.gain[i]
            if (lower <= load <= upper and 0 < count < n and g < change
                    and (step >= tabu[i] or g < aspiration)):
                move, change = (i,), g
        if now() >= deadline:
            break
        if len(one) * len(zero) <= pair_allowance:
            pa, pb = one, zero
        else:
            pa, pb = pool(one), pool(zero)
        if vectorized and pa and pb:
            aa = np.asarray(pa, dtype=np.intp)
            bb = np.asarray(pb, dtype=np.intp)
            ga = np.asarray([current.gain[i] for i in pa])
            gb = np.asarray([current.gain[i] for i in pb])
            if dense_graph is not None:
                cross = dense_graph[np.ix_(aa, bb)]
            else:
                cross = sparse_graph[aa][:, bb].toarray()
            values = ga[:, None] + gb[None, :] + 2.0 * cross
            loads = current.load - wn[aa, None] + wn[bb][None, :]
            ta = np.asarray([step >= tabu[i] for i in pa])
            tb = np.asarray([step >= tabu[i] for i in pb])
            allowed = ((loads >= lower) & (loads <= upper)
                       & ((ta[:, None] & tb[None, :]) | (values < aspiration)))
            values[~allowed] = np.inf
            pos = int(np.argmin(values))
            value = float(values.flat[pos])
            if value < change:
                p, q = divmod(pos, len(pb))
                move, change = (pa[p], pb[q]), value
        else:
            for a in pa:
                if now() >= deadline:
                    break
                for b in pb:
                    load = current.load - weights[a] + weights[b]
                    if lower <= load <= upper:
                        value = current.gain[a] + current.gain[b] + 2 * maps[a].get(b, 0.0)
                        if value < change and ((step >= tabu[a] and step >= tabu[b])
                                               or value < aspiration):
                            move, change = (a, b), value
        if now() >= deadline:
            break
        # Strongly coupled nodes in the same group can also flip as a pair, covering two-node moves missed by ordinary swaps.
        if move is None or change >= 0:
            for a in pool(one) + pool(zero):
                if now() >= deadline:
                    break
                for b, c in adjacency[a]:
                    if a >= b or current.bits[a] != current.bits[b]:
                        continue
                    sign = -1 if current.bits[a] else 1
                    load = current.load + sign * (weights[a] + weights[b])
                    count = current.count + 2 * sign
                    if lower <= load <= upper and 0 < count < n:
                        value = current.gain[a] + current.gain[b] - 2.0 * c
                        if value < change and ((step >= tabu[a] and step >= tabu[b])
                                               or value < aspiration):
                            move, change = (a, b), value
        if move is not None and now() < deadline:
            for i in move:
                current.flip(i, weights, adjacency)
            tenure = max(2, int(root_n + strength / 2) + rng.randrange(1, max(2, int(strength) + 2)))
            for i in move:
                tabu[i] = step + tenure
            if record(current):
                stagnant = 0
            else:
                stagnant += 1
        else:
            stagnant += stagnation_limit

        elapsed = now() - start
        deep_due = (stagnant >= stagnation_limit or move is None)
        periodic = now() - last_deep >= budget / (3 + strength)
        if (deep_due or periodic) and now() < deadline:
            if periodic:
                last_deep = now()
            add_elite(current)
            # Exact blocks are interleaved from early on, but do not take most of the time from the main tabu search.
            if np is not None and lns_spent < 0.40 * max(elapsed, slice_budget):
                serial += 1
                base = _State(best_bits, weights, adjacency)
                if serial % 4 == 0 and elite:
                    base = _State(elite[deep_rng.randrange(len(elite))][1], weights, adjacency)
                selected = select_block(base, block_size, serial % 4)
                begun = now()
                candidate, complete = block_optimize(base, selected, min(deadline, begun + slice_budget))
                spent = now() - begun
                lns_spent += spent
                last_deep = now()
                improved = candidate.value < base.value - tolerance * (1 + abs(base.value))
                if improved:
                    add_elite(candidate)
                    current = candidate
                    tabu = [0] * n
                    stagnant = 0
                if complete and spent < slice_budget * 0.7:
                    block_size = min(max_block, block_size + max(1, math.isqrt(block_size)))
                elif not complete:
                    block_size = max(block_floor, block_size - max(1, math.isqrt(block_size) // 2))
                # When all n - 1 free nodes are covered, global optimality is proven and no further random search is needed.
                if complete and len(selected) == n - 1:
                    break
                if improved and not deep_due:
                    continue
            if now() >= deadline:
                break
            if deep_due:
                add_elite(current)
                restarts += 1
                if elite and rng.random() < 0.35:
                    base = elite[rng.randrange(len(elite))][1]
                else:
                    base = best_bits
                current = _State(base, weights, adjacency)
                # Perturbation time and magnitude are derived from the total budget and stagnation count, with no cap on search rounds.
                kick = min(n, max(2, int(strength + math.sqrt(restarts))))
                if restarts % max(2, int(strength)) == 0:
                    kick = max(kick, n // max(3, int(strength)))
                kick_end = min(deadline, now() + budget / (100 + 10 * strength))
                accepted = 0
                while accepted < kick and now() < kick_end:
                    i, j = rng.randrange(n), rng.randrange(n)
                    if rng.random() < 0.3:
                        sign = -1 if current.bits[i] else 1
                        load = current.load + sign * weights[i]
                        count = current.count + sign
                        if lower <= load <= upper and 0 < count < n:
                            current.flip(i, weights, adjacency)
                            accepted += 1
                            record(current)
                            continue
                    if i == j or current.bits[i] == current.bits[j]:
                        continue
                    if not current.bits[i]:
                        i, j = j, i
                    load = current.load - weights[i] + weights[j]
                    if lower <= load <= upper:
                        current.flip(i, weights, adjacency)
                        current.flip(j, weights, adjacency)
                        accepted += 1
                        record(current)
                tabu = [0] * n
                stagnant = 0
    keep[0] = best_bits


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    n = int(instance['num_nodes'])
    m = int(instance['num_edges'])
    raw_weights = [int(w) for w in instance['node_weights']]
    edges = [(int(u), int(v)) for u, v in instance['edges']]
    costs = [float(c) for c in instance['edge_costs']]
    upper = int(instance['bisection_capacity'])
    lower = sum(raw_weights) - upper
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
    except Exception:
        budget = 0.0
    # Reserve time for output and solver shutdown; the budget is not a fixed ten-second constant.
    reserve = max(0.015 * budget, min(0.04, 0.10 * budget))
    deadline = started + max(0.0, budget - reserve)
    divisor = 0
    for w in raw_weights:
        divisor = math.gcd(divisor, w)
    divisor = max(1, divisor)
    weights = [w // divisor for w in raw_weights]
    lower = (lower + divisor - 1) // divisor
    upper //= divisor
    keep = [[1] + [0] * max(0, n - 1)]
    try:
        if n >= 2:
            keep[0] = _initial(weights, lower, upper,
                               min(deadline, started + 0.35 * budget))
            if not _valid(keep[0], weights, lower, upper) and time.perf_counter() < deadline:
                keep[0] = _initial(weights, lower, upper, deadline)
            if _valid(keep[0], weights, lower, upper) and time.perf_counter() < deadline:
                _search(weights, lower, upper, edges, costs, keep[0], deadline, budget, keep)
    except BaseException:
        # Failure of any search phase never overwrites the feasible solution saved separately before.
        pass
    bits = keep[0][:n]
    return {'node_groups': [1 if b else 2 for b in bits],
            'cut_edges': [int(bits[u] != bits[v]) for u, v in edges[:m]]}
