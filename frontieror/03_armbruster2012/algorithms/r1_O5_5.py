"""Aggregated large neighbourhood with feasible tabu partitioning
Maintains a feasible solution with incremental tabu search and alternates time-limited integer-programming re-optimisation of node neighbourhoods and of aggregated blocks strongly connected within the same group.
"""
import bisect
import heapq
import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
except Exception:
    np = None
    milp = None


def _initial(weights, lower, upper, end):
    n = len(weights)
    if n < 2:
        return [1] * n
    if lower <= 0:
        return [1] + [0] * (n - 1)
    order = sorted(range(n), key=lambda i: -weights[i])
    loads = [0, 0]
    counts = [0, 0]
    bits = [0] * n
    for i in order:
        b = 0 if (loads[0], counts[0]) <= (loads[1], counts[1]) else 1
        if loads[b] + weights[i] > upper:
            b ^= 1
        if loads[b] + weights[i] > upper:
            break
        loads[b] += weights[i]
        counts[b] += 1
        bits[i] = 1 - b
    else:
        if min(counts) > 0:
            return bits
    for seq in (order, order[::-1], range(n)):
        bits = [0] * n
        load = 0
        count = 0
        for i in seq:
            if load + weights[i] <= upper:
                bits[i] = 1
                load += weights[i]
                count += 1
                if lower <= load and count < n:
                    return bits

    divisor = 0
    for w in weights:
        divisor = math.gcd(divisor, w)
    divisor = max(1, divisor)
    lo = (lower + divisor - 1) // divisor
    hi = upper // divisor
    # The bitset size is a memory guard, not a cap on search rounds; reconstruction checkpoints are stored in blocks.
    if hi <= 32000000 and time.perf_counter() < end:
        positive = [i for i in order if weights[i] > 0]
        block = max(1, math.isqrt(len(positive)))
        mask = (1 << (hi + 1)) - 1
        reachable = 1
        checkpoints = [1]
        processed = 0
        target = None
        for i in positive:
            reachable = (reachable | (reachable << (weights[i] // divisor))) & mask
            processed += 1
            available = reachable >> lo
            if available:
                target = lo + (available & -available).bit_length() - 1
                break
            if processed % block == 0:
                checkpoints.append(reachable)
            if time.perf_counter() >= end:
                break
        if target is not None:
            bits = [0] * n
            while processed:
                start = ((processed - 1) // block) * block
                hist = [checkpoints[start // block]]
                for p in range(start, processed):
                    v = weights[positive[p]] // divisor
                    hist.append((hist[-1] | (hist[-1] << v)) & mask)
                for p in range(processed - 1, start - 1, -1):
                    if not ((hist[p - start] >> target) & 1):
                        bits[positive[p]] = 1
                        target -= weights[positive[p]] // divisor
                processed = start
            return bits

    if milp is not None and time.perf_counter() < end:
        try:
            a = np.array([[w // divisor for w in weights], [1] * n], dtype=float)
            result = milp(
                np.zeros(n), integrality=np.ones(n), bounds=Bounds(0, 1),
                constraints=LinearConstraint(a, [lo, 1], [hi, n - 1]),
                options={"time_limit": max(1e-6, end - time.perf_counter()), "presolve": True},
            )
            if result.x is not None:
                bits = [int(x >= 0.5) for x in result.x]
                load = sum(w * b for w, b in zip(weights, bits))
                if lower <= load <= upper and 0 < sum(bits) < n:
                    return bits
        except Exception:
            pass

    suffix = [0] * (n + 1)
    for p in range(n - 1, -1, -1):
        suffix[p] = suffix[p + 1] + weights[order[p]]
    stack = [(0, 0, 0)]
    while stack and time.perf_counter() < end:
        p, load, chosen = stack.pop()
        if lower <= load <= upper:
            return [int((chosen >> i) & 1) for i in range(n)]
        if p == n or load + suffix[p] < lower:
            continue
        i = order[p]
        stack.append((p + 1, load, chosen))
        if load + weights[i] <= upper:
            stack.append((p + 1, load + weights[i], chosen | (1 << i)))
    return None


def solve(instance: dict, time_limit_s: float) -> dict:
    clock = time.perf_counter
    started = clock()
    n = int(instance["num_nodes"])
    m = int(instance["num_edges"])
    weights = [int(w) for w in instance["node_weights"]]
    edges = [(int(e[0]), int(e[1])) for e in instance["edges"]]
    upper = int(instance["bisection_capacity"])
    total = sum(weights)
    lower = total - upper
    best_bits = [1] + [0] * (n - 1) if n else []

    def finish():
        return {
            "node_groups": [1 if b else 2 for b in best_bits],
            "cut_edges": [int(best_bits[u] != best_bits[v]) for u, v in edges],
        }

    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        deadline = started + budget * 0.94
        initial = _initial(weights, lower, upper, min(deadline, started + budget * 0.24))
        if initial is None:
            initial = _initial(weights, lower, upper, deadline)
        if initial is None or n < 2:
            return finish()
        best_bits = initial[:]
        if clock() >= deadline:
            return finish()

        costs = [float(c) for c in instance["edge_costs"]]
        scale = max((abs(c) for c in costs), default=0.0)
        if scale == 0:
            return finish()
        pairs = {}
        for (u, v), cost in zip(edges, costs):
            if u == v:
                continue
            if u > v:
                u, v = v, u
            pairs[u, v] = pairs.get((u, v), 0.0) + cost / scale
        graph = [(u, v, c) for (u, v), c in pairs.items() if c != 0.0]
        maps = [{} for _ in range(n)]
        for u, v, c in graph:
            maps[u][v] = c
            maps[v][u] = c
        adj = [list(row.items()) for row in maps]
        absdegree = [sum(abs(c) for _, c in row) for row in adj]
        strong_edges = sorted(
            ((u, v, c) for u, v, c in graph if c > 0),
            key=lambda e: e[2] / math.sqrt(max(1e-300, absdegree[e[0]] * absdegree[e[1]])),
            reverse=True,
        )
        strength = math.log1p(budget)
        width = max(4, int(7 + 5 * strength))
        neighborhood_size = min(n, max(8, int(18 + 9 * strength)))
        elite_cap = max(2, int(2 + strength))
        rng = random.Random(0xC0FFEE ^ (n * 1000003) ^ (m * 9176) ^ total)
        best_value = sum(c for u, v, c in graph if best_bits[u] != best_bits[v])
        global_lower = sum(min(0.0, c) for _, _, c in graph)
        eps = 1e-10 * max(1.0, sum(abs(c) for _, _, c in graph))
        elite = [(best_value, best_bits[:])]
        solver_enabled = milp is not None

        def feasible(bits):
            return 0 < sum(bits) < n and lower <= sum(w * b for w, b in zip(weights, bits)) <= upper

        def remember(bits, estimate=None):
            nonlocal best_bits, best_value
            if estimate is not None and estimate >= best_value - eps:
                return False
            if not feasible(bits):
                return False
            value = math.fsum(c for u, v, c in graph if bits[u] != bits[v])
            if value < best_value - eps:
                best_value = value
                best_bits = bits[:]
                return True
            return False

        class State:
            def __init__(self, bits):
                self.bits = bits[:]
                self.load = sum(w * b for w, b in zip(weights, bits))
                self.count = sum(bits)
                self.value = math.fsum(c for u, v, c in graph if bits[u] != bits[v])
                self.delta = [sum(c if bits[i] == bits[j] else -c for j, c in row)
                              for i, row in enumerate(adj)]

            def flip(self, i):
                old = self.bits[i]
                change = self.delta[i]
                for j, c in adj[i]:
                    self.delta[j] += -2 * c if self.bits[j] == old else 2 * c
                self.delta[i] = -change
                self.bits[i] ^= 1
                self.value += change
                self.load += -weights[i] if old else weights[i]
                self.count += -1 if old else 1

        state = State(best_bits)
        tabu = [0] * n
        step = 0
        last_gain = clock()

        def pool(nodes, delta):
            if len(nodes) <= 3 * width:
                return nodes
            q = max(1, width // 4)
            chosen = heapq.nsmallest(width, nodes, key=delta.__getitem__)
            chosen += rng.sample(nodes, min(width, len(nodes)))
            chosen += heapq.nsmallest(q, nodes, key=weights.__getitem__)
            chosen += heapq.nlargest(q, nodes, key=weights.__getitem__)
            return list(dict.fromkeys(chosen))

        def tabu_search(until):
            nonlocal state, tabu, step, last_gain
            while clock() < until:
                step += 1
                bits, delta = state.bits, state.delta
                one, zero = [], []
                move = None
                change = float("inf")
                for i in range(n):
                    (one if bits[i] else zero).append(i)
                    load = state.load + (-weights[i] if bits[i] else weights[i])
                    count = state.count + (-1 if bits[i] else 1)
                    d = delta[i]
                    if lower <= load <= upper and 0 < count < n:
                        if (step >= tabu[i] or state.value + d < best_value - eps) and d < change:
                            move, change = (i,), d
                    if (i & 255) == 255 and clock() >= until:
                        return
                left, right = pool(one, delta), pool(zero, delta)
                ordered_zero = sorted(zero, key=weights.__getitem__)
                zero_weights = [weights[i] for i in ordered_zero]
                for a in left:
                    if clock() >= until:
                        break
                    lo = bisect.bisect_left(zero_weights, lower - state.load + weights[a])
                    hi = bisect.bisect_right(zero_weights, upper - state.load + weights[a])
                    compatible = ordered_zero[lo:hi] if hi - lo <= width else [
                        ordered_zero[rng.randrange(lo, hi)] for _ in range(max(1, width // 3))]
                    neighbors = [b for b, c in adj[a] if not bits[b] and c < 0]
                    for b in set(right + compatible + neighbors):
                        if lower <= state.load - weights[a] + weights[b] <= upper:
                            d = delta[a] + delta[b] + 2 * maps[a].get(b, 0.0)
                            if ((step >= tabu[a] and step >= tabu[b]) or
                                    state.value + d < best_value - eps) and d < change:
                                move, change = (a, b), d
                if move is not None:
                    for i in move:
                        state.flip(i)
                        tabu[i] = step + max(2, int(math.sqrt(n) + strength + rng.random() * (2 + strength)))
                    if remember(state.bits, state.value):
                        last_gain = clock()
                if move is None or clock() - last_gain > budget / (7 + strength):
                    # The restart cycle is driven by wall clock; light and heavy perturbations alternate to keep different basins of attraction.
                    base = best_bits if rng.random() < 0.7 else rng.choice(elite)[1]
                    state = State(base)
                    tabu = [0] * n
                    kick_end = min(until, clock() + budget / (120 + 12 * strength))
                    magnitude = max(1, int(math.sqrt(n) * (0.2 + 0.1 * strength) * (1 + rng.random() * strength)))
                    accepted = 0
                    while clock() < kick_end and accepted < magnitude:
                        a, b = rng.randrange(n), rng.randrange(n)
                        if a == b:
                            continue
                        nodes = (a,) if rng.random() < 0.25 else (a, b)
                        load = state.load + sum(-weights[i] if state.bits[i] else weights[i] for i in nodes)
                        count = state.count + sum(-1 if state.bits[i] else 1 for i in nodes)
                        if lower <= load <= upper and 0 < count < n:
                            for i in nodes:
                                state.flip(i)
                            accepted += 1
                            remember(state.bits, state.value)
                    last_gain = clock()

        def choose_blocks(base, k, mode, until):
            if mode == 3:
                parent = list(range(n))
                size = [1] * n

                def root(a):
                    while parent[a] != a:
                        parent[a] = parent[parent[a]]
                        a = parent[a]
                    return a

                remaining = n
                cap = max(2, math.ceil(2.5 * n / max(1, k)))
                for p, (u, v, c) in enumerate(strong_edges):
                    if remaining <= k or clock() >= until:
                        break
                    if base.bits[u] != base.bits[v] or rng.random() < 0.08:
                        continue
                    a, b = root(u), root(v)
                    if a != b and size[a] + size[b] <= cap:
                        if size[a] < size[b]:
                            a, b = b, a
                        parent[b] = a
                        size[a] += size[b]
                        remaining -= 1
                groups = {}
                for i in range(n):
                    groups.setdefault(root(i), []).append(i)
                blocks = list(groups.values())
                if len(blocks) <= k:
                    return blocks
                scored = []
                for block in blocks:
                    inside = set(block)
                    gain = sum(c if base.bits[i] == base.bits[j] else -c
                               for i in block for j, c in adj[i] if j not in inside)
                    scored.append((gain / math.sqrt(len(block)), rng.random(), block))
                scored.sort(key=lambda q: (q[0], q[1]))
                kept = max(1, k * 2 // 3)
                return [q[2] for q in scored[:kept]] + [q[2] for q in rng.sample(scored[kept:], k - kept)]

            if k >= n:
                return [[i] for i in range(n)]
            selected = []
            seen = set()
            sides = [[i for i in range(n) if base.bits[i] == bit] for bit in (0, 1)]
            if mode == 1:
                frontier = []
                counts = [0, 0]
                for side in sides:
                    if side:
                        root_node = rng.choice(heapq.nsmallest(min(len(side), max(2, k // 4)), side,
                                                             key=base.delta.__getitem__))
                        heapq.heappush(frontier, (-float("inf"), root_node))
                while frontier and len(selected) < k and clock() < until:
                    _, i = heapq.heappop(frontier)
                    bit = base.bits[i]
                    if i in seen or counts[bit] >= math.ceil(0.65 * k):
                        continue
                    seen.add(i)
                    selected.append(i)
                    counts[bit] += 1
                    for j, c in adj[i]:
                        if j not in seen:
                            heapq.heappush(frontier, (-abs(c) * (0.75 + 0.5 * rng.random()), j))
            elif mode == 0:
                for side in sides:
                    take = min(len(side), k // 3)
                    selected.extend(heapq.nsmallest(take, side, key=base.delta.__getitem__))
                seen.update(selected)
            else:
                selected = rng.sample(range(n), k // 2)
                seen.update(selected)
            rest = [i for i in range(n) if i not in seen]
            selected.extend(rng.sample(rest, min(k - len(selected), len(rest))))
            return [[i] for i in selected]

        def reoptimize(base, blocks, until):
            nonlocal solver_enabled
            k = len(blocks)
            if k < 1 or clock() >= until:
                return False, False, False
            labels = [-1] * n
            for a, block in enumerate(blocks):
                for i in block:
                    labels[i] = a
            free_count = sum(len(block) for block in blocks)
            bw = [sum(weights[i] for i in block) for block in blocks]
            bc = [len(block) for block in blocks]
            original = [base.bits[block[0]] for block in blocks]
            fixed_load = base.load - sum(w * b for w, b in zip(bw, original))
            fixed_count = base.count - sum(c * b for c, b in zip(bc, original))
            field = [0.0] * k
            couplings = {}
            for p, (u, v, c) in enumerate(graph):
                a, b = labels[u], labels[v]
                if a >= 0 and b >= 0:
                    if a != b:
                        if a > b:
                            a, b = b, a
                        couplings[a, b] = couplings.get((a, b), 0.0) + c
                elif a >= 0:
                    field[a] += c * (1 - 2 * base.bits[v])
                elif b >= 0:
                    field[b] += c * (1 - 2 * base.bits[u])
                if (p & 2047) == 2047 and clock() >= until:
                    return False, False, False
            links = [(a, b, c) for (a, b), c in couplings.items() if c != 0.0]

            if not solver_enabled:
                # Without a solver, conditionally enumerate a neighbourhood whose size is set by the remaining time.
                if k > max(1, int(math.log2(max(2, 40000 * (until - clock()))))):
                    return False, False, False
                local_adj = [[] for _ in range(k)]
                for a, b, c in links:
                    local_adj[a].append((b, c))
                    local_adj[b].append((a, c))
                bits = original[:]
                delta = [(1 - 2 * bits[a]) * field[a] +
                         sum(c if bits[a] == bits[b] else -c for b, c in local_adj[a]) for a in range(k)]
                load, count, value = base.load, base.count, 0.0
                local_best, chosen = 0.0, original[:]
                code = 1
                while code < (1 << k) and clock() < until:
                    a = (code & -code).bit_length() - 1
                    old = bits[a]
                    value += delta[a]
                    for b, c in local_adj[a]:
                        delta[b] += -2 * c if old == bits[b] else 2 * c
                    delta[a] *= -1
                    bits[a] ^= 1
                    load += -bw[a] if old else bw[a]
                    count += -bc[a] if old else bc[a]
                    if lower <= load <= upper and 0 < count < n and value < local_best - eps:
                        local_best, chosen = value, bits[:]
                    code += 1
                candidate = base.bits[:]
                for block, bit in zip(blocks, chosen):
                    for i in block:
                        candidate[i] = bit
                improved = remember(candidate)
                exact = code == (1 << k)
                return improved, exact, exact and free_count == n and k == n

            try:
                nv = k + len(links)
                objective = np.array(field + [c for _, _, c in links])
                rows, cols, values, lows, highs = [], [], [], [], []

                def add(entries, lo, hi):
                    r = len(lows)
                    for col, value in entries:
                        if value != 0:
                            rows.append(r)
                            cols.append(col)
                            values.append(value)
                    lows.append(lo)
                    highs.append(hi)

                divisor = 0
                for w in bw:
                    divisor = math.gcd(divisor, w)
                divisor = max(1, divisor)
                add(enumerate([w // divisor for w in bw]),
                    (lower - fixed_load + divisor - 1) // divisor, (upper - fixed_load) // divisor)
                add(enumerate(bc), 1 - fixed_count, n - 1 - fixed_count)
                for p, (a, b, c) in enumerate(links):
                    y = k + p
                    # Depending on the cost sign, keep the two inequalities that suffice to guarantee the cut relations of the integer optimum.
                    if c > 0:
                        add(((a, 1), (b, -1), (y, -1)), -np.inf, 0)
                        add(((a, -1), (b, 1), (y, -1)), -np.inf, 0)
                    else:
                        add(((a, -1), (b, -1), (y, 1)), -np.inf, 0)
                        add(((a, 1), (b, 1), (y, 1)), -np.inf, 2)
                incumbent = sum(field[a] * original[a] for a in range(k))
                incumbent += sum(c for a, b, c in links if original[a] != original[b])
                add(enumerate(objective), -np.inf, incumbent + eps)
                lb, ub = np.zeros(nv), np.ones(nv)
                if free_count == n:
                    ub[max(range(k), key=bc.__getitem__)] = 0
                matrix = coo_matrix((values, (rows, cols)), shape=(len(lows), nv)).tocsc()
                available = until - clock()
                if available <= 0:
                    return False, False, False
                result = milp(
                    objective, integrality=np.concatenate((np.ones(k), np.zeros(len(links)))),
                    bounds=Bounds(lb, ub), constraints=LinearConstraint(matrix, lows, highs),
                    options={"time_limit": max(1e-6, 0.88 * available), "mip_rel_gap": 0.0, "presolve": True},
                )
                improved = False
                if result.x is not None:
                    candidate = base.bits[:]
                    for a, block in enumerate(blocks):
                        bit = int(result.x[a] >= 0.5)
                        for i in block:
                            candidate[i] = bit
                    improved = remember(candidate)
                exact = result.status == 0 and result.x is not None and feasible(candidate)
                return improved, exact, exact and free_count == n and k == n
            except Exception:
                solver_enabled = False
                return False, False, False

        # First build a good feasible partition, then alternate intensification and large neighbourhood search by wall clock.
        tabu_search(min(deadline, started + budget * 0.16))
        serial = 0
        failures = 0
        while clock() < deadline:
            if best_value <= global_lower + eps:
                break
            before = best_value
            if solver_enabled:
                k = min(n, neighborhood_size + int(strength * math.sqrt(failures)))
            else:
                k = min(n, max(2, int(math.log2(max(2, 12000 * budget)))))
            base = State(best_bits if rng.random() < 0.8 else rng.choice(elite)[1])
            span = budget / (8 + 2 * strength)
            local_end = min(deadline - budget * 0.025, clock() + span * (1 + 0.12 * strength))
            if clock() < local_end:
                blocks = choose_blocks(base, k, serial % 4, local_end)
                improved, exact, optimal = reoptimize(base, blocks, local_end)
                if optimal:
                    break
                if exact and not improved:
                    neighborhood_size = min(n, neighborhood_size + max(1, int(strength)))
                elif not exact and not improved:
                    neighborhood_size = max(8, int(neighborhood_size * 0.96))
                failures = 0 if improved else failures + 1
                if improved:
                    state = State(best_bits)
                    tabu = [0] * n
                    last_gain = clock()
            if clock() < deadline:
                tabu_search(min(deadline, clock() + budget / (15 + 3 * strength)))
            if best_value < before - eps:
                elite.append((best_value, best_bits[:]))
                elite.sort(key=lambda item: item[0])
                del elite[elite_cap:]
            serial += 1
        return finish()
    except Exception:
        return finish()
