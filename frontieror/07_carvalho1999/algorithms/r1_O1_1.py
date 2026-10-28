"""Hybrid bitset-filling and tabu bin-reduction search
Combines first fit and best fit, bitset subset filling, overload tabu repair, bin-pattern crossover and symmetry-pruned local repacking.
"""

import bisect
import math
import random
import time


def _search(weights, capacity, budget, started, incumbent):
    clock = time.perf_counter
    n = len(weights)
    deadline = started + budget * 0.97
    if clock() >= deadline:
        return
    items = list(range(n))
    descending = sorted(items, key=lambda i: (-weights[i], i))
    seed = n * 1000003 + capacity
    for i, w in enumerate(weights):
        seed = (seed * 1000003 + (i + 1) * (w + 17)) & ((1 << 63) - 1)
    rng = random.Random(seed)
    effort = math.log1p(budget)
    quantum = budget / (24.0 + math.sqrt(n))
    archive_limit = max(2, int(4 + 3 * effort * math.sqrt(math.log1p(n))))
    best = [[i] for i in items]
    best_quality = (n, -sum(w * w for w in weights))
    archive = []
    marks = set()

    def loads_of(bins):
        return [sum(weights[i] for i in b) for b in bins]

    def save(bins):
        assignment = [0] * n
        for b, group in enumerate(bins):
            for i in group:
                assignment[i] = b
        incumbent[0] = {
            "bin_activation": [1] * len(bins) + [0] * (n - len(bins)),
            "item_assignments": assignment,
        }

    def admit(bins):
        nonlocal best, best_quality
        if bins is None:
            return
        bins = [list(b) for b in bins if b]
        loads = loads_of(bins)
        quality = (len(bins), -sum(w * w for w in loads))
        if quality < best_quality:
            best = bins
            best_quality = quality
            save(best)
        # Swapping items of equal weight does not count as a new bin pattern.
        mark = tuple(sorted(tuple(sorted(weights[i] for i in b)) for b in bins))
        if mark in marks:
            return
        marks.add(mark)
        archive.append((quality, mark, bins))
        archive.sort(key=lambda entry: entry[0])
        if len(archive) > archive_limit:
            removed = archive.pop()
            marks.remove(removed[1])

    def construct(order, cutoff, initial=None, first=False):
        bins = [] if initial is None else [list(b) for b in initial]
        loads = loads_of(bins)
        if first:
            # A segment tree finds the leftmost available bin, avoiding bin-by-bin scanning in first fit.
            size = 1
            while size < len(bins) + len(order):
                size *= 2
            tree = [-1] * (2 * size)
            for b, load in enumerate(loads):
                tree[size + b] = capacity - load
            for p in range(size - 1, 0, -1):
                tree[p] = max(tree[2 * p], tree[2 * p + 1])
        else:
            spaces = sorted((capacity - load, b) for b, load in enumerate(loads))
        for pos, i in enumerate(order):
            if (pos & 31) == 0 and clock() >= cutoff:
                bins.extend([j] for j in order[pos:])
                return bins
            w = weights[i]
            b = -1
            if first:
                if tree[1] >= w:
                    p = 1
                    while p < size:
                        p *= 2
                        if tree[p] < w:
                            p += 1
                    b = p - size
            else:
                p = bisect.bisect_left(spaces, (w, -1))
                if p < len(spaces):
                    _, b = spaces.pop(p)
            if b < 0:
                b = len(bins)
                bins.append([i])
                loads.append(w)
            else:
                bins[b].append(i)
                loads[b] += w
            if first:
                p = size + b
                tree[p] = capacity - loads[b]
                p //= 2
                while p:
                    tree[p] = max(tree[2 * p], tree[2 * p + 1])
                    p //= 2
            else:
                bisect.insort(spaces, (capacity - loads[b], b))
        return bins

    admit(construct(descending, deadline))
    if clock() >= deadline:
        return
    admit(construct(descending, deadline, first=True))
    total = sum(weights)
    lower = max(1, (total + capacity - 1) // capacity)
    ascending_weights = sorted(weights)
    if n < 3 or sum(ascending_weights[:3]) > capacity:
        order = list(reversed(descending))
        paired = []
        lo, hi = 0, n - 1
        while lo <= hi:
            group = [order[hi]]
            if lo < hi and weights[order[lo]] + weights[order[hi]] <= capacity:
                group.append(order[lo])
                lo += 1
            paired.append(group)
            hi -= 1
        admit(paired)
        return
    if len(best) <= lower or clock() >= deadline:
        return

    # Dedicated bins occupied by large items and space that medium items cannot use give a stronger valid lower bound.
    prefix = [0]
    for w in ascending_weights:
        prefix.append(prefix[-1] + w)
    half = bisect.bisect_right(ascending_weights, capacity // 2)
    previous = -1
    for p in range(half):
        if (p & 63) == 0 and clock() >= deadline:
            return
        alpha = ascending_weights[p]
        if alpha == previous or alpha == 0:
            continue
        previous = alpha
        split = bisect.bisect_right(ascending_weights, capacity - alpha)
        medium_sum = prefix[half] - prefix[p]
        large_slack = (split - half) * capacity - (prefix[split] - prefix[half])
        extra = max(0, (medium_sum - large_slack + capacity - 1) // capacity)
        lower = max(lower, n - half + extra)
    if len(best) <= lower:
        return

    scale = capacity
    for w in weights:
        scale = math.gcd(scale, w)
    scale = max(1, scale)
    # Bitset precision and the space for stored history scale jointly with the budget and the number of candidates.
    bit_ceiling = max(256, int(65536 * (1 + effort)))
    bit_memory = max(4096, int(8388608 * (1 + math.sqrt(budget))))

    def subset(candidates, room, cutoff):
        if room <= 0 or not candidates:
            return []
        limit = min(bit_ceiling, max(256, bit_memory // max(1, len(candidates))))
        unit = scale * max(1, (room + scale * limit - 1) // (scale * limit))
        target = room // unit
        if target <= 0:
            return []
        mask = (1 << (target + 1)) - 1
        reach = 1
        history = []
        for p, i in enumerate(candidates):
            if (p & 15) == 0 and clock() >= cutoff:
                break
            # Rounding up only reserves capacity conservatively and never produces an overloaded subset.
            w = (weights[i] + unit - 1) // unit
            if 0 < w <= target:
                expanded = (reach | (reach << w)) & mask
                if expanded != reach:
                    history.append((i, w, reach))
                    reach = expanded
                    if (reach >> target) & 1:
                        break
        value = reach.bit_length() - 1
        chosen = []
        for i, w, before in reversed(history):
            if not ((before >> value) & 1):
                chosen.append(i)
                value -= w
                if value == 0:
                    break
        return chosen

    def varied_order(pool, strength):
        order = list(pool)
        if rng.random() < strength * strength:
            rng.shuffle(order)
        else:
            noise = capacity * (0.02 + 0.35 * strength)
            order.sort(key=lambda i: -weights[i] + rng.uniform(-noise, noise))
        return order

    def mbs_pack(pool, cutoff, randomized=False):
        remaining = sorted(pool, key=lambda i: (-weights[i], i))
        bins = []
        while remaining and clock() < cutoff:
            seed_pos = 0
            if randomized:
                big = 0
                for i in remaining:
                    if 2 * weights[i] <= capacity:
                        break
                    big += 1
                window = big or min(len(remaining), max(1, int(1 + effort)))
                seed_pos = rng.randrange(window)
            anchor = remaining[seed_pos]
            room = capacity - weights[anchor]
            candidates = [i for i in remaining if i != anchor and weights[i] <= room]
            if randomized:
                rng.shuffle(candidates)
            selected = subset(candidates, room, cutoff)
            chosen = set(selected)
            chosen.add(anchor)
            room -= sum(weights[i] for i in selected)
            # Restore the quantisation slack and place the zero-weight items at the same time.
            for i in remaining:
                if i not in chosen and weights[i] <= room:
                    chosen.add(i)
                    room -= weights[i]
            bins.append(list(chosen))
            remaining = [i for i in remaining if i not in chosen]
        if remaining:
            return construct(remaining, deadline, bins)
        return bins

    admit(mbs_pack(items, min(deadline, clock() + 0.12 * budget)))
    if len(best) <= lower:
        return

    def exact_repack(pool, count, cutoff):
        if not pool:
            return []
        if count <= 0 or sum(weights[i] for i in pool) > count * capacity:
            return None
        order = sorted(pool, key=lambda i: (-weights[i], i))
        candidate = construct(order, cutoff)
        if len(candidate) <= count:
            return candidate
        if clock() >= cutoff:
            return None
        # Two-bin repacking is reduced directly to subset sum instead of repeated trials on a permutation tree.
        if count == 2:
            chosen = subset(order, capacity, cutoff)
            used = set(chosen)
            other = [i for i in order if i not in used]
            if sum(weights[i] for i in other) <= capacity:
                return [chosen, other]
        if clock() >= cutoff:
            return None
        candidate = mbs_pack(order, min(cutoff, clock() + (cutoff - clock()) * 0.18))
        if len(candidate) <= count:
            return candidate
        # An explicit stack removes recursion-depth limits; bins with the same residual capacity are branched on only once.
        m = len(order)
        room = [capacity] * count
        packed = [[] for _ in range(count)]
        choices = [None] * m
        states = [None] * m
        assigned = [-1] * m
        failed = set()
        cache_limit = max(128, int((1 + effort) * 2048))
        pos = 0
        while clock() < cutoff:
            if pos == m:
                return [b for b in packed if b]
            if choices[pos] is None:
                state = (pos, tuple(sorted(room)))
                states[pos] = state
                if state in failed:
                    choices[pos] = []
                else:
                    w = weights[order[pos]]
                    seen = set()
                    available = []
                    for b in sorted(range(count), key=lambda j: room[j]):
                        if room[b] >= w and room[b] not in seen:
                            seen.add(room[b])
                            available.append(b)
                    choices[pos] = available[::-1]
            if choices[pos]:
                b = choices[pos].pop()
                i = order[pos]
                room[b] -= weights[i]
                packed[b].append(i)
                assigned[pos] = b
                pos += 1
                if pos < m:
                    choices[pos] = None
            else:
                if len(failed) < cache_limit:
                    failed.add(states[pos])
                choices[pos] = None
                if pos == 0:
                    return None
                pos -= 1
                b = assigned[pos]
                room[b] += weights[order[pos]]
                packed[b].pop()
        return None

    def tabu_reduce(parent, cutoff):
        """Opens one bin fewer and repairs temporary overload with moves, swaps and one-for-two exchanges."""
        count = len(parent) - 1
        if count < lower:
            return None
        parent_loads = loads_of(parent)
        ranked = sorted(range(len(parent)), key=lambda b: parent_loads[b])
        began = clock()
        span = max(0.0, cutoff - began)
        restart_span = span / (1.5 + effort * 0.3)
        tenure_base = max(2, int(math.sqrt(n) * (0.3 + effort * 0.12)))
        drop_window = min(len(parent), max(1, int(1 + math.sqrt(count) * effort / 8)))
        while clock() < cutoff:
            removed = ranked[0] if rng.random() < 0.5 else ranked[rng.randrange(drop_window)]
            packed = [list(b) for j, b in enumerate(parent) if j != removed]
            loads = [v for j, v in enumerate(parent_loads) if j != removed]
            for i in sorted(parent[removed], key=lambda j: -weights[j]):
                w = weights[i]
                b = min(range(count), key=lambda j: (max(0, loads[j] + w - capacity), abs(capacity - loads[j] - w)))
                packed[b].append(i)
                loads[b] += w
            for b in packed:
                b.sort(key=lambda i: weights[i])
            tabu = {}
            step = 0
            over = [max(0, w - capacity) for w in loads]
            energy = (sum(over), sum(v * v for v in over))
            record = energy
            last_gain = clock()
            while clock() < cutoff:
                if energy[0] == 0:
                    return [b for b in packed if b]
                now = clock()
                if now - last_gain > restart_span:
                    break
                overloaded = [b for b in range(count) if loads[b] > capacity]
                source = max(overloaded, key=lambda b: loads[b]) if rng.random() < 0.6 else rng.choice(overloaded)
                ob = loads[source] - capacity
                best_move = None
                best_score = None
                ties = 0
                pair_search = now - last_gain > restart_span / (3 + effort)
                outgoing = list(packed[source])
                rng.shuffle(outgoing)

                def consider(i, dest, incoming, delta):
                    nonlocal best_move, best_score, ties
                    if delta == 0:
                        return
                    oc = max(0, loads[dest] - capacity)
                    nb = max(0, ob - delta)
                    nc = max(0, loads[dest] + delta - capacity)
                    score = (energy[0] + nb + nc - ob - oc,
                             energy[1] + nb * nb + nc * nc - ob * ob - oc * oc)
                    if best_score is not None and score > best_score:
                        return
                    blocked = tabu.get(i, (-1, -1))
                    if blocked[0] == dest and blocked[1] > step and score >= record:
                        return
                    for j in incoming:
                        blocked = tabu.get(j, (-1, -1))
                        if blocked[0] == source and blocked[1] > step and score >= record:
                            return
                    if best_score is None or score < best_score:
                        best_score = score
                        best_move = (i, dest, incoming)
                        ties = 1
                    else:
                        ties += 1
                        if rng.randrange(ties) == 0:
                            best_move = (i, dest, incoming)

                values = [[weights[i] for i in b] for b in packed]
                for i in outgoing:
                    if clock() >= cutoff:
                        break
                    wi = weights[i]
                    for dest in range(count):
                        if (dest & 15) == 0 and clock() >= cutoff:
                            break
                        if dest == source:
                            continue
                        consider(i, dest, (), wi)
                        vals = values[dest]
                        if not vals:
                            continue
                        # The sum of squared overloads is minimised where the two bins are balanced; check the neighbouring discrete weights.
                        twice_target = 2 * wi - loads[source] + loads[dest]
                        target = twice_target // 2
                        p = bisect.bisect_left(vals, target)
                        for q in (p - 1, p):
                            if 0 <= q < len(vals):
                                j = packed[dest][q]
                                consider(i, dest, (j,), wi - weights[j])
                        if pair_search and len(vals) >= 2:
                            lo, hi = 0, len(vals) - 1
                            while lo < hi:
                                if (lo & 31) == 0 and clock() >= cutoff:
                                    break
                                pair_sum = vals[lo] + vals[hi]
                                consider(i, dest, (packed[dest][lo], packed[dest][hi]), wi - pair_sum)
                                if pair_sum * 2 < twice_target:
                                    lo += 1
                                else:
                                    hi -= 1
                if best_move is None:
                    break
                i, dest, incoming = best_move
                delta = weights[i] - sum(weights[j] for j in incoming)
                packed[source].remove(i)
                for j in incoming:
                    packed[dest].remove(j)
                    packed[source].append(j)
                    tabu[j] = (dest, step + tenure_base + rng.randrange(tenure_base))
                packed[dest].append(i)
                tabu[i] = (source, step + tenure_base + rng.randrange(tenure_base))
                packed[source].sort(key=lambda j: weights[j])
                packed[dest].sort(key=lambda j: weights[j])
                loads[source] -= delta
                loads[dest] += delta
                energy = best_score
                step += 1
                if energy < record:
                    record = energy
                    last_gain = clock()
                if energy[0] == 0:
                    return [b for b in packed if b]
        return None

    def neighborhood(parent, progress):
        loads = loads_of(parent)
        noise = capacity * rng.random() * (0.1 + 0.25 * progress)
        ranked = sorted(range(len(parent)), key=lambda b: loads[b] + rng.random() * noise)
        selected = []
        slack = 0
        for b in ranked:
            selected.append(b)
            slack += capacity - loads[b]
            if slack >= capacity and len(selected) >= 2:
                break
        if slack < capacity:
            return []
        chosen = set(selected)
        gap = max(capacity - loads[b] for b in selected)
        donors = [b for b in range(len(parent)) if b not in chosen]
        donors.sort(key=lambda b: (sum(weights[i] > gap for i in parent[b]),
                                   -sum(weights[i] for i in parent[b] if weights[i] <= gap) + rng.random() * capacity * 0.2))
        depth = max(1, int(1 + effort * (0.5 + progress) + math.sqrt(len(parent)) * progress * 0.15))
        selected.extend(donors[:rng.randint(1, max(1, depth))])
        return selected

    def recombine(a, b, cutoff, progress):
        patterns = [list(group) for group in a + b]
        patterns.sort(key=lambda group: -sum(weights[i] for i in group) + rng.random() * capacity * (0.08 + 0.3 * progress))
        used = set()
        locked = []
        retained = int(n * rng.uniform(0.25, 0.8 - 0.2 * progress))
        for group in patterns:
            if len(used) >= retained or clock() >= cutoff:
                break
            if all(i not in used for i in group):
                used.update(group)
                locked.append(group)
        leftovers = [i for i in items if i not in used]
        order = varied_order(leftovers, rng.random() * (0.4 + 0.5 * progress))
        return construct(order, cutoff, locked)

    phase = 0
    last_improved = clock()
    last_count = len(best)
    while clock() < deadline:
        if len(best) <= lower:
            break
        if len(best) < last_count:
            last_count = len(best)
            last_improved = clock()
        phase += 1
        now = clock()
        progress = min(1.0, (now - started) / max(budget, 1e-12))
        stagnation = min(1.0, (now - last_improved) / max(budget * 0.3, 1e-12))
        cutoff = min(deadline, now + quantum * (1 + progress + stagnation))
        if phase % 5 in (1, 3):
            parent = best if phase % 5 == 1 else archive[rng.randrange(len(archive))][2]
            admit(tabu_reduce(parent, cutoff))
        elif phase % 5 == 2:
            parent = best if rng.random() < 0.7 else archive[rng.randrange(len(archive))][2]
            selected = neighborhood(parent, progress + 0.5 * stagnation)
            if not selected:
                continue
            chosen = set(selected)
            pool = [i for b in selected for i in parent[b]]
            repaired = exact_repack(pool, len(selected) - 1, cutoff)
            if repaired is not None:
                admit([list(group) for b, group in enumerate(parent) if b not in chosen] + repaired)
            elif clock() < deadline:
                # An unsuccessful bin reduction can still yield a feasible new pattern with the same number of bins.
                rebuilt = construct(varied_order(pool, rng.random()), min(deadline, clock() + quantum * 0.1))
                if len(rebuilt) <= len(selected):
                    admit([list(group) for b, group in enumerate(parent) if b not in chosen] + rebuilt)
        elif phase % 5 == 4:
            admit(mbs_pack(items, cutoff, True))
        else:
            a = archive[rng.randrange(len(archive))][2]
            b = archive[rng.randrange(len(archive))][2]
            admit(recombine(a, b, cutoff, progress))


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    incumbent = [{"bin_activation": [], "item_assignments": []}]
    try:
        n = max(0, int(instance["n"]))
        incumbent[0] = {"bin_activation": [1] * n, "item_assignments": list(range(n))}
        if n == 0:
            return incumbent[0]
        weights = [int(w) for w in instance["weights"]]
        capacity = int(instance["capacity"])
        if len(weights) != n or capacity < 0 or any(w < 0 or w > capacity for w in weights):
            return incumbent[0]
        if not any(weights):
            incumbent[0] = {"bin_activation": [1] + [0] * (n - 1), "item_assignments": [0] * n}
            return incumbent[0]
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget <= 0:
            return incumbent[0]
        _search(weights, capacity, budget, started, incumbent)
    except Exception:
        # If the improvement phase exits unexpectedly, the complete feasible solution already saved is still returned.
        pass
    return incumbent[0]
