"""Clique constraint generation with incremental tabu swap algorithm
Accumulates violated cliques and optimises the edge-removal set by weighted covering swaps, combined with resumable colouring branch and bound to certify the maximum clique and multipartite graph construction.
"""
import math
import random
import time


class _CliqueSearch:
    """An explicit search stack supports resumption; threshold mode also collects multiple violated cliques."""

    def __init__(self, adj, deadline, threshold=None, hint=0):
        self.adj = adj
        self.n = len(adj)
        self.threshold = threshold
        self.best = 0
        self.size = 0
        self.hits = set()
        self.stack = []
        self.pending = ((1 << self.n) - 1, 0, 0)
        self.complete = False
        for seed in (hint, 0):
            clique = 0
            candidates = (1 << self.n) - 1
            while seed:
                bit = seed & -seed
                seed ^= bit
                if candidates & bit:
                    v = bit.bit_length() - 1
                    clique |= bit
                    candidates &= adj[v]
            while candidates and time.perf_counter() < deadline:
                scan = candidates
                vertex = (scan & -scan).bit_length() - 1
                degree = -1
                while scan:
                    bit = scan & -scan
                    scan ^= bit
                    v = bit.bit_length() - 1
                    d = (adj[v] & candidates).bit_count()
                    if d > degree:
                        vertex, degree = v, d
                    if time.perf_counter() >= deadline:
                        break
                clique |= 1 << vertex
                candidates &= adj[vertex]
            if clique.bit_count() > self.size:
                self.best, self.size = clique, clique.bit_count()
        if not self.best and self.n:
            self.best, self.size = 1, 1
        if threshold is not None and self.size >= threshold:
            mask, cut = self.best, 0
            for _ in range(threshold):
                bit = mask & -mask
                mask ^= bit
                cut |= bit
            self.hits.add(cut)

    def _color(self, candidates, deadline):
        order, bounds = [], []
        color = 0
        while candidates:
            color += 1
            available = candidates
            while available:
                bit = available & -available
                v = bit.bit_length() - 1
                candidates ^= bit
                available &= ~bit & ~self.adj[v]
                order.append(v)
                bounds.append(color)
                if (len(order) & 63) == 0 and time.perf_counter() >= deadline:
                    return None
            if time.perf_counter() >= deadline:
                return None
        return order, bounds

    def run(self, deadline, batch=1):
        while not self.complete and time.perf_counter() < deadline:
            if self.threshold is not None and len(self.hits) >= batch:
                return
            bound = self.size
            if self.threshold is not None:
                bound = min(bound, self.threshold - 1)
            if self.pending is not None:
                candidates, clique, size = self.pending
                if size + candidates.bit_count() <= bound:
                    self.pending = None
                    continue
                colored = self._color(candidates, deadline)
                if colored is None:
                    return
                self.pending = None
                order, colors = colored
                if order and size + colors[-1] > bound:
                    self.stack.append([candidates, order, colors, len(order) - 1,
                                       clique, size])
                continue
            if not self.stack:
                self.complete = True
                return
            frame = self.stack[-1]
            i = frame[3]
            if i < 0 or frame[5] + frame[2][i] <= bound:
                self.stack.pop()
                continue
            v = frame[1][i]
            bit = 1 << v
            candidates = frame[0] & self.adj[v]
            frame[0] &= ~bit
            frame[3] -= 1
            clique = frame[4] | bit
            size = frame[5] + 1
            if size > self.size:
                self.size, self.best = size, clique
            if self.threshold is not None and size >= self.threshold:
                self.hits.add(clique)
            elif candidates and size + candidates.bit_count() > bound:
                self.pending = (candidates, clique, size)


class _Cover:
    """Remove at least one edge from every discovered clique; incrementally maintain covering gains and undo losses."""

    def __init__(self, adj, pairs):
        self.adj = adj[:]
        self.pairs = pairs
        self.chosen = set()
        self.cuts = []
        self.seen = set()
        self.count = []
        self.xor = []
        self.weight = []
        self.incidence = [[] for _ in pairs]
        self.gain = [0] * len(pairs)
        self.loss = [0] * len(pairs)
        self.open = []
        self.location = []

    def _uncover(self, c):
        self.location[c] = len(self.open)
        self.open.append(c)

    def _cover(self, c):
        p = self.location[c]
        last = self.open.pop()
        if p < len(self.open):
            self.open[p] = last
            self.location[last] = p
        self.location[c] = -1

    def cut(self, mask, edge_ids):
        if mask in self.seen or not edge_ids:
            return
        self.seen.add(mask)
        c = len(self.cuts)
        self.cuts.append(edge_ids)
        selected = [e for e in edge_ids if e in self.chosen]
        self.count.append(len(selected))
        xor = 0
        for e in selected:
            xor ^= e + 1
        self.xor.append(xor)
        self.weight.append(1)
        self.location.append(-1)
        for e in edge_ids:
            self.incidence[e].append(c)
        if not selected:
            self._uncover(c)
            for e in edge_ids:
                self.gain[e] += 1
        elif len(selected) == 1:
            self.loss[selected[0]] += 1

    def add(self, e):
        self.chosen.add(e)
        u, v = self.pairs[e]
        self.adj[u] &= ~(1 << v)
        self.adj[v] &= ~(1 << u)
        for c in self.incidence[e]:
            count, weight = self.count[c], self.weight[c]
            if count == 0:
                self._cover(c)
                for f in self.cuts[c]:
                    self.gain[f] -= weight
                self.loss[e] += weight
            elif count == 1:
                self.loss[self.xor[c] - 1] -= weight
            self.count[c] += 1
            self.xor[c] ^= e + 1

    def drop(self, e):
        self.chosen.remove(e)
        u, v = self.pairs[e]
        self.adj[u] |= 1 << v
        self.adj[v] |= 1 << u
        for c in self.incidence[e]:
            count, weight = self.count[c], self.weight[c]
            self.count[c] -= 1
            self.xor[c] ^= e + 1
            if count == 1:
                self.loss[e] -= weight
                self._uncover(c)
                for f in self.cuts[c]:
                    self.gain[f] += weight
            elif count == 2:
                self.loss[self.xor[c] - 1] += weight

    def penalize(self):
        for c in self.open:
            self.weight[c] += 1
            for e in self.cuts[c]:
                self.gain[e] += 1


def _solve(instance, time_limit_s, answer, started):
    duration = max(0.0, float(time_limit_s))
    deadline = started + duration * 0.97
    vertices = instance['vertices']
    n = len(vertices)
    edges = instance['edges']
    m = len(edges)
    budget = max(0, min(int(instance['interdiction_budget_k']), m))
    position = {v: i for i, v in enumerate(vertices)}
    original = [0] * n
    raw_pairs, raw_indices = [], []
    seen = set()
    for i, edge in enumerate(edges):
        u, v = position[edge[0]], position[edge[1]]
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            continue
        seen.add((u, v))
        raw_pairs.append((u, v))
        raw_indices.append(i)
        original[u] |= 1 << v
        original[v] |= 1 << u
    ordering = sorted(range(n), key=lambda v: original[v].bit_count(), reverse=True)
    inverse = [0] * n
    for i, v in enumerate(ordering):
        inverse[v] = i
    adj = [0] * n
    pairs, pair_index = [], {}
    for e, (u, v) in enumerate(raw_pairs):
        u, v = inverse[u], inverse[v]
        if u > v:
            u, v = v, u
        pairs.append((u, v))
        pair_index[(u, v)] = e
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    edge_count = len(pairs)
    budget = min(budget, edge_count)
    rng = random.Random(0x51C07)

    def save(selected, clique):
        w, x = [0] * m, [0] * n
        for e in selected:
            w[raw_indices[e]] = 1
        while clique:
            bit = clique & -clique
            clique ^= bit
            x[ordering[bit.bit_length() - 1]] = 1
        answer.update(interdiction_status=w, clique_membership=x,
                      max_clique_size=sum(x))

    if not n or budget == edge_count:
        save(range(edge_count), 1 if n else 0)
        return answer

    strength = 1.0 + math.log1p(duration)
    batch = max(1, int(math.sqrt(duration + 1.0) * math.log2(n + 1)))
    oracle_slice = duration / (12.0 + math.sqrt(n + 1))
    warm_slice = duration / (100.0 + n)
    base = _CliqueSearch(adj, min(deadline, time.perf_counter() + warm_slice))
    base.run(min(deadline, started + duration * (0.12 if budget else 0.96)))
    save(set(), base.best)
    certified = base.complete
    best_size = base.size if certified else n + 1
    best_selected = set()
    best_clique = base.best
    original_witness = base.best
    lower_bound = 2

    # Turán lower bound for any clique of the original graph after edge removal; does not depend on whether the original maximum clique is certified.
    c = original_witness.bit_count()
    for r in range(1, c + 1):
        a, b = divmod(c, r)
        required = r * a * (a - 1) // 2 + b * a
        if required <= budget:
            lower_bound = max(lower_bound, r)
            break
    if certified and (best_size <= lower_bound or budget == 0):
        return answer
    if budget == 0:
        base.run(deadline)
        save(set(), base.best)
        return answer

    def edge_ids(mask):
        nodes = []
        while mask:
            bit = mask & -mask
            mask ^= bit
            nodes.append(bit.bit_length() - 1)
        return [pair_index[(u, v)] for i, u in enumerate(nodes) for v in nodes[i + 1:]]

    # Fix a known clique across different groups; deleting all within-group edges allows certification by the colouring upper bound.
    partition_deadline = min(deadline, time.perf_counter() + duration * 0.055)
    r = min(c, best_size - 1)
    while r >= lower_bound and r >= 2 and time.perf_counter() < partition_deadline:
        if budget * r * 2 < edge_count:
            break
        seeds = []
        mask = original_witness
        while mask and len(seeds) < r:
            bit = mask & -mask
            mask ^= bit
            seeds.append(bit.bit_length() - 1)
        groups, labels = [0] * r, [-1] * n
        pinned = set(seeds)
        for g, v in enumerate(seeds):
            groups[g] |= 1 << v
            labels[v] = g
        for v in range(n):
            if labels[v] < 0:
                g = min(range(r), key=lambda j: ((adj[v] & groups[j]).bit_count(),
                                                 groups[j].bit_count(), rng.random()))
                groups[g] |= 1 << v
                labels[v] = g
        cost = sum(labels[u] == labels[v] for u, v in pairs)
        movable = [v for v in range(n) if v not in pinned]
        last_gain = time.perf_counter()
        while cost > budget and movable and time.perf_counter() < partition_deadline:
            v = rng.choice(movable)
            old = labels[v]
            old_cost = (adj[v] & groups[old]).bit_count()
            g = min(range(r), key=lambda j: ((adj[v] & groups[j]).bit_count(), rng.random()))
            new_cost = (adj[v] & groups[g]).bit_count()
            if new_cost <= old_cost:
                groups[old] &= ~(1 << v)
                groups[g] |= 1 << v
                labels[v] = g
                cost += new_cost - old_cost
                if new_cost < old_cost:
                    last_gain = time.perf_counter()
            if time.perf_counter() - last_gain > duration / (40.0 + n):
                v = rng.choice(movable)
                old, g = labels[v], rng.randrange(r)
                cost += (adj[v] & groups[g]).bit_count() - (adj[v] & groups[old]).bit_count()
                groups[old] &= ~(1 << v)
                groups[g] |= 1 << v
                labels[v] = g
                last_gain = time.perf_counter()
        if cost > budget:
            break
        best_selected = {e for e, (u, v) in enumerate(pairs) if labels[u] == labels[v]}
        best_clique = sum(1 << v for v in seeds)
        best_size, certified = r, True
        save(best_selected, best_clique)
        if best_size <= lower_bound:
            return answer
        r -= 1

    cover = _Cover(adj, pairs)
    for e in best_selected:
        cover.add(e)
    target = (best_size if certified else max(2, base.size)) - 1
    seed_mask = best_clique if certified else original_witness
    if seed_mask.bit_count() >= target + 1:
        cover.cut(seed_mask, edge_ids(seed_mask))
    age = [0] * edge_count
    tabu = [0] * edge_count
    centrality = [(adj[u] & adj[v]).bit_count() for u, v in pairs]
    moves = 0
    tenure = max(1, int(math.sqrt(budget + 1) * strength / 2))
    stall_window = duration / (4.0 + math.sqrt(budget + 1))
    last_progress = time.perf_counter()
    smallest_open = len(cover.open)
    pending = None
    hint = best_clique
    next_verify = time.perf_counter() + duration * 0.12

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        if not cover.open:
            if pending is None:
                pending = _CliqueSearch(cover.adj[:], min(deadline, now + warm_slice),
                                        target + 1, hint)
            pending.run(min(deadline, now + oracle_slice), batch)
            hint = pending.best
            if pending.hits:
                for mask in pending.hits:
                    cover.cut(mask, edge_ids(mask))
                pending = None
                smallest_open = len(cover.open)
                last_progress = time.perf_counter()
            elif pending.complete:
                theta = pending.size
                if not certified or theta < best_size:
                    best_size, best_clique = theta, pending.best
                    best_selected = set(cover.chosen)
                    certified = True
                    save(best_selected, best_clique)
                pending = None
                if best_size <= lower_bound:
                    break
                target = best_size - 1
                cover.cut(best_clique, edge_ids(best_clique))
                smallest_open = len(cover.open)
                last_progress = time.perf_counter()
            continue

        # When certifying the original graph is hard, separately certify the residual graph whose budget is already fully used.
        if not certified and len(cover.chosen) == budget and now >= next_verify:
            check = _CliqueSearch(cover.adj[:], min(deadline, now + warm_slice), hint=hint)
            check.run(min(deadline, now + duration * 0.10))
            if check.complete:
                certified, best_size, best_clique = True, check.size, check.best
                best_selected = set(cover.chosen)
                save(best_selected, best_clique)
                if best_size <= lower_bound:
                    break
                target = best_size - 1
                # The old target may be too low; keep only clique constraints still valid for the new target.
                old_masks = [mask for mask in cover.seen if mask.bit_count() > target]
                cover = _Cover(adj, pairs)
                for e in best_selected:
                    cover.add(e)
                for mask in old_masks:
                    cover.cut(mask, edge_ids(mask))
                cover.cut(best_clique, edge_ids(best_clique))
                smallest_open = len(cover.open)
                last_progress = time.perf_counter()
            next_verify = time.perf_counter() + duration * 0.12
            continue

        if now - last_progress >= stall_window:
            if best_selected and rng.random() < 0.5:
                for e in list(cover.chosen - best_selected):
                    cover.drop(e)
                for e in best_selected - cover.chosen:
                    cover.add(e)
            depth = min(len(cover.chosen), max(1, int(math.sqrt(budget + 1) *
                         (1 + (now - started) / max(duration, 1e-9)) * strength / 2)))
            for e in rng.sample(list(cover.chosen), depth):
                cover.drop(e)
                tabu[e] = moves
            last_progress = time.perf_counter()
            smallest_open = len(cover.open)
            if not cover.open:
                continue

        cut = cover.cuts[rng.choice(cover.open)]
        incoming = [e for e in cut if tabu[e] <= moves]
        if not incoming:
            incoming = list(cut)
        if len(cover.chosen) < budget:
            e = max(incoming, key=lambda f: (cover.gain[f], -age[f],
                                             centrality[f] * (0.5 + rng.random())))
            cover.add(e)
            moves += 1
            age[e], tabu[e] = moves, moves + tenure
        else:
            outgoing = [e for e in cover.chosen if tabu[e] <= moves]
            if not outgoing:
                outgoing = list(cover.chosen)
            allowed = set(outgoing)
            default_out = min(outgoing, key=lambda f: (cover.loss[f], age[f], rng.random()))
            best_key, selected_in, selected_out = None, None, None
            rng.shuffle(incoming)
            for e in incoming:
                if time.perf_counter() >= deadline:
                    break
                savings = {}
                for cid in cover.incidence[e]:
                    if cover.count[cid] == 1:
                        f = cover.xor[cid] - 1
                        if f in allowed:
                            savings[f] = savings.get(f, 0) + cover.weight[cid]
                out = default_out
                loss = cover.loss[out] - savings.get(out, 0)
                for f, saved in savings.items():
                    value = cover.loss[f] - saved
                    if (value, age[f]) < (loss, age[out]):
                        out, loss = f, value
                key = (cover.gain[e] - loss, cover.gain[e], -age[e] - age[out],
                       centrality[e] * rng.random())
                if best_key is None or key > best_key:
                    best_key, selected_in, selected_out = key, e, out
            if selected_in is None:
                break
            if best_key[0] <= 0:
                cover.penalize()
            cover.add(selected_in)
            cover.drop(selected_out)
            moves += 1
            age[selected_in] = age[selected_out] = moves
            tabu[selected_in] = moves + tenure
            tabu[selected_out] = moves + max(1, tenure // 2)
        if len(cover.open) < smallest_open:
            smallest_open = len(cover.open)
            last_progress = time.perf_counter()
    return answer


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    n = len(instance.get('vertices', []))
    m = len(instance.get('edges', []))
    answer = {'interdiction_status': [0] * m,
              'clique_membership': [1] + [0] * (n - 1) if n else [],
              'max_clique_size': 1 if n else 0}
    try:
        return _solve(instance, time_limit_s, answer, started)
    except Exception:
        # Keep the most recently fully written solution, so a local search failure cannot break the return structure.
        return answer
