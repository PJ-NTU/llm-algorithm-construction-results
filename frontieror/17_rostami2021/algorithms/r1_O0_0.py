"""Adaptive neighbourhood search for stochastic hub location
Combines hub add/drop/swap, incremental assignment descent, min-cut batch reassignment and elite perturbation, keeping the best feasible solution throughout the wall-clock budget.
"""

import math
import random
import time

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow, breadth_first_order
except Exception:
    csr_matrix = None
    maximum_flow = None
    breadth_first_order = None


class _Search:
    def __init__(self, instance, start, budget, fallback):
        self.start = start
        self.budget = budget
        self.n = int(instance['n'])
        self.s = int(instance['num_scenarios'])
        self.answer = fallback
        self.best = float('inf')
        self.rng = random.Random(731993)
        self.f = np.asarray(instance['fixed_costs'], dtype=float)
        self.prob = np.asarray(instance['scenarios_probability'], dtype=float)
        self.d = np.asarray(instance['distances'], dtype=float)
        self.w = np.asarray(instance['scenarios_demands'], dtype=float)
        self.idx = np.arange(self.n)
        self.diag = np.diag(self.d).copy()
        alpha = float(instance['cost_parameters_alpha'])
        chi = float(instance['cost_parameters_chi'])
        delta = float(instance['cost_parameters_delta'])
        # The input structure provides 2-D distances; here they are used as dist, with
        # c[s,i,k] = dist[i,k] * (chi * outflow[s,i] + delta * inflow[s,i]).
        # pi_factors, p_hubs and hub_capacities do not constitute additional constraints specified by the problem.
        outgoing = self.w.sum(axis=2)
        incoming = self.w.sum(axis=1)
        self.c = (chi * outgoing + delta * incoming)[:, :, None] * self.d[None, :, :]
        self.c[:, self.idx, self.idx] = 0.0
        self.b = alpha * self.w.copy()
        diagonal_flow = self.b[:, self.idx, self.idx].copy()
        self.b[:, self.idx, self.idx] = 0.0
        # Transfer cost of self-demand is folded into the unary term, keeping it correct when the distance diagonal is non-zero.
        self.c += diagonal_flow[:, :, None] * self.diag[None, None, :]
        margin = min(0.08 * budget, 0.003 + 0.003 * budget + 2e-6 * self.s * self.n)
        self.end = start + max(0.0, budget - margin)
        self.elites = []
        self.pool_size = max(2, int(math.sqrt(self.n) * math.log1p(max(0.0, budget))))
        self.evaluation_time = max(1e-5, 1e-7 * self.s * self.n * self.n)
        self.cut_ok = maximum_flow is not None and bool(np.all(self.b >= 0.0))
        self.last_improvement = start

    def value(self, hubs, allocations):
        total = float(self.f[hubs].sum())
        for s in range(self.s):
            a = allocations[s]
            total += float(self.prob[s]) * float(
                self.c[s, self.idx, a].sum()
                + np.sum(self.b[s] * self.d[a[:, None], a[None, :]])
            )
        return total

    def save(self, hubs, allocations, val):
        if math.isfinite(val) and val < self.best:
            self.best = float(val)
            z = [0] * self.n
            for h in hubs:
                z[int(h)] = 1
            self.answer = {
                'hub_selection': z,
                'scenario_allocations': {str(s): allocations[s].tolist() for s in range(self.s)},
            }
            self.last_improvement = time.perf_counter()

    def add_elite(self, hubs, allocations, val):
        key = tuple(int(h) for h in hubs)
        for i, entry in enumerate(self.elites):
            if entry[1] == key:
                if val < entry[0]:
                    self.elites[i] = (float(val), key, allocations.copy())
                break
        else:
            self.elites.append((float(val), key, allocations.copy()))
        self.elites.sort(key=lambda entry: entry[0])
        if len(self.elites) > self.pool_size:
            self.elites.pop()

    def unary_seed(self, hubs):
        a = np.argmin(self.c[:, :, hubs], axis=2)
        a = hubs[a]
        a[:, hubs] = hubs
        return a

    def descend(self, hubs, allocations, until):
        """Fix the hubs, maintain all single-node move costs, and apply the best improvement each time."""
        a = allocations.copy()
        if len(hubs) <= 1:
            a[:] = hubs[0]
            return a
        nonhubs = np.ones(self.n, dtype=bool)
        nonhubs[hubs] = False
        inverse = np.full(self.n, -1, dtype=int)
        inverse[hubs] = np.arange(len(hubs))
        # Handle scenarios with a larger share of expected cost first; zero-probability scenarios only need a valid assignment.
        order = np.argsort(-np.abs(self.prob))
        for s in order:
            if time.perf_counter() >= until:
                break
            if self.prob[s] == 0.0:
                continue
            labels = a[s]
            b = self.b[s]
            score = self.c[s, :, hubs].T.copy()
            score += b @ self.d[hubs[:, None], labels[None, :]].T
            score += b.T @ self.d[labels[:, None], hubs[None, :]]
            sign = 1.0 if self.prob[s] >= 0.0 else -1.0
            tolerance = 1e-10 * max(1.0, float(np.max(np.abs(score))))
            while time.perf_counter() < until:
                best_labels = np.argmin(sign * score, axis=1)
                gains = sign * (score[self.idx, best_labels] - score[self.idx, inverse[labels]])
                gains[~nonhubs] = np.inf
                i = int(np.argmin(gains))
                if gains[i] >= -tolerance:
                    break
                old = int(labels[i])
                new = int(hubs[best_labels[i]])
                score += b[:, i, None] * (self.d[hubs, new] - self.d[hubs, old])[None, :]
                score += b[i, :, None] * (self.d[new, hubs] - self.d[old, hubs])[None, :]
                labels[i] = new
        return a

    def expand(self, s, labels, hubs, target, until):
        """Binary expansion min-cut for one target hub; not used when the submodularity condition fails."""
        if not self.cut_ok or self.prob[s] <= 0.0 or time.perf_counter() >= until:
            return labels
        movable = np.ones(self.n, dtype=bool)
        movable[hubs] = False
        movable[labels == target] = False
        free = np.flatnonzero(movable)
        m = len(free)
        if m == 0:
            return labels
        fixed = np.flatnonzero(~movable)
        old = labels[free]
        b = self.b[s]
        d = self.d
        u0 = self.c[s, free, old].copy()
        u1 = self.c[s, free, target].copy()
        if len(fixed):
            af = labels[fixed]
            out = b[np.ix_(free, fixed)]
            inc = b[np.ix_(fixed, free)].T
            u0 += np.sum(out * d[old[:, None], af[None, :]], axis=1)
            u0 += np.sum(inc * d[af[None, :], old[:, None]], axis=1)
            u1 += out @ d[target, af] + inc @ d[af, target]
        ii, jj = np.triu_indices(m, 1)
        i, j = free[ii], free[jj]
        ai, aj = old[ii], old[jj]
        forward, backward = b[i, j], b[j, i]
        e00 = forward * d[ai, aj] + backward * d[aj, ai]
        e01 = forward * d[ai, target] + backward * d[target, ai]
        e10 = forward * d[target, aj] + backward * d[aj, target]
        e11 = (forward + backward) * d[target, target]
        edge = 0.5 * (e01 + e10 - e00 - e11)
        scale0 = max(1.0, float(np.max(np.abs(e00))) if len(e00) else 1.0)
        if len(edge) and float(np.min(edge)) < -1e-9 * scale0:
            return labels
        edge = np.maximum(edge, 0.0)
        diff = u1 - u0
        diff += np.bincount(ii, weights=e10 - e00 - edge, minlength=m)
        diff += np.bincount(jj, weights=e01 - e00 - edge, minlength=m)
        source_caps = np.maximum(diff, 0.0)
        sink_caps = np.maximum(-diff, 0.0)
        mass = float(2.0 * edge.sum() + source_caps.sum() + sink_caps.sum())
        if not math.isfinite(mass) or mass <= 0.0 or time.perf_counter() >= until:
            return labels
        # Integer capacities are used only to generate candidates; acceptance is always judged by the original floating-point objective.
        scale = (2.0 ** 28) / mass
        vertices = np.arange(m)
        rows = np.concatenate((ii, jj, np.full(m, m), vertices))
        cols = np.concatenate((jj, ii, vertices, np.full(m, m + 1)))
        caps = np.rint(np.concatenate((edge, edge, source_caps, sink_caps)) * scale).astype(np.int64)
        positive = caps > 0
        graph = csr_matrix((caps[positive], (rows[positive], cols[positive])), shape=(m + 2, m + 2))
        try:
            flow = maximum_flow(graph, m, m + 1)
            residual = graph - flow.flow
            residual.data = (residual.data > 0).astype(np.int64)
            residual.eliminate_zeros()
            reached = breadth_first_order(residual, m, directed=True, return_predecessors=False)
        except Exception:
            self.cut_ok = False
            return labels
        on_source = np.zeros(m + 2, dtype=bool)
        on_source[reached] = True
        changed = free[~on_source[:m]]
        if not len(changed):
            return labels
        candidate = labels.copy()
        candidate[changed] = target
        previous = float(self.c[s, self.idx, labels].sum() + np.sum(b * d[labels[:, None], labels[None, :]]))
        updated = float(self.c[s, self.idx, candidate].sum() + np.sum(b * d[candidate[:, None], candidate[None, :]]))
        if updated < previous - 1e-10 * max(1.0, abs(previous)):
            return candidate
        return labels

    def cut_descent(self, hubs, allocations, until):
        a = allocations.copy()
        if not self.cut_ok or len(hubs) < 2:
            return a
        improved = True
        while improved and time.perf_counter() < until:
            improved = False
            for s in np.argsort(-self.prob):
                targets = hubs.tolist()
                self.rng.shuffle(targets)
                for h in targets:
                    if time.perf_counter() >= until:
                        return a
                    candidate = self.expand(int(s), a[s], hubs, int(h), until)
                    if not np.array_equal(candidate, a[s]):
                        a[s] = candidate
                        improved = True
        return a

    def rank_moves(self, hubs, allocations, until):
        """Evaluate whole-cluster relocation exactly, and use single-node costs to predict gains from opening and split-closing."""
        p = len(hubs)
        opened = np.zeros(self.n, dtype=bool)
        opened[hubs] = True
        inverse = np.full(self.n, -1, dtype=int)
        inverse[hubs] = np.arange(p)
        add_score = self.f.copy()
        relocation = -self.f[hubs, None] + np.where(opened, 0.0, self.f)[None, :]
        close_score = -self.f[hubs].copy()
        for s in range(self.s):
            if time.perf_counter() >= until:
                return []
            if self.prob[s] == 0.0:
                continue
            a = allocations[s]
            pos = inverse[a]
            b = self.b[s]
            outgoing = np.zeros((p, self.n))
            incoming = np.zeros((p, self.n))
            np.add.at(outgoing, pos, b.T)
            np.add.at(incoming, pos, b)
            out, inc = outgoing.T, incoming.T
            scores = self.c[s] + out @ self.d[:, hubs].T + inc @ self.d[hubs, :]
            gain = scores - scores[self.idx, a, None]
            opening = np.minimum(gain, 0.0)
            opening[hubs] = 0.0
            add_score += self.prob[s] * (opening.sum(axis=0) + np.maximum(gain[self.idx, self.idx], 0.0))
            grouped = np.zeros((p, self.n))
            np.add.at(grouped, pos, scores)
            internal = np.bincount(pos, weights=out[self.idx, pos], minlength=p)
            move = grouped - grouped[np.arange(p), hubs, None]
            correction = self.diag[None, :] - self.d[:, hubs].T - self.d[hubs, :] + self.diag[hubs, None]
            move += internal[:, None] * correction
            extra = gain[self.idx, self.idx][None, :].copy()
            extra = extra + out.T * (
                (self.diag - self.d[a, self.idx])[None, :]
                - self.d[:, hubs].T + self.d[a[:, None], hubs[None, :]].T
            )
            extra += inc.T * (
                (self.diag - self.d[self.idx, a])[None, :]
                - self.d[hubs, :] + self.d[hubs[:, None], a[None, :]]
            )
            extra[pos, self.idx] = 0.0
            relocation += self.prob[s] * (move + extra)
            if p > 1:
                available = scores[:, hubs].copy()
                available[self.idx, pos] = np.inf
                split = np.min(available, axis=1) - scores[self.idx, a]
                close_score += self.prob[s] * np.bincount(pos, weights=split, minlength=p)
        moves = []
        for k in np.flatnonzero(~opened):
            moves.append((float(add_score[k]), 0, -1, int(k)))
        for q, h in enumerate(hubs):
            for k in range(self.n):
                if k != h:
                    moves.append((float(relocation[q, k]), 1, int(h), k))
            if p > 1:
                moves.append((float(close_score[q]), 2, int(h), -1))
        moves.sort(key=lambda item: item[0])
        return moves

    def propose(self, hubs, allocations, move, until, use_cut):
        _, kind, old, new = move
        opened = np.zeros(self.n, dtype=bool)
        opened[hubs] = True
        a = allocations.copy()
        if kind == 0:
            opened[new] = True
            for s in range(self.s):
                labels = a[s]
                b = self.b[s]
                # First force the new hub to point to itself, then build the expansion start from exact single-node deltas.
                labels[new] = new
                if time.perf_counter() >= until:
                    continue
                current = self.c[s, self.idx, labels].copy()
                current += np.sum(b * self.d[labels[:, None], labels[None, :]], axis=1)
                current += np.sum(b.T * self.d[labels[None, :], labels[:, None]], axis=1)
                target = self.c[s, :, new] + b @ self.d[new, labels] + b.T @ self.d[labels, new]
                change = target < current
                change[opened] = False
                labels[change] = new
        elif kind == 1:
            opened[old] = False
            opened[new] = True
            a[a == old] = new
            a[:, new] = new
        else:
            opened[old] = False
            remaining = np.flatnonzero(opened)
            for s in range(self.s):
                removed = a[s] == old
                if not np.any(removed):
                    continue
                labels = a[s]
                if time.perf_counter() >= until:
                    labels[removed] = remaining[0]
                    continue
                b = self.b[s]
                scores = self.c[s, :, remaining].T.copy()
                scores += b @ self.d[remaining[:, None], labels[None, :]].T
                scores += b.T @ self.d[labels[:, None], remaining[None, :]]
                labels[removed] = remaining[np.argmin(scores[removed], axis=1)]
        h = np.flatnonzero(opened)
        a[:, h] = h
        # A new-label min-cut can move a whole batch of nodes at once, not limited to single-node descent.
        if use_cut and kind != 2 and self.cut_ok:
            for s in np.argsort(-self.prob):
                if time.perf_counter() >= until:
                    break
                a[s] = self.expand(int(s), a[s], h, new, until)
        a = self.descend(h, a, until)
        return h, a

    def perturb(self, hubs, allocations, remaining):
        opened = np.zeros(self.n, dtype=bool)
        opened[hubs] = True
        a = allocations.copy()
        stale = (time.perf_counter() - self.last_improvement) / max(self.budget, 1e-9)
        strength = max(1, int(math.sqrt(math.log1p(max(self.budget, 0.0))) * (1.0 + 3.0 * stale)))
        strength = min(strength, self.n)
        # Perturbation strength varies with the total time limit and stagnation time; the main search has no iteration cap.
        for _ in range(strength):
            if time.perf_counter() >= self.end:
                break
            hs = np.flatnonzero(opened)
            closed = np.flatnonzero(~opened)
            r = self.rng.random()
            if len(closed) and (r < 0.35 or len(hs) == 1):
                k = int(closed[self.rng.randrange(len(closed))])
                opened[k] = True
                a[:, k] = k
            elif len(hs) > 1 and (r < 0.65 or not len(closed)):
                h = int(hs[self.rng.randrange(len(hs))])
                opened[h] = False
                other = np.flatnonzero(opened)
                for s in range(self.s):
                    selected = a[s] == h
                    a[s, selected] = other[np.argmin(self.c[s, selected][:, other], axis=1)]
            elif len(closed):
                h = int(hs[self.rng.randrange(len(hs))])
                k = int(closed[self.rng.randrange(len(closed))])
                opened[h] = False
                opened[k] = True
                a[a == h] = k
                a[:, k] = k
        hs = np.flatnonzero(opened)
        a[:, hs] = hs
        if len(hs) > 1 and self.rng.random() < 0.5:
            for s in range(self.s):
                if time.perf_counter() >= self.end:
                    break
                count = min(self.n, max(1, int(strength * math.sqrt(self.n))))
                nodes = self.rng.sample(range(self.n), count)
                for i in nodes:
                    if not opened[i]:
                        a[s, i] = int(hs[self.rng.randrange(len(hs))])
        return hs, a

    def run(self):
        # Any single-hub plan is necessarily feasible; first compare all single-hub plans and the all-open plan.
        single = self.f.copy()
        for s in range(self.s):
            single += self.prob[s] * (self.c[s].sum(axis=0) + self.b[s].sum() * self.diag)
        h0 = int(np.argmin(single))
        hubs = np.array([h0], dtype=int)
        a = np.full((self.s, self.n), h0, dtype=int)
        val = float(single[h0])
        self.save(hubs, a, val)
        self.add_elite(hubs, a, val)
        if time.perf_counter() >= self.end:
            return
        all_hubs = self.idx.copy()
        all_a = np.tile(self.idx, (self.s, 1))
        all_value = self.value(all_hubs, all_a)
        self.save(all_hubs, all_a, all_value)
        self.add_elite(all_hubs, all_a, all_value)
        if all_value < val:
            hubs, a, val = all_hubs, all_a, all_value
        started = time.perf_counter()
        basin_end = min(self.end, started + max(self.evaluation_time, 0.20 * (self.end - started)))
        tried_cuts = False
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            remaining = self.end - now
            local_until = min(self.end, now + max(2.0 * self.evaluation_time, remaining / (3.0 + math.sqrt(self.n))))
            improved_a = self.descend(hubs, a, local_until)
            improved_val = self.value(hubs, improved_a)
            if improved_val < val:
                a, val = improved_a, improved_val
                self.save(hubs, a, val)
            if time.perf_counter() >= self.end:
                break
            moves = self.rank_moves(hubs, a, self.end)
            accepted = False
            for move in moves:
                now = time.perf_counter()
                if now >= self.end or now >= basin_end:
                    break
                remaining = self.end - now
                allowance = max(2.0 * self.evaluation_time, remaining / (4.0 + math.sqrt(self.n * max(1, self.s))))
                until = min(self.end, basin_end, now + allowance)
                hh, aa = self.propose(hubs, a, move, until, use_cut=tried_cuts)
                vv = self.value(hh, aa)
                elapsed = time.perf_counter() - now
                self.evaluation_time = 0.8 * self.evaluation_time + 0.2 * max(1e-6, elapsed)
                self.save(hh, aa, vv)
                if vv < val - 1e-10 * max(1.0, abs(val)):
                    hubs, a, val = hh, aa, vv
                    self.add_elite(hubs, a, val)
                    accepted = True
                    tried_cuts = False
                    break
            if accepted:
                # Extend the current descent phase when there is real progress, still bounded by the total wall clock.
                now = time.perf_counter()
                basin_end = min(self.end, max(basin_end, now + (self.end - now) / (3.0 + math.log1p(self.n))))
                continue
            now = time.perf_counter()
            if now >= self.end:
                break
            if not tried_cuts and self.cut_ok:
                until = min(self.end, now + (self.end - now) / (3.0 + math.sqrt(max(1, self.s))))
                aa = self.cut_descent(hubs, a, until)
                vv = self.value(hubs, aa)
                self.save(hubs, aa, vv)
                if vv < val - 1e-10 * max(1.0, abs(val)):
                    a, val = aa, vv
                    self.add_elite(hubs, a, val)
                    tried_cuts = False
                else:
                    tried_cuts = True
                now = time.perf_counter()
                basin_end = min(self.end, now + (self.end - now) / (3.0 + math.log1p(self.n)))
                continue
            self.add_elite(hubs, a, val)
            now = time.perf_counter()
            if now >= self.end:
                break
            # The elite pool keeps distinct hub sets; a squared bias makes good solutions more frequent restart points.
            choice = min(len(self.elites) - 1, int(self.rng.random() ** 2 * len(self.elites)))
            entry = self.elites[choice]
            hubs = np.asarray(entry[1], dtype=int)
            a = entry[2].copy()
            hubs, a = self.perturb(hubs, a, self.end - now)
            val = self.value(hubs, a)
            self.save(hubs, a, val)
            tried_cuts = False
            now = time.perf_counter()
            basin_end = min(self.end, now + (self.end - now) / (3.0 + math.log1p(self.n)))


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    # For valid input as specified by the problem, build a complete feasible fallback solution that does not depend on any search.
    try:
        n = max(1, int(instance.get('n', 1)))
        scenarios = max(0, int(instance.get('num_scenarios', 0)))
    except Exception:
        n, scenarios = 1, 0
    fallback = {
        'hub_selection': [1] + [0] * (n - 1),
        'scenario_allocations': {str(s): [0] * n for s in range(scenarios)},
    }
    search = None
    try:
        budget = max(0.0, float(time_limit_s))
        if budget <= 0.0:
            return fallback
        search = _Search(instance, start, budget, fallback)
        search.run()
    except Exception:
        # When time is short or the numerical library is unavailable, the saved feasible solution is not lost.
        pass
    return search.answer if search is not None else fallback
