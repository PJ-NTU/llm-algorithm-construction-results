"""Stochastic hub neighbourhood search with joint flow re-optimisation
Combines add/drop/swap, assignment descent and min-cut, jointly adjusting multiple hubs and scenario assignments in time-limited mixed-integer neighbourhoods.
"""

import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow, breadth_first_order
except Exception:
    csr_matrix = None
    maximum_flow = None
    breadth_first_order = None


try:
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
except Exception:
    milp = None


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
        margin = min(0.09 * budget, 0.004 + 0.018 * budget + 3e-6 * self.s * self.n)
        self.end = start + max(0.0, budget - margin)
        self.elites = []
        self.pool_size = max(2, int(math.sqrt(self.n) * math.log1p(max(0.0, budget))))
        self.evaluation_time = max(1e-5, 1e-7 * self.s * self.n * self.n)
        self.cut_ok = maximum_flow is not None and bool(np.all(self.b >= 0.0))
        self.last_improvement = start
        self.mip_enabled = milp is not None and bool(np.all(self.b >= 0.0))
        self.mip_calls = 0
        self.mip_last_build = 0.0
        self.mip_width = max(2, int(math.ceil(math.sqrt(math.log1p(budget)))))
        self.mip_rate = 3500.0
        self.last_mip = start

    def value(self, hubs, allocations):
        total = float(self.f[hubs].sum())
        for s in range(self.s):
            if time.perf_counter() >= self.end:
                return float('inf')
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
            score += np.einsum('ij,kj->ik', b, self.d[hubs[:, None], labels[None, :]], optimize=False)
            score += np.einsum('ji,jk->ik', b, self.d[labels[:, None], hubs[None, :]], optimize=False)
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

    def joint_reopt(self, hubs, allocations, until):
        """Exactly linearise a multi-label joint neighbourhood using continuous flows aggregated by origin."""
        begin = time.perf_counter()
        allowance = until - begin
        if allowance <= max(0.002, 2.0 * self.mip_last_build):
            return None
        self.mip_calls += 1
        opened = np.zeros(self.n, dtype=bool)
        opened[hubs] = True
        moves = self.rank_moves(hubs, allocations, min(until, begin + allowance * 0.18))
        if time.perf_counter() >= until or not moves:
            return None
        closed = np.flatnonzero(~opened)
        new_rank = []
        old_rank = []
        for _, kind, old, new in moves:
            if new >= 0 and not opened[new] and new not in new_rank:
                new_rank.append(new)
            if old >= 0 and old not in old_rank:
                old_rank.append(old)
        new_rank.extend(int(k) for k in closed if int(k) not in new_rank)
        old_rank.extend(int(k) for k in hubs if int(k) not in old_rank)
        stale = (begin - self.last_improvement) / max(self.budget, 1e-9)
        width = min(len(new_rank), max(2, int(self.mip_width * (1.0 + stale))))
        if self.mip_calls % 3 == 1:
            targets = new_rank[:width]
        else:
            prefix = min(len(new_rank), max(width, int((1.0 + math.log1p(self.budget)) * width)))
            targets = self.rng.sample(new_rank[:prefix], width) if width else []
        pool = np.unique(np.concatenate((hubs, np.asarray(targets, dtype=int))))
        expected_labels = min(len(pool), len(targets) + 2 + int(math.sqrt(math.log1p(self.budget))))
        scenes_used = max(1, int(np.count_nonzero(self.prob)))
        model_space = max(0.0, until - time.perf_counter()) * self.mip_rate
        active_cap = min(self.n, max(len(targets), int(model_space / max(
            1, scenes_used * expected_labels * len(pool)))))
        # Only allow closing old hubs whose entire customer cluster can enter the active region, keeping frozen assignments valid.
        removable = []
        mandatory = np.zeros((self.s, self.n), dtype=bool)
        mandatory[:, targets] = True
        removal_width = max(1, int(self.mip_width * (1.0 + stale)))
        if self.mip_calls % 2 == 0:
            self.rng.shuffle(old_rank)
        for h in old_rank:
            if time.perf_counter() >= until:
                return None
            trial = mandatory | (allocations == h)
            if len(removable) < removal_width and int(trial.sum(axis=1).max()) <= active_cap:
                removable.append(h)
                mandatory = trial
        if not targets and not removable:
            return None
        variable_hubs = np.asarray(sorted(set(targets + removable)), dtype=int)
        frozen = opened.copy()
        frozen[removable] = False
        hub_variables = {int(h): i for i, h in enumerate(variable_hubs)}
        obj = self.f[variable_hubs].tolist()
        integer = [1] * len(variable_hubs)
        lower = [0.0] * len(variable_hubs)
        upper = [1.0] * len(variable_hubs)
        rr, cc, vv, lhs, rhs = [], [], [], [], []
        row_index = 0

        def row(indices, values, lo, hi):
            nonlocal row_index
            rr.extend([row_index] * len(indices))
            cc.extend(indices)
            vv.extend(values)
            lhs.append(lo)
            rhs.append(hi)
            row_index += 1

        if not frozen.any():
            row(list(range(len(variable_hubs))), [1.0] * len(variable_hubs), 1.0, np.inf)
        scene_models = []
        incumbent_part = float(self.f[removable].sum())
        build_end = min(until, begin + allowance * 0.44)
        alternative_width = max(1, int(math.ceil(math.log1p(self.budget) / math.sqrt(max(1, len(pool))))))
        try:
            for s in np.argsort(-np.abs(self.prob)):
                s = int(s)
                if time.perf_counter() >= build_end:
                    return None
                if self.prob[s] == 0.0:
                    continue
                b = self.b[s]
                labels = allocations[s]
                score = self.c[s][:, pool].copy()
                score += np.einsum('ij,kj->ik', b, self.d[pool[:, None], labels[None, :]], optimize=False)
                score += np.einsum('ji,jk->ik', b, self.d[labels[:, None], pool[None, :]], optimize=False)
                pos = np.full(self.n, -1, dtype=int)
                pos[pool] = np.arange(len(pool))
                signed_score = score * self.prob[s]
                gains = signed_score - signed_score[self.idx, pos[labels], None]
                priority = np.min(gains, axis=1)
                if targets:
                    priority = np.min(gains[:, pos[targets]], axis=1)
                active = mandatory[s].copy()
                eligible = np.flatnonzero(~frozen & ~active)
                count = max(0, active_cap - int(active.sum()))
                if count:
                    ranked = eligible[np.argsort(priority[eligible])]
                    active[ranked[:count]] = True
                free = np.flatnonzero(active)
                fixed_nodes = np.flatnonzero(~active)
                if not len(free):
                    continue
                choices = []
                y_indices = []
                folded = self.c[s][np.ix_(free, pool)].copy()
                if len(fixed_nodes):
                    af = labels[fixed_nodes]
                    folded += np.einsum('ij,kj->ik', b[np.ix_(free, fixed_nodes)],
                                        self.d[pool[:, None], af[None, :]], optimize=False)
                    folded += np.einsum('ji,jk->ik', b[np.ix_(fixed_nodes, free)],
                                        self.d[af[:, None], pool[None, :]], optimize=False)
                folded *= self.prob[s]
                q = b[np.ix_(free, free)]
                incumbent_part += float(folded[np.arange(len(free)), pos[labels[free]]].sum())
                incumbent_part += float(self.prob[s] * np.sum(q * self.d[labels[free, None], labels[None, free]]))
                for fi, i in enumerate(free):
                    if time.perf_counter() >= build_end:
                        return None
                    best_labels = pool[np.argsort(signed_score[i])[:alternative_width]]
                    hs = np.asarray(sorted(set([int(labels[i])] + targets + best_labels.tolist())), dtype=int)
                    if i in hub_variables and i not in hs:
                        hs = np.sort(np.append(hs, i))
                    y = np.arange(len(obj), len(obj) + len(hs), dtype=int)
                    obj.extend(folded[fi, pos[hs]].tolist())
                    integer.extend([1] * len(hs))
                    lower.extend([0.0] * len(hs))
                    upper.extend([1.0] * len(hs))
                    row(y.tolist(), [1.0] * len(hs), 1.0, 1.0)
                    for yi, h in zip(y, hs):
                        if int(h) in hub_variables:
                            zi = hub_variables[int(h)]
                            if i == h:
                                # The diagonal label is an alias of z; the other labels correspond to x in the problem.
                                row([int(yi), zi], [1.0, -1.0], 0.0, 0.0)
                            else:
                                row([int(yi), zi], [1.0, -1.0], -np.inf, 0.0)
                    choices.append(hs)
                    y_indices.append(y)
                destinations = np.unique(np.concatenate(choices))
                y_by_hub = {}
                for h in destinations:
                    entries = [(j, int(y[np.flatnonzero(hs == h)[0]]))
                               for j, (hs, y) in enumerate(zip(choices, y_indices)) if h in hs]
                    y_by_hub[int(h)] = entries
                for i, (hs, y) in enumerate(zip(choices, y_indices)):
                    if time.perf_counter() >= build_end:
                        return None
                    mass = float(q[i].sum())
                    if mass <= 0.0:
                        continue
                    weights = q[i] / mass
                    relevant = []
                    col_coeff = []
                    for h in destinations:
                        entries = [(j, yi) for j, yi in y_by_hub[int(h)] if weights[j] > 0.0]
                        if entries:
                            relevant.append(int(h))
                            col_coeff.append(entries)
                    m = len(relevant)
                    if not m:
                        continue
                    flow_ids = np.arange(len(obj), len(obj) + len(hs) * m, dtype=int).reshape(len(hs), m)
                    # Row marginals are origin labels; column marginals are demand fractions flowing to each destination label.
                    # With integer y only one row can be non-zero, so the transfer cost equals the original quadratic term exactly.
                    costs = self.prob[s] * mass * self.d[np.ix_(hs, relevant)]
                    obj.extend(costs.ravel().tolist())
                    integer.extend([0] * costs.size)
                    lower.extend([0.0] * costs.size)
                    upper.extend([1.0] * costs.size)
                    for k in range(len(hs)):
                        row(flow_ids[k].tolist() + [int(y[k])], [1.0] * m + [-1.0], 0.0, 0.0)
                    for l, entries in enumerate(col_coeff):
                        jjs = [j for j, _ in entries]
                        ys = [yi for _, yi in entries]
                        row(flow_ids[:, l].tolist() + ys,
                            [1.0] * len(hs) + (-weights[jjs]).tolist(), 0.0, 0.0)
                scene_models.append((s, free, choices, y_indices))
            if not scene_models:
                return None
            build_elapsed = time.perf_counter() - begin
            self.mip_last_build = build_elapsed
            self.mip_rate = 0.75 * self.mip_rate + 0.25 * min(
                self.mip_rate * 1.5, len(obj) / max(1e-6, build_elapsed) * 0.16)
            if time.perf_counter() >= until:
                return None
            scale = max(1.0, abs(self.best) / max(1, self.n))
            objective = np.asarray(obj, dtype=float) / scale
            nz = np.flatnonzero(objective)
            row(nz.tolist(), objective[nz].tolist(), -np.inf,
                (incumbent_part + 1e-8 * max(1.0, abs(incumbent_part))) / scale)
            matrix = coo_matrix((np.asarray(vv, dtype=float),
                                 (np.asarray(rr, dtype=np.int32), np.asarray(cc, dtype=np.int32))),
                                shape=(row_index, len(obj))).tocsc()
            remaining = until - time.perf_counter()
            reserve = max(build_elapsed * 0.25, (self.end - until) * 0.02)
            seconds = min(remaining - reserve, (self.end - time.perf_counter()) * 0.6)
            if seconds <= max(0.001, build_elapsed * 0.05):
                return None
            result = milp(objective, integrality=np.asarray(integer, dtype=np.uint8),
                          bounds=Bounds(np.asarray(lower), np.asarray(upper)),
                          constraints=LinearConstraint(matrix, np.asarray(lhs), np.asarray(rhs)),
                          options={'time_limit': float(seconds), 'mip_rel_gap': 0.0, 'disp': False})
            if result.x is None or not np.isfinite(result.x).all():
                return None
            z = opened.copy()
            z[variable_hubs] = result.x[:len(variable_hubs)] > 0.5
            hh = np.flatnonzero(z)
            if not len(hh):
                return None
            aa = allocations.copy()
            for s, free, choices, y_indices in scene_models:
                for i, hs, y in zip(free, choices, y_indices):
                    selected = int(np.argmax(result.x[y]))
                    if result.x[y[selected]] < 1.0 - 1e-5:
                        return None
                    aa[s, i] = hs[selected]
            for s in np.flatnonzero(self.prob == 0.0):
                aa[s, ~z[aa[s]]] = hh[0]
                aa[s, hh] = hh
            if not np.all(z[aa]) or not np.all(aa[:, hh] == hh):
                return None
            val = self.value(hh, aa)
            if not math.isfinite(val):
                return None
            self.save(hh, aa, val)
            if time.perf_counter() < until:
                polished = self.descend(hh, aa, until)
                newval = self.value(hh, polished)
                if newval < val:
                    aa, val = polished, newval
            self.last_mip = time.perf_counter()
            return hh, aa, val
        except Exception:
            # Solver or numerical problems abandon only this neighbourhood; the main heuristic and existing solution remain valid.
            self.mip_enabled = False
            return None

    def run(self):
        if self.s == 0:
            hubs = np.flatnonzero(self.f < 0.0)
            if not len(hubs):
                hubs = np.array([int(np.argmin(self.f))])
            self.save(hubs, np.empty((0, self.n), dtype=int), float(self.f[hubs].sum()))
            return
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
            # After stagnation, first try joint multi-hub, multi-label re-optimisation rather than just increasing random perturbation.
            if self.mip_enabled and self.elites:
                entry = self.elites[0] if self.mip_calls % 2 == 0 else (val, tuple(hubs.tolist()), a)
                mh = np.asarray(entry[1], dtype=int)
                ma = entry[2]
                quantum = (self.end - now) / (2.0 + math.sqrt(math.log1p(self.budget)))
                mip_end = min(self.end, now + quantum)
                candidate = self.joint_reopt(mh, ma, mip_end)
                if candidate is not None:
                    nh, na, nv = candidate
                    self.save(nh, na, nv)
                    self.add_elite(nh, na, nv)
                    if nv < val - 1e-10 * max(1.0, abs(val)):
                        hubs, a, val = nh, na, nv
                        tried_cuts = False
                        now = time.perf_counter()
                        basin_end = min(self.end, now + (self.end - now) / (3.0 + math.log1p(self.n)))
                        continue
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
        if not math.isfinite(budget) or budget <= 0.0 or np is None:
            return fallback
        search = _Search(instance, start, budget, fallback)
        search.run()
    except Exception:
        # When time is short or the numerical library is unavailable, the saved feasible solution is not lost.
        pass
    return search.answer if search is not None else fallback
