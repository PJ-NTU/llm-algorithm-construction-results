"""Decomposed pricing with column generation
Allocates offer-set durations by linear programming and generates new columns with add-drop-swap moves, multi-start search and branch-and-bound with segment-wise upper bounds, keeping the corrected feasible solution.
"""

import heapq
import math
import time

try:
    import numpy as np
    from scipy.optimize import linprog
except Exception:
    np = None
    linprog = None


def _ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


class _Pricing:
    """Pricing branch-and-bound over one connected component of interacting customer segments."""

    def __init__(self, indices, w, outside, rates, profit, seed):
        order = np.argsort(-profit)
        self.indices = indices[order]
        self.w = w[:, order]
        self.outside = outside
        self.rates = rates
        self.profit = profit[order]
        self.k = len(order)
        self.tol = 1e-10 * max(1.0, float(rates.sum() * profit.max()))
        self.heap = []
        self.serial = 0
        self.best = 0.0
        self.mask = 0
        self.offer(seed[order])
        self.offer(np.ones(self.k, dtype=bool))
        self._node(np.arange(self.k), np.zeros(len(rates)), outside.copy(), 0)

    def vector(self, mask=None):
        if mask is None:
            mask = self.mask
        return np.fromiter(((mask >> j) & 1 for j in range(self.k)),
                           dtype=bool, count=self.k)

    def value(self, x):
        den = self.outside + self.w @ x
        num = self.w @ (self.profit * x)
        return float(self.rates @ _ratio(num, den))

    def offer(self, x):
        value = self.value(x)
        if value > self.best:
            self.best = value
            mask = 0
            for j in np.flatnonzero(x):
                mask |= 1 << int(j)
            self.mask = mask

    def _offer_mask(self, mask):
        self.offer(self.vector(mask))

    def _node(self, free, numerator, denominator, mask):
        if not len(free):
            self._offer_mask(mask)
            return
        w = self.w[:, free]
        p = self.profit[free]
        den = np.column_stack((denominator,
                               denominator[:, None] + np.cumsum(w, axis=1)))
        num = np.column_stack((numerator,
                               numerator[:, None] + np.cumsum(w * p, axis=1)))
        values = _ratio(num, den)
        segment_bound = values.max(axis=1)
        upper = float(self.rates @ segment_bound)
        if upper <= self.best + self.tol:
            return

        # With the selected products fixed, the optimal free products for a single segment are still a prefix by adjusted fare.
        prefix = int(np.argmax(self.rates @ values))
        candidate = mask
        for j in free[:prefix]:
            candidate |= 1 << int(j)
        self._offer_mask(candidate)

        # Opposite preferences of customer segments for the same product are used to choose the branching variable.
        scale = np.maximum(den[:, -1], 1e-300)
        regret = (self.rates / scale)[:, None] * w * (p - segment_bound[:, None])
        positive = np.maximum(regret, 0.0).sum(axis=0)
        negative = np.maximum(-regret, 0.0).sum(axis=0)
        candidate = mask
        for j in free[positive > negative]:
            candidate |= 1 << int(j)
        self._offer_mask(candidate)
        if upper <= self.best + self.tol:
            return
        conflict = np.minimum(positive, negative)
        branch = int(np.argmax(conflict))
        if float(conflict[branch]) <= self.tol:
            # Keep branching under floating-point approximation; heuristic agreement must not pass as a certificate.
            branch = int(np.argmax(positive + negative))
        self.serial += 1
        heapq.heappush(self.heap, (-upper, self.serial, free, numerator,
                                   denominator, mask, branch))

    def gap(self):
        while self.heap and -self.heap[0][0] <= self.best + self.tol:
            heapq.heappop(self.heap)
        return max(0.0, -self.heap[0][0] - self.best) if self.heap else 0.0

    def expand(self):
        if not self.gap():
            return
        _, _, free, numerator, denominator, mask, branch = heapq.heappop(self.heap)
        j = int(free[branch])
        rest = np.delete(free, branch)
        weight = self.w[:, j]
        # Check the time again only after both subproblems are created, so no unexplored branch is lost.
        self._node(rest, numerator + weight * self.profit[j],
                   denominator + weight, mask | (1 << j))
        self._node(rest, numerator, denominator, mask)


class _Solver:
    def __init__(self, data, budget, started, result):
        self.result = result
        self.budget = budget
        self.deadline = started + budget - min(0.2, max(0.002, budget * 0.035))
        self.best_value = 0.0
        self.T = float(data["booking_horizon_T"])
        self.capacity = np.asarray(data["network_legs_capacity"], dtype=float)
        self.m = int(data["network_num_legs"])
        ids = list(data["products_product_id"])
        n = len(ids)
        fare = np.asarray(data["products_fare"], dtype=float)
        incidence = np.zeros((n, self.m), dtype=float)
        for j, legs in enumerate(data["products_legs_used"]):
            for leg in legs:
                incidence[j, int(leg)] = 1.0

        segment_rates = np.asarray(data["segments_lambda_l"], dtype=float)
        outside = np.asarray(data["segments_no_purchase_preference"], dtype=float)
        w = np.zeros((len(segment_rates), n), dtype=float)
        positions = {int(product): j for j, product in enumerate(ids)}
        for l, considered in enumerate(data["segments_consideration_set"]):
            weights = data["segments_preference_vector"][l]
            for product, weight in zip(considered, weights):
                j = positions.get(int(product))
                if j is not None:
                    w[l, j] += float(weight)

        # Do not sell products that use zero-capacity legs; products without positive-arrival-rate demand need not be offered either.
        allowed = ((w * segment_rates[:, None]).sum(axis=0) > 0)
        if self.m:
            allowed &= ~np.any(incidence[:, self.capacity <= 0] > 0, axis=1)
        self.ids = [int(ids[j]) for j in np.flatnonzero(allowed)]
        self.fare = fare[allowed]
        self.A = incidence[allowed]
        w = w[:, allowed]
        rows = (segment_rates > 0) & np.any(w > 0, axis=1)
        w, outside, segment_rates = w[rows], outside[rows], segment_rates[rows]
        normalizer = outside + w.sum(axis=1)
        normalizer = np.where(normalizer > 0, normalizer, 1.0)
        self.w = w / normalizer[:, None]
        self.outside = outside / normalizer
        # lambda * p_l = lambda_l; do not multiply again by the global arrival rate or alpha.
        self.rates = segment_rates
        self.n = len(self.ids)
        self.capacity_rate = self.capacity / self.T
        self.active_legs = np.flatnonzero((self.capacity > 0) & np.any(self.A > 0, axis=0))
        self.columns = []
        self.loads = []
        self.revenues = []
        self.seen = set()
        self.rng = np.random.default_rng(78371)
        self.last_offer = np.ones(self.n, dtype=bool)
        self.support = []
        self.lp_cost = 0.0

        # Only products considered by the same customer segment interact in the pricing problem.
        parent = list(range(self.n))

        def root(j):
            while parent[j] != j:
                parent[j] = parent[parent[j]]
                j = parent[j]
            return j

        for row in self.w:
            products = np.flatnonzero(row > 0)
            if len(products):
                first = root(int(products[0]))
                for j in products[1:]:
                    parent[root(int(j))] = first
        groups = {}
        for j in range(self.n):
            groups.setdefault(root(j), []).append(j)
        self.components = []
        for products in groups.values():
            products = np.asarray(products, dtype=int)
            rows = np.flatnonzero(np.any(self.w[:, products] > 0, axis=1))
            self.components.append((products, rows))

    def _demand(self, x):
        denominator = self.outside + self.w @ x
        factor = _ratio(self.rates, denominator)
        return x * (factor @ self.w)

    def _accept(self, indices, fractions):
        fractions = np.maximum(np.asarray(fractions, dtype=float), 0.0)
        good = np.isfinite(fractions) & (fractions > 1e-13)
        if not np.any(good):
            return
        indices = np.asarray(indices, dtype=int)[good]
        fractions = fractions[good]
        loads = np.asarray([self.loads[j] for j in indices])
        use = fractions @ loads
        factor = min(1.0, 1.0 / float(fractions.sum()))
        positive = use > 0
        if np.any(positive):
            factor = min(factor, float(np.min(self.capacity_rate[positive] / use[positive])))
        if not math.isfinite(factor) or factor <= 0:
            return
        durations = fractions * (factor * (1.0 - 2e-10) * self.T)
        revenue = float(np.dot(durations, [self.revenues[j] for j in indices]))
        if revenue <= self.best_value:
            return
        allocations = []
        for j, duration in zip(indices, durations):
            allocations.append({"offer_set": [self.ids[k] for k in self.columns[j]],
                                "time_allocated": float(duration)})
        self.best_value = revenue
        self.result["offer_set_allocations"] = allocations

    def _add(self, x):
        x = np.asarray(x, dtype=bool)
        key = np.packbits(x).tobytes()
        if key in self.seen:
            return False
        self.seen.add(key)
        demand = self._demand(x)
        revenue = float(demand @ self.fare)
        if not math.isfinite(revenue) or revenue <= 0:
            return False
        load = demand @ self.A
        j = len(self.columns)
        self.columns.append(np.flatnonzero(x).tolist())
        self.loads.append(load)
        self.revenues.append(revenue)
        # A feasible non-empty candidate solution always exists, even before the first LP completes.
        self._accept([j], [1.0])
        return True

    def _master(self):
        remaining = self.deadline - time.perf_counter()
        if remaining <= 0 or not self.columns:
            return None
        loads = np.asarray(self.loads)
        revenues = np.asarray(self.revenues)
        scale = max(float(revenues.max()), 1e-12)
        rows = self.active_legs
        matrix = np.vstack(((loads[:, rows] / self.capacity_rate[rows]).T,
                            np.ones(len(self.columns))))
        allowance = min(remaining * 0.85,
                        max(self.budget * 0.12, self.lp_cost * 3.0, 0.001))
        before = time.perf_counter()
        solved = linprog(-revenues / scale, A_ub=matrix,
                         b_ub=np.ones(len(rows) + 1), bounds=(0.0, None),
                         method="highs-ds", options={"time_limit": allowance,
                         "primal_feasibility_tolerance": 1e-8,
                         "dual_feasibility_tolerance": 1e-8})
        self.lp_cost = time.perf_counter() - before
        if solved.x is not None:
            self._accept(np.arange(len(self.columns)), solved.x)
            self.support = np.flatnonzero(solved.x > 1e-10).tolist()
        if not solved.success:
            return None
        dual = -np.asarray(solved.ineqlin.marginals) * scale
        prices = np.zeros(self.m)
        prices[rows] = np.maximum(dual[:-1], 0.0) / self.capacity_rate[rows]
        return prices, max(0.0, float(dual[-1]))

    def _local(self, start, profit, end):
        x = np.asarray(start, dtype=bool).copy()
        usable = profit > 0
        x &= usable
        tol = 1e-10 * max(1.0, float(self.rates.sum() * max(0.0, profit.max())))
        width = min(self.n, max(2, int(math.sqrt(self.n * max(self.budget, 0.001)))))
        width = min(width, max(2, int(math.sqrt(250000 / max(1, len(self.rates))))))
        while time.perf_counter() < end:
            den = self.outside + self.w @ x
            num = self.w @ (profit * x)
            average = _ratio(num, den)
            value = float(self.rates @ average)
            sign = 1.0 - 2.0 * x
            new_den = den[:, None] + self.w * sign
            difference = self.w * sign * (profit - average[:, None])
            delta = _ratio(difference, new_den)
            delta = np.where(new_den > 0, delta, -average[:, None])
            gain = self.rates @ delta
            gain[~usable] = -np.inf
            j = int(np.argmax(gain))
            if gain[j] > tol:
                proposed = x.copy()
                proposed[j] = ~proposed[j]
                actual = float(self._demand(proposed) @ profit)
                if actual > value + tol:
                    x = proposed
                    continue
            if time.perf_counter() >= end:
                break
            selected = np.flatnonzero(x)
            absent = np.flatnonzero(usable & ~x)
            if not len(selected) or not len(absent):
                break
            selected = selected[np.argsort(-gain[selected])[:width]]
            absent = absent[np.argsort(-gain[absent])[:width]]
            dw = self.w[:, absent][:, None, :] - self.w[:, selected][:, :, None]
            dn = ((self.w[:, absent] * profit[absent])[:, None, :]
                  - (self.w[:, selected] * profit[selected])[:, :, None])
            values = _ratio(num[:, None, None] + dn, den[:, None, None] + dw)
            swap_values = np.einsum("l,lij->ij", self.rates, values)
            ij = np.unravel_index(int(np.argmax(swap_values)), swap_values.shape)
            if swap_values[ij] <= value + tol:
                break
            proposed = x.copy()
            proposed[selected[ij[0]]] = False
            proposed[absent[ij[1]]] = True
            if float(self._demand(proposed) @ profit) <= value + tol:
                break
            x = proposed
        return x

    def _pricing_states(self, profit):
        states = []
        for indices, rows in self.components:
            if time.perf_counter() >= self.deadline:
                return None
            indices = indices[profit[indices] > 0]
            if not len(indices):
                continue
            w = self.w[np.ix_(rows, indices)]
            relevant = np.any(w > 0, axis=1)
            rows, w = rows[relevant], w[relevant]
            states.append(_Pricing(indices, w, self.outside[rows], self.rates[rows],
                                   profit[indices], self.last_offer[indices]))
        return states

    def _combined(self, states):
        x = np.zeros(self.n, dtype=bool)
        for state in states:
            x[state.indices] = state.vector()
        return x

    def _record_pricing(self, x, states, profit, time_price, tolerance):
        for state in states:
            state.offer(x[state.indices])
        self.last_offer = self._combined(states)
        value = float(self._demand(self.last_offer) @ profit)
        if value > time_price + tolerance:
            return self._add(self.last_offer)
        return False

    def run(self):
        if not self.n or not len(self.rates):
            return self.result
        self._add(np.ones(self.n, dtype=bool))
        init_end = min(self.deadline, time.perf_counter() + self.budget * 0.055)
        prefix = np.zeros(self.n, dtype=bool)
        for j in np.argsort(-self.fare):
            if time.perf_counter() >= init_end:
                break
            single = np.zeros(self.n, dtype=bool)
            single[j] = True
            self._add(single)
            prefix[j] = True
            self._add(prefix)
        for row in self.w:
            if time.perf_counter() >= init_end:
                break
            self._add(row > 0)

        # No cap on column-generation rounds, restarts or branch nodes.
        while time.perf_counter() < self.deadline:
            dual = self._master()
            if dual is None:
                continue
            prices, time_price = dual
            profit = self.fare - self.A @ prices
            tolerance = 1e-8 * max(1.0, time_price, max(self.revenues))
            states = self._pricing_states(profit)
            if states is None:
                break
            if not states:
                break
            x = self._combined(states)
            added = self._record_pricing(x, states, profit, time_price, tolerance)
            remaining = self.deadline - time.perf_counter()
            local_end = min(self.deadline, time.perf_counter() +
                            min(remaining * 0.15, self.budget * 0.015))
            x = self._local(x, profit, local_end)
            added |= self._record_pricing(x, states, profit, time_price, tolerance)
            if added:
                continue
            if not any(state.gap() for state in states):
                break

            # When stuck in a local optimum, apply multi-start perturbation according to the remaining wall-clock budget.
            remaining = self.deadline - time.perf_counter()
            restart_end = min(self.deadline, time.perf_counter() +
                              min(remaining * 0.2, self.budget * 0.04))
            restart = 0
            usable = profit > 0
            while time.perf_counter() < restart_end:
                if restart == 0:
                    seed = np.zeros(self.n, dtype=bool)
                elif restart <= len(self.support):
                    seed = np.zeros(self.n, dtype=bool)
                    seed[self.columns[self.support[restart - 1]]] = True
                elif self.rng.random() < 0.5:
                    seed = self.last_offer.copy()
                    strength = min(self.n, max(1, int(math.sqrt(self.n * self.budget))))
                    count = int(self.rng.integers(1, strength + 1))
                    flipped = self.rng.choice(self.n, size=count, replace=False)
                    seed[flipped] = ~seed[flipped]
                else:
                    seed = self.rng.random(self.n) < self.rng.random()
                seed &= usable
                x = self._local(seed, profit, restart_end)
                restart += 1
                if self._record_pricing(x, states, profit, time_price, tolerance):
                    added = True
                    break
            if added:
                continue

            # The sum of the independent components' upper bounds is a valid upper bound for the full pricing problem.
            certified = False
            while time.perf_counter() < self.deadline:
                gaps = [state.gap() for state in states]
                lower = sum(state.best for state in states)
                if lower + sum(gaps) <= time_price + tolerance:
                    certified = True
                    break
                k = int(np.argmax(gaps))
                if gaps[k] <= 0:
                    certified = True
                    break
                before = states[k].best
                states[k].expand()
                if states[k].best > before + states[k].tol:
                    x = self._combined(states)
                    if self._record_pricing(x, states, profit, time_price, tolerance):
                        added = True
                        break
                    polish_end = min(self.deadline, time.perf_counter() + self.budget * 0.005)
                    x = self._local(x, profit, polish_end)
                    if self._record_pricing(x, states, profit, time_price, tolerance):
                        added = True
                        break
            if certified:
                break
        return self.result


def solve(instance: dict, time_limit_s: float) -> dict:
    result = {"offer_set_allocations": []}
    started = time.perf_counter()
    try:
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget <= 0 or np is None or linprog is None:
            return result
        if float(instance["booking_horizon_T"]) <= 0 or float(instance["lambda"]) <= 0:
            return result
        solver = _Solver(instance, budget, started, result)
        if time.perf_counter() < solver.deadline:
            solver.run()
    except Exception:
        # Every saved candidate passes feasibility correction; a failure does not overwrite the current solution.
        pass
    return result
