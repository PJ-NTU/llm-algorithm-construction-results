def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, (list, tuple)) else []


def _at(value, index, default=None):
    seq = _as_list(value)
    return seq[index] if 0 <= index < len(seq) else default


def _maybe_number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _number(value, default=0.0):
    result = _maybe_number(value)
    return float(default) if result is None else result


def _positive_int(value, default=1):
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        result = int(default)
    return result if result >= 1 else max(1, int(default))


def _dimensions(official_instance):
    data = _as_dict(official_instance)
    demands = _as_list(data.get("demand_scenarios"))
    mean_demands = _as_list(data.get("d_bar"))
    costs = _as_list(data.get("transportation_costs"))
    capacities = _as_list(data.get("L"))

    first_scenario = _at(demands, 0, [])
    first_customer = _at(first_scenario, 0, [])

    inferred_n = len(_as_list(first_scenario))
    if inferred_n == 0:
        inferred_n = len(mean_demands)
    if inferred_n == 0 and costs:
        inferred_n = max(0, len(costs) - 1)
    if inferred_n == 0 and capacities:
        inferred_n = max(0, len(capacities) - 1)

    inferred_t = len(_as_list(first_customer))
    if inferred_t == 0:
        inferred_t = len(_as_list(_at(mean_demands, 0, [])))

    inferred_s = len(demands)
    if inferred_s == 0:
        inferred_s = len(_as_list(data.get("scenario_probabilities")))

    n = _positive_int(data.get("n"), inferred_n or 1)
    periods = _positive_int(data.get("T"), inferred_t or 1)
    vehicles = _positive_int(data.get("m"), 1)
    scenarios = _positive_int(data.get("n_scenarios"), inferred_s or 1)
    return n, periods, vehicles, scenarios


def _probabilities(data, scenarios):
    raw = _as_list(_as_dict(data).get("scenario_probabilities"))
    if len(raw) >= scenarios:
        values = [_number(raw[s], 0.0) for s in range(scenarios)]
        if all(value > 0.0 for value in values):
            return values

    default = 1.0 / scenarios
    values = []
    for s in range(scenarios):
        value = _number(_at(raw, s, default), default)
        values.append(value if value > 0.0 else default)
    total = sum(values)
    if total <= 0.0:
        return [default for _ in range(scenarios)]
    return [value / total for value in values]


def _demand_value(data, scenario, customer, period):
    data = _as_dict(data)
    demands = _as_list(data.get("demand_scenarios"))
    value = _at(
        _at(_at(demands, scenario, []), customer, []),
        period,
        None,
    )
    if value is None:
        value = _at(
            _at(_as_list(data.get("d_bar")), customer, []),
            period,
            0.0,
        )
    return max(0.0, _number(value, 0.0))


def _vector(data, key, length, default=0.0):
    raw = _as_list(_as_dict(data).get(key))
    return [_number(_at(raw, i, default), default) for i in range(length)]


def _transportation_cost(data, i, j):
    matrix = _as_list(_as_dict(data).get("transportation_costs"))
    if i == j:
        return 0.0
    value = _at(_at(matrix, i, []), j, None)
    if value is None:
        value = _at(_at(matrix, j, []), i, 0.0)
    return max(0.0, _number(value, 0.0))


def _route_at(routes, period, vehicle):
    raw = _at(_at(routes, period, []), vehicle, None)
    if not isinstance(raw, (list, tuple)):
        return [0]

    nodes = []
    for value in raw:
        try:
            nodes.append(int(value))
        except (TypeError, ValueError, OverflowError):
            continue
    return nodes if nodes else [0]


def to_ours(official_instance: dict) -> dict:
    data = _as_dict(official_instance)
    n, periods, vehicles, scenarios = _dimensions(data)

    demand_scenarios = []
    for s in range(scenarios):
        scenario_rows = []
        for t in range(periods):
            scenario_rows.append([
                _demand_value(data, s, c, t) for c in range(n)
            ])
        demand_scenarios.append(scenario_rows)

    transportation_costs = []
    for i in range(n + 1):
        transportation_costs.append([
            _transportation_cost(data, i, j) for j in range(n + 1)
        ])

    return {
        "num_periods": periods,
        "num_customers": n,
        "num_vehicles": vehicles,
        "num_scenarios": scenarios,
        "scenario_probabilities": _probabilities(data, scenarios),
        "demand_scenarios": demand_scenarios,
        "initial_inventories": _vector(data, "I0", n + 1, 0.0),
        "inventory_capacities": _vector(data, "L", n + 1, 0.0),
        "production_capacity": max(0.0, _number(data.get("C"), 0.0)),
        "vehicle_capacity": max(0.0, _number(data.get("Q"), 0.0)),
        "setup_cost": max(0.0, _number(data.get("f"), 0.0)),
        "unit_production_cost": max(0.0, _number(data.get("u"), 0.0)),
        "holding_costs": _vector(data, "h", n + 1, 0.0),
        "unmet_demand_penalties": _vector(data, "sigma", n, 0.0),
        "transportation_costs": transportation_costs,
    }


def _objective_from_our_solution(official_instance, our_solution):
    data = _as_dict(official_instance)
    solution = _as_dict(our_solution)
    n, periods, vehicles, scenarios = _dimensions(data)

    setup_raw = _as_list(solution.get("production_setup"))
    setups = [_number(_at(setup_raw, t, 0.0), 0.0) for t in range(periods)]
    setup_cost = _number(data.get("f"), 0.0) * sum(setups)

    routes = _as_list(solution.get("vehicle_routes"))
    transportation_cost = 0.0
    for t in range(periods):
        for k in range(vehicles):
            route = _route_at(routes, t, k)
            for position in range(len(route) - 1):
                i = route[position]
                j = route[position + 1]
                if 0 <= i <= n and 0 <= j <= n and i != j:
                    transportation_cost += _transportation_cost(data, i, j)

    production = _as_list(solution.get("production_quantities"))
    inventories = _as_list(solution.get("end_inventories"))
    unmet = _as_list(solution.get("unmet_demand_quantities"))
    probabilities = _probabilities(data, scenarios)
    production_unit_cost = _number(data.get("u"), 0.0)
    holding_costs = _vector(data, "h", n + 1, 0.0)
    penalties = _vector(data, "sigma", n, 0.0)

    expected_recourse_cost = 0.0
    for s in range(scenarios):
        scenario_cost = 0.0
        for t in range(periods):
            produced = _at(_at(production, s, []), t, 0.0)
            scenario_cost += production_unit_cost * _number(produced, 0.0)

            inventory_row = _at(_at(inventories, s, []), t, [])
            for i in range(n + 1):
                scenario_cost += holding_costs[i] * _number(
                    _at(inventory_row, i, 0.0), 0.0
                )

            unmet_row = _at(_at(unmet, s, []), t, [])
            for c in range(n):
                scenario_cost += penalties[c] * _number(
                    _at(unmet_row, c, 0.0), 0.0
                )

        expected_recourse_cost += probabilities[s] * scenario_cost

    return float(setup_cost + transportation_cost + expected_recourse_cost)


def to_official(official_instance: dict, our_solution: dict) -> dict:
    data = _as_dict(official_instance)
    solution = _as_dict(our_solution)
    n, periods, vehicles, _ = _dimensions(data)

    setup_raw = _as_list(solution.get("production_setup"))
    y = {
        str(t + 1): _number(_at(setup_raw, t, 0.0), 0.0)
        for t in range(periods)
    }

    z = {}
    x = {}
    routes = _as_list(solution.get("vehicle_routes"))

    for t in range(periods):
        official_t = t + 1
        for k in range(vehicles):
            official_k = k + 1
            route = _route_at(routes, t, k)
            customers = [node for node in route if 1 <= node <= n]

            if customers:
                z[f"0_{official_k}_{official_t}"] = 1
            for customer in customers:
                z[f"{customer}_{official_k}_{official_t}"] = 1

            for position in range(len(route) - 1):
                a = route[position]
                b = route[position + 1]
                if not (0 <= a <= n and 0 <= b <= n) or a == b:
                    continue
                i, j = (a, b) if a < b else (b, a)
                key = f"{i}_{j}_{official_k}_{official_t}"
                x[key] = x.get(key, 0) + 1

    objective_value = _objective_from_our_solution(data, solution)
    status = int(_number(solution.get("status"), 2.0))
    gap = _number(solution.get("gap"), 0.0)

    return {
        "objective_value": objective_value,
        "gap": gap,
        "status": status,
        "y": y,
        "z": z,
        "x": x,
    }


def official_objective(official_instance: dict, official_solution: dict) -> float:
    data = _as_dict(official_instance)
    solution = _as_dict(official_solution)

    # The official serialization omits p, q, I, and e. Consequently its
    # objective_value is the only serialized value containing the complete
    # scenario-weighted recourse cost and is authoritative when present.
    reported = _maybe_number(solution.get("objective_value"))
    if reported is not None:
        return float(reported)

    # Safe fallback when a partial official solution omitted objective_value:
    # return the exactly recoverable first-stage setup and route costs.
    n, periods, vehicles, _ = _dimensions(data)
    y = _as_dict(solution.get("y"))
    setup_count = 0.0
    for t in range(1, periods + 1):
        setup_count += _number(y.get(str(t), y.get(t, 0.0)), 0.0)
    objective = _number(data.get("f"), 0.0) * setup_count

    x = _as_dict(solution.get("x"))
    for key, raw_value in x.items():
        if isinstance(key, str):
            parts = key.split("_")
        elif isinstance(key, (list, tuple)):
            parts = list(key)
        else:
            continue
        if len(parts) != 4:
            continue
        try:
            i, j, k, t = (int(part) for part in parts)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= i < j <= n and 1 <= k <= vehicles and 1 <= t <= periods:
            objective += (
                _transportation_cost(data, i, j)
                * _number(raw_value, 0.0)
            )

    return float(objective)


# ─────────────────────────────────────────────────────────────────────────
# _FCO6_CANON (added 2026-08-26)
# Official constraint 18 (Eq. 19) is **vehicle symmetry breaking**: in every period the visit set of vehicle k,
# read as a binary code, must be >= that of vehicle k+1, with code sum(2^(j-i) * z[i,k,t]); **smaller i is the more significant bit**.
# The fleet is homogeneous, so swapping vehicle indices is physically equivalent and leaves the objective unchanged; this is not an algorithm issue
# but a matter of **how the solution is written**. Previously all 25/25 cells were rejected by the official checker; after relabelling, 5/5 pass.
# Only wraps to_official; none of its internal logic is changed.
import collections as _collections


def _fco6_canon_vehicles(sol):
    z = sol.get("z")
    if not isinstance(z, dict):
        return sol
    visits = _collections.defaultdict(set)
    ks, ts, maxi = set(), set(), 0
    for key, v in z.items():
        parts = str(key).split("_")
        if len(parts) != 3:
            continue
        try:
            i, k, t = (int(x) for x in parts)
        except ValueError:
            continue
        ks.add(k); ts.add(t); maxi = max(maxi, i)
        if float(v) >= 0.5 and i >= 1:
            visits[(k, t)].add(i)
    if len(ks) < 2:
        return sol
    kk = sorted(ks)
    remap = {}
    for t in sorted(ts):
        order = sorted(kk, key=lambda k: -sum(2 ** (maxi - i) for i in visits.get((k, t), ())))
        for new_k, old_k in zip(kk, order):
            remap[(old_k, t)] = new_k
    out = dict(sol)
    for field, kpos, arity in (("z", 1, 3), ("x", 2, 4)):
        d = sol.get(field)
        if not isinstance(d, dict):
            continue
        new = {}
        for key, v in d.items():
            parts = str(key).split("_")
            if len(parts) != arity:
                new[key] = v
                continue
            try:
                nums = [int(x) for x in parts]
            except ValueError:
                new[key] = v
                continue
            if (nums[kpos], nums[-1]) in remap:
                nums[kpos] = remap[(nums[kpos], nums[-1])]
            new["_".join(str(n) for n in nums)] = v
        out[field] = new
    return out


_fco6_orig_to_official = to_official


def to_official(official_instance: dict, our_solution: dict) -> dict:
    return _fco6_canon_vehicles(_fco6_orig_to_official(official_instance, our_solution))
