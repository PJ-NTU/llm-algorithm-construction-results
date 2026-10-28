"""Official bard2002 (VRPTW, fleet minimisation) <-> the current portfolio's format (input_format.json /
output_format.json).

A translation only: nothing is repaired or invented. Anything the target format cannot hold raises.
This adapter is for the portfolio updated by continued evolution.

Nodes. Official: depot + customers in list order = nodes 0..n (the official checker indexes by list position).
Ours: depot 0, customer arrays indexed k = 0..n-1 for node k+1, routes written [0, customers..., 0].

Travel times. The official checker uses tau_ij = max(distance_ij (+ service_i unless the instance sets
"travel_time_includes_service_time"), ready_j - due_i). Ours uses c_ij = max(base_ij + service_i, a_j - b_i) for a
customer i. With base = the official distance matrix and service = 0 when the flag is set (the service time is
declared to be inside the matrix already, so nothing is added on top), c_ij equals the official tau_ij on every
customer arc. The revision used here sets the flag in all instances, and its Gurobi references use these times.

Official solution: {"objective_value", "num_vehicles", "routes", "total_distance", "departure_times", "loads"}.
Departure times t and loads y are auxiliary variables of the official model; a routing is feasible iff SOME t, y
satisfy (4)-(7), and the componentwise-minimal ones dominate. As for the recorded algorithms,
they are derived from our routes: t as the earliest departures along each route under the official travel times,
y as the least assignment of the official load constraints (5)-(6), which hold for all ordered customer pairs; if
none exists the running loads are written and the official checker rejects them. Our own schedule labels
(route_departure_times, customer_departure_times, route_return_times, customer_loads) are therefore not copied.
objective_value and num_vehicles are the number of routes, total_distance the route length over the official
distance matrix.
"""
import math


def _distances(d, nodes):
    if d.get("distance_matrix"):
        return d["distance_matrix"]
    return [[0.0 if i == j else math.floor(math.hypot(a["x"] - b["x"], a["y"] - b["y"]) * 10) / 10.0
             for j, b in enumerate(nodes)] for i, a in enumerate(nodes)]


def to_ours(official_instance: dict) -> dict:
    d = official_instance
    depot, customers = d["depot"], list(d["customers"])
    if depot["demand"] != 0 or depot["service_time"] != 0:
        raise ValueError("depot demand and service time must be 0: our format has no field for them")
    fold = d.get("travel_time_includes_service_time", False)
    return {
        "customer_demands": [c["demand"] for c in customers],
        "vehicle_capacity": d["vehicle_capacity"],
        "customer_service_times": [0 if fold else c["service_time"] for c in customers],
        "customer_time_windows": [[c["ready_time"], c["due_date"]] for c in customers],
        "depot_time_window": [depot["ready_time"], depot["due_date"]],
        "depot_coordinates": [depot["x"], depot["y"]],
        "base_travel_times": [list(row) for row in _distances(d, [depot] + customers)],
    }


def _min_loads(routes, q, cap):
    """Least y with y_j >= q_j and, for all customer pairs, y_j >= y_i + q_j (arc i->j used) or
    y_j >= y_i + 2 q_j - cap (otherwise); None when it exceeds cap (no feasible y exists)."""
    cust = [c for r in routes for c in r]
    succ = {a: b for r in routes for a, b in zip(r, r[1:])}
    y = {i: float(q[i]) for i in cust}
    for _ in range(len(cust) + 2):
        changed = False
        for i in cust:
            yi = y[i]
            for j in cust:
                if i != j:
                    need = yi + (q[j] if succ.get(i) == j else 2 * q[j] - cap)
                    if need > y[j] + 1e-9:
                        if need > cap + 1e-6:
                            return None
                        y[j], changed = need, True
        if not changed:
            return y
    return None


def to_official(official_instance: dict, our_solution: dict) -> dict:
    d = official_instance
    nodes = [d["depot"]] + list(d["customers"])
    n = len(nodes) - 1
    q = [v["demand"] for v in nodes]
    ready = [v["ready_time"] for v in nodes]
    due = [v["due_date"] for v in nodes]
    service = [v["service_time"] for v in nodes]
    fold = d.get("travel_time_includes_service_time", False)
    dist = _distances(d, nodes)

    if our_solution.get("status") != "feasible":
        raise ValueError("solution status is %r: no plan to translate" % (our_solution.get("status"),))
    routes = []
    for r, route in enumerate(our_solution["routes"]):
        if (not isinstance(route, list) or len(route) < 3 or route[0] != 0 or route[-1] != 0
                or any(type(c) is not int or not 1 <= c <= n for c in route[1:-1])):
            raise ValueError("routes[%d] is not [0, customers 1..%d, 0]" % (r, n))
        routes.append(route[1:-1])

    departure, running = {}, {}
    for r in routes:
        now, load, prev = 0.0, 0, 0
        for c in r:
            tau = dist[prev][c] + (0 if fold else service[prev])
            tau = max(tau, ready[c] - due[prev])
            now = max(now + tau, ready[c])
            load += q[c]
            departure[c], running[c] = now, load
            prev = c
    loads = _min_loads(routes, q, d["vehicle_capacity"]) or running
    full = [[0] + r + [0] for r in routes]
    total = round(sum(dist[r[p]][r[p + 1]] for r in full for p in range(len(r) - 1)), 1)
    return {"objective_value": float(len(full)), "num_vehicles": len(full), "routes": full, "total_distance": total,
            "departure_times": {str(c): float(t) for c, t in sorted(departure.items())},
            "loads": {str(c): float(y) for c, y in sorted(loads.items())}}


def official_objective(official_instance: dict, official_solution: dict) -> float:
    return float(len({(r[0], r[1]) for r in official_solution["routes"] if len(r) > 2 and r[0] == 0}))
