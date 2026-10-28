"""Feasibility checker for the two-echelon CVRP with grouping constraints.

Only the five fields declared as required in MAT_s4.md are required here.
The checker reconstructs every derived quantity instead of trusting optional
solution summaries such as an objective value.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict


_REL = 1e-9
_REQUIRED_FIELDS = (
    "first_echelon_routes",
    "first_echelon_deliveries",
    "second_echelon_routes",
    "group_assignments",
    "customer_satellite_assignments",
)


def _leq(a: float, b: float) -> bool:
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _geq(a: float, b: float) -> bool:
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _is_int(value: object) -> bool:
    return type(value) is int


def _is_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _read_instance(instance: object, errors: list[str]) -> dict | None:
    """Read only fields needed to evaluate the authoritative constraints."""
    if not isinstance(instance, dict):
        errors.append("FORMAT | instance must be a dict")
        return None

    integer_fields = (
        "Q1",
        "Q2",
        "T1",
        "T2",
        "depot_id",
        "num_customers",
        "num_groups",
        "num_satellites",
    )
    list_fields = (
        "customers_demand",
        "customers_id",
        "groups_customer_ids",
        "groups_group_id",
        "satellites_capacity",
        "satellites_id",
        "satellites_max_vehicles",
    )

    data: dict = {}
    for name in integer_fields:
        if name not in instance:
            errors.append(f"FORMAT | instance is missing {name!r}")
        elif not _is_int(instance[name]):
            errors.append(f"FORMAT | instance field {name!r} must be an int")
        else:
            data[name] = instance[name]

    for name in list_fields:
        if name not in instance:
            errors.append(f"FORMAT | instance is missing {name!r}")
        elif not isinstance(instance[name], list):
            errors.append(f"FORMAT | instance field {name!r} must be a list")
        else:
            data[name] = instance[name]

    if errors:
        return None

    scalar_lists = (
        "customers_demand",
        "customers_id",
        "groups_group_id",
        "satellites_capacity",
        "satellites_id",
        "satellites_max_vehicles",
    )
    for name in scalar_lists:
        for position, value in enumerate(data[name]):
            if not _is_int(value):
                errors.append(
                    f"FORMAT | instance field {name!r}[{position}] must be an int"
                )

    for group_position, members in enumerate(data["groups_customer_ids"]):
        if not isinstance(members, list):
            errors.append(
                "FORMAT | instance field 'groups_customer_ids'"
                f"[{group_position}] must be a list"
            )
            continue
        for member_position, customer_id in enumerate(members):
            if not _is_int(customer_id):
                errors.append(
                    "FORMAT | instance field 'groups_customer_ids'"
                    f"[{group_position}][{member_position}] must be an int"
                )

    expected_lengths = {
        "customers_demand": data["num_customers"],
        "customers_id": data["num_customers"],
        "groups_customer_ids": data["num_groups"],
        "groups_group_id": data["num_groups"],
        "satellites_capacity": data["num_satellites"],
        "satellites_id": data["num_satellites"],
        "satellites_max_vehicles": data["num_satellites"],
    }
    for name, expected in expected_lengths.items():
        if len(data[name]) != expected:
            errors.append(
                f"FORMAT | instance field {name!r} has length {len(data[name])}, "
                f"but its declared axis length is {expected}"
            )

    for name in ("customers_id", "groups_group_id", "satellites_id"):
        if len(set(data[name])) != len(data[name]):
            errors.append(
                f"FORMAT | instance field {name!r} contains duplicate IDs, "
                "so assignments cannot be resolved uniquely"
            )

    customer_ids = set(data["customers_id"])
    for group_position, members in enumerate(data["groups_customer_ids"]):
        if isinstance(members, list):
            unknown = [member for member in members if member not in customer_ids]
            if unknown:
                errors.append(
                    "FORMAT | instance group at position "
                    f"{group_position} contains unknown customer IDs {unknown!r}"
                )

    if errors:
        return None

    data["customer_demand_by_id"] = dict(
        zip(data["customers_id"], data["customers_demand"])
    )
    data["satellite_capacity_by_id"] = dict(
        zip(data["satellites_id"], data["satellites_capacity"])
    )
    data["satellite_vehicle_limit_by_id"] = dict(
        zip(data["satellites_id"], data["satellites_max_vehicles"])
    )
    return data


def _parse_assignment_map(
    raw: object,
    label: str,
    expected_ids: set[int],
    satellite_ids: set[int],
    errors: list[str],
) -> dict[int, int] | None:
    """Parse case/space/underscore/leading-zero variants without ambiguity."""
    if not isinstance(raw, dict):
        errors.append(f"FORMAT | {label}_assignments must be a dict")
        return None

    parsed: dict[int, int] = {}
    for raw_key, satellite_id in raw.items():
        if not isinstance(raw_key, str):
            errors.append(
                f"FORMAT | every key in {label}_assignments must be a string"
            )
            continue

        normalized = re.sub(r"[\s_]+", "", raw_key.casefold())
        match = re.fullmatch(rf"{re.escape(label)}([+-]?\d+)", normalized)
        if match is None:
            errors.append(
                f"FORMAT | key {raw_key!r} cannot be uniquely parsed as "
                f"a {label} ID"
            )
            continue

        item_id = int(match.group(1))
        if item_id not in expected_ids:
            errors.append(
                f"FORMAT | key {raw_key!r} resolves to unknown {label} ID {item_id}"
            )
            continue
        if item_id in parsed:
            errors.append(
                f"FORMAT | multiple keys resolve to {label} ID {item_id}"
            )
            continue
        if not _is_int(satellite_id):
            errors.append(
                f"FORMAT | assignment for {label} ID {item_id} must be an int"
            )
            continue
        if satellite_id not in satellite_ids:
            errors.append(
                f"FORMAT | assignment for {label} ID {item_id} names unknown "
                f"satellite ID {satellite_id}"
            )
            continue
        parsed[item_id] = satellite_id

    missing = sorted(expected_ids - parsed.keys())
    if missing:
        errors.append(
            f"FORMAT | {label}_assignments is missing IDs {missing!r}"
        )
    return parsed


def _has_satellite_cycle(
    satellite_ids: set[int], edges: list[tuple[int, int]]
) -> bool:
    adjacency: dict[int, set[int]] = {satellite_id: set() for satellite_id in satellite_ids}
    for start, end in edges:
        if start in satellite_ids and end in satellite_ids:
            adjacency[start].add(end)

    state = {satellite_id: 0 for satellite_id in satellite_ids}

    def visit(node: int) -> bool:
        state[node] = 1
        for successor in adjacency[node]:
            if state[successor] == 1:
                return True
            if state[successor] == 0 and visit(successor):
                return True
        state[node] = 2
        return False

    return any(state[node] == 0 and visit(node) for node in satellite_ids)


def _route_flow_profiles(
    second_routes: list[list[int]], demand_by_customer: dict[int, int]
) -> list[dict]:
    """Eliminate C10's f variables into one residual-load variable per route."""
    profiles: list[dict] = []
    for route_index, route in enumerate(second_routes):
        customers = route[1:-1]
        if not customers:
            continue

        total_demand = sum(demand_by_customer[customer] for customer in customers)
        offsets = [total_demand]
        remaining = total_demand
        for customer in customers:
            remaining -= demand_by_customer[customer]
            offsets.append(remaining)

        profiles.append(
            {
                "route_index": route_index,
                "satellite": route[0],
                "route": route,
                "customers": customers,
                # If t is the return-arc flow, flow on edge e is t + offsets[e].
                "offsets": offsets,
            }
        )
    return profiles


def _check_solution(instance: object, solution: object) -> list[str]:
    format_errors: list[str] = []
    data = _read_instance(instance, format_errors)

    if not isinstance(solution, dict):
        format_errors.append("FORMAT | solution must be a dict")
        return format_errors

    missing_fields = [field for field in _REQUIRED_FIELDS if field not in solution]
    if missing_fields:
        format_errors.append(
            f"FORMAT | solution is missing required fields {missing_fields!r}"
        )
    if data is None or missing_fields:
        return format_errors

    satellite_ids = set(data["satellites_id"])
    customer_ids = set(data["customers_id"])
    first_node_ids = satellite_ids | {data["depot_id"]}

    first_routes = solution["first_echelon_routes"]
    if not isinstance(first_routes, list):
        format_errors.append("FORMAT | first_echelon_routes must be a list")
    elif len(first_routes) != data["T1"]:
        format_errors.append(
            "FORMAT | first_echelon_routes must contain exactly T1 routes: "
            f"got {len(first_routes)}, expected {data['T1']}"
        )
    else:
        for vehicle_id, route in enumerate(first_routes):
            if not isinstance(route, list) or len(route) < 2:
                format_errors.append(
                    f"FORMAT | first-echelon route {vehicle_id} must be a list "
                    "with at least two node IDs"
                )
                continue
            for position, node_id in enumerate(route):
                if not _is_int(node_id):
                    format_errors.append(
                        f"FORMAT | first-echelon route {vehicle_id} node "
                        f"{position} must be an int"
                    )
                elif node_id not in first_node_ids:
                    format_errors.append(
                        f"FORMAT | first-echelon route {vehicle_id} contains "
                        f"unknown first-echelon node ID {node_id}"
                    )

    deliveries_raw = solution["first_echelon_deliveries"]
    deliveries: dict[tuple[int, int], float] = {}
    if not isinstance(deliveries_raw, list):
        format_errors.append("FORMAT | first_echelon_deliveries must be a list")
    else:
        for position, record in enumerate(deliveries_raw):
            if not isinstance(record, dict):
                format_errors.append(
                    f"FORMAT | delivery record {position} must be a dict"
                )
                continue
            missing = [
                field
                for field in ("satellite_id", "vehicle_id", "quantity")
                if field not in record
            ]
            if missing:
                format_errors.append(
                    f"FORMAT | delivery record {position} is missing {missing!r}"
                )
                continue
            satellite_id = record["satellite_id"]
            vehicle_id = record["vehicle_id"]
            quantity = record["quantity"]
            if not _is_int(satellite_id) or satellite_id not in satellite_ids:
                format_errors.append(
                    f"FORMAT | delivery record {position} has an unknown or "
                    "non-integer satellite_id"
                )
                continue
            if (
                not _is_int(vehicle_id)
                or vehicle_id < 0
                or vehicle_id >= data["T1"]
            ):
                format_errors.append(
                    f"FORMAT | delivery record {position} has vehicle_id outside "
                    "the declared 0-based T1 vehicle IDs"
                )
                continue
            if not _is_number(quantity):
                format_errors.append(
                    f"FORMAT | delivery record {position} quantity must be a "
                    "finite number"
                )
                continue
            pair = (satellite_id, vehicle_id)
            if pair in deliveries:
                format_errors.append(
                    f"FORMAT | duplicate delivery record for satellite "
                    f"{satellite_id}, vehicle {vehicle_id}"
                )
                continue
            deliveries[pair] = quantity

        expected_pairs = {
            (satellite_id, vehicle_id)
            for satellite_id in satellite_ids
            for vehicle_id in range(data["T1"])
        }
        missing_pairs = sorted(expected_pairs - deliveries.keys())
        if missing_pairs:
            format_errors.append(
                "FORMAT | first_echelon_deliveries is missing satellite/vehicle "
                f"pairs {missing_pairs!r}"
            )

    second_routes = solution["second_echelon_routes"]
    if not isinstance(second_routes, list):
        format_errors.append("FORMAT | second_echelon_routes must be a list")
    else:
        for route_index, route in enumerate(second_routes):
            if not isinstance(route, list) or len(route) < 2:
                format_errors.append(
                    f"FORMAT | second-echelon route {route_index} must be a list "
                    "with at least two node IDs"
                )
                continue
            for position, node_id in enumerate(route):
                if not _is_int(node_id):
                    format_errors.append(
                        f"FORMAT | second-echelon route {route_index} node "
                        f"{position} must be an int"
                    )
            if not all(_is_int(node_id) for node_id in route):
                continue
            if route[0] not in satellite_ids or route[-1] not in satellite_ids:
                format_errors.append(
                    f"FORMAT | second-echelon route {route_index} must have "
                    "satellite IDs at both endpoints"
                )
            for node_id in route[1:-1]:
                if node_id not in customer_ids:
                    format_errors.append(
                        f"FORMAT | second-echelon route {route_index} contains "
                        f"non-customer internal node ID {node_id}"
                    )

    group_assignments = _parse_assignment_map(
        solution["group_assignments"],
        "group",
        set(data["groups_group_id"]),
        satellite_ids,
        format_errors,
    )
    customer_assignments = _parse_assignment_map(
        solution["customer_satellite_assignments"],
        "customer",
        customer_ids,
        satellite_ids,
        format_errors,
    )

    if format_errors:
        return format_errors

    violations: list[str] = []

    # The authoritative variable domains say w[s][k] >= 0.  w is continuous,
    # so a relative comparison avoids rejecting solver-scale roundoff at zero.
    for (satellite_id, vehicle_id), quantity in deliveries.items():
        if not _geq(quantity, 0):
            violations.append(
                "DOMAIN(w) | delivery quantity must be nonnegative: satellite "
                f"{satellite_id}, vehicle {vehicle_id}, quantity {quantity!r}"
            )

    first_edges_by_vehicle: list[list[tuple[int, int]]] = []
    first_out_counts: list[Counter] = []
    for vehicle_id, route in enumerate(first_routes):
        edges = [] if route == [data["depot_id"], data["depot_id"]] else list(
            zip(route, route[1:])
        )
        first_edges_by_vehicle.append(edges)
        first_out_counts.append(Counter(start for start, _ in edges))
        if len(edges) != len(set(edges)):
            violations.append(
                "DOMAIN(x) | a binary first-echelon arc is repeated by vehicle "
                f"{vehicle_id}"
            )

    second_edge_occurrences = Counter(
        edge
        for route in second_routes
        if route[1:-1]
        for edge in zip(route, route[1:])
    )
    for edge, count in second_edge_occurrences.items():
        if count > 1:
            violations.append(
                "DOMAIN(y) | binary second-echelon arc "
                f"{edge!r} is traversed {count} times"
            )

    # C1 | First-echelon flow conservation.
    for vehicle_id, route in enumerate(first_routes):
        if route[0] != data["depot_id"] or route[-1] != data["depot_id"]:
            violations.append(
                "C1 | first-echelon vehicle "
                f"{vehicle_id} must start and end at depot {data['depot_id']}"
            )
        incoming = Counter(end for _, end in first_edges_by_vehicle[vehicle_id])
        outgoing = first_out_counts[vehicle_id]
        for node_id in first_node_ids:
            if incoming[node_id] != outgoing[node_id]:
                violations.append(
                    "C1 | first-echelon flow is not conserved for vehicle "
                    f"{vehicle_id} at node {node_id}: in={incoming[node_id]}, "
                    f"out={outgoing[node_id]}"
                )

    # C2 | Each satellite is visited at most once by each first-echelon vehicle.
    for vehicle_id, outgoing in enumerate(first_out_counts):
        for satellite_id in satellite_ids:
            if outgoing[satellite_id] > 1:
                violations.append(
                    "C2 | first-echelon vehicle "
                    f"{vehicle_id} leaves satellite {satellite_id} "
                    f"{outgoing[satellite_id]} times"
                )

    # C3 | MTZ existence is equivalent to no directed satellite-only cycle.
    for vehicle_id, edges in enumerate(first_edges_by_vehicle):
        if _has_satellite_cycle(satellite_ids, edges):
            violations.append(
                "C3 | first-echelon vehicle "
                f"{vehicle_id} contains a directed satellite subtour, so no MTZ "
                "visit-order values can satisfy the constraint"
            )

    # C4 | Positive delivery is linked to a visit by the same vehicle.
    for satellite_id in satellite_ids:
        for vehicle_id in range(data["T1"]):
            quantity = deliveries[(satellite_id, vehicle_id)]
            limit = data["Q1"] * first_out_counts[vehicle_id][satellite_id]
            if not _leq(quantity, limit):
                violations.append(
                    "C4 | delivery exceeds the visit-linked bound for satellite "
                    f"{satellite_id}, vehicle {vehicle_id}: {quantity!r} > "
                    f"Q1*visits={limit!r}"
                )

    # C5 | Total continuous delivery of each first-echelon vehicle <= Q1.
    for vehicle_id in range(data["T1"]):
        carried = sum(
            deliveries[(satellite_id, vehicle_id)]
            for satellite_id in satellite_ids
        )
        if not _leq(carried, data["Q1"]):
            violations.append(
                "C5 | first-echelon vehicle "
                f"{vehicle_id} carries {carried!r}, exceeding Q1={data['Q1']!r}"
            )

    # C6 | Total continuous delivery into each satellite <= its throughput.
    delivered_by_satellite: dict[int, float] = {}
    for satellite_id in satellite_ids:
        delivered = sum(
            deliveries[(satellite_id, vehicle_id)]
            for vehicle_id in range(data["T1"])
        )
        delivered_by_satellite[satellite_id] = delivered
        capacity = data["satellite_capacity_by_id"][satellite_id]
        if not _leq(delivered, capacity):
            violations.append(
                "C6 | satellite "
                f"{satellite_id} receives {delivered!r}, exceeding throughput "
                f"capacity {capacity!r}"
            )

    # C7 | Each customer has exactly one entering and one leaving used route arc.
    customer_occurrences = Counter(
        customer
        for route in second_routes
        for customer in route[1:-1]
    )
    for customer_id in customer_ids:
        if customer_occurrences[customer_id] != 1:
            violations.append(
                "C7 | customer "
                f"{customer_id} occurs {customer_occurrences[customer_id]} "
                "times in second-echelon route interiors; exactly one is required"
            )

    # C8 | A nonempty route contributes one satellite-to-customer departure arc.
    routes_from_satellite = Counter(
        route[0] for route in second_routes if route[1:-1]
    )
    for satellite_id in satellite_ids:
        route_count = routes_from_satellite[satellite_id]
        limit = data["satellite_vehicle_limit_by_id"][satellite_id]
        if route_count > limit:
            violations.append(
                "C8 | satellite "
                f"{satellite_id} starts {route_count} second-echelon vehicles, "
                f"exceeding its limit {limit}"
            )

    # C9 | The same exact route counts summed system-wide cannot exceed T2.
    total_second_vehicles = sum(routes_from_satellite.values())
    if total_second_vehicles > data["T2"]:
        violations.append(
            "C9 | total second-echelon vehicles "
            f"{total_second_vehicles} exceeds T2={data['T2']}"
        )

    # C10 | Eliminate f using its customer balance equations.  On a route,
    # every arc flow is t plus a demand suffix, where t is that route's return
    # flow.  These exact integer suffixes are reused by both C11 and C14.
    flow_profiles = _route_flow_profiles(
        second_routes, data["customer_demand_by_id"]
    )

    # C11 | Intersect every used arc's nonnegativity/lower/upper conditions to
    # obtain the feasible interval for the same residual variable t from C10.
    # customers_demand has only a customer axis; at a satellite endpoint its
    # demand term is therefore extended by the neutral value 0.
    for profile in flow_profiles:
        lower_bounds: list[int] = []
        upper_bounds: list[int] = []
        edges = list(zip(profile["route"], profile["route"][1:]))
        for (start, end), offset in zip(edges, profile["offsets"]):
            destination_demand = data["customer_demand_by_id"].get(end, 0)
            source_demand = data["customer_demand_by_id"].get(start, 0)
            minimum_flow = max(0, destination_demand)
            maximum_flow = data["Q2"] - source_demand
            lower_bounds.append(minimum_flow - offset)
            upper_bounds.append(maximum_flow - offset)

        lower = max(lower_bounds)
        upper = min(upper_bounds)
        profile["lower"] = lower
        profile["upper"] = upper
        profile["interval_feasible"] = lower <= upper
        if lower > upper:
            violations.append(
                "C11 | second-echelon route "
                f"{profile['route_index']} has no load value satisfying all arc "
                f"bounds: residual interval [{lower!r}, {upper!r}] is empty"
            )

    # C12 | Route membership and the reported z assignment must name the route's
    # origin satellite; the route must return to that same satellite.
    for route_index, route in enumerate(second_routes):
        if route[0] != route[-1]:
            violations.append(
                "C12 | second-echelon route "
                f"{route_index} starts at satellite {route[0]} but ends at "
                f"satellite {route[-1]}"
            )
        for customer_id in route[1:-1]:
            assigned_satellite = customer_assignments[customer_id]
            if assigned_satellite != route[0]:
                violations.append(
                    "C12 | customer "
                    f"{customer_id} is on a route from satellite {route[0]} but "
                    f"customer_satellite_assignments reports {assigned_satellite}"
                )

    # C13 | Every member customer's z assignment equals its group's v assignment.
    for group_id, members in zip(
        data["groups_group_id"], data["groups_customer_ids"]
    ):
        group_satellite = group_assignments[group_id]
        for customer_id in members:
            customer_satellite = customer_assignments[customer_id]
            if customer_satellite != group_satellite:
                violations.append(
                    "C13 | group "
                    f"{group_id} is assigned to satellite {group_satellite}, but "
                    f"member customer {customer_id} is assigned to "
                    f"{customer_satellite}"
                )

    # C14 | Couple C10/C11's very same residual variables across all routes from
    # a satellite.  Sum(t_r + route_demand_r) must equal sum_k w[s][k].
    profiles_by_satellite: dict[int, list[dict]] = defaultdict(list)
    for profile in flow_profiles:
        profiles_by_satellite[profile["satellite"]].append(profile)

    for satellite_id in satellite_ids:
        profiles = profiles_by_satellite[satellite_id]
        if any(not profile["interval_feasible"] for profile in profiles):
            # C11 already proves that no common f exists; avoid a derivative C14
            # message based on an empty interval.
            continue

        route_demand = sum(profile["offsets"][0] for profile in profiles)
        residual_target = delivered_by_satellite[satellite_id] - route_demand
        residual_lower = sum(profile["lower"] for profile in profiles)
        residual_upper = sum(profile["upper"] for profile in profiles)
        if not (
            _geq(residual_target, residual_lower)
            and _leq(residual_target, residual_upper)
        ):
            violations.append(
                "C14 | satellite "
                f"{satellite_id} receives {delivered_by_satellite[satellite_id]!r}, "
                "but C10/C11 permit total departing f only when the summed "
                f"residual lies in [{residual_lower!r}, {residual_upper!r}]; "
                f"the delivery requires residual {residual_target!r}"
            )

    return violations


def check_solution(instance: dict, solution: dict) -> list:
    """Return constraint violations; an empty list means feasible.

    The broad guard is intentional: malformed or degenerate inputs must be
    reported as violations rather than escaping through an exception.
    """
    try:
        return _check_solution(instance, solution)
    except Exception as exc:  # pragma: no cover - last-resort robustness boundary
        return [
            "FORMAT | checker could not safely evaluate the supplied data: "
            f"{type(exc).__name__}: {exc}"
        ]
