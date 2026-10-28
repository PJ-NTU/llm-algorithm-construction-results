"""Feasibility checker for the inventory-routing output described in MAT_s4.md."""

from __future__ import annotations

from collections import namedtuple
from decimal import Decimal
import math
import re
from typing import Any


_REL = Decimal("1e-9")
_REQUIRED_FIELDS = (
    "routes",
    "retailer_visit_status",
    "deliveries",
    "retailer_inventory",
    "supplier_inventory",
)


class _Number(namedtuple("_NumberBase", "value integer_chain")):
    __slots__ = ()

    def __add__(self, other: "_Number") -> "_Number":
        return _Number(
            self.value + other.value,
            self.integer_chain and other.integer_chain,
        )

    def __sub__(self, other: "_Number") -> "_Number":
        return _Number(
            self.value - other.value,
            self.integer_chain and other.integer_chain,
        )

    def __mul__(self, other: "_Number") -> "_Number":
        return _Number(
            self.value * other.value,
            self.integer_chain and other.integer_chain,
        )


_ZERO = _Number(Decimal(0), True)


def _number(raw: Any, label: str, violations: list[str]) -> _Number | None:
    """Read a finite JSON number while retaining whether it is an integer chain."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        violations.append(
            f"FORMAT | {label} must be a finite number; got {type(raw).__name__}, "
            "so the stated numeric decision cannot be checked safely"
        )
        return None
    if isinstance(raw, float) and not math.isfinite(raw):
        violations.append(
            f"FORMAT | {label} must be finite; got {raw!r}, so arithmetic constraints "
            "would be undefined"
        )
        return None
    return _Number(Decimal(raw) if isinstance(raw, int) else Decimal(str(raw)), isinstance(raw, int))


def _scale(a: _Number, b: _Number) -> Decimal:
    return max(Decimal(1), abs(a.value), abs(b.value))


def _equal(a: _Number, b: _Number) -> bool:
    if a.integer_chain and b.integer_chain:
        return a.value == b.value
    return abs(a.value - b.value) <= _REL * _scale(a, b)


def _direct_equal(a: _Number, b: _Number) -> bool:
    """Compare an uncomputed instance value with its required copy exactly."""
    return a.value == b.value


def _leq(a: _Number, b: _Number) -> bool:
    if a.integer_chain and b.integer_chain:
        return a.value <= b.value
    return a.value - b.value <= _REL * _scale(a, b)


def _geq(a: _Number, b: _Number) -> bool:
    if a.integer_chain and b.integer_chain:
        return a.value >= b.value
    return b.value - a.value <= _REL * _scale(a, b)


def _sum_numbers(values: Any) -> _Number:
    total = _ZERO
    for value in values:
        total = total + value
    return total


def _parse_index(key: Any) -> int | None:
    """Parse an otherwise-unspecified string spelling containing one index."""
    if not isinstance(key, str):
        return None
    digit_groups = re.findall(r"[0-9]+", key)
    if len(digit_groups) != 1:
        return None
    return int(digit_groups[0])


def _indexed_mapping(
    raw: Any,
    start: int,
    end: int,
    label: str,
    violations: list[str],
) -> dict[int, Any] | None:
    """Normalize string keys and require exactly the described inclusive index domain."""
    before = len(violations)
    if not isinstance(raw, dict):
        violations.append(
            f"FORMAT | {label} must be dict[str, ...]; got {type(raw).__name__}, "
            "which does not match the stated output shape"
        )
        return None

    parsed: dict[int, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            violations.append(
                f"FORMAT | {label} key {key!r} must be str because the stated shape is "
                "dict[str, ...]"
            )
            continue
        index = _parse_index(key)
        if index is None:
            violations.append(
                f"FORMAT | {label} key {key!r} does not contain one uniquely parseable "
                "integer index; the description does not otherwise restrict key spelling"
            )
            continue
        if index in parsed:
            violations.append(
                f"FORMAT | {label} keys collide at index {index}; after accepting flexible "
                "string spelling, each described index must still occur exactly once"
            )
            continue
        parsed[index] = value

    expected_count = max(0, end - start + 1)
    outside = sorted(index for index in parsed if index < start or index > end)
    if outside:
        violations.append(
            f"FORMAT | {label} has indices outside the described domain {start}..{end}: "
            f"{outside}; required decision fields may not contain undeclared indices"
        )
    if len(parsed) != expected_count:
        violations.append(
            f"FORMAT | {label} must contain every index {start}..{end} exactly once; "
            f"found {len(parsed)} distinct parseable indices, expected {expected_count}"
        )

    if len(violations) != before:
        return None
    return parsed


def _instance_data(instance: Any, violations: list[str]) -> dict[str, Any] | None:
    """Read only instance fields needed to evaluate the stated constraints."""
    if not isinstance(instance, dict):
        violations.append(
            f"FORMAT | instance must be a dict; got {type(instance).__name__}, so its "
            "given parameters cannot be read"
        )
        return None

    scalar_names = (
        "supplier_production_rate",
        "supplier_starting_inventory",
        "vehicle_capacity",
    )
    list_names = (
        "retailers_demand",
        "retailers_max_inventory",
        "retailers_starting_inventory",
    )
    required = ("num_retailers", "time_horizon", *scalar_names, *list_names)
    missing = [name for name in required if name not in instance]
    if missing:
        violations.append(
            f"FORMAT | instance is missing fields needed by the authoritative constraints: "
            f"{missing}; without them feasibility cannot be recomputed"
        )
        return None

    n = instance["num_retailers"]
    horizon = instance["time_horizon"]
    if isinstance(n, bool) or not isinstance(n, int):
        violations.append(
            f"FORMAT | instance.num_retailers must be int as stated; got {type(n).__name__}"
        )
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        violations.append(
            f"FORMAT | instance.time_horizon must be int as stated; got "
            f"{type(horizon).__name__}"
        )
    if violations:
        return None

    data: dict[str, Any] = {"n": n, "horizon": horizon}
    for name in scalar_names:
        value = _number(instance[name], f"instance.{name}", violations)
        if value is not None:
            data[name] = value

    needed_retailers = max(0, n)
    for name in list_names:
        raw = instance[name]
        if not isinstance(raw, list):
            violations.append(
                f"FORMAT | instance.{name} must be a list under the stated axis convention; "
                f"got {type(raw).__name__}"
            )
            continue
        if len(raw) < needed_retailers:
            violations.append(
                f"FORMAT | instance.{name} has {len(raw)} entries but retailer indices "
                f"1..{n} require {needed_retailers}; the checker cannot read every parameter"
            )
            continue
        converted: list[_Number] = []
        for offset in range(needed_retailers):
            value = _number(raw[offset], f"instance.{name}[{offset}]", violations)
            if value is not None:
                converted.append(value)
        if len(converted) == needed_retailers:
            data[name] = converted

    if violations:
        return None
    return data


def _solution_data(
    solution: Any,
    data: dict[str, Any],
    violations: list[str],
) -> dict[str, Any] | None:
    """Validate the five required fields and normalize all described decision indices."""
    if not isinstance(solution, dict):
        violations.append(
            f"FORMAT | solution must be a dict; got {type(solution).__name__}, so required "
            "decision fields cannot be read"
        )
        return None
    missing = [name for name in _REQUIRED_FIELDS if name not in solution]
    if missing:
        violations.append(
            f"FORMAT | solution is missing required decision fields {missing}; MAT_s4.md "
            f"lists exactly {_REQUIRED_FIELDS} as required"
        )
        return None

    n = data["n"]
    horizon = data["horizon"]
    routes_raw = _indexed_mapping(solution["routes"], 1, horizon, "routes", violations)
    visits_raw = _indexed_mapping(
        solution["retailer_visit_status"],
        1,
        horizon,
        "retailer_visit_status",
        violations,
    )
    deliveries_raw = _indexed_mapping(
        solution["deliveries"], 1, horizon, "deliveries", violations
    )
    retailer_inventory_raw = _indexed_mapping(
        solution["retailer_inventory"],
        1,
        horizon + 1,
        "retailer_inventory",
        violations,
    )
    supplier_inventory_raw = _indexed_mapping(
        solution["supplier_inventory"],
        1,
        horizon + 1,
        "supplier_inventory",
        violations,
    )
    if any(
        value is None
        for value in (
            routes_raw,
            visits_raw,
            deliveries_raw,
            retailer_inventory_raw,
            supplier_inventory_raw,
        )
    ):
        return None

    routes: dict[int, list[tuple[int, int, int]]] = {}
    visits: dict[int, list[int]] = {}
    deliveries: dict[int, dict[int, _Number]] = {}
    retailer_inventory: dict[int, dict[int, _Number]] = {}
    supplier_inventory: dict[int, _Number] = {}

    for period, raw_period_routes in routes_raw.items():
        if not isinstance(raw_period_routes, list):
            violations.append(
                f"FORMAT | routes[{period}] must be list[list[int]]; got "
                f"{type(raw_period_routes).__name__}"
            )
            continue
        converted_routes: list[tuple[int, int, int]] = []
        for edge_pos, edge in enumerate(raw_period_routes):
            if not isinstance(edge, list) or len(edge) != 3:
                violations.append(
                    f"FORMAT | routes[{period}][{edge_pos}] must be a three-item list "
                    "[start, end, traversal_count] as stated"
                )
                continue
            if any(isinstance(item, bool) or not isinstance(item, int) for item in edge):
                violations.append(
                    f"FORMAT | routes[{period}][{edge_pos}] must contain three integers; "
                    f"got {edge!r}"
                )
                continue
            start, end, count = edge
            if not (0 <= start <= n and 0 <= end <= n):
                violations.append(
                    f"FORMAT | routes[{period}][{edge_pos}] uses node ({start}, {end}) "
                    f"outside supplier 0 and the described retailers 1..{n}"
                )
                continue
            converted_routes.append((start, end, count))
        routes[period] = converted_routes

    for period, raw_period_visits in visits_raw.items():
        if not isinstance(raw_period_visits, list) or len(raw_period_visits) != max(0, n):
            found = len(raw_period_visits) if isinstance(raw_period_visits, list) else "non-list"
            violations.append(
                f"FORMAT | retailer_visit_status[{period}] must list all {max(0, n)} "
                f"retailers exactly once; found {found} entries"
            )
            continue
        if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_period_visits):
            violations.append(
                f"FORMAT | retailer_visit_status[{period}] must be list[int]; booleans and "
                "non-integers do not represent the stated z decisions"
            )
            continue
        visits[period] = list(raw_period_visits)

    for period, raw_period_deliveries in deliveries_raw.items():
        period_map = _indexed_mapping(
            raw_period_deliveries,
            1,
            n,
            f"deliveries[{period}]",
            violations,
        )
        if period_map is None:
            continue
        converted: dict[int, _Number] = {}
        for retailer, raw_value in period_map.items():
            value = _number(
                raw_value, f"deliveries[{period}][{retailer}]", violations
            )
            if value is not None:
                converted[retailer] = value
        if len(converted) == max(0, n):
            deliveries[period] = converted

    for period, raw_period_inventory in retailer_inventory_raw.items():
        period_map = _indexed_mapping(
            raw_period_inventory,
            1,
            n,
            f"retailer_inventory[{period}]",
            violations,
        )
        if period_map is None:
            continue
        converted = {}
        for retailer, raw_value in period_map.items():
            value = _number(
                raw_value,
                f"retailer_inventory[{period}][{retailer}]",
                violations,
            )
            if value is not None:
                converted[retailer] = value
        if len(converted) == max(0, n):
            retailer_inventory[period] = converted

    for period, raw_value in supplier_inventory_raw.items():
        value = _number(raw_value, f"supplier_inventory[{period}]", violations)
        if value is not None:
            supplier_inventory[period] = value

    if violations:
        return None

    # Output completeness (not C10): no deliveries means the original route list is [].
    for period in range(1, horizon + 1):
        no_delivery = all(
            _equal(deliveries[period][retailer], _ZERO)
            for retailer in range(1, n + 1)
        )
        if no_delivery and routes_raw[period] != []:
            violations.append(
                f"FORMAT | routes[{period}] must be the literal empty list [] because every "
                "delivery in that period is zero; MAT_s4.md says 'an empty list if a period has no deliveries'"
            )

    if violations:
        return None
    return {
        "routes": routes,
        "visits": visits,
        "deliveries": deliveries,
        "retailer_inventory": retailer_inventory,
        "supplier_inventory": supplier_inventory,
    }


def _route_variables(
    routes: list[tuple[int, int, int]],
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int, int]]]:
    aggregated: dict[tuple[int, int], int] = {}
    for start, end, count in routes:
        key = (max(start, end), min(start, end))
        aggregated[key] = aggregated.get(key, 0) + count
    return aggregated, routes


def _check_constraints(data: dict[str, Any], sol: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    n = data["n"]
    horizon = data["horizon"]
    deliveries = sol["deliveries"]
    inventory = sol["retailer_inventory"]
    supplier_inventory = sol["supplier_inventory"]
    visits = sol["visits"]

    shipment_totals = {
        period: _sum_numbers(deliveries[period].values())
        for period in range(1, horizon + 1)
    }
    route_vars = {
        period: _route_variables(sol["routes"][period])
        for period in range(1, horizon + 1)
    }

    # C1 | B[t+1] = B[t] + r_0 - sum_s x[t][s], with B[1] given.
    if horizon + 1 >= 1 and 1 in supplier_inventory:
        if not _direct_equal(
            supplier_inventory[1], data["supplier_starting_inventory"]
        ):
            violations.append(
                "C1 | supplier_inventory[1] does not equal the given "
                "supplier_starting_inventory exactly; B[1] is the stated initial condition"
            )
    for period in range(1, horizon + 1):
        expected = (
            supplier_inventory[period]
            + data["supplier_production_rate"]
            - shipment_totals[period]
        )
        if not _equal(supplier_inventory[period + 1], expected):
            violations.append(
                f"C1 | supplier_inventory[{period + 1}]={supplier_inventory[period + 1].value} "
                f"but B[t]+r_0-sum(x)={expected.value}; the supplier inventory balance "
                f"fails after period {period}"
            )
    for period in range(1, horizon + 2):
        if not _geq(supplier_inventory[period], _ZERO):
            violations.append(
                f"C1 | supplier_inventory[{period}]={supplier_inventory[period].value} is "
                "negative, violating the stated B[t] >= 0 decision domain"
            )

    # C2 | B[t] >= sum_s x[t][s].
    for period in range(1, horizon + 1):
        if not _geq(supplier_inventory[period], shipment_totals[period]):
            violations.append(
                f"C2 | period {period} starts with supplier inventory "
                f"{supplier_inventory[period].value} but ships {shipment_totals[period].value}; "
                "current inventory is insufficient for the recomputed deliveries"
            )

    # C3 | I[t+1][s] = I[t][s] + x[t][s] - r_s, with I[1][s] given.
    for retailer in range(1, n + 1):
        if not _direct_equal(
            inventory[1][retailer],
            data["retailers_starting_inventory"][retailer - 1],
        ):
            violations.append(
                f"C3 | retailer_inventory[1][{retailer}] does not equal the given "
                f"starting inventory exactly; I[1][{retailer}] is the stated initial condition"
            )
        for period in range(1, horizon + 1):
            expected = (
                inventory[period][retailer]
                + deliveries[period][retailer]
                - data["retailers_demand"][retailer - 1]
            )
            if not _equal(inventory[period + 1][retailer], expected):
                violations.append(
                    f"C3 | retailer {retailer}, period {period}: next inventory is "
                    f"{inventory[period + 1][retailer].value}, expected {expected.value} from "
                    "I[t][s]+x[t][s]-r_s"
                )

    # C4 | 0 <= I[t][s] <= U_s for all represented H+1 inventory periods.
    for period in range(1, horizon + 2):
        for retailer in range(1, n + 1):
            current = inventory[period][retailer]
            maximum = data["retailers_max_inventory"][retailer - 1]
            if not _geq(current, _ZERO):
                violations.append(
                    f"C4 | retailer_inventory[{period}][{retailer}]={current.value} is below "
                    "zero, violating the lower inventory bound"
                )
            if not _leq(current, maximum):
                violations.append(
                    f"C4 | retailer_inventory[{period}][{retailer}]={current.value} exceeds "
                    f"U_{retailer}={maximum.value}, violating the maximum inventory bound"
                )

    # C5 | Order-up-to inequalities, plus the stated x[t][s] >= 0 domain.
    for period in range(1, horizon + 1):
        for retailer in range(1, n + 1):
            quantity = deliveries[period][retailer]
            maximum = data["retailers_max_inventory"][retailer - 1]
            current = inventory[period][retailer]
            visit = _Number(Decimal(visits[period][retailer - 1]), True)
            lower = maximum * visit - current
            stock_room = maximum - current
            visit_cap = maximum * visit
            if not _geq(quantity, _ZERO):
                violations.append(
                    f"C5 | deliveries[{period}][{retailer}]={quantity.value} is negative, "
                    "violating the stated x[t][s] >= 0 domain"
                )
            if not _geq(quantity, lower):
                violations.append(
                    f"C5 | deliveries[{period}][{retailer}]={quantity.value} is below "
                    f"U_s*z-I={lower.value}, so a visited retailer is not raised to U_s"
                )
            if not _leq(quantity, stock_room):
                violations.append(
                    f"C5 | deliveries[{period}][{retailer}]={quantity.value} exceeds "
                    f"U_s-I={stock_room.value}, so delivery would overfill the retailer"
                )
            if not _leq(quantity, visit_cap):
                violations.append(
                    f"C5 | deliveries[{period}][{retailer}]={quantity.value} exceeds "
                    f"U_s*z={visit_cap.value}, so product is delivered without the required visit"
                )

    # C6 | sum_s x[t][s] <= C.
    for period in range(1, horizon + 1):
        if not _leq(shipment_totals[period], data["vehicle_capacity"]):
            violations.append(
                f"C6 | period {period} ships {shipment_totals[period].value}, exceeding "
                f"vehicle capacity C={data['vehicle_capacity'].value}"
            )

    # C7 | sum_s x[t][s] <= C*z[t][0]; z[t][0] is derived from supplier edges.
    supplier_visits: dict[int, int] = {}
    for period in range(1, horizon + 1):
        edges = route_vars[period][0]
        leaves_supplier = any(
            count > 0 and low == 0 and high != 0
            for (high, low), count in edges.items()
        )
        supplier_visits[period] = int(leaves_supplier)
        allowed = data["vehicle_capacity"] * _Number(
            Decimal(supplier_visits[period]), True
        )
        if not _leq(shipment_totals[period], allowed):
            if supplier_visits[period] == 0:
                violations.append(
                    f"C7 | period {period} ships {shipment_totals[period].value} but no positive "
                    "supplier-retailer edge shows that the vehicle leaves node 0; therefore "
                    "the represented z[t][0] is 0"
                )
            else:
                violations.append(
                    f"C7 | period {period} ships {shipment_totals[period].value}, exceeding "
                    f"C*z[t][0]={allowed.value} even though the represented supplier visit is 1"
                )

    # C8 | Edge/visit domains and degree(i) = 2*z[t][i] at every node.
    for period in range(1, horizon + 1):
        edges, raw_edges = route_vars[period]
        for start, end, count in raw_edges:
            allowed_counts = {0, 1, 2} if 0 in (start, end) else {0, 1}
            if count not in allowed_counts:
                violations.append(
                    f"C8 | route entry ({start}, {end}, {count}) in period {period} violates "
                    f"the stated y domain {sorted(allowed_counts)} for that edge"
                )
        for (high, low), count in edges.items():
            allowed_counts = {0, 1, 2} if low == 0 else {0, 1}
            if count not in allowed_counts:
                violations.append(
                    f"C8 | aggregated y[{period}][{high}][{low}]={count} violates its stated "
                    f"domain {sorted(allowed_counts)}; duplicate route records denote one y variable"
                )
        for retailer, visit in enumerate(visits[period], start=1):
            if visit not in (0, 1):
                violations.append(
                    f"C8 | retailer_visit_status[{period}][{retailer}]={visit} violates "
                    "the stated binary z domain {0,1}"
                )
        for node in range(0, n + 1):
            degree = sum(
                count
                for (high, low), count in edges.items()
                if high != low and node in (high, low)
            )
            visit = supplier_visits[period] if node == 0 else visits[period][node - 1]
            if degree != 2 * visit:
                violations.append(
                    f"C8 | node {node} in period {period} has recomputed incident-edge "
                    f"degree {degree}, but 2*z is {2 * visit}; the required degree equality fails"
                )

    # C9 | Retailer-only subset inequalities, equivalently visited forest + valid endpoints.
    for period in range(1, horizon + 1):
        edges = route_vars[period][0]
        parent = {retailer: retailer for retailer in range(1, n + 1)}

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for (high, low), count in edges.items():
            if count <= 0 or low == 0 or high == low:
                continue
            if visits[period][high - 1] != 1 or visits[period][low - 1] != 1:
                violations.append(
                    f"C9 | retailer edge ({low}, {high}) in period {period} has positive y "
                    "but at least one endpoint has z=0; choosing a subset containing that edge "
                    "violates the stated subtour inequality"
                )
                continue
            root_high = find(high)
            root_low = find(low)
            if root_high == root_low:
                violations.append(
                    f"C9 | retailer edge ({low}, {high}) closes a retailer-only cycle in "
                    f"period {period}; for that cycle S, internal y=|S| exceeds |S|-1"
                )
            else:
                parent[root_high] = root_low

    return violations


def check_solution(instance: dict, solution: dict) -> list:
    """Return concrete feasibility violations; return [] exactly for a feasible solution."""
    try:
        format_violations: list[str] = []
        data = _instance_data(instance, format_violations)
        if data is None:
            return format_violations
        sol = _solution_data(solution, data, format_violations)
        if sol is None:
            return format_violations
        return _check_constraints(data, sol)
    except Exception as exc:  # Defensive API boundary: malformed input must never escape.
        return [
            f"FORMAT | checker could not safely evaluate the supplied data: "
            f"{type(exc).__name__}: {exc}"
        ]
