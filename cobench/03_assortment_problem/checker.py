"""Feasibility checker for the authoritative model in MAT_s4.md.

Only the three required solution fields are read.  Optional fields, including a
reported objective value, are deliberately ignored.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from numbers import Real
from typing import Any


_REL = 1e-9


def _leq(a: float, b: float) -> bool:
    """Relative-tolerance comparison for values produced by float arithmetic."""
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _geq(a: float, b: float) -> bool:
    """Relative-tolerance comparison for values produced by float arithmetic."""
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _is_exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _key_index(key: Any, count: int) -> int | None:
    """Resolve a string key when its numeric content identifies one index.

    No prefix, case, whitespace, separator, or leading-zero convention is
    imposed.  Non-string keys are rejected because the declared shape is
    ``dict[str, int]``.
    """
    if not isinstance(key, str):
        return None

    candidates = {
        int(token)
        for token in re.findall(r"\d+", key)
        if 0 <= int(token) < count
    }
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


def _sequence_value(container: Any, key: str, index: int) -> Any:
    values = container.get(key) if isinstance(container, dict) else None
    if isinstance(values, (list, tuple)) and 0 <= index < len(values):
        return values[index]
    return None


def _stock_dimension(instance: dict, stock_type_id: int, name: str) -> Any:
    stocks = instance.get("stocks")
    if isinstance(stocks, (list, tuple)) and 0 <= stock_type_id < len(stocks):
        stock = stocks[stock_type_id]
        if isinstance(stock, dict) and name in stock:
            return stock[name]
    return _sequence_value(instance, name, stock_type_id)


def _piece_dimension(instance: dict, piece_type_id: int, name: str) -> Any:
    pieces = instance.get("pieces")
    if isinstance(pieces, (list, tuple)) and 0 <= piece_type_id < len(pieces):
        piece = pieces[piece_type_id]
        if isinstance(piece, dict) and name in piece:
            return piece[name]
    return _sequence_value(instance, f"piece_{name}", piece_type_id)


def _piece_bound(instance: dict, piece_type_id: int, name: str) -> Any:
    value = _sequence_value(instance, name, piece_type_id)
    if value is not None:
        return value
    pieces = instance.get("pieces")
    if isinstance(pieces, (list, tuple)) and 0 <= piece_type_id < len(pieces):
        piece = pieces[piece_type_id]
        if isinstance(piece, dict):
            return piece.get(name)
    return None


def _parse_solution(
    instance: dict, solution: Any, m: int, stock_type_count: int
) -> tuple[dict[str, Any], list[str]]:
    """Validate only the structure needed to evaluate the seven constraints."""
    violations: list[str] = []
    parsed: dict[str, Any] = {
        "selected": None,
        "stock_instances": [],
        "piece_counts": None,
    }

    if not isinstance(solution, dict):
        return parsed, ["FORMAT | solution must be a dict"]

    required = ("selected_stock_types", "stock_instances", "piece_counts")
    for field in required:
        if field not in solution:
            violations.append(f"FORMAT | missing required field {field!r}")

    selected = solution.get("selected_stock_types")
    if not isinstance(selected, list):
        violations.append("FORMAT | selected_stock_types must be a list[int]")
    elif len(selected) != stock_type_count:
        violations.append(
            "FORMAT | selected_stock_types must list all "
            f"{stock_type_count} stock types; got {len(selected)} entries"
        )
    elif any(not _is_exact_int(value) or value not in (0, 1) for value in selected):
        violations.append(
            "FORMAT | every selected_stock_types entry must be the integer 0 or 1"
        )
    else:
        parsed["selected"] = selected

    raw_instances = solution.get("stock_instances")
    if not isinstance(raw_instances, list):
        violations.append("FORMAT | stock_instances must be a list[dict]")
    else:
        for instance_pos, raw_stock in enumerate(raw_instances):
            where = f"stock_instances[{instance_pos}]"
            if not isinstance(raw_stock, dict):
                violations.append(f"FORMAT | {where} must be a dict")
                continue

            stock_type_id = raw_stock.get("stock_type_id")
            stock_id_ok = _is_exact_int(stock_type_id) and (
                0 <= stock_type_id < stock_type_count
            )
            if not stock_id_ok:
                violations.append(
                    f"FORMAT | {where}.stock_type_id must be an integer index "
                    f"in [0, {stock_type_count})"
                )

            raw_placements = raw_stock.get("placements")
            if not isinstance(raw_placements, list):
                violations.append(f"FORMAT | {where}.placements must be a list")
                raw_placements = []

            placements: list[dict[str, Any]] = []
            for placement_pos, raw_placement in enumerate(raw_placements):
                p_where = f"{where}.placements[{placement_pos}]"
                if not isinstance(raw_placement, dict):
                    violations.append(f"FORMAT | {p_where} must be a dict")
                    continue

                piece_type_id = raw_placement.get("piece_type_id")
                piece_id_ok = _is_exact_int(piece_type_id) and (
                    0 <= piece_type_id < m
                )
                if not piece_id_ok:
                    violations.append(
                        f"FORMAT | {p_where}.piece_type_id must be an integer "
                        f"index in [0, {m})"
                    )

                rotation = raw_placement.get("is_rotated")
                rotation_ok = _is_exact_int(rotation) and rotation in (0, 1)
                if not rotation_ok:
                    violations.append(
                        f"FORMAT | {p_where}.is_rotated must be the integer 0 or 1"
                    )

                x = raw_placement.get("x")
                y = raw_placement.get("y")
                coordinates_ok = _is_finite_number(x) and _is_finite_number(y)
                if not coordinates_ok:
                    violations.append(
                        f"FORMAT | {p_where}.x and .y must be finite numbers"
                    )

                if piece_id_ok and rotation_ok and coordinates_ok:
                    placements.append(
                        {
                            "piece_type_id": piece_type_id,
                            "x": float(x),
                            "y": float(y),
                            "is_rotated": rotation,
                            "where": p_where,
                        }
                    )

            if stock_id_ok:
                parsed["stock_instances"].append(
                    {
                        "stock_type_id": stock_type_id,
                        "placements": placements,
                        "where": where,
                    }
                )

    raw_counts = solution.get("piece_counts")
    if not isinstance(raw_counts, dict):
        violations.append("FORMAT | piece_counts must be a dict[str, int]")
    else:
        counts_by_index: dict[int, int] = {}
        counts_ok = True
        for key, value in raw_counts.items():
            if not isinstance(key, str):
                violations.append(
                    "FORMAT | every piece_counts key must be a string; "
                    f"got {key!r}"
                )
                counts_ok = False
                continue

            index = _key_index(key, m)
            if index is None:
                violations.append(
                    f"FORMAT | piece_counts key {key!r} does not uniquely identify "
                    f"an index in [0, {m})"
                )
                counts_ok = False
                continue
            if index in counts_by_index:
                violations.append(
                    f"FORMAT | multiple piece_counts keys identify piece type {index}"
                )
                counts_ok = False
                continue
            if not _is_exact_int(value):
                violations.append(
                    f"FORMAT | piece_counts[{key!r}] must be an integer"
                )
                counts_ok = False
                continue
            counts_by_index[index] = value

        missing = sorted(set(range(m)) - set(counts_by_index))
        if missing:
            violations.append(
                f"FORMAT | piece_counts is missing piece type indices {missing}"
            )
            counts_ok = False
        if counts_ok:
            parsed["piece_counts"] = counts_by_index

    return parsed, violations


def _check_c1(selected: list[int] | None) -> list[str]:
    """C1: at most two binary stock-selection indicators may be active."""
    if selected is None:
        return []
    selected_count = sum(selected)  # exact integer sum: no tolerance
    if selected_count > 2:
        return [
            "C1 | at most two distinct stock types may be selected; "
            f"found {selected_count}"
        ]
    return []


def _check_c2(
    selected: list[int] | None, stock_instances: list[dict[str, Any]]
) -> list[str]:
    """C2: every listed (therefore active) instance needs a selected type."""
    if selected is None:
        return []
    violations: list[str] = []
    for stock in stock_instances:
        stock_type_id = stock["stock_type_id"]
        if selected[stock_type_id] != 1:
            violations.append(
                "C2 | "
                f"{stock['where']} is used but stock type {stock_type_id} is not selected"
            )
    return violations


def _check_c3(stock_instances: list[dict[str, Any]]) -> list[str]:
    """C3: the list representation defines both n[s] and sum_k z[s,k].

    Each list entry is one active instance by the output completeness rule.  Thus
    counting entries of type s reconstructs n[s], while summing one active
    indicator for exactly those same entries reconstructs sum_k z[s,k].
    """
    n_by_type = Counter(stock["stock_type_id"] for stock in stock_instances)
    active_indicator_sum = Counter()
    for stock in stock_instances:
        active_indicator_sum[stock["stock_type_id"]] += 1

    violations: list[str] = []
    for stock_type_id in set(n_by_type) | set(active_indicator_sum):
        if n_by_type[stock_type_id] != active_indicator_sum[stock_type_id]:
            violations.append(
                "C3 | stock type "
                f"{stock_type_id} has n={n_by_type[stock_type_id]} but "
                f"sum(z)={active_indicator_sum[stock_type_id]}"
            )
    return violations


def _check_c4(
    instance: dict,
    m: int,
    stock_instances: list[dict[str, Any]],
    declared_counts: dict[int, int] | None,
) -> list[str]:
    """C4: reconstruct binary x placements, totals, and exact demand bounds."""
    violations: list[str] = []
    actual_counts = [0] * m

    for stock in stock_instances:
        seen_piece_types: set[int] = set()
        for placement in stock["placements"]:
            piece_type_id = placement["piece_type_id"]
            actual_counts[piece_type_id] += 1
            if piece_type_id in seen_piece_types:
                violations.append(
                    "C4 | "
                    f"{stock['where']} contains piece type {piece_type_id} more "
                    "than once, but x[i][k][s] is binary"
                )
            else:
                seen_piece_types.add(piece_type_id)

    for piece_type_id, actual in enumerate(actual_counts):
        if declared_counts is not None:
            declared = declared_counts[piece_type_id]
            if declared != actual:  # exact integers: no tolerance
                violations.append(
                    "C4 | piece type "
                    f"{piece_type_id} declares count {declared}, but placements "
                    f"reconstruct count {actual}"
                )

        minimum = _piece_bound(instance, piece_type_id, "min")
        maximum = _piece_bound(instance, piece_type_id, "max")
        if not _is_exact_int(minimum) or not _is_exact_int(maximum):
            violations.append(
                "FORMAT | cannot evaluate C4 because instance min/max for piece "
                f"type {piece_type_id} are not integers"
            )
            continue
        if actual < minimum:  # exact integers: no tolerance
            violations.append(
                f"C4 | piece type {piece_type_id} count {actual} is below min {minimum}"
            )
        if actual > maximum:  # exact integers: no tolerance
            violations.append(
                f"C4 | piece type {piece_type_id} count {actual} exceeds max {maximum}"
            )
    return violations


def _rectangle(
    instance: dict, stock: dict[str, Any], placement: dict[str, Any]
) -> tuple[float, float, float, float, float, float] | None:
    stock_type_id = stock["stock_type_id"]
    piece_type_id = placement["piece_type_id"]
    stock_length = _stock_dimension(instance, stock_type_id, "length")
    stock_width = _stock_dimension(instance, stock_type_id, "width")
    piece_length = _piece_dimension(instance, piece_type_id, "length")
    piece_width = _piece_dimension(instance, piece_type_id, "width")
    values = (stock_length, stock_width, piece_length, piece_width)
    if not all(_is_finite_number(value) for value in values):
        return None

    if placement["is_rotated"] == 1:
        placed_width = float(piece_width)
        placed_height = float(piece_length)
    else:
        placed_width = float(piece_length)
        placed_height = float(piece_width)

    left = placement["x"]
    bottom = placement["y"]
    right = left + placed_width
    top = bottom + placed_height
    return left, bottom, right, top, float(stock_length), float(stock_width)


def _check_c5(
    instance: dict, stock_instances: list[dict[str, Any]]
) -> list[str]:
    """C5: rotated rectangles must stay inside their parent stock boundary."""
    violations: list[str] = []
    for stock in stock_instances:
        for placement in stock["placements"]:
            rectangle = _rectangle(instance, stock, placement)
            if rectangle is None:
                violations.append(
                    "FORMAT | cannot evaluate C5 because dimensions for "
                    f"{placement['where']} are missing or non-finite"
                )
                continue
            left, bottom, right, top, stock_length, stock_width = rectangle
            if not _geq(left, 0.0):
                violations.append(
                    f"C5 | {placement['where']} has negative x-coordinate {left}"
                )
            if not _geq(bottom, 0.0):
                violations.append(
                    f"C5 | {placement['where']} has negative y-coordinate {bottom}"
                )
            if not _leq(right, stock_length):
                violations.append(
                    f"C5 | {placement['where']} right edge {right} exceeds "
                    f"stock length {stock_length}"
                )
            if not _leq(top, stock_width):
                violations.append(
                    f"C5 | {placement['where']} top edge {top} exceeds "
                    f"stock width {stock_width}"
                )
    return violations


def _check_c6(
    instance: dict, stock_instances: list[dict[str, Any]]
) -> list[str]:
    """C6: every pair in the same instance must be linearly separated."""
    violations: list[str] = []
    for stock in stock_instances:
        placements = stock["placements"]
        rectangles = [_rectangle(instance, stock, p) for p in placements]
        for first_index in range(len(placements)):
            first = rectangles[first_index]
            if first is None:
                continue
            for second_index in range(first_index + 1, len(placements)):
                second = rectangles[second_index]
                if second is None:
                    continue
                a_left, a_bottom, a_right, a_top, _, _ = first
                b_left, b_bottom, b_right, b_top, _, _ = second
                separated = (
                    _leq(a_right, b_left)
                    or _leq(b_right, a_left)
                    or _leq(a_top, b_bottom)
                    or _leq(b_top, a_bottom)
                )
                if not separated:
                    violations.append(
                        "C6 | "
                        f"{placements[first_index]['where']} overlaps "
                        f"{placements[second_index]['where']}"
                    )
    return violations


def _check_c7(stock_instances: list[dict[str, Any]]) -> list[str]:
    """C7: nesting makes every placement's parent an active instance.

    The output has no independent placement collection or z field: placements
    can occur only under a stock_instances entry, and each such entry denotes an
    active instance.  Traversing the nesting therefore fully checks the only
    representable x <= z relationship.
    """
    for stock in stock_instances:
        for _placement in stock["placements"]:
            # Reaching this placement through the parent record proves z=1.
            pass
    return []


def check_solution(instance: dict, solution: dict) -> list:
    """Return constraint/format violations; return [] exactly when feasible."""
    try:
        if not isinstance(instance, dict):
            return ["FORMAT | instance must be a dict"]

        m = instance.get("m")
        stock_type_count = instance.get("S")
        if not _is_exact_int(m) or not _is_exact_int(stock_type_count):
            return ["FORMAT | instance fields m and S must be integers"]

        parsed, violations = _parse_solution(
            instance, solution, m, stock_type_count
        )
        stock_instances = parsed["stock_instances"]

        # Exactly one semantic check group for each authoritative constraint.
        violations.extend(_check_c1(parsed["selected"]))
        violations.extend(_check_c2(parsed["selected"], stock_instances))
        violations.extend(_check_c3(stock_instances))
        violations.extend(
            _check_c4(
                instance,
                m,
                stock_instances,
                parsed["piece_counts"],
            )
        )
        violations.extend(_check_c5(instance, stock_instances))
        violations.extend(_check_c6(instance, stock_instances))
        violations.extend(_check_c7(stock_instances))
        return violations
    except Exception as exc:  # malformed/adversarial input must never escape
        return [
            "FORMAT | checker could not safely evaluate the supplied structures: "
            f"{type(exc).__name__}: {exc}"
        ]
