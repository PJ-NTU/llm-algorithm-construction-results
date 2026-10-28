"""Feasibility checker for the 2-D guillotine cutting-stock task."""

from __future__ import annotations

import math
import re
from typing import Any


_REL = 1e-9
_MISSING = object()
_AMBIGUOUS = object()


def _leq(a: float, b: float) -> bool:
    """Relative-tolerance comparison for continuously computed quantities."""
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _geq(a: float, b: float) -> bool:
    """Relative-tolerance comparison for continuously computed quantities."""
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _instance_view(instance: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Read only the instance data needed to judge a solution.

    This deliberately imposes no positivity or other "reasonable instance"
    assumptions that are absent from the description.
    """
    if not isinstance(instance, dict):
        return None, ["FORMAT | cannot read the given instance: instance must be a dict"]

    n = instance.get("num_piece_types")
    if not _is_int(n):
        return None, ["FORMAT | cannot read the given instance: num_piece_types must be an integer"]
    type_count = max(0, n)

    sheet_length = _as_finite_number(instance.get("sheet_length"))
    sheet_width = _as_finite_number(instance.get("sheet_width"))
    if sheet_length is None or sheet_width is None:
        return None, ["FORMAT | cannot read the given instance: stock sheet dimensions must be finite numbers"]

    raw_lengths = instance.get("piece_lengths")
    raw_widths = instance.get("piece_widths")
    raw_demands = instance.get("max_quantities")
    arrays = (
        ("piece_lengths", raw_lengths),
        ("piece_widths", raw_widths),
        ("max_quantities", raw_demands),
    )
    errors: list[str] = []
    for name, values in arrays:
        if not isinstance(values, list) or len(values) < type_count:
            errors.append(
                f"FORMAT | cannot read the given instance: field={name} must cover at least type indices 0..N-1"
            )
    if errors:
        return None, errors

    lengths: list[float] = []
    widths: list[float] = []
    demands: list[int] = []
    for i in range(type_count):
        length = _as_finite_number(raw_lengths[i])
        width = _as_finite_number(raw_widths[i])
        demand = raw_demands[i]
        if length is None:
            errors.append(f"FORMAT | cannot read the given instance: piece_lengths[{i}] is not a finite number")
        else:
            lengths.append(length)
        if width is None:
            errors.append(f"FORMAT | cannot read the given instance: piece_widths[{i}] is not a finite number")
        else:
            widths.append(width)
        if not _is_int(demand):
            errors.append(f"FORMAT | cannot read the given instance: max_quantities[{i}] is not an integer")
        else:
            demands.append(demand)
    if errors:
        return None, errors

    return {
        "n": type_count,
        "sheet_length": sheet_length,
        "sheet_width": sheet_width,
        "lengths": lengths,
        "widths": widths,
        "demands": demands,
    }, []


def _prepare_solution(
    view: dict[str, Any], solution: Any
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the three required output fields before semantic checks."""
    if not isinstance(solution, dict):
        return None, ["FORMAT | solution must be a dict"]

    required = ("selection_status", "placements", "guillotine_cut_tree")
    errors = [f"FORMAT | missing required field {name}" for name in required if name not in solution]
    if errors:
        return None, errors

    # The output specification gives this top-level field shape as ``dict``.
    # Check it before considering any empty-tree marker, including when no piece
    # is selected.  Thus None and string markers can never bypass the shape rule.
    tree = solution["guillotine_cut_tree"]
    if not isinstance(tree, dict):
        errors.append("FORMAT | guillotine_cut_tree top level must be a dict")

    statuses = solution["selection_status"]
    placements = solution["placements"]
    n = view["n"]
    demands = view["demands"]

    if not isinstance(statuses, list):
        errors.append("FORMAT | selection_status must be list[list[int]]")
    elif len(statuses) != n:
        errors.append(
            f"FORMAT | selection_status type row count should be {n}, got {len(statuses)}"
        )

    if not isinstance(placements, list):
        errors.append("FORMAT | placements must be list[list[list[float]]]")
    elif len(placements) != n:
        errors.append(f"FORMAT | placements type row count should be {n}, got {len(placements)}")

    if errors:
        return None, errors

    for i in range(n):
        status_row = statuses[i]
        placement_row = placements[i]
        expected = len(range(demands[i]))

        if not isinstance(status_row, list):
            errors.append(f"FORMAT | selection_status[{i}] must be a list")
        elif len(status_row) != expected:
            errors.append(
                f"FORMAT | selection_status[{i}] should contain {expected} candidate instances, got {len(status_row)}"
            )

        if not isinstance(placement_row, list):
            errors.append(f"FORMAT | placements[{i}] must be a list")
        elif len(placement_row) != expected:
            errors.append(
                f"FORMAT | placements[{i}] should contain {expected} candidate instances, got {len(placement_row)}"
            )

        if not isinstance(status_row, list) or not isinstance(placement_row, list):
            continue
        if len(status_row) != expected or len(placement_row) != expected:
            continue

        for p in range(expected):
            selected = status_row[p]
            if not _is_int(selected) or selected not in (0, 1):
                errors.append(f"FORMAT | selection_status[{i}][{p}] must be the integer 0 or 1")

            coordinate = placement_row[p]
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                errors.append(f"FORMAT | placements[{i}][{p}] must be a coordinate list of length 2")
                continue
            x = _as_finite_number(coordinate[0])
            y = _as_finite_number(coordinate[1])
            if x is None or y is None:
                errors.append(f"FORMAT | placements[{i}][{p}] must have two finite coordinates")
                continue

            # The specified unselected marker is direct output data, not a
            # computed float, so this exact comparison must not gain tolerance.
            if selected == 0 and (x != -1.0 or y != -1.0):
                errors.append(
                    f"FORMAT | unselected instance placements[{i}][{p}] must be filled with [-1.0, -1.0]"
                )

    if errors:
        return None, errors
    return {
        "selection_status": statuses,
        "placements": placements,
        "guillotine_cut_tree": tree,
    }, []


def _selected_indices(prepared: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (i, p)
        for i, row in enumerate(prepared["selection_status"])
        for p, selected in enumerate(row)
        if selected == 1
    ]


def _computed_box(
    view: dict[str, Any], prepared: dict[str, Any], index: tuple[int, int]
) -> tuple[float, float, float, float] | None:
    i, p = index
    x = float(prepared["placements"][i][p][0])
    y = float(prepared["placements"][i][p][1])
    right = x + view["lengths"][i]
    top = y + view["widths"][i]
    if not all(math.isfinite(value) for value in (x, y, right, top)):
        return None
    return x, y, right, top


def _check_c1_within_sheet(
    view: dict[str, Any], prepared: dict[str, Any]
) -> list[str]:
    """C1: every selected piece must lie completely inside the sheet."""
    violations: list[str] = []
    length = view["sheet_length"]
    width = view["sheet_width"]
    for index in _selected_indices(prepared):
        box = _computed_box(view, prepared, index)
        if box is None:
            violations.append(f"C1 | Selected instance {index} has non-finite recomputed bounds")
            continue
        x, y, right, top = box
        # Coordinates and coordinate-plus-size values are continuous floating
        # quantities, so comparisons use the required linear relative tolerance.
        if not (_geq(x, 0.0) and _geq(y, 0.0) and _leq(right, length) and _leq(top, width)):
            violations.append(
                f"C1 | Selected instance {index} has rectangle bounds {(x, y, right, top)} not fully within stock sheet "
                f"[0, {length}] × [0, {width}] bounds"
            )
    return violations


def _check_c2_no_overlap(
    view: dict[str, Any], prepared: dict[str, Any]
) -> list[str]:
    """C2: every pair of selected pieces must be separated on one axis."""
    violations: list[str] = []
    selected = _selected_indices(prepared)
    boxes = {index: _computed_box(view, prepared, index) for index in selected}

    for position, first in enumerate(selected):
        first_box = boxes[first]
        for second in selected[position + 1 :]:
            second_box = boxes[second]
            if first_box is None or second_box is None:
                violations.append(
                    f"C2 | Cannot determine with finite recomputed bounds whether selected instances {first} and {second} overlap"
                )
                continue
            ax, ay, aright, atop = first_box
            bx, by, bright, btop = second_box
            separated = (
                _leq(aright, bx)
                or _leq(bright, ax)
                or _leq(atop, by)
                or _leq(btop, ay)
            )
            if not separated:
                violations.append(f"C2 | Selected instances {first} and {second} overlap in their interiors")
    return violations


def _normal_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\s_-]+", "", value).lower()


def _field(node: dict[Any, Any], *aliases: str) -> Any:
    """Find a uniquely named field without imposing case/underscore styling."""
    wanted = {_normal_key(alias) for alias in aliases}
    matches = [value for key, value in node.items() if _normal_key(key) in wanted]
    if not matches:
        return _MISSING
    if len(matches) > 1:
        return _AMBIGUOUS
    return matches[0]


def _empty_marker(node: Any) -> bool:
    """Recognize blank child regions; their concrete encoding is unspecified."""
    if node is None:
        return True
    if isinstance(node, str):
        return _normal_key(node) in {"empty", "blank"}
    if not isinstance(node, dict):
        return False
    if not node:
        return True
    node_type = _field(node, "type", "node_type")
    if isinstance(node_type, str) and _normal_key(node_type) in {"empty", "blank"}:
        return True
    is_empty = _field(node, "is_empty", "empty")
    return is_empty is True


def _parse_index_component(value: Any) -> int | None:
    if _is_int(value):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    return None


def _parse_piece_index(raw: Any) -> tuple[int, int] | None:
    """Parse any unambiguous two-index representation, including leading zeros."""
    parts: list[Any]
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        parts = list(raw)
    elif isinstance(raw, str):
        numbers = re.findall(r"[+-]?\d+", raw)
        if len(numbers) != 2:
            return None
        parts = numbers
    elif isinstance(raw, dict):
        type_index = _field(raw, "i", "type", "piece_type", "piece_type_index")
        copy_index = _field(raw, "p", "instance", "instance_index", "copy", "copy_index")
        if type_index in (_MISSING, _AMBIGUOUS) or copy_index in (_MISSING, _AMBIGUOUS):
            return None
        parts = [type_index, copy_index]
    else:
        return None

    first = _parse_index_component(parts[0])
    second = _parse_index_component(parts[1])
    return None if first is None or second is None else (first, second)


def _tree_kind(node: dict[Any, Any]) -> str | None:
    raw_type = _field(node, "type", "node_type")
    if raw_type is _AMBIGUOUS:
        return None
    if isinstance(raw_type, str):
        normalized = _normal_key(raw_type)
        if normalized in {"empty", "blank"}:
            return "empty"
        if normalized in {"leaf", "piece", "pieceleaf"}:
            return "leaf"
        if normalized in {"root", "node", "cut", "internal", "branch", "cutnode"}:
            return "cut"
        return None
    if raw_type is not _MISSING:
        return None

    piece = _field(node, "piece_index", "piece", "piece_id", "index")
    direction = _field(node, "cut_direction", "direction", "cut_type", "orientation")
    coordinate = _field(node, "cut_coordinate", "coordinate", "cut_position", "position")
    if piece not in (_MISSING, _AMBIGUOUS):
        return "leaf"
    if direction not in (_MISSING, _AMBIGUOUS) or coordinate not in (_MISSING, _AMBIGUOUS):
        return "cut"
    return "empty" if not node else None


def _check_c3_guillotine(
    view: dict[str, Any], prepared: dict[str, Any]
) -> list[str]:
    """C3: validate a complete recursive guillotine partition of the sheet."""
    violations: list[str] = []
    selected = set(_selected_indices(prepared))
    piece_leaves: set[tuple[int, int]] = set()
    visited_nodes: set[int] = set()
    root = prepared["guillotine_cut_tree"]
    stack: list[tuple[Any, tuple[float, float, float, float], str]] = [
        (root, (0.0, 0.0, view["sheet_length"], view["sheet_width"]), "root")
    ]

    while stack:
        node, region, path = stack.pop()
        if _empty_marker(node):
            continue
        if not isinstance(node, dict):
            violations.append(f"C3 | Cut-tree node {path} is neither a dict node nor a recognizable empty marker")
            continue

        node_id = id(node)
        if node_id in visited_nodes:
            violations.append(f"C3 | Cut-tree node {path} forms a cycle or reuses the same subtree object")
            continue
        visited_nodes.add(node_id)

        kind = _tree_kind(node)
        if kind == "empty":
            continue
        if kind is None:
            violations.append(f"C3 | Cut-tree node {path} has a type that cannot be uniquely resolved")
            continue

        x0, y0, x1, y1 = region
        if kind == "leaf":
            raw_index = _field(node, "piece_index", "piece", "piece_id", "index")
            if raw_index in (_MISSING, _AMBIGUOUS):
                violations.append(f"C3 | Leaf node {path} lacks a uniquely resolvable piece_index")
                continue
            index = _parse_piece_index(raw_index)
            if index is None:
                violations.append(f"C3 | Leaf node {path} has a piece_index that cannot be uniquely resolved into two integer indices")
                continue
            if index not in selected:
                violations.append(f"C3 | Leaf node {path} references an unselected or nonexistent instance {index}")
                continue
            if index in piece_leaves:
                violations.append(f"C3 | Selected instance {index} appears more than once among cut-tree leaf nodes")
                continue
            piece_leaves.add(index)

            box = _computed_box(view, prepared, index)
            if box is None:
                violations.append(f"C3 | Leaf node {path} / instance {index} has no finite recomputed bounds")
                continue
            px0, py0, px1, py1 = box
            if not (
                _geq(px0, x0)
                and _geq(py0, y0)
                and _leq(px1, x1)
                and _leq(py1, y1)
            ):
                violations.append(
                    f"C3 | Leaf node {path} / instance {index} does not lie fully within ancestor-cut region {region} bounds"
                )
            continue

        raw_direction = _field(
            node, "cut_direction", "direction", "cut_type", "orientation"
        )
        raw_coordinate = _field(
            node, "cut_coordinate", "coordinate", "cut_position", "position"
        )
        if raw_direction in (_MISSING, _AMBIGUOUS) or raw_coordinate in (
            _MISSING,
            _AMBIGUOUS,
        ):
            violations.append(f"C3 | Cut node {path} lacks a uniquely resolvable direction or coordinate")
            continue
        if not isinstance(raw_direction, str):
            violations.append(f"C3 | Cut node {path} must have cut_direction horizontal or vertical")
            continue
        direction = _normal_key(raw_direction)
        coordinate = _as_finite_number(raw_coordinate)
        if coordinate is None:
            violations.append(f"C3 | Cut node {path} must have a finite cut_coordinate")
            continue

        if direction in {"vertical", "v"}:
            # A cut must create two smaller rectangles.  These are direct cut
            # coordinates, so an exact strict interior comparison is appropriate.
            if not (x0 < coordinate < x1):
                violations.append(
                    f"C3 | Vertical cut {path} has coordinate {coordinate} not strictly inside the current region ({x0}, {x1})"
                )
                continue
            first = _field(node, "left_child", "left")
            second = _field(node, "right_child", "right")
            first_region = (x0, y0, coordinate, y1)
            second_region = (coordinate, y0, x1, y1)
            first_name, second_name = "left", "right"
        elif direction in {"horizontal", "h"}:
            if not (y0 < coordinate < y1):
                violations.append(
                    f"C3 | Horizontal cut {path} has coordinate {coordinate} not strictly inside the current region ({y0}, {y1})"
                )
                continue
            first = _field(node, "bottom_child", "bottom")
            second = _field(node, "top_child", "top")
            first_region = (x0, y0, x1, coordinate)
            second_region = (x0, coordinate, x1, y1)
            first_name, second_name = "bottom", "top"
        else:
            violations.append(
                f"C3 | Cut node {path} direction {raw_direction!r} is not horizontal/vertical"
            )
            continue

        if first in (_MISSING, _AMBIGUOUS) or second in (_MISSING, _AMBIGUOUS):
            violations.append(f"C3 | Cut node {path} must have two uniquely resolvable subregions")
            continue
        stack.append((second, second_region, f"{path}.{second_name}"))
        stack.append((first, first_region, f"{path}.{first_name}"))

    missing = sorted(selected - piece_leaves)
    if missing:
        violations.append(f"C3 | Cut-tree leaves do not fully cover the selected instances; missing={missing}")
    return violations


def _check_c4_demand(
    view: dict[str, Any], prepared: dict[str, Any]
) -> list[str]:
    """C4: the exact integer selected count may not exceed each demand."""
    violations: list[str] = []
    for i, row in enumerate(prepared["selection_status"]):
        selected_count = sum(row)
        demand = view["demands"][i]
        # Both values are exact integers; tolerance here would admit a real excess.
        if selected_count > demand:
            violations.append(
                f"C4 | Type {i} has {selected_count} selected instances, exceeding the maximum demand {demand}"
            )
    return violations


def check_solution(instance: dict, solution: dict) -> list:
    """Return semantic/format violations; an empty list means feasible."""
    try:
        view, instance_errors = _instance_view(instance)
        if instance_errors:
            return instance_errors
        assert view is not None

        prepared, format_errors = _prepare_solution(view, solution)
        if format_errors:
            return format_errors
        assert prepared is not None

        violations: list[str] = []
        # Exactly one business-checking section for each authoritative constraint.
        violations.extend(_check_c1_within_sheet(view, prepared))
        violations.extend(_check_c2_no_overlap(view, prepared))
        violations.extend(_check_c3_guillotine(view, prepared))
        violations.extend(_check_c4_demand(view, prepared))
        return violations
    except Exception as exc:  # Defensive boundary: malformed inputs must not escape.
        return [f"FORMAT | checker cannot safely parse this solution, error={type(exc).__name__}: {exc}"]
