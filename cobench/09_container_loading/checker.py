"""Feasibility checker for the three-dimensional container-loading task."""

import math
import re
from itertools import combinations
from numbers import Real


_REL = 1e-9
_REQUIRED_FIELDS = ("placement_status", "coordinates", "orientations")
_INTEGER_TEXT = re.compile(r"^[+-]?\d+$")


def _leq(a, b):
    """Relative-tolerance comparison for continuously computed linear values."""
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _geq(a, b):
    """Relative-tolerance comparison for continuously computed linear values."""
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _is_finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


def _as_exact_integer(value):
    """Return the exactly represented integer value, or None."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    if not math.isfinite(float(value)):
        return None
    integer = int(value)
    return integer if value == integer else None


def _parse_forbidden_orientation(value):
    """Parse only representations that identify one integer unambiguously."""
    integer = _as_exact_integer(value)
    if integer is not None:
        return integer
    if isinstance(value, str):
        stripped = value.strip()
        if _INTEGER_TEXT.fullmatch(stripped):
            return int(stripped)
    return None


def _validate_solution_format(instance, solution):
    """Safely read the three required solution fields; optional fields are ignored."""
    errors = []
    if not isinstance(instance, dict):
        return ["FORMAT | instance must be a dict; cannot read problem data"], None
    if not isinstance(solution, dict):
        return ["FORMAT | solution must be a dict"], None

    missing = [key for key in _REQUIRED_FIELDS if key not in solution]
    if missing:
        errors.append(
            "FORMAT | missing required fields: " + ", ".join(missing)
        )

    m = _as_exact_integer(instance.get("total_box_instances"))
    if m is None:
        errors.append(
            "FORMAT | instance.total_box_instances must exactly represent an integer, "
            "otherwise the completeness of the solution cannot be verified"
        )

    if errors:
        return errors, None

    statuses = solution["placement_status"]
    coordinates = solution["coordinates"]
    orientations = solution["orientations"]
    for key, value in (
        ("placement_status", statuses),
        ("coordinates", coordinates),
        ("orientations", orientations),
    ):
        if not isinstance(value, (list, tuple)):
            errors.append(f"FORMAT | {key} must be a list")

    if errors:
        return errors, None

    # For a degenerate negative M, the described index set {0..M-1} is empty.
    # This avoids inventing a separate M > 0 instance-quality requirement.
    expected_count = m if m >= 0 else 0
    for key, value in (
        ("placement_status", statuses),
        ("coordinates", coordinates),
        ("orientations", orientations),
    ):
        if len(value) != expected_count:
            errors.append(
                f"FORMAT | {key} length should be {expected_count}, got {len(value)}"
            )

    if errors:
        return errors, None

    normalized_statuses = []
    normalized_coordinates = []
    normalized_orientations = []
    for j in range(expected_count):
        status = _as_exact_integer(statuses[j])
        if status not in (0, 1):
            errors.append(
                f"FORMAT | placement_status[{j}] must exactly represent the binary value 0 or 1"
            )
        else:
            normalized_statuses.append(status)

        coordinate = coordinates[j]
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 3:
            errors.append(
                f"FORMAT | coordinates[{j}] must be [x, y, z] with exactly three numbers"
            )
        elif not all(_is_finite_number(value) for value in coordinate):
            errors.append(
                f"FORMAT | coordinates[{j}] must have all three coordinates finite"
            )
        else:
            normalized_coordinates.append(tuple(float(value) for value in coordinate))

        orientation = _as_exact_integer(orientations[j])
        if orientation is None:
            errors.append(
                f"FORMAT | orientations[{j}] must exactly represent an integer index"
            )
        else:
            normalized_orientations.append(orientation)

    if errors:
        return errors, None

    return [], (
        expected_count,
        normalized_statuses,
        normalized_coordinates,
        normalized_orientations,
    )


def _container_limits(instance):
    keys = ("container_length", "container_width", "container_height")
    values = []
    for key in keys:
        value = instance.get(key)
        if not _is_finite_number(value):
            return None, f"instance.{key} is not a finite number, cannot compute container bounds"
        values.append(float(value))
    return tuple(values), None


def _selected_dimensions(instance, j, orientation):
    keys = (
        "boxes.rotated_lengths",
        "boxes.rotated_widths",
        "boxes.rotated_heights",
    )
    values = []
    for key in keys:
        matrix = instance.get(key)
        if not isinstance(matrix, (list, tuple)) or j >= len(matrix):
            return None, f"instance.{key} is missing box {j} rotated dimensions"
        row = matrix[j]
        if not isinstance(row, (list, tuple)) or orientation >= len(row):
            return None, (
                f"instance.{key}[{j}] is missing rotated dimensions for orientation {orientation} rotated dimensions"
            )
        value = row[orientation]
        if not _is_finite_number(value):
            return None, (
                f"instance.{key}[{j}][{orientation}] is not a finite number,"
                "cannot compute box boundaries"
            )
        values.append(float(value))
    return tuple(values), None


def _check_c1(instance, statuses, coordinates, orientations):
    """C1: selected boxes are inside; unselected coordinate placeholders are zero."""
    violations = []
    selected_exists = any(status == 1 for status in statuses)
    limits, limit_error = _container_limits(instance) if selected_exists else (None, None)
    if limit_error:
        violations.append(f"C1 | {limit_error}")

    axis_names = ("x", "y", "z")
    for j, status in enumerate(statuses):
        coordinate = coordinates[j]
        if status == 0:
            # The output completeness rule gives an exact zero placeholder.
            if any(value != 0.0 for value in coordinate):
                violations.append(
                    f"C1 | Unselected box {j} must have coordinates exactly [0.0, 0.0, 0.0],"
                    f"got {list(coordinate)}"
                )
            continue

        for axis, start in zip(axis_names, coordinate):
            if not _geq(start, 0.0):
                violations.append(
                    f"C1 | Box {j} has {axis} start {start} less than 0"
                )

        orientation = orientations[j]
        if orientation < 0 or orientation > 5 or limits is None:
            # C3 reports an invalid orientation; the limit error was reported above.
            continue
        dimensions, dimension_error = _selected_dimensions(instance, j, orientation)
        if dimension_error:
            violations.append(f"C1 | {dimension_error}")
            continue
        for axis, start, extent, limit in zip(
            axis_names, coordinate, dimensions, limits
        ):
            endpoint = start + extent
            if not _leq(endpoint, limit):
                violations.append(
                    f"C1 | Box {j} has {axis} endpoint {endpoint} exceeding the container upper bound {limit}"
                )
    return violations


def _check_c2(instance, statuses, coordinates, orientations):
    """C2: each pair of selected boxes has separation on at least one axis."""
    violations = []
    dimensions = {}
    for j, status in enumerate(statuses):
        orientation = orientations[j]
        if status != 1 or orientation < 0 or orientation > 5:
            continue
        box_dimensions, error = _selected_dimensions(instance, j, orientation)
        if error:
            violations.append(f"C2 | {error}")
        else:
            dimensions[j] = box_dimensions

    for j, i in combinations(dimensions, 2):
        xj, yj, zj = coordinates[j]
        xi, yi, zi = coordinates[i]
        dxj, dyj, dzj = dimensions[j]
        dxi, dyi, dzi = dimensions[i]
        separated = (
            _leq(xj + dxj, xi)
            or _leq(xi + dxi, xj)
            or _leq(yj + dyj, yi)
            or _leq(yi + dyi, yj)
            or _leq(zj + dzj, zi)
            or _leq(zi + dzi, zj)
        )
        if not separated:
            violations.append(
                f"C2 | Selected boxes {j} and {i} are not separated along any of the x, y, z axes,"
                "hence a spatial overlap occurs"
            )
    return violations


def _check_c3(statuses, orientations):
    """C3: the compressed orientation field represents sum_k o[j][k] = s[j]."""
    violations = []
    for j, (status, orientation) in enumerate(zip(statuses, orientations)):
        if status == 1 and not 0 <= orientation <= 5:
            violations.append(
                f"C3 | Selected box {j} must give exactly one orientation index in 0..5,"
                f"got {orientation}"
            )
        elif status == 0 and orientation not in (-1, 0):
            violations.append(
                f"C3 | Unselected box {j} may only use -1 or 0 as orientation placeholder,"
                f"got {orientation}"
            )
    return violations


def _check_c4(instance, statuses, orientations):
    """C4: a selected box may not use an instance-declared forbidden orientation."""
    violations = []
    forbidden_rows = instance.get("boxes.forbidden_orientations")
    for j, status in enumerate(statuses):
        orientation = orientations[j]
        if status != 1 or orientation < 0 or orientation > 5:
            continue
        if not isinstance(forbidden_rows, (list, tuple)) or j >= len(forbidden_rows):
            violations.append(
                f"C4 | instance.boxes.forbidden_orientations is missing box {j} record,"
                "cannot verify vertical constraints"
            )
            continue
        row = forbidden_rows[j]
        if not isinstance(row, (list, tuple)):
            violations.append(
                f"C4 | instance.boxes.forbidden_orientations[{j}] is not a list,"
                "cannot verify vertical constraints"
            )
            continue
        forbidden = {
            parsed
            for value in row
            if (parsed := _parse_forbidden_orientation(value)) is not None
        }
        if orientation in forbidden:
            violations.append(
                f"C4 | Box {j} chose orientation forbidden by the vertical-orientation constraint: {orientation}"
            )
    return violations


def check_solution(instance: dict, solution: dict) -> list:
    """Return concrete violations; an empty list means the solution is feasible."""
    try:
        format_errors, parsed = _validate_solution_format(instance, solution)
        if format_errors:
            return format_errors

        _, statuses, coordinates, orientations = parsed
        violations = []
        violations.extend(_check_c1(instance, statuses, coordinates, orientations))
        violations.extend(_check_c2(instance, statuses, coordinates, orientations))
        violations.extend(_check_c3(statuses, orientations))
        violations.extend(_check_c4(instance, statuses, orientations))
        return violations
    except Exception as exc:  # Last-resort guard for adversarial/malformed input.
        return [
            "FORMAT | checker failed to safely parse input: "
            f"{type(exc).__name__}: {exc}"
        ]
