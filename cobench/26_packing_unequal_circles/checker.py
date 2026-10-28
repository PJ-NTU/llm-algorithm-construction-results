"""Feasibility checker for unequal-circle packing solutions."""

import math
from typing import Any, Optional, Tuple


_REL = 1e-9


def _leq(a: float, b: float) -> bool:
    """Compare continuous, computed quantities on their original linear scale."""
    if math.isnan(a) or math.isnan(b):
        return False
    if a <= b:
        return True
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _geq(a: float, b: float) -> bool:
    """Compare continuous, computed quantities on their original linear scale."""
    if math.isnan(a) or math.isnan(b):
        return False
    if a >= b:
        return True
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _finite_number(value: Any) -> Optional[float]:
    """Return a finite JSON-style number as float, or None if unreadable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_inputs(
    instance: Any, solution: Any
) -> Tuple[list, Optional[Tuple[int, float, float, float, list, list, list]]]:
    """Validate only the structure needed to evaluate the three constraints."""
    errors = []
    if not isinstance(instance, dict):
        errors.append("FORMAT | instance must be a dict")
    if not isinstance(solution, dict):
        errors.append("FORMAT | solution must be a dict")
    if errors:
        return errors, None

    missing_instance = [
        key for key in ("n", "cx", "cy", "R", "radii") if key not in instance
    ]
    missing_solution = [
        key for key in ("packing_status", "centers") if key not in solution
    ]
    if missing_instance:
        errors.append(
            "FORMAT | instance is missing fields needed by the constraints: "
            + ", ".join(missing_instance)
        )
    if missing_solution:
        errors.append(
            "FORMAT | solution is missing required fields: "
            + ", ".join(missing_solution)
        )
    if errors:
        return errors, None

    n = instance["n"]
    if isinstance(n, bool) or not isinstance(n, int):
        errors.append("FORMAT | instance.n must be an integer index count")
        return errors, None

    # range(n) is empty when n is negative; no unmentioned n > 0 premise is added.
    circle_count = n if n >= 0 else 0
    radii_raw = instance["radii"]
    statuses = solution["packing_status"]
    centers_raw = solution["centers"]

    if not isinstance(radii_raw, list):
        errors.append("FORMAT | instance.radii must be a list")
    elif len(radii_raw) < circle_count:
        errors.append(
            f"FORMAT | instance.radii has {len(radii_raw)} entries; "
            f"constraints refer to {circle_count} circle indices"
        )
    if not isinstance(statuses, list):
        errors.append("FORMAT | packing_status must be a list")
    elif len(statuses) != circle_count:
        errors.append(
            f"FORMAT | packing_status has {len(statuses)} entries; "
            f"expected {circle_count}"
        )
    if not isinstance(centers_raw, list):
        errors.append("FORMAT | centers must be a list")
    elif len(centers_raw) != circle_count:
        errors.append(
            f"FORMAT | centers has {len(centers_raw)} entries; expected {circle_count}"
        )

    cx = _finite_number(instance["cx"])
    cy = _finite_number(instance["cy"])
    container_radius = _finite_number(instance["R"])
    if cx is None:
        errors.append("FORMAT | instance.cx must be a finite number")
    if cy is None:
        errors.append("FORMAT | instance.cy must be a finite number")
    if container_radius is None:
        errors.append("FORMAT | instance.R must be a finite number")
    if errors:
        return errors, None

    radii = []
    for i in range(circle_count):
        radius = _finite_number(radii_raw[i])
        if radius is None:
            errors.append(f"FORMAT | instance.radii[{i}] must be a finite number")
        else:
            radii.append(radius)

    for i, status in enumerate(statuses):
        # This is exact: alpha is declared binary and the output shape is list[int].
        if isinstance(status, bool) or not isinstance(status, int) or status not in (0, 1):
            errors.append(f"FORMAT | packing_status[{i}] must be integer 0 or 1")

    centers = []
    for i, center in enumerate(centers_raw):
        if not isinstance(center, list) or len(center) != 2:
            errors.append(f"FORMAT | centers[{i}] must be a two-element list")
            continue
        x = _finite_number(center[0])
        y = _finite_number(center[1])
        if x is None or y is None:
            errors.append(f"FORMAT | centers[{i}] coordinates must be finite numbers")
        else:
            centers.append((x, y))

    if errors:
        return errors, None
    return errors, (n, cx, cy, container_radius, radii, statuses, centers)


def check_solution(instance: dict, solution: dict) -> list:
    """Return one message per detected format or hard-constraint violation."""
    try:
        errors, parsed = _read_inputs(instance, solution)
        if parsed is None:
            return errors

        _, cx, cy, container_radius, radii, statuses, centers = parsed
        violations = []

        # C1: Every selected circle must lie entirely inside the container.
        for i, selected in enumerate(statuses):
            if selected == 0:
                continue
            x, y = centers[i]
            center_distance = math.hypot(x - cx, y - cy)
            allowed_distance = container_radius - radii[i]
            if not _leq(center_distance, allowed_distance):
                violations.append(
                    f"C1 | selected circle {i} is not entirely inside the container: "
                    f"center distance {center_distance!r} exceeds R-R_i "
                    f"{allowed_distance!r}"
                )

        # C2: Every pair of selected circles must be non-overlapping.
        selected_indices = [i for i, selected in enumerate(statuses) if selected == 1]
        for position, i in enumerate(selected_indices):
            xi, yi = centers[i]
            for j in selected_indices[position + 1 :]:
                xj, yj = centers[j]
                center_distance = math.hypot(xi - xj, yi - yj)
                required_distance = radii[i] + radii[j]
                if not _geq(center_distance, required_distance):
                    violations.append(
                        f"C2 | selected circles {i} and {j} overlap: center distance "
                        f"{center_distance!r} is below R_i+R_j {required_distance!r}"
                    )

        # C3: Selection statuses must form a prefix in index order.
        # Integer alpha values are exactly representable, so this comparison has no tolerance.
        for i in range(len(statuses) - 1):
            if statuses[i] < statuses[i + 1]:
                violations.append(
                    f"C3 | selection order is broken between circles {i} and {i + 1}: "
                    f"alpha[{i}]={statuses[i]} < alpha[{i + 1}]={statuses[i + 1]}"
                )

        # Optional self-reported fields (including objective) are deliberately ignored.
        # The selected count is derivable exactly from the required binary status list.
        _selected_count = sum(statuses)
        del _selected_count
        return violations
    except Exception as exc:  # Defensive boundary: malformed inputs must not escape.
        return [f"FORMAT | checker could not safely read the inputs: {type(exc).__name__}"]
