"""Feasibility checker for the unequal-circle packing problem."""

import math


_REL = 1e-9
_REQUIRED_SOLUTION_FIELDS = (
    "packing_status",
    "center_x_coordinates",
    "center_y_coordinates",
)


def _leq(a, b):
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _geq(a, b):
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _read_inputs(instance, solution, violations):
    """Validate only the structure needed to evaluate the three constraints."""
    if not isinstance(instance, dict):
        violations.append("FORMAT | instance must be a dict")
        return None
    if not isinstance(solution, dict):
        violations.append("FORMAT | solution must be a dict")
        return None

    missing_instance = [
        key for key in ("n", "cx", "cy", "R", "radii") if key not in instance
    ]
    if missing_instance:
        violations.append(
            "FORMAT | instance is missing fields needed for checking: "
            + ", ".join(missing_instance)
        )
        return None

    n = instance["n"]
    if not isinstance(n, int) or isinstance(n, bool):
        violations.append("FORMAT | instance field 'n' must be an int")
        return None

    # range(n) is empty for every n <= 0.  In particular, n == 0 is accepted;
    # the checker does not invent an N > 0 input-domain requirement.
    circle_count = max(0, n)
    cx = instance["cx"]
    cy = instance["cy"]
    container_radius = instance["R"]
    for key, value in (("cx", cx), ("cy", cy), ("R", container_radius)):
        if not _is_finite_number(value):
            violations.append(
                f"FORMAT | instance field '{key}' must be a finite number so the constraints can be evaluated"
            )

    radii = instance["radii"]
    if not isinstance(radii, list):
        violations.append("FORMAT | instance field 'radii' must be a list")
    elif len(radii) < circle_count:
        violations.append(
            "FORMAT | instance field 'radii' has fewer entries than the index range 0..n-1"
        )
    elif any(not _is_finite_number(radius) for radius in radii[:circle_count]):
        violations.append(
            "FORMAT | each radius used by an index in 0..n-1 must be a finite number"
        )

    missing_solution = [
        key for key in _REQUIRED_SOLUTION_FIELDS if key not in solution
    ]
    if missing_solution:
        violations.append(
            "FORMAT | solution is missing required fields: "
            + ", ".join(missing_solution)
        )

    if violations:
        return None

    status = solution["packing_status"]
    xs = solution["center_x_coordinates"]
    ys = solution["center_y_coordinates"]
    for key, values in (
        ("packing_status", status),
        ("center_x_coordinates", xs),
        ("center_y_coordinates", ys),
    ):
        if not isinstance(values, list):
            violations.append(f"FORMAT | solution field '{key}' must be a list")
        elif len(values) != circle_count:
            violations.append(
                f"FORMAT | solution field '{key}' must contain exactly {circle_count} entries"
            )

    if violations:
        return None

    bad_status = [
        i
        for i, value in enumerate(status)
        if not isinstance(value, int)
        or isinstance(value, bool)
        or value not in (0, 1)
    ]
    if bad_status:
        violations.append(
            "FORMAT | packing_status entries must be integer 0 or 1; invalid indices: "
            + ", ".join(map(str, bad_status))
        )

    for key, values in (
        ("center_x_coordinates", xs),
        ("center_y_coordinates", ys),
    ):
        bad_coordinates = [
            i for i, value in enumerate(values) if not _is_finite_number(value)
        ]
        if bad_coordinates:
            violations.append(
                f"FORMAT | {key} entries must be finite numbers; invalid indices: "
                + ", ".join(map(str, bad_coordinates))
            )

    if violations:
        return None
    return circle_count, cx, cy, container_radius, radii, status, xs, ys


def check_solution(instance: dict, solution: dict) -> list:
    """Return constraint violations; an empty list means the solution is feasible."""
    violations = []
    try:
        parsed = _read_inputs(instance, solution, violations)
        if parsed is None:
            return violations

        n, cx, cy, container_radius, radii, status, xs, ys = parsed

        # C1 | Every circle must satisfy the stated squared containment formula.
        # Work in linear distance so the relative tolerance is not scaled by a
        # squared quantity.  abs() preserves the formula when R - radii[i] < 0.
        for i in range(n):
            center_distance = math.hypot(xs[i] - cx, ys[i] - cy)
            allowed_distance = abs(status[i] * (container_radius - radii[i]))
            if not _leq(center_distance, allowed_distance):
                violations.append(
                    f"C1 | circle {i} has center distance {center_distance!r}, "
                    f"exceeding the formula's linearized limit {allowed_distance!r}"
                )

        # C2 | Only a pair with both binary indicators equal to 1 has a
        # positive right-hand side in the stated non-overlap formula.
        for i in range(n):
            if status[i] != 1:
                continue
            for j in range(i + 1, n):
                if status[j] != 1:
                    continue
                center_distance = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
                required_distance = abs(radii[i] + radii[j])
                if not _geq(center_distance, required_distance):
                    violations.append(
                        f"C2 | packed circles {i} and {j} have center distance "
                        f"{center_distance!r}, below the formula's linearized "
                        f"minimum {required_distance!r}"
                    )

        # C3 | These are direct values, not computed floating-point quantities:
        # an unpacked circle's two coordinates must equal cx and cy exactly.
        for i in range(n):
            if status[i] == 0 and (xs[i] != cx or ys[i] != cy):
                violations.append(
                    f"C3 | unpacked circle {i} must have center ({cx!r}, {cy!r}), "
                    f"but solution gives ({xs[i]!r}, {ys[i]!r})"
                )

        return violations
    except Exception as exc:
        # Malformed/adversarial objects must produce a violation, never escape.
        return violations + [
            f"FORMAT | checker could not safely evaluate the supplied data: {type(exc).__name__}: {exc}"
        ]
