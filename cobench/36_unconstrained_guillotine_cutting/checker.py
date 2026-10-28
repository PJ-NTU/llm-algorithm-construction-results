"""Feasibility checker for the rectangle-packing solution format in MAT_s4.md."""

import math
from numbers import Integral, Real


_REL = 1e-9
_REQUIRED = (
    "selection",
    "bottom_left_x",
    "bottom_left_y",
    "left_relations",
    "below_relations",
)


def _leq(a, b):
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _geq(a, b):
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _is_real(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _read_instance(instance):
    """Return normalized instance data, or format errors if it cannot be read safely."""
    if not isinstance(instance, dict):
        return None, ["FORMAT | instance must be a dictionary"]

    names = (
        "num_parts",
        "sheet_width",
        "sheet_height",
        "part_widths",
        "part_heights",
        "part_values",
        "big_M",
    )
    missing = [name for name in names if name not in instance]
    if missing:
        return None, ["FORMAT | instance is missing field: " + name for name in missing]

    n = instance["num_parts"]
    if not isinstance(n, Integral) or isinstance(n, bool) or n < 0:
        return None, ["FORMAT | instance num_parts must be a non-negative integer"]

    errors = []
    scalars = ("sheet_width", "sheet_height", "big_M")
    for name in scalars:
        if not _is_real(instance[name]):
            errors.append(f"FORMAT | instance {name} must be a finite number")

    vectors = ("part_widths", "part_heights", "part_values")
    for name in vectors:
        value = instance[name]
        if not isinstance(value, list) or len(value) != n:
            errors.append(f"FORMAT | instance {name} must be a list of length {n}")
        elif any(not _is_real(item) for item in value):
            errors.append(f"FORMAT | instance {name} must contain only finite numbers")

    if errors:
        return None, errors
    return {
        "n": int(n),
        "w": instance["part_widths"],
        "h": instance["part_heights"],
        "v": instance["part_values"],
        "W": instance["sheet_width"],
        "H": instance["sheet_height"],
        "M": instance["big_M"],
    }, []


def _read_solution(solution, n):
    """Validate only the structure needed to evaluate the authoritative variables."""
    if not isinstance(solution, dict):
        return None, ["FORMAT | solution must be a dictionary"]

    errors = []
    for name in _REQUIRED:
        if name not in solution:
            errors.append(f"FORMAT | missing required solution field: {name}")
    if errors:
        return None, errors

    selection = solution["selection"]
    x = solution["bottom_left_x"]
    y = solution["bottom_left_y"]
    left = solution["left_relations"]
    below = solution["below_relations"]

    if not isinstance(selection, list) or len(selection) != n:
        errors.append(f"FORMAT | selection must be a list of length {n}")
    elif any(not isinstance(value, Integral) or isinstance(value, bool) or value not in (0, 1)
             for value in selection):
        errors.append("FORMAT | selection entries must be integers 0 or 1")

    for name, vector in (("bottom_left_x", x), ("bottom_left_y", y)):
        if not isinstance(vector, list) or len(vector) != n:
            errors.append(f"FORMAT | {name} must be a list of length {n}")
        elif any(not _is_real(value) for value in vector):
            errors.append(f"FORMAT | {name} entries must be finite numbers")

    for name, matrix in (("left_relations", left), ("below_relations", below)):
        if not isinstance(matrix, list) or len(matrix) != n:
            errors.append(f"FORMAT | {name} must be an {n}x{n} list matrix")
            continue
        if any(not isinstance(row, list) or len(row) != n for row in matrix):
            errors.append(f"FORMAT | {name} must be an {n}x{n} list matrix")
            continue
        if any(not isinstance(value, Integral) or isinstance(value, bool) or value not in (0, 1)
               for row in matrix for value in row):
            errors.append(f"FORMAT | {name} entries must be integers 0 or 1")
            continue
        for i in range(n):
            if matrix[i][i] != 0:
                errors.append(f"FORMAT | {name}[{i}][{i}] must be 0")

    if errors:
        return None, errors
    return {
        "select": selection,
        "x": x,
        "y": y,
        "left": left,
        "below": below,
    }, []


def _check_coordinate_domains(data, variables):
    """Check the x/y lower bounds stated in the decision-variable definitions."""
    violations = []
    for i in range(data["n"]):
        if not _geq(variables["x"][i], 0.0):
            violations.append(f"DOMAIN | bottom_left_x[{i}] must be at least 0")
        if not _geq(variables["y"][i], 0.0):
            violations.append(f"DOMAIN | bottom_left_y[{i}] must be at least 0")
    return violations


def _check_c1(data, variables):
    # C1 | Every selected pair must have at least one declared separation direction.
    violations = []
    s, left, below = variables["select"], variables["left"], variables["below"]
    for i in range(data["n"]):
        for j in range(i + 1, data["n"]):
            separation = left[i][j] + left[j][i] + below[i][j] + below[j][i]
            required = s[i] + s[j] - 1
            if separation < required:  # exact integer comparison
                violations.append(f"C1 | parts {i} and {j} lack a required separation relation")
    return violations


def _check_c2(data, variables):
    # C2 | A declared left relation must satisfy the authoritative big-M inequality.
    violations = []
    x, left = variables["x"], variables["left"]
    for i in range(data["n"]):
        for j in range(data["n"]):
            if i == j:
                continue
            lhs = x[i] + data["w"][i]
            rhs = x[j] + data["M"] * (1 - left[i][j])
            if not _leq(lhs, rhs):
                violations.append(f"C2 | horizontal relation inequality fails for ({i}, {j})")
    return violations


def _check_c3(data, variables):
    # C3 | A declared below relation must satisfy the authoritative big-M inequality.
    violations = []
    y, below = variables["y"], variables["below"]
    for i in range(data["n"]):
        for j in range(data["n"]):
            if i == j:
                continue
            lhs = y[i] + data["h"][i]
            rhs = y[j] + data["M"] * (1 - below[i][j])
            if not _leq(lhs, rhs):
                violations.append(f"C3 | vertical relation inequality fails for ({i}, {j})")
    return violations


def _check_c4(data, variables):
    # C4 | Opposite horizontal relations are mutually exclusive.
    violations = []
    left = variables["left"]
    for i in range(data["n"]):
        for j in range(i + 1, data["n"]):
            if left[i][j] + left[j][i] > 1:  # exact integer comparison
                violations.append(f"C4 | both horizontal directions are set for parts {i} and {j}")
    return violations


def _check_c5(data, variables):
    # C5 | Opposite vertical relations are mutually exclusive.
    violations = []
    below = variables["below"]
    for i in range(data["n"]):
        for j in range(i + 1, data["n"]):
            if below[i][j] + below[j][i] > 1:  # exact integer comparison
                violations.append(f"C5 | both vertical directions are set for parts {i} and {j}")
    return violations


def _check_c6(data, variables):
    # C6 | A relation may be 1 only when both endpoint parts are selected.
    violations = []
    s, left, below = variables["select"], variables["left"], variables["below"]
    for i in range(data["n"]):
        for j in range(data["n"]):
            if i == j:
                continue
            if left[i][j] > s[i] or left[i][j] > s[j]:
                violations.append(f"C6 | left_relations[{i}][{j}] requires both parts selected")
            if below[i][j] > s[i] or below[i][j] > s[j]:
                violations.append(f"C6 | below_relations[{i}][{j}] requires both parts selected")
    return violations


def _check_c7(data, variables):
    # C7 | Coordinates are bounded by remaining sheet extent times selection.
    violations = []
    s, x, y = variables["select"], variables["x"], variables["y"]
    for i in range(data["n"]):
        x_upper = (data["W"] - data["w"][i]) * s[i]
        y_upper = (data["H"] - data["h"][i]) * s[i]
        if not _leq(x[i], x_upper):
            violations.append(f"C7 | bottom_left_x[{i}] exceeds (W - w_i) * selection[{i}]")
        if not _leq(y[i], y_upper):
            violations.append(f"C7 | bottom_left_y[{i}] exceeds (H - h_i) * selection[{i}]")
    return violations


def check_solution(instance: dict, solution: dict) -> list:
    """Return all detected format, domain, and C1-C7 feasibility violations."""
    try:
        data, errors = _read_instance(instance)
        if errors:
            return errors
        variables, errors = _read_solution(solution, data["n"])
        if errors:
            return errors

        # Recompute the optional derived objective; never trust solution['objective'].
        _objective = sum(data["v"][i] * variables["select"][i] for i in range(data["n"]))
        del _objective

        violations = _check_coordinate_domains(data, variables)
        # Exactly one semantic check function corresponds to each authoritative constraint.
        for check in (_check_c1, _check_c2, _check_c3, _check_c4,
                      _check_c5, _check_c6, _check_c7):
            violations.extend(check(data, variables))
        return violations
    except Exception as exc:
        # A malformed or adversarial Python object must produce a violation, never escape.
        return [f"FORMAT | checker could not safely read the supplied data: {type(exc).__name__}"]
