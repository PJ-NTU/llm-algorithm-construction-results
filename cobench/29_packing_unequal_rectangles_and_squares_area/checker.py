"""Check the agreed packing model and its input/output contracts.

An empty violation list means feasible, not necessarily optimal. Geometry is
compared exactly using the supplied Python numbers converted to Fraction;
no feasibility tolerance is added. In particular, a float retains its exact
represented binary value. C1 defines dimensions rather than extra decisions.
"""

import math
from fractions import Fraction


def _is_real(value):
    return type(value) is int or (
        type(value) is float and math.isfinite(value)
    )


def _is_binary(value):
    return type(value) is int and value in (0, 1)


def check(instance: dict, solution: dict) -> list:
    """Return structural, input-domain, and C1--C6 violations as strings."""
    errors = []
    if not isinstance(instance, dict):
        errors.append("Input: instance must be a dictionary.")
    if not isinstance(solution, dict):
        errors.append("Structure: solution must be a dictionary.")
    if errors:
        return errors

    input_keys = ("n", "cx", "cy", "R", "items", "rotation_enabled")
    solution_keys = ("alpha", "x", "y", "r")
    for key in input_keys:
        if key not in instance:
            errors.append(f"Input: missing field {key!r}.")
    for key in solution_keys:
        if key not in solution:
            errors.append(f"Structure: missing solution field {key!r}.")
    for key in solution:
        if key not in solution_keys:
            errors.append(f"Structure: unexpected solution field {key!r}.")
    if errors:
        return errors

    n = instance["n"]
    valid_n = type(n) is int and n >= 0
    if not valid_n:
        errors.append("Input: n must be a nonnegative integer.")
    for key in ("cx", "cy", "R"):
        if not _is_real(instance[key]):
            errors.append(f"Input: {key} must be a finite real number.")
    if _is_real(instance["R"]) and instance["R"] <= 0:
        errors.append("Input: R must be strictly positive.")
    if not _is_binary(instance["rotation_enabled"]):
        errors.append("Input: rotation_enabled must be integer 0 or 1.")

    items = instance["items"]
    if not isinstance(items, list):
        errors.append("Input: items must be a list of [L_i, W_i] lists.")
    else:
        if valid_n and len(items) != n:
            errors.append("Input: items must contain exactly n entries.")
        for i, pair in enumerate(items):
            if not isinstance(pair, list) or len(pair) != 2:
                errors.append(f"Input: items[{i}] must be a two-element list.")
                continue
            for axis, value in enumerate(pair):
                if not _is_real(value) or value <= 0:
                    errors.append(
                        f"Input: items[{i}][{axis}] must be a strictly "
                        "positive finite real number."
                    )

    for key in solution_keys:
        values = solution[key]
        if not isinstance(values, list):
            errors.append(f"Structure: {key} must be a list.")
            continue
        if valid_n and len(values) != n:
            errors.append(f"Structure: {key} must contain exactly n entries.")
        for i, value in enumerate(values):
            if key in ("alpha", "r"):
                if not _is_binary(value):
                    errors.append(f"Domain: {key}[{i}] must be integer 0 or 1.")
            elif not _is_real(value):
                errors.append(f"Domain: {key}[{i}] must be a finite real number.")
    if errors:
        return errors

    alpha = solution["alpha"]
    rotations = solution["r"]
    x = [Fraction(value) for value in solution["x"]]
    y = [Fraction(value) for value in solution["y"]]
    cx, cy = Fraction(instance["cx"]), Fraction(instance["cy"])
    radius_squared = Fraction(instance["R"]) ** 2
    packed = [i for i in range(n) if alpha[i] == 1]

    # C1
    # Enforce the dimension definitions by deriving them, never accepting
    # independent width/height decisions. C2 and C3 use these exact values.
    widths, heights = [], []
    for i, pair in enumerate(items):
        length, width = Fraction(pair[0]), Fraction(pair[1])
        rotated = rotations[i]
        widths.append((1 - rotated) * length + rotated * width)
        heights.append((1 - rotated) * width + rotated * length)

    # C2
    # All four corners of every selected item must lie in the closed circle.
    for i in packed:
        if any(
            (x[i] + sx * widths[i] / 2 - cx) ** 2
            + (y[i] + sy * heights[i] / 2 - cy) ** 2
            > radius_squared
            for sx in (-1, 1)
            for sy in (-1, 1)
        ):
            errors.append(f"C2: packed item {i} has a corner outside the circle.")

    # C3
    # Each selected pair must be separated on at least one axis; equality
    # is permitted, so touching edges or corners are not overlaps.
    for position, i in enumerate(packed):
        for j in packed[position + 1:]:
            separated_x = abs(x[i] - x[j]) >= (widths[i] + widths[j]) / 2
            separated_y = abs(y[i] - y[j]) >= (heights[i] + heights[j]) / 2
            if not (separated_x or separated_y):
                errors.append(f"C3: packed items {i} and {j} overlap internally.")

    # C4
    # Every unselected item must retain all three prescribed dummy values.
    for i in range(n):
        if alpha[i] == 0 and (x[i] != -1 or y[i] != -1 or rotations[i] != 0):
            errors.append(
                f"C4: unpacked item {i} must have x=-1, y=-1, and r=0."
            )

    # C5
    # The global input permission bounds every individual rotation decision.
    for i in range(n):
        if rotations[i] > instance["rotation_enabled"]:
            errors.append(f"C5: item {i} rotates while global rotation is disabled.")

    # C6
    # This strict input condition applies even when n=0 or all items are packed.
    if (-1 - cx) ** 2 + (-1 - cy) ** 2 <= radius_squared:
        errors.append("C6: the fixed dummy point (-1, -1) must be strictly outside the circle.")

    return errors
