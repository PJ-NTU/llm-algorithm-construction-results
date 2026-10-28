"""3D container-loading feasibility check, standard library only.

JSON int/float decimal representations are converted to Fraction before computing; no tolerance is added.
Invalid input, malformed solutions and model violations all return strings; only an empty list means feasible.
Support relations and loads are derived data, not additional decision variables.
"""

import math
from fractions import Fraction
from itertools import combinations


_INPUT_KEYS = (
    "container_length", "container_width", "container_height", "num_box_types",
    "box_quantities", "box_lengths", "box_widths", "box_heights", "box_weights",
    "orientation_allowed", "load_bearing_by_orientation",
)
_SOLUTION_KEYS = ("s", "x", "y", "z", "o")


def _number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _exact(value):
    return Fraction(value) if type(value) is int else Fraction(str(value))


def _instance_errors(instance):
    if not isinstance(instance, dict):
        return ["INPUT: instance must be a dict."]
    errors = [f"INPUT: missing field {key}." for key in _INPUT_KEYS if key not in instance]
    if errors:
        return errors
    n = instance["num_box_types"]
    if type(n) is not int or n < 0:
        return ["INPUT: num_box_types must be a non-negative integer, not a boolean."]
    for key in ("container_length", "container_width", "container_height"):
        value = instance[key]
        if not _number(value) or value <= 0:
            errors.append(f"INPUT: {key} must be a finite positive real number.")
    for key in ("box_quantities", "box_lengths", "box_widths", "box_heights", "box_weights"):
        values = instance[key]
        if not isinstance(values, list) or len(values) != n:
            errors.append(f"INPUT: {key} must be a list of length N={n}.")
            continue
        for i, value in enumerate(values):
            if key == "box_quantities":
                valid = type(value) is int and value >= 0
                requirement = "a non-negative integer, not a boolean"
            else:
                valid = _number(value) and value > 0
                requirement = "a finite positive real number"
            if not valid:
                errors.append(f"INPUT: {key}[{i}] must be {requirement}.")
    for key in ("orientation_allowed", "load_bearing_by_orientation"):
        rows = instance[key]
        if not isinstance(rows, list) or len(rows) != n:
            errors.append(f"INPUT: {key} must be a list of N={n} rows.")
            continue
        for i, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != 3:
                errors.append(f"INPUT: {key}[{i}] must be a list of length 3.")
                continue
            for r, value in enumerate(row):
                if key == "orientation_allowed":
                    valid = type(value) is int and value in (0, 1)
                    requirement = "the integer 0 or 1, not a boolean"
                else:
                    valid = _number(value) and value >= 0
                    requirement = "a finite non-negative real number"
                if not valid:
                    errors.append(f"INPUT: {key}[{i}][{r}] must be {requirement}.")
    return errors


def check(instance: dict, solution: dict) -> list:
    """Return a list of violation strings; an empty list means complete and feasible (optimality is not judged)."""
    errors = _instance_errors(instance)
    if errors:
        return errors
    if not isinstance(solution, dict):
        return ["STRUCTURE: solution must be a dict."]
    errors = [f"STRUCTURE: missing field {key}." for key in _SOLUTION_KEYS if key not in solution]
    if errors:
        return errors
    n = instance["num_box_types"]
    quantities = instance["box_quantities"]
    # Solution structure and basic numeric types; missing fields or wrong types prevent further geometric computation.
    for key in _SOLUTION_KEYS:
        rows = solution[key]
        if not isinstance(rows, list) or len(rows) != n:
            errors.append(f"STRUCTURE: {key} must be a list of N={n} rows.")
            continue
        for i, q in enumerate(quantities):
            row = rows[i]
            if not isinstance(row, list) or len(row) != q:
                errors.append(f"STRUCTURE: {key}[{i}] must have exactly {q} entries, including unloaded boxes.")
                continue
            for k, value in enumerate(row):
                valid = type(value) is int if key in ("s", "o") else _number(value)
                if not valid:
                    errors.append(f"STRUCTURE: {key}[{i}][{k}] has an invalid type; booleans, null and non-finite numbers are not allowed.")
    if errors:
        return errors

    # C1 selection status matches orientation; also check the discrete domains of s and o.
    for i, q in enumerate(quantities):
        for k in range(q):
            s, o = solution["s"][i][k], solution["o"][i][k]
            if s not in (0, 1):
                errors.append(f"C1: s[{i}][{k}] must be 0 or 1.")
            if o not in (0, 1, 2, 3):
                errors.append(f"C1: o[{i}][{k}] must be one of 0, 1, 2, 3.")
            elif (s == 0 and o != 0) or (s == 1 and o == 0):
                errors.append(f"C1: box ({i},{k}) must have orientation 0 when not loaded and 1, 2 or 3 when loaded.")
    if errors:
        return errors

    # boxes[b] is the derived (x,y,z,dx,dy,dz), containing only loaded boxes.
    boxes = {}
    for i, q in enumerate(quantities):
        length = _exact(instance["box_lengths"][i])
        width = _exact(instance["box_widths"][i])
        height = _exact(instance["box_heights"][i])
        orientations = ((width, height, length), (length, height, width), (length, width, height))
        for k in range(q):
            if solution["s"][i][k] == 1:
                xyz = tuple(_exact(solution[axis][i][k]) for axis in ("x", "y", "z"))
                boxes[i, k] = xyz + orientations[solution["o"][i][k] - 1]

    # C2 the orientation of a loaded box must be allowed by its box type.
    for i, k in boxes:
        o = solution["o"][i][k]
        if instance["orientation_allowed"][i][o - 1] != 1:
            errors.append(f"C2: box ({i},{k}) has orientation {o}, which is not allowed.")

    # C3 all candidate box coordinates are non-negative; only loaded boxes must respect the container's upper bounds.
    for axis in ("x", "y", "z"):
        for i, row in enumerate(solution[axis]):
            for k, value in enumerate(row):
                if value < 0:
                    errors.append(f"C3/domain: {axis}[{i}][{k}] must be non-negative, including unloaded boxes.")
    bounds = tuple(_exact(instance[key]) for key in ("container_length", "container_width", "container_height"))
    for b, geometry in boxes.items():
        for r, axis in enumerate(("x", "y", "z")):
            if geometry[r] + geometry[r + 3] > bounds[r]:
                errors.append(f"C3: box {b} exceeds the container's upper bound in the {axis} direction.")

    # C4 loaded count per box type does not exceed its quantity limit; implied by shape and binary domain, but checked explicitly.
    for i, q in enumerate(quantities):
        if sum(solution["s"][i]) > q:
            errors.append(f"C4: loaded count of box type {i} exceeds the limit {q}.")

    # C5 build the complete direct-support sets and require each off-floor box to have exactly one full supporting box.
    parents = {b: [] for b in boxes}
    for b, (x, y, z, dx, dy, dz) in boxes.items():
        for a, (ax, ay, az, adx, ady, adz) in boxes.items():
            if a != b and az + adz == z and ax <= x and x + dx <= ax + adx and ay <= y and y + dy <= ay + ady:
                parents[b].append(a)
        if z > 0 and len(parents[b]) != 1:
            errors.append(f"C5: off-floor box {b} has {len(parents[b])} full direct supporting boxes, must have exactly one; overhang and composite support are not allowed.")

    # C6 any two boxes are separated along at least one axis; non-strict inequalities allow boundary contact.
    for b, c in combinations(boxes, 2):
        x, y, z, dx, dy, dz = boxes[b]
        cx, cy, cz, cdx, cdy, cdz = boxes[c]
        separated = (
            x + dx <= cx or cx + cdx <= x
            or y + dy <= cy or cy + cdy <= y
            or z + dz <= cz or cz + cdz <= z
        )
        if not separated:
            errors.append(f"C6: boxes {b} and {c} overlap in volume.")

    # C7 accumulate the self-weight of all descendants along support relations, each descendant counted once, excluding the box's own weight.
    # Positive heights make support relations strictly increasing in z; propagating sets from top to bottom yields the full transitive closure.
    descendants = {b: set() for b in boxes}
    for b in sorted(boxes, key=lambda item: boxes[item][2], reverse=True):
        for a in parents[b]:
            descendants[a].add(b)
            descendants[a].update(descendants[b])
    weights = [_exact(value) for value in instance["box_weights"]]
    for b, geometry in boxes.items():
        i, k = b
        o = solution["o"][i][k]
        load = sum((weights[c[0]] for c in descendants[b]), Fraction(0))
        capacity = geometry[3] * geometry[4] * _exact(instance["load_bearing_by_orientation"][i][o - 1])
        if load > capacity:
            errors.append(f"C7: box {b} carries a cumulative load {load} above it, exceeding its load capacity {capacity} in the current orientation.")
    return errors
