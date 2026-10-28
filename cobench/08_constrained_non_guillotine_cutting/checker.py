"""Feasibility checker for the constrained non-guillotine cutting problem."""

import math


_REL = 1e-9


def _leq(a, b):
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _geq(a, b):
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _normalise_key(key):
    if not isinstance(key, str):
        return None
    return "".join(character.lower() for character in key if character.isalnum())


def _get_field(mapping, canonical_name, context, violations):
    wanted = _normalise_key(canonical_name)
    matches = [key for key in mapping if _normalise_key(key) == wanted]
    if not matches:
        violations.append(f"FORMAT | {context} is missing field {canonical_name!r}")
        return None, False
    if len(matches) > 1:
        violations.append(
            f"FORMAT | {context} field {canonical_name!r} has multiple resolvable spellings and cannot be uniquely determined"
        )
        return None, False
    return mapping[matches[0]], True


def _is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        return True
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _is_binary(value):
    if isinstance(value, bool):
        return True
    return _is_finite_number(value) and value in (0, 1)


def _as_integer(value):
    if isinstance(value, bool) or not _is_finite_number(value):
        return None
    integer = int(value)
    return integer if value == integer else None


def _read_instance(instance, violations):
    if not isinstance(instance, dict):
        violations.append("INPUT | instance must be a dict; cannot check the solution against it")
        return None

    values = {}
    required = (
        "stock_length",
        "stock_width",
        "num_piece_types",
        "max_slots",
        "piece_lengths",
        "piece_widths",
        "min_usage",
        "max_usage",
    )
    for name in required:
        value, found = _get_field(instance, name, "instance", violations)
        if found:
            values[name] = value
    if len(values) != len(required):
        return None

    n = _as_integer(values["num_piece_types"])
    k = _as_integer(values["max_slots"])
    if n is None:
        violations.append("INPUT | num_piece_types must be an exactly interpretable integer")
    if k is None:
        violations.append("INPUT | max_slots must be an exactly interpretable integer")
    for name in ("stock_length", "stock_width"):
        if not _is_finite_number(values[name]):
            violations.append(f"INPUT | {name} must be a finite number")
    if violations:
        return None

    needed = max(n, 0)
    for name in ("piece_lengths", "piece_widths", "min_usage", "max_usage"):
        sequence = values[name]
        if not isinstance(sequence, (list, tuple)):
            violations.append(f"INPUT | {name} must be a sequence, cannot read values per piece type")
            continue
        if len(sequence) < needed:
            violations.append(
                f"INPUT | {name} has only {len(sequence)} items, cannot read the {n} piece types declared in the description"
            )
            continue
        for index in range(needed):
            if not _is_finite_number(sequence[index]):
                violations.append(f"INPUT | {name}[{index}] must be a finite number")
    if violations:
        return None

    values["num_piece_types"] = n
    values["max_slots"] = k
    return values


def _read_placements(solution, max_slots, violations):
    if not isinstance(solution, dict):
        violations.append("FORMAT | solution must be a dict")
        return None

    placements, found = _get_field(solution, "placements", "solution", violations)
    if not found:
        return None
    if not isinstance(placements, list):
        violations.append("FORMAT | placements must be list[dict]")
        return None
    if len(placements) != max_slots:
        violations.append(
            f"FORMAT | placements must list all K={max_slots} slots, got {len(placements)} entries"
        )
        return None

    parsed = []
    names = ("is_used", "piece_type_id", "x", "y", "is_rotated")
    for slot, placement in enumerate(placements):
        if not isinstance(placement, dict):
            violations.append(f"FORMAT | placements[{slot}] must be a dict")
            continue
        record = {}
        for name in names:
            value, present = _get_field(
                placement, name, f"placements[{slot}]", violations
            )
            if present:
                record[name] = value
        if len(record) != len(names):
            continue

        if not _is_binary(record["is_used"]):
            violations.append(f"C1 | Slot {slot} must have binary 0/1 is_used")
        piece_type = _as_integer(record["piece_type_id"])
        if piece_type is None:
            violations.append(f"C1 | Slot {slot} must have an integer index as piece_type_id")
        else:
            record["piece_type_id"] = piece_type
        if not _is_finite_number(record["x"]):
            violations.append(f"C2 | Slot {slot} must have a finite x")
        if not _is_finite_number(record["y"]):
            violations.append(f"C2 | Slot {slot} must have a finite y")
        if not _is_binary(record["is_rotated"]):
            violations.append(f"C2 | Slot {slot} must have binary 0/1 is_rotated")
        parsed.append(record)

    if violations:
        return None
    return parsed


def _dimensions(record, data):
    if record["is_used"] == 0:
        return 0, 0
    piece_type = record["piece_type_id"]
    length = data["piece_lengths"][piece_type]
    width = data["piece_widths"][piece_type]
    if record["is_rotated"] == 1:
        return width, length
    return length, width


def _check_c1(placements, data):
    """C1: one type exactly when a slot is active."""
    violations = []
    n = data["num_piece_types"]
    for slot, record in enumerate(placements):
        piece_type = record["piece_type_id"]
        if record["is_used"] == 1 and not (0 <= piece_type < n):
            violations.append(
                f"C1 | Activated slot {slot} has piece_type_id={piece_type} which is not a piece type in 0..{n - 1} inclusive"
            )
        elif record["is_used"] == 0 and piece_type != -1:
            violations.append(
                f"C1 | Inactive slot {slot} must have piece_type_id -1, got {piece_type}"
            )
    return violations


def _check_c2(placements, data):
    """C2: every slot obeys the rotated boundary formula."""
    violations = []
    stock_length = data["stock_length"]
    stock_width = data["stock_width"]
    for slot, record in enumerate(placements):
        width, height = _dimensions(record, data)
        x = record["x"]
        y = record["y"]
        if not _geq(x, 0):
            violations.append(f"C2 | Slot {slot} has lower-left x={x} outside the left edge of the stock sheet")
        if not _geq(y, 0):
            violations.append(f"C2 | Slot {slot} has lower-left y={y} outside the bottom edge of the stock sheet")
        if not _leq(x + width, stock_length):
            violations.append(
                f"C2 | Slot {slot} has right edge x+width={x + width} exceeding stock length {stock_length}"
            )
        if not _leq(y + height, stock_width):
            violations.append(
                f"C2 | Slot {slot} has top edge y+height={y + height} exceeding stock width {stock_width}"
            )
    return violations


def _check_c3(placements, data):
    """C3: each pair of used rectangles is separated on at least one axis."""
    violations = []
    for first in range(len(placements)):
        a = placements[first]
        if a["is_used"] == 0:
            continue
        a_width, a_height = _dimensions(a, data)
        for second in range(first + 1, len(placements)):
            b = placements[second]
            if b["is_used"] == 0:
                continue
            b_width, b_height = _dimensions(b, data)
            separated = (
                _leq(a["x"] + a_width, b["x"])
                or _leq(b["x"] + b_width, a["x"])
                or _leq(a["y"] + a_height, b["y"])
                or _leq(b["y"] + b_height, a["y"])
            )
            if not separated:
                violations.append(f"C3 | Placed slots {first} and {second} have overlapping rectangle interiors")
    return violations


def _check_c4(placements, data):
    """C4: exact integer usage counts lie within the supplied bounds."""
    violations = []
    n = data["num_piece_types"]
    counts = [0] * max(n, 0)
    for record in placements:
        if record["is_used"] == 1:
            counts[record["piece_type_id"]] += 1
    for piece_type in range(max(n, 0)):
        count = counts[piece_type]
        minimum = data["min_usage"][piece_type]
        maximum = data["max_usage"][piece_type]
        if count < minimum or count > maximum:
            violations.append(
                f"C4 | Piece type {piece_type} uses {count} pieces, outside the given range [{minimum}, {maximum}] bounds"
            )
    return violations


def _check_solution(instance, solution):
    violations = []
    data = _read_instance(instance, violations)
    if data is None:
        return violations
    placements = _read_placements(solution, data["max_slots"], violations)
    if placements is None:
        return violations

    c1_violations = _check_c1(placements, data)
    violations.extend(c1_violations)
    if c1_violations:
        return violations
    violations.extend(_check_c2(placements, data))
    violations.extend(_check_c3(placements, data))
    violations.extend(_check_c4(placements, data))
    return violations


def check_solution(instance: dict, solution: dict) -> list:
    """Return concrete violations; an empty list means the solution is feasible."""
    try:
        return _check_solution(instance, solution)
    except Exception as error:  # Last-resort containment for adversarial Python objects.
        return [f"FORMAT | encountered data that cannot be read safely during checking, error={type(error).__name__}"]
