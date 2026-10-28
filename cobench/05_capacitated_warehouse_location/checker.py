"""Feasibility checker for the capacitated facility-location solution format."""

import math
import re


_REL = 1e-9


def _leq(a, b):
    """Compare floating-point results without relaxing directly supplied values."""
    if not _is_finite_number(a) or not _is_finite_number(b):
        return False
    try:
        return a - b <= _REL * max(1.0, abs(a), abs(b))
    except OverflowError:
        return a <= b


def _geq(a, b):
    """Compare floating-point results without relaxing directly supplied values."""
    if not _is_finite_number(a) or not _is_finite_number(b):
        return False
    try:
        return b - a <= _REL * max(1.0, abs(a), abs(b))
    except OverflowError:
        return a >= b


def _equal(a, b):
    return _leq(a, b) and _geq(a, b)


def _is_finite_number(value):
    if type(value) is int:
        return True
    return type(value) is float and math.isfinite(value)


def _parse_string_index(key):
    """Return the sole decimal index embedded in a string, or None if ambiguous."""
    stripped = key.strip()
    if re.fullmatch(r"[+-]?\d+", stripped):
        return int(stripped)
    digit_groups = re.findall(r"\d+", stripped)
    if len(digit_groups) != 1:
        return None
    return int(digit_groups[0])


def _read_count(instance, field, violations):
    if field not in instance:
        violations.append(f"FORMAT | instance is missing {field}; constraints cannot be checked")
        return None
    value = instance[field]
    if type(value) is not int:
        violations.append(f"FORMAT | instance.{field} must be an int")
        return None
    return value


def _normalise_indexed_dict(raw, expected_count, path, violations):
    """Map flexible string spellings to the complete, unambiguous index set."""
    if not isinstance(raw, dict):
        violations.append(f"FORMAT | {path} must be a dict with string keys")
        return None

    expected = set(range(expected_count))
    normalised = {}
    start = len(violations)

    for key, value in raw.items():
        # Output shape is dict[str, ...]: integer keys are not valid aliases.
        if type(key) is not str:
            violations.append(
                f"FORMAT | {path} key {key!r} must be a string, got {type(key).__name__}"
            )
            continue

        index = _parse_string_index(key)
        if index is None:
            violations.append(
                f"FORMAT | {path} key {key!r} does not identify exactly one index"
            )
            continue
        if index not in expected:
            violations.append(
                f"FORMAT | {path} key {key!r} maps to out-of-range index {index}"
            )
            continue
        if index in normalised:
            violations.append(
                f"FORMAT | {path} keys collide at index {index}; mapping is ambiguous"
            )
            continue
        normalised[index] = value

    missing = sorted(expected - set(normalised))
    if missing:
        violations.append(f"FORMAT | {path} is missing indices {missing}")

    if len(violations) != start:
        return None
    return normalised


def _read_open_plants(solution, num_plants, violations):
    if "open_plants" not in solution:
        violations.append("FORMAT | missing required field open_plants")
        return None

    raw = solution["open_plants"]
    if not isinstance(raw, list):
        violations.append("FORMAT | open_plants must be a list[int]")
        return None

    expected_length = len(range(num_plants))
    if len(raw) != expected_length:
        violations.append(
            "FORMAT | open_plants must list every plant: "
            f"expected {expected_length} entries, got {len(raw)}"
        )
        return None

    start = len(violations)
    for j, value in enumerate(raw):
        # DECISION VARIABLE y[j] is exactly binary; no tolerance is appropriate.
        if type(value) is not int or value not in (0, 1):
            violations.append(
                f"DOMAIN y | open_plants[{j}] must be integer 0 or 1, got {value!r}"
            )

    if len(violations) != start:
        return None
    return raw


def _read_assignments(solution, num_customers, num_plants, violations):
    if "assignments" not in solution:
        violations.append("FORMAT | missing required field assignments")
        return None

    customers = _normalise_indexed_dict(
        solution["assignments"], num_customers, "assignments", violations
    )
    if customers is None:
        return None

    assignments = {}
    start = len(violations)
    for i in range(num_customers):
        plants = _normalise_indexed_dict(
            customers[i], num_plants, f"assignments[{i}]", violations
        )
        if plants is None:
            continue

        checked_row = {}
        for j in range(num_plants):
            value = plants[j]
            if not _is_finite_number(value):
                violations.append(
                    f"DOMAIN x | assignments[{i}][{j}] must be a finite number"
                )
                continue
            # x[i][j] is supplied directly, so enforce [0, 1] exactly.
            if value < 0 or value > 1:
                violations.append(
                    f"DOMAIN x | assignments[{i}][{j}]={value!r} is outside [0, 1]"
                )
                continue
            checked_row[j] = value
        assignments[i] = checked_row

    if len(violations) != start:
        return None
    return assignments


def _read_numeric_axis(instance, field, needed, violations):
    if field not in instance:
        violations.append(f"FORMAT | instance is missing {field}; C2 cannot be checked")
        return None
    values = instance[field]
    if not isinstance(values, (list, tuple)):
        violations.append(f"FORMAT | instance.{field} must be an indexed sequence")
        return None

    expected_length = len(range(needed))
    if len(values) < expected_length:
        violations.append(
            f"FORMAT | instance.{field} has {len(values)} entries; "
            f"C2 needs {expected_length}"
        )
        return None

    for index in range(needed):
        if not _is_finite_number(values[index]):
            violations.append(
                f"FORMAT | instance.{field}[{index}] must be finite numeric data"
            )
            return None
    return values


def _check_c1(assignments, num_customers, num_plants, violations):
    """C1: every customer's assignment fractions sum exactly to one."""
    for i in range(num_customers):
        total_fraction = sum(assignments[i][j] for j in range(num_plants))
        # The total is computed by addition, so compare it with relative tolerance.
        if not _equal(total_fraction, 1.0):
            violations.append(
                "C1 | customer "
                f"{i} assignment fractions sum to {total_fraction!r}, not 1"
            )


def _check_c2(
    assignments,
    open_plants,
    demands,
    capacities,
    num_customers,
    num_plants,
    violations,
):
    """C2: recompute each plant load and compare it with its open/closed limit."""
    for j in range(num_plants):
        load = sum(demands[i] * assignments[i][j] for i in range(num_customers))

        if open_plants[j] == 0:
            # Products and a cross-customer sum produce a floating-point result.
            if not _equal(load, 0.0):
                violations.append(
                    f"C2 | closed plant {j} has recomputed assigned demand {load!r}, not 0"
                )
        elif not _leq(load, capacities[j]):
            violations.append(
                f"C2 | open plant {j} has recomputed load {load!r}, "
                f"exceeding capacity {capacities[j]!r}"
            )


def check_solution(instance: dict, solution: dict) -> list:
    """Return concrete feasibility violations; an empty list means feasible."""
    violations = []
    try:
        if not isinstance(instance, dict):
            return ["FORMAT | instance must be a dict"]
        if not isinstance(solution, dict):
            return ["FORMAT | solution must be a dict"]

        num_customers = _read_count(instance, "num_customers", violations)
        num_plants = _read_count(instance, "num_plants", violations)
        if num_customers is None or num_plants is None:
            return violations

        open_plants = _read_open_plants(solution, num_plants, violations)
        assignments = _read_assignments(
            solution, num_customers, num_plants, violations
        )

        # C1 depends only on a complete, domain-valid assignments matrix.
        if assignments is not None:
            _check_c1(assignments, num_customers, num_plants, violations)

        # C2 additionally needs y, demands, and capacities. Ignore optional fields.
        demands = _read_numeric_axis(
            instance, "customers_demands", num_customers, violations
        )
        capacities = _read_numeric_axis(
            instance, "plants_capacities", num_plants, violations
        )
        if (
            assignments is not None
            and open_plants is not None
            and demands is not None
            and capacities is not None
        ):
            _check_c2(
                assignments,
                open_plants,
                demands,
                capacities,
                num_customers,
                num_plants,
                violations,
            )

        return violations
    except Exception as exc:
        # Malformed or adversarial objects must never crash the grading process.
        violations.append(
            "FORMAT | checker could not safely evaluate the supplied data: "
            f"{type(exc).__name__}: {exc}"
        )
        return violations
