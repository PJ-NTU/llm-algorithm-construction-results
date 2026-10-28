"""Feasibility checker for balanced attribute partitioning solutions."""

from typing import Optional


def _is_int(value: object) -> bool:
    """Return whether value is an integer, excluding booleans."""

    return isinstance(value, int) and not isinstance(value, bool)


def _validate_instance(instance: object) -> tuple[list[str], Optional[tuple[int, int, int]]]:
    """Validate the instance fields needed to interpret a solution safely."""

    violations: list[str] = []
    if not isinstance(instance, dict):
        return ["FORMAT | instance must be a dict."], None

    required = (
        "num_individuals",
        "num_attributes",
        "num_groups",
        "attribute_matrix",
    )
    missing = [name for name in required if name not in instance]
    if missing:
        violations.append(
            "FORMAT | instance is missing required field(s): " + ", ".join(missing) + "."
        )

    dimensions: dict[str, int] = {}
    for name in ("num_individuals", "num_attributes", "num_groups"):
        if name not in instance:
            continue
        value = instance[name]
        if not _is_int(value):
            violations.append(f"FORMAT | instance.{name} must be an integer (not bool).")
            continue
        dimensions[name] = value

    n = dimensions.get("num_individuals")
    m = dimensions.get("num_attributes")
    k = dimensions.get("num_groups")

    if n is not None and n < 0:
        violations.append("FORMAT | instance.num_individuals cannot be negative.")
    if m is not None and m < 0:
        violations.append("FORMAT | instance.num_attributes cannot be negative.")
    if k is not None and k < 0:
        violations.append("FORMAT | instance.num_groups cannot be negative.")
    elif k == 0:
        violations.append(
            "FORMAT | instance.num_groups cannot be zero because the objective contains 1/K."
        )

    matrix = instance.get("attribute_matrix")
    if "attribute_matrix" in instance:
        if not isinstance(matrix, list):
            violations.append("FORMAT | instance.attribute_matrix must be a list of rows.")
        else:
            if n is not None and n >= 0 and len(matrix) != n:
                violations.append(
                    "FORMAT | instance.attribute_matrix must contain exactly "
                    f"{n} row(s), one per individual; got {len(matrix)}."
                )
            for row_index, row in enumerate(matrix):
                if not isinstance(row, list):
                    violations.append(
                        f"FORMAT | instance.attribute_matrix[{row_index}] must be a list."
                    )
                    continue
                if m is not None and m >= 0 and len(row) != m:
                    violations.append(
                        f"FORMAT | instance.attribute_matrix[{row_index}] must contain "
                        f"exactly {m} attribute value(s); got {len(row)}."
                    )
                for attribute_index, value in enumerate(row):
                    if not _is_int(value) or value not in (0, 1):
                        violations.append(
                            "FORMAT | instance.attribute_matrix"
                            f"[{row_index}][{attribute_index}] must be integer 0 or 1."
                        )

    usable_dimensions: Optional[tuple[int, int, int]] = None
    if (
        n is not None
        and m is not None
        and k is not None
        and n >= 0
        and m >= 0
        and k > 0
    ):
        usable_dimensions = (n, m, k)

    return violations, usable_dimensions


def _validate_solution(solution: object) -> tuple[list[str], Optional[dict]]:
    """Validate only the required output structure."""

    if not isinstance(solution, dict):
        return ["FORMAT | solution must be a dict."], None
    if "group_assignment" not in solution:
        return ["FORMAT | solution is missing required field group_assignment."], None

    assignment = solution["group_assignment"]
    if not isinstance(assignment, dict):
        return ["FORMAT | solution.group_assignment must be a dict[str, int]."], None
    return [], assignment


def _individual_index(identifier: str) -> Optional[int]:
    """Map an indexed string identifier to its nonnegative individual index.

    Bare decimal identifiers (``"0"``) and identifiers whose final
    underscore-delimited component is decimal (``"person_0"``) are accepted.
    This does not make the example's ``person_`` prefix mandatory.
    """

    index_text = identifier.rsplit("_", 1)[-1]
    if not index_text or not index_text.isascii() or not index_text.isdigit():
        return None
    return int(index_text)


def _check_c1_exactly_one_group(assignment: dict, n: int, k: int) -> list[str]:
    """C1: every indexed individual has exactly one valid group assignment."""

    violations: list[str] = []
    index_to_identifier: dict[int, str] = {}

    for identifier, group in assignment.items():
        if not isinstance(identifier, str):
            violations.append(
                "C1 | every group_assignment key must be a string individual identifier; "
                f"got {type(identifier).__name__}."
            )
            continue

        individual_index = _individual_index(identifier)
        if individual_index is None:
            violations.append(
                "C1 | individual identifier "
                f"{identifier!r} does not unambiguously encode a nonnegative index."
            )
        elif individual_index in index_to_identifier:
            violations.append(
                "C1 | individual identifiers "
                f"{index_to_identifier[individual_index]!r} and {identifier!r} both identify "
                f"individual {individual_index}, so that individual is assigned ambiguously."
            )
        else:
            index_to_identifier[individual_index] = identifier

        if not _is_int(group):
            violations.append(
                f"C1 | assignment for {identifier!r} must be one integer group index, not "
                f"{type(group).__name__}."
            )
        elif not 0 <= group < k:
            violations.append(
                f"C1 | assignment for {identifier!r} uses group {group}, but valid group "
                f"indices are 0 through {k - 1}."
            )

    expected_indices = set(range(n))
    actual_indices = set(index_to_identifier)
    missing_indices = sorted(expected_indices - actual_indices)
    extra_indices = sorted(actual_indices - expected_indices)
    if missing_indices:
        violations.append(
            "C1 | group_assignment is missing individual index/indices: "
            + ", ".join(map(str, missing_indices))
            + "."
        )
    if extra_indices:
        violations.append(
            "C1 | group_assignment contains out-of-range individual index/indices: "
            + ", ".join(map(str, extra_indices))
            + f"; valid individual indices are 0 through {n - 1}."
        )

    return violations


def check_solution(instance: dict, solution: dict) -> list:
    """Return constraint and format violations; an empty list means feasible."""

    try:
        instance_violations, dimensions = _validate_instance(instance)
        solution_violations, assignment = _validate_solution(solution)
        violations = instance_violations + solution_violations

        if dimensions is not None and assignment is not None:
            n, _m, k = dimensions
            # C1 | Each individual must be assigned to exactly one group.
            violations.extend(_check_c1_exactly_one_group(assignment, n, k))

        return violations
    except Exception as exc:  # Defensive boundary: malformed inputs must not escape.
        return [f"FORMAT | checker could not safely inspect the input: {type(exc).__name__}."]
