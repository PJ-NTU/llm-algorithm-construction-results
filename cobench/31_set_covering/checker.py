"""Feasibility checker for the set-cover solution format in MAT_s4.md."""


def _is_int(value):
    """Return True only for JSON-style integers, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def check_solution(instance: dict, solution: dict) -> list:
    """Return human-readable violations; an empty list means feasible."""
    violations = []

    # Structural checks needed to read the given instance without exceptions.
    if not isinstance(instance, dict):
        return ["FORMAT | instance must be a dict"]

    n = instance.get("n")
    m = instance.get("m")
    covers = instance.get("covers")
    if not _is_int(n):
        violations.append("FORMAT | instance.n must be an integer")
    if not _is_int(m):
        violations.append("FORMAT | instance.m must be an integer")
    if not isinstance(covers, list):
        violations.append("FORMAT | instance.covers must be a list")
    if violations:
        return violations
    if m > len(covers):
        return [
            "FORMAT | instance.covers must have a row for every element; "
            f"need {m}, got {len(covers)}"
        ]

    # Decision-variable domain: x[j] is binary for every j in {0, ..., n-1}.
    if not isinstance(solution, dict):
        return ["FORMAT | solution must be a dict"]
    if "selection_status" not in solution:
        return ["FORMAT | missing required field selection_status"]

    selection = solution["selection_status"]
    if not isinstance(selection, list):
        return ["FORMAT | selection_status must be a list of integers"]
    if len(selection) != n:
        violations.append(
            "FORMAT | selection_status must contain exactly "
            f"n={n} entries; got {len(selection)}"
        )

    for j, value in enumerate(selection):
        if not _is_int(value) or value not in (0, 1):
            violations.append(
                "VARIABLE | selection_status["
                f"{j}] must be the integer 0 or 1; got {value!r}"
            )

    # C1 | Every element i must be covered by at least one selected set.
    # Only evaluate C1 after the complete binary decision vector is available.
    if violations:
        return violations

    for i in range(m):
        covering_sets = covers[i]
        if not isinstance(covering_sets, list):
            violations.append(
                f"FORMAT | instance.covers[{i}] must be a list of set indices"
            )
            continue

        valid_indices = []
        malformed_row = False
        for raw_j in covering_sets:
            if not _is_int(raw_j) or not (0 <= raw_j < n):
                violations.append(
                    "FORMAT | instance.covers["
                    f"{i}] contains invalid set index {raw_j!r}"
                )
                malformed_row = True
            else:
                valid_indices.append(raw_j)

        if malformed_row:
            continue
        if not any(selection[j] == 1 for j in valid_indices):
            violations.append(
                f"C1 | element {i} is not covered by any selected set"
            )

    return violations
