"""Feasibility checker for the linear assignment problem described in MAT_s4.md."""


def check_solution(instance: dict, solution: dict) -> list:
    """Return concrete format or hard-constraint violations; return [] if feasible."""
    violations = []

    # Structural checks only make the semantic checks below safe to execute.
    if not isinstance(instance, dict):
        return ["FORMAT | instance must be a dict"]
    if not isinstance(solution, dict):
        return ["FORMAT | solution must be a dict"]
    if "n" not in instance:
        return ["FORMAT | instance is missing required field 'n'"]

    n = instance["n"]
    if isinstance(n, bool) or not isinstance(n, int):
        return ["FORMAT | instance['n'] must be an int"]
    if "assignment" not in solution:
        return ["FORMAT | solution is missing the only required field 'assignment'"]

    assignment = solution["assignment"]
    if not isinstance(assignment, list):
        return ["FORMAT | solution['assignment'] must be a list"]

    non_integer_positions = [
        index
        for index, task in enumerate(assignment)
        if isinstance(task, bool) or not isinstance(task, int)
    ]
    if non_integer_positions:
        preview = non_integer_positions[:10]
        suffix = " ..." if len(non_integer_positions) > len(preview) else ""
        return [
            "FORMAT | solution['assignment'] must contain only integer task "
            f"indices; invalid positions: {preview}{suffix}"
        ]

    # C1 | Every worker must be assigned to exactly one task.
    # A list has one slot per represented worker, so it represents all and only
    # the n workers exactly when its length is n.  The slot's integer must also
    # identify one of the n tasks; all comparisons here are exact integer checks.
    c1_reasons = []
    if len(assignment) != n:
        c1_reasons.append(
            f"assignment has {len(assignment)} worker slots, but n is {n}"
        )

    out_of_range_positions = [
        index for index, task in enumerate(assignment) if not (0 <= task < n)
    ]
    if out_of_range_positions:
        preview = out_of_range_positions[:10]
        suffix = " ..." if len(out_of_range_positions) > len(preview) else ""
        c1_reasons.append(
            "task indices must be in the exact integer range [0, n-1]; "
            f"invalid positions: {preview}{suffix}"
        )

    if c1_reasons:
        violations.append("C1 | " + "; ".join(c1_reasons))

    # C2 | Every task must be assigned to exactly one worker.
    # There are exactly n legal integer task IDs.  Thus every task occurs once
    # iff there are n legal occurrences and they comprise n distinct IDs.
    # This avoids iterating range(n), so even an adversarially large n is safe.
    valid_tasks = [task for task in assignment if 0 <= task < n]
    valid_occurrences = len(valid_tasks)
    distinct_valid_tasks = len(set(valid_tasks))
    if valid_occurrences != n or distinct_valid_tasks != n:
        violations.append(
            "C2 | task IDs do not occur exactly once each: "
            f"found {valid_occurrences} legal task occurrences and "
            f"{distinct_valid_tasks} distinct legal task IDs, but both must equal n={n}"
        )

    return violations
