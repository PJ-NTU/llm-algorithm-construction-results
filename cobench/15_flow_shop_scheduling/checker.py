"""Feasibility checker for the permutation flow-shop solution format."""

from collections import Counter


_REL = 1e-9


def _geq(a, b):
    """Compare values produced by continuous floating-point arithmetic."""
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _read_instance(instance):
    if not isinstance(instance, dict):
        return None, None, None, ["FORMAT | instance must be a dict so its fields can be read safely"]

    errors = []
    n = instance.get("n")
    n_machines = instance.get("n_machines")
    proc = instance.get("proc")

    if type(n) is not int:
        errors.append("FORMAT | instance field 'n' must be an int")
    if type(n_machines) is not int:
        errors.append("FORMAT | instance field 'n_machines' must be an int")
    if not isinstance(proc, list):
        errors.append("FORMAT | instance field 'proc' must be a list")
    if errors:
        return None, None, None, errors

    # These checks only ensure that every entry needed by proc[j][i] is readable.
    # Extra rows or columns are irrelevant to the stated constraints and are ignored.
    if n > 0 and len(proc) < n:
        return None, None, None, [
            f"FORMAT | proc has {len(proc)} rows, so proc[j][i] is unavailable for some of the {n} jobs"
        ]

    for j in range(max(0, n)):
        row = proc[j]
        if not isinstance(row, list):
            errors.append(f"FORMAT | proc[{j}] must be a list")
            continue
        if n_machines > 0 and len(row) < n_machines:
            errors.append(
                f"FORMAT | proc[{j}] has {len(row)} entries, so some of {n_machines} machines are unavailable"
            )
            continue
        for i in range(max(0, n_machines)):
            value = row[i]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"FORMAT | proc[{j}][{i}] must be a numeric value")

    if errors:
        return None, None, None, errors
    return n, n_machines, proc, []


def _read_permutation(solution):
    if not isinstance(solution, dict):
        return None, ["FORMAT | solution must be a dict containing the required field 'perm'"]
    if "perm" not in solution:
        return None, ["FORMAT | missing required solution field 'perm'"]

    perm = solution["perm"]
    if not isinstance(perm, list):
        return None, ["FORMAT | solution field 'perm' must be a list"]

    bad = [k for k, job in enumerate(perm) if type(job) is not int]
    if bad:
        shown = bad[:10]
        suffix = "" if len(bad) <= 10 else f" (and {len(bad) - 10} more)"
        return None, [
            f"FORMAT | perm entries must be integer job indices; invalid positions: {shown}{suffix}"
        ]
    return perm, []


def _check_c1(n, perm):
    """C1: every job index in 0..n-1 occurs exactly once."""
    counts = Counter(perm)
    represented = sum(1 for job in counts if 0 <= job < n)
    missing_count = max(0, n - represented)
    duplicated = sorted((job, count) for job, count in counts.items() if 0 <= job < n and count != 1)

    errors = []
    if missing_count:
        errors.append(
            f"C1 | {missing_count} job(s) from 0..{n - 1} are not assigned to any processing position"
        )
    if duplicated:
        shown = duplicated[:10]
        suffix = "" if len(duplicated) <= 10 else f" (and {len(duplicated) - 10} more)"
        errors.append(
            f"C1 | jobs must occur exactly once; repeated job counts: {shown}{suffix}"
        )
    return errors


def _check_c2(n, perm):
    """C2: the list supplies exactly one valid job for each of n positions."""
    errors = []
    if len(perm) != n:
        errors.append(f"C2 | expected exactly {n} processing positions, but perm has {len(perm)}")

    invalid = [(k, job) for k, job in enumerate(perm) if not (0 <= job < n)]
    if invalid:
        shown = invalid[:10]
        suffix = "" if len(invalid) <= 10 else f" (and {len(invalid) - 10} more)"
        errors.append(
            f"C2 | every position must contain a job index in 0..{n - 1}; invalid entries: {shown}{suffix}"
        )
    return errors


def _build_completion_times(n, n_machines, proc, perm):
    completion = [[0 for _ in range(n_machines)] for _ in range(n)]
    exact_value = [[True for _ in range(n_machines)] for _ in range(n)]

    for k in range(n):
        job = perm[k]
        for i in range(n_machines):
            previous_position = completion[k - 1][i] if k > 0 else 0
            previous_machine = completion[k][i - 1] if i > 0 else 0
            previous_position_exact = exact_value[k - 1][i] if k > 0 else True
            previous_machine_exact = exact_value[k][i - 1] if i > 0 else True

            if previous_position >= previous_machine:
                start = previous_position
                start_exact = previous_position_exact
            else:
                start = previous_machine
                start_exact = previous_machine_exact

            duration = proc[job][i]
            completion[k][i] = start + duration
            exact_value[k][i] = start_exact and type(duration) is int

    return completion, exact_value


def _satisfies_geq(value, lower_bound, exact_comparison):
    if exact_comparison:
        return value >= lower_bound
    return _geq(value, lower_bound)


def _check_c3(n, n_machines, proc, perm, completion, exact_value):
    """C3: check both predecessor lower bounds for every completion cell."""
    errors = []
    for k in range(n):
        job = perm[k]
        for i in range(n_machines):
            value = completion[k][i]
            duration = proc[job][i]

            previous_position = completion[k - 1][i] if k > 0 else 0
            previous_position_exact = exact_value[k - 1][i] if k > 0 else True
            position_bound = previous_position + duration
            position_bound_exact = previous_position_exact and type(duration) is int
            if not _satisfies_geq(
                value,
                position_bound,
                exact_value[k][i] and position_bound_exact,
            ):
                errors.append(
                    f"C3 | completion[{k}][{i}]={value!r} is below the previous-position bound {position_bound!r}"
                )

            previous_machine = completion[k][i - 1] if i > 0 else 0
            previous_machine_exact = exact_value[k][i - 1] if i > 0 else True
            machine_bound = previous_machine + duration
            machine_bound_exact = previous_machine_exact and type(duration) is int
            if not _satisfies_geq(
                value,
                machine_bound,
                exact_value[k][i] and machine_bound_exact,
            ):
                errors.append(
                    f"C3 | completion[{k}][{i}]={value!r} is below the previous-machine bound {machine_bound!r}"
                )
    return errors


def _check_c4(n, n_machines, completion):
    """C4: choose Cmax as the independently rebuilt last completion time."""
    # The description does not define a last position/machine for an empty axis.
    # Avoid inventing n > 0 or n_machines > 0 as an input requirement.
    if n <= 0 or n_machines <= 0:
        return []

    last_completion = completion[n - 1][n_machines - 1]
    cmax = last_completion
    if cmax >= last_completion:
        return []
    return [
        f"C4 | reconstructed Cmax={cmax!r} is below the last completion time {last_completion!r}"
    ]


def check_solution(instance: dict, solution: dict) -> list:
    """Return constraint or format violations; an empty list means feasible."""
    n, n_machines, proc, instance_errors = _read_instance(instance)
    if instance_errors:
        return instance_errors

    perm, solution_errors = _read_permutation(solution)
    if solution_errors:
        return solution_errors

    violations = []
    violations.extend(_check_c1(n, perm))
    violations.extend(_check_c2(n, perm))
    if violations:
        return violations

    try:
        completion, exact_value = _build_completion_times(n, n_machines, proc, perm)
        violations.extend(
            _check_c3(n, n_machines, proc, perm, completion, exact_value)
        )
        violations.extend(_check_c4(n, n_machines, completion))
    except (ArithmeticError, IndexError, TypeError, ValueError) as exc:
        violations.append(
            f"FORMAT | completion times could not be reconstructed safely: {type(exc).__name__}: {exc}"
        )
    return violations
