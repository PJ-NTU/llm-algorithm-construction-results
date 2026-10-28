"""Feasibility checker for the job-shop scheduling solution format in MAT_s4.md."""

import math
import re


_REL = 1e-9


def _leq(a, b):
    """Relative comparison for quantities produced by continuous arithmetic."""
    try:
        return a - b <= _REL * max(1.0, abs(a), abs(b))
    except (OverflowError, TypeError, ValueError):
        return False


def _geq(a, b):
    """Relative comparison for quantities produced by continuous arithmetic."""
    try:
        return b - a <= _REL * max(1.0, abs(a), abs(b))
    except (OverflowError, TypeError, ValueError):
        return False


def _is_real(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _all_ints(*values):
    return all(type(value) is int for value in values)


def _completion(start, duration):
    try:
        value = start + duration
    except (ArithmeticError, TypeError, ValueError):
        return None
    return value if _is_real(value) else None


def _starts_after_or_at(start, prior_start, duration):
    """Compare against a recomputed completion time with classified tolerance."""
    completion = _completion(prior_start, duration)
    if completion is None:
        return None
    if _all_ints(start, prior_start, duration):
        return start >= completion
    return _geq(start, completion)


def _read_dimensions(instance, violations):
    if not isinstance(instance, dict):
        violations.append("FORMAT | instance must be a dict")
        return None

    n_jobs = instance.get("n_jobs")
    n_machines = instance.get("n_machines")
    if type(n_jobs) is not int:
        violations.append("FORMAT | instance.n_jobs must be an int")
    if type(n_machines) is not int:
        violations.append("FORMAT | instance.n_machines must be an int")
    if type(n_jobs) is not int or type(n_machines) is not int:
        return None

    # Zero is retained as a legitimate degenerate count.  Negative values cannot
    # define the indexed shapes named by the description, but must not crash us.
    if n_jobs < 0 or n_machines < 0:
        violations.append(
            "FORMAT | negative instance dimensions cannot define the described indexed arrays"
        )
        return None
    return n_jobs, n_machines


def _read_operations(instance, n_jobs, n_machines, violations):
    jobs = instance.get("jobs")
    if not isinstance(jobs, (list, tuple)):
        violations.append("FORMAT | instance.jobs must be a nested list")
        return None
    if len(jobs) < n_jobs:
        violations.append(
            f"FORMAT | instance.jobs has {len(jobs)} job rows; {n_jobs} are needed"
        )
        return None

    operations = []
    valid = True
    for job_index in range(n_jobs):
        row = jobs[job_index]
        if not isinstance(row, (list, tuple)):
            violations.append(
                f"FORMAT | instance.jobs[{job_index}] must be a list of operations"
            )
            valid = False
            continue
        if len(row) < n_machines:
            violations.append(
                f"FORMAT | instance.jobs[{job_index}] has {len(row)} operations; "
                f"{n_machines} are needed"
            )
            valid = False
            continue

        parsed_row = []
        for operation_index in range(n_machines):
            operation = row[operation_index]
            if not isinstance(operation, (list, tuple)) or len(operation) < 2:
                violations.append(
                    f"FORMAT | instance.jobs[{job_index}][{operation_index}] "
                    "must contain a machine id and a processing duration"
                )
                valid = False
                parsed_row.append(None)
                continue

            machine = operation[0]
            duration = operation[1]
            if type(machine) is not int or not 0 <= machine < n_machines:
                violations.append(
                    f"FORMAT | instance.jobs[{job_index}][{operation_index}][0] "
                    "does not identify a described machine index"
                )
                valid = False
            if not _is_real(duration):
                violations.append(
                    f"FORMAT | instance.jobs[{job_index}][{operation_index}][1] "
                    "must be a finite numeric processing duration"
                )
                valid = False
            parsed_row.append((machine, duration))
        operations.append(parsed_row)

    return operations if valid else None


def _read_start_times(solution, n_jobs, n_machines, violations):
    if "start_times" not in solution:
        violations.append("FORMAT | missing required field: start_times")
        return None
    starts = solution["start_times"]
    if not isinstance(starts, (list, tuple)):
        violations.append("FORMAT | start_times must be a list of job rows")
        return None
    if len(starts) != n_jobs:
        violations.append(
            f"FORMAT | start_times must contain exactly {n_jobs} job rows"
        )
        return None

    valid = True
    for job_index, row in enumerate(starts):
        if not isinstance(row, (list, tuple)):
            violations.append(
                f"FORMAT | start_times[{job_index}] must be a list"
            )
            valid = False
            continue
        if len(row) != n_machines:
            violations.append(
                f"FORMAT | start_times[{job_index}] must contain exactly "
                f"{n_machines} operation times"
            )
            valid = False
            continue
        for operation_index, value in enumerate(row):
            if not _is_real(value):
                violations.append(
                    f"FORMAT | start_times[{job_index}][{operation_index}] "
                    "must be a finite number"
                )
                valid = False
            elif value < 0:
                # Direct decision-variable domain: exact comparison, no slack.
                violations.append(
                    f"DOMAIN | start_times[{job_index}][{operation_index}] "
                    "must be >= 0"
                )
                valid = False
    return starts if valid else None


def _normalized_integer_token(token):
    return token.lstrip("0") or "0"


def _parse_machine_key(key, n_machines):
    """Accept any text key containing exactly one unambiguous integer index."""
    if not isinstance(key, str):
        return None, "machine_precedence keys must be strings"
    # Punctuation is treated as naming syntax, not as a numeric sign.  Thus
    # ``machine-01`` and ``machine_01`` both uniquely identify machine 1.
    tokens = re.findall(r"\d+", key)
    if len(tokens) != 1:
        return None, "machine key must contain exactly one unambiguous integer index"

    normalized = _normalized_integer_token(tokens[0])
    # Compare lengths first so even an adversarially long digit run is safe and
    # leading zeroes remain accepted.
    largest = str(n_machines - 1) if n_machines else "-1"
    if n_machines == 0 or len(normalized) > len(largest):
        return None, "machine key index is outside the described machine range"
    machine = int(normalized)
    if machine >= n_machines:
        return None, "machine key index is outside the described machine range"
    return machine, None


def _brief_key(key):
    rendered = repr(key)
    if len(rendered) <= 80:
        return rendered
    return f"{rendered[:77]}... (length={len(key)})"


def _read_precedence(solution, n_jobs, n_machines, violations):
    if "machine_precedence" not in solution:
        violations.append("FORMAT | missing required field: machine_precedence")
        return None
    raw = solution["machine_precedence"]
    if not isinstance(raw, dict):
        violations.append("FORMAT | machine_precedence must be a dict")
        return None

    by_machine = {}
    valid = True
    for key, matrix in raw.items():
        machine, error = _parse_machine_key(key, n_machines)
        if error:
            violations.append(f"FORMAT | {error}: {_brief_key(key)}")
            valid = False
            continue
        if machine in by_machine and matrix != by_machine[machine]:
            violations.append(
                f"FORMAT | conflicting machine_precedence entries map to machine {machine}"
            )
            valid = False
            continue
        by_machine.setdefault(machine, matrix)

    for machine in range(n_machines):
        if machine not in by_machine:
            violations.append(
                f"FORMAT | machine_precedence is missing a matrix for machine {machine}"
            )
            valid = False
            continue
        matrix = by_machine[machine]
        if not isinstance(matrix, (list, tuple)) or len(matrix) != n_jobs:
            violations.append(
                f"FORMAT | precedence matrix for machine {machine} must have "
                f"exactly {n_jobs} rows"
            )
            valid = False
            continue
        for job_index, row in enumerate(matrix):
            if not isinstance(row, (list, tuple)) or len(row) != n_jobs:
                violations.append(
                    f"FORMAT | precedence matrix for machine {machine}, row "
                    f"{job_index}, must contain exactly {n_jobs} values"
                )
                valid = False
                continue
            for other_job, value in enumerate(row):
                if type(value) is not int or value not in (0, 1):
                    # Binary decision-variable domain: exact membership, no slack.
                    violations.append(
                        f"DOMAIN | machine_precedence for machine {machine} at "
                        f"[{job_index}][{other_job}] must be 0 or 1"
                    )
                    valid = False

    return by_machine if valid else None


def _read_makespan(solution, violations):
    if "makespan" not in solution:
        violations.append("FORMAT | missing required field: makespan")
        return None
    makespan = solution["makespan"]
    if not _is_real(makespan):
        violations.append("FORMAT | makespan must be a finite number")
        return None
    if makespan < 0:
        # Direct decision-variable domain: exact comparison, no slack.
        violations.append("DOMAIN | makespan must be >= 0")
        return None
    return makespan


def _check_c1_job_order(operations, starts, violations):
    """C1: every operation starts after its immediate predecessor completes."""
    for job_index, row in enumerate(operations):
        for operation_index in range(len(row) - 1):
            duration = row[operation_index][1]
            relation_holds = _starts_after_or_at(
                starts[job_index][operation_index + 1],
                starts[job_index][operation_index],
                duration,
            )
            if relation_holds is None:
                violations.append(
                    f"C1 | completion time overflowed for job {job_index}, "
                    f"operation {operation_index}"
                )
            elif not relation_holds:
                violations.append(
                    f"C1 | job {job_index} operation {operation_index + 1} "
                    f"starts before operation {operation_index} completes"
                )


def _check_c2_machine_nonoverlap(operations, starts, precedence, violations):
    """C2: all operation pairs sharing a machine are ordered and nonoverlapping."""
    by_machine = {machine: [] for machine in precedence}
    for job_index, row in enumerate(operations):
        for operation_index, (machine, duration) in enumerate(row):
            by_machine[machine].append(
                (job_index, operation_index, starts[job_index][operation_index], duration)
            )

    for machine, assigned in by_machine.items():
        matrix = precedence[machine]
        for first_index in range(len(assigned)):
            first = assigned[first_index]
            for second_index in range(first_index + 1, len(assigned)):
                second = assigned[second_index]
                first_job, first_op, first_start, first_duration = first
                second_job, second_op, second_start, second_duration = second

                first_before_second = _starts_after_or_at(
                    second_start, first_start, first_duration
                )
                second_before_first = _starts_after_or_at(
                    first_start, second_start, second_duration
                )
                if first_before_second is None or second_before_first is None:
                    violations.append(
                        f"C2 | completion time overflowed for operations "
                        f"({first_job}, {first_op}) and ({second_job}, {second_op}) "
                        f"on machine {machine}"
                    )
                    continue
                if not first_before_second and not second_before_first:
                    violations.append(
                        f"C2 | operations ({first_job}, {first_op}) and "
                        f"({second_job}, {second_op}) overlap on machine {machine}"
                    )

                if first_job == second_job:
                    continue

                # z[j][j'][m] == 1 means j precedes j'; 0 means j' precedes j.
                forward = matrix[first_job][second_job]
                forward_holds = (
                    first_before_second if forward == 1 else second_before_first
                )
                if not forward_holds:
                    expected = (
                        "first job before second"
                        if forward == 1
                        else "second job before first"
                    )
                    violations.append(
                        f"C2 | machine {machine} precedence[{first_job}]"
                        f"[{second_job}] declares {expected}, contrary to start_times"
                    )

                # The output requires all ordered job pairs, so verify the reverse
                # cell by its own stated 0/1 meaning as well.
                reverse = matrix[second_job][first_job]
                reverse_holds = (
                    second_before_first if reverse == 1 else first_before_second
                )
                if not reverse_holds:
                    expected = (
                        "second job before first"
                        if reverse == 1
                        else "first job before second"
                    )
                    violations.append(
                        f"C2 | machine {machine} precedence[{second_job}]"
                        f"[{first_job}] declares {expected}, contrary to start_times"
                    )


def _check_c3_makespan(operations, starts, makespan, violations):
    """C3: makespan is no smaller than every job's last completion time."""
    if operations and not operations[0]:
        violations.append(
            "C3 | a job has no last operation, so its completion time cannot be checked"
        )
        return

    for job_index, row in enumerate(operations):
        operation_index = len(row) - 1
        last_start = starts[job_index][operation_index]
        duration = row[operation_index][1]
        relation_holds = _starts_after_or_at(makespan, last_start, duration)
        if relation_holds is None:
            violations.append(
                f"C3 | completion time overflowed for job {job_index}'s last operation"
            )
        elif not relation_holds:
            violations.append(
                f"C3 | makespan is smaller than job {job_index}'s recomputed "
                "last-operation completion time"
            )


def check_solution(instance: dict, solution: dict) -> list:
    """Return concrete violations; an empty list means the solution is feasible."""
    violations = []
    dimensions = _read_dimensions(instance, violations)
    if not isinstance(solution, dict):
        violations.append("FORMAT | solution must be a dict")
        return violations
    if dimensions is None:
        return violations

    n_jobs, n_machines = dimensions
    operations = _read_operations(instance, n_jobs, n_machines, violations)
    starts = _read_start_times(solution, n_jobs, n_machines, violations)
    precedence = _read_precedence(solution, n_jobs, n_machines, violations)
    makespan = _read_makespan(solution, violations)

    # C1 | Precedence constraint on operations within a job
    if operations is not None and starts is not None:
        _check_c1_job_order(operations, starts, violations)

    # C2 | Machine non-overlap constraint (disjunctive constraint)
    if operations is not None and starts is not None and precedence is not None:
        _check_c2_machine_nonoverlap(operations, starts, precedence, violations)

    # C3 | Makespan definition
    if operations is not None and starts is not None and makespan is not None:
        _check_c3_makespan(operations, starts, makespan, violations)

    return violations
