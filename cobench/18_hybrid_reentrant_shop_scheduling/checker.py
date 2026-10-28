"""Feasibility checker for the hybrid reentrant shop schedule in MAT_s4.md."""

from __future__ import annotations

import math
import re
from numbers import Integral, Real


_REL = 1e-9
_JOB_NUMBER = re.compile(r"[0-9]+")
_REQUIRED_FIELDS = (
    "machine_assignment",
    "server_sequence",
    "start_times",
    "makespan",
)


def _leq(a: float, b: float) -> bool:
    """Compare continuously computed values on their original linear scale."""
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _geq(a: float, b: float) -> bool:
    """Compare continuously computed values on their original linear scale."""
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _equal(a: float, b: float) -> bool:
    return _leq(a, b) and _geq(a, b)


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _job_index(value: str) -> int | None:
    """Parse any string that contains exactly one unambiguous integer index."""
    matches = _JOB_NUMBER.findall(value.strip())
    if len(matches) != 1:
        return None
    try:
        return int(matches[0])
    except ValueError:
        return None


def _read_instance(instance: object, violations: list[str]):
    """Read only the structure needed to evaluate a solution; impose no value domain."""
    if not isinstance(instance, dict):
        violations.append("FORMAT | instance must be a dict so its scheduling data can be read")
        return None

    names = (
        "num_jobs",
        "num_machines",
        "init_processing_times",
        "setup_processing_times",
        "main_processing_times",
    )
    missing = [name for name in names if name not in instance]
    if missing:
        violations.append(
            "FORMAT | instance is missing fields needed to recompute feasibility: "
            + ", ".join(missing)
        )
        return None

    n = instance["num_jobs"]
    m = instance["num_machines"]
    if not isinstance(n, Integral) or isinstance(n, bool):
        violations.append("FORMAT | instance num_jobs must be an int")
    if not isinstance(m, Integral) or isinstance(m, bool):
        violations.append("FORMAT | instance num_machines must be an int")
    if violations:
        return None

    arrays = []
    for name in names[2:]:
        values = instance[name]
        if not isinstance(values, list):
            violations.append(f"FORMAT | instance {name} must be a list")
            continue
        if len(values) < max(0, n):
            violations.append(
                f"FORMAT | instance {name} has {len(values)} entries, fewer than num_jobs={n}"
            )
            continue
        bad = next(
            (index for index in range(max(0, n)) if not _finite_number(values[index])),
            None,
        )
        if bad is not None:
            violations.append(
                f"FORMAT | instance {name}[{bad}] must be a finite number for arithmetic checks"
            )
            continue
        arrays.append(values)

    if violations:
        return None
    return int(n), int(m), arrays[0], arrays[1], arrays[2]


def _read_solution(solution: object, n: int, violations: list[str]):
    """Validate the explicitly declared output shapes and normalize job labels."""
    if not isinstance(solution, dict):
        violations.append("FORMAT | solution must be a dict")
        return None

    missing = [name for name in _REQUIRED_FIELDS if name not in solution]
    if missing:
        violations.append(
            "FORMAT | solution is missing required fields: " + ", ".join(missing)
        )
        return None

    assignments_raw = solution["machine_assignment"]
    sequence_raw = solution["server_sequence"]
    starts_raw = solution["start_times"]
    makespan = solution["makespan"]

    if not isinstance(assignments_raw, dict):
        violations.append("FORMAT | machine_assignment must have shape dict[str,int]")
    if not isinstance(sequence_raw, list):
        violations.append("FORMAT | server_sequence must have shape list[str]")
    if not isinstance(starts_raw, dict):
        violations.append("FORMAT | start_times must have shape dict[str,list[float]]")
    if not _finite_number(makespan):
        violations.append("FORMAT | makespan must be a finite numeric value")
    if violations:
        return None

    assignments: dict[int, int] = {}
    for label, machine in assignments_raw.items():
        if not isinstance(label, str):
            violations.append(
                "FORMAT | every machine_assignment key must be a string as required by dict[str,int]"
            )
            continue
        job = _job_index(label)
        if job is None:
            violations.append(
                f"FORMAT | machine_assignment key {label!r} does not contain one uniquely parseable job index"
            )
            continue
        if job in assignments:
            violations.append(
                f"C1 | machine_assignment labels resolve more than once to job {job}, so it has no single assignment"
            )
            continue
        if not isinstance(machine, Integral) or isinstance(machine, bool):
            violations.append(
                f"FORMAT | machine_assignment[{label!r}] must be an int machine index"
            )
            continue
        assignments[job] = int(machine)

    sequence: list[int] = []
    for position, label in enumerate(sequence_raw):
        if not isinstance(label, str):
            violations.append(
                f"FORMAT | server_sequence[{position}] must be a string as required by list[str]"
            )
            continue
        job = _job_index(label)
        if job is None:
            violations.append(
                f"FORMAT | server_sequence[{position}]={label!r} does not contain one uniquely parseable job index"
            )
            continue
        sequence.append(job)

    starts: dict[int, tuple[float, float, float]] = {}
    for label, values in starts_raw.items():
        if not isinstance(label, str):
            violations.append(
                "FORMAT | every start_times key must be a string as required by dict[str,list[float]]"
            )
            continue
        job = _job_index(label)
        if job is None:
            violations.append(
                f"FORMAT | start_times key {label!r} does not contain one uniquely parseable job index"
            )
            continue
        if job in starts:
            violations.append(
                f"FORMAT | start_times labels resolve more than once to job {job}, so its times are ambiguous"
            )
            continue
        if not isinstance(values, list):
            violations.append(
                f"FORMAT | start_times[{label!r}] must be a list, not {type(values).__name__}"
            )
            continue
        if len(values) != 3:
            violations.append(
                f"FORMAT | start_times[{label!r}] must contain exactly [s1, s2, s3]"
            )
            continue
        if any(not _finite_number(value) for value in values):
            violations.append(
                f"FORMAT | start_times[{label!r}] must contain three finite numeric values"
            )
            continue
        starts[job] = (float(values[0]), float(values[1]), float(values[2]))

    if violations:
        return None

    expected_count = max(0, n)
    if len(starts) != expected_count or any(job < 0 or job >= n for job in starts):
        violations.append(
            f"FORMAT | start_times must resolve exactly once to every job index 0..{n - 1}"
        )
    if violations:
        return None

    return assignments, sequence, starts, float(makespan)


def check_solution(instance: dict, solution: dict) -> list:
    """Return concrete violations; an empty list means the schedule is feasible."""
    violations: list[str] = []
    try:
        instance_data = _read_instance(instance, violations)
        if instance_data is None:
            return violations
        n, m, p1, p2, p3 = instance_data

        solution_data = _read_solution(solution, n, violations)
        if solution_data is None:
            return violations
        assignments, sequence, starts, makespan = solution_data

        # Decision-variable domains explicitly stated in MAT_s4.md.
        for job in range(n):
            s1, s2, s3 = starts[job]
            for phase, value in (("s1", s1), ("s2", s2), ("s3", s3)):
                if value < 0:
                    violations.append(
                        f"DOMAIN | {phase}[{job}]={value} is negative, but the declared domain is {phase}[j] >= 0"
                    )

        # C1: every job has exactly one assignment to an existing primary machine.
        if len(assignments) != max(0, n) or any(
            job < 0 or job >= n for job in assignments
        ):
            violations.append(
                f"C1 | machine_assignment must resolve exactly once to every job index 0..{n - 1}"
            )
        for job, machine in assignments.items():
            if machine < 0 or machine >= m:
                violations.append(
                    f"C1 | job {job} is assigned to machine {machine}, outside the declared indices 0..{m - 1}"
                )

        # C2: lower-indexed jobs initialize first whenever they share a machine.
        for later in range(n):
            if later not in assignments:
                continue
            for earlier in range(later):
                if assignments.get(earlier) != assignments[later]:
                    continue
                required = starts[earlier][0] + p1[earlier]
                if not _geq(starts[later][0], required):
                    violations.append(
                        f"C2 | job {later} starts initialization at {starts[later][0]}, before natural-order job "
                        f"{earlier} completes initialization at {required} on machine {assignments[later]}"
                    )

        # C3: no two initialization phases on one primary machine overlap.
        for first in range(n):
            if first not in assignments:
                continue
            for second in range(first + 1, n):
                if assignments.get(second) != assignments[first]:
                    continue
                first_end = starts[first][0] + p1[first]
                second_end = starts[second][0] + p1[second]
                if not (
                    _leq(first_end, starts[second][0])
                    or _leq(second_end, starts[first][0])
                ):
                    violations.append(
                        f"C3 | initialization intervals of jobs {first} and {second} overlap on machine "
                        f"{assignments[first]}"
                    )

        # C4: each setup starts only after that job's initialization completes.
        for job in range(n):
            init_end = starts[job][0] + p1[job]
            if not _geq(starts[job][1], init_end):
                violations.append(
                    f"C4 | job {job} starts setup at {starts[job][1]}, before its initialization completes at {init_end}"
                )

        # C5: setup phases follow server_sequence and never overlap on the server.
        if len(sequence) == max(0, n) and all(0 <= job < n for job in sequence):
            for later_position, later_job in enumerate(sequence):
                for earlier_position in range(later_position):
                    earlier_job = sequence[earlier_position]
                    earlier_end = starts[earlier_job][1] + p2[earlier_job]
                    if not _geq(starts[later_job][1], earlier_end):
                        violations.append(
                            f"C5 | server-position {later_position} job {later_job} starts setup at "
                            f"{starts[later_job][1]}, before earlier-position job {earlier_job} completes at {earlier_end}"
                        )

        # C6: each main-processing phase starts only after that job's setup completes.
        for job in range(n):
            setup_end = starts[job][1] + p2[job]
            if not _geq(starts[job][2], setup_end):
                violations.append(
                    f"C6 | job {job} starts main processing at {starts[job][2]}, before its setup completes at {setup_end}"
                )

        # C7: main phases use natural order, and all same-machine initialization is finished first.
        for later in range(n):
            if later not in assignments:
                continue
            for earlier in range(later):
                if assignments.get(earlier) != assignments[later]:
                    continue
                earlier_main_end = starts[earlier][2] + p3[earlier]
                if not _geq(starts[later][2], earlier_main_end):
                    violations.append(
                        f"C7 | job {later} starts main processing at {starts[later][2]}, before natural-order "
                        f"job {earlier} completes main processing at {earlier_main_end} on machine {assignments[later]}"
                    )
        for main_job in range(n):
            if main_job not in assignments:
                continue
            for init_job in range(n):
                if assignments.get(init_job) != assignments[main_job]:
                    continue
                init_end = starts[init_job][0] + p1[init_job]
                if not _geq(starts[main_job][2], init_end):
                    violations.append(
                        f"C7 | job {main_job} starts main processing at {starts[main_job][2]}, before job "
                        f"{init_job} completes initialization at {init_end} on machine {assignments[main_job]}"
                    )

        # C8: C_max obeys every completion lower bound, its nonnegative domain, and minimization.
        completions = [starts[job][2] + p3[job] for job in range(n)]
        if makespan < 0:
            violations.append(
                f"C8 | makespan={makespan} is negative, but the declared domain is C_max >= 0"
            )
        for job, completion in enumerate(completions):
            if not _geq(makespan, completion):
                violations.append(
                    f"C8 | makespan={makespan} is below job {job}'s recomputed main completion {completion}"
                )
        recomputed_minimum = max([0.0, *completions])
        if not _equal(makespan, recomputed_minimum):
            violations.append(
                f"C8 | makespan={makespan} differs from the recomputed minimum C_max={recomputed_minimum}, "
                "which is max(0, all main-processing completion times)"
            )

        # C9: sequence positions are a duplicate-free permutation covering all jobs.
        expected_count = max(0, n)
        if len(sequence) != expected_count:
            violations.append(
                f"C9 | server_sequence has {len(sequence)} positions, but must have one for each of {n} jobs"
            )
        out_of_range = [job for job in sequence if job < 0 or job >= n]
        if out_of_range:
            violations.append(
                f"C9 | server_sequence contains out-of-range job indices: {out_of_range}"
            )
        seen: set[int] = set()
        duplicate_set: set[int] = set()
        for job in sequence:
            if job in seen:
                duplicate_set.add(job)
            seen.add(job)
        duplicates = sorted(duplicate_set)
        if duplicates:
            violations.append(
                f"C9 | server_sequence repeats job indices {duplicates}, so positions are not AllDifferent"
            )
        if (
            len(sequence) == expected_count
            and not out_of_range
            and len(set(sequence)) != expected_count
        ):
            violations.append(
                "C9 | server_sequence does not cover every job exactly once"
            )

        return violations
    except Exception as exc:  # Last-resort robustness for malformed Python objects.
        return [
            f"FORMAT | checker could not safely inspect the malformed input ({type(exc).__name__}: {exc})"
        ]
