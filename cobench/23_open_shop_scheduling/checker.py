"""Feasibility checker for the OSSP instance described in MAT_s4.md."""

from __future__ import annotations

import math


_REL = 1e-9


def _is_number(value: object) -> bool:
    """Return whether value is a finite JSON-style number, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _geq(a: int | float, b: int | float, *, calculated_with_float: bool) -> bool:
    """Compare exactly for integer arithmetic and relatively for float arithmetic."""
    if not calculated_with_float:
        return a >= b
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _extract_data(instance: dict, solution: dict):
    """Safely obtain the only required decision representation and its dimensions."""
    violations: list[str] = []

    if not isinstance(instance, dict):
        return None, ["FORMAT | instance must be a dictionary"]
    if not isinstance(solution, dict):
        return None, ["FORMAT | solution must be a dictionary"]

    n_jobs = instance.get("n_jobs")
    n_machines = instance.get("n_machines")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
        violations.append("FORMAT | instance['n_jobs'] must be an integer")
    if isinstance(n_machines, bool) or not isinstance(n_machines, int):
        violations.append("FORMAT | instance['n_machines'] must be an integer")
    if violations:
        return None, violations

    ptimes = instance.get("ptimes")
    if not _is_sequence(ptimes):
        return None, ["FORMAT | instance['ptimes'] must be a row sequence"]
    if len(ptimes) != n_jobs:
        return None, [
            "FORMAT | instance['ptimes'] row count "
            f"{len(ptimes)} does not match n_jobs={n_jobs}"
        ]

    normalized_ptimes: list[list[int | float]] = []
    for j, row in enumerate(ptimes):
        if not _is_sequence(row):
            violations.append(f"FORMAT | instance['ptimes'][{j}] must be a sequence")
            continue
        if len(row) != n_machines:
            violations.append(
                f"FORMAT | instance['ptimes'][{j}] has {len(row)} entries; "
                f"expected n_machines={n_machines}"
            )
            continue
        normalized_row: list[int | float] = []
        for m, value in enumerate(row):
            if not _is_number(value):
                violations.append(
                    f"FORMAT | instance['ptimes'][{j}][{m}] must be a finite number"
                )
            else:
                normalized_row.append(value)
        normalized_ptimes.append(normalized_row)

    if "start" not in solution:
        violations.append("FORMAT | missing required solution field 'start'")
        return None, violations
    start = solution["start"]
    if not _is_sequence(start):
        violations.append("FORMAT | solution['start'] must be a row sequence")
        return None, violations
    if len(start) != n_jobs:
        violations.append(
            f"FORMAT | solution['start'] has {len(start)} rows; expected n_jobs={n_jobs}"
        )
        return None, violations

    normalized_start: list[list[int | float]] = []
    for j, row in enumerate(start):
        if not _is_sequence(row):
            violations.append(f"FORMAT | solution['start'][{j}] must be a sequence")
            continue
        if len(row) != n_machines:
            violations.append(
                f"FORMAT | solution['start'][{j}] has {len(row)} entries; "
                f"expected n_machines={n_machines}"
            )
            continue
        normalized_row = []
        for m, value in enumerate(row):
            if not _is_number(value):
                violations.append(
                    f"FORMAT | solution['start'][{j}][{m}] must be a finite number"
                )
                continue
            # Direct solution values are exact inputs: every negative value violates
            # the authoritative domain s[j][m] >= 0, irrespective of its magnitude.
            if value < 0:
                violations.append(
                    f"DOMAIN | start[{j}][{m}]={value!r} violates s[j][m] >= 0"
                )
            normalized_row.append(value)
        normalized_start.append(normalized_row)

    if violations:
        return None, violations
    return (n_jobs, n_machines, normalized_ptimes, normalized_start), []


def _check_c1(
    n_jobs: int,
    n_machines: int,
    ptimes: list[list[int | float]],
    start: list[list[int | float]],
) -> list[str]:
    """C1: construct Cmax and verify every operation finishes no later than it."""
    completions = [
        (start[j][m] + ptimes[j][m], isinstance(start[j][m], float) or isinstance(ptimes[j][m], float))
        for j in range(n_jobs)
        for m in range(n_machines)
    ]
    cmax = max([0, *(completion for completion, _ in completions)])

    violations: list[str] = []
    for index, (completion, used_float) in enumerate(completions):
        if not _geq(cmax, completion, calculated_with_float=used_float or isinstance(cmax, float)):
            j, m = divmod(index, n_machines)
            violations.append(
                f"C1 | completion start[{j}][{m}] + ptimes[{j}][{m}]="
                f"{completion!r} exceeds reconstructed Cmax={cmax!r}"
            )
    return violations


def _check_c2(
    n_jobs: int,
    n_machines: int,
    ptimes: list[list[int | float]],
    start: list[list[int | float]],
    big_m: int | float,
) -> list[str]:
    """C2: existentially recover each omitted zJ and check both inequalities."""
    violations: list[str] = []
    for j in range(n_jobs):
        for m1 in range(n_machines):
            for m2 in range(m1 + 1, n_machines):
                s1, s2 = start[j][m1], start[j][m2]
                p1, p2 = ptimes[j][m1], ptimes[j][m2]
                used_float = any(isinstance(value, float) for value in (s1, s2, p1, p2, big_m))

                has_valid_z = False
                for z_j in (0, 1):
                    first_rhs = s1 + p1 - big_m * z_j
                    second_rhs = s2 + p2 - big_m * (1 - z_j)
                    if _geq(s2, first_rhs, calculated_with_float=used_float) and _geq(
                        s1, second_rhs, calculated_with_float=used_float
                    ):
                        has_valid_z = True
                        break

                if not has_valid_z:
                    violations.append(
                        f"C2 | job {j}, machines ({m1}, {m2}) admit no zJ in {{0,1}} "
                        "that satisfies both finite Big-M inequalities "
                        f"(s1={s1!r}, p1={p1!r}, s2={s2!r}, p2={p2!r}, M={big_m!r})"
                    )
    return violations


def _check_c3(
    n_jobs: int,
    n_machines: int,
    ptimes: list[list[int | float]],
    start: list[list[int | float]],
    big_m: int | float,
) -> list[str]:
    """C3: existentially recover each omitted zM and check both inequalities."""
    violations: list[str] = []
    for m in range(n_machines):
        for j1 in range(n_jobs):
            for j2 in range(j1 + 1, n_jobs):
                s1, s2 = start[j1][m], start[j2][m]
                p1, p2 = ptimes[j1][m], ptimes[j2][m]
                used_float = any(isinstance(value, float) for value in (s1, s2, p1, p2, big_m))

                has_valid_z = False
                for z_m in (0, 1):
                    first_rhs = s1 + p1 - big_m * z_m
                    second_rhs = s2 + p2 - big_m * (1 - z_m)
                    if _geq(s2, first_rhs, calculated_with_float=used_float) and _geq(
                        s1, second_rhs, calculated_with_float=used_float
                    ):
                        has_valid_z = True
                        break

                if not has_valid_z:
                    violations.append(
                        f"C3 | machine {m}, jobs ({j1}, {j2}) admit no zM in {{0,1}} "
                        "that satisfies both finite Big-M inequalities "
                        f"(s1={s1!r}, p1={p1!r}, s2={s2!r}, p2={p2!r}, M={big_m!r})"
                    )
    return violations


def check_solution(instance: dict, solution: dict) -> list:
    """Return all detected violations; an empty list means the solution is feasible."""
    try:
        data, violations = _extract_data(instance, solution)
        if data is None:
            return violations

        n_jobs, n_machines, ptimes, start = data
        big_m = sum(value for row in ptimes for value in row)

        violations.extend(_check_c1(n_jobs, n_machines, ptimes, start))
        violations.extend(_check_c2(n_jobs, n_machines, ptimes, start, big_m))
        violations.extend(_check_c3(n_jobs, n_machines, ptimes, start, big_m))
        return violations
    except Exception as exc:  # Last-resort containment for malformed/hostile inputs.
        return [f"FORMAT | checker could not evaluate solution: {type(exc).__name__}: {exc}"]
