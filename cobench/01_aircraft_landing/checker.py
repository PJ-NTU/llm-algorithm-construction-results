"""Feasibility checker for the aircraft landing problem in MAT_s4.md."""

from __future__ import annotations

import math
from numbers import Real


_REQUIRED_SOLUTION_FIELDS = (
    "runway_assignment",
    "landing_times",
    "landing_sequence",
)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _numeric_vector(
    instance: dict,
    field: str,
    expected_length: int,
    violations: list[str],
) -> list[Real] | None:
    value = instance.get(field)
    if not isinstance(value, list):
        violations.append(f"FORMAT | instance.{field} must be a list")
        return None
    if len(value) != expected_length:
        violations.append(
            f"FORMAT | instance.{field} must have {expected_length} entries, got {len(value)}"
        )
        return None
    if not all(_is_finite_number(item) for item in value):
        violations.append(f"FORMAT | instance.{field} entries must be finite numbers")
        return None
    return value


def _separation_tensor(
    instance: dict,
    plane_count: int,
    runway_count: int,
    violations: list[str],
) -> list | None:
    tensor = instance.get("separation_matrix")
    if not isinstance(tensor, list) or len(tensor) != plane_count:
        violations.append(
            "FORMAT | instance.separation_matrix must have shape "
            f"[{plane_count}][{plane_count}][{runway_count}][{runway_count}]"
        )
        return None

    for i_axis in tensor:
        if not isinstance(i_axis, list) or len(i_axis) != plane_count:
            violations.append(
                "FORMAT | instance.separation_matrix must have shape "
                f"[{plane_count}][{plane_count}][{runway_count}][{runway_count}]"
            )
            return None
        for j_axis in i_axis:
            if not isinstance(j_axis, list) or len(j_axis) != runway_count:
                violations.append(
                    "FORMAT | instance.separation_matrix must have shape "
                    f"[{plane_count}][{plane_count}][{runway_count}][{runway_count}]"
                )
                return None
            for r_axis in j_axis:
                if not isinstance(r_axis, list) or len(r_axis) != runway_count:
                    violations.append(
                        "FORMAT | instance.separation_matrix must have shape "
                        f"[{plane_count}][{plane_count}][{runway_count}][{runway_count}]"
                    )
                    return None
                if not all(_is_finite_number(item) for item in r_axis):
                    violations.append(
                        "FORMAT | instance.separation_matrix entries must be finite numbers"
                    )
                    return None
    return tensor


def _roundoff_tolerance(*values: Real) -> float:
    """Bound a few ULPs of error in arithmetic performed by this checker."""
    return 8.0 * max(math.ulp(float(value)) for value in values)


def _computed_lower_bound_violated(
    actual: Real,
    base: Real,
    increment: Real,
) -> tuple[bool, Real]:
    """Check actual >= base + increment, allowing only addition roundoff."""
    required = base + increment
    if not _is_finite_number(required):
        return True, required
    tolerance = _roundoff_tolerance(actual, base, increment, required)
    return float(actual) + tolerance < float(required), required


def check_solution(instance: dict, solution: dict) -> list:
    """Return concrete constraint violations; an empty list means feasible."""
    violations: list[str] = []

    try:
        if not isinstance(instance, dict):
            return ["FORMAT | instance must be a dict"]
        if not isinstance(solution, dict):
            return ["FORMAT | solution must be a dict"]

        missing = [field for field in _REQUIRED_SOLUTION_FIELDS if field not in solution]
        if missing:
            return ["FORMAT | missing required solution field(s): " + ", ".join(missing)]

        planes = instance.get("planes")
        runways = instance.get("runways")
        if not isinstance(planes, list) or not all(_is_int(plane) for plane in planes):
            violations.append("FORMAT | instance.planes must be a list of integer IDs")
        elif len(set(planes)) != len(planes):
            violations.append("FORMAT | instance.planes must not contain duplicate IDs")
        if not isinstance(runways, list) or not all(_is_int(runway) for runway in runways):
            violations.append("FORMAT | instance.runways must be a list of integer IDs")
        elif len(set(runways)) != len(runways):
            violations.append("FORMAT | instance.runways must not contain duplicate IDs")
        if violations:
            return violations

        plane_count = len(planes)
        runway_count = len(runways)
        plane_labels = [str(plane) for plane in planes]
        expected_plane_labels = set(plane_labels)

        earliest = _numeric_vector(
            instance, "earliest_landing_time", plane_count, violations
        )
        target = _numeric_vector(
            instance, "target_landing_time", plane_count, violations
        )
        latest = _numeric_vector(
            instance, "latest_landing_time", plane_count, violations
        )
        separation = _separation_tensor(
            instance, plane_count, runway_count, violations
        )
        if violations:
            return violations

        assignments = solution["runway_assignment"]
        landing_times = solution["landing_times"]
        sequence = solution["landing_sequence"]
        if not isinstance(assignments, dict):
            violations.append("FORMAT | runway_assignment must be a dict[str, int]")
        if not isinstance(landing_times, dict):
            violations.append("FORMAT | landing_times must be a dict[str, float]")
        if not isinstance(sequence, list):
            violations.append("FORMAT | landing_sequence must be a list[str]")
        if violations:
            return violations

        assignment_keys_valid = all(isinstance(key, str) for key in assignments)
        time_keys_valid = all(isinstance(key, str) for key in landing_times)
        sequence_items_valid = all(isinstance(label, str) for label in sequence)
        if not assignment_keys_valid:
            violations.append("FORMAT | runway_assignment keys must be strings")
        if not time_keys_valid:
            violations.append("FORMAT | landing_times keys must be strings")
        if not sequence_items_valid:
            violations.append("FORMAT | landing_sequence entries must be strings")
        if violations:
            return violations

        # C1 | Each plane has one assignment in this mapping, and it must be a runway in R.
        assignment_labels = set(assignments)
        if assignment_labels != expected_plane_labels:
            missing_labels = sorted(expected_plane_labels - assignment_labels)
            extra_labels = sorted(assignment_labels - expected_plane_labels)
            violations.append(
                "C1 | runway_assignment must contain exactly all planes; "
                f"missing={missing_labels}, extra={extra_labels}"
            )
        for label in sorted(assignment_labels & expected_plane_labels):
            runway = assignments[label]
            if not _is_int(runway) or runway not in runways:
                violations.append(
                    f"C1 | plane {label} is assigned to invalid runway {runway!r}"
                )

        time_labels = set(landing_times)
        if time_labels != expected_plane_labels:
            missing_labels = sorted(expected_plane_labels - time_labels)
            extra_labels = sorted(time_labels - expected_plane_labels)
            violations.append(
                "FORMAT | landing_times must contain exactly all planes; "
                f"missing={missing_labels}, extra={extra_labels}"
            )
        for label in sorted(time_labels & expected_plane_labels):
            if not _is_finite_number(landing_times[label]):
                violations.append(
                    f"FORMAT | landing_times[{label!r}] must be a finite number"
                )

        sequence_is_permutation = (
            len(sequence) == plane_count
            and len(set(sequence)) == len(sequence)
            and set(sequence) == expected_plane_labels
        )
        if not sequence_is_permutation:
            violations.append(
                "C5 | landing_sequence must contain every plane exactly once"
            )

        if violations:
            return violations

        plane_axis = {label: index for index, label in enumerate(plane_labels)}
        runway_axis = {runway: index for index, runway in enumerate(runways)}

        # C2 | Every reported landing time is directly inside its instance time window.
        for label in plane_labels:
            index = plane_axis[label]
            landing = landing_times[label]
            if landing < earliest[index] or landing > latest[index]:
                violations.append(
                    f"C2 | plane {label} lands at {landing!r}, outside "
                    f"[{earliest[index]!r}, {latest[index]!r}]"
                )

        # C3 | e is absent from the required output, so derive e=max(0, T-t) ourselves.
        for label in plane_labels:
            index = plane_axis[label]
            required_earliness = target[index] - landing_times[label]
            derived_earliness = max(0, required_earliness)
            if not _is_finite_number(required_earliness) or not _is_finite_number(
                derived_earliness
            ):
                violations.append(
                    f"C3 | plane {label} has non-finite derived earliness"
                )
            elif derived_earliness < required_earliness:
                violations.append(
                    f"C3 | plane {label} derived earliness does not satisfy e >= T-t"
                )

        # C4 | l is absent from the required output, so derive l=max(0, t-T) ourselves.
        for label in plane_labels:
            index = plane_axis[label]
            required_lateness = landing_times[label] - target[index]
            derived_lateness = max(0, required_lateness)
            if not _is_finite_number(required_lateness) or not _is_finite_number(
                derived_lateness
            ):
                violations.append(
                    f"C4 | plane {label} has non-finite derived lateness"
                )
            elif derived_lateness < required_lateness:
                violations.append(
                    f"C4 | plane {label} derived lateness does not satisfy l >= t-T"
                )

        # C5 | A complete, duplicate-free sequence defines exactly one order per pair.
        for earlier_position in range(plane_count):
            earlier_label = sequence[earlier_position]
            for later_position in range(earlier_position + 1, plane_count):
                later_label = sequence[later_position]
                if landing_times[earlier_label] > landing_times[later_label]:
                    violations.append(
                        f"C5 | sequence puts plane {earlier_label} before {later_label}, "
                        f"but their landing times are {landing_times[earlier_label]!r} "
                        f"and {landing_times[later_label]!r}"
                    )

        # C6 | For the sequence-selected order, enforce S[i,j,r,r] on a shared runway.
        # C7 | For the same order, enforce S[i,j,r,k] when the runways differ.
        for earlier_position in range(plane_count):
            earlier_label = sequence[earlier_position]
            i = plane_axis[earlier_label]
            earlier_runway = assignments[earlier_label]
            r = runway_axis[earlier_runway]
            for later_position in range(earlier_position + 1, plane_count):
                later_label = sequence[later_position]
                j = plane_axis[later_label]
                later_runway = assignments[later_label]
                k = runway_axis[later_runway]
                gap = separation[i][j][r][k]
                is_violated, required_time = _computed_lower_bound_violated(
                    landing_times[later_label], landing_times[earlier_label], gap
                )
                if is_violated:
                    constraint = "C6" if earlier_runway == later_runway else "C7"
                    runway_kind = "same-runway" if constraint == "C6" else "cross-runway"
                    violations.append(
                        f"{constraint} | {runway_kind} separation violated for "
                        f"{earlier_label}->{later_label}: later time "
                        f"{landing_times[later_label]!r} is below required "
                        f"{required_time!r} (S={gap!r})"
                    )

        return violations
    except Exception as error:  # Defensive boundary: malformed inputs must never escape.
        return [
            "FORMAT | checker could not safely read the supplied data: "
            f"{type(error).__name__}: {error}"
        ]
