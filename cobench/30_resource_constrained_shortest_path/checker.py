"""Feasibility checker for the constrained shortest-path output in MAT_s4.md."""

from decimal import Decimal, InvalidOperation
import math
import re


_REL = Decimal("1e-9")
_MISSING = object()
_DIGIT_RUN = re.compile(r"\d+")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value):
    return _is_number(value) and (isinstance(value, int) or math.isfinite(value))


def _brief(value, limit=120):
    try:
        rendered = repr(value)
    except Exception:
        return f"<{type(value).__name__}>"
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def _as_decimal(value):
    try:
        return Decimal(value) if isinstance(value, int) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _leq_continuous(a, b):
    """Relative-tolerance comparison for quantities involving a float."""
    left = _as_decimal(a)
    right = _as_decimal(b)
    if left is None or right is None:
        return False
    return left - right <= _REL * max(Decimal(1), abs(left), abs(right))


def _geq_continuous(a, b):
    """Relative-tolerance comparison for quantities involving a float."""
    left = _as_decimal(a)
    right = _as_decimal(b)
    if left is None or right is None:
        return False
    return right - left <= _REL * max(Decimal(1), abs(left), abs(right))


def _leq_classified(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a <= b
    return _leq_continuous(a, b)


def _geq_classified(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a >= b
    return _geq_continuous(a, b)


def _geq_accumulated(result, *terms):
    """Compare an accumulated resource, exactly for all-int arithmetic."""
    if isinstance(result, int) and all(isinstance(term, int) for term in terms):
        return result >= sum(terms)

    left = _as_decimal(result)
    decimal_terms = [_as_decimal(term) for term in terms]
    if left is None or any(term is None for term in decimal_terms):
        return False
    right = sum(decimal_terms, Decimal(0))
    return right - left <= _REL * max(Decimal(1), abs(left), abs(right))


def _parse_vertex_key(key, n):
    """Return a 1-based vertex id, accepting harmless key spelling variants."""
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        return key if 1 <= key <= n else None
    if not isinstance(key, str):
        return None

    stripped = key.strip()
    runs = _DIGIT_RUN.findall(stripped)
    if len(runs) != 1:
        return None
    if stripped.startswith("-"):
        return None

    digits = runs[0].lstrip("0") or "0"
    n_digits = str(n)
    if len(digits) > len(n_digits):
        return None
    if len(digits) == len(n_digits) and digits > n_digits:
        return None
    try:
        vertex = int(digits)
    except (ValueError, TypeError):
        return None
    return vertex if 1 <= vertex <= n else None


def _read_count(instance, field, violations):
    value = instance.get(field, _MISSING)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        violations.append(
            f"FORMAT | instance field {field!r} cannot be used as a nonnegative integer count"
        )
        return None
    return value


def _read_arcs(instance, violations):
    raw_arcs = instance.get("arcs", _MISSING)
    if not isinstance(raw_arcs, list):
        violations.append("FORMAT | instance field 'arcs' is unavailable as a list")
        return []

    arcs = []
    for index, raw_arc in enumerate(raw_arcs):
        if not isinstance(raw_arc, (list, tuple)) or len(raw_arc) < 2:
            violations.append(f"FORMAT | instance arc {index} has no readable endpoints")
            continue
        tail, head = raw_arc[0], raw_arc[1]
        if (
            not isinstance(tail, int)
            or isinstance(tail, bool)
            or not isinstance(head, int)
            or isinstance(head, bool)
        ):
            violations.append(f"FORMAT | instance arc {index} endpoints are not integers")
            continue
        arcs.append((tail, head, index))
    return arcs


def _read_matrix_number(instance, field, row, column, constraint, violations):
    matrix = instance.get(field, _MISSING)
    try:
        value = matrix[row][column]
    except (KeyError, IndexError, TypeError):
        violations.append(
            f"{constraint} | instance field {field!r} has no value at [{row}][{column}], "
            "so this constraint cannot be evaluated"
        )
        return _MISSING
    if not _is_finite_number(value):
        violations.append(
            f"{constraint} | instance field {field!r}[{row}][{column}] is not a finite number, "
            "so this constraint cannot be evaluated"
        )
        return _MISSING
    return value


def _normalize_resources(raw_resources, n, resource_types, violations):
    normalized = {}
    if not isinstance(raw_resources, dict):
        violations.append("FORMAT | 'resource_accumulations' must be a dictionary")
        return normalized

    for raw_key, values in raw_resources.items():
        vertex = _parse_vertex_key(raw_key, n)
        if vertex is None:
            violations.append(
                f"FORMAT | resource key {_brief(raw_key)} cannot be uniquely parsed "
                f"as a vertex in 1..{n}"
            )
            continue
        if vertex in normalized:
            violations.append(
                f"FORMAT | resource keys are ambiguous because more than one key maps to vertex {vertex}"
            )
            continue
        if not isinstance(values, list):
            violations.append(f"FORMAT | resources for vertex {vertex} must be a list")
            continue
        if len(values) != resource_types:
            violations.append(
                f"FORMAT | vertex {vertex} has {len(values)} resource values; "
                f"expected {resource_types}"
            )
            continue
        bad_dimensions = [
            k for k, value in enumerate(values) if not _is_finite_number(value)
        ]
        if bad_dimensions:
            violations.append(
                f"FORMAT | vertex {vertex} has non-finite or non-numeric resource values "
                f"at dimensions {bad_dimensions}"
            )
            continue
        normalized[vertex] = values

    for vertex in range(1, n + 1):
        if vertex not in normalized:
            violations.append(
                f"FORMAT | resource_accumulations is missing a usable value for vertex {vertex}"
            )
    return normalized


def _check_c1(path, n, arcs, violations):
    """C1: selected binary arcs form the described source-to-destination flow."""
    if n is None:
        violations.append("C1 | vertex count is unavailable, so path flow cannot be evaluated")
        return
    if not isinstance(path, list):
        violations.append("C1 | 'path' must be a list of vertex integers")
        return
    if not path:
        violations.append("C1 | path is empty and therefore does not run from vertex 1 to vertex n")
        return

    bad_positions = [
        index
        for index, vertex in enumerate(path)
        if not isinstance(vertex, int) or isinstance(vertex, bool)
    ]
    if bad_positions:
        violations.append(f"C1 | path has non-integer vertices at positions {bad_positions}")
        return

    outside = sorted({vertex for vertex in path if not 1 <= vertex <= n})
    if outside:
        violations.append(f"C1 | path contains vertices outside 1..{n}: {outside}")

    if path[0] != 1:
        violations.append(f"C1 | path starts at vertex {path[0]!r}, not vertex 1")
    if path[-1] != n:
        violations.append(f"C1 | path ends at vertex {path[-1]!r}, not vertex {n}")

    available_pairs = {(tail, head) for tail, head, _ in arcs}
    transitions = list(zip(path, path[1:]))
    selected_pairs = set()
    seen_pairs = set()
    for tail, head in transitions:
        pair = (tail, head)
        if pair not in available_pairs:
            violations.append(f"C1 | path transition {tail}->{head} is not an arc in A")
            continue
        if pair in seen_pairs:
            violations.append(
                f"C1 | directed arc {tail}->{head} is used more than once, but x[{tail}][{head}] is binary"
            )
        seen_pairs.add(pair)
        selected_pairs.add(pair)

    source_out = sum(1 for tail, _ in selected_pairs if tail == 1)
    if source_out != 1:
        violations.append(f"C1 | selected arcs give source vertex 1 outdegree {source_out}, not 1")

    destination_in = sum(1 for _, head in selected_pairs if head == n)
    if destination_in != 1:
        violations.append(
            f"C1 | selected arcs give destination vertex {n} indegree {destination_in}, not 1"
        )

    for vertex in range(1, n + 1):
        if vertex in (1, n):
            continue
        indegree = sum(1 for _, head in selected_pairs if head == vertex)
        outdegree = sum(1 for tail, _ in selected_pairs if tail == vertex)
        if indegree != outdegree:
            violations.append(
                f"C1 | vertex {vertex} violates flow balance: indegree {indegree}, outdegree {outdegree}"
            )


def _check_c2(resources, n, resource_types, instance, violations):
    """C2: every w is nonnegative and lies within its vertex/resource bounds."""
    if n is None or resource_types is None:
        violations.append("C2 | dimensions are unavailable, so resource bounds cannot be evaluated")
        return
    for vertex in range(1, n + 1):
        values = resources.get(vertex)
        if values is None:
            continue
        for k in range(resource_types):
            value = values[k]
            lower = _read_matrix_number(
                instance, "lower_bounds", vertex - 1, k, "C2", violations
            )
            upper = _read_matrix_number(
                instance, "upper_bounds", vertex - 1, k, "C2", violations
            )
            if not _geq_classified(value, 0):
                violations.append(
                    f"C2 | w[{vertex}][{k}]={value!r} is negative, contrary to the domain w[i][k] >= 0"
                )
            if lower is not _MISSING and not _geq_classified(value, lower):
                violations.append(
                    f"C2 | w[{vertex}][{k}]={value!r} is below lower bound {lower!r}"
                )
            if upper is not _MISSING and not _leq_classified(value, upper):
                violations.append(
                    f"C2 | w[{vertex}][{k}]={value!r} exceeds upper bound {upper!r}"
                )


def _check_c3(path, resources, resource_types, arcs, instance, violations):
    """C3: each path transition admits one arc whose resource propagation holds."""
    if not isinstance(path, list) or resource_types is None:
        violations.append("C3 | path or resource dimension is unavailable, so propagation cannot be evaluated")
        return
    if any(not isinstance(vertex, int) or isinstance(vertex, bool) for vertex in path):
        return

    arc_rows_by_pair = {}
    for tail, head, row in arcs:
        arc_rows_by_pair.setdefault((tail, head), []).append(row)

    for tail, head in zip(path, path[1:]):
        candidate_rows = arc_rows_by_pair.get((tail, head), [])
        if not candidate_rows:
            continue
        tail_values = resources.get(tail)
        head_values = resources.get(head)
        if tail_values is None or head_values is None:
            continue

        candidate_failures = []
        candidate_satisfied = False
        for arc_row in candidate_rows:
            failed_dimensions = []
            candidate_readable = True
            for k in range(resource_types):
                vertex_use = _read_matrix_number(
                    instance,
                    "vertex_resource_consumption",
                    tail - 1,
                    k,
                    "C3",
                    violations,
                )
                arc_use = _read_matrix_number(
                    instance,
                    "arc_resource_consumption",
                    arc_row,
                    k,
                    "C3",
                    violations,
                )
                if vertex_use is _MISSING or arc_use is _MISSING:
                    candidate_readable = False
                    break
                if not _geq_accumulated(
                    head_values[k], tail_values[k], vertex_use, arc_use
                ):
                    failed_dimensions.append(k)
            if candidate_readable and not failed_dimensions:
                candidate_satisfied = True
                break
            if candidate_readable:
                candidate_failures.append((arc_row, failed_dimensions))

        if not candidate_satisfied and candidate_failures:
            detail = ", ".join(
                f"arc row {row}: dimensions {dimensions}"
                for row, dimensions in candidate_failures
            )
            violations.append(
                f"C3 | transition {tail}->{head} has no single arc satisfying resource propagation ({detail})"
            )


def _check_c4(resources, resource_types, instance, violations):
    """C4: source accumulations equal the source vertex's direct resource values."""
    if resource_types is None:
        violations.append("C4 | resource dimension is unavailable, so source initialization cannot be evaluated")
        return
    source_values = resources.get(1)
    if source_values is None:
        return
    for k in range(resource_types):
        expected = _read_matrix_number(
            instance, "vertex_resource_consumption", 0, k, "C4", violations
        )
        if expected is not _MISSING and source_values[k] != expected:
            violations.append(
                f"C4 | w[1][{k}]={source_values[k]!r} does not exactly equal v_res[1][{k}]={expected!r}"
            )


def check_solution(instance: dict, solution: dict) -> list:
    """Return human-readable violations; an empty list means the solution is feasible."""
    violations = []
    try:
        if not isinstance(instance, dict):
            return ["FORMAT | instance must be a dictionary"]
        if not isinstance(solution, dict):
            return ["FORMAT | solution must be a dictionary"]

        n = _read_count(instance, "n", violations)
        resource_types = _read_count(instance, "resource_types", violations)
        arcs = _read_arcs(instance, violations)

        if "path" not in solution:
            violations.append("FORMAT | required field 'path' is missing")
        if "resource_accumulations" not in solution:
            violations.append("FORMAT | required field 'resource_accumulations' is missing")

        path = solution.get("path", _MISSING)
        raw_resources = solution.get("resource_accumulations", _MISSING)
        resources = {}
        if n is not None and resource_types is not None:
            resources = _normalize_resources(
                raw_resources, n, resource_types, violations
            )
        elif raw_resources is not _MISSING and not isinstance(raw_resources, dict):
            violations.append("FORMAT | 'resource_accumulations' must be a dictionary")

        # Exactly four semantic checks, one for each authoritative constraint.
        _check_c1(path, n, arcs, violations)
        _check_c2(resources, n, resource_types, instance, violations)
        _check_c3(path, resources, resource_types, arcs, instance, violations)
        _check_c4(resources, resource_types, instance, violations)
    except Exception as exc:  # Last-resort containment for adversarial Python objects.
        violations.append(
            f"FORMAT | checker could not safely read the supplied data: {type(exc).__name__}: {exc}"
        )
    return violations
