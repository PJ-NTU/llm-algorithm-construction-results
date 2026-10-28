"""Feasibility checker for the maximum independent set solution format."""

from typing import Any


def _read_instance(instance: Any, violations: list[str]) -> tuple[int | None, list | None]:
    """Read only the instance structure needed to evaluate C1."""
    if not isinstance(instance, dict):
        violations.append("FORMAT | instance must be a dict")
        return None, None

    num_vertices = instance.get("num_vertices")
    if type(num_vertices) is not int:
        violations.append("FORMAT | instance.num_vertices must be an int")
        num_vertices = None

    edges = instance.get("edges")
    if not isinstance(edges, list):
        violations.append("FORMAT | instance.edges must be a list")
        edges = None

    return num_vertices, edges


def _read_selection(solution: Any, violations: list[str]) -> list | None:
    """Read the sole required solution field without requiring optional fields."""
    if not isinstance(solution, dict):
        violations.append("FORMAT | solution must be a dict")
        return None

    if "vertex_selection" not in solution:
        violations.append("FORMAT | missing required field: vertex_selection")
        return None

    selection = solution["vertex_selection"]
    if not isinstance(selection, list):
        violations.append("FORMAT | vertex_selection must be a list")
        return None

    return selection


def _selection_has_valid_domain(
    selection: list, num_vertices: int | None, violations: list[str]
) -> bool:
    """Check completeness and the authoritative binary decision-variable domain."""
    valid = True

    if num_vertices is not None and len(selection) != num_vertices:
        violations.append(
            "DECISION | vertex_selection must contain exactly "
            f"{num_vertices} entries, got {len(selection)}"
        )
        valid = False

    for index, value in enumerate(selection):
        # bool is deliberately excluded: the declared output is list[int].
        if type(value) is not int or value not in (0, 1):
            violations.append(
                "DECISION | vertex_selection["
                f"{index}] must be integer 0 or 1, got {value!r}"
            )
            valid = False

    return valid


def _check_c1_edge_independence(
    num_vertices: int, edges: list, selection: list, violations: list[str]
) -> None:
    """C1: for every edge, at most one endpoint may be selected."""
    for edge_index, edge in enumerate(edges):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            violations.append(
                f"FORMAT | edges[{edge_index}] cannot be evaluated for C1: "
                "an edge must contain exactly two endpoints"
            )
            continue

        left, right = edge
        if type(left) is not int or type(right) is not int:
            violations.append(
                f"FORMAT | edges[{edge_index}] cannot be evaluated for C1: "
                "edge endpoints must be ints"
            )
            continue
        if not (0 <= left < num_vertices and 0 <= right < num_vertices):
            violations.append(
                f"FORMAT | edges[{edge_index}] cannot be evaluated for C1: "
                f"endpoint indices {edge!r} must refer to vertices 0..{num_vertices - 1}"
            )
            continue

        # C1 uses binary integers and an integer sum, so exact comparison is required.
        if selection[left] + selection[right] > 1:
            violations.append(
                f"C1 | edge ({left}, {right}) has both endpoints selected"
            )


def check_solution(instance: dict, solution: dict) -> list:
    """Return constraint violations; an empty list means the solution is feasible."""
    violations: list[str] = []

    try:
        num_vertices, edges = _read_instance(instance, violations)
        selection = _read_selection(solution, violations)

        if selection is None:
            return violations

        valid_selection = _selection_has_valid_domain(
            selection, num_vertices, violations
        )

        # C1 | Edge independence can be indexed safely only after its inputs pass
        # the structural, completeness, and binary-domain checks above.
        if (
            num_vertices is not None
            and edges is not None
            and valid_selection
            and len(selection) == num_vertices
        ):
            _check_c1_edge_independence(
                num_vertices, edges, selection, violations
            )
    except Exception as exc:  # Defensive boundary: malformed objects must not escape.
        violations.append(
            f"FORMAT | checker could not safely inspect the supplied data: {type(exc).__name__}"
        )

    return violations
