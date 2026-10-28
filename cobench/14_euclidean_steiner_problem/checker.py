"""Feasibility checker for the Euclidean Steiner tree output schema."""

import math
from numbers import Integral, Real


def _is_finite_real(value):
    """Return whether value is a finite real coordinate (booleans are not coordinates)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _read_inputs(instance, solution):
    """Validate only the structure needed to interpret the two required fields."""
    violations = []

    if not isinstance(instance, dict):
        return None, ["FORMAT | instance must be a dictionary"]
    if not isinstance(solution, dict):
        return None, ["FORMAT | solution must be a dictionary"]

    if "points" not in instance:
        violations.append("FORMAT | instance is missing 'points'")
    elif not isinstance(instance["points"], (list, tuple)):
        violations.append("FORMAT | instance 'points' must be a list")

    if "max_steiner_points" not in instance:
        violations.append("FORMAT | instance is missing 'max_steiner_points'")
    elif (
        isinstance(instance["max_steiner_points"], bool)
        or not isinstance(instance["max_steiner_points"], Integral)
    ):
        violations.append("FORMAT | instance 'max_steiner_points' must be an integer")

    if "steiner_points" not in solution:
        violations.append("FORMAT | solution is missing required field 'steiner_points'")
    elif not isinstance(solution["steiner_points"], (list, tuple)):
        violations.append("FORMAT | solution 'steiner_points' must be a list")

    if "edges" not in solution:
        violations.append("FORMAT | solution is missing required field 'edges'")
    elif not isinstance(solution["edges"], (list, tuple)):
        violations.append("FORMAT | solution 'edges' must be a list")

    if violations:
        return None, violations

    steiner_points = solution["steiner_points"]
    for index, point in enumerate(steiner_points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            violations.append(
                f"FORMAT | steiner_points[{index}] must contain exactly two coordinates"
            )
            continue
        if not _is_finite_real(point[0]) or not _is_finite_real(point[1]):
            violations.append(
                f"FORMAT | steiner_points[{index}] coordinates must be finite real numbers"
            )

    terminal_count = len(instance["points"])
    vertex_count = terminal_count + len(steiner_points)
    normalized_edges = []
    seen_edges = set()

    for index, edge in enumerate(solution["edges"]):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            violations.append(
                f"FORMAT | edges[{index}] must contain exactly two endpoint indices"
            )
            continue

        u, v = edge
        if (
            isinstance(u, bool)
            or isinstance(v, bool)
            or not isinstance(u, Integral)
            or not isinstance(v, Integral)
        ):
            violations.append(f"FORMAT | edges[{index}] endpoints must be integers")
            continue
        u, v = int(u), int(v)
        if not (0 <= u < vertex_count) or not (0 <= v < vertex_count):
            violations.append(
                f"FORMAT | edges[{index}] endpoint is outside index range "
                f"0..{vertex_count - 1}"
            )
            continue
        if u == v:
            violations.append(f"FORMAT | edges[{index}] is a self-loop, not an edge u<v")
            continue

        normalized = (min(u, v), max(u, v))
        if normalized in seen_edges:
            violations.append(
                f"FORMAT | edges[{index}] duplicates undirected edge {normalized}"
            )
            continue
        seen_edges.add(normalized)
        normalized_edges.append(normalized)

    if violations:
        return None, violations

    return {
        "terminal_count": terminal_count,
        "vertex_count": vertex_count,
        "steiner_count": len(steiner_points),
        "max_steiner_points": int(instance["max_steiner_points"]),
        "edges": normalized_edges,
    }, []


def _check_c1_connected(data):
    """C1: selected edges form one connected graph on P union S."""
    vertex_count = data["vertex_count"]
    if vertex_count == 0:
        return []

    adjacency = [[] for _ in range(vertex_count)]
    for u, v in data["edges"]:
        adjacency[u].append(v)
        adjacency[v].append(u)

    visited = {0}
    pending = [0]
    while pending:
        u = pending.pop()
        for v in adjacency[u]:
            if v not in visited:
                visited.add(v)
                pending.append(v)

    if len(visited) == vertex_count:
        return []

    unreachable = sorted(set(range(vertex_count)) - visited)
    return [
        "C1 | selected edges do not form a connected graph on P ∪ S; "
        f"vertices unreachable from vertex 0: {unreachable}"
    ]


def _check_c2_edge_count(data):
    """C2: a tree on all returned points has exactly |P|+|S|-1 edges."""
    actual = len(data["edges"])
    expected = data["vertex_count"] - 1
    if actual == expected:
        return []
    return [
        "C2 | selected edge count must equal (terminals + returned Steiner points) - 1; "
        f"got {actual}, expected {expected}"
    ]


def _check_c3_steiner_limit(data):
    """C3: the number of returned Steiner points does not exceed M."""
    actual = data["steiner_count"]
    maximum = data["max_steiner_points"]
    if actual <= maximum:
        return []
    return [
        "C3 | returned Steiner point count exceeds max_steiner_points; "
        f"got {actual}, maximum {maximum}"
    ]


def check_solution(instance: dict, solution: dict) -> list:
    """Return a list of format or semantic violations; [] means feasible."""
    try:
        data, violations = _read_inputs(instance, solution)
        if violations:
            return violations

        # Exactly one semantic check for each authoritative constraint: C1, C2, C3.
        violations.extend(_check_c1_connected(data))
        violations.extend(_check_c2_edge_count(data))
        violations.extend(_check_c3_steiner_limit(data))
        return violations
    except Exception as exc:  # Defensive boundary: malformed objects must not escape.
        return [
            "FORMAT | checker could not safely inspect the supplied objects: "
            f"{type(exc).__name__}"
        ]
