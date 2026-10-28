"""Feasibility checker for the graph-coloring solution format in MAT_s4.md."""


def _read_instance(instance):
    """Return (n, edges, violations) without letting malformed input raise."""
    violations = []
    if not isinstance(instance, dict):
        return None, None, ["FORMAT | instance must be a dict; cannot read n and edges"]

    if "n" not in instance:
        violations.append("FORMAT | instance is missing n; cannot determine the number of vertices and the upper bound of color indices")
        n = None
    else:
        n = instance["n"]
        if isinstance(n, bool) or not isinstance(n, int):
            violations.append("FORMAT | instance.n must be an integer; cannot interpret vertex and color indices")
            n = None

    if "edges" not in instance:
        violations.append("FORMAT | instance is missing edges, cannot check C2")
        edges = None
    else:
        edges = instance["edges"]
        if not isinstance(edges, list):
            violations.append("FORMAT | instance.edges must be a list of edges, cannot check C2")
            edges = None

    if violations:
        return n, edges, violations

    parsed_edges = []
    for edge_index, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 2:
            violations.append(
                f"FORMAT | edges[{edge_index}] must be a two-element list [u, v], cannot check C2"
            )
            continue
        u, v = edge
        if (
            isinstance(u, bool)
            or not isinstance(u, int)
            or isinstance(v, bool)
            or not isinstance(v, int)
        ):
            violations.append(
                f"FORMAT | edges[{edge_index}] must have two integer vertex indices, cannot check C2"
            )
            continue
        if not (0 <= u < n) or not (0 <= v < n):
            violations.append(
                f"FORMAT | edges[{edge_index}]={edge!r} contains a vertex that coloring cannot index, cannot check C2"
            )
            continue
        parsed_edges.append((u, v))

    return n, parsed_edges, violations


def _read_coloring(solution):
    """Return (coloring, violations) for the sole required solution field."""
    if not isinstance(solution, dict):
        return None, ["FORMAT | solution must be a dict"]
    if "coloring" not in solution:
        return None, ["FORMAT | solution is missing the only required field coloring"]
    coloring = solution["coloring"]
    if not isinstance(coloring, list):
        return None, ["FORMAT | coloring must be list[int]"]
    return coloring, []


def _check_c1(n, coloring):
    """C1: every vertex has exactly one allowed integer color."""
    violations = []

    # C1 | A length-n list gives one slot to each vertex 0..n-1.
    if len(coloring) != n:
        violations.append(
            f"C1 | coloring has length {len(coloring)}, but must give exactly one color to each of all {n} vertices"
        )

    # C1 | Each slot must be an integer color index in the exact range 0..K-1;
    # MAT_s4.md maps the instance field n to K.
    for vertex, color in enumerate(coloring):
        if isinstance(color, bool) or not isinstance(color, int):
            violations.append(
                f"C1 | Vertex {vertex} has color {color!r} which is not an integer color index"
            )
            continue
        if not (0 <= color < n):
            violations.append(
                f"C1 | Vertex {vertex} has color {color} which is outside the description-specified range 0..{n - 1}"
            )

    return violations


def _check_c2(edges, coloring):
    """C2: the endpoints of every edge have different colors."""
    violations = []

    # C2 | Endpoint colors are exact integers, so exact inequality is required;
    # adding numeric tolerance here could only admit a real violation.
    for edge_index, (u, v) in enumerate(edges):
        if coloring[u] == coloring[v]:
            violations.append(
                f"C2 | Edge edges[{edge_index}]=[{u}, {v}] has both adjacent vertices with the same color {coloring[u]}"
            )

    return violations


def _check_c3(coloring):
    """C3: construct and verify a y witness from the submitted coloring."""
    violations = []

    # C3 | y is not a required output field.  Reconstruct the witness
    # y[c] = 1 exactly when some submitted vertex uses c, rather than trusting
    # a solution-declared y or objective value.
    used_colors = set(coloring)
    y = {color: 1 for color in used_colors}
    for vertex, color in enumerate(coloring):
        if y.get(color, 0) < 1:
            violations.append(
                f"C3 | Vertex {vertex} uses color {color}, but reconstructed y[{color}] is not marked as 1"
            )

    return violations


def _check_c4(coloring):
    """C4: account for the explicitly optional symmetry-breaking constraint."""
    # C4 | MAT_s4.md calls this constraint "(optional) symmetry-breaking constraint".  Whether the
    # labels in used_colors are consecutive is therefore not a feasibility
    # condition.  Compute the property to make the correspondence explicit,
    # but do not reject an otherwise valid coloring when an optional modeling
    # aid was omitted.
    used_colors = sorted(set(coloring))
    _uses_consecutive_low_labels = used_colors == list(range(len(used_colors)))
    return []


def check_solution(instance: dict, solution: dict) -> list:
    """Return concrete violations; an empty list means the solution is feasible."""
    n, edges, instance_violations = _read_instance(instance)
    coloring, solution_violations = _read_coloring(solution)
    violations = instance_violations + solution_violations
    if violations:
        return violations

    c1_violations = _check_c1(n, coloring)
    violations.extend(c1_violations)

    # C2/C3/C4 may use colors as indices or set members.  Only run them after
    # C1 establishes a complete, integer, in-range coloring.
    if c1_violations:
        return violations

    violations.extend(_check_c2(edges, coloring))
    violations.extend(_check_c3(coloring))
    violations.extend(_check_c4(coloring))
    return violations
