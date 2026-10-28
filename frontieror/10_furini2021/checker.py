"""Checker for the edge-interdiction maximum-clique problem."""


_REQUIRED_FIELDS = (
    "interdiction_status",
    "clique_membership",
    "max_clique_size",
)


def _is_plain_int(value):
    """JSON booleans are not integer decisions even though bool subclasses int."""
    return isinstance(value, int) and not isinstance(value, bool)


def _popcount(value):
    """Count set bits without requiring a particular Python minor version."""
    return bin(value).count("1")


def _read_instance(instance, violations):
    if not isinstance(instance, dict):
        violations.append("FORMAT | instance must be a dict so the graph can be read")
        return None

    missing = [key for key in ("vertices", "edges", "interdiction_budget_k") if key not in instance]
    for key in missing:
        violations.append(f"FORMAT | instance is missing required graph field {key!r}")
    if missing:
        return None

    vertices = instance["vertices"]
    edges = instance["edges"]
    budget = instance["interdiction_budget_k"]

    if not isinstance(vertices, list):
        violations.append("FORMAT | instance field 'vertices' must be a list")
        return None
    if not all(_is_plain_int(vertex) for vertex in vertices):
        violations.append("FORMAT | every entry in instance field 'vertices' must be an integer")
        return None
    if len(set(vertices)) != len(vertices):
        violations.append(
            "FORMAT | instance field 'vertices' contains duplicate indices, so clique_membership positions are ambiguous"
        )
        return None
    if not isinstance(edges, list):
        violations.append("FORMAT | instance field 'edges' must be a list")
        return None
    if not _is_plain_int(budget):
        violations.append("FORMAT | instance field 'interdiction_budget_k' must be an integer")
        return None

    vertex_position = {vertex: position for position, vertex in enumerate(vertices)}
    positioned_edges = []
    for edge_index, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 2:
            violations.append(
                f"FORMAT | edges[{edge_index}] must be a two-entry list naming its endpoints"
            )
            continue
        u, v = edge
        if not _is_plain_int(u) or not _is_plain_int(v):
            violations.append(f"FORMAT | endpoints in edges[{edge_index}] must be integers")
            continue
        if u not in vertex_position or v not in vertex_position:
            violations.append(
                f"FORMAT | edges[{edge_index}] uses an endpoint absent from instance field 'vertices'"
            )
            continue
        positioned_edges.append((vertex_position[u], vertex_position[v]))

    if len(positioned_edges) != len(edges):
        return None
    return vertices, positioned_edges, budget


def _read_binary_list(solution, field, expected_length, violations):
    value = solution.get(field)
    if not isinstance(value, list):
        violations.append(f"FORMAT | required field {field!r} must be a list")
        return None
    if len(value) != expected_length:
        violations.append(
            f"FORMAT | {field!r} has length {len(value)}, but its instance axis has length {expected_length}"
        )
        return None

    bad_positions = [
        position
        for position, item in enumerate(value)
        if not _is_plain_int(item) or item not in (0, 1)
    ]
    if bad_positions:
        violations.append(
            f"FORMAT | {field!r} must contain only integer binary values 0 or 1; bad positions: {bad_positions}"
        )
        return None
    return value


def _build_adjacency(num_vertices, positioned_edges, removed_mask):
    adjacency = [0] * num_vertices
    for edge_index, (u, v) in enumerate(positioned_edges):
        if removed_mask & (1 << edge_index) or u == v:
            continue
        adjacency[u] |= 1 << v
        adjacency[v] |= 1 << u
    return adjacency


def _maximum_clique_size(adjacency):
    """Return the exact clique number using Bron-Kerbosch branch-and-bound."""
    best = 0

    def search(chosen_count, candidates, excluded):
        nonlocal best
        if chosen_count + _popcount(candidates) <= best:
            return
        if not candidates and not excluded:
            best = max(best, chosen_count)
            return

        pivot_pool = candidates | excluded
        pivot = None
        pivot_score = -1
        cursor = pivot_pool
        while cursor:
            bit = cursor & -cursor
            vertex = bit.bit_length() - 1
            score = _popcount(candidates & adjacency[vertex])
            if score > pivot_score:
                pivot = vertex
                pivot_score = score
            cursor ^= bit

        branch_vertices = candidates if pivot is None else candidates & ~adjacency[pivot]
        while branch_vertices:
            bit = branch_vertices & -branch_vertices
            vertex = bit.bit_length() - 1
            search(
                chosen_count + 1,
                candidates & adjacency[vertex],
                excluded & adjacency[vertex],
            )
            candidates &= ~bit
            excluded |= bit
            branch_vertices &= ~bit
            if chosen_count + _popcount(candidates) <= best:
                return

    search(0, (1 << len(adjacency)) - 1, 0)
    return best






def _check_solution(instance, solution):
    violations = []
    graph = _read_instance(instance, violations)

    if not isinstance(solution, dict):
        violations.append("FORMAT | solution must be a dict containing the three required fields")
        return violations

    missing = [field for field in _REQUIRED_FIELDS if field not in solution]
    for field in missing:
        violations.append(f"FORMAT | solution is missing required field {field!r}")
    if missing or graph is None:
        return violations

    vertices, positioned_edges, budget = graph
    interdiction_status = _read_binary_list(
        solution,
        "interdiction_status",
        len(positioned_edges),
        violations,
    )
    clique_membership = _read_binary_list(
        solution,
        "clique_membership",
        len(vertices),
        violations,
    )

    theta = solution.get("max_clique_size")
    if not _is_plain_int(theta):
        violations.append("FORMAT | required field 'max_clique_size' must be an integer")
        theta = None

    if interdiction_status is None or clique_membership is None or theta is None:
        return violations

    # C1: sum_{(u,v) in E} w[u][v] <= k. This is an exact integer sum.
    removed_count = sum(interdiction_status)
    budget_satisfied = removed_count <= budget
    if not budget_satisfied:
        violations.append(
            f"C1 | removed {removed_count} edges, which exceeds interdiction budget {budget}"
        )

    removed_mask = 0
    for edge_index, removed in enumerate(interdiction_status):
        if removed:
            removed_mask |= 1 << edge_index
    remaining_adjacency = _build_adjacency(
        len(vertices), positioned_edges, removed_mask
    )

    # C2: theta and x must identify the true maximum clique after interdiction.
    selected_count = sum(clique_membership)
    if theta != selected_count:
        violations.append(
            f"C2 | max_clique_size is {theta}, but clique_membership selects {selected_count} vertices"
        )
    recomputed_clique_size = _maximum_clique_size(remaining_adjacency)
    theta_matches_graph = theta == recomputed_clique_size
    if not theta_matches_graph:
        violations.append(
            f"C2 | max_clique_size is {theta}, but the interdicted graph's recomputed maximum clique size is {recomputed_clique_size}"
        )

    # C3: x[u] + x[v] <= 2 - w[u][v] for every listed original edge.
    for edge_index, ((u, v), removed) in enumerate(
        zip(positioned_edges, interdiction_status)
    ):
        if clique_membership[u] + clique_membership[v] > 2 - removed:
            violations.append(
                f"C3 | edges[{edge_index}] is removed, but both endpoint vertices are selected in clique_membership"
            )

    # C4: x[u] + x[v] <= 1 for every non-edge in the original undirected graph.
    original_pairs = {
        (u, v) if u < v else (v, u)
        for u, v in positioned_edges
        if u != v
    }
    selected_vertices = [
        position for position, selected in enumerate(clique_membership) if selected
    ]
    for left in range(len(selected_vertices)):
        for right in range(left + 1, len(selected_vertices)):
            u, v = selected_vertices[left], selected_vertices[right]
            pair = (u, v) if u < v else (v, u)
            if pair not in original_pairs:
                violations.append(
                    "C4 | selected vertices "
                    f"{vertices[u]!r} and {vertices[v]!r} have no edge in the original graph"
                )

    # Feasible leader plans are ranked by objective.py. Only the follower
    # response must be optimal; C2 above verifies its exact clique number.

    return violations


def check_solution(instance: dict, solution: dict) -> list:
    """Return violations for an infeasible plan or an incorrect follower maximum."""
    try:
        return _check_solution(instance, solution)
    except Exception as exc:  # The checker contract forbids propagating malformed-input errors.
        return [
            "CHECKER | could not safely evaluate the supplied data: "
            f"{type(exc).__name__}: {exc}"
        ]
