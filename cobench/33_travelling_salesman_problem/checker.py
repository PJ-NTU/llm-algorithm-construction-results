"""Feasibility checker for the tour representation described in MAT_s4.md."""

from collections import Counter


def _check_c1(num_nodes: int, arcs: list[tuple[int, int]]) -> list[str]:
    """C1: every node has exactly one selected outgoing arc."""
    out_degree = Counter(origin for origin, _ in arcs)
    bad = [(node, out_degree[node]) for node in range(num_nodes) if out_degree[node] != 1]
    if bad:
        return [
            "C1 | Each node must have exactly one outgoing arc; "
            f"nodes whose actual out-degree is not 1 (node, out-degree): {bad[:10]}"
        ]
    return []


def _check_c2(num_nodes: int, arcs: list[tuple[int, int]]) -> list[str]:
    """C2: every node has exactly one selected incoming arc."""
    in_degree = Counter(destination for _, destination in arcs)
    bad = [(node, in_degree[node]) for node in range(num_nodes) if in_degree[node] != 1]
    if bad:
        return [
            "C2 | Each node must have exactly one incoming arc; "
            f"nodes whose actual in-degree is not 1 (node, in-degree): {bad[:10]}"
        ]
    return []


def _check_c3(num_nodes: int, arcs: list[tuple[int, int]]) -> list[str]:
    """C3: the selected arcs form one complete cycle containing depot 0."""
    # C3 is vacuous when there are no target nodes. C1/C2 still decide whether
    # the required depot arcs exist, without inventing a num_nodes > 0 premise.
    if num_nodes <= 1:
        return []

    successors: dict[int, list[int]] = {node: [] for node in range(num_nodes)}
    for origin, destination in arcs:
        successors[origin].append(destination)

    # A unique successor is needed before traversal is well-defined. If it is
    # absent, the selected arcs cannot be the single complete loop C3 describes.
    if any(len(successors[node]) != 1 for node in range(num_nodes)):
        return [
            "C3 | The selected arcs cannot form a single complete cycle containing start node 0: "
            "at least one node has no unique successor"
        ]

    visited: set[int] = set()
    current = 0
    for _ in range(num_nodes):
        if current in visited:
            break
        visited.add(current)
        current = successors[current][0]

    expected = set(range(num_nodes))
    if current != 0 or visited != expected:
        missing = sorted(expected - visited)
        return [
            "C3 | The selected arcs form an incomplete cycle or a subtour not containing the start node; "
            f"reachable from start node 0: {len(visited)}/{num_nodes} nodes,"
            f"uncovered nodes: {missing[:10]}, traversal ends at {current}"
        ]
    return []


def check_solution(instance: dict, solution: dict) -> list:
    """Return constraint violations; an empty list means the tour is feasible."""
    try:
        # These checks are only the structural/type foundation needed to read
        # the representation safely. Extra solution fields are intentionally ignored.
        if not isinstance(instance, dict):
            return ["FORMAT | instance must be a dict"]
        if not isinstance(solution, dict):
            return ["FORMAT | solution must be a dict"]
        if "tour" not in solution:
            return ["FORMAT | missing the only required field 'tour'"]

        num_nodes = instance.get("num_nodes")
        if not isinstance(num_nodes, int) or isinstance(num_nodes, bool):
            return ["FORMAT | instance.num_nodes must be an integer, cannot read node indices from it"]

        tour = solution["tour"]
        if not isinstance(tour, list):
            return ["FORMAT | tour must match list[int] as in the description"]
        if len(tour) != num_nodes:
            return [
                "FORMAT | tour must list all num_nodes nodes without explicitly repeating the end node 0; "
                f"expected length {num_nodes}, actual length {len(tour)}"
            ]
        if not tour:
            return ["FORMAT | tour cannot place start node 0 first as required by the description"]

        non_integer_positions = [
            index
            for index, node in enumerate(tour)
            if not isinstance(node, int) or isinstance(node, bool)
        ]
        if non_integer_positions:
            return [
                "FORMAT | tour must match list[int]; "
                f"non-integer element positions: {non_integer_positions[:10]}"
            ]

        out_of_range = [
            (index, node)
            for index, node in enumerate(tour)
            if node < 0 or node >= num_nodes
        ]
        if out_of_range:
            return [
                "FORMAT | node indices in tour must come from 0..N given in the description; "
                f"out-of-range items (position, value): {out_of_range[:10]}"
            ]
        if tour[0] != 0:
            return ["FORMAT | the description requires start node 0 to be first in tour"]

        # The closing edge is implicit in the output contract. Self-transitions
        # are excluded because x[i][j] is defined only for i != j.
        arcs = [
            (tour[index], tour[(index + 1) % num_nodes])
            for index in range(num_nodes)
            if tour[index] != tour[(index + 1) % num_nodes]
        ]

        violations: list[str] = []
        violations.extend(_check_c1(num_nodes, arcs))
        violations.extend(_check_c2(num_nodes, arcs))
        violations.extend(_check_c3(num_nodes, arcs))
        return violations
    except Exception as exc:  # Defensive boundary: malformed objects must not escape.
        return [f"FORMAT | exception while checking input, error={type(exc).__name__}: {exc}"]
