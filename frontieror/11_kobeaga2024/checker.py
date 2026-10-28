import math


def _is_binary(value: object) -> bool:
    return type(value) is int and value in (0, 1)


def check(instance: dict, solution: dict) -> list:
    violations = []

    if not isinstance(solution, dict):
        return ["structure: solution must be a dictionary"]

    n = instance["n_locations"]
    depot = instance["depot_index"]

    if "visited" not in solution:
        violations.append("structure: missing decision field 'visited'")
    else:
        visited = solution["visited"]
        if not isinstance(visited, list):
            violations.append("structure: 'visited' must be a list")
        elif len(visited) != n:
            violations.append(
                f"structure: 'visited' must have length {n}, got {len(visited)}"
            )
        elif not all(_is_binary(value) for value in visited):
            violations.append(
                "structure: every entry of 'visited' must be the integer 0 or 1"
            )

    if "selected_links" not in solution:
        violations.append("structure: missing decision field 'selected_links'")
    else:
        links = solution["selected_links"]
        if not isinstance(links, list):
            violations.append("structure: 'selected_links' must be a list")
        elif len(links) != n:
            violations.append(
                f"structure: 'selected_links' must have {n} rows, got {len(links)}"
            )
        elif any(not isinstance(row, list) for row in links):
            violations.append(
                "structure: every row of 'selected_links' must be a list"
            )
        elif any(len(row) != n for row in links):
            violations.append(
                f"structure: every row of 'selected_links' must have length {n}"
            )
        elif any(not _is_binary(value) for row in links for value in row):
            violations.append(
                "structure: every entry of 'selected_links' must be the integer 0 or 1"
            )
        else:
            if any(links[i][i] != 0 for i in range(n)):
                violations.append(
                    "structure: the diagonal of 'selected_links' must be zero"
                )
            if any(
                links[i][j] != links[j][i]
                for i in range(n)
                for j in range(i + 1, n)
            ):
                violations.append(
                    "structure: 'selected_links' must be symmetric because links are undirected"
                )

    if violations:
        return violations

    visited = solution["visited"]
    links = solution["selected_links"]

    # C1: The depot must be visited.
    if visited[depot] != 1:
        violations.append(
            f"C1: depot location {depot} must be visited"
        )

    # C2: Every visited location has degree two, and every unvisited location has degree zero.
    for i in range(n):
        degree = sum(links[i][j] for j in range(n))
        required_degree = 2 * visited[i]
        if degree != required_degree:
            violations.append(
                f"C2: location {i} has selected-link degree {degree}, "
                f"but its required degree is {required_degree}"
            )

    # C3: Every visited location must be connected to the depot through selected links.
    reachable = {depot}
    stack = [depot]
    while stack:
        current = stack.pop()
        for neighbor in range(n):
            if links[current][neighbor] == 1 and neighbor not in reachable:
                reachable.add(neighbor)
                stack.append(neighbor)

    disconnected = [
        i for i in range(n) if visited[i] == 1 and i not in reachable
    ]
    if disconnected:
        violations.append(
            "C3: visited locations disconnected from the depot through selected links: "
            + str(disconnected)
        )

    # C4: The total distance of selected undirected links must not exceed the budget.
    distances = instance["distance_matrix"]
    budget = instance["distance_budget"]
    total_distance = math.fsum(
        distances[i][j] * links[i][j]
        for i in range(n)
        for j in range(i + 1, n)
    )
    if total_distance > budget:
        violations.append(
            f"C4: total route distance {total_distance} exceeds budget {budget}"
        )

    return violations
