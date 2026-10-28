def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, (list, tuple)) else []


def _nonnegative_int(value, default=None):
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if result >= 0 else default


def _instance_size(instance):
    instance = _as_dict(instance)
    n = _nonnegative_int(instance.get("num_nodes"))
    if n is not None:
        return n

    return max(
        len(_as_list(instance.get("scores"))),
        len(_as_list(instance.get("coordinates"))),
        len(_as_list(instance.get("distance_matrix"))),
    )


def _matrix_value(matrix, i, j, default=0.0):
    rows = _as_list(matrix)
    if i >= len(rows):
        return default
    row = _as_list(rows[i])
    if j >= len(row):
        return default
    value = row[j]
    return default if value is None else value


def _numeric(value, default=0.0):
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _official_score(scores, vertex):
    scores = _as_list(scores)
    if vertex < 0 or vertex >= len(scores):
        return 0
    try:
        return int(scores[vertex])
    except (TypeError, ValueError, OverflowError):
        return 0


def to_ours(official_instance: dict) -> dict:
    instance = _as_dict(official_instance)
    n = _instance_size(instance)

    depot = _nonnegative_int(instance.get("depot_vertex"), 0)
    if n > 0 and depot >= n:
        depot = 0

    source_scores = _as_list(instance.get("scores"))
    scores = [
        source_scores[i] if i < len(source_scores) and source_scores[i] is not None else 0.0
        for i in range(n)
    ]

    source_matrix = instance.get("distance_matrix")
    distance_matrix = [
        [_matrix_value(source_matrix, i, j, 0.0) for j in range(n)]
        for i in range(n)
    ]

    distance_budget = instance.get("distance_limitation_d0", 0.0)
    if distance_budget is None:
        distance_budget = 0.0

    return {
        "n_locations": n,
        "depot_index": depot,
        "scores": scores,
        "distance_matrix": distance_matrix,
        "distance_budget": distance_budget,
    }


def to_official(official_instance: dict, our_solution: dict) -> dict:
    instance = _as_dict(official_instance)
    solution = _as_dict(our_solution)
    n = _instance_size(instance)

    visited = _as_list(solution.get("visited"))
    visited_vertices = [
        i for i in range(n)
        if i < len(visited) and visited[i] == 1
    ]

    selected_links = _as_list(solution.get("selected_links"))
    tour_edges = []
    for i in range(n):
        row = _as_list(selected_links[i]) if i < len(selected_links) else []
        for j in range(i + 1, n):
            if j < len(row) and row[j] == 1:
                tour_edges.append([i, j])

    scores = instance.get("scores")
    objective_value = float(sum(
        _official_score(scores, vertex)
        for vertex in visited_vertices
    ))

    distances = instance.get("distance_matrix")
    tour_distance = sum(
        _numeric(_matrix_value(distances, u, v, 0.0), 0.0)
        for u, v in tour_edges
    )

    return {
        "objective_value": objective_value,
        "status": solution.get("status", 2),
        "status_name": solution.get("status_name", "OPTIMAL"),
        "runtime": solution.get("runtime", 0.0),
        "mip_gap": solution.get("mip_gap", 0.0),
        "visited_vertices": visited_vertices,
        "tour_edges": tour_edges,
        "tour_distance": tour_distance,
    }


def official_objective(official_instance: dict, official_solution: dict) -> float:
    instance = _as_dict(official_instance)
    solution = _as_dict(official_solution)
    n = _instance_size(instance)

    selected = set()
    if "tour" in solution:
        vertices = _as_list(solution.get("tour"))
    elif "visited_vertices" in solution and "tour_edges" in solution:
        vertices = _as_list(solution.get("visited_vertices"))
    else:
        vertices = []

    for vertex in vertices:
        if isinstance(vertex, (int, float)) and int(vertex) == vertex:
            vertex = int(vertex)
            if 0 <= vertex < n:
                selected.add(vertex)

    scores = instance.get("scores")
    return float(sum(_official_score(scores, vertex) for vertex in selected))