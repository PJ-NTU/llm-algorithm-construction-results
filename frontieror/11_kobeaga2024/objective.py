import math


def objective(instance: dict, solution: dict) -> float:
    n = instance["n_locations"]
    visited = solution["visited"]

    if not isinstance(visited, list) or len(visited) != n:
        raise ValueError(f"'visited' must be a list of length {n}")
    if any(type(value) is not int or value not in (0, 1) for value in visited):
        raise ValueError("every entry of 'visited' must be the integer 0 or 1")

    scores = instance["scores"]
    total_score = math.fsum(scores[i] for i in range(n) if visited[i] == 1)

    # The original objective is maximization; negate it for minimization ranking.
    return -total_score
