def objective(instance, solution):
    tour = solution['tour']
    d = instance['distance_matrix']
    total = 0.0
    n = len(tour)
    for i in range(n):
        j = (i + 1) % n
        total += d[tour[i]][tour[j]]
    return total