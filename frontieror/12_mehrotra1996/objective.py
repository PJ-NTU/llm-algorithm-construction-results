def objective(instance, solution):
    coloring = solution['coloring']
    if not coloring:
        return 0.0
    unique_colors = set(coloring)
    # The objective is the number of distinct colors used, 
    # which corresponds to the sum of the binary variables y[c].
    # Even if the symmetry breaking constraint y[c] >= y[c+1] is used,
    # the total count of used colors is the number of unique values in the coloring list.
    return float(len(unique_colors))