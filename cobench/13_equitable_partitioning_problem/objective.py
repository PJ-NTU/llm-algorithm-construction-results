def objective(instance, solution):
    N = instance['num_individuals']
    M = instance['num_attributes']
    K = instance['num_groups']
    A = instance['attribute_matrix']
    group_assignment = solution['group_assignment']

    # Reconstruct x[i][j] from group_assignment
    # group_assignment is {"person_i": group_index}
    x = [[0 for _ in range(K)] for _ in range(N)]
    for i in range(N):
        group_idx = group_assignment[f"person_{i}"]
        x[i][group_idx] = 1

    total_objective = 0.0
    for m in range(M):
        # Calculate the mean count for attribute m across all groups
        total_attr_count = sum(A[i][m] for i in range(N))
        mean_m = total_attr_count / K
        
        sum_abs_deviations = 0.0
        for j in range(K):
            # Calculate the count of attribute m in group j
            group_attr_count = sum(A[i][m] * x[i][j] for i in range(N))
            sum_abs_deviations += abs(group_attr_count - mean_m)
        
        # Add the average absolute deviation for this attribute to the total
        total_objective += (1.0 / K) * sum_abs_deviations

    return float(total_objective)