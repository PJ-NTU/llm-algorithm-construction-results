def objective(instance, solution):
    n = instance['n']
    cost = instance['cost']
    assignment = solution['assignment']
    
    total_cost = 0.0
    for i in range(n):
        task_j = assignment[i]
        total_cost += cost[i][task_j]
        
    return float(total_cost)