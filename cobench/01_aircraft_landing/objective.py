def objective(instance, solution):
    planes = instance['planes']
    target_times = instance['target_landing_time']
    alpha = instance['earliness_penalty_cost']
    beta = instance['lateness_penalty_cost']
    landing_times = solution['landing_times']
    
    total_penalty = 0.0
    for i in planes:
        # Convert plane index to string to match solution dictionary keys
        plane_id = str(i)
        t_i = landing_times[plane_id]
        T_i = target_times[i]
        
        # Calculate earliness e[i] = max(0, T_i - t_i)
        e_i = max(0.0, T_i - t_i)
        # Calculate lateness l[i] = max(0, t_i - T_i)
        l_i = max(0.0, t_i - T_i)
        
        total_penalty += alpha[i] * e_i + beta[i] * l_i
        
    return float(total_penalty)