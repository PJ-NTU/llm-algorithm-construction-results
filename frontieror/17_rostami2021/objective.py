def objective(instance, solution):
    n = instance['n']
    num_scenarios = instance['num_scenarios']
    alpha = instance['cost_parameters_alpha']
    chi = instance['cost_parameters_chi']
    delta = instance['cost_parameters_delta']
    dist = instance['distances']
    fixed_costs = instance['fixed_costs']
    probs = instance['scenarios_probability']
    w = instance['scenarios_demands']
    
    z = solution['hub_selection']
    allocations = solution['scenario_allocations']
    
    total_fixed_cost = sum(fixed_costs[k] * z[k] for k in range(n))
    
    expected_transport_cost = 0.0
    for s in range(num_scenarios):
        s_key = str(s)
        h = allocations[s_key] # h[i] is the hub node i is assigned to
        
        # Collection cost: sum_{i} sum_{k!=i} c[s][i][k] * x[s][i][k]
        # x[s][i][k] = 1 if h[i] == k and z[i] == 0
        collection_cost = 0.0
        for i in range(n):
            if z[i] == 0:
                k = h[i]
                # Based on solver logic: c[s][i][k] = chi * dist[i][k] * sum_j w[s][i][j]
                flow_out = sum(w[s][i][j] for j in range(n))
                collection_cost += chi * dist[i][k] * flow_out
        
        # Distribution cost: sum_j sum_{l!=j} d[s][l][j] * x[s][j][l]
        # x[s][j][l] = 1 if h[j] == l and z[j] == 0
        distribution_cost = 0.0
        for j in range(n):
            if z[j] == 0:
                l = h[j]
                # Based on solver logic: d[s][l][j] = delta * dist[l][j] * sum_i w[s][i][j]
                flow_in = sum(w[s][i][j] for i in range(n))
                distribution_cost += delta * dist[l][j] * flow_in
        
        # Inter-hub cost: alpha * sum_i sum_j w[s][i][j] * dist[h(i)][h(j)]
        inter_hub_cost = 0.0
        for i in range(n):
            for j in range(n):
                inter_hub_cost += alpha * w[s][i][j] * dist[h[i]][h[j]]
        
        scenario_total = collection_cost + distribution_cost + inter_hub_cost
        expected_transport_cost += probs[s] * scenario_total
        
    return float(total_fixed_cost + expected_transport_cost)