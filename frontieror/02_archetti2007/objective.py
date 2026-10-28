def objective(instance, solution):
    h0 = instance['supplier_inventory_cost']
    hs = instance['retailers_inventory_cost']
    dist = instance['distance_matrix']
    H = instance['time_horizon']
    N = instance['num_retailers']

    # Inventory holding costs
    # Sum over t=1 to H+1
    total_inventory_cost = 0.0
    for t_str in solution['supplier_inventory']:
        total_inventory_cost += h0 * solution['supplier_inventory'][t_str]
    
    for t_str in solution['retailer_inventory']:
        for s_idx in range(1, N + 1):
            total_inventory_cost += hs[s_idx-1] * solution['retailer_inventory'][t_str][str(s_idx)]

    # Transportation costs
    # Sum over t=1 to H
    total_transport_cost = 0.0
    for t_idx in range(1, H + 1):
        t_str = str(t_idx)
        if t_str in solution['routes']:
            for edge in solution['routes'][t_str]:
                i, j, count = edge[0], edge[1], edge[2]
                # The solver uses y[t][i, j] where i > j
                # The routes format is [node_a, node_b, count]
                total_transport_cost += dist[i][j] * count

    return float(total_inventory_cost + total_transport_cost)