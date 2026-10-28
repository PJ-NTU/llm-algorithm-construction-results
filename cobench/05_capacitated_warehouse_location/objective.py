def objective(instance, solution):
    num_customers = instance['num_customers']
    num_plants = instance['num_plants']
    plants_fixed_costs = instance['plants_fixed_costs']
    shipping_costs = instance['shipping_costs']
    
    open_plants = solution['open_plants']
    assignments = solution['assignments']
    
    fixed_cost_total = 0.0
    for j in range(num_plants):
        fixed_cost_total += plants_fixed_costs[j] * open_plants[j]
        
    variable_cost_total = 0.0
    for i in range(num_customers):
        customer_assignments = assignments[str(i)]
        for j in range(num_plants):
            x_ij = customer_assignments[str(j)]
            variable_cost_total += shipping_costs[i][j] * x_ij
            
    return float(fixed_cost_total + variable_cost_total)