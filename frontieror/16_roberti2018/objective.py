def objective(instance, solution):
    import math

    def dist(id1, id2):
        c1 = coords[id1]
        c2 = coords[id2]
        return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    # Build coordinate map
    coords = {instance['depot_id']: (instance['depot_x'], instance['depot_y'])}
    for i in range(instance['num_satellites']):
        coords[instance['satellites_id'][i]] = (instance['satellites_x'][i], instance['satellites_y'][i])
    for i in range(instance['num_customers']):
        coords[instance['customers_id'][i]] = (instance['customers_x'][i], instance['customers_y'][i])

    # 1. First-echelon travel costs
    e1_cost = 0.0
    for route in solution['first_echelon_routes']:
        for i in range(len(route) - 1):
            e1_cost += dist(route[i], route[i+1])

    # 2. Satellite handling costs
    handling_cost = 0.0
    sat_h_cost_map = {instance['satellites_id'][i]: instance['satellites_handling_cost'][i] for i in range(instance['num_satellites'])}
    for delivery in solution['first_echelon_deliveries']:
        s_id = delivery['satellite_id']
        qty = delivery['quantity']
        handling_cost += sat_h_cost_map[s_id] * qty

    # 3. Second-echelon travel costs
    e2_cost = 0.0
    for route in solution['second_echelon_routes']:
        for i in range(len(route) - 1):
            e2_cost += dist(route[i], route[i+1])

    return float(e1_cost + handling_cost + e2_cost)