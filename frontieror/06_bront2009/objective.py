def objective(instance, solution):
    arrival_rate = instance['lambda']
    fares = instance['products_fare']
    seg_lambdas = instance['segments_lambda_l']
    seg_consideration = instance['segments_consideration_set']
    seg_preferences = instance['segments_preference_vector']
    seg_no_purchase = instance['segments_no_purchase_preference']
    num_segments = len(seg_lambdas)
    total_seg_lambda = sum(seg_lambdas)

    # Pre-calculate segment-product preference mapping
    pref_map = []
    for l in range(num_segments):
        p_map = {}
        for idx, prod_id in enumerate(seg_consideration[l]):
            p_map[prod_id] = seg_preferences[l][idx]
        pref_map.append(p_map)

    total_expected_revenue = 0.0
    for allocation in solution['offer_set_allocations']:
        S = allocation['offer_set']
        t_S = allocation['time_allocated']
        if t_S <= 0:
            continue
        
        expected_revenue_S = 0.0
        for l in range(num_segments):
            alpha_l = seg_lambdas[l] / total_seg_lambda
            denom = seg_no_purchase[l]
            for j in S:
                denom += pref_map[l].get(j, 0.0)
            
            if denom > 0:
                segment_revenue_in_S = 0.0
                for j in S:
                    pref_lj = pref_map[l].get(j, 0.0)
                    if pref_lj > 0:
                        prob_lj = pref_lj / denom
                        segment_revenue_in_S += prob_lj * fares[j]
                expected_revenue_S += alpha_l * segment_revenue_in_S
        
        total_expected_revenue += arrival_rate * expected_revenue_S * t_S

    return float(total_expected_revenue)

# Legacy raw utility is maximized; cycle4 scores are minimized.
_legacy_raw_objective = objective
def objective(instance, solution):
    return -_legacy_raw_objective(instance, solution)
