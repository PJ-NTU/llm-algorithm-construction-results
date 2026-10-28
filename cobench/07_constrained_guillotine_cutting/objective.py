def objective(instance, solution):
    piece_values = instance['piece_values']
    selection_status = solution['selection_status']
    
    total_value = 0.0
    for i in range(len(selection_status)):
        v_i = piece_values[i]
        for p_status in selection_status[i]:
            if p_status == 1:
                total_value += v_i
                
    return float(total_value)