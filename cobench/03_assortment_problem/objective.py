def objective(instance, solution):
    stock_areas = instance['area']
    piece_areas = instance['piece_area']
    
    total_stock_area = 0.0
    for instance_data in solution['stock_instances']:
        s_id = instance_data['stock_type_id']
        total_stock_area += stock_areas[s_id]
    
    total_piece_area = 0.0
    for i in range(instance['m']):
        count = solution['piece_counts'].get(f"piece_{i}", 0)
        total_piece_area += count * piece_areas[i]
        
    if total_stock_area == 0:
        return 0.0
        
    return (total_stock_area - total_piece_area) / total_stock_area