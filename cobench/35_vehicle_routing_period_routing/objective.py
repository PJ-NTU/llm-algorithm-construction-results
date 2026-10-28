"""Sole PVRP objective: total Euclidean travel distance over all vehicles and all days, lower is better.

Callers should first decide feasibility with checker.check; this module does not repeat the C1-C10 checks,
does not prove optimality and computes no penalty. An infeasible declaration returns positive infinity, but this does not verify
its infeasibility. Finite JSON numbers are summed exactly via their decimal representation and converted only once at the end
to float; if the total exceeds the representable float range, positive infinity is returned.
"""

import math
from fractions import Fraction


def objective(instance: dict, solution: dict) -> float:
    """Compute sum(d[i][j] * y[i][j][h][t]); no sign flip or additional objective."""
    if not isinstance(solution, dict) or set(solution) != {'status', 'x', 'y', 'u'}:
        raise ValueError('the solution dict must contain exactly status, x, y, u')
    if solution['status'] == 'infeasible':
        if any(solution[key] is not None for key in ('x', 'y', 'u')):
            raise ValueError('when status=infeasible, x, y, u must all be null')
        return math.inf
    if solution['status'] != 'feasible':
        raise ValueError('status must be feasible or infeasible')
    if not isinstance(instance, dict):
        raise ValueError('instance must be a dict')
    for key, minimum in (('num_customers', 0), ('num_vehicles', 1),
                         ('planning_period', 1)):
        if type(instance.get(key)) is not int or instance[key] < minimum:
            raise ValueError(f'{key}: an integer of at least {minimum} is required')
    n = instance['num_customers']
    k_count = instance['num_vehicles']
    t_count = instance['planning_period']
    distances = instance.get('distance_matrix')
    y = solution['y']
    if type(distances) is not list or len(distances) != n + 1:
        raise ValueError('distance_matrix has an incorrect node-axis shape')
    if type(y) is not list or len(y) != n + 1:
        raise ValueError('y has an incorrect origin-axis shape')
    total = Fraction(0)
    for i in range(n + 1):
        if type(distances[i]) is not list or len(distances[i]) != n + 1:
            raise ValueError(f'distance_matrix[{i}] has an incorrect shape')
        if type(y[i]) is not list or len(y[i]) != n + 1:
            raise ValueError(f'y[{i}] has an incorrect destination-axis shape')
        for j in range(n + 1):
            distance = distances[i][j]
            numeric = (type(distance) is int or
                       (type(distance) is float and math.isfinite(distance)))
            if not numeric or distance < 0:
                raise ValueError(f'distance_matrix[{i}][{j}] must be a non-negative finite number')
            cars = y[i][j]
            if type(cars) is not list or len(cars) != k_count:
                raise ValueError(f'y[{i}][{j}] has an incorrect vehicle-axis shape')
            traversals = 0
            for h, days in enumerate(cars):
                if type(days) is not list or len(days) != t_count:
                    raise ValueError(f'y[{i}][{j}][{h}] has an incorrect day-axis shape')
                if any(type(value) is not int or value not in (0, 1) for value in days):
                    raise ValueError(f'y[{i}][{j}][{h}] must contain only the integers 0 or 1')
                traversals += sum(days)
            total += Fraction(str(distance)) * traversals
    try:
        return float(total)
    except OverflowError:
        return math.inf
