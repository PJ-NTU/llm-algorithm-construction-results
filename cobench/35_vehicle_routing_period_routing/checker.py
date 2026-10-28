"""PVRP feasibility check; an empty list means only that a complete assignment satisfies all hard constraints.

The input must satisfy the Euclidean-distance commitment of the fixed input_schema.json. This module checks the input's
fields, shapes and direct numeric domains; it does not reconstruct coordinates to prove the Euclidean origin of the distance matrix.
Load computations convert the decimal representation of finite JSON numbers to Fraction and compare exactly, with no
feasibility tolerance; the solution is not rounded, missing components are not patched, and the input object is not modified.
status='infeasible' is not a feasible assignment; this checker does not attempt a proof of global infeasibility.
"""

import math
from fractions import Fraction


_INPUT_FIELDS = {
    'num_customers', 'planning_period', 'num_vehicles', 'vehicle_capacity',
    'distance_matrix', 'demands', 'candidate_schedules', 'schedule_vectors',
}
_SOLUTION_FIELDS = {'status', 'x', 'y', 'u'}


def _number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _bit(value):
    return type(value) is int and value in (0, 1)


def _tensor(value, shape, predicate, path):
    """Check a complete dense array; raise ValueError on any structural or leaf-value error."""
    if not shape:
        if not predicate(value):
            raise ValueError(f'{path}: value or type does not match the field convention')
        return
    if type(value) is not list or len(value) != shape[0]:
        raise ValueError(f'{path}: a list of length {shape[0]} is required')
    for index, child in enumerate(value):
        _tensor(child, shape[1:], predicate, f'{path}[{index}]')


def _instance_dimensions(instance):
    if not isinstance(instance, dict):
        raise ValueError('instance must be a dict')
    missing = _INPUT_FIELDS - instance.keys()
    if missing:
        raise ValueError(f'missing input fields {sorted(missing)}')
    for key, minimum in (('num_customers', 0), ('num_vehicles', 1),
                         ('planning_period', 1)):
        if type(instance[key]) is not int or instance[key] < minimum:
            raise ValueError(f'{key}: an integer of at least {minimum} is required')
    n = instance['num_customers']
    k = instance['num_vehicles']
    t = instance['planning_period']
    capacity = instance['vehicle_capacity']
    if not _number(capacity) or capacity <= 0:
        raise ValueError('vehicle_capacity: must be a strictly positive finite number')
    _tensor(instance['demands'], (n,), lambda v: _number(v) and v > 0,
            'demands')
    distances = instance['distance_matrix']
    _tensor(distances, (n + 1, n + 1),
            lambda v: _number(v) and v >= 0, 'distance_matrix')
    for i in range(n + 1):
        if distances[i][i] != 0:
            raise ValueError(f'distance_matrix[{i}][{i}]: must be 0')
        for j in range(i):
            if distances[i][j] != distances[j][i]:
                raise ValueError(f'distance_matrix: distance between nodes {i} and {j} is asymmetric')
    patterns = instance['schedule_vectors']
    if type(patterns) is not list:
        raise ValueError('schedule_vectors: must be a list')
    m = len(patterns)
    _tensor(patterns, (m, t), _bit, 'schedule_vectors')
    for s, pattern in enumerate(patterns):
        if not any(pattern):
            raise ValueError(f'schedule_vectors[{s}]: all-zero schedules are not allowed')
    candidates = instance['candidate_schedules']
    if type(candidates) is not list or len(candidates) != n:
        raise ValueError(f'candidate_schedules: a list of length {n} is required')
    referenced = set()
    for p, row in enumerate(candidates):
        if type(row) is not list or not row:
            raise ValueError(f'candidate_schedules[{p}]: must be a non-empty list')
        if any(type(s) is not int or not 0 <= s < m for s in row):
            raise ValueError(f'candidate_schedules[{p}]: schedule index out of range or not an integer')
        if len(set(row)) != len(row):
            raise ValueError(f'candidate_schedules[{p}]: the same schedule index must not repeat')
        referenced.update(row)
    if referenced != set(range(m)):
        raise ValueError('schedule_vectors: there are schedule rows not referenced by any customer')
    return n, k, t


def check(instance: dict, solution: dict) -> list:
    """Return violation strings for structure, value domains and C1-C10; optimality is not judged."""
    try:
        n, k_count, t_count = _instance_dimensions(instance)
    except ValueError as exc:
        return [f'input: {exc}']
    if not isinstance(solution, dict):
        return ['structure: solution must be a dict']
    if set(solution) != _SOLUTION_FIELDS:
        return ['structure: the solution dict must contain exactly the four fields status, x, y, u']
    if solution['status'] == 'infeasible':
        if any(solution[key] is not None for key in ('x', 'y', 'u')):
            return ['structure: when status=infeasible, x, y, u must all be null']
        return ['status: an infeasible declaration contains no complete feasible assignment; this checker does not prove global infeasibility']
    if solution['status'] != 'feasible':
        return ['structure: status must be feasible or infeasible']

    x, y, u = solution['x'], solution['y'], solution['u']
    candidates = instance['candidate_schedules']
    patterns = instance['schedule_vectors']
    try:
        if type(x) is not list or len(x) != n:
            raise ValueError(f'x: a list of length {n} is required')
        for p, row in enumerate(x):
            _tensor(row, (len(candidates[p]),), _bit, f'x[{p}]')
        _tensor(y, (n + 1, n + 1, k_count, t_count), _bit, 'y')
        _tensor(u, (n, k_count, t_count), _number, 'u')
    except ValueError as exc:
        return [f'structure or value domain: {exc}']

    q = [Fraction(str(value)) for value in instance['demands']]
    capacity = Fraction(str(instance['vehicle_capacity']))
    loads = [[[Fraction(str(value)) for value in days] for days in cars]
             for cars in u]
    incoming = [[[sum(y[j][i][h][t] for j in range(n + 1) if j != i)
                  for t in range(t_count)] for h in range(k_count)]
                for i in range(n + 1)]
    violations = []

    # C1
    for p, row in enumerate(x):
        if sum(row) != 1:
            violations.append(f'C1: customer {p + 1} must select exactly one candidate schedule')

    # C2
    for p in range(n):
        for t in range(t_count):
            required = sum(patterns[s][t] * x[p][r]
                           for r, s in enumerate(candidates[p]))
            actual = sum(incoming[p + 1][h][t] for h in range(k_count))
            if actual != required:
                violations.append(f'C2: customer {p + 1} day {t}: visit count '
                                  f'{actual} does not equal the requirement of the selected schedule, {required}')

    # C3
    for i in range(n + 1):
        for h in range(k_count):
            for t in range(t_count):
                outgoing = sum(y[i][j][h][t] for j in range(n + 1) if j != i)
                if incoming[i][h][t] != outgoing:
                    violations.append(f'C3: node {i} vehicle {h + 1} day {t}: '
                                      'number of entries differs from number of exits')

    # C4
    for h in range(k_count):
        for t in range(t_count):
            if sum(y[0][j][h][t] for j in range(1, n + 1)) > 1:
                violations.append(f'C4: vehicle {h + 1} day {t} departs from the depot more than once')

    # C5
    for h in range(k_count):
        for t in range(t_count):
            total = sum(q[p] * incoming[p + 1][h][t] for p in range(n))
            if total > capacity:
                violations.append(f'C5: vehicle {h + 1} day {t}: total demand exceeds capacity')

    # C6
    for p in range(n):
        for r in range(n):
            if p == r:
                continue
            for h in range(k_count):
                for t in range(t_count):
                    arc = y[p + 1][r + 1][h][t]
                    if loads[p][h][t] + q[r] > loads[r][h][t] + capacity * (1 - arc):
                        violations.append(f'C6: customer {p + 1}→{r + 1} '
                                          f'vehicle {h + 1} day {t} violates the load-increase lower bound')

    # C7
    for p in range(n):
        for h in range(k_count):
            for t in range(t_count):
                if not q[p] <= loads[p][h][t] <= capacity:
                    violations.append(f'C7: customer {p + 1} vehicle {h + 1} day {t}: '
                                      'load is not between the customer demand and the vehicle capacity')

    # C8
    for i in range(n + 1):
        for h in range(k_count):
            for t in range(t_count):
                if y[i][i][h][t] != 0:
                    violations.append(f'C8: node {i} vehicle {h + 1} day {t} has a self-loop')

    # C9
    for p in range(n):
        for h in range(k_count):
            for t in range(t_count):
                limit = q[p] + capacity * (1 - y[0][p + 1][h][t])
                if loads[p][h][t] > limit:
                    violations.append(f'C9: customer {p + 1} vehicle {h + 1} day {t} '
                                      'violates the cumulative-load upper bound for the first customer')

    # C10
    for p in range(n):
        for r in range(n):
            if p == r:
                continue
            for h in range(k_count):
                for t in range(t_count):
                    arc = y[p + 1][r + 1][h][t]
                    limit = loads[p][h][t] + q[r] + capacity * (1 - arc)
                    if loads[r][h][t] > limit:
                        violations.append(f'C10: customer {p + 1}→{r + 1} '
                                          f'vehicle {h + 1} day {t} violates the exact-increment upper bound')

    return violations
