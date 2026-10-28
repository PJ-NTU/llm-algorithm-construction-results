"""Converter for uncapacitated_warehouse_location, for the portfolio updated by continued evolution (its format differs from the original).

Official: m warehouses, n customers, warehouses[i]['fixed_cost'], customers[j]['costs'][i] (cost of serving all of
customer j from warehouse i); solution warehouse_open = m values (1 = open), assignments = n x m
(assignments[j][i] = 1 iff customer j is served by warehouse i).
Ours (input_format.json / output_format.json in the problem folder): n, m_facilities, fixed_cost (m),
alloc_cost (m x n, facility x customer); solution y = m values (1 = open), x = m x n (x[i][j] = 1 iff customer j
is served by facility i).

Input: the original entry produces exactly these fields and is reused unchanged.
Output: y is passed through as warehouse_open; x is transposed to assignments. Values are not converted, so
the official eval_func judges them as given. A solution that is not an m-vector / m x n matrix has no faithful
transpose and raises (the official eval_func ignores entries beyond m / n, so it must not see such a solution).
"""
from adapters import _lst, _uwl_to_lib as to_lib


def to_off(sol, off, lib):
    m, n = int(off["m"]), int(off["n"])
    y = _lst(sol.get("y"))
    x = [_lst(row) for row in _lst(sol.get("x"))]
    if len(y) != m or len(x) != m or any(len(row) != n for row in x):
        raise ValueError("solution shape is not y[%d], x[%d][%d]: len(y)=%d, x rows=%d, row lengths=%s"
                         % (m, m, n, len(y), len(x), sorted({len(row) for row in x})))
    return {"warehouse_open": y, "assignments": [[x[i][j] for i in range(m)] for j in range(n)]}
