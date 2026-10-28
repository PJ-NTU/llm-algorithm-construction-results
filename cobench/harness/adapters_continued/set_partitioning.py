"""Converter for set_partitioning, for the portfolio updated by continued evolution (its format differs from the original).

Official: num_rows, num_columns, columns_info {column 1..num_columns: (cost, {rows 1..num_rows})};
solution selected_columns = strictly increasing list of column numbers 1..num_columns.
Ours (input_format.json / output_format.json in the problem folder): n = columns, m = rows, costs[n],
covers[n] (0-based row lists); solution feasible (bool), x = n values 0/1, total_cost.

Input: the original entry produces exactly these fields and is reused unchanged.
Output: x[j] == 1 becomes column j + 1 (ascending by construction). total_cost is not passed on: the official
eval_func recomputes the cost. A solution reporting no feasible partition (feasible not true), or an x that is
not n values 0/1, has no official representation and raises.
"""
from adapters import _lst, _sp_to_lib as to_lib


def to_off(sol, off, lib):
    n = int(off["num_columns"])
    if sol.get("feasible") is not True:
        raise ValueError("solution reports no feasible partition (feasible=%r)" % (sol.get("feasible"),))
    x = _lst(sol.get("x"))
    if len(x) != n or any(v not in (0, 1) for v in x):
        raise ValueError("x is not %d values 0/1: len(x)=%d, values=%s" % (n, len(x), sorted({repr(v) for v in x})[:5]))
    return {"selected_columns": [j + 1 for j, v in enumerate(x) if v == 1]}
