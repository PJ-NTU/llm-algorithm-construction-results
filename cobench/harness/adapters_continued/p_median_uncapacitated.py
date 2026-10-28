"""Converter for p_median_uncapacitated, for the portfolio updated by continued evolution (its format differs from the original).

Official: n, m, p, dist (n x n shortest-path matrix of floats, rows = served vertex, columns = median);
solution medians = list of exactly p distinct int vertex numbers 1..n.
Ours (input_format.json / output_format.json in the problem folder): n, p, dist (same orientation, ints legal,
null = unreachable); solution medians = list of p distinct 0-based vertex indices.

Input: the field names and meaning are those of the original corrected entry, which is reused unchanged
(the 40 official instances are connected and integral, so no +inf / null arises).
Output: only re-based 0 -> 1; length, order and duplicates are passed through for the official eval_func to judge.
"""
import operator

from adapters import _lst
from adapters_corrected import _pmu_to_lib as to_lib


def to_off(sol, off, lib):
    return {"medians": [operator.index(v) + 1 for v in _lst(sol.get("medians"))]}
