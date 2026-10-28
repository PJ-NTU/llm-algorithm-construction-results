"""Converter for p_median_capacitated, for the portfolio updated by continued evolution (its format
requires an integer capacity and integer demands).

Official: n, p, Q, customers [id, x, y, demand]; solution medians / assignments as customer ids.
Ours: the original corrected entry (n, p, capacity, demands, floored Euclidean distance_matrix), with capacity and
demands typed as integers. The official data are integral, so this is an exact re-typing; a non-integral value
raises instead of being rounded. The solution side is the original one.
"""
from adapters_corrected import _pmc_to_lib
from adapters_corrected import _pmc_to_off as to_off


def _int(v, what):
    if float(v) != int(float(v)):
        raise ValueError("%s %r is not an integer, which the updated format cannot represent" % (what, v))
    return int(float(v))


def to_lib(off):
    lib = _pmc_to_lib(off)
    lib["capacity"] = _int(lib["capacity"], "capacity")
    lib["demands"] = [_int(d, "demand") for d in lib["demands"]]
    return lib
