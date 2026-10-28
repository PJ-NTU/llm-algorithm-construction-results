"""Converter for packing_unequal_rectangles_and_squares_area, for the portfolio updated by continued evolution (its format differs from the original).

Official: n, container centre (cx, cy), radius R, items [(L, W)] (L along x, W along y), shape, rotation
(may items turn by 90 degrees); solution placements = n triples (x, y, theta) of item centres, theta 0 or 90,
(-1, -1, 0) marking an item that is not packed (matched within 1e-5). The value is the packed area.
Ours (input_format.json / output_format.json in the problem folder): n, cx, cy, R, items [[L, W]] and
rotation_enabled (0/1) in the official frame; solution alpha (packed 0/1), x, y (centre, -1 when not packed),
r (1 = turned by 90 degrees).

Input: the original entry passed the official instance through unchanged, which lacks rotation_enabled and
gives items as tuples; this one writes the same values in our field names (rotation -> rotation_enabled).
Output: alpha = 0 -> (-1, -1, 0); alpha = 1 -> (x, y, 90 if r = 1 else 0), same frame. Wrong lengths,
values other than 0/1, or a packed centre that the official evaluator would read as its (-1, -1) marker raise.
"""
from adapters import _lst


def to_lib(off):
    return {"n": int(off["n"]), "cx": float(off["cx"]), "cy": float(off["cy"]), "R": float(off["R"]),
            "items": [[float(L), float(W)] for L, W in off["items"]],
            "rotation_enabled": 1 if off["rotation"] else 0}


def to_off(sol, off, lib):
    n = int(off["n"])
    alpha, xs, ys, rs = (_lst(sol.get(k)) for k in ("alpha", "x", "y", "r"))
    if any(len(v) != n for v in (alpha, xs, ys, rs)):
        raise ValueError("alpha, x, y, r must have %d entries each" % n)
    out = []
    for i in range(n):
        if alpha[i] not in (0, 1) or rs[i] not in (0, 1):
            raise ValueError("item %d: alpha = %r, r = %r are not 0/1" % (i, alpha[i], rs[i]))
        if alpha[i] == 0:
            out.append((-1.0, -1.0, 0))
            continue
        if abs(xs[i] + 1) < 1e-5 and abs(ys[i] + 1) < 1e-5:
            raise ValueError("item %d is packed at the official 'not packed' marker (-1, -1)" % i)
        out.append((xs[i], ys[i], 90 if rs[i] == 1 else 0))
    return {"placements": out}
