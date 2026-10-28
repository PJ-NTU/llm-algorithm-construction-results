"""Converter for packing_unequal_rectangles_and_squares, for the portfolio updated by continued evolution (its format differs from the original).

Official: n, container centre (cx, cy), radius R, items [(L, W)] (L along x, W along y), shape, rotation
(may items turn by 90 degrees); solution placements = n triples (x, y, theta) of item centres, theta 0 or 90,
(-1, -1, theta) marking an item that is not packed. The value is the number of packed items.
Ours (input_format.json / output_format.json in the problem folder): num_rectangles, widths (along x), heights
(along y), container_radius with the centre at the origin, big_m; solution alpha (packed 0/1), x, y (centre),
r (1 = turned by 90 degrees, widths and heights swap). Turning is always allowed in our format.

Input: the original entry is reused (centre moved to the origin). A case that forbids turning and has an item
with L != W cannot be expressed (our format always allows turning), so it raises; this is the rotation-free
case 0 of rect1-3. A square turned by 90 degrees covers the same region, so the square cases are faithful.
Output: alpha = 0 -> (-1, -1, 0); alpha = 1 -> (x + cx, y + cy, 90 if r = 1 else 0), except that a turned square
in a case without turning is written with theta 0 (same region). Coordinates of unpacked items are not passed
on (the official format has none). Wrong lengths, values other than 0/1, or a packed centre that would land
exactly on the official (-1, -1) marker raise.
"""
from adapters import _lst, _purs_to_lib


def to_lib(off):
    if not off["rotation"] and any(L != W for L, W in off["items"]):
        raise ValueError("not representable in the evolved format: rotation not allowed for non-square items")
    return _purs_to_lib(off)


def to_off(sol, off, lib):
    n, cx, cy = int(off["n"]), float(off["cx"]), float(off["cy"])
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
        L, W = off["items"][i]
        if rs[i] == 1 and not off["rotation"]:
            if L != W:
                raise ValueError("item %d is turned but the case does not allow turning" % i)
            theta = 0
        else:
            theta = 90 if rs[i] == 1 else 0
        px, py = xs[i] + cx, ys[i] + cy
        if px == -1 and py == -1:
            raise ValueError("item %d is packed at the official 'not packed' marker (-1, -1)" % i)
        out.append((px, py, theta))
    return {"placements": out}
