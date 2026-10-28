"""Converter for container_loading_with_weight_restrictions, for the portfolio updated by continued evolution (its format differs from the original).

Official: container (L, W, H), box_types [{length, width, height, length_flag / width_flag / height_flag, count,
weight, lb1 / lb2 / lb3}]; solution placements = [{box_type 1..n, orientation 1..3, x, y, z}] (front-bottom-left
corner), plus the required but unused arguments instance, util, m.
Ours (input_format.json / output_format.json in the problem folder): the fields of the original corrected entry;
solution s, o, x, y, z, each [box type i][copy k] with box_quantities[i] copies; s = 1 places the copy.
Orientation codes and the corner convention are the same on both sides (1 / 2 / 3 = original length / width /
height vertical, with the same (dx, dy, dz) per code).

Input: the original corrected entry is reused unchanged.
Output: every copy with s = 1 becomes one placement with box_type i + 1 and its o, x, y, z as given.
Unplaced copies are not listed (the official format lists placed boxes only). A solution whose s, o, x, y, z do
not have one entry per copy, or whose s is not 0/1, has no faithful reading and raises.
"""
from adapters import _lst
from adapters_corrected import _clw_to_lib as to_lib


def to_off(sol, off, lib):
    qty = lib["box_quantities"]
    rows = {key: [_lst(r) for r in _lst(sol.get(key))] for key in ("s", "o", "x", "y", "z")}
    for key, r in rows.items():
        if len(r) != len(qty) or any(len(r[i]) != q for i, q in enumerate(qty)):
            raise ValueError("%s is not one entry per box copy %s" % (key, qty))
    out = []
    for i, q in enumerate(qty):
        for k in range(q):
            s = rows["s"][i][k]
            if s not in (0, 1):
                raise ValueError("s[%d][%d] = %r is not 0/1" % (i, k, s))
            if s == 1:
                out.append({"box_type": i + 1, "orientation": rows["o"][i][k],
                            "x": rows["x"][i][k], "y": rows["y"][i][k], "z": rows["z"][i][k]})
    # `instance` and `util` are required arguments of the official eval_func but are not used by it.
    return {"placements": out, "instance": 0, "util": 0.0, "m": len(out)}
