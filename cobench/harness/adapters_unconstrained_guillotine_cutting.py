"""Adapter for unconstrained_guillotine_cutting (official format <-> prototype problem format)."""


# ── Problem 36 unconstrained_guillotine_cutting ────────────────────────────────
# Official instance {m, stock_width, stock_height, pieces:{pid->{l,w,value}}} (pid 1-indexed)
# Official solution {placements:[{piece_id,x,y,orientation}]}
#          orientation=0 → width=l height=w; =1 → width=w height=l (config.py:177-183)
# Library instance {num_parts, sheet_width, sheet_height, part_widths[], part_heights[],
#           part_values[], big_M}
# Library solution {selection[0/1], bottom_left_x[], bottom_left_y[], left_relations, below_relations}
# The library problem type is the [rotation-forbidden] one (lib_77867f3818b7), so orientation is always 0.

def _ugc_ids(off_inst):
    return sorted(off_inst["pieces"], key=lambda k: int(k))


def _ugc_to_lib(off_inst):
    ids = _ugc_ids(off_inst)
    p = off_inst["pieces"]
    sw, sh = float(off_inst["stock_width"]), float(off_inst["stock_height"])
    return {
        "num_parts": int(off_inst["m"]),
        "sheet_width": sw,
        "sheet_height": sh,
        "part_widths": [float(p[k]["l"]) for k in ids],    # orientation 0: width is l
        "part_heights": [float(p[k]["w"]) for k in ids],   # height is w
        "part_values": [float(p[k]["value"]) for k in ids],
        "big_M": max(sw, sh) * 10.0,
    }


def _ugc_to_off(lib_sol, off_inst, lib_inst):
    ids = _ugc_ids(off_inst)
    sel = (lib_sol or {}).get("selection") or []
    xs = (lib_sol or {}).get("bottom_left_x") or []
    ys = (lib_sol or {}).get("bottom_left_y") or []
    out = []
    for i, pid in enumerate(ids):
        if i >= len(sel):
            break
        try:
            on = int(round(float(sel[i])))
        except Exception:
            continue
        if on != 1:
            continue
        x = int(round(float(xs[i]))) if i < len(xs) else 0
        y = int(round(float(ys[i]))) if i < len(ys) else 0
        out.append({"piece_id": int(pid), "x": x, "y": y, "orientation": 0})
    return {"placements": out}


ENTRY = {"to_lib": _ugc_to_lib, "to_off": _ugc_to_off}
