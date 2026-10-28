"""Corrected adapters for five problems (official format <-> prototype problem format).

These entries replace the entries of the same problems in `adapters.py` (see `run.get_adapter`):

- job_shop_scheduling, generalised_assignment_problem: the packaged algorithms are the ones of
  `adapters.py`, but its adapter for these two problems had two format bugs that were fixed before the
  reported results were produced: official machine ids of the job shop instances are 1-based and were not
  re-based to the 0-based ids of the problem format, and the official generalised assignment data are floats
  while the problem format (and its checker) use integers. The solution direction of the generalised
  assignment adapter also always maps the 0-based machine index to the official 1-based agent number.
- container_loading_with_weight_restrictions, p_median_capacitated, p_median_uncapacitated: the packaged
  algorithms were built for corrected versions of these problem types, whose instance and solution formats
  differ from the ones `adapters.py` was written for. The adapters below map the official data to the formats
  in each problem's `formulation.md` / `input_format.json` / `output_format.json`.

As in `adapters.py`, the adapters only move, rename, re-index and re-type fields.
"""
import math

from adapters import _g, _lst, _js_to_off


def _num(v):
    """Official integral floats -> int (the problem formats use integers); other values unchanged."""
    v = float(v)
    return int(v) if v.is_integer() else v


# ═══════════════ job_shop_scheduling ═══════════════
# Official: times[j][k], machines[j][k] (machine ids 1..n_machines in the Taillard files).
# Problem format: jobs[j][k] = [machine (0-based), processing time]; n = number of operations.
def _js_to_lib(off):
    t, m = off["times"], off["machines"]
    base = min(int(x) for r in m for x in r)      # machine id base (1 for the official files)
    jobs = [[[int(m[j][k]) - base, t[j][k]] for k in range(len(t[j]))] for j in range(len(t))]
    nj, nm = int(off["n_jobs"]), int(off["n_machines"])
    return {"n": nj * nm, "n_jobs": nj, "n_machines": nm, "jobs": jobs}


# The solution direction is unchanged: start_times[j][k] of the k-th operation of job j (`adapters._js_to_off`).


# ═══════════════ generalised_assignment_problem ═══════════════
# Official: m agents, n jobs, cost_matrix / consumption_matrix (m x n), capacities (m), all floats;
# solution assignments = list of n agent numbers (1..m).
# Problem format: num_machines, num_jobs, cost_matrix / resource_matrix (machine x job), capacities as integers;
# solution assignments = {"<job>": machine index 0..num_machines-1}.
def _gap_to_lib(off):
    return {"num_machines": int(off["m"]), "num_jobs": int(off["n"]),
            "cost_matrix": [[_num(v) for v in row] for row in off["cost_matrix"]],
            "resource_matrix": [[_num(v) for v in row] for row in off["consumption_matrix"]],
            "capacities": [_num(v) for v in _lst(off["capacities"])]}


def _gap_to_off(sol, off, lib):
    a = _g(sol, "assignments", "assignment") or {}
    n = int(off["n"])
    if isinstance(a, dict):
        out = [a.get(str(j), a.get(j)) for j in range(n)]
    else:
        out = _lst(a)[:n]
    return {"assignments": [int(v) + 1 if v is not None else 0 for v in out]}   # 0-based machine -> agent 1..m


# ═══════════════ container_loading_with_weight_restrictions ═══════════════
# Official box type: length, width, height, length_flag / width_flag / height_flag (orientation 1 / 2 / 3
# allowed), count, weight, lb1 / lb2 / lb3 (load-bearing pressure per orientation). Orientation codes are the
# same on both sides: 1 = length vertical, 2 = width vertical, 3 = height vertical.
# Official solution: flat list of placements {box_type (1-based), orientation, x, y, z}.
# Problem format solution: selection_status / box_positions / box_orientations indexed [box type][copy].
def _clw_to_lib(off):
    L, W, H = [float(x) for x in off["container"]]
    bt = off["box_types"]
    return {"container_length": L, "container_width": W, "container_height": H,
            "num_box_types": len(bt),
            "box_quantities": [int(b["count"]) for b in bt],
            "box_lengths": [float(b["length"]) for b in bt],
            "box_widths": [float(b["width"]) for b in bt],
            "box_heights": [float(b["height"]) for b in bt],
            "box_weights": [float(b["weight"]) for b in bt],
            "orientation_allowed": [[int(b["length_flag"]), int(b["width_flag"]), int(b["height_flag"])]
                                    for b in bt],
            "load_bearing_by_orientation": [[float(b["lb1"]), float(b["lb2"]), float(b["lb3"])] for b in bt]}


def _clw_to_off(sol, off, lib):
    sel = _g(sol, "selection_status") or []
    pos = _g(sol, "box_positions") or []
    ors = _g(sol, "box_orientations") or []
    out = []
    for t in range(len(sel)):
        row = _lst(sel[t])
        for i in range(len(row)):
            if not int(row[i]):
                continue
            x, y, z = (list(pos[t][i]) + [0, 0, 0])[:3]
            out.append({"box_type": t + 1, "orientation": int(ors[t][i]),
                        "x": float(x), "y": float(y), "z": float(z)})
    # `instance` and `util` are required arguments of the official eval_func but are not used by it.
    return {"placements": out, "instance": 0, "util": 0.0, "m": len(out)}


# ═══════════════ p_median_capacitated ═══════════════
# Official: customers = [(customer_id, x, y, demand)], p, capacity Q; the official cost of assigning customer i
# to median j is floor(Euclidean distance). Solution: medians and assignments as customer ids.
# Problem format: n, p, capacity, demands, distance_matrix[i][j] (customer i -> median candidate j = customer j);
# solution medians / assignments as 0-based customer indices.
def _pmc_to_lib(off):
    cu = off["customers"]
    xs = [float(c[1]) for c in cu]
    ys = [float(c[2]) for c in cu]
    n = len(cu)
    dm = [[math.floor(math.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)) for j in range(n)]
          for i in range(n)]
    return {"n": int(off["n"]), "p": int(off["p"]), "capacity": float(off["Q"]),
            "demands": [float(c[3]) for c in cu], "distance_matrix": dm}


def _pmc_to_off(sol, off, lib):
    ids = [int(c[0]) for c in off["customers"]]
    med = [ids[int(v)] for v in _lst(_g(sol, "medians"))]
    asg = [ids[int(v)] for v in _lst(_g(sol, "assignments"))]
    # `objective` must be present for the official eval_func, which recomputes the cost itself.
    return {"medians": med, "assignments": asg, "objective": 0.0}


# ═══════════════ p_median_uncapacitated ═══════════════
# Official: n, p, dist (n x n shortest-path matrix of floats); solution medians = p vertex numbers 1..n.
# Problem format: same n, p, dist (integers); solution chosen_medians = 0/1 vector of length n.
def _pmu_to_lib(off):
    return {"n": int(off["n"]), "p": int(off["p"]), "dist": [[_num(v) for v in row] for row in off["dist"]]}


def _pmu_to_off(sol, off, lib):
    ch = _lst(_g(sol, "chosen_medians"))
    return {"medians": [i + 1 for i, v in enumerate(ch) if int(v) == 1]}


TABLE = {
    "job_shop_scheduling": {"to_lib": _js_to_lib, "to_off": _js_to_off},
    "generalised_assignment_problem": {"to_lib": _gap_to_lib, "to_off": _gap_to_off},
    "container_loading_with_weight_restrictions": {"to_lib": _clw_to_lib, "to_off": _clw_to_off},
    "p_median_capacitated": {"to_lib": _pmc_to_lib, "to_off": _pmc_to_off},
    "p_median_uncapacitated": {"to_lib": _pmu_to_lib, "to_off": _pmu_to_off},
}
