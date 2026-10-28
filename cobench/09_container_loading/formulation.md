# Container loading: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
Given a single container of dimensions (L, W, H) and N types of boxes, where each type i has dimensions (l_i, w_i, h_i), a total quantity Q_i, and orientation constraints, the goal is to determine the selection, orientation, and 3D coordinates of boxes to maximize space utilization. All indices start from 0. Let M be the total number of potential box instances (sum of Q_i).

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[j], y[j], z[j]: The coordinates of the front-bottom-left corner of box instance j in the container.  |  x[j] ≥ 0, y[j] ≥ 0, z[j] ≥ 0, continuous
- s[j]: Binary indicator whether box instance j is selected to be placed in the container.  |  s[j] ∈ {0, 1}
- o[j][k]: Binary indicator for the orientation of box instance j, where k represents one of the 6 possible rotation states (respecting vertical constraints).  |  o[j][k] ∈ {0, 1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each selected box must lie entirely within the container boundary.  |  s[j] * (x[j] + \sum_{k=0}^5 o[j][k] \cdot l'_{j,k}) \le L, \\ s[j] * (y[j] + \sum_{k=0}^5 o[j][k] \cdot w'_{j,k}) \le W, \\ s[j] * (z[j] + \sum_{k=0}^5 o[j][k] \cdot h'_{j,k}) \le H, \quad \forall j \in \{0..M-1\}
- [hard] No two selected boxes may overlap in space.  |  \forall j, i \in \{0..M-1\}, j \neq i: \\ (s[j] \cdot s[i] = 1) \implies (x[j] + dx_j \le x[i]) \lor (x[i] + dx_i \le x[j]) \lor \\ (y[j] + dy_j \le y[i]) \lor (y[i] + dy_i \le y[j]) \lor \\ (z[j] + dz_j \le z[i]) \lor (z[i] + dz_i \le z[j]) \\ \text{where } dx_j = \sum_{k=0}^5 o[j][k] \cdot l'_{j,k} \text{ (and similarly for } dy, dz\text{)}
- [hard] If a box is selected, exactly one valid rotation orientation must be chosen for it.  |  \sum_{k=0}^5 o[j][k] = s[j], \quad \forall j \in \{0..M-1\}
- [hard] Box rotation must respect its vertical orientation constraint (i.e. certain axes must stay vertical).  |  o[j][k] = 0 \quad \forall k \in \text{InvalidOrientations}_j

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- container_length: L; unit=mm
- container_width: W; unit=mm
- container_height: H; unit=mm
- total_box_instances: M
- boxes.volume: v[j]; unit=mm^3; axis: box instance
- boxes.rotated_lengths: l'_{j,k}; unit=mm; axis: box instance × rotation state
- boxes.rotated_widths: w'_{j,k}; unit=mm; axis: box instance × rotation state
- boxes.rotated_heights: h'_{j,k}; unit=mm; axis: box instance × rotation state
- boxes.forbidden_orientations: InvalidOrientations_j; axis: box instance

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: maximize
Formula: max (sum_{j=0}^{M-1} s[j] * v[j]) / (L * W * H)
Plain: The total volume of all successfully placed boxes divided by the total volume of the container.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"container_length": float, "container_width": float, "container_height": float, "total_box_instances": int, "boxes.volume": [float, ...], "boxes.rotated_lengths": [[float, ...], ...], "boxes.rotated_widths": [[float, ...], ...], "boxes.rotated_heights": [[float, ...], ...], "boxes.forbidden_orientations": [[any, ...]|[int, ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "placement_status": list[int] — whether each box instance is placed into the container. example: [1, 1, 0, 1]. Lists the selection status of all M candidate box instances (1 = selected, 0 = not selected)
- "coordinates": list[list[float]] — 3D coordinates of each box in the container (front-left-bottom corner). example: [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 40.0, 10.0]]. Lists the [x, y, z] coordinates of all M box instances; unselected boxes are filled with [0.0, 0.0, 0.0]
- "orientations": list[int] — rotation orientation index (0-5) chosen for each box. example: [0, 2, -1, 5]. Lists the orientation index of all M box instances; unselected boxes may use -1 or 0
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: max

**Formula**
```
max (sum_{j=0}^{M-1} s[j] * v[j]) / (L * W * H)
```

**Plain-language statement**
The total volume of all successfully placed boxes divided by the total volume of the container.

**Rationale**
The goal is to maximize the volume utilization ratio, which is equivalent to maximizing the total volume of placed boxes since the container volume is constant.
