# Packing unequal circles: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
The problem involves packing a subset of unequal circles into a fixed circular container of radius R centered at (cx, cy). Given N circles with radii R_i (sorted non-decreasingly), the goal is to select and position as many circles as possible such that each selected circle is entirely within the container and no two selected circles overlap. All indices start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- alpha[i]: Binary indicator whether circle i is packed into the container  |  alpha[i] ∈ {0, 1}
- x[i]: The x-coordinate of the center of circle i  |  x[i] ∈ ℝ
- y[i]: The y-coordinate of the center of circle i  |  y[i] ∈ ℝ

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each selected circle must lie entirely inside the container, i.e. the distance from the circle center to the container center plus the circle's radius must not exceed the container radius.  |  alpha[i] \cdot \sqrt{(x[i] - cx)^2 + (y[i] - cy)^2} \le alpha[i] \cdot (R - R_i), \forall i \in \{0, \dots, N-1\}
- [hard] No two selected circles may overlap, i.e. the distance between their centers must be greater than or equal to the sum of their radii.  |  \sqrt{(x[i] - x[j])^2 + (y[i] - y[j])^2} \ge alpha[i] \cdot alpha[j] \cdot (R_i + R_j), \forall i, j \in \{0, \dots, N-1\}, i < j
- [hard] Circles must be packed in order of increasing radius: if circle i is selected, all circles with smaller radius (smaller index) must also be selected.  |  alpha[i] \ge alpha[i+1], \forall i \in \{0, \dots, N-2\}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n: N
- cx: cx
- cy: cy
- R: R
- radii: R_i; axis: circle_index

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: maximize
Formula: max \sum_{i=0}^{N-1} alpha[i]
Plain: Maximize the total number of circles successfully packed into the container.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "cx": float, "cy": float, "R": float, "radii": [float, ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "packing_status": list[int] — Whether each circle is selected and packed into the container. example: [1, 1, 1, 0, 0]. List the selection status of all n circles (1 = selected, 0 = not selected)
- "centers": list[list[float]] — Coordinates of each circle's center. example: [[0.5, 0.5], [1.2, -0.3], [-0.8, 0.1], [-1.0, -1.0], [-1.0, -1.0]]. List the (x, y) coordinates of all n circles; unselected circles default to (-1.0, -1.0)
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: max

**Formula**
```
max \sum_{i=0}^{N-1} alpha[i]
```

**Plain-language statement**
Maximize the total number of circles successfully packed into the container.

**Rationale**
The primary goal stated is to maximize the count of circles that fit within the constraints.
