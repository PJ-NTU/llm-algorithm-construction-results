# Packing unequal circles area: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
The problem involves selecting a subset of unequal circles from a given set and placing them into a fixed circular container of radius R centered at (cx, cy) such that no two circles overlap and all selected circles are contained within the container's boundary. The goal is to maximize the total area of the packed circles. Indices for circles range from 0 to N-1.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- alpha[i]: Binary indicator whether circle i is selected to be packed  |  alpha[i] ∈ {0, 1}
- x[i]: The x-coordinate of the center of circle i  |  x[i] ∈ ℝ
- y[i]: The y-coordinate of the center of circle i  |  y[i] ∈ ℝ

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each packed circle must lie entirely within the circular container. If a circle is not packed, its center is forced to the container's center.  |  (x[i] - cx)^2 + (y[i] - cy)^2 <= (alpha[i] * (R - radii[i]))^2 + (1 - alpha[i]) * 0
- [hard] Any two packed circles must not overlap with each other.  |  (x[i] - x[j])^2 + (y[i] - y[j])^2 >= (alpha[i] * alpha[j] * (radii[i] + radii[j]))^2
- [hard] If a circle is not packed, its coordinates must be set to the container's center (cx, cy).  |  x[i] = alpha[i] * x[i] + (1 - alpha[i]) * cx, y[i] = alpha[i] * y[i] + (1 - alpha[i]) * cy

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n: N
- cx: cx
- cy: cy
- R: R
- radii: radii; axis: circle index

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: maximize
Formula: max ∑_{i=0}^{N-1} alpha[i] * π * radii[i]^2
Plain: Maximize the total area of all circles successfully packed into the container.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "cx": float, "cy": float, "R": float, "radii": [float, ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "packing_status": list[int] — Whether each circle is placed in the container. example: [1, 0, 1, 1]. List the selection status of all n circles; 1 = placed, 0 = not placed
- "center_x_coordinates": list[float] — x-coordinate of each circle's center. example: [0.5, 0.0, -1.2, 2.1]. List the x-coordinates of all n circles; if not placed, fill in the container center coordinate cx
- "center_y_coordinates": list[float] — y-coordinate of each circle's center. example: [0.5, 0.0, 1.5, -0.8]. List the y-coordinates of all n circles; if not placed, fill in the container center coordinate cy
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: max

**Formula**
```
max ∑_{i=0}^{N-1} alpha[i] * π * radii[i]^2
```

**Plain-language statement**
Maximize the total area of all circles successfully packed into the container.

**Rationale**
The goal is to utilize the container's area as efficiently as possible by selecting the best combination of circles.
