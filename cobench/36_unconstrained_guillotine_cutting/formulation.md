# Unconstrained guillotine cutting: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Given a rectangular sheet of width W and height H, and N rectangular parts (each part i has width w_i, height h_i and value v_i), parts cannot be rotated, must be placed entirely inside the sheet, and must not overlap each other. Select a subset of the parts and determine their positions so that the total value of the selected parts is maximised. Indices start from 0. N, W, H, w_i, h_i and v_i are symbolic quantities; their concrete values are given by the input data.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- select[i]: 1 if part i is selected, otherwise 0.  |  select[i] ∈ {0,1}
- x[i]: x coordinate (horizontal) of the bottom-left corner of part i, ranging from 0 to W-w_i. Meaningful only when select[i]=1.  |  x[i] ∈ [0, W - w_i]
- y[i]: y coordinate (vertical) of the bottom-left corner of part i, ranging from 0 to H-h_i. Meaningful only when select[i]=1.  |  y[i] ∈ [0, H - h_i]
- left[i][j]: 1 if part i lies entirely to the left of part j (i.e. x[i] + w_i ≤ x[j]), otherwise 0. Used to prevent overlap.  |  left[i][j] ∈ {0,1}
- below[i][j]: 1 if part i lies entirely below part j (i.e. y[i] + h_i ≤ y[j]), otherwise 0. Used to prevent overlap.  |  below[i][j] ∈ {0,1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Any two distinct parts are separated in at least one direction (left/right/above/below), guaranteeing no overlap  |  left[i][j] + left[j][i] + below[i][j] + below[j][i] >= select[i] + select[j] - 1,  ∀ i < j
- [hard] Horizontal coordinate linking: if left[i][j]=1, part i must be to the left of part j  |  x[i] + w_i <= x[j] + M * (1 - left[i][j]),  ∀ i ≠ j
- [hard] Vertical coordinate linking: if below[i][j]=1, part i must be below part j  |  y[i] + h_i <= y[j] + M * (1 - below[i][j]),  ∀ i ≠ j
- [hard] Horizontal exclusivity: left and right relations cannot hold simultaneously  |  left[i][j] + left[j][i] <= 1,  ∀ i < j
- [hard] Vertical exclusivity: above and below relations cannot hold simultaneously  |  below[i][j] + below[j][i] <= 1,  ∀ i < j
- [hard] left/below variables can be 1 only when both corresponding parts are selected  |  left[i][j] <= select[i]; left[i][j] <= select[j]; below[i][j] <= select[i]; below[i][j] <= select[j],  ∀ i ≠ j
- [hard] Coordinate variables may be non-zero only when the part is selected  |  x[i] <= (W - w_i) * select[i]; y[i] <= (H - h_i) * select[i],  ∀ i

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- num_parts: N
- sheet_width: W; unit=mm
- sheet_height: H; unit=mm
- part_widths: w_i; unit=mm; axis: N
- part_heights: h_i; unit=mm; axis: N
- part_values: v_i; unit=CNY; axis: N
- big_M: M

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: maximize
Formula: maximize ∑_{i=0}^{N-1} v_i * select[i]
Plain: Maximise the sum of the values of all selected parts.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"num_parts": int, "sheet_width": float, "sheet_height": float, "part_widths": [float, ...], "part_heights": [float, ...], "part_values": [float, ...], "big_M": float}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "selection": list[int] (length equals the total number of parts N, elements are 0 or 1) — whether each part is selected for cutting. example: [1, 1, 0]. List all parts, including unselected ones (value 0)
- "bottom_left_x": list[float] (length N) — horizontal coordinate of the bottom-left corner of each part (meaningful only when selected; forced to 0 when unselected). example: [0.0, 30.0, 0.0]. List all parts; coordinates are set to 0 when unselected
- "bottom_left_y": list[float] (length N) — vertical coordinate of the bottom-left corner of each part. example: [0.0, 0.0, 0.0]. List all parts; coordinates are set to 0 when unselected
- "left_relations": list[list[int]] (N×N matrix, rows correspond to i and columns to j, elements are 0 or 1, 0 on the diagonal) — whether part i lies entirely to the left of part j (x[i]+w_i ≤ x[j]). example: [[0, 1, 0], [0, 0, 0], [0, 0, 0]]. Includes all pairs i,j (0≤i,j<N), with the diagonal set to 0
- "below_relations": list[list[int]] (N×N matrix, rows correspond to i and columns to j, elements are 0 or 1, 0 on the diagonal) — whether part i lies entirely below part j (y[i]+h_i ≤ y[j]). example: [[0, 0, 0], [0, 0, 0], [0, 0, 0]]. Includes all pairs i,j
Fill every value for every entity; keep the keys EXACTLY as above.
