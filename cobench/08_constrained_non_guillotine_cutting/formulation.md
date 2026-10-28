# Constrained non-guillotine cutting: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
The problem involves placing a subset of rectangular pieces from N available types onto a single rectangular stock of dimensions (L, W). Each piece type i has dimensions (l_i, w_i), a value v_i, and usage bounds [min_i, max_i]. Pieces can be rotated 90 degrees and must not overlap or exceed stock boundaries. The goal is to maximize the total value of placed pieces. All indices start from 0. Let K be the maximum possible number of pieces that could fit (an upper bound on total items).

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[k]: Binary variable indicating if the k-th placement slot is used  |  x[k] ∈ {0,1}
- t[k][i]: Binary variable indicating if the k-th placement slot contains piece type i  |  t[k][i] ∈ {0,1}
- pos_x[k], pos_y[k]: Continuous coordinates of the bottom-left corner of the piece in slot k  |  pos_x[k] ≥ 0, pos_y[k] ≥ 0
- r[k]: Binary variable indicating if the piece in slot k is rotated 90 degrees  |  r[k] ∈ {0,1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each placement slot can be associated with at most one piece type, and only when that slot is activated.  |  ∑_{i=0}^{N-1} t[k][i] = x[k], ∀k
- [hard] Pieces must lie entirely within the stock sheet boundary (accounting for rotation).  |  pos_x[k] + ∑_{i=0}^{N-1} t[k][i] * (l_i * (1 - r[k]) + w_i * r[k]) ≤ L, ∀k
pos_y[k] + ∑_{i=0}^{N-1} t[k][i] * (w_i * (1 - r[k]) + l_i * r[k]) ≤ W, ∀k
- [hard] No two placed pieces may overlap.  |  Let W_k = ∑_{i=0}^{N-1} t[k][i] * (l_i * (1 - r[k]) + w_i * r[k])
Let H_k = ∑_{i=0}^{N-1} t[k][i] * (w_i * (1 - r[k]) + l_i * r[k])
For k < m: 
pos_x[k] + W_k ≤ pos_x[m] + M(3 - x[k] - x[m] - b1_{km})
pos_x[m] + W_m ≤ pos_x[k] + M(3 - x[k] - x[m] - b2_{km})
pos_y[k] + H_k ≤ pos_y[m] + M(3 - x[k] - x[m] - b3_{km})
pos_y[m] + H_m ≤ pos_y[k] + M(3 - x[k] - x[m] - b4_{km})
b1_{km} + b2_{km} + b3_{km} + b4_{km} ≥ 1
- [hard] The number of pieces used of each type must be within the specified minimum and maximum.  |  min_q_i ≤ ∑_{k=0}^{K-1} t[k][i] ≤ max_q_i, ∀i

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- stock_length: L; unit=mm
- stock_width: W; unit=mm
- piece_values: v_i; axis: piece_type
- piece_lengths: l_i; unit=mm; axis: piece_type
- piece_widths: w_i; unit=mm; axis: piece_type
- min_usage: min_q_i; unit=pcs; axis: piece_type
- max_usage: max_q_i; unit=pcs; axis: piece_type
- max_slots: K
- num_piece_types: N
- big_m: M

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: maximize
Formula: max ∑_{k=0}^{K-1} ∑_{i=0}^{N-1} (v_i * t[k][i])
Plain: Maximize the sum of values of all pieces successfully placed on the stock.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"stock_length": float, "stock_width": float, "num_piece_types": int, "max_slots": int, "piece_lengths": [float, ...], "piece_widths": [float, ...], "piece_values": [float, ...], "min_usage": [int, ...], "max_usage": [int, ...], "big_m": float}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "placements": list[dict] — status of each potential placement slot (whether used, piece type, coordinates and rotation). example: [{"is_used": 1, "piece_type_id": 0, "x": 0.0, "y": 0.0, "is_rotated": 0}, {"is_used": 1, "piece_type_id": 1, "x": 50.5, "y": 20.0, "is_rotated": 1}, {"is_used": 0, "piece_type_id": -1, "x": 0.0, "y": 0.0, "is_rotated": 0}]. Lists all K potential placement slots; for unused slots set is_used to 0 and piece_type_id to -1
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: max

**Formula**
```
max ∑_{k=0}^{K-1} ∑_{i=0}^{N-1} (v_i * t[k][i])
```

**Plain-language statement**
Maximize the sum of values of all pieces successfully placed on the stock.

**Rationale**
The objective is to find the most valuable combination and arrangement of pieces within the given constraints.
