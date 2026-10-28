# Constrained guillotine cutting: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
The problem is a two-dimensional guillotine cutting stock problem where rectangular pieces must be placed on a single stock sheet of dimensions (L, W) to maximize total value. Each piece type i has dimensions (l_i, w_i), a value v_i, and a maximum demand d_i. A placement is valid only if it follows a recursive guillotine cutting pattern, meaning every rectangular region containing pieces can be split into two smaller rectangles by a straight horizontal or vertical cut without intersecting any piece. All indices start from 0. N denotes the number of piece types.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[i][p]: Binary variable indicating if the p-th instance of piece type i is selected and placed on the sheet.  |  x[i][p] ∈ {0,1}, for i ∈ {0..N-1}, p ∈ {0..d_i-1}
- pos_x[i][p], pos_y[i][p]: The bottom-left coordinates (x, y) of the p-th instance of piece type i on the stock sheet.  |  pos_x[i][p] ∈ [0, L - l_i], pos_y[i][p] ∈ [0, W - w_i]
- cut_structure: A recursive tree structure representing the sequence and orientation (horizontal/vertical) of guillotine cuts.  |  Tree of (cut_type, coordinate)

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each rectangular piece must lie entirely within the stock sheet boundary.  |  x[i][p] = 1 \implies (pos\_x[i][p] + l_i \le L) \land (pos\_y[i][p] + w_i \le W)
- [hard] No two selected rectangular pieces may overlap.  |  x[i][p]=1 \land x[j][q]=1 \implies (pos\_x[i][p] + l_i \le pos\_x[j][q]) \lor (pos\_x[j][q] + l_j \le pos\_x[i][p]) \lor (pos\_y[i][p] + w_i \le pos\_y[j][q]) \lor (pos\_y[j][q] + w_j \le pos\_y[i][p])
- [hard] The layout of all selected rectangular pieces must satisfy the guillotine constraint, i.e. they can be separated by a sequence of straight cuts running across the whole region.  |  is\_guillotine(x, pos\_x, pos\_y, cut\_structure) = True
- [hard] The number of selected pieces of each type must not exceed its maximum demand.  |  \sum_{p=0}^{d_i-1} x[i][p] \le d_i, \forall i \in \{0..N-1\}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- sheet_length: L; unit=mm; axis: scalar
- sheet_width: W; unit=mm; axis: scalar
- num_piece_types: N; axis: scalar
- piece_lengths: l_i; unit=mm; axis: piece type index
- piece_widths: w_i; unit=mm; axis: piece type index
- max_quantities: d_i; unit=pieces; axis: piece type index
- piece_values: v_i; unit=value unit; axis: piece type index

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: maximize
Formula: max Σ_{i=0}^{N-1} Σ_{p=0}^{d_i-1} (v_i * x[i][p])
Plain: Maximize the total value of all pieces successfully placed on the stock sheet.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"sheet_length": float, "sheet_width": float, "num_piece_types": int, "piece_lengths": [float, ...], "piece_widths": [float, ...], "max_quantities": [int, ...], "piece_values": [float, ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "selection_status": list[list[int]] — whether each candidate rectangular piece instance is selected into the cutting pattern. example: [[1, 1, 0], [1, 0], [0, 0, 0, 0]]. Contains the 0/1 status for every piece type i and its d_i instances (its maximum demand)
- "placements": list[list[list[float]]] — bottom-left coordinates (x, y) of each selected rectangular piece instance on the stock sheet. example: [[[0.0, 0.0], [0.0, 50.0], [-1.0, -1.0]], [[100.0, 0.0], [-1.0, -1.0]], [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]]. Covers all instances in selection_status; unselected instances are filled with [-1.0, -1.0]
- "guillotine_cut_tree": dict — recursive cut tree satisfying the guillotine constraint (describes how all selected pieces are separated by horizontal or vertical cuts). example: {"type": "root", "cut_direction": "vertical", "cut_coordinate": 100.0, "left_child": {"type": "leaf", "piece_index": [0, 0]}, "right_child": {"type": "node", "cut_direction": "horizontal", "cut_coordinate": 50.0, "top": {"type": "leaf", "piece_index": [0, 1]}, "bottom": {"type": "leaf", "piece_index": [1, 0]}}}. Describes the complete cutting path until every sub-region contains exactly one selected piece instance or is empty
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: max

**Formula**
```
max Σ_{i=0}^{N-1} Σ_{p=0}^{d_i-1} (v_i * x[i][p])
```

**Plain-language statement**
Maximize the total value of all pieces successfully placed on the stock sheet.

**Rationale**
The goal is to extract the maximum economic value from the available stock material given the piece values and quantity limits.
