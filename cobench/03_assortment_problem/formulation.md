# Assortment problem: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
The problem involves selecting at most two types of stock rectangles from a set of available types and packing a required quantity of rectangular pieces into instances of these stocks. Each piece type has a minimum and maximum demand. Pieces can be rotated by 90 degrees and must not overlap or exceed stock boundaries. The goal is to minimize the total waste area percentage across all used stock instances. All indices for stock types, piece types, and instances start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- y[s]: Binary variable indicating if stock type s is selected for use.  |  y[s] ∈ {0, 1}
- n[s]: Integer variable representing the number of instances of stock type s used.  |  n[s] ≥ 0, integer
- x[i][k][s]: Binary variable indicating if piece i is placed in the k-th instance of stock type s.  |  x[i][k][s] ∈ {0, 1}
- pos_x[i][k][s], pos_y[i][k][s]: Continuous variables representing the bottom-left coordinates of piece i in instance k of stock s.  |  pos_x, pos_y ≥ 0
- r[i][k][s]: Binary variable indicating if piece i is rotated 90 degrees in instance k of stock s.  |  r[i][k][s] ∈ {0, 1}
- q[i]: Integer variable representing the total number of pieces of type i placed across all stocks.  |  min_i ≤ q[i] ≤ max_i

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] At most two distinct stock types can be selected.  |  sum_{s=1}^{S} y[s] \le 2
- [hard] A stock instance can only be used if its stock type is selected.  |  z_{s,k} \le y[s], \forall s, k
- [hard] The number of instances used for a stock type must match the active instance indicators.  |  n[s] = sum_{k=1}^{K_{max}} z_{s,k}, \forall s
- [hard] The total number of pieces of type i placed must be within its min and max limits.  |  q[i] = sum_{s, k} x[i][k][s], \text{ and } MinPieces_i \le q[i] \le MaxPieces_i, \forall i
- [hard] Pieces must be placed within the boundaries of the stock instance, accounting for rotation.  |  pos_x[i][k][s] + ( (1-r[i][k][s]) \cdot L_i + r[i][k][s] \cdot W_i ) \le L_s + M(1 - x[i][k][s]), \forall i,k,s; \\ pos_y[i][k][s] + ( (1-r[i][k][s]) \cdot W_i + r[i][k][s] \cdot L_i ) \le W_s + M(1 - x[i][k][s]), \forall i,k,s
- [hard] Pieces within the same stock instance must not overlap.  |  \text{For any two pieces } i, j \text{ in instance } (s,k): \\ (pos_x[i] + w'_i \le pos_x[j]) \lor (pos_x[j] + w'_j \le pos_x[i]) \lor (pos_y[i] + h'_i \le pos_y[j]) \lor (pos_y[j] + h'_j \le pos_y[i])
- [hard] A piece can only be placed in an instance that is actually used.  |  x[i][k][s] \le z_{s,k}, \forall i, k, s

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- area: Area_i; axis: m
- length: L_i; axis: m
- width: W_i; axis: m
- min: MinPieces_i; axis: m
- max: MaxPieces_i; axis: m
- stock_instance_active: z_{s,k}; axis: S×K_max

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min ( (sum_{s, k} Area_s * z_{s,k} - sum_{i} q[i] * Area_i) / (sum_{s, k} Area_s * z_{s,k}) )
Plain: The ratio of the total unused (waste) area to the total area of all stock instances used.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"m": int, "S": int, "length": [float, ...], "width": [float, ...], "area": [float, ...], "piece_length": [float, ...], "piece_width": [float, ...], "piece_area": [float, ...], "min": [int, ...], "max": [int, ...], "stocks": [{"length": float, "width": float, "fixed_cost": float}, ...], "pieces": [{"length": float, "width": float, "min": int, "max": int, "value": float}, ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "selected_stock_types": list[int] — whether each stock sheet type is selected (at most two). example: [1, 0, 1]. List all stock sheet types; 1 means selected, 0 means not selected
- "stock_instances": list[dict] — details of the pieces placed in each stock sheet instance (including position, rotation and piece type). example: [{"stock_type_id": 1, "placements": [{"piece_type_id": 1, "x": 0.0, "y": 0.0, "is_rotated": 0}, {"piece_type_id": 2, "x": 10.5, "y": 0.0, "is_rotated": 1}]}, {"stock_type_id": 3, "placements": []}]. Includes all opened stock sheet instances and the geometric layout of the pieces inside them
- "piece_counts": dict[str, int] — total number of placed pieces for each piece type. example: {"piece_1": 5, "piece_2": 10, "piece_3": 0}. Lists all piece types with their total placed counts; must satisfy the min/max constraints
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min ( (sum_{s, k} Area_s * z_{s,k} - sum_{i} q[i] * Area_i) / (sum_{s, k} Area_s * z_{s,k}) )
```

**Plain-language statement**
The ratio of the total unused (waste) area to the total area of all stock instances used.

**Rationale**
The objective is to minimize the waste area percentage, which is equivalent to maximizing the area utilization of the selected stocks.
