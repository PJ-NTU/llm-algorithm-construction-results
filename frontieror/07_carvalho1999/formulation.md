# carvalho1999 · Mathematical model

> One-dimensional bin packing problem
>
> Sense: min

---

Given N items with weights W[i] and an unlimited supply of bins each with capacity C, the goal is to assign each item to exactly one bin such that the total weight in each bin does not exceed C, while minimizing the total number of bins used. Indices for items i and bins b start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- y[b]: A binary indicator whether bin b is used.  |  y[b] ∈ {0,1}, for b ∈ {0...N-1}
- x[i][b]: A binary indicator whether item i is assigned to bin b.  |  x[i][b] ∈ {0,1}, for i ∈ {0...N-1}, b ∈ {0...N-1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each item must be packed into exactly one bin.  |  Σ_{b=0}^{n-1} x[i][b] = 1, ∀i ∈ {0...n-1}
- [hard] The total weight of the items packed into each bin must not exceed the bin's capacity, and items can be packed into a bin only if the bin is activated.  |  Σ_{i=0}^{n-1} weights[i] · x[i][b] ≤ capacity · y[b], ∀b ∈ {0...n-1}
- [hard] To break symmetry, bins with smaller indices are used first (optional).  |  y[b] ≥ y[b+1], ∀b ∈ {0...n-2}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n: n; axis: scalar
- capacity: capacity; unit=weight unit; axis: scalar
- weights: weights; unit=weight unit; axis: item index

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: minimize Σ_{b=0}^{N-1} y[b]
Plain: Minimize the total number of bins activated/used.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "capacity": int, "weights": [int, ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "bin_activation": list[int] — whether each bin is activated (1 means activated, 0 means not activated). example: [1, 1, 1, 0, 0]. List the activation status of all n potential bins
- "item_assignments": list[int] — index of the bin to which each item is assigned (0 to n-1). example: [0, 2, 0, 1, 1]. List the bin index to which each of the n items belongs
Fill every value for every entity; keep the keys EXACTLY as above.