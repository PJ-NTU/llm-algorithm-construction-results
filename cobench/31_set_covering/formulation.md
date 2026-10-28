# Set covering: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
Given m elements to be covered and n candidate sets, each set j has a specific cost costs[j] and covers a subset of the elements. The goal is to select a collection of sets such that all m elements are covered by at least one selected set and the total cost of the selected sets is minimized. Indices start at 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[j]: whether candidate set j is selected  |  x[j] ∈ {0,1}, ∀j ∈ {0..n-1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each element must be covered by at least one selected set  |  Σ_{j ∈ covers[i]} x[j] ≥ 1, ∀i ∈ {0..m-1}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n: n; axis: total number of sets
- m: m; axis: total number of elements
- costs: costs; axis: set index j
- covers: covers; axis: element index i

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: minimize Σ_{j=0}^{n-1} costs[j] * x[j]
Plain: Minimize the total cost of all selected sets
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "m": int, "costs": [int, ...], "covers": [[int, ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "selection_status": list[int] — Whether each candidate set is selected (1 = selected, 0 = not selected). example: [1, 0, 1, 1, 0]. List the selection status of all n candidate sets, including unselected sets (value 0)
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
minimize Σ_{j=0}^{n-1} costs[j] * x[j]
```

**Plain-language statement**
Minimize the total cost of all selected sets

**Rationale**
The goal of the problem is to find the most economical solution while satisfying the full-coverage constraint, i.e., to minimize the total cost.
