# Aircraft landing: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
The problem involves scheduling the landing times and runway assignments for a set of planes P. Each plane i must land within its time window [E_i, L_i] and is ideally scheduled at its target time T_i. If a plane lands before T_i, an earliness penalty α_i per unit time is incurred; if it lands after T_i, a lateness penalty β_i per unit time is incurred. For every pair of planes (i, j), a minimum separation time S_{i,j,r,k} must be maintained if plane i lands before plane j, where r and k are the runways assigned to i and j respectively. All indices start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- t[i]: The scheduled landing time for plane i  |  t[i] ∈ [E_i, L_i], ∀i ∈ P
- y[i][r]: Binary variable indicating if plane i is assigned to runway r  |  y[i][r] ∈ {0,1}, ∀i ∈ P, ∀r ∈ R
- z[i][j]: Binary variable indicating if plane i lands before plane j  |  z[i][j] ∈ {0,1}, ∀i, j ∈ P, i ≠ j
- e[i]: The earliness of plane i (time units before target)  |  e[i] ≥ 0, ∀i ∈ P
- l[i]: The lateness of plane i (time units after target)  |  l[i] ≥ 0, ∀i ∈ P

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each flight must be assigned to exactly one runway.  |  ∑_{r ∈ R} y[i][r] = 1, ∀i ∈ P
- [hard] Each flight's actual landing time must lie within its earliest/latest allowed landing time window.  |  E_i ≤ t[i] ≤ L_i, ∀i ∈ P
- [hard] Compute each flight's earliness.  |  e[i] ≥ T_i - t[i], ∀i ∈ P
- [hard] Compute each flight's lateness.  |  l[i] ≥ t[i] - T_i, ∀i ∈ P
- [hard] Any two flights i and j must have a precedence order (i before j or j before i).  |  z[i][j] + z[j][i] = 1, ∀i, j ∈ P, i < j
- [hard] If flights i and j are assigned to the same runway r and i lands before j, the minimum separation required by that runway must hold between them.  |  t[j] ≥ t[i] + S_{i,j,r,r} - M * (2 - y[i][r] - y[j][r] + 1 - z[i][j]), ∀i, j ∈ P, i ≠ j, ∀r ∈ R
- [hard] If flights i and j are assigned to different runways r and k and i lands before j, the cross-runway minimum separation must hold between them.  |  t[j] ≥ t[i] + S_{i,j,r,k} - M * (2 - y[i][r] - y[j][k] + 1 - z[i][j]), ∀i, j ∈ P, i ≠ j, ∀r, k ∈ R, r ≠ k

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- planes: P; axis: set of flights
- runways: R; axis: set of runways
- earliest_landing_time: E_i; unit=s; axis: flight
- target_landing_time: T_i; unit=s; axis: flight
- latest_landing_time: L_i; unit=s; axis: flight
- earliness_penalty_cost: α_i; unit=cost/s; axis: flight
- lateness_penalty_cost: β_i; unit=cost/s; axis: flight
- separation_matrix: S_{i,j,r,k}; unit=s; axis: flight i × flight j × runway r × runway k
- big_m: M

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min ∑_{i ∈ P} (α_i * e[i] + β_i * l[i])
Plain: Minimize the total weighted penalty cost associated with planes landing earlier or later than their preferred target times.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"planes": [int, ...], "runways": [int, ...], "earliest_landing_time": [float, ...], "target_landing_time": [float, ...], "latest_landing_time": [float, ...], "earliness_penalty_cost": [float, ...], "lateness_penalty_cost": [float, ...], "separation_matrix": [[[[float, ...], ...], ...], ...], "big_m": float}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "runway_assignment": dict[str, int] — landing runway assigned to each flight. example: {"UA123": 1, "CA456": 2, "DL789": 1}. List all P flights with their corresponding runway numbers
- "landing_times": dict[str, float] — actual landing time determined for each flight. example: {"UA123": 540.5, "CA456": 552.0, "DL789": 565.0}. List the landing times of all P flights
- "landing_sequence": list[str] — landing order of the flights. example: ["UA123", "CA456", "DL789"]. List of all flight IDs sorted by landing time from earliest to latest
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min ∑_{i ∈ P} (α_i * e[i] + β_i * l[i])
```

**Plain-language statement**
Minimize the total weighted penalty cost associated with planes landing earlier or later than their preferred target times.

**Rationale**
The goal is to achieve a schedule as close to the target times as possible while respecting safety and operational constraints, with costs representing the fuel or passenger dissatisfaction impact of deviations.
