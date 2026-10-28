# Equitable partitioning problem: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
Partition a set of N individuals, each with M binary attributes, into exactly K groups to ensure the distribution of '1's for each attribute is balanced across all groups. The imbalance is measured by the sum of mean absolute deviations of attribute counts from their respective averages. All indices start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[i][j]: Binary indicator whether individual i is assigned to group j  |  x[i][j] ∈ {0, 1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each individual must be assigned to exactly one group.  |  Σ_{j=0}^{K-1} x[i][j] = 1, ∀i ∈ {0, ..., N-1}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- num_individuals: N; unit=person; axis: scalar
- num_attributes: M; unit=attribute; axis: scalar
- num_groups: K; unit=group; axis: scalar
- attribute_matrix: A[i][m]; unit=binary flag; axis: individual × attribute

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min Σ_{m=0}^{M-1} ( (1/K) * Σ_{j=0}^{K-1} | Σ_{i=0}^{N-1} (A[i][m] * x[i][j]) - (1/K) * Σ_{i=0}^{N-1} A[i][m] | )
Plain: Minimize the sum over all attributes of the average absolute difference between the count of individuals with that attribute in a group and the expected mean count for that attribute across all groups.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"num_individuals": int, "num_attributes": int, "num_groups": int, "attribute_matrix": [[int, ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "group_assignment": dict[str, int] — Group number (0-7) that each individual belongs to. example: {"person_0": 2, "person_1": 0, "person_2": 7, "person_3": 2, "person_4": 5}. Must include the groups of all N individuals; keys are individual identifiers, values are the assigned group indices
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min Σ_{m=0}^{M-1} ( (1/K) * Σ_{j=0}^{K-1} | Σ_{i=0}^{N-1} (A[i][m] * x[i][j]) - (1/K) * Σ_{i=0}^{N-1} A[i][m] | )
```

**Plain-language statement**
Minimize the sum over all attributes of the average absolute difference between the count of individuals with that attribute in a group and the expected mean count for that attribute across all groups.

**Rationale**
The goal is to achieve a balanced distribution across groups for every attribute simultaneously, using the mean absolute deviation as the penalty for imbalance.
