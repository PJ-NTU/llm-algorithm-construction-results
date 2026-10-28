# Corporate structuring: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Given N countries with specific tax codes, domestic tax rates, foreign income tax rates, and profits, the goal is to construct a tree-structured corporate hierarchy rooted at a designated target country. Each country with positive profit must be included exactly once. The hierarchy determines the flow of dividends from children to parents, where each node pays domestic tax on its own profit and additional taxes (withholding and foreign income tax) on dividends received from its children based on its specific tax code (Exemption, Deduction, Source-by-source Pooling, or World-wide Pooling). The objective is to maximize the total net income reaching the root country. Indices for countries are from 1 to N, with 0 representing the external parent of the root.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[i][j]: Binary indicator if country i is a child of country j in the hierarchy  |  x[i][j] ∈ {0,1}, i ∈ {1..N}, j ∈ {0..N}
- F[i]: The total net dividend amount sent from country i to its parent  |  F[i] ≥ 0, i ∈ {1..N}
- T_foreign[i]: The extra foreign income tax paid by country i on dividends received from its children  |  T_foreign[i] ≥ 0, i ∈ {1..N}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Every country with profit greater than 0 must have exactly one parent node in the hierarchy.  |  \sum_{j=0, j 
eq i}^{N} x[i][j] = 1, \quad \forall i \in \{1..N\} \text{ s.t. } P_i > 0
- [hard] The parent of the target country (root node) must be the virtual node 0.  |  x[target][0] = 1
- [hard] The structure must be acyclic (a tree structure).  |  u_i - u_j + N \cdot x[i][j] \le N-1, \quad \forall i, j \in \{1..N\}, j 
eq target
- [hard] The amount F[i] that country i sends to its parent equals its own after-tax profit plus the after-tax dividends received from its children, minus the additional foreign income tax.  |  F[i] = P_i(1-D_i) + \sum_{k=1, k 
eq i}^{N} (x[k][i] \cdot F[k] \cdot (1-W_{ki})) - T\_foreign[i], \quad \forall i \in \{1..N\}
- [hard] Additional foreign income tax by tax code: code 1 (exemption) has no additional tax.  |  C_i = 1 \implies T\_foreign[i] = 0
- [hard] Additional foreign income tax by tax code: code 2 (deduction) is computed on the total remitted by the children.  |  C_i = 2 \implies T\_foreign[i] = F\_rate_i \cdot \sum_{k=1}^{N} (x[k][i] \cdot F[k] \cdot (1-W_{ki}))
- [hard] Additional foreign income tax by tax code: code 3 (per-country credit) computes the difference separately for each child branch.  |  C_i = 3 \implies T\_foreign[i] = \sum_{k=1}^{N} x[k][i] \cdot \max(0, F[k](1-W_{ki}) - (1-F\_rate_i) \cdot \sum_{s \in Subtree(k)} P_s)
- [hard] Additional foreign income tax by tax code: code 4 (worldwide credit) computes the difference aggregated over all child branches.  |  C_i = 4 \implies T\_foreign[i] = \max(0, \sum_{k=1}^{N} (x[k][i] \cdot F[k] \cdot (1-W_{ki})) - (1-F\_rate_i) \cdot \sum_{s \in Subtree(i), s 
eq i} P_s)

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- N: N
- target: target
- countries: P_i; unit=currency; axis: country_id
- withholding: W_{ki}; axis: from_country × to_country

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: maximize
Formula: max F[target]
Plain: Maximize the total net income (after all domestic, withholding, and foreign taxes) that reaches the designated root country.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"N": int, "target": int, "countries": {"1": [float, ...], "2": [float, ...], "3": [float, ...], "4": [float, ...]}, "withholding": {"1": {"1": float, "2": float, "3": float, "4": float}, "2": {"1": float, "2": float, "3": float, "4": float}, "3": {"1": float, "2": float, "3": float, "4": float}, "4": {"1": float, "2": float, "3": float, "4": float}}}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "structure": dict[int, int] — the direct parent node of each country (0 means the virtual root, i.e. the country is the top-level parent company). example: {"1": 0, "2": 1, "3": 1, "4": 2}. List all N countries with profit greater than 0; the target country must point to 0
- "net_dividends": dict[int, float] — the net dividend amount each country sends to its parent. example: {"1": 1500.5, "2": 800.0, "3": 450.2, "4": 200.0}. List all N countries with profit greater than 0
- "foreign_tax_paid": dict[int, float] — the additional foreign income tax each country pays on dividends received from its children. example: {"1": 45.0, "2": 0.0, "3": 12.5, "4": 0.0}. List all N countries with profit greater than 0
Fill every value for every entity; keep the keys EXACTLY as above.
