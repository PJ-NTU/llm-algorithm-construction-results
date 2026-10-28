# pedersen2024 · Mathematical model

## Sets and parameters

Text: The nodes of the undirected graph are partitioned into three classes: substations, candidate turbine sites, and Steiner nodes.

Formula: \(G=(V,E),\quad V=S\mathbin{\dot\cup}T\mathbin{\dot\cup}H\)

Text: Each undirected edge is expanded into two auxiliary arcs, one in each direction, to express the connectivity flow.

Formula: \(A=\{(i,j),(j,i):\{i,j\}\in E\}\)

Text: An arbitrary substation is fixed as the root node of the auxiliary flow.

Formula: \(r\in S\)

Text: Each node has given spatial coordinates.

Formula: \(\boldsymbol q_v,\quad \forall v\in V\)

Text: Candidate turbine sites have an installation cost, an annual energy production, and a node landscape impact, whereas these parameters are all zero for substations and Steiner nodes.

Formula: \(a_v,p_v,\ell_v>0\ \forall v\in T;\qquad a_v=p_v=\ell_v=0\ \forall v\in S\cup H\)

Text: Each edge has a given cable cost and edge landscape impact.

Formula: \(c_e\ge 0,\ \ell_e^{E}\ge 0,\quad \forall e\in E\)

Text: Between any two distinct substations there is an edge with zero cost and zero landscape impact.

Formula: \(\{s,t\}\in E,\ c_{\{s,t\}}=\ell^{E}_{\{s,t\}}=0,\quad \forall s,t\in S,\ s\ne t\)

Text: The instance gives a minimum annual energy production quota and a cost weight.

Formula: \(Q\ge 0,\qquad 0\le \alpha\le 1\)

## Decision variables

Text: Whether node \(v\) lies in the selected tree; for a candidate turbine site, this is exactly whether a turbine is built.

Formula: \(y_v\in\{0,1\},\quad \forall v\in V\)

Text: Whether a cable is laid on undirected edge \(e\) and the edge is included in the selected tree.

Formula: \(x_e\in\{0,1\},\quad \forall e\in E\)

Text: Nonnegative connectivity flow sent from the root node on auxiliary arc \((i,j)\).

Formula: \(f_{ij}\ge 0,\quad \forall(i,j)\in A\)

## Objective

Text: Minimize the combined value of total financial cost and total landscape impact, weighted by \(\alpha\).

Formula: \[\min\ \alpha\left(\sum_{e\in E}c_e x_e+\sum_{v\in T}a_vy_v\right)+(1-\alpha)\left(\sum_{e\in E}\ell_e^{E}x_e+\sum_{v\in T}\ell_vy_v\right)\]

## Constraints

C1  Text: Every substation must lie in the selected tree.

Formula: \[y_s=1,\quad \forall s\in S\]

C2  Text: An edge can be selected only if both of its endpoints lie in the selected tree.

Formula: \[x_e\le y_v,\quad \forall e\in E,\ \forall v\in e\]

C3  Text: The total annual energy production of the built turbines must reach the minimum quota.

Formula: \[\sum_{v\in T}p_vy_v\ge Q\]

C4  Text: Each selected non-root node receives a net one unit of auxiliary flow, and each unselected non-root node receives a net zero units of auxiliary flow.

Formula: \[\sum_{i:(i,v)\in A}f_{iv}-\sum_{j:(v,j)\in A}f_{vj}=y_v,\quad \forall v\in V\setminus\{r\}\]

C5  Text: The net auxiliary flow sent out by the root node equals the number of selected non-root nodes.

Formula: \[\sum_{j:(r,j)\in A}f_{rj}-\sum_{i:(i,r)\in A}f_{ir}=\sum_{v\in V\setminus\{r\}}y_v\]

C6  Text: Auxiliary flow can pass only through selected undirected edges, and the capacity upper bound is large enough to carry the flow of all selected non-root nodes.

Formula: \[0\le f_{ij}\le (|V|-1)x_{\{i,j\}},\quad \forall(i,j)\in A\]

C7  Text: The number of selected edges equals the number of selected nodes minus one; together with the connectivity flow constraints, this ensures that the selected nodes and edges are connected, acyclic, and form a tree.

Formula: \[\sum_{e\in E}x_e=\sum_{v\in V}y_v-1\]
