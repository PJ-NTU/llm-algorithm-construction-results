# kobeaga2024 · Mathematical model

## Data and notation

Text: Let \(V\) be the set of locations and \(r\in V\) the designated depot location.

Formula: \(V=\{r\}\cup N\), where \(N=V\setminus\{r\}\) is the set of non-depot locations.

Text: There is an undirected direct link between any two distinct locations.

Formula: \(E=\bigl\{\{i,j\}:i,j\in V,\ i\neq j\bigr\}\).

Text: Each link \(e\in E\) has a known positive travel distance.

Formula: \(d_e>0,\quad \forall e\in E\).

Text: Each location \(i\in V\) has a known positive score.

Formula: \(p_i>0,\quad \forall i\in V\).

Text: \(B\) denotes the given total travel distance budget.

Formula: \(B\in\mathbb{R}\).

Text: For any subset of locations \(S\subseteq V\), \(\delta(S)\) denotes the set of links with exactly one endpoint in \(S\), and we write \(\delta(i)=\delta(\{i\})\) for short.

Formula: \(\delta(S)=\bigl\{\{i,j\}\in E:i\in S,\ j\in V\setminus S\bigr\}\).

## Decision variables

Text: Binary variable \(y_i\) indicates whether location \(i\) is visited by the route.

Formula: \(y_i\in\{0,1\},\quad \forall i\in V\).

Text: Binary variable \(x_e\) indicates whether undirected link \(e\) belongs to the route.

Formula: \(x_e\in\{0,1\},\quad \forall e\in E\).

## Objective

Text: Maximize the total score of all visited locations, including the score of the depot location, which is always visited.

Formula: \(\max\displaystyle\sum_{i\in V}p_i y_i\).

## Constraints

C1  Text: The depot location must be visited.

Formula: \(y_r=1\).

C2  Text: Each visited location is incident to exactly two selected links, and an unvisited location must not be incident to any selected link.

Formula: \(\displaystyle\sum_{e\in\delta(i)}x_e=2y_i,\quad \forall i\in V\).

C3  Text: Every subset of locations that does not contain the depot but contains some visited location must be connected to the outside of the subset by at least two selected links, which excludes subtours separated from the depot and ensures that the selected route forms a single connected cycle.

Formula: \(\displaystyle\sum_{e\in\delta(S)}x_e\ge 2y_i,\quad \forall\,\emptyset\neq S\subseteq V\setminus\{r\},\ \forall i\in S\).

C4  Text: The total travel distance of all selected links must not exceed the given distance budget.

Formula: \(\displaystyle\sum_{e\in E}d_e x_e\le B\).