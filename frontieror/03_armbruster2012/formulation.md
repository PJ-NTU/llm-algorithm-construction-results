# armbruster2012 · Mathematical model

## Sets and parameters

Text: The undirected graph consists of a node set \(V\) and an edge set \(E\) whose edges are distinguished by input record; the two endpoints of each edge \(e\) are \(u_e,v_e\).

Formula: \(G=(V,E),\quad n=|V|,\quad m=|E|,\quad e=\{u_e,v_e\}\in E\)

Text: The nonnegative integer weight of node \(i\) is \(w_i\), and the real-valued cost of edge \(e\) is \(c_e\).

Formula: \(w_i\in\mathbb Z_{\ge 0}\ (\forall i\in V),\quad c_e\in\mathbb R\ (\forall e\in E)\)

Text: The total node weight is \(W\), and the bisection capacity \(B\) lies within the range specified by the problem.

Formula: \(W=\sum_{i\in V}w_i,\quad B\in\mathbb Z,\quad \lceil W/2\rceil\le B\le W\)

## Decision variables

Text: The binary variable \(x_i\) indicates the group of node \(i\): value 1 means it belongs to group 1, and value 0 means it belongs to group 2.

Formula: \(x_i\in\{0,1\},\quad \forall i\in V\)

Text: The binary variable \(y_e\) indicates whether edge \(e\) is cut; it equals 1 if and only if its two endpoints belong to different groups.

Formula: \(y_e\in\{0,1\},\quad \forall e\in E\)

## Objective

Text: Minimize the total cost of all cut edges.

Formula: \(\displaystyle \min\ \sum_{e\in E}c_e y_e\)

## Constraints

C1  Text: The total weight of all nodes in group 1 must not exceed the bisection capacity.

Formula: \(\displaystyle \sum_{i\in V}w_i x_i\le B\)

C2  Text: The total weight of all nodes in group 2 must not exceed the bisection capacity.

Formula: \(\displaystyle \sum_{i\in V}w_i(1-x_i)\le B\)

C3  Text: Group 1 must contain at least one node.

Formula: \(\displaystyle \sum_{i\in V}x_i\ge 1\)

C4  Text: Group 2 must contain at least one node.

Formula: \(\displaystyle \sum_{i\in V}(1-x_i)\ge 1\)

C5  Text: If endpoint \(u_e\) of edge \(e\) belongs to group 1 and endpoint \(v_e\) belongs to group 2, the edge must be marked as cut.

Formula: \(y_e\ge x_{u_e}-x_{v_e},\quad \forall e\in E\)

C6  Text: If endpoint \(v_e\) of edge \(e\) belongs to group 1 and endpoint \(u_e\) belongs to group 2, the edge must be marked as cut.

Formula: \(y_e\ge x_{v_e}-x_{u_e},\quad \forall e\in E\)

C7  Text: If both endpoints of edge \(e\) belong to group 2, the edge must not be marked as cut.

Formula: \(y_e\le x_{u_e}+x_{v_e},\quad \forall e\in E\)

C8  Text: If both endpoints of edge \(e\) belong to group 1, the edge must not be marked as cut.

Formula: \(y_e\le 2-x_{u_e}-x_{v_e},\quad \forall e\in E\)
