# nagy2015 · Mathematical model

## Sets and parameters

Text: The set of customers is $N$, the depot is indexed $0$, and all nodes form the set $V$.

Formula: $N=\{1,\ldots,n\},\quad V=\{0\}\cup N$

Text: The allowed directed arcs connect any two distinct nodes.

Formula: $A=\{(i,j):i,j\in V,\ i\ne j\}$

Text: An unlimited fleet is represented by at most $2n$ candidate vehicles, because there are at most $2n$ unsplittable delivery or pickup requests and each request can be served on its own route; if a single request exceeds the capacity, the model deems the instance infeasible as the problem specifies.

Formula: $K=\{1,\ldots,2n\}$

Text: Parameter $C$ is the capacity shared by all vehicles, and $q_i^D$ and $q_i^P$ are the delivery demand and pickup demand of customer $i$, respectively.

Formula: $C>0,\quad q_i^D\ge0,\quad q_i^P\ge0,\qquad \forall i\in N$

Text: Parameter $d_{ij}$ is the inter-node distance given in the input, which is the symmetric Euclidean distance rounded to the nearest integer.

Formula: $d_{ij}=d_{ji}\in\mathbb Z_{\ge0},\quad d_{ii}=0,\qquad \forall i,j\in V$

## Decision variables

Text: Binary variable $x_{ijk}$ indicates whether candidate vehicle $k$ travels directly from node $i$ to node $j$.

Formula: $x_{ijk}\in\{0,1\},\qquad \forall (i,j)\in A,\ k\in K$

Text: Binary variable $u_k$ indicates whether candidate vehicle $k$ is used and performs a depot-to-depot route.

Formula: $u_k\in\{0,1\},\qquad \forall k\in K$

Text: Binary variable $z_{ik}$ indicates whether vehicle $k$ visits customer $i$.

Formula: $z_{ik}\in\{0,1\},\qquad \forall i\in N,\ k\in K$

Text: Binary variables $y^D_{ik}$ and $y^P_{ik}$ indicate whether the entire delivery request and the entire pickup request of customer $i$, respectively, are served by vehicle $k$.

Formula: $y^D_{ik},y^P_{ik}\in\{0,1\},\qquad \forall i\in N,\ k\in K$

Text: Continuous variables $f^D_{ijk}$ and $f^P_{ijk}$ denote, respectively, the quantity of not-yet-delivered goods and the quantity of already-picked-up goods carried by vehicle $k$ on arc $(i,j)$.

Formula: $f^D_{ijk},f^P_{ijk}\in\mathbb R_{\ge0},\qquad \forall(i,j)\in A,\ k\in K$

Text: Continuous variable $s_{ik}$ is the visit-order variable of customer $i$ on the route of vehicle $k$.

Formula: $s_{ik}\in[0,n],\qquad \forall i\in N,\ k\in K$

## Objective

Text: Minimize the total distance traveled by all used vehicles over all route arcs.

Formula: $\displaystyle \min \sum_{k\in K}\sum_{(i,j)\in A}d_{ij}x_{ijk}$

## Constraints

C1  Text: The entire delivery request of each customer must be assigned to exactly one vehicle, so deliveries cannot be split or duplicated.

Formula: $\displaystyle \sum_{k\in K}y^D_{ik}=1,\qquad \forall i\in N$

C2  Text: The entire pickup request of each customer must be assigned to exactly one vehicle, so pickups cannot be split or duplicated.

Formula: $\displaystyle \sum_{k\in K}y^P_{ik}=1,\qquad \forall i\in N$

C3  Text: A vehicle can serve a customer's delivery request only if it visits that customer.

Formula: $y^D_{ik}\le z_{ik},\qquad \forall i\in N,\ k\in K$

C4  Text: A vehicle can serve a customer's pickup request only if it visits that customer.

Formula: $y^P_{ik}\le z_{ik},\qquad \forall i\in N,\ k\in K$

C5  Text: A customer visit must serve the customer's delivery request, pickup request, or both; hence, when the same vehicle handles both requests, it visits the customer only once.

Formula: $z_{ik}\le y^D_{ik}+y^P_{ik},\qquad \forall i\in N,\ k\in K$

C6  Text: If a vehicle visits a customer, it must leave that customer exactly once; otherwise it must not leave that customer.

Formula: $\displaystyle \sum_{j:(i,j)\in A}x_{ijk}=z_{ik},\qquad \forall i\in N,\ k\in K$

C7  Text: If a vehicle visits a customer, it must enter that customer exactly once; otherwise it must not enter that customer.

Formula: $\displaystyle \sum_{j:(j,i)\in A}x_{jik}=z_{ik},\qquad \forall i\in N,\ k\in K$

C8  Text: Each used vehicle must depart from the depot exactly once, and an unused vehicle must not depart.

Formula: $\displaystyle \sum_{j\in N}x_{0jk}=u_k,\qquad \forall k\in K$

C9  Text: Each used vehicle must return to the depot exactly once, and an unused vehicle must not return.

Formula: $\displaystyle \sum_{i\in N}x_{i0k}=u_k,\qquad \forall k\in K$

C10  Text: The visit-order variable takes a positive value only for customers visited by the vehicle.

Formula: $z_{ik}\le s_{ik}\le n z_{ik},\qquad \forall i\in N,\ k\in K$

C11  Text: Every used customer-to-customer arc must strictly increase the visit order, thereby eliminating subtours that do not pass through the depot.

Formula: $s_{jk}\ge s_{ik}+1-(n+1)(1-x_{ijk}),\qquad \forall k\in K,\ i,j\in N,\ i\ne j$

C12  Text: The delivery quantity unloaded by a vehicle at a customer must equal that customer's entire delivery demand handled by the vehicle.

Formula: $\displaystyle \sum_{j:(j,i)\in A}f^D_{jik}-\sum_{j:(i,j)\in A}f^D_{ijk}=q_i^D y^D_{ik},\qquad \forall i\in N,\ k\in K$

C13  Text: The pickup quantity added by a vehicle at a customer must equal that customer's entire pickup demand handled by the vehicle.

Formula: $\displaystyle \sum_{j:(i,j)\in A}f^P_{ijk}-\sum_{j:(j,i)\in A}f^P_{jik}=q_i^P y^P_{ik},\qquad \forall i\in N,\ k\in K$

C14  Text: A vehicle must not carry any picked-up goods when leaving the depot.

Formula: $f^P_{0jk}=0,\qquad \forall j\in N,\ k\in K$

C15  Text: A vehicle must not still carry any undelivered goods when returning to the depot.

Formula: $f^D_{i0k}=0,\qquad \forall i\in N,\ k\in K$

C16  Text: On each traveled arc, the sum of undelivered goods and picked-up goods carried by a vehicle must not exceed the common capacity, and unused arcs must carry no goods.

Formula: $f^D_{ijk}+f^P_{ijk}\le Cx_{ijk},\qquad \forall(i,j)\in A,\ k\in K$
