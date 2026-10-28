# Vehicle routing_ period routing: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

For each customer, select one service schedule from its candidate schedules, and for each day of the planning period assign vehicle routes that start from the central depot and finally return to it. Subject to all service, capacity and cumulative-load requirements, minimise the total Euclidean travel distance of all vehicles over all days.

## Parameters and sets
| Symbol | Meaning | Unit or range |
|---|---|---|
| \(N\) | Number of customers, excluding the depot; field `num_customers` | Non-negative integer, 0 allowed |
| \(\mathcal C=\{1,\ldots,N\}\) | Set of customers | Empty when \(N=0\) |
| \(\mathcal V=\{0,\ldots,N\}\) | Set of all nodes; node 0 is the central depot | Node index |
| \(T\) | Number of days in the planning period; field `planning_period` | Days; positive integer, 1 allowed |
| \(\mathcal T=\{0,\ldots,T-1\}\) | Set of days in the planning period | Day index |
| \(K\) | Number of vehicles available each day; field `num_vehicles` | Vehicles; positive integer |
| \(\mathcal K=\{1,\ldots,K\}\) | Set of vehicles available each day | Vehicle index |
| \(S_i\) | Set of candidate schedules of customer \(i\); field `candidate_schedules`, axes customer × candidate schedule index | Each customer's set is a non-empty finite set |
| \(\mathcal S=\bigcup_{i\in\mathcal C}S_i\) | Set of schedule identifiers referenced by the customers' candidate sets | Schedule index |
| \(a_{st}\) | Whether schedule \(s\) requires service on day \(t\); field `schedule_vectors`, axes schedule × day | \(a_{st}\in\{0,1\}\); 1 means service is required, 0 means no service; every candidate schedule satisfies \(\sum_{t\in\mathcal T}a_{st}\ge1\) |
| \(d_{ij}\) | Euclidean distance from node \(i\) to node \(j\); field `distance_matrix`, axes node × node | Non-negative real; common distance unit, specific unit unspecified |
| \(q_i\) | Demand of customer \(i\) counted against that vehicle's capacity for the day on each visit; field `demands`, axis customer, excluding the depot | Strictly positive real; common capacity unit, specific physical unit unspecified |
| \(Q\) | Common capacity of all vehicles; field `vehicle_capacity` | Strictly positive real, same unit as \(q_i\) |

The distance matrix gives the Euclidean distance for every node pair, hence \(d_{ii}=0\), \(d_{ij}=d_{ji}\), and the triangle inequality required of Euclidean distances hold. No additional coordinate input is needed. Candidate schedule vectors cover all \(T\) days and cannot be all-zero vectors. The input domain above does not guarantee that a feasible solution exists; \(q_i\le Q\) is not additionally imposed as a precondition of input validity.

## Decision variables
Only the following three classes of authoritative decision variables are used; no other decision variables are added.

| Variable | Meaning | Range and indices |
|---|---|---|
| \(x_{is}\) (`x[i][s]`) | Whether customer \(i\) selects candidate schedule \(s\) | \(x_{is}\in\{0,1\},\ i\in\mathcal C,\ s\in S_i\) |
| \(y_{ijkt}\) (`y[i][j][k][t]`) | Whether vehicle \(k\) travels directly from node \(i\) to node \(j\) on day \(t\) | \(y_{ijkt}\in\{0,1\},\ i,j\in\mathcal V,\ k\in\mathcal K,\ t\in\mathcal T\) |
| \(u_{ikt}\) (`u[i][k][t]`) | Cumulative load of vehicle \(k\) after visiting customer \(i\) on day \(t\), computed as the demand of customers served since leaving the depot | Continuous; \(u_{ikt}\ge0,\ i\in\mathcal C,\ k\in\mathcal K,\ t\in\mathcal T\); same unit as \(Q\) |

At visited customers \(u\) must be the exact cumulative value. Customers not visited by that vehicle on that day still keep their \(u\) components and the given bounds, but such a component does not represent an actual service. The depot has no \(u\) variable.

## Objective function
Minimise the total Euclidean distance of the arcs actually travelled by all vehicles over the whole planning period; this is the only optimisation objective, with no secondary objective or penalty.
\[
\min\ \sum_{t\in\mathcal T}\sum_{k\in\mathcal K}
\sum_{i\in\mathcal V}\sum_{j\in\mathcal V}
 d_{ij}\,y_{ijkt}.
\]

## Constraints
All of C1-C10 below are hard constraints. C1-C7 retain all hard constraints given in the problem statement; C8-C10 implement the confirmed self-loop and exact cumulative-load requirements.

### C1 Unique selection of a candidate schedule
Hard constraint: each customer must select exactly one schedule from its candidate set.
\[
\sum_{s\in S_i}x_{is}=1,
\qquad \forall i\in\mathcal C.
\]

### C2 Exact visits according to the selected schedule
Hard constraint: on the days required by its selected schedule, a customer must be visited exactly once by exactly one vehicle, and on other days it must not be visited.
\[
\sum_{k\in\mathcal K}\sum_{\substack{j\in\mathcal V\\j\ne i}}y_{jikt}
=\sum_{s\in S_i}a_{st}x_{is},
\qquad \forall i\in\mathcal C,\ t\in\mathcal T.
\]

### C3 Per-vehicle, per-day node flow conservation
Hard constraint: for each vehicle on each day, the number of entries into each node (including the depot) must equal the number of exits.
\[
\sum_{\substack{j\in\mathcal V\\j\ne i}}y_{jikt}
-\sum_{\substack{j\in\mathcal V\\j\ne i}}y_{ijkt}=0,
\qquad \forall i\in\mathcal V,\ k\in\mathcal K,\ t\in\mathcal T.
\]

### C4 At most one depot route per vehicle per day
Hard constraint: each vehicle leaves the depot at most once per day and may stay idle; together with C3 and C6, any route used must start from and return to the depot.
\[
\sum_{j\in\mathcal C}y_{0jkt}\le1,
\qquad \forall k\in\mathcal K,\ t\in\mathcal T.
\]

### C5 Total demand capacity limit per vehicle per day
Hard constraint: the sum of demands of all customers visited by a vehicle on the same day must not exceed the capacity \(Q\).
\[
\sum_{i\in\mathcal C}q_i
\left(\sum_{\substack{j\in\mathcal V\\j\ne i}}y_{jikt}\right)\le Q,
\qquad \forall k\in\mathcal K,\ t\in\mathcal T.
\]

### C6 Cumulative-load increase and customer subtour elimination
Hard constraint: retains the given load inequality; along any used arc between customers, the cumulative load increases by at least the demand of the next customer. Since every customer demand is strictly positive, this constraint excludes customer cycles that do not pass through the depot.
\[
u_{ikt}+q_j\le u_{jkt}+Q(1-y_{ijkt}),
\qquad \forall i,j\in\mathcal C,\ i\ne j,\ k\in\mathcal K,\ t\in\mathcal T.
\]

### C7 Bounds on all customer load components
Hard constraint: for every customer, the cumulative-load component for each vehicle and each day must be no lower than that customer's demand and no higher than the capacity; unvisited components are subject to the same limits.
\[
q_i\le u_{ikt}\le Q,
\qquad \forall i\in\mathcal C,\ k\in\mathcal K,\ t\in\mathcal T.
\]

### C8 No node self-loops
Hard constraint: no vehicle on any day may travel from a node directly back to that same node, including depot self-loops; the corresponding variables are kept and must be 0.
\[
y_{iikt}=0,
\qquad \forall i\in\mathcal V,\ k\in\mathcal K,\ t\in\mathcal T.
\]

### C9 Exact cumulative load of the first customer
Hard constraint: for the first customer reached directly from the depot, the cumulative load must equal its own demand; the upper bound below, together with the lower bound of C7, fixes this starting point.
\[
u_{ikt}\le q_i+Q(1-y_{0ikt}),
\qquad \forall i\in\mathcal C,\ k\in\mathcal K,\ t\in\mathcal T.
\]

### C10 Exact load increment between consecutive customers
Hard constraint: along any used arc between customers, the increase in cumulative load must equal exactly the demand of the next customer; the upper bound below, together with C6, enforces the exact increase.
\[
u_{jkt}\le u_{ikt}+q_j+Q(1-y_{ijkt}),
\qquad \forall i,j\in\mathcal C,\ i\ne j,\ k\in\mathcal K,\ t\in\mathcal T.
\]

## Solution and boundary conventions
- A complete feasible solution gives all \(x\), \(y\), \(u\) components and satisfies their value domains and C1-C10. It determines each customer's selected schedule, each vehicle's ordered depot route or idle status on each day, and all customer load components. An optimal solution is a complete feasible solution with minimum total distance.
- Every selected schedule contains at least one service day, so no customer may be omitted, nor any service required by the selected schedule. A customer's service on one day cannot be split across several vehicles.
- Every visit counts the customer's full demand \(q_i\); demand is not apportioned by the number of visits in the selected schedule. Cumulative load is accumulated from 0 separately for each vehicle on each day; C9 fixes the value at the first customer, and C10 together with C6 fixes subsequent values.
- \(u\) at unvisited customers must not simply be set to zero, because C7 remains in force for these components. Unvisited components only need to satisfy all relevant constraints and are not required to correspond to an actual cumulative service amount.
- \(N=0\) is allowed: customers, demands and customer candidate sets are then empty, \(x\) and \(u\) have no components, all vehicles are idle every day, depot self-loops are 0, and the total distance is 0. \(K\), \(T\), \(Q\) must still satisfy the specified input domain.
- \(N=1\), \(K=1\), \(T=1\) are allowed, as is a route whose total demand equals exactly \(Q\); feasibility still requires satisfying all hard constraints.
- An empty customer candidate set, an all-zero candidate schedule, non-positive demand or non-positive capacity are invalid inputs; such input-domain violations are not accepted merely because infeasible instances are allowed.
- A valid input does not guarantee feasibility. For example, \(q_i>Q\) does not violate the fixed numeric input domain but makes C7 unsatisfiable. When no complete feasible solution exists, infeasibility must be returned explicitly; a partially covering solution must not be submitted and hard constraints must not be relaxed.
- All vehicles have the same capacity. Vehicle and capacity constraints are satisfied separately for each day; no cross-day vehicle state constraints are added.
- Complete optimal solutions with equal total distance are all acceptable; vehicle count used, schedule indices or route lexicographic order are not used as secondary optimisation criteria.

Number of numbered constraints: `n_constraints = 10`.
