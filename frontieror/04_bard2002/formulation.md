# bard2002 · Mathematical model

> Vehicle routing with time windows · minimizing the number of vehicles (Bard et al. 2002)
>
> Sense: min

---

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Decide the grouping and visiting order of customers, and the depot departure, customer departure and depot return times of each route. Minimise the number of vehicles dispatched from the depot, subject to every customer being visited exactly once and all vehicle capacity and time constraints being satisfied.
## Parameters and sets
| Symbol | Meaning | Unit or range |
|---|---|---|
| \(N=\{1,\ldots,n\}\) | Set of customers | \(n\) is a non-negative integer; the empty set and a single customer are allowed |
| \(0\), \(V=N\cup\{0\}\) | Depot index and set of all locations | Single depot |
| \(E=\{(i,j)\in V\times V:i\ne j\}\) | Ordered location pairs without self-loops | Includes both directions between the depot and customers |
| \(E_C=\{(i,j)\in N\times N:i\ne j\}\) | Ordered customer pairs without self-loops | \(E_C\subseteq E\) |
| \(p_0\) | Known depot coordinates | Coordinate dimension and distance unit name unspecified |
| \(\ell_{ij}\) | Euclidean distance between locations \(i,j\) before truncation | Non-negative; distance unit name unspecified |
| \(\tau_{ij}\) | Known base travel time, i.e. the Euclidean distance truncated to one decimal place | Non-negative, an integer multiple of \(0.1\); speed is 1, so distance and time values are interchangeable |
| \(q_i\) | Pickup demand of customer \(i\) | Positive real; quantity unit name unspecified |
| \(Q\) | Capacity, identical for all vehicles | Positive real; same quantity unit as demand |
| \(s_i\) | Service time of customer \(i\) | Non-negative real; same time unit as travel time, no time grid imposed |
| \([a_i,b_i]\), \(i\in N\) | Time window in which the customer may be departed, i.e. earliest ready time and latest due time | \(0\le a_i\le b_i\); real, no time grid imposed |
| \([a_0,b_0]\) | Common depot scheduling horizon; vehicles choose a departure time within it and return to the depot no later than its end | Real, \(a_0\le b_0\); shares the time axis and unit with customers, no other range specified for the endpoints |
| \(q_0=s_0=0\) | Demand and service time of the depot | Zero quantity and zero time, respectively |
| \(L_0=0\) | Initial load of each vehicle when leaving the depot | Zero quantity; not the load when returning to the depot |
| \(c_{ij}\) | Effective travel time after the processing stated in the problem | Non-negative real; defined below, possibly asymmetric |
| \(F\) | Set of arcs excluded in advance because the customer pair is incompatible in time windows or capacity | \(F\subseteq E_C\), defined below |

The base travel time of every location pair is known and follows the data definition below; truncation is not rounding:
\[
\tau_{ij}=\frac{\lfloor 10\ell_{ij}\rfloor}{10},\qquad (i,j)\in E.
\]
Euclidean distances come from the location positions stated in the problem; the model uses the known base travel times, does not additionally require customer coordinates as a separate input field, and does not assume a coordinate dimension.

Service time is added only to the outgoing travel time of the departing customer. Using the depot horizon start \(a_0\) as the ready time when the destination is the depot, the effective travel time is defined as:
\[
c_{ij}=\max\{\tau_{ij}+s_i,\ a_j-b_i\},
\qquad i\in N,\ j\in V\setminus\{i\},
\]
\[
c_{0j}=\tau_{0j},\qquad j\in N.
\]
The latter formula does not extend to the depot the time-window lifting rule that applies only when "the origin is a customer". The effective travel time is not truncated again once formed. The problem guarantees that these data satisfy the triangle inequality:
\[
c_{ij}\le c_{ih}+c_{hj},\qquad i,h,j\in V\text{ pairwise distinct}.
\]
This is a structural guarantee of valid input, not a decision constraint to be satisfied by adjusting travel times.

The pre-exclusion condition for customer pairs is:
\[
F=\{(i,j)\in E_C:
 a_i+c_{ij}>b_j\ \text{or}\ q_i+q_j>Q\}.
\]
These conditions mean, respectively: even when leaving customer \(i\) at its earliest allowed time, customer \(j\) cannot be departed in time; or the total quantity of the two customers already exceeds capacity. Equality cases are not excluded by this condition. Arcs that are not excluded must still satisfy all constraints of a complete route.

To switch off the time recursion on unselected arcs, define the following constants derived from the known time windows; no new business parameters are introduced:
\[
M^{C}_{ij}=\max\{0,b_i+c_{ij}-a_j\},\qquad (i,j)\in E_C,
\]
\[
M^{D}_i=\max\{0,b_0+c_{0i}-a_i\},\qquad i\in N,
\]
\[
M^{R}_i=\max\{0,b_i+c_{i0}-a_0\},\qquad i\in N.
\]
## Decision variables
| Symbol | Meaning | Domain |
|---|---|---|
| \(x_{ij}\) | Whether a vehicle travels directly from location \(i\) to location \(j\) | \(\{0,1\}\), \((i,j)\in E\) |
| \(t_i\) | Time at which the vehicle leaves customer \(i\) | Real, \(i\in N\); time window imposed by C4 |
| \(L_i\) | Cumulative load label when leaving customer \(i\), recorded by the lower-bound recursion of the problem | Non-negative real, \(i\in N\); restricted by C10-C12 |
| \(d_i\) | If \(i\) is the first customer, the time the route leaves the depot | Real, \(i\in N\); for non-first customers the value is only an auxiliary value with no business meaning |
| \(r_i\) | If \(i\) is the last customer, the time the route returns to the depot | Real, \(i\in N\); for non-last customers the value is only an auxiliary value with no business meaning |

Each route has its own depot departure and return times; a single common depot time variable is not used for all vehicles. \(L_0\) is a fixed parameter, not a customer load variable.
## Objective
Minimise only the number of vehicles actually dispatched from the depot, i.e. the number of routes; no distance, time or other secondary objective is used.
\[
\min\ K=\sum_{i\in N}x_{0i}.
\]
No separate upper limit on the number of available vehicles is imposed.
## Constraints
The model has 12 numbered constraints, \(\texttt{n\_constraints}=12\), all of them hard constraints.

### C1 Each customer has exactly one predecessor
Each customer must be entered by exactly one selected arc; customers may not be missed or entered more than once.
\[
\sum_{j\in V\setminus\{i\}}x_{ji}=1,
\qquad i\in N.
\]
### C2 Each customer has exactly one successor
Each customer must be left by exactly one selected arc; together with C1 this gives each customer a unique visiting order.
\[
\sum_{j\in V\setminus\{i\}}x_{ij}=1,
\qquad i\in N.
\]
### C3 Pre-excluded customer arcs are forbidden
Customer-to-customer arcs excluded because of time-window or capacity incompatibility may never be selected.
\[
x_{ij}=0,\qquad (i,j)\in F.
\]
### C4 Customer departure times satisfy the time windows
The time of leaving each customer must be no earlier than its ready time and no later than its due time; equality is allowed at both ends.
\[
a_i\le t_i\le b_i,\qquad i\in N.
\]
### C5 Depot departure times lie within the scheduling horizon
Each used route may choose its depot departure time independently, but that time must lie within the depot horizon; auxiliary departure times of non-first customers are also placed in the same non-empty interval and do not affect route feasibility.
\[
a_0\le d_i\le b_0,\qquad i\in N.
\]
### C6 Effective travel time recursion between customers
When the arc from customer \(i\) to customer \(j\) is selected, the departure time of the latter must be no earlier than the departure time of the former plus the effective travel time; no precedence is imposed on unselected arcs.
\[
t_j\ge t_i+c_{ij}-M^{C}_{ij}(1-x_{ij}),
\qquad (i,j)\in E_C.
\]
### C7 Travel time from the depot to the first customer
When customer \(i\) is the first customer of a route, its departure time must be no earlier than the time the route leaves the depot plus the effective travel time from the depot to that customer.
\[
t_i\ge d_i+c_{0i}-M^{D}_i(1-x_{0i}),
\qquad i\in N.
\]
### C8 Effective travel time for the last customer returning to the depot
When customer \(i\) is the last customer of a route, the route return time must be no earlier than the time of leaving that customer plus the effective travel time from the customer to the depot, which already includes the service time of that customer and any applicable time-window correction.
\[
r_i\ge t_i+c_{i0}-M^{R}_i(1-x_{i0}),
\qquad i\in N.
\]
### C9 Depot return times lie within the scheduling horizon
Each route must return to the depot within the depot horizon, and returning exactly at the end of the horizon is allowed; auxiliary return times of non-last customers are also placed in the same non-empty interval and do not affect route feasibility.
\[
a_0\le r_i\le b_0,\qquad i\in N.
\]
### C10 Customer departure load covers its own demand
When leaving any customer, the cumulative load label must be at least the pickup demand of that customer.
\[
L_i\ge q_i,\qquad i\in N.
\]
### C11 Cumulative load does not exceed vehicle capacity
The cumulative load label when leaving any customer must not exceed the vehicle capacity; vehicles start empty and only pick up, so this also bounds the load along the whole route.
\[
L_i\le Q,\qquad i\in N.
\]
### C12 Accumulate customer demand along selected arcs
When customer \(j\) is reached directly from the depot or a preceding customer, its departure load label must be at least the predecessor load plus its own demand; the depot departure load uses the fixed value \(L_0=0\). Return arcs do not reset the return load to zero before applying this recursion.
\[
L_j\ge L_i+q_j-Q(1-x_{ij}),
\qquad (i,j)\in E,\ j\in N,
\qquad L_0=0.
\]
## Solution and boundary conventions
- **Complete solution and vehicles.** A complete solution is a set of routes covering all customers, together with feasible times and load labels. Each route leaves the depot, visits customers in order and finally returns to the depot, and is executed by one separate vehicle; a route does not revisit the depot in between. The set of routes must satisfy all constraints above; partial coverage, splitting the demand of a customer, or replacing service by violating constraints is not allowed.
- **Routes really connect to the depot.** C1 and C2 give each customer exactly one predecessor and successor. Any directed cycle containing only customers would lead, by C12, to the contradiction \(0\ge\sum q_i>0\), so no customer subtour detached from the depot exists. Hence the selected arcs decompose into routes that leave and return to the depot, and \(\sum_{i\in N}x_{0i}=\sum_{i\in N}x_{i0}\). Strictly positive travel times are not needed to eliminate subtours.
- **Original meaning of the load lower bounds.** C10 and C12 keep the "greater than or equal to" of the problem and are not changed into equalities. For any feasible route \((0,i_1,\ldots,i_m,0)\), the labels can be set to the actual cumulative pickup \(L_{i_k}=\sum_{h=1}^{k}q_{i_h}\); this choice satisfies all load constraints. Thus optional slack in the labels does not represent additional goods, discarded goods or extra loading permission; the actual vehicle leaves empty and collects all demand along the way.
- **Original meaning of times and service times.** \(t_i\) is always called the customer departure time, as in the problem, and the given outgoing effective travel time recursion is applied directly. Service time is not counted again anywhere else, and no additional condition such as \(t_i+s_i\le b_i\) is introduced. The "greater than or equal to" in the recursion keeps slack for waiting and does not require all vehicles to depart simultaneously.
- **Linear terms for unselected arcs.** The three kinds of \(M\) are given by the known time-window bounds of the variables, which guarantees that the time constraints of unselected arcs hold automatically. C12 also holds automatically on unselected arcs because \(L_i\le Q\) and \(L_j\ge q_j\); the initial depot load is zero. Hence these terms do not relax any requirement on selected arcs.
- **Ties.** Every complete feasible route set attaining the minimum number of vehicles is an optimal answer; base distance, effective travel time, waiting time and other metrics are not compared.
- **Empty set and single customer.** The empty customer set is valid; all customer-related constraints are then empty, and the complete solution has zero routes and zero vehicles, with objective value zero. A single-customer instance is valid and must use the route from the depot to that customer and back, satisfying the same time and capacity requirements.
- **Equalities and zeros.** Using capacity exactly, leaving a customer exactly at a time-window endpoint and returning exactly at the end of the depot horizon are all allowed. Service times may be zero and time windows may degenerate to a single instant; the problem does not require travel times between distinct locations to be strictly positive, and zero travel times do not change the rules above.
- **Infeasibility and input validity.** Inputs that satisfy the data definitions and structural guarantees of the problem may have no complete feasible plan, including when a single demand exceeds capacity or time constraints cannot be met; infeasibility should then be reported, without omitting customers or violating constraints. Violating the data definitions or problem guarantees (for example a negative customer demand or failing the required effective-travel-time triangle inequality) is not the same as route infeasibility and is not remedied by modifying the input. Customer demands and vehicle capacity are always positive reals, and an empty customer set does not lift this requirement on vehicle capacity.
