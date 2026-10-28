# desaulniers2016 · Mathematical model

## Sets and indices

- Text: The planning horizon consists of discrete periods; the set of customers excludes the distribution center; the set of locations includes the distribution center, indexed zero; the set of candidate vehicles corresponds to the available identical vehicles; and the set of arcs excludes self-loops.
  Formula: $P=\{1,\ldots,T\},\ C=\{1,\ldots,n\},\ N=\{0\}\cup C,\ K=\{1,\ldots,m\},\ A=\{(i,j)\in N\times N:i\ne j\}$

## Parameters

- Text: The common load capacity of each vehicle is $Q$, and the number of available vehicles is $m$.
  Formula: $Q\ge 0,\quad |K|=m$
- Text: The coordinates of location $i$ are $(a_i,b_i)$, and the input travel cost $c_{ij}$ of arc $(i,j)$ is a symmetric cost obtained by rounding the Euclidean distance between the coordinates.
  Formula: $c_{ij}=c_{ji}\ge 0,\quad \forall(i,j)\in A$
- Text: The known production of the distribution center in period $t$ is $p_t$.
  Formula: $p_t\ge 0,\quad \forall t\in P$
- Text: The known demand of customer $i$ in period $t$ is $d_{it}$.
  Formula: $d_{it}\ge 0,\quad \forall i\in C,\ t\in P$
- Text: The inventory capacity, unit end-of-period holding cost, and initial inventory of location $i$ are $S_i$, $h_i$, and $I_i^{\mathrm{init}}$, respectively, and the initial inventory is feasible.
  Formula: $S_i\ge 0,\ h_i\ge 0,\ 0\le I_i^{\mathrm{init}}\le S_i,\quad \forall i\in N$
- Text: The inventory in period zero is fixed to the input initial inventory.
  Formula: $I_{i0}:=I_i^{\mathrm{init}},\quad \forall i\in N$

## Decision variables

- Text: $x_{ijkt}$ indicates whether vehicle $k$ travels directly from location $i$ to location $j$ in period $t$.
  Formula: $x_{ijkt}\in\{0,1\},\quad \forall(i,j)\in A,\ k\in K,\ t\in P$
- Text: $y_{ikt}$ indicates whether customer $i$ is visited by vehicle $k$ in period $t$.
  Formula: $y_{ikt}\in\{0,1\},\quad \forall i\in C,\ k\in K,\ t\in P$
- Text: $z_{kt}$ indicates whether vehicle $k$ operates a route in period $t$.
  Formula: $z_{kt}\in\{0,1\},\quad \forall k\in K,\ t\in P$
- Text: $q_{ikt}$ denotes the quantity of goods delivered by vehicle $k$ to customer $i$ in period $t$.
  Formula: $q_{ikt}\in\mathbb{R}_{+},\quad \forall i\in C,\ k\in K,\ t\in P$
- Text: $I_{it}$ denotes the inventory at location $i$ at the end of period $t$; its nonnegativity and capacity requirements are imposed by constraints.
  Formula: $I_{it}\in\mathbb{R},\quad \forall i\in N,\ t\in P$

## Objective

- Text: Minimize the sum of the total travel cost of all vehicles over the whole planning horizon and the end-of-period inventory holding costs at the distribution center and all customers.
  Formula: $\displaystyle \min\ \sum_{t\in P}\sum_{k\in K}\sum_{(i,j)\in A}c_{ij}x_{ijkt}+\sum_{t\in P}\sum_{i\in N}h_iI_{it}$

## Constraints

- C1 Text: Each activated vehicle departs from the distribution center exactly once in the corresponding period, and a vehicle that is not activated does not depart from the distribution center.
  Formula: $\displaystyle \sum_{j\in C}x_{0jkt}=z_{kt},\quad \forall k\in K,\ t\in P$
- C2 Text: Each activated vehicle returns to the distribution center exactly once in the corresponding period, and a vehicle that is not activated does not return to the distribution center.
  Formula: $\displaystyle \sum_{i\in C}x_{i0kt}=z_{kt},\quad \forall k\in K,\ t\in P$
- C3 Text: When a vehicle visits a customer, it must leave that customer exactly once; when it does not visit, it must not leave that customer.
  Formula: $\displaystyle \sum_{j\in N\setminus\{i\}}x_{ijkt}=y_{ikt},\quad \forall i\in C,\ k\in K,\ t\in P$
- C4 Text: When a vehicle visits a customer, it must enter that customer exactly once; when it does not visit, it must not enter that customer.
  Formula: $\displaystyle \sum_{j\in N\setminus\{i\}}x_{jikt}=y_{ikt},\quad \forall i\in C,\ k\in K,\ t\in P$
- C5 Text: Each customer is visited by at most one vehicle in each period.
  Formula: $\displaystyle \sum_{k\in K}y_{ikt}\le 1,\quad \forall i\in C,\ t\in P$
- C6 Text: The number of routes operated in each period must not exceed the number of available vehicles.
  Formula: $\displaystyle \sum_{k\in K}z_{kt}\le m,\quad \forall t\in P$
- C7 Text: No vehicle in any period may form a customer subtour that does not pass through the distribution center.
  Formula: $\displaystyle \sum_{i\in S}\sum_{j\in S\setminus\{i\}}x_{ijkt}\le |S|-1,\quad \forall\varnothing\ne S\subseteq C,\ k\in K,\ t\in P$
- C8 Text: A vehicle may deliver only to customers it actually visits, while the problem statement allows a zero delivery quantity when a customer is visited.
  Formula: $\displaystyle q_{ikt}\le Qy_{ikt},\quad \forall i\in C,\ k\in K,\ t\in P$
- C9 Text: The total delivery quantity of each activated vehicle in each period must not exceed the common vehicle capacity, and a vehicle that is not activated must not deliver.
  Formula: $\displaystyle \sum_{i\in C}q_{ikt}\le Qz_{kt},\quad \forall k\in K,\ t\in P$
- C10 Text: A customer's end-of-period inventory equals the end-of-previous-period inventory plus the deliveries by all vehicles in the current period minus the current-period demand.
  Formula: $\displaystyle I_{it}=I_{i,t-1}+\sum_{k\in K}q_{ikt}-d_{it},\quad \forall i\in C,\ t\in P$
- C11 Text: The distribution center's end-of-period inventory equals the end-of-previous-period inventory plus the current-period production minus all shipments in the current period.
  Formula: $\displaystyle I_{0t}=I_{0,t-1}+p_t-\sum_{k\in K}\sum_{i\in C}q_{ikt},\quad \forall t\in P$
- C12 Text: A customer's inventory after receiving the current-period delivery at the beginning of the period and before consuming the current-period demand must not exceed its inventory capacity.
  Formula: $\displaystyle I_{i,t-1}+\sum_{k\in K}q_{ikt}\le S_i,\quad \forall i\in C,\ t\in P$
- C13 Text: Each customer's inventory at the end of each period must be nonnegative, so no stockout may occur.
  Formula: $\displaystyle I_{it}\ge 0,\quad \forall i\in C,\ t\in P$
- C14 Text: The distribution center's inventory at the end of each period must be nonnegative and must not exceed its inventory capacity.
  Formula: $\displaystyle 0\le I_{0t}\le S_0,\quad \forall t\in P$
