# adulyasak2015 · Mathematical model

## Sets, indices, and scenario information

Text: The planning horizon consists of a finite, ordered set of periods.

Formula: $$T=\{1,\ldots,H\}$$

Text: The set of customers is $C$, the plant is represented by node $0$, and the set of all locations and the set of allowed directed arcs are $V$ and $A$, respectively.

Formula: $$V=\{0\}\cup C,\qquad A=\{(i,j)\in V\times V:i\ne j\},\qquad n=|C|$$

Text: The set of homogeneous vehicles, which can be reused for deliveries in every period, is $K$.

Formula: $$k\in K$$

Text: The set of demand scenarios is $\Omega$, and the probability of scenario $s$ is $\pi_s$; deterministic demand is the special case $|\Omega|=1$, $\pi_s=1$.

Formula: $$\pi_s>0\quad(\forall s\in\Omega),\qquad \sum_{s\in\Omega}\pi_s=1$$

Text: The demand history of scenario $s$ observed up to period $t$ consists of the demands of all customers in periods $1$ through $t$.

Formula: $$\mathcal H_t(s)=\bigl(d_{i\tau}^{s}:i\in C,\ \tau=1,\ldots,t\bigr)$$

## Parameters

Text: $d_{it}^{s}$ is the demand of customer $i$ in period $t$ under scenario $s$.

Formula: $$d_{it}^{s}\ge 0\qquad(\forall i\in C,t\in T,s\in\Omega)$$

Text: $I_v^{\mathrm{init}}$ and $\bar I_v$ are the initial inventory and the maximum inventory capacity of location $v$, respectively, and the period-zero inventory in every scenario is fixed to this initial inventory.

Formula: $$I_{v0}^{s}:=I_v^{\mathrm{init}},\qquad 0\le I_v^{\mathrm{init}}\le \bar I_v\qquad(\forall v\in V,s\in\Omega)$$

Text: $\bar P$ is the maximum production quantity of the plant per period, and $Q$ is the loading capacity of each homogeneous vehicle.

Formula: $$\bar P\ge0,
\qquad Q>0$$

Text: $f$, $c^p$, $h_v$, and $\rho_i$ are, respectively, the production setup cost per setup, the unit production cost, the unit end-of-period holding cost at location $v$, and the unit penalty cost for unmet demand of customer $i$.

Formula: $$f,c^p,h_v,\rho_i\ge0$$

Text: $c_{ij}$ is the transportation cost of a vehicle traveling directly from location $i$ to a different location $j$, and transportation costs are symmetric.

Formula: $$c_{ij}=c_{ji}\ge0\qquad(\forall(i,j)\in A)$$

## Decision variables

Text: $y_t$ indicates whether production is set up in period $t$; it is determined once before demand is realized and carries no scenario index.

Formula: $$y_t\in\{0,1\}\qquad(\forall t\in T)$$

Text: $w_{kt}$ indicates whether vehicle $k$ is dispatched in period $t$; it is determined once before demand is realized.

Formula: $$w_{kt}\in\{0,1\}\qquad(\forall k\in K,t\in T)$$

Text: $z_{ikt}$ indicates whether vehicle $k$ visits customer $i$ in period $t$; it is determined once before demand is realized.

Formula: $$z_{ikt}\in\{0,1\}\qquad(\forall i\in C,k\in K,t\in T)$$

Text: $x_{ijkt}$ indicates whether vehicle $k$ travels directly from location $i$ to location $j$ in period $t$; it is determined once before demand is realized.

Formula: $$x_{ijkt}\in\{0,1\}\qquad(\forall(i,j)\in A,k\in K,t\in T)$$

Text: $u_{ikt}$ is an auxiliary visiting-order variable used to eliminate customer subtours; it likewise does not vary with the scenario.

Formula: $$1\le u_{ikt}\le n\qquad(\forall i\in C,k\in K,t\in T)$$

Text: $P_t^s$ is the production quantity in period $t$ under scenario $s$.

Formula: $$P_t^s\ge0\qquad(\forall t\in T,s\in\Omega)$$

Text: $q_{ikt}^s$ is the quantity delivered by vehicle $k$ to customer $i$ in period $t$ under scenario $s$.

Formula: $$q_{ikt}^s\ge0\qquad(\forall i\in C,k\in K,t\in T,s\in\Omega)$$

Text: $I_{vt}^s$ is the inventory at location $v$ at the end of period $t$ under scenario $s$.

Formula: $$I_{vt}^s\ge0\qquad(\forall v\in V,t\in T,s\in\Omega)$$

Text: $L_{it}^s$ is the demand of customer $i$ in period $t$ under scenario $s$ that is not satisfied and incurs a penalty cost in that period.

Formula: $$L_{it}^s\ge0\qquad(\forall i\in C,t\in T,s\in\Omega)$$

## Objective

Text: Minimize the production setup costs and route transportation costs over the whole planning horizon, plus the expected total cost, weighted by scenario probabilities, of production, plant and customer inventory holding, and unmet-demand penalties.

Formula: $$\min\quad f\sum_{t\in T}y_t+\sum_{t\in T}\sum_{k\in K}\sum_{(i,j)\in A}c_{ij}x_{ijkt}+\sum_{s\in\Omega}\pi_s\sum_{t\in T}\left[c^pP_t^s+h_0I_{0t}^s+\sum_{i\in C}\left(h_iI_{it}^s+\rho_iL_{it}^s\right)\right]$$

## Constraints

C1  Text: Production is possible only if a production setup has been made in the current period, and the production quantity must not exceed the plant's per-period production capacity.

Formula: $$P_t^s\le \bar P\,y_t\qquad(\forall t\in T,s\in\Omega)$$

C2  Text: The plant's end-of-previous-period inventory plus the current production quantity must equal the total quantity shipped to all customers in the current period plus the plant's end-of-current-period inventory.

Formula: $$I_{0,t-1}^s+P_t^s=\sum_{i\in C}\sum_{k\in K}q_{ikt}^s+I_{0t}^s\qquad(\forall t\in T,s\in\Omega)$$

C3  Text: The plant's end-of-period inventory must not exceed the plant's inventory capacity.

Formula: $$I_{0t}^s\le\bar I_0\qquad(\forall t\in T,s\in\Omega)$$

C4  Text: For each customer, the sum of the end-of-previous-period inventory, all deliveries in the current period, and the unmet demand in the current period must equal the current-period demand plus the end-of-current-period inventory.

Formula: $$I_{i,t-1}^s+\sum_{k\in K}q_{ikt}^s+L_{it}^s=d_{it}^s+I_{it}^s\qquad(\forall i\in C,t\in T,s\in\Omega)$$

C5  Text: A customer's unmet demand in the current period must not exceed that customer's demand in the current period.

Formula: $$L_{it}^s\le d_{it}^s\qquad(\forall i\in C,t\in T,s\in\Omega)$$

C6  Text: Following the capacity rule given in the problem statement, under which arrivals precede demand consumption, the sum of a customer's end-of-period inventory and current-period demand must not exceed its inventory capacity.

Formula: $$I_{it}^s+d_{it}^s\le\bar I_i\qquad(\forall i\in C,t\in T,s\in\Omega)$$

C7  Text: The total delivery quantity loaded on each vehicle in each period must not exceed the vehicle capacity, and a vehicle that is not dispatched must not be loaded.

Formula: $$\sum_{i\in C}q_{ikt}^s\le Qw_{kt}\qquad(\forall k\in K,t\in T,s\in\Omega)$$

C8  Text: A vehicle may deliver product to a customer only if its fixed route visits that customer.

Formula: $$q_{ikt}^s\le Qz_{ikt}\qquad(\forall i\in C,k\in K,t\in T,s\in\Omega)$$

C9  Text: The total quantity delivered to a customer in the current period must not exceed the remaining storage space before arrival, as determined by its end-of-previous-period inventory.

Formula: $$\sum_{k\in K}q_{ikt}^s\le\bar I_i-I_{i,t-1}^s\qquad(\forall i\in C,t\in T,s\in\Omega)$$

C10  Text: The total quantity delivered to a customer in the current period must not exceed the cumulative demand in this scenario from the current period to the end of the planning horizon.

Formula: $$\sum_{k\in K}q_{ikt}^s\le\sum_{\tau=t}^{H}d_{i\tau}^s\qquad(\forall i\in C,t\in T,s\in\Omega)$$

C11  Text: Each customer is visited by at most one vehicle in each period.

Formula: $$\sum_{k\in K}z_{ikt}\le1\qquad(\forall i\in C,t\in T)$$

C12  Text: If a vehicle visits a customer, its route must have exactly one arc entering that customer and exactly one arc leaving that customer, and vice versa.

Formula: $$\sum_{j\in V\setminus\{i\}}x_{jikt}=z_{ikt}=\sum_{j\in V\setminus\{i\}}x_{ijkt}\qquad(\forall i\in C,k\in K,t\in T)$$

C13  Text: Each dispatched vehicle must leave the plant exactly once and return to the plant exactly once, and a vehicle that is not dispatched must not use arcs incident to the plant.

Formula: $$\sum_{j\in C}x_{0jkt}=w_{kt}=\sum_{i\in C}x_{i0kt}\qquad(\forall k\in K,t\in T)$$

C14  Text: Ordering constraints on arcs between customers eliminate customer subtours disconnected from the plant, so that each dispatched vehicle forms only a single tour that starts and ends at the plant.

Formula: $$u_{ikt}-u_{jkt}+n x_{ijkt}\le n-1\qquad(\forall i,j\in C:i\ne j,\ k\in K,t\in T)$$

C15  Text: Production, delivery, inventory, and unmet-demand decisions may depend only on the demand history observed up to the current period; therefore, scenarios with identical histories must adopt identical current-period operational decisions.

Formula: $$\mathcal H_t(s)=\mathcal H_t(s')\ \Longrightarrow\ \left(P_t^s,(q_{ikt}^s)_{i\in C,k\in K},(I_{vt}^s)_{v\in V},(L_{it}^s)_{i\in C}\right)=\left(P_t^{s'},(q_{ikt}^{s'})_{i\in C,k\in K},(I_{vt}^{s'})_{v\in V},(L_{it}^{s'})_{i\in C}\right)\quad(\forall t\in T,s,s'\in\Omega)$$
