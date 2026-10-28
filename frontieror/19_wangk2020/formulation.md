# wangk2020 · Mathematical model

## Notation and input data

- Sets: flight set $F$, airport set $A$, set of actual time periods $T={1,…,H}$, set of weather scenarios $Ω$, set of event types $K={D,A}$, where $D$ denotes departure and $A$ denotes arrival. Let $T⁺={1,…,H+1}$; $H+1$ is a sentinel period used only to mark the end of a sequence.
- Flight data: $orig(f)$ and $dest(f)$ are the departure and arrival airports of flight $f$; $r_f^D$ and $r_f^A$ are the requested departure and arrival periods; $τ_f^S=r_f^A-r_f^D$ is the fixed scheduled flight duration; $τ_f^{min}$ and $τ_f^{max}$ are the lower and upper bounds on the actual flight duration; $L_f^D$ and $L_f^A$ are the maximum departure and arrival delays; $c_f^D$ and $c_f^A$ are the departure and arrival delay costs per period.
- Schedule data: $Δ$ is the maximum scheduled departure displacement shared by all flights; $ρ∈[0,1]$ is the weight of the schedule displacement cost.
- Connection data: $C^S⊆F×F$ is the set of flight pairs used in the scheduling stage, containing both aircraft turnaround and passenger itinerary connections, and $m_{fg}^S$ is the minimum connection time of $(f,g)∈C^S$; $C^{AC}⊆F×F$ is the set of aircraft turnaround flight pairs used in the actual operations stage, and $m_{fg}^{AC}$ is its minimum connection time.
- Scenario data: $p_ω$ is the probability of scenario $ω$, satisfying $p_ω≥0$ and $Σ_{ω∈Ω}p_ω=1$; $q_{a,t,ω}∈{VMC,IMC}$ is the operating condition of airport $a$ in period $t$ under scenario $ω$.
- Capacity data: $L_{a,q}$ is the set of capacity envelope segments of airport $a$ under condition $q$; the departure coefficient, arrival coefficient, and capacity limit of segment $ℓ∈L_{a,q}$ are $α_{a,ℓ}^{q}$, $β_{a,ℓ}^{q}$, and $κ_{a,ℓ}^{q}$, respectively. The input guarantees that the IMC envelope is tighter than the VMC envelope.

Define the following derived quantities:

$$S_f^k:=Σ_{t∈T}x_{f,t}^k,\qquad O_{f,ω}^k:=Σ_{t∈T}y_{f,t,ω}^k,$$

$$N_{a,t,ω}^D:=Σ_{f:orig(f)=a}(y_{f,t,ω}^D-y_{f,t+1,ω}^D),\qquad N_{a,t,ω}^A:=Σ_{f:dest(f)=a}(y_{f,t,ω}^A-y_{f,t+1,ω}^A).$$

Under the endpoint and monotonicity constraints, $S_f^D,S_f^A,O_{f,ω}^D,O_{f,ω}^A$ equal the period indices of the corresponding events, and the difference between adjacent indicators indicates exactly whether the event occurs in that period.

## Objective

Text: Minimize the weighted sum of the total absolute displacement cost of the scheduled departure periods relative to the requested departure periods and the expected operational delay cost over the weather scenarios.

Formula:
$$min\; ρΣ_{f∈F}Σ_{t∈T}|t-r_f^D|(x_{f,t}^D-x_{f,t+1}^D)+(1-ρ)Σ_{ω∈Ω}p_ωΣ_{f∈F}(c_f^Dδ_{f,ω}^D+c_f^Aδ_{f,ω}^A).$$

## Decision variables

$x_{f,t}^k∈{0,1}$  Text: Equals 1 if scheduled event $k$ of flight $f$ occurs no earlier than period $t$, and 0 otherwise, where $f∈F$, $k∈K$, $t∈T⁺$.

$y_{f,t,ω}^k∈{0,1}$  Text: Equals 1 if actual operational event $k$ of flight $f$ under scenario $ω$ occurs no earlier than period $t$, and 0 otherwise, where $f∈F$, $k∈K$, $t∈T⁺$, $ω∈Ω$.

$δ_{f,ω}^D≥0$  Text: Nonnegative number of departure delay periods of flight $f$ under scenario $ω$.

$δ_{f,ω}^A≥0$  Text: Nonnegative number of arrival delay periods of flight $f$ under scenario $ω$.

## Constraints

**C1**  Text: Each scheduled event indicator sequence takes value 1 in the first period and 0 in the sentinel period.

Formula:
$$x_{f,1}^k=1,\qquad x_{f,H+1}^k=0,\qquad ∀f∈F,\;k∈K.$$

**C2**  Text: The scheduled departure and scheduled arrival indicators are nonincreasing over periods.

Formula:
$$x_{f,t}^k≥x_{f,t+1}^k,\qquad ∀f∈F,\;k∈K,\;t∈T.$$

**C3**  Text: In each scenario, each actual operational event indicator sequence takes value 1 in the first period and 0 in the sentinel period.

Formula:
$$y_{f,1,ω}^k=1,\qquad y_{f,H+1,ω}^k=0,\qquad ∀f∈F,\;k∈K,\;ω∈Ω.$$

**C4**  Text: In each scenario, the actual departure and actual arrival indicators are nonincreasing over periods.

Formula:
$$y_{f,t,ω}^k≥y_{f,t+1,ω}^k,\qquad ∀f∈F,\;k∈K,\;t∈T,\;ω∈Ω.$$

**C5**  Text: For each flight, the difference between the scheduled departure period and the requested departure period must not exceed the common maximum displacement.

Formula:
$$-Δ≤S_f^D-r_f^D≤Δ,\qquad ∀f∈F.$$

**C6**  Text: For each flight, the scheduled arrival period minus the scheduled departure period must equal its fixed scheduled flight duration.

Formula:
$$S_f^A-S_f^D=τ_f^S,\qquad ∀f∈F.$$

**C7**  Text: Every flight pair in the first connection set must satisfy its minimum connection time in the scheduling stage.

Formula:
$$S_g^D-S_f^A≥m_{fg}^S,\qquad ∀(f,g)∈C^S.$$

**C8**  Text: In each scenario, the departure delay equals the actual departure period minus the scheduled departure period.

Formula:
$$δ_{f,ω}^D=O_{f,ω}^D-S_f^D,\qquad ∀f∈F,\;ω∈Ω.$$

**C9**  Text: In each scenario, the arrival delay must be no less than the actual arrival period minus the scheduled arrival period.

Formula:
$$δ_{f,ω}^A≥O_{f,ω}^A-S_f^A,\qquad ∀f∈F,\;ω∈Ω.$$

**C10**  Text: The departure delay of each flight in each scenario must not exceed its maximum allowed departure delay.

Formula:
$$δ_{f,ω}^D≤L_f^D,\qquad ∀f∈F,\;ω∈Ω.$$

**C11**  Text: The arrival delay of each flight in each scenario must not exceed its maximum allowed arrival delay.

Formula:
$$δ_{f,ω}^A≤L_f^A,\qquad ∀f∈F,\;ω∈Ω.$$

**C12**  Text: Every flight pair in the aircraft turnaround connection set must satisfy its minimum connection time in the actual operations of each scenario.

Formula:
$$O_{g,ω}^D-O_{f,ω}^A≥m_{fg}^{AC},\qquad ∀(f,g)∈C^{AC},\;ω∈Ω.$$

**C13**  Text: The actual flight duration of each flight in each scenario must lie between its allowed minimum and maximum.

Formula:
$$τ_f^{min}≤O_{f,ω}^A-O_{f,ω}^D≤τ_f^{max},\qquad ∀f∈F,\;ω∈Ω.$$

**C14**  Text: Every airport, period, and weather scenario must satisfy each segment inequality of the capacity envelope under the operating condition prevailing at that time.

Formula:
$$α_{a,ℓ}^{q_{a,t,ω}}N_{a,t,ω}^D+β_{a,ℓ}^{q_{a,t,ω}}N_{a,t,ω}^A≤κ_{a,ℓ}^{q_{a,t,ω}},\qquad ∀a∈A,\;t∈T,\;ω∈Ω,\;ℓ∈L_{a,q_{a,t,ω}}.$$
