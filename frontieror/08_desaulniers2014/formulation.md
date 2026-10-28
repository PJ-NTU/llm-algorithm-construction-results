# desaulniers2014 · Mathematical model

## Sets and indices

- Text: The planning horizon consists of H discrete periods arranged in chronological order.
  Formula: T = {1,…,H}.
- Text: C is the set of customers, index 0 denotes the supplier, V is the set of all locations, and A is the set of directed arcs without self-loops.
  Formula: V = {0} ∪ C, A = {(i,j) ∈ V × V : i ≠ j}.
- Text: There are M identical vehicles available, and K is the set of their labels.
  Formula: K = {1,…,M}.

## Known parameters

- Text: The loading capacity of each vehicle is Q.
  Formula: Q > 0.
- Text: The known production of the supplier in period t is p_t, and the known demand of customer i in period t is d_it.
  Formula: p_t ≥ 0, d_it ≥ 0, ∀t ∈ T, i ∈ C.
- Text: The inventory capacity of location ℓ is U_ℓ, its initial inventory is I⁰_ℓ, and the initial inventory does not exceed the capacity.
  Formula: 0 ≤ I⁰_ℓ ≤ U_ℓ, ∀ℓ ∈ V.
- Text: The unit end-of-period inventory holding cost at location ℓ is h_ℓ.
  Formula: h_ℓ ≥ 0, ∀ℓ ∈ V.
- Text: The unit travel cost between locations i and j is c_ij, and travel costs are symmetric.
  Formula: c_ij = c_ji ≥ 0, ∀(i,j) ∈ A.

## Decision variables

- Text: x_ijkt indicates whether vehicle k travels directly from location i to location j in period t.
  Formula: x_ijkt ∈ {0,1}, ∀(i,j) ∈ A, k ∈ K, t ∈ T.
- Text: y_ikt indicates whether customer i is visited by vehicle k in period t.
  Formula: y_ikt ∈ {0,1}, ∀i ∈ C, k ∈ K, t ∈ T.
- Text: z_kt indicates whether vehicle k performs a route in period t.
  Formula: z_kt ∈ {0,1}, ∀k ∈ K, t ∈ T.
- Text: q_ikt is the quantity of goods loaded by vehicle k for customer i and delivered to it in period t.
  Formula: q_ikt ∈ ℝ₊, ∀i ∈ C, k ∈ K, t ∈ T.
- Text: I_ℓt is the inventory at location ℓ at the end of period t, initialized with the given initial inventory.
  Formula: I_ℓt ∈ ℝ, I_ℓ0 = I⁰_ℓ, ∀ℓ ∈ V, t ∈ T.

## Objective

- Text: Minimize the sum of the total travel cost of all vehicle routes over all periods and the end-of-period inventory holding costs at all locations.
  Formula: min Z = Σ_{t∈T} [Σ_{k∈K} Σ_{(i,j)∈A} c_ij x_ijkt + h_0 I_0t + Σ_{i∈C} h_i I_it].

## Constraints

- C1  Text: If vehicle k visits customer i in period t, it must leave that customer exactly once; otherwise it must not leave that customer.
  Formula: Σ_{j∈V:j≠i} x_ijkt = y_ikt, ∀i ∈ C, k ∈ K, t ∈ T.
- C2  Text: If vehicle k visits customer i in period t, it must enter that customer exactly once; otherwise it must not enter that customer.
  Formula: Σ_{j∈V:j≠i} x_jikt = y_ikt, ∀i ∈ C, k ∈ K, t ∈ T.
- C3  Text: A vehicle performing a route must depart from the supplier exactly once; a vehicle not performing a route must not depart from the supplier.
  Formula: Σ_{j∈C} x_0jkt = z_kt, ∀k ∈ K, t ∈ T.
- C4  Text: A vehicle performing a route must return to the supplier exactly once; a vehicle not performing a route must not return to the supplier.
  Formula: Σ_{i∈C} x_i0kt = z_kt, ∀k ∈ K, t ∈ T.
- C5  Text: Each customer is visited by at most one vehicle in each period.
  Formula: Σ_{k∈K} y_ikt ≤ 1, ∀i ∈ C, t ∈ T.
- C6  Text: A vehicle may deliver goods only to customers it actually visits, and the quantity delivered to a single customer cannot exceed the vehicle capacity.
  Formula: 0 ≤ q_ikt ≤ Q y_ikt, ∀i ∈ C, k ∈ K, t ∈ T.
- C7  Text: The total quantity loaded and delivered by each vehicle in a period must not exceed its loading capacity, and a vehicle not performing a route must not be loaded.
  Formula: Σ_{i∈C} q_ikt ≤ Q z_kt, ∀k ∈ K, t ∈ T.
- C8  Text: The total number of routes performed in each period must not exceed the number of available vehicles.
  Formula: Σ_{k∈K} z_kt ≤ M, ∀t ∈ T.
- C9  Text: Every vehicle route must be connected to the supplier; subtours consisting only of customers are not allowed.
  Formula: Σ_{i∈S} Σ_{j∈V∖S} x_ijkt ≥ y_rkt, ∀t ∈ T, k ∈ K, ∅ ≠ S ⊆ C, r ∈ S.
- C10  Text: A customer's end-of-previous-period inventory plus the delivery at the beginning of the current period must be sufficient to satisfy the current-period demand, and must not exceed the customer's inventory capacity before demand is consumed.
  Formula: d_it ≤ I_i,t−1 + Σ_{k∈K} q_ikt ≤ U_i, ∀i ∈ C, t ∈ T.
- C11  Text: A customer's end-of-current-period inventory equals the end-of-previous-period inventory plus the current-period delivery minus the current-period demand.
  Formula: I_it = I_i,t−1 + Σ_{k∈K} q_ikt − d_it, ∀i ∈ C, t ∈ T.
- C12  Text: The supplier's end-of-current-period inventory equals the end-of-previous-period inventory plus the current-period production minus the total quantity loaded on all vehicles in the current period.
  Formula: I_0t = I_0,t−1 + p_t − Σ_{k∈K} Σ_{i∈C} q_ikt, ∀t ∈ T.
- C13  Text: The supplier's end-of-period inventory in each period must be nonnegative and must not exceed its inventory capacity.
  Formula: 0 ≤ I_0t ≤ U_0, ∀t ∈ T.
