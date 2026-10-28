# bront2009 · Mathematical model

> Choice-based network revenue management (CDLP)
>
> Sense: max

---

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

The airline aims to maximize total expected revenue over a discrete booking horizon T by deciding which subset of products (offer sets) to display to customers in each time period. Customers from different segments arrive with specific probabilities and choose products from the offered set based on a Multinomial Logit (MNL) model. The problem is formulated as a Choice-Based Deterministic Linear Program (CDLP) where the decision is the duration each possible offer set is active, subject to aircraft leg capacity constraints and the total time horizon. All indices for products, legs, and segments start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- t[S]: The total number of time periods (duration) for which a specific offer set S (a subset of products) is made available to customers.  |  t[S] ≥ 0, continuous

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] The total consumption of each flight leg's capacity by all products across all offer sets must not exceed the leg's initial capacity.  |  sum_{S \subseteq N} (lambda * Q_i(S) * t[S]) <= c_i, \forall i \in {1, \dots, m}
- [hard] The total time allocated to all offer sets cannot exceed the total number of periods in the booking horizon.  |  sum_{S \subseteq N} t[S] <= T

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- lambda: lambda; unit=probability; axis: scalar
- booking_horizon_T: T; unit=periods; axis: scalar
- network_legs_capacity: c_i; unit=seats; axis: leg_id
- network_num_legs: m; axis: scalar
- products_product_id: N; axis: product_id
- products_fare: r_j; unit=currency; axis: product_id
- products_legs_used: A; axis: product_id x leg_ids
- alpha: p_l; unit=probability; axis: Note: In this instance, segment shares are derived from global arrival rate and segment_lambda_l
- segments_lambda_l: lambda_l; unit=probability; axis: segment_id
- segments_consideration_set: C_l; axis: segment_id x product_ids
- segments_preference_vector: v_lj; unit=weight; axis: segment_id x product_index_in_consideration_set
- segments_no_purchase_preference: v_l0; unit=weight; axis: segment_id

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: maximize
Formula: max ∑_{S ⊆ N} (lambda * R(S) * t[S])
Plain: Maximize the total expected revenue across all selected offer sets over the booking horizon.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"alpha": float, "booking_horizon_T": int, "lambda": float, "network_city_names": [str, ...], "network_legs_capacity": [int, ...], "network_legs_destination": [str, ...], "network_legs_leg_id": [int, ...], "network_legs_origin": [str, ...], "network_num_cities": int, "network_num_legs": int, "products_class": [str, ...], "products_fare": [float, ...], "products_legs_used": [[int, ...], ...], "products_origin_destination": [str, ...], "products_product_id": [int, ...], "segments_consideration_set": [[int, ...], ...], "segments_description": [str, ...], "segments_lambda_l": [float, ...], "segments_no_purchase_preference": [float, ...], "segments_preference_vector": [[float, ...], ...], "segments_segment_id": [int, ...], "source": str}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "offer_set_allocations": list[dict] — the sales duration (in time periods) allocated to each possible product combination (offer set). example: [{"offer_set": [1, 2, 5], "time_allocated": 45.5}, {"offer_set": [3, 4], "time_allocated": 120.0}, {"offer_set": [], "time_allocated": 34.5}]. List every offer set allocated a non-zero duration; for completeness, unselected offer sets default to a duration of 0, and column-generation results usually show only positive entries, but all activated columns must be included.
Fill every value for every entity; keep the keys EXACTLY as above.
