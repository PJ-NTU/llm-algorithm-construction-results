# watermeyer2020 · Mathematical model

> Resource-constrained project scheduling with general temporal constraints and partially renewable resources
>
> Sense: min

---

The Resource-Constrained Project Scheduling Problem with Partially Renewable Resources and General Temporal Constraints (RCPSP/max-π) involves scheduling a set of activities V = {0, 1, ..., n, n+1} to minimize the project completion time. Activity 0 is the project start and n+1 is the project end. Each activity i has a fixed duration p_i and resource demands. Partially renewable resources k have a total capacity R_k over a specific subset of periods Π_k. General temporal constraints (minimum and maximum time lags) exist between pairs of activities. All indices for activities and resources are 0-indexed.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- start_times: The scheduled start time for each activity i in the set V.  |  S_i ∈ ℤ≥0

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each activity must start at a non-negative time, with the project-start activity fixed at time zero.  |  S_0 = 0
- [hard] General temporal constraints between activities must be satisfied, where the difference between the start times of activity j and activity i must be at least a specified time lag delta.  |  S_j - S_i \geq \delta_{ij} \quad \forall (i, j) \in E
- [hard] For each partially renewable resource, the total consumption across all activities over the resource's designated period subset must not exceed its capacity.  |  \sum_{i=0}^{n+1} (r_{ik}^{d} \cdot |\{t \in \mathbb{Z} \mid S_i < t \leq S_i + p_i \text{ and } t \in \Pi_k\}|) \leq R_k \quad \forall k \in \mathcal{R}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n: n
- activities_processing_time: p_i; unit=time periods; axis: activity_id
- activities_resource_demands_{k}: r_{ik}^{d}; unit=resource units per period; axis: resource_id, activity_id
- temporal_constraints_delta: \delta_{ij}; unit=time periods; axis: constraint_index
- temporal_constraints_i: i; axis: constraint_index
- temporal_constraints_j: j; axis: constraint_index
- resources_capacity: R_k; unit=resource units; axis: resource_id
- resources_periods: \Pi_k; unit=period index; axis: resource_id, periods
- d_bar: \bar{d}; unit=time periods

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min S_{n+1}
Plain: Minimize the start time of the fictitious project-end activity.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "num_activities": int, "num_resources": int, "d_bar": int, "activities_processing_time": [int, ...], "temporal_constraints_i": [int, ...], "temporal_constraints_j": [int, ...], "temporal_constraints_delta": [int, ...], "resources_capacity": [int, ...], "resources_periods": [[int, ...], ...], "parameters_cardinality_factor_CF": float, "parameters_horizon_factor_HF": float, "parameters_interval_factor_IF": float, "parameters_order_strength_OS": float, "parameters_resource_factor_RF": float, "parameters_resource_strength_RS": float, "activities_resource_demands_0": [int, ...], "activities_resource_demands_1": [int, ...], "activities_resource_demands_2": [int, ...], "activities_resource_demands_3": [int, ...], "activities_resource_demands_4": [int, ...], "activities_resource_demands_5": [int, ...], "activities_resource_demands_6": [int, ...], "activities_resource_demands_7": [int, ...], "activities_resource_demands_8": [int, ...], "activities_resource_demands_9": [int, ...], "activities_resource_demands_10": [int, ...], "activities_resource_demands_11": [int, ...], "activities_resource_demands_12": [int, ...], "activities_resource_demands_13": [int, ...], "activities_resource_demands_14": [int, ...], "activities_resource_demands_15": [int, ...], "activities_resource_demands_16": [int, ...], "activities_resource_demands_17": [int, ...], "activities_resource_demands_18": [int, ...], "activities_resource_demands_19": [int, ...], "activities_resource_demands_20": [int, ...], "activities_resource_demands_21": [int, ...], "activities_resource_demands_22": [int, ...], "activities_resource_demands_23": [int, ...], "activities_resource_demands_24": [int, ...], "activities_resource_demands_25": [int, ...], "activities_resource_demands_26": [int, ...], "activities_resource_demands_27": [int, ...], "activities_resource_demands_28": [int, ...], "activities_resource_demands_29": [int, ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "start_times": dict[str, int] — scheduled start time of each activity (including the fictitious start and end activities). example: {"0": 0, "1": 5, "2": 12, "3": 8, "4": 15, "5": 20, "6": 22, "7": 25, "8": 30, "9": 35, "10": 40, "11": 45}. Must include all activity IDs from 0 to n+1, where 0 is the project start and n+1 is the project end
Fill every value for every entity; keep the keys EXACTLY as above.