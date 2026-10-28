# Uncapacitated warehouse location: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Decide which of \(M\) candidate facilities to open and assign each of \(N\) customers to one open facility for service; facilities have no capacity limit, and the goal is to minimise the sum of fixed opening costs and assignment costs.

## Parameters and sets

| Symbol | Meaning | Unit or range |
|---|---|---|
| \(N\) | Total number of customers, data field `n` | customers; integer, \(N \ge 1\) |
| \(M\) | Total number of candidate facilities, data field `m_facilities` | count; integer, \(M \ge 1\) |
| \(I=\{0,1,\dots,M-1\}\) | Set of candidate facilities, facility index \(i \in I\), starting from 0 | index, dimensionless |
| \(J=\{0,1,\dots,N-1\}\) | Set of customers, customer index \(j \in J\), starting from 0 | index, dimensionless |
| \(f_i\) | Fixed cost of opening facility \(i\), `fixed_cost[i]`, ordered by facility, length \(M\) | currency units; finite real, \(f_i \ge 0\), may be fractional or 0 |
| \(c_{ij}\) | Assignment cost when customer \(j\) is served entirely by facility \(i\), `alloc_cost[i][j]`, \(M\) rows by \(N\) columns, rows are facilities and columns are customers | currency units; finite real, \(c_{ij} \ge 0\), may be fractional or 0 |

## Decision variables

| Symbol | Meaning | Range |
|---|---|---|
| \(y_i\) (\(i \in I\)) | Whether facility \(i\) is open: 1 means open, 0 means not open | \(y_i \in \{0,1\}\) |
| \(x_{ij}\) (\(i \in I,\ j \in J\)) | Whether customer \(j\) is served by facility \(i\): 1 means served by facility \(i\), 0 means not | \(x_{ij} \in [0,1]\) (as defined in the problem statement); a delivered solution takes only 0 or 1, see "Solution and boundary conventions" |

## Objective function
Minimise total cost, i.e. the sum of fixed costs of all open facilities plus the sum of each customer's assignment cost to its assigned facility; this is the only objective, both parts are in currency units and are added directly with weight 1.
\[ \min\; Z=\sum_{i \in I} f_i\, y_i+\sum_{i \in I}\sum_{j \in J} c_{ij}\, x_{ij} \]

## Constraints
**C1 Each customer is assigned exactly once (hard constraint)**: each customer's assignments over all facilities must sum to exactly 1, i.e. every customer must be served, none may be left out, and each is served by only one facility (delivered solutions are not split).
\[ \sum_{i \in I} x_{ij}=1,\quad \forall j \in J \]

**C2 Only open facilities may serve (hard constraint)**: a facility \(i\) that is not open (\(y_i=0\)) cannot serve any customer; an open facility may serve any number of customers (no capacity limit).
\[ x_{ij} \le y_i,\quad \forall i \in I,\ \forall j \in J \]

## Solution and boundary conventions
- **Valid complete solution (no splitting)**: a delivered solution must give all \(y_i\) and \(x_{ij}\), satisfy C1 and C2, and have \(x_{ij} \in \{0,1\}\) (\(\forall i \in I,\ j \in J\)), i.e. each customer \(j\) has exactly one facility \(i\) with \(x_{ij}=1\), and that facility has \(y_i=1\); solutions with fractional assignments (e.g. 0.5 and 0.5) are invalid.
- **The 0/1 requirement does not change the optimal value**: for any given \(y\), the minimum assignment cost of customer \(j\) is \(\min_{i \in I:\, y_i=1} c_{ij}\), achieved by setting \(x_{ij}=1\) for one facility attaining that minimum (others 0), so the model with \(x_{ij} \in [0,1]\) always has a 0/1-valued optimal solution.
- **Lowest-cost assignment is a property, not a constraint**: the statement "each customer is always assigned to the open facility with the lowest assignment cost" describes a property of optimal solutions; a solution that assigns a customer to a more expensive open facility is still valid, only with a higher objective value, which is computed from the actual \(y\), \(x\) in the solution.
- **Idle facilities**: a facility that is open but serves no customer is valid, and its fixed cost is still counted in the objective.
- **Number opened**: since \(N \ge 1\), C1 and C2 force any feasible solution to open at least one facility; beyond that the number opened is unrestricted, and all \(M\) facilities may be opened.
- **Ties**: there is no secondary objective; any valid solution attaining the minimum objective value is optimal, e.g. whether to open an idle facility with fixed cost 0, or which of several open facilities with equal assignment cost to choose.
- **Valid input**: \(N\) and \(M\) are positive integers (\(N=1\) or \(M=1\) is valid); `fixed_cost` has length \(M\) and `alloc_cost` has \(M\) rows and \(N\) columns; all costs are finite non-negative reals; any facility can serve any customer, with no forbidden assignments. \(N=0\) or \(M=0\), negative costs, infinity, NaN or missing values, or array dimensions inconsistent with \(N\) and \(M\) are treated as invalid input and rejected outright.
- **Feasibility**: valid input always has a feasible solution (e.g. open any single facility and assign all customers to it); there are no infeasible instances.
- **Numerics**: costs and objective values are reals; comparisons of objective values should allow for floating-point rounding error.
