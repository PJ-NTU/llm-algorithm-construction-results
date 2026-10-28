# Container loading with weight restrictions: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Decide which boxes of each box type are loaded into the container, and the position and allowed orientation of each loaded box. Subject to the hard constraints of boundaries, quantities, full single-box support, non-overlap and layer-by-layer cumulative load bearing, maximise volume utilisation.

## Parameters and Sets
Each instance uses one self-consistent set of units. Denote the length unit by \(U_\ell\) and the weight unit by \(U_w\); the physical units are not fixed. All edge lengths and coordinates use \(U_\ell\), and load-bearing limits use \(U_w/U_\ell^2\).

| Symbol | Meaning | Unit or range |
| --- | --- | --- |
| \(L\) | `container_length`, length of the container along the x axis | \(U_\ell\), strictly positive real |
| \(W\) | `container_width`, width of the container along the y axis | \(U_\ell\), strictly positive real |
| \(H\) | `container_height`, height of the container along the z axis | \(U_\ell\), strictly positive real |
| \(N\) | `num_box_types`, number of box types | non-negative integer, zero allowed |
| \(\mathcal I=\{0,\ldots,N-1\}\) | set of box-type indices | empty when \(N=0\) |
| \(q_i\) | `box_quantities[i]`, quantity limit of box type i | non-negative integer, zero allowed |
| \(\mathcal K_i=\{0,\ldots,q_i-1\}\) | set of candidate box indices of box type i | empty when \(q_i=0\) |
| \(\mathcal B=\{(i,k):i\in\mathcal I,\ k\in\mathcal K_i\}\) | set of all candidate boxes | finite set |
| \(i(b)\) | box type of candidate box \(b=(i,k)\) | \(i(b)=i\) |
| \(\ell_i\) | `box_lengths[i]`, original length | \(U_\ell\), strictly positive real |
| \(w_i\) | `box_widths[i]`, original width | \(U_\ell\), strictly positive real |
| \(h_i\) | `box_heights[i]`, original height | \(U_\ell\), strictly positive real |
| \(m_i\) | `box_weights[i]`, weight of one box | \(U_w\), strictly positive real |
| \(\mathcal O=\{1,2,3\}\) | set of orientation codes usable by loaded boxes | meanings in the table below |
| \(a_{i,o}\) | `orientation_allowed[i][o-1]`, whether box type i allows orientation o | \(\{0,1\}\) |
| \(\lambda_{i,o}\) | `load_bearing_by_orientation[i][o-1]`, per-unit-area load-bearing limit of box type i in orientation o | \(U_w/U_\ell^2\), non-negative real, zero allowed |

Box-type arrays are indexed by i; orientation data are indexed by box type and orientation, each row having exactly three components. Box-type and box indices start at 0; orientation codes are 1, 2, 3, corresponding to array indices 0, 1, 2. Real-valued data may be fractional; no other rounding requirement is imposed.

## Decision Variables
For each \(b=(i,k)\in\mathcal B\), use the following and only the following decision variables; the subscript b abbreviates the original index \([i][k]\).

| Symbol | Meaning | Range |
| --- | --- | --- |
| \(s_b=s[i][k]\) | whether the box is loaded | \(\{0,1\}\) |
| \(x_b=x[i][k]\) | coordinate of the front-bottom-left corner along the container length | non-negative continuous real, unit \(U_\ell\) |
| \(y_b=y[i][k]\) | coordinate of the front-bottom-left corner along the container width | non-negative continuous real, unit \(U_\ell\) |
| \(z_b=z[i][k]\) | coordinate of the front-bottom-left corner vertically upward | non-negative continuous real, unit \(U_\ell\) |
| \(o_b=o[i][k]\) | orientation code; 0 when not loaded | \(\{0,1,2,3\}\), correspondence with selection status given by C1 |

Define the derived set of loaded boxes:
\[
\mathcal S=\{b\in\mathcal B:s_b=1\}.
\]
For \(b\in\mathcal S\), the occupied dimensions \((d_b^x,d_b^y,d_b^z)\) are uniquely determined by box type and orientation and are not additional decision variables:
\[
(d_b^x,d_b^y,d_b^z)=
\begin{cases}
(w_{i(b)},h_{i(b)},\ell_{i(b)}),&o_b=1,\\
(\ell_{i(b)},h_{i(b)},w_{i(b)}),&o_b=2,\\
(\ell_{i(b)},w_{i(b)},h_{i(b)}),&o_b=3.
\end{cases}
\]

| Orientation | Original edge pointing vertically up | Occupied size in x | Occupied size in y | Occupied size in z |
| --- | --- | --- | --- | --- |
| 1 | original length | original width | original height | original length |
| 2 | original width | original length | original height | original width |
| 3 | original height | original length | original width | original height |

This mapping fixes the occupied sizes in all three directions; swapping the two horizontal edges or any other rotation is not allowed. Unloaded boxes need no occupied sizes. The support sets and upper-descendant sets below are likewise only derived definitions from the decision variables above; no independent support or load decision variables are introduced.

## Objective Function
Maximise the total original volume of loaded boxes as a fraction of the container volume, i.e. volume utilisation; this is the only optimisation objective.
\[
\max\quad
\frac{\displaystyle\sum_{i\in\mathcal I}\sum_{k\in\mathcal K_i}
 s[i][k]\,\ell_i w_i h_i}{LWH}.
\]
No secondary objectives such as number of boxes loaded, weight or stacking height are added.

## Constraints
There are 7 numbered constraints, n_constraints = 7, all hard.

### C1 Selection status consistent with orientation
Hard constraint: an unloaded box must have orientation 0; a loaded box must have orientation 1, 2 or 3.
\[
\forall b\in\mathcal B:\quad
(s_b=0\ \Longrightarrow\ o_b=0)
\ \land\
(s_b=1\ \Longrightarrow\ o_b\in\{1,2,3\}).
\]

### C2 Orientation permission
Hard constraint: each loaded box must use an orientation allowed by its box type.
\[
\forall b\in\mathcal S:\quad a_{i(b),o_b}=1.
\]

### C3 Container boundaries
Hard constraint: loaded boxes must lie entirely inside the container and may touch the container boundaries exactly.
\[
\forall b\in\mathcal S:\quad
0\le x_b,\quad x_b+d_b^x\le L,\qquad
0\le y_b,\quad y_b+d_b^y\le W,\qquad
0\le z_b,\quad z_b+d_b^z\le H.
\]

### C4 Quantity limits
Hard constraint: the number of boxes actually loaded of each box type must not exceed that type's quantity limit.
\[
\forall i\in\mathcal I:\quad
\sum_{k\in\mathcal K_i}s[i][k]\le q_i.
\]

### C5 Unique full single-box support
Hard constraint: a box whose base is off the floor must be fully supported by exactly one loaded box directly below it; overhang is not allowed, nor is composite support by several boxes.
\[
\mathcal P_b=
\left\{
 a\in\mathcal S\setminus\{b\}:\begin{array}{l}
 z_a+d_a^z=z_b,\\
 x_a\le x_b,\quad x_b+d_b^x\le x_a+d_a^x,\\
 y_a\le y_b,\quad y_b+d_b^y\le y_a+d_a^y
 \end{array}
\right\},\qquad b\in\mathcal S,
\]
\[
\forall b\in\mathcal S:\quad z_b>0\ \Longrightarrow\ |\mathcal P_b|=1.
\]
Here \(\mathcal P_b\) is the geometrically determined set of full direct supporting boxes. Boxes with \(z_b=0\) are supported by the floor and need no supporting box. One supporting box may carry several boxes, but each upper box must individually satisfy this and all other constraints.

### C6 Non-overlap
Hard constraint: the interiors of any two loaded boxes must not intersect; boundary contact does not count as volume intersection.
\[
\begin{aligned}
\forall b,c\in\mathcal S,\ b\ne c:\quad
 &(x_b+d_b^x\le x_c)
 \ \lor\ (x_c+d_c^x\le x_b)\\
 &\lor\ (y_b+d_b^y\le y_c)
 \ \lor\ (y_c+d_c^y\le y_b)\\
 &\lor\ (z_b+d_b^z\le z_c)
 \ \lor\ (z_c+d_c^z\le z_b).
\end{aligned}
\]

### C7 Layer-by-layer cumulative load bearing
Hard constraint: each loaded box bears the total self-weight of all boxes above it transmitted along support relations, excluding its own weight; this cumulative load must not exceed its top-face area times the per-unit-area load-bearing limit of its current orientation.
\[
\mathcal D_b=
\left\{
 c\in\mathcal S\setminus\{b\}:\begin{array}{l}
 \exists t\in\{1,\ldots,|\mathcal S|-1\},\
 \exists (b_0,\ldots,b_t)\in\mathcal S^{t+1},\\
 b_0=b,\quad b_t=c,\\
 \forall r\in\{1,\ldots,t\}:\ b_{r-1}\in\mathcal P_{b_r}
 \end{array}
\right\},\qquad b\in\mathcal S,
\]
\[
\forall b\in\mathcal S:\quad
\sum_{c\in\mathcal D_b}m_{i(c)}
\le d_b^x d_b^y\,\lambda_{i(b),o_b}.
\]
\(\mathcal D_b\) is the set of all boxes above supported directly or indirectly by b, including all descendants on every branch, each box counted once in the sum. Box heights are strictly positive and heights strictly increase along support relations, so no support cycles exist; the finite-path definition above covers all transmission levels. With no boxes above, the load is the empty sum, i.e. 0.

## Solutions and Boundary Conventions
- A complete solution gives values \(s[i][k],x[i][k],y[i][k],z[i][k],o[i][k]\) for every candidate box and satisfies the variable ranges and C1-C7. The derived occupied sizes, support relations and cumulative loads are not extra decision variables.
- Loading only some boxes, or none at all, is allowed; quantities are upper limits, not loading demands that must be met. Unloaded boxes must have orientation 0 and their coordinates must still be non-negative continuous reals, but they are not required to be 0 and take no part in the geometric or load-bearing constraints of loaded boxes.
- \(N=0\), a box type with quantity limit 0, and all quantity limits 0 are all valid cases; the corresponding index sets are empty and empty sums are 0. A single box type or a single candidate box is also valid.
- All container and box edge lengths and box weights are strictly positive; load-bearing limits may be 0. A loaded box in a zero-capacity orientation cannot bear any positive weight from boxes above, but may itself be loaded as long as the other constraints hold.
- Orientation permissions are 0 or 1 entry-wise; each type is not required to allow at least one orientation. If a box type allows no orientation or has no feasible placement, boxes of that type may simply not be selected.
- Whether or not candidate boxes can be loaded, selecting nothing with all coordinates 0 and all orientations 0 gives a feasible solution. When no box can be loaded, the optimal volume utilisation is 0.
- Exact fit to container boundaries, support base exactly flush with top-face boundaries, and cumulative load exactly equal to capacity are all allowed; contact and containment use the model's exact equalities or non-strict inequalities with no relaxation tolerance.
- Boxes on the floor must still satisfy C7 and bear the cumulative load above them; the problem sets no separate load-bearing limit for the floor.
- Volume utilisation is dimensionless. When several feasible solutions reach the same maximum volume utilisation, all are optimal; no tie-breaking rule is defined.
