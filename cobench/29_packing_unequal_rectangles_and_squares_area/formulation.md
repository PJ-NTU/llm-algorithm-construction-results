# Packing unequal rectangles and squares area: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Select a subset of the given rectangular items and decide each selected item's centre position and whether it is rotated by 90 degrees, so that the items lie entirely inside a fixed circular container and do not overlap. Rotation is allowed only when the global rotation permission is on. The objective is to maximise the total area of the packed items.
## Parameters and Sets
All geometric lengths and coordinates use the same length unit, which is unspecified; the area unit is the square of that length unit.

| Symbol | Meaning | Unit or range |
| --- | --- | --- |
| \(n=N\) | Input field `n`, number of candidate items, equal to the number of entries in `items` | \(n\in\mathbb Z_{\ge 0}\), items |
| \(I\) | Item index set \(\{0,\ldots,n-1\}\); empty when \(n=0\) | Integer indices starting at 0 |
| \(\mathrm{items}\) | Input field `items`; the \(i\)-th entry is the ordered size pair \((L_i,W_i)\) | \(n\) pairs in total |
| \(L_i\) | `items[i][0]`, side length of the item along the x-axis when unrotated | \(L_i\in\mathbb R_{>0}\), length unit unspecified |
| \(W_i\) | `items[i][1]`, side length of the item along the y-axis when unrotated | \(W_i\in\mathbb R_{>0}\), same unit as \(L_i\) |
| \(c_x,c_y\) | Input fields `cx`, `cy`, centre coordinates of the fixed circular container | Real, same unit as the side lengths; must also satisfy C6 |
| \(R\) | Input field `R`, radius of the fixed circular container | \(R\in\mathbb R_{>0}\), same unit as the side lengths; must also satisfy C6 |
| \(g\) | Confirmed global rotation permission input; 1 means rotation allowed, 0 means rotation forbidden. \(g\) is a mathematical symbol; the original problem does not specify a field name for this input | \(g\in\{0,1\}\), dimensionless |

Geometric inputs may be non-integer reals; apart from the conditions above and C6, no further numerical bounds are imposed.
## Decision Variables
The following variables are defined for each \(i\in I\), and only these four kinds of decision variables specified by the problem are used.

| Variable | Meaning | Range |
| --- | --- | --- |
| \(\alpha[i]\) | Whether item \(i\) is selected and packed; 1 means packed, 0 means not packed | \(\{0,1\}\) |
| \(x[i]\) | x-coordinate of the centre of item \(i\); takes the prescribed placeholder value when not packed | \(\mathbb R\), same unit as the input lengths |
| \(y[i]\) | y-coordinate of the centre of item \(i\); takes the prescribed placeholder value when not packed | \(\mathbb R\), same unit as the input lengths |
| \(r[i]\) | Whether item \(i\) is rotated by 90 degrees; 1 means rotated, 0 means not rotated | \(\{0,1\}\) |

Below, \(w_i^{\mathrm{eff}}\) and \(h_i^{\mathrm{eff}}\) are only shorthand for size expressions, not additional decision variables. Item edges are always parallel to the coordinate axes; no other rotation angles are allowed.
## Objective Function
Maximise the total area of all items selected and packed into the container. This is the only optimisation objective; there are no secondary objectives or extra weights.
\[
\max_{\alpha,x,y,r}\ \sum_{i=0}^{n-1} L_iW_i\alpha[i].
\]
## Constraints
There are 6 numbered constraints, `n_constraints = 6`; all are hard constraints, and C6 is a condition that every valid input must satisfy.

### C1 Effective dimensions
Hard constraint: when unrotated, the effective width and height are the original length and width respectively; after a 90-degree rotation the two are swapped.
\[
\begin{aligned}
w_i^{\mathrm{eff}}&\equiv(1-r[i])L_i+r[i]W_i,\\
h_i^{\mathrm{eff}}&\equiv(1-r[i])W_i+r[i]L_i,
\end{aligned}
\qquad \forall i\in I.
\]
### C2 Containment in the circular container
Hard constraint: all four corners of each packed item must lie inside the circle or on its circumference, i.e. the distance from each corner to the centre does not exceed the radius.
\[
\alpha[i]=1\ \Longrightarrow\
\left(x[i]+\frac{s\,w_i^{\mathrm{eff}}}{2}-c_x\right)^2+
\left(y[i]+\frac{t\,h_i^{\mathrm{eff}}}{2}-c_y\right)^2
\le R^2,
\qquad
\forall i\in I,\ \forall(s,t)\in\{-1,1\}^2.
\]
### C3 No overlap between items
Hard constraint: any two distinct packed items must be separated at least up to boundary contact along the x-axis or the y-axis; no interior overlap is allowed.
\[
\alpha[i]\alpha[j]=1\ \Longrightarrow\
\left(
|x[i]-x[j]|\ge\frac{w_i^{\mathrm{eff}}+w_j^{\mathrm{eff}}}{2}
\right)
\ \lor\
\left(
|y[i]-y[j]|\ge\frac{h_i^{\mathrm{eff}}+h_j^{\mathrm{eff}}}{2}
\right),
\qquad \forall i,j\in I,\ i<j.
\]
### C4 Fixed state of unpacked items
Hard constraint: an unpacked item must use the fixed placeholder centre \((-1,-1)\) and must not be marked as rotated.
\[
\alpha[i]=0\ \Longrightarrow\
\bigl(x[i]=-1\ \land\ y[i]=-1\ \land\ r[i]=0\bigr),
\qquad \forall i\in I.
\]
### C5 Global rotation permission
Hard constraint: when the global permission is off, no item may be rotated; when it is on, each selected item may choose whether to rotate, and unpacked items still obey C4.
\[
r[i]\le g,\qquad \forall i\in I.
\]
### C6 Fixed placeholder point strictly outside the circle
Hard constraint (input validity): every valid input must guarantee that the fixed placeholder point \((-1,-1)\) lies strictly outside the circular container; a placeholder point on the circumference does not qualify.
\[
(-1-c_x)^2+(-1-c_y)^2>R^2.
\]
## Solution and Boundary Conventions
- A complete solution must give \(\alpha[i],x[i],y[i],r[i]\) for every index \(i\in I\); unpacked items must also be given the state prescribed by C4, and their variable assignments may not be omitted.
- Any subset may be selected, including packing no items. For a valid input, packing nothing and assigning values per C4 is a feasible solution with zero area; it is not necessarily optimal. Items that cannot fit even alone may be left unselected, and this does not make the input invalid.
- \(n=0\) with empty `items` is a valid empty input; all four kinds of variables in its complete solution are empty and the total packed area is zero; all input conditions must still hold, including C6. \(n=1\) is also valid, in which case there are no C3 constraints between item pairs.
- Lengths, widths and the radius must all be strictly positive; zero-size items or a zero-radius container are not accepted.
- Equality is allowed in C2 and C3, so items may touch the container circumference and may touch each other at edges or corners; no extra gap is required. C6 uses a strict inequality, unlike the contact rules above.
- Unpacked items are not subject to the containment condition of C2; as long as one item of a pair is unpacked, that pair is not subject to the separation condition of C3. The placeholder state of C4 must still hold.
- No constraint may be relaxed via penalties. If several feasible solutions have the same maximum total packed area, all of them are optimal; no tie-breaking rule is defined.
