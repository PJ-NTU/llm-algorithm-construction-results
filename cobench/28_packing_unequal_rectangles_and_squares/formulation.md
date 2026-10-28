# Packing unequal rectangles and squares: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Select a subset of the given rectangles and decide, for each rectangle, whether it is rotated by 90 degrees and where its centre lies, so that the selected rectangles lie entirely inside a fixed circular container and do not overlap. The objective is to maximise the number of packed rectangles; the centre of each unselected rectangle must be placed at the origin as a dummy position.

## Parameters and sets
All real-valued data are finite reals. Widths, heights, the radius and coordinates use the same length unit; the specific unit is unspecified.

| Symbol | Meaning | Unit or range |
| --- | --- | --- |
| \(N\) | Input field `num_rectangles`, the number of rectangles | count; \(N\in\mathbb{Z}_{\ge 0}\) |
| \(I\) | Set of rectangle indices, all 0-based | \(I=\{0,\ldots,N-1\}\); \(\varnothing\) when \(N=0\) |
| \(w_i\) | Item \(i\) of input field `widths`, the rectangle's width along the x-axis in its original orientation | length; \(w_i\in\mathbb{R}_{>0}\), \(i\in I\) |
| \(h_i\) | Item \(i\) of input field `heights`, the rectangle's height along the y-axis in its original orientation | length; \(h_i\in\mathbb{R}_{>0}\), \(i\in I\) |
| \(R\) | Input field `container_radius`, the radius of the fixed circular container | length; \(R\in\mathbb{R}_{>0}\) |
| \(O\) | Centre of the fixed circular container | length coordinates; \(O=(0,0)\) |
| \(M\) | Input field `big_m`, the given constant in the original non-overlap disjunction | unit unspecified; \(M\in\mathbb{R}\), zero, negative and non-integer values allowed |

`widths` and `heights` each have exactly \(N\) items, and the two items with the same index belong to the same rectangle; dimensions and radius need not be integers.

## Decision variables
Only the following four kinds of decision variables are used; no other decision variables are introduced.

| Symbol | Meaning | Range |
| --- | --- | --- |
| \(\alpha_i\) (`alpha[i]`) | Whether rectangle \(i\) is packed: 1 means packed, 0 means not packed | \(\alpha_i\in\{0,1\}\), \(i\in I\) |
| \(x_i\) (`x[i]`) | x-coordinate of the centre of rectangle \(i\) | \(x_i\in\mathbb{R}\), length unit, \(i\in I\) |
| \(y_i\) (`y[i]`) | y-coordinate of the centre of rectangle \(i\) | \(y_i\in\mathbb{R}\), length unit, \(i\in I\) |
| \(r_i\) (`r[i]`) | Rotation flag: 0 for the original orientation, 1 for rotated by 90 degrees | \(r_i\in\{0,1\}\), \(i\in I\) |

Rotation only swaps the rectangle's side lengths along the x- and y-axes; no other angles are allowed. For a given orientation, the horizontal side length is \((1-r_i)w_i+r_i h_i\) and the vertical side length is \((1-r_i)h_i+r_i w_i\); these are expressions, not additional decision variables.

## Objective function
Maximise the total number of rectangles successfully packed into the circular container; there is no area objective or other secondary objective.
\[
\max_{\alpha,x,y,r}\;\sum_{i=0}^{N-1}\alpha_i.
\]

## Constraints
There are 3 numbered constraints (`n_constraints = 3`), all hard constraints.

### C1: Selected rectangles lie entirely inside the circle
Hard constraint: for every selected rectangle, the distance from each of its four corners to the origin must not exceed \(R\); the four corner conditions must hold simultaneously.
\[
\begin{aligned}
\alpha_i\left[\left(x_i+\frac{(1-r_i)w_i}{2}+\frac{r_i h_i}{2}\right)^2
+\left(y_i+\frac{(1-r_i)h_i}{2}+\frac{r_i w_i}{2}\right)^2\right]&\le R^2,
&&\forall i\in I,\\
\alpha_i\left[\left(x_i-\frac{(1-r_i)w_i}{2}-\frac{r_i h_i}{2}\right)^2
+\left(y_i+\frac{(1-r_i)h_i}{2}+\frac{r_i w_i}{2}\right)^2\right]&\le R^2,
&&\forall i\in I,\\
\alpha_i\left[\left(x_i+\frac{(1-r_i)w_i}{2}+\frac{r_i h_i}{2}\right)^2
+\left(y_i-\frac{(1-r_i)h_i}{2}-\frac{r_i w_i}{2}\right)^2\right]&\le R^2,
&&\forall i\in I,\\
\alpha_i\left[\left(x_i-\frac{(1-r_i)w_i}{2}-\frac{r_i h_i}{2}\right)^2
+\left(y_i-\frac{(1-r_i)h_i}{2}-\frac{r_i w_i}{2}\right)^2\right]&\le R^2,
&&\forall i\in I.
\end{aligned}
\]

### C2: Selected rectangles must not overlap
Hard constraint: for every pair of distinct rectangles, at least one of the following three branches holds. When both are selected, they must be separated along the x-axis or the y-axis; when they are not both selected, the first branch satisfies the constraint.
\[
\begin{gathered}
\left(\alpha_i+\alpha_j\le 1\right)\\
\lor\\
\left(
|x_i-x_j|\ge
\frac{(1-r_i)w_i+r_i h_i+(1-r_j)w_j+r_j h_j}{2}
-M(2-\alpha_i-\alpha_j)
\right)\\
\lor\\
\left(
|y_i-y_j|\ge
\frac{(1-r_i)h_i+r_i w_i+(1-r_j)h_j+r_j w_j}{2}
-M(2-\alpha_i-\alpha_j)
\right),\\
\forall i,j\in I,\quad i<j.
\end{gathered}
\]
The full disjunction from the problem statement is kept. When both are selected the coefficient of \(M\) is zero, and when they are not both selected the first branch holds, so any finite real \(M\) leaves the feasibility meaning of the constraint unchanged; no "sufficiently large" or non-negativity requirement is imposed.

### C3: Unselected rectangles are placed at the dummy position
Hard constraint: both coordinates of an unselected rectangle must be zero; this constraint does not restrict the coordinates of selected rectangles.
\[
\begin{aligned}
(1-\alpha_i)x_i&=0,&&\forall i\in I,\\
(1-\alpha_i)y_i&=0,&&\forall i\in I.
\end{aligned}
\]

## Solutions and boundary conventions
- A complete feasible solution must give \((\alpha_i,x_i,y_i,r_i)\) for every \(i\in I\), satisfying the variable ranges and C1-C3; an optimal solution maximises the number of selected rectangles among all feasible solutions.
- Packing only a subset is allowed; not all rectangles need to fit. Unselected rectangles must still provide values for all four variables, with their centre fixed at \((0,0)\); their rotation flag may be 0 or 1 and is not otherwise fixed.
- Unselected rectangles are not subject to the containment and non-overlap requirements of actual packing; their dummy positions may coincide with one another or with the region occupied by packed rectangles. A selected rectangle may also be centred at the origin; whether it is packed is determined only by \(\alpha_i\).
- The non-strict inequalities allow rectangles to touch the circle and allow the boundaries of two packed rectangles to touch, but interiors must not overlap.
- \(N=0\) is valid input; then the width and height lists and the four variable arrays are empty, the complete solution is empty, and the optimal objective value is 0; the radius must still be strictly positive and \(M\) must still be a finite real.
- \(N=1\) is valid input; then there are no rectangle pairs subject to C2, and the other requirements are unchanged.
- It is not required that every rectangle fits into the container on its own. Rectangles that cannot fit may be left unselected; selecting no rectangle with all centres at the origin is always a feasible solution.
- Zero or negative widths, heights or radius are not allowed; degenerate rectangles or degenerate circles are not accepted.
- When several solutions reach the same maximum packed count, any of them is an acceptable optimal solution; no tie-breaking rule on area, position, rotation or index is imposed.
