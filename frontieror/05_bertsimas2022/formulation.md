# bertsimas2022 · Mathematical model

## Sets and data

Text: Let the set of securities be $I$, and the number of securities be $n$.

Formula: $$I=\{1,\ldots,n\}.$$

Text: The expected return per unit investment of each security $i$ is $\mu_i$.

Formula: $$\mu=(\mu_i)_{i\in I}\in\mathbb R^n.$$

Text: The security covariance matrix $Q$ is either given directly, or constructed from a loading matrix $B$, whose rows correspond to factors and whose columns correspond to securities, and an idiosyncratic variance vector $d$.

Formula: $$Q=Q^{\mathrm{direct}}\in\mathbb S_+^n\quad\text{or}\quad Q=B^\top B+\operatorname{Diag}(d),\qquad B\in\mathbb R^{p\times n},\ d\in\mathbb R_+^n.$$

Text: A nonnegative return weight $\lambda$ and a strictly positive concentration parameter $\theta$ are given.

Formula: $$\lambda\ge 0,\qquad \theta>0.$$

Text: The maximum number of securities allowed to be held, $K$, is given.

Formula: $$K\in\{1,\ldots,n\}.$$

Text: The index set of the optional linear rules is $R$; rule $r$ has coefficients $A_{ri}$, lower bound $\ell_r$, and upper bound $u_r$; if there are no rules, let $R=\varnothing$.

Formula: $$R=\{1,\ldots,m\},\qquad \ell_r\le u_r\quad(\forall r\in R),\qquad m\ge0.$$

Text: The minimum and maximum investment proportions of security $i$ once enabled are $m_i$ and $M_i$, respectively; if thresholds are not provided, they are taken as zero and one, respectively.

Formula: $$m_i=\begin{cases}m_i^{\mathrm{given}},&\text{if a minimum threshold is provided},\\0,&\text{otherwise},\end{cases}\qquad M_i=\begin{cases}M_i^{\mathrm{given}},&\text{if a maximum threshold is provided},\\1,&\text{otherwise}.\end{cases}$$

## Decision variables

Text: $x_i$ denotes the nonnegative proportion of total capital invested in security $i$.

Formula: $$x_i\in\mathbb R_+,\\qquad \forall i\in I.$$

Text: $z_i$ is the binary enabling variable of security $i$; any positive investment proportion requires the security to be enabled.

Formula: $$z_i\in\{0,1\},\qquad \forall i\in I.$$

## Objective

Text: Minimize the sum of half the portfolio variance and the concentration penalty, minus the return weight times the expected portfolio return.

Formula: $$\min_{x,z}\ \frac12\sum_{i\in I}\sum_{j\in I}Q_{ij}x_ix_j+\frac{1}{2\theta}\sum_{i\in I}x_i^2-\lambda\sum_{i\in I}\mu_i x_i.$$

## Constraints

**C1**  Text: All capital must be invested exactly.

Formula: $$\sum_{i\in I}x_i=1.$$

**C2**  Text: The investment proportion of an enabled security must not be lower than its minimum threshold.

Formula: $$x_i\ge m_i z_i,\qquad \forall i\in I.$$

**C3**  Text: The investment proportion of a security that is not enabled must be zero, and the investment proportion of an enabled security must not exceed its maximum threshold.

Formula: $$x_i\le M_i z_i,\qquad \forall i\in I.$$

**C4**  Text: The number of enabled securities must not exceed the given limit; hence the number of securities with a positive investment proportion must not exceed this limit either.

Formula: $$\sum_{i\in I}z_i\le K.$$

**C5**  Text: For each given linear rule, the weighted sum of security weights must not be lower than the rule's lower bound.

Formula: $$\sum_{i\in I}A_{ri}x_i\ge \ell_r,\qquad \forall r\in R.$$

**C6**  Text: For each given linear rule, the weighted sum of security weights must not exceed the rule's upper bound.

Formula: $$\sum_{i\in I}A_{ri}x_i\le u_r,\qquad \forall r\in R.$$
