# pinnoi1997 · Mathematical model

## Sets and parameters

- Text: The task set consists of all tasks to be assigned.
  Formula: \(T=\{1,\ldots,n\}\)
- Text: The station set consists of a fixed number \(m\) of stations numbered in line order.
  Formula: \(S=\{1,\ldots,m\}\)
- Text: Each task has a given integer processing time.
  Formula: \(p_i\in\mathbb Z_{>0},\quad \forall i\in T\)
- Text: The maximum allowed load of each station is the given cycle time.
  Formula: \(C\in\mathbb Z_{>0}\)
- Text: The given number of stations \(m\) is the minimum feasible number of stations for the line and is kept fixed in this model.
  Formula: \(m\in\mathbb Z_{>0}\)
- Text: An arc \((i,j)\) in the immediate precedence relation set means that task \(i\) is an immediate predecessor of task \(j\).
  Formula: \(A\subseteq T\times T\)
- Text: \(P_i\) and \(Q_i\) denote, respectively, all transitive predecessors and all transitive successors of task \(i\) in the precedence graph.
  Formula: \(P_i=\{j\in T:j\leadsto i\},\quad Q_i=\{j\in T:i\leadsto j\}\)
- Text: The earliest station of task \(i\) is obtained by dividing the cumulative processing time of the task itself and all its predecessors by the cycle time and rounding up.
  Formula: \(E_i=\left\lceil\dfrac{p_i+\sum_{j\in P_i}p_j}{C}\right\rceil\)
- Text: The latest station of task \(i\) is obtained by subtracting, from the fixed number of stations, the rounded-up number of cycles corresponding to the cumulative processing time of the task itself and all its successors, and then adding one.
  Formula: \(L_i=m-\left\lceil\dfrac{p_i+\sum_{j\in Q_i}p_j}{C}\right\rceil+1\)
- Text: Task \(i\) can only be assigned to a station between its earliest station and its latest station.
  Formula: \(S_i=\{s\in S:E_i\le s\le L_i\}\)
- Text: The given lower bound on the worst idle time equals the total slack divided by the cycle time, rounded down.
  Formula: \(B=\left\lfloor\dfrac{mC-\sum_{i\in T}p_i}{C}\right\rfloor\)

## Decision variables

- Text: Binary variable \(x_{is}\) indicates whether task \(i\) is assigned to its feasible station \(s\).
  Formula: \(x_{is}\in\{0,1\},\quad \forall i\in T,\ s\in S_i\)
- Text: Nonnegative integer variable \(z\) denotes the maximum station idle time over the whole line.
  Formula: \(z\in\mathbb Z_{\ge 0}\)

## Objective

- Text: Minimize the maximum station idle time over the whole line, so that the workloads of the stations are as balanced as possible.
  Formula: \(\min z\)

## Constraints

- **C1** Text: Each task must be assigned exactly once to a station between its earliest and latest feasible stations.
  Formula: \(\displaystyle \sum_{s\in S_i}x_{is}=1,\quad \forall i\in T\)
- **C2** Text: For each immediate precedence arc, the index of the station of the predecessor task must not exceed the index of the station of the successor task.
  Formula: \(\displaystyle \sum_{s\in S_i}s\,x_{is}\le \sum_{s\in S_j}s\,x_{js},\quad \forall(i,j)\in A\)
- **C3** Text: The total processing time of all tasks assigned to each station must not exceed the cycle time.
  Formula: \(\displaystyle \sum_{i\in T:\,s\in S_i}p_i x_{is}\le C,\quad \forall s\in S\)
- **C4** Text: The worst idle time must be no less than the idle time of each station, i.e., the cycle time minus the total processing time at that station.
  Formula: \(\displaystyle \sum_{i\in T:\,s\in S_i}p_i x_{is}+z\ge C,\quad \forall s\in S\)
- **C5** Text: The worst idle time must not be lower than the total-slack lower bound given by the problem.
  Formula: \(\displaystyle z\ge B=\left\lfloor\dfrac{mC-\sum_{i\in T}p_i}{C}\right\rfloor\)
