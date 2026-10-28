# -*- coding: utf-8 -*-
"""
Stage-4 checker: workload_balancing_nsswd (equivalent to identical parallel machine load balancing / NSSWD)

Single source of truth = the authoritative block of MAT_s4.md. The description's == CONSTRAINTS == has only two entries:

  C1 [hard] Each job must be assigned to exactly one machine.
            sum_{i=0}^{num_machines-1} x[i][j] = 1, \\forall j \\in {0,...,num_jobs-1}
  C2 [hard] The completion time of each machine is the sum of the processing times
            of all jobs assigned to it.
            C[i] = sum_{j=0}^{num_jobs-1} p[j] * x[i][j], \\forall i \\in {0,...,num_machines-1}

Decision variable domains (DECISION VARIABLES block, authoritative):
  x[i][j] ∈ {0,1}, i ∈ {0,...,num_machines-1}, j ∈ {0,...,num_jobs-1}
  C[i] ≥ 0, continuous (derived from integer processing times)
  → Variable domains are not separate constraint entries, so they are merged into C1 (0/1 and index domain of x) and C2 (C[i] ≥ 0 and index domain) respectively.

The only required fields are ['assignment', 'machine_completion_times']; everything else (objective / feasible / status ...)
is optional redundant information: missing ones are not violations, and **no derivable value is trusted** (the objective is only recomputed, never compared).

Violation message prefix convention:
  "C1 | ..."     → description constraint 1 (including 0/1 and index domain of x)
  "C2 | ..."     → description constraint 2 (including C[i] ≥ 0 and index domain)
  "FORMAT | ..." → structure/type/completeness issues (needed to read the solution safely; not semantic constraints)
"""

import math
import re

# ---------------------------------------------------------------------------
# Numeric comparison: classify first, then decide whether to use a tolerance
#   · integers, sums of integers, values taken directly from the instance without computation → exactly representable → exact comparison (no tolerance)
#   · quantities obtained by continuous floating-point accumulation                        → relative tolerance (nominal 1e-9)
# ---------------------------------------------------------------------------
_REL = 1e-9


def _leq(a, b):
    return a - b <= _REL * max(1.0, abs(a), abs(b))


def _geq(a, b):
    return b - a <= _REL * max(1.0, abs(a), abs(b))


def _integral(x):
    """Whether x lies on an "exactly representable" integer value."""
    try:
        return float(x).is_integer()
    except (TypeError, ValueError, OverflowError):
        return False


def _num_equal(a, b):
    """
    Equality comparison for C2.
    When both a and b are integer-valued → exact comparison (sums of integers are exactly representable; a tolerance would let real overruns pass);
    if either side is a true fraction → it went through floating-point accumulation, so use a relative tolerance.
    """
    if a == b:
        return True
    if _integral(a) and _integral(b):
        return False
    return _leq(a, b) and _geq(a, b)


# ---------------------------------------------------------------------------
# Lenient but unambiguous parsing helpers
#   Key name format (leading zeros / case / underscores / whitespace / key order) is never specified by the description;
#   anything that uniquely resolves to the correct index must be accepted; only fail when no unique index can be resolved.
# ---------------------------------------------------------------------------
_KEY_RE = re.compile(r"[^0-9]*?([0-9]+)\s*")


def _to_index(k):
    """Parse a key / element into an integer index; return None if it cannot be resolved uniquely."""
    if isinstance(k, bool):
        return None
    if isinstance(k, int):
        return k
    if isinstance(k, float):
        return int(k) if k.is_integer() else None
    if isinstance(k, str):
        s = k.strip()
        if not s:
            return None
        try:
            return int(s)                      # "0" "00" " 3 " "+2" "-1"
        except ValueError:
            pass
        try:
            f = float(s)                       # "0.0" "3.0" "1e1"
            return int(f) if f.is_integer() else None
        except ValueError:
            pass
        m = _KEY_RE.fullmatch(s)               # "machine_01" "M0" "<machine in Chinese> 2"
        if m:
            return int(m.group(1))
        return None
    return None


def _to_num(x):
    """Parse a value into float; return None if unparseable / NaN / Inf."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        f = float(x)
    elif isinstance(x, str):
        try:
            f = float(x.strip())
        except ValueError:
            return None
    else:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _entries(obj):
    """
    Normalize dict / list forms into [(original key, index or None, value)].
    dict is the form required by the output format; the list form (k-th element = machine k) also resolves uniquely to indices,
    so it is accepted as well -- this is leniency, not a new rule.
    """
    if isinstance(obj, dict):
        return [(k, _to_index(k), val) for k, val in obj.items()]
    if isinstance(obj, (list, tuple)):
        return [(i, i, val) for i, val in enumerate(obj)]
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def check_solution(instance, solution):
    """Return a list of violation strings; empty list = feasible. Never raises under any circumstances."""
    violations = []
    try:
        _check(instance, solution, violations)
    except Exception as exc:                                  # Fallback: never propagate
        violations.append(
            "FORMAT | unexpected exception during checking (treated as infeasible): %s: %s"
            % (type(exc).__name__, exc)
        )
    return violations


def _check(instance, solution, v):
    # ======================================================================
    # 0. Read the instance (only check whether it is readable; do not impose plausibility thresholds the description does not specify:
    #    no requirement of N>0 / M>0 / p[j]≥0 -- the instance is given, not under review)
    # ======================================================================
    if not isinstance(instance, dict):
        v.append("FORMAT | instance is not a dict; cannot read num_machines/num_jobs/processing_times, cannot judge")
        return

    m_idx = _to_index(instance.get("num_machines"))
    if m_idx is None:
        v.append("FORMAT | instance.num_machines is missing or not parseable as an integer; cannot determine the machine index set {0,...,num_machines-1}")
        return
    num_machines = m_idx

    p_raw = instance.get("processing_times")
    p = []
    p_ok = isinstance(p_raw, (list, tuple))
    if p_ok:
        for x in p_raw:
            p.append(_to_num(x))               # Unparseable elements are set to None and reported when used
    else:
        v.append("FORMAT | instance.processing_times is missing or not a list; cannot recompute C[i] per C2")

    n_idx = _to_index(instance.get("num_jobs"))
    num_jobs = n_idx if n_idx is not None else len(p)   # If num_jobs is missing, use the length of processing_times as the job axis

    # ======================================================================
    # 1. Read the solution structure (format validation: the foundation needed to "read the solution safely")
    # ======================================================================
    if solution is None:
        v.append("FORMAT | solution is None; no solution provided")
        return
    if not isinstance(solution, dict):
        v.append("FORMAT | solution is not a dict (actual type %s); cannot read required fields assignment / machine_completion_times"
                 % type(solution).__name__)
        return

    missing = [k for k in ("assignment", "machine_completion_times") if k not in solution]
    if missing:
        v.append("FORMAT | missing required field %s (required fields are ['assignment','machine_completion_times'])" % missing)

    asg_entries = _entries(solution.get("assignment")) if "assignment" in solution else None
    if "assignment" in solution and asg_entries is None:
        v.append("FORMAT | assignment is neither dict[str,list[int]] nor list; cannot parse the machine→job assignment")

    mct_entries = _entries(solution.get("machine_completion_times")) if "machine_completion_times" in solution else None
    if "machine_completion_times" in solution and mct_entries is None:
        v.append("FORMAT | machine_completion_times is neither dict[str,float] nor list; cannot read C[i]")

    if asg_entries is None and mct_entries is None:
        return                                  # Neither required field could be read; subsequent semantic checks are impossible

    # -- assignment: keys parsed into machine indices, values into lists of job indices -------------------
    asg = {}            # machine index -> [job indices (with duplicates)]
    if asg_entries is not None:
        seen_keys = {}
        for raw_key, mid, val in asg_entries:
            if mid is None:
                v.append("FORMAT | assignment key %r cannot be uniquely resolved to a machine index" % (raw_key,))
                continue
            if mid in seen_keys:
                v.append("FORMAT | assignment keys %r and %r both resolve to machine index %d; ownership is ambiguous"
                         % (raw_key, seen_keys[mid], mid))
                continue
            seen_keys[mid] = raw_key

            if val is None:                     # null is treated as an empty list (unambiguous leniency)
                items = []
            elif isinstance(val, (list, tuple)):
                items = list(val)
            else:
                v.append("FORMAT | assignment[%r] is not a list of job indices (actual type %s)"
                         % (raw_key, type(val).__name__))
                continue

            jobs = []
            bad = False
            for it in items:
                j = _to_index(it)
                if j is None:
                    v.append("FORMAT | assignment[%r]: element %r cannot be uniquely resolved to a job index" % (raw_key, it))
                    bad = True
                    continue
                jobs.append(j)
            asg[mid] = jobs

    # -- machine_completion_times: keys parsed into machine indices, values into numbers -------------
    mct = {}
    if mct_entries is not None:
        seen_keys = {}
        for raw_key, mid, val in mct_entries:
            if mid is None:
                v.append("FORMAT | machine_completion_times key %r cannot be uniquely resolved to a machine index" % (raw_key,))
                continue
            if mid in seen_keys:
                v.append("FORMAT | machine_completion_times keys %r and %r both resolve to machine index %d; ownership is ambiguous"
                         % (raw_key, seen_keys[mid], mid))
                continue
            seen_keys[mid] = raw_key
            c = _to_num(val)
            if c is None:
                v.append("FORMAT | machine_completion_times[%r] = %r is not a parseable finite number" % (raw_key, val))
                continue
            mct[mid] = c

    # ======================================================================
    # C1  [hard] Each job must be assigned to exactly one machine.
    #     sum_{i} x[i][j] = 1, ∀ j ∈ {0,...,num_jobs-1}
    #     Meaning: the total number of occurrences of each job across all machines is exactly 1 (none missing, none duplicated).
    #     Solution representation: x[i][j]=1 ⟺ j ∈ assignment[i]; so the multiset of all lists concatenated
    #             must equal exactly {0,...,num_jobs-1}, each element exactly once.
    #     Also variable domains: x[i][j] ∈ {0,1} → the same job must not be listed twice in one machine's list;
    #                 i ∈ {0,...,M-1}, j ∈ {0,...,N-1} → indices must be within the domain.
    # ======================================================================
    occ = {}            # job index -> [machine indices where it appears (with duplicates)]
    for mid in sorted(asg.keys()):
        jobs = asg[mid]

        # Variable domain: machine index must be in {0,...,num_machines-1}
        if not (0 <= mid < num_machines):
            v.append("C1 | assignment contains machine index %d, outside the machine index domain {0,...,%d} (num_machines=%d)"
                     % (mid, num_machines - 1, num_machines))

        # Variable domain: x[i][j] ∈ {0,1} → the same job must not appear twice within one machine
        cnt_local = {}
        for j in jobs:
            cnt_local[j] = cnt_local.get(j, 0) + 1
        for j in sorted(k for k, c in cnt_local.items() if c > 1):
            v.append("C1 | in the job list of machine %d, job %d appears %d times, equivalent to x[%d][%d]=%d, violating x[i][j] ∈ {0,1}"
                     % (mid, j, cnt_local[j], mid, j, cnt_local[j]))

        for j in jobs:
            # Variable domain: job index must be in {0,...,num_jobs-1}
            if not (0 <= j < num_jobs):
                v.append("C1 | machine %d is assigned job index %d, outside the job index domain {0,...,%d} (num_jobs=%d)"
                         % (mid, j, num_jobs - 1, num_jobs))
                continue
            occ.setdefault(j, []).append(mid)

    # Main check: each job is assigned exactly once (counts are integers; exact comparison, no tolerance)
    if asg_entries is not None:
        for j in range(num_jobs):
            where = occ.get(j, [])
            if len(where) != 1:
                if not where:
                    v.append("C1 | job %d is not assigned to any machine (sum_i x[i][%d]=0, required =1)" % (j, j))
                else:
                    v.append("C1 | job %d is assigned %d times (on machines %s; sum_i x[i][%d]=%d, required =1)"
                             % (j, len(where), sorted(where), j, len(where)))

    # ======================================================================
    # C2  [hard] C[i] = sum_{j} p[j] * x[i][j], ∀ i ∈ {0,...,num_machines-1}
    #     Meaning: the completion time reported for each machine must exactly equal the sum of processing times of its assigned jobs;
    #           under-reporting load to make the objective look better is not allowed.
    #     Solution representation: machine_completion_times[i] vs. the sum of p over the jobs in assignment[i].
    #     Also variable domains: C[i] ≥ 0; machine index i ∈ {0,...,M-1}; and C[i] must be given for every machine.
    #     Recompute independently; do not trust any derivable value reported by the solution.
    # ======================================================================
    for mid in sorted(mct.keys()):
        if not (0 <= mid < num_machines):
            v.append("C2 | machine_completion_times contains machine index %d, outside the machine index domain {0,...,%d} (num_machines=%d)"
                     % (mid, num_machines - 1, num_machines))

    for i in range(num_machines):
        jobs = asg.get(i)

        # The output format requires "list all num_machines machines, keeping an empty list even for machines with no jobs"
        if asg_entries is not None and jobs is None:
            v.append("FORMAT | assignment does not list machine %d (the output format requires listing all %d machines, keeping an empty list for machines with no jobs)"
                     % (i, num_machines))
            jobs = []
        elif jobs is None:
            jobs = []

        # Recompute C[i]: only accumulate jobs within the index domain that actually exist in processing_times
        load = 0.0
        gap = False
        for j in jobs:
            if not (0 <= j < num_jobs):
                gap = True
                continue
            if j >= len(p) or p[j] is None:
                gap = True
                continue
            load += p[j]

        if i not in mct:
            v.append("C2 | machine %d: completion time C[%d] is not given (machine_completion_times must cover all %d machines; recomputed from the assignment it should be %s)"
                     % (i, i, num_machines, _fmt(load)))
            continue

        reported = mct[i]

        # Variable domain C[i] ≥ 0 (recomputed load is a floating-point accumulation, so decide with relative-tolerance direction)
        if not _geq(reported, 0.0):
            v.append("C2 | machine %d: completion time C[%d]=%s is negative, violating the variable domain C[i] ≥ 0" % (i, i, _fmt(reported)))

        if not _num_equal(reported, load):
            v.append("C2 | inconsistent completion time for machine %d: solution reports %s, recomputed from assignment sum_j p[j]*x[%d][j] = %s%s"
                     % (i, _fmt(reported), i, _fmt(load),
                        " (this machine has job entries whose p[j] is unavailable; the recomputed value is based only on valid entries)" if gap else ""))


def _fmt(x):
    """Number to string: integer values without a decimal tail, for easier manual review."""
    try:
        if float(x).is_integer():
            return str(int(x))
        return repr(float(x))
    except (TypeError, ValueError, OverflowError):
        return repr(x)
