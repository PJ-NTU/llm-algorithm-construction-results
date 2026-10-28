# -*- coding: utf-8 -*-
"""
Two-stage stochastic single-allocation hub location -- solution feasibility checker.

Source of rules: the problem description in MAT_s4.md (single source of truth). This file judges only the **solution** against the description; it does not audit instance quality.

The description's == CONSTRAINTS == has 3 entries, mapped one-to-one to 3 check functions:
  C1  sum_{k!=i} x[s][i][k] = 1 - z[i]  -> _check_c1_exactly_one_assignment
  C2  x[s][i][k] <= z[k]                -> _check_c2_assign_only_to_open_hub
  C3  sum_k z[k] >= 1                   -> _check_c3_at_least_one_hub

Solution representation -> decision variables (given by the description's "output format"):
  hub_selection[k] == 1            <=>  z[k] = 1
  scenario_allocations[s][i] == k  <=>  x[s][i][k] = 1 (k != i)
  scenario_allocations[s][i] == i  <=>  that row is all 0 ("if the node itself is a hub, it points to itself")

Numerical tolerance notes (guard against errors in both directions):
  Every quantity this checker compares is an **exactly representable discrete quantity** -- 0/1 indicators, node indices, and
  their integer counts and sums (sum z, number of assignments per node). Following "exactly representable -> exact comparison, no
  tolerance", all comparisons (==, >=, in {0,1}, index range) are exact, with no relaxation.
  The only exception is in the **parsing layer**, not the comparison layer: _to_int allows snapping float integer values such as
  0.9999999999 to integers (window _INT_SNAP=1e-6). Rationale: MILP solvers often read back binaries as 0.999999999998,
  while this problem's decision space is discrete and any two distinct legal values are >= 1 apart (0 and 1, index k and k+1);
  1e-6 << 1, so the snap window **cannot hide any real violation**; after snapping, all constraint comparisons remain exact integer comparisons.
"""

import re

# Required fields specified by the description's "output format"; all other fields (objective / num_hubs / feasible / notes ...)
# are optional, missing ones are not violations, and their self-reported values are **never trusted** (all needed quantities are recomputed).
_REQUIRED_KEYS = ("hub_selection", "scenario_allocations")

# Integer snap window for the parsing layer; see the file header. Not used in any constraint comparison.
_INT_SNAP = 1e-6

# Max number of violation details listed per constraint; the rest is summarized in one line (affects readability only, not the verdict).
_MSG_CAP = 20

_DIGITS = re.compile(r"\d+")


# --------------------------------------------------------------------------
# Violation collector
# --------------------------------------------------------------------------
class _Sink(object):
    """Collect violations grouped by constraint code, to avoid flooding output on large instances."""

    def __init__(self, cap=_MSG_CAP):
        self.cap = cap
        self.msgs = []
        self.count = {}
        self.order = []

    def add(self, code, msg):
        c = self.count.get(code, 0) + 1
        if c == 1:
            self.order.append(code)
        self.count[code] = c
        if c <= self.cap:
            self.msgs.append("%s | %s" % (code, msg))

    def finish(self):
        for code in self.order:
            c = self.count[code]
            if c > self.cap:
                self.msgs.append(
                    "%s | %d violations of this kind in total; only the first %d are listed above" % (code, c, self.cap)
                )
        return self.msgs


# --------------------------------------------------------------------------
# Lenient parsing helpers (only responsible for uniquely resolving values/keys to indices; do not invent format rules absent from the description)
# --------------------------------------------------------------------------
def _to_int(v):
    """Parse a value that "should be an integer" into int; return None if it cannot be parsed.

    Accepts int / bool / integer-valued float / strings parseable as integers (with surrounding whitespace, leading zeros, '+1').
    Floats are snapped per _INT_SNAP (see rationale in the file header).
    """
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
            return None
        r = round(v)
        return int(r) if abs(v - r) <= _INT_SNAP else None
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return None
        try:
            return int(t)
        except ValueError:
            pass
        try:
            f = float(t)
        except ValueError:
            return None
        return _to_int(f)
    return None


def _parse_index_key(key):
    """Parse a container key into an index.

    The description never specifies the key format, so "anything that uniquely resolves to the correct index must be accepted":
    '0' / '00' / ' 2 ' / 0(int) / 'scenario_1' / 's3' are all accepted;
    keys with multiple digit groups (e.g. 's1_2') cannot be resolved uniquely, return None (ignored as an irrelevant extra key).
    """
    v = _to_int(key)
    if v is not None:
        return v
    if isinstance(key, str):
        runs = _DIGITS.findall(key)
        if len(runs) == 1:
            return int(runs[0])
    return None


def _as_indexed_seq(obj, length):
    """Normalize a list/tuple or an "index-keyed dict" into a list (length `length`, missing slots are None).

    Returns (seq_or_None, kind); kind is used to produce human-readable diagnostics.
    """
    if isinstance(obj, (list, tuple)):
        return list(obj), "seq"
    if isinstance(obj, dict):
        if length is None:
            return None, "dict-unknown-length"
        out = [None] * length
        for k, v in obj.items():
            idx = _parse_index_key(k)
            if idx is not None and 0 <= idx < length:
                out[idx] = v
        return out, "dict"
    return None, "bad-type"


# --------------------------------------------------------------------------
# Get n / S from the instance (the instance is given and not audited; degrade gracefully if unavailable, never flag the solution for it)
# --------------------------------------------------------------------------
def _infer_n(inst):
    v = _to_int(inst.get("n"))
    if v is not None and v >= 0:
        return v
    # Fallback: infer from fields with axis=node in DATA FIELD SEMANTICS
    for f in ("fixed_costs", "node_coordinates", "base_demands", "distances"):
        val = inst.get(f)
        if isinstance(val, (list, tuple)):
            return len(val)
    return None


def _infer_s(inst):
    v = _to_int(inst.get("num_scenarios"))
    if v is not None and v >= 0:
        return v
    for f in ("scenarios_probability", "scenarios_demands", "scenarios_pi_factors"):
        val = inst.get(f)
        if isinstance(val, (list, tuple)):
            return len(val)
    return None


# --------------------------------------------------------------------------
# Structure parsing: hub_selection -> z ; scenario_allocations -> per-scenario assignment arrays
# --------------------------------------------------------------------------
def _parse_hub_selection(raw, n, sink):
    """Parse z. Returns (z_list, n); z_list elements are 0/1, invalid positions are None (violation already recorded)."""
    seq, kind = _as_indexed_seq(raw, n)
    if seq is None:
        sink.add("FORMAT", "hub_selection must be a list of length n (or a dict keyed by node index), "
                           "actual type %s" % type(raw).__name__)
        return None, n
    if n is None:
        n = len(seq)  # When n is unavailable from the instance, use the solution's own length; do not flag violations out of nothing
    elif len(seq) != n:
        # Basis: output format completeness "list all n nodes"
        sink.add("FORMAT", "hub_selection length should be n=%d, got %d" % (n, len(seq)))
    z = [None] * max(n, len(seq))
    for k, v in enumerate(seq):
        iv = _to_int(v)
        if iv is None or iv not in (0, 1):
            # Basis: decision variable block z[k] ∈ {0,1}
            sink.add("FORMAT", "hub_selection[%d]=%r is not 0/1 (z[k] ∈ {0,1})" % (k, v))
        else:
            z[k] = iv
    return z[:max(n, len(seq))], n


def _parse_allocations(raw, n, s_count, sink):
    """Parse scenario_allocations; return {scenario index: [hub index assigned to each node, or None]}."""
    by_s = {}
    if isinstance(raw, dict):
        groups = {}
        for key, val in raw.items():
            idx = _parse_index_key(key)
            if idx is None:
                continue  # Keys that cannot be resolved uniquely: ignored as irrelevant extra information
            groups.setdefault(idx, []).append((key, val))
        for idx, pairs in groups.items():
            vals = [p[1] for p in pairs]
            if len(vals) > 1:
                first = vals[0]
                if any(v != first for v in vals[1:]):
                    sink.add("FORMAT", "multiple keys in scenario_allocations (%s) refer to scenario %d with inconsistent contents, "
                                       "cannot uniquely determine the assignment for this scenario" % (", ".join(repr(p[0]) for p in pairs), idx))
                    continue
            by_s[idx] = vals[0]
    elif isinstance(raw, (list, tuple)):
        # The description gives the shape dict[str, list[int]]; here an equivalent "list ordered by scenario index" is also tolerated
        for idx, val in enumerate(raw):
            by_s[idx] = val
    else:
        sink.add("FORMAT", "scenario_allocations must be a dict (key = scenario index, value = list of hub indices of length n), "
                           "actual type %s" % type(raw).__name__)
        return None

    if s_count is None:
        s_count = (max(by_s.keys()) + 1) if by_s else 0

    parsed = {}
    for s in range(s_count):
        if s not in by_s:
            # C1 is a constraint over all s: a missing scenario => "each node assigned exactly once" cannot hold in that scenario
            sink.add("C1", "missing assignment for scenario %d (the description requires ∀ s ∈ {0,...,S-1}, S=%d)" % (s, s_count))
            continue
        seq, kind = _as_indexed_seq(by_s[s], n)
        if seq is None:
            sink.add("FORMAT", "the assignment for scenario %d must be a list of length n (or a dict keyed by node index), "
                               "actual type %s" % (s, type(by_s[s]).__name__))
            continue
        if n is not None and len(seq) != n:
            # Basis: output format completeness "the list for each scenario has length n"
            sink.add("FORMAT", "assignment length for scenario %d should be n=%d, got %d" % (s, n, len(seq)))
        width = n if n is not None else len(seq)
        row = [None] * width
        for i in range(min(width, len(seq))):
            iv = _to_int(seq[i])
            if iv is None:
                sink.add("FORMAT", "scenario %d node %d: assignment %r is not an integer index" % (s, i, seq[i]))
            elif not (0 <= iv < width):
                # x[s][i][k] is defined only for k ∈ {0,...,N-1}; an out-of-range index has no corresponding decision variable
                sink.add("FORMAT", "scenario %d node %d: assigned hub index %d is out of range (should be in 0..%d)"
                                   % (s, i, iv, width - 1))
            else:
                row[i] = iv
        for i in range(len(seq), width):
            sink.add("FORMAT", "scenario %d is missing the assignment for node %d" % (s, i))
        parsed[s] = row
    return parsed, s_count


# --------------------------------------------------------------------------
# The three constraints (one-to-one with the description's == CONSTRAINTS ==)
# --------------------------------------------------------------------------
def _check_c1_exactly_one_assignment(z, alloc, s_count, n, sink):
    """C1  sum_{k=0,k!=i}^{N-1} x[s][i][k] = 1 - z[i],  ∀ i, ∀ s

    Meaning: in each scenario a non-hub node is assigned to exactly 1 (other) hub; a hub node is not assigned out (RHS = 0).
    In the solution representation: alloc[s][i] holds exactly one value per node ("exactly once" is guaranteed by the per-node single-value structure of length n + per-position presence),
    so what must be checked here is whether the RHS value matches:
        z[i]==1  ->  the row must be all 0  ->  alloc[s][i] must point to itself i (description: "if the node itself is a hub, it points to itself")
        z[i]==0  ->  exactly one k!=i ->  alloc[s][i] must point to another node (!= i)
    All comparisons are exact integer comparisons, no tolerance.
    """
    for s in range(s_count):
        row = alloc.get(s)
        if row is None:
            continue  # A missing scenario was already recorded as a C1 violation during parsing
        for i in range(n):
            k = row[i] if i < len(row) else None
            if k is None:
                continue  # This position already recorded a FORMAT violation during parsing
            zi = z[i] if (z is not None and i < len(z)) else None
            if zi is None:
                continue  # Invalid z[i] already recorded as a FORMAT violation
            if zi == 1 and k != i:
                sink.add("C1", "scenario %d: node %d is a hub (z[%d]=1, RHS 1-z=0) but is assigned to hub %d, "
                               "i.e. sum_{k!=i} x[%d][%d][k]=1 != 0" % (s, i, i, k, s, i))
            elif zi == 0 and k == i:
                sink.add("C1", "scenario %d: node %d is not a hub (z[%d]=0, RHS 1-z=1) but points to itself (not assigned to any hub), "
                               "i.e. sum_{k!=i} x[%d][%d][k]=0 != 1" % (s, i, i, s, i))


def _check_c2_assign_only_to_open_hub(z, alloc, s_count, n, sink):
    """C2  x[s][i][k] <= z[k],  ∀ i != k, ∀ s

    Meaning: nodes may only be assigned to **opened** hubs.
    In the solution representation: whenever alloc[s][i]=k and k!=i (i.e. x[s][i][k]=1), hub_selection[k]==1 is required.
    Exact integer comparison, no tolerance.
    """
    for s in range(s_count):
        row = alloc.get(s)
        if row is None:
            continue
        for i in range(n):
            k = row[i] if i < len(row) else None
            if k is None or k == i:
                continue  # k==i does not correspond to any x variable (x is defined only for i!=k); C2 holds vacuously here
            zk = z[k] if (z is not None and k < len(z)) else None
            if zk is None:
                continue  # Invalid z[k] already recorded as a FORMAT violation
            if zk != 1:
                sink.add("C2", "scenario %d: node %d is assigned to node %d, but z[%d]=%d (that node is not opened as a hub), "
                               "violating x[%d][%d][%d]=1 <= z[%d]=0" % (s, i, k, k, zk, s, i, k, k))


def _check_c3_at_least_one_hub(z, sink):
    """C3  sum_{k=0}^{N-1} z[k] >= 1

    Meaning: at least one hub is opened in the network.
    In the solution representation: hub_selection contains at least one 1.
    sum is an integer sum of 0/1 values -> exactly representable -> exact comparison >= 1, no tolerance.
    """
    if z is None:
        return
    total = sum(1 for v in z if v == 1)
    if total < 1:
        sink.add("C3", "number of opened hubs in hub_selection is %d, violating sum_k z[k] >= 1 (at least one hub must be opened)" % total)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def check_solution(instance, solution):
    """Return a list of violation strings; an empty list means the solution is feasible. Never raises on any input."""
    sink = _Sink()
    try:
        _run(instance, solution, sink)
    except Exception as exc:  # Fallback: never propagate exceptions to the caller
        sink.add("FORMAT", "unexpected exception during checking: %r" % (exc,))
    return sink.finish()


def _run(instance, solution, sink):
    # When the instance is unreadable, degrade gracefully (fall back to the solution's own length); never flag the solution for it
    inst = instance if isinstance(instance, dict) else {}
    n = _infer_n(inst)
    s_count = _infer_s(inst)

    if not isinstance(solution, dict):
        sink.add("FORMAT", "solution must be a dict, actual type %s" % type(solution).__name__)
        return

    missing = [k for k in _REQUIRED_KEYS if k not in solution or solution[k] is None]
    if missing:
        sink.add("FORMAT", "missing required field %s (the only required fields are %s)"
                 % (", ".join(missing), list(_REQUIRED_KEYS)))
        return

    z, n = _parse_hub_selection(solution["hub_selection"], n, sink)
    if z is None:
        return
    if n is None:
        n = len(z)

    parsed = _parse_allocations(solution["scenario_allocations"], n, s_count, sink)
    if parsed is None:
        # If the assignments cannot be parsed, C3 can still be checked (depends only on hub_selection)
        _check_c3_at_least_one_hub(z, sink)
        return
    alloc, s_count = parsed

    _check_c1_exactly_one_assignment(z, alloc, s_count, n, sink)
    _check_c2_assign_only_to_open_hub(z, alloc, s_count, n, sink)
    _check_c3_at_least_one_hub(z, sink)
