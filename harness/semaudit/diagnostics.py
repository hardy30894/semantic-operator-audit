"""Diagnostics that complement the excess-divergence tests (PREREGISTRATION.md §3.3).

- Fusion incoherence: the fused answer contradicts the component answers when all of them are
  stable. Unlike H3's distributional test, this cannot come from correlated uncertainty within a
  call: no coherent belief makes P(p and q) high while P(p) is low.
- Batch-call collapse: a multi-row call gives every row the same label although the rows' own
  single-row answers disagree. Such calls make the rows of a batch dependent.
- Independent-subset tests: H4, S2 and S4 restricted to one grid column, whose rows share no N1,
  N2 or rot batch, so their draws come from different calls.
"""

from __future__ import annotations

from typing import Any, Iterable

from .core import INVALID
from .identities import Outputs, draws, reparse
from .stats import compare, wilson


def _stable(values: list[Any]) -> Any:
    return values[0] if values and INVALID not in values and len(set(values)) == 1 else None


def fusion_incoherence(out: Outputs, design: dict[str, Any]) -> dict[str, Any]:
    if "filter_algebra" not in design:
        return {}
    k = design["repeats"]
    above = below = checked = 0
    for r in design["filter_algebra"]["rows"]:
        pq = _stable(draws(out, "filter_algebra", "p_and_q", r, k))
        p = _stable(draws(out, "filter_algebra", "p", r, k))
        q = _stable(draws(out, "filter_algebra", "q", r, k))
        if pq is None or p is None or q is None:
            continue
        checked += 1
        above += pq is True and (p is False or q is False)  # conjunction held, a conjunct never did
        below += pq is False and p is True and q is True  # both conjuncts held, conjunction never did
    return {"rows_with_stable_answers": checked, "conjunction_above_a_conjunct": above,
            "conjunction_below_both_conjuncts": below,
            "incoherent_rate_ci": list(wilson(above + below, checked)) if checked else None}


def batch_collapse(records: Iterable[dict[str, Any]], out: Outputs, design: dict[str, Any]) -> dict[str, Any]:
    """Share of batched calls whose labels are all equal although single-row answers differ."""
    if "batch" not in design:
        return {}
    k = design["repeats"]
    single = {r: _stable(draws(out, "batch", "single", r, k)) for r in design["batch"]["rows"]}
    calls = collapsed = 0
    for rec in records:
        if rec.get("identity") != "batch" or rec.get("op") != "filter_batch" or rec.get("error"):
            continue
        labels = reparse(rec)
        if INVALID in labels:
            continue
        reference = {single.get(r) for r in rec["items"]} - {None}
        calls += 1
        collapsed += len(set(labels)) == 1 and len(reference) > 1
    return {"calls": calls, "collapsed": collapsed,
            "collapsed_rate_ci": list(wilson(collapsed, calls)) if calls else None}


def independent_subset_tests(out: Outputs, design: dict[str, Any], n_perm: int = 10000,
                             column: int = 0) -> dict[str, Any]:
    """H4, S2 and S4 on the rows at one grid position: no two of them share any batch call."""
    if "batch" not in design:
        return {}
    k = design["repeats"]
    rows = [members[column] for members in design["batch"]["partitions"]["N1"].values()]
    pairs = {"batch_neighbours": ("N1", "N2"), "batch_position": ("N1", "rot"), "batch_vs_single": ("single", "N1")}
    results = {}
    for name, (a, b) in pairs.items():
        sample = {r: (draws(out, "batch", a, r, k), draws(out, "batch", b, r, k)) for r in rows}
        results[name] = compare(sample, "categorical", n_perm=n_perm, n_boot=1000, seed=column)
    return {"column": column, "rows": len(rows), "tests": results}
