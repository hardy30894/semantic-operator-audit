"""Offline previews of the mitigations, computed from existing logs (no new calls).

The mitigation conditions need new runs to be measured properly. Three of them can be previewed from
v1's logs, because v1 already issued what they would send:

- **Self-consistency voting.** Every request was issued k = 5 times. For a binary comparison, the
  probability that a majority vote over v of an item's draws says True is estimated without bias by
  averaging over all v-subsets of its k draws (a U-statistic); for v = k it is the majority of all
  draws. The voted disagreement between the plans,

      D_v = π_v(A) (1 - π_v(B)) + (1 - π_v(A)) π_v(B),

  is estimated without bias at every v: it is the share of items whose voted answer changes under
  the rewrite, noise included. It is compared with E0, its exact expectation when the item's 2k draws
  are split between the plans at random (exchangeable plans: the hypergeometric distribution of the
  True draws), and a permutation test asks whether the excess D_v - E0 is above 0.
  The excess tests exchangeability at each v; it is not an effect size, and its size is not
  comparable across v. E0 is computed from the pooled draws, which carry the plan's effect. At
  v = 1 the excess is (k-1)/(2k-1) of semaudit's excess divergence (4/9 for k = 5); at v = k it is 0
  for an item whose plans always disagree. For 1 <= v <= k its ratio to the voted operator's excess
  divergence ranges from 0 to above 3 (exact computation over a grid of True rates, k = 5). The
  voted operator's excess divergence is (π_v(A) - π_v(B))^2 per item, a polynomial of degree 2v
  in the True rates, so it has an unbiased estimate only when k >= 2v: with k = 5, only at v = 1.
  To size the effect of voting, compare D_v across v (it falls by the noise that voting removes) or
  run the `vote` segment, or use k >= 2v draws. The permutation treats items as independent: it
  does not resample matching blocks or batch calls.
- **Symmetric matching.** Both operand orders of every pair were issued in every repeat, so
  sym(a,b) = match(a,b) AND match(b,a) (or OR) can be formed per repeat. H1 holds for it by
  construction; the preview reports the transitivity it would have and its call cost (accuracy is
  in accuracy.py).
- **Negation normalization.** NOT F(p) replaces F(not p): H2's excess is removed by construction.
  The preview reports the excess removed, how many rows a NOT p query would return under each plan,
  and the calls that reusing F(p) would save.

Previews are exploratory. A voted run (the `vote` segment) measures the voted operator's excess
divergence directly, from independent voted draws.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from semaudit.clusters import transitivity_report
from semaudit.core import INVALID, stable_seed
from semaudit.identities import comparisons
from semaudit.stats import compare

from .backends import record_cost
from .logs import collect_ext, load_design, read_compact, run_logs

BINARY = ("symmetry", "negation", "fusion", "conjunct_order", "batch_neighbours", "batch_position",
          "batch_vs_single", "placement_full", "placement_names")


def vote_table(k: int, v: int) -> np.ndarray:
    """Entry t: the unbiased estimate of P(a majority of v draws is True) from t True draws out of k,
    i.e. the share of v-subsets of the k draws with a True majority."""
    total = math.comb(k, v)
    return np.array([sum(math.comb(t, j) * math.comb(k - t, v - j) for j in range(v // 2 + 1, v + 1)) / total
                     for t in range(k + 1)])


def voted_disagreement(a: np.ndarray, b: np.ndarray, k: int, v: int) -> np.ndarray:
    table = vote_table(k, v)
    pa, pb = table[a], table[b]
    return pa * (1 - pb) + (1 - pa) * pb


def null_expectation(a: np.ndarray, b: np.ndarray, k: int, v: int) -> np.ndarray:
    """E[D_v] when an item's 2k draws are split between the plans at random: the number of True draws
    that land in plan A is hypergeometric."""
    n1 = a + b
    expected = np.zeros(len(a), dtype=float)
    denom = math.comb(2 * k, k)
    pmf = np.array([[math.comb(n, x) * math.comb(2 * k - n, k - x) / denom if 0 <= n - x <= k else 0.0
                     for x in range(k + 1)] for n in range(2 * k + 1)])
    for a_star in range(k + 1):
        b_star = np.clip(n1 - a_star, 0, k)
        expected += pmf[n1, a_star] * voted_disagreement(np.full(len(a), a_star), b_star, k, v)
    return expected


def voting_preview(a: np.ndarray, b: np.ndarray, k: int, votes: tuple[int, ...], n_perm: int,
                   rng: np.random.Generator) -> dict[int, dict[str, float]]:
    """a, b: True draws per item under each plan (k draws each)."""
    out = {}
    n1 = a + b
    for v in votes:
        observed = voted_disagreement(a, b, k, v)
        expected = null_expectation(a, b, k, v)
        stat = float(observed.mean())
        hits = 0
        chunk = max(1, int(2e6 // max(1, len(a))))
        for start in range(0, n_perm, chunk):
            m = min(chunk, n_perm - start)
            a_star = rng.hypergeometric(n1, 2 * k - n1, k, size=(m, len(a)))
            perm = voted_disagreement(a_star.ravel(), (n1 - a_star).ravel(), k, v).reshape(m, len(a)).mean(axis=1)
            hits += int((perm >= stat - 1e-12).sum())
        out[v] = {"observed": stat, "expected_if_exchangeable": float(expected.mean()),
                  "excess": stat - float(expected.mean()), "p_perm": (hits + 1) / (n_perm + 1)}
    return out


def _binary_counts(draws: dict[str, tuple[list[Any], list[Any]]], k: int) -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    for item, (xa, xb) in sorted(draws.items()):
        if len(xa) != k or len(xb) != k or any(v == INVALID or v is None for v in (*xa, *xb)):
            continue
        if not all(isinstance(v, bool) for v in (*xa, *xb)):
            continue
        a.append(sum(xa))
        b.append(sum(xb))
    return np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)


def preview_log(records: list[dict[str, Any]], design: dict[str, Any], votes: tuple[int, ...], n_perm: int,
                seed: int, label: str, counts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per comparison: the voting preview for this log. `counts`, if given, collects each binary
    comparison's (k, True draws under A, True draws under B) for pooling across runs."""
    k = design["repeats"]
    out = collect_ext(records)
    result: dict[str, Any] = {"voting": {}, "symmetric": None, "negation": None}
    comps = {c.name: c for c in comparisons(out, design)}
    for name in BINARY:
        if name not in comps:
            continue
        a, b = _binary_counts(comps[name].draws, k)
        if len(a) < 2:
            continue
        if counts is not None:
            counts.setdefault(name, {"hypothesis": comps[name].hypothesis, "parts": []})["parts"].append((k, a, b))
        rng = np.random.default_rng(stable_seed(seed, label, "vote", name) % 2**32)
        result["voting"][name] = {"hypothesis": comps[name].hypothesis, "n_items": int(len(a)),
                                  "by_votes": voting_preview(a, b, k, tuple(v for v in votes if v <= k), n_perm, rng)}
    if "symmetry" in design:
        trans = transitivity_report(out, design, n_boot=200, seed=stable_seed(seed, label, "trans") % 2**32)
        h1 = compare(comps["symmetry"].draws, "categorical", n_perm=2000, n_boot=200,
                     clusters=comps["symmetry"].clusters)
        result["symmetric"] = {
            "h1_delta_removed": h1.get("delta"),
            "calls_per_answer": {"single order": 1, "symmetric": 2},
            "transitivity": {p: {"violation_rate": trans[p]["violation_rate"],
                                 "random_error_baseline": trans[p]["random_error_baseline"]["violation_rate"],
                                 "f1": trans[p]["clusters"]["f1"], "positive_rate": trans[p]["positive_rate"]}
                             for p in ("ab", "ba", "and", "or")}}
    if "negation" in comps:
        neg = comps["negation"]
        h2 = compare(neg.draws, "categorical", n_perm=2000, n_boot=200)
        rows = design["filter_algebra"]["rows"]
        not_p = [r for r in records if r["identity"] == "filter_algebra" and r["condition"] == "not_p"]
        returned = {"F(not p)": [], "NOT F(p)": []}
        for rep in range(k):
            fp = [out.get(("filter_algebra", "p", r), {}).get(rep, INVALID) for r in rows]
            fnp = [out.get(("filter_algebra", "not_p", r), {}).get(rep, INVALID) for r in rows]
            returned["F(not p)"].append(sum(1 for v in fnp if v is True))
            returned["NOT F(p)"].append(sum(1 for v in fp if v is False))
        result["negation"] = {"h2_delta_removed": h2.get("delta"), "rows": len(rows),
                              "rows_returned_per_repeat": returned,
                              "not_p_calls_saved_with_reuse": len(not_p),
                              "usd_saved_with_reuse": sum(record_cost(r) for r in not_p)}
    return result


def preview_report(run_dirs: list[Path], backends: list[str] | None = None, votes: tuple[int, ...] = (1, 3, 5),
                   n_perm: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Per run and arm, and pooled over the runs per arm and identity (`pooled_voting`): items from
    every run with the same k are concatenated, and the test reshuffles within items as before."""
    report: dict[str, Any] = {"runs": {}, "votes": list(votes), "pooled_voting": {}}
    counts: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        design = load_design(run_dir)
        report["runs"][run_dir.name] = {
            backend: preview_log(read_compact(path), design, votes, n_perm, seed, f"{run_dir.name}/{backend}",
                                 counts.setdefault(backend, {}))
            for backend, path in run_logs(run_dir, backends)}
    for backend, by_name in counts.items():
        pooled = {}
        for name, entry in by_name.items():
            ks = {k for k, _, _ in entry["parts"]}
            if len(ks) != 1:  # different repeats cannot share one exact null
                pooled[name] = {"hypothesis": entry["hypothesis"], "status": f"runs differ in repeats {sorted(ks)}"}
                continue
            k = ks.pop()
            a = np.concatenate([p[1] for p in entry["parts"]])
            b = np.concatenate([p[2] for p in entry["parts"]])
            rng = np.random.default_rng(stable_seed(seed, "pooled", backend, name) % 2**32)
            pooled[name] = {"hypothesis": entry["hypothesis"], "runs": len(entry["parts"]), "n_items": int(len(a)),
                            "k": k, "by_votes": voting_preview(a, b, k, tuple(v for v in votes if v <= k), n_perm, rng)}
        report["pooled_voting"][backend] = pooled
    return report


VOTING_NOTE = (
    "D_v: share of items whose majority-of-v answers differ between the two plans, estimated without bias "
    "(noise included); v = 1 is the operator without voting. E0: its exact expectation if the plans were "
    "exchangeable, from each item's pooled draws. Excess = D_v - E0, with a one-sided permutation p, tests "
    "exchangeability at each v. It is not an effect size, and its size is not comparable across v: E0 absorbs "
    "part of the plan's effect, more as v approaches k (at v = 1 the excess is (k-1)/(2k-1) of semaudit's excess "
    "divergence, 4/9 for k = 5; at v = k it is 0 for an item whose plans always disagree). The voted operator's "
    "own excess divergence has an unbiased estimate only for v <= k/2. Compare D_v across v to see what voting "
    "removes. The permutation treats items as independent (it does not resample matching blocks or batch "
    "calls), so its p-values for H1, H4, S2 and S4 may be too small."
)


def render_pooled_voting(report: dict[str, Any]) -> str:
    votes = report["votes"]
    lines = ["## Voting, pooled over the runs (per arm and identity)", "",
             VOTING_NOTE, "",
             "| Arm | Hyp. | Comparison | Runs | Items | " + " | ".join(f"v={v}: D / E0 / excess (p)" for v in votes)
             + " |", "|" + "---|" * (5 + len(votes))]
    for backend, pooled in report["pooled_voting"].items():
        for name, row in pooled.items():
            if "by_votes" not in row:
                lines.append(f"| {backend} | {row['hypothesis']} | {name} | | | {row.get('status')} |")
                continue
            cells = []
            for v in votes:
                r = row["by_votes"].get(v, row["by_votes"].get(str(v)))
                cells.append("" if r is None else f"{r['observed']:.3f} / {r['expected_if_exchangeable']:.3f} / "
                                                  f"{r['excess']:+.3f} ({r['p_perm']:.3g})")
            lines.append(f"| {backend} | {row['hypothesis']} | {name} | {row['runs']} | {row['n_items']} | "
                         + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render(report: dict[str, Any]) -> str:
    votes = report["votes"]
    lines = ["# Offline previews of the mitigations (logged runs, no new calls)", "",
             render_pooled_voting(report),
             "## Per run", "",
             "Voting columns as in the pooled table above.", ""]
    for run, by_backend in report["runs"].items():
        lines += [f"### {run}", ""]
        header = "| Backend | Comparison | Items | " + " | ".join(f"v={v}: D / E0 / excess (p)" for v in votes) + " |"
        lines += [header, "|" + "---|" * (3 + len(votes))]
        for backend, res in by_backend.items():
            for name, row in res["voting"].items():
                cells = []
                for v in votes:
                    r = row["by_votes"].get(str(v), row["by_votes"].get(v))
                    cells.append("" if r is None else
                                 f"{r['observed']:.3f} / {r['expected_if_exchangeable']:.3f} / {r['excess']:+.3f} "
                                 f"({r['p_perm']:.3g})")
                lines.append(f"| {backend} | {row['hypothesis']} {name} | {row['n_items']} | " + " | ".join(cells) + " |")
        lines.append("")
        for backend, res in by_backend.items():
            sym = res.get("symmetric")
            if sym:
                t = sym["transitivity"]
                lines.append(f"- {backend}, symmetric matching: H1 excess removed {sym['h1_delta_removed']:.4f} at 2 calls "
                             "per answer; triangle-violation rate (random-error baseline) "
                             + ", ".join(f"{p} {t[p]['violation_rate']:.4f} ({t[p]['random_error_baseline']:.4f})"
                                         for p in ("ab", "ba", "and", "or"))
                             + "; pairwise F1 of the deduplication clusters (transitive closure, against gold) "
                             + ", ".join(f"{p} {t[p]['f1']:.3f}" for p in ("ab", "ba", "and", "or")))
            neg = res.get("negation")
            if neg:
                ret = neg["rows_returned_per_repeat"]
                lines.append(f"- {backend}, negation normalization: H2 excess removed {neg['h2_delta_removed']:.4f}; "
                             f"a NOT p query over {neg['rows']} rows returns {np.mean(ret['F(not p)']):.1f} rows as "
                             f"F(not p) and {np.mean(ret['NOT F(p)']):.1f} as NOT F(p) (mean per repeat); reusing F(p) saves "
                             f"{neg['not_p_calls_saved_with_reuse']} calls (${neg['usd_saved_with_reuse']:.4f})")
        lines.append("")
    return "\n".join(lines)


def write(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "previews.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "previews.md").write_text(render(report), encoding="utf-8")
