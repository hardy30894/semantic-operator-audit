"""Accuracy in the consistent grid (RQ5), with the paper's own functions (extension/accuracy.py: claim truths,
per-item accuracy, the paired cluster-robust t-test; extension/meta.py: REML pooling with the HKSJ interval).

- Filter rewrites on the grid's labeled conditions: truth of p for CUAD-A, CUAD-B, ArgQ-A, ArgQ-B, arXiv-A and
  arXiv-B; of p and q for the first five. Signed change in points, query B minus query A, per condition, pooled.
- Matching: symmetry against the datasets' labeled matches on WDC, Abt-Buy and ArgKP; placement and renamed columns
  against the truth of the condition on ArgKP (Abt-Buy's and WDC's conditions have none).
- Would an accuracy benchmark notice? Per model and labeled condition, the six rewrites' answer tests (grid
  per-condition p) and accuracy tests, each Holm-adjusted over the rewrites (scripts/review_accuracy_blind.py's rule).
Run from the project root:  python3 grid/analyze_accuracy.py     writes grid/results/accuracy.json
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "harness"
sys.path.insert(0, str(H))
sys.path.insert(0, str(ROOT / "grid"))
from extension.accuracy import _groups_for, _item_accuracy, claim_truths, comparison_truth, paired_test  # noqa: E402
from extension.meta import random_effects  # noqa: E402
from semaudit.core import stable_seed  # noqa: E402
from semaudit.data import load_dataset_dir  # noqa: E402
from semaudit.identities import comparisons  # noqa: E402
from semaudit.stats import holm  # noqa: E402

import analyze as A  # noqa: E402

REWRITES = ["H2", "H3", "S3", "H4", "S2", "S4"]
ORDER = ["H2", "H3", "H4", "H7", "S2", "S3", "S4"]


def accuracy_rows(model: str, where: str, unit: dict, sources: dict, ds, truths: dict) -> dict:
    rows = {}
    for key, (_, design) in A.designs(where, unit).items():
        run = sources.get(where, {}).get(key)
        out = A.outputs(run, model) if run else None
        if out is None:
            continue
        design = copy.deepcopy(design)
        if where in A.MATCH:
            design["gold"] = {rid: rec["cluster"] for rid, rec in ds.records.items() if rec.get("cluster")}
        for comp in comparisons(out, design):
            if comp.name not in A.WANT[key]:
                continue
            if comp.name == "symmetry":
                comp = copy.copy(comp)
                comp.clusters = unit["symmetry"]["cluster_of"]
            truth = comparison_truth(comp.name, list(comp.draws), design, truths)
            if not truth:
                continue
            acc_a, acc_b = {}, {}
            for item, (a, b) in comp.draws.items():
                if item in truth:
                    ra, rb = _item_accuracy(a, truth[item]), _item_accuracy(b, truth[item])
                    if ra is not None and rb is not None:
                        acc_a[item], acc_b[item] = ra[0], rb[0]
            t = paired_test(acc_a, acc_b, _groups_for(comp), 2000,
                            stable_seed(0, "grid-acc", model, where, comp.name) % 2**32)
            rows[comp.hypothesis] = t
    return rows


def main() -> int:
    sources = json.loads((ROOT / "grid" / "coverage.json").read_text(encoding="utf-8"))["sources"]
    grid = {r["model"]: r for r in json.loads((A.OUT / "grid.json").read_text())}
    models = [m for m in A.MODELS if m != "deepseek-v3.2"]
    result = {"filter": {}, "matching": {}, "pooled": {}, "blind": {}, "per_condition_counts": {}}
    for m in models:
        print(f"accuracy: {m}", flush=True)
        per = {}
        for where in A.PAIRS + A.MATCH:
            unit = json.loads((ROOT / "grid" / "units" / (f"match-{where}.json" if where in A.MATCH else f"{where}.json"))
                              .read_text(encoding="utf-8"))
            try:
                ds = load_dataset_dir(H / ".." / "data" / (unit["pair_config"] if where in A.MATCH else where))
            except FileNotFoundError:  # WDC Products and Abt-Buy are not redistributed (docs/DATA.md)
                result.setdefault("skipped_no_records", [])
                if where not in result["skipped_no_records"]:
                    result["skipped_no_records"].append(where)
                    print(f"  {where}: records not found, skipped (only symmetry's accuracy there needs them)", flush=True)
                continue
            truths = claim_truths(ds)
            if where not in A.MATCH and "p" not in truths:
                continue
            rows = accuracy_rows(m, where, unit, sources[m], ds, truths)
            if rows:
                per[where] = rows
        A._cache.clear()
        result["filter"][m] = {w: r for w, r in per.items() if w not in A.MATCH}
        result["matching"][m] = {w: r for w, r in per.items() if w in A.MATCH}
        # pooled signed change over the labeled conditions (points), REML with the HKSJ interval
        pooled = {}
        for hyp in ORDER:
            ok = [(w, r[hyp]) for w, r in result["filter"][m].items() if hyp in r and r[hyp].get("status") == "ok"
                  and r[hyp].get("se")]
            if len(ok) < 2:
                continue
            fit = random_effects([100 * r["difference"] for _, r in ok], [100 * r["se"] for _, r in ok], "REML")
            fit["conditions"] = [w for w, _ in ok]
            pooled[hyp] = fit
        result["pooled"][m] = pooled
        # per-condition tests, Holm within model and condition over the identities with accuracy (as in the paper)
        counts = {}
        blind = []
        for where, r in result["filter"][m].items():
            hyps = [h for h in ORDER if h in r]
            adj = dict(zip(hyps, holm([r[h]["p_two_sided"] if r[h].get("status") == "ok" else 1.0 for h in hyps])))
            for h in hyps:
                c = counts.setdefault(h, {"conditions": 0, "changes": 0, "lowers": 0, "raises": 0})
                c["conditions"] += 1
                if adj[h] < 0.05:
                    c["changes"] += 1
                    c["lowers" if r[h]["difference"] < 0 else "raises"] += 1
            rew = [h for h in REWRITES if h in r]
            ident_p = {}
            for row in grid[m]["comparisons"]:
                if row["hypothesis"] in rew:
                    b = next((x for x in row.get("by_where", []) if x["where"] == where), None)
                    ident_p[row["hypothesis"]] = (b or {}).get("p", 1.0) if (b or {}).get("status") == "ok" else 1.0
            ip = dict(zip(rew, holm([ident_p.get(h, 1.0) for h in rew])))
            ap = dict(zip(rew, holm([r[h]["p_two_sided"] if r[h].get("status") == "ok" else 1.0 for h in rew])))
            for h in rew:
                blind.append({"where": where, "rewrite": h, "breaks": ip[h] < 0.05, "accuracy_moves": ap[h] < 0.05})
        result["per_condition_counts"][m] = counts
        result["blind"][m] = blind
    allb = [c for m in models for c in result["blind"][m]]
    result["blind_summary"] = {"cells": len(allb), "break": sum(c["breaks"] for c in allb),
                               "accuracy_moves": sum(c["accuracy_moves"] for c in allb),
                               "break_and_accuracy": sum(c["breaks"] and c["accuracy_moves"] for c in allb),
                               "h4_break": sum(c["breaks"] for c in allb if c["rewrite"] == "H4"),
                               "h4_accuracy": sum(c["accuracy_moves"] for c in allb if c["rewrite"] == "H4")}
    (A.OUT / "accuracy.json").write_text(json.dumps(result, indent=1, default=float) + "\n",
                                                            encoding="utf-8")
    print(json.dumps(result["blind_summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
