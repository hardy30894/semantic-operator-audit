"""Accuracy of each plan against gold labels, where a dataset provides them, with paired tests.

Ground truth comes from two places:

- gold entities (`cluster` column): a pair matches iff both records belong to one entity. This
  scores matching plans: match(a,b), match(b,a), symmetric matching (both orders combined by AND
  or OR), and majority votes over the k repeats of one order or over both orders.
- machine-checkable predicate rules (`rule` in dataset.toml, evaluated with semaudit.data.eval_rule):
  a claim's truth for a record. In the study data only wdc-A's q (listed price above 100) has a
  rule, so every filter identity whose truth needs p is skipped there; the synthetic dataset has
  rules for both predicates and exercises all of them in the tests.

For every semaudit comparison whose items have a truth (H1-H5, H7, S2-S5), plan A and plan B are
scored on the same items. The paired test is on the per-item difference in accuracy (mean over that
item's valid draws): its mean, a bootstrap interval that resamples items or their clusters (matching
blocks, batch calls) and a two-sided cluster-robust t-test; an exact McNemar test on the per-item
majority-correct indicator is reported alongside. "Does the rewrite change accuracy?" is a different
question from "does it change answers?" (semaudit's Δ): a rewrite can move answers in both
directions and leave accuracy unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from semaudit.core import INVALID, stable_seed
from semaudit.data import eval_rule, load_data
from semaudit.identities import comparisons

from .inference import fmt_ci, mapping_ways, paired_inference
from .logs import collect_ext, load_config, load_design, read_compact, run_logs
from .mitigations import symmetric_value, vote_value


def claim_truths(ds: Any) -> dict[str, dict[str, bool]]:
    """Truth per claim and record, for the claims whose rules the dataset provides."""
    p, q = ds.p.rule, ds.q.rule
    rules: dict[str, Any] = {}
    if p:
        rules["p"] = p
        rules["not_p"] = {"not": p}
    if q:
        rules["q"] = q
    if p and q:
        rules["p_and_q"] = rules["q_and_p"] = {"and": [p, q]}
    return {claim: {rid: eval_rule(rule, rec) for rid, rec in ds.records.items()} for claim, rule in rules.items()}


def gold_pairs(design: dict[str, Any]) -> dict[str, bool]:
    gold = design.get("gold", {})
    out = {}
    for pid in design.get("symmetry", {}).get("pairs", []):
        a, b = pid.split("|")
        out[pid] = gold.get(a) is not None and gold.get(a) == gold.get(b)
    return out


def comparison_truth(name: str, items: list[str], design: dict[str, Any],
                     truths: dict[str, dict[str, bool]]) -> dict[str, Any] | None:
    """Truth for each item of a comparison (both of its plans answer the same logical query)."""
    if name == "symmetry":
        gold = gold_pairs(design)
        return gold if design.get("gold") else None
    if name in ("negation", "batch_neighbours", "batch_position", "batch_vs_single"):
        return truths.get("p")
    if name in ("placement_full", "placement_names"):
        if "p" not in truths:
            return None
        return {t: truths["p"][t.split("|")[0]] for t in items}
    if name in ("fusion", "conjunct_order"):
        return truths.get("p_and_q")
    if name == "count":
        if "p" not in truths:
            return None
        batches = design["filter_algebra"]["count_batches"]
        return {bid: sum(truths["p"][r] for r in members) for bid, members in batches.items()}
    return None


TIE = "tie"  # a vote between equally many True and False labels; scored as a fair coin would be


def _correct(d: Any, truth: Any) -> float:
    return 0.5 if d == TIE else float(d == truth)


def _item_accuracy(draws: list[Any], truth: Any) -> tuple[float, int] | None:
    valid = [d for d in draws if d != INVALID and d is not None]
    if not valid:
        return None
    return sum(_correct(d, truth) for d in valid) / len(valid), len(valid)


def _vote(draws: list[Any]) -> Any:
    """Majority label; a tie among valid labels is kept as TIE rather than dropped, because dropping
    ties would score the voted plan on the easier pairs only."""
    value = vote_value(draws)
    valid = [d for d in draws if d != INVALID and d is not None]
    if value == INVALID and valid and 2 * sum(bool(d) for d in valid) == len(valid) == len(draws):
        return TIE
    return value


def paired_test(acc_a: dict[str, float], acc_b: dict[str, float], groups: dict[str, Any] | None,
                n_boot: int, seed: int, mcnemar: bool = True) -> dict[str, Any]:
    """B minus A in per-item accuracy: mean, t interval and two-sided p from the cluster-robust SE
    (crossed groupings for H4), a bootstrap interval (secondary), and an exact McNemar test on the
    per-item majority-correct indicator (skipped when the two plans read the same draws)."""
    items = sorted(set(acc_a) & set(acc_b))
    if len(items) < 2:
        return {"n_items": len(items), "status": "insufficient data"}
    d = np.array([acc_b[i] - acc_a[i] for i in items])
    res = paired_inference(d, mapping_ways(items, groups), n_boot, np.random.default_rng(seed))
    out = {"n_items": len(items), "accuracy_a": float(np.mean([acc_a[i] for i in items])),
           "accuracy_b": float(np.mean([acc_b[i] for i in items])), **res}
    if not mcnemar:
        out["mcnemar"] = None
        out["mcnemar_note"] = "uninformative: both plans read the same draws"
        return out
    b = sum(1 for i in items if acc_a[i] > 0.5 and acc_b[i] < 0.5)  # A right, B wrong (by majority)
    c = sum(1 for i in items if acc_b[i] > 0.5 and acc_a[i] < 0.5)
    n = b + c
    p_mcnemar = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)
    out["mcnemar"] = {"a_right_b_wrong": b, "b_right_a_wrong": c, "p": p_mcnemar}
    return out


def _groups_for(comp: Any) -> dict[str, Any] | None:
    """semaudit's grouping of a comparison's items: both call groups for H4 (crossed), else one."""
    if comp.call_clusters:
        return {i: tuple(groups) if len(groups) > 1 else groups[0] for i, groups in comp.call_clusters.items()}
    return comp.clusters


def matching_plans(out: dict, design: dict[str, Any]) -> dict[str, dict[str, list[Any]]]:
    """Per plan and pair: its answers (one per repeat; one per pair for the vote plans)."""
    k = design["repeats"]
    plans: dict[str, dict[str, list[Any]]] = {name: {} for name in
                                               ("ab", "ba", "sym_and", "sym_or", "vote_ab", "vote_ba", "vote_both")}
    for pid in design["symmetry"]["pairs"]:
        ab = [out.get(("symmetry", "ab", pid), {}).get(r, INVALID) for r in range(k)]
        ba = [out.get(("symmetry", "ba", pid), {}).get(r, INVALID) for r in range(k)]
        plans["ab"][pid], plans["ba"][pid] = ab, ba
        plans["sym_and"][pid] = [symmetric_value(x, y, "and") for x, y in zip(ab, ba)]
        plans["sym_or"][pid] = [symmetric_value(x, y, "or") for x, y in zip(ab, ba)]
        plans["vote_ab"][pid] = [_vote(ab)]
        plans["vote_ba"][pid] = [_vote(ba)]
        plans["vote_both"][pid] = [_vote(ab + ba)]
    return plans


def _prf(answers: dict[str, list[Any]], gold: dict[str, bool]) -> dict[str, float]:
    """Pooled confusion counts; a TIE counts half as a positive and half as a negative answer."""
    tp = fp = fn = tn = 0.0
    for pid, values in answers.items():
        for v in values:
            if v == INVALID or v is None:
                continue
            positive = 0.5 if v == TIE else float(bool(v))
            if gold[pid]:
                tp += positive
                fn += 1 - positive
            else:
                fp += positive
                tn += 1 - positive
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"answers": int(round(n)), "accuracy": (tp + tn) / n if n else float("nan"), "precision": precision,
            "recall": recall, "f1": f1, "positive_rate": (tp + fp) / n if n else float("nan")}


def accuracy_for_log(records: list[dict[str, Any]], design: dict[str, Any], truths: dict[str, dict[str, bool]],
                     n_boot: int = 2000, seed: int = 0, label: str = "") -> dict[str, Any]:
    out = collect_ext(records)
    result: dict[str, Any] = {"comparisons": [], "conditions": [], "matching": None}
    for comp in comparisons(out, design):
        truth = comparison_truth(comp.name, list(comp.draws), design, truths)
        if not truth:
            continue
        acc_a, acc_b = {}, {}
        for item, (a, b) in comp.draws.items():
            if item not in truth:
                continue
            ra, rb = _item_accuracy(a, truth[item]), _item_accuracy(b, truth[item])
            if ra is not None and rb is not None:
                acc_a[item], acc_b[item] = ra[0], rb[0]
        test = paired_test(acc_a, acc_b, _groups_for(comp), n_boot,
                           stable_seed(seed, label, "accuracy", comp.name) % 2**32)
        row = {"comparison": comp.name, "hypothesis": comp.hypothesis, "a": comp.a, "b": comp.b, **test}
        if comp.kind == "numeric":
            mae = {}
            for side, idx in (("a", 0), ("b", 1)):
                errs = [abs(v - truth[i]) for i, pair in comp.draws.items() if i in truth
                        for v in pair[idx] if v != INVALID and v is not None]
                mae[side] = float(np.mean(errs)) if errs else float("nan")
            row["mean_absolute_error"] = mae
        result["comparisons"].append(row)
    # Single conditions whose claim has a truth (e.g. wdc-A's q).
    by_condition: dict[tuple[str, str], list[float]] = {}
    for (identity, condition, item), slot in out.items():
        if identity != "filter_algebra" or condition not in truths:
            continue
        ra = _item_accuracy(list(slot.values()), truths[condition][item])
        if ra is not None:
            by_condition.setdefault((identity, condition), []).append(ra[0])
    for (identity, condition), accs in sorted(by_condition.items()):
        result["conditions"].append({"identity": identity, "condition": condition, "n_items": len(accs),
                                     "accuracy": float(np.mean(accs))})
    if "symmetry" in design and design.get("gold"):
        gold = gold_pairs(design)
        plans = matching_plans(out, design)
        block_of = {r: f"block{n}" for n, block in enumerate(design["symmetry"]["blocks"]) for r in block}
        clusters = {pid: block_of[pid.split("|")[0]] for pid in gold}
        summary = {name: _prf(answers, gold) for name, answers in plans.items()}
        per_item = {name: {pid: r[0] for pid, values in answers.items()
                           if (r := _item_accuracy(values, gold[pid])) is not None}
                    for name, answers in plans.items()}
        tests = {name: paired_test(per_item["ab"], per_item[name], clusters, n_boot,
                                   stable_seed(seed, label, "matching", name) % 2**32,
                                   mcnemar=name != "vote_ab")  # the majority of match(a,b)'s own draws
                 for name in plans if name != "ab"}
        result["matching"] = {"pairs": len(gold), "positives": sum(gold.values()), "plans": summary,
                              "versus_ab": tests}
    return result


def accuracy_report(run_dirs: list[Path], backends: list[str] | None = None, n_boot: int = 2000) -> dict[str, Any]:
    report: dict[str, Any] = {"runs": {}}
    for run_dir in run_dirs:
        design = load_design(run_dir)
        config = load_config(run_dir)
        data_cfg = config.get("data")
        truths = claim_truths(load_data(data_cfg)) if data_cfg else {}
        has_truth = bool(truths) or bool(design.get("gold") and "symmetry" in design)
        entry = {"dataset": design.get("dataset"), "claims_with_truth": sorted(truths), "backends": {}}
        if has_truth:
            for backend, path in run_logs(run_dir, backends):
                entry["backends"][backend] = accuracy_for_log(read_compact(path), design, truths, n_boot,
                                                              label=f"{run_dir.name}/{backend}")
        report["runs"][run_dir.name] = entry
    return report


def _f(x: Any, d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


PLAN_LABELS = {"ab": "match(a,b)", "ba": "match(b,a)", "sym_and": "symmetric, AND", "sym_or": "symmetric, OR",
               "vote_ab": "majority of k, (a,b)", "vote_ba": "majority of k, (b,a)",
               "vote_both": "majority of 2k, both orders (ties count half)"}


def _test_cells(t: dict[str, Any] | None) -> str:
    """Difference [t CI] | p | bootstrap CI | McNemar p."""
    if not t or "difference" not in t:
        return " | | |"
    if t.get("status") != "ok":
        return f"{t['difference']:+.4f} ({t.get('status')}) | {_f(t.get('p_two_sided'), 3)} | | "
    mc = t.get("mcnemar")
    mc_text = "n/a (same draws)" if mc is None else f"{mc['p']:.3g}"
    return (f"{t['difference']:+.4f} {fmt_ci(t['difference_ci'])} | {t['p_two_sided']:.3g} | "
            f"{fmt_ci(t['difference_ci_bootstrap'])} | {mc_text}")


def render(report: dict[str, Any]) -> str:
    lines = ["# Plan accuracy against gold labels", "",
             "Matching is scored against gold entities; filters against machine-checkable predicate rules where the "
             "dataset has them (in the study data only wdc-A's q). Differences are plan minus match(a,b) (matching) or "
             "plan B minus plan A (identities), in per-item accuracy. Primary: the t interval and two-sided p from the "
             "cluster-robust standard error (matching blocks; crossed N1 and N2 batches for H4), df = clusters - 1. "
             "The bootstrap interval is secondary. McNemar is exact, on per-item majority correctness; it is not "
             "reported for the majority of match(a,b)'s own draws, which cannot disagree with it. A majority vote "
             "that ties is scored as a fair coin (half right), not dropped. No multiplicity correction.", ""]
    for run, entry in report["runs"].items():
        if not entry["backends"]:
            lines += [f"## {run} ({entry['dataset']})", "", "No ground truth for this run's plans.", ""]
            continue
        lines += [f"## {run} ({entry['dataset']})", "",
                  f"Claims with a rule: {', '.join(entry['claims_with_truth']) or 'none'}.", ""]
        for backend, res in entry["backends"].items():
            m = res["matching"]
            if m:
                lines += [f"### {backend}: matching ({m['pairs']} pairs, {m['positives']} gold matches)", "",
                          "| Plan | Answers | Accuracy | Precision | Recall | F1 | Positive rate | Accuracy minus "
                          "match(a,b) [95% t CI] | p | Bootstrap 95% CI | McNemar p |",
                          "|---|---|---|---|---|---|---|---|---|---|---|"]
                for name, s_ in m["plans"].items():
                    lines.append(f"| {PLAN_LABELS[name]} | {s_['answers']} | {_f(s_['accuracy'])} | "
                                 f"{_f(s_['precision'])} | {_f(s_['recall'])} | {_f(s_['f1'])} | "
                                 f"{_f(s_['positive_rate'])} | {_test_cells(m['versus_ab'].get(name))} |")
                lines.append("")
            if res["comparisons"]:
                lines += [f"### {backend}: identities with a truth", "",
                          "| Hyp. | Comparison | Items | Accuracy A | Accuracy B | B minus A [95% t CI] | p | "
                          "Bootstrap 95% CI | McNemar p |", "|---|---|---|---|---|---|---|---|---|"]
                for row in res["comparisons"]:
                    if "difference" not in row:
                        continue
                    lines.append(f"| {row['hypothesis']} | {row['comparison']} | {row['n_items']} | "
                                 f"{_f(row['accuracy_a'])} | {_f(row['accuracy_b'])} | {_test_cells(row)} |")
                lines.append("")
            if res["conditions"]:
                lines.append(f"{backend}, single conditions: " + "; ".join(
                    f"{c['identity']}/{c['condition']} accuracy {c['accuracy']:.3f} over {c['n_items']} rows"
                    for c in res["conditions"]))
                lines.append("")
    return "\n".join(lines)


def write(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "accuracy.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "accuracy.md").write_text(render(report), encoding="utf-8")
