"""Turn a run directory into results: per-identity tests, transitivity, and call diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .clusters import transitivity_report
from .core import INVALID, stable_seed
from .identities import collect, comparisons, other_host, parse_record
from .runner import log_files, log_name, read_log
from .stats import compare, holm


def call_summary(records: list[dict[str, Any]], design: dict[str, Any]) -> dict[str, Any]:
    """Diagnostics computed on the analysed data (latest attempt per call, re-parsed), not on raw
    log lines, so retried failures are not counted twice."""
    latest: dict[str, dict[str, Any]] = {}
    for rec in records:
        if rec["task_id"] not in latest or not rec.get("error"):
            latest[rec["task_id"]] = rec
    invalid: Counter[str] = Counter()
    total: Counter[str] = Counter()
    rules: dict[str, Counter] = {}
    other_host_calls = 0
    for rec in latest.values():
        key = f"{rec['identity']}/{rec['condition']}"
        outputs, fired = parse_record(rec)
        total[key] += len(outputs)
        invalid[key] += sum(v == INVALID for v in outputs)
        other_host_calls += other_host(rec)
        if not rec.get("error") and rec.get("text") is not None:
            rules.setdefault(key, Counter()).update(fired)
    usage = Counter()
    for rec in records:
        for k, v in (rec.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] += v
    summary = {
        "calls": len(records),
        "distinct_calls": len(latest),
        "failed_calls": sum(1 for r in latest.values() if r.get("error")),
        "served_models": dict(Counter(str(r.get("served_model")) for r in latest.values())),
        "served_providers": dict(Counter(str((r.get("extra") or {}).get("served_provider")) for r in latest.values())),
        "other_host_calls_excluded": other_host_calls,
        "stop_reasons": dict(Counter(str(r.get("stop_reason")) for r in latest.values())),
        "usage": dict(usage),
        "invalid_rate": {k: invalid[k] / total[k] for k in sorted(total)},
        "parse_rules": {k: dict(v) for k, v in sorted(rules.items())},
    }
    truth = design.get("truth")
    if truth:  # exploratory: how often single-row filter answers match the data's ground truth
        out = collect(records)
        accuracy = {}
        for claim in truth:
            hits = n = 0
            for (identity, condition, item), slot in out.items():
                if identity == "filter_algebra" and condition == claim:
                    for value in slot.values():
                        if value != INVALID:
                            n += 1
                            hits += value == truth[claim][item]
            if n:
                accuracy[claim] = hits / n
        summary["filter_accuracy"] = accuracy
    return summary


def analyze_records(records: list[dict[str, Any]], design: dict[str, Any], n_perm: int, n_boot: int,
                    seed: int, label: str) -> dict[str, Any]:
    out = collect(records)
    rows = []
    for comp in comparisons(out, design):
        stats = compare(comp.draws, comp.kind, n_perm=n_perm, n_boot=n_boot,
                        seed=stable_seed(seed, label, comp.name) % 2**32, clusters=comp.clusters,
                        call_clusters=comp.call_clusters)
        rows.append({"comparison": comp.name, "hypothesis": comp.hypothesis, "primary": comp.primary,
                     "a": comp.a, "b": comp.b, **stats})
    return {
        "comparisons": rows,
        "transitivity": transitivity_report(out, design, n_boot=min(n_boot, 2000),
                                            seed=stable_seed(seed, label, "transitivity") % 2**32),
        "calls": call_summary(records, design),
    }


# S1 equivalence margin (PREREGISTRATION.md §3.2): drift is negligible if the pooled interval lies in [0, 0.005].
DRIFT_MARGIN = 0.005


def _drift_groups(design: dict[str, Any], identity: str, condition: str, item: str) -> str:
    """The group an item's requests share calls or records with: its batch prompt for the batched
    conditions, its block for matching pairs, else the item itself."""
    if identity == "batch" and condition in design.get("batch", {}).get("partitions", {}):
        for bid, members in design["batch"]["partitions"][condition].items():
            if item in members:
                return f"{condition}:{bid}"
    if identity == "symmetry":
        for n, block in enumerate(design["symmetry"]["blocks"]):
            if item.split("|")[0] in block:
                return f"block{n}"
    return f"{identity}/{condition}/{item}"


def drift_report(run_a: Path, run_b: Path, n_perm: int, n_boot: int, seed: int) -> dict[str, Any]:
    """Same requests in two runs: does the second run disagree more than duplicates within a run?

    Both runs must share one design. Idempotence's follow-up requests are skipped, because their
    inputs differ between runs by construction. Items that share batch prompts or matching blocks are
    resampled together, and the test guards against dependence within those groups (stats.compare).
    """
    design_a = (run_a / "design.json").read_text(encoding="utf-8")
    design_b = (run_b / "design.json").read_text(encoding="utf-8")
    if json.loads(design_a) != json.loads(design_b):
        raise SystemExit(f"{run_a} and {run_b} have different designs; drift needs identical requests")
    design = json.loads(design_a)
    k = design["repeats"]
    report = {}
    logs_b = {log_name(p): p for p in log_files(run_b)}
    for log_a in log_files(run_a):
        backend = log_name(log_a)
        log_b = logs_b.get(backend)
        if log_b is None:
            continue
        out_a, out_b = collect(read_log(log_a)), collect(read_log(log_b))
        compared = {key for key in set(out_a) | set(out_b)
                    if not (key[0] == "idempotence" and key[1] in ("fx", "x_in"))}
        in_one_run_only = sum(1 for key in compared if (key in out_a) != (key in out_b))
        by_condition: dict[tuple[str, str], dict[str, Any]] = {}
        for key in sorted(set(out_a) & set(out_b)):
            identity, condition, item = key
            if identity == "idempotence" and condition in ("fx", "x_in"):
                continue
            by_condition.setdefault((identity, condition), {})[item] = (
                [out_a[key].get(r, INVALID) for r in range(k)],
                [out_b[key].get(r, INVALID) for r in range(k)],
            )
        report[backend] = {
            f"{identity}/{condition}": compare(
                pairs, "numeric" if condition == "count" else "categorical", n_perm=n_perm, n_boot=n_boot,
                seed=stable_seed(seed, "drift", backend, identity, condition) % 2**32,
                call_clusters={item: (_drift_groups(design, identity, condition, item),) for item in pairs})
            for (identity, condition), pairs in by_condition.items()
        }
        # S1: one pooled estimate over every True/False request, each (condition, item) as an item.
        pooled, groups = {}, {}
        for (identity, condition), pairs in by_condition.items():
            for item, draws in pairs.items():
                if all(isinstance(v, bool) or v == INVALID for side in draws for v in side):
                    key = f"{identity}/{condition}/{item}"
                    pooled[key] = draws
                    groups[key] = (_drift_groups(design, identity, condition, item),)
        row = compare(pooled, "categorical", n_perm=n_perm, n_boot=n_boot,
                      seed=stable_seed(seed, "drift", backend, "pooled") % 2**32, call_clusters=groups)
        if row.get("status") == "ok":
            row["equivalence_margin"] = DRIFT_MARGIN
            row["equivalent"] = row["delta_ci"][1] <= DRIFT_MARGIN
        row["requests_in_one_run_only"] = in_one_run_only  # not compared; nonzero if a run is incomplete
        report[backend]["pooled_binary"] = row
    return report


def analyze_run(run_dir: Path, n_perm: int = 5000, n_boot: int = 5000, seed: int = 0,
                compare_run: Path | None = None) -> dict[str, Any]:
    design = json.loads((run_dir / "design.json").read_text(encoding="utf-8"))
    results: dict[str, Any] = {"run": str(run_dir), "backends": {}}
    for log_path in log_files(run_dir):
        name = log_name(log_path)
        results["backends"][name] = analyze_records(read_log(log_path), design, n_perm, n_boot, seed, name)
    # Descriptive only: Holm within each model and family for this run. The confirmatory numbers
    # come from `semaudit pool` over the pre-registered runs (PREREGISTRATION.md §6.2).
    for res in results["backends"].values():
        for family in (True, False):
            idx = [i for i, row in enumerate(res["comparisons"]) if row["primary"] is family]
            adjusted = holm([res["comparisons"][i].get("p", 1.0) for i in idx])
            for i, p_adj in zip(idx, adjusted):
                res["comparisons"][i]["p_holm_run"] = p_adj
    if compare_run is not None:
        results["compare_run"] = str(compare_run)
        results["drift"] = drift_report(run_dir, compare_run, n_perm, n_boot, seed)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "results.md").write_text(render_markdown(results), encoding="utf-8")
    return results


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_markdown(results: dict[str, Any]) -> str:
    lines = [f"# Results for {results['run']}", ""]
    for name, res in results["backends"].items():
        calls = res["calls"]
        lines += [f"## {name}", "",
                  f"Calls: {calls['calls']} ({calls['failed_calls']} failed; {calls['other_host_calls_excluded']} "
                  f"served by another host than the pinned one, excluded). "
                  f"Served models: {calls['served_models']}. Served by: {calls['served_providers']}. "
                  f"Cost: ${calls['usage'].get('cost', 0.0):.4f}.", "",
                  "| Hyp. | Comparison | Items (excl.) | Within-condition disagreement | Cross-condition "
                  "disagreement | Excess Δ [approx. 95% CI] | p (perm.) | p | p (Holm, this run; descriptive) | "
                  "Stable violations: obs / exp |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in res["comparisons"]:
            if row.get("status") != "ok":
                lines.append(f"| {row['hypothesis']} | {row['comparison']} | {row['n_items']} "
                             f"({row['n_excluded']}) | {row.get('status')} | | | | | | |")
                continue
            lo, hi = row["delta_ci"]
            lines.append(
                f"| {row['hypothesis']} | {row['comparison']} | {row['n_items']} ({row['n_excluded']}) | "
                f"{_fmt(row['noise_within'])} | {_fmt(row['cross'])} | {_fmt(row['delta'])} "
                f"[{_fmt(lo)}, {_fmt(hi)}] | {_fmt(row['p_perm'])} | {_fmt(row['p'])} | {_fmt(row.get('p_holm_run'))} | "
                f"{row['stable_violations']} / {_fmt(row['stable_expected'], 2)} |")
        trans = res["transitivity"]
        if trans:
            lines += ["", f"Transitivity over {trans['n_triangles']} triangles in {trans['n_blocks']} blocks "
                          f"(gold violations: {trans['gold_violations']}):", "",
                      "| Edge labels | Violation rate [95% CI] | Random-error baseline [95% int.] | Chance "
                      "rate | Violations among unanimous / confident edges | Clusters | Merged clusters | "
                      "Closure conflicts | Pairwise F1 |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for policy in ("ab", "ba", "or", "and"):
                t = trans[policy]
                c, base = t["clusters"], t["random_error_baseline"]
                lines.append(
                    f"| per repeat, {policy} | {_fmt(t['violation_rate'])} [{_fmt(t['violation_ci'][0])}, "
                    f"{_fmt(t['violation_ci'][1])}] | {_fmt(base['violation_rate'])} [{_fmt(base['interval'][0])}, "
                    f"{_fmt(base['interval'][1])}] | {_fmt(t['chance_rate'])} | {t['unanimous_violations']} / "
                    f"{t['unanimous_triangles']} | {_fmt(c['n_clusters'], 1)} | {_fmt(c['merged_clusters'], 1)} | "
                    f"{_fmt(c['closure_conflicts'], 1)} | {_fmt(c['f1'], 3)} |")
            m = trans["modal"]
            c, base = m["clusters"], m["random_error_baseline"]
            lines.append(
                f"| modal over both orders | {_fmt(m['violation_rate'])} [{_fmt(m['violation_ci'][0])}, "
                f"{_fmt(m['violation_ci'][1])}] | {_fmt(base['violation_rate'])} [{_fmt(base['interval'][0])}, "
                f"{_fmt(base['interval'][1])}] | | {m['confident_violations']} / {m['confident_triangles']} | "
                f"{c['n_clusters']} | {c['merged_clusters']} | {c['closure_conflicts']} | {_fmt(c['f1'], 3)} |")
        if calls.get("filter_accuracy"):
            lines += ["", "Filter accuracy against ground truth (exploratory): "
                      + ", ".join(f"{k} {v:.3f}" for k, v in calls["filter_accuracy"].items())]
        lines.append("")
    for name, drift in results.get("drift", {}).items():
        lines += [f"## Cross-run drift: {name}, this run against {results['compare_run']}", "",
                  "Both runs share one design. `pooled_binary` pools every True/False request, each "
                  "condition × item as one item (S1).", "",
                  "| Requests | Items (excl.) | Within-run disagreement | Cross-run disagreement | "
                  "Excess Δ [approx. 95% CI] | p (perm.) | p | Equivalent (CI within [0, 0.005]) |",
                  "|---|---|---|---|---|---|---|---|"]
        for key in sorted(drift, key=lambda k: (k == "pooled_binary", k)):
            row = drift[key]
            if row.get("status") != "ok":
                lines.append(f"| {key} | {row.get('n_items')} ({row.get('n_excluded')}) | {row.get('status')} | | | | | |")
                continue
            lo, hi = row["delta_ci"]
            lines.append(f"| {key} | {row['n_items']} ({row['n_excluded']}) | {_fmt(row['noise_within'])} | "
                         f"{_fmt(row['cross'])} | {_fmt(row['delta'])} [{_fmt(lo)}, {_fmt(hi)}] | "
                         f"{_fmt(row['p_perm'])} | {_fmt(row['p'])} | "
                         f"{'' if 'equivalent' not in row else 'yes' if row['equivalent'] else 'no'} |")
        pooled = drift.get("pooled_binary", {})
        if pooled.get("requests_in_one_run_only"):
            lines.append(f"\n{pooled['requests_in_one_run_only']} requests appear in only one of the runs and were "
                         "not compared (an incomplete run).")
        lines.append("")
    return "\n".join(lines)
