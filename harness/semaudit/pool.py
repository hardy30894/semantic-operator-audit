"""Pool each identity's items across run directories (datasets and predicate pairs), per model.

This is the confirmatory analysis (PREREGISTRATION.md §6.2):

- **Pooling.** The primary test of an identity pools its items from every run where it is
  measured, with item and cluster ids kept distinct per run.
- **Holm.** Holm-Bonferroni is applied within each model, separately for the primary family
  (H1-H7) and the secondary family (S2-S5), always over all of their slots. A test with too little
  data, or not measured in the runs, enters its family as p = 1 and is flagged.
- **Completeness.** Every model must have a log in every run; otherwise pooling refuses.
- **Generality.** An identity is "broken in general" if at least two of the three primary arms
  support it.

Runs must not repeat a (dataset, prompt style) pair, and must share one prompt style.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .analyze import call_summary
from .core import INVALID, stable_seed
from .diagnostics import batch_collapse, fusion_incoherence, independent_subset_tests
from .identities import collect, comparisons, other_host
from .runner import log_files, log_name, read_log
from .stats import compare, holm

PRIMARY_HYPOTHESES = ("H1", "H2", "H3", "H4", "H5", "H6", "H7")
SECONDARY_HYPOTHESES = ("S2", "S3", "S4", "S5")
SLOT_NAMES = {"H1": "symmetry", "H2": "negation", "H3": "fusion", "H4": "batch_neighbours",
              "H5": "placement_full", "H6": "idempotence", "H7": "count", "S2": "batch_position",
              "S3": "conjunct_order", "S4": "batch_vs_single", "S5": "placement_names"}
DEFAULT_PRIMARY_ARMS = ("haiku-5.5", "gpt-5.6-luna", "gemini-3.5-flash-lite")
# Smallest effects of interest: an RMS shift of 5 points in P(True), or half a row per COUNT batch.
SESOI = {"categorical": 0.05, "numeric": 0.5}


def _check_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    infos = []
    for run_dir in run_dirs:
        raw = (run_dir / "design.json").read_text(encoding="utf-8")
        design = json.loads(raw)
        infos.append({"run": run_dir.name, "dataset": design.get("dataset"), "style": design.get("style", "lotus"),
                      "design_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(), "design": design})
    seen: dict[str, str] = {}
    for info in infos:
        if info["design_sha256"] in seen:
            raise SystemExit(f"runs {seen[info['design_sha256']]} and {info['run']} have the same design; pooling "
                             "both would count every item twice (a replicate run belongs in --compare-run)")
        seen[info["design_sha256"]] = info["run"]
    styles = {info["style"] for info in infos}
    if len(styles) > 1:
        raise SystemExit(f"runs mix prompt styles {sorted(styles)}; analyze each style separately")
    datasets = Counter(info["dataset"] for info in infos)
    for info in infos:
        info["warning"] = (f"dataset {info['dataset']} appears in {datasets[info['dataset']]} runs; items may repeat records"
                           if datasets[info["dataset"]] > 1 else None)
    return infos


def planned_calls(run_dir: Path) -> int | None:
    """Calls a semaudit run plans per model: every task plus one follow-up per stage-0 input. None
    for directories written by other tools (e.g. the engine-native arm)."""
    try:
        from .data import load_data
        from .identities import build

        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        run = config["run"]
        tasks, _ = build(load_data(config["data"]), config["identities"], run["repeats"], run["seed"],
                         run.get("prompt_style", "lotus"))
    except (OSError, KeyError, ValueError, TypeError):
        return None
    return len(tasks) + sum(t.condition == "x_in" for t in tasks)


def _check_logs(run_dirs: list[Path], backends: list[str] | None, allow_missing: bool) -> dict[str, list[str]]:
    """Each model's log in each run. Pooling refuses if a model is missing from a run, because its
    pooled tests would silently rest on fewer items than pre-registered."""
    present = {run_dir.name: {log_name(p) for p in log_files(run_dir)} for run_dir in run_dirs}
    models = sorted(set().union(*present.values()))
    if backends:
        unknown = sorted(set(backends) - set(models))
        if unknown:
            raise SystemExit(f"--backend {unknown} has no log in these runs (models found: {models})")
        models = [m for m in models if m in backends]
    missing = {m: [run for run, names in present.items() if m not in names] for m in models}
    missing = {m: runs for m, runs in missing.items() if runs}
    if missing and not allow_missing:
        raise SystemExit(f"models missing from some runs: {missing}; finish those runs, or pass "
                         "--allow-missing to pool anyway (flagged in the output)")
    return missing


def pool_runs(run_dirs: list[Path], n_perm: int = 10000, n_boot: int = 10000, seed: int = 0,
              backends: list[str] | None = None, n_perm_binary: int = 100000,
              allow_missing: bool = False) -> dict[str, Any]:
    infos = _check_runs(run_dirs)
    missing = _check_logs(run_dirs, backends, allow_missing)
    slots: dict[str, dict[str, dict[str, Any]]] = {}
    usage: dict[str, Counter] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    parsing: dict[str, dict[str, Any]] = {}
    for info, run_dir in zip(infos, run_dirs):
        design = info["design"]
        planned = planned_calls(run_dir)
        for log_path in log_files(run_dir):
            backend = log_name(log_path)
            if backends and backend not in backends:
                continue
            records = read_log(log_path)
            out = collect(records)
            u = usage.setdefault(backend, Counter())
            u["calls"] += len(records)
            for rec in records:
                for key, value in (rec.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        u[key] += value
                u["served_by:" + str((rec.get("extra") or {}).get("served_provider"))] += 1
                u["calls_other_host"] += other_host(rec)
            u["draws"] += sum(len(v) for v in out.values())
            u["invalid_draws"] += sum(value == INVALID for v in out.values() for value in v.values())
            summary = call_summary(records, design)
            parsing.setdefault(backend, {})[run_dir.name] = {
                **{k: summary[k] for k in ("invalid_rate", "parse_rules", "stop_reasons", "failed_calls",
                                           "other_host_calls_excluded", "distinct_calls")},
                "planned_calls": planned,
                "calls_not_issued": None if planned is None else max(0, planned - summary["distinct_calls"])}
            if planned is not None and summary["distinct_calls"] < planned:
                u["calls_not_issued"] += planned - summary["distinct_calls"]
            diag = diagnostics.setdefault(backend, {})
            diag[run_dir.name] = {"fusion_incoherence": fusion_incoherence(out, design),
                                  "batch_collapse": batch_collapse(records, out, design),
                                  "independent_subset": independent_subset_tests(out, design)}
            for comp in comparisons(out, design):
                slot = slots.setdefault(backend, {}).setdefault(
                    comp.name, {"comp": comp, "draws": {}, "clusters": {}, "call_clusters": {}, "by_run": {}})
                prefix = run_dir.name
                slot["draws"].update({f"{prefix}/{item}": d for item, d in comp.draws.items()})
                if comp.clusters:
                    slot["clusters"].update({f"{prefix}/{item}": f"{prefix}/{c}" for item, c in comp.clusters.items()})
                if comp.call_clusters:
                    slot["call_clusters"].update({f"{prefix}/{item}": tuple(f"{prefix}/{c}" for c in groups)
                                                  for item, groups in comp.call_clusters.items()})
                slot["by_run"][prefix] = comp
    results: dict[str, Any] = {"runs": [{k: v for k, v in info.items() if k != "design"} for info in infos],
                               "backends": {}}
    for backend, comps in slots.items():
        rows = []
        for name, slot in comps.items():
            c = slot["comp"]
            seed_c = stable_seed(seed, "pool", backend, name) % 2**32
            stats = compare(slot["draws"], c.kind, n_perm=n_perm, n_boot=n_boot, seed=seed_c,
                            clusters=slot["clusters"] or None, n_perm_binary=n_perm_binary,
                            call_clusters=slot["call_clusters"] or None)
            row = {"comparison": name, "hypothesis": c.hypothesis, "primary": c.primary, "a": c.a, "b": c.b,
                   "runs": list(slot["by_run"]), **stats}
            if stats.get("status") == "ok":
                threshold = SESOI[c.kind]
                row["practically_relevant"] = stats["sqrt_delta"] >= threshold
                row["sesoi"] = threshold
            if c.kind == "categorical":  # pre-registered sensitivity: INVALID as its own answer
                sens = compare(slot["draws"], c.kind, n_perm=n_perm, n_boot=200, seed=seed_c,
                               clusters=slot["clusters"] or None, invalid_as_category=True,
                               n_perm_binary=n_perm_binary, call_clusters=slot["call_clusters"] or None)
                row["sensitivity_invalid_as_category"] = {k: sens.get(k) for k in
                                                          ("n_items", "delta", "p", "status")}
            row["by_run"] = []
            for run_name, comp in slot["by_run"].items():
                part = compare(comp.draws, comp.kind, n_perm=2000, n_boot=500,
                               seed=stable_seed(seed, "byrun", backend, name, run_name) % 2**32,
                               clusters=comp.clusters, call_clusters=comp.call_clusters)
                row["by_run"].append({"run": run_name, **{k: part.get(k) for k in
                                                          ("n_items", "n_excluded", "delta", "p", "status")}})
            positive = [r for r in row["by_run"] if r.get("delta") is not None and r["delta"] > 0]
            row["runs_with_positive_delta"] = f"{len(positive)}/{len(row['by_run'])}"
            rows.append(row)
        measured = {row["hypothesis"] for row in rows}
        for hyp in PRIMARY_HYPOTHESES + SECONDARY_HYPOTHESES:
            if hyp not in measured:
                rows.append({"comparison": SLOT_NAMES[hyp], "hypothesis": hyp, "primary": hyp.startswith("H"),
                             "status": "not measured", "n_items": 0, "n_excluded": 0, "runs": []})
        for family, slots_ in ((True, PRIMARY_HYPOTHESES), (False, SECONDARY_HYPOTHESES)):
            idx = [i for i, row in enumerate(rows) if row["primary"] is family]
            assert sorted(rows[i]["hypothesis"] for i in idx) == sorted(slots_), (backend, family)
            pvals = []
            for i in idx:
                if "p" not in rows[i]:
                    rows[i]["holm_note"] = f"{rows[i]['status']}: entered the Holm family as p = 1"
                pvals.append(rows[i].get("p", 1.0))
            for i, p_adj in zip(idx, holm(pvals)):
                rows[i]["p_holm"] = p_adj
        rows.sort(key=lambda r: (r["primary"] is not True, r["primary"] is None, r["hypothesis"]))
        u = usage[backend]
        u["invalid_rate"] = u["invalid_draws"] / u["draws"] if u["draws"] else math.nan
        results["backends"][backend] = {"comparisons": rows, "usage": dict(u),
                                        "diagnostics": diagnostics.get(backend, {}),
                                        "parsing": parsing.get(backend, {}),
                                        "missing_runs": missing.get(backend, [])}
    return results


def primary_arms_for(results: dict[str, Any], requested: list[str] | None) -> list[str]:
    """The arms for the generality verdict. Named arms must all be present. By default, the three
    pre-registered primary arms if the runs hold them, and none if they hold none of them."""
    present = set(results["backends"])
    if requested is not None:
        unknown = [a for a in requested if a not in present]
        if unknown:
            raise SystemExit(f"--primary-arms {unknown} have no logs in these runs (models: {sorted(present)})")
        return list(requested)
    found = [a for a in DEFAULT_PRIMARY_ARMS if a in present]
    if found and len(found) != len(DEFAULT_PRIMARY_ARMS):
        raise SystemExit(f"only {found} of the primary arms {list(DEFAULT_PRIMARY_ARMS)} are in these runs; "
                         "name the arms with --primary-arms")
    return found


def generality(results: dict[str, Any], primary_arms: list[str], alpha: float = 0.05) -> dict[str, Any]:
    """Per primary identity: which primary arms support it after Holm, and the 2-of-3 verdict."""
    verdict = {}
    for hyp in PRIMARY_HYPOTHESES:
        supported = []
        for arm in primary_arms:
            rows = results["backends"].get(arm, {}).get("comparisons", [])
            row = next((r for r in rows if r["hypothesis"] == hyp), None)
            if row is not None and row.get("p_holm", 1.0) < alpha:
                supported.append(arm)
        name = next((r["comparison"] for b in results["backends"].values()
                     for r in b["comparisons"] if r["hypothesis"] == hyp), hyp)
        verdict[hyp] = {"identity": name, "supported_by": supported,
                        "broken_in_general": len(supported) >= 2}
    return verdict


def render_parsing(results: dict[str, Any]) -> str:
    """Per model and run: invalid rate and parse-rule mix for every condition (PREREGISTRATION.md §6.1)."""
    lines = ["# Parsing and exclusions", ""]
    for backend, res in results["backends"].items():
        lines += [f"## {backend}", "", "| Run | Condition | Invalid | Parse rules | ", "|---|---|---|---|"]
        for run, info in res.get("parsing", {}).items():
            for cond, rate in info["invalid_rate"].items():
                rules = info["parse_rules"].get(cond, {})
                total = sum(rules.values()) or 1
                mix = ", ".join(f"{k} {v / total:.1%}" for k, v in sorted(rules.items(), key=lambda kv: -kv[1]))
                lines.append(f"| {run} | {cond} | {rate:.2%} | {mix} |")
            lines.append(f"| {run} | (calls) | failed {info['failed_calls']}; other host {info['other_host_calls_excluded']}; "
                         f"logged {info['distinct_calls']} of {info['planned_calls']} planned "
                         f"| stop reasons {info['stop_reasons']} |")
        lines.append("")
    return "\n".join(lines)


def render(results: dict[str, Any], verdict: dict[str, Any] | None = None) -> str:
    runs = ", ".join(f"{r['run']} ({r['dataset']}, {r['style']})" for r in results["runs"])
    lines = ["# Pooled results", "", f"Runs: {runs}", ""]
    if verdict:
        lines += ["| Hyp. | Identity | Supported by (Holm p < 0.05) | Broken in general (>= 2 of 3 primary arms) |",
                  "|---|---|---|---|"]
        for hyp, v in verdict.items():
            lines.append(f"| {hyp} | {v['identity']} | {', '.join(v['supported_by']) or 'none'} | "
                         f"{'yes' if v['broken_in_general'] else 'no'} |")
        lines.append("")
    for backend, res in results["backends"].items():
        u = res["usage"]
        served = {k.split(":", 1)[1]: v for k, v in u.items() if k.startswith("served_by:")}
        lines += [f"## {backend}", "",
                  f"Calls: {u.get('calls', 0):,}. Cost: ${u.get('cost', 0.0):.4f}. Invalid draws: "
                  f"{u.get('invalid_rate', float('nan')):.2%}. Served by: {served}; calls served by another host "
                  f"than the pinned one (excluded): {u.get('calls_other_host', 0)}. Planned calls never issued "
                  f"(an interrupted or capped run): {u.get('calls_not_issued', 0)}."
                  + (f" **Missing from runs: {res['missing_runs']}.**" if res.get("missing_runs") else ""), "",
                  "p is the permutation p, or max(permutation, cluster-robust) for the batched identities "
                  "(H4, S2, S4). Holm uses p.", "",
                  "| Hyp. | Comparison | Items (excl.) | Within | Cross | Excess Δ [approx. 95% CI] | √Δ (≥ SESOI?) | "
                  "p (perm. ± MC) | p (cluster-robust) | p | p (Holm, model) | Stable viol. obs / exp | Runs with Δ > 0 | "
                  "Result-set gap between / within | INVALID as a category: Δ, p |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in res["comparisons"]:
            if row.get("status") != "ok":
                lines.append(f"| {row['hypothesis']} | {row['comparison']} | {row['n_items']} ({row['n_excluded']}) "
                             f"| {row.get('status')} | | | | | | | {row.get('p_holm', '')} | | | | |")
                continue
            lo, hi = row["delta_ci"]
            gap = (f"{row['count_gap_between']:.1f} / {row['count_gap_within']:.1f}"
                   if "count_gap_between" in row else "")
            relevant = "yes" if row.get("practically_relevant") else "no"
            p_holm = row.get("p_holm")
            p_cr = row.get("p_cluster_robust")
            sens = row.get("sensitivity_invalid_as_category") or {}
            sens_txt = (f"{sens['delta']:.4f}, {sens['p']:.5f}" if sens.get("status") == "ok" else sens.get("status", ""))
            lines.append(
                f"| {row['hypothesis']} | {row['comparison']} | {row['n_items']} ({row['n_excluded']}) | "
                f"{row['noise_within']:.4f} | {row['cross']:.4f} | {row['delta']:.4f} [{lo:.4f}, {hi:.4f}] | "
                f"{row['sqrt_delta']:.3f} ({relevant}) | {row['p_perm']:.5f} ± {row['p_perm_mc_se']:.5f} | "
                f"{'' if p_cr is None else f'{p_cr:.5f}'} | {row['p']:.5f} | "
                f"{'' if p_holm is None else f'{p_holm:.5f}'} | {row['stable_violations']} / {row['stable_expected']:.2f} | "
                f"{row['runs_with_positive_delta']} | {gap} | {sens_txt} |")
        lines.append("")
        for run_name, diag in res["diagnostics"].items():
            fi, bc = diag["fusion_incoherence"], diag["batch_collapse"]
            parts = []
            if fi:
                parts.append(f"fusion incoherence {fi['conjunction_above_a_conjunct'] + fi['conjunction_below_both_conjuncts']}"
                             f"/{fi['rows_with_stable_answers']} stable rows")
            if bc:
                parts.append(f"collapsed batch calls {bc['collapsed']}/{bc['calls']}")
            if diag["independent_subset"]:
                tests = diag["independent_subset"]["tests"]
                parts.append("independent-column p: " + ", ".join(
                    f"{k} {v.get('p_perm', float('nan')):.3f}" for k, v in tests.items()))
            if parts:
                lines.append(f"- {run_name}: " + "; ".join(parts))
        lines.append("")
    return "\n".join(lines)
