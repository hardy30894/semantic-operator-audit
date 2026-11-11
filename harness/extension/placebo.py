"""Addendum G: rewording placebos and a three-valued prompt for filter_algebra, per run and pooled.

Items are the rows of filter_algebra. Every comparison uses semaudit's own per-item estimator,
permutation test and bootstrap (`semaudit.stats.compare`), as the base identities do, and every paired
contrast uses extension/inference.py's `paired_inference` (mean of the per-item differences, standard
error from their variance across items, two-sided t-test with n - 1 df, t interval, and an item bootstrap
interval). Per arm:

- **Base identities (reference):** H2 (negation), H3 (fusion) and S3 (conjunct order) of the base segment,
  with semaudit's definitions (`semaudit.identities.comparisons`).
- **Rewording comparisons:** R_para1 = base F(p) vs reword F(p_para1); R_para2 = base F(p) vs F(p_para2);
  R_tmpl = base F(p) vs F(p_tmpl); R_order = base F(p) vs F(p_order); R_pqpara = base F(p and q) vs
  F(pq_para). Holm over these five.
- **Logic against rewording, paired per row (confirmatory):** C1 = Δ_H2,i - (Δ_para1,i + Δ_para2,i)/2 and
  C2 = Δ_H3,i - Δ_pqpara,i; Holm over C1 and C2. **Secondary** (Holm over these four): Δ_H2 - Δ_tmpl,
  Δ_H2 - Δ_order, Δ_S3 - Δ_pqpara, Δ_H3 - Δ_S3. A positive difference means the logical rewrite moves
  answers more than the meaning-preserving rewording does.
- **Three-valued negation and fusion:** U_H2 = unknown F3(p) vs NOT3 F3(not p); U_H3 = unknown
  F3(p and q) vs AND3(F3(p), F3(q)), with Kleene's NOT3 and AND3 (`not3`, `and3`; INVALID propagates as in
  semaudit's _not and _and). The kernel compares distributions over {True, False, UNKNOWN}. Holm over the
  two. **Paired contrasts:** Δ_H2 - Δ_U_H2 and Δ_H3 - Δ_U_H3, two-sided, Holm over the two.
- **Descriptive:** the share of UNKNOWN answers per condition; among rows whose base answers are a stable
  double rejection (all k base F(p) draws False and all k base F(not p) draws False), the share where most
  of the unknown segment's F3(p) draws, or most of its F3(not p) draws, are UNKNOWN; for claims with a truth
  rule, accuracy with UNKNOWN as an abstention (coverage = answered / valid draws, accuracy among answered,
  and accuracy counting abstentions as wrong), over draws; INVALID rates and parse rules per condition.
- **Arm pairs:** an arm whose roster entry names `pairs_with` (a temperature-0 arm, a reasoning arm) is
  compared with that arm as extension/analysis.py compares reasoning arms: per identity, the paired per-row
  difference Δ_arm,i - Δ_reference,i (`analysis.paired_difference`, with the same seeds), Holm over H2, H3
  and S3. The rewording and three-valued comparisons are paired the same way, without Holm (exploratory).

**Pooling** (`pool_placebo`): items from all runs together, kept distinct per run, as semaudit pools
identities: the permutation test permutes draws within items; the bootstrap interval of each pooled mean
resamples items within each run (stratified by run), reported beside semaudit's own interval (which resamples
items across runs). Pooled contrasts and arm pairs: the mean of the per-item differences over all items, the
standard error from their variance, a two-sided t-test, and a run-stratified bootstrap interval. Per-run values
are reported alongside. Holm families as above, per arm; a test that cannot be computed enters its family as
p = 1 and is flagged.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from semaudit.core import INVALID, stable_seed
from semaudit.data import load_data
from semaudit.identities import Comparison, comparisons, draws
from semaudit.pool import SESOI, _check_logs
from semaudit.stats import compare, holm

from .accuracy import claim_truths
from .analysis import _item_deltas, paired_difference
from .inference import fmt_ci, paired_inference
from .logs import collect_ext, latest_attempts, load_config, load_design, read_compact, run_logs
from .mitigations import segment_design, segment_outputs
from .placebo_segments import REWORD_RESTATES
from .styles import UNKNOWN_ANSWER, parse_record_ext

FA = "filter_algebra"


# --- Kleene's three-valued logic ------------------------------------------------------------------

def not3(v: Any) -> Any:
    """Kleene's NOT: True and False swap, UNKNOWN stays UNKNOWN; INVALID stays INVALID (semaudit's _not)."""
    if v == INVALID:
        return INVALID
    if v == UNKNOWN_ANSWER:
        return UNKNOWN_ANSWER
    return not v


def and3(x: Any, y: Any) -> Any:
    """Kleene's AND: INVALID if either is INVALID (semaudit's _and); else False if either is False, True if
    both are True, otherwise UNKNOWN."""
    if INVALID in (x, y):
        return INVALID
    a, b = (UNKNOWN_ANSWER if v == UNKNOWN_ANSWER else bool(v) for v in (x, y))
    if a is False or b is False:
        return False
    if a is True and b is True:
        return True
    return UNKNOWN_ANSWER


# --- the comparisons ------------------------------------------------------------------------------

BASE = {"negation": "H2", "fusion": "H3", "conjunct_order": "S3"}
# (hypothesis, comparison, base condition, reword condition, label of B)
REWORD = [("R_para1", "reword_para1", "p", "p_para1", "F(p), paraphrase 1"),
          ("R_para2", "reword_para2", "p", "p_para2", "F(p), paraphrase 2"),
          ("R_tmpl", "reword_template", "p", "p_tmpl", "F(p), other prompt template"),
          ("R_order", "reword_field_order", "p", "p_order", "F(p), fields in reversed order"),
          ("R_pqpara", "reword_conjunction", "p_and_q", "pq_para", "F(p and q), paraphrase")]
THREE = {"unknown_negation": "U_H2", "unknown_fusion": "U_H3"}
# (name, family, [(weight, comparison)], definition)
CONTRASTS = [
    ("C1", "confirmatory", [(1.0, "negation"), (-0.5, "reword_para1"), (-0.5, "reword_para2")],
     "Δ_H2 - (Δ_para1 + Δ_para2)/2"),
    ("C2", "confirmatory", [(1.0, "fusion"), (-1.0, "reword_conjunction")], "Δ_H3 - Δ_pqpara"),
    ("H2_vs_tmpl", "secondary", [(1.0, "negation"), (-1.0, "reword_template")], "Δ_H2 - Δ_tmpl"),
    ("H2_vs_order", "secondary", [(1.0, "negation"), (-1.0, "reword_field_order")], "Δ_H2 - Δ_order"),
    ("S3_vs_pqpara", "secondary", [(1.0, "conjunct_order"), (-1.0, "reword_conjunction")], "Δ_S3 - Δ_pqpara"),
    ("H3_vs_S3", "secondary", [(1.0, "fusion"), (-1.0, "conjunct_order")], "Δ_H3 - Δ_S3"),
    ("H2_vs_U_H2", "three-valued", [(1.0, "negation"), (-1.0, "unknown_negation")], "Δ_H2 - Δ_U_H2"),
    ("H3_vs_U_H3", "three-valued", [(1.0, "fusion"), (-1.0, "unknown_fusion")], "Δ_H3 - Δ_U_H3"),
]
COMPARISON_FAMILIES = {"rewording": [name for _, name, *_ in REWORD], "three-valued": list(THREE)}
CONTRAST_FAMILIES = ("confirmatory", "secondary", "three-valued")
BASE_CONDITIONS = ("p", "not_p", "q", "p_and_q", "q_and_p")


def _segments(design: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "ext" not in design:
        raise SystemExit("not an extension run (its design has no [ext] segments)")
    return {s["mitigation"]: s for s in design["ext"]["segments"]}


def is_placebo_design(design: dict[str, Any]) -> bool:
    segs = {s["mitigation"] for s in design.get("ext", {}).get("segments", [])}
    return "none" in segs and FA in design and bool(segs & {"reword", "unknown"})


def arm_comparisons(out: dict, design: dict[str, Any]) -> dict[str, Comparison]:
    """Addendum G's comparisons for one arm's outputs in one run: the base segment's H2, H3 and S3, the five
    rewording comparisons and the two three-valued ones (for the segments the run has)."""
    segs = _segments(design)
    base_seg = segs["none"]
    base_out = segment_outputs(out, base_seg)
    comps = {c.name: c for c in comparisons(base_out, segment_design(design, base_seg)) if c.name in BASE}
    rows = design[FA]["rows"]
    k = base_seg["repeats"]
    if "reword" in segs:
        r_out, kr = segment_outputs(out, segs["reword"]), segs["reword"]["repeats"]
        for hyp, name, a_cond, b_cond, label in REWORD:
            comps[name] = Comparison(name, hyp, None, "categorical", f"F({a_cond.replace('_and_', ' and ')})", label,
                                     {r: (draws(base_out, FA, a_cond, r, k), draws(r_out, FA, b_cond, r, kr))
                                      for r in rows})
    if "unknown" in segs:
        u_out, ku = segment_outputs(out, segs["unknown"]), segs["unknown"]["repeats"]

        def d(cond: str, r: str) -> list[Any]:
            return draws(u_out, FA, cond, r, ku)

        comps["unknown_negation"] = Comparison(
            "unknown_negation", "U_H2", None, "categorical", "F3(p)", "NOT3 F3(not p)",
            {r: (d("p", r), [not3(v) for v in d("not_p", r)]) for r in rows})
        comps["unknown_fusion"] = Comparison(
            "unknown_fusion", "U_H3", None, "categorical", "F3(p and q)", "AND3(F3(p), F3(q))",
            {r: (d("p_and_q", r), [and3(x, y) for x, y in zip(d("p", r), d("q", r))]) for r in rows})
    return comps


def _seed_label(backend: str, name: str) -> str:
    """The seed label of a comparison: as extension/analysis.py labels the base segment's comparisons, so the
    base identities here equal `python3 -m extension analyze`'s base rows."""
    segment = "base" if name in BASE else "reword" if name.startswith("reword_") else "unknown"
    return f"{backend}/{segment}"


def _comparison_row(comp: Comparison, stats: dict[str, Any]) -> dict[str, Any]:
    row = {"comparison": comp.name, "hypothesis": comp.hypothesis, "a": comp.a, "b": comp.b, **stats}
    if stats.get("status") == "ok":
        row["practically_relevant"] = stats["sqrt_delta"] >= SESOI["categorical"]
        ci = stats.get("delta_ci")
        if ci and ci[0] is not None:  # the minimum-effect reading: the interval's lower bound clears the SESOI
            row["min_effect"] = ci[0] >= SESOI["categorical"] ** 2
    return row


def apply_holm(rows: list[dict[str, Any]], p_key: str) -> None:
    """Holm-adjusted p (`p_holm`) over `rows`, in place; a row without a usable p enters as p = 1, flagged."""
    ps = []
    for row in rows:
        p = row.get(p_key) if row.get("status") == "ok" else None
        if p is None:
            row["holm_note"] = f"{row.get('status', 'not measured')}: entered the Holm family as p = 1"
            p = 1.0
        ps.append(p)
    for row, adjusted in zip(rows, holm(ps)):
        row["p_holm"] = adjusted


def stratified_bootstrap_ci(values: np.ndarray, strata: np.ndarray, n_boot: int, rng: np.random.Generator,
                            alpha: float = 0.05) -> list[float]:
    """Percentile interval for the mean over all items, resampling items within each stratum (run)."""
    values = np.asarray(values, dtype=float)
    _, inverse = np.unique(np.asarray(strata), return_inverse=True)
    sums = np.zeros(n_boot)
    for g in range(int(inverse.max()) + 1):
        v = values[inverse == g]
        n = len(v)
        chunk = max(1, int(5e6 // max(1, n)))
        for start in range(0, n_boot, chunk):
            m = min(chunk, n_boot - start)
            sums[start:start + m] += v[rng.integers(0, n, size=(m, n))].sum(axis=1)
    lo, hi = np.quantile(sums / len(values), [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def contrast_values(deltas: dict[str, dict[str, float]], terms: list[tuple[float, str]]) -> dict[str, float]:
    """Σ weight × Δ_comparison,i for every item that all the terms measured."""
    if any(name not in deltas for _, name in terms):
        return {}
    items = set.intersection(*(set(deltas[name]) for _, name in terms))
    return {i: sum(w * deltas[name][i] for w, name in terms) for i in sorted(items)}


def contrast_row(name: str, family: str, terms: list[tuple[float, str]], definition: str,
                 deltas: dict[str, dict[str, float]], n_boot: int, seed: int,
                 strata: dict[str, str] | None = None) -> dict[str, Any]:
    values = contrast_values(deltas, terms)
    items = sorted(values)
    row: dict[str, Any] = {"contrast": name, "family": family, "definition": definition,
                           "terms": [[w, c] for w, c in terms], "n_items": len(items)}
    if len(items) < 2:
        row["status"] = "insufficient data"
        return row
    d = np.array([values[i] for i in items])
    rng = np.random.default_rng(seed)
    row.update(paired_inference(d, [np.array(items)], n_boot, rng))
    row["term_means"] = {c: float(np.mean([deltas[c][i] for i in items])) for _, c in terms}
    if strata is not None:
        row["difference_ci_bootstrap_stratified"] = stratified_bootstrap_ci(d, np.array([strata[i] for i in items]),
                                                                            n_boot, rng)
    return row


# --- descriptive tallies --------------------------------------------------------------------------

def _tally(values: list[Any], truth: dict[str, bool] | None, items: list[str]) -> Counter:
    t: Counter = Counter()
    for item, v in zip(items, values):
        t["draws"] += 1
        if v == INVALID:
            t["invalid"] += 1
            continue
        if v == UNKNOWN_ANSWER:
            t["unknown"] += 1
        else:
            t["true" if v else "false"] += 1
            if truth is not None and item in truth:
                t["answered_with_truth"] += 1
                t["correct"] += bool(v) == truth[item]
        if truth is not None and item in truth:
            t["valid_with_truth"] += 1
    return t


def tallies(out: dict, design: dict[str, Any], truths: dict[str, dict[str, bool]],
            records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Counts that add up across runs: answers per segment and condition (with accuracy where the claim has a
    truth), stable double rejections, and parse rules."""
    segs = _segments(design)
    rows = design[FA]["rows"]
    conditions: dict[str, tuple[dict[str, Any], tuple[str, ...]]] = {"base": (segs["none"], BASE_CONDITIONS)}
    for kind in ("reword", "unknown"):
        if kind in segs:
            conditions[kind] = (segs[kind], tuple(segs[kind]["conditions"]))
    answers: dict[str, Counter] = {}
    for seg_name, (seg, conds) in conditions.items():
        seg_out = segment_outputs(out, seg)
        k = seg["repeats"]
        for cond in conds:
            claim = REWORD_RESTATES[cond] if seg_name == "reword" else cond
            values, items = [], []
            for r in rows:
                vs = draws(seg_out, FA, cond, r, k)
                values += vs
                items += [r] * len(vs)
            answers[f"{seg_name}/{cond}"] = _tally(values, truths.get(claim), items)
    double: Counter = Counter()
    if "unknown" in segs:
        base_out, u_out = segment_outputs(out, segs["none"]), segment_outputs(out, segs["unknown"])
        k, ku = segs["none"]["repeats"], segs["unknown"]["repeats"]
        double.update(dict.fromkeys(("stable_double_rejections", "majority_unknown_p", "majority_unknown_not_p",
                                     "majority_unknown_p_or_not_p", "majority_unknown_both"), 0))
        double["rows"] = len(rows)
        for r in rows:
            if all(v is False for v in draws(base_out, FA, "p", r, k) + draws(base_out, FA, "not_p", r, k)):
                double["stable_double_rejections"] += 1
                up = sum(v == UNKNOWN_ANSWER for v in draws(u_out, FA, "p", r, ku)) * 2 > ku
                un = sum(v == UNKNOWN_ANSWER for v in draws(u_out, FA, "not_p", r, ku)) * 2 > ku
                double["majority_unknown_p"] += up
                double["majority_unknown_not_p"] += un
                double["majority_unknown_p_or_not_p"] += up or un
                double["majority_unknown_both"] += up and un
    rules: dict[str, Counter] = {}
    if records is not None:
        prefixes = {seg["prefix"]: name for name, (seg, _) in conditions.items()}
        for rec in latest_attempts(records).values():
            if rec.get("error") or rec.get("text") is None:
                continue
            identity = rec["identity"]
            prefix = identity[: identity.index("/") + 1] if "/" in identity else ""
            if prefix not in prefixes or identity[len(prefix):] != FA:
                continue
            rules.setdefault(f"{prefixes[prefix]}/{rec['condition']}", Counter()).update(parse_record_ext(rec)[1])
    return {"answers": answers, "double_rejection": double, "parse_rules": rules}


def summarize(t: dict[str, Any]) -> dict[str, Any]:
    """Shares from summed tallies."""
    def share(a: float, b: float) -> float | None:
        return a / b if b else None

    answers = {}
    for key, c in sorted(t["answers"].items()):
        valid = c["draws"] - c["invalid"]
        answered = c["true"] + c["false"]
        row = {"draws": c["draws"], "invalid_rate": share(c["invalid"], c["draws"]),
               "unknown_share": share(c["unknown"], valid), "true_share": share(c["true"], valid)}
        if c["valid_with_truth"]:
            row.update(coverage=share(c["answered_with_truth"], c["valid_with_truth"]),
                       accuracy_answered=share(c["correct"], c["answered_with_truth"]),
                       accuracy_abstain_wrong=share(c["correct"], c["valid_with_truth"]))
        rules = t["parse_rules"].get(key)
        if rules:
            total = sum(rules.values())
            row["parse_rules"] = {r: n / total for r, n in sorted(rules.items(), key=lambda kv: -kv[1])}
        answers[key] = row
    d = t["double_rejection"]
    double = None
    if d.get("rows"):
        n = d.get("stable_double_rejections", 0)
        double = {"rows": d["rows"], "stable_double_rejections": n,
                  "share_majority_unknown_p_or_not_p": share(d["majority_unknown_p_or_not_p"], n),
                  "share_majority_unknown_p": share(d["majority_unknown_p"], n),
                  "share_majority_unknown_not_p": share(d["majority_unknown_not_p"], n),
                  "share_majority_unknown_both": share(d["majority_unknown_both"], n)}
    return {"answers": answers, "double_rejection": double}


def _sum(a: Counter, b: Counter) -> Counter:
    total = Counter(a)
    total.update(b)  # unlike a + b, keeps keys whose count is 0
    return total


def _add(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = {"answers": dict(a["answers"]), "double_rejection": _sum(a["double_rejection"], b["double_rejection"]),
           "parse_rules": dict(a["parse_rules"])}
    for part in ("answers", "parse_rules"):
        for key, c in b[part].items():
            out[part][key] = _sum(out[part].get(key, Counter()), c)
    return out


# --- one run --------------------------------------------------------------------------------------

def _truths(config: dict[str, Any]) -> dict[str, dict[str, bool]]:
    return claim_truths(load_data(config["data"])) if config.get("data") else {}


def analyze_arm(records: list[dict[str, Any]], design: dict[str, Any], truths: dict[str, dict[str, bool]],
                n_perm: int, n_boot: int, seed: int, backend: str) -> dict[str, Any]:
    out = collect_ext(records)
    comps = arm_comparisons(out, design)
    rows = {name: _comparison_row(comp, compare(comp.draws, comp.kind, n_perm=n_perm, n_boot=n_boot,
                                                seed=stable_seed(seed, _seed_label(backend, name), name) % 2**32,
                                                clusters=comp.clusters, call_clusters=comp.call_clusters))
            for name, comp in comps.items()}
    for family, names in COMPARISON_FAMILIES.items():
        apply_holm([rows[n] if n in rows else rows.setdefault(n, {"comparison": n, "status": "not measured"})
                    for n in names], "p")
    deltas = {name: _item_deltas(comp) for name, comp in comps.items()}
    contrasts = [contrast_row(name, family, terms, definition, deltas, n_boot,
                              stable_seed(seed, backend, "placebo-contrast", name) % 2**32)
                 for name, family, terms, definition in CONTRASTS]
    for family in CONTRAST_FAMILIES:
        apply_holm([c for c in contrasts if c["family"] == family], "p_two_sided")
    t = tallies(out, design, truths, records)
    return {"comparisons": rows, "contrasts": contrasts, "descriptive": summarize(t),
            "_deltas": deltas, "_comps": comps}


def arm_pair(res: dict[str, Any], ref: dict[str, Any], backend: str, partner: str, n_boot: int,
             seed: int) -> dict[str, Any]:
    """Δ_arm,i - Δ_reference,i per comparison, as extension/analysis.py pairs reasoning arms (same seeds);
    Holm over H2, H3 and S3."""
    effects, exploratory = {}, {}
    for name, deltas in res["_deltas"].items():
        base = ref["_deltas"].get(name)
        if base is None:
            continue
        e = paired_difference(base, deltas, ref["_comps"].get(name), n_boot,
                              stable_seed(seed, backend, partner, name) % 2**32)
        e["hypothesis"] = res["_comps"][name].hypothesis
        (effects if name in BASE else exploratory)[name] = e
    for name in BASE:
        effects.setdefault(name, {"n_items": 0, "status": "not measured", "hypothesis": BASE[name]})
    apply_holm([effects[n] for n in BASE], "p_two_sided")
    return {"reference": partner, "effects": effects, "exploratory": exploratory}


def analyze_placebo_run(run_dir: Path, n_perm: int = 5000, n_boot: int = 2000, seed: int = 0,
                        backends: list[str] | None = None) -> dict[str, Any]:
    design = load_design(run_dir)
    if not is_placebo_design(design):
        raise SystemExit(f"{run_dir} has no reword or unknown segment (not an Addendum G design)")
    config = load_config(run_dir)
    truths = _truths(config)
    roster = {b["name"]: b for b in config.get("backends", [])}
    results: dict[str, Any] = {"run": str(run_dir), "dataset": design.get("dataset"), "rows": len(design[FA]["rows"]),
                               "repeats": design["repeats"],
                               "segments": [s["name"] for s in design["ext"]["segments"]],
                               "claims_with_truth": sorted(truths), "n_perm": n_perm, "n_boot": n_boot,
                               "backends": {}}
    for backend, path in run_logs(run_dir, backends):
        results["backends"][backend] = analyze_arm(read_compact(path), design, truths, n_perm, n_boot, seed, backend)
    results["arm_pairs"] = {}
    for backend, res in results["backends"].items():
        partner = (roster.get(backend) or {}).get("pairs_with")
        if partner and partner in results["backends"]:
            results["arm_pairs"][backend] = arm_pair(res, results["backends"][partner], backend, partner, n_boot, seed)
    for res in results["backends"].values():
        res.pop("_deltas")
        res.pop("_comps")
    return results


# --- pooled over runs -----------------------------------------------------------------------------

def _check_runs(run_dirs: list[Path]) -> list[dict[str, Any]]:
    infos, seen = [], {}
    for run_dir in run_dirs:
        raw = (run_dir / "design.json").read_text(encoding="utf-8")
        design = json.loads(raw)
        if not is_placebo_design(design):
            raise SystemExit(f"{run_dir} has no reword or unknown segment (not an Addendum G design)")
        sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if sha in seen:
            raise SystemExit(f"runs {seen[sha]} and {run_dir.name} have the same design; pooling both would count "
                             "every item twice")
        seen[sha] = run_dir.name
        infos.append({"run": run_dir.name, "dataset": design.get("dataset"), "design_sha256": sha,
                      "base_style": design["ext"]["base_style"], "rows": len(design[FA]["rows"]),
                      "segments": [s["name"] for s in design["ext"]["segments"]], "design": design})
    styles = {i["base_style"] for i in infos}
    if len(styles) > 1:
        raise SystemExit(f"runs mix base prompt styles {sorted(styles)}; pool each style separately")
    if len({i["run"] for i in infos}) != len(infos):
        raise SystemExit("two runs share a directory name; item ids would collide")
    datasets = Counter(i["dataset"] for i in infos)
    for i in infos:
        i["warning"] = (f"dataset {i['dataset']} appears in {datasets[i['dataset']]} runs; items may repeat records"
                        if datasets[i["dataset"]] > 1 else None)
    return infos


def _prefixed(mapping: dict[str, Any], run: str) -> dict[str, Any]:
    return {f"{run}/{item}": value for item, value in mapping.items()}


def _pooled_comparison(name: str, by_run: dict[str, Comparison], backend: str, n_perm: int, n_boot: int,
                       n_perm_binary: int, seed: int) -> dict[str, Any]:
    first = next(iter(by_run.values()))
    pooled = {}
    for run, comp in by_run.items():
        pooled.update(_prefixed(comp.draws, run))
    seed_c = stable_seed(seed, "placebo-pool", backend, name) % 2**32
    stats = compare(pooled, first.kind, n_perm=n_perm, n_boot=n_boot, seed=seed_c, n_perm_binary=n_perm_binary)
    row = _comparison_row(first, stats)
    row["runs"] = list(by_run)
    if stats.get("status") == "ok":
        row["delta_ci_unstratified"] = row["delta_ci"]
        deltas = {}
        for run, comp in by_run.items():
            deltas.update(_prefixed(_item_deltas(comp), run))
        items = sorted(deltas)
        lo, hi = stratified_bootstrap_ci(np.array([deltas[i] for i in items]),
                                         np.array([i.split("/", 1)[0] for i in items]), n_boot,
                                         np.random.default_rng(stable_seed(seed, "placebo-strata", backend, name) % 2**32))
        row["delta_ci_stratified_raw"] = [lo, hi]
        row["delta_ci_stratified"] = [max(lo, 0.0), max(hi, 0.0)]
        sens = compare(pooled, first.kind, n_perm=n_perm, n_boot=200, seed=seed_c, invalid_as_category=True,
                       n_perm_binary=n_perm_binary)
        row["sensitivity_invalid_as_category"] = {k: sens.get(k) for k in ("n_items", "delta", "p", "status")}
    row["by_run"] = []
    for run, comp in by_run.items():
        part = compare(comp.draws, comp.kind, n_perm=2000, n_boot=500,
                       seed=stable_seed(seed, "placebo-byrun", backend, name, run) % 2**32)
        row["by_run"].append({"run": run, **{k: part.get(k) for k in ("n_items", "n_excluded", "delta", "sqrt_delta",
                                                                      "p", "stable_violations", "status")}})
    positive = [r for r in row["by_run"] if r.get("delta") is not None and r["delta"] > 0]
    row["runs_with_positive_delta"] = f"{len(positive)}/{len(row['by_run'])}"
    return row


def _by_run_contrast(values_by_run: dict[str, dict[str, float]], n_boot: int, seed: int,
                     label: tuple[Any, ...]) -> list[dict[str, Any]]:
    out = []
    for run, values in values_by_run.items():
        items = sorted(values)
        entry: dict[str, Any] = {"run": run, "n_items": len(items)}
        if len(items) >= 2:
            res = paired_inference(np.array([values[i] for i in items]), [np.array(items)], n_boot,
                                   np.random.default_rng(stable_seed(seed, *label, run) % 2**32))
            entry.update({k: res.get(k) for k in ("difference", "difference_ci", "p_two_sided", "status")})
        else:
            entry["status"] = "insufficient data"
        out.append(entry)
    return out


def pool_placebo(run_dirs: list[Path], n_perm: int = 10000, n_boot: int = 10000, seed: int = 0,
                 backends: list[str] | None = None, n_perm_binary: int = 100000,
                 allow_missing: bool = False) -> dict[str, Any]:
    infos = _check_runs(run_dirs)
    missing = _check_logs(run_dirs, backends, allow_missing)
    comps: dict[str, dict[str, dict[str, Comparison]]] = {}  # backend -> run -> comparison -> Comparison
    counts: dict[str, dict[str, dict[str, Any]]] = {}  # backend -> run -> tallies
    partners: dict[str, set[str]] = {}
    for info, run_dir in zip(infos, run_dirs):
        design, run = info["design"], info["run"]
        config = load_config(run_dir)
        truths = _truths(config)
        for spec in config.get("backends", []):
            if spec.get("pairs_with"):
                partners.setdefault(spec["name"], set()).add(spec["pairs_with"])
        for backend, path in run_logs(run_dir, backends):
            records = read_compact(path)
            out = collect_ext(records)
            comps.setdefault(backend, {})[run] = arm_comparisons(out, design)
            counts.setdefault(backend, {})[run] = tallies(out, design, truths, records)
            del records, out
    bad = {b: sorted(p) for b, p in partners.items() if len(p) > 1}
    if bad:
        raise SystemExit(f"runs name different pairs_with partners for the same arm: {bad}")
    results: dict[str, Any] = {"runs": [{k: v for k, v in i.items() if k != "design"} for i in infos],
                               "n_perm": n_perm, "n_perm_binary": n_perm_binary, "n_boot": n_boot, "backends": {}}
    deltas_by_backend: dict[str, dict[str, dict[str, float]]] = {}
    for backend, by_run in comps.items():
        names = []
        for run_comps in by_run.values():
            names += [n for n in run_comps if n not in names]
        rows = {}
        for name in names:
            per_run = {run: rc[name] for run, rc in by_run.items() if name in rc}
            rows[name] = _pooled_comparison(name, per_run, backend, n_perm, n_boot, n_perm_binary, seed)
        for family, members in COMPARISON_FAMILIES.items():
            apply_holm([rows.setdefault(n, {"comparison": n, "status": "not measured", "runs": []}) for n in members],
                       "p")
        deltas: dict[str, dict[str, float]] = {}
        per_run_deltas: dict[str, dict[str, dict[str, float]]] = {}
        for run, run_comps in by_run.items():
            for name, comp in run_comps.items():
                d = _item_deltas(comp)
                per_run_deltas.setdefault(run, {})[name] = d
                deltas.setdefault(name, {}).update(_prefixed(d, run))
        deltas_by_backend[backend] = deltas
        strata = {item: item.split("/", 1)[0] for d in deltas.values() for item in d}
        contrasts = []
        for name, family, terms, definition in CONTRASTS:
            row = contrast_row(name, family, terms, definition, deltas, n_boot,
                               stable_seed(seed, "placebo-pool", backend, "contrast", name) % 2**32, strata)
            row["by_run"] = _by_run_contrast({run: contrast_values(d, terms) for run, d in per_run_deltas.items()},
                                             min(n_boot, 2000), seed, ("placebo-byrun", backend, "contrast", name))
            contrasts.append(row)
        for family in CONTRAST_FAMILIES:
            apply_holm([c for c in contrasts if c["family"] == family], "p_two_sided")
        total = None
        for t in counts[backend].values():
            total = t if total is None else _add(total, t)
        results["backends"][backend] = {
            "comparisons": rows, "contrasts": contrasts,
            "descriptive": summarize(total),
            "descriptive_by_run": {run: summarize(t) for run, t in counts[backend].items()},
            "missing_runs": missing.get(backend, [])}
    results["arm_pairs"] = {}
    for backend in comps:
        partner = next(iter(partners.get(backend, [])), None)
        if partner is None or partner not in comps:
            continue
        mine, ref = deltas_by_backend[backend], deltas_by_backend[partner]
        effects, exploratory = {}, {}
        for name in mine:
            if name not in ref:
                continue
            e = paired_difference(ref[name], mine[name], None, n_boot,
                                  stable_seed(seed, "placebo-pool", backend, partner, name) % 2**32)
            items = sorted(set(ref[name]) & set(mine[name]))
            if e.get("status") == "ok":
                d = np.array([mine[name][i] - ref[name][i] for i in items])
                e["difference_ci_bootstrap_stratified"] = stratified_bootstrap_ci(
                    d, np.array([i.split("/", 1)[0] for i in items]), n_boot,
                    np.random.default_rng(stable_seed(seed, "placebo-pool-strata", backend, partner, name) % 2**32))
            by_run_values: dict[str, dict[str, float]] = {}
            for item in items:
                run = item.split("/", 1)[0]
                by_run_values.setdefault(run, {})[item] = mine[name][item] - ref[name][item]
            e["by_run"] = _by_run_contrast(by_run_values, min(n_boot, 2000), seed,
                                           ("placebo-byrun", backend, partner, name))
            first = next(iter(comps[backend].values())).get(name)
            e["hypothesis"] = first.hypothesis if first else BASE.get(name)
            (effects if name in BASE else exploratory)[name] = e
        for name in BASE:
            effects.setdefault(name, {"n_items": 0, "status": "not measured", "hypothesis": BASE[name]})
        apply_holm([effects[n] for n in BASE], "p_two_sided")
        results["arm_pairs"][backend] = {"reference": partner, "effects": effects, "exploratory": exploratory}
    return results


# --- rendering ------------------------------------------------------------------------------------

def _f(x: Any, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def _p(x: Any) -> str:
    return "n/a" if x is None else f"{x:.4g}"


COMPARISON_HEAD = ["| Hyp. | Comparison (A vs B) | Items (excl.) | Within | Cross | Excess Δ [approx. 95% CI] | √Δ | "
                   "p (perm.) | p (Holm) | Stable viol. obs / exp | Result-set gap between / within |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]


def _comparison_line(row: dict[str, Any], pooled: bool = False) -> str:
    name = f"{row.get('comparison')}" + (f" ({row['a']} vs {row['b']})" if "a" in row else "")
    if row.get("status") != "ok":
        return (f"| {row.get('hypothesis', '')} | {name} | {row.get('n_items', 0)} ({row.get('n_excluded', 0)}) | "
                f"{row.get('status')} | | | | | {_p(row.get('p_holm'))} | | |")
    ci = row["delta_ci_stratified"] if pooled and "delta_ci_stratified" in row else row["delta_ci"]
    gap = (f"{row['count_gap_between']:.1f} / {row['count_gap_within']:.1f}" if "count_gap_between" in row else "")
    return (f"| {row['hypothesis']} | {name} | {row['n_items']} ({row['n_excluded']}) | {_f(row['noise_within'])} | "
            f"{_f(row['cross'])} | {_f(row['delta'])} [{_f(ci[0])}, {_f(ci[1])}] | {row['sqrt_delta']:.3f} | "
            f"{_p(row['p'])} | {_p(row.get('p_holm'))} | {row['stable_violations']} / {row['stable_expected']:.2f} | "
            f"{gap} |")


CONTRAST_HEAD = ["| Contrast | Definition | Family | Rows | Mean of each Δ | Difference [95% t CI] | p (two-sided) | "
                 "p (Holm, family) | Bootstrap 95% CI |", "|---|---|---|---|---|---|---|---|---|"]


def _contrast_line(row: dict[str, Any], pooled: bool = False) -> str:
    if row.get("status") != "ok":
        return (f"| {row['contrast']} | {row['definition']} | {row['family']} | {row['n_items']} | | "
                f"{row.get('status')} | | {_p(row.get('p_holm'))} | |")
    means = ", ".join(f"{c} {v:.4f}" for c, v in row["term_means"].items())
    boot = row.get("difference_ci_bootstrap_stratified") if pooled else row.get("difference_ci_bootstrap")
    return (f"| {row['contrast']} | {row['definition']} | {row['family']} | {row['n_items']} | {means} | "
            f"{row['difference']:+.4f} {fmt_ci(row['difference_ci'])} | {_p(row['p_two_sided'])} | "
            f"{_p(row.get('p_holm'))} | {fmt_ci(boot)} |")


PAIR_HEAD = ["| Hyp. | Comparison | Items | Δ reference | Δ this arm | Difference [95% t CI] | p (two-sided) | "
             "p (Holm over H2, H3, S3) | Bootstrap 95% CI |", "|---|---|---|---|---|---|---|---|---|"]


def _pair_line(name: str, e: dict[str, Any], pooled: bool = False) -> str:
    if e.get("status") != "ok":
        return (f"| {e.get('hypothesis', '')} | {name} | {e.get('n_items')} | {_f(e.get('delta_base'))} | "
                f"{_f(e.get('delta_other'))} | {e.get('status')} | | {_p(e.get('p_holm'))} | |")
    boot = e.get("difference_ci_bootstrap_stratified") if pooled else e.get("difference_ci_bootstrap")
    return (f"| {e.get('hypothesis', '')} | {name} | {e['n_items']} | {_f(e['delta_base'])} | {_f(e['delta_other'])} | "
            f"{e['difference']:+.4f} {fmt_ci(e['difference_ci'])} | {_p(e['p_two_sided'])} | "
            f"{_p(e.get('p_holm')) if 'p_holm' in e else 'exploratory'} | {fmt_ci(boot)} |")


def _descriptive_lines(desc: dict[str, Any]) -> list[str]:
    lines = ["| Segment/condition | Draws | INVALID | UNKNOWN (share of valid) | True (share of valid) | Coverage | "
             "Accuracy, answered | Accuracy, abstentions wrong | Parse rules |", "|---|---|---|---|---|---|---|---|---|"]
    for key, a in desc["answers"].items():
        rules = ", ".join(f"{r} {v:.1%}" for r, v in (a.get("parse_rules") or {}).items())
        lines.append(f"| {key} | {a['draws']} | {_f(a['invalid_rate'], 3)} | {_f(a['unknown_share'], 3)} | "
                     f"{_f(a['true_share'], 3)} | {_f(a.get('coverage'), 3)} | {_f(a.get('accuracy_answered'), 3)} | "
                     f"{_f(a.get('accuracy_abstain_wrong'), 3)} | {rules} |")
    d = desc.get("double_rejection")
    if d:
        lines += ["", f"Stable double rejections in the base segment (all base F(p) and F(not p) draws False): "
                      f"{d['stable_double_rejections']} of {d['rows']} rows. Among them, the share where most "
                      f"F3(p) or most F3(not p) draws are UNKNOWN: {_f(d['share_majority_unknown_p_or_not_p'], 3)} "
                      f"(F3(p) {_f(d['share_majority_unknown_p'], 3)}, F3(not p) "
                      f"{_f(d['share_majority_unknown_not_p'], 3)}, both {_f(d['share_majority_unknown_both'], 3)})."]
    return lines


INTRO = ("Items are filter_algebra's rows. Δ is semaudit's excess divergence (cross-condition disagreement minus "
         "within-condition disagreement), its p a permutation p within items. Rewording comparisons set the base "
         "segment's answers against meaning-preserving rewordings of the same claim (Holm over the five); the "
         "three-valued comparisons use Kleene's NOT3 and AND3 on True, False and UNKNOWN (Holm over the two). "
         "Contrasts are paired per row: a positive difference means the logical rewrite moves answers more than the "
         "rewording (or than the three-valued prompt's). Their p is a two-sided t-test with n - 1 df; Holm within each "
         "family (confirmatory: C1, C2; secondary: four; three-valued: two). The base identities are a reference "
         "without Holm.")


def _render_arm(res: dict[str, Any], pooled: bool) -> list[str]:
    rows = res["comparisons"]
    lines = ["### Base segment (reference)", "", *COMPARISON_HEAD]
    lines += [_comparison_line(rows[n], pooled) for n in BASE if n in rows]
    lines += ["", "### Rewording comparisons (Holm over the five)", "", *COMPARISON_HEAD]
    lines += [_comparison_line(rows[n], pooled) for n in COMPARISON_FAMILIES["rewording"]]
    lines += ["", "### Three-valued comparisons (Holm over the two)", "", *COMPARISON_HEAD]
    lines += [_comparison_line(rows[n], pooled) for n in COMPARISON_FAMILIES["three-valued"]]
    lines += ["", "### Paired contrasts", "", *CONTRAST_HEAD]
    lines += [_contrast_line(c, pooled) for c in res["contrasts"]]
    lines += ["", "### Descriptive", "", *_descriptive_lines(res["descriptive"]), ""]
    return lines


def render(results: dict[str, Any]) -> str:
    lines = [f"# Addendum G, one run: {results['run']}", "",
             f"Dataset {results['dataset']}, {results['rows']} rows, k = {results['repeats']}; segments "
             f"{', '.join(results['segments'])}; claims with a truth rule: "
             f"{', '.join(results['claims_with_truth']) or 'none'}. n_perm {results['n_perm']}, n_boot "
             f"{results['n_boot']}.", "", INTRO, ""]
    for backend, res in results["backends"].items():
        lines += [f"## {backend}", "", *_render_arm(res, pooled=False)]
    for backend, pair in results.get("arm_pairs", {}).items():
        lines += [f"## {backend} vs {pair['reference']} (paired per row)", "", *PAIR_HEAD]
        lines += [_pair_line(n, e) for n, e in pair["effects"].items()]
        lines += [_pair_line(n, e) for n, e in pair["exploratory"].items()]
        lines.append("")
    return "\n".join(lines)


def render_pooled(results: dict[str, Any]) -> str:
    runs = ", ".join(f"{r['run']} ({r['dataset']}, {r['rows']} rows)" for r in results["runs"])
    lines = ["# Addendum G, pooled over runs", "", f"Runs: {runs}. n_perm {results['n_perm']} "
             f"({results['n_perm_binary']} for binary comparisons), n_boot {results['n_boot']}.", "", INTRO,
             "Pooled: items from every run, kept distinct per run; the interval of Δ shown is the bootstrap stratified by "
             "run (the JSON also holds semaudit's unstratified interval), and contrasts and arm pairs pool the per-row "
             "differences of every run, with a run-stratified bootstrap interval.", ""]
    for backend, res in results["backends"].items():
        lines += [f"## {backend}", ""]
        if res.get("missing_runs"):
            lines += [f"**Missing from runs: {res['missing_runs']}.**", ""]
        lines += _render_arm(res, pooled=True)
        lines += ["Per run (Δ, p): " + "; ".join(
            f"{row['comparison']}: " + ", ".join(f"{b['run']} {_f(b.get('delta'))} ({_p(b.get('p'))})"
                                                 for b in row.get("by_run", []))
            for row in res["comparisons"].values() if row.get("by_run")), "",
                  "Per run, contrasts (difference, p): " + "; ".join(
                      f"{c['contrast']}: " + ", ".join(f"{b['run']} {_f(b.get('difference'))} ({_p(b.get('p_two_sided'))})"
                                                       for b in c.get("by_run", []))
                      for c in res["contrasts"]), ""]
    for backend, pair in results.get("arm_pairs", {}).items():
        lines += [f"## {backend} vs {pair['reference']} (paired per row, pooled)", "", *PAIR_HEAD]
        lines += [_pair_line(n, e, pooled=True) for n, e in pair["effects"].items()]
        lines += [_pair_line(n, e, pooled=True) for n, e in pair["exploratory"].items()]
        lines += ["", "Per run (difference, p): " + "; ".join(
            f"{n}: " + ", ".join(f"{b['run']} {_f(b.get('difference'))} ({_p(b.get('p_two_sided'))})"
                                 for b in e.get("by_run", []))
            for n, e in pair["effects"].items() if e.get("by_run")), ""]
    return "\n".join(lines)


def write(results: dict[str, Any], out_dir: Path, pooled: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "placebo.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out_dir / "placebo.md").write_text(render_pooled(results) if pooled else render(results), encoding="utf-8")
