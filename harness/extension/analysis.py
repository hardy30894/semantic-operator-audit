"""Effect of each mitigation on excess divergence and on cost.

For an extension run (extension.runner.run_ext), per backend:

- every segment's identity comparisons are computed with semaudit's own `comparisons` and `compare`
  on that segment's outputs: explain- and format-parsed answers for those segments, voted answers for
  a vote segment, NOT F(p) for not_p under negation normalization (fresh calls), and sym(a,b) for
  symmetric matching;
- a mitigation's effect on an identity is the paired difference Delta_mitigated - Delta_base over the
  items both segments measured (they share units by construction), with semaudit's grouping of the
  items (crossed N1/N2 batches for H4): the mean, a t interval and two-sided p from the
  cluster-robust standard error (primary), and a cluster bootstrap interval (secondary). See
  extension/inference.py;
- the explain segment is also compared with the format segment: base -> format is the effect of the
  answer format, format -> explain the effect of the instruction to reason;
- symmetric matching and negation normalization with reused base calls send no new requests: both
  plans then read the same draws, so the identity holds by construction. Their Delta is reported as 0
  (semaudit's Delta on identical draws would read -within/k), and their effect is minus the base
  segment's Delta, item by item;
- cost is counted from the logs: calls (every attempt, as billed), input, output and reasoning
  tokens, and USD (logged cost; for providers that bill no cost per call, tokens at roster prices),
  per segment, identity and condition, per measured item, with the ratio to the base segment.

Reasoning-on arms are backends of their own; an arm whose roster entry names `pairs_with` is
compared with that arm on the base segment, item by item, in the same way.

Addendum G's segments (reword, unknown) send conditions that semaudit's comparisons do not read; they
are analyzed by extension/placebo.py and skipped here. A base segment without COUNT requests
(base_count = false) has no H7 row.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from semaudit.clusters import transitivity_report
from semaudit.core import stable_seed
from semaudit.identities import Outputs, comparisons
from semaudit.stats import compare, encode, item_deltas

from .backends import record_cost, spec_prices
from .inference import comparison_ways, fmt_ci, paired_inference
from .logs import collect_ext, load_config, load_design, read_compact, run_logs
from .mitigations import negation_normalized_outputs, segment_design, segment_outputs, vote_outputs
from .placebo_segments import KINDS as PLACEBO_SEGMENTS

TARGETS = {"negation_normalized": ("negation",), "symmetric": ("symmetry",)}


def _item_deltas(comp: Any) -> dict[str, float]:
    items, a, b, _, _ = encode(comp.draws, comp.kind)
    if len(items) < 1 or a is None or a.shape[1] < 2 or b.shape[1] < 2:
        return {}
    return dict(zip(items, item_deltas(a, b, comp.kind)[0].tolist()))


def paired_difference(base: dict[str, float], other: dict[str, float], comp: Any, n_boot: int,
                      seed: int) -> dict[str, Any]:
    """other - base per shared item, grouped as semaudit groups the comparison's items."""
    items = sorted(set(base) & set(other))
    if len(items) < 2:
        return {"n_items": len(items), "status": "insufficient data"}
    d = np.array([other[i] - base[i] for i in items])
    res = paired_inference(d, comparison_ways(comp, items), n_boot, np.random.default_rng(seed))
    return {"n_items": len(items), "delta_base": float(np.mean([base[i] for i in items])),
            "delta_other": float(np.mean([other[i] for i in items])), **res}


def _comparison_rows(out: Outputs, design: dict[str, Any], n_perm: int, n_boot: int, seed: int,
                     label: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows, comps = {}, {}
    for comp in comparisons(out, design):
        stats = compare(comp.draws, comp.kind, n_perm=n_perm, n_boot=n_boot,
                        seed=stable_seed(seed, label, comp.name) % 2**32, clusters=comp.clusters,
                        call_clusters=comp.call_clusters)
        rows[comp.name] = {"hypothesis": comp.hypothesis, **{k: stats.get(k) for k in (
            "status", "n_items", "n_excluded", "noise_within", "cross", "delta", "delta_ci", "sqrt_delta", "p")}}
        comps[comp.name] = comp
    return rows, comps


def _segment_of(rec: dict[str, Any]) -> str:
    return (rec.get("context") or {}).get("segment", "base")


def cost_table(records: list[dict[str, Any]], prices: tuple[float, float] | None) -> dict[str, dict[str, Any]]:
    """{segment/identity and segment/identity/condition: calls, tokens, USD}, counting every logged
    attempt (failed attempts are billed or at least sent)."""
    table: dict[str, Counter] = {}
    for rec in records:
        identity = rec["identity"].split("/")[-1]
        usage = rec.get("usage") or {}
        for key in (f"{_segment_of(rec)}/{identity}", f"{_segment_of(rec)}/{identity}/{rec['condition']}"):
            c = table.setdefault(key, Counter())
            c["calls"] += 1
            c["failed_calls"] += bool(rec.get("error"))
            c["input_tokens"] += usage.get("input_tokens") or 0
            c["output_tokens"] += usage.get("output_tokens") or 0
            c["reasoning_tokens"] += usage.get("reasoning_tokens") or 0
            c["usd"] += record_cost(rec, prices)
    return {k: dict(v) for k, v in sorted(table.items())}


def _units(design: dict[str, Any], identity: str) -> int:
    units = design.get(identity, {})
    if identity == "symmetry":
        return len(units.get("pairs", []))
    if identity in ("filter_algebra", "batch", "idempotence"):
        return len(units.get("rows", []))
    if identity == "placement":
        return len(units.get("tuples", []))
    return 0


def _with_ratios(costs: dict[str, dict[str, Any]], design: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-item figures, and ratios to the base segment's same identity (or identity/condition)."""
    out = {}
    for key, c in costs.items():
        segment, identity, *condition = key.split("/")
        n = _units(design, identity) or 1
        row = {**c, "items": n, "calls_per_item": c["calls"] / n,
               "tokens_per_item": (c["input_tokens"] + c["output_tokens"] + c["reasoning_tokens"]) / n,
               "usd_per_item": c["usd"] / n}
        base = costs.get("/".join(["base", identity, *condition]))
        if base and segment != "base":
            for metric in ("calls", "input_tokens", "output_tokens", "usd"):
                row[f"{metric}_ratio"] = c[metric] / base[metric] if base[metric] else None
        out[key] = row
    return out


def _roster(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {b["name"]: b for b in config.get("backends", [])}


def _by_construction(name: str, rows: dict[str, Any], comp_name: str, base_deltas: dict[str, float],
                     how: str) -> dict[str, float]:
    rows[comp_name] = {"hypothesis": {"symmetry": "H1", "negation": "H2"}[comp_name], "status": "ok",
                       "n_items": len(base_deltas), "n_excluded": 0, "delta": 0.0, "by_construction": how}
    return {i: 0.0 for i in base_deltas}


def analyze_log(records: list[dict[str, Any]], design: dict[str, Any], spec: dict[str, Any] | None,
                n_perm: int, n_boot: int, seed: int, label: str) -> dict[str, Any]:
    segments = design["ext"]["segments"]
    out = collect_ext(records)
    base_seg = next((s for s in segments if s["mitigation"] == "none"), None)
    base_out = segment_outputs(out, base_seg) if base_seg else {}
    result: dict[str, Any] = {"segments": {}, "effects": {}, "costs": _with_ratios(
        cost_table(records, spec_prices(spec)), design)}
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    comps_by_segment: dict[str, dict[str, Any]] = {}
    for seg in segments:  # the base segment comes first
        name, mitigation = seg["name"], seg["mitigation"]
        if mitigation in PLACEBO_SEGMENTS:  # Addendum G's segments: extension/placebo.py
            continue
        reuse = seg.get("issues_tasks") is False
        if mitigation == "symmetric":
            source = base_out if reuse else segment_outputs(out, seg)
            seg_design = segment_design(design, seg)
            trans = transitivity_report(source, seg_design, n_boot=min(n_boot, 500),
                                        seed=stable_seed(seed, label, "trans") % 2**32)
            rows: dict[str, Any] = {}
            how = f"both plans send the same two requests and combine them with {seg.get('combine', 'and').upper()}"
            deltas[name] = {"symmetry": _by_construction(name, rows, "symmetry",
                                                         deltas.get("base", {}).get("symmetry", {}), how)}
            rows["transitivity"] = {p: {"violation_rate": trans[p]["violation_rate"], "f1": trans[p]["clusters"]["f1"]}
                                    for p in ("ab", "ba", "and", "or")}
            result["segments"][name] = rows
            comps_by_segment[name] = {}
            continue
        if mitigation == "negation_normalized" and reuse:
            rows = {}
            deltas[name] = {"negation": _by_construction(
                name, rows, "negation", deltas.get("base", {}).get("negation", {}),
                "NOT F(p) is read from the base segment's F(p) calls: NOT p is never sent")}
            result["segments"][name] = rows
            comps_by_segment[name] = {}
            continue
        if mitigation == "none":
            seg_out, seg_design = base_out, segment_design(design, seg)
        elif mitigation == "vote":
            seg_out = vote_outputs(segment_outputs(out, seg), seg["votes"], seg["voted_repeats"])
            seg_design = segment_design(design, seg)
        elif mitigation == "negation_normalized":  # fresh F(p) calls
            if base_seg is None:
                continue
            seg_out = negation_normalized_outputs(base_out, segment_outputs(out, seg))
            seg_design = segment_design(design, {**seg, "repeats": base_seg["repeats"]})
        else:  # explain, format
            seg_out, seg_design = segment_outputs(out, seg), segment_design(design, seg)
        rows, comps = _comparison_rows(seg_out, seg_design, n_perm, n_boot, seed, f"{label}/{name}")
        if mitigation in TARGETS:
            rows = {k: v for k, v in rows.items() if k in TARGETS[mitigation]}
        if seg.get("count") is False:  # base_count = false: no COUNT requests were sent
            rows.pop("count", None)
            comps.pop("count", None)
        result["segments"][name] = rows
        deltas[name] = {c: _item_deltas(comp) for c, comp in comps.items()}
        comps_by_segment[name] = comps
    if base_seg is not None:
        pairs = [(seg["name"], "base", seg["mitigation"]) for seg in segments if seg["mitigation"] != "none"]
        names = {seg["name"] for seg in segments}
        if "explain" in names and "format" in names:
            pairs.append(("explain", "format", "explain"))  # the instruction to reason, format held fixed
        for other_name, ref_name, mitigation in pairs:
            if other_name not in deltas or ref_name not in deltas:
                continue
            effects = {}
            for cname, other in deltas[other_name].items():
                ref = deltas[ref_name].get(cname)
                if ref is None or (mitigation in TARGETS and cname not in TARGETS[mitigation]):
                    continue
                comp = comps_by_segment["base"].get(cname)
                effects[cname] = paired_difference(ref, other, comp, n_boot,
                                                   stable_seed(seed, label, other_name, ref_name, cname) % 2**32)
            key = other_name if ref_name == "base" else f"{other_name} vs {ref_name}"
            result["effects"][key] = effects
    result["_item_deltas_base"] = deltas.get("base", {})
    result["_comps_base"] = comps_by_segment.get("base", {})
    return result


def analyze_run_ext(run_dir: Path, n_perm: int = 5000, n_boot: int = 2000, seed: int = 0,
                    backends: list[str] | None = None) -> dict[str, Any]:
    design = load_design(run_dir)
    config = load_config(run_dir)
    roster = _roster(config)
    results: dict[str, Any] = {"run": str(run_dir), "segments": design["ext"]["segments"], "backends": {}}
    for backend, path in run_logs(run_dir, backends):
        results["backends"][backend] = analyze_log(read_compact(path), design, roster.get(backend), n_perm, n_boot,
                                                   seed, backend)
    # Reasoning-on (or any other setting) arms paired with their reference arm, on the base segment.
    results["arm_pairs"] = {}
    for backend, res in results["backends"].items():
        partner = (roster.get(backend) or {}).get("pairs_with")
        if not partner or partner not in results["backends"]:
            continue
        ref = results["backends"][partner]
        effects = {}
        for cname, deltas in res["_item_deltas_base"].items():
            base = ref["_item_deltas_base"].get(cname)
            if base is not None:
                effects[cname] = paired_difference(base, deltas, ref["_comps_base"].get(cname), n_boot,
                                                   stable_seed(seed, backend, partner, cname) % 2**32)
        costs = {}
        for key, row in res["costs"].items():
            other = ref["costs"].get(key)
            if other and key.startswith("base/") and key.count("/") == 1:
                costs[key] = {m: (row[m] / other[m] if other[m] else None)
                              for m in ("calls", "input_tokens", "output_tokens", "reasoning_tokens", "usd")}
        results["arm_pairs"][backend] = {"reference": partner, "effects": effects, "cost_ratios": costs}
    for res in results["backends"].values():
        res.pop("_item_deltas_base", None)
        res.pop("_comps_base", None)
    return results


def _f(x: Any, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


EFFECT_HEAD = ["| Comparison | Items | Clusters | Delta reference | Delta this | Difference [95% t CI] | p | "
               "Bootstrap 95% CI (secondary) |", "|---|---|---|---|---|---|---|---|"]


def _effect_row(cname: str, e: dict[str, Any]) -> str:
    if e.get("status") != "ok":
        return (f"| {cname} | {e.get('n_items')} | {e.get('n_clusters', '')} | {_f(e.get('delta_base'))} | "
                f"{_f(e.get('delta_other'))} | {e.get('status')} | {_f(e.get('p_two_sided'), 3)} | |")
    return (f"| {cname} | {e['n_items']} | {'x'.join(map(str, e['n_clusters']))} | {_f(e['delta_base'])} | "
            f"{_f(e['delta_other'])} | {e['difference']:+.4f} {fmt_ci(e['difference_ci'])} | "
            f"{e['p_two_sided']:.3g} | {fmt_ci(e['difference_ci_bootstrap'])} |")


def render(results: dict[str, Any]) -> str:
    lines = [f"# Mitigation effects: {results['run']}", "",
             "Delta is semaudit's excess divergence on each segment's answers. The effect is the paired difference "
             "Delta(this) - Delta(reference) over shared items, grouped as semaudit groups them (crossed N1 and N2 "
             "batches for H4). Primary: the t interval and two-sided p from the cluster-robust standard error, with "
             "df = clusters - 1 (the smaller grouping for H4). The bootstrap interval is secondary; it under-covers "
             "with few clusters. 'explain vs format' isolates the instruction to reason. Costs count every logged call.",
             ""]
    for backend, res in results["backends"].items():
        lines += [f"## {backend}", ""]
        for seg, effects in res["effects"].items():
            if not effects:
                continue
            title = seg if " vs " in seg else f"{seg} vs base"
            lines += [f"### {title}", "", *EFFECT_HEAD]
            lines += [_effect_row(cname, e) for cname, e in effects.items()]
            lines.append("")
        lines += ["### Cost per segment and identity", "",
                  "| Segment/identity | Calls | Items | Calls per item | Tokens per item | USD | Ratio to base: calls, "
                  "output tokens, USD |", "|---|---|---|---|---|---|---|"]
        for key, c in res["costs"].items():
            if key.count("/") > 1 and not key.startswith("negnorm/"):
                continue  # per-condition rows only where a segment replaces one condition
            ratios = ", ".join(_f(c.get(f"{m}_ratio"), 2) for m in ("calls", "output_tokens", "usd")) \
                if "calls_ratio" in c else ""
            lines.append(f"| {key} | {c['calls']} | {c['items']} | {c['calls_per_item']:.2f} | "
                         f"{c['tokens_per_item']:.0f} | {c['usd']:.4f} | {ratios} |")
        lines.append("")
    for backend, pair in results.get("arm_pairs", {}).items():
        lines += [f"## {backend} vs {pair['reference']} (base segment)", "", *EFFECT_HEAD]
        lines += [_effect_row(cname, e) for cname, e in pair["effects"].items()]
        lines.append("")
        for key, ratios in pair["cost_ratios"].items():
            lines.append(f"- {key}: " + ", ".join(f"{m} x{_f(v, 2)}" for m, v in ratios.items()))
        lines.append("")
    return "\n".join(lines)


def write(results: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mitigations.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out_dir / "mitigations.md").write_text(render(results), encoding="utf-8")
