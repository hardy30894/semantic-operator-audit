"""Analyse the consistent grid: every test for every model, from exactly the runs grid/coverage.json names.

Follows semaudit.pool.pool_runs step for step: items of every condition (or matching dataset) are pooled per
model and test with the condition as a prefix, compared with semaudit.stats.compare (permutation test, the
batch rule for batched prompts, bootstrap intervals over shared records or calls), and Holm-adjusted within each
model over the seven identities H1 to H7 and separately over S2 to S5. Per-condition estimates are reported
beside the pooled one. Units are the curated ones in grid/units, so every model is measured on the same items.

Run from the project root:  python3 grid/analyze.py [--models ...] [--quick]
Writes grid/results/grid.json and grid/results/GRID.md. Makes no model calls.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "harness"
sys.path.insert(0, str(H))
sys.path.insert(0, str(ROOT / "grid"))
from semaudit.core import stable_seed  # noqa: E402
from semaudit.identities import collect, comparisons  # noqa: E402
from semaudit.pool import PRIMARY_HYPOTHESES, SECONDARY_HYPOTHESES, SESOI  # noqa: E402
from semaudit.stats import compare, holm  # noqa: E402

ALL_PAIRS = ["wdc-A", "wdc-B", "abt-A", "abt-B", "cuad-A", "cuad-B", "argq-A", "argq-B", "arxiv-A", "arxiv-B", "argkp-A"]
# GRID_SCOPE=a-pairs restricts every analysis to the six A pairs (one condition pair per dataset, the submitted
# paper's scope) and writes to grid/results/a-pairs/; the default is all 11 pairs into grid/results/.
SCOPE = os.environ.get("GRID_SCOPE", "")
PAIRS = [p for p in ALL_PAIRS if p.endswith("-A")] if SCOPE == "a-pairs" else ALL_PAIRS
OUT = ROOT / "grid" / "results" / SCOPE if SCOPE else ROOT / "grid" / "results"
MATCH = ["wdc", "abt", "argkp"]
MODELS = ["haiku-5.5", "sonnet-5.5", "gpt-5.6-luna", "gpt-5.6-terra", "gemini-3.5-flash-lite", "gemini-3.5-flash",
          "qwen-3.7-flash", "deepseek-v3.2"]
SHORT = {"haiku-5.5": "Haiku", "sonnet-5.5": "Sonnet", "gpt-5.6-luna": "Luna", "gpt-5.6-terra": "Terra",
         "gemini-3.5-flash-lite": "Flash-Lite", "gemini-3.5-flash": "Flash", "qwen-3.7-flash": "Qwen",
         "deepseek-v3.2": "DeepSeek"}
NAMES = {"H1": "symmetry", "H2": "negation", "H3": "fusion", "H4": "batch_neighbours", "H5": "placement_full",
         "H6": "idempotence", "H7": "count", "S2": "batch_position", "S3": "conjunct_order", "S4": "batch_vs_single",
         "S5": "placement_names"}
LABEL = {"H1": "Symmetry", "H2": "Negation", "H3": "Fusion", "H4": "Batch neighbors", "H5": "Placement",
         "H6": "Idempotence", "H7": "COUNT (rows per batch)", "S2": "Position", "S3": "Conjunct order",
         "S4": "Batching", "S5": "Renamed columns"}
K = 5

_cache: dict[tuple[str, str], Any] = {}


def outputs(run: str, model: str):
    key = (run, model)
    if key not in _cache:
        path = H / "runs" / run / f"{model}.jsonl.gz"
        recs = []
        if path.exists():
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # identity tests read only the base calls; segment calls (reword/, unknown/) are the
                        # rewording analysis's, and the unknown style has its own parser
                        if "/" not in rec["identity"]:
                            recs.append(rec)
            except EOFError:  # a log still being written
                pass
        _cache[key] = collect(recs) if recs else None
    return _cache[key]


def designs(where: str, unit: dict) -> dict[str, tuple[str, dict]]:
    """For each source key of a cell, the design semaudit.identities.comparisons reads, on the grid's units."""
    base = {"repeats": K}
    if where in MATCH:
        sym = unit["symmetry"]
        return {"symmetry": ("symmetry", {**base, "symmetry": {"pairs": sym["pairs"],
                                                                "blocks": [p.split("|") for p in sym["pairs"]]}}),
                "placement": ("placement", {**base, "placement": {"tuples": unit["placement"]["tuples"]}})}
    out = {"logic and rewordings": ("logic and rewordings",
                                    {**base, "filter_algebra": {"rows": unit["filter"]["rows"], "count_batches": {}}}),
           "COUNT": ("COUNT", {**base, "filter_algebra": {"rows": [], "count_batches": unit["filter"]["count_batches"]}}),
           "batching": ("batching", {**base, "batch": {"rows": unit["batch"]["analysed_rows"],
                                                       "partitions": unit["batch"]["prompts"]}})}
    if "idempotence" in unit:
        out["idempotence"] = ("idempotence", {**base, "idempotence": {"rows": unit["idempotence"]["rows"]}})
    return out


WANT = {"logic and rewordings": {"negation", "fusion", "conjunct_order"}, "COUNT": {"count"},
        "batching": {"batch_neighbours", "batch_position", "batch_vs_single"}, "idempotence": {"idempotence"},
        "symmetry": {"symmetry"}, "placement": {"placement_full", "placement_names"}}


def analyse_model(model: str, sources: dict, n_perm: int, n_boot: int, n_perm_binary: int) -> dict:
    slots: dict[str, dict] = {}
    for where in PAIRS + MATCH:
        unit = json.loads((ROOT / "grid" / "units" / (f"match-{where}.json" if where in MATCH else f"{where}.json"))
                          .read_text(encoding="utf-8"))
        cell = sources.get(where, {})
        for key, (_, design) in designs(where, unit).items():
            run = cell.get(key)
            out = outputs(run, model) if run else None
            if out is None:
                continue
            for comp in comparisons(out, design):
                if comp.name not in WANT[key]:
                    continue
                clusters = comp.clusters
                if comp.name == "symmetry":  # pairs that share records or come from one block
                    clusters = unit["symmetry"]["cluster_of"]
                s = slots.setdefault(comp.name, {"comp": comp, "draws": {}, "clusters": {}, "call_clusters": {},
                                                 "by_where": {}})
                s["draws"].update({f"{where}/{i}": d for i, d in comp.draws.items()})
                if clusters:
                    s["clusters"].update({f"{where}/{i}": f"{where}/{clusters[i]}" for i in comp.draws if i in clusters})
                if comp.call_clusters:
                    s["call_clusters"].update({f"{where}/{i}": tuple(f"{where}/{c}" for c in g)
                                               for i, g in comp.call_clusters.items()})
                s["by_where"][where] = (comp, clusters, run)
    rows = []
    for hyp, name in NAMES.items():
        s = slots.get(name)
        if not s or not s["draws"]:
            rows.append({"hypothesis": hyp, "comparison": name, "status": "not measured"})
            continue
        c = s["comp"]
        seed = stable_seed(0, "grid", model, name) % 2**32
        st = compare(s["draws"], c.kind, n_perm=n_perm, n_boot=n_boot, seed=seed, clusters=s["clusters"] or None,
                     n_perm_binary=n_perm_binary, call_clusters=s["call_clusters"] or None)
        row = {"hypothesis": hyp, "comparison": name, "kind": c.kind, **st}
        if st.get("status") == "ok":
            row["sesoi"] = SESOI[c.kind]
            lo = (st.get("delta_ci") or [None, None])[0]
            row["sqrt_ci_low"] = math.sqrt(max(lo, 0.0)) if lo is not None else None
        row["by_where"] = []
        for where, (comp, clusters, run) in s["by_where"].items():
            part = compare(comp.draws, comp.kind, n_perm=2000, n_boot=500,
                           seed=stable_seed(0, "grid-where", model, name, where) % 2**32,
                           clusters=clusters, call_clusters=comp.call_clusters)
            row["by_where"].append({"where": where, "run": run, **{k: part.get(k) for k in
                                    ("n_items", "n_excluded", "delta", "sqrt_delta", "p", "status")}})
        rows.append(row)
    for family in (PRIMARY_HYPOTHESES, SECONDARY_HYPOTHESES):
        idx = [i for i, r in enumerate(rows) if r["hypothesis"] in family]
        for i, adj in zip(idx, holm([rows[i].get("p", 1.0) for i in idx])):
            rows[i]["p_holm"] = adj
            r = rows[i]
            r["broken"] = r.get("status") == "ok" and adj < 0.05
            r["above_sesoi"] = r.get("status") == "ok" and r["sqrt_delta"] >= r["sesoi"]
            r["min_effect"] = r["broken"] and (r.get("sqrt_ci_low") or 0) >= r.get("sesoi", math.inf)
    return {"model": model, "comparisons": rows}


def fmt(r: dict) -> str:
    if r.get("status") != "ok":
        return "–"
    v = f"{r['sqrt_delta']:.2f}"
    mark = "" if not r["broken"] else ("" if r["min_effect"] else "ᵐ")
    return (f"**{v}**{mark}" if r["broken"] else v) + ("ˢ" if r["broken"] and not r["above_sesoi"] else "")


def render(results: list[dict]) -> str:
    models = [r["model"] for r in results]
    out = ["# The consistent grid: results", "",
           "Excess divergence √Δ̄ pooled over the grid's units, per test and model. Bold: broken (Holm p < 0.05 "
           "within the model). ˢ: broken but below the smallest effect of interest. ᵐ: broken and above it, but the "
           "lower 95% bound is below it. COUNT in rows per batch of ten. Generated by `grid/analyze.py`.", "",
           "| Test | " + " | ".join(SHORT[m] for m in models) + " |", "|---|" + "---|" * len(models)]
    for hyp in list(PRIMARY_HYPOTHESES) + list(SECONDARY_HYPOTHESES):
        cells = []
        for res in results:
            r = next(x for x in res["comparisons"] if x["hypothesis"] == hyp)
            cells.append(fmt(r))
        out.append(f"| {hyp} {LABEL[hyp]} | " + " | ".join(cells) + " |")
    out += ["", "## Items per test (after leaving out items with an unreadable answer)", "",
            "| Test | " + " | ".join(SHORT[m] for m in models) + " |", "|---|" + "---|" * len(models)]
    for hyp in list(PRIMARY_HYPOTHESES) + list(SECONDARY_HYPOTHESES):
        cells = []
        for res in results:
            r = next(x for x in res["comparisons"] if x["hypothesis"] == hyp)
            cells.append(f"{r.get('n_items', 0):,}" if r.get("status") == "ok" else "–")
        out.append(f"| {hyp} {LABEL[hyp]} | " + " | ".join(cells) + " |")
    out += ["", "## Per condition and dataset (√Δ̄)", ""]
    for hyp in list(PRIMARY_HYPOTHESES) + list(SECONDARY_HYPOTHESES):
        wheres = MATCH if hyp in ("H1", "H5", "S5") else PAIRS
        out += [f"**{hyp} {LABEL[hyp]}**", "", "| Model | " + " | ".join(wheres) + " |", "|---|" + "---|" * len(wheres)]
        for res in results:
            r = next(x for x in res["comparisons"] if x["hypothesis"] == hyp)
            bw = {b["where"]: b for b in r.get("by_where", [])}
            cells = [f"{bw[w]['sqrt_delta']:.2f}" if w in bw and bw[w].get("sqrt_delta") is not None else "–"
                     for w in wheres]
            out.append(f"| {SHORT[res['model']]} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[m for m in MODELS if m != "deepseek-v3.2"])
    ap.add_argument("--quick", action="store_true", help="fewer permutations, for checks")
    args = ap.parse_args(argv)
    sources = json.loads((ROOT / "grid" / "coverage.json").read_text(encoding="utf-8"))["sources"]
    n_perm, n_boot, n_bin = (500, 200, 2000) if args.quick else (10000, 10000, 100000)
    results = []
    for m in args.models:
        print(f"analysing {m}", flush=True)
        results.append(analyse_model(m, sources[m], n_perm, n_boot, n_bin))
        _cache.clear()
    outdir = OUT
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "-quick" if args.quick else ""
    (outdir / f"grid{suffix}.json").write_text(json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8")
    (outdir / f"GRID{suffix}.md").write_text(render(results), encoding="utf-8")
    print(f"wrote grid/results/GRID{suffix}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
