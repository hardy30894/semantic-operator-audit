"""Variation across conditions in the consistent grid (RQ3): per model and test, each condition's Delta and its
dependence-robust standard error (extension.meta, unchanged: cluster-robust over shared records or calls, with
the design-based floor), pooled by REML on the Delta scale (extension.meta.random_effects) and on the square-root
scale (scripts/review_meta.meta_sqrt). Writes grid/results/meta.json and meta_sqrt.json in the formats those
scripts write, so the paper's figure code reads them unchanged. Makes no model calls.
Run from the project root:  python3 grid/analyze_meta.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "harness"
sys.path.insert(0, str(H))
sys.path.insert(0, str(H / "scripts"))
sys.path.insert(0, str(ROOT / "grid"))
import review_meta  # noqa: E402
from extension.meta import _se, random_effects, se_floor  # noqa: E402
from semaudit.identities import comparisons  # noqa: E402
from semaudit.stats import encode, item_deltas  # noqa: E402

import analyze as A  # noqa: E402


def estimates(model: str, sources: dict) -> dict:
    out: dict = {}
    for where in A.PAIRS + A.MATCH:
        unit = json.loads((ROOT / "grid" / "units" / (f"match-{where}.json" if where in A.MATCH else f"{where}.json"))
                          .read_text(encoding="utf-8"))
        for key, (_, design) in A.designs(where, unit).items():
            run = sources.get(where, {}).get(key)
            o = A.outputs(run, model) if run else None
            if o is None:
                continue
            for comp in comparisons(o, design):
                if comp.name not in A.WANT[key]:
                    continue
                if comp.name == "symmetry":
                    comp = copy.copy(comp)
                    comp.clusters = unit["symmetry"]["cluster_of"]
                items, a, b, excluded, _ = encode(comp.draws, comp.kind)
                if len(items) < 2:
                    continue
                d = item_deltas(a, b, comp.kind)[0]
                raw = _se(d, items, comp)
                floor = se_floor(items, comp, a.shape[1], b.shape[1])
                entry = out.setdefault(comp.name, {"hypothesis": comp.hypothesis, "kind": comp.kind,
                                                   "primary": comp.primary, "studies": []})
                # review_meta reads the dataset from the run name, "study-<pair>"
                entry["studies"].append({"run": f"study-{where}", "n_items": len(items), "n_excluded": excluded,
                                         "delta": float(d.mean()), "se": max(raw, floor), "se_raw": raw,
                                         "se_floored": raw < floor})
    for name, entry in out.items():
        y, se = [s["delta"] for s in entry["studies"]], [s["se"] for s in entry["studies"]]
        entry["reml"], entry["dl"] = random_effects(y, se, "REML"), random_effects(y, se, "DL")
    return out


def main() -> int:
    sources = json.loads((ROOT / "grid" / "coverage.json").read_text(encoding="utf-8"))["sources"]
    models = [m for m in A.MODELS if m != "deepseek-v3.2"]
    meta = {"runs": A.PAIRS + A.MATCH, "backends": {}}
    for m in models:
        print(f"meta: {m}", flush=True)
        meta["backends"][m] = estimates(m, sources[m])
        A._cache.clear()
    outdir = A.OUT
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    msq = review_meta.meta_sqrt(meta)
    (outdir / "meta_sqrt.json").write_text(json.dumps(msq, indent=1, default=float) + "\n", encoding="utf-8")
    print("wrote grid/results/meta.json and meta_sqrt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
