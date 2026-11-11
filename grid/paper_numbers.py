"""The paper's derived numbers that the analyses do not print directly, from the A-pairs outputs and the logs.

- Fig. 3 (heatmap): sqrt(Delta) per identity, model and dataset, and whether each cell is above the noise floor
  (permutation test with 10,000 permutations per cell, Holm within the model over its 57 cells).
- Fig. 4 (records returned): per identity and model, the mean absolute difference in the number of records returned
  between the two queries (every pair of draws) and between two draws of one query, per 1,000 items, computed per
  dataset and averaged with each dataset weighted by its items (COUNT per 1,000 records, a batch holding ten).
- Section V-E: accuracy changes per model and dataset (Holm within the model and dataset over the identities with
  truth), and how many combinations change answers (above the noise floor of Fig. 3) and accuracy.
- Section V-F: per dataset, whether a violation is smaller, equal or larger with reasoning on.
Run from the project root after the analyses:  GRID_SCOPE=a-pairs python3 grid/paper_numbers.py
Writes grid/results/a-pairs/paper_numbers.json and PAPER-NUMBERS.md. Makes no model calls.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GRID_SCOPE", "a-pairs")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "grid"))
import analyze as A  # noqa: E402
from semaudit.core import stable_seed  # noqa: E402
from semaudit.identities import comparisons  # noqa: E402
from semaudit.stats import compare, holm  # noqa: E402

MODELS = ["haiku-5.5", "gpt-5.6-luna", "gemini-3.5-flash-lite", "qwen-3.7-flash", "gemini-3.5-flash", "sonnet-5.5",
          "gpt-5.6-terra"]
REASON = {"haiku-5.5-reasoning-medium": "haiku-5.5", "gpt-5.6-luna-reasoning-medium": "gpt-5.6-luna",
          "sonnet-5.5-reasoning-medium": "sonnet-5.5"}
NAME2H = {v: k for k, v in A.NAMES.items()}
ORDER = ["H2", "H3", "H4", "H7", "S2", "S3", "S4"]


def dataset(where: str) -> str:
    return where.split("-")[0]


def numeric(x) -> bool:
    return isinstance(x, (bool, int, float))


def gaps(draws: list) -> tuple[float, float, int] | None:
    ok = [(a, b) for a, b in draws if len(a) == len(b) and all(numeric(x) for x in list(a) + list(b))]
    if not ok:
        return None
    k = len(ok[0][0])
    sa = [sum(float(a[u]) for a, _ in ok) for u in range(k)]
    sb = [sum(float(b[u]) for _, b in ok) for u in range(k)]
    between = sum(abs(x - y) for x in sa for y in sb) / (k * k)
    within = [abs(s[i] - s[j]) for s in (sa, sb) for i in range(k) for j in range(i + 1, k)]
    return between * 1000 / len(ok), sum(within) / len(within) * 1000 / len(ok), len(ok)


def main() -> int:
    sources = json.loads((ROOT / "grid" / "coverage.json").read_text(encoding="utf-8"))["sources"]
    heat, prop = {}, {}
    for m in MODELS:
        keys, ps, vals = [], [], []
        per: dict[str, list] = {}
        for where in A.PAIRS + A.MATCH:
            unit = json.loads((ROOT / "grid" / "units" / (f"match-{where}.json" if where in A.MATCH else f"{where}.json"))
                              .read_text(encoding="utf-8"))
            for key, (_, design) in A.designs(where, unit).items():
                run = sources[m].get(where, {}).get(key)
                out = A.outputs(run, m) if run else None
                if out is None:
                    continue
                for comp in comparisons(out, design):
                    if comp.name not in A.WANT[key]:
                        continue
                    clusters = unit["symmetry"]["cluster_of"] if comp.name == "symmetry" else comp.clusters
                    st = compare(comp.draws, comp.kind, n_perm=10000, n_boot=100,
                                 seed=stable_seed(0, "grid-where", m, comp.name, where) % 2**32, clusters=clusters,
                                 call_clusters=comp.call_clusters)
                    if st.get("status") == "ok":
                        keys.append(f"{NAME2H[comp.name]}|{dataset(where)}")
                        ps.append(st["p"])
                        vals.append(st["sqrt_delta"])
                    if comp.name != "idempotence":
                        g = gaps(list(comp.draws.values()))
                        if g:
                            per.setdefault(NAME2H[comp.name], []).append(g)
        A._cache.clear()
        heat[m] = {k: {"sqrt_delta": v, "p": p, "p_holm": a, "above_noise_floor": a < 0.05}
                   for k, v, p, a in zip(keys, vals, ps, holm(ps))}
        for hyp, v in per.items():
            n = sum(x[2] for x in v)
            f = 0.1 if hyp == "H7" else 1.0  # COUNT: per 1,000 records, a batch holding ten
            prop[f"{m}|{hyp}"] = {"between": f * sum(x[0] * x[2] for x in v) / n,
                                  "within": f * sum(x[1] * x[2] for x in v) / n}
        print(f"{m}: {sum(c['above_noise_floor'] for c in heat[m].values())} of {len(heat[m])} cells above the noise floor",
              flush=True)

    acc = json.loads((A.OUT / "accuracy.json").read_text(encoding="utf-8"))
    accsig = {}
    for m in MODELS:
        for where, rows in acc["filter"][m].items():
            hyps = [h for h in ORDER if h in rows]
            adj = holm([rows[h]["p_two_sided"] if rows[h].get("status") == "ok" else 1.0 for h in hyps])
            for h, a in zip(hyps, adj):
                accsig[f"{m}|{h}|{dataset(where)}"] = {"difference_points": 100 * rows[h]["difference"],
                                                       "p_holm": a, "changes": a < 0.05}
    with_truth = list(accsig)
    answers = [k for k in with_truth if heat[k.split("|")[0]].get("|".join(k.split("|")[1:]), {}).get("above_noise_floor")]

    grid = {r["model"]: {c["hypothesis"]: c for c in r["comparisons"]} for r in json.loads((A.OUT / "grid.json").read_text())}
    reason = {r["model"]: r for r in json.loads((A.OUT / "reason.json").read_text())}
    per_dataset = {}
    for rm, off in REASON.items():
        tally = {"smaller": 0, "equal": 0, "larger": 0, "larger_cells": []}
        for c in reason[rm]["comparisons"]:
            offb = {b["where"]: b for b in grid[off][c["hypothesis"]].get("by_where", [])}
            for b in c.get("by_where", []):
                o = offb.get(b["where"])
                if not o or b.get("sqrt_delta") is None or o.get("sqrt_delta") is None:
                    continue
                on, of = b["sqrt_delta"], o["sqrt_delta"]
                key = "smaller" if on < of else "equal" if on == of else "larger"
                tally[key] += 1
                if key == "larger":
                    tally["larger_cells"].append([c["hypothesis"], b["where"], round(of, 2), round(on, 2)])
        per_dataset[rm] = tally

    result = {"heatmap": heat, "records_returned": prop, "accuracy_per_dataset": accsig,
              "accuracy_summary": {"combinations_with_truth": len(with_truth), "answers_change": len(answers),
                                   "accuracy_changes": sum(v["changes"] for v in accsig.values()),
                                   "accuracy_changes_among_answer_changes": sum(accsig[k]["changes"] for k in answers)},
              "reasoning_per_dataset": per_dataset}
    (A.OUT / "paper_numbers.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    lines = ["# Derived numbers of the paper", "", "Generated by `grid/paper_numbers.py` from the A-pairs outputs and the logs.", "",
             "## Cells above the noise floor (Fig. 3)", "", "| Model | Cells above |", "|---|---|"]
    for m in MODELS:
        lines.append(f"| {A.SHORT[m]} | {sum(c['above_noise_floor'] for c in heat[m].values())} of {len(heat[m])} |")
    lines += ["", "| Dataset | Cells above |", "|---|---|"]
    for d in ["wdc", "abt", "cuad", "argq", "arxiv", "argkp"]:
        cells = [c for m in MODELS for k, c in heat[m].items() if k.split("|")[1] == d]
        lines.append(f"| {d} | {sum(c['above_noise_floor'] for c in cells)} of {len(cells)} |")
    lines += ["", "## Records returned per 1,000 items (Fig. 4): between the two queries / between two draws", "",
              "| Identity | " + " | ".join(A.SHORT[m] for m in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    for hyp in ["H1", "H2", "H3", "H4", "H5", "H7", "S2", "S3", "S4", "S5"]:
        cells = [f"{prop[f'{m}|{hyp}']['between']:.1f} / {prop[f'{m}|{hyp}']['within']:.1f}" if f"{m}|{hyp}" in prop
                 else "–" for m in MODELS]
        lines.append(f"| {hyp} | " + " | ".join(cells) + " |")
    s = result["accuracy_summary"]
    lines += ["", "## Accuracy (Section V-E)", "",
              f"{s['combinations_with_truth']} combinations of identity, model and dataset with truth; "
              f"{s['answers_change']} change answers beyond the noise floor; {s['accuracy_changes']} change accuracy "
              f"(Holm within the model and dataset), {s['accuracy_changes_among_answer_changes']} of them among those "
              f"that change answers.", "", "## Reasoning per dataset (Section V-F)", ""]
    for rm, t in per_dataset.items():
        lines.append(f"- {rm}: smaller {t['smaller']}, equal {t['equal']}, larger {t['larger']}")
    (A.OUT / "PAPER-NUMBERS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", A.OUT / "PAPER-NUMBERS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
