"""Plan and price ArgKP as the grid's 11th condition pair (argkp-A), and record where its cells come from.

Added after the grid's first runs (grid/RUN.md). No model has run ArgKP's filter tests, batching, COUNT,
idempotence or rewordings before (study-argkp-A ran only symmetry and placement), so every cell runs, for all
seven models, priced like the rest of the grid (grid/coverage.py's prices: each model's mean logged cost per call
of the operation, on ArgKP where it exists, otherwise scaled from product data by the prompt-length ratio).

grid/coverage.py itself is not re-run: it predates the grid's runs and does not read them, so it would list every
cell the grid already ran as missing. This script writes grid/coverage-argkp.json (the runs grid/run.py executes
with --coverage grid/coverage-argkp.json) and adds argkp-A to the sources in grid/coverage.json, the runs the
analyses read; the 10 earlier pairs' entries are left as they are. Makes no model calls.

Run from the project root:  python3 grid/add_argkp.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grid"))
import coverage as C  # noqa: E402

PAIR = "argkp-A"
MODELS = [m for m in C.MODELS if m != "deepseek-v3.2"]  # DeepSeek is not in the grid (grid/RUN.md)


def main() -> int:
    unit = json.loads((C.U / f"{PAIR}.json").read_text(encoding="utf-8"))
    gt = C.grid_tasks(PAIR, unit)
    conds = unit["rewording"]["conditions"]
    for model in MODELS:  # nothing of argkp-A may exist yet: the analyses must read only the runs below
        assert not C.have(gt["filter"], C.log_index(f"study-{PAIR}", model)), model
        assert not C.reword_done(model, PAIR, conds), model
    costs = C.mean_costs()
    n_single = sum(t.condition == "single" for t in gt["batch"])
    cells = [("logic + rewordings (one run)", 1000 * (5 + len(conds)), "filter"),
             ("COUNT, in the same run", len(gt["count"]), "count"),
             ("batching", len(gt["batch"]) - n_single, "filter_batch"),
             ("batching", n_single, "filter"),
             ("idempotence", 3 * len(gt["idempotence"]) // 2, "map")]
    to_run = []
    for model in MODELS:
        for test, calls, op in cells:
            to_run.append({"model": model, "where": PAIR, "test": test, "calls": calls,
                           "usd": calls * C.price(costs, model, "argkp", op)})
    sources = {"logic and rewordings": f"grid-{PAIR}-lr", "COUNT": f"grid-{PAIR}-lr",
               "batching": f"grid-{PAIR}-batch", "idempotence": f"grid-{PAIR}-idem"}
    total_calls = sum(r["calls"] for r in to_run)
    total_usd = sum(r["usd"] for r in to_run)
    out = ROOT / "grid" / "coverage-argkp.json"
    out.write_text(json.dumps({"to_run": to_run, "sources": {m: {PAIR: sources} for m in MODELS},
                               "total_calls": total_calls, "total_usd": total_usd}, indent=1) + "\n",
                   encoding="utf-8")
    cov_path = ROOT / "grid" / "coverage.json"
    cov = json.loads(cov_path.read_text(encoding="utf-8"))
    for m in MODELS:
        cov["sources"].setdefault(m, {})[PAIR] = sources
    cov_path.write_text(json.dumps(cov, indent=1) + "\n", encoding="utf-8")
    for m in MODELS:
        rows = [r for r in to_run if r["model"] == m]
        print(f"{m:24s} {sum(r['calls'] for r in rows):7,} calls  est. US${sum(r['usd'] for r in rows):6.2f}")
    print(f"{'total':24s} {total_calls:7,} calls  est. US${total_usd:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
