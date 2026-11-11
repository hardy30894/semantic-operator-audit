"""What the consistent grid already has, what it still needs, and what the missing calls cost.

For every model, pair and test, a unit counts as done only if the model's log holds a successful call for
every task the grid sends (same task id, byte-identical system and user prompt), all five repeats, and
all of a comparison's conditions come from that one run (one session). Everything else is to run.
Costs are measured from the logs: each model's mean cost per call of the same operation on the same
dataset where it exists; otherwise the model's mean on the product data, scaled by how much longer that
dataset's prompts are for the other models (a Flash-Lite ratio). Makes no model calls.

Run from the project root:  python3 grid/coverage.py     writes grid/COVERAGE.md and grid/coverage.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "harness"
RUNS = H / "runs"
U = ROOT / "grid" / "units"
sys.path.insert(0, str(H))
sys.path.insert(0, str(ROOT / "grid"))
from semaudit.data import load_dataset_dir  # noqa: E402
from semaudit.identities import build  # noqa: E402

PAIRS = ["wdc-A", "wdc-B", "abt-A", "abt-B", "cuad-A", "cuad-B", "argq-A", "argq-B", "arxiv-A", "arxiv-B"]
MODELS = ["haiku-5.5", "gpt-5.6-luna", "gemini-3.5-flash-lite", "qwen-3.7-flash", "deepseek-v3.2",
          "gemini-3.5-flash", "sonnet-5.5", "gpt-5.6-terra"]
SHORT = {"haiku-5.5": "Haiku", "gpt-5.6-luna": "Luna", "gemini-3.5-flash-lite": "Flash-Lite",
         "qwen-3.7-flash": "Qwen", "deepseek-v3.2": "DeepSeek", "gemini-3.5-flash": "Flash",
         "sonnet-5.5": "Sonnet", "gpt-5.6-terra": "Terra"}
# The grid runs Qwen with default sampling like every other model; the logged Qwen arm ran at temperature 0,
# so none of its calls count for the grid. Its token costs are the same, so they price the new Qwen calls.
PRICE_AS = {"qwen-3.7-flash": "qwen-3.7-flash-t0"}
K = 5


def study_run(model: str, pair: str) -> str:
    if model == "gemini-3.5-flash":
        return f"study-{pair}-flash"
    if model in ("gpt-5.6-terra", "sonnet-5.5"):
        return f"study-{pair}-tier"
    return f"study-{pair}"


_cache: dict[tuple[str, str], dict] = {}


def log_index(run: str, model: str) -> dict:
    """task_id -> (system, user) of successful calls in one run's log (latest wins)."""
    key = (run, model)
    if key in _cache:
        return _cache[key]
    idx: dict = {}
    p = RUNS / run / f"{model}.jsonl.gz"
    if model in MODELS and p.exists():
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("error") or r.get("text") is None:
                    continue
                idx[r["task_id"]] = _h(r.get("system"), r.get("user"))
    _cache[key] = idx
    return idx


def _h(system, user) -> str:
    return hashlib.sha1(f"{system}\x1f{user}".encode("utf-8")).hexdigest()


def have(tasks, idx) -> bool:
    return bool(tasks) and all(idx.get(t.task_id) == _h(t.system, t.user) for t in tasks)


def grid_tasks(pair: str, unit: dict) -> dict[str, list]:
    """The grid's tasks for one pair, from the harness builders (the prompts the grid would send)."""
    ds = load_dataset_dir(H / ".." / "data" / pair)
    seed = unit["seed"]
    cfg = {"filter_algebra": {"enabled": True, "n_rows": 200, "count_batch": 10},
           "batch": {"enabled": True, "n_rows": 300, "batch_size": 10}}
    if "idempotence" in unit:
        cfg["idempotence"] = {"enabled": True, "n_rows": 200}
    tasks, _ = build(ds, cfg, K, seed, "lotus")
    keep_b = set(unit["batch"]["prompts"]["N1"]) | set(unit["batch"]["prompts"]["rot"]) \
        | set(unit["batch"]["prompts"]["N2"])
    analysed = set(unit["batch"]["analysed_rows"])
    out = defaultdict(list)
    for t in tasks:
        if t.identity == "filter_algebra":
            out["count" if t.condition == "count" else "filter"].append(t)
            if t.condition == "p":
                out["count_p"].append(t)
        elif t.identity == "batch":
            if t.condition == "single" and t.items[0] in analysed:
                out["batch"].append(t)
            elif t.condition != "single" and t.unit in keep_b:
                out["batch"].append(t)
        elif t.identity == "idempotence":
            out["idempotence"].append(t)  # stage 0 and condition x; the fx follow-ups mirror stage 0
    return out


def match_tasks(name: str, unit: dict) -> dict[str, list]:
    import symmetry

    pair = unit["pair_config"]
    ds = load_dataset_dir(H / ".." / "data" / pair)
    out = defaultdict(list)
    out["symmetry"] = symmetry.tasks(ds, unit["symmetry"]["pairs"], K)
    if name != "wdc":
        tasks, _ = build(ds, {"placement": {"enabled": True, "n_tuples": len(unit["placement"]["tuples"])}},
                         K, unit["seed"], "lotus")
        out["placement"] = tasks
    return out


def reword_done(model: str, pair: str, conds: list[str]) -> bool:
    """The rewording run holds the five logical claims and every rewording, 1,000 calls each (200 rows x 5)."""
    p = RUNS / f"ext-placebo-{pair}" / f"{model}.jsonl.gz"
    if model not in MODELS or not p.exists():
        return False
    n = defaultdict(int)
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("error") and r.get("text") is not None:
                n[(r["identity"], r["condition"])] += 1
    need = [("filter_algebra", c) for c in ("p", "not_p", "q", "p_and_q", "q_and_p")] + \
           [("reword/filter_algebra", c) for c in conds]
    return all(n[k] >= 1000 for k in need)


# --- prices -------------------------------------------------------------------------------------------

def mean_costs() -> dict:
    """(model, dataset, op) -> mean logged cost per call, over the study and rewording runs."""
    acc = defaultdict(list)
    for run_dir in sorted(RUNS.glob("study-*")) + sorted(RUNS.glob("ext-placebo-*")):
        design = run_dir / "design.json"
        if not design.exists():
            continue
        ds = json.loads(design.read_text())["dataset"].split("-")[0]
        for f in run_dir.glob("*.jsonl.gz"):
            model = f.name.split(".jsonl")[0]
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 20000:
                        break
                    r = json.loads(line)
                    c = (r.get("usage") or {}).get("cost")
                    if c is not None and not r.get("error"):
                        acc[(model, ds, r["op"])].append(float(c))
    return {k: statistics.fmean(v) for k, v in acc.items() if v}


def price(costs: dict, model: str, dataset: str, op: str) -> float:
    m = PRICE_AS.get(model, model)
    if (m, dataset, op) in costs:
        return costs[(m, dataset, op)]
    prod = [costs[(m, d, op)] for d in ("wdc", "abt") if (m, d, op) in costs]
    base = statistics.fmean(prod) if prod else max(v for (mm, _, oo), v in costs.items() if mm == m)
    ref = "gemini-3.5-flash-lite"
    r_new = costs.get((ref, dataset, op))
    r_prod = [costs[(ref, d, op)] for d in ("wdc", "abt") if (ref, d, op) in costs]
    scale = r_new / statistics.fmean(r_prod) if r_new and r_prod else 1.0
    return base * scale


# --- main ---------------------------------------------------------------------------------------------

def main() -> int:
    costs = mean_costs()
    plan = []  # rows: model, where, test, status, calls, op, dataset
    for pair in PAIRS:
        _cache.clear()
        unit = json.loads((U / f"{pair}.json").read_text())
        gt = grid_tasks(pair, unit)
        dsname = pair.split("-")[0]
        conds = unit["rewording"]["conditions"]
        for model in MODELS:
            idx = log_index(study_run(model, pair), model)
            f_done = have(gt["filter"], idx)
            c_done = have(gt["count"], idx) and f_done  # COUNT is compared with p from the same run
            b_done = have(gt["batch"], idx)
            i_done = "idempotence" in gt and have(gt["idempotence"], idx)
            r_done = reword_done(model, pair, conds)
            n_rw = 1000 * (5 + len(conds))
            # The rewording run carries the five logical claims, so a model without it gets one combined
            # run: logic + rewordings (+ COUNT batches when COUNT is missing, answered beside p in that run).
            if not r_done:
                plan.append((model, pair, "logic + rewordings (one run)", "run", n_rw, "filter", dsname))
                if not c_done:
                    plan.append((model, pair, "COUNT, in the same run", "run", 100, "count", dsname))
            else:
                plan.append((model, pair, "logic + rewordings (one run)", "have", n_rw, "filter", dsname))
                if not c_done:
                    # COUNT needs p answered in the same session: 20 batches plus p on the 200 rows
                    plan.append((model, pair, "COUNT with p, new session", "run", 100, "count", dsname))
                    plan.append((model, pair, "COUNT with p, new session", "run", 1000, "filter", dsname))  # p, one row
            plan.append((model, pair, "filter tests (identity runs)", "have" if f_done else "from the combined run",
                         0, "filter", dsname))
            if c_done:
                plan.append((model, pair, "COUNT", "have", 100, "count", dsname))
            n_single = sum(t.condition == "single" for t in gt["batch"])
            plan.append((model, pair, "batching", "have" if b_done else "run", len(gt["batch"]) - n_single,
                         "filter_batch", dsname))
            plan.append((model, pair, "batching", "have" if b_done else "run", n_single, "filter", dsname))
            if "idempotence" in gt:
                plan.append((model, pair, "idempotence", "have" if i_done else "run",
                             3 * len(gt["idempotence"]) // 2, "map", dsname))
    for name in ("wdc", "abt", "argkp"):
        unit = json.loads((U / f"match-{name}.json").read_text())
        mt = match_tasks(name, unit)
        run_base = {"wdc": "wdc-A", "abt": "abt-A", "argkp": "argkp-A"}[name]
        for model in MODELS:
            run = study_run(model, run_base) if name != "argkp" else "study-argkp-A"
            idx = log_index(run, model)
            s_done = have(mt["symmetry"], idx)
            plan.append((model, name, "symmetry", "have" if s_done else "run", len(mt["symmetry"]), "match", name))
            n_pl = 3 * K * len(unit["placement"]["tuples"])
            p_done = name != "wdc" and have(mt["placement"], idx)
            plan.append((model, name, "placement", "have" if p_done else "run", n_pl, "filter", name))

    # where each cell's data comes from (the analysis reads exactly these runs)
    sources = {}
    for model, where, test, status, calls, op, ds in plan:
        match = where in ("wdc", "abt", "argkp")
        if test.startswith("logic + rewordings") or test == "filter tests (identity runs)":
            # negation, fusion, conjunct order and the rewordings all come from the run that holds the rewordings
            src = f"ext-placebo-{where}" if reword_done(model, where, json.loads((U / f"{where}.json").read_text())["rewording"]["conditions"]) else f"grid-{where}-lr"
            key = "logic and rewordings"
        elif test.startswith("COUNT"):
            src = study_run(model, where) if status == "have" else (
                f"grid-{where}-lr" if test == "COUNT, in the same run" else f"grid-{where}-count")
            key = "COUNT"
        elif match:
            base = "argkp-A" if where == "argkp" else f"{where}-A"
            src = (("study-argkp-A" if where == "argkp" else study_run(model, base)) if status == "have"
                   else f"grid-{where}-{'sym' if test == 'symmetry' else 'place'}")
            key = test
        else:
            src = study_run(model, where) if status == "have" else f"grid-{where}-{'batch' if test == 'batching' else 'idem'}"
            key = test
        sources.setdefault(model, {}).setdefault(where, {})[key] = src

    # cost of the calls to run
    total_calls, total_usd = 0, 0.0
    by_model = defaultdict(lambda: [0, 0.0])
    rows = []
    for model, where, test, status, calls, op, ds in plan:
        if status != "run":
            continue
        usd = calls * price(costs, model, ds if ds in ("wdc", "abt", "cuad", "argq", "arxiv", "argkp") else "wdc", op)
        total_calls += calls
        total_usd += usd
        by_model[model][0] += calls
        by_model[model][1] += usd
        rows.append((model, where, test, calls, usd))

    out = ["# Grid coverage and quote", "",
           "Generated by `grid/coverage.py` from the run logs. A unit counts as done only if every grid task has a "
           "successful logged call with a byte-identical prompt, all repeats, from one run.", ""]
    out += ["## What exists and what runs", "",
            "| Model | " + " | ".join(PAIRS) + " | WDC match | Abt match | ArgKP |",
            "|---|" + "---|" * (len(PAIRS) + 3)]
    cell = defaultdict(list)
    for model, where, test, status, calls, op, ds in plan:
        if status == "run":
            short = {"logic + rewordings (one run)": "L+R", "COUNT, in the same run": "C",
                     "COUNT with p, new session": "C", "batching": "B", "idempotence": "I",
                     "symmetry": "S", "placement": "P"}[test]
            cell[(model, where)].append(short)
    for model in MODELS:
        cells = [",".join(cell[(model, p)]) or "done" for p in PAIRS] + \
                [",".join(cell[(model, m)]) or "done" for m in ("wdc", "abt", "argkp")]
        out.append(f"| {SHORT[model]} | " + " | ".join(cells) + " |")
    out += ["", "L+R: the five logical claims and the rewordings in one run (200 rows); C: COUNT batches (with p in "
            "the same session); B: batching; I: idempotence; S: symmetry; P: placement. done: every grid unit is "
            "already logged.", "", "## Calls and cost to run", "", "| Model | Calls | Estimated US$ |", "|---|---|---|"]
    for model in MODELS:
        c, u = by_model[model]
        out.append(f"| {SHORT[model]} | {c:,} | {u:.2f} |")
    out.append(f"| **Total** | **{total_calls:,}** | **{total_usd:.2f}** |")
    out += ["", "Prices: each model's mean logged cost per call of the same operation on the same dataset; where the "
            "model never ran that dataset, its product-data mean scaled by the dataset's prompt-length ratio for "
            "Flash-Lite. Qwen is priced from its logged temperature-0 calls (same model and prices)."]
    (ROOT / "grid" / "COVERAGE.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (ROOT / "grid" / "coverage.json").write_text(json.dumps(
        {"plan": [dict(zip(["model", "where", "test", "status", "calls", "op", "dataset"], p)) for p in plan],
         "to_run": [dict(zip(["model", "where", "test", "calls", "usd"], r)) for r in rows],
         "sources": sources, "total_calls": total_calls, "total_usd": total_usd}, indent=1) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
