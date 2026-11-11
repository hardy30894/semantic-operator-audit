"""Run the consistent grid's missing cells (grid/DESIGN.md).

Every run is built with the harness's own builders from the pair's frozen seed, checked against the
curated units in grid/units/, and executed with the harness's own resume-safe machinery: one compressed
log per model, flushed after every call, a lock against a second writer, resume that skips calls that
succeeded, one seeded random order per run, stops on a spending cap, on failures and on another host.
Runs that pair the logical claims with the rewordings go through extension.runner.run_ext unchanged.

Usage (from the project root):
  python3 grid/run.py plan                       runs, calls, estimates and caps per model
  python3 grid/run.py check                      build every run's tasks and check them against the units
  python3 grid/run.py smoke --model NAME         20 calls of the model's first run, into harness/runs-smoke/
  python3 grid/run.py run --model NAME           every run of one model, in plan order (resume: run it again)
  python3 grid/run.py run --model NAME --offline the same with a simulated backend into a scratch root (tests)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "harness"
GRID = ROOT / "grid"
sys.path.insert(0, str(H))
sys.path.insert(0, str(GRID))

from semaudit import __version__  # noqa: E402
from semaudit.core import stable_seed  # noqa: E402
from semaudit.data import load_dataset_dir  # noqa: E402
from semaudit.identities import build, idempotence_followups  # noqa: E402
from semaudit.prompts import TEMPLATE_VERSION  # noqa: E402
from semaudit.runner import CallLog, _versions, _write_atomic, execute, find_log, read_log  # noqa: E402

from backends import GridBackend  # noqa: E402

PAIRS = ["wdc-A", "wdc-B", "abt-A", "abt-B", "cuad-A", "cuad-B", "argq-A", "argq-B", "arxiv-A", "arxiv-B", "argkp-A"]
MODELS = ["haiku-5.5", "gpt-5.6-luna", "gemini-3.5-flash-lite", "qwen-3.7-flash", "deepseek-v3.2",
          "gemini-3.5-flash", "sonnet-5.5", "gpt-5.6-terra"]
K = 5
CAP_FACTOR = 1.5
CAP_FLOOR = 0.05
MATCH_CONFIG = {"wdc": "wdc-A", "abt": "abt-A", "argkp": "argkp-A"}
KIND_OF_TEST = {"logic + rewordings (one run)": "lr", "COUNT, in the same run": "lr", "COUNT with p, new session": "count",
                "batching": "batch", "idempotence": "idem", "symmetry": "sym", "placement": "place"}
ORDER = ["lr", "count", "batch", "idem", "sym", "place"]


# --- roster -----------------------------------------------------------------------------------------

def roster() -> dict[str, dict[str, Any]]:
    """The grid's 7 models. Six are the logged arms' roster entries unchanged; Qwen is its logged entry
    without temperature 0, the fixed seed and logprobs, so it samples at the provider default like the rest."""
    specs: dict[str, dict[str, Any]] = {}
    for name in ("models-v2.toml", "models-v3.toml"):
        for b in tomllib.loads((H / "configs" / name).read_text(encoding="utf-8"))["backends"]:
            specs[b["name"]] = b
    out = {m: copy.deepcopy(specs[m]) for m in MODELS if m in specs}
    qwen = copy.deepcopy(specs["qwen-3.7-flash-t0"])
    qwen["name"] = "qwen-3.7-flash"
    for key in ("temperature", "seed", "top_logprobs"):
        qwen["params"].pop(key, None)
    out["qwen-3.7-flash"] = qwen
    missing = [m for m in MODELS if m not in out]
    if missing:
        raise SystemExit(f"roster entries missing: {missing}")
    return {m: out[m] for m in MODELS}


def make(model: str, offline: bool) -> Any:
    from extension import priced
    from extension.backends import make_backend_ext

    priced.install()
    spec = ({"kind": "simulated", "name": model, "params": {"seed": 7, "standin_truth": True}}
            if offline else roster()[model])
    return GridBackend(make_backend_ext(spec))


# --- runs -------------------------------------------------------------------------------------------

def unit_of(where: str) -> dict[str, Any]:
    name = f"match-{where}.json" if where in MATCH_CONFIG else f"{where}.json"
    return json.loads((GRID / "units" / name).read_text(encoding="utf-8"))


def plan(cap_factor: float = CAP_FACTOR, coverage: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Per model: its runs (kind, where, run id, estimated US$), from grid/coverage.json (or, for ArgKP's
    pair added later, grid/coverage-argkp.json)."""
    cov = json.loads((coverage or GRID / "coverage.json").read_text(encoding="utf-8"))
    runs: dict[tuple[str, str, str], float] = defaultdict(float)
    for r in cov["to_run"]:
        runs[(r["model"], KIND_OF_TEST[r["test"]], r["where"])] += r["usd"]
    out: dict[str, list[dict[str, Any]]] = {m: [] for m in MODELS}
    for (model, kind, where), usd in runs.items():
        out[model].append({"kind": kind, "where": where, "run_id": f"grid-{where}-{kind}", "usd": usd,
                           "cap": max(CAP_FLOOR, cap_factor * usd)})
    for m in out:
        out[m].sort(key=lambda r: (ORDER.index(r["kind"]), r["where"]))
    return out


def placebo_config(pair: str) -> Path:
    """The pair's rewording config: the earlier runs' (harness/configs), or the grid's own for ArgKP (grid/configs)."""
    own = GRID / "configs" / f"ext-placebo-{pair}.toml"
    return own if own.exists() else H / "configs" / f"ext-placebo-{pair}.toml"


def lr_config(pair: str) -> dict[str, Any]:
    """The rewording run's config with only the reword segment and the COUNT batches on (the logical
    claims, every rewording and COUNT on the same 200 rows, in one session)."""
    cfg = tomllib.loads(placebo_config(pair).read_text(encoding="utf-8"))
    cfg.pop("include", None)
    cfg.pop("backends", None)
    cfg["identities"] = {"filter_algebra": {"enabled": True, "n_rows": 200, "count_batch": 10}}
    cfg["ext"]["segments"] = ["reword"]
    cfg["ext"]["base_count"] = True
    cfg.pop("reasoning", None)
    return cfg


def tasks_for(kind: str, where: str) -> tuple[Any, list, dict[str, Any], int, dict[str, Any]]:
    """Dataset, tasks, design, seed and config of one non-rewording run, checked against the units."""
    unit = unit_of(where)
    if kind in ("sym", "place"):
        pair = MATCH_CONFIG[where]
        seed = tomllib.loads((H / "configs" / f"study-{pair}.toml").read_text(encoding="utf-8"))["run"]["seed"]
    else:
        pair, seed = where, unit["seed"]
    ds = load_dataset_dir(H / ".." / "data" / pair)
    if kind == "count":
        cfg = {"filter_algebra": {"enabled": True, "n_rows": 200, "count_batch": 10}}
    elif kind == "batch":
        cfg = {"batch": {"enabled": True, "n_rows": 300, "batch_size": 10}}
    elif kind == "idem":
        cfg = {"idempotence": {"enabled": True, "n_rows": 200}}
    elif kind == "sym":
        cfg = {}
    elif kind == "place":
        cfg = {"placement": {"enabled": True, "n_tuples": len(unit["placement"]["tuples"])}}
        if where == "wdc":  # WDC records carry no shop: every offer is its own source (a self-join)
            for rid, rec in ds.records.items():
                rec["source"] = rid
    else:
        raise ValueError(kind)
    if kind == "sym":  # 100 matches and 100 look-alikes (grid/symmetry.py), the harness's own requests
        import symmetry

        tasks = symmetry.tasks(ds, unit["symmetry"]["pairs"], K)
        design = {"dataset": ds.name, "repeats": K, "style": "lotus", "symmetry": symmetry.design(unit)}
    else:
        tasks, design = build(ds, cfg, K, seed, "lotus")
    design.pop("gold", None)
    if kind == "count":
        assert design["filter_algebra"]["rows"] == unit["filter"]["rows"], where
        assert design["filter_algebra"]["count_batches"] == unit["filter"]["count_batches"], where
        tasks = [t for t in tasks if t.condition in ("p", "count")]
    elif kind == "batch":
        assert design["batch"]["rows"] == unit["batch"]["grid_rows"], where
        prompts = unit["batch"]["prompts"]
        keep = set(prompts["N1"]) | set(prompts["rot"]) | set(prompts["N2"])
        analysed = set(unit["batch"]["analysed_rows"])
        tasks = [t for t in tasks if (t.condition == "single" and t.items[0] in analysed)
                 or (t.condition != "single" and t.unit in keep)]
        design["batch"]["analysed_rows"] = unit["batch"]["analysed_rows"]
    elif kind == "idem":
        assert design["idempotence"]["rows"] == unit["idempotence"]["rows"], where
    elif kind == "place":
        assert list(design["placement"]["tuples"]) == unit["placement"]["tuples"], where
    design["grid"] = {"kind": kind, "where": where, "units": f"grid/units/{'match-' if kind in ('sym', 'place') else ''}{where}.json"}
    config = {"run": {"seed": seed, "repeats": K, "concurrency": 16, "prompt_style": "lotus"},
              "data": {"kind": "dir", "path": f"../data/{pair}"}, "identities": cfg,
              "grid": design["grid"]}
    return ds, tasks, design, seed, config


def run_tasks(run_id: str, ds: Any, tasks: list, design: dict[str, Any], seed: int, config: dict[str, Any],
              backend: Any, root: Path, cap: float | None) -> None:
    """semaudit.runner.run_experiment's per-backend loop for a given task list (design check, resume,
    settings check, stage 0 then the main phase in one seeded order, manifest)."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    design_path = run_dir / "design.json"
    canonical = json.loads(json.dumps(design))
    if design_path.exists():
        if json.loads(design_path.read_text(encoding="utf-8")) != canonical:
            raise RuntimeError(f"{run_dir} holds a different design; refusing to mix")
    else:
        _write_atomic(design_path, json.dumps(canonical))
    _write_atomic(run_dir / "config.json", json.dumps(config, indent=2))
    backend.bind_run(run_id)
    log_path = find_log(run_dir, backend.name)
    previous = read_log(log_path)
    desc = backend.describe()
    if any((r.get("model_requested"), r.get("params")) != (desc["model"], desc["params"]) for r in previous):
        raise RuntimeError(f"{log_path} holds calls made with other settings")
    done = {r["task_id"] for r in previous if not r["error"]}
    spent_before = sum(float((r.get("usage") or {}).get("cost") or 0) for r in previous)
    started = time.time()
    with CallLog(log_path) as log:
        conc = getattr(backend, "concurrency", None) or config["run"]["concurrency"]
        stage0 = [t for t in tasks if t.stage == 0 and t.task_id not in done]
        new = execute(stage0, backend, log, run_id, conc, stable_seed(seed, run_id, backend.name, 0),
                      budget=None if cap is None else cap - spent_before)
        latest = {r["task_id"]: r for r in previous + new if not r["error"]}
        main = [t for t in tasks if t.stage != 0] + idempotence_followups(ds, latest.values())
        spent0 = sum(float((r.get("usage") or {}).get("cost") or 0) for r in new)
        execute([t for t in main if t.task_id not in done], backend, log, run_id, conc,
                stable_seed(seed, run_id, backend.name, 1),
                budget=None if cap is None else cap - spent_before - spent0)
    manifest = {"run_id": run_id, "harness_version": __version__, "template_version": TEMPLATE_VERSION,
                "packages": _versions(), "stage1_tasks": len(tasks), "backend": desc,
                "started": started, "finished": time.time()}
    _write_atomic(run_dir / f"manifest-{backend.name}.json", json.dumps(manifest, indent=2))


def check_lr(pair: str) -> tuple[dict[str, Any], int]:
    """Build the combined run's tasks offline and check its rows, COUNT batches and rewordings."""
    from semaudit.data import load_data
    from extension.mitigations import build_ext

    cwd = os.getcwd()
    os.chdir(H)
    try:
        cfg = lr_config(pair)
        tasks, design = build_ext(load_data(cfg["data"]), cfg)
    finally:
        os.chdir(cwd)
    unit = unit_of(pair)
    assert design["filter_algebra"]["rows"] == unit["filter"]["rows"], pair
    assert design["filter_algebra"]["count_batches"] == unit["filter"]["count_batches"], pair
    conds = sorted({t.condition for t in tasks if t.identity == "reword/filter_algebra"})
    assert conds == sorted(unit["rewording"]["conditions"]), (pair, conds)
    return cfg, len(tasks)


def run_one(r: dict[str, Any], model: str, offline: bool, root: Path, smoke: int | None = None) -> None:
    backend = make(model, offline)
    cap = None if offline else r["cap"]
    if r["kind"] == "lr":
        from extension.runner import run_ext

        cfg, _ = check_lr(r["where"])
        cwd = os.getcwd()
        os.chdir(H)
        try:
            if smoke:
                import extension.runner as er

                real_build = er.build_ext

                def small(ds, config):
                    tasks, design = real_build(ds, config)
                    return random.Random(0).sample(tasks, smoke), design
                er.build_ext = small
                try:
                    run_ext(cfg, r["run_id"], [backend], root, max_cost={model: cap} if cap else None)
                finally:
                    er.build_ext = real_build
            else:
                run_ext(cfg, r["run_id"], [backend], root, max_cost={model: cap} if cap else None)
        finally:
            os.chdir(cwd)
        return
    ds, tasks, design, seed, config = tasks_for(r["kind"], r["where"])
    if smoke:
        tasks = random.Random(0).sample([t for t in tasks if t.stage != 0], smoke)
    cwd = os.getcwd()
    os.chdir(H)
    try:
        run_tasks(r["run_id"], ds, tasks, design, seed, config, backend, root, cap)
    finally:
        os.chdir(cwd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["plan", "check", "smoke", "run"])
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--offline", action="store_true", help="simulated backend into a scratch root (no calls)")
    ap.add_argument("--root", type=Path, help="run root (default harness/runs; smoke: harness/runs-smoke)")
    ap.add_argument("--coverage", type=Path, help="the plan to run (default grid/coverage.json)")
    ap.add_argument("--cap-factor", type=float, default=CAP_FACTOR,
                    help="cap per run as a multiple of its estimate (default 1.5); a resume after a cap stop may raise it")
    args = ap.parse_args(argv)
    p = plan(args.cap_factor, args.coverage)
    if args.command == "plan":
        tot = 0.0
        for m in MODELS:
            usd = sum(r["usd"] for r in p[m])
            tot += usd
            print(f"{m}: {len(p[m])} runs, est. US${usd:.2f}, caps US${sum(r['cap'] for r in p[m]):.2f}")
            for r in p[m]:
                print(f"   {r['run_id']:28s} est. ${r['usd']:.2f}  cap ${r['cap']:.2f}")
        print(f"total est. US${tot:.2f}")
        return 0
    if args.command == "check":
        n = 0
        for pair in PAIRS:
            _, k = check_lr(pair)
            n += k
        for kinds in ({r["kind"], r["where"]} for m in MODELS for r in p[m]):
            pass
        seen = {(r["kind"], r["where"]) for m in MODELS for r in p[m] if r["kind"] != "lr"}
        for kind, where in sorted(seen):
            _, tasks, _, _, _ = tasks_for(kind, where)
            n += len(tasks)
        print(f"every run's tasks match grid/units ({len(seen)} runs plus {len(PAIRS)} combined runs checked)")
        return 0
    if not args.model:
        raise SystemExit("--model is required")
    if args.command == "smoke":
        root = args.root or (H / "runs-smoke")
        r = p[args.model][0]
        run_one(r, args.model, args.offline, root, smoke=20)
        print(f"smoke: 20 calls of {r['run_id']} for {args.model} in {root}")
        return 0
    root = args.root or (H / "runs")
    if args.offline and args.root is None:
        raise SystemExit("--offline needs --root (a scratch directory), so simulated calls never reach harness/runs")
    capped = []
    for r in p[args.model]:
        print(f"== {args.model}: {r['run_id']} (cap ${r['cap']:.2f})", flush=True)
        try:
            run_one(r, args.model, args.offline, root)
        except RuntimeError as exc:  # a cap stop ends this run only; every call sent is logged
            if "spend cap" not in str(exc):
                raise
            capped.append(r["run_id"])
            print(f"!! {r['run_id']}: {exc}; continuing with the next run", flush=True)
    if capped:
        print(f"{args.model}: stopped on a cap: {capped}; resume with a higher --cap-factor", flush=True)
        return 3
    print(f"{args.model}: every run done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
