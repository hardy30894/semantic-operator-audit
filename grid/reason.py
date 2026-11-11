"""Reasoning on: the grid's identities again, with reasoning at medium effort, for Sonnet 5.5, Haiku 5.5 and Luna.

Every cell of the grid except the rewordings, on the same units (grid/units/), with byte-identical prompts and task
ids (the harness's own builders and the grid's own run machinery, grid/run.py), five repeats:
  logic   negation, fusion and conjunct order (p, not p, q, p and q, q and p) and COUNT, 200 rows and 20 batches,
          on each of the 11 condition pairs
  batch   batch neighbors, position and batching, on each of the 11 condition pairs
  idem    idempotence, 200 values, on the six pairs ending in -A (one per dataset)
  sym     symmetry, 200 record pairs, on WDC, Abt-Buy and ArgKP
  place   placement and renamed columns, on WDC (200), Abt-Buy (200) and ArgKP (120 tuples)
The reasoning-off side of each comparison is the grid's existing runs (grid/coverage.json, coverage-argkp.json).

Models (the earlier reasoning arms' roster entries, at medium effort, 4,096 output tokens):
  sonnet-5.5-reasoning-medium   Anthropic API direct (ANTHROPIC_API_KEY), adaptive thinking, effort medium
  haiku-5.5-reasoning-medium    Anthropic API direct (ANTHROPIC_API_KEY), adaptive thinking, effort medium
  gpt-5.6-luna-reasoning-medium OpenRouter pinned to OpenAI without fallbacks, reasoning effort medium
  sonnet-5.5-reasoning-high     the alternative for Sonnet: adaptive thinking, effort high, 8,192 output tokens
                                (the earlier high arms' entry), since at medium Sonnet did not think in the smoke test
Logs: harness/runs/rgrid-<where>-<kind>/<model>.jsonl.gz, one compressed record per call (prompt, reply, parse,
tokens, cost, served model and host, the full raw API response with any thinking block), flushed per call;
a lock against a second writer; resume skips calls that succeeded; one seeded order per run; a spending cap per
run (a cap stop ends that run only, and the model goes on with its next run).

Guards (ReasonBackend): reasoning must be on in the request (checked before any call); a call is turned into a
logged failure and the model stops when the served model or host differs from the pinned one, or the account hits
a limit. A call without reasoning is NOT a failure: these models decide per call whether to reason, so the runs
record how often they did (reasoning tokens for Luna, a thinking block or output beyond the answer for Claude).

Usage (from the project root):
  python3 grid/reason.py plan                        runs, calls, estimates and caps per model
  python3 grid/reason.py check                       build every run's tasks and check them against the units
  python3 grid/reason.py smoke --model NAME          20 calls of one run of each kind, into harness/runs-smoke/
  python3 grid/reason.py run --model NAME            every run of one model, in order (resume: run it again)
  python3 grid/reason.py run --model NAME --offline --root DIR    the same with a simulated backend (tests)
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grid"))
import run  # noqa: E402  (the grid's runner: units, task builders, run_tasks, paths)
from backends import LIMIT_MARKERS, GridBackend  # noqa: E402
from semaudit.core import Completion  # noqa: E402
from semaudit.identities import _host_slug, build  # noqa: E402

H, GRID = run.H, run.GRID
PAIRS = run.PAIRS  # the 11 condition pairs
MATCH = ["wdc", "abt", "argkp"]
K = run.K
OFF = {"sonnet-5.5-reasoning-medium": "sonnet-5.5", "haiku-5.5-reasoning-medium": "haiku-5.5",
       "gpt-5.6-luna-reasoning-medium": "gpt-5.6-luna", "sonnet-5.5-reasoning-high": "sonnet-5.5"}
MODELS = list(OFF)  # Sonnet at medium or at high: at medium it did not think in the smoke test (grid/REASON.md)
SPECS = {
    "sonnet-5.5-reasoning-medium": {"name": "sonnet-5.5-reasoning-medium", "kind": "anthropic",
                                    "model": "claude-sonnet-5-5",
                                    "params": {"max_tokens": 4096, "thinking": "adaptive", "effort": "medium"},
                                    "price_in_per_mtok": 2.00, "price_out_per_mtok": 10.00},
    "haiku-5.5-reasoning-medium": {"name": "haiku-5.5-reasoning-medium", "kind": "anthropic",
                                   "model": "claude-haiku-5-5",
                                   "params": {"max_tokens": 4096, "thinking": "adaptive", "effort": "medium"},
                                   "price_in_per_mtok": 0.10, "price_out_per_mtok": 0.50},
    "sonnet-5.5-reasoning-high": {"name": "sonnet-5.5-reasoning-high", "kind": "anthropic",
                                  "model": "claude-sonnet-5-5",
                                  "params": {"max_tokens": 8192, "thinking": "adaptive", "effort": "high"},
                                  "price_in_per_mtok": 2.00, "price_out_per_mtok": 10.00},
    "gpt-5.6-luna-reasoning-medium": {"name": "gpt-5.6-luna-reasoning-medium", "kind": "openrouter",
                                      "model": "openai/gpt-5.6-luna",
                                      "params": {"upstream": "openai", "reasoning_effort": "medium", "max_tokens": 4096},
                                      "price_in_per_mtok": 0.20, "price_out_per_mtok": 1.20},
}
# Cost of a reasoning call over the same call without reasoning, measured on the earlier reasoning arms
# (ext-reason-*: the arm against its own reasoning-off arm, same prompts, same session); the larger of the
# measured ratios, so estimates lean high. Sonnet's is its high-effort ratio (it never ran at medium on a full design).
RATIO = {"sonnet-5.5-reasoning-medium": 3.4, "sonnet-5.5-reasoning-high": 3.4, "haiku-5.5-reasoning-medium": 2.9, "gpt-5.6-luna-reasoning-medium": 2.3}
CAP_FACTOR = 2.0
CAP_FLOOR = 0.10
ORDER = ["logic", "batch", "idem", "sym", "place"]


def reasoning_on(params: dict[str, Any], effort: str) -> bool:
    """The model's reasoning effort in the request as the backend sends it: Anthropic's adaptive thinking with that
    effort (thinking.type and output_config.effort), or OpenRouter's reasoning effort."""
    thinking = params.get("thinking")
    anthropic = (thinking == "adaptive" or (isinstance(thinking, dict) and thinking.get("type") == "adaptive")) \
        and effort in (params.get("effort"), (params.get("output_config") or {}).get("effort"))
    openrouter = effort in (params.get("reasoning_effort"), (params.get("reasoning") or {}).get("effort"))
    return anthropic or openrouter


class ReasonBackend(GridBackend):
    """GridBackend with reasoning expected: the reasoning-token and thinking-block guards are off; the served-model,
    host and account-limit guards stay. Reasoning must be on in the request before any call is made."""

    def __init__(self, inner: Any, offline: bool = False) -> None:
        super().__init__(inner)
        effort = self.name.rsplit("-", 1)[-1]  # ...-reasoning-medium or ...-reasoning-high
        if not offline and not reasoning_on(self.describe()["params"], effort):
            raise SystemExit(f"{self.name}: reasoning is not on at {effort} in {self.describe()['params']}")

    def _guard(self, comp: Completion, raw: Any) -> str | None:
        if comp.error:
            low = comp.error.lower()
            return f"account limit: {comp.error[:200]}" if any(m in low for m in LIMIT_MARKERS) else None
        requested = self.describe()["model"]
        if comp.served_model is not None and comp.served_model != requested:
            return f"served model {comp.served_model!r}, requested {requested!r}"
        pinned = ((self.describe()["params"].get("provider") or {}).get("order") or [None])[0]
        served = (comp.extra or {}).get("served_provider")
        if pinned and served and _host_slug(str(served)) != str(pinned).split("/")[0].lower():
            return f"served by {served!r}, pinned to {pinned!r}"
        return None


def make(model: str, offline: bool) -> Any:
    from extension import priced
    from extension.backends import make_backend_ext

    priced.install()
    spec = ({"kind": "simulated", "name": model, "params": {"seed": 7, "standin_truth": True}}
            if offline else SPECS[model])
    return ReasonBackend(make_backend_ext(spec), offline=offline)


# --- runs -------------------------------------------------------------------------------------------

def runs_of() -> list[dict[str, Any]]:
    out = []
    for pair in PAIRS:
        out.append({"kind": "logic", "where": pair})
        out.append({"kind": "batch", "where": pair})
        if pair.endswith("-A"):
            out.append({"kind": "idem", "where": pair})
    for m in MATCH:
        out.append({"kind": "sym", "where": m})
        out.append({"kind": "place", "where": m})
    for r in out:
        r["run_id"] = f"rgrid-{r['where']}-{r['kind']}"
    return sorted(out, key=lambda r: (ORDER.index(r["kind"]), r["where"]))


def tasks_for(kind: str, where: str):
    """Dataset, tasks, design, seed and config of one run: run.tasks_for for the grid's kinds, plus 'logic' (the
    five claims and COUNT of the combined run, without its rewordings), checked against the units."""
    if kind != "logic":
        return run.tasks_for(kind, where)
    unit = run.unit_of(where)
    from semaudit.data import load_dataset_dir

    ds = load_dataset_dir(H / ".." / "data" / where)
    cfg = {"filter_algebra": {"enabled": True, "n_rows": 200, "count_batch": 10}}
    tasks, design = build(ds, cfg, K, unit["seed"], "lotus")
    design.pop("gold", None)
    assert design["filter_algebra"]["rows"] == unit["filter"]["rows"], where
    assert design["filter_algebra"]["count_batches"] == unit["filter"]["count_batches"], where
    design["grid"] = {"kind": kind, "where": where, "units": f"grid/units/{where}.json"}
    config = {"run": {"seed": unit["seed"], "repeats": K, "concurrency": 16, "prompt_style": "lotus"},
              "data": {"kind": "dir", "path": f"../data/{where}"}, "identities": cfg, "grid": design["grid"]}
    return ds, tasks, design, unit["seed"], config


def n_calls(kind: str, where: str) -> int:
    _, tasks, _, _, _ = tasks_for(kind, where)
    return len(tasks) + (sum(t.stage == 0 for t in tasks) if kind == "idem" else 0)  # idempotence follow-ups


def off_rates() -> dict[tuple[str, str, str], float]:
    """(off model, dataset, op) -> mean cost per call, from the grid's and the earlier logs the grid reuses."""
    acc: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for run_dir in sorted(H.joinpath("runs").glob("grid-*")) + sorted(H.joinpath("runs").glob("study-*")):
        design = run_dir / "design.json"
        if not design.exists():
            continue
        ds = json.loads(design.read_text())["dataset"].split("-")[0]
        for off in set(OFF.values()):
            f = run_dir / f"{off}.jsonl.gz"
            if not f.exists():
                continue
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 5000:
                        break
                    r = json.loads(line)
                    c = (r.get("usage") or {}).get("cost")
                    if c is not None and not r.get("error"):
                        acc[(off, ds, r["op"])].append(float(c))
    return {k: statistics.fmean(v) for k, v in acc.items() if v}


def plan(cap_factor: float = CAP_FACTOR) -> dict[str, list[dict[str, Any]]]:
    rates = off_rates()
    per_run = {}
    for r in runs_of():
        _, tasks, _, _, _ = tasks_for(r["kind"], r["where"])
        ds = (r["where"] if r["kind"] in ("sym", "place") else r["where"].split("-")[0])
        ops = defaultdict(int)
        for t in tasks:
            ops[t.op] += 1
        if r["kind"] == "idem":
            ops["map"] += sum(t.stage == 0 for t in tasks)
        per_run[r["run_id"]] = (ds, dict(ops))
    out = {}
    for m in MODELS:
        off = OFF[m]
        rows = []
        for r in runs_of():
            ds, ops = per_run[r["run_id"]]
            usd = 0.0
            for op, n in ops.items():
                rate = rates.get((off, ds, op)) or statistics.fmean(
                    [v for (mm, _, oo), v in rates.items() if mm == off and oo == op] or [0.001])
                usd += n * rate * RATIO[m]
            rows.append({**r, "calls": sum(ops.values()), "usd": usd, "cap": max(CAP_FLOOR, cap_factor * usd)})
        out[m] = rows
    return out


def run_one(r: dict[str, Any], model: str, offline: bool, root: Path, smoke: int | None = None,
            cap: float | None = None) -> None:
    import os

    backend = make(model, offline)
    ds, tasks, design, seed, config = tasks_for(r["kind"], r["where"])
    config = {**config, "reasoning": {"model": model, "effort": model.rsplit("-", 1)[-1], "reasoning_off_side": OFF[model]}}
    if smoke:
        tasks = random.Random(0).sample([t for t in tasks if t.stage != 0], smoke)
    cwd = os.getcwd()
    os.chdir(H)
    try:
        run.run_tasks(r["run_id"], ds, tasks, design, seed, config, backend, root, None if offline else cap)
    finally:
        os.chdir(cwd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["plan", "check", "smoke", "run"])
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--root", type=Path)
    ap.add_argument("--cap-factor", type=float, default=CAP_FACTOR)
    args = ap.parse_args(argv)
    if args.command == "check":
        n = 0
        for r in runs_of():
            n += n_calls(r["kind"], r["where"])
        print(f"every run's tasks match grid/units: {len(runs_of())} runs, {n:,} calls per model")
        return 0
    p = plan(args.cap_factor)
    if args.command == "plan":
        tot_c, tot_u = 0, 0.0
        for m in MODELS:
            c, u = sum(r["calls"] for r in p[m]), sum(r["usd"] for r in p[m])
            tot_c, tot_u = tot_c + c, tot_u + u
            print(f"{m}: {len(p[m])} runs, {c:,} calls, est. US${u:.2f}, caps US${sum(r['cap'] for r in p[m]):.2f}")
        print(f"total: {tot_c:,} calls, est. US${tot_u:.2f}")
        (GRID / "reason-plan.json").write_text(json.dumps(p, indent=1) + "\n", encoding="utf-8")
        return 0
    if not args.model:
        raise SystemExit("--model is required")
    if args.command == "smoke":
        root = args.root or (H / "runs-smoke")
        first = {}
        for x in p[args.model]:
            first.setdefault(x["kind"], x)  # one run of each kind: logic, batch, idem, sym, place
        for r in first.values():
            run_one(r, args.model, args.offline, root, smoke=20, cap=1.0)
        print(f"smoke: 20 calls of one run of each kind for {args.model} in {root}")
        return 0
    root = args.root or (H / "runs")
    if args.offline and args.root is None:
        raise SystemExit("--offline needs --root (a scratch directory), so simulated calls never reach harness/runs")
    capped = []
    for r in p[args.model]:
        print(f"== {args.model}: {r['run_id']} (est. ${r['usd']:.2f}, cap ${r['cap']:.2f})", flush=True)
        try:
            run_one(r, args.model, args.offline, root, cap=r["cap"])
        except RuntimeError as exc:
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
