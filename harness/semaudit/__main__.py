"""Command line: plan, run, analyze, power.

    python -m semaudit plan configs/pilot.toml
    python -m semaudit run configs/pilot.toml --run-id pilot-1 --yes
    python -m semaudit analyze runs/pilot-1
    python -m semaudit power
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

from .backends import PAID_KINDS, make_backend
from .data import load_data
from .identities import build

# Rough output sizes, in tokens, for the cost estimate.
_OUTPUT_TOKENS = {"filter": 2, "match": 2, "count": 3, "map": 25}


def load_config(path: str) -> dict[str, Any]:
    """A TOML config; `include = ["models.toml"]` appends the included files' [[backends]]."""
    config = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    for name in config.pop("include", []):
        included = tomllib.loads((Path(path).parent / name).read_text(encoding="utf-8"))
        config.setdefault("backends", []).extend(included.get("backends", []))
    return config


def selected_backends(config: dict[str, Any], names: list[str] | None) -> list[dict[str, Any]]:
    specs = [b for b in config.get("backends", []) if b.get("enabled", True)]
    if names:
        specs = [b for b in specs if b["name"] in names]
        missing = set(names) - {b["name"] for b in specs}
        if missing:
            raise SystemExit(f"unknown or disabled backends: {sorted(missing)}")
    if not specs:
        raise SystemExit("no enabled backends selected")
    return specs


def plan_summary(config: dict[str, Any], specs: list[dict[str, Any]]) -> str:
    return plan_details(config, specs)[0]


def plan_details(config: dict[str, Any], specs: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    """The plan as text, and each backend's estimated cost in USD (the measured pilot rate when
    there is one, else list prices)."""
    ds = load_data(config["data"])
    tasks, _ = build(ds, config["identities"], config["run"]["repeats"], config["run"]["seed"],
                     config["run"].get("prompt_style", "lotus"))
    stage2 = sum(t.condition == "x_in" for t in tasks)  # map(map(x)): one per stage-0 input draw
    per_identity = Counter(t.identity for t in tasks)
    if stage2:
        per_identity["idempotence"] += stage2
    calls = len(tasks) + stage2
    chars = sum(len(t.system) + len(t.user) for t in tasks)
    chars += sum(len(t.system) + len(t.user) for t in tasks if t.condition == "x_in")
    input_tokens = chars / 4
    output_tokens = sum(_OUTPUT_TOKENS.get(t.op, 6 * len(t.items)) for t in tasks)
    output_tokens += _OUTPUT_TOKENS["map"] * stage2
    lines = [f"Dataset {ds.name}: {len(ds.records)} records.",
             f"Calls per backend: {calls} ({', '.join(f'{k} {v}' for k, v in sorted(per_identity.items()))}).",
             f"Approx. tokens per backend: {input_tokens:,.0f} input, {output_tokens:,.0f} visible output."]
    estimates: dict[str, float] = {}
    for spec in specs:
        line = f"  {spec['name']} ({spec['kind']}{', ' + spec['model'] if 'model' in spec else ''})"
        if "price_in_per_mtok" in spec:
            thinking = spec.get("est_thinking_tokens_per_call", 0) * calls
            cost = (input_tokens * spec["price_in_per_mtok"]
                    + (output_tokens + thinking) * spec.get("price_out_per_mtok", 0)) / 1e6
            estimates[spec["name"]] = cost
            line += f": about ${cost:,.2f} at list prices" + (f" including {thinking:,.0f} thinking tokens" if thinking else "")
        if "measured_usd_per_1k_calls" in spec:
            estimates[spec["name"]] = calls * spec["measured_usd_per_1k_calls"] / 1000
            line += f"; about ${estimates[spec['name']]:,.2f} at the measured pilot rate"
        lines.append(line)
    return "\n".join(lines), estimates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="semaudit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan", help="count calls and estimate cost; makes no API calls")
    p_plan.add_argument("config")
    p_plan.add_argument("--backend", action="append")
    p_run = sub.add_parser("run", help="execute a run")
    p_run.add_argument("config")
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--backend", action="append", help="limit to these backend names")
    p_run.add_argument("--yes", action="store_true", help="confirm calls to paid APIs")
    p_run.add_argument("--max-cost-factor", type=float,
                       help="stop a backend once its logged spend in this run exceeds this multiple of its "
                            "estimate (at least $0.10)")
    p_an = sub.add_parser("analyze", help="tests, transitivity and diagnostics for a run")
    p_an.add_argument("run_dir")
    p_an.add_argument("--compare-run", help="second run of the same design, for the cross-run noise floor")
    p_an.add_argument("--n-perm", type=int, default=5000)
    p_an.add_argument("--n-boot", type=int, default=5000)
    p_pow = sub.add_parser("power", help="power of the binary test by simulation")
    p_pow.add_argument("--n-sims", type=int, default=300)
    p_pow.add_argument("--n-perm", type=int, default=2000)
    p_pow.add_argument("--repeats", type=int, nargs="+", default=[3])
    p_pool = sub.add_parser("pool", help="pool identities across runs per model; Holm within model")
    p_pool.add_argument("run_dirs", nargs="+")
    p_pool.add_argument("--out", required=True, help="directory for pooled.json and pooled.md")
    p_pool.add_argument("--primary-arms", nargs="*", default=None,
                        help="arms for the 2-of-3 generality verdict (default: the three primary arms, if present)")
    p_pool.add_argument("--backend", action="append")
    p_pool.add_argument("--allow-missing", action="store_true",
                        help="pool even if a model has no log in some run (flagged in the output)")
    p_pool.add_argument("--n-perm", type=int, default=10000)
    p_pool.add_argument("--n-boot", type=int, default=10000)
    p_probe = sub.add_parser("probe", help="one live filter request per backend: settings, served host, cost")
    p_probe.add_argument("config")
    p_probe.add_argument("--backend", action="append")
    p_cost = sub.add_parser("cost", help="actual spend recorded in run logs")
    p_cost.add_argument("run_dirs", nargs="+")
    p_cal = sub.add_parser("calibrate", help="false-positive rate of the full pipeline under the null")
    p_cal.add_argument("config")
    p_cal.add_argument("--seeds", type=int, default=20)
    p_cal.add_argument("--n-perm", type=int, default=1000)
    p_cal.add_argument("--sim", default="{}", help='simulator settings as JSON, e.g. {"call_failure": 0.1}')
    p_cal.add_argument("--first-seed", type=int, default=1)
    args = parser.parse_args(argv)

    if args.cmd == "plan":
        config = load_config(args.config)
        print(plan_summary(config, selected_backends(config, args.backend)))
        return 0

    if args.cmd == "run":
        from .runner import run_experiment

        config = load_config(args.config)
        specs = selected_backends(config, args.backend)
        text, estimates = plan_details(config, specs)
        print(text)
        if any(s["kind"] in PAID_KINDS for s in specs) and not args.yes:
            print("\nThese backends call paid APIs. Re-run with --yes to proceed.")
            return 2
        caps = None
        if args.max_cost_factor:
            caps = {s["name"]: max(0.10, args.max_cost_factor * estimates.get(s["name"], 0.0)) for s in specs}
            print("Spend caps for this run: " + ", ".join(f"{k} ${v:.2f}" for k, v in caps.items()))
        root = Path(config["run"].get("output_dir", "runs"))
        run_dir = run_experiment(config, args.run_id, [make_backend(s) for s in specs], root, max_cost=caps)
        print(f"Run written to {run_dir}")
        return 0

    if args.cmd == "analyze":
        from .analyze import analyze_run

        compare_run = Path(args.compare_run) if args.compare_run else None
        analyze_run(Path(args.run_dir), n_perm=args.n_perm, n_boot=args.n_boot, compare_run=compare_run)
        print((Path(args.run_dir) / "results.md").read_text(encoding="utf-8"))
        return 0

    if args.cmd == "power":
        from .power import power_grid, render

        for k in args.repeats:
            print(render(power_grid(k=k, n_sims=args.n_sims, n_perm=args.n_perm)), end="\n\n")
        return 0

    if args.cmd == "pool":
        import json

        from .pool import generality, pool_runs, primary_arms_for, render, render_parsing

        results = pool_runs([Path(d) for d in args.run_dirs], n_perm=args.n_perm, n_boot=args.n_boot,
                            backends=args.backend, allow_missing=args.allow_missing)
        arms = primary_arms_for(results, args.primary_arms)
        verdict = generality(results, arms) if arms else None
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "pooled.json").write_text(json.dumps({**results, "primary_arms": arms, "generality": verdict},
                                                    indent=2), encoding="utf-8")
        text = render(results, verdict)
        (out / "pooled.md").write_text(text, encoding="utf-8")
        (out / "parsing.md").write_text(render_parsing(results), encoding="utf-8")
        print(text)
        return 0

    if args.cmd == "probe":
        from .core import FatalBackendError, Task
        from .prompts import filter_prompt, parse_bool, serialize

        config = load_config(args.config)
        ds = load_data(config["data"])
        rid = sorted(ds.records)[0]
        system, user = filter_prompt(serialize(ds.records[rid], ds.filter_fields), ds.p.text)
        task = Task(task_id="probe", identity="probe", condition="p", unit=rid, repeat=0, op="filter",
                    items=(rid,), system=system, user=user, payload={})
        total = 0.0
        print("| Backend | Answer | Parsed | Served model | Served by | Tokens in / out / reasoning | Cost (USD) | Seconds | Error |")
        print("|---|---|---|---|---|---|---|---|---|")
        for spec in selected_backends(config, args.backend):
            try:
                comp = make_backend(spec).complete(task)
            except FatalBackendError as e:
                print(f"| {spec['name']} | | | | | | | | FATAL: {e} |")
                continue
            u = comp.usage
            total += float(u.get("cost") or 0)
            answer = (comp.text or "").strip().replace("\n", " ")[:40]
            print(f"| {spec['name']} | {answer} | {parse_bool(comp.text)} | {comp.served_model} | "
                  f"{comp.extra.get('served_provider')} | {u.get('input_tokens')} / {u.get('output_tokens')} / "
                  f"{u.get('reasoning_tokens')} | {u.get('cost')} | {comp.latency_s:.2f} | {comp.error or ''} |")
        print(f"\nProbe cost: ${total:.6f}")
        return 0

    if args.cmd == "cost":
        from .runner import log_files, log_name, read_log

        total = 0.0
        print("| Run | Backend | Calls | Failed | Input tokens | Output tokens | Reasoning tokens | Cost (USD) |")
        print("|---|---|---|---|---|---|---|---|")
        for run_dir in map(Path, args.run_dirs):
            for log_path in log_files(run_dir):
                recs = read_log(log_path)
                sums = Counter()
                for r in recs:
                    for k, v in (r.get("usage") or {}).items():
                        if isinstance(v, (int, float)):
                            sums[k] += v
                total += sums["cost"]
                print(f"| {run_dir.name} | {log_name(log_path)} | {len(recs)} | {sum(bool(r.get('error')) for r in recs)} | "
                      f"{sums['input_tokens']:,} | {sums['output_tokens']:,} | {sums['reasoning_tokens']:,} | "
                      f"{sums['cost']:.4f} |")
        print(f"\nTotal: ${total:.4f}")
        return 0

    if args.cmd == "calibrate":
        from .power import pipeline_calibration

        import json as _json

        sim = _json.loads(args.sim)
        rates = pipeline_calibration(load_config(args.config), range(args.first_seed, args.first_seed + args.seeds),
                                     n_perm=args.n_perm, sim_params=sim)
        print(f"Simulator settings: {sim or 'null (no effects)'}; seeds: {args.seeds}\n")
        print("p is the reported p-value (max of permutation and cluster-robust for H4, S2 and S4).\n")
        print("| Comparison | Runs | False-positive rate of p at 0.05 (± SE) | at 0.05/7 | Permutation p alone at 0.05 "
              "| CI lower bound > 0 | Mean Δ |\n|---|---|---|---|---|---|---|")
        for name, r in rates.items():
            se = (r["false_positive_rate"] * (1 - r["false_positive_rate"]) / max(1, r["runs"])) ** 0.5
            ci = "" if r["ci_lower_above_zero"] is None else f"{r['ci_lower_above_zero']:.3f}"
            print(f"| {name} | {r['runs']} | {r['false_positive_rate']:.3f} ± {se:.3f} | "
                  f"{r['false_positive_rate_strict']:.3f} | {r['false_positive_rate_permutation_only']:.3f} | {ci} | "
                  f"{r['mean_delta']:+.5f} |")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
