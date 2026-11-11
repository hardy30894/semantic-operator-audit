"""Command line for the extension: `python3 -m extension <command>` from harness/.

    plan        quote calls, tokens and cost for a config (no API calls)
    run         run a config with mitigation segments (refuses paid backends without --yes)
    analyze     effect of each mitigation on excess divergence and cost, for one extension run
    meta        random-effects meta-analysis of excess divergence across predicate pairs
    accuracy    plan accuracy against gold labels, with paired tests
    preview     offline previews of the mitigations from existing logs
    calibrate   measure tokens and cost per call from existing logs, for `plan --calibration`
    provenance  the offline evaluation of three answer caches (extension/provenance_eval.py)
    placebo     Addendum G for one run: rewording placebos, the three-valued prompt, arm pairs (extension/placebo.py)
    placebo-pool  Addendum G pooled over runs, per arm, with its Holm families

Configs without an [ext] table run unchanged through `python3 -m extension.cli run` (semaudit's
command line with priced Anthropic backends); this command line adds the mitigation segments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from semaudit.__main__ import load_config

from .backends import is_paid, make_backend_ext, settings_warnings

HARNESS = Path(__file__).resolve().parents[1]
ANALYSIS_OUT = HARNESS.parent / "extension" / "analysis"
DEFAULT_CALIBRATION = HARNESS / "configs" / "ext-calibration.json"


def _select(config: dict, names: list[str] | None, include_disabled: bool = False) -> list[dict]:
    specs = [b for b in config.get("backends", []) if include_disabled or b.get("enabled", True)]
    if names:
        specs = [b for b in specs if b["name"] in names]
        missing = set(names) - {b["name"] for b in specs}
        if missing:
            raise SystemExit(f"unknown or disabled backends: {sorted(missing)}")
    if not specs:
        raise SystemExit("no enabled backends selected")
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m extension")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="calls, tokens and cost for a config; makes no API calls")
    p.add_argument("config")
    p.add_argument("--backend", action="append")
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION),
                   help="token and cost measurements from earlier logs ('' to use plan_details's sizes)")
    p.add_argument("--include-disabled", action="store_true", help="also quote roster entries with enabled = false")
    p.add_argument("--json", action="store_true")
    r = sub.add_parser("run", help="run a config with its mitigation segments")
    r.add_argument("config")
    r.add_argument("--run-id", required=True)
    r.add_argument("--backend", action="append")
    r.add_argument("--yes", action="store_true", help="confirm calls to paid APIs")
    r.add_argument("--max-cost-factor", type=float,
                   help="stop a backend once its logged spend exceeds this multiple of its quote (at least $0.10)")
    r.add_argument("--max-cost", type=float, help="stop each backend once its logged spend exceeds this many USD")
    r.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    a = sub.add_parser("analyze", help="mitigation effects for one extension run")
    a.add_argument("run_dir")
    a.add_argument("--backend", action="append")
    a.add_argument("--n-perm", type=int, default=5000)
    a.add_argument("--n-boot", type=int, default=2000)
    m = sub.add_parser("meta", help="random-effects meta-analysis across predicate pairs")
    m.add_argument("run_dirs", nargs="+")
    m.add_argument("--backend", action="append")
    m.add_argument("--out", default=str(ANALYSIS_OUT))
    m.add_argument("--name", default="meta")
    m.add_argument("--title", default="Random-effects meta-analysis of excess divergence across predicate pairs")
    c = sub.add_parser("accuracy", help="plan accuracy against gold labels")
    c.add_argument("run_dirs", nargs="+")
    c.add_argument("--backend", action="append")
    c.add_argument("--out", default=str(ANALYSIS_OUT))
    c.add_argument("--n-boot", type=int, default=2000)
    v = sub.add_parser("preview", help="offline previews of the mitigations from existing logs")
    v.add_argument("run_dirs", nargs="+")
    v.add_argument("--backend", action="append")
    v.add_argument("--out", default=str(ANALYSIS_OUT))
    v.add_argument("--n-perm", type=int, default=2000)
    v.add_argument("--votes", type=int, nargs="+", default=[1, 3, 5], help="votes per voted answer (at most k)")
    v.add_argument("--voting-only", action="store_true",
                   help="print only the voting table pooled over the runs, per arm and identity")
    k = sub.add_parser("calibrate", help="tokens and cost per call, measured from existing logs")
    k.add_argument("run_dirs", nargs="+")
    k.add_argument("--out", default=str(DEFAULT_CALIBRATION))
    sub.add_parser("provenance", help="offline evaluation of three answer caches", add_help=False)
    g = sub.add_parser("placebo", help="Addendum G for one run (writes placebo.json and placebo.md into the run)")
    g.add_argument("run_dir")
    g.add_argument("--backend", action="append")
    g.add_argument("--n-perm", type=int, default=5000)
    g.add_argument("--n-boot", type=int, default=2000)
    gp = sub.add_parser("placebo-pool", help="Addendum G pooled over runs, per arm")
    gp.add_argument("run_dirs", nargs="+")
    gp.add_argument("--out", required=True, help="directory for placebo.json and placebo.md")
    gp.add_argument("--backend", action="append")
    gp.add_argument("--allow-missing", action="store_true",
                    help="pool even if an arm has no log in some run (flagged in the output)")
    gp.add_argument("--n-perm", type=int, default=10000)
    gp.add_argument("--n-perm-binary", type=int, default=100000)
    gp.add_argument("--n-boot", type=int, default=10000)
    args, rest = parser.parse_known_args(argv)

    if args.cmd == "provenance":
        from .provenance_eval import main as provenance_main

        return provenance_main(rest)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")

    if args.cmd in ("plan", "run"):
        from .planner import base_crosscheck, load_calibration, quote, render

        config = load_config(args.config)
        specs = _select(config, args.backend, getattr(args, "include_disabled", False))
        cal_path = Path(args.calibration) if args.calibration else None
        if cal_path is not None and not cal_path.is_file():
            # A missing file would quote, and cap, without calibration (Addendum G review, M1).
            print(f"Refused: the calibration file {cal_path} does not exist.", file=sys.stderr)
            return 2
        calibration = load_calibration(cal_path)
        q = quote(config, specs, calibration)
        if getattr(args, "json", False):
            print(json.dumps(q, indent=2))
            return 0
        if cal_path is not None:
            import hashlib

            print(f"Calibration: {cal_path} (SHA-256 {hashlib.sha256(cal_path.read_bytes()).hexdigest()})")
        print(render(config, specs, q))
        if args.cmd == "plan":
            if not config.get("ext"):
                print("\nsemaudit's own plan (plan_details) for this config:")
                print(base_crosscheck(config, specs)[0])
            return 0
        paid = [s["name"] for s in specs if is_paid(s)]
        if paid and not args.yes:
            print("\nThese backends call paid APIs. Re-run with --yes to proceed.")
            return 2
        if paid and not (args.max_cost_factor or args.max_cost):
            print(f"\nRefused: paid backends {paid} need a spend cap. Pass --max-cost-factor (a multiple of the "
                  "quote) or --max-cost (USD per backend).")
            return 2
        for spec in specs:
            for note in settings_warnings(spec):
                print(f"note: {note}")
        caps = None
        if args.max_cost_factor or args.max_cost:
            caps = {}
            for s_ in specs:
                limits = []
                if args.max_cost_factor:
                    limits.append(max(0.10, args.max_cost_factor * (q["backends"][s_["name"]].get("usd") or 0.0)))
                if args.max_cost:
                    limits.append(args.max_cost)
                caps[s_["name"]] = min(limits)
            print("Spend caps for this run: " + ", ".join(f"{n} ${c:.2f}" for n, c in caps.items()))
        from .runner import run_ext

        root = Path(config["run"].get("output_dir", "runs"))
        run_dir = run_ext(config, args.run_id, [make_backend_ext(s) for s in specs], root, max_cost=caps)
        print(f"Run written to {run_dir}")
        return 0

    if args.cmd == "analyze":
        from .analysis import analyze_run_ext, render, write

        run_dir = Path(args.run_dir)
        results = analyze_run_ext(run_dir, n_perm=args.n_perm, n_boot=args.n_boot, backends=args.backend)
        write(results, run_dir)
        print(render(results))
        return 0

    if args.cmd == "meta":
        from .meta import meta_report, render, write

        report = meta_report([Path(d) for d in args.run_dirs], args.backend)
        write(report, Path(args.out), args.name, args.title)
        print(render(report, args.title))
        return 0

    if args.cmd == "accuracy":
        from .accuracy import accuracy_report, render, write

        report = accuracy_report([Path(d) for d in args.run_dirs], args.backend, n_boot=args.n_boot)
        write(report, Path(args.out))
        print(render(report))
        return 0

    if args.cmd == "preview":
        from .preview import preview_report, render, render_pooled_voting, write

        report = preview_report([Path(d) for d in args.run_dirs], args.backend, votes=tuple(args.votes),
                                n_perm=args.n_perm)
        if args.voting_only:
            print(render_pooled_voting(report))
            return 0
        write(report, Path(args.out))
        print(render(report))
        return 0

    if args.cmd == "placebo":
        from .placebo import analyze_placebo_run, render, write as write_placebo

        run_dir = Path(args.run_dir)
        results = analyze_placebo_run(run_dir, n_perm=args.n_perm, n_boot=args.n_boot, backends=args.backend)
        write_placebo(results, run_dir)
        print(render(results))
        return 0

    if args.cmd == "placebo-pool":
        from .placebo import pool_placebo, render_pooled, write as write_placebo

        results = pool_placebo([Path(d) for d in args.run_dirs], n_perm=args.n_perm, n_boot=args.n_boot,
                               backends=args.backend, n_perm_binary=args.n_perm_binary,
                               allow_missing=args.allow_missing)
        write_placebo(results, Path(args.out), pooled=True)
        print(render_pooled(results))
        return 0

    if args.cmd == "calibrate":
        from .planner import calibrate

        result = calibrate([Path(d) for d in args.run_dirs])
        out = Path(args.out)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out} ({len(result['backends'])} backends)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
