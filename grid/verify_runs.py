"""Check the grid's new runs call by call: every planned task has a successful call with the exact prompt the
design specifies, and every call was made in the standard state (no reasoning, the requested model, the pinned
host, the raw response stored). Run on the box or after the copy-down, from the project root:
  python3 grid/verify_runs.py [--model NAME]      writes grid/VERIFY.md
Makes no model calls. A log still being written is read up to its last complete call.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "grid"))
import run  # noqa: E402
from semaudit.identities import other_host  # noqa: E402


def _h(system, user) -> str:
    return hashlib.sha1(f"{system}\x1f{user}".encode("utf-8")).hexdigest()


def read(path: Path) -> list[dict]:
    recs = []
    if not path.exists():
        return recs
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except EOFError:
        pass
    return recs


def tasks_of(r: dict) -> list:
    if r["kind"] == "lr":
        from semaudit.data import load_data
        from extension.mitigations import build_ext

        cwd = os.getcwd()
        os.chdir(run.H)
        try:
            cfg = run.lr_config(r["where"])
            tasks, _ = build_ext(load_data(cfg["data"]), cfg)
        finally:
            os.chdir(cwd)
        return tasks
    return run.tasks_for(r["kind"], r["where"])[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--root", type=Path, default=run.H / "runs")
    ap.add_argument("--coverage", type=Path, help="the plan to verify (default grid/coverage.json)")
    ap.add_argument("--out", type=Path, default=ROOT / "grid" / "VERIFY.md")
    args = ap.parse_args(argv)
    plan = run.plan(coverage=args.coverage)
    lines = ["# Verification of the grid's new runs", "",
             "| Model | Run | Planned | Done | Missing | Failed calls | Wrong prompt | Reasoning | Other model | Other host | No raw |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    totals = Counter()
    for model in run.MODELS:
        if args.model and model != args.model:
            continue
        for r in plan[model]:
            tasks = tasks_of(r)
            recs = read(args.root / r["run_id"] / f"{model}.jsonl.gz")
            ok = {}
            c = Counter()
            for rec in recs:
                if rec.get("error"):
                    c["failed"] += 1
                    continue
                ok[rec["task_id"]] = _h(rec.get("system"), rec.get("user"))
                c["reasoning"] += int((rec.get("usage") or {}).get("reasoning_tokens") or 0) > 0
                c["model"] += rec.get("served_model") not in (None, rec.get("model_requested"))
                c["host"] += other_host(rec)
                c["raw"] += not (rec.get("extra") or {}).get("raw_response")
            planned = [t for t in tasks]
            done = sum(t.task_id in ok for t in planned)
            wrong = sum(t.task_id in ok and ok[t.task_id] != _h(t.system, t.user) for t in planned)
            fx = sum(1 for k in ok if k.startswith("idempotence:fx:"))
            note = f" (+{fx} follow-ups)" if r["kind"] == "idem" else ""
            missing = len(planned) - done
            for k, v in (("planned", len(planned)), ("done", done), ("missing", missing), ("wrong", wrong)):
                totals[k] += v
            for k in ("failed", "reasoning", "model", "host", "raw"):
                totals[k] += c[k]
            lines.append(f"| {model} | {r['run_id']} | {len(planned):,} | {done:,}{note} | {missing:,} | {c['failed']} | "
                         f"{wrong} | {c['reasoning']} | {c['model']} | {c['host']} | {c['raw']} |")
    lines += ["", f"Total: {totals['done']:,} of {totals['planned']:,} planned calls done, {totals['missing']:,} missing, "
              f"{totals['wrong']} with a wrong prompt, {totals['failed']} failed calls logged (retried on resume), "
              f"{totals['reasoning']} with reasoning, {totals['model']} from another model, {totals['host']} from "
              f"another host, {totals['raw']} without a raw response."]
    out = args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[-1])
    return 0 if not (totals["missing"] or totals["wrong"] or totals["reasoning"] or totals["model"]
                     or totals["host"] or totals["raw"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
