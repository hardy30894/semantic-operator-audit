"""Verify the reasoning runs (grid/reason.py): every planned call present and successful with the exact prompt,
from the requested model on the pinned host, with its raw response, and how often the model actually reasoned.

A call reasoned when it reports reasoning tokens (OpenRouter, Luna), thinking tokens in the raw Anthropic usage
(output_tokens_details.thinking_tokens), or a thinking block in the raw Anthropic content.
Run from the project root:  python3 grid/verify_reason.py [--model NAME]     writes grid/VERIFY-reason.md
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grid"))
import reason  # noqa: E402

RUN_MODELS = ["sonnet-5.5-reasoning-medium", "haiku-5.5-reasoning-medium", "gpt-5.6-luna-reasoning-medium"]


def _h(system, user) -> str:
    return hashlib.sha1(f"{system}\x1f{user}".encode("utf-8")).hexdigest()


def thinking(rec: dict) -> int:
    raw = (rec.get("extra") or {}).get("raw_response") or {}
    t = ((raw.get("usage") or {}).get("output_tokens_details") or {}).get("thinking_tokens") or 0 \
        if isinstance(raw, dict) else 0
    block = isinstance(raw, dict) and any(isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
                                          for b in raw.get("content") or [])
    return max(int(t), int((rec.get("usage") or {}).get("reasoning_tokens") or 0), int(block))


def read(path: Path) -> list[dict]:
    out = []
    if path.exists():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    out.append(json.loads(line))
        except EOFError:
            pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=RUN_MODELS)
    args = ap.parse_args(argv)
    lines = ["# Verification of the reasoning runs", "",
             "| Model | Run | Planned | Done | Missing | Failed calls | Wrong prompt | Reasoned | Mean reasoning tokens | "
             "Other model | Other host | No raw | Spent (US$) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    totals = Counter()
    per_model = {}
    for model in RUN_MODELS:
        if args.model and model != args.model:
            continue
        backend_desc = reason.SPECS[model]
        pinned = (backend_desc["params"].get("upstream") or "").lower() or None
        mt = Counter()
        for r in reason.runs_of():
            _, tasks, _, _, _ = reason.tasks_for(r["kind"], r["where"])
            recs = read(reason.H / "runs" / r["run_id"] / f"{model}.jsonl.gz")
            ok, c = {}, Counter()
            think_sum = 0
            for rec in recs:
                c["spent"] += float((rec.get("usage") or {}).get("cost") or 0)
                if rec.get("error"):
                    c["failed"] += 1
                    continue
                ok[rec["task_id"]] = _h(rec.get("system"), rec.get("user"))
                th = thinking(rec)
                c["calls"] += 1
                c["reasoned"] += th > 0
                think_sum += (rec.get("usage") or {}).get("reasoning_tokens") or \
                    (((rec.get("extra") or {}).get("raw_response") or {}).get("usage") or {}).get(
                        "output_tokens_details", {}).get("thinking_tokens") or 0
                c["model"] += rec.get("served_model") not in (None, rec.get("model_requested"))
                served = (rec.get("extra") or {}).get("served_provider")
                c["host"] += bool(pinned and served and pinned not in str(served).lower())
                c["raw"] += not (rec.get("extra") or {}).get("raw_response")
            done = sum(t.task_id in ok for t in tasks)
            wrong = sum(t.task_id in ok and ok[t.task_id] != _h(t.system, t.user) for t in tasks)
            missing = len(tasks) - done
            fx = sum(1 for k in ok if k.startswith("idempotence:fx:"))
            note = f" (+{fx} follow-ups)" if r["kind"] == "idem" else ""
            for k, v in (("planned", len(tasks)), ("done", done), ("missing", missing), ("wrong", wrong)):
                totals[k] += v
                mt[k] += v
            for k in ("failed", "model", "host", "raw", "calls", "reasoned", "spent"):
                totals[k] += c[k]
                mt[k] += c[k]
            mean = think_sum / c["calls"] if c["calls"] else 0
            lines.append(f"| {model} | {r['run_id']} | {len(tasks):,} | {done:,}{note} | {missing:,} | {c['failed']} | "
                         f"{wrong} | {c['reasoned']:,} of {c['calls']:,} | {mean:.0f} | {c['model']} | {c['host']} | "
                         f"{c['raw']} | {c['spent']:.2f} |")
        per_model[model] = mt
    lines += ["", "| Model | Planned | Done | Missing | Wrong prompt | Calls that reasoned | Spent (US$) |",
              "|---|---|---|---|---|---|---|"]
    for model, mt in per_model.items():
        share = 100 * mt["reasoned"] / mt["calls"] if mt["calls"] else 0
        lines.append(f"| {model} | {mt['planned']:,} | {mt['done']:,} | {mt['missing']:,} | {mt['wrong']} | "
                     f"{mt['reasoned']:,} of {mt['calls']:,} ({share:.0f}%) | {mt['spent']:.2f} |")
    lines += ["", f"Total: {totals['done']:,} of {totals['planned']:,} planned calls done, {totals['missing']:,} missing, "
              f"{totals['wrong']} with a wrong prompt, {totals['failed']} failed calls logged (re-sent on resume), "
              f"{totals['model']} from another model, {totals['host']} from another host, {totals['raw']} without a "
              f"raw response; US${totals['spent']:.2f} spent."]
    (ROOT / "grid" / "VERIFY-reason.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-(len(per_model) + 4):]))
    return 0 if not (totals["missing"] or totals["wrong"] or totals["model"] or totals["host"] or totals["raw"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
