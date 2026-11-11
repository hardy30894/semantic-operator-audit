"""Call and cost quotes for extension configs, made before any paid run (no API calls).

The quote applies `semaudit.__main__.plan_details`'s logic to the exact task list that
`extension.runner.run_ext` will issue (`extension.mitigations.build_ext`, every segment, plus one
map(map(x)) follow-up per stage-0 draw):

- calls: exact;
- input tokens: prompt characters / 4, as plan_details, or / the backend's measured characters per
  token when a calibration file has that backend;
- visible output tokens: plan_details's per-operator sizes for semaudit's styles, or the backend's
  measured mean per operator and style from a calibration file; for the explain style, the
  assumed sizes in EXPLAIN_OUTPUT_TOKENS until a pilot measures them; for the format style, the
  backend's measured lotus size plus FORMAT_LABEL_TOKENS (plan_details's size plus the label for an
  uncalibrated backend) until a pilot measures it; for the `unknown` style (Addendum G; one word, like
  lotus), the backend's measured lotus size, or plan_details's for an uncalibrated backend;
- thinking tokens: included in the measured output tokens when the backend has a calibration entry
  (Anthropic and OpenRouter count thinking in output_tokens), so the roster's
  est_thinking_tokens_per_call is used only for an uncalibrated backend;
- cost: list prices per million tokens (for Claude through Anthropic directly this is also how the
  runner computes spend). Without a calibration, plan_details's measured pilot rate per call is used
  where the roster has one and the calls are in a style and setting the rate was measured for; with
  one, calibrated tokens are used and the pilot rate and the billed cost per call measured in the
  logs are printed as cross-checks. An arm with `pairs_with` and no calibration entry of its own (a
  reasoning arm before its pilot) borrows its partner's prompt-token ratio, since it is the same
  model; its output sizes are not borrowed.

With no calibration and only the base segment, the numbers equal plan_details's exactly (tested).
`calibrate` builds the calibration file from existing logs.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from semaudit.__main__ import _OUTPUT_TOKENS, plan_details
from semaudit.data import load_data
from semaudit.runner import log_files, log_name

from .backends import settings_warnings
from .logs import iter_log
from .mitigations import build_ext, plan_segments
from .styles import EXPLAIN, FORMAT, UNKNOWN

# Assumed visible output per call in the explain style (reasoning plus the answer line), in tokens.
# Replace with measured means after a small pilot (`calibrate` measures them from its logs).
EXPLAIN_OUTPUT_TOKENS = {"filter": 70, "match": 90, "count": 140, "map": 80}
EXPLAIN_BATCH_TOKENS_PER_ROW = 20
# The format style's answers are the lotus answers plus their "Answer:" label (an assumption until measured).
FORMAT_LABEL_TOKENS = 3


def _assumed_tokens(op: str, n_items: int, style: str) -> tuple[float, str]:
    """Visible output tokens per call when no measurement exists, and where the size comes from."""
    if style == EXPLAIN:
        if op == "filter_batch":
            return 40 + EXPLAIN_BATCH_TOKENS_PER_ROW * n_items, "explain assumption"
        return EXPLAIN_OUTPUT_TOKENS[op], "explain assumption"
    base = _OUTPUT_TOKENS.get(op, 6 * n_items)
    if style == FORMAT:  # the same answer plus the "Answer:" / "Answers:" label
        return base + FORMAT_LABEL_TOKENS, "format assumption"
    return base, "plan_details sizes"


def quote(config: dict[str, Any], specs: list[dict[str, Any]],
          calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    """Calls, tokens and USD per backend for an extension config, with the source of each number."""
    ds = load_data(config["data"])
    tasks, design = build_ext(ds, config)
    stage0 = [t for t in tasks if t.condition == "x_in"]
    calls = len(tasks) + len(stage0)
    per_segment: Counter[str] = Counter()
    per_identity: Counter[str] = Counter()
    for t in tasks + stage0:  # each stage-0 draw has one follow-up in the same identity
        per_segment[t.context.get("segment", "base")] += 1
        per_identity[t.identity] += 1
    # Each follow-up map(map(x)) is counted as a copy of its stage-0 request, as plan_details does.
    chars_by_style: Counter[str] = Counter()
    ops_by_style: Counter[tuple[str, str, int]] = Counter()
    for t in tasks + stage0:
        style = t.context.get("style", "lotus")
        chars_by_style[style] += len(t.system) + len(t.user)
        ops_by_style[(style, t.op, len(t.items))] += 1
    table = (calibration or {}).get("backends", {})
    backends = {}
    for spec in specs:
        own = table.get(spec["name"])
        partner = spec.get("pairs_with") if own is None and spec.get("pairs_with") in table else None
        cpt = (own or table.get(partner) or {}).get("chars_per_input_token")
        input_tokens = sum(chars / (cpt or 4.0) for chars in chars_by_style.values())
        input_source = (f"calibrated on {spec['name']}" if own and cpt else
                        f"calibrated on {partner} (same model)" if partner and cpt else "characters / 4")
        output_tokens, sources = 0.0, Counter()
        measured_out = (own or {}).get("output_tokens", {})
        for (style, op, items), n in ops_by_style.items():
            key = f"{style}/{op}"
            if key in measured_out:
                output_tokens += n * measured_out[key]
                sources[f"measured on {spec['name']}"] += n
            elif style == FORMAT and f"lotus/{op}" in measured_out:
                # A format-style reply is the lotus reply plus its "Answer:" label (an assumption).
                output_tokens += n * (measured_out[f"lotus/{op}"] + FORMAT_LABEL_TOKENS)
                sources[f"format assumption (lotus size measured on {spec['name']}, plus label)"] += n
            elif style == UNKNOWN and f"lotus/{op}" in measured_out:
                # A one-word reply, as in the lotus style (an assumption until a pilot measures it).
                output_tokens += n * measured_out[f"lotus/{op}"]
                sources[f"unknown-style assumption (lotus size measured on {spec['name']})"] += n
            else:
                size, source = _assumed_tokens(op, items, style)
                output_tokens += n * size
                sources[source] += n
        if own:
            # Measured output tokens already include thinking (Anthropic and OpenRouter both bill it as
            # output and count it in output_tokens), so the roster's assumption is not added.
            thinking, thinking_source = 0.0, "included in measured output"
        else:
            thinking = spec.get("est_thinking_tokens_per_call", 0) * calls
            thinking_source = (f"roster assumption, {spec['est_thinking_tokens_per_call']} per call"
                               if thinking else "none")
        row: dict[str, Any] = {"calls": calls, "input_tokens": input_tokens, "output_tokens": output_tokens,
                               "thinking_tokens": thinking, "calibrated": bool(own),
                               "sources": {"input": input_source,
                                           "output": ", ".join(f"{src} ({n:,} calls)" for src, n in sources.items()),
                                           "thinking": thinking_source},
                               "warnings": settings_warnings(spec)}
        if "price_in_per_mtok" in spec:
            row["usd_list_price"] = (input_tokens * spec["price_in_per_mtok"]
                                     + (output_tokens + thinking) * spec.get("price_out_per_mtok", 0)) / 1e6
            row["usd"] = row["usd_list_price"]
            row["method"] = "list price x tokens"
        per_call = (own or {}).get("usd_per_call", {})
        needed = {f"{style}/{op}": n for (style, op, _), n in ops_by_style.items()}
        if per_call and all(key in per_call for key in needed):
            # Cross-check: the billed cost per call measured for each style and operator.
            row["usd_billed_per_call"] = sum(n * per_call[key] for key, n in needed.items())
        rate = spec.get("measured_usd_per_1k_calls")
        new_style = sum(n for (style, _, _), n in ops_by_style.items() if style in (EXPLAIN, FORMAT))
        if rate is not None and not new_style and not spec.get("est_thinking_tokens_per_call"):
            row["usd_measured_rate"] = calls * rate / 1000
            if not own:  # plan_details's rule; calibrated tokens are more exact when available
                row["usd"] = row["usd_measured_rate"]
                row["method"] = "measured pilot rate per call"
        backends[spec["name"]] = row
    return {"dataset": ds.name, "records": len(ds.records), "calls": calls,
            "segments": [s["name"] for s in plan_segments(config)], "per_segment": dict(per_segment),
            "per_identity": dict(sorted(per_identity.items())), "backends": backends}


def render(config: dict[str, Any], specs: list[dict[str, Any]], q: dict[str, Any]) -> str:
    lines = [f"Dataset {q['dataset']}: {q['records']} records. Segments: {', '.join(q['segments'])}.",
             f"Calls per backend: {q['calls']:,} ("
             + ", ".join(f"{k} {v:,}" for k, v in q["per_identity"].items()) + ")."]
    total = 0.0
    for spec in specs:
        row = q["backends"][spec["name"]]
        usd = row.get("usd")
        total += usd or 0.0
        line = (f"  {spec['name']} ({spec['kind']}{', ' + spec['model'] if 'model' in spec else ''}): "
                f"{row['input_tokens']:,.0f} input, {row['output_tokens']:,.0f} output"
                + (f", {row['thinking_tokens']:,.0f} assumed thinking" if row["thinking_tokens"] else "")
                + " tokens")
        if usd is not None:
            line += f"; about ${usd:,.2f} ({row['method']})"
        checks = []
        if row.get("usd_billed_per_call") is not None:
            checks.append(f"${row['usd_billed_per_call']:,.2f} at the billed cost per call measured in the logs")
        if row.get("usd_measured_rate") is not None and row.get("method") != "measured pilot rate per call":
            checks.append(f"${row['usd_measured_rate']:,.2f} at the pilot rate per call")
        if checks:
            line += "; cross-checks: " + ", ".join(checks)
        lines.append(line)
        src = row["sources"]
        lines.append(f"    sources: input tokens {src['input']}; output tokens {src['output']}; "
                     f"thinking {src['thinking']}")
        lines += [f"    note: {w}" for w in row["warnings"]]
    lines.append(f"Total: about ${total:,.2f}.")
    return "\n".join(lines)


def base_crosscheck(config: dict[str, Any], specs: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    """semaudit's own plan for the base identities of the config (plan_details, unchanged)."""
    return plan_details(config, specs)


# --- calibration from existing logs -----------------------------------------------------------

def calibrate(run_dirs: list[Path]) -> dict[str, Any]:
    """Per backend: prompt characters per input token, and mean visible output, reasoning tokens and
    billed cost per call by style and operator, from every successful call in the logs."""
    sums: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for run_dir in run_dirs:
        for path in log_files(run_dir):
            backend = log_name(path)
            for rec in iter_log(path):
                usage = rec.get("usage") or {}
                if rec.get("error") or usage.get("input_tokens") is None:
                    continue
                style = (rec.get("context") or {}).get("style", "lotus")
                key = f"{style}/{rec['op']}"
                s = sums[backend]
                s["all"]["chars"] += len(rec.get("system") or "") + len(rec.get("user") or "")
                s["all"]["input_tokens"] += usage.get("input_tokens") or 0
                s[key]["calls"] += 1
                s[key]["output_tokens"] += usage.get("output_tokens") or 0
                s[key]["reasoning_tokens"] += usage.get("reasoning_tokens") or 0
                s[key]["cost"] += float(usage.get("cost") or 0.0)
    out: dict[str, Any] = {"runs": [str(d) for d in run_dirs], "backends": {}}
    for backend, s in sorted(sums.items()):
        entry: dict[str, Any] = {
            "chars_per_input_token": s["all"]["chars"] / s["all"]["input_tokens"] if s["all"]["input_tokens"] else None,
            "output_tokens": {}, "reasoning_tokens": {}, "usd_per_call": {}, "calls": {}}
        for key, c in s.items():
            if key == "all" or not c["calls"]:
                continue
            entry["calls"][key] = c["calls"]
            entry["output_tokens"][key] = c["output_tokens"] / c["calls"]
            entry["reasoning_tokens"][key] = c["reasoning_tokens"] / c["calls"]
            entry["usd_per_call"][key] = c["cost"] / c["calls"]
        out["backends"][backend] = entry
    return out


def load_calibration(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
