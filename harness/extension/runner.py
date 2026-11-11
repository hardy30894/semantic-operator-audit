"""Running extension configs: semaudit's call log and call records, with extension segments and
interleaved arms.

`run_ext` follows `semaudit.runner.run_experiment` step for step (design check and resume, stage 0
before the main phase, seeded shuffles, one manifest per backend) and logs every call with
semaudit's `CallLog` and `call_record`, so the JSONL format is unchanged. The differences:

- tasks come from `extension.mitigations.build_ext` (all segments, shuffled together);
- map(map(x)) follow-ups come from `extension.mitigations.ext_followups`;
- replies in the extension styles (explain, format) are logged with the answer-line parser's
  outputs and its parser version;
- **several backends run interleaved**: one executor (`execute_arms`) shuffles every arm's tasks
  into a single seeded order on one thread pool, and each backend's calls go to its own log. Arms
  that are compared item by item (a reasoning arm and its `pairs_with` reference) are therefore
  measured at the same time. One backend runs through semaudit's own `execute`;
- spend already logged is summed with `extension.backends.record_cost` (token-priced for kinds
  that report no cost), and caps apply to the cost written into each completion.
"""

from __future__ import annotations

import contextlib
import json
import platform
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from semaudit import __version__
from semaudit.core import Task, stable_seed
from semaudit.data import load_data
from semaudit.identities import other_host
from semaudit.prompts import TEMPLATE_VERSION
from semaudit.runner import CallLog, _call, _versions, _write_atomic, call_record, execute, find_log, read_log

from .backends import record_cost
from .mitigations import EXT_VERSION, build_ext, ext_followups
from .styles import (EXPLAIN_PARSER_VERSION, EXT_STYLES, TEMPLATE_VERSIONS, THREE_VALUED_STYLES, UNKNOWN_PARSER_VERSION,
                     parse_explain, parse_unknown)


class ExtCallLog(CallLog):
    """semaudit's call log; records in the extension styles get the answer-line parser's outputs, and
    records in the `unknown` style the three-valued parser's."""

    def append(self, record: dict[str, Any]) -> None:
        context = record.get("context") or {}
        if context.get("style") in EXT_STYLES and not record.get("error"):
            count_max = len(context["batch"]) if record["op"] == "count" and context.get("batch") else None
            outputs, rules = parse_explain(record["op"], record.get("text"), len(record["items"]),
                                           record.get("stop_reason"), count_max)
            record["outputs"], record["parse_rules"] = outputs, rules
            record["parser"] = EXPLAIN_PARSER_VERSION
        elif context.get("style") in THREE_VALUED_STYLES and not record.get("error"):
            outputs, rules = parse_unknown(record["op"], record.get("text"), len(record["items"]),
                                           record.get("stop_reason"))
            record["outputs"], record["parse_rules"] = outputs, rules
            record["parser"] = UNKNOWN_PARSER_VERSION
        super().append(record)


def _prices(backend: Any) -> tuple[float, float] | None:
    if hasattr(backend, "price_in") and hasattr(backend, "price_out"):
        return backend.price_in, backend.price_out
    return None


def logged_spend(records: list[dict[str, Any]], backend: Any) -> float:
    """USD already spent in a log: reported costs, or token-priced costs where none was reported."""
    prices = _prices(backend)
    return sum(record_cost(r, prices) for r in records)


def execute_arms(jobs: list[tuple[Any, Task]], logs: dict[str, CallLog], run_id: str, concurrency: int, seed: int,
                 budgets: dict[str, float | None], max_error_rate: float = 0.5, progress_every: int | None = 250,
                 max_other_host: float = 0.5) -> dict[str, list[dict[str, Any]]]:
    """semaudit.runner.execute for several backends at once: every (backend, task) pair in one seeded
    random order on one thread pool, each call logged in its backend's log. The stop rules (error
    rate, calls served by another host, spend cap) apply per backend, and any of them stops the whole
    run, because the arms are compared item by item; calls already sent are waited for and logged,
    because they are billed."""
    order = list(jobs)
    names = sorted({backend.name for backend, _ in order})
    for name in names:
        budget = budgets.get(name)
        if budget is not None and budget <= 0:
            raise RuntimeError(f"{name}: spend cap already reached; nothing was sent")
    random.Random(seed).shuffle(order)
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    done, errors, foreign = Counter(), Counter(), Counter()
    spent: Counter = Counter()
    logged: set[Any] = set()

    def record(fut: Any, comp: Any) -> tuple[Any, dict[str, Any]]:
        backend, task = futures[fut]
        rec = call_record(run_id, backend, task, comp)
        logs[backend.name].append(rec)
        records[backend.name].append(rec)
        logged.add(fut)
        return backend, rec

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_call, backend, task): (backend, task) for backend, task in order}
        try:
            for n_done, fut in enumerate(as_completed(futures), 1):
                comp = fut.result()
                backend, rec = record(fut, comp)
                name = backend.name
                done[name] += 1
                errors[name] += comp.error is not None
                foreign[name] += other_host(rec)
                spent[name] += float((comp.usage or {}).get("cost") or 0)
                if done[name] >= 20 and errors[name] / done[name] > max_error_rate:
                    raise RuntimeError(f"{name}: {errors[name]} of {done[name]} calls failed; last: {comp.error}")
                if done[name] >= 20 and foreign[name] / done[name] > max_other_host:
                    raise RuntimeError(f"{name}: {foreign[name]} of {done[name]} calls were served by another host "
                                       "than the pinned one; check the pin and identities.HOST_SLUGS")
                budget = budgets.get(name)
                if budget is not None and spent[name] > budget:
                    raise RuntimeError(f"{name}: spend cap reached (${spent[name]:.4f} spent, ${budget:.4f} allowed); "
                                       "every call sent is logged")
                if progress_every and (n_done % progress_every == 0 or n_done == len(order)):
                    print(f"[{' + '.join(names)}] {n_done}/{len(order)} calls, "
                          + ", ".join(f"{n} ${spent[n]:.4f}" for n in names), flush=True)
        except BaseException:
            for f in futures:
                f.cancel()
            for f in futures:  # calls already sent are billed: wait for them and log them
                if f in logged or f.cancelled():
                    continue
                try:
                    record(f, f.result())
                except BaseException:  # noqa: BLE001 (the original error is re-raised below)
                    continue
            raise
    return records


def _check_settings(log_path: Path, previous: list[dict[str, Any]], backend: Any) -> None:
    desc = backend.describe()
    mismatched = {(r.get("model_requested"), json.dumps(r.get("params"), sort_keys=True)) for r in previous
                  if (r.get("model_requested"), r.get("params")) != (desc["model"], desc["params"])}
    if mismatched:
        raise RuntimeError(f"{log_path} holds calls made with other settings {sorted(mismatched)[:2]}; "
                           "use a new --run-id or a new backend name")


def run_ext(config: dict[str, Any], run_id: str, backends: list[Any], root: Path,
            progress_every: int | None = 250, max_cost: dict[str, float] | None = None) -> Path:
    ds = load_data(config["data"])
    run_cfg = config["run"]
    seed = run_cfg["seed"]
    tasks, design = build_ext(ds, config)

    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    design_path = run_dir / "design.json"
    canonical_design = json.loads(json.dumps(design))
    if design_path.exists():
        stored = json.loads(design_path.read_text(encoding="utf-8"))
        if stored != canonical_design:
            raise RuntimeError(f"{run_dir} holds a different design (units, segments or the extension styles' "
                               "prompt text changed); use a new --run-id")
    else:
        _write_atomic(design_path, json.dumps(canonical_design))
    _write_atomic(run_dir / "config.json", json.dumps(config, indent=2))
    environment = dict(harness_version=__version__, ext_design=EXT_VERSION, template_version=TEMPLATE_VERSION,
                       extension_template_versions=TEMPLATE_VERSIONS, python=platform.python_version(),
                       platform=platform.platform(), packages=_versions(), stage1_tasks=len(tasks))
    caps = max_cost or {}

    if len(backends) == 1:
        backend = backends[0]
        if hasattr(backend, "bind_run"):
            backend.bind_run(run_id)
        log_path = find_log(run_dir, backend.name)
        previous = read_log(log_path)
        _check_settings(log_path, previous, backend)
        done = {r["task_id"] for r in previous if not r["error"]}
        started = time.time()
        cap = caps.get(backend.name)
        spent_before = logged_spend(previous, backend)
        with ExtCallLog(log_path) as log:
            concurrency = getattr(backend, "concurrency", None) or run_cfg.get("concurrency", 8)
            stage0 = [t for t in tasks if t.stage == 0 and t.task_id not in done]
            new = execute(stage0, backend, log, run_id, concurrency,
                          stable_seed(seed, run_id, backend.name, 0), progress_every=progress_every,
                          budget=None if cap is None else cap - spent_before)
            latest = {r["task_id"]: r for r in previous + new if not r["error"]}
            main = [t for t in tasks if t.stage != 0] + ext_followups(ds, latest.values(), design)
            execute([t for t in main if t.task_id not in done], backend, log, run_id, concurrency,
                    stable_seed(seed, run_id, backend.name, 1), progress_every=progress_every,
                    budget=None if cap is None else cap - spent_before - logged_spend(new, backend))
        manifest = {"run_id": run_id, **environment, "backend": backend.describe(),
                    "segments": [s["name"] for s in design["ext"]["segments"]],
                    "started": started, "finished": time.time()}
        _write_atomic(run_dir / f"manifest-{backend.name}.json", json.dumps(manifest, indent=2))
        return run_dir

    # Several arms: interleaved in one order, each logged in its own file.
    names = [b.name for b in backends]
    if len(set(names)) != len(names):
        raise ValueError(f"backend names must be distinct: {names}")
    started = time.time()
    previous, done, spent_before = {}, {}, {}
    with contextlib.ExitStack() as stack:
        logs: dict[str, CallLog] = {}
        for backend in backends:
            if hasattr(backend, "bind_run"):
                backend.bind_run(run_id)
            log_path = find_log(run_dir, backend.name)
            previous[backend.name] = read_log(log_path)
            _check_settings(log_path, previous[backend.name], backend)
            done[backend.name] = {r["task_id"] for r in previous[backend.name] if not r["error"]}
            spent_before[backend.name] = logged_spend(previous[backend.name], backend)
            logs[backend.name] = stack.enter_context(ExtCallLog(log_path))
        concurrency = sum(getattr(b, "concurrency", None) or run_cfg.get("concurrency", 8) for b in backends)

        def budgets(spent_now: dict[str, float]) -> dict[str, float | None]:
            return {b.name: (None if caps.get(b.name) is None else caps[b.name] - spent_now.get(b.name, 0.0))
                    for b in backends}

        stage0 = [(b, t) for b in backends for t in tasks if t.stage == 0 and t.task_id not in done[b.name]]
        new = execute_arms(stage0, logs, run_id, concurrency, stable_seed(seed, run_id, *names, 0),
                           budgets(spent_before), progress_every=progress_every)
        spent_now = {b.name: spent_before[b.name] + logged_spend(new.get(b.name, []), b) for b in backends}
        main = []
        for b in backends:
            latest = {r["task_id"]: r for r in previous[b.name] + new.get(b.name, []) if not r["error"]}
            arm_tasks = [t for t in tasks if t.stage != 0] + ext_followups(ds, latest.values(), design)
            main += [(b, t) for t in arm_tasks if t.task_id not in done[b.name]]
        execute_arms(main, logs, run_id, concurrency, stable_seed(seed, run_id, *names, 1), budgets(spent_now),
                     progress_every=progress_every)
    for backend in backends:
        manifest = {"run_id": run_id, **environment, "backend": backend.describe(),
                    "segments": [s["name"] for s in design["ext"]["segments"]],
                    "interleaved_with": [n for n in names if n != backend.name],
                    "started": started, "finished": time.time()}
        _write_atomic(run_dir / f"manifest-{backend.name}.json", json.dumps(manifest, indent=2))
    return run_dir
