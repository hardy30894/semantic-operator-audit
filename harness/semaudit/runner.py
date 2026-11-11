"""Executing tasks against a backend and logging every call.

Within a run, all requests for a backend are issued in one seeded random order, so conditions and
repeats are interleaved in time and slow provider drift cannot masquerade as a condition effect.
Every call is appended to `<run>/<backend>.jsonl.gz` with its full prompt, the request hash and the
provenance context, so a run can be resumed, audited and re-analysed without re-calling the model.
Logs are gzip streams flushed after every call (about 6x smaller than plain JSONL); plain `.jsonl`
logs from earlier runs are still read.
"""

from __future__ import annotations

import fcntl
import gzip
import json
import os
import platform
import random
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__
from .core import INVALID, Completion, FatalBackendError, Task, request_hash, stable_seed
from .data import load_data
from .identities import build, idempotence_followups
from .prompts import PARSER_VERSION, TEMPLATE_VERSION, parse_with_rules


LOG_SUFFIXES = (".jsonl.gz", ".jsonl")


def log_name(path: Path) -> str:
    """The backend a call log belongs to: `haiku.jsonl.gz` and `haiku.jsonl` both give `haiku`."""
    for suffix in LOG_SUFFIXES:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    raise ValueError(f"{path} is not a call log")


def log_files(run_dir: Path) -> list[Path]:
    """The run's call logs, one per backend, sorted by backend name."""
    found: dict[str, Path] = {}
    for path in (run_dir.iterdir() if run_dir.is_dir() else []):
        if path.name.startswith(".") or not path.name.endswith(LOG_SUFFIXES):
            continue
        name = log_name(path)
        if name in found:
            raise RuntimeError(f"{run_dir} has two logs for {name}: {found[name].name} and {path.name}")
        found[name] = path
    return [found[name] for name in sorted(found)]


def find_log(run_dir: Path, backend_name: str) -> Path:
    """The backend's existing log in this run, or the path of a new gzip log."""
    existing = [p for p in log_files(run_dir) if log_name(p) == backend_name]
    return existing[0] if existing else run_dir / f"{backend_name}.jsonl.gz"


def _gunzip_prefix(data: bytes) -> tuple[bytes, bool]:
    """Decompress concatenated gzip members. A last member cut off by an interrupted write gives what
    was written before the cut; also returns whether the last member was complete."""
    chunks = []
    while data:
        member = zlib.decompressobj(16 + zlib.MAX_WBITS)
        chunks.append(member.decompress(data))
        if not member.eof:
            return b"".join(chunks), False
        data = member.unused_data
    return b"".join(chunks), True


def _repair_log(path: Path) -> None:
    """Cut a log whose last write was interrupted back to its complete records, so that new records
    can be appended after it (a gzip log also needs its unfinished member closed)."""
    raw = path.read_bytes()
    if path.name.endswith(".gz"):
        data, complete = _gunzip_prefix(raw)
        if complete and (not data or data.endswith(b"\n")):
            return
        keep = data[: data.rfind(b"\n") + 1]
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with gzip.open(tmp, "wb") as fh:
            fh.write(keep)
        os.replace(tmp, path)
    else:
        if not raw or raw.endswith(b"\n"):
            return
        keep = raw[: raw.rfind(b"\n") + 1]
        os.truncate(path, len(keep))
    n = keep.count(b"\n")
    print(f"warning: {path} ended in an interrupted write; kept its {n} complete records", flush=True)


class CallLog:
    """Append-only JSONL, one line per call, safe to write from worker threads.

    A `.jsonl.gz` log is a gzip stream flushed (Z_SYNC_FLUSH) after every call, so all logged calls
    survive an interruption. Reopening a log whose last write was cut off first rewrites it to its
    complete records. A lock file stops a second process from writing the same log.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._lockfile = open(path.with_name(f".{log_name(path)}.lock"), "a")
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lockfile.close()
            raise RuntimeError(f"another process is writing {path}") from None
        if path.exists():
            _repair_log(path)
        self._fh = gzip.open(path, "ab") if path.name.endswith(".gz") else open(path, "ab")

    def append(self, record: dict[str, Any]) -> None:
        line = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            self._lockfile.close()

    def __enter__(self) -> "CallLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_log(path: Path) -> list[dict[str, Any]]:
    """All records. A last line without its newline (an interrupted write) is skipped; any other
    unreadable line raises."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if path.name.endswith(".gz"):
        raw, _ = _gunzip_prefix(raw)
    *lines, tail = raw.split(b"\n")
    if tail.strip():
        print(f"warning: skipped a truncated last line in {path}", flush=True)
    return [json.loads(line) for line in lines if line.strip()]


def _call(backend: Any, task: Task) -> Completion:
    start = time.monotonic()
    try:
        comp = backend.complete(task)
    except FatalBackendError:
        raise
    except Exception as exc:  # an unexpected failure is logged as a failed call, not lost
        return Completion(text=None, error=f"{type(exc).__name__}: {exc}",
                          latency_s=time.monotonic() - start)
    if not comp.latency_s:
        comp.latency_s = time.monotonic() - start
    return comp


def call_record(run_id: str, backend: Any, task: Task, comp: Completion) -> dict[str, Any]:
    desc = backend.describe()
    if comp.error:
        outputs, rules = [INVALID] * len(task.items), ["error"] * len(task.items)
    else:
        count_max = len(task.context["batch"]) if task.op == "count" and task.context.get("batch") else None
        outputs, rules = parse_with_rules(task.op, comp.text, len(task.items),
                                          task.context.get("style", "lotus"), comp.stop_reason, count_max)
    return {
        "run_id": run_id,
        "backend": backend.name,
        "provider": desc["provider"],
        "model_requested": desc["model"],
        "params": desc["params"],
        "task_id": task.task_id,
        "identity": task.identity,
        "condition": task.condition,
        "unit": task.unit,
        "repeat": task.repeat,
        "stage": task.stage,
        "op": task.op,
        "items": list(task.items),
        "context": task.context,
        "system": task.system,
        "user": task.user,
        "request_hash": request_hash(desc["provider"], desc["model"], desc["params"], task.system, task.user),
        "text": comp.text,
        "outputs": outputs,
        "parse_rules": rules,
        "parser": PARSER_VERSION,
        "served_model": comp.served_model,
        "request_id": comp.request_id,
        "response_id": comp.response_id,
        "stop_reason": comp.stop_reason,
        "usage": comp.usage,
        "logprobs": comp.logprobs,
        "latency_s": round(comp.latency_s, 4),
        "error": comp.error,
        "extra": comp.extra,
        "ts": time.time(),
    }


def execute(tasks: list[Task], backend: Any, log: CallLog, run_id: str, concurrency: int, seed: int,
            max_error_rate: float = 0.5, progress_every: int | None = 250,
            budget: float | None = None, max_other_host: float = 0.5) -> list[dict[str, Any]]:
    """Issue the tasks in one seeded random order. Stops if more than `max_error_rate` of the calls
    fail, if more than `max_other_host` are served by a host other than the pinned one (analysis
    would exclude them), or once the logged cost exceeds `budget` USD. On a stop, calls already
    sent are waited for and logged, because they are billed."""
    from .identities import other_host

    order = list(tasks)
    if order and budget is not None and budget <= 0:
        raise RuntimeError(f"{backend.name}: spend cap already reached; nothing was sent")
    random.Random(seed).shuffle(order)
    records: list[dict[str, Any]] = []
    errors = foreign = 0
    spent = 0.0
    logged: set[Any] = set()

    def record(fut: Any, comp: Completion) -> dict[str, Any]:
        rec = call_record(run_id, backend, futures[fut], comp)
        log.append(rec)
        records.append(rec)
        logged.add(fut)
        return rec

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_call, backend, t): t for t in order}
        try:
            for done, fut in enumerate(as_completed(futures), 1):
                comp = fut.result()
                rec = record(fut, comp)
                errors += comp.error is not None
                foreign += other_host(rec)
                spent += float(comp.usage.get("cost") or 0)
                if done >= 20 and errors / done > max_error_rate:
                    raise RuntimeError(f"{backend.name}: {errors} of {done} calls failed; last: {comp.error}")
                if done >= 20 and foreign / done > max_other_host:
                    raise RuntimeError(f"{backend.name}: {foreign} of {done} calls were served by another host "
                                       "than the pinned one; check the pin and identities.HOST_SLUGS")
                if budget is not None and spent > budget:
                    raise RuntimeError(f"{backend.name}: spend cap reached (${spent:.4f} spent, ${budget:.4f} "
                                       "allowed); every call sent is logged")
                if progress_every and (done % progress_every == 0 or done == len(order)):
                    print(f"[{backend.name}] {done}/{len(order)} calls, {errors} failed, "
                          f"${spent:.4f} spent", flush=True)
        except BaseException:
            for f in futures:
                f.cancel()
            for f in futures:  # calls already sent are billed: wait for them and log them
                if f in logged or f.cancelled():
                    continue
                try:
                    record(f, f.result())
                except BaseException:  # noqa: BLE001 (a fatal error is re-raised below)
                    continue
            raise
    return records


def _write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file and rename, so parallel runs never read a half-written file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _versions() -> dict[str, str]:
    found = {}
    for pkg in ("anthropic", "openai", "google-genai", "numpy"):
        try:
            found[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return found


def run_experiment(config: dict[str, Any], run_id: str, backends: list[Any], root: Path,
                   progress_every: int | None = 250, max_cost: dict[str, float] | None = None) -> Path:
    ds = load_data(config["data"])
    run_cfg = config["run"]
    repeats, seed = run_cfg["repeats"], run_cfg["seed"]
    tasks, design = build(ds, config["identities"], repeats, seed, run_cfg.get("prompt_style", "lotus"))

    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    design_path = run_dir / "design.json"
    canonical_design = json.loads(json.dumps(design))
    if design_path.exists():
        stored = json.loads(design_path.read_text(encoding="utf-8"))
        stored.setdefault("style", "lotus")  # designs written before prompt styles existed
        if stored != canonical_design:
            raise RuntimeError(f"{run_dir} holds a different design; use a new --run-id")
    else:
        _write_atomic(design_path, json.dumps(canonical_design))
    _write_atomic(run_dir / "config.json", json.dumps(config, indent=2))
    environment = dict(harness_version=__version__, template_version=TEMPLATE_VERSION,
                       python=platform.python_version(), platform=platform.platform(),
                       packages=_versions(), stage1_tasks=len(tasks))

    for backend in backends:
        if hasattr(backend, "bind_run"):
            backend.bind_run(run_id)
        log_path = find_log(run_dir, backend.name)
        previous = read_log(log_path)
        desc = backend.describe()
        mismatched = {(r.get("model_requested"), json.dumps(r.get("params"), sort_keys=True)) for r in previous
                      if (r.get("model_requested"), r.get("params")) != (desc["model"], desc["params"])}
        if mismatched:
            raise RuntimeError(f"{log_path} holds calls made with other settings {sorted(mismatched)[:2]}; "
                               "use a new --run-id or a new backend name")
        done = {r["task_id"] for r in previous if not r["error"]}
        started = time.time()
        cap = (max_cost or {}).get(backend.name)
        spent_before = sum(float((r.get("usage") or {}).get("cost") or 0) for r in previous)
        with CallLog(log_path) as log:
            concurrency = getattr(backend, "concurrency", None) or run_cfg.get("concurrency", 8)
            # Stage 0: inputs that later requests depend on (idempotence's first application).
            stage0 = [t for t in tasks if t.stage == 0 and t.task_id not in done]
            new = execute(stage0, backend, log, run_id, concurrency,
                          stable_seed(seed, run_id, backend.name, 0), progress_every=progress_every,
                          budget=None if cap is None else cap - spent_before)
            latest = {r["task_id"]: r for r in previous + new if not r["error"]}
            # Main phase: every other request plus the dependent ones, shuffled together so that
            # no condition is confounded with time.
            followups = idempotence_followups(ds, latest.values())
            main = [t for t in tasks if t.stage != 0] + followups
            spent_stage0 = sum(float((r.get("usage") or {}).get("cost") or 0) for r in new)
            execute([t for t in main if t.task_id not in done], backend, log, run_id, concurrency,
                    stable_seed(seed, run_id, backend.name, 1), progress_every=progress_every,
                    budget=None if cap is None else cap - spent_before - spent_stage0)
        # One manifest per backend, so backends can run in parallel processes on the same run.
        manifest = {"run_id": run_id, **environment, "backend": backend.describe(),
                    "started": started, "finished": time.time()}
        _write_atomic(run_dir / f"manifest-{backend.name}.json", json.dumps(manifest, indent=2))
    return run_dir
