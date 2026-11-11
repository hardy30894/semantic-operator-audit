"""Memory-friendly reading of semaudit call logs.

`semaudit.runner.read_log` loads a whole log into memory, prompts included (36-53 MB of JSON per
model and run in the study). The analyses here stream a log one record at a time and keep only the
fields they need, so a full study log fits in a few MB.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from semaudit.core import INVALID
from semaudit.identities import Outputs
from semaudit.runner import log_files, log_name

from .styles import parse_record_ext

# Fields kept by `compact`; the prompts (system, user) and logprobs are dropped. `text` stays because
# analysis always re-parses the raw reply with the current parser.
KEEP = ("run_id", "backend", "provider", "model_requested", "params", "task_id", "identity", "condition", "unit",
        "repeat", "stage", "op", "items", "context", "request_hash", "text", "outputs", "stop_reason", "usage",
        "error", "extra", "ts", "parser")


def iter_log(path: Path) -> Iterator[dict[str, Any]]:
    """Records of a call log (gzip stream or plain JSONL), one at a time. A last line cut off by an
    interrupted write is skipped, as in semaudit's read_log."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as fh:
        try:
            for line in fh:
                if not line.endswith(b"\n"):
                    break  # an interrupted last write
                if line.strip():
                    yield json.loads(line)
        except EOFError:  # a gzip member cut off by an interrupted write
            return


def compact(rec: dict[str, Any], prompt_chars: bool = False) -> dict[str, Any]:
    out = {k: rec.get(k) for k in KEEP}
    if prompt_chars:
        out["prompt_chars"] = len(rec.get("system") or "") + len(rec.get("user") or "")
    return out


def read_compact(path: Path, prompt_chars: bool = False) -> list[dict[str, Any]]:
    return [compact(rec, prompt_chars) for rec in iter_log(path)]


def latest_attempts(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The attempt analysis uses for each task: the latest successful one, else the latest failed one
    (the rule of semaudit.identities.collect: a failed call never overwrites a success)."""
    latest: dict[str, dict[str, Any]] = {}
    for rec in records:
        prev = latest.get(rec["task_id"])
        if prev is not None and rec.get("error") and not prev.get("error"):
            continue
        latest[rec["task_id"]] = rec
    return latest


def collect_ext(records: Iterable[dict[str, Any]]) -> Outputs:
    """semaudit.identities.collect with the extension's parser (adds the explain style)."""
    out: Outputs = {}
    for rec in records:
        for item, value in zip(rec["items"], parse_record_ext(rec)[0]):
            slot = out.setdefault((rec["identity"], rec["condition"], item), {})
            if rec.get("error") and rec["repeat"] in slot:
                continue
            slot[rec["repeat"]] = value
    return out


def run_logs(run_dir: Path, backends: list[str] | None = None) -> list[tuple[str, Path]]:
    return [(log_name(p), p) for p in log_files(run_dir) if not backends or log_name(p) in backends]


def load_design(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "design.json").read_text(encoding="utf-8"))


def load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def valid(value: Any) -> bool:
    return value != INVALID and value is not None
