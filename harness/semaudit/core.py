"""Shared types for the audit harness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# Marker for an output that could not be parsed, or for a call that failed. A string, so it
# survives JSON round-trips and never compares equal to a boolean, integer or rewritten value.
INVALID = "__INVALID__"


class FatalBackendError(RuntimeError):
    """A provider error that will not go away on retry (bad request, auth, unknown model)."""


@dataclass
class Task:
    """One operator invocation, i.e. one LLM request."""

    task_id: str
    identity: str
    condition: str
    unit: str  # scheduling unit: an item id, or a batch id for multi-row prompts
    repeat: int
    op: str  # filter | match | map | count | filter_batch
    items: tuple[str, ...]  # analysis items whose outputs this call yields, in prompt order
    system: str
    user: str
    payload: dict[str, Any]  # structured inputs for the offline backends; never sent to providers
    context: dict[str, Any] = field(default_factory=dict)  # provenance fields logged with the call
    stage: int = 1


@dataclass
class Completion:
    text: str | None
    served_model: str | None = None
    request_id: str | None = None
    response_id: str | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    logprobs: Any = None
    latency_s: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def request_hash(provider: str, model: str, params: dict[str, Any], system: str, user: str) -> str:
    """Hash of everything the provider sees. Equal hashes mean byte-identical requests."""
    blob = json.dumps(
        {"provider": provider, "model": model, "params": params, "system": system, "user": user},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    """Process-independent 64-bit seed (Python's hash() is salted per process)."""
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
