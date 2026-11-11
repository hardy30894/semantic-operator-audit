"""Offline backends for extension runs (no API calls).

semaudit's control arm and simulator answer in the bare format of the `lotus` and `plain` styles.
`ExplainAware` wraps either of them so that requests in the extension styles get replies in their
answer-line format, as a model following the style would answer:

- `explain`: a line of reasoning, then "Answer: True" (or an "Answers:" block for batches);
- `format`: only the answer line ("Answer: True", or the "Answers:" block).

Other requests pass through unchanged.
"""

from __future__ import annotations

from typing import Any

from semaudit.core import Completion, Task
from semaudit.offline import ControlBackend, SimulatedBackend

from .styles import EXPLAIN, EXT_STYLES


class ExplainAware:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = inner.name
        self.concurrency = getattr(inner, "concurrency", None)

    def describe(self) -> dict[str, Any]:
        return self.inner.describe()

    def bind_run(self, run_id: str) -> None:
        if hasattr(self.inner, "bind_run"):
            self.inner.bind_run(run_id)

    def complete(self, task: Task) -> Completion:
        style = task.context.get("style")
        if style not in EXT_STYLES:
            return self.inner.complete(task)
        # The wrapped backends answer in a style's own words; the extension styles use lotus's.
        plain = Task(**{**task.__dict__, "context": {**task.context, "style": "lotus"}})
        comp = self.inner.complete(plain)
        answer = f"Answers:\n{comp.text}" if task.op == "filter_batch" else f"Answer: {comp.text}"
        comp.text = f"The record was checked against the instruction.\n{answer}" if style == EXPLAIN else answer
        return comp


def offline_backend(spec: dict[str, Any]) -> Any:
    """An offline backend for a roster entry. Two optional settings (Addendum G, extension/placebo_offline.py):
    params `standin_truth = true` gives payloads without rules a stand-in truth, and any of
    placebo_offline.SIM_PARAMS makes a simulated arm a PlaceboSimulator. Entries without them are built as before."""
    kind, name, params = spec["kind"], spec["name"], dict(spec.get("params", {}))
    from .placebo_offline import SIM_PARAMS, PlaceboSimulator, StandInTruth

    standin = bool(params.pop("standin_truth", False))
    extra = {key: params.pop(key) for key in SIM_PARAMS if key in params}
    if kind == "control":
        if extra:
            raise ValueError(f"{name}: simulator parameters {sorted(extra)} on a control arm")
        inner = ControlBackend(name)
    elif kind == "simulated":
        inner = PlaceboSimulator(name, **extra, **params) if extra else SimulatedBackend(name, **params)
    else:
        raise ValueError(f"not an offline kind: {kind}")
    inner.concurrency = spec.get("concurrency")
    if standin:
        inner = StandInTruth(inner)
    return ExplainAware(inner)
