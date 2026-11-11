"""The grid's backend wrapper: the full raw API response of every call, and guards on the settings.

`GridBackend` wraps a backend built by the harness (extension.backends.make_backend_ext), so requests,
settings, costs and the logged record format stay the harness's own. It adds two things:

- **Raw responses.** The SDK call (`client.chat.completions.create` for OpenRouter,
  `client.messages.create` for Anthropic) is wrapped per thread, and the response body it returns is
  stored in the call record's `extra["raw_response"]`, next to the fields the harness already logs.
- **Guards.** A call is turned into a logged failure, and the backend is tripped so its next call raises
  FatalBackendError and the run stops, when the call reports reasoning tokens or a thinking block, the
  served model differs from the one requested, or OpenRouter reports a host other than the pinned one.
  The offending call is therefore logged before the run stops; a resume after the cause is fixed re-sends it.
"""
from __future__ import annotations

import threading
from typing import Any

from semaudit.core import Completion, FatalBackendError, Task
from semaudit.identities import _host_slug

GUARD_PREFIX = "grid guard: "
# A spending or key limit is not a transient failure: stop the model at once instead of logging thousands of
# failed calls (OpenRouter answers 403 "Key limit exceeded ... does not reset"; Anthropic reports low credit).
LIMIT_MARKERS = ("key limit exceeded", "credit balance", "insufficient credit", "credits exhausted",
                 "spending limit", "usage limit", "billing", "quota")


def _dump(resp: Any) -> Any:
    try:
        return resp.model_dump(mode="json")
    except Exception:  # noqa: BLE001 (an unexpected SDK object is stored as text, never lost)
        return repr(resp)


class GridBackend:
    _OWN = ("inner", "name", "concurrency", "tripped", "_local", "_client_owner")

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "name", inner.name)
        object.__setattr__(self, "concurrency", getattr(inner, "concurrency", None))
        object.__setattr__(self, "tripped", None)
        object.__setattr__(self, "_local", threading.local())
        owner = inner
        while not hasattr(owner, "client") and hasattr(owner, "inner"):
            owner = owner.inner
        object.__setattr__(self, "_client_owner", owner if hasattr(owner, "client") else None)
        self._wrap_client()

    # attributes such as params and model are the wrapped backend's
    def __getattr__(self, attr: str) -> Any:
        return getattr(self.inner, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr in self._OWN:
            object.__setattr__(self, attr, value)
        else:
            setattr(self.inner, attr, value)

    def describe(self) -> dict[str, Any]:
        return self.inner.describe()

    def bind_run(self, run_id: str) -> None:
        if hasattr(self.inner, "bind_run"):
            self.inner.bind_run(run_id)

    def _wrap_client(self) -> None:
        owner = self._client_owner
        if owner is None:
            return
        client = owner.client
        resource = getattr(getattr(client, "chat", None), "completions", None) or getattr(client, "messages", None)
        if resource is None or getattr(resource, "_grid_wrapped", False):
            return
        original = resource.create
        local = self._local

        def create(*args: Any, **kwargs: Any) -> Any:
            resp = original(*args, **kwargs)
            local.raw = _dump(resp)
            return resp

        resource.create = create
        resource._grid_wrapped = True

    def _guard(self, comp: Completion, raw: Any) -> str | None:
        if comp.error:
            low = comp.error.lower()
            return f"account limit: {comp.error[:200]}" if any(m in low for m in LIMIT_MARKERS) else None
        usage = comp.usage or {}
        if (usage.get("reasoning_tokens") or 0) > 0:
            return f"{usage['reasoning_tokens']} reasoning tokens"
        if isinstance(raw, dict) and any(isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
                                         for b in raw.get("content") or []):
            return "a thinking block"
        requested = self.describe()["model"]
        if comp.served_model is not None and comp.served_model != requested:
            return f"served model {comp.served_model!r}, requested {requested!r}"
        pinned = ((self.describe()["params"].get("provider") or {}).get("order") or [None])[0]
        served = (comp.extra or {}).get("served_provider")
        if pinned and served and _host_slug(str(served)) != str(pinned).split("/")[0].lower():
            return f"served by {served!r}, pinned to {pinned!r}"
        return None

    def complete(self, task: Task) -> Completion:
        if self.tripped:
            raise FatalBackendError(f"{self.name}: stopped by the grid guard: {self.tripped}")
        self._local.raw = None
        comp = self.inner.complete(task)
        raw = getattr(self._local, "raw", None)
        comp.extra = {**(comp.extra or {}), "raw_response": raw}
        problem = self._guard(comp, raw)
        if problem:
            self.tripped = problem
            if not comp.error:
                comp.error = GUARD_PREFIX + problem
        return comp
