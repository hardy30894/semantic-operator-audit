"""Backend construction for extension runs, on top of extension/priced.py.

- **Logged cost for every paid kind.** semaudit's spend caps read usage["cost"]. OpenRouter reports
  the billed cost. Anthropic's, OpenAI's and Google's APIs report tokens only, so:
  - kind "anthropic" is built as Addendum A's PricedAnthropicBackend (extension/priced.py), which
    also loads ANTHROPIC_API_KEY from the project .env when the environment lacks it;
  - kinds "openai" and "google" are wrapped in TokenPriced, which writes the same computed cost
    (input tokens x input price + output tokens x output price, per million, from the roster) into
    every completion that carries token counts and no cost.
  Every non-OpenRouter paid kind therefore needs roster prices, or the backend is refused.
- **Key.** `ensure_anthropic_key` copies ANTHROPIC_API_KEY from the environment or a .env file
  (semaudit.env.env_value) into os.environ. priced.py does the same on construction; this keeps
  the extension's entry points independent of that. The value is never printed or logged.
- **Offline kinds.** "control" and "simulated" are wrapped with extension.offline.ExplainAware, so
  they answer the extension styles' requests in their answer-line format.
- **Usage summaries.** `record_cost` returns a logged call's cost, computing it from tokens at the
  roster's prices for a provider that does not bill per call (Anthropic, OpenAI, Google) when a
  record carries no cost.
- **Roster checks.** `settings_warnings` flags settings that differ from what the arm's name
  promises, without blocking them.
"""

from __future__ import annotations

import os
from typing import Any

from semaudit import backends as v1_backends
from semaudit.core import Completion, FatalBackendError, Task
from semaudit.env import env_value

from . import priced

ANTHROPIC_KEY = "ANTHROPIC_API_KEY"
TOKEN_PRICED_KINDS = ("anthropic", "openai", "google")  # paid kinds whose API reports no cost
COST_SOURCE = "list price x tokens"

# How to turn thinking off, per Claude model that thinks when a request has no `thinking` parameter
# (Anthropic API docs, models of 2026-10). None: the model cannot turn thinking off; only effort
# lowers it. v1's haiku-5.5 arm ran with thinking off (3 output tokens per one-word answer).
THINKING_OFF = {"claude-haiku-5-5": "disabled", "claude-sonnet-5-5": "between_tools", "claude-opus-5-5": None,
                "claude-opus-5": "disabled", "claude-fable-5": None, "claude-fable-5-1": None}


def ensure_anthropic_key() -> None:
    """Make ANTHROPIC_API_KEY available to the Anthropic SDK; reads only that variable."""
    if os.environ.get(ANTHROPIC_KEY):
        return
    value = env_value(ANTHROPIC_KEY)
    if not value:
        raise FatalBackendError("no Anthropic API key: set ANTHROPIC_API_KEY in the environment or the project .env")
    os.environ[ANTHROPIC_KEY] = value


def token_cost(usage: dict[str, Any], price_in: float, price_out: float) -> float | None:
    """USD for one call from its token counts (thinking is billed as output), or None without counts."""
    if usage.get("input_tokens") is None and usage.get("output_tokens") is None:
        return None
    return ((usage.get("input_tokens") or 0) * price_in + (usage.get("output_tokens") or 0) * price_out) / 1e6


def spec_prices(spec: dict[str, Any] | None) -> tuple[float, float] | None:
    if not spec or "price_in_per_mtok" not in spec:
        return None
    return float(spec["price_in_per_mtok"]), float(spec.get("price_out_per_mtok", 0.0))


class TokenPriced:
    """Writes a list-price cost into completions that report tokens but no cost (OpenAI, Google).
    Requests and the logged settings are the wrapped backend's own."""

    _OWN = ("inner", "name", "price_in", "price_out", "concurrency")

    def __init__(self, inner: Any, price_in: float, price_out: float) -> None:
        self.inner = inner
        self.name = inner.name
        self.price_in, self.price_out = price_in, price_out
        self.concurrency = getattr(inner, "concurrency", None)

    def __getattr__(self, attr: str) -> Any:  # client, params, ... are the wrapped backend's
        return getattr(self.inner, attr)

    def __setattr__(self, attr: str, value: Any) -> None:  # e.g. a test replacing the client
        if attr in self._OWN:
            object.__setattr__(self, attr, value)
        else:
            setattr(self.inner, attr, value)

    def describe(self) -> dict[str, Any]:
        return self.inner.describe()

    def bind_run(self, run_id: str) -> None:
        if hasattr(self.inner, "bind_run"):
            self.inner.bind_run(run_id)

    def complete(self, task: Task) -> Completion:
        comp = self.inner.complete(task)
        usage = comp.usage or {}
        if usage.get("cost") is None:
            cost = token_cost(usage, self.price_in, self.price_out)
            if cost is not None:
                usage["cost"] = cost
                usage["cost_source"] = COST_SOURCE
                comp.usage = usage
        return comp


def make_backend_ext(spec: dict[str, Any]) -> Any:
    """A backend for a roster entry: offline kinds explain-aware; anthropic priced (Addendum A),
    after loading the key; openai and google priced from tokens; openrouter as semaudit builds it."""
    kind = spec["kind"]
    if kind in ("control", "simulated"):
        from .offline import offline_backend

        return offline_backend(spec)
    if kind in TOKEN_PRICED_KINDS and not {"price_in_per_mtok", "price_out_per_mtok"} <= set(spec):
        raise SystemExit(f"{spec['name']}: kind {kind!r} reports tokens but no cost; the roster needs "
                         "price_in_per_mtok and price_out_per_mtok so its spend can be logged and capped")
    if kind == "anthropic":
        ensure_anthropic_key()
        backend = priced._build_backend(spec)
        backend.concurrency = spec.get("concurrency")
        return backend
    backend = v1_backends.make_backend(spec)
    if kind in TOKEN_PRICED_KINDS:
        return TokenPriced(backend, *spec_prices(spec))
    return backend


def record_cost(rec: dict[str, Any], prices: tuple[float, float] | None = None) -> float:
    """USD for a logged call: its logged usage["cost"] (OpenRouter's billed cost, or a list-price
    cost written at run time), else its tokens at `prices` for a provider that does not bill per call
    (Anthropic, OpenAI, Google), else 0. A missing OpenRouter cost is never guessed."""
    usage = rec.get("usage") or {}
    if usage.get("cost") is not None:
        return float(usage["cost"])
    if prices is not None and rec.get("provider", "anthropic") in TOKEN_PRICED_KINDS:
        cost = token_cost(usage, *prices)
        if cost is not None:
            return cost
    return 0.0


def settings_warnings(spec: dict[str, Any]) -> list[str]:
    """Non-blocking notes about a roster entry's settings."""
    notes = []
    params = spec.get("params", {})
    model = spec.get("model")
    if spec.get("kind") == "anthropic" and model in THINKING_OFF and "thinking" not in params:
        off = THINKING_OFF[model]
        how = (f"add thinking = \"{off}\" to params to turn it off" if off else
               "it cannot be turned off; set effort to lower it")
        notes.append(f"{spec['name']}: {model} runs adaptive thinking when a request has no thinking parameter; "
                     f"{how}. v1's haiku-5.5 arm ran with thinking off (3 output tokens per one-word answer). "
                     f"Thinking also counts toward max_tokens ({params.get('max_tokens', 1024)}).")
    thinking = params.get("thinking")
    if spec.get("kind") == "anthropic" and thinking == "enabled":
        notes.append(f"{spec['name']}: thinking \"enabled\" needs budget_tokens, which current Claude models reject; "
                     "use \"adaptive\" with an effort level")
    if spec.get("kind") == "anthropic" and model in THINKING_OFF and thinking in ("disabled", "between_tools"):
        off = THINKING_OFF[model]
        if off is None:
            notes.append(f"{spec['name']}: {model} cannot turn thinking off; thinking \"{thinking}\" is rejected")
        elif thinking != off:
            notes.append(f"{spec['name']}: {model} turns thinking off with \"{off}\", not \"{thinking}\"")
        elif off == "disabled" and params.get("effort") in ("xhigh", "max"):
            notes.append(f"{spec['name']}: thinking \"disabled\" is rejected at effort xhigh or max")
    if spec.get("kind") == "anthropic" and {"temperature", "top_p", "top_k"} & set(params):
        notes.append(f"{spec['name']}: current Claude models reject non-default temperature, top_p and top_k")
    return notes


def is_paid(spec: dict[str, Any]) -> bool:
    return spec["kind"] in v1_backends.PAID_KINDS
