"""Direct Anthropic calls with a logged cost, so the harness's spend caps apply to them.

Anthropic's API returns token counts, not a billed amount. PricedAnthropicBackend adds usage["cost"]
at the roster's list prices; the runner's caps and `semaudit cost` read that field as they read
OpenRouter's billed cost. Requests are exactly those of AnthropicBackend. No cache_control is sent,
so every input token is billed at the base input price. The SDK reads ANTHROPIC_API_KEY from the
environment only, so the key is taken from the project .env (semaudit.env) when the environment lacks it.
"""

from __future__ import annotations

import os
from typing import Any

from semaudit import backends
from semaudit.backends import AnthropicBackend
from semaudit.core import Completion, Task
from semaudit.env import env_value

COST_SOURCE = "list price x tokens"


class PricedAnthropicBackend(AnthropicBackend):
    def __init__(self, name: str, model: str, price_in_per_mtok: float, price_out_per_mtok: float,
                 **params: Any) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            key = env_value("ANTHROPIC_API_KEY")
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key
        super().__init__(name, model, **params)
        self.price_in, self.price_out = price_in_per_mtok, price_out_per_mtok

    def complete(self, task: Task) -> Completion:
        comp = super().complete(task)
        usage = comp.usage or {}
        if "input_tokens" in usage:  # failed calls carry no usage and are not billed
            usage["cost"] = (usage["input_tokens"] * self.price_in
                             + usage.get("output_tokens", 0) * self.price_out) / 1e6
            usage["cost_source"] = COST_SOURCE
            comp.usage = usage
        return comp


_build_v1 = backends._build_backend


def _build_backend(spec: dict[str, Any]):
    if spec["kind"] == "anthropic":
        return PricedAnthropicBackend(spec["name"], spec["model"], spec["price_in_per_mtok"],
                                      spec["price_out_per_mtok"], **spec.get("params", {}))
    return _build_v1(spec)


def install() -> None:
    """Make semaudit build priced backends for roster entries of kind "anthropic"."""
    backends._build_backend = _build_backend
