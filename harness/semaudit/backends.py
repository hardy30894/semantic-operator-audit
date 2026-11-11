"""Provider backends. Each turns a Task into a Completion and describes its exact settings.

Errors that retrying cannot fix (authentication, unknown model; a bad request on the direct
provider SDKs) raise FatalBackendError so a misconfigured run stops at once. Through OpenRouter, a
rejected request is logged as one failed call, because it can be specific to one prompt; a run in
which most calls fail stops at its error-rate limit. Rate limits and server errors are retried by
the SDKs; whatever survives those retries is logged as a failed call and the run continues.
"""

from __future__ import annotations

import random
import time
from typing import Any

from .core import Completion, FatalBackendError, Task


def _failed(message: str, start: float) -> Completion:
    return Completion(text=None, error=message, latency_s=time.monotonic() - start)


class AnthropicBackend:
    """Claude via the official SDK.

    Current Claude models reject non-default sampling parameters, so no temperature is sent, and
    the API returns no logprobs. Refusal fallbacks are deliberately left off: a fallback answers
    with a different model, which would contaminate per-model measurements. A refusal is logged
    (stop_reason "refusal") and its output is counted as invalid.
    """

    provider = "anthropic"

    def __init__(self, name: str, model: str, max_tokens: int = 1024, thinking: str | None = None,
                 effort: str | None = None, max_retries: int = 8, timeout: float = 120.0) -> None:
        import anthropic

        self._sdk = anthropic
        self.client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)
        self.name = name
        self.model = model
        self.params: dict[str, Any] = {"max_tokens": max_tokens}
        if thinking:
            self.params["thinking"] = {"type": thinking}
        if effort:
            self.params["output_config"] = {"effort": effort}

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "params": self.params}

    def complete(self, task: Task) -> Completion:
        sdk = self._sdk
        start = time.monotonic()
        try:
            msg = self.client.messages.create(
                model=self.model,
                system=task.system,
                messages=[{"role": "user", "content": task.user}],
                **self.params,
            )
        except (sdk.BadRequestError, sdk.AuthenticationError, sdk.PermissionDeniedError,
                sdk.NotFoundError) as e:
            raise FatalBackendError(f"{self.name}: {type(e).__name__}: {e.message}") from e
        except sdk.APIStatusError as e:  # 429 / 5xx still failing after the SDK's retries
            return _failed(f"{type(e).__name__}({e.status_code}): {e.message}", start)
        except sdk.APIConnectionError as e:
            return _failed(f"{type(e).__name__}: {e}", start)
        text = "".join(block.text for block in msg.content if block.type == "text")
        details = getattr(msg, "stop_details", None)
        return Completion(
            text=text or None,
            served_model=msg.model,
            request_id=getattr(msg, "_request_id", None),
            response_id=msg.id,
            stop_reason=msg.stop_reason,
            usage={"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens},
            latency_s=time.monotonic() - start,
            extra={"stop_details": details.model_dump() if details is not None else None},
        )


class OpenAIBackend:
    """OpenAI chat completions. Sampling and logprob settings are sent only when configured,
    because reasoning models reject temperature and logprobs."""

    provider = "openai"

    def __init__(self, name: str, model: str, max_completion_tokens: int = 1024,
                 temperature: float | None = None, seed: int | None = None,
                 top_logprobs: int | None = None, reasoning_effort: str | None = None,
                 max_retries: int = 8, timeout: float = 120.0) -> None:
        import openai

        self._sdk = openai
        self.client = openai.OpenAI(max_retries=max_retries, timeout=timeout)
        self.name = name
        self.model = model
        self.params: dict[str, Any] = {"max_completion_tokens": max_completion_tokens}
        if temperature is not None:
            self.params["temperature"] = temperature
        if seed is not None:
            self.params["seed"] = seed
        if top_logprobs is not None:
            self.params.update(logprobs=True, top_logprobs=top_logprobs)
        if reasoning_effort:
            self.params["reasoning_effort"] = reasoning_effort

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "params": self.params}

    def complete(self, task: Task) -> Completion:
        sdk = self._sdk
        start = time.monotonic()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": task.system},
                          {"role": "user", "content": task.user}],
                **self.params,
            )
        except (sdk.BadRequestError, sdk.AuthenticationError, sdk.PermissionDeniedError,
                sdk.NotFoundError) as e:
            raise FatalBackendError(f"{self.name}: {type(e).__name__}: {e.message}") from e
        except sdk.APIStatusError as e:
            return _failed(f"{type(e).__name__}({e.status_code}): {e.message}", start)
        except sdk.APIConnectionError as e:
            return _failed(f"{type(e).__name__}: {e}", start)
        choice = resp.choices[0]
        logprobs = None
        if choice.logprobs is not None and choice.logprobs.content:
            logprobs = [{"token": t.token, "logprob": t.logprob,
                         "top": [[a.token, a.logprob] for a in (t.top_logprobs or [])]}
                        for t in choice.logprobs.content]
        usage = ({"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
                 if resp.usage else {})
        return Completion(
            text=choice.message.content,
            served_model=resp.model,
            request_id=getattr(resp, "_request_id", None),
            response_id=resp.id,
            stop_reason=choice.finish_reason,
            usage=usage,
            logprobs=logprobs,
            latency_s=time.monotonic() - start,
            extra={"system_fingerprint": resp.system_fingerprint},
        )


class GoogleBackend:
    """Gemini via google-genai.

    UNTESTED: google-genai is not installed in this environment. Install it and run a small pilot
    before relying on this adapter.
    """

    provider = "google"
    _RETRYABLE = {429, 500, 502, 503, 504}
    _FATAL = {400, 401, 403, 404}

    def __init__(self, name: str, model: str, max_output_tokens: int = 1024,
                 temperature: float | None = None, seed: int | None = None,
                 thinking_budget: int | None = None, logprobs: int | None = None,
                 attempts: int = 8) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        self.name = name
        self.model = model
        self.attempts = attempts
        self.params: dict[str, Any] = {"max_output_tokens": max_output_tokens}
        if temperature is not None:
            self.params["temperature"] = temperature
        if seed is not None:
            self.params["seed"] = seed
        if logprobs is not None:
            self.params.update(response_logprobs=True, logprobs=logprobs)
        if thinking_budget is not None:
            self.params["thinking_budget"] = thinking_budget

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "params": self.params}

    def complete(self, task: Task) -> Completion:
        types = self._types
        settings = {k: v for k, v in self.params.items() if k != "thinking_budget"}
        if "thinking_budget" in self.params:
            settings["thinking_config"] = types.ThinkingConfig(thinking_budget=self.params["thinking_budget"])
        config = types.GenerateContentConfig(system_instruction=task.system, **settings)
        start = time.monotonic()
        for attempt in range(self.attempts):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=task.user, config=config)
                break
            except Exception as e:  # google-genai raises errors.APIError carrying an HTTP `code`
                code = getattr(e, "code", None)
                if code in self._FATAL:
                    raise FatalBackendError(f"{self.name}: {type(e).__name__}({code}): {e}") from e
                if attempt == self.attempts - 1 or (code is not None and code not in self._RETRYABLE):
                    return _failed(f"{type(e).__name__}({code}): {e}", start)
                time.sleep(min(60.0, 2 ** attempt + random.random()))
        candidate = resp.candidates[0] if resp.candidates else None
        meta = resp.usage_metadata
        return Completion(
            text=resp.text,
            served_model=resp.model_version,
            response_id=resp.response_id,
            stop_reason=str(candidate.finish_reason) if candidate else None,
            usage=({"input_tokens": meta.prompt_token_count, "output_tokens": meta.candidates_token_count}
                   if meta else {}),
            latency_s=time.monotonic() - start,
        )


class OpenRouterBackend:
    """Any model through OpenRouter's OpenAI-compatible API, pinned to one upstream host.

    OpenRouter load-balances a model across hosts (Anthropic, Vertex, Bedrock, Azure, ...), which
    would add serving-stack variance to every condition. `upstream` pins one host with fallbacks
    disabled, and the host that served each call is logged. Reasoning is set through OpenRouter's
    `reasoning.effort` ("none" turns it off where the model allows). The cost OpenRouter reports
    for each call is logged in usage["cost"], in USD.
    """

    provider = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"
    KEY_NAMES = ("OPENROUTER_API_KEY", "OPENROUTER_TOKEN")

    def __init__(self, name: str, model: str, upstream: str | None = None, max_tokens: int = 1024,
                 reasoning_effort: str | None = None, temperature: float | None = None,
                 seed: int | None = None, top_logprobs: int | None = None, max_retries: int = 8,
                 timeout: float = 120.0, budget_waits: int = 20) -> None:
        import openai

        from .env import env_value

        key = env_value(*self.KEY_NAMES)
        if not key:
            raise FatalBackendError(f"{name}: no OpenRouter key; set OPENROUTER_API_KEY or "
                                    "OPENROUTER_TOKEN in the environment or the project .env")
        self._sdk = openai
        self.client = openai.OpenAI(api_key=key, base_url=self.BASE_URL, max_retries=max_retries,
                                    timeout=timeout)
        self.name = name
        self.model = model
        self.budget_waits = budget_waits
        self.params: dict[str, Any] = {"max_completion_tokens": max_tokens}
        if temperature is not None:
            self.params["temperature"] = temperature
        if seed is not None:
            self.params["seed"] = seed
        if top_logprobs is not None:
            self.params.update(logprobs=True, top_logprobs=top_logprobs)
        self.extra_body: dict[str, Any] = {}
        if upstream:
            self.extra_body["provider"] = {"order": [upstream], "allow_fallbacks": False}
        if reasoning_effort:
            self.extra_body["reasoning"] = {"effort": reasoning_effort}

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "params": {**self.params, **self.extra_body}}

    def complete(self, task: Task) -> Completion:
        sdk = self._sdk
        start = time.monotonic()
        for wait in range(self.budget_waits + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": task.system},
                              {"role": "user", "content": task.user}],
                    extra_body=self.extra_body or None,
                    **self.params,
                )
                break
            except (sdk.AuthenticationError, sdk.NotFoundError) as e:
                raise FatalBackendError(f"{self.name}: {type(e).__name__}: {e.message}") from e
            except sdk.APIStatusError as e:
                # A rejected request (400, a 403 moderation flag) is one failed call: it is logged and
                # retried on resume. If every request is rejected, the run's error-rate limit stops it.
                if e.status_code != 402:
                    return _failed(f"{type(e).__name__}({e.status_code}): {e.message}", start)
                delay = _in_flight_wait(e)
                if delay is None or wait == self.budget_waits:
                    raise FatalBackendError(f"{self.name}: OpenRouter credits exhausted ({e.message})") from e
                # Concurrent requests reserve budget; wait for them to settle, then retry.
                time.sleep(delay + random.random() * 5)
            except sdk.APIConnectionError as e:
                return _failed(f"{type(e).__name__}: {e}", start)
        top = resp.model_extra or {}
        if not resp.choices:  # OpenRouter can answer 200 with an error object instead of choices
            return _failed(f"no choices: {top.get('error')}", start)
        choice = resp.choices[0]
        choice_extra = choice.model_extra or {}
        if choice.finish_reason == "error":
            return _failed(f"upstream error: {choice_extra.get('error')}", start)
        usage: dict[str, Any] = {}
        if resp.usage is not None:
            usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
            details = resp.usage.completion_tokens_details
            if details is not None and details.reasoning_tokens is not None:
                usage["reasoning_tokens"] = details.reasoning_tokens
            cost = (resp.usage.model_extra or {}).get("cost")
            if cost is not None:
                usage["cost"] = cost
            cached = getattr(resp.usage.prompt_tokens_details, "cached_tokens", None) \
                if resp.usage.prompt_tokens_details is not None else None
            if cached is not None:
                usage["cached_tokens"] = cached  # provider prompt caching could lower the noise floor
        logprobs = None
        if choice.logprobs is not None and choice.logprobs.content:
            logprobs = [{"token": t.token, "logprob": t.logprob,
                         "top": [[a.token, a.logprob] for a in (t.top_logprobs or [])]}
                        for t in choice.logprobs.content]
        return Completion(
            text=choice.message.content,
            served_model=resp.model,
            request_id=getattr(resp, "_request_id", None),
            response_id=resp.id,
            stop_reason=choice.finish_reason,
            usage=usage,
            logprobs=logprobs,
            latency_s=time.monotonic() - start,
            extra={"served_provider": top.get("provider"),
                   "native_finish_reason": choice_extra.get("native_finish_reason"),
                   "system_fingerprint": resp.system_fingerprint},
        )


def _in_flight_wait(e: Any) -> float | None:
    """Seconds to wait if a 402 means "in-flight requests reserve too much budget", else None.

    OpenRouter reserves each concurrent request's maximum cost against the balance; that limit
    clears as requests settle. Any other 402 (credits actually exhausted) returns None.
    """
    body = e.body if isinstance(e.body, dict) else {}
    error = body.get("error", body) if isinstance(body.get("error", body), dict) else body
    meta = error.get("metadata") or {}
    if meta.get("reason") != "in_flight_budget_exhausted":
        return None
    retry = (meta.get("headers") or {}).get("Retry-After") or e.response.headers.get("retry-after")
    try:
        return float(retry)
    except (TypeError, ValueError):
        return 120.0


PAID_KINDS = {"anthropic", "openai", "google", "openrouter"}


def make_backend(spec: dict[str, Any]):
    """Build a backend from a `[[backends]]` config table."""
    backend = _build_backend(spec)
    backend.concurrency = spec.get("concurrency")  # optional per-backend override of [run] concurrency
    return backend


def _build_backend(spec: dict[str, Any]):
    from .offline import ControlBackend, SimulatedBackend

    kind, name, params = spec["kind"], spec["name"], spec.get("params", {})
    if kind == "control":
        return ControlBackend(name)
    if kind == "simulated":
        return SimulatedBackend(name, **params)
    if kind == "anthropic":
        return AnthropicBackend(name, spec["model"], **params)
    if kind == "openai":
        return OpenAIBackend(name, spec["model"], **params)
    if kind == "google":
        return GoogleBackend(name, spec["model"], **params)
    if kind == "openrouter":
        return OpenRouterBackend(name, spec["model"], **params)
    raise ValueError(f"unknown backend kind: {kind}")
