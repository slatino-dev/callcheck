"""HTTP client for OpenAI-compatible model endpoints (vLLM, mockserver, etc.).

Wraps httpx with retry logic, timeout configuration, and response
normalisation so that the rest of callcheck never needs to deal with raw
HTTP concerns.

Configuration is picked from the environment when not supplied explicitly:

  OPENAI_BASE_URL  (or LLM_BASE_URL)  — e.g. http://127.0.0.1:9876
  OPENAI_API_KEY                       — passed as Bearer token; use "none" for local

Usage::

    from callcheck.client import ModelClient
    client = ModelClient(base_url="http://127.0.0.1:9876", api_key="none")
    response = client.complete(
        messages=[{"role": "user", "content": "What's the weather in Dublin?"}],
        tools=[...],
        model="mock-model",
    )
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 60.0  # seconds
_DEFAULT_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_BASE_DELAY = 0.5  # seconds, doubled on each attempt


def _get_base_url() -> str:
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or "http://127.0.0.1:9876"
    )


def _get_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or "none"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalise_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Ensure the response has the standard OpenAI chat.completion shape.

    vLLM and other compatible servers are generally well-behaved, but some
    edge cases (missing ``choices``, missing ``message``) are patched here so
    downstream code always has a consistent dict to work with.
    """
    raw.setdefault("choices", [])
    raw.setdefault("usage", {})
    for choice in raw["choices"]:
        choice.setdefault("index", 0)
        choice.setdefault("finish_reason", "stop")
        msg = choice.setdefault("message", {})
        msg.setdefault("role", "assistant")
        msg.setdefault("content", None)
        msg.setdefault("tool_calls", None)
    return raw


# ---------------------------------------------------------------------------
# Synchronous client
# ---------------------------------------------------------------------------


class ModelClient:
    """Synchronous OpenAI-compatible client.

    Parameters
    ----------
    base_url:
        Root URL of the API server (no trailing slash).  Defaults to
        ``OPENAI_BASE_URL`` / ``LLM_BASE_URL`` env vars, then
        ``http://127.0.0.1:9876``.
    api_key:
        Bearer token.  Defaults to ``OPENAI_API_KEY`` env var, then
        ``"none"`` (suitable for local/unauthenticated servers).
    timeout:
        Request timeout in seconds.
    max_retries:
        Number of retries on 429 / 5xx responses.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self.base_url = (base_url or _get_base_url()).rstrip("/")
        self.api_key = api_key or _get_api_key()
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        model: str = "default",
        temperature: float = 0.0,
        **extra: Any,
    ) -> dict[str, Any]:
        """Send a chat-completion request and return the normalised response.

        Parameters
        ----------
        messages:
            The conversation so far (OpenAI messages array).
        tools:
            OpenAI-format tool definitions.  When ``None`` / empty no tools
            are sent and the model replies in plain text.
        tool_choice:
            ``"auto"``, ``"none"``, or a specific ``{"type": "function", ...}``
            dict.  Ignored when ``tools`` is empty.
        model:
            Model identifier.  Passed as-is to the endpoint.
        temperature:
            Sampling temperature.  Defaults to 0 for deterministic evaluation.
        **extra:
            Any additional fields forwarded verbatim to the request body
            (e.g. ``max_tokens``).
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **extra,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice

        url = f"{self.base_url}/v1/chat/completions"
        delay = _RETRY_BASE_DELAY

        for attempt in range(self.max_retries + 1):
            try:
                resp = httpx.post(
                    url,
                    json=body,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            except httpx.TransportError:
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

            if resp.status_code in _RETRY_STATUSES and attempt < self.max_retries:
                # Respect Retry-After if present
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                time.sleep(wait)
                delay *= 2
                continue

            resp.raise_for_status()
            return _normalise_response(resp.json())

        # Should be unreachable, but satisfy type checker
        raise RuntimeError("Exhausted retries without a successful response")


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncModelClient:
    """Async variant using :class:`httpx.AsyncClient`.

    API mirrors :class:`ModelClient`.  Use with ``await client.acomplete(...)``
    inside an asyncio event loop.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self.base_url = (base_url or _get_base_url()).rstrip("/")
        self.api_key = api_key or _get_api_key()
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        model: str = "default",
        temperature: float = 0.0,
        **extra: Any,
    ) -> dict[str, Any]:
        """Async equivalent of :meth:`ModelClient.complete`."""
        import asyncio

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **extra,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice

        url = f"{self.base_url}/v1/chat/completions"
        delay = _RETRY_BASE_DELAY

        async with httpx.AsyncClient(timeout=self.timeout) as http_client:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await http_client.post(
                        url, json=body, headers=self._headers()
                    )
                except httpx.TransportError:
                    if attempt < self.max_retries:
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    raise

                if resp.status_code in _RETRY_STATUSES and attempt < self.max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else delay
                    await asyncio.sleep(wait)
                    delay *= 2
                    continue

                resp.raise_for_status()
                return _normalise_response(resp.json())

        raise RuntimeError("Exhausted retries without a successful response")
