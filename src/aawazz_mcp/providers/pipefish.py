"""Built-in Pipefish LLM provider — HTTP client to seahorse pipefish.

Phase 1 of v1.4 (SPEC §3). Single concrete :class:`LlmProvider` for
v1.4.0. Talks to the captain's pipefish HTTP server (default
``http://127.0.0.1:11450/v1``) via OpenAI-compat ``/v1/chat/completions``.
All backend routing — Local llama.cpp-FFI, Ollama, Gemini, Ravan,
LlamaCppBackend — happens upstream inside pipefish per the captain's
seahorse-first directive (memory ``seahorse_pipefish_canonical``).

Reachability semantics
----------------------
``capabilities()`` does a one-shot ``GET /v1/models`` with a 1 s timeout
on first call. Result caches for 30 s; subsequent calls re-probe lazily.

* Success → ``available=True``, ``backend_models=(...)`` from the response.
* ConnectionRefused / DNS / 5xx → ``available=False``, ``backend_models=()``;
  the router skips us so ``respond`` errors cleanly instead of hanging.

Streaming arrives in v1.4 phase 2; phase 1 ``stream()`` raises
:class:`ProviderError`.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from aawazz_mcp.provider_base import (
    LlmCapabilities,
    LlmRequest,
    LlmResult,
    ProviderError,
)
from aawazz_mcp.registry import register_llm

log = logging.getLogger("aawazz_mcp.providers.pipefish")


_DEFAULT_URL = "http://127.0.0.1:11450/v1"
_REACHABILITY_CACHE_TTL_S = 30.0
_REACHABILITY_PROBE_TIMEOUT_S = 1.0


def _httpx_version() -> str:
    try:
        from importlib.metadata import version
        return version("httpx")
    except Exception:
        return "unknown"


def _probe_httpx() -> bool:
    try:
        import httpx  # noqa: F401, PLC0415
        return True
    except Exception:
        return False


@register_llm("pipefish")
class PipefishLlmProvider:
    name = "pipefish"

    def __init__(self) -> None:
        self._available = _probe_httpx()
        self._version = _httpx_version() if self._available else "not-installed"
        self._base_url = (
            os.environ.get("AAWAZZ_PIPEFISH_URL") or _DEFAULT_URL
        ).rstrip("/")
        self._default_model = os.environ.get("AAWAZZ_PIPEFISH_MODEL") or None
        self._token = os.environ.get("AAWAZZ_PIPEFISH_TOKEN") or None
        self._cache_expires: float = 0.0
        self._cached_caps: LlmCapabilities | None = None

    @property
    def version(self) -> str:
        return self._version

    @property
    def base_url(self) -> str:
        return self._base_url

    def capabilities(self) -> LlmCapabilities:
        if not self._available:
            return LlmCapabilities(
                available=False,
                requires_network=False,
                supports_streaming=False,
                supports_system_prompt=False,
                backend_models=(),
                notes=(
                    "httpx not installed; install via "
                    "``pip install aawazz-mcp[llm]``"
                ),
            )

        now = time.time()
        if self._cached_caps is not None and now < self._cache_expires:
            return self._cached_caps

        caps = self._probe()
        self._cached_caps = caps
        self._cache_expires = now + _REACHABILITY_CACHE_TTL_S
        return caps

    def _probe(self) -> LlmCapabilities:
        """One-shot synchronous reachability probe. Results cached by caller."""
        import httpx  # noqa: PLC0415

        url = f"{self._base_url}/models"
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            resp = httpx.get(
                url, headers=headers, timeout=_REACHABILITY_PROBE_TIMEOUT_S
            )
            resp.raise_for_status()
            doc = resp.json()
        except Exception as e:  # noqa: BLE001 — any error → unavailable
            log.debug("pipefish reachability probe failed for %s: %s", url, e)
            return LlmCapabilities(
                available=False,
                requires_network=True,
                supports_streaming=False,
                supports_system_prompt=False,
                backend_models=(),
                notes=(
                    f"pipefish unreachable at {self._base_url}: {e}. "
                    "Start pipefish or set AAWAZZ_PIPEFISH_URL."
                ),
            )

        # OpenAI-compat ``/v1/models`` response: {"data": [{"id": ...}, ...]}
        models: list[str] = []
        if isinstance(doc, dict) and isinstance(doc.get("data"), list):
            for entry in doc["data"]:
                if isinstance(entry, dict) and "id" in entry:
                    models.append(str(entry["id"]))

        return LlmCapabilities(
            available=True,
            requires_network=True,
            supports_streaming=True,
            supports_system_prompt=True,
            backend_models=tuple(models),
            notes=f"endpoint={self._base_url}",
        )

    async def complete(self, request: LlmRequest) -> LlmResult:
        if not self._available:
            msg = (
                "httpx not installed; install via "
                "``pip install aawazz-mcp[llm]``"
            )
            raise ProviderError(msg)

        # Build messages with optional system prompt prepended.
        messages = list(request.messages)
        if request.system_prompt and not _has_system_role(messages):
            messages.insert(
                0, {"role": "system", "content": request.system_prompt}
            )

        body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": int(request.max_tokens),
            "temperature": float(request.temperature),
            "top_p": float(request.top_p),
            "stream": False,
        }
        if request.stop:
            body["stop"] = list(request.stop)
        model = request.model or self._default_model
        if model:
            body["model"] = model

        # Provider-specific kwargs (e.g. extra={"headers": ...}).
        extra_headers = (request.extra or {}).get("headers") or {}

        import httpx  # noqa: PLC0415

        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        headers.update(extra_headers)

        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=request.timeout_s) as client:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                doc = resp.json()
        except httpx.TimeoutException as e:
            msg = (
                f"pipefish timeout after {request.timeout_s}s at {url}: {e}"
            )
            raise ProviderError(msg, hint="raise timeout_s or check pipefish load") from e
        except httpx.HTTPStatusError as e:
            msg = (
                f"pipefish returned HTTP {e.response.status_code}: "
                f"{e.response.text[:200]}"
            )
            raise ProviderError(msg) from e
        except Exception as e:  # noqa: BLE001
            msg = f"pipefish request failed: {e}"
            raise ProviderError(msg, hint=f"endpoint: {url}") from e

        latency_ms = int((time.time() - t0) * 1000)

        return _parse_chat_response(doc, latency_ms)

    async def stream(self, request: LlmRequest):  # noqa: ARG002
        """Async generator stub — phase 2 wires SSE. The lone ``yield``
        below is unreachable; it makes Python treat this as an async
        generator so callers can use ``async for`` and receive the
        ProviderError on the first ``__anext__``."""
        msg = "PipefishLlmProvider streaming arrives in v1.4 phase 2"
        raise ProviderError(msg)
        yield  # noqa: B901, RET504  - unreachable; marks as async generator

    async def aclose(self) -> None:
        pass


def _has_system_role(messages: list[dict[str, str]]) -> bool:
    return any(m.get("role") == "system" for m in messages)


def _parse_chat_response(doc: Any, latency_ms: int) -> LlmResult:
    """Parse an OpenAI-compat ``chat/completions`` response into LlmResult."""
    if not isinstance(doc, dict):
        msg = f"pipefish returned non-dict response: {type(doc).__name__}"
        raise ProviderError(msg)

    choices = doc.get("choices")
    if not isinstance(choices, list) or not choices:
        msg = "pipefish response missing 'choices'"
        raise ProviderError(msg)

    first = choices[0]
    if not isinstance(first, dict):
        msg = "pipefish response 'choices[0]' is not an object"
        raise ProviderError(msg)

    message = first.get("message") or {}
    text = (message.get("content") or "").strip() if isinstance(message, dict) else ""
    finish_reason = str(first.get("finish_reason") or "stop")

    usage = doc.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0
    completion_tokens = (
        int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0
    )

    model = str(doc.get("model") or "")

    return LlmResult(
        text=text,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )
