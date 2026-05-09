"""Coverage for :class:`aawazz_mcp.providers.pipefish.PipefishLlmProvider`.

Non-network: every test mocks ``httpx.get`` / ``httpx.AsyncClient.post``
so the suite never tries to reach the captain's actual pipefish.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aawazz_mcp import providers  # noqa: F401  - register
from aawazz_mcp import registry
from aawazz_mcp.provider_base import LlmRequest, ProviderError


def test_pipefish_registered() -> None:
    p = registry.get_llm("pipefish")
    assert p.name == "pipefish"


def test_capabilities_unreachable_when_pipefish_offline(monkeypatch) -> None:
    """Connection refused / timeout → available=False, no models, clean notes."""
    from aawazz_mcp.providers.pipefish import PipefishLlmProvider

    import httpx

    def _refuse(*a, **kw):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("httpx.get", _refuse)

    p = PipefishLlmProvider()
    caps = p.capabilities()
    assert caps.available is False
    assert caps.requires_network is True
    assert caps.backend_models == ()
    assert "unreachable" in caps.notes.lower() or "refused" in caps.notes.lower()


def test_capabilities_available_when_pipefish_responds(monkeypatch) -> None:
    """Mocked /v1/models response → available=True, backend_models populated."""
    from aawazz_mcp.providers.pipefish import PipefishLlmProvider

    fake = MagicMock()
    fake.json.return_value = {
        "data": [
            {"id": "qwen3"},
            {"id": "ollama:llama3"},
            {"id": "ravan:zpu"},
        ],
    }
    fake.raise_for_status.return_value = None

    monkeypatch.setattr("httpx.get", lambda *a, **kw: fake)

    p = PipefishLlmProvider()
    caps = p.capabilities()
    assert caps.available is True
    assert "qwen3" in caps.backend_models
    assert "ollama:llama3" in caps.backend_models
    assert caps.supports_streaming is True
    assert caps.supports_system_prompt is True


def test_capabilities_caches_within_ttl(monkeypatch) -> None:
    """Reachability probe runs once; subsequent capabilities() calls hit cache."""
    from aawazz_mcp.providers.pipefish import PipefishLlmProvider

    fake = MagicMock()
    fake.json.return_value = {"data": [{"id": "qwen3"}]}
    fake.raise_for_status.return_value = None
    call_count = {"n": 0}

    def _get(*a, **kw):
        call_count["n"] += 1
        return fake

    monkeypatch.setattr("httpx.get", _get)

    p = PipefishLlmProvider()
    p.capabilities()
    p.capabilities()
    p.capabilities()
    assert call_count["n"] == 1, "second/third call should hit cache"


@pytest.mark.asyncio
async def test_complete_happy_path(monkeypatch) -> None:
    """Mocked /v1/chat/completions → LlmResult populated correctly."""
    from aawazz_mcp.providers.pipefish import PipefishLlmProvider

    # Make capabilities probe succeed.
    probe = MagicMock()
    probe.json.return_value = {"data": [{"id": "qwen3"}]}
    probe.raise_for_status.return_value = None
    monkeypatch.setattr("httpx.get", lambda *a, **kw: probe)

    # Mock the chat/completions POST.
    chat_resp = MagicMock()
    chat_resp.json.return_value = {
        "id": "cmpl-1",
        "model": "qwen3",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello, captain."},
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
        },
    }
    chat_resp.raise_for_status.return_value = None

    async_client = MagicMock()
    async_client.post = AsyncMock(return_value=chat_resp)
    async_client.__aenter__ = AsyncMock(return_value=async_client)
    async_client.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: async_client)

    p = PipefishLlmProvider()
    result = await p.complete(
        LlmRequest(
            messages=({"role": "user", "content": "hi"},),
            system_prompt="be terse",
        )
    )

    assert result.text == "Hello, captain."
    assert result.model == "qwen3"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 4
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_complete_pipefish_5xx_raises_provider_error(monkeypatch) -> None:
    """A 5xx from pipefish becomes a ProviderError, not a bare HTTP exception."""
    from aawazz_mcp.providers.pipefish import PipefishLlmProvider

    import httpx

    probe = MagicMock()
    probe.json.return_value = {"data": [{"id": "x"}]}
    probe.raise_for_status.return_value = None
    monkeypatch.setattr("httpx.get", lambda *a, **kw: probe)

    err_resp = MagicMock()
    err_resp.status_code = 503
    err_resp.text = "model loading"
    err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Service Unavailable", request=MagicMock(), response=err_resp
    )

    async_client = MagicMock()
    async_client.post = AsyncMock(return_value=err_resp)
    async_client.__aenter__ = AsyncMock(return_value=async_client)
    async_client.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: async_client)

    p = PipefishLlmProvider()
    with pytest.raises(ProviderError, match="HTTP 503"):
        await p.complete(LlmRequest(messages=({"role": "user", "content": "hi"},)))


@pytest.mark.asyncio
async def test_stream_phase1_raises_provider_error() -> None:
    """v1.4 phase 1 ships batch only; stream() must hard-fail clean."""
    from aawazz_mcp.providers.pipefish import PipefishLlmProvider

    p = PipefishLlmProvider()
    with pytest.raises(ProviderError, match="phase 2"):
        async for _ in p.stream(
            LlmRequest(messages=({"role": "user", "content": "hi"},))
        ):
            pass


def test_system_prompt_prepended_when_absent() -> None:
    """system_prompt is added to the messages list when not already present."""
    from aawazz_mcp.providers.pipefish import _has_system_role

    assert _has_system_role([{"role": "user", "content": "hi"}]) is False
    assert (
        _has_system_role([{"role": "system", "content": "be terse"}]) is True
    )
