"""End-to-end coverage for the v1.4 ``respond`` flow at LocalBackend layer.

Mocks the LLM provider via the registry so we don't reach pipefish; the
TTS and post_process paths use the real v1.3 plumbing.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pytest
import soundfile as sf

from aawazz_mcp import providers  # noqa: F401  - register
from aawazz_mcp import registry
from aawazz_mcp.backends.local import LocalBackend
from aawazz_mcp.config import AawazzConfig
from aawazz_mcp.provider_base import (
    LlmCapabilities,
    LlmRequest,
    LlmResult,
    ProviderError,
)


class _MockLlm:
    """Deterministic LLM stub. Captures the last LlmRequest for assertions."""

    name = "mock-llm"
    version = "0.0"

    def __init__(self) -> None:
        self.last_request: LlmRequest | None = None
        self.reply_text = "Hello world."

    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(
            available=True,
            requires_network=False,
            supports_streaming=False,
            supports_system_prompt=True,
        )

    async def complete(self, request: LlmRequest) -> LlmResult:
        self.last_request = request
        return LlmResult(
            text=self.reply_text,
            model="mock-1",
            prompt_tokens=5,
            completion_tokens=3,
            latency_ms=12,
            finish_reason="stop",
        )

    async def stream(self, request):  # noqa: ARG002
        raise ProviderError("not implemented")
        yield  # noqa: B901, RET504

    async def aclose(self) -> None:
        pass


@pytest.fixture
def isolated_llm():
    """Save/restore the real llm registry; install a clean MockLlm during tests."""
    saved = dict(registry._REGISTRY.llm)
    registry._REGISTRY.llm.clear()

    instance = _MockLlm()
    registry._REGISTRY.llm["mock-llm"] = instance
    try:
        yield instance
    finally:
        registry._REGISTRY.llm.clear()
        registry._REGISTRY.llm.update(saved)


def _backend_with_llm_chain(llm_chain: tuple[str, ...]) -> LocalBackend:
    """LocalBackend with a routing config pointing at the named llm chain."""
    cfg = AawazzConfig.from_args(
        argparse.Namespace(
            remote=None, transport="stdio", host="127.0.0.1", port=7860,
            warm=False, log_level="WARNING",
            routing_config=None, tts_default=None, stt_default=None,
            llm_default=None,
        )
    )
    # Override the routing chain post-construction via a fresh RoutingConfig
    # rebuild — Config is frozen so we replace the whole field.
    from dataclasses import replace
    from aawazz_mcp.routing import RoutingConfig
    new_routing = RoutingConfig(
        tts=cfg.routing.tts, stt=cfg.routing.stt, llm=llm_chain,
    )
    cfg = replace(cfg, routing=new_routing)
    return LocalBackend(cfg)


@pytest.mark.asyncio
async def test_respond_full_path_to_speak(
    isolated_llm: _MockLlm, tmp_path: Path,
) -> None:
    """LLM produces text → speak() turns it into a WAV; response carries
    both LLM and TTS metadata."""
    backend = _backend_with_llm_chain(("mock-llm",))
    out = tmp_path / "respond.wav"

    result = await backend.respond(
        prompt="Say hi",
        system_prompt="be terse",
        output_path=str(out),
        play=False,
    )

    assert "error" not in result, result
    assert result["text"] == "Hello world."
    assert result["llm_provider"] == "mock-llm"
    assert result["tts_provider"] == "tiny-tts"
    assert result["llm_latency_ms"] == 12
    assert result["tts_latency_ms"] >= 0
    assert result["total_latency_ms"] == result["llm_latency_ms"] + result["tts_latency_ms"]
    assert result["prompt_tokens"] == 5
    assert result["completion_tokens"] == 3
    assert result["finish_reason"] == "stop"
    assert out.exists()

    # WAV is real RIFF (sanity).
    info = sf.info(str(out))
    assert info.duration > 0


@pytest.mark.asyncio
async def test_respond_system_prompt_passes_through(
    isolated_llm: _MockLlm, tmp_path: Path,
) -> None:
    backend = _backend_with_llm_chain(("mock-llm",))
    await backend.respond(
        prompt="hi",
        system_prompt="be terse",
        output_path=str(tmp_path / "out.wav"),
        play=False,
    )
    req = isolated_llm.last_request
    assert req is not None
    assert req.system_prompt == "be terse"
    assert req.messages[0] == {"role": "user", "content": "hi"}


@pytest.mark.asyncio
async def test_respond_messages_form_for_multiturn(
    isolated_llm: _MockLlm, tmp_path: Path,
) -> None:
    backend = _backend_with_llm_chain(("mock-llm",))
    await backend.respond(
        messages=[
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "earlier reply"},
            {"role": "user", "content": "follow-up"},
        ],
        output_path=str(tmp_path / "out.wav"),
        play=False,
    )
    req = isolated_llm.last_request
    assert len(req.messages) == 3
    assert req.messages[-1]["content"] == "follow-up"


@pytest.mark.asyncio
async def test_respond_requires_prompt_or_messages(
    isolated_llm: _MockLlm, tmp_path: Path,
) -> None:
    backend = _backend_with_llm_chain(("mock-llm",))
    result = await backend.respond(output_path=str(tmp_path / "out.wav"))
    assert "error" in result
    assert "prompt or messages" in result["error"]


@pytest.mark.asyncio
async def test_respond_rejects_both_prompt_and_messages(
    isolated_llm: _MockLlm, tmp_path: Path,
) -> None:
    backend = _backend_with_llm_chain(("mock-llm",))
    result = await backend.respond(
        prompt="hi",
        messages=[{"role": "user", "content": "hi"}],
        output_path=str(tmp_path / "out.wav"),
    )
    assert "error" in result
    assert "not both" in result["error"]


@pytest.mark.asyncio
async def test_respond_no_llm_available_returns_error(tmp_path: Path) -> None:
    """Empty llm chain → no provider available → respond returns clean error."""
    backend = _backend_with_llm_chain(())
    result = await backend.respond(
        prompt="hi", output_path=str(tmp_path / "out.wav")
    )
    assert "error" in result
    assert "no llm provider" in result["error"]


@pytest.mark.asyncio
async def test_respond_with_post_process_chain(
    isolated_llm: _MockLlm, tmp_path: Path,
) -> None:
    """The TTS-side post_process chain is applied after LLM produces text."""
    isolated_llm.reply_text = "Hello world."
    backend = _backend_with_llm_chain(("mock-llm",))

    result = await backend.respond(
        prompt="say hi",
        post_process=["dsp:DEEP", "gain:auto"],
        output_path=str(tmp_path / "out.wav"),
        play=False,
    )
    assert "error" not in result
    assert result["post_process_chain"] == ["dsp:DEEP", "gain:auto"]


@pytest.mark.asyncio
async def test_respond_empty_llm_text_errors_out(
    isolated_llm: _MockLlm, tmp_path: Path,
) -> None:
    """Don't synthesize empty audio — bail with a clear error."""
    isolated_llm.reply_text = ""
    backend = _backend_with_llm_chain(("mock-llm",))
    result = await backend.respond(
        prompt="hi", output_path=str(tmp_path / "out.wav")
    )
    assert "error" in result
    assert "empty text" in result["error"]
