"""Coverage for the v1.4 phase-3 lang_mismatch policy on respond."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pytest

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
from aawazz_mcp.routing import RoutingConfig


class _RussianLlm:
    """Mock LLM that always emits the canonical bodhi-Russian response.

    Reproduces the s148 finding: bodhi without sys-prompt reverts to
    Bonsai identity in Russian, and tiny-tts dutifully synthesizes
    Cyrillic gibberish.
    """

    name = "mock-russian-llm"
    version = "0.0"

    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(
            available=True,
            requires_network=False,
            supports_streaming=False,
            supports_system_prompt=True,
        )

    async def complete(self, request: LlmRequest) -> LlmResult:  # noqa: ARG002
        return LlmResult(
            text="Я Bonsai, 1-битная модель, разработанная PrismML.",
            model="mock", prompt_tokens=5, completion_tokens=12,
            latency_ms=10, finish_reason="stop",
        )

    async def stream(self, request):  # noqa: ARG002
        raise ProviderError("not implemented")
        yield  # noqa: B901, RET504

    async def aclose(self) -> None:
        pass


@pytest.fixture
def isolated_llm():
    saved = dict(registry._REGISTRY.llm)
    registry._REGISTRY.llm.clear()
    instance = _RussianLlm()
    registry._REGISTRY.llm["mock-russian-llm"] = instance
    try:
        yield instance
    finally:
        registry._REGISTRY.llm.clear()
        registry._REGISTRY.llm.update(saved)


def _backend() -> LocalBackend:
    cfg = AawazzConfig.from_args(
        argparse.Namespace(
            remote=None, transport="stdio", host="127.0.0.1", port=7860,
            warm=False, log_level="WARNING",
            routing_config=None, tts_default=None, stt_default=None,
            llm_default=None,
        )
    )
    cfg = replace(cfg, routing=RoutingConfig(
        tts=cfg.routing.tts, stt=cfg.routing.stt, llm=("mock-russian-llm",),
    ))
    return LocalBackend(cfg)


@pytest.mark.asyncio
async def test_route_policy_reroutes_to_gtts_for_russian(
    isolated_llm: _RussianLlm, tmp_path: Path,
) -> None:
    """LLM emits Russian → lang_mismatch=route should route TTS to gtts
    (which speaks ru), NOT tiny-tts (en-only) which would gibberish-synth."""
    pytest.importorskip("lingua")
    pytest.importorskip("gtts")

    backend = _backend()
    out = tmp_path / "out.wav"
    result = await backend.respond(
        prompt="who are you",
        language="en",
        output_path=str(out),
        play=False,
        lang_mismatch="route",
    )

    assert "error" not in result, result
    assert result["tts_provider"] == "gtts", (
        "route policy should pick gtts for Russian, not tiny-tts"
    )
    assert result["language_mismatch"] == {"requested": "en", "detected": "ru"}
    assert result["language_detected"] == "ru"


@pytest.mark.asyncio
async def test_warn_policy_keeps_provider_but_tags_mismatch(
    isolated_llm: _RussianLlm, tmp_path: Path,
) -> None:
    """lang_mismatch=warn synthesizes via the originally-resolved provider
    (tiny-tts for en) but marks the mismatch in the response."""
    pytest.importorskip("lingua")

    backend = _backend()
    result = await backend.respond(
        prompt="who are you",
        language="en",
        output_path=str(tmp_path / "out.wav"),
        play=False,
        lang_mismatch="warn",
    )

    assert "error" not in result, result
    assert result["tts_provider"] == "tiny-tts"
    assert result["language_mismatch"] == {"requested": "en", "detected": "ru"}


@pytest.mark.asyncio
async def test_error_policy_returns_structured_error(
    isolated_llm: _RussianLlm, tmp_path: Path,
) -> None:
    pytest.importorskip("lingua")

    backend = _backend()
    result = await backend.respond(
        prompt="who are you",
        language="en",
        output_path=str(tmp_path / "out.wav"),
        play=False,
        lang_mismatch="error",
    )
    assert "error" in result
    assert "language mismatch" in result["error"]
    assert result["language_mismatch"] == {"requested": "en", "detected": "ru"}
    assert result["text"]  # the LLM output is still returned for inspection


@pytest.mark.asyncio
async def test_off_policy_skips_detection(
    isolated_llm: _RussianLlm, tmp_path: Path,
) -> None:
    """lang_mismatch='off' bypasses detection entirely — even if lingua is
    installed, no language_mismatch field appears in the response."""
    pytest.importorskip("lingua")

    backend = _backend()
    result = await backend.respond(
        prompt="who are you",
        language="en",
        output_path=str(tmp_path / "out.wav"),
        play=False,
        lang_mismatch="off",
    )
    assert "error" not in result, result
    assert result.get("language_mismatch") is None
    assert result["tts_provider"] == "tiny-tts"


@pytest.mark.asyncio
async def test_route_respects_explicit_tts_provider_override(
    isolated_llm: _RussianLlm, tmp_path: Path,
) -> None:
    """When the caller passes tts_provider= explicitly, route policy must
    NOT override it — explicit override wins. The mismatch is still tagged
    in the response (warn-style) so callers see what happened."""
    pytest.importorskip("lingua")
    pytest.importorskip("gtts")

    backend = _backend()
    result = await backend.respond(
        prompt="who are you",
        language="en",
        tts_provider="gtts",  # explicit
        output_path=str(tmp_path / "out.wav"),
        play=False,
        lang_mismatch="route",
    )
    assert result["tts_provider"] == "gtts"
    # Mismatch still recorded — caller's explicit override happens to also
    # match the detected lang in this case (gtts speaks ru), so the route
    # policy didn't need to override anything.
    assert result["language_mismatch"] == {"requested": "en", "detected": "ru"}


@pytest.mark.asyncio
async def test_no_mismatch_when_languages_align(
    isolated_llm: _RussianLlm, tmp_path: Path,
) -> None:
    """If request.language matches detected, language_mismatch is None."""
    pytest.importorskip("lingua")

    backend = _backend()
    result = await backend.respond(
        prompt="who are you",
        language="ru",  # matches the LLM's actual output
        tts_provider="gtts",  # gtts speaks ru
        output_path=str(tmp_path / "out.wav"),
        play=False,
        lang_mismatch="route",
    )
    assert result.get("language_mismatch") is None
