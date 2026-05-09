"""Coverage for the v1.4 LLM routing layer (RoutingConfig.llm + Router.resolve_llm)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aawazz_mcp import providers  # noqa: F401  - register
from aawazz_mcp.provider_base import (
    LlmCapabilities,
    LlmRequest,
    LlmResult,
    ProviderError,
)
from aawazz_mcp.routing import RoutingConfig, Router


# ── RoutingConfig.llm shape ─────────────────────────────────────────────────


def test_builtin_default_includes_pipefish() -> None:
    cfg = RoutingConfig.builtin_default()
    assert cfg.llm == ("pipefish",)


def test_load_with_toml_llm_section(tmp_path: Path) -> None:
    f = tmp_path / "aawazz.toml"
    f.write_text(
        """
        [tts.routing]
        en = ["tiny-tts"]

        [llm.routing]
        default = ["my-cloud", "pipefish"]
        """
    )
    cfg = RoutingConfig.load(file_path=f, env={})
    assert cfg.llm == ("my-cloud", "pipefish")


def test_load_llm_default_string_form(tmp_path: Path) -> None:
    """TOML scalar string for llm.routing.default coerces to a 1-tuple."""
    f = tmp_path / "aawazz.toml"
    f.write_text(
        """
        [llm.routing]
        default = "pipefish"
        """
    )
    cfg = RoutingConfig.load(file_path=f, env={})
    assert cfg.llm == ("pipefish",)


def test_load_env_var_overrides_default(tmp_path: Path) -> None:
    cfg = RoutingConfig.load(
        file_path=None,
        env={"AAWAZZ_LLM_PROVIDER": "my-direct-anthropic"},
    )
    assert cfg.llm == ("my-direct-anthropic",)


def test_load_cli_override_wins_over_env(tmp_path: Path) -> None:
    cfg = RoutingConfig.load(
        file_path=None,
        llm_default_override="cli-winner",
        env={"AAWAZZ_LLM_PROVIDER": "env-loser"},
    )
    assert cfg.llm == ("cli-winner",)


# ── Router.resolve_llm ──────────────────────────────────────────────────────


class _AvailableLlm:
    name = "stub-llm"
    version = "0.1"

    def capabilities(self):
        return LlmCapabilities(
            available=True,
            requires_network=False,
            supports_streaming=False,
            supports_system_prompt=True,
        )

    async def complete(self, request: LlmRequest) -> LlmResult:
        return LlmResult(
            text="ok", model="stub", prompt_tokens=1, completion_tokens=1,
            latency_ms=1, finish_reason="stop",
        )

    async def stream(self, request):  # noqa: ARG002
        raise ProviderError("not implemented")

    async def aclose(self) -> None:
        pass


class _UnavailableLlm:
    name = "down-llm"
    version = "0.1"

    def capabilities(self):
        return LlmCapabilities(
            available=False,
            requires_network=True,
            supports_streaming=False,
            supports_system_prompt=False,
            notes="endpoint unreachable",
        )

    async def complete(self, request):  # noqa: ARG002
        raise ProviderError("offline")

    async def stream(self, request):  # noqa: ARG002
        raise ProviderError("offline")

    async def aclose(self) -> None:
        pass


@pytest.fixture
def isolated_llm_registry():
    """Save current llm registry, swap to clean for the test, restore after."""
    from aawazz_mcp import registry

    saved = dict(registry._REGISTRY.llm)
    registry._REGISTRY.llm.clear()
    try:
        yield registry
    finally:
        registry._REGISTRY.llm.clear()
        registry._REGISTRY.llm.update(saved)


def test_resolve_llm_picks_first_available(isolated_llm_registry) -> None:
    """Chain ['down-llm', 'stub-llm']: down-llm capabilities.available=False
    so resolve_llm skips to stub-llm."""
    isolated_llm_registry.register_llm("down-llm")(_UnavailableLlm)
    isolated_llm_registry.register_llm("stub-llm")(_AvailableLlm)

    cfg = RoutingConfig(
        tts={"en": ("tiny-tts",), "default": ("gtts",)},
        stt={"default": ("moonshine",)},
        llm=("down-llm", "stub-llm"),
    )
    r = Router(cfg)
    p = r.resolve_llm()
    assert p.name == "stub-llm"


def test_resolve_llm_no_available_provider_raises(isolated_llm_registry) -> None:
    isolated_llm_registry.register_llm("down-llm")(_UnavailableLlm)

    cfg = RoutingConfig(
        tts={"default": ("gtts",)},
        stt={"default": ("moonshine",)},
        llm=("down-llm",),
    )
    r = Router(cfg)
    with pytest.raises(ProviderError, match="no llm provider"):
        r.resolve_llm()


def test_resolve_llm_override_unknown_raises(isolated_llm_registry) -> None:
    isolated_llm_registry.register_llm("stub-llm")(_AvailableLlm)
    cfg = RoutingConfig(
        tts={"default": ("gtts",)}, stt={"default": ("moonshine",)},
        llm=("stub-llm",),
    )
    r = Router(cfg)
    with pytest.raises(ProviderError, match="not registered"):
        r.resolve_llm(override="not-a-real-llm")


def test_resolve_llm_override_unavailable_raises(isolated_llm_registry) -> None:
    """Per-call override hard-fails when the named provider is registered
    but unavailable — no silent fallback to chain."""
    isolated_llm_registry.register_llm("down-llm")(_UnavailableLlm)
    isolated_llm_registry.register_llm("stub-llm")(_AvailableLlm)

    cfg = RoutingConfig(
        tts={"default": ("gtts",)}, stt={"default": ("moonshine",)},
        llm=("stub-llm",),  # would resolve to stub-llm if we fell back
    )
    r = Router(cfg)
    with pytest.raises(ProviderError, match="unavailable"):
        r.resolve_llm(override="down-llm")
