"""Coverage for :mod:`aawazz_mcp.registry`.

Tests the registration / lookup / conflict / plugin-discovery surface in
isolation. The registry singleton is reset between tests via the
``isolated_registry`` fixture so test order doesn't matter.
"""

from __future__ import annotations

import pytest

from aawazz_mcp import registry
from aawazz_mcp.provider_base import (
    SttCapabilities,
    SttRequest,
    SttResult,
    TtsCapabilities,
    TtsRequest,
    TtsResult,
    VoiceCatalogEntry,
)


@pytest.fixture
def isolated_registry():
    """Save current registry state, yield a clean slate, restore after."""
    saved = (
        dict(registry._REGISTRY.tts),
        dict(registry._REGISTRY.stt),
        dict(registry._REGISTRY.post),
        dict(registry._REGISTRY.capture),
        dict(registry._REGISTRY.playback),
    )
    registry.reset()
    try:
        yield registry
    finally:
        registry._REGISTRY.tts.clear()
        registry._REGISTRY.tts.update(saved[0])
        registry._REGISTRY.stt.clear()
        registry._REGISTRY.stt.update(saved[1])
        registry._REGISTRY.post.clear()
        registry._REGISTRY.post.update(saved[2])
        registry._REGISTRY.capture.clear()
        registry._REGISTRY.capture.update(saved[3])
        registry._REGISTRY.playback.clear()
        registry._REGISTRY.playback.update(saved[4])


class _StubTts:
    name = "stub-tts"
    version = "0.1"

    def capabilities(self) -> TtsCapabilities:
        return TtsCapabilities(
            languages=frozenset({"en"}),
            voices=(VoiceCatalogEntry(id="stub-tts:default", language="en"),),
        )

    async def synthesize(self, request: TtsRequest) -> TtsResult:
        return TtsResult(
            audio_path="/tmp/stub.wav",
            sample_rate=44100,
            duration_s=1.0,
            latency_ms=10,
        )

    async def aclose(self) -> None:
        pass


class _StubStt:
    name = "stub-stt"
    version = "0.1"

    def capabilities(self) -> SttCapabilities:
        return SttCapabilities(
            languages=frozenset({"en"}),
            model_archs={"en": ("base",)},
        )

    async def transcribe(self, request: SttRequest) -> SttResult:
        return SttResult(
            text="ok",
            audio_duration_s=1.0,
            sample_rate=16000,
            latency_ms=5,
        )

    async def aclose(self) -> None:
        pass


def test_register_and_lookup(isolated_registry) -> None:
    registry.register_tts("stub-tts")(_StubTts)
    got = registry.get_tts("stub-tts")
    assert got.name == "stub-tts"
    assert got in registry.list_tts()


def test_register_stt_lookup(isolated_registry) -> None:
    registry.register_stt("stub-stt")(_StubStt)
    got = registry.get_stt("stub-stt")
    assert got.name == "stub-stt"


def test_register_conflict_raises(isolated_registry) -> None:
    registry.register_tts("stub-tts")(_StubTts)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_tts("stub-tts")(_StubTts)


def test_register_name_mismatch_raises(isolated_registry) -> None:
    class _MisnamedTts(_StubTts):
        name = "different"

    with pytest.raises(ValueError, match="declares .name="):
        registry.register_tts("stub-tts")(_MisnamedTts)


def test_get_unknown_raises(isolated_registry) -> None:
    with pytest.raises(KeyError, match="not registered"):
        registry.get_tts("does-not-exist")


def test_get_unknown_lists_available_in_message(isolated_registry) -> None:
    registry.register_tts("stub-tts")(_StubTts)
    with pytest.raises(KeyError, match=r"\['stub-tts'\]"):
        registry.get_tts("does-not-exist")


def test_reset_clears_all_kinds(isolated_registry) -> None:
    registry.register_tts("stub-tts")(_StubTts)
    registry.register_stt("stub-stt")(_StubStt)
    assert len(registry.list_tts()) == 1
    assert len(registry.list_stt()) == 1
    registry.reset()
    assert len(registry.list_tts()) == 0
    assert len(registry.list_stt()) == 0


def test_register_failed_instantiation_raises(isolated_registry) -> None:
    class _BadTts:
        name = "bad-tts"
        version = "0.1"

        def __init__(self) -> None:
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="failed to instantiate"):
        registry.register_tts("bad-tts")(_BadTts)


def test_discover_plugins_idempotent(isolated_registry, monkeypatch) -> None:
    """Second call without force=True is a no-op."""
    calls: list[str] = []

    def fake_entry_points(group: str):
        calls.append(group)
        return []

    monkeypatch.setattr(
        "importlib.metadata.entry_points", fake_entry_points
    )
    registry.discover_plugins()
    n = len(calls)
    registry.discover_plugins()  # no-op
    assert len(calls) == n

    registry.discover_plugins(force=True)  # rescans
    assert len(calls) > n


def test_discover_plugins_collision_skips(isolated_registry, monkeypatch) -> None:
    """Built-in name collisions warn-and-skip; built-in wins."""
    registry.register_tts("stub-tts")(_StubTts)

    class _OverrideStub(_StubTts):
        version = "999"

    class _FakeEntryPoint:
        name = "stub-tts"
        value = "fake.module:_OverrideStub"

        @staticmethod
        def load():
            return _OverrideStub

    def fake_entry_points(group: str):
        if group == "aawazz.tts_providers":
            return [_FakeEntryPoint()]
        return []

    monkeypatch.setattr(
        "importlib.metadata.entry_points", fake_entry_points
    )
    registry.discover_plugins(force=True)

    # Built-in retained; plugin skipped.
    assert registry.get_tts("stub-tts").version == "0.1"
