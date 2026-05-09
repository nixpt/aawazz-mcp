"""voices_list v1.3 response shape — providers + routing keys.

Phase 1 introduces the v2 keys (``providers``, ``routing``) while keeping the
v1 keys (``tts``, ``stt``, ``capabilities``) as flattened alias views for
back-compat. Both must remain present until v2.0 is cut.
"""

from __future__ import annotations

import argparse
import asyncio

import pytest

from aawazz_mcp.config import AawazzConfig
from aawazz_mcp.dispatcher import Dispatcher


@pytest.fixture
def dispatcher() -> Dispatcher:
    cfg = AawazzConfig.from_args(
        argparse.Namespace(
            remote=None,
            transport="stdio",
            host="127.0.0.1",
            port=7860,
            warm=False,
            log_level="WARNING",
        )
    )
    return Dispatcher(cfg)


def test_voices_list_has_v2_top_level_keys(dispatcher: Dispatcher) -> None:
    result = asyncio.run(dispatcher.voices_list())
    for key in ("providers", "routing", "capabilities", "tts", "stt"):
        assert key in result, f"missing top-level key {key!r}"


def test_providers_block_lists_built_ins(dispatcher: Dispatcher) -> None:
    result = asyncio.run(dispatcher.voices_list())
    tts_names = {p["name"] for p in result["providers"]["tts"]}
    stt_names = {p["name"] for p in result["providers"]["stt"]}
    assert "tiny-tts" in tts_names
    assert "gtts" in tts_names
    assert "moonshine" in stt_names
    assert "whisper" in stt_names


def test_provider_entry_shape(dispatcher: Dispatcher) -> None:
    result = asyncio.run(dispatcher.voices_list())
    tiny = next(p for p in result["providers"]["tts"] if p["name"] == "tiny-tts")
    assert tiny["version"]
    assert tiny["languages"] == ["en"]
    assert tiny["accepts_dsp_profiles"] is True
    assert tiny["voices"][0]["id"].startswith("tiny-tts:")


def test_routing_phase1_hardcoded(dispatcher: Dispatcher) -> None:
    """Phase 1 emits the static routing chain LocalBackend uses today."""
    result = asyncio.run(dispatcher.voices_list())
    assert result["routing"]["tts"] == {"en": ["tiny-tts"], "default": ["gtts"]}
    assert result["routing"]["stt"] == {"ne": ["whisper"], "default": ["moonshine"]}


def test_v1_alias_view_preserved(dispatcher: Dispatcher) -> None:
    """v1.0 / v1.2 callers must keep working — DSP voices flat, lang_models
    union of all STT providers."""
    result = asyncio.run(dispatcher.voices_list())

    # tts.voices: 8 DSP profiles (MALE + 7 effects).
    voice_ids = {v["id"] for v in result["tts"]["voices"]}
    assert {"MALE", "DEEP", "BRIGHT", "SOFT", "GRAVEL", "ROBOT", "ECHO", "WIDE"} <= voice_ids

    # stt.languages: union covers Moonshine + Whisper langs.
    langs = set(result["stt"]["languages"])
    assert {"en", "es", "ne"} <= langs

    # lang_models["ne"] surfaces the whisper-small marker.
    assert "whisper-small" in result["stt"]["lang_models"]["ne"]


def test_capabilities_block_present(dispatcher: Dispatcher) -> None:
    result = asyncio.run(dispatcher.voices_list())
    cap = result["capabilities"]
    assert "listen" in cap
    assert "play" in cap
    assert "backend_mode" in cap
    assert cap["backend_mode"] == "local"
