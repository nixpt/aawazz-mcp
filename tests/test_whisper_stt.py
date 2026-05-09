"""Smoke coverage for :mod:`aawazz_mcp.models.whisper_stt`.

We don't load the actual Whisper model in unit tests — that requires the
``[multilingual]`` extras (transformers + torch) and pulls weights over the
network. Instead we exercise the lazy registry surface.
"""

from __future__ import annotations

import pytest


def test_supported_languages_includes_nepali() -> None:
    from aawazz_mcp.models.whisper_stt import supported_languages

    langs = supported_languages()
    assert "ne" in langs


def test_loader_initial_state() -> None:
    from aawazz_mcp.models.whisper_stt import WhisperSttLoader

    loader = WhisperSttLoader()
    assert loader.loaded is False


@pytest.mark.asyncio
async def test_load_unknown_language_raises_before_import() -> None:
    """An unsupported language must raise *before* attempting to import
    ``transformers`` — otherwise users without the multilingual extras get an
    ImportError instead of a clean validation error."""
    from aawazz_mcp.models.whisper_stt import WhisperSttLoader

    loader = WhisperSttLoader()
    with pytest.raises(ValueError, match="no Whisper model registered"):
        await loader.load("xx")
