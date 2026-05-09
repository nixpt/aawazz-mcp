"""Coverage for :mod:`aawazz_mcp.audio.lang_detect`."""

from __future__ import annotations

import pytest

from aawazz_mcp.audio.lang_detect import detect_language, is_available


def test_lingua_available_in_dev_env() -> None:
    """The dev install includes [langdetect]; lingua should be importable."""
    pytest.importorskip("lingua")
    assert is_available()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world.", "en"),
        ("Я Bonsai, 1-битная модель.", "ru"),
        ("Hola mundo, esto es una prueba.", "es"),
        ("Bonjour le monde, comment ça va?", "fr"),
        ("こんにちは、これは日本語のテストです。", "ja"),
        ("你好，这是一个中文测试。", "zh"),
        ("नमस्कार, यो परीक्षण हो।", "hi"),  # Devanagari, lingua tags as hi
        ("Witaj świecie", "pl"),
    ],
)
def test_detects_common_languages(text: str, expected: str) -> None:
    pytest.importorskip("lingua")
    assert detect_language(text) == expected


def test_returns_none_on_empty_text() -> None:
    assert detect_language("") is None


def test_min_chars_skip_short_text() -> None:
    """min_chars guards against unreliable short-text detection."""
    pytest.importorskip("lingua")
    assert detect_language("hi", min_chars=4) is None
    assert detect_language("a", min_chars=4) is None


def test_no_op_when_lingua_missing(monkeypatch) -> None:
    """If lingua isn't installed, detect_language returns None silently —
    the lang_mismatch policy then falls through to 'off'."""
    import aawazz_mcp.audio.lang_detect as ld

    monkeypatch.setattr(ld, "_LINGUA_AVAILABLE", False)
    monkeypatch.setattr(ld, "_DETECTOR", None)
    assert ld.detect_language("Hello world.") is None
    assert ld.is_available() is False
