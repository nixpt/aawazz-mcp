"""Coverage for :mod:`aawazz_mcp.routing` — RoutingConfig + Router.

Focused unit tests for the layering logic and the chain-resolution
algorithm. End-to-end smoke through LocalBackend lives in
``test_smoke_local.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aawazz_mcp import providers  # noqa: F401  - register built-ins
from aawazz_mcp.provider_base import ProviderError
from aawazz_mcp.routing import RoutingConfig, Router


# ── RoutingConfig ───────────────────────────────────────────────────────────


def test_builtin_default_matches_v1_2_behavior() -> None:
    cfg = RoutingConfig.builtin_default()
    assert cfg.tts == {"en": ("tiny-tts",), "default": ("gtts",)}
    assert cfg.stt == {"ne": ("whisper",), "default": ("moonshine",)}


def test_load_no_file_no_env_no_cli_returns_builtin(tmp_path: Path) -> None:
    """Built-in defaults apply when no config file is resolvable.

    Passing ``env={}`` looks like "no env" but the loader still calls
    ``Path.home()`` internally, so on hosts where ``~/.config/aawazz/
    aawazz.toml`` exists, the file would be read. Pin AAWAZZ_ROUTING_FILE
    to a non-existent path so the resolution is host-independent.
    """
    cfg = RoutingConfig.load(
        file_path=None,
        env={"AAWAZZ_ROUTING_FILE": str(tmp_path / "no-such.toml")},
    )
    assert cfg.tts == {"en": ("tiny-tts",), "default": ("gtts",)}


def test_load_toml_overrides_per_language(tmp_path: Path) -> None:
    f = tmp_path / "aawazz.toml"
    f.write_text(
        """
        [tts.routing]
        en = ["piper", "tiny-tts"]
        ja = "gtts"
        default = ["gtts"]

        [stt.routing]
        ne = ["whisper"]
        es = ["moonshine"]
        default = ["moonshine"]
        """
    )
    cfg = RoutingConfig.load(file_path=f, env={})
    assert cfg.tts["en"] == ("piper", "tiny-tts")
    assert cfg.tts["ja"] == ("gtts",)
    assert cfg.stt["es"] == ("moonshine",)


def test_load_cli_override_wins_over_env(tmp_path: Path) -> None:
    cfg = RoutingConfig.load(
        file_path=None,
        tts_default_override="piper",
        env={"AAWAZZ_TTS_PROVIDER": "kokoro"},
    )
    assert cfg.tts["default"] == ("piper",)


def test_load_env_overrides_default_when_no_cli(tmp_path: Path) -> None:
    cfg = RoutingConfig.load(
        file_path=None, env={"AAWAZZ_TTS_PROVIDER": "kokoro"}
    )
    assert cfg.tts["default"] == ("kokoro",)


def test_load_per_language_survives_env_default_override(tmp_path: Path) -> None:
    """Env var changes only the ``default`` chain — per-language entries
    from the config file are preserved."""
    f = tmp_path / "aawazz.toml"
    f.write_text(
        """
        [tts.routing]
        en = ["tiny-tts"]
        default = ["gtts"]
        """
    )
    cfg = RoutingConfig.load(
        file_path=f, env={"AAWAZZ_TTS_PROVIDER": "kokoro"}
    )
    assert cfg.tts["en"] == ("tiny-tts",)
    assert cfg.tts["default"] == ("kokoro",)


def test_load_missing_file_silently_falls_back(tmp_path: Path) -> None:
    cfg = RoutingConfig.load(
        file_path=tmp_path / "does-not-exist.toml", env={}
    )
    assert cfg == RoutingConfig.builtin_default()


def test_load_malformed_toml_logs_and_falls_back(tmp_path: Path) -> None:
    f = tmp_path / "bad.toml"
    f.write_text("this is = not valid = toml [[[")
    cfg = RoutingConfig.load(file_path=f, env={})
    # Built-in default applied — neither file nor crash.
    assert cfg.tts["en"] == ("tiny-tts",)


# ── Router.resolve_tts ──────────────────────────────────────────────────────


def test_resolve_tts_en_default() -> None:
    r = Router(RoutingConfig.builtin_default())
    p = r.resolve_tts("en")
    assert p.name == "tiny-tts"


def test_resolve_tts_non_en_falls_through_to_default() -> None:
    r = Router(RoutingConfig.builtin_default())
    p = r.resolve_tts("es")
    assert p.name == "gtts"


def test_resolve_tts_override_succeeds_when_supported() -> None:
    """gtts supports en, so override='gtts' for en is valid."""
    r = Router(RoutingConfig.builtin_default())
    p = r.resolve_tts("en", override="gtts")
    assert p.name == "gtts"


def test_resolve_tts_override_unknown_provider_raises() -> None:
    r = Router(RoutingConfig.builtin_default())
    with pytest.raises(ProviderError, match="not registered"):
        r.resolve_tts("en", override="not-a-real-provider")


def test_resolve_tts_override_unsupported_language_raises() -> None:
    """tiny-tts only supports en; override='tiny-tts' for es must hard-fail."""
    r = Router(RoutingConfig.builtin_default())
    with pytest.raises(ProviderError, match="does not support language"):
        r.resolve_tts("es", override="tiny-tts")


def test_resolve_tts_chain_skips_lang_incompatible_first_choice() -> None:
    """Chain ['tiny-tts', 'gtts'] for es: tiny-tts skipped, gtts wins."""
    cfg = RoutingConfig(
        tts={"es": ("tiny-tts", "gtts"), "default": ("gtts",)},
        stt=RoutingConfig.builtin_default().stt,
    )
    r = Router(cfg)
    p = r.resolve_tts("es")
    assert p.name == "gtts"


def test_resolve_tts_chain_skips_unregistered_provider() -> None:
    """Chain ['nopiper', 'tiny-tts'] for en: nopiper not registered, tiny-tts wins."""
    cfg = RoutingConfig(
        tts={"en": ("nopiper", "tiny-tts"), "default": ("gtts",)},
        stt=RoutingConfig.builtin_default().stt,
    )
    r = Router(cfg)
    p = r.resolve_tts("en")
    assert p.name == "tiny-tts"


def test_resolve_tts_no_compatible_provider_raises() -> None:
    """Chain ['tiny-tts'] for ja: tiny-tts doesn't support ja, no fallback."""
    cfg = RoutingConfig(
        tts={"ja": ("tiny-tts",), "default": ("tiny-tts",)},
        stt=RoutingConfig.builtin_default().stt,
    )
    r = Router(cfg)
    with pytest.raises(ProviderError, match="no tts provider"):
        r.resolve_tts("ja")


# ── Router.resolve_stt ──────────────────────────────────────────────────────


def test_resolve_stt_en_default() -> None:
    r = Router(RoutingConfig.builtin_default())
    p = r.resolve_stt("en")
    assert p.name == "moonshine"


def test_resolve_stt_ne_routes_to_whisper() -> None:
    r = Router(RoutingConfig.builtin_default())
    p = r.resolve_stt("ne")
    assert p.name == "whisper"


def test_resolve_stt_override_unknown_raises() -> None:
    r = Router(RoutingConfig.builtin_default())
    with pytest.raises(ProviderError, match="not registered"):
        r.resolve_stt("en", override="vosk")


# ── Router inspection helpers ───────────────────────────────────────────────


def test_router_routing_inspection() -> None:
    cfg = RoutingConfig(
        tts={"en": ("tiny-tts",), "default": ("gtts",)},
        stt={"ne": ("whisper",), "default": ("moonshine",)},
    )
    r = Router(cfg)
    assert r.tts_routing() == {"en": ["tiny-tts"], "default": ["gtts"]}
    assert r.stt_routing() == {"ne": ["whisper"], "default": ["moonshine"]}
