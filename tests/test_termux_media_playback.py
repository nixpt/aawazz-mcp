"""Coverage for the termux-media playback provider + default selection."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from aawazz_mcp import providers  # noqa: F401  — registration side-effect
from aawazz_mcp import registry
from aawazz_mcp.audio import playback as playback_audio


# ── Registration ────────────────────────────────────────────────────────────


def test_termux_media_registered() -> None:
    p = registry.get_playback("termux-media")
    assert p.name == "termux-media"


def test_termux_media_has_player_observes_monkeypatch(monkeypatch) -> None:
    """Provider does a module-level lookup so tests can patch the canonical
    helper without rebinding the provider's reference."""
    p = registry.get_playback("termux-media")

    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_media.has_player", lambda: False
    )
    assert p.has_player() is False

    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_media.has_player", lambda: True
    )
    assert p.has_player() is True


# ── play() success / failure modes ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_play_returns_false_when_binary_missing(monkeypatch) -> None:
    p = registry.get_playback("termux-media")
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_media.shutil.which", lambda name: None
    )
    assert await p.play("/tmp/never.wav") is False


@pytest.mark.asyncio
async def test_play_returns_true_on_exit_0(monkeypatch, tmp_path) -> None:
    p = registry.get_playback("termux-media")
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_media.shutil.which",
        lambda name: "/usr/bin/termux-media-player",
    )

    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_media.subprocess.run", fake_run
    )

    out = tmp_path / "x.wav"
    out.write_bytes(b"")
    assert await p.play(str(out)) is True
    assert seen["argv"][0] == "/usr/bin/termux-media-player"
    assert seen["argv"][1] == "play"
    assert seen["argv"][2] == str(out)


@pytest.mark.asyncio
async def test_play_returns_false_on_nonzero_exit(monkeypatch, tmp_path) -> None:
    p = registry.get_playback("termux-media")
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_media.shutil.which",
        lambda name: "/usr/bin/termux-media-player",
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_media.subprocess.run",
        lambda argv, **kw: SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"service unavailable"
        ),
    )
    assert await p.play(str(tmp_path / "x.wav")) is False


@pytest.mark.asyncio
async def test_play_returns_false_on_timeout(monkeypatch, tmp_path) -> None:
    p = registry.get_playback("termux-media")
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_media.shutil.which",
        lambda name: "/usr/bin/termux-media-player",
    )

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="termux-media-player", timeout=10.0)

    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_media.subprocess.run", raise_timeout
    )
    assert await p.play(str(tmp_path / "x.wav")) is False


# ── Default-provider selection ──────────────────────────────────────────────


def test_default_falls_back_to_shell(monkeypatch) -> None:
    monkeypatch.delenv("AAWAZZ_PLAYBACK_PROVIDER", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setattr(
        "aawazz_mcp.audio.playback.os.path.isdir", lambda p: False
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.playback.shutil.which", lambda name: None
    )
    assert playback_audio.default_provider_name() == "shell"


def test_default_picks_termux_media_on_native_termux(monkeypatch) -> None:
    """Native Termux shell: $PREFIX is set, dir lookup not strictly needed."""
    monkeypatch.delenv("AAWAZZ_PLAYBACK_PROVIDER", raising=False)
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(
        "aawazz_mcp.audio.playback.os.path.isdir", lambda p: False
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.playback.shutil.which",
        lambda name: f"/data/data/com.termux/files/usr/bin/{name}"
        if name == "termux-media-player"
        else None,
    )
    assert playback_audio.default_provider_name() == "termux-media"


def test_default_picks_termux_media_inside_proot_distro(monkeypatch) -> None:
    """proot-distro chroots with a clean env — $PREFIX is unset but the
    Termux PREFIX directory is bind-mounted in, and termux-* binaries are
    reachable on PATH. The probe must catch this case."""
    monkeypatch.delenv("AAWAZZ_PLAYBACK_PROVIDER", raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.setattr(
        "aawazz_mcp.audio.playback.os.path.isdir",
        lambda p: p == "/data/data/com.termux/files/usr",
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.playback.shutil.which",
        lambda name: f"/data/data/com.termux/files/usr/bin/{name}"
        if name == "termux-media-player"
        else None,
    )
    assert playback_audio.default_provider_name() == "termux-media"


def test_default_falls_back_when_termux_but_no_binary(monkeypatch) -> None:
    """Termux without the API addon installed — termux-media-player isn't on
    PATH, so we shouldn't auto-pick it (would hard-fail every speak(play=true))."""
    monkeypatch.delenv("AAWAZZ_PLAYBACK_PROVIDER", raising=False)
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(
        "aawazz_mcp.audio.playback.os.path.isdir", lambda p: True
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.playback.shutil.which", lambda name: None
    )
    assert playback_audio.default_provider_name() == "shell"


def test_env_override_wins(monkeypatch) -> None:
    """AAWAZZ_PLAYBACK_PROVIDER short-circuits the auto-probe. No validation
    here — caller can name a provider that doesn't exist; the registry
    lookup will surface that error later."""
    monkeypatch.setenv("AAWAZZ_PLAYBACK_PROVIDER", "custom-provider")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(
        "aawazz_mcp.audio.playback.shutil.which",
        lambda name: "/data/data/com.termux/files/usr/bin/termux-media-player",
    )
    assert playback_audio.default_provider_name() == "custom-provider"
