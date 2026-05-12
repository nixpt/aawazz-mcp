"""Coverage for the Android/Termux clipboard path in aawazz-dictate."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from aawazz_mcp import dictate


def _which_fake(binaries: set[str]):
    return lambda name: f"/bin/{name}" if name in binaries else None


# ── _detect_session_type ───────────────────────────────────────────────────


def test_detect_termux_via_prefix_env(monkeypatch) -> None:
    monkeypatch.setattr("aawazz_mcp.dictate.sys.platform", "linux")
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr("aawazz_mcp.dictate.os.path.isdir", lambda p: False)
    assert dictate._detect_session_type() == "termux"


def test_detect_termux_via_proot_distro_dir(monkeypatch) -> None:
    """proot-distro starts with a clean env — $PREFIX is unset but the
    Termux PREFIX dir is bind-mounted in. The probe must catch this."""
    monkeypatch.setattr("aawazz_mcp.dictate.sys.platform", "linux")
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.setattr(
        "aawazz_mcp.dictate.os.path.isdir",
        lambda p: p == "/data/data/com.termux/files/usr",
    )
    assert dictate._detect_session_type() == "termux"


def test_detect_wayland_wins_over_termux(monkeypatch) -> None:
    """Termux:X11 / Termux:Wayland users keep their display-clipboard path
    when a display is set up. termux fallback only fires when nothing else
    matches — otherwise wl-copy / xclip target the right surface."""
    monkeypatch.setattr("aawazz_mcp.dictate.sys.platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr("aawazz_mcp.dictate.os.path.isdir", lambda p: True)
    assert dictate._detect_session_type() == "wayland"


def test_detect_x11_wins_over_termux(monkeypatch) -> None:
    monkeypatch.setattr("aawazz_mcp.dictate.sys.platform", "linux")
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr("aawazz_mcp.dictate.os.path.isdir", lambda p: True)
    assert dictate._detect_session_type() == "x11"


def test_detect_unknown_when_no_signals(monkeypatch) -> None:
    monkeypatch.setattr("aawazz_mcp.dictate.sys.platform", "linux")
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.setattr("aawazz_mcp.dictate.os.path.isdir", lambda p: False)
    assert dictate._detect_session_type() == "unknown"


# ── _resolve_clipboarder("termux") ─────────────────────────────────────────


def test_resolve_clipboarder_termux_with_binary(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.dictate.shutil.which",
        _which_fake({"termux-clipboard-set"}),
    )
    result = dictate._resolve_clipboarder("termux")
    assert result is not None
    cmd, name = result
    assert cmd == ["termux-clipboard-set"]
    assert name == "termux-clipboard-set"


def test_resolve_clipboarder_termux_without_binary(monkeypatch) -> None:
    """Termux:API addon not installed — fall through to None."""
    monkeypatch.setattr(
        "aawazz_mcp.dictate.shutil.which", lambda name: None
    )
    assert dictate._resolve_clipboarder("termux") is None


def test_resolve_typer_termux_returns_none(monkeypatch) -> None:
    """No keystroke-injection path on Android. Auto-mode falls through to
    clipboard via _resolve_default_mode."""
    monkeypatch.setattr(
        "aawazz_mcp.dictate.shutil.which",
        _which_fake({"termux-clipboard-set"}),
    )
    assert dictate._resolve_typer("termux") is None


def test_default_mode_picks_clipboard_on_termux(monkeypatch) -> None:
    """Auto mode on Termux: no typer entry, clipboard wins."""
    monkeypatch.setattr(
        "aawazz_mcp.dictate.shutil.which",
        _which_fake({"termux-clipboard-set"}),
    )
    assert dictate._resolve_default_mode("termux") == "clipboard"


def test_default_mode_falls_to_stdout_without_termux_api(monkeypatch) -> None:
    """Termux session but no Termux:API installed."""
    monkeypatch.setattr(
        "aawazz_mcp.dictate.shutil.which", lambda name: None
    )
    assert dictate._resolve_default_mode("termux") == "stdout"


# ── _dispatch wiring ───────────────────────────────────────────────────────


def test_dispatch_termux_clipboard_invokes_binary(monkeypatch) -> None:
    """End-to-end argv shape for the termux clipboard path."""
    monkeypatch.setattr(
        "aawazz_mcp.dictate.shutil.which",
        _which_fake({"termux-clipboard-set"}),
    )

    seen: dict = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["input"] = kw.get("input")
        seen["text"] = kw.get("text")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("aawazz_mcp.dictate.subprocess.run", fake_run)

    ok, name = dictate._dispatch("hello world", "clipboard", "termux")
    assert ok is True
    assert name == "termux-clipboard-set"
    assert seen["cmd"] == ["termux-clipboard-set"]
    assert seen["input"] == "hello world"
    assert seen["text"] is True


def test_dispatch_termux_clipboard_propagates_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.dictate.shutil.which",
        _which_fake({"termux-clipboard-set"}),
    )

    def fake_run(cmd, **kw):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr("aawazz_mcp.dictate.subprocess.run", fake_run)

    ok, name = dictate._dispatch("hello", "clipboard", "termux")
    assert ok is False
    assert name == "termux-clipboard-set"
