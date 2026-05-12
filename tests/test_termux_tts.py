"""Coverage for the termux-tts (Android TextToSpeech) helper module."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from aawazz_mcp.audio import termux_tts


# ── Capability probes ───────────────────────────────────────────────────────


def test_has_engine_false_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.shutil.which", lambda name: None
    )
    assert termux_tts.has_engine() is False


def test_has_engine_true_when_on_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.shutil.which",
        lambda name: f"/data/data/com.termux/files/usr/bin/{name}"
        if name == "termux-tts-speak"
        else None,
    )
    assert termux_tts.has_engine() is True


def test_available_engines_parses_json(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.shutil.which",
        lambda name: f"/bin/{name}" if name == "termux-tts-engines" else None,
    )
    payload = json.dumps(
        [
            {"name": "com.samsung.SMT", "label": "Samsung TTS", "default": True},
            {"name": "com.google.android.tts", "label": "Google TTS", "default": False},
        ]
    ).encode()
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.subprocess.run",
        lambda argv, **kw: SimpleNamespace(returncode=0, stdout=payload, stderr=b""),
    )
    engines = termux_tts.available_engines()
    assert len(engines) == 2
    assert engines[0]["name"] == "com.samsung.SMT"


def test_available_engines_empty_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.shutil.which", lambda name: None
    )
    assert termux_tts.available_engines() == []


def test_available_engines_empty_on_garbage_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.shutil.which", lambda name: "/bin/x"
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.subprocess.run",
        lambda argv, **kw: SimpleNamespace(
            returncode=0, stdout=b"not json", stderr=b""
        ),
    )
    assert termux_tts.available_engines() == []


def test_default_engine_name_picks_default_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.available_engines",
        lambda: [
            {"name": "com.google.android.tts", "default": False},
            {"name": "com.samsung.SMT", "default": True},
        ],
    )
    assert termux_tts.default_engine_name() == "com.samsung.SMT"


def test_default_engine_name_none_when_no_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.available_engines",
        lambda: [{"name": "foo", "default": False}],
    )
    assert termux_tts.default_engine_name() is None


# ── speak() input validation ────────────────────────────────────────────────


def test_speak_rejects_empty_text() -> None:
    out = termux_tts.speak("")
    assert "error" in out
    assert "non-empty" in out["error"]


def test_speak_rejects_oversize_text() -> None:
    out = termux_tts.speak("x" * 5000)
    assert "error" in out
    assert "exceeds max" in out["error"]


def test_speak_rejects_bad_pitch() -> None:
    out = termux_tts.speak("hi", pitch=3.0)
    assert "error" in out
    assert "pitch" in out["error"]


def test_speak_rejects_bad_rate() -> None:
    out = termux_tts.speak("hi", rate=0.1)
    assert "error" in out
    assert "rate" in out["error"]


def test_speak_rejects_bad_stream() -> None:
    out = termux_tts.speak("hi", stream="BOGUS")
    assert "error" in out
    assert "invalid stream" in out["error"]
    assert "NOTIFICATION" in out["hint"]


def test_speak_returns_error_when_binary_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.shutil.which", lambda name: None
    )
    out = termux_tts.speak("hi")
    assert "error" in out
    assert "termux-tts-speak" in out["error"]


# ── speak() success / failure paths ─────────────────────────────────────────


def test_speak_returns_played_true_on_exit_0(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.shutil.which",
        lambda name: f"/bin/{name}",
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.default_engine_name",
        lambda: "com.samsung.SMT",
    )

    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        seen["input"] = kw.get("input")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("aawazz_mcp.audio.termux_tts.subprocess.run", fake_run)

    out = termux_tts.speak("hello world", language="en", pitch=1.2, rate=0.9)
    assert out["played"] is True
    assert out["engine"] == "com.samsung.SMT"
    assert out["language"] == "en"
    assert out["pitch"] == 1.2
    assert out["rate"] == 0.9
    assert out["stream"] == "NOTIFICATION"
    assert out["text_length"] == len("hello world")
    assert isinstance(out["latency_ms"], int)

    # argv shape: [bin, -l, en, -p, 1.2, -r, 0.9, -s, NOTIFICATION]
    assert seen["argv"][0].endswith("/termux-tts-speak")
    assert "-l" in seen["argv"] and "en" in seen["argv"]
    assert "-p" in seen["argv"] and "1.2" in seen["argv"]
    assert "-r" in seen["argv"] and "0.9" in seen["argv"]
    assert "-s" in seen["argv"] and "NOTIFICATION" in seen["argv"]
    assert seen["input"] == b"hello world"


def test_speak_passes_engine_when_specified(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.shutil.which", lambda name: f"/bin/{name}"
    )
    seen: dict = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("aawazz_mcp.audio.termux_tts.subprocess.run", fake_run)

    out = termux_tts.speak("hi", engine="com.google.android.tts")
    assert out["engine"] == "com.google.android.tts"
    assert "-e" in seen["argv"]
    e_idx = seen["argv"].index("-e")
    assert seen["argv"][e_idx + 1] == "com.google.android.tts"


def test_speak_returns_error_on_nonzero_exit(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.shutil.which", lambda name: f"/bin/{name}"
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.subprocess.run",
        lambda argv, **kw: SimpleNamespace(
            returncode=2, stdout=b"", stderr=b"engine not found"
        ),
    )
    out = termux_tts.speak("hi")
    assert "error" in out
    assert "exited 2" in out["error"]
    assert out["stderr"] == "engine not found"
    assert "latency_ms" in out


def test_speak_returns_error_on_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.shutil.which", lambda name: f"/bin/{name}"
    )

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="termux-tts-speak", timeout=60.0)

    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_tts.subprocess.run", raise_timeout
    )
    out = termux_tts.speak("hi")
    assert "error" in out
    assert "timed out" in out["error"]
    assert "hint" in out
