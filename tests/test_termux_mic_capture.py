"""Coverage for the termux-mic capture provider + default selection."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from aawazz_mcp import providers  # noqa: F401 — register
from aawazz_mcp import registry
from aawazz_mcp.audio import capture as capture_audio
from aawazz_mcp.audio import termux_mic
from aawazz_mcp.provider_base import CaptureRequest, ProviderError


def _which_fake(binaries: set[str]):
    """shutil.which mock: returns a path for names in `binaries`, else None."""
    return lambda name: f"/bin/{name}" if name in binaries else None


def _make_wav(path, duration_s: float = 1.0, sample_rate: int = 16000) -> None:
    audio = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


@pytest.fixture(autouse=True)
def _redirect_termux_mic_tmp_dir(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect the intermediate-opus directory away from /sdcard for tests."""
    monkeypatch.setenv("AAWAZZ_TERMUX_MIC_TMP_DIR", str(tmp_path))


# ── Registration ────────────────────────────────────────────────────────────


def test_termux_mic_registered() -> None:
    p = registry.get_capture("termux-mic")
    assert p.name == "termux-mic"


def test_termux_mic_has_input_device_observes_monkeypatch(monkeypatch) -> None:
    p = registry.get_capture("termux-mic")
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.has_microphone", lambda: False
    )
    assert p.has_input_device() is False
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.has_microphone", lambda: True
    )
    assert p.has_input_device() is True


# ── record_to_wav() input validation / quick errors ─────────────────────────


def test_record_to_wav_errors_when_record_bin_missing(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.shutil.which", lambda name: None
    )
    out = termux_mic.record_to_wav(2.0, str(tmp_path / "x.wav"))
    assert "error" in out
    assert "termux-microphone-record" in out["error"]
    assert "Termux:API" in out["hint"]


def test_record_to_wav_errors_when_ffmpeg_missing(monkeypatch, tmp_path) -> None:
    """termux-microphone-record is on PATH but ffmpeg isn't — fail-loud."""
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.shutil.which",
        _which_fake({"termux-microphone-record"}),
    )
    out = termux_mic.record_to_wav(2.0, str(tmp_path / "x.wav"))
    assert "error" in out
    assert "ffmpeg" in out["error"]
    assert "pkg install ffmpeg" in out["hint"]


def test_record_to_wav_errors_on_nonzero_exit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.shutil.which",
        _which_fake({"termux-microphone-record", "ffmpeg"}),
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.subprocess.run",
        lambda argv, **kw: SimpleNamespace(
            returncode=1,
            stdout=b"",
            stderr=b"Recording is already in progress",
        ),
    )
    out = termux_mic.record_to_wav(2.0, str(tmp_path / "x.wav"))
    assert "error" in out
    assert "exited 1" in out["error"]
    assert "in progress" in out["stderr"]
    assert "another termux-microphone-record" in out["hint"]


def test_record_to_wav_errors_on_start_timeout(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.shutil.which",
        _which_fake({"termux-microphone-record", "ffmpeg"}),
    )

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="termux-microphone-record", timeout=10.0)

    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.subprocess.run", raise_timeout
    )
    out = termux_mic.record_to_wav(2.0, str(tmp_path / "x.wav"))
    assert "error" in out
    assert "timed out" in out["error"]
    assert "hint" in out


def test_record_to_wav_errors_when_file_missing(monkeypatch, tmp_path) -> None:
    """termux-microphone-record exited 0 but the service never wrote a file
    (typical when the user denied the mic permission)."""
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.shutil.which",
        _which_fake({"termux-microphone-record", "ffmpeg"}),
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.subprocess.run",
        lambda argv, **kw: SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    )
    monkeypatch.setattr("aawazz_mcp.audio.termux_mic.time.sleep", lambda s: None)
    out = termux_mic.record_to_wav(1.0, str(tmp_path / "x.wav"))
    assert "error" in out
    assert "missing or empty" in out["error"]
    assert "permission" in out["hint"]


def test_record_to_wav_errors_when_ffmpeg_fails(monkeypatch, tmp_path) -> None:
    """Capture succeeded but ffmpeg refused the file (corrupt / unsupported)."""
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.shutil.which",
        _which_fake({"termux-microphone-record", "ffmpeg"}),
    )
    monkeypatch.setattr("aawazz_mcp.audio.termux_mic.time.sleep", lambda s: None)

    def fake_run(argv, **kw):
        if "-q" in argv:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if argv[0].endswith("/ffmpeg") or argv[0] == "ffmpeg":
            return SimpleNamespace(
                returncode=1, stdout=b"", stderr=b"Invalid data found in stream"
            )
        # Recording call — pretend the service wrote a non-empty file.
        tmp = argv[argv.index("-f") + 1]
        with open(tmp, "wb") as f:
            f.write(b"\x00" * 100)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("aawazz_mcp.audio.termux_mic.subprocess.run", fake_run)
    out = termux_mic.record_to_wav(1.0, str(tmp_path / "x.wav"))
    assert "error" in out
    assert "ffmpeg" in out["error"] and "exited 1" in out["error"]
    assert "Invalid data" in out["stderr"]


# ── record_to_wav() argv shape + format pass-through ────────────────────────


def test_record_to_wav_invokes_expected_argv(monkeypatch, tmp_path) -> None:
    """Both the recording start and the ffmpeg transcode use the right flags."""
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.shutil.which",
        _which_fake({"termux-microphone-record", "ffmpeg"}),
    )
    monkeypatch.setattr("aawazz_mcp.audio.termux_mic.time.sleep", lambda s: None)

    seen_record: dict = {}
    seen_ffmpeg: dict = {}

    def fake_run(argv, **kw):
        if "-q" in argv:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if argv[0].endswith("/ffmpeg") or argv[0] == "ffmpeg":
            seen_ffmpeg["argv"] = argv
            # Pretend ffmpeg wrote a valid 16 kHz mono WAV at the output path.
            _make_wav(argv[-1])
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        # Recording call — pretend the service wrote a non-empty file.
        seen_record["argv"] = argv
        tmp = argv[argv.index("-f") + 1]
        with open(tmp, "wb") as f:
            f.write(b"\x00" * 100)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("aawazz_mcp.audio.termux_mic.subprocess.run", fake_run)

    out_path = tmp_path / "out.wav"
    result = termux_mic.record_to_wav(2.0, str(out_path))

    assert "error" not in result
    assert result["audio_path"] == str(out_path)
    assert result["sample_rate"] == 16000

    # Recording argv: -l 2 -r 16000 -c 1 -e opus -f <tmp>.
    rec = seen_record["argv"]
    assert "-l" in rec and "2" in rec
    assert "-r" in rec and "16000" in rec
    assert "-c" in rec and "1" in rec
    assert "-e" in rec and "opus" in rec
    assert "-f" in rec

    # ffmpeg argv: -ar 16000 -ac 1 -f wav into the requested output.
    ff = seen_ffmpeg["argv"]
    assert "-ar" in ff and "16000" in ff
    assert "-ac" in ff and "1" in ff
    assert "-f" in ff and "wav" in ff
    assert ff[-1] == str(out_path)


def test_record_to_wav_cleans_up_intermediate(monkeypatch, tmp_path) -> None:
    """The intermediate capture file is removed regardless of ffmpeg outcome."""
    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.shutil.which",
        _which_fake({"termux-microphone-record", "ffmpeg"}),
    )
    monkeypatch.setattr("aawazz_mcp.audio.termux_mic.time.sleep", lambda s: None)

    intermediate_paths: list[str] = []

    def fake_run(argv, **kw):
        if "-q" in argv:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if argv[0].endswith("/ffmpeg") or argv[0] == "ffmpeg":
            _make_wav(argv[-1])
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        tmp = argv[argv.index("-f") + 1]
        intermediate_paths.append(tmp)
        with open(tmp, "wb") as f:
            f.write(b"\x00" * 100)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("aawazz_mcp.audio.termux_mic.subprocess.run", fake_run)
    result = termux_mic.record_to_wav(1.0, str(tmp_path / "out.wav"))

    assert "error" not in result
    # The intermediate file was created during the run but is now gone.
    assert intermediate_paths
    import os
    assert not os.path.exists(intermediate_paths[0])


# ── CaptureProvider record() wrapping ───────────────────────────────────────


@pytest.mark.asyncio
async def test_provider_requires_save_path() -> None:
    p = registry.get_capture("termux-mic")
    with pytest.raises(ProviderError, match="requires save_path"):
        await p.record(CaptureRequest(duration_s=1.0, save_path=None))


@pytest.mark.asyncio
async def test_provider_propagates_hint_on_error(monkeypatch, tmp_path) -> None:
    p = registry.get_capture("termux-mic")

    def fake_helper(duration_s, output_path):
        return {
            "error": "termux-microphone-record not on PATH",
            "hint": "install Termux:API",
        }

    monkeypatch.setattr(
        "aawazz_mcp.audio.termux_mic.record_to_wav", fake_helper
    )
    with pytest.raises(ProviderError) as excinfo:
        await p.record(
            CaptureRequest(duration_s=1.0, save_path=str(tmp_path / "x.wav"))
        )
    assert "not on PATH" in str(excinfo.value)
    assert excinfo.value.hint == "install Termux:API"


# ── default_provider_name() ─────────────────────────────────────────────────


def test_default_falls_back_to_sounddevice(monkeypatch) -> None:
    monkeypatch.delenv("AAWAZZ_CAPTURE_PROVIDER", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.os.path.isdir", lambda p: False
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.shutil.which", lambda name: None
    )
    assert capture_audio.default_provider_name() == "sounddevice"


def test_default_picks_termux_mic_on_native_termux(monkeypatch) -> None:
    """Auto-select treats Termux + termux-microphone-record as termux-mic.

    Ffmpeg is enforced at record time, not at default-selection time —
    keeps the provider visible even when ffmpeg is missing, so the user
    gets a helpful error rather than silent fallback to sounddevice
    (which itself can't reach PortAudio through proot).
    """
    monkeypatch.delenv("AAWAZZ_CAPTURE_PROVIDER", raising=False)
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.os.path.isdir", lambda p: False
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.shutil.which",
        lambda name: f"/data/data/com.termux/files/usr/bin/{name}"
        if name == "termux-microphone-record"
        else None,
    )
    assert capture_audio.default_provider_name() == "termux-mic"


def test_default_picks_termux_mic_inside_proot_distro(monkeypatch) -> None:
    """proot-distro chroots with a clean env — $PREFIX is unset but the
    Termux PREFIX directory is bind-mounted in. The probe must catch it."""
    monkeypatch.delenv("AAWAZZ_CAPTURE_PROVIDER", raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.os.path.isdir",
        lambda p: p == "/data/data/com.termux/files/usr",
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.shutil.which",
        lambda name: f"/data/data/com.termux/files/usr/bin/{name}"
        if name == "termux-microphone-record"
        else None,
    )
    assert capture_audio.default_provider_name() == "termux-mic"


def test_default_falls_back_when_termux_but_no_binary(monkeypatch) -> None:
    """Termux without Termux:API addon — fall back to sounddevice."""
    monkeypatch.delenv("AAWAZZ_CAPTURE_PROVIDER", raising=False)
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.os.path.isdir", lambda p: True
    )
    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.shutil.which", lambda name: None
    )
    assert capture_audio.default_provider_name() == "sounddevice"


def test_default_env_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("AAWAZZ_CAPTURE_PROVIDER", "custom-mic")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.shutil.which",
        lambda name: "/path/to/termux-microphone-record",
    )
    assert capture_audio.default_provider_name() == "custom-mic"
