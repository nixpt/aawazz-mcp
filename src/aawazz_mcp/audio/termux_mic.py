"""Mic capture via Termux:API's ``termux-microphone-record``.

Works inside proot-distro on Android, where ``sounddevice`` can't enumerate
PortAudio devices through the chroot. Talks to Android's MediaRecorder
service via Termux:API.

Android's MediaRecorder wraps every available encoder (opus, aac, amr_*)
in an ISO Media container (MP4 / 3GP), not the bare formats soundfile
can decode. So this provider records OPUS-in-MP4 and shells out to
``ffmpeg`` to transcode to 16 kHz mono PCM WAV — the format Moonshine
expects. ffmpeg is required at runtime; the provider's
:func:`has_microphone` probe checks both binaries and reports the
specific missing one.

**proot-distro path trap:** termux-microphone-record runs in Termux's
namespace (outside proot). proot's ``/tmp`` is NOT visible to it. The
intermediate audio file lives under ``/sdcard/`` (or
``$AAWAZZ_TERMUX_MIC_TMP_DIR``), which is reachable from both sides; the
final WAV goes to whatever path the caller asked for. Override with
``$AAWAZZ_TERMUX_MIC_TMP_DIR`` if your device's /sdcard mount differs.

Contract:
    has_microphone() -> bool
    record_to_wav(duration_s, output_wav) -> dict
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

_LOG = logging.getLogger("aawazz.audio.termux_mic")

_RECORD_BIN: str = "termux-microphone-record"
_FFMPEG_BIN: str = "ffmpeg"

# Format aawazz feeds Moonshine: 16 kHz mono PCM. termux-microphone-record
# honours -r / -c via Android MediaRecorder; OPUS is the lightest encoder
# (smaller files than AAC, better quality than AMR at this bitrate).
_TARGET_SAMPLE_RATE: int = 16000
_TARGET_CHANNELS: int = 1
_ENCODER: str = "opus"

# Wall-time padding for the service round-trip after a recording's -l
# timer elapses. The Termux:API service returns from `-l` non-blocking,
# so we sleep duration + grace before reading the file.
_FINALIZE_GRACE_S: float = 2.0

# Subprocess timeout for the start command itself — the call returns
# almost instantly after dispatching to the service.
_START_TIMEOUT_S: float = 10.0

# Bounds matching the listen() tool's hard caps.
_MIN_DURATION_S: float = 0.5
_MAX_DURATION_S: float = 30.0

# Intermediate-file directory that must be visible both to Termux:API
# (outside proot) and to this code (inside proot). /sdcard is the
# universal answer on stock Android; override via env for unusual mounts.
_DEFAULT_TMP_DIR: str = "/sdcard"


def _tmp_dir() -> str:
    return os.environ.get("AAWAZZ_TERMUX_MIC_TMP_DIR", _DEFAULT_TMP_DIR)


def _stop_active_recording(bin_path: str) -> None:
    """Stop any in-progress recording before starting a new one.

    A previous call that left the Termux:API service in a recording
    state (or another process competing for the mic) will otherwise
    fail the next start with ``Recording already in progress``.
    Best-effort — ignore non-zero exits, the next start command will
    surface a useful error if the service is genuinely wedged.
    """
    try:
        subprocess.run(
            [bin_path, "-q"], capture_output=True, timeout=3.0
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        _LOG.debug("pre-stop probe (%s -q) failed: %s", _RECORD_BIN, exc)


def has_microphone() -> bool:
    """True iff the binaries we need are on PATH.

    Requires both ``termux-microphone-record`` (Termux:API addon) AND
    ``ffmpeg`` — Android MediaRecorder wraps its encoders in ISO Media
    containers that need ffmpeg to transcode. Doesn't check Android mic
    permission; that fails at record time with a structured error.
    """
    return (
        shutil.which(_RECORD_BIN) is not None
        and shutil.which(_FFMPEG_BIN) is not None
    )


def record_to_wav(duration_s: float, output_wav: str) -> dict:
    """Record ``duration_s`` of mic audio; write 16 kHz mono PCM to ``output_wav``.

    Returns a result dict — never raises. On failure returns
    ``{"error": str, "hint": str?, ...}``; on success returns
    ``{"audio_path", "sample_rate", "duration_s", "latency_ms"}``.
    """
    bin_path = shutil.which(_RECORD_BIN)
    if bin_path is None:
        return {
            "error": f"{_RECORD_BIN} not on PATH",
            "hint": "install the Termux:API addon (Android/Termux only)",
        }
    ffmpeg_path = shutil.which(_FFMPEG_BIN)
    if ffmpeg_path is None:
        return {
            "error": f"{_FFMPEG_BIN} not on PATH",
            "hint": "install via Termux: `pkg install ffmpeg`",
        }

    duration_s = max(_MIN_DURATION_S, min(_MAX_DURATION_S, float(duration_s)))
    # ``-l <seconds>`` is integer in the termux-microphone-record CLI. Round
    # up so callers requesting 0.6 s actually get ~1 s rather than 0
    # (which means "unlimited" — would wedge the call).
    record_seconds: int = max(1, round(duration_s))

    out = Path(output_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The intermediate opus file MUST be in a path the Termux:API service
    # (outside proot) can write — proot's /tmp is invisible to it. /sdcard
    # is reachable from both sides; we use a unique filename to avoid
    # collisions between concurrent listen() calls on the same device.
    tmp_dir = Path(_tmp_dir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Suffix is just an Android filename hint; the actual container is
    # whatever MediaRecorder produces for the chosen encoder (MP4/3GP).
    tmp_capture = tmp_dir / f"aawazz-mic-{uuid.uuid4().hex[:8]}.mp4"

    # Stop any in-progress recording from a prior failed run, otherwise
    # the next start command fails with "Recording already in progress".
    _stop_active_recording(bin_path)

    t0 = time.time()
    argv = [
        bin_path,
        "-l", str(record_seconds),
        "-r", str(_TARGET_SAMPLE_RATE),
        "-c", str(_TARGET_CHANNELS),
        "-e", _ENCODER,
        "-f", str(tmp_capture),
    ]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            timeout=_START_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {
            "error": f"{_RECORD_BIN} start command timed out",
            "hint": "termux-api service may be wedged; try `pkill -f termux-api`",
        }
    except OSError as exc:
        return {"error": f"{_RECORD_BIN} failed to spawn: {exc}"}

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        # Concurrent-recording error is a frequent gotcha — surface it cleanly.
        hint = None
        if "already" in stderr.lower() or "in progress" in stderr.lower():
            hint = "another termux-microphone-record session is active; stop it with `termux-microphone-record -q`"
        return {
            "error": f"{_RECORD_BIN} exited {result.returncode}",
            "stderr": stderr,
            **({"hint": hint} if hint else {}),
        }

    # Recording runs asynchronously in the Termux:API service. Wait the
    # requested duration plus the service-finalize grace before reading
    # the file.
    time.sleep(record_seconds + _FINALIZE_GRACE_S)

    if not tmp_capture.exists() or tmp_capture.stat().st_size == 0:
        return {
            "error": f"recording file {tmp_capture} missing or empty",
            "hint": "Termux:API mic permission may be denied — check Android Settings",
        }

    # Transcode the MediaRecorder-wrapped audio to 16 kHz mono PCM WAV
    # via ffmpeg. ``-ar`` resamples, ``-ac 1`` down-mixes; ``-y`` overwrites.
    transcode_argv = [
        ffmpeg_path,
        "-y",
        "-i", str(tmp_capture),
        "-ar", str(_TARGET_SAMPLE_RATE),
        "-ac", str(_TARGET_CHANNELS),
        "-f", "wav",
        str(out),
    ]
    try:
        transcode = subprocess.run(
            transcode_argv,
            capture_output=True,
            timeout=record_seconds + 30.0,
        )
    except subprocess.TimeoutExpired:
        tmp_capture.unlink(missing_ok=True)
        return {
            "error": f"{_FFMPEG_BIN} timed out during transcode",
            "hint": "extreme — file may be corrupt or ffmpeg wedged",
        }
    except OSError as exc:
        tmp_capture.unlink(missing_ok=True)
        return {"error": f"{_FFMPEG_BIN} failed to spawn: {exc}"}
    finally:
        tmp_capture.unlink(missing_ok=True)

    if transcode.returncode != 0:
        return {
            "error": f"{_FFMPEG_BIN} exited {transcode.returncode}",
            "stderr": transcode.stderr.decode("utf-8", errors="replace").strip()[-500:],
        }

    import soundfile as sf  # noqa: PLC0415

    info = sf.info(str(out))
    return {
        "audio_path": str(out),
        "sample_rate": int(info.samplerate),
        "duration_s": float(info.duration),
        "latency_ms": int((time.time() - t0) * 1000),
    }
