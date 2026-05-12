"""Android TextToSpeech via ``termux-tts-speak``.

Wraps the Termux:API binary. Distinct from aawazz's WAV-producing TTS
providers because Android's TTS engine plays audio directly to a system
audio stream and returns no file. Useful for low-latency one-shot speech
that doesn't need a persisted artifact.

Contract:
    has_engine() -> bool
    available_engines() -> list[dict]      # [{name, label, default}, ...]
    default_engine_name() -> str | None
    speak(...) -> dict                     # blocks until playback finishes
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time

_LOG = logging.getLogger("aawazz.audio.termux_tts")

_SPEAK_BIN: str = "termux-tts-speak"
_ENGINES_BIN: str = "termux-tts-engines"

# Android's AudioManager.STREAM_* values exposed by termux-tts-speak -s.
VALID_STREAMS: frozenset[str] = frozenset({
    "ALARM",
    "MUSIC",
    "NOTIFICATION",
    "RING",
    "SYSTEM",
    "VOICE_CALL",
})

# termux-tts-speak blocks until Android's TTS engine finishes synthesizing
# AND playing the utterance. 60 s is a generous cap for the longest
# reasonable single speech call; an agent shouldn't spin longer.
_SPEAK_TIMEOUT_S: float = 60.0

# Engine probe is cheap (just enumerates installed engines) but a wedged
# termux-api service could hang it; cap conservatively.
_ENGINES_TIMEOUT_S: float = 5.0

# Mirror speak()'s text-length contract so behaviour is consistent across
# the two tools.
_MAX_TEXT_LEN: int = 4000

# Pitch / rate bounds match speak()'s 0.5..2.0 speed range. Android accepts
# wider values but extremes produce unintelligible output.
_MIN_PITCH: float = 0.5
_MAX_PITCH: float = 2.0
_MIN_RATE: float = 0.5
_MAX_RATE: float = 2.0


def has_engine() -> bool:
    """True iff ``termux-tts-speak`` is on PATH (Termux:API addon installed)."""
    return shutil.which(_SPEAK_BIN) is not None


def available_engines() -> list[dict]:
    """Enumerate installed Android TTS engines via ``termux-tts-engines``.

    Returns a list of ``{"name": str, "label": str, "default": bool}``
    dicts; empty list when the binary is missing or fails. Used both as
    a capability probe and to resolve the default engine name when the
    caller of :func:`speak` didn't pass ``engine=``.
    """
    bin_path = shutil.which(_ENGINES_BIN)
    if bin_path is None:
        return []
    try:
        result = subprocess.run(
            [bin_path], capture_output=True, timeout=_ENGINES_TIMEOUT_S
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        _LOG.warning("%s probe failed: %s", _ENGINES_BIN, exc)
        return []
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def default_engine_name() -> str | None:
    """Return the package name of the engine marked ``default=true``, or None."""
    for engine in available_engines():
        if engine.get("default"):
            name = engine.get("name")
            return name if isinstance(name, str) else None
    return None


def speak(
    text: str,
    engine: str | None = None,
    language: str | None = None,
    region: str | None = None,
    variant: str | None = None,
    pitch: float = 1.0,
    rate: float = 1.0,
    stream: str = "NOTIFICATION",
) -> dict:
    """Speak ``text`` via Android TextToSpeech. Returns a result dict.

    Blocks until ``termux-tts-speak`` returns (typically after Android's
    TTS engine finishes synthesizing and playing). Never raises — all
    failures come back as a dict with an ``error`` key.
    """
    if not isinstance(text, str) or not text:
        return {"error": "text must be a non-empty string"}
    if len(text) > _MAX_TEXT_LEN:
        return {
            "error": f"text length {len(text)} exceeds max {_MAX_TEXT_LEN}",
        }
    if not (_MIN_PITCH <= pitch <= _MAX_PITCH):
        return {
            "error": f"pitch {pitch} out of range {_MIN_PITCH}..{_MAX_PITCH}",
        }
    if not (_MIN_RATE <= rate <= _MAX_RATE):
        return {
            "error": f"rate {rate} out of range {_MIN_RATE}..{_MAX_RATE}",
        }
    if stream not in VALID_STREAMS:
        return {
            "error": f"invalid stream {stream!r}",
            "hint": f"expected one of {sorted(VALID_STREAMS)}",
        }

    bin_path = shutil.which(_SPEAK_BIN)
    if bin_path is None:
        return {
            "error": f"{_SPEAK_BIN} not on PATH",
            "hint": "install the Termux:API addon (Android/Termux only)",
        }

    argv: list[str] = [bin_path]
    if engine:
        argv += ["-e", engine]
    if language:
        argv += ["-l", language]
    if region:
        argv += ["-n", region]
    if variant:
        argv += ["-v", variant]
    argv += ["-p", str(pitch), "-r", str(rate), "-s", stream]

    t0 = time.time()
    try:
        result = subprocess.run(
            argv,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=_SPEAK_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {
            "error": f"{_SPEAK_BIN} timed out after {_SPEAK_TIMEOUT_S:.0f}s",
            "hint": "long text, slow rate, or termux-api service is wedged",
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except OSError as exc:
        return {"error": f"{_SPEAK_BIN} failed to spawn: {exc}"}

    latency_ms = int((time.time() - t0) * 1000)

    if result.returncode != 0:
        return {
            "error": f"{_SPEAK_BIN} exited {result.returncode}",
            "stderr": result.stderr.decode("utf-8", errors="replace").strip(),
            "latency_ms": latency_ms,
        }

    return {
        "engine": engine or default_engine_name(),
        "language": language,
        "region": region,
        "variant": variant,
        "pitch": pitch,
        "rate": rate,
        "stream": stream,
        "text_length": len(text),
        "latency_ms": latency_ms,
        "played": True,
    }
