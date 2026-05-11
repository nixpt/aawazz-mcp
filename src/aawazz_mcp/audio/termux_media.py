"""Audio playback via the Termux:API ``termux-media-player`` binary.

Talks to Android's MediaPlayer service. No PulseAudio / PipeWire / ALSA
daemon required, so this works inside proot-distro on Termux without
the cross-boundary ``module-native-protocol-tcp`` setup that
``paplay`` needs.

Contract (mirrors :mod:`aawazz_mcp.audio.playback`):
    has_player() -> bool        # ``termux-media-player`` on PATH
    play(audio_path: str) -> bool

Returns True iff the request was dispatched to the Android service with
exit code 0. ``termux-media-player play`` is fire-and-forget — playback
runs asynchronously on the device, so a True return means "service
accepted the request," matching the shell provider's contract.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

_LOG = logging.getLogger("aawazz.audio.termux_media")

_BIN: str = "termux-media-player"

# Hard cap on a single play() dispatch. termux-media-player normally returns
# in milliseconds (the Android service handles playback async), but a wedged
# termux-api socket could in principle hang the call.
_PLAY_TIMEOUT_S: float = 10.0


def has_player() -> bool:
    """True iff ``termux-media-player`` is on PATH (Termux:API installed)."""
    return shutil.which(_BIN) is not None


def play(audio_path: str) -> bool:
    """Dispatch ``termux-media-player play <audio_path>``. Return True on exit 0."""
    bin_path = shutil.which(_BIN)
    if bin_path is None:
        _LOG.warning("%s not on PATH — install the Termux:API addon", _BIN)
        return False

    try:
        result = subprocess.run(
            [bin_path, "play", audio_path],
            capture_output=True,
            timeout=_PLAY_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        _LOG.warning(
            "%s timed out after %.0fs on %s", _BIN, _PLAY_TIMEOUT_S, audio_path
        )
        return False
    except OSError as exc:
        _LOG.warning("%s failed to spawn for %s: %s", _BIN, audio_path, exc)
        return False

    if result.returncode != 0:
        _LOG.warning(
            "%s exited %d on %s: %s",
            _BIN,
            result.returncode,
            audio_path,
            result.stderr.decode("utf-8", errors="replace").strip(),
        )
        return False
    return True
