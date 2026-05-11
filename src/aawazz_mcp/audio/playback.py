"""Audio playback via subprocess shellout. paplay → aplay → afplay probe order.

The probe order matches the conventional Linux/macOS desktop fallback chain:
PulseAudio first (Linux desktop), ALSA second (Linux server / fallback),
CoreAudio third (macOS).

Contract:
    has_player() -> bool                          # any of paplay/aplay/afplay on PATH
    play(audio_path: str) -> bool                 # True on success, False on no-player
    default_provider_name() -> str                # host-aware provider name
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

_LOG = logging.getLogger("aawazz.audio")

# Probe order: PulseAudio first (Linux desktop), ALSA next (Linux fallback /
# server / sandbox), CoreAudio last (macOS).
_PLAYERS: tuple[str, ...] = ("paplay", "aplay", "afplay")

# Hard cap on a single play() call. We don't know the audio duration here, so
# this guards against an infinite hang if the player wedges (e.g. PulseAudio
# socket gone). 60s is plenty for any reasonable TTS/STT clip.
_PLAY_TIMEOUT_S: float = 60.0


def _resolve_player() -> str | None:
    """Return absolute path of the first available player, or None."""
    for name in _PLAYERS:
        path = shutil.which(name)
        if path:
            return path
    return None


def has_player() -> bool:
    """True iff at least one of paplay / aplay / afplay is on PATH.

    Used by ``voices_list().capabilities.play``. Must not raise.
    """
    return _resolve_player() is not None


def play(audio_path: str) -> bool:
    """Spawn a subprocess to play `audio_path`. Return True on success.

    Probe order: paplay → aplay → afplay. Returns True iff a player was found
    AND its return code was 0. Logs warnings to ``aawazz.audio`` on failure
    rather than printing — stdout would corrupt the FastMCP stdio transport.
    """
    player = _resolve_player()
    if player is None:
        _LOG.warning("no audio player found (tried %s)", ", ".join(_PLAYERS))
        return False

    try:
        result = subprocess.run(
            [player, audio_path],
            capture_output=True,
            timeout=_PLAY_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        _LOG.warning("player %s timed out after %.0fs on %s", player, _PLAY_TIMEOUT_S, audio_path)
        return False
    except OSError as exc:
        _LOG.warning("player %s failed to spawn for %s: %s", player, audio_path, exc)
        return False

    if result.returncode != 0:
        _LOG.warning(
            "player %s exited %d on %s: %s",
            player,
            result.returncode,
            audio_path,
            result.stderr.decode("utf-8", errors="replace").strip(),
        )
        return False
    return True


# ── Default-provider selection ──────────────────────────────────────────────

# Termux on Android exposes its PREFIX as ``/data/data/com.termux/files/usr``.
# Detecting that prefix is a more reliable Android signal than ``uname`` —
# proot-distro fakes uname to look like a regular Linux box, but $PREFIX
# survives because the Termux shell sets it before chrooting.
_TERMUX_PREFIX: str = "/data/data/com.termux/files/usr"


def default_provider_name() -> str:
    """Auto-select the playback provider for this host.

    Precedence:
    1. ``$AAWAZZ_PLAYBACK_PROVIDER`` env override — any value, no validation.
    2. Termux/Android with ``termux-media-player`` on PATH → ``"termux-media"``
       (no PulseAudio/ALSA daemon needed on Android).
    3. Fallback → ``"shell"`` (the paplay/aplay/afplay probe).

    Per-call ``speak(playback_provider=...)`` still wins over this default.

    Termux detection: ``$PREFIX`` is set in Termux's native shell, but
    ``proot-distro`` chroots into a fresh env where ``$PREFIX`` is unset.
    Probe the directory too — proot bind-mounts ``/data/data/com.termux``
    into the distro so termux-* binaries are reachable on PATH, which is
    the actual condition we care about.
    """
    override = os.environ.get("AAWAZZ_PLAYBACK_PROVIDER")
    if override:
        return override

    prefix = os.environ.get("PREFIX", "")
    is_termux = (
        prefix.startswith(_TERMUX_PREFIX) or os.path.isdir(_TERMUX_PREFIX)
    )
    if is_termux and shutil.which("termux-media-player"):
        return "termux-media"

    return "shell"
