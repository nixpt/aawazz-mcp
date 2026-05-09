"""Audio playback via subprocess shellout. paplay → aplay → afplay probe order.

Wave 1C owns this module. Reference: existing joker-mcp ``maybe_autoplay`` in
``crates/ai/services/joker-mcp/src/modalities.rs`` for the player probe order
and the captain's ``AAWAZZ_AUTOPLAY=1`` opt-in convention.

Contract:
    has_player() -> bool                          # any of paplay/aplay/afplay on PATH
    play(audio_path: str) -> bool                 # True on success, False on no-player
"""

from __future__ import annotations


def has_player() -> bool:
    """True iff at least one of paplay / aplay / afplay is on PATH.

    Used by ``voices_list().capabilities.play``. Must not raise.
    """
    raise NotImplementedError("Wave 1C: shutil.which() probe")


def play(audio_path: str) -> bool:
    """Spawn a subprocess to play `audio_path`. Return True on success.

    Wave 1C:
        - probe paplay → aplay → afplay
        - subprocess.run([player, audio_path], capture_output=True, timeout=...)
        - True iff returncode 0 AND a player was found
    """
    raise NotImplementedError("Wave 1C")
