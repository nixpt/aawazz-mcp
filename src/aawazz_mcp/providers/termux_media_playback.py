"""Built-in playback provider — shell-out to ``termux-media-player``.

Mirrors :mod:`aawazz_mcp.providers.shell_playback` but targets Android's
MediaPlayer service via the Termux:API addon. The win on Termux/Android
is that no PulseAudio / PipeWire / ALSA daemon needs to be started
out-of-band — the Termux:API service handles audio routing itself.

Auto-selection: see :func:`aawazz_mcp.audio.playback.default_provider_name`
— on a Termux host with this binary present, this provider becomes the
implicit default for ``speak(play=True)`` instead of ``shell``.
"""

from __future__ import annotations

import asyncio

from aawazz_mcp.audio import termux_media as _termux_module
from aawazz_mcp.registry import register_playback


@register_playback("termux-media")
class TermuxMediaPlaybackProvider:
    name = "termux-media"
    version = "1.0"

    def has_player(self) -> bool:
        # Module-level lookup so monkeypatched probes are honored.
        return bool(_termux_module.has_player())

    async def play(self, audio_path: str) -> bool:
        return bool(
            await asyncio.to_thread(_termux_module.play, audio_path)
        )

    async def aclose(self) -> None:
        pass
