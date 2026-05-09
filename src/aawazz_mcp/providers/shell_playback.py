"""Built-in playback provider — shell-out to paplay/aplay/afplay.

Phase 6 of v1.3 (SPEC §1.5). Wraps :func:`aawazz_mcp.audio.playback.play`,
which probes for a desktop audio player on PATH and spawns it. Plugin
surface exists so users on headless / sandboxed boxes can register an
alternative (e.g. push to MPD, write to a named pipe, etc.) without
forking aawazz-mcp.
"""

from __future__ import annotations

import asyncio

from aawazz_mcp.audio import playback as _playback_module
from aawazz_mcp.registry import register_playback


@register_playback("shell")
class ShellPlaybackProvider:
    name = "shell"
    version = "1.0"

    def has_player(self) -> bool:
        # Module-level lookup so monkeypatched probes are honored.
        return bool(_playback_module.has_player())

    async def play(self, audio_path: str) -> bool:
        return bool(
            await asyncio.to_thread(_playback_module.play, audio_path)
        )

    async def aclose(self) -> None:
        pass
