"""Local | remote dispatcher with fail-loud-no-fallback policy.

Wave 1B owns this module. Wave 0 ships the type signatures so other waves can
import without circular blocking.

Resolution: when ``cfg.mode == "remote"``, ``speak`` and ``transcribe`` route to
the FastAPI server via :class:`aawazz_mcp.backends.remote.RemoteBackend`. If the
server is unreachable, the tool returns a structured error containing the URL
attempted and a hint pointing the user at systemd. **No silent fallback.**

``listen`` always routes to the local backend regardless of mode — the mic is on
the host running this MCP server, the FastAPI server has no path to it.
"""

from __future__ import annotations

from aawazz_mcp.backends.base import Backend
from aawazz_mcp.config import AawazzConfig


class Dispatcher:
    """Tool-call router. Holds at most one local + one remote backend instance."""

    def __init__(self, cfg: AawazzConfig) -> None:
        self.cfg = cfg
        self._local: Backend | None = None
        self._remote: Backend | None = None
        # Wave 1B: instantiate the appropriate backend(s) based on cfg.mode.

    async def warm(self) -> None:
        """Eagerly load both backends' models if applicable. Called by --warm."""
        raise NotImplementedError("Wave 1B")

    async def speak(self, **kwargs) -> dict:
        raise NotImplementedError("Wave 1B")

    async def transcribe(self, **kwargs) -> dict:
        raise NotImplementedError("Wave 1B")

    async def listen(self, **kwargs) -> dict:
        """Always local — mic lives on this host."""
        raise NotImplementedError("Wave 1B")

    async def voices_list(self) -> dict:
        """Pure metadata; no model load. Wave 1B: synthesize from cfg + sounddevice probe."""
        raise NotImplementedError("Wave 1B")

    async def health(self) -> dict:
        """Backing for the ``aawazz://health`` resource."""
        raise NotImplementedError("Wave 1B")
