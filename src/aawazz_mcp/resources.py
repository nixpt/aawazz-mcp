"""MCP resources exposed by the server.

The ``aawazz://health`` resource handler is registered directly in
:func:`aawazz_mcp.server.build_server` (Wave 2) — keeping it next to the
FastMCP instance avoids a circular import (the handler closes over the
dispatcher created inside ``build_server``).

Schema for ``aawazz://health`` (SPEC §1.5)::

    {
        "version": "1.0.0",
        "mode": "local" | "remote",
        "remote_url": {"mouth": str | None, "ears": str | None},
        "models_loaded": {"tts": bool, "stt_archs": ["tiny_streaming", ...]},
        "capabilities": {"listen": bool, "play": bool}
    }

The actual JSON is produced by :meth:`aawazz_mcp.dispatcher.Dispatcher.health`.
"""

from __future__ import annotations
