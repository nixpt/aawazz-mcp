"""MCP resources exposed by the server.

Wave 2 wires this in — for now, just the schema for ``aawazz://health``:

    {
        "version": "1.0.0",
        "mode": "local" | "remote",
        "remote_url": {"mouth": str | None, "ears": str | None},
        "models_loaded": {"tts": bool, "stt_archs": ["tiny_streaming", ...]},
        "capabilities": {"listen": bool, "play": bool}
    }
"""

from __future__ import annotations

# Wave 2: register on the FastMCP instance via @mcp.resource("aawazz://health").
