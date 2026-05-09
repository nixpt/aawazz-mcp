"""Backends: local (bundled tiny-tts + Moonshine + sounddevice) | remote (httpx → FastAPI)."""

from aawazz_mcp.backends.base import Backend

__all__ = ["Backend"]
