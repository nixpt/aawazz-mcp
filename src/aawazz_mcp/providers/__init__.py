"""Built-in providers package.

Importing this package registers each built-in provider as a side effect.
The registry singleton lives in :mod:`aawazz_mcp.registry`.

Add new built-ins by:
1. Writing a provider module under this package (e.g. ``piper.py``).
2. Decorating the class with the matching ``register_*`` factory.
3. Importing the module here so the decorator runs.

Third-party providers register via Python entry points; see
:func:`aawazz_mcp.registry.discover_plugins`.
"""

from __future__ import annotations

# Import order doesn't matter — registration is idempotent within each module.
from aawazz_mcp.providers import gtts_provider  # noqa: F401
from aawazz_mcp.providers import kokoro  # noqa: F401
from aawazz_mcp.providers import moonshine  # noqa: F401
from aawazz_mcp.providers import pipefish  # noqa: F401
from aawazz_mcp.providers import piper  # noqa: F401
from aawazz_mcp.providers import shell_playback  # noqa: F401
from aawazz_mcp.providers import sounddevice_capture  # noqa: F401
from aawazz_mcp.providers import termux_media_playback  # noqa: F401
from aawazz_mcp.providers import termux_mic_capture  # noqa: F401
from aawazz_mcp.providers import tiny_tts  # noqa: F401
from aawazz_mcp.providers import whisper  # noqa: F401
from aawazz_mcp.providers import xtts  # noqa: F401
