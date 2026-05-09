"""Backend ABC.

Local and remote backends share this interface. Dispatcher routes per-tool.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    """Abstract backend exposing the four tool primitives."""

    @abstractmethod
    async def speak(
        self,
        text: str,
        voice: str = "MALE",
        speed: float = 1.0,
        output_path: str | None = None,
        play: bool = False,
    ) -> dict:
        """Synthesize speech, write a WAV, return metadata."""

    @abstractmethod
    async def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        model_arch: str = "tiny_streaming",
    ) -> dict:
        """Transcribe a local WAV (or http(s) URL) and return text + metadata."""

    @abstractmethod
    async def listen(
        self,
        duration_s: float = 5.0,
        language: str = "en",
        model_arch: str = "tiny_streaming",
        save_audio: bool = False,
    ) -> dict:
        """Capture mic for `duration_s`, transcribe, return text + metadata.

        Note: only :class:`LocalBackend` implements this meaningfully — remote
        backends have no mic access on the MCP server's host. The dispatcher
        routes ``listen`` straight to local regardless of cfg.mode.
        """

    @abstractmethod
    async def warm(self) -> None:
        """Eagerly load models. No-op for remote (servers warm themselves)."""
