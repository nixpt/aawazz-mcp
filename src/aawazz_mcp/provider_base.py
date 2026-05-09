"""Protocols and request/response types for the v1.3 pluggable backends.

See ``SPEC_v1.3.md`` §1 for the design narrative.

This module defines five Protocol surfaces — TtsProvider, SttProvider,
PostProcessor, CaptureProvider, PlaybackProvider — and the frozen-dataclass
request/response types they exchange. No runtime logic lives here; everything
is types only. Concrete providers live in sibling modules under ``providers/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "VoiceCatalogEntry",
    "TtsCapabilities",
    "TtsRequest",
    "TtsResult",
    "TtsProvider",
    "SttCapabilities",
    "SttRequest",
    "SttResult",
    "SttProvider",
    "PostProcessor",
    "CaptureRequest",
    "CaptureResult",
    "CaptureProvider",
    "PlaybackProvider",
    "ProviderError",
]


class ProviderError(Exception):
    """Typed error raised by providers for caller-facing failures.

    Distinct from generic exceptions so the dispatcher can convert into
    structured ``{error, hint, ...}`` responses instead of crashing the tool
    call. Providers should still raise plain exceptions for programmer errors
    (passing wrong types, etc.) — only wrap user-facing failures here.

    The optional ``hint`` is surfaced separately by the dispatcher as the
    response's ``hint`` field, so callers don't have to parse it out of the
    primary message.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


# ── TTS ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VoiceCatalogEntry:
    """One voice exposed by a TtsProvider."""

    id: str
    """Namespaced voice ID (``"<provider>:<voice>"``). The provider strips its
    own prefix internally; the registry presents IDs with prefix in voices_list.
    """

    language: str = ""
    """ISO 639-1 code; ``""`` means voice covers many/all languages (e.g. XTTS)."""

    description: str = ""
    default: bool = False


@dataclass(frozen=True)
class TtsCapabilities:
    """Static capability profile for a TtsProvider — no model load required."""

    languages: frozenset[str]
    """ISO 639-1 codes the provider can synthesize. Empty set = "any" (gTTS-like)."""

    voices: tuple[VoiceCatalogEntry, ...]
    """Available voices. Empty tuple is acceptable for one-voice-per-language
    providers (e.g. gTTS) — the routing layer treats them as having a default."""

    requires_network: bool = False
    sample_rate: int = 0
    """Native output sample rate. ``0`` means variable per call."""

    accepts_dsp_profiles: bool = False
    """If True, DSP voice profiles (DEEP/BRIGHT/...) post-process this provider's
    output. Set False only when the provider has its own style control that DSP
    would muddy (e.g. XTTS expressivity)."""

    speed_range: tuple[float, float] = (1.0, 1.0)
    """``(min, max)`` multiplier supported by ``synthesize.request.speed``. Equal
    bounds mean fixed speed."""

    notes: str = ""


@dataclass(frozen=True)
class TtsRequest:
    """Inputs for a single ``synthesize`` call."""

    text: str
    language: str = "en"
    voice: str | None = None
    """Voice ID without provider prefix (provider strips namespace before dispatch)."""

    speed: float = 1.0
    output_path: str | None = None
    """Absolute path; if None the provider chooses a default location."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Provider-specific kwargs (XTTS ``speaker_wav``, Piper ``noise_scale``, etc.)."""


@dataclass(frozen=True)
class TtsResult:
    """Outputs from a single ``synthesize`` call."""

    audio_path: str
    sample_rate: int
    duration_s: float
    latency_ms: int
    voice_used: str = ""
    """Voice ID actually used (after fallback resolution)."""

    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TtsProvider(Protocol):
    """Synthesizes text to a WAV file."""

    name: str
    """Stable, lowercase identifier used in routing and registry. Examples:
    ``"tiny-tts"``, ``"gtts"``, ``"piper"``, ``"kokoro"``, ``"xtts"``."""

    version: str
    """Provider version (independent of aawazz-mcp version). Surfaced via
    ``voices_list`` so users can correlate behavior with library bumps."""

    def capabilities(self) -> TtsCapabilities: ...

    async def synthesize(self, request: TtsRequest) -> TtsResult: ...

    async def aclose(self) -> None: ...
    """Best-effort resource release. May be a no-op."""


# ── STT ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SttCapabilities:
    """Static capability profile for an SttProvider."""

    languages: frozenset[str]
    model_archs: dict[str, tuple[str, ...]]
    """Per-language available arch identifiers. Provider-specific strings;
    opaque to the dispatcher. Empty tuple = "provider chooses default"."""

    accepts_url: bool = False
    """True if the provider can pull http(s) URLs itself; False means the
    dispatcher must download the URL to a tempfile first."""

    cold_load_seconds_estimate: float = 5.0
    notes: str = ""


@dataclass(frozen=True)
class SttRequest:
    audio_path: str
    """Absolute path to a WAV; URL handling happens upstream of the provider
    unless ``accepts_url=True`` in capabilities."""

    language: str = "en"
    model_arch: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SttResult:
    text: str
    audio_duration_s: float
    sample_rate: int
    latency_ms: int
    model_arch: str = ""
    language_detected: str | None = None
    confidence: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SttProvider(Protocol):
    name: str
    version: str

    def capabilities(self) -> SttCapabilities: ...

    async def transcribe(self, request: SttRequest) -> SttResult: ...

    async def aclose(self) -> None: ...


# ── Post-process ────────────────────────────────────────────────────────────


@runtime_checkable
class PostProcessor(Protocol):
    """Audio buffer transform. Composes left-to-right via a pipeline list.

    Used both for TTS output (DSP profiles) and STT input (VAD trim, gain).
    Pure: should not have meaningful side effects beyond returning the
    transformed buffer.
    """

    name: str
    """e.g. ``"dsp:DEEP"``, ``"vad:webrtc"``, ``"gain:auto"``."""

    direction: Literal["tts", "stt", "both"]

    def process(
        self, audio: "np.ndarray", sample_rate: int
    ) -> "np.ndarray": ...


# ── Capture ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CaptureRequest:
    duration_s: float
    sample_rate: int = 16000
    save_path: str | None = None
    """Absolute path; ``None`` discards the buffer after caller consumes."""

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptureResult:
    audio_path: str | None
    sample_rate: int
    duration_s: float
    latency_ms: int


@runtime_checkable
class CaptureProvider(Protocol):
    name: str

    def has_input_device(self) -> bool: ...

    async def record(self, request: CaptureRequest) -> CaptureResult: ...

    async def aclose(self) -> None: ...


# ── Playback ────────────────────────────────────────────────────────────────


@runtime_checkable
class PlaybackProvider(Protocol):
    name: str

    def has_player(self) -> bool: ...

    async def play(self, audio_path: str) -> bool:
        """Return True if playback was kicked off successfully."""

    async def aclose(self) -> None: ...
