"""Local | remote dispatcher with fail-loud-no-fallback policy.

Resolution: when ``cfg.mode == "remote"``, ``speak`` and ``transcribe`` route to
the FastAPI server via :class:`aawazz_mcp.backends.remote.RemoteBackend`. If the
server is unreachable, :class:`RemoteBackend` returns a structured error
(URL + hint) — **no silent fallback** to the bundled backend. Fail loud is the
contract: the operator opted into remote, masking misconfig is worse than a
clean error.

``listen`` ALWAYS routes to the local backend regardless of mode. The mic is
on the host running this MCP server; the FastAPI server has no path to it.

**Split mode** (one URL set, the other unset) is supported: the unset side
falls back to the bundled local backend. The dispatcher checks
``cfg.remote_mouth_url`` / ``cfg.remote_ears_url`` per-tool rather than relying
on ``cfg.mode``, so a ``mode=remote`` config with only ``mouth`` set still
serves ``transcribe`` from the local backend.

Backend instantiation is lazy on both sides: pure-remote setups never spin up
the heavy ``LocalBackend`` model loaders unless ``listen`` is actually called.
Pure-local setups never open an httpx client.
"""

from __future__ import annotations

from aawazz_mcp.backends.base import Backend
from aawazz_mcp.backends.remote import RemoteBackend
from aawazz_mcp.config import AawazzConfig


class Dispatcher:
    """Tool-call router. Holds at most one local + one remote backend instance."""

    def __init__(self, cfg: AawazzConfig) -> None:
        self.cfg = cfg
        self._local: Backend | None = None
        self._remote: RemoteBackend | None = None
        if cfg.remote_mouth_url is not None or cfg.remote_ears_url is not None:
            self._remote = RemoteBackend(cfg.remote_mouth_url, cfg.remote_ears_url)

    # ----- Lazy backend accessors -------------------------------------------

    def _ensure_local(self) -> Backend:
        """Instantiate the LocalBackend on demand. Heavy: triggers model loaders."""
        if self._local is None:
            # Imported here to keep importing this module cheap. LocalBackend's
            # __init__ is itself light (it lazies its loaders), but the chain
            # pulls in tiny_tts / moonshine_voice — heavy on first call.
            from aawazz_mcp.backends.local import LocalBackend  # noqa: PLC0415
            self._local = LocalBackend(self.cfg)
        return self._local

    def _mouth_backend(self) -> Backend:
        """Pick mouth-side backend. Remote if URL set, else local fallback."""
        if self._remote is not None and self.cfg.remote_mouth_url is not None:
            return self._remote
        return self._ensure_local()

    def _ears_backend(self) -> Backend:
        """Pick ears-side backend. Remote if URL set, else local fallback."""
        if self._remote is not None and self.cfg.remote_ears_url is not None:
            return self._remote
        return self._ensure_local()

    # ----- Tool surface ------------------------------------------------------

    async def warm(self) -> None:
        """Eagerly load both backends' models if applicable. Called by --warm.

        Remote backend warm is a no-op (servers warm themselves). Local backend
        warm only fires if at least one side is local.
        """
        if self._remote is not None:
            await self._remote.warm()
        # Touch local only if any side actually needs it.
        needs_local = (
            self.cfg.remote_mouth_url is None or self.cfg.remote_ears_url is None
        )
        if needs_local:
            await self._ensure_local().warm()

    async def speak(self, **kwargs) -> dict:
        # Non-English TTS routes through gTTS (in local backend).
        # Remote backend also passes language through if present.
        return await self._mouth_backend().speak(**kwargs)

    async def transcribe(self, **kwargs) -> dict:
        return await self._ears_backend().transcribe(**kwargs)

    async def listen(self, **kwargs) -> dict:
        """Always local — mic lives on this host."""
        return await self._ensure_local().listen(**kwargs)

    # ----- Metadata / probes -------------------------------------------------

    async def voices_list(self) -> dict:
        """Pure metadata; no model load. v1.3 response shape:

        - ``providers.{tts,stt,post_processors}`` — each provider's
          capabilities scraped from the registry.
        - ``routing.{tts,stt}`` — the resolved routing chain. Phase 1 emits
          the hardcoded chain (en→tiny-tts, ne→whisper, default→gtts/moonshine);
          phase 2 will read the configurable chain.
        - ``capabilities`` — listen/play probes + backend mode (back-compat).
        - ``tts``, ``stt`` — flattened v1 alias views for callers still on the
          v1.0 / v1.2 response shape. Will be removed in v2.0.
        """
        from aawazz_mcp.audio.capture import has_input_device  # noqa: PLC0415
        from aawazz_mcp.audio.playback import has_player  # noqa: PLC0415
        from aawazz_mcp import providers  # noqa: F401, PLC0415  - register builtins
        from aawazz_mcp import registry as _registry  # noqa: PLC0415

        # ── Provider catalog ────────────────────────────────────────────────
        tts_providers = []
        for p in _registry.list_tts():
            caps = p.capabilities()
            tts_providers.append({
                "name": p.name,
                "version": p.version,
                "languages": sorted(caps.languages),
                "voices": [
                    {
                        "id": v.id,
                        "language": v.language,
                        "description": v.description,
                        "default": v.default,
                    }
                    for v in caps.voices
                ],
                "requires_network": caps.requires_network,
                "sample_rate": caps.sample_rate,
                "accepts_dsp_profiles": caps.accepts_dsp_profiles,
                "speed_range": list(caps.speed_range),
                "notes": caps.notes,
            })

        stt_providers = []
        for p in _registry.list_stt():
            caps = p.capabilities()
            stt_providers.append({
                "name": p.name,
                "version": p.version,
                "languages": sorted(caps.languages),
                "model_archs": {
                    lang: list(archs)
                    for lang, archs in caps.model_archs.items()
                },
                "accepts_url": caps.accepts_url,
                "cold_load_seconds_estimate": caps.cold_load_seconds_estimate,
                "notes": caps.notes,
            })

        post_processors = []
        for p in _registry.list_post():
            post_processors.append({
                "name": p.name,
                "direction": p.direction,
            })

        # ── Live routing chain from cfg ─────────────────────────────────────
        routing = {
            "tts": {k: list(v) for k, v in self.cfg.routing.tts.items()},
            "stt": {k: list(v) for k, v in self.cfg.routing.stt.items()},
        }

        # ── v1 alias views (back-compat for callers on the old shape) ──────
        # DSP profiles still surface here as voice IDs — phase 5 graduates them
        # to post_processors.
        v1_tts_voices = [
            {"id": "MALE", "language": "en", "default": True},
            {"id": "DEEP", "description": "Lower pitch, warm lowpass"},
            {"id": "BRIGHT", "description": "Higher pitch, airy highpass"},
            {"id": "SOFT", "description": "Warm lowpass, smoothed"},
            {"id": "GRAVEL", "description": "Subtle saturation, pitch-down"},
            {"id": "ROBOT", "description": "Rectify + bandpass", "fun": True},
            {"id": "ECHO", "description": "Single echo tap at 300ms"},
            {"id": "WIDE", "description": "Pitch-up + reverb tail"},
        ]
        v1_stt_lang_models: dict[str, list[str]] = {}
        for p in _registry.list_stt():
            caps = p.capabilities()
            for lang, archs in caps.model_archs.items():
                merged = v1_stt_lang_models.setdefault(lang, [])
                for arch in archs:
                    if arch not in merged:
                        merged.append(arch)

        return {
            "providers": {
                "tts": tts_providers,
                "stt": stt_providers,
                "post_processors": post_processors,
            },
            "routing": routing,
            "capabilities": {
                "listen": bool(has_input_device()),
                "play": bool(has_player()),
                "backend_mode": self.cfg.mode,
                "remote_url": {
                    "mouth": self.cfg.remote_mouth_url,
                    "ears": self.cfg.remote_ears_url,
                },
            },
            # v1 alias views — back-compat for v1.0 / v1.2 callers.
            "tts": {
                "backend": "tiny-tts + DSP profiles",
                "voices": v1_tts_voices,
                "voice_profiles": True,
                "note": "Voice profiles are DSP post-processing effects applied to tiny-tts output. Zero additional models required.",
            },
            "stt": {
                "backend": "moonshine",
                "languages": sorted(v1_stt_lang_models.keys()),
                "lang_models": v1_stt_lang_models,
                "note": "Languages marked 'whisper-small' use a Whisper-based model; others use Moonshine.",
                "model_archs": [
                    "tiny",
                    "tiny_streaming",
                    "base",
                    "base_streaming",
                    "small_streaming",
                    "medium_streaming",
                ],
            },
        }

    async def health(self) -> dict:
        """Backing for the ``aawazz://health`` resource (SPEC §1.5).

        Reports models loaded (only meaningful for local backend), backend
        mode, remote URLs, and capability probe.
        """
        from aawazz_mcp import __version__  # noqa: PLC0415
        from aawazz_mcp.audio.capture import has_input_device  # noqa: PLC0415
        from aawazz_mcp.audio.playback import has_player  # noqa: PLC0415

        models_loaded: dict = {"tts": False, "stt_archs": []}
        if self._local is not None:
            # Best-effort introspection — LocalBackend may expose flags. Don't
            # crash health if attributes are absent; treat as "not loaded".
            tts_loader = getattr(self._local, "_tts_loader", None)
            stt_loader = getattr(self._local, "_stt_loader", None)
            models_loaded = {
                "tts": bool(getattr(tts_loader, "loaded", False)) if tts_loader else False,
                "stt_archs": list(getattr(stt_loader, "loaded_archs", []) or []) if stt_loader else [],
            }

        return {
            "version": __version__,
            "mode": self.cfg.mode,
            "remote_url": {
                "mouth": self.cfg.remote_mouth_url,
                "ears": self.cfg.remote_ears_url,
            },
            "models_loaded": models_loaded,
            "capabilities": {
                "listen": bool(has_input_device()),
                "play": bool(has_player()),
            },
        }

    # ----- Lifecycle ---------------------------------------------------------

    async def aclose(self) -> None:
        """Release any open resources. Called from FastMCP lifespan teardown."""
        if self._remote is not None:
            await self._remote.aclose()
