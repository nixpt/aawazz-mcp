"""Provider registry — central in-memory map of all known providers.

Two registration paths (SPEC_v1.3 §2):

- **Built-in providers** register at import time via the
  ``@register_tts`` / ``@register_stt`` / ``@register_post`` /
  ``@register_capture`` / ``@register_playback`` decorators. Module
  ``aawazz_mcp.providers.__init__`` imports each built-in module so
  registration runs as a side-effect of importing the package.

- **Third-party plugins** publish Python entry points under the groups
  ``aawazz.tts_providers`` / ``aawazz.stt_providers`` / etc.
  :func:`discover_plugins` scans these at server startup. Built-in name
  collisions warn and skip (built-ins win).

Provider classes are instantiated **once** at registration. ``__init__``
must be cheap — no model loads — because the registry holds the instance
for the lifetime of the process. Heavy work belongs in
``synthesize`` / ``transcribe`` (today's lazy-first-call pattern).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from aawazz_mcp.provider_base import (
    CaptureProvider,
    LlmProvider,
    PlaybackProvider,
    PostProcessor,
    SttProvider,
    TtsProvider,
)

log = logging.getLogger("aawazz_mcp.registry")


ENTRY_POINT_GROUPS: dict[str, str] = {
    "tts": "aawazz.tts_providers",
    "stt": "aawazz.stt_providers",
    "llm": "aawazz.llm_providers",
    "post": "aawazz.post_processors",
    "capture": "aawazz.capture_providers",
    "playback": "aawazz.playback_providers",
}


@dataclass
class _Registry:
    tts: dict[str, TtsProvider] = field(default_factory=dict)
    stt: dict[str, SttProvider] = field(default_factory=dict)
    llm: dict[str, LlmProvider] = field(default_factory=dict)
    post: dict[str, PostProcessor] = field(default_factory=dict)
    capture: dict[str, CaptureProvider] = field(default_factory=dict)
    playback: dict[str, PlaybackProvider] = field(default_factory=dict)


_REGISTRY = _Registry()
_PLUGINS_DISCOVERED = False


_T = TypeVar("_T")


def _instantiate_and_check_name(cls: type, expected_name: str, kind: str) -> object:
    try:
        instance = cls()
    except Exception as e:
        msg = (
            f"failed to instantiate {kind} provider {expected_name!r} "
            f"({cls.__name__}): {e}"
        )
        raise RuntimeError(msg) from e

    declared = getattr(instance, "name", None)
    if declared and declared != expected_name:
        msg = (
            f"register_{kind}({expected_name!r}) but provider class "
            f"{cls.__name__} declares .name={declared!r}; the two must match"
        )
        raise ValueError(msg)
    if not declared:
        try:
            instance.name = expected_name  # type: ignore[attr-defined]
        except AttributeError:
            msg = (
                f"{kind} provider {cls.__name__} has no settable .name and "
                f"didn't declare one — set ``name = {expected_name!r}`` on the class"
            )
            raise ValueError(msg) from None
    return instance


def _make_decorator(
    kind: str, store: dict
) -> Callable[[str], Callable[[type[_T]], type[_T]]]:
    def register(name: str) -> Callable[[type[_T]], type[_T]]:
        def decorator(cls: type[_T]) -> type[_T]:
            if name in store:
                msg = (
                    f"{kind} provider {name!r} already registered "
                    f"(previous: {type(store[name]).__name__}, new: {cls.__name__})"
                )
                raise ValueError(msg)
            instance = _instantiate_and_check_name(cls, name, kind)
            store[name] = instance
            log.debug("registered %s provider %r (%s)", kind, name, cls.__name__)
            return cls

        return decorator

    return register


register_tts = _make_decorator("tts", _REGISTRY.tts)
register_stt = _make_decorator("stt", _REGISTRY.stt)
register_llm = _make_decorator("llm", _REGISTRY.llm)
register_post = _make_decorator("post", _REGISTRY.post)
register_capture = _make_decorator("capture", _REGISTRY.capture)
register_playback = _make_decorator("playback", _REGISTRY.playback)


# ── Lookup helpers ───────────────────────────────────────────────────────────


def _store_for(kind: str) -> dict:
    return getattr(_REGISTRY, kind)


def get_tts(name: str) -> TtsProvider:
    if name not in _REGISTRY.tts:
        msg = (
            f"tts provider {name!r} not registered. "
            f"Available: {sorted(_REGISTRY.tts.keys())}"
        )
        raise KeyError(msg)
    return _REGISTRY.tts[name]


def get_stt(name: str) -> SttProvider:
    if name not in _REGISTRY.stt:
        msg = (
            f"stt provider {name!r} not registered. "
            f"Available: {sorted(_REGISTRY.stt.keys())}"
        )
        raise KeyError(msg)
    return _REGISTRY.stt[name]


def get_llm(name: str) -> LlmProvider:
    if name not in _REGISTRY.llm:
        msg = (
            f"llm provider {name!r} not registered. "
            f"Available: {sorted(_REGISTRY.llm.keys())}"
        )
        raise KeyError(msg)
    return _REGISTRY.llm[name]


def get_post(name: str) -> PostProcessor:
    if name not in _REGISTRY.post:
        msg = (
            f"post-processor {name!r} not registered. "
            f"Available: {sorted(_REGISTRY.post.keys())}"
        )
        raise KeyError(msg)
    return _REGISTRY.post[name]


def get_capture(name: str) -> CaptureProvider:
    if name not in _REGISTRY.capture:
        msg = (
            f"capture provider {name!r} not registered. "
            f"Available: {sorted(_REGISTRY.capture.keys())}"
        )
        raise KeyError(msg)
    return _REGISTRY.capture[name]


def get_playback(name: str) -> PlaybackProvider:
    if name not in _REGISTRY.playback:
        msg = (
            f"playback provider {name!r} not registered. "
            f"Available: {sorted(_REGISTRY.playback.keys())}"
        )
        raise KeyError(msg)
    return _REGISTRY.playback[name]


def list_tts() -> list[TtsProvider]:
    return list(_REGISTRY.tts.values())


def list_stt() -> list[SttProvider]:
    return list(_REGISTRY.stt.values())


def list_llm() -> list[LlmProvider]:
    return list(_REGISTRY.llm.values())


def list_post() -> list[PostProcessor]:
    return list(_REGISTRY.post.values())


def list_capture() -> list[CaptureProvider]:
    return list(_REGISTRY.capture.values())


def list_playback() -> list[PlaybackProvider]:
    return list(_REGISTRY.playback.values())


# ── Plugin discovery ────────────────────────────────────────────────────────


def discover_plugins(force: bool = False) -> None:
    """Scan ``importlib.metadata`` entry points and register third-party providers.

    Idempotent unless ``force=True``. Built-in name collisions warn and skip;
    plugin load failures log-warn and skip (other plugins still load).
    """
    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED and not force:
        return

    from importlib.metadata import entry_points

    for kind, group in ENTRY_POINT_GROUPS.items():
        store = _store_for(kind)
        try:
            eps = entry_points(group=group)
        except Exception:
            log.exception("entry_points lookup failed for %s", group)
            continue

        for ep in eps:
            if ep.name in store:
                log.warning(
                    "third-party plugin %r in %s collides with built-in; skipping",
                    ep.name,
                    group,
                )
                continue
            try:
                cls = ep.load()
                instance = _instantiate_and_check_name(cls, ep.name, kind)
                store[ep.name] = instance
                log.info("loaded plugin %s/%s from %s", group, ep.name, ep.value)
            except Exception:
                log.exception("plugin %s/%s failed to load", group, ep.name)

    _PLUGINS_DISCOVERED = True


def reset() -> None:
    """Clear the registry. Tests use this to isolate registration state."""
    global _PLUGINS_DISCOVERED
    _REGISTRY.tts.clear()
    _REGISTRY.stt.clear()
    _REGISTRY.llm.clear()
    _REGISTRY.post.clear()
    _REGISTRY.capture.clear()
    _REGISTRY.playback.clear()
    _PLUGINS_DISCOVERED = False
