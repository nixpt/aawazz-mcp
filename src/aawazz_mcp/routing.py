"""Provider routing chain — phase 2 of v1.3 (``SPEC_v1.3.md`` §3).

Layers four config sources to produce a per-language preference list:

1. Built-in default (matches v1.2.x: en→tiny-tts, default→gtts; ne→whisper, default→moonshine).
2. TOML config file (``~/.config/aawazz/aawazz.toml`` or ``$AAWAZZ_ROUTING_FILE``).
3. Env vars ``AAWAZZ_TTS_PROVIDER`` / ``AAWAZZ_STT_PROVIDER`` (override the
   ``default`` chain only — per-language entries from config are preserved).
4. CLI flags ``--tts-default`` / ``--stt-default`` (same shape as env, takes precedence).

Per-call ``tts_provider=`` / ``stt_provider=`` overrides bypass the chain
entirely and hard-fail if the provider is missing or doesn't support the
requested language (SPEC §13 Q7 — no silent fallback).

Example ``aawazz.toml``::

    [tts.routing]
    en      = ["piper", "tiny-tts"]
    es      = ["piper", "gtts"]
    default = ["gtts"]

    [stt.routing]
    en      = ["moonshine"]
    ne      = ["whisper"]
    default = ["moonshine"]
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from aawazz_mcp import registry as _registry
from aawazz_mcp.provider_base import (
    ProviderError,
    SttProvider,
    TtsProvider,
)

log = logging.getLogger("aawazz_mcp.routing")


# Phase-1 hardcoded routing reproduced as the v1.2.x-compatible built-in
# default. Loading no config gives identical behavior to v1.2.x.
_BUILTIN_DEFAULT_TTS: dict[str, tuple[str, ...]] = {
    "en": ("tiny-tts",),
    "default": ("gtts",),
}
_BUILTIN_DEFAULT_STT: dict[str, tuple[str, ...]] = {
    "ne": ("whisper",),
    "default": ("moonshine",),
}


@dataclass(frozen=True)
class RoutingConfig:
    """Per-stage preference lists keyed by language; ``"default"`` is fallback."""

    tts: Mapping[str, tuple[str, ...]]
    stt: Mapping[str, tuple[str, ...]]

    @classmethod
    def builtin_default(cls) -> "RoutingConfig":
        return cls(tts=dict(_BUILTIN_DEFAULT_TTS), stt=dict(_BUILTIN_DEFAULT_STT))

    @classmethod
    def load(
        cls,
        file_path: Path | str | None = None,
        *,
        tts_default_override: str | None = None,
        stt_default_override: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> "RoutingConfig":
        """Layer four config sources into a resolved routing map.

        ``file_path`` defaults to ``$AAWAZZ_ROUTING_FILE`` if set, else
        ``~/.config/aawazz/aawazz.toml`` (or under ``$XDG_CONFIG_HOME``).
        Missing file is not an error — built-in default applies.

        ``*_override`` parameters represent CLI flags and replace the
        ``default`` chain only; per-language config entries survive.
        """
        env = env if env is not None else os.environ

        tts: dict[str, tuple[str, ...]] = dict(_BUILTIN_DEFAULT_TTS)
        stt: dict[str, tuple[str, ...]] = dict(_BUILTIN_DEFAULT_STT)

        # Layer 2: config file.
        resolved_path = _resolve_config_path(file_path, env)
        if resolved_path is not None and resolved_path.exists():
            try:
                file_tts, file_stt = _parse_toml(resolved_path)
            except Exception:
                log.exception("failed to parse routing config %s", resolved_path)
            else:
                tts.update(file_tts)
                stt.update(file_stt)
                log.debug("loaded routing config from %s", resolved_path)

        # Layer 3: env-var defaults.
        env_tts_default = (env.get("AAWAZZ_TTS_PROVIDER") or "").strip()
        env_stt_default = (env.get("AAWAZZ_STT_PROVIDER") or "").strip()
        if env_tts_default:
            tts["default"] = (env_tts_default,)
        if env_stt_default:
            stt["default"] = (env_stt_default,)

        # Layer 4: CLI overrides win.
        if tts_default_override:
            tts["default"] = (tts_default_override,)
        if stt_default_override:
            stt["default"] = (stt_default_override,)

        return cls(tts=tts, stt=stt)


def _resolve_config_path(
    explicit: Path | str | None, env: Mapping[str, str]
) -> Path | None:
    if explicit:
        return Path(str(explicit)).expanduser()
    env_path = env.get("AAWAZZ_ROUTING_FILE")
    if env_path:
        return Path(env_path).expanduser()
    xdg = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg) / "aawazz" / "aawazz.toml"


def _parse_toml(path: Path) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    import tomllib  # noqa: PLC0415

    with path.open("rb") as fh:
        doc = tomllib.load(fh)
    tts_raw = (doc.get("tts") or {}).get("routing") or {}
    stt_raw = (doc.get("stt") or {}).get("routing") or {}
    return _coerce(tts_raw), _coerce(stt_raw)


def _coerce(d: dict) -> dict[str, tuple[str, ...]]:
    """Accept either a string (single provider) or list (chain) per language."""
    out: dict[str, tuple[str, ...]] = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = (v,)
        elif isinstance(v, (list, tuple)):
            out[k] = tuple(str(x) for x in v)
        else:
            log.warning("ignoring routing entry %r (not str or list)", k)
    return out


class Router:
    """Resolves a TtsProvider / SttProvider for a given (language, override).

    Reads the registry on each call so dynamically-registered providers
    (e.g. via ``discover_plugins`` after init) are immediately routable.
    """

    def __init__(self, config: RoutingConfig) -> None:
        self.config = config

    # ── TTS ─────────────────────────────────────────────────────────────────

    def resolve_tts(self, language: str, override: str | None = None) -> TtsProvider:
        if override:
            return self._resolve_override_tts(override, language)
        chain = self.config.tts.get(language) or self.config.tts.get("default") or ()
        return self._resolve_chain_tts(chain, language)

    def _resolve_override_tts(self, name: str, language: str) -> TtsProvider:
        try:
            provider = _registry.get_tts(name)
        except KeyError as e:
            registered = sorted(p.name for p in _registry.list_tts())
            msg = f"tts_provider {name!r} not registered; registered: {registered}"
            raise ProviderError(msg) from e
        caps = provider.capabilities()
        if language not in caps.languages:
            supported = sorted(caps.languages)
            head = supported if len(supported) <= 20 else supported[:20] + ["..."]
            msg = (
                f"tts_provider {name!r} does not support language "
                f"{language!r}; supports: {head}"
            )
            raise ProviderError(msg)
        return provider

    def _resolve_chain_tts(
        self, chain: tuple[str, ...], language: str
    ) -> TtsProvider:
        tried: list[str] = []
        for name in chain:
            tried.append(name)
            try:
                provider = _registry.get_tts(name)
            except KeyError:
                log.debug("routing chain skip: tts %r not registered", name)
                continue
            if language in provider.capabilities().languages:
                return provider
            log.debug(
                "routing chain skip: tts %r does not support %r", name, language
            )
        msg = (
            f"no tts provider in routing chain supports language "
            f"{language!r}; chain tried: {tried}"
        )
        raise ProviderError(msg)

    # ── STT ─────────────────────────────────────────────────────────────────

    def resolve_stt(self, language: str, override: str | None = None) -> SttProvider:
        if override:
            return self._resolve_override_stt(override, language)
        chain = self.config.stt.get(language) or self.config.stt.get("default") or ()
        return self._resolve_chain_stt(chain, language)

    def _resolve_override_stt(self, name: str, language: str) -> SttProvider:
        try:
            provider = _registry.get_stt(name)
        except KeyError as e:
            registered = sorted(p.name for p in _registry.list_stt())
            msg = f"stt_provider {name!r} not registered; registered: {registered}"
            raise ProviderError(msg) from e
        caps = provider.capabilities()
        if language not in caps.languages:
            msg = (
                f"stt_provider {name!r} does not support language "
                f"{language!r}; supports: {sorted(caps.languages)}"
            )
            raise ProviderError(msg)
        return provider

    def _resolve_chain_stt(
        self, chain: tuple[str, ...], language: str
    ) -> SttProvider:
        tried: list[str] = []
        for name in chain:
            tried.append(name)
            try:
                provider = _registry.get_stt(name)
            except KeyError:
                log.debug("routing chain skip: stt %r not registered", name)
                continue
            if language in provider.capabilities().languages:
                return provider
            log.debug(
                "routing chain skip: stt %r does not support %r", name, language
            )
        msg = (
            f"no stt provider in routing chain supports language "
            f"{language!r}; chain tried: {tried}"
        )
        raise ProviderError(msg)

    # ── Inspection (for voices_list, ops UX) ────────────────────────────────

    def tts_routing(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self.config.tts.items()}

    def stt_routing(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self.config.stt.items()}
