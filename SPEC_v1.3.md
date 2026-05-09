# aawazz-mcp v1.3 — pluggable audio backends

**Status:** draft, not yet implemented
**Targets:** v1.3.0 release
**Supersedes:** the hardcoded TTS routing in `LocalBackend.speak` shipped in v1.2.x
**Companion:** [`SPEC.md`](./SPEC.md) (v1.0 baseline still authoritative for the four-tool surface)

---

## 0. Goal

Make every stage of the audio pipeline — TTS synthesis, STT recognition, mic capture, post-processing, playback — pluggable via a small provider interface, so adding a new model means writing one class, not editing `LocalBackend`.

### Concrete drivers

- Today, English speak is locked to tiny-tts (one voice, one language). Non-English speak is locked to gTTS (network-dependent, no voice cloning).
- Captain wants to experiment with Piper, Kokoro, XTTS, Whisper variants, and future small models without forking aawazz or maintaining a parallel server.
- Joker / squadron consumers want to route text from any LLM through a chosen voice without the aawazz tool surface fighting them.

### Non-goals

- **Streaming synthesis.** v1.3.0 stays batch-only (return a complete WAV). Add to v1.4 if a provider justifies it.
- **GPU autodetect.** Providers declare their compute requirements; ops choose what to install. No magic CUDA probing.
- **Voice cloning UX.** XTTS supports cloning from a reference WAV; that capability is exposed as a provider-specific kwarg, not generalized into the core interface.
- **Cross-provider voice equivalence.** "MALE" on tiny-tts and "en_US-amy-medium" on Piper are not interchangeable; we namespace voice IDs (§6) rather than try to unify them.

---

## 1. Pipeline stages and abstractions

The audio pipeline has five logical stages. v1.3 introduces a provider interface for each. The first two are mandatory; the last three start with a single built-in and a clean extension point.

```
                    ┌──────────────┐         ┌──────────────┐
   speak(text)  →   │ TtsProvider  │   →     │ PostProcessor│   →  WAV
                    └──────────────┘         └──────────────┘
                    ┌──────────────┐         ┌──────────────┐
   listen()     →   │ Capture      │   →     │ PostProcessor│   →   ┐
                    └──────────────┘         └──────────────┘       │
                                                                    ↓
                                                            ┌──────────────┐
                                                            │ SttProvider  │  →  text
                                                            └──────────────┘
   transcribe(wav) →                                                 ↑
                                                                     ┘
                    ┌──────────────┐
   play=True    →   │ Playback     │
                    └──────────────┘
```

### 1.1 `TtsProvider`

```python
class TtsProvider(Protocol):
    name: str                          # "tiny-tts" / "piper" / "gtts" / "kokoro" / "xtts"
    version: str                       # provider version, not aawazz version

    def capabilities(self) -> TtsCapabilities: ...
    async def synthesize(self, request: TtsRequest) -> TtsResult: ...
    async def aclose(self) -> None: ...  # best-effort resource release


@dataclass(frozen=True)
class TtsCapabilities:
    languages: frozenset[str]          # {"en", "es", ...}; empty = all (Bark)
    voices: tuple[VoiceCatalogEntry, ...]
    requires_network: bool
    sample_rate: int                   # native rate; dispatcher resamples if needed
    accepts_dsp_profiles: bool         # gates whether DSP post-process is offered
    speed_range: tuple[float, float]   # (min, max); (1.0, 1.0) for fixed-speed providers
    notes: str = ""                    # free-form, surfaced in voices_list


@dataclass(frozen=True)
class TtsRequest:
    text: str
    language: str = "en"
    voice: str | None = None           # see §6 voice IDs
    speed: float = 1.0
    extra: dict = field(default_factory=dict)  # provider-specific kwargs (e.g. xtts speaker_wav)


@dataclass(frozen=True)
class TtsResult:
    audio: AudioBuffer                 # see §7 — always normalized to PCM int16/float32
    sample_rate: int
    duration_s: float
    latency_ms: int
```

### 1.2 `SttProvider`

```python
class SttProvider(Protocol):
    name: str                          # "moonshine" / "whisper" / "vosk"
    version: str

    def capabilities(self) -> SttCapabilities: ...
    async def transcribe(self, request: SttRequest) -> SttResult: ...
    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class SttCapabilities:
    languages: frozenset[str]          # supported language codes
    model_archs: tuple[str, ...]       # provider-specific arch names; opaque to dispatcher
    accepts_url: bool                  # provider can pull http(s) URLs itself
    cold_load_seconds_estimate: float  # ops display
    notes: str = ""


@dataclass(frozen=True)
class SttRequest:
    audio: AudioBuffer | str           # buffer OR file path OR URL (if accepts_url)
    language: str = "en"
    model_arch: str | None = None      # provider default if None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SttResult:
    text: str
    language_detected: str | None      # for providers that auto-detect
    audio_duration_s: float
    sample_rate: int
    latency_ms: int
    confidence: float | None           # if provider exposes it
```

### 1.3 `PostProcessor`

A unary `(audio, sr) → audio` transform. Composes left-to-right via a pipeline list. Used both for TTS output (e.g., DSP voice profiles) and STT input (e.g., VAD trim, noise gate, gain normalize).

```python
class PostProcessor(Protocol):
    name: str                          # "dsp:DEEP" / "vad:webrtc" / "gain:auto"
    direction: Literal["tts", "stt", "both"]

    def process(self, audio: NDArray, sample_rate: int) -> NDArray: ...
```

The existing eight DSP profiles in `audio/dsp.py` graduate into post-processors named `dsp:MALE` / `dsp:DEEP` / etc. They're TTS-direction. Providers with `accepts_dsp_profiles=False` skip them silently (with an info-level log).

### 1.4 `CaptureProvider`

Mic capture. Initial built-in is sounddevice (today's `audio/capture.py`). Plugin surface exists so ops who want PortAudio direct, PipeWire native, or ffmpeg subprocess can swap. Not expected to see heavy use in v1.3.

```python
class CaptureProvider(Protocol):
    name: str

    async def record(
        self, duration_s: float, sample_rate: int, save_path: Path | None
    ) -> CaptureResult: ...
```

### 1.5 `PlaybackProvider`

Plays a WAV. Initial built-in is the existing paplay/aplay/afplay shell-out. Same plugin shape as capture; included for completeness.

---

## 2. Registration and discovery

Two registration paths. Both write into the same in-process registry; conflicts (two providers claiming the same `name`) raise at server start.

### 2.1 Built-in providers (in-tree)

`src/aawazz_mcp/providers/__init__.py` imports each shipped provider module. Adding a new built-in is one import + one decorator:

```python
# src/aawazz_mcp/providers/piper.py

from aawazz_mcp.registry import register_tts

@register_tts(name="piper")
class PiperTtsProvider(TtsProvider):
    ...
```

### 2.2 Third-party providers (external packages)

Python entry points. Any pip-installed package can publish:

```toml
# pyproject.toml of a third-party plugin
[project.entry-points."aawazz.tts_providers"]
my-fancy-tts = "my_pkg:MyFancyTtsProvider"
```

aawazz-mcp scans `aawazz.tts_providers`, `aawazz.stt_providers`, `aawazz.post_processors`, `aawazz.capture_providers`, `aawazz.playback_providers` at startup. Discovery cost is one `importlib.metadata.entry_points()` call (~ms).

### 2.3 Lazy import

Provider classes are imported eagerly to register their `name`/`capabilities()`. Heavy model loading stays inside the provider's `synthesize` / `transcribe` first call (today's lazy pattern). Cold-load cost still belongs to first invocation, not server startup.

---

## 3. Routing

### 3.1 Per-language preference list

Server config (CLI / env / config file) specifies a preference order per stage:

```toml
# ~/.config/aawazz/aawazz.toml
[tts.routing]
en      = ["piper", "tiny-tts"]
es      = ["piper", "gtts"]
ja      = ["piper", "gtts"]
ne      = ["xtts", "gtts"]
default = ["gtts"]                 # fallback for languages not listed

[stt.routing]
en      = ["moonshine"]
es      = ["moonshine", "whisper"]
ne      = ["whisper"]
default = ["whisper"]
```

The dispatcher iterates the list, picks the first registered + capability-compatible provider, and routes. If none match, returns the v1.0 structured error format `{error, hint, available_providers, ...}`.

### 3.2 Per-call override

`speak()` and `transcribe()` gain optional `tts_provider` / `stt_provider` parameters. Caller-specified provider must exist and support the requested language/voice; otherwise structured error.

```python
speak({"text": "Hola", "language": "es", "tts_provider": "xtts"})
```

### 3.3 CLI flag

`aawazz-mcp --tts-default piper --stt-default whisper` overrides the per-language routing for any unmatched language.

### 3.4 Env vars

`AAWAZZ_TTS_PROVIDER`, `AAWAZZ_STT_PROVIDER`, `AAWAZZ_TTS_ROUTING_FILE` for the toml above. Env overrides config; CLI overrides env; per-call overrides CLI.

### 3.5 Resolution order

```
per-call provider arg → CLI flag → env var → config file → built-in default
                                                          (en→tiny-tts, other→gtts)
```

Built-in default exists so `pip install aawazz-mcp && aawazz-mcp` keeps working with zero config. Backwards-compat (§10).

---

## 4. Tool surface changes

### 4.1 `speak`

Adds two optional parameters; existing callers unchanged:

```python
speak(
    text: str,
    voice: str = "MALE",                      # see §6 voice IDs
    speed: float = 1.0,
    output_path: str | None = None,
    play: bool = False,
    language: str = "en",
    # v1.3 additions:
    tts_provider: str | None = None,          # override routing
    post_process: list[str] | None = None,    # ["dsp:DEEP", "gain:auto"]
)
```

Response shape adds `provider: str` and `post_process_chain: list[str]`. Backward-compat: existing fields unchanged.

### 4.2 `transcribe`

```python
transcribe(
    audio_path: str,
    language: str = "en",
    model_arch: str = "tiny_streaming",
    # v1.3 additions:
    stt_provider: str | None = None,
    pre_process: list[str] | None = None,     # ["vad:webrtc", "gain:auto"]
)
```

Response adds `provider: str`, `pre_process_chain: list[str]`.

### 4.3 `listen`

Same additions as `transcribe`, plus optional `capture_provider`. Defaults preserve v1.2.x behavior.

### 4.4 `voices_list`

Becomes the canonical capabilities discovery surface:

```jsonc
{
  "providers": {
    "tts": [
      {
        "name": "tiny-tts",
        "version": "0.3.2",
        "languages": ["en"],
        "voices": [{"id": "tiny-tts:MALE", "language": "en", "default": true}],
        "requires_network": false,
        "accepts_dsp_profiles": true,
        "speed_range": [0.5, 2.0]
      },
      {
        "name": "piper",
        "version": "1.2.0",
        "languages": ["en", "es", "fr", "de", "..."],
        "voices": [
          {"id": "piper:en_US-amy-medium", "language": "en"},
          {"id": "piper:en_GB-jenny-medium", "language": "en"},
          {"id": "piper:es_ES-davefx-medium", "language": "es"}
        ],
        "requires_network": false,
        "accepts_dsp_profiles": true,
        "speed_range": [0.5, 2.0]
      }
    ],
    "stt": [
      {"name": "moonshine", "languages": ["en", "es", "..."], "model_archs": ["tiny", "base", "..."]},
      {"name": "whisper",    "languages": ["ne"],            "model_archs": ["small", "medium", "large-v3"]}
    ],
    "post_processors": [
      {"name": "dsp:DEEP", "direction": "tts"},
      {"name": "vad:webrtc", "direction": "stt"}
    ]
  },
  "routing": {
    "tts": {"en": ["piper", "tiny-tts"], "es": ["piper", "gtts"], "...": "..."},
    "stt": {"en": ["moonshine"], "ne": ["whisper"], "...": "..."}
  },
  "capabilities": {"listen": true, "play": true, "backend_mode": "local"}
}
```

This subsumes v1.0's `voices_list`. Old shape (`tts.voices`, `stt.languages`) preserved as a top-level alias for back-compat (§10).

---

## 5. Provider lifecycle

```
register   — at import time, provider class enters the registry
discover   — at server start, registry scans built-ins + entry-points
warm       — optional: --warm flag eagerly loads default providers' models
synthesize — lazy first-call loads the model (still the v1.0 pattern)
aclose     — provider releases resources on server shutdown
```

`Backend.warm()` (existing in `base.py`) becomes a thin wrapper that walks the routing table's defaults and calls each provider's eager-load entry point.

---

## 6. Voice ID convention

Voice IDs become namespaced: `<provider>:<voice_id>`. Examples:

```
tiny-tts:MALE                     # current MALE
piper:en_US-amy-medium            # Piper, US English, Amy, medium quality
piper:en_GB-jenny-medium
kokoro:af_bella                   # Kokoro, American Female, Bella
xtts:cloned-from-/path/ref.wav    # XTTS clone request via extra={speaker_wav: ...}
gtts:default                      # gTTS one voice per language
```

Unprefixed legacy IDs (`MALE`, `DEEP`, ...) resolve under tiny-tts (back-compat). DSP profiles stop being voice IDs (they were a v1.2 hack); they move to `post_process=["dsp:DEEP"]`. The unprefixed `voice="DEEP"` keeps working as a back-compat alias for `voice="tiny-tts:MALE"` + `post_process=["dsp:DEEP"]`.

---

## 7. Audio format normalization

Different providers emit different formats. The dispatcher normalizes at the provider boundary:

| Stage              | Output to dispatcher           | Dispatcher action                                  |
| ------------------ | ------------------------------ | -------------------------------------------------- |
| `TtsProvider`      | PCM `np.ndarray` + sample rate | Resample to caller's requested rate if specified; otherwise pass through. Always write final WAV via `soundfile` PCM-16. |
| `gTTS`             | MP3 bytes (current)            | Provider transcodes to PCM internally (already shipped in v1.2.1). |
| `Coqui XTTS`       | numpy waveform                 | Native — no transcode.                              |
| `Piper`            | PCM int16 stream               | Concatenate to final buffer.                        |
| `SttProvider` input | WAV path or `np.ndarray`      | Provider declares which it accepts; dispatcher converts as needed. |

Outcome: the file at `audio_path` always has a true RIFF/WAVE header and matches the declared sample rate. The bug class fixed in v1.2.1 doesn't reappear.

---

## 8. Initial built-in providers (v1.3.0)

| Provider     | Stage | Status     | Optional extra | Notes                                                     |
| ------------ | ----- | ---------- | -------------- | --------------------------------------------------------- |
| `tiny-tts`   | TTS   | existing   | (default)      | English MALE only; DSP-compatible.                        |
| `gtts`       | TTS   | existing   | `[gtts]`       | Network. Many languages. No voice selection per language. |
| `piper`      | TTS   | **new**    | `[piper]`      | ~100 MB ONNX/voice, ~30 languages, multi-voice, fast CPU. |
| `kokoro`     | TTS   | **new**    | `[kokoro]`     | ~330 MB, 8 voices, English-only, very fast.               |
| `xtts`       | TTS   | **new**    | `[xtts]`       | ~2 GB, 17 languages, voice cloning via reference WAV.     |
| `moonshine`  | STT   | existing   | (default)      | Languages per current `moonshine_voice` distribution.     |
| `whisper`    | STT   | existing   | `[whisper]`    | HF transformers pipeline. Currently registered for `ne`. |
| `dsp:*`      | post  | existing   | (default)      | 7 profiles graduating from `audio/dsp.py`.                |
| `vad:webrtc` | post  | **new**    | `[vad]`        | webrtcvad. Used by listen for auto-stop (cross-link: v1.3 VAD ticket). |
| `gain:auto`  | post  | **new**    | (default)      | Pure-numpy peak-normalize.                                |
| `sounddevice`| capture| existing  | (default)      | Today's `record_to_wav_hard_timeout`.                     |
| `paplay`     | playback| existing | (default)      | paplay/aplay/afplay shell-out wrapper.                    |

Total optional extras: `[piper]`, `[kokoro]`, `[xtts]`, `[vad]`, plus existing `[gtts]`, `[whisper]`. Combined "everything" extra: `[all]`.

The v1.2.x `[multilingual]` extra alias stays as `gtts + whisper` (back-compat).

---

## 9. Routing defaults shipped in v1.3.0

The default config — the one a fresh `pip install aawazz-mcp && aawazz-mcp` runs with — preserves v1.2.x behavior exactly:

```toml
[tts.routing]
en      = ["tiny-tts"]
default = ["gtts"]

[stt.routing]
ne      = ["whisper"]
default = ["moonshine"]
```

Captain (or any user) opts into Piper / Kokoro / XTTS by either:

- installing the extra (`pip install aawazz-mcp[piper]`) and overriding routing, or
- running with `--tts-default piper`, or
- using the `tts_provider` per-call override.

Rationale: zero surprise for existing users; opt-in to model variety.

---

## 10. Backwards compatibility

| Surface            | v1.2.x behavior                       | v1.3 behavior                                                                                |
| ------------------ | ------------------------------------- | -------------------------------------------------------------------------------------------- |
| `speak({voice: "MALE"})` | tiny-tts MALE                   | tiny-tts MALE (legacy ID resolves to `tiny-tts:MALE`)                                        |
| `speak({voice: "DEEP"})` | tiny-tts MALE + DSP DEEP        | tiny-tts MALE + post_process=[dsp:DEEP] (transparent rewrite)                                |
| `speak({language: "es"})` | gTTS es                         | Default route: gTTS. New route configurable.                                                 |
| `transcribe({language: "ne"})` | Whisper                     | Default route: Whisper.                                                                       |
| `voices_list()` response | flat tts/stt sections           | new `providers` and `routing` keys; old keys preserved as a flattened alias view.            |
| `--warm` flag       | warms tiny-tts + Moonshine            | warms whatever the resolved default-route providers are.                                      |

No breaking changes for callers using only documented v1.0 / v1.2 surfaces.

---

## 11. Testing strategy

| Test layer            | Coverage                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| `tests/registry/`     | `register_tts` / `register_stt` accept-and-reject paths; conflict detection; entry-point discovery |
| `tests/routing/`      | per-language preference resolution; CLI/env/config layering; per-call override; missing-provider error |
| `tests/providers/<name>/` | One smoke per provider (synthesize → check WAV header + duration > 0). All marked `@pytest.mark.slow`. |
| `tests/post_process/`  | DSP profile registry passthrough; gain:auto peak normalize; VAD trim shape                       |
| `tests/integration/`   | End-to-end: speak(piper, en) → post_process(dsp:DEEP) → file → transcribe(moonshine, en); ditto for xtts/kokoro round-trips |
| `tests/back_compat/`   | All v1.0 / v1.2 example tool calls still produce matching response shapes                         |

Mock providers in `tests/conftest.py` make the routing/registry tests fast (no real model loads).

---

## 12. Migration plan (implementation phases)

Implementation lands across multiple PRs to keep each diff reviewable:

1. **Phase 1 — abstractions and registry.** `providers/base.py` (the Protocols), `registry.py`, entry-points scanning, `voices_list` v2 response. No new providers yet. The existing tiny-tts + Moonshine + gTTS + Whisper code wraps as built-in providers without changing behavior.
2. **Phase 2 — routing layer.** Replace `LocalBackend.speak`'s if/else language fork with the routing chain. Server-time CLI flags + env vars. Per-call override on tools.
3. **Phase 3 — Piper.** First new provider. Drives any holes in the abstraction.
4. **Phase 4 — Kokoro and XTTS.** Round out the TTS shipping set.
5. **Phase 5 — post-process pipeline.** Move DSP into post-processors; add `gain:auto` and `vad:webrtc`. Wire `post_process=` parameter on `speak` / `pre_process=` on `transcribe`/`listen`.
6. **Phase 6 — capture/playback pluggability.** Lower priority; in this phase only because it closes the spec.

Each phase is tagged-ship-able. v1.3.0 releases when phases 1–5 land; capture/playback (phase 6) can defer to v1.3.1 if time-pressured.

---

## 13. Open questions

| #  | Question                                                                                        | Default if not answered                              |
| -- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Q1 | Streaming synthesis — punt to v1.4 or include now?                                              | Punt to v1.4.                                        |
| Q2 | Should `voice="DEEP"` (legacy) auto-rewrite to `post_process=["dsp:DEEP"]` or be a deprecation warning? | Auto-rewrite, no warning. Deprecation in v1.4.       |
| Q3 | For XTTS voice cloning, where does the reference WAV live? Per-call only, or registered cloned voices? | Per-call only via `extra={speaker_wav: ...}`.       |
| Q4 | Do third-party providers need a CLA / license disclosure surface?                               | No — entry-point discovery is opt-in by `pip install`. |
| Q5 | Should the post-process pipeline support cross-stage (e.g., a DSP profile that runs on STT input)? | Yes — `direction="both"` in `PostProcessor` Protocol. |
| Q6 | Do we expose per-provider warmth state (loaded vs lazy) in `voices_list`?                        | Yes — small `loaded: bool` field per provider.       |
| Q7 | When per-call `tts_provider` is unavailable, fall back to routing chain or hard-fail?           | Hard-fail with structured error. No silent fallback (consistent with v1.0 hybrid policy §4 of SPEC.md). |
| Q8 | Should DSP profiles work on Piper / Kokoro output, not just tiny-tts?                            | Yes — DSP is provider-agnostic post-process. Provider opts-out via `accepts_dsp_profiles=False` only if the model already does style control. |

---

## 14. Future research — local LLM integration

Out of scope for v1.3 implementation, but the v1.3 abstractions are designed so the directions below can land in v1.4+ without rework. Each item lists what v1.3 must keep open for the integration to be cheap later.

### 14.1 Direct local-LLM endpoints (Ollama / llama.cpp / pipefish)

The captain runs three flavours of local text-LLM server already: **Ollama** (`127.0.0.1:11434`, OpenAI-compat `/v1`), **llama.cpp** (`/v1` and native), and **pipefish** (`127.0.0.1:11450` fronting `llama-server` 11451 with Qwen3). Today they're consumed via joker-mcp's routing or by hand-written HTTP. The integration question is whether aawazz should call them directly or stay text-only.

Two shapes worth prototyping in v1.4:

- **`LlmProvider` Protocol** — sibling to `TtsProvider` / `SttProvider`. One built-in subclass each for Ollama / llama.cpp / pipefish (all OpenAI-compat → one HTTP client, three URL configs). New MCP tool `respond(prompt, llm_provider?, tts_provider?, language?, play?) → audio_path + text` fuses LLM→TTS in one round-trip.
- **Pure-bridge alternative** — no `LlmProvider` in aawazz at all; just a `respond` tool that takes an HTTP URL + OpenAI-compat schema and pipes the streamed text to `speak()`. Thinner; offloads model selection to the caller.

**Open Q:** does aawazz duplicate joker's routing or stay deliberately ignorant and let joker compose? Probably the latter — aawazz remains a voice surface, joker remains the LLM router — but the `respond` shape is worth prototyping to see if the ergonomics warrant it.

**v1.3 keeps this open by:** the routing chain (§3) is per-stage; adding a fourth stage (`llm.routing`) is a strict extension. The post-process pipeline (§1.3) accepts text/audio dual-direction — a future "stream LLM tokens to TTS chunks" composer fits.

### 14.2 Streaming token-to-speech

Today `speak()` is batch-only. For local-LLM integration, perceived latency matters: synthesizing the first sentence as soon as the LLM emits it cuts time-to-first-audio from "wait for full response" to "wait for first sentence break". Most local servers stream by default (`stream: true` in OpenAI-compat); the question is whether aawazz exposes a streaming `speak`/`respond` or chunks at the dispatcher.

**v1.3 keeps this open by:** Q1 in §13 explicitly defers streaming. Adding it doesn't require breaking changes — `TtsResult` becomes a union with an `AsyncIterator[TtsChunk]` variant.

### 14.3 Conversational mode (listen → local-LLM → speak loop)

The captain's wet-hand UX is currently push-to-talk-only (`aawazz-dictate` v1.1.2 + Super+V hotkey). A natural extension is a held-open conversational session: VAD-gated listen → captain finishes → local-LLM responds → speak the answer → VAD-gated listen again. Same primitives, new orchestrator.

This composes the v1.3 VAD post-processor (`vad:webrtc`) with §14.1's `respond` and an interrupt detector. It belongs in `aawazz-converse` — a new console script alongside `aawazz-mcp` and `aawazz-dictate` — not as an MCP tool, since the loop is local-only and the MCP overhead per turn is wrong shape.

**v1.3 keeps this open by:** capture and post-process pluggability (§1.4, §1.3) means `aawazz-converse` consumes the same providers as the MCP server. No code duplication.

### 14.4 Voice identity per persona

Sibling research, lower priority. Squadron agents (foreman, codex, kiro, antigravity, …) currently share whatever default voice the captain configures. With Piper / Kokoro / XTTS providing voice variety, each persona could carry a `voice_id` in its persona.yaml; aawazz reads it via a new `persona` extra param on `speak`. Small surface, real ergonomic win once the captain wants speaking-named-agent context.

**v1.3 keeps this open by:** namespaced voice IDs (§6) and per-call `tts_provider` override mean a persona just specifies `voice="piper:en_US-amy-medium"` and the routing resolves it.

### 14.5 Audio-native LLMs (see-also, deferred)

Distinct from §14.1 — Qwen2-Audio / Qwen2.5-Omni / AudioLM-class models fuse text and speech in one stack and are 7B+ parameters, GPU-shaped. Tracked here for completeness; integration design (fused `AudioLmProvider`? wrapping `TtsProvider`+`SttProvider`?) is a v1.5+ question once the local-text-LLM patterns are settled. Not the captain's near-term framing per s148 clarification.

---

## 15. What this spec is not

- A commitment that all listed providers (Piper / Kokoro / XTTS) ship together. If one's integration drags, it can slip out of v1.3.0 without affecting the others — they're independent provider modules behind the same registry.
- A promise of behavioral parity between providers. Voice quality, latency, and feature support vary widely; users compose per-language routes that match their preferences.
- A replacement for `joker_text_to_speech`. Joker's TTS arm stays (per the s145 ship note); aawazz and joker remain parallel surfaces with overlapping coverage. Cross-routing between the two is a v1.4+ design.

---

## 16. Cross-links

- Existing baseline: [`SPEC.md`](./SPEC.md) §4 (hybrid mode dispatcher) — the local/remote split this spec layers on top of.
- v1.2.0 multilingual: [`README.md`](./README.md) → "Voice profiles & multilingual" section — what the registry will replace.
- v1.3 VAD parking-lot ticket: `workspace-meta/FOREMAN_THREADS.md` → `aawazz v1.3 — VAD-based auto-stop for wet-hand UX` — converges with phase 5 (`vad:webrtc` post-processor).
- Whisper drift / brand-name OOD memory: `~/.claude/projects/-home-nixp-WORKSPACE/memory/aawazz_brand_stt_drift.md` — informs the test phrase choices in phase 3+ smoke tests.
