# aawazz-mcp v1.4 — pipefish-bridged voice loop

**Status:** draft, not yet implemented
**Targets:** v1.4.0 release
**Builds on:** [`SPEC_v1.3.md`](./SPEC_v1.3.md) §14 (research items now graduated, but reshaped per s148 architectural review)
**Companion specs:** [`SPEC.md`](./SPEC.md) (v1.0 baseline); seahorse-side ticket for `TransformersBackend` (filed in `workspace-meta/FOREMAN_THREADS.md`)

---

## 0. Goal

Three TTS-side adds, gated behind a thin LLM-bridge to pipefish:

1. **A single `pipefish` LLM provider** — HTTP client against the captain's existing pipefish server. All backend routing (Ollama / llama.cpp / Gemini / Ravan / future Transformers) happens **inside seahorse/pipefish**, never in aawazz.
2. **Streaming token-to-speech** — first-audio latency drops from *"wait for full response"* (~15 s on CPU 1.7B) to *"wait for first sentence"* (~3-5 s).
3. **`aawazz-converse`** — console script closing the listen → LLM → speak loop with VAD interrupts. Wet-hand UX, hands-free conversational sessions.

### What changed since the s148 draft

The s148 draft proposed building `OpenAICompatLlmProvider` (Ollama / llamacpp / pipefish, each as a separate provider) plus a `TransformersLlmProvider` for in-process HF + PEFT (the bodhi pattern). That contradicts the captain's load-bearing directive (`workspace-meta/FOREMAN_THREADS.md` s130, line 154):

> *"seahorse + rama are the priority. Any new backend goes THROUGH seahorse, not parallel to it. seahorse is already the Sym inference abstraction layer."*

Pipefish (`libs/seahorse/apps/pipefish/rust-server/src/models.rs:20`) already implements the `ModelBackend` trait with five built-in adapters: `LocalBackend` (Seahorse C++ → llama.cpp FFI), `OllamaBackend`, `GeminiBackend`, `RavanBackend` (rama-zpu arena), `LlamaCppBackend`. Multi-turn chat works. **aawazz speaks to pipefish, not to backends.**

The transformers / PEFT loading we proved out in s148 (bodhi adapter on Bonsai-1.7B base) **belongs as a sixth `TransformersBackend` inside pipefish**, not as a parallel provider in aawazz. That's a seahorse-side ticket; aawazz reaches it the same way it reaches every other backend — through pipefish's HTTP `/v1/chat/completions`.

### Drivers from the s148 ship + bodhi experiment

- **Bridge shape was dead simple** even when composed manually — `inference.chat_completion()` + `LocalBackend.speak()` worked in 50 lines, no aawazz changes required. Confirms the §14.1 v1.3 finding that **pure-bridge is sufficient** at the aawazz level.
- **End-to-end latency on CPU is ~15-25 s per turn** (10 s gen + 5-15 s TTS). Streaming first-sentence cuts perceived latency 4–5×.
- **Bodhi-without-system-prompt produced Russian** ("Я Bonsai…") — tiny-tts dutifully synthesized 2.89 s of pure gibberish, no warning. This is the canary for language-mismatch detection.
- **PEFT identity inversion is real** — the system prompt is load-bearing for adapter-flavoured models. The `respond` tool must accept `system_prompt` as first-class, not buried in `extra=`.

### Non-goals

- **Building any inference backend in aawazz.** Even a "fallback" direct-Ollama path is rejected. If pipefish is down, `respond` returns a structured error pointing at pipefish.
- **Audio-native LLMs** (Qwen2-Audio, Qwen2.5-Omni, AudioLM-class). Carries from v1.3 §14.5; revisit in v1.5+ once pipefish/seahorse settles a story for them.
- **Wake-word.** Always-on background process is a separate concern from `aawazz-converse`.
- **Multi-turn persistence.** `aawazz-converse` keeps in-memory conversation; flush on exit.

---

## 1. Architecture

```
                                    ┌─────────────────────────────────┐
   captain types or speaks          │ pipefish (libs/seahorse/.../    │
        │                            │   apps/pipefish/rust-server)    │
        ↓                            │                                 │
   ┌──────────────────┐              │   ModelBackend trait:           │
   │ aawazz-mcp       │              │   - LocalBackend (llama.cpp FFI)│
   │  ┌────────────┐  │   HTTP /v1   │   - OllamaBackend               │
   │  │ Pipefish   │──┼──────────────│   - GeminiBackend               │
   │  │ LlmProvider│  │              │   - RavanBackend (rama-zpu)     │
   │  └────────────┘  │              │   - LlamaCppBackend             │
   │                  │              │   - TransformersBackend (NEW;   │
   │  ┌────────────┐  │              │       seahorse v1.4 ticket)     │
   │  │ TtsProvider│←─┼─ generated   │                                 │
   │  │  + stream  │   text          └─────────────────────────────────┘
   │  └────────────┘
   │  ┌────────────┐
   │  │ Lang-detect│   (gates which TTS provider runs)
   │  │ post-step  │
   │  └────────────┘
   └──────────────────┘
```

aawazz adds ONE LLM provider (`pipefish`) and three TTS-side concerns (streaming, language-detection, conversational loop). All "which model? which backend?" decisions live upstream in pipefish.

---

## 2. New abstraction: `LlmProvider`

A small Protocol surface, sized for the v1.4 needs but designed compatibly with future `LlmProvider`s (cloud APIs, direct Anthropic, etc.) that may legitimately not go through pipefish.

```python
class LlmProvider(Protocol):
    name: str                          # "pipefish" in v1.4.0
    version: str

    def capabilities(self) -> LlmCapabilities: ...
    async def complete(self, request: LlmRequest) -> LlmResult: ...
    async def stream(self, request: LlmRequest) -> AsyncIterator[LlmChunk]: ...
    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class LlmCapabilities:
    available: bool                    # endpoint reachable / model loaded
    requires_network: bool
    supports_streaming: bool
    supports_system_prompt: bool
    backend_models: tuple[str, ...]    # discovered via /v1/models on the endpoint
    notes: str = ""


@dataclass(frozen=True)
class LlmRequest:
    messages: tuple[dict[str, str], ...]   # OpenAI-compat: [{role, content}, ...]
    system_prompt: str | None = None        # convenience; prepended as system role
    model: str | None = None                # let pipefish pick its default if None
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: tuple[str, ...] = ()
    timeout_s: float = 30.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LlmResult:
    text: str
    model: str                         # whatever pipefish reports back
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    finish_reason: str                 # "stop" / "length" / "error"
    language_detected: str | None = None


@dataclass(frozen=True)
class LlmChunk:
    text: str                          # incremental delta since last chunk
    is_final: bool
    language_detected: str | None = None
```

---

## 3. Built-in: `PipefishLlmProvider`

Single concrete provider in v1.4.0. HTTP client (`httpx.AsyncClient`) against `/v1/chat/completions`, SSE streaming.

```python
@register_llm("pipefish")
class PipefishLlmProvider:
    name = "pipefish"

    # Per-instance config:
    #   AAWAZZ_PIPEFISH_URL    = "http://127.0.0.1:11450/v1"  (default)
    #   AAWAZZ_PIPEFISH_MODEL  = unset → pipefish picks; or named model
    #   AAWAZZ_PIPEFISH_TOKEN  = unset (pipefish is local; auth optional)
```

### Reachability semantics

`capabilities()` does a one-shot `GET /v1/models` with a 1 s timeout at first call:

- **Success** → `available=True`, `backend_models=(...)` populated from response.
- **ConnectionRefused / DNS / 5xx** → `available=False`, `backend_models=()`. The router skips us (so `respond` returns a clean error instead of silently hanging).

The probe result caches for 30 s; subsequent calls re-probe lazily. This keeps `voices_list()` cheap while still reflecting pipefish lifecycle changes within a session.

### Streaming

`stream(request)` POSTs `/v1/chat/completions` with `"stream": true`, yields `LlmChunk` per SSE line. Whether the underlying backend can stream or not is pipefish's problem — its OpenAI-compat surface presents streaming uniformly. Captain's pipefish-multi-turn-chat surface (per s111 research, line 341) handles this already.

---

## 4. Streaming token-to-speech

### 4.1 The trade-off

Batch (today): 10 s gen + 6 s TTS = **16 s to first audio**.
Streaming (v1.4): 2 s gen-to-first-sentence + 1 s TTS chunk = **3 s to first audio**, ongoing playback as more text arrives.

### 4.2 `TtsProvider.synthesize_stream`

`TtsProvider` gains an optional streaming path:

```python
async def synthesize_stream(
    self, request: TtsRequest, text_stream: AsyncIterator[str]
) -> AsyncIterator[TtsChunk]: ...

@property
def supports_streaming(self) -> bool: ...
```

```python
@dataclass(frozen=True)
class TtsChunk:
    audio: np.ndarray
    sample_rate: int
    is_final: bool
```

Default: `supports_streaming=False`, `synthesize_stream` raises `NotImplementedError`. Provider authors opt in.

Phase-3 candidates (v1.4.0 ships streaming for):
- **tiny-tts** — synth-per-chunk, write final WAV from concatenated chunks at end.
- **Piper** — already streams natively (`PiperVoice.synthesize` returns an iterable).
- **Kokoro** — `Kokoro.create_stream` is exposed by `kokoro_onnx`.

gTTS (network-based, batch-only by upstream) and XTTS (long context, complex inference) keep `supports_streaming=False`. Dispatcher transparently falls back to batch synthesis after collecting the full LLM output.

### 4.3 Sentence-boundary chunking

The dispatcher reads the LLM token stream, accumulates until it hits a sentence boundary (`.`, `?`, `!`, `\n\n`), flushes the chunk to `TtsProvider.synthesize_stream`. Final chunk on EOS regardless of boundary.

A small `[chunking]` extra adds `pysbd>=0.3` for robust multi-language sentence segmentation. Without it we use a regex fallback that handles English correctly and most Latin-script languages reasonably.

### 4.4 Playback during streaming

When `play=True` and streaming is active, each TtsChunk plays immediately via the `PlaybackProvider`. `paplay` / `aplay` block until done; the dispatcher synthesizes the next chunk in parallel via `asyncio.to_thread` so playback gaps stay small.

---

## 5. Language-mismatch detection

### 5.1 The bug surfaced in s148

Bodhi-without-system-prompt → Russian → tiny-tts (English-only) → 2.89 s of pure gibberish, no warning. Bridge has no checkpoint between LLM output and TTS input.

### 5.2 Fix: optional language detection step

`lingua-language-detector` runs on either the full LLM output (batch) or each streamed chunk (streaming). Three policy options, configurable per call and per server:

| Policy   | Behavior |
|----------|----------|
| `route` (default) | If detected lang ≠ `request.language`, re-resolve TTS via `Router.resolve_tts(detected_lang)`. Falls back to `language` chain if no provider supports `detected_lang`. |
| `warn`   | Synthesize anyway, but include `language_mismatch: {requested, detected}` in the response and log a WARNING. |
| `error`  | Hard-fail with structured `{error: "language mismatch", requested, detected, hint}`. |

Pure routing-chain users who don't install `[langdetect]` opt out — language detection silently no-ops. For `respond` callers, `route` is the v1.4.0 default since it preserves "user gets audio they can understand."

### 5.3 `[langdetect]` extra

```toml
langdetect = ["lingua-language-detector>=2.0"]
```

`lingua-py` chosen over `langdetect` for short-text accuracy (LLM-streaming chunks are ~50 chars on average). Pure-Python, no C deps, ~5 ms / chunk.

---

## 6. New MCP tool: `respond`

Single high-level tool fuses LLM → TTS → playback. Thin pure-bridge surface.

```python
@mcp.tool()
async def respond(
    prompt: str,
    *,
    system_prompt: str | None = None,
    messages: list[dict] | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    tts_provider: str | None = None,
    language: str = "en",
    voice: str | None = None,
    speed: float = 1.0,
    play: bool = False,
    stream: bool = False,
    post_process: list[str] | None = None,
    output_path: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    lang_mismatch: Literal["route", "warn", "error"] = "route",
) -> dict:
    """Generate text via the configured LLM provider, then synthesize and
    optionally play it. Returns ``{text, audio_path, llm_provider, llm_model,
    tts_provider, llm_latency_ms, tts_latency_ms, total_latency_ms,
    language_detected, post_process_chain, ...}``.

    Either ``prompt`` (one-shot) or ``messages`` (multi-turn caller-managed
    state). System prompt is prepended automatically if not present in messages.
    """
```

`respond` does **not** subsume `speak` — `speak` stays the canonical text-in / WAV-out tool. `respond` is the convenience verb for *"have a model say this for me."*

`respond` is **opt-in via extras** — a pip install of `aawazz-mcp[multilingual]` (no LLM extras) doesn't expose `respond` at the MCP surface. This avoids surprising existing v1.3 callers.

---

## 7. `aawazz-converse` console script

A new entry point alongside `aawazz-mcp` and `aawazz-dictate`. Local-only, no MCP surface — round-trip overhead per turn is wrong shape for a tool surface.

### 7.1 Loop

```
captain Super+V (or VAD auto-stop)
  → listen(pre_process=["vad:webrtc"])
  → respond(messages=<history>, system_prompt=<persona>, stream=True)
  → audio plays (streaming chunks)
  → next listen with VAD interrupt detection
```

Multi-turn state lives in-process — `messages` array maintained for the session, flushed on exit. Pipefish handles per-backend chat-template formatting.

### 7.2 Interrupt detection

`vad:webrtc` polls the mic during playback. Voiced frame mid-speech → `paplay` killed (SIGTERM), loop returns to listening. Mid-flight LLM stream is cancelled cleanly via `httpx`'s `aclose()`.

### 7.3 Personas

Squadron agents already carry persona configs. `aawazz-converse --persona foreman` reads `projects/squadron/personas/foreman/persona.yaml` and uses:

- `system_prompt:` → LLM `system_prompt`
- `voice:` → TTS voice ID (e.g. `piper:en_GB-cori-high`)
- `tts_provider:` → optional override

Closes SPEC_v1.3 §14.4 (per-persona voices).

### 7.4 Hotkey integration

The captain's existing Super+V wrapper (`~/.local/bin/aawazz-dictate-toggle`, s147) gets a sibling: `aawazz-converse-toggle`. Same SIGUSR1 graceful-stop pattern.

---

## 8. Routing additions

### 8.1 `RoutingConfig.llm`

```toml
[llm.routing]
default = ["pipefish"]
```

Per-language LLM routing is **out of scope** — LLMs are language-emergent, you pick a model not a "language → model" map. The chain is provider preference order; first reachable wins.

### 8.2 `Router.resolve_llm(override?)`

Mirrors `resolve_tts` / `resolve_stt`. Override hard-fails if not registered. Default chain: try in order, first whose `capabilities().available=True` wins. v1.4.0 ships with one LLM provider so the chain is degenerate — but the Protocol surface is ready for v1.5+ additions (direct Anthropic, direct OpenAI, etc.) that may need to coexist with pipefish.

---

## 9. Tool surface changes

### 9.1 New: `respond` — see §6

### 9.2 `speak` gains streaming

```python
speak(..., stream: bool = False)
```

Streaming primarily consumed by `respond` and `aawazz-converse`. Default `stream=False` so existing v1.3 callers see no change.

### 9.3 `voices_list` adds `providers.llm`

```jsonc
{
  "providers": {
    "tts": [...], "stt": [...], "post_processors": [...],
    "capture": [...], "playback": [...],
    "llm": [
      {
        "name": "pipefish",
        "version": "...",
        "available": true,
        "supports_streaming": true,
        "backend_models": ["llamacpp:qwen3", "ollama:llama3", "ravan:..."],
        "endpoint": "http://127.0.0.1:11450/v1"
      }
    ]
  },
  "routing": {"tts": {...}, "stt": {...}, "llm": {"default": ["pipefish"]}}
}
```

---

## 10. Backwards compatibility

| Surface | v1.3 | v1.4 |
|---------|------|------|
| `speak()` | ✅ | unchanged + new `stream`, defaults preserved |
| `transcribe()` / `listen()` | ✅ | unchanged |
| `voices_list()` v2 | ✅ | gains `providers.llm`, `routing.llm` |
| Default routing config | en→tiny-tts / default→gtts / ne→whisper / default→moonshine | unchanged + `llm.default = ["pipefish"]` (only meaningful when `[llm]` extra installed) |
| `respond()` | n/a | new tool; only registered when an LLM extra is installed |
| Default install (no extras) | text-and-WAV | text-and-WAV; no LLM tools surface, no LLM in voices_list |

`respond` is **opt-in via extras**. v1.3 callers see zero change.

---

## 11. Initial built-in providers (v1.4.0)

| Provider | Stage | Status | Optional extra | Notes |
|----------|-------|--------|----------------|-------|
| `pipefish` | LLM | new | `[llm]` | httpx-based OpenAI-compat client; 1 s reachability probe at first capability call |
| (lingua) | lang-detect post-step | new | `[langdetect]` | short-text language detection |
| (pysbd) | sentence segmentation | new | `[chunking]` | streaming chunk boundaries |

Other LLM providers (direct Anthropic, direct OpenAI cloud, vLLM-direct) are deferred. Adding them in v1.5+ is straightforward via the same `register_llm` decorator — but per the captain's directive, **anything that can reach an inference backend should reach it through pipefish first**, with direct-cloud only as an explicit exception.

---

## 12. Sibling work in seahorse (NOT v1.4 of aawazz)

The s148 bodhi experiment proved out an in-process transformers + PEFT loading recipe that's **load-bearing for pipefish, not for aawazz**. To make bodhi-class adapter models reachable through pipefish, seahorse needs:

1. **`TransformersBackend` adapter** — sixth implementation of the `ModelBackend` trait in `libs/seahorse/apps/pipefish/rust-server/src/models.rs` alongside Local / Ollama / Gemini / Ravan / LlamaCpp. Calls into a small Python sidecar via PyO3 or an out-of-process `transformers-server` (decision left to seahorse-side review).
2. **Adapter config schema** — pipefish needs to know "for model X, base = `prism-ml/Bonsai-1.7B-unpacked`, adapter = `nixprabin/bodhi`, dtype = float16, device = cpu". TOML extension to pipefish's existing model registry.
3. **Reachability discovery** — `/v1/models` should report transformers-loaded checkpoints alongside the existing backends.

This is filed as a separate ticket in `workspace-meta/FOREMAN_THREADS.md` under the seahorse cluster. **aawazz v1.4 doesn't block on it** — pipefish's existing five backends already cover the bridge surface (the bodhi experiment can re-run via pipefish's `LlamaCppBackend` against a quantized Bonsai GGUF if/when one becomes available).

The s148 transformers + PEFT bridge code (`/tmp/bodhi-aawazz-bridge.py`) becomes the prototype seahorse pulls from, not a permanent aawazz file.

---

## 13. Testing strategy

| Test layer | Coverage |
|------------|----------|
| `tests/llm_providers/` | `PipefishLlmProvider` against an httpx mock; reachability cache; capability discovery via mocked `/v1/models` |
| `tests/streaming/` | sentence-boundary chunker; `synthesize_stream` mock provider; first-audio-latency harness |
| `tests/lang_detect/` | `lingua` round-trip on canonical phrases; `route` / `warn` / `error` policy paths |
| `tests/integration/respond/` | end-to-end `respond` against a real running pipefish, gated by env (`AAWAZZ_TEST_PIPEFISH_URL`); skipped in CI by default |
| `tests/converse/` | listen → respond → speak loop with mocked LLM and mocked playback; interrupt detection |
| `tests/back_compat/` | every v1.3 example call still produces the matching response shape |

A small `MockPipefishLlmProvider` in `tests/conftest.py` (deterministic 5-token replies, configurable `available` flag) keeps the routing/streaming tests fast.

---

## 14. Migration plan (implementation phases)

Same shape as v1.3 — each phase ship-able, tag-able, slip-able:

1. **Phase 1 — `LlmProvider` Protocol + registry + `PipefishLlmProvider` + batch `respond`.** No streaming yet. Verify against the captain's running pipefish (`:11450`).
2. **Phase 2 — Streaming token-to-speech.** Sentence chunker + `synthesize_stream` extension on `TtsProvider`. Implement on tiny-tts + Piper + Kokoro. `speak(stream=True)` and `respond(stream=True)` light up.
3. **Phase 3 — Language-mismatch detection.** `[langdetect]` extra + `route` / `warn` / `error` policies.
4. **Phase 4 — `aawazz-converse` console script.** listen → respond → speak loop with VAD interrupts and persona system prompts. Hotkey wrapper (`aawazz-converse-toggle`). Wires into the captain's Super+V workflow (s147).

v1.4.0 releases when phases 1–3 land. Phase 4 can ship as v1.4.1 if the converse-loop UX needs more iteration.

---

## 15. Open questions

| #  | Question | Default if not answered |
| -- | -------- | ----------------------- |
| Q1 | Does `respond` accept `messages=[{role,content}]` or just `prompt: str`? | Both — `messages` for multi-turn, `prompt` for one-shot. |
| Q2 | Should `aawazz-converse` keep multi-turn history? | Yes, in-memory only; flush on session end. |
| Q3 | `lingua-py` vs `langdetect`? | `lingua-py` — better short-text accuracy, no Java runtime. |
| Q4 | Streaming when `output_path` is set and the user wants the full file too — buffer or skip? | Buffer chunks to RAM, write WAV at end. Bounded by `max_tokens`. |
| Q5 | Should `PipefishLlmProvider` support custom auth headers (Bearer tokens)? | Yes, via `extra={"headers": {...}}` per call. |
| Q6 | Reachability cache TTL? | 30 s. Long enough that voices_list doesn't probe per call; short enough that pipefish lifecycle changes are reflected within a session. |
| Q7 | `respond` timeout when pipefish is slow (>30 s first-token)? | Configurable via `timeout_s` per call, default 30 s. |
| Q8 | Streaming chunk-size lower bound — does shorter than ~5 words cause synthesis artifacts? | Spike during phase 2. If yes, accumulate until min-chunk-size hit. |
| Q9 | If pipefish reports `LlamaCppBackend` is loaded but the model file is missing, who reports the error? | Pipefish — surface as 5xx. Aawazz forwards the error string verbatim into the `respond` response. |
| Q10 | `aawazz-converse` interrupt vs. let-llm-finish during user re-speak? | Interrupt — the captain's intent overrides the in-flight response. Cancel mid-stream. |

---

## 16. Future research — v1.5+ directions

### 16.1 Audio-native LLMs

Inherits from v1.3 §14.5 (deferred). Once seahorse settles a story (likely a `MultimodalBackend` extension to pipefish's `ModelBackend` trait), aawazz exposes it via the same `pipefish` provider — no new aawazz-side abstraction needed.

### 16.2 Direct-cloud LLM providers

Captain may eventually want direct Anthropic / OpenAI access from aawazz when pipefish doesn't make sense for the use case (e.g., calling a frontier-class model that pipefish doesn't proxy). The `LlmProvider` Protocol shape supports this — register a second provider, route via `respond(llm_provider="anthropic")`. Per the captain's directive this is an **explicit exception**, not the default — ops decide per-call.

### 16.3 Persistent multi-turn state

`aawazz-converse` keeps in-memory history. A v1.5 add could persist conversations as `~/.local/share/aawazz/sessions/<id>.jsonl` with replay, search, and RAG hooks. Squadron agents already log dialogue separately; we'd reuse that surface or coexist.

### 16.4 Wake-word

"Hey aawazz" needs an always-on background process polling the mic. Different lifecycle than `aawazz-converse` (which the captain triggers via hotkey). A small `aawazz-wakeword` daemon + IPC-handoff to `aawazz-converse` on detection. Likely a separate optional subpackage.

### 16.5 Cross-modal grounding

If the LLM emits a `<emotion>excited</emotion>` token or similar tag, can the TTS layer respond? Out of scope until any of our LLMs naturally emit such tags (or pipefish exposes a structured-event channel alongside token streams).

---

## 17. What this spec is not

- A commitment to ship every listed phase together. Phases 1–3 are v1.4.0; phase 4 (converse) can slip to v1.4.1 if the loop UX needs more iteration.
- A new MCP server. Everything below adds to the existing aawazz-mcp surface; no new transport, no new resource URI.
- An MCP-side conversational-mode tool. `aawazz-converse` is local-only. The MCP `respond` tool is for one-shot LLM→TTS, not stateful sessions.
- A replacement for joker-mcp's LLM routing. Joker stays the squadron-level LLM router; aawazz's `LlmProvider` is for self-contained voice-loop use cases. Joker calling `respond` to vocalize a model output stays composable.
- A path to add inference backends in aawazz. Per the captain's directive: backends live in seahorse. aawazz reaches them through pipefish.

---

## 18. Cross-links

- [`SPEC_v1.3.md`](./SPEC_v1.3.md) §14 — research items reshaped here. §14.5 (audio-native LLMs) carries forward unchanged.
- s148 bodhi experiment (this session) — validated the bridge shape, exposed the language-mismatch bug, and produced the latency numbers. The transformers + PEFT recipe is captured for re-use **inside seahorse**, not aawazz.
- `workspace-meta/FOREMAN_THREADS.md` s130 (line 154) — captain's load-bearing directive: *"Any new backend goes THROUGH seahorse, not parallel to it."*
- `workspace-meta/FOREMAN_THREADS.md` s111 parked threads (lines 339, 343, 345) — pipefish-as-canonical-dispatch convergence: pipefish + ollama dispatch session, capsule inference via pipefish, crush-ide model selection via pipefish. v1.4 of aawazz is the fourth client of the same pattern.
- `libs/seahorse/apps/pipefish/rust-server/src/models.rs` — the `ModelBackend` trait aawazz consumes via HTTP. Source of truth for backend coverage.
- s147 voice stack (`workspace-meta/FOREMAN_THREADS.md` aawazz cluster) — the Super+V hotkey + voice-inbox sink that `aawazz-converse-toggle` plugs into.
- Memory `pipefish_llamacpp_backend` — pipefish ↔ llama-server ↔ Qwen3 setup the captain runs locally.
