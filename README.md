# aawazz-mcp

> **आवाज़** — Hindi/Urdu/Nepali for *voice / sound*.

A portable, local-CPU **TTS + STT MCP server** for any agent runtime that speaks the Model Context Protocol — Claude Code, Claude Desktop, Codex, Cursor, Zed, Cline, Continue, Goose, Gemini CLI. One `pip install` and four tools (`speak`, `transcribe`, `listen`, `voices_list`) light up across every runtime simultaneously. Bundles [tiny-tts](https://github.com/backtracking/tiny-tts) (~3.4 MB ONNX) and [Useful Sensors / Moonshine](https://github.com/usefulsensors/moonshine) (~80 MB ONNX) so it runs offline once weights are cached; an optional `--remote` mode delegates to a separately-running FastAPI mouth/ears so model load doesn't double up on machines that already have those services.

---

## Status & support matrix

| | |
|---|---|
| **Version** | v1.0 (first release) |
| **OS** | Linux, macOS — Moonshine ships only `.so` and `.dylib` |
| **Python** | 3.10, 3.11, 3.12 |
| **Distribution** | `pip install git+...` (PyPI release queued for v1.1) |
| **Transport** | stdio default; opt-in `streamable-http` |

**Windows users**: install [WSL2](https://learn.microsoft.com/windows/wsl/install) and follow the Linux instructions inside it. Native Windows support is gated on Moonshine shipping `.dll` artifacts — see [usefulsensors/moonshine#TBD](https://github.com/usefulsensors/moonshine).

---

## Install

```bash
pip install git+https://github.com/nixpt/aawazz-mcp.git
```

That's it. The `aawazz-mcp` console script lands on your `PATH`. PyPI publication is queued for v1.1; until then the git URL is the canonical install path.

---

## Quickstart

1. **Verify the install.** The CLI prints its flags:

   ```bash
   aawazz-mcp --help
   ```

2. **Wire it into one runtime.** Claude Code is the canonical example; drop this into `~/.claude.json` and restart Claude Code:

   ```json
   {
     "mcpServers": {
       "aawazz": {
         "command": "aawazz-mcp",
         "args": []
       }
     }
   }
   ```

3. **(Optional) Pre-warm models.** First-run downloads tiny-tts, Moonshine, and a couple of NLTK datasets totalling ~130 MB plus a torch dep weighing ~600 MB on disk. On offline / air-gapped boxes, run the prefetch script while you still have a network:

   ```bash
   python -m aawazz_mcp.scripts.prefetch_models
   ```

   Models live in standard upstream caches (`~/.cache/huggingface/hub/`, `~/.cache/moonshine_voice/`, `~/nltk_data/`) — no `~/.cache/aawazz/` directory is invented.

Once Claude Code reloads, ask the model to *"say hello using aawazz"* and you should get a `.wav` back.

---

## Multi-runtime config grid

Every runtime below talks to `aawazz-mcp` over stdio. The bare-minimum invocation is identical (`command: aawazz-mcp`, no args); the differences live in **where the config file lives** and **how the runtime picks up changes**. Copy-paste blocks below match exactly the files in [`examples/clients/`](examples/clients/) — pull from there if your editor mangles JSON.

The optional `env:` table on each block enables [hybrid mode](#hybrid-mode-advanced) — uncomment if you have an `aawazz-mouth` / `aawazz-ears` FastAPI pair already running on this host.

### Claude Code

| | |
|---|---|
| **Config file** | `~/.claude.json` (user-level) or `.mcp.json` (per-project, repo root) |
| **Restart** | Manual — quit & relaunch the Claude Code CLI |
| **Example** | [`examples/clients/claude_code.json`](examples/clients/claude_code.json) |

```json
{
  "mcpServers": {
    "aawazz": {
      "command": "aawazz-mcp",
      "args": []
      // "env": {
      //   "AAWAZZ_MOUTH_URL": "http://127.0.0.1:7861/tts",
      //   "AAWAZZ_EARS_URL": "http://127.0.0.1:7862/transcribe"
      // }
    }
  }
}
```

**Gotchas**
- `~/.claude.json` is shared with other Claude Code state — merge into the existing `mcpServers` table, don't overwrite the file.
- Per-project `.mcp.json` overrides user-level for the cwd; useful for pinning hybrid-mode env to a workspace.

### Claude Desktop

| | |
|---|---|
| **Config file** | Linux: `~/.config/Claude/claude_desktop_config.json` · macOS: `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Restart** | Manual — fully quit Claude Desktop and relaunch (menu bar quit on macOS, not just the window close) |
| **Example** | [`examples/clients/claude_desktop.json`](examples/clients/claude_desktop.json) |

```json
{
  "mcpServers": {
    "aawazz": {
      "command": "aawazz-mcp",
      "args": []
    }
  }
}
```

**Gotchas**
- If `aawazz-mcp` isn't on the GUI app's `PATH` (common on macOS), use the absolute path: `which aawazz-mcp` from your shell, then paste the result into `command`.
- Claude Desktop on macOS sandboxes the subprocess; the bundled mic-capture path (`listen`) may not have access. Check `voices_list().capabilities.listen`.

### Codex (OpenAI Codex CLI)

| | |
|---|---|
| **Config file** | `~/.codex/config.toml` |
| **Restart** | None — Codex auto-detects on the next `codex` invocation |
| **Example** | [`examples/clients/codex.toml`](examples/clients/codex.toml) |

```toml
[mcp_servers.aawazz]
command = "aawazz-mcp"
args = []

# [mcp_servers.aawazz.env]
# AAWAZZ_MOUTH_URL = "http://127.0.0.1:7861/tts"
# AAWAZZ_EARS_URL = "http://127.0.0.1:7862/transcribe"
```

**Gotchas**
- TOML — table form `[mcp_servers.<name>]`, not the JSON `mcpServers: { ... }` shape used elsewhere.
- `args = []` is required even when empty; Codex won't infer it.

### Cursor

| | |
|---|---|
| **Config file** | `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per-project) |
| **Restart** | Auto-reload on save — Cursor watches the file |
| **Example** | [`examples/clients/cursor.json`](examples/clients/cursor.json) |

```json
{
  "mcpServers": {
    "aawazz": {
      "command": "aawazz-mcp",
      "args": []
    }
  }
}
```

**Gotchas**
- Cursor supports `${env:VAR}` and `${workspaceFolder}` interpolation in command/args/env values — useful for per-machine paths.
- On some setups the GUI launcher's `PATH` doesn't include user-installed scripts; if `aawazz-mcp` isn't found, swap to the absolute path from `which aawazz-mcp`.

### Zed

| | |
|---|---|
| **Config file** | `~/.config/zed/settings.json` |
| **Restart** | Auto-applied — Zed reloads MCP servers on settings save |
| **Example** | [`examples/clients/zed.json`](examples/clients/zed.json) |

```json
{
  "context_servers": {
    "aawazz": {
      "command": "aawazz-mcp",
      "args": []
    }
  }
}
```

**Gotchas**
- The top-level key is `context_servers`, **not** `mcpServers` — Zed uses its own naming. The inner shape (command / args / env) matches.
- Zed's settings are merge-friendly; you can drop `context_servers` next to your existing `theme`/`buffer_font_size`/etc. without losing them.

### Cline (VS Code)

| | |
|---|---|
| **Config file** | UI-driven — Cline panel → MCP Servers → Add. Persisted in VS Code's MCP storage. |
| **Restart** | Click the per-server **Restart Server** button in the Cline UI |
| **Example** | [`examples/clients/cline.json`](examples/clients/cline.json) |

```json
{
  "command": "aawazz-mcp",
  "args": [],
  "alwaysAllow": [],
  "disabled": false
}
```

**Gotchas**
- Cline is primarily UI-driven; the JSON above is what the dialog produces under the hood. Easiest path is to add via the panel rather than hand-editing.
- `alwaysAllow` is a Cline-specific allowlist of tool names the agent can call without per-call approval — leaving it `[]` keeps the human-in-the-loop prompt.

### Continue

| | |
|---|---|
| **Config file** | `~/.continue/config.yaml` (user-level) or `.continue/config.yaml` (per-project) |
| **Restart** | Hot-reload on save |
| **Example** | [`examples/clients/continue.yaml`](examples/clients/continue.yaml) |

```yaml
modelContextProtocolServers:
  - name: aawazz
    type: stdio
    command: aawazz-mcp
    # env:
    #   AAWAZZ_MOUTH_URL: http://127.0.0.1:7861/tts
    #   AAWAZZ_EARS_URL: http://127.0.0.1:7862/transcribe
```

**Gotchas**
- YAML, not JSON — top-level key is `modelContextProtocolServers` (a list of mappings), not `mcpServers`.
- Continue supports `${{ secrets.X }}` interpolation; if you already keep your remote URLs in the secrets store, prefer that over inline values.

### Goose

| | |
|---|---|
| **Config file** | Linux/macOS: `~/.config/goose/config.yaml` · Windows-WSL: `%APPDATA%\Block\goose\config\config.yaml` |
| **Restart** | Manual — restart the `goose` CLI after editing |
| **Example** | [`examples/clients/goose.yaml`](examples/clients/goose.yaml) |

```yaml
extensions:
  aawazz:
    cmd: aawazz-mcp
    args: []
    enabled: true
    type: stdio
    timeout: 300
    # envs:
    #   AAWAZZ_MOUTH_URL: http://127.0.0.1:7861/tts
    #   AAWAZZ_EARS_URL: http://127.0.0.1:7862/transcribe
```

**Gotchas**
- Goose uses `cmd` (not `command`) and `envs` (not `env`) — copy-paste from another runtime's block won't work, use the exact keys above.
- `timeout` is in seconds and applies to tool calls — bump it (e.g. `timeout: 600`) if first-run model download bites you.
- `enabled: false` keeps the extension registered but inactive — handy when toggling between hybrid and bundled mode without deleting the block.

### Gemini CLI

| | |
|---|---|
| **Config file** | `~/.gemini/settings.json` (user-level) or `./.gemini/settings.json` (per-project) |
| **Restart** | Auto-detected on next `gemini` invocation |
| **Example** | [`examples/clients/gemini_cli.json`](examples/clients/gemini_cli.json) |

```json
{
  "mcpServers": {
    "aawazz": {
      "command": "aawazz-mcp",
      "args": []
    }
  }
}
```

**Gotchas**
- Same `mcpServers` shape as Claude Code — copy-paste interchangeable.
- Gemini CLI's folder-trust prompt may block stdio spawn on first run; accept the trust prompt or pass `--skip-trust` (see [`agent-launch`](https://github.com/nixpt/squadron) for an example wrapper).

---

## Hybrid mode (advanced)

If you have `aawazz-mouth` and `aawazz-ears` running as separate FastAPI services on this host (typical setup: systemd-user units on `:7861` and `:7862`), you don't want this MCP server to load its own copies of tiny-tts and Moonshine on every runtime spawn. Hybrid mode delegates to those FastAPI services instead.

**Why bother**
- Existing FastAPI services stay useful — other clients on the same host can keep hitting them in parallel.
- Model load happens once, in the long-running service — not per MCP-runtime subprocess.
- ~600 MB of torch state is held in one place, not duplicated across every Claude Code / Cursor / Zed window.

**How**

CLI flag form (joint base for both services, overrideable per-service via env):

```bash
aawazz-mcp --remote http://127.0.0.1:7861,http://127.0.0.1:7862
```

Per-service env-var form (independent overrides for mouth and ears):

```bash
AAWAZZ_MOUTH_URL=http://127.0.0.1:7861/tts \
AAWAZZ_EARS_URL=http://127.0.0.1:7862/transcribe \
  aawazz-mcp
```

In an MCP-runtime config, drop the env table into the server entry — see the commented block in any of the [`examples/clients/*`](examples/clients/) files.

**`listen` is always local.** The mic lives on the MCP server's host; tunneling raw audio over the FastAPI request/response cycle isn't a clean fit and no shipping runtime in scope consumes streaming partials. Use `transcribe` against a pre-recorded WAV if you need remote STT.

**Failure mode is fail-loud.** If `--remote` is set and the FastAPI server is unreachable, the tool returns a structured error instead of silently falling back to local:

```json
{
  "error": "remote aawazz-mouth at http://127.0.0.1:7861/tts unreachable: connection refused",
  "hint": "is aawazz-mouth running? `systemctl --user status aawazz-mouth`. Or pass --no-remote.",
  "backend": "remote",
  "url": "http://127.0.0.1:7861/tts"
}
```

Silent fallback would mask misconfig; you explicitly opted into remote. v1.1 may add `--remote-fallback=local` if there's demand.

---

## Tools

Four tools and one resource. Tool docstrings become MCP tool descriptions verbatim — what your agent sees in `tools/list` mirrors the contracts below.

### `speak(text, voice="MALE", speed=1.0, output_path=None, play=False)`

Render text to speech and write a `.wav`. Returns `{audio_path, duration_s, sample_rate, latency_ms, voice, speed, text_hash, played, backend}`.

- `text` — required, 1–4000 chars.
- `voice` — tiny-tts ships only `"MALE"`. Anything else returns a structured error with `available_voices: ["MALE"]`.
- `speed` — 0.5–2.0 multiplier.
- `output_path` — absolute path; default `~/.local/share/aawazz/mouth/<ts>-<hash>.wav`.
- `play` — autoplay via `paplay` / `aplay` / `afplay` if any are on `PATH`.

Example (mcp-inspector):
```
speak({"text": "Hello world", "voice": "MALE", "play": true})
```

### `transcribe(audio_path, language="en", model_arch="tiny_streaming")`

Transcribe a WAV file (local path or `http(s)://` URL). Returns `{text, audio_duration_s, sample_rate, latency_ms, model_arch, language, audio_path, backend}`.

- `audio_path` — absolute path or `http(s)://` URL. URL inputs are downloaded to `${TMPDIR}/aawazz-stt-<sha8>.wav` and unlinked after.
- `model_arch` — `tiny | tiny_streaming | base | base_streaming | small_streaming | medium_streaming`. Default `tiny_streaming` is the recommended balance of speed/accuracy for English on CPU.

Example:
```
transcribe({"audio_path": "/tmp/note.wav", "language": "en"})
```

### `listen(duration_s=5.0, language="en", model_arch="tiny_streaming", save_audio=False)`

Bounded mic capture, then transcribe. Returns same shape as `transcribe` plus `audio_path: str | None` (only set when `save_audio=true`). `backend` is always `"local"`.

- `duration_s` — 0.5–30.0 hard cap.
- `save_audio` — keep the captured WAV at `~/.local/share/aawazz/ears/<ts>.wav`.

Example:
```
listen({"duration_s": 4.0, "save_audio": true})
```

### `voices_list()`

Cheap probe — does **not** load models. Returns `{tts: {backend, voices}, stt: {backend, languages, model_archs}, capabilities: {listen, play, backend_mode, remote_url}}`. Use it to discover whether `listen` will work on the current host (mic-less / sandboxed runtimes report `capabilities.listen: false`).

Example:
```
voices_list()
```

### Resource: `aawazz://health`

Returns JSON `{models_loaded: {tts, stt_archs}, mode, remote_url, version}`. Read it from the resource panel in your MCP runtime, or via `mcp-inspector` `resources/read aawazz://health`.

Full contract: [`SPEC.md` §1](SPEC.md).

---

## Caveats

- **Linux + macOS only.** Moonshine ships only `libmoonshine.so` and `libonnxruntime.*.dylib`; native Windows is gated upstream. WSL2 works.
- **First-run network requirement.** Three caches get populated on first call: `~/.cache/huggingface/hub/` (tiny-tts G.pth, ~50 MB via HF hub), `~/.cache/moonshine_voice/` (~80 MB ONNX), `~/nltk_data/` (`g2p_en`, `cmudict`, `averaged_perceptron_tagger_eng`). Run `python -m aawazz_mcp.scripts.prefetch_models` ahead of time on offline / air-gapped boxes.
- **One TTS voice.** `tiny-tts` ships exactly one (`MALE`). Asking for a different voice returns a structured error — no silent downgrade. Multi-voice TTS is deferred to v1.1 (Moonshine TTS).
- **`listen` needs a mic + audio server.** `sounddevice` requires PulseAudio / PipeWire / CoreAudio access. Headless boxes, SSH sessions without `--enable-audio`, and most Docker containers will see `voices_list().capabilities.listen: false`. The tool itself returns a clean error rather than crashing.
- **torch dependency.** `tiny-tts` pulls PyTorch — ~600 MB on disk. v1.1 may carve out a `[lite]` extras for users who only need STT.
- **stdio-safe by construction.** All logging goes to stderr; tiny-tts's stdout `print()` is wrapped in a `redirect_stdout` context manager. Don't add `print()` calls to source — see the lint rule in `pyproject.toml`.

---

## Development

```bash
git clone https://github.com/nixpt/aawazz-mcp
cd aawazz-mcp
pip install -e ".[dev]"
pytest
```

Test markers (configured in `pyproject.toml`):

- `@pytest.mark.slow` — model-load round-trips. Run with `pytest -m slow`; skip in CI default.
- `@pytest.mark.remote` — requires the FastAPI mouth/ears running on `:7861` / `:7862`. Auto-skip when unreachable.
- `@pytest.mark.mic` — requires a working audio input device. Auto-skip when `sounddevice.query_devices(kind="input")` is empty.

Lint: `ruff check src tests`. Format: `ruff format src tests`. The ruff config bans `print()` in `src/` to keep stdio safe.

---

## License & credits

MIT — see [`LICENSE`](LICENSE).

Bundled / linked work, each under its own license:

- **[Useful Sensors / Moonshine](https://github.com/usefulsensors/moonshine)** — MIT — STT model + ONNX runtime wrapper.
- **[backtracking / tiny-tts](https://github.com/backtracking/tiny-tts)** — TTS model + inference.
- **[Anthropic / `mcp` Python SDK](https://github.com/modelcontextprotocol/python-sdk)** — MIT — `FastMCP` server framework, the protocol layer.
