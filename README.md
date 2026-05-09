# aawazz-mcp

> **आवाज़** (Hindi/Urdu — *voice / sound*)

A portable, local-CPU **TTS + STT MCP server** for any agent runtime — Claude Code, Codex, Cursor, Zed, Cline, Continue, Goose. One `pip install`, zero cloud calls.

- 🗣️ **`speak`** — text-to-speech via [tiny-tts](https://github.com/backtracking/tiny-tts) (~3.4 MB ONNX)
- 👂 **`transcribe`** — file-based STT via [Moonshine](https://github.com/usefulsensors/moonshine) (~80 MB ONNX, MIT, Useful Sensors)
- 🎤 **`listen`** — push-to-talk mic capture → transcription
- 📋 **`voices_list`** — voice / language / model catalog + capability probe

Bundle-by-default. Optional `--remote` mode delegates to a separately-running FastAPI server (e.g. captain's `aawazz-mouth` / `aawazz-ears` systemd-user services).

---

## Status

**v1.0** — first release. Linux + macOS only (Moonshine ships only `.so` and `.dylib`). Windows users: WSL recommended.

## Install

```bash
pip install git+https://github.com/nixpt/aawazz-mcp.git
```

PyPI release is queued for v1.1.

## Quickstart

After install, the `aawazz-mcp` console script is on your PATH. Run it to verify:

```bash
aawazz-mcp --help
```

Then add it to your MCP-runtime config (see below). On first invocation, models are downloaded lazily into:

- `~/.cache/huggingface/hub/` (tiny-tts via `huggingface_hub`)
- `~/.cache/moonshine_voice/` (Moonshine ONNX weights)
- `~/nltk_data/` (NLTK `g2p_en`, `cmudict`, `averaged_perceptron_tagger_eng`)

To pre-warm everything before first use:

```bash
python -m aawazz_mcp.scripts.prefetch_models
```

## Multi-runtime config grid

> See `examples/clients/` for copy-paste files.

### Claude Code (`~/.claude.json`)

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

### Cursor (`~/.cursor/mcp.json`)

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

### Zed (`~/.config/zed/settings.json`)

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

### Codex (`~/.codex/config.toml`)

```toml
[mcp_servers.aawazz]
command = "aawazz-mcp"
args = []
```

### Continue (`~/.continue/config.yaml`)

```yaml
modelContextProtocolServers:
  - name: aawazz
    type: stdio
    command: aawazz-mcp
```

### Cline (VS Code MCP UI)

```json
{
  "command": "aawazz-mcp",
  "args": []
}
```

### Goose (`~/.config/goose/config.yaml`)

```yaml
extensions:
  aawazz:
    cmd: aawazz-mcp
    args: []
    enabled: true
    type: stdio
    timeout: 300
```

> _Wave 1D will replace this section with the polished, runtime-by-runtime detail (config-file paths, restart behavior, gotchas)._

## Hybrid mode (advanced)

If you already run [`aawazz-mouth`](https://github.com/nixpt/aawazz-mcp/blob/main/SPEC.md) and `aawazz-ears` as separate FastAPI services and don't want this MCP server to load its own model copies, point at them:

```bash
aawazz-mcp --remote http://127.0.0.1:7861,http://127.0.0.1:7862
# or
AAWAZZ_MOUTH_URL=http://127.0.0.1:7861/tts \
AAWAZZ_EARS_URL=http://127.0.0.1:7862/transcribe \
  aawazz-mcp
```

`listen` always runs locally — the mic is on the MCP server's host.

## Tools

> _Auto-generated tool reference will land in Wave 2 from FastMCP-introspected schemas. See `SPEC.md` for the v1.0 contract._

## Development

```bash
git clone https://github.com/nixpt/aawazz-mcp
cd aawazz-mcp
pip install -e ".[dev]"
pytest
```

## License

MIT — see `LICENSE`. Bundled models keep their own licenses (tiny-tts: see upstream; Moonshine: MIT).

## Credits

Built on:
- [Useful Sensors / Moonshine](https://github.com/usefulsensors/moonshine) — MIT
- [backtracking / tiny-tts](https://github.com/backtracking/tiny-tts)
- [Anthropic / `mcp` Python SDK](https://github.com/modelcontextprotocol/python-sdk)
