# aawazz-mcp assets

Brand artifacts for the project. All MIT-licensed alongside the rest of the repo.

## What's here

| File | Purpose | Used in |
|---|---|---|
| `banner.png` | Wide-format brand mark — mic icon + "aawazz / TTS · STT · MCP / आवाज़ / voice for every agent runtime". | [`README.md`](../README.md) header |
| `logo.png` | Square primary logo — mic + radial sound waves + "aawazz / आवाज़ / mcp" stack. | PyPI listing (when v1.1 publishes), GitHub social card candidate |
| `icon.png` | Rounded-square app-icon style — bidirectional sound waves (mouth + ears symmetry). | Client-config UIs that show an icon next to the server entry, favicon source |

All three are 1024×1024, JPEG-encoded inside `.png` containers. Re-export to true PNG (or SVG) if a downstream consumer requires it.

## Color palette

- Background: deep navy / black (`#0a0b1f` ish)
- Primary accent: cyan-to-purple gradient (`#3ecde6` → `#8a5cf6`)
- Devanagari script accent: pink/magenta (`#e36ad9` ish)
- Wordmark: white

## Reuse

Embed in markdown:

```markdown
<p align="center">
  <img src="assets/banner.png" alt="aawazz-mcp banner" width="100%">
</p>
```

Or for a tighter logo-only header:

```markdown
<p align="center">
  <img src="assets/logo.png" alt="aawazz logo" width="200">
</p>
```

## License

MIT — same as the rest of the repo. Generated/commissioned art is dedicated under the same terms unless noted otherwise inline.
