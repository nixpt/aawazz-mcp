# aawazz-mcp assets

Branding artifacts go here. Populated interactively — drop generated files in.

## Drop-in slots

| File | Purpose | Reference in repo |
|---|---|---|
| `logo.svg` | Primary logo, vector. Used in README header. | [`README.md`](../README.md) |
| `logo.png` | Raster fallback (1024×1024) for places that don't render SVG (PyPI, some Markdown renderers). | [`README.md`](../README.md), eventual PyPI listing |
| `logo-dark.svg` | Optional dark-mode variant. GitHub renders the right one based on viewer theme via `<picture>` tags. | optional |
| `social-card.png` | 1280×640 OpenGraph / Twitter card for the GitHub repo "social preview" setting. | repo settings → Options → Social preview |
| `icon.png` | 256×256 favicon-style square; useful for client config UIs that show an icon next to the server entry. | optional |

## Suggested prompt for image-gen

> A minimalist logo for "aawazz" — Hindi/Urdu/Nepali word आवाज़ meaning *voice / sound*. Two paired motifs: an open mouth on the left and a listening ear on the right, connected by a sound wave. Flat geometric style, single accent color (suggested: warm coral or ochre — references the South Asian script tradition without being literal). Square 1024×1024 frame, clean negative space, looks good as both a 32×32 favicon and a 256×256 client-config icon. Avoid: realistic faces, microphone clipart, generic speech bubbles.

Tweak as you iterate. The Devanagari glyph **आवाज़** itself is a strong design element — a calligraphic treatment may work better than an illustrated mouth/ear pair. Worth trying both directions.

## Once dropped

Update `README.md` to reference the logo at the top of the elevator section:

```markdown
<p align="center">
  <img src="assets/logo.svg" width="160" alt="aawazz logo">
</p>
```

And set `social-card.png` via the repo's *Settings → General → Social preview* upload.

## License

Whatever you generate / commission belongs under the same MIT license as the rest of the repo unless you note otherwise inline.
