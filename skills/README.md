# Skills

`skills/` holds agent skills for eremite. Each skill is written without the
name of any agent harness, vendor or model, and follows the standard skill
layout: a short `SKILL.md`, `references/` and `scripts/` where a task needs
them, and `agents/openai.yaml` for interface metadata (display name, short
description and default prompt).

## `image-generation`

`skills/image-generation` guides an agent through generating and editing raster
images against eremite's image models (such as MiniMax-H3-Image):

- `SKILL.md`: when to use raster generation instead of vector or code, the
  workflow for generating, inspecting and saving assets, what the server
  controls, and prose prompt rules (no labeled lines or keyword lists, quoted
  text, no GIFs).
- `agents/openai.yaml`: display name, short description and default prompt.
- `references/editing.md`: multi-reference `<Picture N>` editing rules.
- `references/transparency.md`: prompt guidance for green-key transparent
  cutouts.

## Install

Copy `skills/image-generation` into your agent's skills folder. For Codex:

```sh
cp -R skills/image-generation "${CODEX_HOME:-$HOME/.codex}/skills/"
```

See `docs/server.md` (Codex's image tool) for the Codex `config.toml` lines
that enable the image tool with eremite.
