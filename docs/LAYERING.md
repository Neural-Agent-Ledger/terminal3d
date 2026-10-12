# Layering

The rules that decide which crate may depend on which. They keep the engine usable by
any game, keep the editor out of shipped game binaries, and keep the terminal and the
audio device behind one door each.

## The three layers

```
   apps/*  ─────────►  engine/*  ◄─────────  editor/*
 (runnable consumers)  (the reusable runtime)  (authoring and dev tooling)
```

Arrows point from the crate that depends to the crate it uses.

| Directory | Holds | Crates today |
|---|---|---|
| `engine/` | The reusable runtime: data types, level format, renderer, present layer, audio, effects, play-session systems | `re_core`, `re_world`, `re_level`, `re_puzzle`, `re_render`, `re_present`, `re_audio`, `re_postfx`, `re_glitch`, `re_asciifx`, `re_runtime`, `ts_net` (experimental) |
| `editor/` | Authoring and development tools. They are never linked into a game's normal build | `ts_editor`, `re_lvlcheck`, `re_voximport`, `re_voxscene`, `re_imgqa`, `re_golden` |
| `apps/` | Runnable programs built only on the engine | `ts_netplay` |

Every crate sits exactly two directories below the repository root
(`engine/<crate>`, `editor/<crate>`, `apps/<crate>`). Tests find `levels/` and
`assets/` through `CARGO_MANIFEST_DIR/../..`, so a new crate must follow the same
depth.

## Dependency rules

| From | May depend on (normal and build) | May depend on (dev) |
|---|---|---|
| `engine/*` | `engine/*`, external crates | `engine/*`, external crates |
| `editor/*` | `engine/*`, `editor/*`, external crates | same |
| `apps/*` | `engine/*`, external crates | same |

And for a game built on this engine (for example
[terminal-state](https://github.com/Neural-Agent-Ledger/terminal-state)):

| From | Normal and build | Dev |
|---|---|---|
| a game crate | `engine/*`, external crates | `engine/*` and `editor/*` (for example `re_golden`, `re_lvlcheck`) |

Further rules:

- **Nothing here depends on anything outside this repository.** Every `path =` in a
  manifest resolves inside the repo root.
- **Engine never depends on editor or apps, not even as a dev-dependency.** If an
  engine test needs a helper that lives in `editor/`, move the helper down into the
  engine or write the test in the editor crate.
- **The engine never names a game.** No engine crate knows a host's state enum, its
  menus, its save format or its settings. `re_runtime` plugins are generic over the
  host's `S: States` and receive the "a level is live" value from the host
  (`ZoomPlugin { live: MyState::Playing }`). Values the host owns cross the seam as
  plain resources the host writes: `FieldContext`, `FieldSignals` and
  `re_audio::FieldSettings`. Runtime crates never read a host's settings file.
- **One Bevy.** A game's resolved dependency tree must contain exactly one
  `bevy_ecs`. Keep the Bevy version in `[workspace.dependencies]` in step with the
  games that use this engine.

## Boundary rules inside the engine

- **Only `re_present` touches the terminal.** No other crate depends on crossterm,
  ratatui or bevy_ratatui. `clippy.toml` lists their types under `disallowed-types`
  as a second line of defence. Everything else is pure compute and compiles and tests
  with no terminal attached.
- **Only `re_audio::playback` opens an audio device.** Synthesis, the clock, the
  director, acoustics and sound fields are pure and unit-tested. With no device, or
  with `TS_AUDIO_SILENT=1`, playback is a silent no-op.
- **`Framebuffer` and `Cell` belong to `re_core`.** Every stage reads and writes that
  one type. Extend it there; never redefine it.
- **`re_level` owns the `.lvl` format.** The game loads levels, so the format cannot
  live under `editor/`. The writer (`to_lvl_text`) currently lives in
  `re_voximport`; moving it into `re_level` is on the [roadmap](../ROADMAP.md).
- **The editor may drive a player, never link one.** `ts_editor`'s playtest command
  spawns a separate process: `$TS_PLAY_CMD` if set, otherwise a sibling
  `terminal-state` or `terminal3d-play` binary next to the editor, otherwise it asks
  you to set `TS_PLAY_CMD`. (`terminal3d-play` does not exist yet; see the roadmap.)

## The graph-puzzle kit (why `node`, `pipe` and `door` appear in the engine)

The `.lvl` format has sections for a small directed graph: `[node]`, `[pipe]`,
`[door]`, `[pedestal]`, `[inputs]`, `[sequence]`. `re_puzzle` evaluates that graph
with a topological forward pass (switches toggle edges, threshold nodes fire, a door
opens when its target node matches). The renderer draws graph state (lit pipes,
node sprites, pipe text), and the audio crate has sounds for it.

This vocabulary is **part of the engine**, not a game leaking into it. It is the
engine's built-in graph-puzzle kit, and a second game can use it or ignore it. A
level with no graph sections renders and plays as a walkable scene.

Some names in the lower crates still read as one specific puzzle rather than as a
generic kit. They are listed under "good first issues" in the
[roadmap](../ROADMAP.md): `re_core::pool::{NodePool, pipe_count, pool_for_pipe}`,
`re_core::trigger::activate_node`, `re_world::{at_door, at_door_within}`,
`re_render::scene_text::with_pipe_pool`,
`re_render::ascii_shader::MaterialRegistry::push_puzzle_builtins` and
`re_audio::SoundId::{Door, Pipe}`. Renaming them, or documenting each one as part of
the kit, is welcome.

## How the rules are checked

- `cargo xtask layers` classifies every workspace member by its first path segment
  (`engine`, `editor`, `apps`, `xtask`) and checks every internal edge, including
  dev edges, against the table above. The same check runs as a unit test inside
  `cargo test --workspace`.
- `cargo clippy --workspace --all-targets` denies `disallowed_types`, which catches
  terminal types outside `re_present`.
- Games check their own side. For example the game repo asserts that its normal
  dependencies contain no `editor/` crate and exactly one `bevy_ecs`, and that no
  networking crate reaches its build.

If a change needs an edge the table forbids, stop and open an issue first. The usual
fix is a seam: a trait, a registry or a resource that the higher layer fills.
`re_render::scene_text` (text panels supplied by the host) is the model to follow.
