# CLAUDE.md

Guidance for Claude Code (and other coding assistants) working in this repository.

terminal3d is a first-person 3D engine that renders into a terminal: a Rust cargo
workspace on **headless Bevy 0.19** (MinimalPlugins + ScheduleRunnerPlugin +
bevy_state; no wgpu, winit or `bevy_render`), **ratatui / crossterm** through
`bevy_ratatui`, and **kira** for audio. The raycaster, shading, materials, synth,
puzzle kit, editor and tools are original code.

Read `README.md` for the overview, `docs/ARCHITECTURE.md` for the structure,
`docs/LAYERING.md` for the dependency rules and `CONTRIBUTING.md` for what PRs are
held to. This file is the short operational version.

## Layout

```
engine/   the reusable runtime        re_core re_world re_level re_puzzle re_render re_present
                                      re_audio re_postfx re_glitch re_asciifx re_runtime ts_net
editor/   authoring and dev tools     ts_editor re_lvlcheck re_voximport re_voxscene re_imgqa re_golden
apps/     programs built on engine/   ts_netplay
levels/   example and fixture levels  assets/ authoring/  generated fixtures and samples
tests/golden/  golden hash store      docs/  architecture, layering, level format, materials, lighting, emitters
```

Every crate is exactly two directories below the root; tests reach `levels/` via
`CARGO_MANIFEST_DIR/../..`. A new crate must keep that depth.

## Rules you must not break

1. **Layering.** `engine/*` depends only on `engine/*` and external crates, even in
   dev-dependencies. `editor/*` may use `engine/*` and `editor/*`. `apps/*` may use
   `engine/*` only. Nothing depends on a path outside this repository.
2. **Only `re_present` touches the terminal.** No crossterm, ratatui or bevy_ratatui
   anywhere else; `clippy.toml` bans their types. Everything else must compile and
   test with no terminal.
3. **Only `re_audio::playback` opens an audio device.**
4. **The engine never names a game.** No game state enum, menu, save or settings
   type in engine code. Seams are traits, registries, or resources the host writes;
   `re_runtime` plugins are generic over the host's `S: States`.
5. **`Framebuffer` / `Cell` are owned by `re_core`.** Extend them there; never
   redefine them.
6. **No `unsafe`.** `unsafe_code = "forbid"` workspace-wide. The only exception is
   the existing OS FFI in `re_present`. Do not add more.
7. **Golden hashes must not move by accident.** Never re-bless to make a failure go
   away (see below).

## Commands

The root is a virtual workspace, so always pass `-p`.

```sh
cargo build --workspace
cargo test -p <crate>                                  # prefer crate-scoped runs
cargo test --workspace --no-fail-fast
cargo clippy --workspace --all-targets                 # must show no `error:`
cargo test -p re_golden --test engine_corpus           # golden framebuffer check
cargo run --release -p re_lvlcheck -- levels/          # lint levels
cargo run --release -p re_imgqa -- levels/22-lit-cellar.lvl   # headless render report
cargo run -p re_render --example turn_shots -- <out_dir>      # plain-glyph frames to files
cargo xtask layers                                     # layering check
```

**Always set these when running tests or tools from a non-interactive shell:**

- `TS_AUDIO_SILENT=1`: never open an audio device, unless the user explicitly wants
  sound.
- `TS_SAVE_DIR=<fresh temp dir>`: keeps any save or settings write away from the
  user's real profile.

`ts_editor` (and `ts_netplay`'s client) take over the terminal and need a real TTY.
Do not start them from a non-interactive shell; ask the user to run them.

## Verifying visual changes

You cannot see the terminal, so verify with frames written to text:

1. Before the change, capture the affected levels: `turn_shots`, a `renderqa` run,
   or the reference game's `terminal-state --dump <level> --size 100x32` if the user
   has it built (`--ansi` when colour matters).
2. Make the change, capture again, and diff.
3. Describe what moved and show the relevant rows. "It compiles" is not a visual
   check.
4. For motion, input or timing, say plainly that it needs a live run in a real
   terminal, and ask the user to do it.

## Golden tests

`re_golden` compares a shade digest and a geom digest per curated level against
`tests/golden/engine_fb_hashes.txt`.

- Shade moved and geom did not: a look change. Both moved: treat it as a raycaster
  regression until proven otherwise.
- **Never bless (`TS_BLESS_GOLDEN=1`) without first showing the user before/after
  frames for every moved level and getting agreement.** Never bless to silence an
  unexpected failure, a missing level or a missing hash; those are coverage holes.
- When a bless is agreed, commit the store with the change and add a dated entry to
  `tests/golden/README.md` naming the levels and the reason.

## Working rules

- **Bug fixes: write the failing test first**, confirm it fails, then fix.
- **Cross-crate changes:** if you change a public signature, grep for and update
  every caller in the workspace in the same change.
- **Clippy `all` is deny.** Do not add `#[allow]` to hide a real finding. Tests may
  allow `float_cmp` at the crate root, as existing crates do. Do not add new pedantic
  warnings in code you write.
- **Match the surrounding style.** The tree is not uniformly rustfmt-formatted; do
  **not** run `cargo fmt` on the workspace or on whole files.
- **Docs:** new or moved modules get a `//!` header (purpose, where the crate sits in
  the layering); public items get `///`. Comments say *why*, not *what*.
- **Determinism:** render, synth, director and effect paths are seeded and
  reproducible. No wall-clock time or unseeded randomness there.
- **Dependencies:** prefer a maintained crate over hand-rolling a subsystem, but
  every new dependency needs a stated reason, must be headless (no wgpu, winit,
  `bevy_render`) and must pass `deny.toml`.
- **No downloaded assets.** Never fetch images, models, fonts or audio from the web,
  even for tests. Assets are self-made, generated by code here, or CC0 / CC BY with
  the licence file committed, and every file under `assets/`, `authoring/` and
  `levels/` has a `PROVENANCE.toml` entry. Sounds are synth recipes.
- **Levels must pass `lvlcheck`.**
- **Do not weaken tests to make them pass.** If a premise is wrong, explain why and
  change it deliberately.
- **Commits are signed off** (`git commit -s`); the project uses the DCO.

## Where to look

| Working on | Read first |
|---|---|
| Rendering, performance | `docs/ARCHITECTURE.md`, `re_render` crate docs, `engine/re_render/examples/perfbench.rs` |
| Materials, glyphs | `docs/MATERIALS.md`, `engine/re_render/src/glyph_family.rs` |
| Lights, particles | `docs/LIGHTING.md`, `docs/EMITTERS.md` |
| Sky | `docs/SKY.md`, `engine/re_render/src/sky/` |
| Level format | `docs/LEVEL_FORMAT.md`, `re_level` crate docs |
| `.vox` import | `docs/VOX_IMPORT.md`, `editor/re_voximport` crate docs |
| Audio, sound fields | `re_audio` crate docs (`engine/re_audio/src/lib.rs`) |
| Play session, HUD probe | `re_runtime` crate docs (`engine/re_runtime/src/lib.rs`) |
| Editor | `editor/ts_editor/src/keymap.rs` (the single binding table) |
| Layering | `docs/LAYERING.md` |
| Golden tests | `tests/golden/README.md` |
| What to work on | `ROADMAP.md` |
