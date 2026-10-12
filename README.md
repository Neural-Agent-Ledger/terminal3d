# terminal3d

A first-person 3D engine that renders into your terminal. Every frame is ray-cast
from a voxel world into text glyphs with 24-bit foreground and background colour,
and every sound is synthesized at runtime. There is no GPU, no window and no audio
file.

terminal3d is written in Rust on **Bevy 0.19** (headless ECS: `MinimalPlugins`, no
`bevy_render`, no `wgpu`, no `winit`), **ratatui / crossterm** for the terminal and
**kira** for audio. It ships with a terminal level editor, a level linter, a
MagicaVoxel importer, a headless render-quality report and golden-framebuffer
regression tests.

The game [TERMINAL STATE](https://github.com/Neural-Agent-Ledger/terminal-state), a
first-person puzzle game, is built on it.

## Try it in five minutes

You need Rust from [rustup](https://rustup.rs) (on Windows also the MSVC build
tools; on Linux `pkg-config libasound2-dev libudev-dev`). Then, in Windows Terminal
or any modern Linux terminal of at least 120×40:

```sh
git clone https://github.com/Neural-Agent-Ledger/terminal3d
cd terminal3d
cargo run --release -p ts_editor -- levels/22-lit-cellar.lvl   # open the level editor
```

The editor shows the level as a 2D grid next to a live 3D preview of the same
level. Press `?` for every key binding (also searchable with `:`), and `s` to save:
every save is checked by `lvlcheck`, and `V` opens its report. `F5` play-tests the
level in first person once a player is available: build
[TERMINAL STATE](https://github.com/Neural-Agent-Ledger/terminal-state) and set
`TS_PLAY_CMD` to its `terminal-state` binary.

To check that everything works on your machine:

```sh
cargo test --workspace                          # all tests
cargo test -p re_golden --test engine_corpus    # rendering matches the reference frames
cargo run --release -p re_lvlcheck -- levels/   # every example level is valid
```

On Linux/macOS run the tests with `TS_AUDIO_SILENT=1` set if you have no audio
device. The full setup is under [Quick start](#quick-start).

## What a frame looks like

A frame is a grid of terminal cells. Each cell carries a glyph, a truecolor
foreground and a truecolor background. The renderer picks all three per cell from
the surface it hit: depth, face orientation, material recipe, lights, fog and sky.
Near walls read as dense textured glyphs (brick seams, circuit traces, scan lines),
floors as a sparse world-pinned grain that fades with distance, and open sky as a
procedural field (stars, aurora, clouds, data rain and others). Silhouette edges get
sub-cell block glyphs, so shapes stay crisp at terminal resolution.

To see one, open any example level in the editor, whose right-hand pane is the live
renderer:

```sh
cargo run --release -p ts_editor -- levels/22-lit-cellar.lvl
```

Games built on the engine can print a single frame to stdout with no terminal
takeover; see [How a game uses the engine](#how-a-game-uses-the-engine).

## Features

**Rendering** (`re_render`, `re_postfx`, `re_glitch`)
- Continuous DDA voxel raycaster with floor and ceiling casting, multi-storey maps,
  stairs, ramps and 45-degree wedges; pitch up to about ±60 degrees.
- Tiered composite: cached far panorama, near layer recast only when something
  changes, back-to-front transparency, per-frame emitters. An idle camera costs
  almost nothing.
- Procedural ASCII materials authored per level: noise, fBm, Worley, brick, circuit,
  scanline, Truchet, fractal, cosine palettes, animation, glyph families and
  super-glyph selectors. Image textures with height relief.
- Point lights, key light, ambient, height fog, shadows, a distance-depth effect and
  an Oklab legibility floor, so dark scenes stay readable.
- Procedural skies, particle emitters (several motion kinds, trails, word particles,
  light-gated dust, colour-only light shafts), warp portals, holograms, mirrors,
  walls of flowing code and sign banners.
- A deterministic post-process stack (`off`, `clean`, `subtle`, `cinematic`, `toon`,
  `crt`, `dream`) and a separate deterministic glitch layer.

**Terminal and input** (`re_present`)
- Truecolor with a 256-colour fallback. One write per frame, a dirty-diff that sends
  only changed cells, and an optional colour deadband that trades colour steps for
  bandwidth on slow terminals.
- Keyboard, mouse-look and gamepad (`gilrs`), all delivered as backend-neutral events.

**Audio** (`re_audio`)
- Deterministic synth feeding kira on a musical clock: BPM grid, beat quantization,
  drone beds, a generative director.
- Room acoustics, spatial voices, and per-level sound fields bound to level objects.

**World, levels and play** (`re_world`, `re_level`, `re_puzzle`, `re_runtime`, `re_asciifx`)
- Voxel map, camera, collision and gravity, with coyote time and a jump buffer.
- A plain-text `.lvl` format and a built-in graph-puzzle kit (switches, threshold
  nodes, a topological forward pass, doors).
- Generic play-session systems: puzzle→render state, powered props, triggers, note
  placards, sound fields, zoom, and a HUD probe that labels the nearest interactable.
- Seeded 2D ASCII effects (eye, plasma, tunnel, radar, …) for menus and splashes.

**Tools** (`editor/`)
- `ts_editor`: a terminal-native level editor with a top-down slice grid beside a
  live 3D preview, mouse painting, geometry, stair, object, material, emitter and
  sound-field tools, `.vox` import, undo/redo, a command palette (`:`) and a
  generated help sheet (`?`).
- `lvlcheck`: a static level linter (structure, solvability, reachability, "can the
  player get stuck", lights, props, sound fields). Exits non-zero on failure.
- `vox2lvl` / `lvl2vox`: MagicaVoxel `.vox` ⇄ `.lvl`.
- `renderqa`: headless render-quality report (HTML + JSON).
- `re_golden`: golden-framebuffer hashes (look and geometry) that fail closed.

**Experimental**
- `ts_net` and `ts_netplay`: a multiplayer replication testbed on `bevy_replicon`.
  Nothing else depends on them.

For what comes next, see [ROADMAP.md](ROADMAP.md).

## Quick start

You need a **stable Rust toolchain** from [rustup](https://rustup.rs).
`rust-toolchain.toml` selects the channel and components.

- **Windows 10+:** the MSVC build tools ("Desktop development with C++"). Binaries
  link the C runtime statically.
- **Linux:** `sudo apt install pkg-config libasound2-dev libudev-dev` (or your
  distribution's equivalents).

```sh
cargo build --workspace
cargo run --release -p ts_editor -- levels/21-emitter-showcase.lvl   # editor + live preview
cargo run --release -p re_lvlcheck -- levels/                        # lint every example level
cargo run --release -p re_imgqa -- levels/22-lit-cellar.lvl          # writes target/renderqa/report.html
```

Run the tests with no audio device and a throwaway save directory:

```sh
export TS_AUDIO_SILENT=1 TS_SAVE_DIR="$(mktemp -d)"
cargo test --workspace --no-fail-fast
```

A terminal with truecolor, UTF-8 and a monospace font with good Unicode coverage
works best: Windows Terminal, or any modern Linux terminal. 120×40 or larger looks
best; 80×24 is the minimum.

## Crate map

Dependencies point one way: `editor/` and `apps/` use `engine/`, never the reverse.
Only `re_present` touches the terminal. Details: [docs/LAYERING.md](docs/LAYERING.md).

**engine/** — the reusable runtime

| Crate | What it is |
|---|---|
| `re_core` | Shared types: `Framebuffer` / `Cell`, colour (Oklab), overlays, pools, triggers |
| `re_world` | Voxel `Map`, `CameraPose`, collision and gravity, material/emitter/sky/light specs |
| `re_level` | The `.lvl` format: parser, asset resolver, props and staircase expansion |
| `re_puzzle` | The graph-puzzle kit: switches, threshold nodes, forward pass, power budget, win |
| `re_render` | Layer A: raycast → shade → composite, materials, lighting, sky, emitters, portals |
| `re_present` | Layer B: framebuffer → terminal via `bevy_ratatui`, input and gamepad. The only terminal crate |
| `re_audio` | Deterministic synth, kira playback, clock, director, acoustics, sound fields |
| `re_postfx` | Deterministic post-process stack over the shaded framebuffer |
| `re_glitch` | Deterministic visual-corruption layer |
| `re_asciifx` | Seeded 2D ASCII effects and text-painting helpers |
| `re_runtime` | Generic play-session systems for a loaded level, generic over the host's Bevy `States` |
| `ts_net` | *Experimental* replication types for the multiplayer testbed |

**editor/** — authoring and development tools

| Crate | What it is |
|---|---|
| `ts_editor` | The level editor (binary `terminal-state-editor`) |
| `re_lvlcheck` | The level linter (library + binary `lvlcheck`) |
| `re_voximport` | `.vox` + `.toml` → `.lvl` (`vox2lvl`), `lvl2vox`, `tex_import`, `gen_texfixture`, and the `.lvl` writer |
| `re_voxscene` | Assembles `.vox` models into a scene and renders it (`voxscene`, `gen_voxmodels`) |
| `re_imgqa` | Headless render-quality linter (`renderqa`) |
| `re_golden` | Golden-framebuffer harness and the engine corpus |

**apps/** — runnable programs built on the engine

| Crate | What it is |
|---|---|
| `ts_netplay` | *Experimental* multiplayer testbed (binary `terminal-state-netplay`) |

**Data**

| Path | What |
|---|---|
| `levels/` | Example, showcase and test-fixture levels |
| `assets/texture/` | Generated texture fixtures |
| `authoring/voxelmodels/generated/` | Generated `.vox` sample models |
| `tests/golden/` | Golden hashes and their curation log |
| `docs/` | [Architecture](docs/ARCHITECTURE.md), [layering](docs/LAYERING.md), [level format](docs/LEVEL_FORMAT.md), [materials](docs/MATERIALS.md), [lighting](docs/LIGHTING.md), [emitters](docs/EMITTERS.md), [sky](docs/SKY.md), [vox import](docs/VOX_IMPORT.md) |

## How a game uses the engine

A game is its own cargo workspace that depends on the engine crates by git tag:

```toml
[workspace.dependencies]
re_core    = { git = "https://github.com/Neural-Agent-Ledger/terminal3d", tag = "v0.1.0" }
re_render  = { git = "https://github.com/Neural-Agent-Ledger/terminal3d", tag = "v0.1.0" }
re_present = { git = "https://github.com/Neural-Agent-Ledger/terminal3d", tag = "v0.1.0", default-features = false }
re_runtime = { git = "https://github.com/Neural-Agent-Ledger/terminal3d", tag = "v0.1.0" }
# … one line per engine crate you use; editor crates only as dev-dependencies
```

The game owns its Bevy app, its state enum, menus, saves and settings. It adds the
engine plugins, passes its own "a level is live" state value to the `re_runtime`
plugins, and writes the few seam resources the runtime reads. It may use editor
crates such as `re_golden` and `re_lvlcheck` in tests, but never in its normal build.

[TERMINAL STATE](https://github.com/Neural-Agent-Ledger/terminal-state) is the reference game.
Its `terminal-state --dump <level.lvl>` prints one frame of any level, including the
examples here, to stdout, and its repository documents working against a local clone
of this engine.

## Contributing

**Contributions are welcome, and we keep them upstream.** Pull requests are reviewed,
merged here and maintained here; you do not need to carry a fork. Engine features,
effects, editor work, levels, docs and tests are all in scope.

Start with [CONTRIBUTING.md](CONTRIBUTING.md) (setup, tests, goldens, the layering
rule, provenance, DCO sign-off) and the good first issues in [ROADMAP.md](ROADMAP.md).
Coding assistants: see [CLAUDE.md](CLAUDE.md). Community rules:
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security reports: [SECURITY.md](SECURITY.md).

Unless you explicitly state otherwise, any contribution you intentionally submit is
licensed as described below, without any additional terms or conditions.

## Licence

- **Code** is dual-licensed under **MIT OR Apache-2.0**, at your option
  ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)). That covers
  everything under `engine/`, `editor/` and `apps/`, the documentation, every
  level under `levels/` (examples and test fixtures alike) and the generated
  assets, so you can copy any example level into your own project.
- [PROVENANCE.toml](PROVENANCE.toml) lists the origin and licence of every file under
  `assets/`, `authoring/` and `levels/`.
- Third-party crates keep their own licences, as declared in each crate's manifest.
- Games built on the engine choose their own licence. The reference game, TERMINAL
  STATE, is licensed GPL-3.0-or-later for its code and CC BY-SA 4.0 for its content;
  that does not change the terms of anything in this repository.

## Credits

The raycaster, shading, materials, synth, puzzle kit and tools are original code in
this repository, built on [Bevy](https://bevyengine.org), [ratatui](https://ratatui.rs)
and [crossterm](https://github.com/crossterm-rs/crossterm) via `bevy_ratatui`,
[kira](https://github.com/tesselode/kira), `glam`, `noise`, `image`, `dot_vox`,
`gilrs` and `bevy_replicon`.
