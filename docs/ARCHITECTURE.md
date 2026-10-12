# Architecture

A one-page map of how terminal3d is put together. The dependency rules are in
[LAYERING.md](LAYERING.md); this page is about what each part does and how a frame
is made.

## Stack

- **Bevy 0.19, headless.** Bevy is a cargo dependency with `MinimalPlugins`,
  `ScheduleRunnerPlugin` and `bevy_state`, and no `wgpu`, `winit` or `bevy_render`.
  We use its ECS, schedules and states, and nothing that draws.
- **ratatui + crossterm** through `bevy_ratatui` for raw mode, input, resize and
  output. All of it is confined to `re_present`.
- **kira** is the audio output and clock. Every sample it plays comes from our own
  synth.
- Also `glam` (math), `noise` (procedural fields), `image` (PNG/GIF/WebP textures),
  `dot_vox` (MagicaVoxel), `gilrs` (gamepad), `bevy_rand` (seeded ECS randomness),
  `serde` / `toml`, and `bevy_replicon` for the experimental networking crates.

## Crates

```
                 ┌──────────────── editor/ ────────────────┐   ┌── apps/ ──┐
                 │ ts_editor  re_lvlcheck  re_voximport     │   │ ts_netplay│
                 │ re_voxscene  re_imgqa  re_golden         │   └─────┬─────┘
                 └───────────────────┬──────────────────────┘         │
                                     ▼                                ▼
 engine/   re_runtime (play-session systems, generic over the host's States)
              │  uses everything below
   re_level (.lvl format) ── re_puzzle (graph-puzzle kit)
   re_render (Layer A) ── re_postfx ── re_glitch        re_asciifx (2D effects)
   re_present (Layer B: the ONLY terminal crate)        re_audio (synth + kira)
   re_world (voxel Map, camera, collision, specs)       ts_net (experimental)
   re_core (Framebuffer, Cell, colour, overlays, shared types)
```

`Framebuffer` and `Cell` (glyph, fg, bg, depth, side, surface, material, style and
more) live in `re_core`. Every stage reads and writes that one type, which is why the
renderer can be tested, hashed and dumped with no terminal.

## Frame flow

```
Update       input (re_present: keys/mouse/gamepad → backend-neutral events)
             → puzzle (re_puzzle: switch toggles → forward pass → door state)
             → camera move + collision + gravity (re_world: coyote time, jump buffer)
             → session systems (re_runtime: puzzle→render bus, prop power, triggers,
               sound fields, zoom, colour drift)

PostUpdate   raycast (re_render)        camera + Map → per-cell depth/side/surface/material
             → shade (re_render)        materials, lights, fog, sky, depth effect → {glyph, fg, bg}
             → composite (re_render)    far panorama (cached) + near (recast when dirty)
                                        + transparent layers + emitters, back to front
             → HUD and host overlays    re_runtime::context_line, notes; the host's menus
             → post-FX (re_postfx) → glitch (re_glitch)

Last         present (re_present)       diff against the last presented grid,
                                        write only changed cells, one flush per frame
```

What keeps it fast:

- **Dirty flags.** An unchanged camera, world and material set means no recast and
  zero terminal bytes. A puzzle player stands still most of the time.
- **Tiered caching.** The far panorama is cached and sampled as the camera turns,
  and the near layer is recast only when dirty. Animated cells are re-shaded
  selectively.
- **Cell-quantized motion.** Internally the camera moves in discrete steps, which
  keeps the caches valid and the golden hashes stable.
- **Own dirty-diff in `re_present`**, plus a temporal colour deadband
  (`PresentConfig::color_deadband`) that a host can expose as a "colour steps"
  setting. Terminal cost is the number of bytes written, not the area of the grid.

`re_render::composite_dump_once` runs the same raycast, shade and composite path once
into a fixed-size framebuffer. The golden tests (`re_golden`) hash exactly that
output, and a host's `--dump` flag prints it.

Deeper: the `re_render` crate docs, [MATERIALS.md](MATERIALS.md),
[LIGHTING.md](LIGHTING.md), [EMITTERS.md](EMITTERS.md), [SKY.md](SKY.md).

## Audio graph

```
 host events (switch, door, footstep, solve, …)
        │  re_audio::cue — event → voiced layers (the sound grammar)
        ▼
 synth (pure)                oscillators / ADSR / noise → mono f32 PCM, deterministic
 transport + policy (pure)   BPM grid, strong/weak beats, snap window, humanize;
                             which sounds play at once and which wait for the beat
 director (pure, seeded)     generative notes, drone accents, glitches per theme
 ambience (pure)             drone beds, intensity → dread curve, equal-power crossfades
 acoustics / fields (pure)   room RT60 + early reflections; distance, pan, occlusion;
                             per-level sound fields bound to nodes, pipes, doors, regions
        │
        ▼
 playback (kira; the ONLY device code)
        master track (reverb + low-pass) · accent track (distortion + delay) · musical clock
        → a silent no-op when no device exists or TS_AUDIO_SILENT is set
```

Everything above `playback` is pure and unit-tested. The engine ships no audio files;
sounds are synth recipes.

## Level pipeline

```
 MagicaVoxel .vox + .toml ──vox2lvl──┐
                                     ▼
 ts_editor ◄──────────────► levels/*.lvl  (plain text, see LEVEL_FORMAT.md)
   (edit + live preview)             │
                                     ▼
                     re_level::load → Level → build Map + camera spawn
                                     │
         ┌───────────────────┬───────┴──────────┬────────────────────┐
         ▼                   ▼                  ▼                    ▼
   re_lvlcheck          re_runtime +        re_golden            re_imgqa
   (lint: solvable,     a host game         (golden hashes)      (renderqa report)
   reachable, no-stuck) (play the level)
```

- `re_puzzle` holds the graph rules. Switches toggle edges, a topological forward
  pass pushes activations through threshold nodes, and the door opens when its target
  node matches. `lvlcheck` uses the same code to prove a level solvable, and models
  the same jump envelope as play (including coyote time and the jump buffer), so it
  cannot disagree with the player about reachability.
- The writer (`to_lvl_text`) lives in `re_voximport` today, and round-trips are
  tested per section.

Deeper: [LEVEL_FORMAT.md](LEVEL_FORMAT.md), [VOX_IMPORT.md](VOX_IMPORT.md).

## Play session (`re_runtime`)

`re_runtime` brings a loaded `Level` to life while the host says a level is live.
Each plugin is generic over the host's `S: States` and takes the "live" value from
the host (`ZoomPlugin { live: MyState::Playing }`). It holds:

- `state_bridge`: carries puzzle state to the renderer (lit pipes, powered props);
- `prop_power`, `emit_parts`, `color_drift_sources`, `zoom`, `unstuck`;
- `triggers` (trigger fields), `notes` (wall placards), `sound_fields` + `audio_world`;
- `hud_probe` and `context_line`: what the player faces, plus a proximity-picked
  label for the nearest interactable in range, so switches can be found without
  pixel-aiming.

Values the host owns (story counters, movement phase, player audio settings) cross
the seam as resources the host writes. Level load and teardown still live in the
host; a generic `LevelSessionPlugin` and a `terminal3d-play` binary are on the
[roadmap](../ROADMAP.md).

## 2D effects (`re_asciifx`)

Seeded, deterministic animated ASCII effects (eye, plasma, tunnel, radar,
oscilloscope, circuit, DNA, heartbeat, overgrowth and others) painted into a
`Framebuffer`, plus the `paint` text helpers. Each effect is a pure function of
`(seed, rect, t)`, so it can be seeked to any time with no state. Menus, backdrops
and splash screens use them.

## Editor

`ts_editor` (binary `terminal-state-editor`) is a Bevy app that uses `re_present`
through its `editor` input seam. Its screen has:

- a top-down grid of the current z-slice, with rulers, a z-slider and height
  overlays;
- a live first-person preview drawn by the same `re_render` a game ships, with a
  flyable preview camera;
- panels: toolbar, inspector, outliner, validation drawer, level-settings form.

Every binding lives in one table (`editor/ts_editor/src/keymap.rs`). The status-bar
hints, the `?` help sheet and the `:` command palette are generated from it, so the
three cannot drift apart. Saving runs `re_lvlcheck`.

## Tests

- **Unit tests** in every crate. The renderer, synth, puzzle, parser and editor
  logic are pure, so they are tested without a terminal or an audio device.
- **Integration tests**: `re_runtime/tests/` (lit levels, staircases, triggers, prop
  power, image textures), `re_lvlcheck/tests/example_levels.rs` (every level in
  `levels/` lints), `re_voximport/tests/` (round-trips, `.vox` end to end),
  `ts_netplay/tests/` (replication).
- **Golden framebuffer tests** (`re_golden`) hash each curated level's frame twice,
  once for the look (`shade`) and once for the geometry (`geom`), and fail closed.
  See [tests/golden/README.md](../tests/golden/README.md).
