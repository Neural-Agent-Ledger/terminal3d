# Roadmap

Where terminal3d stands and what comes next. The
[good first issues](#good-first-issues) are small and self-contained; the
[bigger items](#bigger-items) need a design discussion first. See
[how to pick one up](#how-to-pick-one-up) before you start.

Last updated 2026-09-28.

---

## Done

**Engine**
- Headless Bevy 0.19 app (MinimalPlugins + ScheduleRunner): no GPU, no window.
- Continuous DDA voxel raycaster with floor/ceiling casting, multi-storey maps,
  stairs, ramps and 45° wedges.
- Tiered composite renderer: cached far panorama, near layer recast only when dirty,
  back-to-front transparency, per-frame emitters.
- Procedural ASCII material system (noise, fBm, Worley, brick, circuit, scanline,
  Truchet, fractal; palettes, animation, glyph families, super-glyph selectors).
- Image textures with height relief, sub-cell cutouts and a quadrant blitter for
  silhouettes.
- Point lights, key/ambient light, height fog, shadows, distance-depth effect, Oklab
  legibility floor.
- Procedural skies: stars, constellation, aurora, clouds, storm, data rain, void,
  fractal, watcher, mirror.
- Particle emitters: motion kinds (curl, attract, Lorenz, flock, ripple), trails,
  distributions, word particles, light-gated dust, and colour-only light-shaft
  emitters.
- Warp portals, fractal infinite-zoom material, holograms, mirrors, code walls, pipe
  text, banners.
- Post-FX stack (`off/clean/subtle/cinematic/toon/crt/dream`) and a deterministic
  glitch layer.
- Terminal present with its own dirty-diff: one write per frame, truecolor plus
  256-colour mode, clean resize handling, and a temporal colour deadband a host can
  expose as "colour steps".
- Keyboard, mouse-look and gamepad input as backend-neutral events.
- Procedural audio: deterministic synth, kira playback, musical clock, beat
  quantization, drone beds, generative director, room acoustics, spatial voices,
  per-level sound fields.
- Graph-puzzle kit: switches, threshold nodes, topological forward pass, power budget,
  door targets, sequences.

**Finished 2026-09-28**
- **Two-repo split.** The engine and editor now live here in `engine/`, `editor/` and
  `apps/`; the game lives in its own repository and depends on this one by tag.
  Layering rules are written down in [docs/LAYERING.md](docs/LAYERING.md).
- **`re_runtime`**: the generic `.lvl` play session, extracted from the game. Every
  plugin is generic over the host's `States`; host-owned values cross as resources.
- **`re_asciifx`**: seeded 2D ASCII effects and the text-paint helpers, extracted
  from the game.
- **Coyote time and a jump buffer** in `re_world`'s player motion, modelled by
  `re_lvlcheck`'s reachability check so the linter and the player agree.
- **Proximity interaction prompt** in `re_runtime::context_line`: a label for the
  nearest interactable in range, not just the one under the crosshair.
- Engine hooks behind the reference game's new settings: "colour steps"
  (off/light/strong) drives `re_present::PresentConfig::color_deadband`, and
  brightness drives the `re_postfx` lift/gain grade. The game's in-game guide,
  first-steps checklist and scrolling settings window shipped the same day on top of
  `re_asciifx` and `re_runtime`.
- Dependency refresh (Bevy 0.19 ecosystem, `toml` 1.x) and LF line-ending
  normalisation across the tree (`.gitattributes`).

**Tools**
- `ts_editor`: terminal level editor with live preview, geometry/stair/object/
  material/emitter/field tools, `.vox` import, undo/redo, command palette, help sheet.
- `lvlcheck`: solvability, reachability, no-stuck, lights, props, fields.
- `vox2lvl`, `lvl2vox`, `tex_import`, `renderqa`, `voxscene`.
- `re_golden`: golden framebuffer harness with separate shade and geometry digests.

---

## Good first issues

Small, contained, and each one teaches you a corner of the codebase.

**Naming and docs**
- **Graph-puzzle vocabulary in the lower crates.** These names read as one specific
  puzzle rather than a generic kit: `re_core::pool::{NodePool, pipe_count,
  pool_for_pipe}`, `re_core::trigger::activate_node`,
  `re_core::cell_state::decode_pipe_tag`, `re_world::{at_door, at_door_within}`,
  `re_render::scene_text::with_pipe_pool`,
  `MaterialRegistry::push_puzzle_builtins`, `re_audio::SoundId::{Door, Pipe}`. Rename
  one (updating every caller) or document it as part of the graph-puzzle kit. One PR
  per crate. See [docs/LAYERING.md](docs/LAYERING.md#the-graph-puzzle-kit-why-node-pipe-and-door-appear-in-the-engine).
- **A complete `.lvl` reference page.** [docs/LEVEL_FORMAT.md](docs/LEVEL_FORMAT.md)
  is a map; turn it into a full key-by-key reference from the `re_level` docs.
- **Crate docs.** `ts_editor`'s crate header is one line; several engine crates still
  describe themselves in terms of one game. Rewrite a `//!` header so it says what the
  crate is, where it sits in the layering, and its entry points.
- **Level headers.** Several example levels open with comments that point at
  documents not in this repository. Rewrite a header as a plain technical description.

**Tooling and CI**
- **Provenance guard.** A test or CI step that fails when a file under `assets/`,
  `authoring/` or `levels/` has no entry in [PROVENANCE.toml](PROVENANCE.toml), or an
  entry points at a missing file.
- **Third-party licence notice.** Generate `THIRD_PARTY_LICENSES` with `cargo-about`
  from `Cargo.lock` and check it in CI.
- **Trim ratatui features** so `termwiz` (and `terminfo`) drop out of the dependency
  tree; verify with `cargo tree`.
- **`lvlcheck` no-stuck fallback.** The waypoint-chase fallback in
  `re_lvlcheck/src/no_stuck.rs` is probably redundant now; try removing it and run the
  staircase and example-level tests.

**Engine**
- **Node sprite:** make its floor contact shadow honour `ShadowQuality::Off`, and make
  the sprite portal-aware.
- **Emitter `color_jitter.dance`** is parsed and stored but not read by the shader.
  Wire it (see [docs/EMITTERS.md](docs/EMITTERS.md#known-gaps)).
- **Emitter parser tests** in `re_level`: one success and one error case per variety
  key.
- **Parallelise the remaining synth renders** on the audio device-open path to cut
  startup time.

**Editor**
- **F2 rename for sound fields** should cascade to every `bind =` / `drive =` that
  names the field (`editor/ts_editor/src/rename.rs` already does this for nodes).
- **Box and capsule shapes** in the field tool (the model supports them; the tool
  only authors spheres).

---

## Bigger items

Pick one of these after a conversation in an issue.

- **Generic level player, `terminal3d-play`.** A `LevelSessionPlugin<S>` in
  `re_runtime` that loads a level on `OnEnter(live)` (map, camera, puzzle, emitters,
  notes, fields) and tears it down on `OnExit`, plus a ~150-line app in `apps/` that
  plays any `.lvl` and has a `--dump` flag. The editor already looks for a sibling
  `terminal3d-play` binary for playtests. This gives the engine a "play my level"
  command without a game.
- **Split the giant files into modules.** `engine/re_render/src/ascii_shader.rs`
  (~21k lines) and `engine/re_render/src/lib.rs` (~16k), then
  `editor/ts_editor/src/chrome.rs` (~7k) and `entity.rs` (~6k). One file per PR, no
  behaviour change, golden hashes must not move.
- **Move the `.lvl` writer into `re_level`.** `to_lvl_text` / `to_lvl_text_with_ids`
  live in `re_voximport`; the format crate should own both directions. Unify the two
  writers so an editor save of a `.vox`-imported level keeps `[color_cell_ids]`.
- **Mantle / ledge grab.** Snap onto a floor edge the jump almost reached (reach
  0.2–0.3 cells, no free climbing). `re_lvlcheck` must model it in the same PR, and
  it can open sequence breaks in multi-floor levels.
- **Screen-space light shafts ("god rays").** An opt-in `re_postfx` pass: a radial
  march from the projected key light over the depth buffer, brightening unoccluded
  cells. Default off, marked as costly. (Particle light shafts already exist as the
  `godrays` emitter kind.)
- **An opt-in "painted" look.** A per-level or per-material register where the
  background carries the lit albedo and the glyph is the same hue with a small
  lightness offset, so large planes read as solid forms from a distance. Never a
  default change; it moves goldens only on levels that opt in.
- **Weather fields.** Rain or snow that fills a region, instead of per-level snow
  emitters.
- **Off-slice pipes** still render as full text cubes; give them a lighter form.
  A second, farther wall in the same plane can show a dark strip at its base
  (`engine/re_render/src/dark_strip_audit.rs` reproduces it).
- **Golden coverage for code walls.** The golden harness does not load code-wall text
  yet.
- **Audio:** interaural time difference on the live playback path; re-bake the drone
  when the player changes room; per-position absorption.
- **Web build (study only).** A browser target would need a cell-grid backend for
  `re_present` (a WebGL instanced-cell renderer over `ratatui-core`), a
  requestAnimationFrame frame loop, browser storage, and audio unlock on first input.
  `re_present` being the only terminal crate makes it plausible. Nobody should start
  code for it yet. Open question: is a browser build of the engine wanted at all? It adds a CI target and a second present backend to maintain. Until someone decides, this item is a study: a short write-up of the backend options and the work list is welcome as an issue.
- **Publishing on crates.io**, once the crate names and APIs settle.

## Open questions

These need a decision before code. Discussion is welcome in the issues.

- Sub-cell filled shapes (tables, tanks, seat rows): worth the renderer cost?
- Per-voxel unique materials, or keep per-level surface skinning?
- Crate names: keep the `re_*` prefix at the first public tag, or rename before a
  crates.io publish?
- Binary names: `terminal-state-editor` and `terminal-state-netplay` carry the game's
  name; rename them to `terminal3d-*`?

---

## How to pick one up

1. **Find or open the issue** for the item, and say you are taking it. For a bigger
   item, write a short plan there first: which crates, which files, what the test is.
2. **Read** [CONTRIBUTING.md](CONTRIBUTING.md) (build, tests, goldens, clippy, DCO)
   and [docs/LAYERING.md](docs/LAYERING.md). If you use a coding assistant, point it
   at [CLAUDE.md](CLAUDE.md).
3. **Start with a test.** For a bug, a test that fails; for a feature, a test that
   pins the new behaviour.
4. **Keep it small.** One topic per PR. A rename touching three crates is three PRs.
5. **Check the goldens.** `cargo test -p re_golden --test engine_corpus`. If a hash
   moves and you did not intend it, find out why before you go further.
6. **Open the PR** with before/after frames for anything visual, signed-off commits,
   and a line saying which roadmap item it closes. We update this page when it lands.
