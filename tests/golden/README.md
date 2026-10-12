# tests/golden — Golden Framebuffer Corpus

A regression suite that pins what the renderer draws. For each curated level it
renders one frame and compares two SHA-256 digests of that frame against the
values stored in `engine_fb_hashes.txt`. Any change to glyph choice, colour,
shading, geometry or camera placement in a curated level fails the suite until
someone looks at it and decides.

## Running it

```sh
cargo test -p re_golden --test engine_corpus
```

The harness lives in the `re_golden` crate (`editor/re_golden`); the curated list
is in `editor/re_golden/tests/engine_corpus.rs`. Another project built on this
engine can pin its own levels by calling `re_golden::check_corpus(store_path,
levels)` from a test of its own, with its own store file.

## What is hashed

Each curated level is loaded with `re_level::load`, the camera is placed at the
level's spawn point (`spawn_camera()`), and one frame is rendered at time 0 with
the default FOV through `re_render::composite::composite_dump_once` — the same
entry point a `--dump` uses — into a fixed 100 × 32 grid (3200 cells). Two digests
are taken over that grid:

| digest | computed over | answers |
|--------|---------------|---------|
| `shade` | row-major `{glyph u32, fg u32, bg u32}` — 12 bytes/cell, little-endian (38 400 bytes/level) | did the **look** change? |
| `geom` | row-major `{depth, side u8, surface u32, material u16}` — 11 bytes/cell, little-endian; `depth` via `f32::to_bits` (35 200 bytes/level) | did the **cast** change? |

`depth` is hashed through `to_bits` because a ray that hits nothing is
`f32::INFINITY` and `NaN != NaN`; the bit pattern is stable where a float
comparison is not.

Not hashed: ANSI escape bytes, `tex_u`/`tex_v`/`hit`, and the `style` channel
(attributes and coloured underline). A style-only change moves neither digest;
that is a known limit of the suite.

Because the digests are taken before any terminal output is produced, they are
the same on Windows and Linux, in any terminal emulator and at any real terminal
size.

## Store format

`engine_fb_hashes.txt` is plain text. Lines starting with `#` are comments. Every
other line is

```
<key> = <shade-sha256> <geom-sha256>
```

where `<key>` is the curated level key (the file name without `.lvl`). A line
with only one digest predates the `geom` channel; its `geom` is reported as a
coverage gap, never as a regression. Do not edit the store by hand; it is
written by the bless run described below.

## Reading a failure

The suite fails closed: every curated level must be positively matched. A failing
run lists each problem level under one of these categories.

| category | meaning | what to do |
|----------|---------|------------|
| `MISMATCH` | the level rendered differently | see the table below |
| `UNREADABLE` | the level file is missing or no longer parses | fix or restore the level; blessing cannot clear it |
| `NO-GOLDEN` | curated, but no hash is stored | bless only if this is a deliberate new anchor |
| `ORPHAN-GOLDEN` | a hash is stored for a level that is no longer curated | restore the entry, or retire the hash deliberately by blessing |

For a `MISMATCH`, which digest moved tells you what kind of change it was:

| moved | means |
|-------|-------|
| `shade` only | The cast is unchanged; a shader, palette or glyph change did this. |
| both | The geometry moved — the DDA, the map or the camera. Treat it as a raycaster regression until proven otherwise. |
| `geom` only | Geometry changed with no visible consequence (for example a material swap behind an identical glyph and colour). Unusual; read it before blessing. |

## Curated levels

23 levels, all in `levels/`:

| Key | Group | Why pinned |
|-----|-------|-----------|
| `01`–`08` | Archive puzzles | Core puzzle-room regression anchors |
| `08-material-demo` | Materials | Authored materials |
| `09-code-wall` | Materials | Flowing monospace code-wall text fitting |
| `10-calibration-vault` | Materials | Floor and ceiling fog |
| `10-hologram-demo` | Materials | Hologram wireframe |
| `TX5-genplate` | Texture | Image sampler and 2×4 Braille blitter on a generated plate |
| `U1-underline` | Feature | Underline substyles (curly, dotted, dashed) |
| `A1-zone-materials` | Overlay | Scalable material overlay zones |
| `A2-lines` | Overlay | Vector line strokes |
| `A5-inset` | Overlay | Inset panels |
| `97-semantic-surfaces` | Feature | Semantic materials |
| `92-fractal-portal` | Feature | Fractal zoom and portal |
| `21-emitter-showcase` | Feature | Particle emitters |
| `20-stairs-straight` | Multi-floor | Straight 45° staircase: per-storey raycasting and ramp climb |
| `21-stairs-spiral` | Multi-floor | Spiral staircase: multi-turn ramp geometry |
| `22-tower` | Multi-floor | Three-storey tower with two staircases: camera elevation across storeys |

Other levels in `levels/` (the sky, emitter and showcase demos, `00-test-load`,
`93-warp-portal`, `94`/`95`/`99` demos, `A3`/`A4`/`T0` and so on) are not pinned:
they were not part of the stable set when the corpus was captured, or their
feature is already covered by a level above. Any of them can be added as a new
anchor (a `NO-GOLDEN` bless of a reviewed frame).

## Re-blessing after an intentional render change

Bless only when a change to the rendered output is intended and has been
reviewed: a glyph-shade constant, a new material recipe, a fix that changes what
a level looks like.

1. **Run the suite unchanged first** and confirm which levels moved and which
   digest moved for each. If you are on a new toolchain, confirm the suite is
   green before your change; a bless on a red baseline silently absorbs whatever
   the toolchain changed.
2. Capture before/after frames for every moved level (for example with
   `cargo run -p re_render --example turn_shots -- <out_dir>`) and include them,
   or a description of the difference, in the pull request.
3. Re-bless:
   ```sh
   TS_BLESS_GOLDEN=1 cargo test -p re_golden --test engine_corpus
   ```
   This re-renders every curated level and rewrites the store. It still fails on
   `UNREADABLE` levels, so it cannot hide a missing file.
4. Commit the updated `engine_fb_hashes.txt` in the same signed-off commit as the
   render change, and add an entry to *History* below:

   ```
   ### YYYY-MM-DD — <levels that moved>, <shade | geom | both>
   Author: <name>. Change: <what changed in the renderer or level, and why>.
   Reviewed: <PR link or reviewer>.
   ```

**Never bless to silence an unexpected failure.** If a level you did not intend to
change now hashes differently, that is a regression to investigate, not a hash to
update.

## History

- **2026-09-28** — The store was split in two: this repository keeps the 23
  engine, example and archive levels; the seven campaign levels of the
  TERMINAL STATE game are pinned by that game's own repository. Every hash kept
  its previous value (a split, not a re-render). Three levels that depended on
  third-party assets without a clear licence were removed from the corpus along
  with those assets.
- **2026-09-24 – 2026-09-27** — Several deliberate re-blesses during a render
  quality pass: set dressing, distance-based depth shading on by default, the
  puzzle-node sprite renderer, and fixes to dark strips and band mottling.
- **2026-08-22** — Format change: a second digest (`geom`) was added per level so
  that a look change and a geometry change can be told apart. No `shade` digest
  moved.
- **2026-08-06 – 2026-08-08** — `09-code-wall` and `TX5-genplate` added as anchors;
  a batch of levels re-blessed after renderer fixes.
- **2026-06-21** — Initial corpus, with the multi-floor staircase and tower levels
  added the same day.
