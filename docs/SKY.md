# `[sky]` — Procedural Sky Program Authoring

Owner crates: `re_render::sky` (the sampler + per-layer implementations),
`re_world::sky_spec` (the pure-data spec + `.lvl` `[sky]` grammar), `re_level`
(section dispatch).

A level's `[sky]` section can author **two independent things**, and both may
appear in the same section:

1. **Image panels** (pre-existing) — `cylinder = <path>` / `cap_top = <path>`
   / `cap_bottom = <path>`, a 360° cylindrical panorama plus a circular top
   cap sampled by ray direction (`re_render::sky::sample_sky`).
2. **A procedural sky PROGRAM** — a time-aware gradient plus a stack of
   composited layers: stars, constellation, aurora, clouds, storm, data-rain,
   void, fractal, watcher, mirror, skyline.

A level with neither set of keys renders exactly as before (no `[sky]`
section, or an `[sky]` with only image paths and no procedural keys) — no
golden churn on existing levels.

## Why the sky shows at all: the open-column gate

The sky (image OR procedural) only paints a cell when that ray's column is
genuinely open overhead: either the level explicitly authors `[sky]` (image
paths or a procedural program), or the camera's column has no capping ceiling
at all (`re_render::column_sky_open`). A closed room never leaks sky,
regardless of what a level authors. Every demo level in this doc uses a
`preset = void` `surface = ceiling` material (see `levels/SKY-*.lvl`) so the
ceiling itself is transparent to the raycaster and the sky is actually
reachable.

## Section syntax

```ini
[sky]
cylinder    = assets/texture/foo_cylinder.png   # optional, image panel
cap_top     = assets/texture/foo_cap_top.png    # optional, image panel
cap_bottom  = assets/texture/foo_cap_bottom.png # optional, image panel

kind        = stars                              # ONE-LINE PRESET, or:
base        = gradient horizon=#101018 zenith=#04060E glow=#2A2040 glow_width=0.15
layer       = stars density=0.02 twinkle=0.4
layer       = aurora curtains=3 palette=oxygen
seed        = 7
hz          = 12
```

- `kind = <preset>` expands to a whole, ready-to-use `SkySpec` in one line
  (see [Presets](#presets)). **Cannot be combined** with `base`/`layer` on the
  same section — authoring both is a parse error (ambiguous which wins), not
  a silent override.
- `base = gradient ...` sets the sky's background colour ramp (horizon to
  zenith, with an optional glow band). At most one `base` key per section; a
  second one overwrites the first.
- `layer = <kind> ...` is **repeatable** — one call per authored layer,
  applied in FILE ORDER (each layer composites over the ones before it,
  matching `SkyProgram::sample`'s fold order). Mix and match freely, e.g.
  `layer = stars ...` then `layer = aurora ...` for stars-through-the-aurora.
- `seed = <u32>` — the deterministic hashing seed for every layer (default
  `0`). Change it to get a different-looking sky from the same layer stack.
- `hz = <u32>` — the sky's own tick rate in Hz (default `12`, i.e. every
  layer re-evaluates 12 times per simulated second; see
  [Byte budget](#byte-budget-why-every-sky-looks-a-little-choppy)). Must be
  `> 0`.
- An unknown key anywhere in `[sky]`, or a malformed `layer`/`base` value, is
  a parse error with line context (`re_level`'s standard `err_at` — same
  convention as every other section).

## Presets

`kind = <name>` is the fastest way to get a working sky; each expands to a
small, sensible layer stack (see `re_world::sky_spec::SkySpec::from_kind`):

| `kind` | Expands to |
|---|---|
| `stars` | `layer = stars` (defaults) |
| `constellation` | a dim star field + `layer = constellation` (default 3-star triangle, inert link tags — see [Constellation hook](#constellation-hook)) |
| `aurora` | a very sparse star field + `layer = aurora` (oxygen palette) |
| `clouds` | `layer = clouds` (defaults) |
| `storm` | `layer = storm` (cloud deck + lightning) |
| `datarain` | `layer = datarain` (binary source) |
| `void` | `layer = void` (defaults) |
| `fractal` | `layer = fractal` (defaults) |
| `watcher` | `layer = watcher` (defaults) |
| `mirror` | `layer = mirror` (defaults) |
| `skyline` | a sparse star field + `layer = skyline` (defaults) |

Every demo level under `levels/SKY-<kind>.lvl` uses exactly `kind = <kind>`
(all presets except `skyline`, which is demonstrated in
`levels/101-facade-skyline-courtyard.lvl`) — open one for a complete,
minimal, winnable, open-top example.

## The channel grammar

Every layer writes into the same four independent channels every other
surface in the game uses:

- **glyph** = structure (a star, a cloud edge, a bolt streak, a falling
  digit) — never used to fake colour.
- **fg** = accent (the bright colour ON the glyph).
- **bg** = mass (the flat colour behind/around the glyph — the sky's "fill").
- **underline** (`CellStyle::underline_style`/`underline_color`) = a SECOND
  independent colour channel — the aurora's rippling hem (`Curly`/`Double`),
  the watcher's iris rim (`Dotted`), used nowhere else in a sky layer so it
  never competes with fg/bg for attention.

## `base` — the gradient

```ini
base = gradient horizon=#RRGGBB zenith=#RRGGBB [glow=#RRGGBB] [glow_width=0.12]
```

| Key | Meaning | Default (no `base` authored) |
|---|---|---|
| `horizon` | colour at the bottom of the visible dome | `#1A2238` |
| `zenith` | colour at the very top | `#04060E` |
| `glow` | optional warm/cool band lerped in near the horizon | none |
| `glow_width` | how wide the glow band is (fraction of dome height) | `0.12` |

## Layers

Every layer key below is optional; an unspecified key falls back to a
documented default. All colours are `#RRGGBB`.

### `stars`

```ini
layer = stars density=0.025 twinkle=0.5 milkyway=0.35 pole_tilt=24 wheel=0.01 meteors=0.0
```

| Key | Meaning | Default |
|---|---|---|
| `density` | fraction of grid cells eligible to be a star | `0.025` |
| `twinkle` | brightness/glyph flicker amount | `0.5` |
| `milkyway` | how much brighter/denser the milky-way band is | `0.35` |
| `pole_tilt` | the star wheel's pole tilt, degrees | `24` |
| `wheel` | how fast the whole field slowly wheels (rad/sec-ish) | `0.01` |
| `meteors` | meteor streak rate (0 = none) | `0.0` |

### `constellation`

```ini
layer = constellation stars=(0.31,0.42) (0.34,0.47) (0.29,0.51) link=tag:star_a,star_b,star_c
```

Three normalized `(u, v)` sky positions (`u` = wrapped yaw fraction, `v` =
dome-height fraction, both `[0,1)`) plus three `link` tag names. Each star
renders **dim/unpowered** until its tag reads "on" (see
[Constellation hook](#constellation-hook)); when all three are powered, a
dotted-underline line is drawn connecting them.

### `aurora`

```ini
layer = aurora curtains=3 band=0.18..0.55 height=0.22 decay=3.0 palette=oxygen \
        activity=0.6 shimmer_hz=2.0 surge_rate=0.15 drift_period=180.0
```

| Key | Meaning | Default |
|---|---|---|
| `curtains` | number of independent curtain bands | `3` |
| `band` | `lo..hi` dome-height fraction the curtains' feet wander within | `0.18..0.55` |
| `height` | curtain ray height (dome-height units) | `0.22` |
| `decay` | exponential falloff above the foot | `3.0` |
| `palette` | `oxygen` \| `violet` \| `mono` | `oxygen` |
| `activity` | overall intensity multiplier | `0.6` |
| `shimmer_hz` | authored but folded into the layer's own tick cadence, not a separate oscillator | `2.0` |
| `surge_rate` | how often a bright "surge" sweeps along a curtain | `0.15` |
| `drift_period` | seconds for one full hue-drift cycle | `180.0` |

Stars already drawn underneath show through, dimmed by the aurora's local
intensity. The hem (where a curtain's foot crosses the current row) gets a
`Curly` (or `Double` at high intensity) underline in the same altitude-ramp
colour.

### `clouds`

```ini
layer = clouds cover=0.4 soft=0.15 scale=1.0 height=60.0 wind=0.05,0.0 \
        layers=1 lit=#FFF0D0 shade=#202030 haze=40.0 bake=off
```

| Key | Meaning | Default |
|---|---|---|
| `cover` | density threshold (0 = no cloud, 1 = solid deck) | `0.4` |
| `soft` | edge softness | `0.15` |
| `scale` | noise feature size (bigger = larger cloud clumps) | `1.0` |
| `height` | the cloud plane's world height | `60.0` |
| `wind` | `dx,dy` world-space drift per second | `0.05,0.0` |
| `layers` | `1` (cumulus only) or `2` (+ a faster, higher cirrus deck) | `1` |
| `lit` / `shade` | the two colour stops (silver-lining rim / body) | `#FFF0D0` / `#202030` |
| `haze` | distance a cloud fades into by (bigger = clouds stay solid farther) | `40.0` |
| `bake` | reserved for a future far-tier panorama cache; currently accepted, not yet wired to a distinct code path | `off` |

Density maps to a glyph ramp: the interior of a dense cloud is a flat `bg`
fill (no glyph); only a thin edge band gets a glyph (`.` -> `·` -> `:` -> `░`
as density rises). A cloud occludes whatever the layer below drew (stars,
aurora) once dense enough.

`wind = 0,0` makes the layer static (`is_time_animated() == false`) — useful
for a deliberately frozen sky.

### `storm`

```ini
layer = storm cover=0.85 rate=0.3
```

| Key | Meaning | Default |
|---|---|---|
| `cover` | the deck's own cloud cover (reuses `clouds`' model) | `0.85` |
| `rate` | bolts/second, BEFORE the WCAG cap | `0.3` |

Bolts are rate-limited to the WCAG 2.3.1 ceiling of **at most 3
flashes/second** regardless of the authored `rate`. A bolt shows a short
streak glyph (`│`/`╱`/`╲`) near its own position and a small, bounded
afterglow lift **around** the bolt (not the whole dome — see
`re_render::sky::storm::AFTERGLOW_RADIUS`). Pushing `rate` toward the 3/s
ceiling trades away byte-budget headroom for flash frequency — keep it well
under the ceiling unless you have re-measured the sky's own budget tests at
your level's actual `[sky]` configuration.

**The room-flash hook.** Actually folding a bolt into the room's own lighting
means editing the shading pipeline directly, which is a bigger change than a
sky layer should make on its own. Instead, `re_render::sky::storm` exposes
the event as a thread-local:

```rust
pub struct LightningFlash { pub azimuth: f32, pub intensity: f32 }
pub fn set_active_lightning_flash(flash: Option<LightningFlash>); // called by StormLayer::shade
pub fn active_lightning_flash() -> Option<LightningFlash>;        // read by any lighting-owning code
```

Nothing in this crate installs it into the room's own light environment
today — **this is a hook, not a finished feature.** Code that owns room
lighting can read `active_lightning_flash()` once per frame and fold a
transient key light in.

### `datarain`

```ini
layer = datarain source=binary speed0=0.5 speed1=1.5 trail=0.35
```

| Key | Meaning | Default |
|---|---|---|
| `source` | `binary` \| `hex` \| `pool` \| `shards` | `binary` |
| `speed0`/`speed1` | per-column fall speed range (hashed per column) | `0.5` / `1.5` |
| `trail` | trail length (fraction of the dome height) | `0.35` |

Columns are **world-yaw-anchored, not screen-fixed** — turning the camera
turns the columns WITH the world (the matrix is in the SKY, not on the
screen). `pool`/`shards` are meant to draw from a level's own vocabulary of
short strings; wiring that real text registry crosses into a text-authoring
layer outside this module's scope, so both currently fall back to a wider
deterministic letter set — visually distinct from `binary`/`hex`, but not yet
a level-specific vocabulary.

### `void`

```ini
layer = void drift=0.15
```

A near-black dome with a sparse, slowly drifting `·` field. `drift = 0`
freezes it.

### `fractal`

```ini
layer = fractal speed=0.12
```

The zenith cap becomes an infinite log-polar spin-zoom tunnel, reusing
`re_render::fractal::{spinzoom_phase, log_polar_field}` READ-ONLY (the same
machinery the `fractal` wall/portal material recipe uses elsewhere in the
renderer). `speed = 0` freezes it.

### `watcher`

```ini
layer = watcher iris=#2A3040
```

A giant iris/pupil rendered into the zenith cap; the pupil tracks the
**player's own facing yaw**, so it always seems to look back at you no matter
which way you turn. Not time-animated — a yaw change already forces a
structural recast on its own, so no independent per-tick drift is needed. The
eye is a simple procedural shape (concentric iris ring + pupil disc).

### `mirror`

```ini
layer = mirror dim=0.4
```

**Scope cut, documented rather than hidden:** the intent is "the level's own
footprint, seen from below, rotating the wrong way" — a literal reflection of
the authored floor plan. A sky layer's `shade` signature is a pure function
of `(ctx, seed)` with no access to the level's map, so a literal geometry
reflection isn't reachable from inside a layer today. This ships a
deterministic silhouette-noise field keyed on the **mirrored yaw**
(`-ctx.yaw`), so it is still direction-anchored and stable under rotation —
just not a literal reflection of the level's own geometry. Extending the
layer trait to also carry the map is the natural follow-up for whoever wants
the literal version.

### `skyline`

```ini
layer = skyline blocks=96 density=0.75 height=0.12..0.55 style=office lit=0.3 landmark_rate=0.08
```

| Key | Meaning | Default |
|---|---|---|
| `blocks` | azimuth slots around the full turn; each is a candidate tower | `96` |
| `density` | fraction of blocks that actually carry a tower; the rest show sky/gap | `0.75` |
| `height` | `lo..hi` dome-height fraction a tower's height is seeded within | `0.12..0.55` |
| `style` | `office` \| `residential` \| `industrial` \| `server_rack` \| `lockers` — reuses the facade material's window-unit table | `office` |
| `lit` | fraction of a tower's windows that hash lit | `0.3` |
| `landmark_rate` | fraction of towers that carry a spire + slow blinking antenna above the roofline | `0.08` |

A distant-city horizon. Each candidate tower's height and hue are picked by a
seeded hash, and `style` reuses the same window-unit table the `facade`
material recipe uses, so a tower's windows read with the same rhythm a nearby
facade wall would. PARALLAX-FREE: like every sky layer this is sampled by ray
direction only, never by camera position — a "distant" skyline does not need
real-world distance to read as far away; it only needs to never scroll under
camera translation.

## Constellation hook

`re_render::sky::stars::tag_id_for_name(name: &str) -> u16` is the SAME hash
any puzzle-wiring code must use to publish a tag's powered/unpowered state
under the id this layer reads via `re_render::state_bus` (which in turn
reflects the active `re_core::CellStateMap`). A `constellation` layer's three
`link` tag names are hashed through this function; whatever writes that state
controls that star's "powered" look. None of the shipped `SKY-*.lvl` demo levels wire
this — every star in `levels/SKY-constellation.lvl` renders in its
dim/unpowered default, by design (no puzzle wiring is authored there).

## Byte budget: why every sky looks a little choppy

Every layer hashes `(seed, index, param_id)` only — no RNG state — and the
p95 count of sky cells that change from one tick to the next must stay small
(at most 800 at 322x86 — `sky::budget_tests::p95_changed_cells_per_tick_is_within_budget`
enforces this in CI). A continuous drifting field (wind-driven clouds, an
aurora's phase, the fractal spin-zoom, a data-rain column's fall) sampled at
the *live* tick would recolour nearly every covered cell **every single
tick** — far over budget, and also wasteful (every changed sky cell forces
that column through the renderer's near-tier full re-cast path, not the cheap
selective reshade).

The fix, applied uniformly: **stagger** the continuous driver. Each layer
quantizes a coarse per-cell/per-column "bucket" (a yaw bucket for aurora, a
`(u,v)` bucket for the fractal cap, a `(yaw, height)` bucket for clouds, the
data-rain column index itself) and only lets that bucket's animation clock
advance once every `N` ticks (`re_render::sky::util::{staggered_tick,
staggered_qt}`, offset by the bucket's own hash so buckets desynchronise).
Between its own updates a bucket replays **byte-identically** — this is also
what makes `sky::program_tests::same_tick_is_byte_static` meaningful: the
same `tick` always produces the same `SkyCell`, and most calls are, in
effect, the same tick as last time from that bucket's point of view.

This generalises the twinkle/meteor staggering technique to every continuous
sky layer. The visible side-effect: clouds drift and auroras shimmer in small
discrete jumps rather than perfectly smooth motion. That is the deliberate
trade a real-time terminal raycaster makes for a bounded re-cast rate — not a
bug.

**`storm`'s lightning is the one deliberate exception.** A bolt's afterglow
lift is allowed to touch a meaningfully large patch of sky at once (a real
flash IS supposed to look big) — see `AFTERGLOW_RADIUS` in
`re_render::sky::storm`. That's safe under the p95 budget only because the
patch is bounded (not literally the whole dome) AND flashes stay a small
minority of ticks even at the WCAG ceiling; see the constant's own doc
comment for the exact arithmetic before turning `rate` up.

## Known scope cuts (honest, not hidden)

- **No authored sun/moon disc key.** Nothing in this module renders an
  explicit sun or moon disc today. A future pass should get an explicit key
  name from a deliberate design decision, not a guess.
- **`mirror` is an approximation**, not a literal floor-plan reflection — see
  the `mirror` section above.
- **`datarain`'s `pool`/`shards` sources** fall back to a generic letter set,
  not a level's own real vocabulary — see the `datarain` section above.
- **The storm room-flash hook** (see the `storm` section above) publishes the
  event; nothing consumes it into room lighting yet.
- **The constellation `state_bus` wiring** is a hook only (see
  [Constellation hook](#constellation-hook)) — no puzzle in the shipped demo
  levels actually drives it.

## Demo levels

One open-top vault per sky kind (except `skyline`, demonstrated separately),
each a minimal, fully winnable one-node puzzle (source to sink, one switch)
so the level corpus's solvability gate passes: `levels/SKY-stars.lvl`,
`SKY-constellation.lvl`, `SKY-aurora.lvl`, `SKY-clouds.lvl`, `SKY-storm.lvl`,
`SKY-datarain.lvl`, `SKY-void.lvl`, `SKY-fractal.lvl`, `SKY-watcher.lvl`,
`SKY-mirror.lvl`. `levels/101-facade-skyline-courtyard.lvl` demonstrates
`kind = skyline` alongside the `facade` material recipe.

```
cargo run --release -p re_imgqa -- levels/SKY-aurora.lvl
```
