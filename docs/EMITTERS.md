# `[emitter]` — Particle-Emitter Authoring

A `.lvl` file can declare one or more **particle emitters**. Each `[emitter]`
section is parsed by `re_level` into one `re_world::EmitterSpec` and exposed on
the parsed level as `Level::emitters` (a `Vec<EmitterSpec>`). The renderer
(`re_render`) later resolves each spec into a live `Emitter`.

Emitters are **additive and back-compatible**: a level with no `[emitter]`
section has an empty `emitters` Vec and renders exactly as before.

## Section syntax

```ini
[emitter NAME]          # NAME is optional, for readability only
key = value
key = value
...
```

- The header may be `[emitter]` or `[emitter some_name]`. The name is **not**
  stored on the spec — it is purely a label for the author.
- **Multiple `[emitter]` sections** are allowed; each one produces a separate
  `EmitterSpec`, in file order (exactly like multiple `[material]` sections).
- Every key is optional. Unspecified keys fall through to the `EmitterSpec`
  default (a gentle upward haze — immediately visible and clearly working).
- An **unknown key** is a parse error with line context. A **malformed value**
  (e.g. a non-numeric `rate`) is a parse error naming the key and line.
- After parsing, every emitter is run through `EmitterSpec::validate()`, which
  **clamps** out-of-range values so a hand-written level can never hand the
  renderer a degenerate spec (see [Clamping](#clamping-validate)).

## Keys

Each key maps to exactly one `EmitterSpec` field.

| Key | Value | Field | Default | Notes |
|-----|-------|-------|---------|-------|
| `origin_world` | `X Y Z` (floats) | `origin` = `World(Vec3)` | `World(3.5, 3.5, 1.5)` | World-space cell coords. Particles project through the camera and are occluded by walls. |
| `origin_screen` | `COL ROW` (floats) | `origin` = `Screen{col,row}` | — | Fractional terminal cell. Particles are pinned to the viewport (HUD sparks) and never occluded. |
| `colors` | `0xRRGGBB 0xRRGGBB 0xRRGGBB` | `colors: [u32;3]` | `0x9FB8D8 0x4A7AA8 0x2A3A52` | Exactly **three** stops. `#RRGGBB` and bare `RRGGBB` also accepted. |
| `color_mode` | `RampOverAge` \| `RandomPick` \| `Fixed` | `color_mode` | `RampOverAge` | How the three stops apply over a particle's life. |
| `family` | `<name>` | `family: String` | `haze_dot` | Glyph-family name (see [Families](#glyph-families)). Passed through **verbatim**; an unknown name falls back to `growth_dot` at render time. |
| `glyph_set` | `<char> <char> …` | `glyph_set: Some(Vec<char>)` | `None` | Explicit glyph sequence; **bypasses** `family`. Each space/comma-separated token contributes its first character (multibyte OK). |
| `bold` | `Off` \| `On` \| `ByProximity` \| `ByAge` | `bold` | `Off` | When the terminal cell renders bold/bright. |
| `direction` | `X Y Z` (floats) | `direction: Vec3` | `0 0 1` (up) | Mean emission direction (need not be normalised). Zero vector ⇒ omni. |
| `spread_deg` | `<f32>` | `spread_deg` | `40.0` | Cone half-angle in degrees. **Clamped to `[0, 180]`.** `0` = pencil beam, `180` = omni. |
| `size` | `<f32>` | `size` | `1.0` | Base size at birth. **Clamped to `≥ ε`.** |
| `growth` | `<f32>` | `growth` | `1.2` | Size scale over age (`>1` grows, `<1` shrinks). **Clamped to `≥ ε`.** |
| `rate` | `<f32>` | `rate` | `14.0` | Particles spawned per second. **Clamped to `≥ 0`.** |
| `burst` | `<u32>` | `burst: Some(u32)` | `None` | One-shot: emit exactly N at activation, then stop. Omit for a continuous stream. |
| `speed` | `<f32>` | `speed` | `0.4` | Mean initial speed (cells/sec). **Clamped to `≥ 0`.** |
| `speed_spread` | `<f32>` | `speed_spread` | `0.15` | Symmetric speed spread. **Clamped to `≥ 0`.** |
| `path_jitter` | `<f32>` | `path_jitter` | `0.0` | Per-step turbulence amplitude. **Clamped to `≥ 0`.** |
| `seed` | `<u64>` | `seed` | a fixed constant | Deterministic RNG seed (golden-frame safe). |
| `swirl` | `<f32>` | `swirl` | `0.0` | Curl/rotational drift. Positive = counter-clockwise (from above). |
| `lifetime` | `<f32>` | `lifetime` | `3.5` | Mean particle lifetime (sec). **Clamped to `≥ 0.01`.** |
| `lifetime_spread` | `<f32>` | `lifetime_spread` | `0.5` | Symmetric lifetime spread. **Clamped** so the shortest life stays `≥ 0.01 s`. |
| `accel` | `X Y Z` (floats) | `accel: Vec3` | `0 0 -0.02` | Constant acceleration (cells/sec²). `0 0 -9.8` = gravity, `0 0 +0.4` = buoyancy. |
| `max_particles` | `<u32>` | `max_particles` | `256` | Hard per-emitter cap. **Clamped to `≥ 1`.** |
| `blend` | `KeepBg` \| `Opaque` | `blend` | `KeepBg` | `KeepBg` = glyph+fg only (bg shows through); `Opaque` = glyph+fg+bg replace. |

### Variety & randomisation keys

Every key below is **opt-in and additive**: a level that never sets one leaves the
corresponding `EmitterSpec` field at its neutral default, and its frame is
unaffected. Two behaviours that do apply to every emitter are described under
[Shared-behaviour notes](#shared-behaviour-notes).

| Key | Value | Field | Default | Notes |
|-----|-------|-------|---------|-------|
| `motion` | `ballistic` \| `curl scale=… strength=… evolve=…` \| `attract X Y Z strength=… swirl=…` \| `lorenz scale=…` \| `flock n=… sep=… align=… coh=… seek_light=…` \| `ripple speed=…` | `motion: MotionKind` | `ballistic` | Replaces (or, for `curl`, layers atop) the vel/accel integration. See [Motion kinds](#motion-kinds). |
| `lifetime_dist` | `a..b` (uniform) \| `m~s` (Gaussian) \| `a..b^k` (power law) | `lifetime_dist: Option<Distribution>` | `None` | Overrides `lifetime ± lifetime_spread` when present. |
| `speed_dist` | same three forms | `speed_dist: Option<Distribution>` | `None` | Overrides `speed ± speed_spread` when present. |
| `size_dist` | same three forms | `size_dist: Option<Distribution>` | `None` | Overrides plain `size` at birth when present. |
| `randomize` | `<f32>` | `randomize` | `1.0` | Scales the SPREAD of the plain lifetime/colour-jitter draws (`0` = no spread, `1` = authored spread, `>1` = wider). Has no effect on a field once a `*_dist` key overrides it outright. |
| `color_jitter` | `h=<deg> s=<0..1> l=<0..1> dance=<0..1>` (any subset, unset = `0`) | `color_jitter: Option<ColorJitter>` | `None` | One coherent HSL delta applied to every ramp stop, fixed per particle at birth from its seed (not per-stop noise — the ramp keeps its shape, just hue/sat/light-shifted as a whole). `dance` is authored but not yet consumed by the renderer — see [Known gaps](#known-gaps). |
| `color_ramp` | `0xRRGGBB 0xRRGGBB …` (2 to 8 stops) | `color_ramp: Option<Vec<u32>>` | `None` | Overrides `colors`/`color_mode` entirely with an evenly-spaced N-stop ramp. Fewer than 2 or more than 8 stops is a parse error. |
| `glyph_pick` | `Sequential` \| `Random` | `glyph_pick: GlyphPick` | `Sequential` | `Sequential` = today's age-driven family pick; `Random` fixes one family member at birth, held for the particle's life. |
| `glyph_source` | `family` \| `text "<string>"` | `glyph_source: GlyphSource` | `family` | `text "HELLO"`: each spawned particle claims the next character of the string, cycling, instead of a family glyph. |
| `trail` | `<count> fade=<0..1> dir_glyphs=<on\|off>` | `trail: Option<TrailSpec>` | `None` | Ghost trail of `count` (1–8) prior positions, each dimmer by `fade` per step; `dir_glyphs=on` draws a directional glyph (`-\|/\` chosen from screen-space motion) on the head instead of the family glyph. |
| `bounce` | `<f32>` (restitution, `0..1`) | `bounce: Option<f32>` | `None` | When falling through the emitter's own origin height (`floor_z`), reflects vertical velocity scaled by this restitution and applies horizontal friction. |
| `visibility` | `always` \| `lit gain=<f32>` | `visibility: DustVisibility` | `always` | `lit`: a particle's opacity is sampled ONCE at birth from the nearest baked `PointLight`s, scaled by `gain`, and held for its life — invisible in the dark. Needs the runtime to bake lights in via `Emitter::from_spec_lit` (see [Known gaps](#known-gaps)). |

#### Motion kinds

- **`curl scale=0.6 strength=0.8 evolve=0.2`** — divergence-free 2D curl-noise
  drift in the horizontal plane, layered ON TOP of the emitter's ordinary
  accel (so buoyancy/gravity still applies). `scale` is the noise-field
  wavelength (world cells), `strength` the drift speed, `evolve` how fast the
  field itself churns over time. A documented simplification of Bridson's 3D
  curl noise, restricted to the horizontal plane — smoke/spores/dust drifting
  near-level, at a third of the noise evaluations.
- **`attract 5 3 1.5 strength=2.0 swirl=0.5`** — pulled toward the world point
  `(5, 3, 1.5)` by an inverse-square-ish force (`strength`); a nonzero `swirl`
  adds a tangential term so particles orbit rather than fall straight in.
- **`lorenz scale=0.08`** — the classic chaotic attractor (σ=10, ρ=28, β=8/3,
  RK2-integrated); the particle's position IS the (scaled) attractor state —
  `vel`/`accel`/`direction`/`speed` are all ignored for this motion. `scale`
  maps the attractor's natural ~[-20,20] range into world cells.
- **`flock n=40 sep=0.4 align=0.3 coh=0.2 seek_light=0.0`** — Reynolds boids
  (separation/alignment/cohesion) among this emitter's OWN live particles,
  capped at `n` (higher `n` is O(n²) per tick — see [Perf](#perf-particle-budget)).
  `seek_light > 0` adds a pull toward the nearest baked light (needs
  `from_spec_lit`, same caveat as `visibility = lit`).
- **`ripple speed=1.2`** — an expanding ring on the emitter's horizontal
  plane: `radius(age) = speed · age`, with each particle's angle around the
  ring fixed at birth from its own seed. `vel`/`accel` are ignored; height
  stays pinned at the origin's `z`. See [Ripples](#ripples).


#### Ripples

A ripple's leading edge carries a `curly` underline hem on top of its glyph and colour
fade (`'∘' → '·' → ' '`). The emitter composite slot passes a particle's non-empty
cell style through to the final frame (`re_core::layer::apply_emitter_blend`), so the
hem is visible live. The ring math and why ripples are a motion kind rather than a
new `.lvl` section are documented in `engine/re_render/src/ripple.rs`.

#### Shared-behaviour notes

Two behaviours apply to **every** emitter, including ones that use none of the
variety keys:

- **Accumulating splat.** When two or more standard particles land in the same
  screen cell in one frame, the renderer accumulates every contributor and, for a
  cluster, draws a density-ramp glyph (`·` → `:` → `*` → `%` as the cluster grows)
  over the cluster's averaged colour. A single occupant keeps its own glyph and
  colour. This replaced a last-write-wins rule that flickered as the "winner"
  changed from frame to frame.
- **Deterministic per-particle randomness.** Every draw (lifetime, speed, size,
  colour jitter) is hashed from the emitter seed and the particle's birth index, so
  the same level renders the same particles every run. The `*_dist`, `color_jitter`
  and `randomize` keys default to values that reproduce the plain spread keys
  exactly.

### Value formats

- **Float / int triples** (`origin_world`, `direction`, `accel`) accept
  `X Y Z`, `X,Y,Z`, or `(X,Y,Z)`.
- **Colours** accept `0xRRGGBB`, `#RRGGBB`, or bare `RRGGBB` (6 hex digits).
- **Enum keywords** accept the `PascalCase` form shown above plus `snake_case`
  and short aliases (e.g. `color_mode = ramp`, `bold = age`, `blend = keep_bg`).

### Enum aliases

| Field | Accepted spellings |
|-------|--------------------|
| `color_mode` | `RampOverAge` / `ramp_over_age` / `ramp`; `RandomPick` / `random_pick` / `random`; `Fixed` / `fixed` |
| `bold` | `Off`/`off`; `On`/`on`; `ByProximity`/`by_proximity`/`proximity`; `ByAge`/`by_age`/`age` |
| `blend` | `KeepBg`/`keep_bg`/`keepbg`; `Opaque`/`opaque` |
| `glyph_pick` | `Sequential`/`sequential`/`seq`; `Random`/`random`/`rand` |
| `visibility` | `always`; `lit gain=<f32>` (case-insensitive keyword) |

### Distribution formats (`lifetime_dist` / `speed_dist` / `size_dist`)

Three forms, all sampled deterministically from the particle's own birth seed
(same seed + birth index ⇒ same draw, every run):

| Form | Example | Meaning |
|------|---------|---------|
| `a..b` | `0.5..1.5` | Uniform in `[a, b]`. |
| `m~s` | `1.0~0.3` | Gaussian, mean `m`, std-dev `s`, clamped at ±3σ (never an unbounded tail). |
| `a..b^k` | `0.2..2.0^2.5` | Power-law over `[a, b]` with shape `k` (`k>1` biases toward `a`, i.e. many small + few large — e.g. sparks). |

### `motion` knob grammar

`motion = <kind> [knob=value ...]` — the kind keyword is case-insensitive;
every knob after it is `name=value`, any order, and an unrecognised knob or
kind is a parse error naming the `.lvl` line (never a silently-ignored typo).
`attract` is the one exception: its target point is three bare floats
(`X Y Z`) before the `knob=value` list, matching every other `X Y Z` triple in
this file's grammar.

## Glyph families

`family =` selects a glyph family from the `re_render` registry. The canonical
names are:

```
haze_o   haze_dot   haze_wave   haze_colon   haze_bar
growth_dot   growth_density   growth_bars
anim_spin   anim_arc   anim_sweep
lean_fill   lean_stub
```

- `haze_*` — soft, sparse motes for mist/fog.
- `growth_*` — glyphs that read as a particle visibly growing/shrinking with age.
- `anim_*` — animated cycles (spin/arc/sweep) for sparks.
- `lean_*` — thin/streak glyphs.

An **unknown family name is not an error** — it passes through verbatim and the
renderer falls back to `growth_dot` at resolve time. Use `glyph_set` to bypass
the registry with a bespoke one-off sequence.

## Clamping (validate)

After parsing, each emitter is clamped by `EmitterSpec::validate()`:

- `spread_deg` → `[0, 180]`
- `lifetime` → `≥ 0.01`
- `lifetime_spread` → non-negative, capped so the shortest life is `≥ 0.01 s`
- `max_particles` → `≥ 1`
- `rate`, `speed`, `speed_spread`, `path_jitter` → `≥ 0`
- `size`, `growth` → `≥ ε`

So `spread_deg = 999`, `rate = -50`, `size = 0`, or `max_particles = 0` are all
silently corrected to sane values rather than crashing the renderer.

## Worked examples

### 1. Rising ember (a wall torch)

A buoyant upward stream, bright gold → orange → ember-red over age, with a quick
bold flash at birth.

```ini
[emitter ember_torch]
origin_world    = 2.5 2.0 1.0
colors          = 0xFFD27F 0xFF7A1A 0x6E1200
color_mode      = RampOverAge
family          = growth_dot
bold            = ByAge
direction       = 0.0 0.0 1.0
spread_deg      = 25.0
size            = 1.0
growth          = 1.4
rate            = 26.0
speed           = 0.9
speed_spread    = 0.2
path_jitter     = 0.05
lifetime        = 2.2
lifetime_spread = 0.3
accel           = 0.0 0.0 0.35
max_particles   = 256
seed            = 101
blend           = KeepBg
```

### 2. Drifting haze curtain

A nearly weightless cool mist that wanders (`path_jitter`) and curls
(`swirl`), long-lived, with the wall/holo background showing through (`KeepBg`).

```ini
[emitter drift_haze]
origin_world    = 6.5 5.0 1.2
colors          = 0x9FB8D8 0x4A7AA8 0x2A3A52
color_mode      = RampOverAge
family          = haze_dot
bold            = Off
direction       = 0.0 0.0 1.0
spread_deg      = 60.0
speed           = 0.35
speed_spread    = 0.12
path_jitter     = 0.4
swirl           = 0.6
lifetime        = 4.5
lifetime_spread = 0.6
accel           = 0.0 0.0 -0.01
max_particles   = 256
seed            = 404
blend           = KeepBg
```

### 3. Swirling spark burst (one-shot)

A single explosion: `burst` emits exactly N sparks at activation (no continuous
`rate`), fired omni-directionally (`spread_deg = 180`) and pulled back down by
gravity. `Opaque` so the bright sparks read solid.

```ini
[emitter spark_burst]
origin_world    = 5.0 3.0 1.6
colors          = 0xFFFFFF 0xFFD060 0x803000
color_mode      = RampOverAge
family          = anim_spin
bold            = On
direction       = 0.0 0.0 1.0
spread_deg      = 180.0
burst           = 64
speed           = 2.6
speed_spread    = 0.9
lifetime        = 1.1
lifetime_spread = 0.2
accel           = 0.0 0.0 -3.0
max_particles   = 128
seed            = 707
blend           = Opaque
```

### Bonus: screen-pinned HUD spark

`origin_screen` pins the emitter to a fixed terminal cell (never occluded by
geometry) — useful for HUD/overlay effects.

```ini
[emitter hud_spark]
origin_screen   = 6.0 3.0
colors          = 0x60FF60 0x30A030 0x103810
family          = haze_wave
direction       = 0.0 1.0 0.0
spread_deg      = 20.0
rate            = 8.0
lifetime        = 2.5
max_particles   = 96
seed            = 808
blend           = KeepBg
```

## Known gaps

- **`color_jitter.dance`** is parsed, validated and stored, but no shading code reads
  it yet. The birth-time jitter (`h`/`s`/`l`) is live. Wiring `dance` (let the
  jittered hue wander over the particle's life) is a small, self-contained task.
- **Weather fields** (ambient rain or snow that fills a region rather than
  streaming from one origin) are not built. Snow today is produced per level by
  `re_level`'s occupancy-aware snow emitters.
- **Parser tests** for the variety keys in `re_level` are thin. The keys are covered
  indirectly by the `re_world` / `re_render` emitter tests and by the showcase levels
  loading and passing `lvlcheck`. Direct parse-success and parse-error tests per key
  are welcome.
- `visibility = lit` and `flock … seek_light` need the level's `[light]` list. The
  host builds its live emitter set, and it must resolve each spec through
  `Emitter::from_spec_lit(spec, &level.lights)`; `Emitter::from_spec` passes an empty
  light list, which leaves those emitters dark. The reference game does this.
  Moving emitter-set construction into `re_runtime` belongs with the generic level
  player on the [roadmap](../ROADMAP.md).

## Perf (particle budget)

No new per-particle cost is unconditional: `Curl`/`Lorenz`/`Attract` motion add
a handful of scalar ops per particle per tick (same order as the existing
turbulence/swirl path); `Flock` is the one that changes complexity — it is
O(k²) per tick among an emitter's own live particles (`k` = that emitter's
live count, capped by both `max_particles` and the `flock n=` knob), evaluated
once per tick in a pre-pass (`apply_flocking`) before the ordinary integrate
step. A level author who wants many independent flocks should keep each
flock's own `n`/`max_particles` modest (tens, not hundreds) rather than one
giant flock — the showcase `EM-flock.lvl` stays at `n=24`. `splat_particles_kinds`
now allocates one `CellAccum` per screen cell every frame (`cols*rows`, e.g.
~2,000 for an 80×26 terminal) instead of writing directly into the output
layer — a fixed, frame-sized scratch buffer, not proportional to particle
count.

## See also

- `engine/re_world/src/emitter_spec.rs`: the `EmitterSpec` contract type, field docs,
  defaults and `validate()` rules.
- `engine/re_render/src/emitter.rs`: the runtime `Emitter` / `ParticlePool` /
  `tick_emitters` / `splat_particles_kinds` implementation.
- `engine/re_render/src/ripple.rs`: the ripple ring math.
- `engine/re_level/src/lib.rs`: the `[emitter]` parser (`Section::Emitter`).
- `levels/21-emitter-showcase.lvl`: a tech-demo room exercising the full control set.
- `levels/EM-*.lvl`: one showcase level per behaviour family (motion kinds, trails,
  bounce, word particles, dust visibility, dense splat).
- `levels/GODRAYS-demo.lvl`: the colour-only `kind = godrays` emitter, which tints
  the cells a light shaft crosses instead of drawing glyphs.
- [LIGHTING.md](LIGHTING.md) for the `[light]` sections that `lit` emitters read.
