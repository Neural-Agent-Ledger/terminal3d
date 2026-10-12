# Smart-material catalog

The terminal-render look is built on **smart materials**: a [`Material`] is pure
DATA — a procedural `recipe` + two iq-cosine palettes (`palette` / `bg_palette`) +
a relief/lighting model (`light`) + an `animation` + a glyph-stability tier +
optional reflectivity + an underline accent. A "material" is a *config*, not code;
adding a look is a new data row, not a new shader branch.

This file catalogs the named **built-in materials** the renderer ships and how a
level authors them by name. The catalog lives in
`engine/re_render/src/ascii_shader.rs` (`catalog_specs`, registered by
`MaterialRegistry::with_catalog`); the surface/puzzle/hologram built-ins (slots
0–7) are unchanged and the catalog is appended at slots 8+.

## Composable fields (the "knobs")

| Field | What it controls |
|-------|------------------|
| `recipe` | The pattern primitive: `Brick`, `Circuit`, `Grain`, `Sky`, `Hologram`, `CodeWall`, `Scanline`, `Truchet`, `Fractal`, `Substrate`, `Voice`, `Image`. Every kind is authorable directly with `recipe = KIND k=v …` — see § "Authoring a recipe directly" below. |
| `palette` / `bg_palette` | iq cosine gradients (12 floats each) → the fg glyph colour and the bg base colour. |
| `light` (`LightModel`) | `height_scale`, `normal_strength`, `shininess`, `specular`, `spec_color`, `ao_strength`, `emissive` — the relief + Blinn-Phong sheen + self-illumination. |
| `animation` (`Animation`) | `kind` = `None`/`Pulse`/`Shift`/`Flicker`, plus `speed` + `amplitude`. |
| `stability` (`GlyphStability`) | `Responsive` / `Stable` / `Frozen` — the anti-flicker glyph deadband. |
| `reflective` + `clarity` | Turn the surface into a one-bounce mirror; `clarity` 1 = crisp, 0 = collapsed. |
| `underline_accent` | A subtle colored underline substyle (scan-glow / specular glint) under inked cells. |
| `family` (`Option<FamilyLayer>`) | The KINDRED-SYMBOL floating-glyph layer: a `GlyphFamily` + one of the four `time`/`distance`/`normal`/`state` drivers, REPLACING the recipe glyph. `None` = unchanged. |
| `super_glyph` (`Option<SuperGlyphLayer>`) | The SUPER-GLYPH layer: a `GlyphFamily` + a `SuperSelector` (the richer drivers below). Runs right after `family` with the same frozen/quantized stability, and likewise OWNS the cell glyph. `None` = unchanged. |

### The `Scanline` recipe (new)

A FILLED directional-band primitive added for looks the void-bg `Hologram` and the
cellular `Brick`/`Circuit` recipes don't cover cleanly (CRT rasters, brushed metal,
woven fabric, grates). Unlike the hologram, band gaps keep a real dark `bg` fill, so
it reads as a solid panel. Params: `freq` (bands per world-cell), `duty` (inked
fraction `[0,1]`), `warp` (fBm jitter that breaks straight lines into a grain),
`axis` (`0` = horizontal raster, `1` = vertical grate, `2` = woven cross-hatch).
Self-lit (the relief pass leaves it alone), so `emissive` drives the glow.

## Authoring a recipe directly — `recipe = KIND k=v …` (2026-09-02)

`preset =` pulls a whole catalog row (palette, relief, animation, stability…).
`recipe =` authors the PATTERN alone and combines with the explicit `fg`/`bg`/`glyph`/
`animation`/`reflective` keys. Until 2026-09-02 only seven kinds were reachable this
way; the five pattern recipes below could be had only by naming a preset. All twelve
kinds now parse (`re_level::parse_recipe`), serialize (`re_voximport::to_lvl_text`,
every knob written explicitly so save→load is the identity), and cycle in the editor.

Knob syntax: floats as decimals (`duty=0.55`), integers bare (`axis=0`, `seed=7`),
keywords from a named set (`mode=hold`). Unknown keys are ignored; an unknown KIND or
an unknown keyword value is a parse error — a level never silently gets a different
surface than it asked for. Defaults are the values of the matching preset, so
`recipe = scanline` and `preset = monitor` draw the same pattern.

| `recipe =` | knobs (defaults) | same pattern as `preset =` | example |
|---|---|---|---|
| `brick` | `aspect=2 mortar_w=0.16 grain=0.6` | the `wall_solid` built-in | `recipe = brick aspect=3` |
| `circuit` | `scale=1.6 trace_w=0.12` | — | `recipe = circuit scale=2` |
| `grain` | `scale=3 speck=0.3 tile_eps=0.06` | — | `recipe = grain speck=0.5` |
| `sky` | `scale=6 density=0.1` | — | `recipe = sky density=0.2` |
| `hologram` | `line_freq=6 scan_w=0.5` | — | `recipe = hologram line_freq=8` |
| `codewall` | `panel=0` | — | `recipe = codewall panel=2` |
| `scanline` | `freq=9 duty=0.55 warp=0.05 axis=0` (`axis` 0 h / 1 v / 2 woven, clamped) | `monitor` | `recipe = scanline freq=9 duty=0.55 warp=0.05 axis=0` |
| `truchet` | `seed=0 density=0.5 scale=4` | `conduit` | `recipe = truchet seed=7 density=0.5 scale=4` |
| `fractal` | `arms=0 octave_warp=0.3 twist=0` | `portal_zoom` | `recipe = fractal arms=3 octave_warp=0.3 twist=0.2` |
| `substrate` | `scale=3 held=0.55 released=0.3` | `substrate` | `recipe = substrate scale=3 held=0.55 released=0.3` |
| `voice` | `scale=3 mode=hold` (`mode` = `hold` \| `release`) | `mnemos_hold` / `lethe_release` | `recipe = voice scale=3 mode=release` |
| `image` | bare keyword; the picture comes from `texture = NAME` | — | `recipe = image` + `texture = rock` |

Fixture: `levels/91-recipe-kinds.lvl` skins its five surfaces with the five pattern
recipes. Open it in the editor to see it in the live preview:
`cargo run --release -p ts_editor -- levels/91-recipe-kinds.lvl`.

## Authoring by name — `preset = NAME`

A level's `[material]` section can pick a catalog material **by name** and bind it
to a surface:

```ini
[material myrust]
preset  = rust          # pull the whole `rust` config
surface = wall_solid

[material screen]
preset  = monitor       # CRT phosphor + flicker
glyph   = @             # explicit keys still override the preset
surface = wall_trace
```

A bare `preset = NAME` pulls the catalog material's full config. Any explicit key on
the same `[material]` (`fg`, `bg`, `glyph`, `recipe`, `animation`, `reflective`,
`clarity`, `rank`) **overrides on top**. The legacy `recipe = …` / `fg = …` path is
unchanged — `preset` is purely additive. Surfaces bindable: `wall_solid`,
`wall_trace`, `wall_matrix`, `floor`, `ceiling`.

See the demo level in the editor's live preview:

```sh
cargo run --release -p ts_editor -- levels/90-material-catalog.lvl
```

## Image textures — `[texture NAME]`

An `[material] recipe = image` takes its picture from a named `[texture]` block:

```ini
[texture pic]
file     = assets/texture/genplate_tex.png
max_size = 1024          # optional: cap the LONGEST EDGE at decode time
```

| Key | Type | Default | What it does |
|---|---|---|---|
| `file` | path | — | resolved against the level's own directory, then the level tree's parent, then the CWD |
| `max_size` | integer ≥ 1 | unset (native size) | box-downsamples the image on load so its longest edge is at most `N`, keeping the aspect ratio |

`max_size` is a **memory** lever, not a look lever. An ASCII cell samples a handful
of texels, so a 9428×3548 photo — about **134 MB** of RGBA held for the whole
session — carries no more legible detail than its 1024-wide reduction (~1.6 MB).
The reduction uses a triangle (box-average) filter, which is what a minifying
sampler wants; nearest-neighbour would alias exactly the detail the cap folds away.
Unset, the image is kept at its native size and nothing changes.

Unrecognised keys inside `[texture]` are reported by `re_lvlcheck` (they used to be
swallowed in silence, so a typo looked accepted and did nothing).

### Relief — making a photo read as a lit surface (2026-09-02)

By default an image material is **self-lit**: the sampled pixel *is* the final
colour, so a photo on a wall reads as a flat decal pasted on. Two keys turn it into
a surface the level's light actually falls on.

```ini
[texture pic]
file   = assets/texture/brick.png
height = assets/texture/brick_height.png   # optional 8-bit relief map

[material wall_pic]
recipe = image
texture = pic
tile   = 4 2                                # repeat 4x across, 2x up the span
relief = height_scale=1.2 normal_strength=0.9
```

| Key | Where | Type | Default | What it does |
|---|---|---|---|---|
| `height` | `[texture]` | path | unset | an 8-bit relief map: `0` = lowest, `255` = highest. Resolved, `max_size`-capped and (if its dimensions differ) **resampled to the colour image's dimensions**, so texel *(x, y)* of both channels is the same point on the surface |
| `tile` | `[material]` | `A B`, both finite and `> 0` | `1 1` | how many times the figure repeats across the mapped span. A tiled UV **wraps**; an untiled one is clamped exactly as before |
| `relief` | `[material]` | `height_scale=… normal_strength=…` | unset | flips the image **off** self-lit and routes its height field through the texture's `height` map. Both knobs optional (a bare `relief =` means full strength); an **unknown knob name is a parse error**, not a silent default |

Notes:

* `height` alone changes nothing — the surface only becomes lit relief when a
  material bound to that texture also authors `relief =`. That is deliberate: one
  texture can be shared by a self-lit sign and a lit wall.
* `relief` on an image with **no** height map is not an error: the height field is a
  constant, so the surface lights as a flat face. Defined, not garbage.
* A malformed `tile` (`0`, negative, non-numeric, wrong arity) is a **line-numbered
  parse error**. A `tile = 0 4` that silently became `1 1` would be debugged in the
  renderer instead of in the level.
* An image material that authors none of these keys shades **byte-identically** to
  before the keys existed — the golden framebuffer corpus is unchanged by them.

## The catalog

| Name | Evokes | Recipe | Palette mood | Relief | Emissive | Animation | Reflective | Stability |
|------|--------|--------|--------------|--------|----------|-----------|------------|-----------|
| `metal` | Brushed steel panel | Scanline (h-grain, warp) | Cool grey | Low, high spec sheen | ~0 | — | — | Stable |
| `gold` | Brass / gold bar | Scanline (v-grain) | Warm amber | Low, very high spec | faint lustre | — | — | Stable |
| `glass` | Clean window pane | Scanline (woven, sparse) | Cool faint blue | Flat, crisp glint | faint | — | **yes** (clarity 0.92) | Stable |
| `stone` | Concrete / masonry | Brick (wide mortar) | Cool grey | Deep relief, **deep AO** | 0 | — | — | Stable |
| `rust` | Corrosion / oxide | Scanline (h, **high warp**) | Warm orange-brown | Rough, matte | ~0 | — | — | Stable |
| `ice` | Frost / crystal | Circuit (fine) | Pale blue | Faceted, glint | faint | — | yes (clarity 0.55) | Stable |
| `wood` | Timber grain | Scanline (h-grain) | Warm brown | Soft satin | 0 | — | — | Stable |
| `fabric` | Cloth / weave | Scanline (**woven**) | Muted dusty | Soft, low | 0 | — | — | Stable |
| `water` | Liquid surface | Circuit | Cool blue | Rippled, glint | faint caustic | **Shift** (flow) | yes (clarity 0.45) | Responsive |
| `lava` | Molten energy | Scanline (woven, warp) | Hot orange-red | Mid | **strong glow** | **Pulse** | — | Responsive |
| `neon` | Sign / tube | Scanline (v-bands) | Saturated magenta | Flat | **strong** | **Flicker** | — | Responsive |
| `monitor` | CRT / screen | Scanline (**h raster**) | Phosphor green | Flat | **strong** | **Flicker** | — | Responsive |
| `flesh` | Organic / skin | Grain | Warm pink-red | Soft | subsurface | **Pulse** (slow) | — | Stable |
| `void` | Deep space | Sky (sparse) | Near-black + stars | Flat | speck glow | — | — | **Frozen** |
| `holo_amber` | Amber hologram | Hologram | Amber/gold | Flat (self-lit) | strong | Shift (scroll) | — | Responsive |
| `circuit_red` | Hot circuit board | Circuit (dense) | Red-orange | Pad/trace relief | mid | Pulse | — | Responsive |
| `haze` | Drifting mist | Grain + HAZE_O float | Soft blue-grey | Flat | faint | **family (Time)** | — | Stable |
| `flow` | Moving stream | Grain + ANIM_ARC float | Cool teal | Flat | faint | **family (Time)** | — | Stable |
| `spark` | Embers (growth) | Grain + GROWTH_DOT | Hot orange | Flat | strong | **family (Distance)** + Pulse | — | Stable |
| **`vitrine`** | **Clear glass pane** | **Hologram (sparse)** | **Cool faint blue** | Flat (self-lit) | faint | Shift (slow) | — | Responsive |
| **`screen`** | **Translucent HUD display** | **Hologram (dense)** | **Phosphor green** | Flat (self-lit) | strong | Shift (fast) | — | Responsive |
| **`veil`** | **Gauze curtain** | **Hologram (mid)** | **Pale violet** | Flat (self-lit) | mid | Shift (slow) | — | Responsive |
| `moss` | Organic lichen crust | Grain + HAZE_DOT float | Green | Soft relief | ~0 | **family (Time, slow)** | — | Stable |
| `ember_field` | Bed of coals (LOD growth) | Grain + GROWTH_DENSITY | Hot orange | Low | strong | **family (Distance)** + Pulse | — | Stable |
| `static_noise` | TV static / interference | Grain + ANIM_SPIN float | Grey | Flat | faint | **family (Time, fast)** + Flicker | — | Responsive |
| `lattice` | Woven steel mesh | Scanline (woven) | Steel-blue | Mid, sheeny | ~0 | — | — | Stable |
| `beat_grid` | Pulse-on-beat panel | Circuit + GROWTH_DENSITY | Cool blue | Low, faint glow | mid | **super-glyph `Bpm` (120)** | — | Stable |
| `auto_tile` | Auto-tiling woven panel | Grain + GROWTH_BARS | Warm sandstone | Soft relief | ~0 | **super-glyph `NeighborAware`** (static) | — | Stable |
| `boil` | Boiling procedural field | Grain + HAZE_DOT | Sickly green-grey | Low | faint | **super-glyph `Procedural`** | — | Stable |

> **Transparent materials (bold above — `vitrine` / `screen` / `veil`)** route into
> the layer composite's TRANSPARENT slot (`OverKeepBg`): their glyph + fg draw OVER
> the scene behind them and the resolver keeps that background. They use the
> glyph-only `Hologram` recipe (see [`Material::is_transparent`]), so most of each
> cell is a void gap the background shows through. `glass` stays a REFLECTIVE-opaque
> mirror by design (it replaces the cell with a one-bounce reflection), not a
> see-through pane — use `vitrine` for a clear window.

## Super-glyph selectors

Beyond the four `glyph_family::FamilyDriver` kinds (`time` / `distance` / `normal` /
`state`), `ascii_shader` provides the super-glyph SELECTORS as the
[`SuperSelector`] / [`SuperGlyphLayer`] API — each picks a member of a `GlyphFamily`
from a richer, ECONOMICAL (one quantized table lookup) and TEMPORALLY STABLE
(deadbanded / steppy) driver:

| Selector | Driver | Animated? |
|----------|--------|-----------|
| `Bpm { bpm }` | beat-quantized cycle (on-beat, steppy) | yes |
| `Density` | the quantized LOD density (sparse→dense member) | no (cacheable) |
| `NeighborAware` | hash of the cell + 4 axis-neighbour world buckets (auto-tiling) | no (cacheable) |
| `MultiplyComposite { speed }` | distance picks the window × time floats within it | yes |
| `PriorityComposite { speed }` | `state != 0` pins an interrupt frame; else a slow idle float | yes |
| `Procedural { speed }` | world-hash × quantized time step (endless boil) | yes |

Every selector is deterministic and stable under a sub-cell camera / sub-step time
nudge (it reads the same quantized drivers the rest of the shader uses). The
distance, time, normal/incidence and state selectors already live on a material via
its `family` field (`FamilyLayer`); these six are the selectors beyond that set.

These selectors are now **authorable on a material** through the `super_glyph:
Option<SuperGlyphLayer>` field, wired in `Material::sample` right after the `family`
block with the IDENTICAL frozen / quantized-density / stability handling, so far and
idle cells do not flicker. Three catalog materials exercise them and resolve by
`preset = NAME` like any other:

- **`beat_grid`** — `Bpm { bpm: 120 }` over `GROWTH_DENSITY` (`. : ░ ▒ ▓ █`): the
  glyph STEPS on the beat (phase from `ctx.time`), pulsing with the audio layer.
- **`auto_tile`** — `NeighborAware` over `GROWTH_BARS` (`▁▂▃▄▅▆▇█`): adjoining cells
  pick CONNECTED members from a world-lattice hash (auto-tiling); static, cacheable.
- **`boil`** — `Procedural { speed: 1.5 }` over `HAZE_DOT` (`. · ∙ • ●`): a
  world-seeded "boiling" field that advances in stable quantized time-steps.

> **Beat phase note:** the `Bpm` selector derives its beat phase from `ctx.time`
> (there is no `bpm_phase` on `SampleCtx`). When the audio transport's beat clock is
> threaded into the render context, swap that `time` for the quantized beat phase in
> `Material::super_glyph_glyph` — a one-line change confined to that helper.

Notes:

- **Accents:** `metal`/`gold` carry a dotted **specular-ridge** under-glint
  (`LitRidge` gate); `lava`/`neon`/`monitor` carry a curly **scan-glow** under every
  inked cell (`Inked` gate); `holo_amber` inherits the hologram scan-glow.
- **Variants** (`holo_amber`, `circuit_red`) deliberately re-skin an existing recipe
  with only a different palette/animation, demonstrating the config-as-data point.
- Everything is deterministic and allocation-free in the shade path; an animated
  catalog material (e.g. `water`, `monitor`) flips the registry's `any_animated`
  flag so the renderer keeps re-shading.

## Enumerating the catalog in code

```rust
use re_render::ascii_shader::{ids, MaterialRegistry};

for name in ids::CATALOG_NAMES {
    let m = MaterialRegistry::catalog_material(name).unwrap();
    // m.recipe, m.light.emissive, m.animation.kind, m.reflective, …
}
```
