# Per-level lighting and atmosphere

The authoring reference for `[light]` and `[atmosphere]` in a `.lvl`.
Demo level: `levels/22-lit-cellar.lvl`
(`cargo run --release -p ts_editor -- levels/22-lit-cellar.lvl` shows it in the
editor's live preview).

A level that authors **none** of these sections shades exactly as it always has.
Every default below is the engine's own built-in, and the golden corpus is
byte-identical with the feature present.

---

## `[light]` — punctual (point) lights

Repeatable: one section per lamp, up to **4** live lamps
(`re_render::ascii_shader::MAX_POINT_LIGHTS`). Extras are dropped, and
`re_lvlcheck` warns at author time rather than letting them vanish silently.

```
[light]
pos       = 4.0, 5.0, 3.2     # x, y on the floor plane; z is HEIGHT
color     = #FFB060           # packed #RRGGBB
radius    = 5.0               # > 0, in world cells
intensity = 1.1               # >= 0; 1.0 = the authored colour at full
```

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `pos` (alias `at`) | `x, y, z` | required in practice | **`z` is height**, same convention as `CameraPose::pos` |
| `color` | `#RRGGBB` | `#FFFFFF` | |
| `radius` | f32 > 0 | `6.0` | at `d == radius` the lamp is at **half** strength |
| `intensity` | f32 ≥ 0 | `1.0` | `0` makes the lamp inert: it is dropped, and lvlcheck warns |

**The shading term** (`ascii_shader::point_light_sum`, unshadowed):

```
Σ  radiance · max(N·L, 0) / (1 + d² / radius²)
```

`radiance = color × intensity`. The `+1` in the denominator keeps a lamp finite as
`d → 0`. Point lights are added to BOTH the colour term and the relief term that
drives the glyph ramp: a cell inside a pool gets brighter **and** denser ink, which
is the whole reason to author one.

They are **unshadowed** in this revision (shadowed point lights are not built yet),
so a lamp inside a wall or inside an `Object` cube still lights the room through it.
Hang lamps in open air; `re_lvlcheck` warns when one is buried.

Because a point light is a function of position and normal only, it survives the
renderer's view-independent mode, unlike the headlamp.

## `[atmosphere]` — key light, ambient, height fog, post-fx

One section per level. Every key is optional; an unset key keeps the built-in.

```
[atmosphere]
key_dir     = -0.3, -0.5, 0.8   # points TOWARD the light; normalised on load
key_color   = #6070A0
ambient     = 0.12              # [0,1] grey floor — nothing goes pure black
fog_height  = 0.4 0.6           # base z, falloff per world unit above it
postfx      = clean             # a re_postfx preset name
```

`fog_color` / `fog_near` / `fog_far` are accepted here too, as **aliases** of the
`[palette]` keys. They land in the same fields, so author fog in one section or the
other, never both.

| Key | Type | Default |
|-----|------|---------|
| `key_dir` | `x, y, z`, non-zero | the built-in upper-front-left rake |
| `key_color` | `#RRGGBB` | near-white `(0.95, 0.96, 1.0)` |
| `ambient` | f32 in `[0,1]` | `0.18` |
| `fog_height` | `base falloff`, `falloff >= 0` | off |
| `postfx` | one of `re_world::POSTFX_PRESETS` | none |

**Height fog** multiplies the existing distance fog by a clarity factor

```
exp(-falloff · max(0, z − base))
```

so the murk thins with height: a room gains vertical air instead of a flat grey
wash. Unauthored, the term is a literal `1.0` and costs nothing.

**Post-fx precedence.** `postfx =` is what the level *asks* for. The engine only
carries the request (`re_world::Atmosphere`); resolving it against the player's own
settings is the host game's job. A host should let an explicit player choice win,
resolve the chain on every level load (so a preset never leaks from the previous
room), and restore the player's preset when the level is torn down.

## Authoring notes

- Lamps read best hung **above** the geometry they light, at `z` a little under the
  ceiling, with `radius` around half the room's short axis, so pools overlap faintly
  instead of flooding the floor evenly.
- Pull `ambient` down (0.10–0.15) before adding lamps. At the default 0.18 the room
  is already lit and a lamp only washes it out.
- A dimmed, tinted `key_color` (the cellar uses a cold `#6070A0`) lets the lamps own
  the warm end of the frame.
- The whole set round-trips through `re_voximport::to_lvl_text`, so an editor save
  never drops a lamp.

## Where it lives in the code

| Piece | Where |
|-------|-------|
| Authored data | `re_world::{PointLight, HeightFog, Atmosphere}` |
| Parser | `re_level`, `Section::Light` / `Section::Atmosphere` |
| Runtime light set | `re_render::light_env_for_map` → `ascii_shader::{PointLights, LightEnv}` |
| Shading | `ascii_shader::{point_light_sum, light_surface_mode, height_fog}` |
| World anchor | `LightCtx::at(world_pos)`, applied once in `re_render::shade_cell` |
| Serializer | `re_voximport::to_lvl_text` |
| Linter | `re_lvlcheck`, the `lighting` check (WARN only) |
| Editor | `ts_editor` Settings rows for `ambient`, `key_color`, `fog_height` (base + falloff) and `postfx`. `key_dir` and the `[light]` sections stay hand-authored; the editor loads and re-saves them untouched. |
