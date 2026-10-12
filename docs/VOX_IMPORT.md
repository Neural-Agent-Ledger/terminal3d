# VOX import pipeline — contract

Authoring a level in a voxel editor (Vengi, MagicaVoxel, or Goxel) and turning it
into a playable `.lvl`.

```
  voxel editor  ──►  level.vox  ─┐
  (Vengi/Magica)                 ├─►  re_voximport  ──►  level.lvl  ──►  re_level ──► Map/puzzle
  text editor   ──►  level.toml ─┘     (dot_vox +          (canonical           (existing engine,
                     (metadata)         tag derivation)     text format)         unchanged)
```

`re_voximport` does **not** invent a new runtime format. `.lvl` is already the
composable, sectioned, extensible scene format (`docs/LEVEL_FORMAT.md`). The
converter's whole job is `level.vox + level.toml → level.lvl`, after which the
*unchanged* `re_level` parser + `finalize` + `re_world::stamp_geometry` take
over. This reuses the entire validated level path instead of a second route
into `Map`.

## Why this shape

- **Editor-agnostic.** The loader binds to `.vox` (the universal exchange
  format) + an MIT Rust loader (`dot_vox`), never to one editor's project
  format. Author in Vengi, MagicaVoxel, or Goxel — the loader is the same.
- **Future-proof.** `.lvl` already carries optional typed sections beyond raw
  geometry (`wall_decal`, `effects`, `palette`, `note`, `prop`, `field`, …).
  New content types become new optional `.lvl` sections plus new `.toml`
  tables, never a format rewrite. The voxel grid feeds the *geometry + graph*
  layer only; everything else is data layered on top.

## Geometry is the voxel grid; the GRAPH is color-tagged voxels

`.vox` stores, per voxel, a palette **index** into a 256-colour palette
(RGBA). The converter treats specific palette **colours** as semantic tags so
the level *reads as the puzzle* while you sculpt it. Matching is by RGB
(nearest within a small tolerance, default exact); every tag's RGB is
overridable in the `.toml`.

### Default tag palette

| Tag           | Default RGB | Palette idx | Becomes in `.lvl`                                           |
|---------------|-------------|-------------|-------------------------------------------------------------|
| `wall_solid`  | `#808080`   | 0           | perimeter / decal `CellKind::WallSolid`                     |
| `wall_trace`  | `#00A0FF`   | 1           | `[wall_decal] surface = trace` region                       |
| `wall_matrix` | `#00FF50`   | 2           | `[wall_decal] surface = matrix` region                      |
| `floor`       | `#404040`   | 3           | floor slab (optional — auto z=0 if omitted)                 |
| `ceiling`     | `#202020`   | 4           | ceiling slab (optional — auto z=top if omitted)             |
| `node_input`  | `#FFD000`   | 5           | `[node] is_input = yes`                                     |
| `node_output` | `#FF3030`   | 6           | `[node] is_output = yes`                                    |
| `node_hidden` | `#FFFFFF`   | 7           | `[node]` (intermediate)                                     |
| `pipe`        | `#8000FF`   | 8           | traced into `[pipe A->B] path = ...`                        |
| `switch`      | `#FF00FF`   | 9           | `[pipe] switch_at = ...` on the pipe it sits on             |
| `door`        | `#00FFFF`   | 10          | `[door] pos = ...`                                          |
| `player`      | `#FFA500`   | 11          | `[player] spawn = ...`                                      |
| `pedestal`    | `#606060`   | 12          | `[pedestal] center/volume` bounding box                     |
| `hologram`    | `#40FFFF`   | 13          | transparent hologram cells (`CellKind::Hologram`); glowing wireframe in renderer |
| `portal`      | `#FF40C0`   | 14          | non-euclidean warp-portal face (`CellKind::Portal`); DDA stops, ray re-based |
| `object`      | `#C06000`   | 15          | generic solid object cell (`CellKind::Object`); `side_data = 0` |

**Palette index contract**: indices 0–15 are append-only and frozen. A `.vox`
file exported by any version of the converter is re-importable by any later
version. New tags always append; existing indices never change.

Any voxel whose colour matches no tag (within tolerance) is treated as
`wall_solid` (a safe, visible default) and a warning is emitted listing the
unknown colours.

### Derivation rules

1. **Room size** = voxel bounding box (`max+1` per axis), unless `[room] size`
   is set explicitly in the `.toml`.
2. **Shell** = perimeter `wall_solid` at every z + floor(z=0) + ceiling(z=top),
   exactly as `stamp_geometry` already does. Authored wall/floor/ceiling
   voxels only need to mark *decal* deviations; a fully-walled room can be
   left to the shell.
3. **Decals**: contiguous boxes of `wall_trace` / `wall_matrix` voxels become
   one `[wall_decal]` region each (axis-aligned bounding box of the connected
   run).
4. **Nodes**: each `node_*` voxel becomes one `[node]`. Name/threshold/pools/
   colour come from the `.toml` `[[node]]` table, matched by `pos = [x,y,z]`.
   Role (`is_input`/`is_output`) comes from the tag colour (the `.toml` may
   override).
5. **Pipes**: every maximal 6-connected run of `pipe` voxels must terminate
   (be 6-adjacent) at **exactly two** node voxels, becoming one edge. The
   expanded run becomes `path`. Orientation: `from` is the endpoint that
   `is_input` (else the non-output endpoint); ties are resolved by `.toml`
   `[[edge]] from/to`. A run that does not end at exactly two nodes is an
   error (reported with its cells).
6. **Switch**: each `switch` voxel must be on (or 6-adjacent to) exactly one
   pipe path, setting that pipe's `switch_at` to the matching path index.
7. **Door / player / pedestal**: from their single tagged voxel (or the
   `.toml`).
8. Everything non-spatial (`meta name/title`, `room ambient`, `inputs`,
   `effects`, `palette`, notes, props) is copied straight from the `.toml`.

## Companion `.toml` schema

```toml
[meta]
name    = "01"
title   = "FIRST CONTACT"
ambient = "trace"            # -> [room] ambient

# Optional: override the tag -> RGB table. Omit to use defaults above.
[tags]
pipe = "#8000FF"

[[node]]
pos       = [3, 3, 2]        # matches the node voxel
name      = "in_a"
glyph     = "A"
pool      = ["0", "1"]
threshold = 1
# is_input/is_output inferred from tag colour; override here if needed.

[[edge]]                     # OPTIONAL - only to disambiguate pipe direction
from = "in_a"
to   = "out"

[inputs]                     # -> [inputs] initial activations
in_a = 1

[effects]                    # -> [effects]   (optional)
matrix = 0.3

[palette]                    # -> [palette]   (optional fog override)
fog_color = "#0a0f14"
```

Tables not present mean that `.lvl` section is simply omitted (the engine
already treats them as optional). New future sections attach the same way: a
new `.toml` table plus a new optional `.lvl` section, with no format break.

## Materials — `color = material KEY` (per-level surface skinning)

A level can re-skin its five bindable surfaces (`wall_solid` / `wall_trace` /
`wall_matrix` / `floor` / `ceiling`) by authoring a material set; surfaces it
leaves alone keep the engine built-ins. See `docs/MATERIALS.md` for the full
recipe catalog, preset list, and animation kinds — this section only covers
how the `.toml` maps onto it.

### Authoring (`.toml`)

```toml
[[material]]
name      = "floor_neon"
fg        = "#20FF80"          # foreground colour (#RRGGBB)
bg        = "#04140C"          # background colour (dual-channel)
glyph     = "+"                # OPTIONAL fixed base glyph (else recipe ramp)
recipe    = "grain scale=3.0 speck=0.5 tile_eps=0.08"   # OPTIONAL; recipe + params
rank      = 7                  # OPTIONAL detail-rank bias

[[material]]
name      = "wall_pulse"
fg        = "#66CCFF"
bg        = "#041018"
recipe    = "brick aspect=2.0 mortar_w=0.16 grain=0.6"
animation = { kind = "pulse", speed = 0.5, amplitude = 0.18 }  # OPTIONAL

# Bind a surface kind to a material NAME. Keys: wall_solid / wall_trace /
# wall_matrix / floor / ceiling. A name MUST reference a defined [[material]]
# (a typo is a hard error, never a silent fallback).
[surface]
floor      = "floor_neon"
wall_solid = "wall_pulse"
```

`recipe` names one of the recipe kinds in `docs/MATERIALS.md`'s catalog with
optional ` key=value` float/int/keyword params (omitted params use the
recipe's own default). `animation.kind` is `none` / `pulse` / `shift` /
`flicker`. A material the author leaves without a `recipe` inherits the bound
surface's built-in recipe and rank strategy; an animation left `none`
inherits the built-in's animation, so a flat `fg`/`bg` override alone still
rides the built-in look.

### `.lvl` section (emitted by the converter, parsed by `re_level`)

The converter emits one `[material NAME]` section per authored material, with
its `surface = KIND` binding, and the *unchanged* `re_level` parser reads it
into `Level.materials` + `Level.surface_materials` (both empty for a level
with no material section — fully back-compatible):

```
[material floor_neon]
fg = #20FF80
bg = #04140C
glyph = +
recipe = grain scale=3 speck=0.5 tile_eps=0.08
rank = 7
surface = floor
```

`Level::build_map` forwards the bound set to the `Map` (`Map::set_materials`)
as a fixed, **slot-ordered** list of five `re_world::MaterialDef`s — slot
order `wall_solid, wall_trace, wall_matrix, floor, ceiling`. An unbound
surface emits a `MaterialDef::fallback()` marker so the renderer keeps its
built-in for that exact slot. `re_render::registry_for_map` then builds the
`MaterialRegistry` from those, replacing only the authored slots. An authored
material is visible end-to-end, including in
`cargo run -p re_imgqa -- <level>`. See `levels/08-material-demo.lvl` for a
working fixture (a neon `+` floor).

### `[[color_material]]` — per-voxel colour-keyed materials

The surface-keyed set above binds one material to an entire surface kind (one
floor material, one brick material, …). `[[color_material]]` is the per-voxel
upgrade: paint individual voxels with specific palette colours in your voxel
editor; each colour maps to a full material definition, so two voxels of the
same surface kind but different colours carry **different materials**.

```toml
[[color_material]]
color  = "#AA0000"          # the palette colour this entry matches (REQUIRED)
name   = "red_wall"         # optional; auto-derived as "cm_AA0000" when absent
fg     = "#FF4444"          # same fields as [[material]]
bg     = "#200000"
recipe = "brick aspect=2.0"

[[color_material]]
color  = "#00AA00"
name   = "green_wall"
fg     = "#44FF44"
bg     = "#002000"
recipe = "grain scale=3.0 speck=0.4"
```

Matching is nearest-RGB within `color_material_tolerance` (squared Euclidean;
default 36 = plus/minus 2 per channel); set a custom tolerance in the `.toml`
root:

```toml
color_material_tolerance = 100   # wider tolerance (optional)
```

**What the converter produces:**

1. Each `[[color_material]]` entry is emitted as a standard `[material NAME]`
   section in the `.lvl` (consumed by the existing `re_level` parser).
2. Every geometry voxel that matched a `[[color_material]]` colour gets a
   **1-based material id** (index into the `[[color_material]]` list) written
   to the `[color_cell_ids]` section of the `.lvl`:

   ```
   [color_cell_ids]
   1,1,1 = 1
   3,1,1 = 2
   ```

`re_level` parses `[color_cell_ids]` and stamps each id into its cell's
`side_data` during `build_map`; `re_render` then resolves a non-zero id
through `MaterialRegistry::color_material` instead of the surface-kind
default, so the per-voxel look is applied end to end. The surface-keyed
materials (`[surface]` binding) continue to work as before —
`[[color_material]]` is additive and takes priority per cell where both
apply.

**API surface:**

```rust
// Full result with per-cell ids:
let DerivedLevel { level, cell_material_ids } =
    re_voximport::derive_level_full(&voxels, size, &cfg)?;

// Serialise to .lvl with the [color_cell_ids] section:
let lvl_text = re_voximport::to_lvl_text_with_ids(&level, &cell_material_ids);

// Back-compat paths (no [[color_material]] -> identical output):
let level = re_voximport::derive_level(&voxels, size, &cfg)?;
let lvl_text = re_voximport::to_lvl_text(&level);
```

## Crate

`editor/re_voximport` (depends on `re_level` + `dot_vox` + `serde`/`toml`).
- lib: `vox_to_level(vox_bytes, toml_str) -> Result<re_level::Level, ImportError>`
  and a `to_lvl_text(&Level) -> String` serializer (the inverse of the
  parser).
- bin `vox2lvl`: `vox2lvl level.vox level.toml > level.lvl`.

Tests build a tiny in-memory voxel grid (no external `.vox` file needed), run
the derivation, and assert the resulting `Level` + `build_map()` match
expectations — the same "world is not empty / graph wired correctly" guard
class used elsewhere.
