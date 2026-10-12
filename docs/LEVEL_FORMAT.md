# The `.lvl` level format

A short guide to the format and where each part is documented. A single complete
reference page does not exist yet; writing it is a
[good first issue](../ROADMAP.md#good-first-issues).

The authority is the parser, `re_level` (`engine/re_level/src/lib.rs`). Its crate
docs (`cargo doc -p re_level --open`) and the `Level` struct's field docs describe
every key it reads. When this page and the parser disagree, the parser is right;
please open an issue.

## Shape of a file

A level is a UTF-8 text file with INI-style sections. `#` starts a comment. Keys are
`key = value`. Some sections take a name (`[material pulse_wall]`,
`[emitter ember_torch]`) and may repeat.

```ini
# A minimal room: 12×8 cells, 4 high, one player spawn.
[meta]
name  = my-room
title = My Room

[room]
size    = 12 8 4
ambient = solid

[player]
spawn  = 3 3 1
facing = +x
```

The coordinate convention is shared by every section: `x` runs west to east, `y`
north to south, `z` is height (floor at `z = 0`). See the `re_world` crate docs.

Error policy: an unknown key or a malformed value in most sections is a
**line-numbered parse error**, so a typo never silently becomes a default. A few
legacy sections are read leniently; `re_lvlcheck` reports what the parser let
through.

## Sections

| Section | Purpose | Where to read more |
|---|---|---|
| `[meta]` | Name, title, format revision tags, `prop_style` | `re_level::Level` field docs |
| `[room]` | Size, perimeter wall kind, `storey_height`, `power_budget`, `render_pipes` | `re_level::Level` field docs |
| `[cells]` | Explicit dense per-cell geometry (the editor's save format): one ASCII block per z-layer, optional `material_ids` | `re_level` docs |
| `[player]` | Spawn position and facing | `re_level` docs |
| `[node]`, `[pipe]`, `[inputs]`, `[door]`, `[pedestal]`, `[sequence]` | The graph-puzzle kit: nodes, wires with switches, initial activations, the goal | [LAYERING.md](LAYERING.md#the-graph-puzzle-kit-why-node-pipe-and-door-appear-in-the-engine), `re_puzzle` docs |
| `[part]`, `[emit]` | What an output node shows when it fires | `re_level` docs |
| `[material NAME]` | Surface looks: recipes, presets, palettes, relief, animation | [MATERIALS.md](MATERIALS.md) |
| `[texture NAME]` | Image textures and relief maps for `recipe = image` | [MATERIALS.md](MATERIALS.md#image-textures--texture-name) |
| `[palette]`, `[effects]` | Level colour palette, distance fog, per-level render effects | `re_level` docs |
| `[light]`, `[atmosphere]` | Point lights, key light, ambient, height fog, post-fx request | [LIGHTING.md](LIGHTING.md) |
| `[sky]` | Image sky (cylinder and caps); the procedural sky is chosen elsewhere | `re_render::sky` docs |
| `[emitter NAME]` | Particle emitters | [EMITTERS.md](EMITTERS.md) |
| `[wall_decal NAME]` | Decals on wall regions (for example a matrix panel) | `re_level` docs |
| `[holo NAME]` | Transparent hologram objects | `re_render::holo` docs |
| `[portal NAME]` | Non-euclidean warp-portal pairs | `re_render::portal` docs |
| `[vox NAME]` | Stamp a MagicaVoxel `.vox` model into the map | `re_level` docs, `re_voximport` docs |
| `[zone NAME]`, `[line NAME]`, `[inset NAME]` | Scalable material overlay: material zones, crisp lines, bounded insets | `re_level` docs |
| `[zone_input NAME]` | A region that drives a puzzle input while the player stands in it | `re_level` docs |
| `[trigger NAME]` | Trigger fields: region + action + once | `re_runtime::triggers` docs |
| `[note NAME]` | Readable text placards on walls | `re_runtime::notes` docs |
| `[prop NAME]` | Set-dressing props (powered or static) | `re_level::props` docs |
| `[field NAME]` | Sound fields bound to nodes, pipes, doors or regions | `re_level::fields`, `re_audio::field` docs |
| `[hierarchy]` | Object nesting: `child_kind index -> parent_kind index` | `re_level` docs |
| `[color_cell_ids]` | Per-cell colour-material ids (`x,y,z = id`) | `re_level` docs |
| `[map]`, `[win_when]` | Legacy blocks; read and skipped | — |

## Tools that read and write the format

- `ts_editor` edits levels with a live 3D preview and saves through the writer.
- `re_voximport::to_lvl_text` is the writer (`vox2lvl` and the editor both use it).
  Round-trips (load → save → load) are tested per section.
- `lvlcheck` (`cargo run --release -p re_lvlcheck -- levels/`) validates a level:
  structure, solvability of the puzzle graph, reachability, "can the player get
  stuck", lights, props and sound fields.
- `vox2lvl` builds a level from a `.vox` model plus a small `.toml`.

## Examples to learn from

`levels/00-test-load.lvl` exercises most section types in a few dozen lines.
`levels/22-lit-cellar.lvl` (lights), `levels/21-emitter-showcase.lvl` (particles),
`levels/90-material-catalog.lvl` (materials), `levels/STAIRS-showcase.lvl` (stairs
and storeys) and `levels/93-warp-portal.lvl` (portals) each focus on one feature.
