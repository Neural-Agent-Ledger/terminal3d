//! Regression tests for the `.lvl` parser + geometry stamping. Ported from the
//! legacy `tests/test_level.c` spirit: parse a real level, assert the network
//! shape, and guard against the empty-world bug after geometry stamping.

use super::*;
use re_world::CellKind;
use std::path::PathBuf;

/// Path to a repo-root level file (the workspace root is two dirs above the
/// crate manifest).
fn level_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../levels")
        .join(name)
}

#[test]
fn parse_01_network_shape() {
    let lv = load(level_path("01.lvl")).expect("01.lvl must parse");
    assert_eq!(lv.name, "level-01-first-pulse");
    assert_eq!(lv.size, (12, 8, 4));
    // Two nodes: R (input) and output.
    assert_eq!(lv.nodes.len(), 2, "expected 2 nodes");
    assert_eq!(lv.pipes.len(), 1, "expected 1 pipe");
    // Door present at the parsed anchor.
    assert_eq!((lv.door.x, lv.door.y, lv.door.z), (11, 3, 1));
    // target_parts auto-resolves to the single output node (index 1).
    assert_eq!(lv.door.target_parts, vec![1]);
    // The input + output flags landed.
    assert!(lv.nodes[0].is_input);
    assert!(lv.nodes[1].is_output);
    // Pipe endpoints resolved.
    assert_eq!(lv.pipes[0].from_node, 0);
    assert_eq!(lv.pipes[0].to_node, 1);
    // Player spawn + facing. Spawn nudged one cell west (7,3,1) so it faces the
    // switch and clears the lvlcheck spawn-facing WARN (still -x facing).
    assert_eq!((lv.player.x, lv.player.y, lv.player.z), (7, 3, 1));
    assert_eq!(lv.player.facing_yaw_col, 360); // facing = -x
}

#[test]
fn parse_06_polysemantic() {
    // 06.lvl is "L6 · Polysemantic": one hidden node (poly, T=2) feeding TWO
    // required outputs (out_face AND out_grid). Replaces the prior "Constellation"
    // content (commit 3fc1512).
    let lv = load(level_path("06.lvl")).expect("06.lvl must parse");
    assert_eq!(lv.name, "level-06-polysemantic");
    assert_eq!(lv.size, (16, 14, 5));
    // 5 nodes: s1 s2 (inputs) poly (hidden) out_face out_grid (outputs).
    assert_eq!(lv.nodes.len(), 5, "expected 5 nodes");
    // 4 pipes: s1->poly, s2->poly, poly->out_face, poly->out_grid.
    assert_eq!(lv.pipes.len(), 4, "expected 4 pipes");
    // The polysemantic hidden node has threshold 2 (fires on s1 + s2).
    let poly = lv.nodes.iter().find(|n| n.name == "poly").unwrap();
    assert_eq!(poly.threshold, 2);
    // The two outputs are both present and both threshold 1.
    let out_face = lv.nodes.iter().find(|n| n.name == "out_face").unwrap();
    let out_grid = lv.nodes.iter().find(|n| n.name == "out_grid").unwrap();
    assert!(out_face.is_output && out_grid.is_output);
    assert_eq!(out_face.threshold, 1);
    assert_eq!(out_grid.threshold, 1);
    // effects + palette parsed (matrix drift over indigo fog).
    assert!((lv.effects.matrix - 0.12).abs() < 1e-6);
    assert!(lv.palette.has_fog);
    // Two notes captured (model_thought then system_log), both with body text.
    assert_eq!(lv.notes.len(), 2);
    assert_eq!(lv.notes[0].kind, NoteKind::ModelThought);
    assert_eq!(lv.notes[1].kind, NoteKind::SystemLog);
    assert!(!lv.notes[0].text.is_empty());
    assert!(!lv.notes[1].text.is_empty());
}

/// The empty-world guard — after stamping geometry the map MUST have walls,
/// floor and ceiling (the same class of check that caught the legacy bug).
#[test]
fn stamp_01_is_not_empty() {
    let lv = load(level_path("01.lvl")).expect("01.lvl must parse");
    let map = lv.build_map();
    assert_eq!(map.size(), (12, 8, 4));
    let walls = map.count(CellKind::WallTrace); // ambient = trace-pattern
    assert!(walls > 0, "stamped 01 must have wall cells, got {walls}");
    assert!(map.count(CellKind::Floor) > 0);
    assert!(map.count(CellKind::Ceiling) > 0);
    // 2 node cells + 1 door anchor = 3 Object cells total.
    // Node cells have side_data = node index; door anchor has side_data =
    // DOOR_OBJECT_TAG (not 0 — that used to collide with node 0's tag).
    assert_eq!(map.count(CellKind::Object), 3);
    // The door anchor is a generic Object cell at its known position.
    assert_eq!(map.cell_at(11, 3, 1), CellKind::Object);
    // Node cells carry their index in side_data (node 0 and node 1).
    // (positions are level-specific; we only assert the total Object count above)
}

#[test]
fn stamp_06_is_not_empty() {
    let lv = load(level_path("06.lvl")).expect("06.lvl must parse");
    let map = lv.build_map();
    assert!(map.count(CellKind::WallTrace) > 0);
    assert!(map.count(CellKind::Floor) > 0);
    assert!(map.count(CellKind::Ceiling) > 0);
    // 5 node cells (s1 s2 poly out_face out_grid) + 1 door anchor = 6 Object cells.
    assert_eq!(map.count(CellKind::Object), 6);
}

/// A pipe's expanded path is contiguous (each step moves exactly one cell in
/// one axis — 6-connected). Optional per the spec, but cheap insurance.
#[test]
fn pipe_path_is_contiguous() {
    let lv = load(level_path("06.lvl")).expect("06.lvl must parse");
    for pipe in &lv.pipes {
        assert!(
            pipe.path.len() >= 2,
            "pipe {}->{} empty path",
            pipe.from,
            pipe.to
        );
        for w in pipe.path.windows(2) {
            let (ax, ay, az) = w[0];
            let (bx, by, bz) = w[1];
            let manhattan = (ax - bx).abs() + (ay - by).abs() + (az - bz).abs();
            assert_eq!(
                manhattan, 1,
                "non-contiguous step in pipe {}->{}: {:?} -> {:?}",
                pipe.from, pipe.to, w[0], w[1]
            );
        }
    }
    // Every 06 pipe declares a switch_at; ensure each resolved onto its path.
    for pipe in &lv.pipes {
        let idx = pipe
            .switch_cell
            .unwrap_or_else(|| panic!("{}->{} switch unresolved", pipe.from, pipe.to));
        assert!(idx < pipe.path.len(), "switch index in range");
    }
    // The polysemantic node feeds two outputs — both outgoing pipes exist.
    assert!(lv
        .pipes
        .iter()
        .any(|p| p.from == "poly" && p.to == "out_face"));
    assert!(lv
        .pipes
        .iter()
        .any(|p| p.from == "poly" && p.to == "out_grid"));
}

/// `[effects] glitch` parses into `effects.glitch`, clamps out-of-range, and
/// defaults to 0.0 when absent — mirroring the `matrix` knob exactly.
#[test]
fn effects_glitch_parses_clamps_and_defaults() {
    let base = "\
[meta]
name = g
[room]
size = 6 4 4
[node a]
pos = 1 1 1
is_input = yes
[node b]
pos = 4 1 1
is_output = yes
[pipe a->b]
path = (1,1,1) (4,1,1)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";

    // In-range value parses verbatim.
    let with = format!("{base}[effects]\nglitch = 0.4\n");
    let lv = load_string(&with, "<glitch>").expect("glitch level parses");
    assert!((lv.effects.glitch - 0.4).abs() < 1e-6, "glitch = 0.4");

    // Out-of-range clamps to [0, 1].
    let hi = format!("{base}[effects]\nglitch = 9.0\n");
    let lv = load_string(&hi, "<glitch-hi>").expect("clamped glitch parses");
    assert_eq!(lv.effects.glitch, 1.0, "glitch clamps to 1.0");
    let lo = format!("{base}[effects]\nglitch = -3.0\n");
    let lv = load_string(&lo, "<glitch-lo>").expect("clamped glitch parses");
    assert_eq!(lv.effects.glitch, 0.0, "glitch clamps to 0.0");

    // Absent → 0.0.
    let lv = load_string(base, "<glitch-absent>").expect("no-effects level parses");
    assert_eq!(lv.effects.glitch, 0.0, "absent glitch defaults to 0.0");
}

#[test]
fn missing_required_section_errors() {
    // No [pedestal] → finalize must reject.
    let src =
        "[meta]\nname = x\n[room]\nsize = 4 4 4\n[door]\npos = 0 1 1\n[player]\nspawn = 1 1 1\n";
    let r = load_string(src, "<test>");
    assert!(r.is_err(), "missing [pedestal] should error");
}

/// All shipped levels (01-07) must continue to parse after the switch_at
/// second-pass change (R-H6) and the mask-overflow fix (R-M8).
#[test]
fn all_shipped_levels_load() {
    for n in 1..=7 {
        let name = format!("{n:02}.lvl");
        let lv = load(level_path(&name)).unwrap_or_else(|e| panic!("{name} must load: {e}"));
        // every pipe with a declared switch resolved onto its path
        for p in &lv.pipes {
            if p.switch_at_raw.is_some() {
                let idx = p.switch_cell.unwrap_or_else(|| {
                    panic!("{name}: switch_at unresolved on {}->{}", p.from, p.to)
                });
                assert!(idx < p.path.len(), "{name}: switch index out of path range");
            }
        }
    }
}

/// R-H6: `switch_at` declared BEFORE `path` must still resolve (second pass),
/// where the old order-dependent inline resolution would silently drop it.
#[test]
fn switch_at_before_path_resolves() {
    let src = "\
[meta]
name = sw
[room]
size = 6 4 4
[node a]
pos = 1 1 1
is_input = yes
[node b]
pos = 4 1 1
is_output = yes
[pipe a->b]
switch_at = 2,1,1
path = (1,1,1) (4,1,1)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<sw>").expect("switch_at before path must resolve");
    let p = &lv.pipes[0];
    let idx = p.switch_cell.expect("switch resolved in second pass");
    assert_eq!(p.path[idx], (2, 1, 1));
}

/// R-H6: an off-path `switch_at` is a HARD error (matches legacy level.c).
#[test]
fn switch_at_off_path_is_error() {
    let src = "\
[meta]
name = sw
[room]
size = 6 4 4
[node a]
pos = 1 1 1
is_input = yes
[node b]
pos = 4 1 1
is_output = yes
[pipe a->b]
path = (1,1,1) (4,1,1)
switch_at = 1,3,1
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let err = load_string(src, "<sw>").unwrap_err();
    assert!(
        err.msg.contains("switch_at cell not on expanded path"),
        "expected off-path error, got: {}",
        err.msg
    );
}

/// R-M8: a `[part]` mask with more rows than `d` allows is a HARD error
/// (previously the extra row was silently dropped by the `is_mask_row` gate).
#[test]
fn part_mask_too_many_rows_is_error() {
    // d = 1, h = 1, w = 2 → exactly ONE mask row expected. Supply two.
    let src = "\
[meta]
name = pm
[room]
size = 6 4 4
[part blk]
size = 2,1,1
mask =
##
##
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let err = load_string(src, "<pm>").unwrap_err();
    assert!(
        err.msg.contains("[part] mask too many rows for d"),
        "expected mask overflow error, got: {}",
        err.msg
    );
}

/// The optional `[material]` section parses into `Level.materials`, binds to a
/// surface, and reaches the `Map` via `build_map` — while a level WITHOUT one
/// stays empty (back-compat). This is the parser end of the "color = material
/// KEY" pipeline.
#[test]
fn material_section_parses_and_binds_to_surface() {
    let src = "\
[meta]
name = mat
[room]
size = 6 4 4
ambient = solid
[material floor_neon]
fg = #20FF80
bg = #04140C
glyph = +
recipe = grain scale=3.0 speck=0.5 tile_eps=0.08
animation = pulse speed=0.5 amplitude=0.2
rank = 12
surface = floor
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<mat>").expect("level with [material] must parse");
    assert_eq!(lv.materials.len(), 1);
    let m = &lv.materials[0];
    assert_eq!(m.name, "floor_neon");
    assert_eq!(m.fg, 0x20FF80);
    assert_eq!(m.bg, 0x04140C);
    assert_eq!(m.glyph, Some(u32::from('+')));
    assert_eq!(m.rank, 12);
    assert!(matches!(m.recipe, Some(MaterialRecipe::Grain { .. })));
    assert_eq!(m.anim.kind, MaterialAnimKind::Pulse);
    assert!((m.anim.speed - 0.5).abs() < 1e-6);
    // The surface binding resolved.
    assert_eq!(lv.surface_materials.floor.as_deref(), Some("floor_neon"));

    // It reaches the Map: 5 slot-ordered defs, floor (slot 3) is the authored one.
    let map = lv.build_map();
    let mats = map.materials();
    assert_eq!(
        mats.len(),
        5,
        "authored map carries 5 slot-ordered materials"
    );
    assert!(mats[0].fallback, "unbound wall_solid → fallback");
    assert!(!mats[3].fallback, "floor slot is authored");
    assert_eq!(mats[3].name, "floor_neon");
    assert_eq!(mats[3].glyph, Some(u32::from('+')));
}

/// COLOUR→MATERIAL SUBSTRATE (reactive-materials spec §7, R1): a `[color_cell_ids]`
/// section parses into `Level.color_cell_ids`, `build_map` stamps each 1-based id
/// into the target geometry cell's `side_data`, and the colour palette is appended
/// to the renderer-facing material set AFTER the five surface slots (id `N` ⇒
/// `materials[5 + (N-1)]`). Two perimeter wall cells get DIFFERENT ids → DIFFERENT
/// side_data, so the renderer dispatches two different colour materials.
#[test]
fn color_cell_ids_parse_and_stamp_side_data() {
    let src = "\
[meta]
name = cm
[room]
size = 6 4 4
ambient = solid
[material red_wall]
fg = #FF4444
bg = #200000
recipe = brick
[material green_wall]
fg = #44FF44
bg = #002000
recipe = grain
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
[color_cell_ids]
0,1,1 = 1
0,2,1 = 2
";
    let lv = load_string(src, "<cm>").expect("level with [color_cell_ids] must parse");
    // Parsed into the level.
    assert_eq!(
        lv.color_cell_ids,
        vec![(0, 1, 1, 1u16), (0, 2, 1, 2u16)],
        "the two cell assignments parse in order"
    );
    assert_eq!(lv.materials.len(), 2, "two colour-materials present");

    let map = lv.build_map();
    // Both target cells are perimeter walls and now carry the colour ids in side_data.
    assert_eq!(
        map.cell_at(0, 1, 1),
        CellKind::WallSolid,
        "target is a wall cell"
    );
    assert_eq!(
        map.cell_at(0, 2, 1),
        CellKind::WallSolid,
        "target is a wall cell"
    );
    assert_eq!(map.side_data_at(0, 1, 1), 1, "cell (0,1,1) ⇒ colour id 1");
    assert_eq!(map.side_data_at(0, 2, 1), 2, "cell (0,2,1) ⇒ colour id 2");
    // An untouched wall cell keeps side_data 0 (no override → surface default).
    assert_eq!(map.side_data_at(0, 3, 1), 0, "an unpainted wall keeps id 0");

    // The renderer-facing material set: 5 surface slots followed by the 2-entry
    // colour palette (id 1 ⇒ slot 5, id 2 ⇒ slot 6).
    let mats = map.materials();
    assert_eq!(mats.len(), 7, "5 surface slots + 2 colour palette entries");
    assert_eq!(
        mats[5].name, "red_wall",
        "colour id 1 ⇒ palette entry 0 (slot 5)"
    );
    assert_eq!(
        mats[6].name, "green_wall",
        "colour id 2 ⇒ palette entry 1 (slot 6)"
    );
    assert_eq!(mats[5].fg, 0xFF4444);
    assert_eq!(mats[6].fg, 0x44FF44);
}

/// Back-compat: a level with NO `[color_cell_ids]` carries no per-cell ids and the
/// renderer-facing material set keeps the historical five-slot shape (the colour
/// palette is appended ONLY when colour ids are present).
#[test]
fn no_color_cell_ids_is_byte_identical() {
    let src = "\
[meta]
name = nocm
[room]
size = 6 4 4
ambient = solid
[material red_wall]
fg = #FF4444
bg = #200000
recipe = brick
surface = wall_solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<nocm>").expect("level must parse");
    assert!(lv.color_cell_ids.is_empty(), "no colour ids");
    let map = lv.build_map();
    // No palette appended: a material-authoring level keeps exactly 5 surface slots.
    assert_eq!(
        map.materials().len(),
        5,
        "no colour ids ⇒ five-slot surface set"
    );
    // No geometry cell carries a colour id.
    assert_eq!(map.side_data_at(0, 1, 1), 0, "wall side_data untouched");
}

/// The glyph-only transparent HOLOGRAM is authorable: `recipe = hologram` bound to
/// a wall surface parses into a `MaterialRecipe::Hologram` carrying its line knobs,
/// and reaches the `Map`. (The renderer forces the void bg + vertical scroll.)
#[test]
fn hologram_material_parses_and_binds() {
    let src = "\
[meta]
name = holo
[room]
size = 6 4 4
ambient = solid
[material holo_panel]
fg = #58F0FF
bg = #020608
recipe = hologram line_freq=8.0 scan_w=0.4
surface = wall_solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<holo>").expect("level with a hologram material must parse");
    let m = &lv.materials[0];
    assert_eq!(m.name, "holo_panel");
    match m.recipe {
        Some(MaterialRecipe::Hologram { line_freq, scan_w }) => {
            assert!(
                (line_freq - 8.0).abs() < 1e-6,
                "line_freq override must parse"
            );
            assert!((scan_w - 0.4).abs() < 1e-6, "scan_w override must parse");
        }
        other => panic!("expected a Hologram recipe, got {other:?}"),
    }
    // It reaches the Map bound to the wall_solid slot (slot 0).
    let map = lv.build_map();
    let mats = map.materials();
    assert!(
        !mats[0].fallback,
        "wall_solid slot is the authored hologram"
    );
    assert!(matches!(
        mats[0].recipe,
        Some(MaterialRecipe::Hologram { .. })
    ));
}

/// A MIRROR is authorable: `[material]` with `reflective = true` + `clarity = N`
/// bound to a wall surface parses, carries the flags, and reaches the `Map` as a
/// reflective `MaterialDef`. A material that omits the flags is NON-reflective
/// (back-compat).
#[test]
fn mirror_material_parses_reflective_and_clarity() {
    let src = "\
[meta]
name = mir
[room]
size = 6 4 4
ambient = solid
[material mirror_glass]
fg = #C8D2FF
bg = #05060A
reflective = true
clarity = 0.4
surface = wall_trace
[material plain_wall]
fg = #808080
bg = #101010
surface = wall_matrix
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<mir>").expect("mirror level must parse");
    let mirror = lv
        .materials
        .iter()
        .find(|m| m.name == "mirror_glass")
        .expect("mirror material present");
    assert!(mirror.reflective, "reflective = true must parse");
    assert!((mirror.clarity - 0.4).abs() < 1e-6, "clarity must parse");
    // The non-mirror material defaults to non-reflective (back-compat).
    let plain = lv
        .materials
        .iter()
        .find(|m| m.name == "plain_wall")
        .unwrap();
    assert!(
        !plain.reflective,
        "a material without `reflective` is not a mirror"
    );
    assert_eq!(plain.clarity, MaterialDef::DEFAULT_CLARITY);

    // The flags reach the Map: wall_trace (slot 1) is the reflective one.
    let map = lv.build_map();
    let mats = map.materials();
    assert!(
        mats[1].reflective,
        "wall_trace slot must be a mirror on the Map"
    );
    assert!((mats[1].clarity - 0.4).abs() < 1e-6);
    assert!(!mats[2].reflective, "wall_matrix slot is not a mirror");
}

/// `[material]` with `detail = on` parses the opt-in world-anchored
/// detail-tier flag onto the authored `LevelMaterial`. A material that omits
/// the key defaults to `false` (back-compat: the historical single-tier
/// shading). The flag is later forwarded to `re_world::MaterialDef::detail`
/// (see `build_map`), which `re_render` gates its detail-LOD sampling on.
#[test]
fn material_parses_the_opt_in_detail_flag() {
    let src = "\
[meta]
name = det
[room]
size = 6 4 4
ambient = solid
[material fine_brick]
fg = #A05030
bg = #201008
recipe = brick
detail = on
surface = wall_trace
[material plain_wall]
fg = #808080
bg = #101010
surface = wall_matrix
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<det>").expect("detail level must parse");
    let fine = lv
        .materials
        .iter()
        .find(|m| m.name == "fine_brick")
        .expect("fine_brick material present");
    assert!(fine.detail, "detail = on must parse");
    let plain = lv
        .materials
        .iter()
        .find(|m| m.name == "plain_wall")
        .unwrap();
    assert!(
        !plain.detail,
        "a material without `detail` defaults to off (back-compat)"
    );
}

/// A level with NO `[material]` section leaves `materials` empty and the Map
/// carries no authored materials (renderer keeps built-ins). Back-compat guard.
#[test]
fn no_material_section_keeps_map_default() {
    // Inline fixture (not a shipped level, which are now color-skinned) so the
    // back-compat intent — absent [material] ⇒ no authored materials — is pinned
    // to a stable source that future level re-skins cannot perturb.
    let src = "\
[meta]
name = no-mat
[room]
size = 6 4 4
ambient = solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<no-mat>").expect("material-free level must parse");
    assert!(lv.materials.is_empty(), "fixture authors no materials");
    assert!(
        lv.build_map().materials().is_empty(),
        "a level without [material] must carry no authored materials"
    );
}

/// The shipped levels split into a COLOR-FORWARD skinned set that authors a
/// `[material]` palette (01/02/04/05/06 — re-skinned per the user-requested
/// color pass, commits efe8c16/503b596/6db631a + the L6 Polysemantic rewrite)
/// and an UNSKINNED set that authors none (03/07 — the additive section stays a
/// true no-op for them). Guards both halves of the new reality.
#[test]
fn shipped_levels_material_skinning_matches_expected_set() {
    // Levels that intentionally carry an authored [material] palette.
    let skinned = [1, 2, 4, 5, 6];
    // Levels that still author no [material] section (back-compat no-op).
    let unskinned = [3, 7];
    for n in skinned {
        let name = format!("{n:02}.lvl");
        let lv = load(level_path(&name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            !lv.build_map().materials().is_empty(),
            "{name} is a skinned level and must carry authored materials"
        );
    }
    for n in unskinned {
        let name = format!("{n:02}.lvl");
        let lv = load(level_path(&name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            lv.build_map().materials().is_empty(),
            "{name} authors no [material] (additive no-op, back-compat)"
        );
    }
}

/// Minimal well-formed level body the lighting fixtures append their
/// `[light]` / `[atmosphere]` sections to. Keeping it in one place means a
/// grammar change to the *required* sections does not have to be edited into
/// each lighting test.
#[cfg(test)]
fn lit_fixture(extra: &str) -> String {
    format!(
        "[meta]
name = lit-fixture
[room]
size = 6 4 4
ambient = solid
{extra}
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
"
    )
}

/// §1.3: repeatable `[light]` sections and the `[atmosphere]` block parse, land
/// on the `Level`, and reach the built `Map`'s `Atmosphere`.
#[test]
fn light_and_atmosphere_sections_reach_map() {
    let src = lit_fixture(
        "[light]
pos = 2.5, 1.5, 1.2
color = #FFB060
radius = 6
intensity = 1.25

[light]
pos = 4 2 1
color = #6080FF
radius = 3.5

[atmosphere]
key_dir = -0.4, -1.0, 0.3
key_color = #C0C8FF
ambient = 0.12
fog_height = 0.0 0.6
postfx = clean",
    );
    let lv = load_string(&src, "<lit>").expect("lighting fixture must parse");
    assert_eq!(lv.lights.len(), 2, "two [light] sections");
    assert_eq!(lv.lights[0].color, 0x00FF_B060);
    assert_eq!(lv.lights[0].radius, 6.0);
    assert_eq!(lv.lights[0].intensity, 1.25);
    // intensity omitted ⇒ the documented default of 1.0, not 0.
    assert_eq!(lv.lights[1].intensity, 1.0);
    assert_eq!(lv.lights[1].pos, glam::Vec3::new(4.0, 2.0, 1.0));

    let atmo = lv.atmosphere();
    assert!(atmo.has_lighting());
    assert_eq!(atmo.lights.len(), 2);
    assert_eq!(atmo.key_color, Some(0x00C0_C8FF));
    assert_eq!(atmo.ambient, Some(0.12));
    assert_eq!(atmo.fog_height.unwrap().falloff, 0.6);
    assert_eq!(atmo.postfx.as_deref(), Some("clean"));

    let map = lv.build_map();
    let ma = map.atmosphere();
    assert_eq!(ma.lights.len(), 2, "lights reach the built Map");
    assert_eq!(ma.postfx.as_deref(), Some("clean"));
}

/// `[atmosphere]` accepts the fog-DISTANCE keys as aliases of the `[palette]`
/// ones (one storage location, both spellings), and a level with neither is
/// still `is_none()`.
#[test]
fn atmosphere_accepts_fog_aliases_and_stays_additive() {
    let src = lit_fixture(
        "[atmosphere]
fog_color = #0A0E1A
fog_near = 2.0
fog_far = 16.0",
    );
    let lv = load_string(&src, "<atmo-fog>").expect("must parse");
    assert!(lv.palette.has_fog && lv.palette.has_fog_dist);
    let a = lv.atmosphere();
    assert_eq!(a.fog_color, Some(0x000A_0E1A));
    assert_eq!(a.fog_far, Some(16.0));
    assert!(!a.has_lighting(), "fog aliases are not lighting");

    // No sections at all ⇒ nothing authored (shipped-level back-compat).
    let bare = load_string(&lit_fixture(""), "<bare>").expect("must parse");
    assert!(bare.atmosphere().is_none());
    assert!(bare.lights.is_empty());
}

/// Out-of-range / malformed lighting values are parse ERRORS with the offending
/// line number, never silent defaults.
#[test]
fn bad_lighting_values_are_errors_with_line_numbers() {
    let cases: [(&str, &str); 5] = [
        (
            "[light]
pos = 1 1 1
radius = 0",
            "radius",
        ),
        (
            "[light]
pos = 1 1 1
intensity = -2",
            "intensity",
        ),
        (
            "[light]
pos = 1 1",
            "pos",
        ),
        (
            "[atmosphere]
ambient = 1.7",
            "ambient",
        ),
        (
            "[atmosphere]
postfx = sparkle",
            "postfx",
        ),
    ];
    for (extra, needle) in cases {
        let err = load_string(&lit_fixture(extra), "<bad>")
            .expect_err(&format!("`{extra}` must be rejected"));
        let msg = format!("{err}");
        assert!(
            msg.contains(needle),
            "error for `{extra}` should mention `{needle}`, got: {msg}"
        );
        assert!(
            msg.contains("<bad>") || msg.contains("line"),
            "error must locate itself: {msg}"
        );
    }
}

/// A negative height-fog falloff is refused (it would AMPLIFY fog with height).
#[test]
fn negative_fog_height_falloff_is_rejected() {
    let err = load_string(
        &lit_fixture(
            "[atmosphere]
fog_height = 0.0 -0.5",
        ),
        "<bad2>",
    )
    .expect_err("negative falloff must be refused");
    assert!(format!("{err}").contains("fog_height"));
}

/// A level's `[palette]` fog + `[effects] matrix` reach the `Map` via
/// `build_map` (the seam the renderer reads back to tint distant cells).
#[test]
fn palette_fog_reaches_map_atmosphere() {
    // Inline fixture authoring fog_color/near/far AND matrix, so the seam being
    // tested (present palette/effects ⇒ atmosphere reaches the built Map) is
    // pinned to a stable source rather than a shipped level whose palette may be
    // re-skinned. The shipped levels are still exercised by `all_shipped_levels_load`.
    let src = "\
[meta]
name = fog-fixture
[room]
size = 6 4 4
ambient = solid
[effects]
matrix = 0.1
[palette]
fog_color = #0A0E1A
fog_near = 2.0
fog_far = 16.0
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<fog>").expect("fog fixture must parse");
    assert!(lv.palette.has_fog && lv.palette.has_fog_dist);

    // The level-level accessor mirrors the parsed palette/effects.
    let atmo = lv.atmosphere();
    assert!(!atmo.is_none());
    assert_eq!(atmo.fog_color, Some(0x000A_0E1A));
    assert_eq!(atmo.fog_near, Some(2.0));
    assert_eq!(atmo.fog_far, Some(16.0));
    assert_eq!(atmo.matrix, Some(0.1));

    // And it is forwarded onto the built Map.
    let map = lv.build_map();
    let ma = map.atmosphere();
    assert_eq!(ma.fog_color, Some(0x000A_0E1A));
    assert_eq!(ma.fog_near, Some(2.0));
    assert_eq!(ma.fog_far, Some(16.0));
    assert_eq!(ma.matrix, Some(0.1));
}

/// The atmosphere fixture: 08-material-demo authors a vivid magenta fog (tight
/// band) + a matrix knob, all of which must reach the built Map.
#[test]
fn fixture_08_fog_reaches_map() {
    let lv = load(level_path("08-material-demo.lvl")).expect("08 must parse");
    let built = lv.build_map();
    let atmo = built.atmosphere();
    assert_eq!(atmo.fog_color, Some(0x00FF_00FF), "magenta fog colour");
    assert_eq!(atmo.fog_near, Some(1.0));
    assert_eq!(atmo.fog_far, Some(4.0));
    assert_eq!(atmo.matrix, Some(0.4));
}

/// A level with no `[palette]` fog and no `[effects] matrix` yields the default
/// (none) atmosphere → the Map carries the engine defaults (no behaviour change).
#[test]
fn no_palette_keeps_map_atmosphere_default() {
    // Inline fixture with NO [palette] fog and NO [effects] matrix, so the
    // "absent atmosphere ⇒ map default" guarantee is pinned to a stable source
    // (the shipped levels now all author fog as part of the color-forward pass).
    let src = "\
[meta]
name = no-atmo
[room]
size = 6 4 4
ambient = solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<no-atmo>").expect("atmosphere-free level must parse");
    assert!(!lv.palette.has_fog, "fixture authors no fog");
    assert!(lv.atmosphere().is_none(), "fixture yields no atmosphere");
    assert!(
        lv.build_map().atmosphere().is_none(),
        "a level without fog/matrix must leave the map atmosphere default"
    );
}

/// The optional `[node] anchor = N` parses into `Node.anchor`; absent it
/// defaults to 0 (additive + back-compatible). A non-numeric value falls back
/// to 0 rather than erroring.
#[test]
fn node_anchor_parses_and_defaults() {
    let src = "\
[meta]
name = anchortest
[room]
size = 6 4 4
[node A]
pos = 1,1,1
pool = FIRE BLOOD ROSE
anchor = 2
[node B]
pos = 2,1,1
pool = LEAF MOSS
[node C]
pos = 3,1,1
pool = SEA SKY
anchor = oops
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<anchor>").expect("anchor level parses");
    assert_eq!(lv.nodes[0].anchor, 2, "explicit anchor = 2");
    assert_eq!(lv.nodes[1].anchor, 0, "absent anchor defaults to 0");
    assert_eq!(lv.nodes[2].anchor, 0, "non-numeric anchor falls back to 0");
}

/// The shipped levels parse with the default anchor 0 — no level authors an
/// override, so the field is purely additive.
#[test]
fn shipped_levels_default_anchor_zero() {
    for n in 1..=7 {
        let lv = load(level_path(&format!("{n:02}.lvl"))).expect("shipped level parses");
        for node in &lv.nodes {
            assert_eq!(node.anchor, 0, "L{n} {} keeps anchor 0", node.name);
        }
    }
}

/// R-M8 companion: a correctly-sized mask still parses (no false positive).
#[test]
fn part_mask_exact_rows_ok() {
    let src = "\
[meta]
name = pm
[room]
size = 6 4 4
[part blk]
size = 2,2,1
mask =
##
.#
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<pm>").expect("well-formed 2x2x1 mask must parse");
    let part = &lv.parts[0];
    assert_eq!(part.mask, vec![1, 1, 0, 1]);
}

/// A `[holo]` section authors a transparent HOLOGRAM object: it parses into a
/// `LevelHolo` carrying its shape + region, and `build_map` stamps the shape's
/// cells as `CellKind::Hologram` (the map reports `has_holograms`).
#[test]
fn holo_section_parses_and_stamps_hologram_cells() {
    use re_world::HoloShape;
    let src = "\
[meta]
name = holo
[room]
size = 10 10 5
ambient = solid
[holo cube]
shape = box
region = (4,4,1)-(6,6,3)
style = 2
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 8 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (8,1,2)
[pedestal]
center = 8 1 1
volume = 1 1 1
[door]
pos = 9 5 1
[player]
spawn = 5 8 1
";
    let lv = load_string(src, "<holo>").expect("level with [holo] must parse");
    assert_eq!(lv.holos.len(), 1);
    let h = &lv.holos[0];
    assert_eq!(h.name, "cube");
    assert_eq!(h.shape, HoloShape::Box);
    assert_eq!((h.x0, h.y0, h.z0), (4, 4, 1));
    assert_eq!((h.x1, h.y1, h.z1), (6, 6, 3));
    assert_eq!(h.style, 2);

    // It reaches the Map: a 3×3×3 box of hologram cells, and the map flags it.
    let map = lv.build_map();
    assert!(
        map.has_holograms(),
        "stamped holo map must report has_holograms"
    );
    assert_eq!(
        map.count(CellKind::Hologram),
        27,
        "3×3×3 box = 27 hologram cells"
    );
    assert_eq!(map.cell_at(5, 5, 2), CellKind::Hologram);
    assert_eq!(map.side_data_at(5, 5, 2), 2, "style tag rides in side_data");
}

/// A level with NO `[holo]` section has no hologram objects and stamps no
/// hologram cells (the back-compat guarantee for the shipped levels).
#[test]
fn level_without_holo_has_none() {
    let lv = load(level_path("01.lvl")).expect("01.lvl must parse");
    assert!(lv.holos.is_empty());
    let map = lv.build_map();
    assert!(!map.has_holograms());
    assert_eq!(map.count(CellKind::Hologram), 0);
}

/// The shipped demo levels (13/13b) parse and stamp their hologram objects.
#[test]
fn demo_holo_levels_parse_and_stamp() {
    let cube = load(level_path("13-holocube.lvl")).expect("13-holocube.lvl must parse");
    assert_eq!(cube.holos.len(), 1);
    let m = cube.build_map();
    assert!(m.has_holograms());
    assert!(m.count(CellKind::Hologram) > 0);

    let mesh = load(level_path("13b-holomesh.lvl")).expect("13b-holomesh.lvl must parse");
    assert_eq!(mesh.holos.len(), 2, "pyramid + sphere");
    let mm = mesh.build_map();
    assert!(mm.has_holograms());
    assert!(mm.count(CellKind::Hologram) > 0);
}

// ===========================================================================
// [pipe] spline = ... — Catmull-Rom + supercover 6-connectivity (Task A)
// ===========================================================================

/// Assert every consecutive pair in a path differs by exactly one cell on a
/// single axis (face-adjacent ⇒ 6-connected, no diagonal gaps).
fn assert_6connected(path: &[(i32, i32, i32)]) {
    for w in path.windows(2) {
        let (a, b) = (w[0], w[1]);
        let d = (a.0 - b.0).abs() + (a.1 - b.1).abs() + (a.2 - b.2).abs();
        assert_eq!(d, 1, "non-6-connected step {a:?} -> {b:?} (manhattan {d})");
    }
}

/// Build a minimal one-pipe level whose pipe uses a `spline = ...` value.
fn spline_level(room: &str, spline: &str, switch_at: Option<&str>) -> String {
    let sw = switch_at
        .map(|s| format!("switch_at = {s}\n"))
        .unwrap_or_default();
    format!(
        "[meta]\nname = t\n[room]\nsize = {room}\n\
         [node A]\npos = 2,2,1\nglyph = A\nis_input = yes\n\
         [node B]\npos = 6,6,1\nglyph = B\nis_output = yes\nthreshold = 1\n\
         [pipe A->B]\nspline = {spline}\n{sw}\
         [pedestal]\ncenter = 4 4 1\nvolume = 2 2 2\n\
         [door]\npos = 7 4 1\ntarget_parts = B\n\
         [player]\nspawn = 5 4 1\nfacing = -x\n"
    )
}

/// A spline through 3 control points yields a contiguous 6-connected path that
/// passes through (near) each control point.
#[test]
fn spline_three_points_contiguous_and_passes_near_controls() {
    let lv = load_string(
        &spline_level("12 12 4", "(2,2,1) (7,2,3) (7,9,1)", None),
        "spline3",
    )
    .expect("3-point spline must parse");
    let path = &lv.pipes[0].path;
    assert!(path.len() > 3, "spline should produce a dense path");
    assert_6connected(path);
    // The curve passes through its endpoints exactly and near the middle ctrl.
    assert_eq!(*path.first().unwrap(), (2, 2, 1));
    assert_eq!(*path.last().unwrap(), (7, 9, 1));
    let near_mid = path
        .iter()
        .any(|&(x, y, z)| (x - 7).abs() <= 1 && (y - 2).abs() <= 1 && (z - 3).abs() <= 1);
    assert!(near_mid, "path must pass near the (7,2,3) control point");
}

/// A 2-control-point spline degenerates to a near-straight routed line: every
/// cell stays within ~1 of the straight chord between the endpoints.
#[test]
fn spline_two_points_is_near_straight() {
    let lv = load_string(&spline_level("12 12 4", "(1,1,1) (9,5,1)", None), "spline2")
        .expect("2-point spline must parse");
    let path = &lv.pipes[0].path;
    assert_6connected(path);
    assert_eq!(*path.first().unwrap(), (1, 1, 1));
    assert_eq!(*path.last().unwrap(), (9, 5, 1));
    // Distance from each cell to the infinite line through the endpoints stays
    // within one cell — a straight routed run, no bulging.
    let (ax, ay) = (1.0f64, 1.0f64);
    let (bx, by) = (9.0f64, 5.0f64);
    let len = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
    for &(x, y, _z) in path {
        let num = ((bx - ax) * (ay - y as f64) - (ax - x as f64) * (by - ay)).abs();
        assert!(
            num / len <= 1.0 + 1e-6,
            "cell {:?} bows off the chord",
            (x, y)
        );
    }
}

/// An out-of-bounds control point errors clearly.
#[test]
fn spline_out_of_bounds_control_errors() {
    let err = load_string(
        &spline_level("8 8 4", "(2,2,1) (20,2,1) (6,6,1)", None),
        "splineoob",
    )
    .expect_err("out-of-bounds control point must error");
    assert!(
        err.msg.contains("out of room bounds"),
        "unexpected error: {}",
        err.msg
    );
}

/// `switch_at` on a generated spline path resolves to an index on that path.
#[test]
fn spline_switch_at_resolves_on_path() {
    let lv = load_string(
        &spline_level("12 12 4", "(2,2,1) (7,2,1) (7,9,1)", Some("(7,2,1)")),
        "splinesw",
    )
    .expect("spline with switch_at must parse");
    let p = &lv.pipes[0];
    let idx = p
        .switch_cell
        .expect("switch_cell must resolve on the spline path");
    assert_eq!(p.path[idx], (7, 2, 1));
}

/// A `switch_at` cell NOT on the generated spline path is a hard error
/// (the existing finalize off-path check covers spline paths too).
#[test]
fn spline_switch_at_off_path_errors() {
    let err = load_string(
        &spline_level("12 12 4", "(2,2,1) (7,2,1) (7,9,1)", Some("(0,0,3)")),
        "splineswoff",
    )
    .expect_err("off-path switch must error");
    assert!(
        err.msg.contains("not on expanded path"),
        "unexpected error: {}",
        err.msg
    );
}

/// Back-compat: an existing `path =` level is byte-unchanged in behaviour — its
/// expanded path matches the legacy x-then-y-then-z L-expansion exactly.
#[test]
fn waypoint_path_still_l_expands() {
    let lv = load(level_path("03.lvl")).expect("03.lvl must parse");
    // First pipe R->h1: path = (2,3,2) (8,3,2) (8,5,2) — pure L-expansion.
    let expected = expand_path(&[(2, 3, 2), (8, 3, 2), (8, 5, 2)]);
    assert_eq!(lv.pipes[0].path, expected);
    assert_6connected(&lv.pipes[0].path);
}

// ===========================================================================
// L12 finale parse: [door] target_off (inverted) + [room] cycle_keep
// ===========================================================================

/// A two-output level whose `[door]` declares an inverted target. `target_off`
/// resolves to the named node index; `target_parts` (EXIT) stays positive.
#[test]
fn door_target_off_parses_to_index() {
    let src = "[meta]\nname = t\n[room]\nsize = 10 10 4\n\
        [node G]\npos = 2,2,1\nglyph = G\nis_input = yes\n\
        [node REC]\npos = 5,2,1\nglyph = R\nis_output = yes\nthreshold = 1\n\
        [node EXIT]\npos = 5,6,1\nglyph = E\nis_output = yes\nthreshold = 1\n\
        [pipe G->REC]\npath = (2,2,1) (5,2,1)\nswitch_at = (3,2,1)\n\
        [pipe G->EXIT]\npath = (2,2,1) (2,6,1) (5,6,1)\nswitch_at = (2,4,1)\n\
        [inputs]\nG = 1\n\
        [pedestal]\ncenter = 4 4 1\nvolume = 2 2 2\n\
        [door]\npos = 7 4 1\ntarget_parts = EXIT\ntarget_off = REC\n\
        [player]\nspawn = 5 4 1\nfacing = -x\n";
    let lv = load_string(src, "<exit0>").expect("inverted-target level must parse");
    let rec = lv.node_index("REC").unwrap();
    let exit = lv.node_index("EXIT").unwrap();
    assert_eq!(
        lv.door.target_parts,
        vec![exit],
        "EXIT is the positive target"
    );
    assert_eq!(lv.door.target_off, vec![rec], "REC is the inverted target");
    // cycle_keep defaults off.
    assert!(!lv.cycle_keep);
}

/// A `target_off` naming a nonexistent node is a hard load error (validated like
/// target_parts), not a silent no-op.
#[test]
fn door_target_off_unknown_node_errors() {
    let src = "[meta]\nname = t\n[room]\nsize = 10 10 4\n\
        [node G]\npos = 2,2,1\nglyph = G\nis_input = yes\n\
        [node EXIT]\npos = 5,6,1\nglyph = E\nis_output = yes\nthreshold = 1\n\
        [pipe G->EXIT]\npath = (2,2,1) (2,6,1) (5,6,1)\nswitch_at = (2,4,1)\n\
        [inputs]\nG = 1\n\
        [pedestal]\ncenter = 4 4 1\nvolume = 2 2 2\n\
        [door]\npos = 7 4 1\ntarget_parts = EXIT\ntarget_off = NOPE\n\
        [player]\nspawn = 5 4 1\nfacing = -x\n";
    let err = load_string(src, "<badoff>").expect_err("unknown target_off must error");
    assert!(
        err.msg.contains("target_off"),
        "unexpected error: {}",
        err.msg
    );
}

/// A level with NO `target_off` leaves the field empty (byte-identical default).
#[test]
fn no_target_off_is_empty() {
    let lv = load(level_path("07.lvl")).expect("07.lvl must parse");
    assert!(
        lv.door.target_off.is_empty(),
        "default level has no inverted targets"
    );
    assert!(!lv.cycle_keep, "default level does not keep cycles");
}

/// `[room] cycle_keep = true` parses to the flag; truthy variants all work and a
/// missing/other value stays false.
#[test]
fn room_cycle_keep_parses() {
    let mk = |val: &str| {
        format!(
            "[meta]\nname = t\n[room]\nsize = 8 8 4\ncycle_keep = {val}\n\
             [node A]\npos = 2,2,1\nglyph = A\nis_input = yes\n\
             [node B]\npos = 6,6,1\nglyph = B\nis_output = yes\nthreshold = 1\n\
             [pipe A->B]\npath = (2,2,1) (6,2,1) (6,6,1)\nswitch_at = (4,2,1)\n\
             [inputs]\nA = 1\n\
             [pedestal]\ncenter = 4 4 1\nvolume = 2 2 2\n\
             [door]\npos = 7 4 1\ntarget_parts = B\n\
             [player]\nspawn = 5 4 1\nfacing = -x\n"
        )
    };
    for val in ["true", "1", "on", "yes", "TRUE", "On"] {
        let lv = load_string(&mk(val), "<ck>").unwrap_or_else(|e| panic!("cycle_keep={val}: {e}"));
        assert!(lv.cycle_keep, "cycle_keep = {val} must be true");
    }
    for val in ["false", "0", "off", "no", ""] {
        let lv = load_string(&mk(val), "<ck>").unwrap_or_else(|e| panic!("cycle_keep={val}: {e}"));
        assert!(!lv.cycle_keep, "cycle_keep = {val} must be false");
    }
}

// ===========================================================================
// [vox] = .vox model stamp (Task A)
// ===========================================================================

/// Build a minimal `.vox` byte blob (VOX 150: MAIN -> SIZE + XYZI) for the given
/// model dimensions and voxel list, so the `[vox]` tests need no asset on disk
/// and never reimplement `.vox` PARSING (only the trivial writer side). Each
/// voxel is `(x, y, z, color_index)`. `dot_vox::load_bytes` parses the result.
fn make_vox_bytes(size: (u32, u32, u32), voxels: &[(u8, u8, u8, u8)]) -> Vec<u8> {
    fn chunk(id: &[u8; 4], content: &[u8], children: &[u8]) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend_from_slice(id);
        c.extend_from_slice(&(content.len() as u32).to_le_bytes());
        c.extend_from_slice(&(children.len() as u32).to_le_bytes());
        c.extend_from_slice(content);
        c.extend_from_slice(children);
        c
    }
    let mut size_content = Vec::new();
    size_content.extend_from_slice(&size.0.to_le_bytes());
    size_content.extend_from_slice(&size.1.to_le_bytes());
    size_content.extend_from_slice(&size.2.to_le_bytes());
    let size_chunk = chunk(b"SIZE", &size_content, &[]);
    let mut xyzi_content = Vec::new();
    xyzi_content.extend_from_slice(&(voxels.len() as u32).to_le_bytes());
    for &(x, y, z, i) in voxels {
        xyzi_content.extend_from_slice(&[x, y, z, i]);
    }
    let xyzi_chunk = chunk(b"XYZI", &xyzi_content, &[]);
    let mut main_children = Vec::new();
    main_children.extend_from_slice(&size_chunk);
    main_children.extend_from_slice(&xyzi_chunk);
    let main_chunk = chunk(b"MAIN", &[], &main_children);
    let mut out = Vec::new();
    out.extend_from_slice(b"VOX ");
    out.extend_from_slice(&150u32.to_le_bytes());
    out.extend_from_slice(&main_chunk);
    out
}

/// Write a `.vox` blob to a unique temp path and return it (each test uses a
/// distinct name to stay parallel-safe).
fn write_temp_vox(tag: &str, size: (u32, u32, u32), voxels: &[(u8, u8, u8, u8)]) -> PathBuf {
    let bytes = make_vox_bytes(size, voxels);
    let path = std::env::temp_dir().join(format!("re_level_voxtest_{tag}.vox"));
    std::fs::write(&path, &bytes).expect("write temp .vox");
    path
}

/// The `[vox]` section parses (file/offset/material/scale) and `build_map` stamps
/// each SOLID voxel as the chosen surface, offset into the room.
#[test]
fn vox_section_parses_and_stamps_solids() {
    let voxels: Vec<(u8, u8, u8, u8)> = (0..2)
        .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| (x, y, z, 1u8))))
        .collect();
    let vox = write_temp_vox("cube2", (2, 2, 2), &voxels);
    let vox_str = vox.to_string_lossy().replace('\\', "/");

    let src = format!(
        "[meta]\nname = voxlvl\n[room]\nsize = 10 10 5\nambient = solid\n\
         [vox building]\nfile = {vox_str}\noffset = 3,3,1\nmaterial = trace\nscale = 1\n\
         [node a]\npos = 1 1 2\nis_input = yes\n\
         [node b]\npos = 8 1 2\nis_output = yes\n\
         [pipe a->b]\npath = (1,1,2) (8,1,2)\n\
         [pedestal]\ncenter = 8 1 1\nvolume = 1 1 1\n\
         [door]\npos = 9 5 1\n\
         [player]\nspawn = 5 8 1\n"
    );
    let lv = load_string(&src, "<vox>").expect("level with [vox] must parse");
    assert_eq!(lv.voxes.len(), 1);
    let v = &lv.voxes[0];
    assert_eq!(v.name, "building");
    assert_eq!((v.off_x, v.off_y, v.off_z), (3, 3, 1));
    assert_eq!(v.material, VoxMaterial::Trace);
    assert_eq!(v.scale, 1);

    let map = lv.build_map();
    assert_eq!(
        map.cell_at(3, 3, 1),
        CellKind::WallTrace,
        "voxel (0,0,0) -> (3,3,1)"
    );
    assert_eq!(
        map.cell_at(4, 4, 2),
        CellKind::WallTrace,
        "voxel (1,1,1) -> (4,4,2)"
    );
    assert_eq!(
        map.count(CellKind::WallTrace),
        8,
        "2x2x2 stamp = 8 trace cells"
    );

    let _ = std::fs::remove_file(&vox);
}

/// `scale = 2` keeps every other voxel on each axis and divides the coordinate,
/// so a 4x4x4 source stamps as a 2x2x2 footprint.
#[test]
fn vox_scale_downsamples() {
    let voxels: Vec<(u8, u8, u8, u8)> = (0..4)
        .flat_map(|x| (0..4).flat_map(move |y| (0..4).map(move |z| (x, y, z, 1u8))))
        .collect();
    let vox = write_temp_vox("cube4", (4, 4, 4), &voxels);
    let vox_str = vox.to_string_lossy().replace('\\', "/");
    let src = format!(
        "[meta]\nname = voxscale\n[room]\nsize = 10 10 5\nambient = solid\n\
         [vox b]\nfile = {vox_str}\noffset = 2,2,1\nscale = 2\n\
         [node a]\npos = 1 1 2\nis_input = yes\n\
         [node b]\npos = 8 1 2\nis_output = yes\n\
         [pipe a->b]\npath = (1,1,2) (8,1,2)\n\
         [pedestal]\ncenter = 8 1 1\nvolume = 1 1 1\n\
         [door]\npos = 9 5 1\n\
         [player]\nspawn = 5 8 1\n"
    );
    let lv = load_string(&src, "<voxscale>").expect("scaled [vox] parses");
    let map = lv.build_map();
    // 4x4x4 source @ scale 2: kept voxels have coords in {0,2} per axis → /2 →
    // {0,1}, a full 2x2x2 footprint at offset (2,2,1) ⇒ x∈{2,3}, y∈{2,3}, z∈{1,2}.
    assert_eq!(
        map.cell_at(2, 2, 1),
        CellKind::WallSolid,
        "voxel (0,0,0)/2 -> (2,2,1)"
    );
    assert_eq!(
        map.cell_at(3, 3, 2),
        CellKind::WallSolid,
        "voxel (2,2,2)/2 -> (3,3,2)"
    );
    // Count: exactly 8 interior solids were added by the stamp. The room's own
    // interior is otherwise empty air (perimeter is WallSolid), so count the stamp
    // footprint precisely by probing a cell just OUTSIDE it.
    assert_eq!(
        map.cell_at(4, 4, 2),
        CellKind::Empty,
        "cell beyond the 2x2x2 footprint is air"
    );
    assert_eq!(
        map.cell_at(3, 3, 3),
        CellKind::Empty,
        "z beyond the footprint is air"
    );
    let _ = std::fs::remove_file(&vox);
}

/// Voxels stamped outside the room are CLIPPED, not an error.
#[test]
fn vox_out_of_bounds_is_clipped() {
    let voxels: Vec<(u8, u8, u8, u8)> = (0..4).map(|x| (x, 0, 0, 1u8)).collect();
    let vox = write_temp_vox("row4", (4, 1, 1), &voxels);
    let vox_str = vox.to_string_lossy().replace('\\', "/");
    let src = format!(
        "[meta]\nname = voxclip\n[room]\nsize = 6 4 4\nambient = solid\n\
         [vox r]\nfile = {vox_str}\noffset = 4,2,2\n\
         [node a]\npos = 1 1 2\nis_input = yes\n\
         [node b]\npos = 4 1 2\nis_output = yes\n\
         [pipe a->b]\npath = (1,1,2) (4,1,2)\n\
         [pedestal]\ncenter = 4 1 1\nvolume = 1 1 1\n\
         [door]\npos = 5 1 1\n\
         [player]\nspawn = 3 1 1\n"
    );
    let lv = load_string(&src, "<voxclip>").expect("clipped [vox] parses");
    let map = lv.build_map();
    assert_eq!(map.cell_at(4, 2, 2), CellKind::WallSolid, "x=4 in bounds");
    assert_eq!(
        map.cell_at(5, 2, 2),
        CellKind::WallSolid,
        "x=5 in bounds (interior)"
    );
    assert!(!map.in_bounds(6, 2, 2));
    assert!(lv.build_map_checked().is_ok());
    let _ = std::fs::remove_file(&vox);
}

/// A `[vox]` with no `file` key is a finalize error (an authoring mistake).
#[test]
fn vox_missing_file_errors() {
    let src = "\
[meta]
name = voxnofile
[room]
size = 6 4 4
[vox b]
offset = 1,1,1
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let err = load_string(src, "<voxnofile>").unwrap_err();
    assert!(err.msg.contains("[vox] missing 'file'"), "got: {}", err.msg);
}

/// A `[vox]` naming a missing file PARSES, but `build_map_checked` reports the
/// load failure while `build_map` stays best-effort (no panic).
#[test]
fn vox_missing_file_on_disk_is_reported_not_panicked() {
    let src = "\
[meta]
name = voxghost
[room]
size = 6 4 4
ambient = solid
[vox ghost]
file = /no/such/path/definitely_missing.vox
offset = 1,1,1
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<voxghost>").expect("a missing-on-disk path still parses");
    let _ = lv.build_map();
    let err = lv
        .build_map_checked()
        .expect_err("missing file must be reported");
    assert!(err.contains("cannot read"), "got: {err}");
}

/// Back-compat: a level with NO `[vox]` section has no voxes and stamps nothing
/// new (the shipped levels are unchanged).
#[test]
fn level_without_vox_has_none() {
    let lv = load(level_path("01.lvl")).expect("01.lvl must parse");
    assert!(lv.voxes.is_empty());
    assert!(lv.build_map_checked().is_ok());
}

/// `[material] preset = NAME` selects a renderer CATALOG material by name. The
/// preset is parsed onto the `LevelMaterial` and forwarded to the renderer encoded
/// in the emitted `MaterialDef.name` (the sentinel-framed `preset = NAME`), while
/// explicit `fg`/`glyph` keys still override on top.
#[test]
fn material_preset_selects_catalog_by_name() {
    let src = "\
[meta]
name = preset-lvl
[room]
size = 6 4 4
ambient = solid
[material myrust]
preset = rust
surface = wall_solid
[material screen]
preset = monitor
glyph = @
surface = wall_trace
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<preset>").expect("preset level must parse");
    // The preset is captured on the LevelMaterial.
    let rust = lv.materials.iter().find(|m| m.name == "myrust").unwrap();
    assert_eq!(rust.preset.as_deref(), Some("rust"));
    let screen = lv.materials.iter().find(|m| m.name == "screen").unwrap();
    assert_eq!(screen.preset.as_deref(), Some("monitor"));
    assert_eq!(
        screen.glyph,
        Some(u32::from('@')),
        "an explicit glyph overrides the preset"
    );

    // It reaches the Map encoded in the slot-ordered MaterialDef names.
    let map = lv.build_map();
    let mats = map.materials();
    assert_eq!(
        mats[0].name,
        format!("{PRESET_NAME_PREFIX}rust"),
        "wall_solid slot carries the rust preset sentinel"
    );
    assert_eq!(mats[1].name, format!("{PRESET_NAME_PREFIX}monitor"));
    assert_eq!(
        mats[1].glyph,
        Some(u32::from('@')),
        "the explicit override survives to the Map"
    );
    // A material WITHOUT a preset still emits its plain section name (back-compat).
    assert!(mats[3].fallback, "an unbound surface stays a fallback");
}

/// Back-compat: a material with no `preset` key leaves `preset = None` and emits
/// its section name unchanged (the historical `recipe=`/`fg=` path).
#[test]
fn material_without_preset_emits_section_name() {
    let src = "\
[meta]
name = no-preset
[room]
size = 6 4 4
ambient = solid
[material plain]
fg = #808080
surface = wall_solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<noprep>").expect("level must parse");
    assert!(lv.materials[0].preset.is_none());
    let map = lv.build_map();
    let mats = map.materials();
    assert_eq!(mats[0].name, "plain", "no preset → the plain section name");
}

// ===========================================================================
// [emitter] — particle-emitter authoring → re_world::EmitterSpec
// ===========================================================================

use re_world::{BoldMode, ColorMode, EmitterBlend, EmitterOrigin, EmitterSpec};

/// Wrap one-or-more `[emitter]` section blocks in an otherwise minimal, valid
/// level (so the parser's required-section checks pass). `body` is the raw
/// emitter section text appended after the room/network scaffold.
fn emitter_level(body: &str) -> String {
    format!(
        "[meta]\nname = emit\n[room]\nsize = 12 8 4\nambient = solid\n\
         [node a]\npos = 1 1 2\nis_input = yes\n\
         [node b]\npos = 8 1 2\nis_output = yes\n\
         [pipe a->b]\npath = (1,1,2) (8,1,2)\n\
         [pedestal]\ncenter = 8 1 1\nvolume = 1 1 1\n\
         [door]\npos = 9 5 1\n\
         [player]\nspawn = 5 6 1\n\
         {body}"
    )
}

/// A single `[emitter]` section parses EVERY documented key into the matching
/// `EmitterSpec` field, exactly per the contract's key→field mapping.
#[test]
fn emitter_parses_every_key() {
    let src = emitter_level(
        "[emitter torch_1]\n\
         origin_world  = 3.5 2.0 1.2\n\
         colors        = 0xFFD27F 0xFF7A1A 0x6E1200\n\
         color_mode    = RampOverAge\n\
         family        = growth_dot\n\
         glyph_set     = . o O @\n\
         bold          = ByAge\n\
         direction     = 0.0 0.0 1.0\n\
         spread_deg    = 30.0\n\
         size          = 1.0\n\
         growth        = 1.5\n\
         rate          = 24.0\n\
         burst         = 12\n\
         speed         = 0.9\n\
         speed_spread  = 0.2\n\
         path_jitter   = 0.15\n\
         seed          = 42\n\
         swirl         = 0.3\n\
         lifetime      = 2.2\n\
         lifetime_spread = 0.3\n\
         accel         = 0.0 0.0 0.35\n\
         max_particles = 256\n\
         blend         = KeepBg\n",
    );
    let lv = load_string(&src, "<emit>").expect("emitter level must parse");
    assert_eq!(lv.emitters.len(), 1);
    let e = &lv.emitters[0];

    assert_eq!(
        e.origin,
        EmitterOrigin::World(glam::Vec3::new(3.5, 2.0, 1.2))
    );
    assert_eq!(e.colors, [0xFFD27F, 0xFF7A1A, 0x6E1200]);
    assert_eq!(e.color_mode, ColorMode::RampOverAge);
    assert_eq!(e.family, "growth_dot");
    assert_eq!(e.glyph_set, Some(vec!['.', 'o', 'O', '@']));
    assert_eq!(e.bold, BoldMode::ByAge);
    assert_eq!(e.direction, glam::Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(e.spread_deg, 30.0);
    assert_eq!(e.size, 1.0);
    assert_eq!(e.growth, 1.5);
    assert_eq!(e.rate, 24.0);
    assert_eq!(e.burst, Some(12));
    assert_eq!(e.speed, 0.9);
    assert_eq!(e.speed_spread, 0.2);
    assert_eq!(e.path_jitter, 0.15);
    assert_eq!(e.seed, 42);
    assert_eq!(e.swirl, 0.3);
    assert_eq!(e.lifetime, 2.2);
    assert!((e.lifetime_spread - 0.3).abs() < 1e-6);
    assert_eq!(e.accel, glam::Vec3::new(0.0, 0.0, 0.35));
    assert_eq!(e.max_particles, 256);
    assert_eq!(e.blend, EmitterBlend::KeepBg);
    // A fully-authored, in-range spec passes validation unchanged.
    assert!(e.is_valid(), "authored in-range spec must be valid: {e:?}");
}

/// `origin_screen` selects a screen-pinned origin (mutually exclusive with
/// `origin_world`).
#[test]
fn emitter_origin_screen() {
    let src = emitter_level("[emitter hud]\norigin_screen = 10.5 4.0\n");
    let lv = load_string(&src, "<emit-screen>").expect("must parse");
    assert_eq!(lv.emitters.len(), 1);
    assert!(
        matches!(lv.emitters[0].origin, EmitterOrigin::Screen { col, row } if col == 10.5 && row == 4.0)
    );
}

/// MULTIPLE `[emitter]` sections each produce one spec, in file order — just
/// like multiple `[material]` sections.
#[test]
fn multiple_emitters_each_produce_a_spec() {
    let src = emitter_level(
        "[emitter ember]\nfamily = growth_dot\nrate = 24.0\n\
         [emitter haze]\nfamily = haze_wave\nrate = 6.0\n\
         [emitter spark]\nfamily = anim_spin\nburst = 40\n",
    );
    let lv = load_string(&src, "<emit-multi>").expect("must parse");
    assert_eq!(
        lv.emitters.len(),
        3,
        "three [emitter] sections → three specs"
    );
    assert_eq!(lv.emitters[0].family, "growth_dot");
    assert_eq!(lv.emitters[1].family, "haze_wave");
    assert_eq!(lv.emitters[2].family, "anim_spin");
    assert_eq!(lv.emitters[2].burst, Some(40));
}

/// A header may be a bare `[emitter]` (no name) — the name is optional.
#[test]
fn emitter_header_name_is_optional() {
    let src = emitter_level("[emitter]\nfamily = haze_dot\n");
    let lv = load_string(&src, "<emit-noname>").expect("bare [emitter] must parse");
    assert_eq!(lv.emitters.len(), 1);
    assert_eq!(lv.emitters[0].family, "haze_dot");
}

/// A level with NO `[emitter]` section exposes an empty `emitters` Vec
/// (additive; the shipped levels are unchanged).
#[test]
fn level_without_emitter_has_empty_vec() {
    let lv = load(level_path("01.lvl")).expect("01.lvl must parse");
    assert!(lv.emitters.is_empty(), "no [emitter] section ⇒ empty Vec");
}

/// Unspecified keys fall through to the `EmitterSpec` default (a gentle haze).
#[test]
fn emitter_unspecified_keys_take_default() {
    let src = emitter_level("[emitter d]\nfamily = haze_bar\n");
    let lv = load_string(&src, "<emit-default>").expect("must parse");
    let e = &lv.emitters[0];
    let def = EmitterSpec::default();
    assert_eq!(e.family, "haze_bar"); // the one override
    assert_eq!(e.colors, def.colors); // everything else is the default
    assert_eq!(e.rate, def.rate);
    assert_eq!(e.max_particles, def.max_particles);
}

/// Out-of-range authored values are CLAMPED by `EmitterSpec::validate` at parse
/// time — a hand-written `.lvl` can never hand the renderer a degenerate spec.
#[test]
fn emitter_out_of_range_values_are_clamped() {
    let src = emitter_level(
        "[emitter bad]\n\
         spread_deg    = 999.0\n\
         rate          = -50.0\n\
         speed         = -3.0\n\
         size          = 0.0\n\
         growth        = -1.0\n\
         lifetime      = 0.0\n\
         max_particles = 0\n",
    );
    let lv = load_string(&src, "<emit-clamp>").expect("must parse");
    let e = &lv.emitters[0];
    assert!(e.is_valid(), "clamped spec must be valid: {e:?}");
    assert_eq!(e.spread_deg, 180.0, "spread_deg > 180 clamps to 180");
    assert_eq!(e.rate, 0.0, "negative rate clamps to 0");
    assert_eq!(e.speed, 0.0, "negative speed clamps to 0");
    assert!(e.size >= f32::EPSILON, "zero size clamps up");
    assert!(e.growth >= f32::EPSILON, "negative growth clamps up");
    assert!(e.lifetime >= 0.01, "zero lifetime clamps up");
    assert_eq!(e.max_particles, 1, "zero max_particles clamps to 1");
}

/// A malformed numeric value errors with the offending line number (line
/// context), and leaves nothing half-applied.
#[test]
fn emitter_malformed_number_errors_with_line() {
    // The bad `rate` sits on line 12 of the assembled source (8 scaffold lines +
    // [emitter] header + 3 emitter keys → `rate` is on line 12).
    let src = emitter_level("[emitter bad]\nfamily = haze_dot\nrate = not_a_number\n");
    let err = load_string(&src, "<emit-bad>").expect_err("malformed rate must error");
    assert!(
        err.msg.contains("rate"),
        "error message should name the key: {}",
        err.msg
    );
    assert!(err.line > 0, "error must carry a 1-based line number");
    // The line the parser reports must be the actual `rate = ...` line.
    let bad_line = src
        .lines()
        .position(|l| l.trim_start().starts_with("rate"))
        .map(|i| i + 1)
        .unwrap();
    assert_eq!(err.line, bad_line, "line context must point at the bad key");
}

/// An unknown key inside `[emitter]` errors (same strictness as the contract's
/// documented key set), with line context.
#[test]
fn emitter_unknown_key_errors() {
    let src = emitter_level("[emitter bad]\nwiggle = 3\n");
    let err = load_string(&src, "<emit-unknown>").expect_err("unknown key must error");
    assert!(
        err.msg.contains("unknown key") && err.msg.contains("wiggle"),
        "error should name the unknown key: {}",
        err.msg
    );
}

/// A malformed colour list (not three valid stops) errors clearly.
#[test]
fn emitter_bad_colors_errors() {
    let src = emitter_level("[emitter bad]\ncolors = 0xFF0000 0x00FF00\n");
    let err = load_string(&src, "<emit-color>").expect_err("two-stop colors must error");
    assert!(
        err.msg.contains("colors"),
        "error should name colors: {}",
        err.msg
    );
}

/// The `family` string passes through verbatim — even a non-canonical name
/// (the renderer falls back to a default at resolve time; the parser must not
/// reject or rewrite it).
#[test]
fn emitter_family_name_passes_through_verbatim() {
    let src = emitter_level("[emitter custom]\nfamily = my_bespoke_family_42\n");
    let lv = load_string(&src, "<emit-fam>").expect("must parse");
    assert_eq!(lv.emitters[0].family, "my_bespoke_family_42");
}

/// Enum keywords parse for every documented variant (color_mode / bold / blend),
/// including the snake_case aliases.
#[test]
fn emitter_enum_keywords_parse() {
    for (kw, want) in [
        ("RandomPick", ColorMode::RandomPick),
        ("Fixed", ColorMode::Fixed),
        ("ramp", ColorMode::RampOverAge),
    ] {
        let src = emitter_level(&format!("[emitter e]\ncolor_mode = {kw}\n"));
        let lv = load_string(&src, "<cm>").expect("must parse");
        assert_eq!(lv.emitters[0].color_mode, want, "color_mode = {kw}");
    }
    for (kw, want) in [
        ("On", BoldMode::On),
        ("ByProximity", BoldMode::ByProximity),
        ("ByAge", BoldMode::ByAge),
    ] {
        let src = emitter_level(&format!("[emitter e]\nbold = {kw}\n"));
        let lv = load_string(&src, "<bm>").expect("must parse");
        assert_eq!(lv.emitters[0].bold, want, "bold = {kw}");
    }
    let src = emitter_level("[emitter e]\nblend = Opaque\n");
    let lv = load_string(&src, "<bl>").expect("must parse");
    assert_eq!(lv.emitters[0].blend, EmitterBlend::Opaque);
}

/// Inline copy of the eight `[emitter]` sections from `21-emitter-showcase.lvl`
/// (the authored tech-demo), wrapped in a minimal valid level so the test does
/// not depend on that file's UNRELATED `[material] surface = wall` line (a
/// material-binder concern outside this task). Proves the full emitter
/// vocabulary the demo exercises — every colour/bold/blend mode, a tight
/// direction cone, size+growth, rate vs. burst, jitter+swirl, an explicit
/// (multibyte) glyph_set, and a screen-pinned origin — parses end-to-end.
#[test]
fn emitter_showcase_vocabulary_parses() {
    let body = "\
[emitter ember_torch]
origin_world = 2.5 2.0 1.0
colors = 0xFFD27F 0xFF7A1A 0x6E1200
color_mode = RampOverAge
family = growth_dot
bold = ByAge
direction = 0.0 0.0 1.0
spread_deg = 25.0
size = 1.0
growth = 1.4
rate = 26.0
speed = 0.9
speed_spread = 0.2
path_jitter = 0.05
lifetime = 2.2
lifetime_spread = 0.3
accel = 0.0 0.0 0.35
max_particles = 256
seed = 101
blend = KeepBg
[emitter growth_bloom]
family = growth_bars
bold = On
size = 1.0
growth = 2.4
[emitter spark_jet]
direction = 1.0 0.0 0.25
spread_deg = 8.0
speed = 3.2
speed_spread = 0.8
bold = ByProximity
blend = Opaque
[emitter drift_haze]
family = haze_dot
path_jitter = 0.4
swirl = 0.6
lifetime = 4.5
[emitter confetti_pick]
color_mode = RandomPick
colors = 0xFF3CA0 0x3CD0FF 0xFFC83C
spread_deg = 75.0
[emitter rune_motes]
color_mode = Fixed
glyph_set = · : * ✦ ✧
bold = On
[emitter spark_burst]
spread_deg = 180.0
burst = 64
blend = Opaque
[emitter hud_spark]
origin_screen = 6.0 3.0
family = haze_wave
direction = 0.0 1.0 0.0
";
    let src = emitter_level(body);
    let lv = load_string(&src, "<showcase>").expect("showcase vocabulary must parse");
    assert_eq!(lv.emitters.len(), 8, "eight emitters");
    for e in &lv.emitters {
        assert!(e.is_valid(), "every emitter clamps valid: {e:?}");
    }
    // Fixed color_mode + an explicit multibyte glyph_set override (rune_motes).
    let rune = lv
        .emitters
        .iter()
        .find(|e| e.color_mode == ColorMode::Fixed)
        .expect("a Fixed-mode emitter");
    let glyphs = rune.glyph_set.as_ref().expect("rune_motes has glyph_set");
    assert!(glyphs.contains(&'✦') && glyphs.contains(&'✧'));
    // A screen-pinned origin (hud_spark).
    assert!(
        lv.emitters
            .iter()
            .any(|e| matches!(e.origin, EmitterOrigin::Screen { .. })),
        "a screen-pinned emitter is present"
    );
    // A one-shot burst (spark_burst).
    let burst = lv.emitters.iter().find(|e| e.burst.is_some()).unwrap();
    assert_eq!(burst.burst, Some(64));
}

// ===========================================================================
// [portal] — non-euclidean warp-portal pair (Stage 3)
// ===========================================================================

/// A minimal level body with a single `[portal]` pair. `room` is the size line,
/// `portal` the body of the `[portal gate]` section.
fn portal_level(room: &str, portal: &str) -> String {
    format!(
        "[meta]\nname = wp\n[room]\nsize = {room}\nambient = solid\n\
         [portal gate]\n{portal}\
         [node a]\npos = 4,13,2\nis_input = yes\n\
         [node b]\npos = 7,13,2\nis_output = yes\nthreshold = 1\n\
         [pipe a->b]\npath = (4,13,2) (7,13,2)\n\
         [pedestal]\ncenter = 7 13 1\nvolume = 1 1 1\n\
         [door]\npos = 7 14 1\ntarget_parts = b\n\
         [player]\nspawn = 6 14 1\nfacing = -y\n"
    )
}

/// `[portal]` parses into a `LevelPortal` carrying both regions, the rotate, and
/// the optional skin; `build_map` registers a reciprocal pair and stamps both
/// faces as NON-SOLID-for-collision Portal cells.
#[test]
fn portal_section_parses_and_builds_pair() {
    let src = portal_level(
        "12 16 10",
        "region_a = (1,0,1)-(10,0,8)\nregion_b = (0,1,1)-(0,10,8)\nrotate = 0\nskin = portal_spiral\n",
    );
    let lv = load_string(&src, "<portal>").expect("level with [portal] must parse");
    assert_eq!(lv.portals.len(), 1);
    let p = &lv.portals[0];
    assert_eq!(p.name, "gate");
    assert_eq!(p.region_a(), ((1, 0, 1), (10, 0, 8)));
    assert_eq!(p.region_b(), ((0, 1, 1), (0, 10, 8)));
    assert_eq!(p.rotate, 0.0);
    assert_eq!(p.skin, "portal_spiral");

    // build_map registers exactly 2 reciprocal links and stamps both faces.
    let map = lv.build_map();
    assert_eq!(map.portals.len(), 2, "one [portal] pair => 2 PortalLinks");
    // delta_inv is the inverse of delta (reciprocity).
    let id = map.portals[0].delta * map.portals[0].delta_inv;
    for c in 0..4 {
        for r in 0..4 {
            let want = if r == c { 1.0 } else { 0.0 };
            assert!(
                (id.col(c)[r] - want).abs() < 1e-4,
                "delta*delta_inv not identity"
            );
        }
    }
    // North face cell is a Portal: solid for the raycaster, NOT collidable.
    assert_eq!(map.cell_at(5, 0, 4), CellKind::Portal);
    assert!(
        map.cell_at(5, 0, 4).is_solid(),
        "portal cell stays solid for the raycaster"
    );
    assert!(
        !map.cell_at(5, 0, 4).is_collidable(),
        "portal cell is walkable for the player"
    );
    assert_eq!(map.portal_for_cell(5, 0, 4), Some(0));
    // West face maps to the second link.
    assert_eq!(map.cell_at(0, 5, 4), CellKind::Portal);
    assert_eq!(map.portal_for_cell(0, 5, 4), Some(1));
}

/// A `[portal]` whose two openings have DIFFERENT areas is rejected with a clear
/// error (a mismatch would tear the seam).
#[test]
fn portal_mismatched_area_is_rejected() {
    // region_a opening is 9 (x) × 7 (z); region_b is 11 (y) × 7 (z) — areas
    // differ, but both are in-bounds flat wall faces (so the area check, not the
    // bounds check, is what rejects it).
    let src = portal_level(
        "12 16 10",
        "region_a = (1,0,1)-(10,0,8)\nregion_b = (0,1,1)-(0,12,8)\n",
    );
    let err = load_string(&src, "<portal-bad>").expect_err("mismatched portal must fail");
    assert!(
        err.to_string().contains("area"),
        "error should mention area mismatch; got: {err}"
    );
}

/// A `[portal]` region that is NOT a flat wall face (a 3-D box, or a flat face in
/// mid-air) is rejected.
#[test]
fn portal_non_face_region_is_rejected() {
    // region_a is a 3-D box (no flat axis).
    let src = portal_level(
        "12 16 10",
        "region_a = (1,1,1)-(3,3,3)\nregion_b = (0,1,1)-(0,3,3)\n",
    );
    let err = load_string(&src, "<portal-box>").expect_err("non-face portal must fail");
    assert!(
        err.to_string().contains("flat wall face"),
        "error should mention flat wall face; got: {err}"
    );

    // A flat face that is NOT on a boundary (mid-air at y=5).
    let src2 = portal_level(
        "12 16 10",
        "region_a = (1,5,1)-(10,5,8)\nregion_b = (0,1,1)-(0,10,8)\n",
    );
    let err2 = load_string(&src2, "<portal-midair>").expect_err("mid-air face must fail");
    assert!(
        err2.to_string().contains("boundary"),
        "error should mention room boundary; got: {err2}"
    );
}

/// The shipped levels/93-warp-portal.lvl loads, stamps exactly 2 portals, and
/// its portal cells are non-solid-for-collision (walkable). A CameraPose walked
/// straight into the north opening teleports (reusing the Stage-2 path).
#[test]
fn level_93_loads_and_is_walkable() {
    use re_world::{CameraPose, PLAYER_RADIUS};
    use std::f32::consts::PI;

    let lv = load(level_path("93-warp-portal.lvl")).expect("93-warp-portal.lvl must parse");
    assert_eq!(lv.portals.len(), 1, "level 93 authors one portal pair");

    let map = lv.build_map();
    assert_eq!(map.portals.len(), 2, "two reciprocal PortalLinks");
    // Some portal cells were stamped, walkable for the player.
    assert_eq!(map.cell_at(5, 0, 4), CellKind::Portal);
    assert!(!map.cell_at(5, 0, 4).is_collidable());

    // Walk straight north into the opening and confirm a teleport fires (yaw
    // changes by ~90° — the non-euclidean corner turn).
    let mut cam = CameraPose {
        pos: glam::Vec3::new(6.5, 14.5, 1.5),
        yaw: PI,
        pitch: 0.0,
    };
    let start_yaw = cam.yaw;
    let mut teleported = false;
    for _ in 0..40 {
        let before = cam.pos;
        let (dx, dy) = cam.dir();
        cam.move_with_collision(dx * 0.5, dy * 0.5, &map, PLAYER_RADIUS);
        cam.apply_portal_crossings(before, &map);
        if (cam.yaw - start_yaw).abs() > 0.1 {
            teleported = true;
            break;
        }
    }
    assert!(
        teleported,
        "walking north into the portal must teleport (yaw changed)"
    );
}

// ---------------------------------------------------------------------------
// [zone_input] floor-zone input (player-position-as-input, L5)
// ---------------------------------------------------------------------------

/// A minimal level with a `[zone_input]` driving an input node. Asserts the zone
/// parses, resolves its node index, normalises a swapped region, and bounds-tests
/// occupancy via `contains`.
#[test]
fn zone_input_parses_and_resolves() {
    let src = "\
[meta]
name = zone-test
[room]
size = 8 8 6
[node present]
pos = 4,4,3
is_input = yes
[node output]
pos = 2,4,3
is_output = yes
threshold = 1
[pipe present->output]
state = ON
path = (4,4,3) (2,4,3)
[inputs]
present = 1
[zone_input warm_square]
node = present
region = (5,5,1)-(3,3,1)
[pedestal]
center = 3 4 1
volume = 2 2 2
[door]
pos = 1 4 1
target_parts = output
[player]
spawn = 6 4 1
facing = -x
";
    let lv = load_string(src, "<zone>").expect("level with [zone_input] must parse");
    assert_eq!(lv.zone_inputs.len(), 1);
    let z = &lv.zone_inputs[0];
    assert_eq!(z.name, "warm_square");
    assert_eq!(z.node, "present");
    assert_eq!(z.node_index, lv.node_index("present").unwrap());
    // Region corners NORMALISED (authored max-then-min swapped above).
    assert_eq!((z.x0, z.y0, z.z0), (3, 3, 1));
    assert_eq!((z.x1, z.y1, z.z1), (5, 5, 1));
    // Occupancy: inclusive box.
    assert!(z.contains(4, 4, 1), "centre is inside");
    assert!(
        z.contains(3, 3, 1) && z.contains(5, 5, 1),
        "corners are inside"
    );
    assert!(!z.contains(2, 4, 1), "x below min is outside");
    assert!(!z.contains(4, 4, 2), "wrong z layer is outside");
}

/// A `[zone_input]` naming a node that does not exist is a hard load error.
#[test]
fn zone_input_unknown_node_fails() {
    let src = "\
[meta]
name = zone-bad
[room]
size = 8 8 6
[node present]
pos = 4,4,3
is_input = yes
[node output]
pos = 2,4,3
is_output = yes
threshold = 1
[pipe present->output]
state = ON
path = (4,4,3) (2,4,3)
[zone_input warm]
node = ghost
region = (3,3,1)-(5,5,1)
[pedestal]
center = 3 4 1
volume = 2 2 2
[door]
pos = 1 4 1
target_parts = output
[player]
spawn = 6 4 1
facing = -x
";
    let err = load_string(src, "<zone>").unwrap_err();
    assert!(err.msg.contains("unknown node"), "got: {}", err.msg);
}

/// A `[zone_input]` naming a NON-input node is a hard load error (a zone can only
/// drive an input — the forward pass would overwrite a hidden/output node).
#[test]
fn zone_input_non_input_node_fails() {
    let src = "\
[meta]
name = zone-bad2
[room]
size = 8 8 6
[node present]
pos = 4,4,3
is_input = yes
[node output]
pos = 2,4,3
is_output = yes
threshold = 1
[pipe present->output]
state = ON
path = (4,4,3) (2,4,3)
[zone_input warm]
node = output
region = (3,3,1)-(5,5,1)
[pedestal]
center = 3 4 1
volume = 2 2 2
[door]
pos = 1 4 1
target_parts = output
[player]
spawn = 6 4 1
facing = -x
";
    let err = load_string(src, "<zone>").unwrap_err();
    assert!(err.msg.contains("not an input node"), "got: {}", err.msg);
}

// ─────────────────────── R2 pipe-cell tagging ───────────────────────────────

/// A tiny level with one pipe through interior air, used to exercise the
/// `render_pipes` opt-in. The pipe path (2,2,1)→(4,2,1) runs through cells that
/// are Empty after geometry stamping, so the stamper can fill them.
const PIPE_LEVEL: &str = "[meta]
name = pipe-tag-test
[room]
size = 7 6 3
[node A]
pos = 2,2,1
is_input = 1
[node B]
pos = 4,2,1
is_output = 1
[pipe A->B]
state = ON
path = (2,2,1) (4,2,1)
[pedestal]
center = 4 2 1
volume = 1 1 1
[door]
pos = 1 2 1
target_parts = B
[player]
spawn = 5 2 1
facing = -x
";

#[test]
fn render_pipes_default_off_leaves_geometry_untagged() {
    let lv = load_string(PIPE_LEVEL, "<pipe>").expect("parses");
    assert!(!lv.render_pipes, "render_pipes defaults to OFF");
    let map = lv.build_map();
    // Without the opt-in, the mid-path cell (3,2,1) is untouched air — no pipe
    // tag, geometry byte-identical to today.
    assert_eq!(
        map.cell_at(3, 2, 1),
        CellKind::Empty,
        "mid-path cell stays Empty"
    );
    assert_eq!(map.side_data_at(3, 2, 1), 0, "no pipe tag when opt-out");
}

#[test]
fn render_pipes_on_stamps_tagged_walltrace_traces() {
    let mut src = PIPE_LEVEL.replace("[room]\n", "[room]\nrender_pipes = true\n");
    // Sanity: the replacement actually inserted the flag.
    assert!(src.contains("render_pipes = true"));
    let lv = load_string(&src, "<pipe>").expect("parses");
    assert!(lv.render_pipes, "render_pipes opt-in parsed");
    let map = lv.build_map();
    // The mid-path air cell is now a tagged WallTrace pipe cell.
    assert_eq!(
        map.cell_at(3, 2, 1),
        CellKind::WallTrace,
        "pipe cell is WallTrace"
    );
    let sd = map.side_data_at(3, 2, 1);
    assert_eq!(
        re_core::decode_pipe_tag(sd),
        Some(0),
        "mid-path cell carries pipe-0 state-bus tag"
    );
    // Endpoint cells are occupied by the node Objects — the pipe must NOT clobber
    // them (they stay Object cells, untouched by the stamper).
    assert_eq!(
        map.cell_at(2, 2, 1),
        CellKind::Object,
        "node A cell preserved"
    );
    assert_eq!(
        map.cell_at(4, 2, 1),
        CellKind::Object,
        "node B cell preserved"
    );
    // unused now that we built from a mutated copy.
    let _ = &mut src;
}

// ---------------------------------------------------------------------------
// [sequence] sequence-lock parsing (L11)
// ---------------------------------------------------------------------------

/// A two-output base with an opt-in `[sequence]` of three ordered steps over the
/// {o1,o2} window. Each step lists the ON nodes; the parser fills in the OFF
/// constraints for the rest of the observe window.
const SEQ_BASE: &str = "\
[meta]
name = seq
[room]
size = 8 4 4
[node a]
pos = 1 1 1
is_input = yes
[node o1]
pos = 5 1 1
is_output = yes
threshold = 1
[node o2]
pos = 5 2 1
is_output = yes
threshold = 1
[pipe a->o1]
path = (1,1,1) (5,1,1)
switch_at = 3,1,1
[pipe a->o2]
path = (1,1,1) (1,2,1) (5,2,1)
switch_at = 3,2,1
[inputs]
a = 1
[pedestal]
center = 5 1 1
volume = 1 1 1
[door]
pos = 6 1 1
target_parts = o2
[player]
spawn = 2 1 1
";

#[test]
fn sequence_section_parses_into_ordered_patterns() {
    let src =
        format!("{SEQ_BASE}[sequence]\nobserve = o1 o2\nstep = o1\nstep = o2\nstep = o1 o2\n");
    let lv = load_string(&src, "<seq>").expect("level with [sequence] must parse");
    let o1 = lv.node_index("o1").unwrap();
    let o2 = lv.node_index("o2").unwrap();
    assert_eq!(lv.sequence.patterns.len(), 3, "three ordered steps");
    // step 1: o1 ON, o2 OFF.
    assert_eq!(lv.sequence.patterns[0].bits, vec![(o1, true), (o2, false)]);
    // step 2: o1 OFF, o2 ON.
    assert_eq!(lv.sequence.patterns[1].bits, vec![(o1, false), (o2, true)]);
    // step 3: both ON.
    assert_eq!(lv.sequence.patterns[2].bits, vec![(o1, true), (o2, true)]);
}

#[test]
fn no_sequence_section_is_inactive() {
    let lv = load_string(SEQ_BASE, "<noseq>").expect("level without [sequence] parses");
    assert!(
        lv.sequence.patterns.is_empty(),
        "no [sequence] ⇒ inactive (single-target win), unchanged"
    );
}

#[test]
fn sequence_alias_lock_section_works() {
    // `[lock]` is an accepted alias for `[sequence]`.
    let src = format!("{SEQ_BASE}[lock]\nobserve = o1\nstep = o1\n");
    let lv = load_string(&src, "<lock>").expect("[lock] alias parses");
    assert_eq!(lv.sequence.patterns.len(), 1);
}

#[test]
fn sequence_unknown_observe_node_errors() {
    let src = format!("{SEQ_BASE}[sequence]\nobserve = o1 ghost\nstep = o1\n");
    assert!(
        load_string(&src, "<seq>").is_err(),
        "unknown observe node must error"
    );
}

#[test]
fn sequence_step_node_outside_window_errors() {
    let src = format!("{SEQ_BASE}[sequence]\nobserve = o1\nstep = o2\n");
    assert!(
        load_string(&src, "<seq>").is_err(),
        "a step naming a node outside the observe window must error"
    );
}

#[test]
fn sequence_missing_observe_or_steps_errors() {
    let no_observe = format!("{SEQ_BASE}[sequence]\nstep = o1\n");
    assert!(
        load_string(&no_observe, "<seq>").is_err(),
        "missing observe errors"
    );
    let no_steps = format!("{SEQ_BASE}[sequence]\nobserve = o1 o2\n");
    assert!(load_string(&no_steps, "<seq>").is_err(), "no steps errors");
}

#[test]
fn sequence_duplicate_patterns_error() {
    // Two identical output states cannot be distinct ordered targets.
    let src = format!("{SEQ_BASE}[sequence]\nobserve = o1 o2\nstep = o1\nstep = o1\n");
    assert!(
        load_string(&src, "<seq>").is_err(),
        "identical ordered targets must be rejected"
    );
}

// `l11_level_file_declares_three_distinct_patterns` moved to the game repo's
// `campaign_level_parse.rs` — it asserts a property OF THE CAMPAIGN
// (levels/L11-inner-circuit.lvl declares three distinct ordered steps), not of
// the `[sequence]` parser in general (that's covered by the inline-fixture
// tests above): a campaign-content assertion has no business in the
// engine-agnostic parser's own test suite.

// ---------------------------------------------------------------------------
// Scalable Material Overlay — [zone]/[line]/[inset] parsing (P0 seam)
// ---------------------------------------------------------------------------

/// A minimal, valid base level WITHOUT any overlay sections — used to assert the
/// empty-overlay default and to append overlay sections onto.
const OVERLAY_BASE: &str = "\
[meta]
name = ov
[room]
size = 8 8 4
[node a]
pos = 1 1 1
is_input = yes
[node b]
pos = 5 1 1
is_output = yes
[pipe a->b]
path = (1,1,1) (5,1,1)
[pedestal]
center = 5 1 1
volume = 1 1 1
[door]
pos = 6 1 1
[player]
spawn = 3 1 1
";

#[test]
fn no_overlay_sections_parse_to_empty_overlay() {
    let lv = load_string(OVERLAY_BASE, "<ov-empty>").expect("base level parses");
    assert!(
        lv.overlay.is_empty(),
        "a level with no [zone]/[line]/[inset] must carry an empty Overlay"
    );
    assert_eq!(
        lv.overlay,
        Overlay::default(),
        "empty == default (byte-identical rung)"
    );
}

#[test]
fn zone_section_parses_into_zone_primitive() {
    let src = format!(
        "{OVERLAY_BASE}\
[zone hot]
region = (1,1,0)-(3,4,2)
material = metal
scale = crisp
"
    );
    let lv = load_string(&src, "<ov-zone>").expect("level with [zone] parses");
    assert_eq!(lv.overlay.zones.len(), 1, "one zone");
    assert!(lv.overlay.lines.is_empty());
    assert!(lv.overlay.insets.is_empty());
    let z = &lv.overlay.zones[0];
    assert_eq!(z.name, "hot");
    assert_eq!(
        z.region,
        Region::Box {
            min: (1, 1, 0),
            max: (3, 4, 2)
        }
    );
    assert_eq!(z.material, MaterialId("metal".to_string()));
    assert_eq!(z.scale, ScaleRule::Crisp);
}

#[test]
fn zone_self_similar_scale_maps_fractal_params() {
    let src = format!(
        "{OVERLAY_BASE}\
[zone deep]
region = (0,0,0)-(1,1,0)
material = void
scale = self_similar arms:5 twist:0.9 warp:0.3
insets = inner edge
"
    );
    let lv = load_string(&src, "<ov-ss>").expect("self_similar zone parses");
    let z = &lv.overlay.zones[0];
    match &z.scale {
        ScaleRule::SelfSimilar(p) => {
            assert_eq!(p.arms, 5);
            assert!((p.twist - 0.9).abs() < 1e-6);
            assert!((p.octave_warp - 0.3).abs() < 1e-6);
        }
        other => panic!("expected SelfSimilar, got {other:?}"),
    }
    // Inset references carried by name (resolution deferred to a later phase).
    assert_eq!(
        z.insets,
        vec![InsetId("inner".to_string()), InsetId("edge".to_string())]
    );
}

// ---------------------------------------------------------------------------
// Scalable Material Overlay — P1: zones RENDER by id (DP1-A/DP3-A region fill)
// ---------------------------------------------------------------------------

/// A base level WITH two authored materials (`floor_base` bound to the floor,
/// `floor_tile` free) and an `[color_cell_ids]` hook point — the substrate the P1
/// zone region-fill reuses. Room is `8 8 4`, so floor cells are the interior
/// `(1..6, 1..6, z=0)`. Append `[zone]`/`[color_cell_ids]` sections onto this.
const ZONE_FILL_BASE: &str = "\
[meta]
name = zf
[room]
size = 8 8 4
ambient = solid
[material floor_base]
fg = #6FA6CE
bg = #08111E
recipe = grain
surface = floor
[material floor_tile]
fg = #E8B14A
bg = #1E1405
recipe = brick
[node a]
pos = 1 1 1
is_input = yes
[node b]
pos = 5 1 1
is_output = yes
[pipe a->b]
path = (1,1,1) (5,1,1)
[pedestal]
center = 5 1 1
volume = 1 1 1
[door]
pos = 6 1 1
[player]
spawn = 3 1 1
";

/// (a) A `[zone]` stamps its resolved palette id into the `side_data` of SOLID
/// cells INSIDE its region and leaves cells outside untouched. (b) The zone's
/// material name resolves to the correct 1-based palette slot.
#[test]
fn zone_stamps_resolved_palette_id_into_region_only() {
    // `floor_tile` is the SECOND authored material (index 1) ⇒ palette id 2.
    let src = format!(
        "{ZONE_FILL_BASE}\
[zone tile]
region = (2,2,0)-(4,4,0)
material = floor_tile
"
    );
    let lv = load_string(&src, "<zf-one>").expect("zone level parses");
    assert_eq!(lv.overlay.zones.len(), 1);

    let map = lv.build_map();

    // (b) The renderer-facing palette resolves: id 1 ⇒ floor_base, id 2 ⇒ floor_tile.
    let mats = map.materials();
    assert_eq!(
        mats.len(),
        7,
        "5 surface slots + 2 palette entries (zone needs the palette)"
    );
    assert_eq!(mats[5].name, "floor_base", "palette id 1 ⇒ slot 5");
    assert_eq!(mats[6].name, "floor_tile", "palette id 2 ⇒ slot 6");

    // (a) Floor cells INSIDE the region carry the zone's id (2); they ARE solid floor.
    assert_eq!(
        map.cell_at(3, 3, 0),
        CellKind::Floor,
        "zone target is a floor cell"
    );
    assert_eq!(map.side_data_at(2, 2, 0), 2, "corner of region ⇒ id 2");
    assert_eq!(map.side_data_at(3, 3, 0), 2, "centre of region ⇒ id 2");
    assert_eq!(map.side_data_at(4, 4, 0), 2, "far corner of region ⇒ id 2");
    // Floor cells OUTSIDE the region keep id 0 (surface-kind default).
    assert_eq!(
        map.side_data_at(1, 1, 0),
        0,
        "floor outside region untouched"
    );
    assert_eq!(
        map.side_data_at(5, 5, 0),
        0,
        "floor outside region untouched"
    );
    // The AIR above a zone floor cell is preserved (solid-only stamp).
    assert_eq!(
        map.cell_at(3, 3, 1),
        CellKind::Empty,
        "air above the zone is not stamped"
    );
    assert_eq!(
        map.side_data_at(3, 3, 1),
        0,
        "air above the zone keeps id 0"
    );
}

/// PRECEDENCE 1: an explicit `[color_cell_ids]` entry is MORE SPECIFIC and WINS
/// over a zone region-fill covering the same cell (zones stamp first, then
/// color_cell_ids overwrite).
#[test]
fn color_cell_id_wins_over_overlapping_zone() {
    // Zone fills (2..4)² of the floor with floor_tile (id 2); an explicit
    // color_cell_ids entry repaints one of those cells with id 1 (floor_base).
    let src = format!(
        "{ZONE_FILL_BASE}\
[zone tile]
region = (2,2,0)-(4,4,0)
material = floor_tile
[color_cell_ids]
3,3,0 = 1
"
    );
    let lv = load_string(&src, "<zf-prec1>").expect("parses");
    let map = lv.build_map();
    // The overlapping cell takes the explicit id, NOT the zone id.
    assert_eq!(
        map.side_data_at(3, 3, 0),
        1,
        "explicit color_cell_id wins over the zone"
    );
    // A neighbour the explicit id did not touch keeps the zone id.
    assert_eq!(
        map.side_data_at(2, 2, 0),
        2,
        "non-overridden zone cell keeps zone id"
    );
}

/// PRECEDENCE 2: among two overlapping zones, the LATER-listed zone wins
/// (last-writer; authored `[zone]` order is the iteration order).
#[test]
fn later_zone_wins_over_earlier_overlapping_zone() {
    // Both zones cover (3,3,0). `first` binds floor_base (id 1); `second`,
    // listed after, binds floor_tile (id 2) → id 2 must win.
    let src = format!(
        "{ZONE_FILL_BASE}\
[zone first]
region = (2,2,0)-(3,3,0)
material = floor_base
[zone second]
region = (3,3,0)-(4,4,0)
material = floor_tile
"
    );
    let lv = load_string(&src, "<zf-prec2>").expect("parses");
    let map = lv.build_map();
    // Overlap cell: later zone (floor_tile, id 2) wins.
    assert_eq!(
        map.side_data_at(3, 3, 0),
        2,
        "later-listed zone wins the overlap"
    );
    // A cell only the first zone covers keeps id 1.
    assert_eq!(
        map.side_data_at(2, 2, 0),
        1,
        "first-only cell keeps the first zone id"
    );
    // A cell only the second zone covers keeps id 2.
    assert_eq!(
        map.side_data_at(4, 4, 0),
        2,
        "second-only cell keeps the second zone id"
    );
}

/// A zone naming a material with NO matching `[material NAME]` section is an
/// authoring error: `build_map_checked` reports it, while `build_map` skips the
/// zone with no panic (the cell keeps id 0).
#[test]
fn zone_unknown_material_is_reported_not_panicked() {
    let src = format!(
        "{ZONE_FILL_BASE}\
[zone ghost]
region = (2,2,0)-(4,4,0)
material = no_such_material
"
    );
    let lv = load_string(&src, "<zf-ghost>").expect("parses (resolution deferred)");
    // build_map does not panic and leaves the cell unstamped.
    let map = lv.build_map();
    assert_eq!(
        map.side_data_at(3, 3, 0),
        0,
        "unresolved zone leaves the cell at id 0"
    );
    // build_map_checked surfaces the authoring error.
    let err = lv
        .build_map_checked()
        .expect_err("unknown zone material must be reported");
    assert!(err.contains("no_such_material"), "got: {err}");
    assert!(
        err.contains("[zone ghost]"),
        "error names the offending zone: {err}"
    );
}

/// A zone whose box hangs OFF the map bounds (and into negatives) is clamped: only
/// the in-bounds solid cells are stamped; no panic, no out-of-bounds write.
#[test]
fn zone_region_is_clamped_to_map_bounds() {
    let src = format!(
        "{ZONE_FILL_BASE}\
[zone big]
region = (-3,-3,0)-(100,100,0)
material = floor_tile
"
    );
    let lv = load_string(&src, "<zf-clamp>").expect("parses");
    let map = lv.build_map();
    // In-bounds interior floor takes the id; perimeter walls (also solid) at the
    // floor plane are stamped too, but out-of-bounds reads are simply 0.
    assert_eq!(
        map.side_data_at(3, 3, 0),
        2,
        "in-bounds floor cell is stamped"
    );
    assert_eq!(
        map.side_data_at(-1, -1, 0),
        0,
        "out-of-bounds read is 0 (no panic)"
    );
}

/// BACK-COMPAT: a real shipped level with NO `[zone]` sections builds a map whose
/// floor `side_data` is byte-identical to its pre-P1 value (all zeros — no zone
/// ever stamps it). Snapshots a known level's floor-plane side_data.
#[test]
fn no_zone_sections_leave_side_data_byte_identical() {
    let lv = load(level_path("01.lvl")).expect("01.lvl must parse");
    assert!(lv.overlay.zones.is_empty(), "01.lvl declares no zones");
    let map = lv.build_map();
    let (sx, sy, _sz) = map.size();
    // Every floor cell (z = 0) must carry side_data 0 — no zone touched it.
    for y in 0..sy as i32 {
        for x in 0..sx as i32 {
            assert_eq!(
                map.side_data_at(x, y, 0),
                0,
                "floor cell ({x},{y},0) side_data must be 0 with no zones"
            );
        }
    }
}

/// The shipped P1 demo level loads, builds, and stamps its zone — the literal
/// "two floor cells read differently" proof at the file level.
#[test]
fn a1_demo_level_zone_stamps_distinct_floor() {
    let lv = load(level_path("A1-zone-materials.lvl")).expect("A1 demo loads");
    assert_eq!(lv.overlay.zones.len(), 1, "A1 has one zone");
    let map = lv
        .build_map_checked()
        .expect("A1 builds cleanly (zone resolves)");
    // A floor cell INSIDE the zone (4..7)² reads from the zone material; a floor
    // cell OUTSIDE reads the base — two floor cells, two different side_data.
    assert_eq!(map.cell_at(5, 5, 0), CellKind::Floor);
    let inside = map.side_data_at(5, 5, 0);
    let outside = map.side_data_at(2, 2, 0);
    assert!(inside > 0, "zone floor cell carries a palette id");
    assert_eq!(outside, 0, "non-zone floor cell stays surface default");
    assert_ne!(
        inside, outside,
        "two floor cells read differently — the P1 proof"
    );
}

#[test]
fn line_section_parses_polyline_path() {
    let src = format!(
        "{OVERLAY_BASE}\
[line trace]
path = 1,1,0; 4,1,0; 4,5,0
material = neon
scale = none
"
    );
    let lv = load_string(&src, "<ov-line>").expect("level with [line] parses");
    assert_eq!(lv.overlay.lines.len(), 1, "one line");
    let ln = &lv.overlay.lines[0];
    assert_eq!(ln.name, "trace");
    assert_eq!(ln.material, LineMaterialId("neon".to_string()));
    assert_eq!(ln.scale, ScaleRule::None);
    assert_eq!(ln.path.points.len(), 3);
    assert_eq!(ln.path.points[0], glam::Vec3::new(1.0, 1.0, 0.0));
    assert_eq!(ln.path.points[2], glam::Vec3::new(4.0, 5.0, 0.0));
}

#[test]
fn inset_section_parses_child_scene_reference() {
    let src = format!(
        "{OVERLAY_BASE}\
[inset врезка]
at_octave = 2-4
region = (0,0,0)-(2,2,0)
child = levels/child.lvl
"
    );
    let lv = load_string(&src, "<ov-inset>").expect("level with [inset] parses");
    assert_eq!(lv.overlay.insets.len(), 1, "one inset");
    let ins = &lv.overlay.insets[0];
    assert_eq!(ins.name, "врезка");
    assert_eq!(ins.at_octave, (2, 4));
    assert_eq!(
        ins.region,
        Region::Box {
            min: (0, 0, 0),
            max: (2, 2, 0)
        }
    );
    assert_eq!(
        ins.content,
        InsetContent::ChildScene("levels/child.lvl".to_string())
    );
}

#[test]
fn inset_single_octave_parses_as_degenerate_range() {
    let src = format!(
        "{OVERLAY_BASE}\
[inset i]
at_octave = 3
child = c.lvl
"
    );
    let lv = load_string(&src, "<ov-inset1>").expect("single-octave inset parses");
    assert_eq!(lv.overlay.insets[0].at_octave, (3, 3));
}

#[test]
fn overlay_sections_do_not_disturb_existing_parse() {
    // A level that ALSO authors overlay sections must keep its geometry/puzzle
    // parse identical to the same level without them.
    let base = load_string(OVERLAY_BASE, "<ov-base>").expect("base parses");
    let with = format!(
        "{OVERLAY_BASE}\
[zone z]
region = (1,1,0)-(2,2,0)
material = metal
[line l]
path = 0,0,0; 1,1,0
[inset n]
at_octave = 0
child = k.lvl
"
    );
    let lv = load_string(&with, "<ov-with>").expect("level with overlay parses");
    // Existing geometry/puzzle data is unchanged by the presence of the overlay.
    assert_eq!(lv.size, base.size);
    assert_eq!(lv.nodes.len(), base.nodes.len());
    assert_eq!(lv.pipes.len(), base.pipes.len());
    assert_eq!(lv.door.x, base.door.x);
    assert_eq!(
        (lv.player.x, lv.player.y, lv.player.z),
        (base.player.x, base.player.y, base.player.z)
    );
    // And the overlay itself is populated.
    assert_eq!(lv.overlay.zones.len(), 1);
    assert_eq!(lv.overlay.lines.len(), 1);
    assert_eq!(lv.overlay.insets.len(), 1);
    // While the base carries an empty overlay.
    assert!(base.overlay.is_empty());
}

#[test]
fn zone_bad_region_errors() {
    let src = format!("{OVERLAY_BASE}[zone z]\nregion = nonsense\n");
    assert!(
        load_string(&src, "<ov-bad>").is_err(),
        "a malformed [zone] region must error, not silently drop"
    );
}

// ── Scalable Material Overlay P4 — INSETS (bounded phase-band reveal) ─────────

/// (DP7) The `at_octave → phase_band` mapping is DETERMINISTIC and lands distinct
/// octaves at distinct phases, all reduced into `[0, LN2)`.
#[test]
fn phase_band_mapping_is_deterministic_and_spread() {
    // Pure function: same input → same band, every call.
    assert_eq!(phase_band_for_octave((0, 0)), phase_band_for_octave((0, 0)));
    let b0 = phase_band_for_octave((0, 0));
    let b1 = phase_band_for_octave((1, 1));
    let b2 = phase_band_for_octave((2, 2));
    // All endpoints are inside [0, LN2).
    for (lo, hi) in [b0, b1, b2] {
        assert!((0.0..LN2).contains(&lo), "lo {lo} out of [0,LN2)");
        assert!((0.0..LN2).contains(&hi), "hi {hi} out of [0,LN2)");
    }
    // Octave 0 maps to lo = 0, width = 0.2·LN2 (a single octave).
    assert!((b0.0 - 0.0).abs() < 1e-6, "octave 0 lo must be 0");
    assert!(
        (b0.1 - 0.2 * LN2).abs() < 1e-4,
        "single-octave width = 0.2·LN2"
    );
    // The golden-ratio spread puts successive octaves at DISTINCT lo phases.
    assert!(
        (b0.0 - b1.0).abs() > 1e-3,
        "octaves 0 and 1 must differ in phase"
    );
    assert!(
        (b1.0 - b2.0).abs() > 1e-3,
        "octaves 1 and 2 must differ in phase"
    );
    // A wide octave RANGE opens a wider window than a single octave.
    let wide = phase_band_for_octave((0, 2)); // span 3 ⇒ 0.6·LN2 wide
    assert!(
        (wide.1 - 0.6 * LN2).abs() < 1e-4,
        "3-octave width = 0.6·LN2"
    );
    // A full-octave-or-wider span collapses to the explicit always-on [0, LN2) band.
    assert_eq!(
        phase_band_for_octave((0, 4)),
        (0.0, LN2),
        "span 5 ⇒ whole loop"
    );
}

/// The representative-colour rule: first `[material].fg`, else `[palette]` fog tint,
/// else the neutral fallback — all masked to 24 bits and DETERMINISTIC.
#[test]
fn representative_colour_follows_documented_fallback_chain() {
    // 1) first [material] fg wins.
    let with_mat = load(level_path("A5-child.lvl")).expect("child parses");
    assert_eq!(
        representative_colour(&with_mat),
        0x00F2_A33C,
        "first material fg"
    );

    // 2) no material, but a [palette] fog tint ⇒ that tint.
    let src_pal =
        format!("{OVERLAY_BASE}[palette]\nfog_color = #123456\nfog_near = 1.0\nfog_far = 9.0\n");
    let pal = load_string(&src_pal, "<pal>").expect("palette level parses");
    assert!(pal.materials.is_empty());
    assert_eq!(representative_colour(&pal), 0x0012_3456, "palette fog tint");

    // 3) neither ⇒ the documented neutral fallback.
    let plain = load_string(OVERLAY_BASE, "<plain>").expect("plain parses");
    assert_eq!(
        representative_colour(&plain),
        INSET_FALLBACK_RGB,
        "neutral fallback"
    );
}

/// (DP7) Resolving the SAME parent twice yields BIT-IDENTICAL `ResolvedInset`s
/// (region + phase-band + rgb) — the determinism the renderer relies on.
#[test]
fn inset_resolution_is_deterministic() {
    let lv = load(level_path("A5-inset.lvl")).expect("A5-inset parses");
    let m1 = lv.build_map();
    let m2 = lv.build_map();
    let r1 = m1.resolved_insets();
    let r2 = m2.resolved_insets();
    assert_eq!(r1.len(), 1, "one resolved inset");
    assert_eq!(r1, r2, "same parent must resolve to identical insets");
    // The resolved inset carries the CHILD's amber colour (first material fg), the
    // authored region, and octave-0's [0, 0.2·LN2) band.
    let ins = &r1[0];
    assert_eq!(ins.rgb, 0x00F2_A33C, "child's first-material colour");
    assert_eq!(
        ins.region,
        Region::Box {
            min: (4, 0, 1),
            max: (7, 0, 1)
        }
    );
    assert!((ins.phase_band.0 - 0.0).abs() < 1e-6 && (ins.phase_band.1 - 0.2 * LN2).abs() < 1e-4);
}

/// A level with NO `[inset]` resolves to an EMPTY `resolved_insets` — the renderer's
/// override loop never runs (byte-identical back-compat).
#[test]
fn no_inset_level_resolves_empty() {
    let lv = load(level_path("A3-selfsimilar.lvl")).expect("A3 parses");
    assert!(lv.overlay.insets.is_empty(), "A3 has no [inset]");
    let map = lv.build_map();
    assert!(
        map.resolved_insets().is_empty(),
        "no inset ⇒ empty resolved_insets"
    );
    // Level 01 (no overlay at all) likewise.
    let m01 = load(level_path("01.lvl")).expect("01 parses").build_map();
    assert!(m01.resolved_insets().is_empty(), "01 has no insets");
}

/// THE CYCLE GUARD: a self-referential `[inset]` chain (parent → parent) must
/// TERMINATE with the fallback colour — no infinite recursion, no stack overflow.
#[test]
fn cycle_guard_terminates_self_reference() {
    use std::io::Write;
    let dir = std::env::temp_dir().join(format!("ts_p4_cycle_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let parent_path = dir.join("cycle_parent.lvl");
    // A parent whose inset references ITSELF (by file name) — the worst case.
    let parent_src = format!(
        "{OVERLAY_BASE}\
[zone z]
region = (1,0,1)-(5,0,3)
material = m
scale = self_similar arms:3 twist:0.0 warp:0.0
[inset loop]
at_octave = 0
region = (2,0,1)-(4,0,3)
child = cycle_parent.lvl
"
    );
    {
        let mut f = std::fs::File::create(&parent_path).expect("write parent");
        f.write_all(parent_src.as_bytes()).unwrap();
    }
    // Building the map must NOT recurse forever; it terminates and falls back.
    let lv = load(&parent_path).expect("cyclic parent parses");
    let map = lv.build_map(); // would stack-overflow without the guard
    let r = map.resolved_insets();
    assert_eq!(r.len(), 1, "the inset still resolves (to a fallback)");
    assert_eq!(
        r[0].rgb, INSET_FALLBACK_RGB,
        "a cyclic child uses the fallback colour"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A MISSING child reference also terminates with the fallback (every input shape
/// produces a defined output, never garbage / a panic).
#[test]
fn missing_inset_child_falls_back() {
    let src = format!(
        "{OVERLAY_BASE}\
[zone z]
region = (1,0,1)-(5,0,3)
material = m
scale = self_similar arms:3 twist:0.0 warp:0.0
[inset gone]
at_octave = 0
region = (2,0,1)-(4,0,3)
child = this_child_does_not_exist.lvl
"
    );
    // load_string ⇒ source_path = None ⇒ resolves relative to CWD; the file is absent.
    let lv = load_string(&src, "<missing-child>").expect("parses");
    let map = lv.build_map();
    let r = map.resolved_insets();
    assert_eq!(r.len(), 1);
    assert_eq!(
        r[0].rgb, INSET_FALLBACK_RGB,
        "a missing child uses the fallback colour"
    );
}

// ===========================================================================
// [cells] explicit geometry — the editor save format (spec §5)
// ===========================================================================

/// A minimal level shell (meta/room/pedestal/door/player) the `[cells]` tests
/// append a `[cells]` body onto.
fn cells_shell(body: &str) -> String {
    format!(
        "[meta]\nname = cells-test\n[room]\nsize = 3 2 2\nambient = solid\n\
         [pedestal]\ncenter = 1 1 1\nvolume = 1 1 1\n[door]\npos = 2 1 1\n\
         [player]\nspawn = 1 1 1\n{body}"
    )
}

#[test]
fn cells_parse_into_grid() {
    // 3x2x2 grid. z=0 all floor, z=1 a wall ring with empty centre.
    let body = "\
[cells]
size = 3,2,2
___
___

#.#
###
";
    let lv = load_string(&cells_shell(body), "<cells>").expect("[cells] parses");
    let g = lv.cells.as_ref().expect("Level.cells populated");
    assert_eq!(g.size, (3, 2, 2));
    // z=0 layer all Floor.
    assert_eq!(g.kind_at(0, 0, 0), CellKind::Floor);
    assert_eq!(g.kind_at(2, 1, 0), CellKind::Floor);
    // z=1 layer: (1,0,1) is the empty centre, the rest WallSolid.
    assert_eq!(g.kind_at(0, 0, 1), CellKind::WallSolid);
    assert_eq!(g.kind_at(1, 0, 1), CellKind::Empty);
    assert_eq!(g.kind_at(2, 0, 1), CellKind::WallSolid);
    assert_eq!(g.kind_at(1, 1, 1), CellKind::WallSolid);
}

#[test]
fn cells_are_authoritative_over_procedural() {
    // The procedural shell stamps a perimeter wall ring + floor/ceiling. The
    // [cells] block clears EVERY cell to Empty — and because cells stamp LAST,
    // the resulting map is empty (cells win over the procedural geometry).
    let body = "\
[cells]
size = 3,2,2
...
...

...
...
";
    let lv = load_string(&cells_shell(body), "<auth>").expect("parses");
    let map = lv.build_map();
    assert_eq!(
        map.count(CellKind::WallSolid),
        0,
        "cells cleared the procedural walls"
    );
    assert_eq!(
        map.count(CellKind::Floor),
        0,
        "cells cleared the procedural floor"
    );
}

#[test]
fn cells_material_ids_round_trip() {
    // A cell with a non-default side_data material id must survive save→load.
    // Coordinates avoid (2,1,1) — `cells_shell`'s `[door] pos` — whose Object cell
    // is stamped AFTER `[cells]`/material_ids with the DERIVED `DOOR_OBJECT_TAG`
    // (RC5 fix), so an authored id there is correctly NOT reapplied (the door's own
    // tag is non-zero) rather than testing this fixture's coincidental door overlap.
    let body = "\
[cells]
size = 3,2,2
___
___

#.#
###
material_ids =
0,0,1 = 3
0,1,1 = 7
";
    let lv = load_string(&cells_shell(body), "<mid>").expect("parses");
    let g = lv.cells.as_ref().unwrap();
    assert_eq!(g.material_ids, vec![(0, 0, 1, 3u16), (0, 1, 1, 7u16)]);
    // The ids reach the Map via build_map (stamped into side_data).
    let map = lv.build_map();
    assert_eq!(map.side_data_at(0, 0, 1), 3);
    assert_eq!(map.side_data_at(0, 1, 1), 7);
}

#[test]
fn cells_from_map_and_stamp_onto_round_trip() {
    // The editor save path: capture a Map → CellGrid → stamp back → identical Map.
    let body = "\
[cells]
size = 3,2,2
_O_
.#.

#/#
%@%
material_ids =
1,1,1 = 4
";
    let lv = load_string(&cells_shell(body), "<sync>").expect("parses");
    let map = lv.build_map();
    let grid = CellGrid::from_map(&map);
    let rebuilt = grid.stamp_onto(&map);
    let (sx, sy, sz) = map.size();
    for z in 0..sz {
        for y in 0..sy {
            for x in 0..sx {
                let (xi, yi, zi) = (x as i32, y as i32, z as i32);
                assert_eq!(
                    map.cell_at(xi, yi, zi),
                    rebuilt.cell_at(xi, yi, zi),
                    "kind at {xi},{yi},{zi}"
                );
                assert_eq!(
                    map.side_data_at(xi, yi, zi),
                    rebuilt.side_data_at(xi, yi, zi),
                    "side_data at {xi},{yi},{zi}"
                );
            }
        }
    }
}

#[test]
fn cells_edge_cases() {
    // Empty grid (0 cells): a `size = 0,0,0` [cells] with no rows.
    let body_empty = "[cells]\nsize = 0,0,0\n";
    let lv = load_string(&cells_shell(body_empty), "<empty>").expect("empty grid parses");
    let g = lv.cells.as_ref().unwrap();
    assert_eq!(g.size, (0, 0, 0));
    assert!(g.kinds.is_empty());

    // Single non-empty cell in a 1x1x1 grid.
    let body_one = "[cells]\nsize = 1,1,1\n#\n";
    let lv = load_string(&cells_shell(body_one), "<one>").expect("single-cell parses");
    let g = lv.cells.as_ref().unwrap();
    assert_eq!(g.size, (1, 1, 1));
    assert_eq!(g.kind_at(0, 0, 0), CellKind::WallSolid);

    // Full grid (every cell WallSolid).
    let body_full = "[cells]\nsize = 2,2,1\n##\n##\n";
    let lv = load_string(&cells_shell(body_full), "<full>").expect("full grid parses");
    let g = lv.cells.as_ref().unwrap();
    assert_eq!(
        g.kinds
            .iter()
            .filter(|&&k| k == CellKind::WallSolid)
            .count(),
        4
    );
}

#[test]
fn cells_shapes_readable_syntax_parses() {
    // M3c: the `shapes =` sub-table uses READABLE direction names.
    use re_core::{DiagXY, RampDir, Shape};
    let body = "\
[cells]
size = 2,2,1
##
##
shapes =
0 0 0 wedgexy nw
1 1 0 rampz posy
";
    let lv = load_string(&cells_shell(body), "<shapes>").expect("shapes parse");
    let g = lv.cells.as_ref().unwrap();
    assert_eq!(
        g.shapes,
        vec![
            (0, 0, 0, Shape::WedgeXY(DiagXY::NW)),
            (1, 1, 0, Shape::RampZ(RampDir::PosY)),
        ]
    );
}

// ---------------------------------------------------------------------------
// IMAGE TEXTURES: `[texture NAME]` + `[material] recipe = image texture = NAME`.
// ---------------------------------------------------------------------------

/// Encode a tiny solid-red RGBA PNG to a temp file and return its absolute path.
/// (Absolute so `decode_texture_file`'s cwd-relative `fs::read` resolves it from
/// any test working directory.)
fn write_tiny_red_png() -> PathBuf {
    // 2×2 fully-opaque red.
    let rgba: Vec<u8> = (0..4).flat_map(|_| [255u8, 0, 0, 255]).collect();
    let img: image::RgbaImage = image::ImageBuffer::from_raw(2, 2, rgba).expect("2x2 rgba buffer");
    let mut path = std::env::temp_dir();
    // Unique name PER CALL so parallel tests in this binary never share a file
    // (process::id() alone collides across threads — one test's remove_file would
    // yank the fixture out from under another mid-decode).
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    path.push(format!("ts_tex_fixture_{}_{}.png", std::process::id(), seq));
    img.save(&path).expect("write png fixture");
    path
}

/// PARSE: `[texture NAME] file = …` populates `Level.textures`, and a
/// `[material] recipe = image texture = NAME` stores the image-recipe placeholder
/// plus the bound texture name (resolved to a real `tex_id` only in `build_map`).
#[test]
fn texture_section_and_image_material_parse() {
    let src = "\
[meta]
name = tx
[room]
size = 6 4 4
ambient = solid
[texture pic]
file = assets/texture/wall_tex.png
[material img_wall]
recipe = image
texture = pic
surface = wall_solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<tx>").expect("level with [texture] must parse");
    assert_eq!(lv.textures.len(), 1, "one [texture] section");
    assert_eq!(lv.textures[0].name, "pic");
    assert_eq!(lv.textures[0].file, "assets/texture/wall_tex.png");
    let m = &lv.materials[0];
    assert_eq!(m.texture.as_deref(), Some("pic"), "texture name bound");
    // Placeholder image recipe (tex_id resolved later in build_map).
    assert!(
        matches!(m.recipe, Some(MaterialRecipe::Image { .. })),
        "recipe = image parses to MaterialRecipe::Image"
    );
    assert_eq!(lv.surface_materials.wall_solid.as_deref(), Some("img_wall"));
    // No `span` key ⇒ legacy per-cell tiling.
    assert_eq!(m.image_span, None, "no span key ⇒ legacy per-cell tiling");
}

/// PARSE (M3a): `[material] span = ALONG_MIN,ALONG_SIZE UP_TOP,UP_SIZE` parses the
/// world-anchored image span (ONE figure across the surface). A malformed/degenerate
/// span (zero size) is a level error, not silently dropped.
#[test]
fn image_material_span_parses() {
    let base = "\
[meta]
name = tx
[room]
size = 6 4 4
ambient = solid
[texture pic]
file = assets/texture/wall_tex.png
[material img_wall]
recipe = image
texture = pic
surface = wall_solid
span = 2,7 3,2
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(base, "<txspan>").expect("level with image span must parse");
    let m = &lv.materials[0];
    assert_eq!(
        m.image_span,
        Some(re_world::ImageSpan {
            along_min: 2.0,
            along_size: 7.0,
            up_top: 3.0,
            up_size: 2.0
        }),
        "span = 2,7 3,2 must parse to the world-anchored ImageSpan"
    );

    // A zero-size span is rejected (a collapsed figure is a level error).
    let bad = base.replace("span = 2,7 3,2", "span = 2,0 3,2");
    assert!(
        load_string(&bad, "<txbad>").is_err(),
        "a zero ALONG_SIZE must be a parse error, not a silent drop"
    );
}

// ── helpers shared by the fit / subcell tests below ──────────────────────────

fn image_material_level(extra_keys: &str) -> String {
    format!(
        "\
[meta]
name = fitsc
[room]
size = 6 4 4
ambient = solid
[texture pic]
file = assets/texture/wall_tex.png
[material img_wall]
recipe = image
texture = pic
surface = wall_solid
{extra_keys}
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
"
    )
}

/// PARSE: `fit = contain` is accepted and stored; absent `fit` defaults to `Contain`.
#[test]
fn image_material_fit_parses() {
    // Explicit `fit = contain`
    let src = image_material_level("fit = contain");
    let lv = load_string(&src, "<fitc>").expect("fit=contain must parse");
    let m = &lv.materials[0];
    assert_eq!(m.image_fit, re_world::Fit::Contain, "fit = contain");

    // Explicit `fit = stretch`
    let src = image_material_level("fit = stretch");
    let lv = load_string(&src, "<fits>").expect("fit=stretch must parse");
    assert_eq!(
        lv.materials[0].image_fit,
        re_world::Fit::Stretch,
        "fit = stretch"
    );

    // Explicit `fit = cover`
    let src = image_material_level("fit = cover");
    let lv = load_string(&src, "<fitcv>").expect("fit=cover must parse");
    assert_eq!(
        lv.materials[0].image_fit,
        re_world::Fit::Cover,
        "fit = cover"
    );

    // Case-insensitive
    let src = image_material_level("fit = COVER");
    let lv = load_string(&src, "<fitcvup>").expect("fit=COVER must parse case-insensitively");
    assert_eq!(
        lv.materials[0].image_fit,
        re_world::Fit::Cover,
        "fit = COVER (case-insensitive)"
    );

    // Absent `fit` key ⇒ default `Contain`
    let src = image_material_level("");
    let lv = load_string(&src, "<fitnone>").expect("no fit key must default");
    assert_eq!(
        lv.materials[0].image_fit,
        re_world::Fit::Contain,
        "absent fit ⇒ Contain default"
    );

    // Unknown value is a parse error
    let bad = image_material_level("fit = none_of_the_above");
    assert!(
        load_string(&bad, "<fitbad>").is_err(),
        "an unknown fit value must be a parse error"
    );
}

/// PARSE: `fit = contain` + `subcell = contour` round-trips into the right
/// `MaterialRecipe::Image` (the core requirement from the task brief).
#[test]
fn image_material_fit_subcell_contour_round_trip() {
    let src = image_material_level("fit = contain\nsubcell = contour");
    let lv = load_string(&src, "<fitcont>").expect("fit=contain subcell=contour must parse");
    let m = &lv.materials[0];
    assert_eq!(m.image_fit, re_world::Fit::Contain, "fit field");
    assert_eq!(
        m.image_subcell,
        re_world::SubcellMode::Contour,
        "subcell field"
    );
    assert!(
        matches!(m.recipe, Some(MaterialRecipe::Image { .. })),
        "recipe is Image variant"
    );
}

/// PARSE: `subcell = ascii`, `subcell = contour`, `subcell = lines` parse correctly.
#[test]
fn image_material_subcell_extended_parses() {
    let src = image_material_level("subcell = ascii");
    let lv = load_string(&src, "<scascii>").expect("subcell=ascii must parse");
    assert_eq!(
        lv.materials[0].image_subcell,
        re_world::SubcellMode::Ascii,
        "subcell = ascii"
    );

    let src = image_material_level("subcell = contour");
    let lv = load_string(&src, "<sccontour>").expect("subcell=contour must parse");
    assert_eq!(
        lv.materials[0].image_subcell,
        re_world::SubcellMode::Contour,
        "subcell = contour"
    );

    let src = image_material_level("subcell = lines");
    let lv = load_string(&src, "<sclines>").expect("subcell=lines must parse");
    assert_eq!(
        lv.materials[0].image_subcell,
        re_world::SubcellMode::Lines,
        "subcell = lines"
    );

    // Case-insensitive
    let src = image_material_level("subcell = ASCII");
    let lv = load_string(&src, "<scasciiup>").expect("subcell=ASCII must parse case-insensitively");
    assert_eq!(
        lv.materials[0].image_subcell,
        re_world::SubcellMode::Ascii,
        "subcell = ASCII (case-insensitive)"
    );

    // Unknown value remains a parse error (pattern-matching existing behaviour)
    let bad = image_material_level("subcell = dissolve");
    assert!(
        load_string(&bad, "<scbad>").is_err(),
        "an unknown subcell value must be a parse error"
    );
}

/// PARSE: the two rungs added to the sub-cell ladder — `half` (2×1) and `sextant`
/// (3×2) — plus their geometry aliases, which exist so a level can be authored in the
/// same `RxC` vocabulary the ladder is discussed in.
#[test]
fn image_material_subcell_half_and_sextant_parse() {
    for (kw, want) in [
        ("half", re_world::SubcellMode::Half),
        ("2x1", re_world::SubcellMode::Half),
        ("sextant", re_world::SubcellMode::Sextant),
        ("3x2", re_world::SubcellMode::Sextant),
        // The pre-existing aliases must still resolve — the ladder grew, it did not move.
        ("quad", re_world::SubcellMode::Quad),
        ("2x2", re_world::SubcellMode::Quad),
        ("braille", re_world::SubcellMode::Braille),
        ("2x4", re_world::SubcellMode::Braille),
    ] {
        let src = image_material_level(&format!("subcell = {kw}"));
        let lv = load_string(&src, "<subcell>")
            .unwrap_or_else(|e| panic!("subcell = {kw} must parse: {e}"));
        assert_eq!(lv.materials[0].image_subcell, want, "subcell = {kw}");
    }

    // Case-insensitive, like every other subcell keyword.
    let src = image_material_level("subcell = SEXTANT");
    let lv = load_string(&src, "<scsexup>").expect("subcell=SEXTANT must parse");
    assert_eq!(
        lv.materials[0].image_subcell,
        re_world::SubcellMode::Sextant
    );
}

/// PARSE: `fit` is forwarded into the resolved `MaterialRecipe::Image` after
/// `renderer_material_defs` (the late-resolve step). This exercises the full
/// `LevelMaterial → MaterialDef` path, not just the parse field.
#[test]
fn image_material_fit_forwarded_to_material_def() {
    let src = image_material_level("fit = cover\nsubcell = ascii");
    let lv =
        load_string(&src, "<fwdfitcv>").expect("level with fit=cover subcell=ascii must parse");
    // renderer_material_defs resolves tex_id (0 placeholder — no real PNG, that's fine)
    // and folds in fit + subcell.
    let tex_ids: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let defs = lv.renderer_material_defs(&tex_ids);
    // Find the img_wall def.
    let def = defs
        .iter()
        .find(|d| d.name == "img_wall")
        .expect("img_wall must appear in defs");
    assert_eq!(
        def.recipe,
        Some(MaterialRecipe::Image {
            tex_id: 0,
            span: None,
            subcell: re_world::SubcellMode::Ascii,
            fit: re_world::Fit::Cover,
            tile: re_world::IMAGE_TILE_NONE,
            relief: None,
        }),
        "fit=Cover and subcell=Ascii must be forwarded to the resolved MaterialRecipe"
    );
}

/// BUILD: decoding a real PNG fixture populates `Map::textures` and the bound
/// `[material]` resolves to `MaterialRecipe::Image { tex_id }` pointing at the
/// pushed texture (id 0, the first decoded texture). The decoded pixels are the
/// fixture's red.
#[test]
fn build_map_decodes_texture_and_resolves_image_material() {
    let png = write_tiny_red_png();
    let png_str = png.to_string_lossy().replace('\\', "/");
    let src = format!(
        "\
[meta]
name = txb
[room]
size = 6 4 4
ambient = solid
[texture pic]
file = {png_str}
[material img_wall]
recipe = image
texture = pic
surface = wall_solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
"
    );
    let lv = load_string(&src, "<txb>").expect("level parses");
    let map = lv.build_map();
    // The PNG decoded and was pushed onto the map's texture table.
    assert_eq!(map.textures.len(), 1, "one decoded texture on the map");
    let tex = map.texture(0).expect("texture 0 present");
    assert_eq!((tex.width, tex.height), (2, 2));
    assert_eq!(
        tex.sample_nearest(0.5, 0.5),
        [255, 0, 0, 255],
        "red fixture"
    );
    // The wall_solid surface (slot 0) resolved to an Image recipe pointing at id 0.
    let mats = map.materials();
    assert_eq!(mats.len(), 5);
    assert!(!mats[0].fallback, "wall_solid slot authored");
    assert_eq!(
        mats[0].recipe,
        // No `span` / `subcell` / `fit` key in this fixture ⇒ legacy per-cell tiling +
        // solid block + default Contain fit.
        Some(MaterialRecipe::Image {
            tex_id: 0,
            span: None,
            subcell: re_world::SubcellMode::Off,
            fit: re_world::Fit::Contain,
            tile: re_world::IMAGE_TILE_NONE,
            relief: None,
        }),
        "image recipe resolved to the decoded texture's id"
    );
    let _ = std::fs::remove_file(&png);
}

// ---------------------------------------------------------------------------
// IMAGE RELIEF: `[texture] height =`, `[material] tile =`, `[material] relief =`
// (engine temporal-stability + lighting spec 1.4).
// ---------------------------------------------------------------------------

/// Encode a greyscale PNG of the given size to a temp file and return its path.
/// `f(x, y)` supplies the 8-bit height of each texel, so a test can author a ramp,
/// a checker, or a single bright texel and then assert on what the shader read.
fn write_height_png(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> PathBuf {
    let img: image::GrayImage = image::ImageBuffer::from_fn(w, h, |x, y| image::Luma([f(x, y)]));
    let mut path = std::env::temp_dir();
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    path.push(format!(
        "ts_height_fixture_{}_{}.png",
        std::process::id(),
        seq
    ));
    img.save(&path).expect("write height png fixture");
    path
}

/// A minimal, valid level shell with one `[texture pic]` (whose body the caller
/// supplies) and one `recipe = image` material bound to `wall_solid`.
fn image_level(tex_body: &str, mat_body: &str) -> String {
    format!(
        "\
[meta]
name = relief
[room]
size = 6 4 4
ambient = solid
[texture pic]
{tex_body}[material img_wall]
recipe = image
texture = pic
{mat_body}surface = wall_solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
"
    )
}

/// PROTECTS: `[texture NAME] height = <path>` actually DECODES and rides into the
/// renderer on the colour image. Without this the whole relief feature is a parsed
/// string that never reaches a texel.
#[test]
fn an_authored_height_map_decodes_onto_the_colour_image() {
    let png = write_tiny_red_png(); // 2x2 colour
                                    // Same 2x2 dims: left column low, right column high.
    let hpng = write_height_png(2, 2, |x, _| if x == 0 { 0 } else { 255 });
    let src = image_level(
        &format!(
            "file = {}\nheight = {}\n",
            png.to_string_lossy().replace('\\', "/"),
            hpng.to_string_lossy().replace('\\', "/")
        ),
        "",
    );
    let lv = load_string(&src, "<relief>").expect("level parses");
    assert!(lv.textures[0].height.is_some(), "height key parsed");
    let map = lv.build_map();
    let tex = map.texture(0).expect("texture 0 present");
    assert!(
        tex.has_height_map(),
        "the decoded image must carry the relief channel"
    );
    assert_eq!(tex.sample_height_nearest(0.25, 0.5), 0.0, "left column low");
    assert_eq!(
        tex.sample_height_nearest(0.75, 0.5),
        1.0,
        "right column high"
    );
    let _ = std::fs::remove_file(&png);
    let _ = std::fs::remove_file(&hpng);
}

/// PROTECTS: an author will not always export the relief at the colour image's exact
/// size. A dimension MISMATCH must resample, never panic and never silently drop the
/// channel - and the resampled field must still put the high side on the high side.
#[test]
fn a_height_map_of_different_dimensions_is_resampled_not_rejected() {
    let png = write_tiny_red_png(); // 2x2 colour
                                    // 8x8 height: left half low, right half high. Nothing about it matches 2x2.
    let hpng = write_height_png(8, 8, |x, _| if x < 4 { 0 } else { 255 });
    let src = image_level(
        &format!(
            "file = {}\nheight = {}\n",
            png.to_string_lossy().replace('\\', "/"),
            hpng.to_string_lossy().replace('\\', "/")
        ),
        "",
    );
    let lv = load_string(&src, "<relief-mismatch>").expect("level parses");
    let map = lv.build_map();
    let tex = map.texture(0).expect("texture 0 present");
    assert_eq!(
        (tex.width, tex.height),
        (2, 2),
        "colour dims are authoritative"
    );
    assert!(
        tex.has_height_map(),
        "a mismatch must resample, not drop the channel"
    );
    assert_eq!(
        tex.height_map.as_ref().unwrap().len(),
        4,
        "the channel is resampled to the COLOUR image's texel count"
    );
    let (lo, hi) = (
        tex.sample_height_nearest(0.25, 0.5),
        tex.sample_height_nearest(0.75, 0.5),
    );
    assert!(
        lo < 0.25 && hi > 0.75,
        "orientation survives the resample: {lo} vs {hi}"
    );
    let _ = std::fs::remove_file(&png);
    let _ = std::fs::remove_file(&hpng);
}

/// PROTECTS: a broken `height =` costs only the relief, not the whole texture. The
/// colour image must still decode and render (the same soft-skip rule `file` has),
/// while `build_map_checked` still reports the failure so it is not invisible.
#[test]
fn a_missing_height_file_is_a_soft_skip_but_a_checked_build_reports_it() {
    let png = write_tiny_red_png();
    let src = image_level(
        &format!(
            "file = {}\nheight = /no/such/dir/definitely_missing_height.png\n",
            png.to_string_lossy().replace('\\', "/")
        ),
        "",
    );
    let lv = load_string(&src, "<relief-missing>").expect("level parses");
    let map = lv.build_map();
    assert_eq!(map.textures.len(), 1, "the colour texture still decoded");
    assert!(
        !map.texture(0).unwrap().has_height_map(),
        "a bad relief file leaves the image flat rather than losing it"
    );
    let err = lv
        .build_map_checked()
        .expect_err("checked build reports the bad height");
    assert!(
        err.contains("height"),
        "the error names the height channel: {err}"
    );
    let _ = std::fs::remove_file(&png);
}

/// PROTECTS: `tile = a b` reaches the resolved recipe. A parsed-but-dropped tile
/// would look exactly like "tiling does not work" in the renderer.
#[test]
fn tile_parses_and_reaches_the_resolved_image_recipe() {
    let src = image_level("file = nope.png\n", "tile = 4 2\n");
    let lv = load_string(&src, "<tile>").expect("level parses");
    assert_eq!(lv.materials[0].image_tile, Some((4.0, 2.0)));
    let defs = lv.material_defs(&std::collections::HashMap::new());
    let def = defs
        .iter()
        .find(|d| d.name == "img_wall")
        .expect("img_wall in defs");
    match def.recipe {
        Some(MaterialRecipe::Image { tile, .. }) => assert_eq!(tile, (4.0, 2.0)),
        other => panic!("expected an Image recipe carrying the tile, got {other:?}"),
    }
}

/// PROTECTS: an image material with NONE of the new keys resolves to EXACTLY the
/// recipe it resolved to before they existed. This is the parser half of the
/// no-regression guarantee the golden corpus checks end-to-end.
#[test]
fn an_image_material_without_the_new_keys_is_unchanged() {
    let src = image_level("file = nope.png\n", "");
    let lv = load_string(&src, "<plain>").expect("level parses");
    assert_eq!(lv.materials[0].image_tile, None);
    assert_eq!(lv.materials[0].image_relief, None);
    assert_eq!(lv.textures[0].height, None);
    let defs = lv.material_defs(&std::collections::HashMap::new());
    let def = defs
        .iter()
        .find(|d| d.name == "img_wall")
        .expect("img_wall in defs");
    assert_eq!(
        def.recipe,
        Some(MaterialRecipe::Image {
            tex_id: 0,
            span: None,
            subcell: re_world::SubcellMode::Off,
            fit: re_world::Fit::Contain,
            tile: re_world::IMAGE_TILE_NONE,
            relief: None,
        })
    );
}

/// PROTECTS: `relief = height_scale=… normal_strength=…` parses both knobs, in the
/// same KNOB=VALUE style `recipe = KIND k=v …` uses, and reaches the recipe.
#[test]
fn relief_parses_both_knobs_and_reaches_the_resolved_image_recipe() {
    let src = image_level(
        "file = nope.png\n",
        "relief = height_scale=1.2 normal_strength=0.9\n",
    );
    let lv = load_string(&src, "<relief-knobs>").expect("level parses");
    let r = lv.materials[0].image_relief.expect("relief authored");
    assert!((r.height_scale - 1.2).abs() < 1e-6);
    assert!((r.normal_strength - 0.9).abs() < 1e-6);
    let defs = lv.material_defs(&std::collections::HashMap::new());
    let def = defs
        .iter()
        .find(|d| d.name == "img_wall")
        .expect("img_wall in defs");
    match def.recipe {
        Some(MaterialRecipe::Image {
            relief: Some(rr), ..
        }) => {
            assert!((rr.height_scale - 1.2).abs() < 1e-6);
            assert!((rr.normal_strength - 0.9).abs() < 1e-6);
        }
        other => panic!("expected an Image recipe carrying the relief, got {other:?}"),
    }
}

/// PROTECTS: a bare `relief =` means "relief on, full strength" rather than an
/// error - the knobs are optional, only their SPELLING is strict.
#[test]
fn a_bare_relief_key_turns_relief_on_at_the_documented_default() {
    let src = image_level("file = nope.png\n", "relief =\n");
    let lv = load_string(&src, "<relief-bare>").expect("level parses");
    assert_eq!(
        lv.materials[0].image_relief,
        Some(re_world::ImageRelief::DEFAULT)
    );
}

/// PROTECTS: every malformed value is a LINE-NUMBERED parse error, never a silent
/// default. A `tile = 0 4` that quietly became `1 1` would be debugged in the
/// renderer, and a misspelled relief knob that quietly did nothing would be debugged
/// in the light model - both are hours spent in the wrong crate.
#[test]
fn bad_tile_and_relief_values_are_line_numbered_parse_errors() {
    // (material body, a fragment the message must mention)
    let cases: [(&str, &str); 7] = [
        ("tile = 0 4\n", "tile"),
        ("tile = -1 1\n", "tile"),
        ("tile = 2\n", "tile"),
        ("tile = 1 2 3\n", "tile"),
        ("tile = a b\n", "tile"),
        ("relief = height=2\n", "unknown knob"),
        ("relief = height_scale=nope\n", "not a number"),
    ];
    for (body, needle) in cases {
        let src = image_level("file = nope.png\n", body);
        let err = load_string(&src, "<bad>").expect_err(&format!("{body:?} must not parse"));
        let msg = format!("{err}");
        assert!(
            msg.contains(needle),
            "{body:?} -> message must mention {needle:?}: {msg}"
        );
        // The offending key sits on the line the shell puts it on; the error must
        // point AT a line, not at the file as a whole.
        let line = src
            .lines()
            .position(|l| l.starts_with(body.trim_end_matches('\n').split(' ').next().unwrap()))
            .expect("the bad key is in the fixture")
            + 1;
        // `ParseError` renders as `LEVEL: <src>:<line>: <msg>`, so the source name
        // pins the number to the LINE field and not to some digit in the text.
        assert!(
            msg.contains(&format!("<bad>:{line}:")),
            "{body:?} -> message must carry line {line}: {msg}"
        );
    }
}

/// BUILD (failure path): a `[texture]` whose file is missing is SKIPPED — the map
/// still builds with no texture, and the image material keeps its placeholder id
/// (no panic). `build_map_checked` surfaces the error.
#[test]
fn build_map_missing_texture_is_nonfatal() {
    let src = "\
[meta]
name = txm
[room]
size = 6 4 4
ambient = solid
[texture pic]
file = /no/such/file/definitely_missing.png
[material img_wall]
recipe = image
texture = pic
surface = wall_solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<txm>").expect("level parses");
    // build_map does NOT panic; the missing texture is simply absent.
    let map = lv.build_map();
    assert!(map.textures.is_empty(), "missing texture decoded nothing");
    // build_map_checked surfaces the decode error.
    assert!(
        lv.build_map_checked().is_err(),
        "a missing [texture] file is reported by build_map_checked"
    );
}

/// The in-repo TX5 demo level loads and surfaces the generated plate on the walls.
#[test]
fn tx5_genplate_demo_builds_with_texture() {
    let lv = load(level_path("TX5-genplate.lvl")).expect("TX5-genplate.lvl must parse");
    assert_eq!(lv.textures.len(), 1);
    assert_eq!(lv.textures[0].name, "genplate");
    assert_eq!(lv.textures[0].file, "assets/texture/genplate_tex.png");
    // THE ANCHOR TEST. The authored path is relative to the level TREE, not to the
    // process CWD — and under `cargo test` the CWD is `crates/re_level`, where it
    // does not resolve. This assertion used to be impossible to write: the test had
    // to rebuild the path itself from `CARGO_MANIFEST_DIR` and decode the file by
    // hand, which proved the ASSET was valid but said nothing about whether the
    // level could actually load it. It could not — the texture was silently skipped
    // and the level rendered its untextured fallback.
    assert!(
        !PathBuf::from(&lv.textures[0].file).exists(),
        "precondition: the authored path must NOT resolve against this test's cwd, \
         or the anchor is not being exercised"
    );
    assert!(
        lv.build_map_checked().is_ok(),
        "the level resolves its own texture regardless of cwd: {:?}",
        lv.build_map_checked().err()
    );
    let map = lv.build_map();
    let img = map
        .texture(0)
        .expect("the decoded genplate image is on the built map");
    // `re_voximport::texfixture::PLATE_SIZE` — re_level cannot dev-dep on
    // re_voximport (which already dev-deps on re_level), so the size is pinned
    // here as a plain literal; `texfixture`'s own tests pin the generator itself.
    assert_eq!((img.width, img.height), (64, 64), "genplate asset dims");
    // The first material is bound to the image recipe and its `tex_id` now
    // resolves against a texture that is really there.
    let m = &lv.materials[0];
    assert!(matches!(m.recipe, Some(MaterialRecipe::Image { .. })));
    assert_eq!(m.texture.as_deref(), Some("genplate"));
}

#[test]
fn no_cells_section_parses_as_before() {
    // A level WITHOUT [cells] carries no explicit grid (back-compat).
    let lv = load(level_path("01.lvl")).expect("01.lvl parses");
    assert!(lv.cells.is_none(), "01.lvl has no [cells] section");
}

// ───────────────────────────── object nesting ─────────────────────────────

/// A small valid level with three nodes (a/b/c) + a door/spawn/pedestal — enough
/// to wire a multi-level hierarchy in tests.
fn hier_base() -> &'static str {
    "[meta]
name = h
[room]
size = 8 8 4
[node a]
pos = 1 1 1
is_input = yes
[node b]
pos = 4 1 1
[node c]
pos = 6 1 1
is_output = yes
[pedestal]
center = 6 1 1
volume = 1 1 1
[door]
pos = 7 1 1
[player]
spawn = 3 1 1
"
}

#[test]
fn hier_set_parent_and_descendants() {
    use crate::{HierKind, HierRef};
    let mut lv = load_string(hier_base(), "<h>").expect("parses");
    let a = HierRef::new(HierKind::Node, 0);
    let b = HierRef::new(HierKind::Node, 1);
    let c = HierRef::new(HierKind::Node, 2);
    // b -> a, c -> b (a multi-level chain).
    assert!(lv.set_hier_parent(b, a));
    assert!(lv.set_hier_parent(c, b));
    assert_eq!(lv.hier_parent_of(b), Some(a));
    assert_eq!(lv.hier_parent_of(c), Some(b));
    // Descendants of a are {b, c} (transitive).
    let desc = lv.hier_descendants(a);
    assert!(desc.contains(&b) && desc.contains(&c) && desc.len() == 2);
}

#[test]
fn hier_refuses_self_and_cycle() {
    use crate::{HierKind, HierRef};
    let mut lv = load_string(hier_base(), "<h>").expect("parses");
    let a = HierRef::new(HierKind::Node, 0);
    let b = HierRef::new(HierKind::Node, 1);
    // Self-parent refused.
    assert!(!lv.set_hier_parent(a, a));
    assert!(lv.hier_parent_of(a).is_none());
    // b -> a OK; then a -> b would close a cycle → refused.
    assert!(lv.set_hier_parent(b, a));
    assert!(lv.hier_would_cycle(a, b));
    assert!(!lv.set_hier_parent(a, b));
    assert_eq!(
        lv.hier_parent_of(b),
        Some(a),
        "the original link is untouched"
    );
}

#[test]
fn hier_one_parent_per_child() {
    use crate::{HierKind, HierRef};
    let mut lv = load_string(hier_base(), "<h>").expect("parses");
    let a = HierRef::new(HierKind::Node, 0);
    let b = HierRef::new(HierKind::Node, 1);
    let c = HierRef::new(HierKind::Node, 2);
    // Re-parenting c replaces its single parent link.
    assert!(lv.set_hier_parent(c, a));
    assert!(lv.set_hier_parent(c, b));
    assert_eq!(lv.hier_parent_of(c), Some(b));
    assert_eq!(lv.hierarchy.iter().filter(|(ch, _)| *ch == c).count(), 1);
}

#[test]
fn hier_fixup_after_remove_drops_and_shifts() {
    use crate::{HierKind, HierRef};
    let mut lv = load_string(hier_base(), "<h>").expect("parses");
    let a = HierRef::new(HierKind::Node, 0);
    let b = HierRef::new(HierKind::Node, 1);
    let c = HierRef::new(HierKind::Node, 2);
    lv.set_hier_parent(b, a); // b(1) -> a(0)
    lv.set_hier_parent(c, b); // c(2) -> b(1)
                              // Remove node a (index 0): its link is dropped, and b/c shift down to 0/1.
    lv.nodes.remove(0);
    lv.hier_fixup_after_remove(HierKind::Node, 0);
    // a's links gone; the surviving c->b link now reads (1)->(0).
    let b_new = HierRef::new(HierKind::Node, 0);
    let c_new = HierRef::new(HierKind::Node, 1);
    assert_eq!(
        lv.hier_parent_of(c_new),
        Some(b_new),
        "c->b shifted down by one"
    );
    assert!(
        lv.hier_parent_of(b_new).is_none(),
        "b's parent (a) was removed"
    );
}

#[test]
fn hier_unparent_clears_only_one_edge() {
    use crate::{HierKind, HierRef};
    let mut lv = load_string(hier_base(), "<h>").expect("parses");
    let a = HierRef::new(HierKind::Node, 0);
    let b = HierRef::new(HierKind::Node, 1);
    let c = HierRef::new(HierKind::Node, 2);
    lv.set_hier_parent(b, a);
    lv.set_hier_parent(c, b);
    assert!(lv.clear_hier_parent(b));
    assert!(lv.hier_parent_of(b).is_none());
    assert_eq!(lv.hier_parent_of(c), Some(b), "c keeps its own link");
}

#[test]
fn hier_prune_drops_out_of_range_and_cyclic() {
    // A hand-written [hierarchy] with an out-of-range index + a self link is pruned
    // on load (the level still parses; bad links just vanish).
    let src = format!(
        "{}[hierarchy]\nnode 1 -> node 0\nnode 0 -> node 0\nnode 5 -> node 0\n",
        hier_base()
    );
    let lv = load_string(&src, "<hp>").expect("parses with pruning");
    use crate::{HierKind, HierRef};
    let a = HierRef::new(HierKind::Node, 0);
    let b = HierRef::new(HierKind::Node, 1);
    // The valid b->a link survives; the self link and the index-5 link are gone.
    assert_eq!(lv.hier_parent_of(b), Some(a));
    assert_eq!(lv.hierarchy.len(), 1);
}

// ───────────────────────── pipe authoring helpers (M4b) ─────────────────────────

#[test]
fn push_pipe_between_auto_routes_l_shaped_path() {
    let mut lv = Level {
        name: "p".into(),
        title: "p".into(),
        size: (16, 16, 4),
        ..Default::default()
    };
    let a = lv.push_node(1, 1, 0);
    let b = lv.push_node(4, 3, 0);
    let pi = lv.push_pipe_between(a, b).expect("pipe created");
    let p = &lv.pipes[pi];
    // Endpoints recorded by node index + name.
    assert_eq!((p.from_node, p.to_node), (a, b));
    assert_eq!(p.from, lv.nodes[a].name);
    assert_eq!(p.to, lv.nodes[b].name);
    // Path starts at A, ends at B, is contiguous (6-connected, |delta|==1 steps),
    // and is the x-then-y expansion (Manhattan length = dx+dy+1 cells).
    assert_eq!(p.path.first(), Some(&(1, 1, 0)));
    assert_eq!(p.path.last(), Some(&(4, 3, 0)));
    assert_eq!(p.path.len(), (3 + 2) + 1, "manhattan dx+dy+1 cells");
    for w in p.path.windows(2) {
        let d = (w[1].0 - w[0].0).abs() + (w[1].1 - w[0].1).abs() + (w[1].2 - w[0].2).abs();
        assert_eq!(d, 1, "each step is one face-adjacent cell");
    }
}

#[test]
fn push_pipe_between_refuses_self_and_duplicate() {
    let mut lv = Level {
        name: "p".into(),
        title: "p".into(),
        size: (16, 16, 4),
        ..Default::default()
    };
    let a = lv.push_node(1, 1, 0);
    let b = lv.push_node(4, 3, 0);
    // Self-pipe refused, level unchanged.
    assert!(lv.push_pipe_between(a, a).is_none());
    assert!(lv.pipes.is_empty());
    // First a->b ok.
    assert!(lv.push_pipe_between(a, b).is_some());
    // Duplicate a->b refused.
    assert!(lv.push_pipe_between(a, b).is_none());
    assert_eq!(lv.pipes.len(), 1);
    // Reverse b->a is a DISTINCT wire and allowed.
    assert!(lv.push_pipe_between(b, a).is_some());
    assert_eq!(lv.pipes.len(), 2);
}

#[test]
fn push_pipe_between_out_of_range_is_none() {
    let mut lv = Level {
        name: "p".into(),
        title: "p".into(),
        size: (8, 8, 2),
        ..Default::default()
    };
    let a = lv.push_node(1, 1, 0);
    assert!(lv.push_pipe_between(a, 99).is_none());
    assert!(lv.pipes.is_empty());
}

// ───────────────────────── node-delete index fixup (B2) ─────────────────────────

/// `remove_node` must shift/drop EVERY index-based node reference, not just the
/// `[hierarchy]` Vec. Deleting n1 (below the win-target n2) must leave the door
/// still targeting the SAME logical node (now at index 1), keep the zone_input and
/// pipe pointing at it, and DROP refs to the deleted node — never dangle.
#[test]
fn remove_node_fixes_up_all_index_references() {
    let mut lv = Level {
        name: "p".into(),
        title: "p".into(),
        size: (16, 16, 4),
        ..Default::default()
    };
    let n0 = lv.push_node(1, 1, 0);
    let n1 = lv.push_node(2, 1, 0);
    let n2 = lv.push_node(3, 1, 0);
    let n2_name = lv.nodes[n2].name.clone();

    // Door wins on n2 (index-based target), plus an inverted target on n1.
    lv.door.target_parts = vec![n2];
    lv.door.target_off = vec![n1];
    // A zone_input driving n2.
    lv.zone_inputs.push(ZoneInput {
        name: "z".into(),
        node: n2_name.clone(),
        node_index: n2,
        ..Default::default()
    });
    // A zone_input driving the soon-to-be-deleted n1 (must be dropped).
    lv.zone_inputs.push(ZoneInput {
        name: "zdead".into(),
        node: lv.nodes[n1].name.clone(),
        node_index: n1,
        ..Default::default()
    });
    // Pipe n0 -> n2 (survives, decremented), and pipe n0 -> n1 (dropped).
    let keep_pipe = lv.push_pipe_between(n0, n2).expect("pipe n0->n2");
    let _dead_pipe = lv.push_pipe_between(n0, n1).expect("pipe n0->n1");
    assert_eq!(lv.pipes.len(), 2);

    // Delete n1 (index 1, below the n2 targets).
    lv.remove_node(n1);

    // n2 shifted down to index 1; its name is preserved.
    assert_eq!(lv.nodes.len(), 2);
    let new_n2 = lv.node_index(&n2_name).expect("n2 still present");
    assert_eq!(new_n2, 1, "n2 shifted from index 2 to 1");

    // Door win target follows n2 to its new index (NOT the wrong node).
    assert_eq!(lv.door.target_parts, vec![new_n2], "door still targets n2");
    // target_off referenced the DELETED node ⇒ dropped (not dangling).
    assert!(
        lv.door.target_off.is_empty(),
        "inverted target on deleted node dropped"
    );

    // zone_input on n2 survives + re-indexed; the one on n1 is dropped.
    assert_eq!(lv.zone_inputs.len(), 1, "zone on deleted node dropped");
    assert_eq!(lv.zone_inputs[0].name, "z");
    assert_eq!(
        lv.zone_inputs[0].node_index, new_n2,
        "zone re-indexed to n2"
    );
    assert_eq!(lv.zone_inputs[0].node, n2_name, "zone node NAME unchanged");

    // Pipe n0->n2 survives with endpoints decremented; pipe to deleted node dropped.
    assert_eq!(lv.pipes.len(), 1, "pipe to deleted node dropped");
    let p = &lv.pipes[0];
    assert_eq!(p.from_node, n0, "n0 (< deleted idx) unchanged");
    assert_eq!(p.to_node, new_n2, "pipe endpoint follows n2");
    assert_eq!(p.from, lv.nodes[n0].name, "pipe from-name unchanged");
    assert_eq!(p.to, n2_name, "pipe to-name unchanged");
    // The kept pipe is the n0->n2 one (sanity: index 0 since the dead one was after).
    let _ = keep_pipe;
}

/// `remove_node` on an out-of-range index is a defined no-op (validate-before-mutate).
#[test]
fn remove_node_out_of_range_is_noop() {
    let mut lv = Level {
        name: "p".into(),
        title: "p".into(),
        size: (8, 8, 2),
        ..Default::default()
    };
    let _n0 = lv.push_node(1, 1, 0);
    lv.door.target_parts = vec![0];
    lv.remove_node(99);
    assert_eq!(lv.nodes.len(), 1, "no node removed");
    assert_eq!(lv.door.target_parts, vec![0], "references untouched");
}

// ---- [note stencil] — the verbatim FIXTURE placard -------------------------

/// `[note stencil]` parses, and its body reaches `LevelNote::text` with the drawing
/// INTACT: line breaks preserved, and the indentation that distinguishes one row of
/// the frame from the next still there.
///
/// This is the parse half of the stencil contract; the render half (dedent + verbatim
/// layout) is `re_render::note_style`. If the parser ever word-joined or trimmed these
/// lines the decal would arrive at the wall as rubble, and the render tests — which
/// build from a `&str` literal — would not notice.
#[test]
fn note_stencil_parses_and_keeps_its_drawing() {
    let src = "\
[meta]
name = stencil-fixture
[room]
size = 6 4 4
[note stencil pos=1,2,2 facing=-x]
    +--[E]--+
    |  ON   |
    +-------+
[node a]
pos = 1 1 1
is_input = yes
[node b]
pos = 4 1 1
is_output = yes
[pipe a->b]
path = (1,1,1) (4,1,1)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<stencil>").expect("stencil level parses");
    assert_eq!(lv.notes.len(), 1, "the stencil note was captured");
    let n = &lv.notes[0];
    assert_eq!(n.kind, NoteKind::Stencil);
    assert_eq!((n.x, n.y, n.z), (1, 2, 2));

    // Three body rows, in order, with the authored indent still attached — the
    // renderer's `dedent` is what removes it, precisely so the parser does not have
    // to guess whether leading space is layout or content.
    let rows: Vec<&str> = n.text.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(
        rows,
        vec!["    +--[E]--+", "    |  ON   |", "    +-------+"],
        "the drawing must survive parsing verbatim, indentation included"
    );

    // The `[E]` is mid-line, so the column-0 `[` body terminator must not eat it.
    assert!(
        n.text.contains("[E]"),
        "a bracket INSIDE a body line is placard text, not a section header"
    );
}

/// `[note banner]` is a KEYWORD ALIAS for `stencil`, not a new `NoteKind` — it parses
/// to `NoteKind::Stencil` exactly like `[note stencil]`, and its body reaches
/// `LevelNote::text` verbatim so `re_render::banner` can recognise its `--- banner ---`
/// marker at render time. See `parse_note_header`'s doc comment for why this stays an
/// alias rather than a real variant.
#[test]
fn note_banner_is_a_stencil_keyword_alias() {
    let src = "\
[meta]
name = banner-fixture
[room]
size = 6 4 4
[note banner pos=1,2,2 facing=-x]
    --- banner ---
    word = EXIT
[node a]
pos = 1 1 1
is_input = yes
[node b]
pos = 4 1 1
is_output = yes
[pipe a->b]
path = (1,1,1) (4,1,1)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<banner>").expect("banner note level parses");
    assert_eq!(lv.notes.len(), 1);
    let n = &lv.notes[0];
    assert_eq!(
        n.kind,
        NoteKind::Stencil,
        "banner is a stencil-kind alias, not a new kind"
    );
    assert!(n.text.contains("--- banner ---"));
    assert!(n.text.contains("word = EXIT"));
}

/// An unknown note kind is still rejected — adding `stencil` must not have turned
/// the kind field into "anything goes".
#[test]
fn note_unknown_kind_is_still_rejected() {
    let src = "\
[meta]
name = bad-kind
[room]
size = 6 4 4
[note wallpaper pos=1,2,2 facing=-x]
    nope
[node a]
pos = 1 1 1
is_input = yes
[node b]
pos = 4 1 1
is_output = yes
[pipe a->b]
path = (1,1,1) (4,1,1)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";
    let lv = load_string(src, "<bad-kind>").expect("level with an unknown note kind still parses");
    assert!(
        lv.notes.is_empty(),
        "an unrecognised note kind yields NO note (the pre-existing rule)"
    );
}

// ---- [room] storey_height (multi-floor vertical model, design §4) -----------

/// A minimal but complete level body WITHOUT a `storey_height` key. Tests inject
/// the key (or a bad value) by inserting a line after `[room]`. Kept small so the
/// byte-identical Map comparison is cheap and obvious.
const STOREY_BASE: &str = "\
[meta]
name = storey-fixture
[room]
size = 6 4 9
[node a]
pos = 1 1 1
is_input = yes
[node b]
pos = 4 1 1
is_output = yes
[pipe a->b]
path = (1,1,1) (4,1,1)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
";

/// Insert `line` immediately after the `[room]` header so the key lands inside
/// the room section. Returns the modified source.
fn with_room_line(line: &str) -> String {
    STOREY_BASE.replacen("[room]\n", &format!("[room]\n{line}\n"), 1)
}

/// A present, valid `storey_height` parses to exactly the stored value, and the
/// accessor reports the same.
#[test]
fn storey_height_present_parses() {
    let src = with_room_line("storey_height = 6");
    let lv = load_string(&src, "<storey-present>").expect("level with storey_height must parse");
    assert_eq!(lv.storey_height, 6, "raw field is the authored value");
    assert_eq!(lv.storey_height(), 6, "accessor agrees with the raw field");
}

/// An ABSENT `storey_height` normalises to the default (4) on a LOADED level, and
/// the accessor reports 4 as well.
#[test]
fn storey_height_absent_defaults_to_four() {
    assert_eq!(DEFAULT_STOREY_HEIGHT, 4, "the default is 4 per the spec");
    let lv =
        load_string(STOREY_BASE, "<storey-absent>").expect("level without storey_height parses");
    assert_eq!(
        lv.storey_height, DEFAULT_STOREY_HEIGHT,
        "finalize normalises an absent key to the default"
    );
    assert_eq!(lv.storey_height(), DEFAULT_STOREY_HEIGHT);
}

/// The accessor maps the unset (`0`) sentinel of a programmatically-built level to
/// the default, even though `finalize` never ran.
#[test]
fn storey_height_accessor_maps_zero_to_default() {
    let lv = Level::default();
    assert_eq!(
        lv.storey_height, 0,
        "default-built level leaves the raw field unset"
    );
    assert_eq!(
        lv.storey_height(),
        DEFAULT_STOREY_HEIGHT,
        "accessor maps the 0 sentinel to the default"
    );
}

/// `storey_height = 0` (and any value < 1) is a HARD parse error — never silently
/// clamped to the default.
#[test]
fn storey_height_zero_is_error() {
    let src = with_room_line("storey_height = 0");
    let err = load_string(&src, "<storey-zero>").unwrap_err();
    assert!(
        err.msg.contains("storey_height") && err.msg.contains(">= 1"),
        "expected a >= 1 rejection, got: {}",
        err.msg
    );
}

/// A negative `storey_height` is rejected with the same `>= 1` error.
#[test]
fn storey_height_negative_is_error() {
    let src = with_room_line("storey_height = -3");
    let err = load_string(&src, "<storey-neg>").unwrap_err();
    assert!(
        err.msg.contains(">= 1"),
        "expected a >= 1 rejection, got: {}",
        err.msg
    );
}

/// A non-numeric `storey_height` is a hard error (not silently ignored), so a typo
/// can never leave the level quietly on the default.
#[test]
fn storey_height_non_numeric_is_error() {
    let src = with_room_line("storey_height = tall");
    let err = load_string(&src, "<storey-nan>").unwrap_err();
    assert!(
        err.msg.contains("storey_height") && err.msg.contains("integer"),
        "expected an integer-parse error, got: {}",
        err.msg
    );
}

// ---- [room] power_budget (the simultaneously-ON pipe cap, v0.2 re-scope) ----
//
// Parse-side spec only. The RULE (refuse an ON-switch at the cap) lives in
// `re_puzzle`; here we only prove the number reaches `Level` intact, that an
// absent key means UNLIMITED, and that a malformed one is loud.

/// A present, non-zero `power_budget` parses to exactly the authored value.
#[test]
fn power_budget_present_parses() {
    let lv = load_string(&with_room_line("power_budget = 3"), "<pb-present>")
        .expect("level with power_budget must parse");
    assert_eq!(lv.power_budget, 3);
}

/// An ABSENT `power_budget` is `0` — the UNLIMITED sentinel. This is the
/// back-compat guarantee: every level authored before the rule existed keeps
/// behaving exactly as it did, so no golden can move.
#[test]
fn power_budget_absent_means_unlimited() {
    let lv = load_string(STOREY_BASE, "<pb-absent>").expect("level without power_budget parses");
    assert_eq!(lv.power_budget, 0, "absent ⇒ 0 ⇒ unlimited");
    assert_eq!(Level::default().power_budget, 0, "and so does a bare Level");
}

/// An EXPLICIT `power_budget = 0` is legal and means the same thing as absent
/// (unlimited) — never "a zero cap", which would ship an unplayable level.
#[test]
fn power_budget_zero_is_explicit_unlimited() {
    let lv = load_string(&with_room_line("power_budget = 0"), "<pb-zero>").expect("0 parses");
    assert_eq!(lv.power_budget, 0);
}

/// A negative `power_budget` is a HARD parse error.
#[test]
fn power_budget_negative_is_error() {
    let err = load_string(&with_room_line("power_budget = -1"), "<pb-neg>").unwrap_err();
    assert!(
        err.msg.contains("power_budget") && err.msg.contains(">= 0"),
        "expected a >= 0 rejection, got: {}",
        err.msg
    );
}

/// A non-numeric `power_budget` is a HARD parse error, never a silent fall-back
/// to unlimited — a typo must not quietly delete the level's only constraint.
#[test]
fn power_budget_non_numeric_is_error() {
    let err = load_string(&with_room_line("power_budget = lots"), "<pb-nan>").unwrap_err();
    assert!(
        err.msg.contains("power_budget") && err.msg.contains("integer"),
        "expected an integer-parse error, got: {}",
        err.msg
    );
}

/// `power_budget` is PUZZLE metadata: it must not touch geometry. A level with
/// no key builds a Map cell-for-cell identical to the same level with one.
#[test]
fn power_budget_does_not_change_the_map() {
    let a = load_string(STOREY_BASE, "<pb-map-absent>").expect("absent parses");
    let b = load_string(&with_room_line("power_budget = 2"), "<pb-map-present>")
        .expect("present parses");
    let (ma, mb) = (a.build_map(), b.build_map());
    assert_eq!(ma.size(), mb.size());
    let (sx, sy, sz) = ma.size();
    for z in 0..sz as i32 {
        for y in 0..sy as i32 {
            for x in 0..sx as i32 {
                assert_eq!(
                    ma.cell_at(x, y, z),
                    mb.cell_at(x, y, z),
                    "cell ({x},{y},{z})"
                );
                assert_eq!(
                    ma.side_data_at(x, y, z),
                    mb.side_data_at(x, y, z),
                    "side_data ({x},{y},{z})"
                );
            }
        }
    }
}

// `campaign_levels_declare_a_budget_and_nothing_else_does` MOVED to the game
// repo's `campaign_lint.rs`: it asserted a property OF THE CAMPAIGN ("exactly
// these twelve carry a power_budget, and no other shipped level does"), which
// only worked by coincidence while campaign and engine levels shared one
// `levels/` directory. Now that the campaign `L*.lvl` files live in the game
// repo and not in `terminal3d/levels/`, this test's own premise ("scan
// levels/, find exactly the twelve") can never hold here — see
// `campaign_levels_declare_a_budget_and_nothing_else_does` in the game repo.

/// `storey_height` is METADATA ONLY: a level with NO key builds a Map that is
/// cell-for-cell identical to the SAME level with an explicit `storey_height`.
/// This proves the key never changes grid storage, dimensions, kinds, side_data,
/// or shapes — the back-compat guarantee from the spec.
#[test]
fn storey_height_does_not_change_the_map() {
    let lv_absent = load_string(STOREY_BASE, "<map-absent>").expect("absent parses");
    let lv_present =
        load_string(&with_room_line("storey_height = 3"), "<map-present>").expect("present parses");

    let map_a = lv_absent.build_map();
    let map_p = lv_present.build_map();

    assert_eq!(
        map_a.size(),
        map_p.size(),
        "grid dimensions are unchanged by storey_height"
    );
    let (sx, sy, sz) = map_a.size();
    for z in 0..sz as i32 {
        for y in 0..sy as i32 {
            for x in 0..sx as i32 {
                assert_eq!(
                    map_a.cell_at(x, y, z),
                    map_p.cell_at(x, y, z),
                    "cell kind differs at ({x},{y},{z})"
                );
                assert_eq!(
                    map_a.side_data_at(x, y, z),
                    map_p.side_data_at(x, y, z),
                    "side_data differs at ({x},{y},{z})"
                );
                assert_eq!(
                    map_a.shape_at(x, y, z),
                    map_p.shape_at(x, y, z),
                    "shape differs at ({x},{y},{z})"
                );
            }
        }
    }
}

/// A SHIPPED level (01.lvl) with no `storey_height` reports the default and builds
/// the SAME Map whether or not an explicit default key is injected — the real-file
/// regression guard.
#[test]
fn storey_height_shipped_level_byte_identical() {
    // Normalize to LF before the textual injection below. The repo file is LF, but a
    // Windows checkout with `core.autocrlf=true` materializes CRLF, and then
    // `replacen("[room]\n", …)` matches NOTHING — the injection silently becomes a
    // no-op and the `assert_ne!` guard (correctly) fires. The parser itself is CRLF
    // tolerant; only this test's own string surgery is not. Same normalization
    // `ts_editor`'s byte-identity round-trip test already applies for the same reason.
    let raw = std::fs::read_to_string(level_path("01.lvl"))
        .expect("read 01.lvl")
        .replace("\r\n", "\n");

    let lv_plain = load_string(&raw, "<01-plain>").expect("01 parses");
    assert_eq!(
        lv_plain.storey_height(),
        DEFAULT_STOREY_HEIGHT,
        "a shipped single-storey level reports the default height"
    );

    // Inject an explicit `storey_height = 4` after the [room] header.
    let injected = raw.replacen("[room]\n", "[room]\nstorey_height = 4\n", 1);
    assert_ne!(injected, raw, "[room] header must exist to inject after");
    let lv_inj = load_string(&injected, "<01-injected>").expect("injected 01 parses");

    let map_plain = lv_plain.build_map();
    let map_inj = lv_inj.build_map();
    assert_eq!(map_plain.size(), map_inj.size());
    let (sx, sy, sz) = map_plain.size();
    for z in 0..sz as i32 {
        for y in 0..sy as i32 {
            for x in 0..sx as i32 {
                assert_eq!(map_plain.cell_at(x, y, z), map_inj.cell_at(x, y, z));
                assert_eq!(
                    map_plain.side_data_at(x, y, z),
                    map_inj.side_data_at(x, y, z)
                );
                assert_eq!(map_plain.shape_at(x, y, z), map_inj.shape_at(x, y, z));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// [sky] section — image-cubemap paths.
// ---------------------------------------------------------------------------

/// Minimal level text used as a base for sky tests: a small room that meets
/// the `finalize` requirements (pedestal, door, player spawn, two nodes, pipe)
/// without any sky section. Exactly the pattern the texture tests use.
fn sky_base_level() -> String {
    "\
[meta]
name = sky_test
[room]
size = 6 4 4
ambient = solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
"
    .to_string()
}

/// PARSE: a `[sky]` section with all three keys populates `Level.sky` correctly.
#[test]
fn sky_section_all_keys_parse() {
    let src = format!(
        "{}[sky]\ncylinder = sky/horizon.png\ncap_top = sky/top.png\ncap_bottom = sky/bottom.png\n",
        sky_base_level()
    );
    let lv = load_string(&src, "<sky-all>").expect("[sky] with all keys must parse");
    assert_eq!(
        lv.sky.cylinder.as_deref(),
        Some("sky/horizon.png"),
        "cylinder path"
    );
    assert_eq!(
        lv.sky.cap_top.as_deref(),
        Some("sky/top.png"),
        "cap_top path"
    );
    assert_eq!(
        lv.sky.cap_bottom.as_deref(),
        Some("sky/bottom.png"),
        "cap_bottom path"
    );
}

/// PARSE: a `[sky]` section with only `cylinder` and `cap_top` leaves
/// `cap_bottom` as `None`.
#[test]
fn sky_section_partial_keys_parse() {
    let src = format!(
        "{}[sky]\ncylinder = sky/h.png\ncap_top = sky/t.png\n",
        sky_base_level()
    );
    let lv = load_string(&src, "<sky-partial>").expect("[sky] partial must parse");
    assert_eq!(lv.sky.cylinder.as_deref(), Some("sky/h.png"));
    assert_eq!(lv.sky.cap_top.as_deref(), Some("sky/t.png"));
    assert!(
        lv.sky.cap_bottom.is_none(),
        "cap_bottom must be None when absent"
    );
}

/// PARSE: a level with NO `[sky]` section leaves all sky paths as `None`
/// (additive, byte-identical to pre-sky levels).
#[test]
fn sky_section_absent_leaves_all_none() {
    let lv = load_string(&sky_base_level(), "<sky-none>").expect("base level must parse");
    assert!(lv.sky.is_empty(), "no [sky] section ⇒ SkyDef must be empty");
    assert!(lv.sky.cylinder.is_none());
    assert!(lv.sky.cap_top.is_none());
    assert!(lv.sky.cap_bottom.is_none());
}

/// BUILD: decoding a real PNG for the cylinder slot populates `Map.sky_images`.
/// Uses the same `write_tiny_red_png` fixture the texture tests use.
#[test]
fn build_map_decodes_sky_cylinder() {
    let png = write_tiny_red_png();
    let png_str = png.to_string_lossy().replace('\\', "/");
    let src = format!("{}[sky]\ncylinder = {png_str}\n", sky_base_level());
    let lv = load_string(&src, "<sky-build>").expect("level with [sky] must parse");
    let map = lv.build_map();
    let sky = map.sky_images();
    assert!(
        sky.cylinder.is_some(),
        "cylinder slot must be populated after decode"
    );
    let stored = sky.cylinder.as_ref().unwrap();
    assert_eq!((stored.width, stored.height), (2, 2));
    // Red fixture: first pixel R=255, G=0, B=0, A=255.
    assert_eq!(stored.rgba[0], 255, "R channel of decoded cylinder");
    assert_eq!(stored.rgba[1], 0, "G channel of decoded cylinder");
    assert!(
        sky.cap_top.is_none(),
        "cap_top must stay None when not authored"
    );
    assert!(
        sky.cap_bottom.is_none(),
        "cap_bottom must stay None when not authored"
    );
    let _ = std::fs::remove_file(&png);
}

/// BUILD: a missing `[sky]` file is SKIPPED — the map still builds with the slot
/// at `None` (no panic). `build_map_checked` surfaces the error.
#[test]
fn build_map_missing_sky_file_is_nonfatal() {
    let src = format!(
        "{}[sky]\ncylinder = /no/such/sky_definitely_missing.png\n",
        sky_base_level()
    );
    let lv = load_string(&src, "<sky-miss>").expect("level parses even with bad sky path");
    // best-effort: map builds, sky slot stays None.
    let map = lv.build_map();
    assert!(
        map.sky_images().cylinder.is_none(),
        "missing file ⇒ slot stays None"
    );
    // checked: error surfaced.
    let err = lv.build_map_checked().unwrap_err();
    assert!(
        err.contains("[sky]") && err.contains("cylinder"),
        "error must mention the slot: {err}"
    );
}

/// BUILD: a level with no `[sky]` section keeps all sky slots at `None` even
/// after `build_map` — procedural sky is the unchanged default.
#[test]
fn build_map_no_sky_section_all_slots_none() {
    let lv = load_string(&sky_base_level(), "<sky-def>").expect("base level parses");
    let map = lv.build_map();
    let sky = map.sky_images();
    assert!(
        sky.cylinder.is_none(),
        "cylinder must be None — no [sky] section"
    );
    assert!(
        sky.cap_top.is_none(),
        "cap_top must be None — no [sky] section"
    );
    assert!(
        sky.cap_bottom.is_none(),
        "cap_bottom must be None — no [sky] section"
    );
}

// ── D1: level-file `#` comments must never reach a lore placard ──────────────

/// A `[note]` body runs "verbatim until the next header", so every dev comment an
/// author wrote between the placard and the next section used to be captured into
/// the note and PRINTED ON THE WALL. Column-0 `#` lines are level-file comments and
/// are dropped; the indented body — including a MID-LINE `#` — survives untouched.
#[test]
fn note_body_drops_column_zero_comments_and_keeps_mid_line_hash() {
    let src = "\
[meta]
name = note-comment-fixture

[room]
size = 6 6 3

[player]
spawn = 2 2 1

[note system_log pos=1,2,1 facing=+x]
  MINISTRY CORPUS #9 — retained.
  the lattice remembers {time}.

# ---- nodes ----------------------------------------------------------------
# this is dev commentary, not placard text ({time} must NOT interpolate)

[node a]
pos = 2,2,1
is_input = yes

[node b]
pos = 4,2,1
is_output = yes

[pipe a->b]
state = OFF
path = (2,2,1) (4,2,1)

[pedestal]
center = 4 2 1
volume = 1 1 1

[door]
pos = 5 3 1
";
    let lv = load_string(src, "<note-comment>").expect("fixture parses");
    assert_eq!(lv.notes.len(), 1, "one placard");
    let body = &lv.notes[0].text;
    assert!(
        !body.lines().any(|l| l.starts_with('#')),
        "a column-0 `#` comment leaked into the placard body:\n{body}"
    );
    assert!(
        !body.contains("dev commentary"),
        "trailing dev comments must not reach the wall:\n{body}"
    );
    assert!(
        body.contains("MINISTRY CORPUS #9"),
        "a MID-LINE `#` must survive verbatim:\n{body}"
    );
    assert!(
        body.contains("the lattice remembers {time}."),
        "authored body lines must survive verbatim:\n{body}"
    );
}

/// The shipped campaign + numbered levels carry ZERO leaked comment lines in any
/// placard body. This is the corpus-wide form of a defect once found in review
/// (55 leaked lines across 8 levels).
#[test]
fn no_shipped_level_leaks_a_comment_into_a_placard() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../levels");
    let mut leaked: Vec<String> = Vec::new();
    let mut with_notes = 0usize;
    for entry in std::fs::read_dir(&dir).expect("levels/ readable") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("lvl") {
            continue;
        }
        let Ok(lv) = load(&path) else { continue };
        if !lv.notes.is_empty() {
            with_notes += 1;
        }
        for note in &lv.notes {
            for line in note.text.lines().filter(|l| l.starts_with('#')) {
                leaked.push(format!("{}: {line}", path.display()));
            }
        }
    }
    assert!(
        with_notes > 0,
        "fixture assumption: some level ships placards"
    );
    assert!(
        leaked.is_empty(),
        "{} comment line(s) leaked onto placards:\n{}",
        leaked.len(),
        leaked.join("\n")
    );
}

// ── D4: `[effects] snow` reaches the renderer as ambient emitters ────────────

/// A minimal `[room]`-built cold-level fixture: a plain sealed box carrying only
/// the `[effects] snow` density under test, no props, no other authored
/// content. Used by [`effects_snow_synthesizes_ambient_emitters`].
fn cold_fixture(size: (u32, u32, u32), density: f32) -> String {
    let (sx, sy, sz) = size;
    format!(
        "[meta]\nname = cold-fixture\n\n\
         [room]\nsize = {sx} {sy} {sz}\nambient = solid\n\n\
         [node source]\npos = 2,2,1\nis_input = yes\n\n\
         [node sink]\npos = 5,2,1\nis_output = yes\nthreshold = 1\n\n\
         [pipe source->sink]\npath = (2,2,1) (3,2,1) (4,2,1) (5,2,1)\nswitch_at = (3,2,1)\n\n\
         [effects]\nsnow = {density}\n\n\
         [inputs]\nsource = 1\n\n\
         [pedestal]\ncenter = 5 2 1\nvolume = 1 1 1\n\n\
         [door]\npos = 6 2 1\n\n\
         [player]\nspawn = {} {} 0\nfacing = -y\n",
        sx / 2,
        sy / 2,
    )
}

/// A COLD level's `snow` knob must produce real, valid ambient emitters — it was
/// parsed, round-tripped and editable while NOTHING read it, so a cold level rendered
/// exactly like a warm one.
///
/// Originally exercised via the campaign's `levels/L3-full-spectrum.lvl` and
/// `levels/L8-cold-room.lvl` ("snow in L8"). Re-pointed to two inline fixtures
/// carrying the SAME densities and comparable room footprints — an engine
/// test should not depend on shipped campaign content, which lives with the
/// game, not the engine.
#[test]
fn effects_snow_synthesizes_ambient_emitters() {
    for (name, size, density) in [
        ("cold-fixture-a", (18u32, 16u32, 9u32), 0.4_f32),
        ("cold-fixture-b", (20, 16, 5), 0.35),
    ] {
        let src = cold_fixture(size, density);
        let lv = load_string(&src, name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            (lv.effects.snow - density).abs() < 1e-4,
            "{name}: fixture assumption — snow = {density}, got {}",
            lv.effects.snow
        );
        let synth = snow_emitters_of(&lv);
        assert!(
            !synth.is_empty(),
            "{name}: snow = {density} must synthesize emitters, got none"
        );
        assert!(
            synth.len() <= 48,
            "{name}: the snow lattice is capped, got {}",
            synth.len()
        );
        let (sx, sy, sz) = lv.size;
        for e in synth {
            assert!(
                e.is_valid(),
                "{name}: a synthesized snow emitter must validate"
            );
            assert!(e.rate > 0.0, "{name}: snow must actually emit");
            assert!(
                e.accel.z < 0.0 && e.direction.z < 0.0,
                "{name}: snow falls DOWN"
            );
            let re_world::EmitterOrigin::World(p) = e.origin else {
                panic!("{name}: snow is world-space (so walls occlude it)");
            };
            assert!(
                p.x > 1.0
                    && p.x < (sx - 1) as f32
                    && p.y > 1.0
                    && p.y < (sy - 1) as f32
                    && p.z > 1.0
                    && p.z < (sz - 1) as f32,
                "{name}: a snow emitter was born inside the shell at {p:?} (room {sx}×{sy}×{sz})"
            );
        }
        // Deterministic: two loads of the same source agree exactly.
        let again = load_string(&src, name).expect("reload");
        assert_eq!(
            again.emitters, lv.emitters,
            "{name}: snow must be deterministic"
        );
    }
}

/// The SNOW share of a level's synthesized emitters.
///
/// `synthesized` is not a snow marker: since the prop kit (set-dressing 2026-09-24) a
/// `[prop]` with an emitter folds its spec into `Level::emitters` flagged
/// `synthesized` too (so the writer never serializes it back). L8's `cold_vent`-side
/// prop and L4's props now contribute such specs, which made "every synthesized
/// emitter is snow" false — L8 counted 48 snow + 1 prop = 49 against the 48 cap, and
/// warm L4 was no longer "nothing synthesized". The prop specs are appended BEFORE
/// the snow lattice, in `expand_level` order, so they are exactly the synthesized
/// entries a fresh prop expansion yields; the rest is snow.
fn snow_emitters_of(lv: &Level) -> Vec<&re_world::EmitterSpec> {
    let prop_specs: Vec<re_world::EmitterSpec> = if lv.props.is_empty() {
        Vec::new()
    } else {
        props::expand_level(lv, &lv.build_map_base()).0.emitters
    };
    let synth: Vec<&re_world::EmitterSpec> = lv.emitters.iter().filter(|e| e.synthesized).collect();
    assert!(
        synth.len() >= prop_specs.len(),
        "every prop emitter is folded into the level"
    );
    for (got, want) in synth.iter().zip(&prop_specs) {
        assert_eq!(
            *got, want,
            "the prop emitters come first, in expansion order"
        );
    }
    synth[prop_specs.len()..].to_vec()
}

/// A WARM level (no `snow`) synthesizes no snow; the only synthesized emitters it may
/// carry are its props' (see [`snow_emitters_of`]).
///
/// Originally `levels/L4-the-angle.lvl` (a campaign level). Re-pointed to the
/// engine archive's `01.lvl` — an engine test should not depend on shipped
/// campaign content — keeping the same fixture assumption (no `[effects]` at
/// all ⇒ warm, no props).
#[test]
fn no_snow_synthesizes_no_emitters() {
    let lv = load(level_path("01.lvl")).expect("01.lvl loads");
    assert_eq!(
        lv.effects.snow, 0.0,
        "fixture assumption: 01.lvl is a WARM level"
    );
    assert!(
        snow_emitters_of(&lv).is_empty(),
        "no snow ⇒ no snow emitters"
    );
    // And a prop-free warm level synthesizes nothing at all.
    let bare = load_string(ZONE_FILL_BASE, "<warm>").expect("parses");
    assert!(
        bare.emitters.iter().all(|e| !e.synthesized),
        "no snow, no props ⇒ nothing synthesized"
    );
}

/// The density SCALES the field: a denser knob emits faster.
#[test]
fn snow_density_scales_the_emission_rate() {
    let base = |snow: f32| -> f32 {
        let src = format!(
            "[meta]\nname = snow-scale\n\n[room]\nsize = 10 10 4\n\n[player]\nspawn = 5 5 1\n\n\
             [node a]\npos = 2,2,1\nis_input = yes\n\n[node b]\npos = 6,2,1\nis_output = yes\n\n\
             [pipe a->b]\nstate = OFF\npath = (2,2,1) (6,2,1)\n\n\
             [pedestal]\ncenter = 6 2 1\nvolume = 1 1 1\n\n[door]\npos = 9 5 1\n\n\
             [effects]\nsnow = {snow}\n"
        );
        let lv = load_string(&src, "<snow>").expect("parses");
        lv.emitters
            .iter()
            .filter(|e| e.synthesized)
            .map(|e| e.rate)
            .sum()
    };
    let light = base(0.25);
    let heavy = base(1.0);
    assert!(light > 0.0, "0.25 snow must emit");
    assert!(
        heavy > light * 3.0,
        "density must scale emission (0.25 ⇒ {light}/s, 1.0 ⇒ {heavy}/s)"
    );
}

/// `[effects]` INTENSITY knobs accept the boolean KEYWORDS their sibling `steam`
/// accepts, instead of silently collapsing to `0.0`.
///
/// Regression: `snow`/`matrix`/`glitch` parsed with a bare
/// `parse::<f32>().unwrap_or(0.0)`, while `steam` right beside them in the same
/// match used `is_truthy`. The campaign's `levels/L9-stereopsis.lvl` shipped
/// `snow = on` and got `0.0` — the authored effect never rendered, and nothing
/// failed to say so.
#[test]
fn effects_intensity_accepts_keywords_as_well_as_floats() {
    let case = |line: &str| {
        let src = format!(
            "[meta]\nname = fx\n[room]\nsize = 8 8 4\n\
             [pedestal]\ncenter = 4 4 1\nvolume = 1 1 1\n\
             [door]\npos = 1 4 1\n[player]\nspawn = 6 4 1\n[effects]\n{line}\n"
        );
        load_string(&src, "fx.lvl").expect("level parses").effects
    };

    // The affirmative keywords mean FULL intensity, matching `is_truthy`.
    for word in ["on", "ON", "yes", "true", "TRUE"] {
        assert_eq!(
            case(&format!("snow = {word}")).snow,
            1.0,
            "`snow = {word}` must read as full intensity, not 0.0"
        );
    }
    // The negative ones stay off — as does anything unparsable.
    for word in ["off", "false", "no", "garbage"] {
        assert_eq!(
            case(&format!("snow = {word}")).snow,
            0.0,
            "`snow = {word}` must read as off"
        );
    }
    // A numeric value still wins and is still clamped to 0..=1.
    assert_eq!(case("matrix = 0.35").matrix, 0.35);
    assert_eq!(case("glitch = 4.0").glitch, 1.0, "out-of-range clamps high");
    assert_eq!(case("glitch = -2.0").glitch, 0.0, "out-of-range clamps low");
    // NaN/inf must not leak into the render path.
    assert_eq!(
        case("snow = NaN").snow,
        0.0,
        "NaN must not reach the renderer"
    );
    assert_eq!(
        case("snow = inf").snow,
        0.0,
        "inf must not reach the renderer"
    );

    // The regression's own shape, through the real `load_string` pipeline (not
    // just the `case` closure's mini-effects-only parse): a level authoring
    // exactly the keyword the campaign's `levels/L9-stereopsis.lvl` originally
    // shipped (`snow = on`) must actually carry snow after a full load.
    let motivated = load_string(
        "[meta]\nname = stereopsis-fixture\n[room]\nsize = 8 8 4\n\
         [pedestal]\ncenter = 4 4 1\nvolume = 1 1 1\n\
         [door]\npos = 1 4 1\n[player]\nspawn = 6 4 1\n[effects]\nsnow = on\n",
        "<stereopsis-fixture>",
    )
    .expect("fixture parses");
    assert!(
        motivated.effects.snow > 0.0,
        "`snow = on` must render as snow — the bug this regression covers"
    );
}

// ---------------------------------------------------------------------------
// strip_comment: the lone-`#` value exception
// ---------------------------------------------------------------------------

/// A `#` that is the last non-whitespace character of a line whose value would
/// otherwise be EMPTY is a VALUE, not a comment.
///
/// `levels/12-mirror.lvl` authors `[material green_matrix] glyph = #`. The old
/// rule read that `#` as a comment and reduced the line to `glyph       =`, so
/// the glyph silently became `None` and the material rendered with the default.
/// An empty comment carries no information, so reading one could only ever
/// destroy data — which is the whole justification for the exception.
#[test]
fn lone_trailing_hash_is_a_value_not_a_comment() {
    // THE BUG: an empty value is never what anyone wrote.
    assert_eq!(strip_comment("glyph       = #"), "glyph       = #");
    assert_eq!(strip_comment("glyph = #"), "glyph = #");
    assert_eq!(strip_comment("glyph = #   "), "glyph = #   ");

    // UNCHANGED — the exception needs BOTH conditions, so its blast radius is
    // zero for every other shape a `#` can take.
    assert_eq!(
        strip_comment("pos = 1,2,3 #"),
        "pos = 1,2,3",
        "non-empty value ⇒ comment"
    );
    assert_eq!(
        strip_comment("speed = 2 # fast"),
        "speed = 2",
        "comment with a body"
    );
    assert_eq!(
        strip_comment("glyph = # yes"),
        "glyph =",
        "empty value but a real comment body"
    );
    assert_eq!(
        strip_comment("node 0 -> node 1 #"),
        "node 0 -> node 1",
        "no `=` ⇒ comment"
    );
    assert_eq!(strip_comment("# whole line"), "", "full-line comment");
    assert_eq!(
        strip_comment("color = #FF0000"),
        "color = #FF0000",
        "hex colour survives"
    );
    assert_eq!(
        strip_comment("color = #FF0000 # warm"),
        "color = #FF0000",
        "hex colour plus a real trailing comment"
    );
}

/// End-to-end on the shipped file: `12-mirror.lvl`'s `green_matrix` really does
/// come back with the `#` glyph now, not `None`.
#[test]
fn shipped_material_hash_glyph_survives_the_parse() {
    let lv = load(level_path("12-mirror.lvl")).expect("12-mirror.lvl must parse");
    let m = lv
        .materials
        .iter()
        .find(|m| m.name == "green_matrix")
        .expect("12-mirror carries [material green_matrix]");
    assert_eq!(
        m.glyph,
        Some(u32::from('#')),
        "the authored `glyph = #` reaches the material"
    );
}

// ---------------------------------------------------------------------------
// Unknown-key diagnostics
// ---------------------------------------------------------------------------

/// Every `.lvl` under `levels/`, sorted, as `(file name, path)`.
fn shipped_level_paths() -> Vec<(String, PathBuf)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../levels");
    let mut v: Vec<(String, PathBuf)> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("lvl"))
        .map(|p| (p.file_name().unwrap().to_string_lossy().to_string(), p))
        .collect();
    v.sort();
    v
}

/// Every unknown-key/section finding across the shipped corpus, as
/// `"<file>:<line>: [<section>] unknown key '<key>'"`, in file order.
fn corpus_unknown_findings() -> (usize, Vec<String>) {
    let mut checked = 0usize;
    let mut findings = Vec::new();
    for (name, path) in shipped_level_paths() {
        let Ok(lv) = load(&path) else { continue };
        checked += 1;
        for u in &lv.unknown_keys {
            findings.push(format!("{name}:{u}"));
        }
    }
    (checked, findings)
}

/// EVERY shipped level parses with ZERO unknown KEYS.
///
/// This is the test that actually enforces the diagnostic. [`UnknownKey`] is
/// deliberately not an error (a stray key must never make a hand-authored level
/// unloadable), so without a corpus assertion the collection would just be a
/// quieter place for the same silence.
///
/// A failure here is one of exactly two things and both want a human:
/// * a TYPO — the author wrote a key the parser has never had, and it has been
///   doing nothing ever since; or
/// * a key that is real, correctly spelled, and NOT WIRED UP — the
///   "authored, parsed, tested, never on screen" defect this project keeps
///   re-hitting.
///
/// Do not silence an entry to make this green. Either wire the key up or fix the
/// level. (Unknown SECTIONS are held separately — see
/// [`unknown_section_findings_are_a_registered_defect_list`].)
#[test]
fn no_shipped_level_has_an_unknown_key() {
    let (checked, findings) = corpus_unknown_findings();
    let keys: Vec<&String> = findings.iter().filter(|f| !f.contains("[?]")).collect();
    assert!(
        checked >= 60,
        "expected the ~63-level shipped corpus, only found {checked}"
    );
    assert!(
        keys.is_empty(),
        "{} unknown key(s) across the shipped corpus — each is either a typo that \
         has never done anything, or a real key nobody wired up:\n{}",
        keys.len(),
        keys.iter()
            .map(|s| format!("  {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The unknown-SECTION findings across the corpus, as a DEFECT REGISTER pinned to
/// its exact contents.
///
/// This is not an exemption list and it does not silence anything: it fails if a
/// finding is ADDED **and** if one is REMOVED, so the single entry below cannot be
/// forgotten and cannot be quietly grown.
///
/// ## It is EMPTY, and that emptiness was earned
///
/// The register opened with one entry: `levels/L7-hidden-depths.lvl:369`, a
/// `[note]` BODY line reading `    [ STEWARDSHIP - DO NOT POWER DOWN ]`.
/// [`Parser::run`] ended a `[note]` body at ANY line that looked like a section
/// header, so this authored placard line terminated its note, was dropped from the
/// note text, and opened a bogus section named `STEWARDSHIP`. No player had ever
/// seen it — the same "authored, parsed, never on screen" defect as the rest of
/// the class, except eaten by the grammar rather than by the renderer.
///
/// It is fixed at the grammar: inside a `[note]` body a `[…]` line ends the note
/// only in COLUMN 0 ([`note_body_line_is_header`]), mirroring the column-0 `#`
/// rule the body already had, with [`escape_note_body_line`] escaping both markers
/// and `ts_editor`'s capture/re-attach mirroring it. This assertion going from
/// one entry to zero is what proves the fix landed.
#[test]
fn unknown_section_findings_are_a_registered_defect_list() {
    const REGISTERED: &[&str] = &[];

    let (checked, findings) = corpus_unknown_findings();
    assert!(
        checked >= 60,
        "expected the ~63-level shipped corpus, only found {checked}"
    );
    let sections: Vec<String> = findings.into_iter().filter(|f| f.contains("[?]")).collect();
    assert_eq!(
        sections, REGISTERED,
        "the unknown-SECTION register changed. A NEW entry is a fresh defect — a \
         mistyped section header that silently does nothing. A MISSING entry means \
         someone fixed one: delete it from REGISTERED in the same commit."
    );
}

/// The diagnostic fires, is NOT an error, and names the section, line and key.
#[test]
fn unknown_keys_are_collected_not_rejected() {
    let src = "\
[meta]
name = uk-test
verzion = 1

[room]
size = 6 6 4
celing = dim

[node a]
pos = 1,2,1
is_input = yes
colour = #FF0000

[node b]
pos = 4,2,1
is_output = yes

[pipe a->b]
state = ON
path = (1,2,1) (4,2,1)

[pedestal]
center = 3 3 1
volume = 2 2 2

[door]
pos = 0 3 1

[player]
spawn = 3 3 0
";
    let lv = load_string(src, "<uk>").expect("an unknown key must NOT fail the load");
    // The level itself parsed correctly — the diagnostics are additive.
    assert_eq!(lv.name, "uk-test");
    assert_eq!(lv.size, (6, 6, 4));
    assert_eq!(lv.nodes.len(), 2);

    let got: Vec<String> = lv.unknown_keys.iter().map(ToString::to_string).collect();
    assert_eq!(
        got,
        vec![
            "line 3: [meta] unknown key 'verzion'".to_string(),
            "line 7: [room] unknown key 'celing'".into(),
            "line 12: [node] unknown key 'colour'".into(),
        ],
        "in file order, with section + line + key"
    );
}

/// The five INERT authoring keys and `[room] floor`/`ceiling` are now STORED, so
/// they must not be reported. A diagnostic that fired on a key the parser does
/// keep would train everyone to ignore the report.
#[test]
fn stored_authoring_keys_are_not_reported_as_unknown() {
    let lv = load(level_path("00-test-load.lvl")).expect("00-test-load parses");
    assert!(
        lv.unknown_keys.is_empty(),
        "00-test-load carries all five inert keys and must report none: {:?}",
        lv.unknown_keys
    );
    assert_eq!(lv.version, "1");
    assert_eq!(lv.spec_rev, "2026-06-15");
    assert_eq!(lv.room_floor, "grid");
    assert_eq!(lv.room_ceiling, "dim");
    assert_eq!(lv.door.target_model, "test");
}

/// An unrecognized SECTION is reported ONCE, at its header — not once per key
/// inside it, which would bury the finding under its own block.
#[test]
fn an_unknown_section_is_reported_once_at_its_header() {
    let src = "\
[meta]
name = us-test

[room]
size = 6 6 4

[triger zone_a]
region = (1,1,1)-(2,2,2)
action = activate a
once = yes

[node a]
pos = 1,2,1
is_input = yes

[node b]
pos = 4,2,1
is_output = yes

[pipe a->b]
state = ON
path = (1,2,1) (4,2,1)

[pedestal]
center = 3 3 1
volume = 2 2 2

[door]
pos = 0 3 1

[player]
spawn = 3 3 0
";
    let lv = load_string(src, "<us>").expect("an unknown section must NOT fail the load");
    let got: Vec<String> = lv.unknown_keys.iter().map(ToString::to_string).collect();
    assert_eq!(
        got,
        vec!["line 7: [?] unknown key 'triger'".to_string()],
        "one finding for the header, none for its three body keys"
    );
    assert!(
        lv.triggers.is_empty(),
        "the mistyped section really did nothing"
    );
}

// ---------------------------------------------------------------------------
// [note] bodies: COLUMN 0 means structure, an indent means content
// ---------------------------------------------------------------------------

/// Inside a `[note]` body a bracketed line ends the note only in COLUMN 0. An
/// INDENTED bracketed line is placard TEXT.
///
/// The body already used column 0 as the discriminator for `#`; it now uses it for
/// `[` as well. Before this, ANY bracketed line ended the body, which silently ate
/// `levels/L7-hidden-depths.lvl:369` — an authored placard line no player had ever
/// seen — and opened a bogus section from its first word.
#[test]
fn an_indented_bracket_line_is_note_text_not_a_header() {
    let src = "\
[meta]
name = nb-test

[room]
size = 6 6 4

[note researcher pos=1,1,2 facing=+x]
    FORM 19 - POWER CONTINUITY DIRECTIVE
    [ STEWARDSHIP - DO NOT POWER DOWN ]

[node a]
pos = 1,2,1
is_input = yes

[node b]
pos = 4,2,1
is_output = yes

[pipe a->b]
state = ON
path = (1,2,1) (4,2,1)

[pedestal]
center = 3 3 1
volume = 2 2 2

[door]
pos = 0 3 1

[player]
spawn = 3 3 0
";
    let lv = load_string(src, "<nb>").expect("parses");
    assert_eq!(lv.notes.len(), 1);
    assert_eq!(
        lv.notes[0].text,
        "    FORM 19 - POWER CONTINUITY DIRECTIVE\n    [ STEWARDSHIP - DO NOT POWER DOWN ]",
        "the indented bracket line is the note's LAST LINE, not a section header"
    );
    // …and it did not open a phantom section.
    assert!(
        lv.unknown_keys.is_empty(),
        "no bogus section: {:?}",
        lv.unknown_keys
    );
    // The real sections after it still parsed.
    assert_eq!(lv.nodes.len(), 2);
    assert_eq!(lv.pipes.len(), 1);
}

/// The runaway-note protection is untouched: a COLUMN-0 header still ends the body.
/// Every real header in every shipped level (and every header `re_voximport` writes)
/// sits at column 0, so an unterminated note still cannot swallow the file.
#[test]
fn a_column_zero_header_still_ends_a_note_body() {
    let src = "\
[meta]
name = nb2

[room]
size = 6 6 4

[note researcher pos=1,1,2 facing=+x]
    body line
[node a]
pos = 1,2,1
is_input = yes

[node b]
pos = 4,2,1
is_output = yes

[pipe a->b]
state = ON
path = (1,2,1) (4,2,1)

[pedestal]
center = 3 3 1
volume = 2 2 2

[door]
pos = 0 3 1

[player]
spawn = 3 3 0
";
    let lv = load_string(src, "<nb2>").expect("parses");
    assert_eq!(
        lv.notes[0].text, "    body line",
        "the column-0 header ended the body"
    );
    assert_eq!(lv.nodes.len(), 2, "the section after it really opened");
}

/// `escape_note_body_line` escapes BOTH column-0 markers, so an authored body line
/// starting with `[` (or `#`) in column 0 round-trips instead of truncating its own
/// placard on the next load — and the escape is idempotent.
#[test]
fn note_body_escape_covers_the_bracket_marker() {
    assert_eq!(escape_note_body_line("[ STEWARDSHIP ]"), " [ STEWARDSHIP ]");
    assert_eq!(escape_note_body_line("# 9 MINISTRY"), " # 9 MINISTRY");
    // Idempotent: the escaped form needs no further escaping.
    assert_eq!(
        escape_note_body_line(" [ STEWARDSHIP ]"),
        " [ STEWARDSHIP ]"
    );
    assert_eq!(escape_note_body_line(" # 9 MINISTRY"), " # 9 MINISTRY");
    // Untouched: no column-0 marker.
    assert_eq!(escape_note_body_line("    plain text"), "    plain text");
    assert_eq!(
        escape_note_body_line("MINISTRY CORPUS #9"),
        "MINISTRY CORPUS #9"
    );
}

/// THE MOTIVATING SHAPE, end to end through the real `load_string` pipeline (not
/// the closer unit check above): a placard line reading exactly like the one
/// that shipped as `levels/L7-hidden-depths.lvl:369` — "FORM 19 - POWER
/// CONTINUITY DIRECTIVE" followed by an indented `[ STEWARDSHIP - DO NOT POWER
/// DOWN ]` line — must still reach the note text as its last line.
///
/// Re-pointed off the campaign file — an engine test should not depend on
/// shipped campaign content — onto a fixture that carries the exact placard
/// text that motivated the fix, so this stays the "closed on real data"
/// assertion without a campaign `.lvl` reference.
#[test]
fn stewardship_placard_line_reaches_the_note_text() {
    let src = "\
[meta]
name = stewardship-fixture

[room]
size = 6 6 4

[note researcher pos=1,1,2 facing=+x]
    FORM 19 - POWER CONTINUITY DIRECTIVE
    [ STEWARDSHIP - DO NOT POWER DOWN ]

[node a]
pos = 1,2,1
is_input = yes

[node b]
pos = 4,2,1
is_output = yes

[pipe a->b]
path = (1,2,1) (4,2,1)

[pedestal]
center = 3 3 1
volume = 2 2 2

[door]
pos = 0 3 1

[player]
spawn = 3 3 0
";
    let lv = load_string(src, "<stewardship>").expect("parses");
    let note = lv
        .notes
        .iter()
        .find(|n| n.text.contains("POWER CONTINUITY DIRECTIVE"))
        .expect("fixture carries FORM 19");
    assert!(
        note.text.contains("[ STEWARDSHIP - DO NOT POWER DOWN ]"),
        "the authored placard line is in the note body:\n{}",
        note.text
    );
    assert_eq!(
        note.text.lines().last().map(str::trim),
        Some("[ STEWARDSHIP - DO NOT POWER DOWN ]"),
        "…as its last line"
    );
}

// ---------------------------------------------------------------------------
// [wall_decal] surface keywords
// ---------------------------------------------------------------------------

/// A minimal legal level with one `[wall_decal]` whose `surface` is `word`.
fn decal_surface_src(word: &str) -> String {
    format!(
        "[meta]\nname = decal_surface\n[room]\nsize = 12 16 10\nambient = solid\n\
         [wall_decal d0]\nsurface = {word}\nregion = (1,2,1)-(2,2,3)\n\
         [node a]\npos = 4,13,2\nis_input = yes\n\
         [node b]\npos = 7,13,2\nis_output = yes\nthreshold = 1\n\
         [pipe a->b]\npath = (4,13,2) (7,13,2)\n\
         [pedestal]\ncenter = 7 13 1\nvolume = 1 1 1\n\
         [door]\npos = 7 14 1\ntarget_parts = b\n\
         [player]\nspawn = 6 14 1\nfacing = -y\n"
    )
}

/// The parser accepts every `surface` spelling `re_voximport::to_lvl_text`
/// emits, and maps each to the matching `CellKind`. `hologram`/`portal`/
/// `object` are produced by the `.vox` import path (tags 13/14/15); rejecting
/// them made an imported level unloadable after a save.
#[test]
fn wall_decal_surface_accepts_all_six_kinds() {
    for (word, want) in [
        ("trace", CellKind::WallTrace),
        ("matrix", CellKind::WallMatrix),
        ("solid", CellKind::WallSolid),
        ("hologram", CellKind::Hologram),
        ("portal", CellKind::Portal),
        ("object", CellKind::Object),
    ] {
        let lv = load_string(&decal_surface_src(word), "decal")
            .unwrap_or_else(|e| panic!("`surface = {word}` must parse: {e}"));
        assert_eq!(lv.decals.len(), 1, "one decal for `{word}`");
        assert_eq!(lv.decals[0].surface_kind, want, "`surface = {word}` kind");
        // Load-bearing: the region is painted with that kind.
        assert_eq!(
            lv.build_map().cell_at(1, 2, 1),
            want,
            "`surface = {word}` paints the decal region"
        );
    }
}

/// Widening the accepted set must NOT make the parser lenient: a genuinely
/// unknown spelling still hard-errors.
#[test]
fn wall_decal_surface_rejects_unknown_kind() {
    let err = load_string(&decal_surface_src("glitter"), "decal")
        .expect_err("`surface = glitter` must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("[wall_decal] surface: unknown kind"),
        "error must name the offending key, got: {msg}"
    );
}

// ── `recipe =` covers every pattern kind (spec 2026-09-02 §1.5) ─────────────────
//
// Until 2026-09-02 `parse_recipe` knew seven kinds; Scanline/Truchet/Fractal/
// Substrate/Voice were reachable only through `preset =`. These pin the direct
// grammar: defaults equal the catalog preset's knobs, integer knobs keep their
// bits, keyword knobs reject unknown words, and an unknown KIND is still a hard
// error at the section level.
mod recipe_kinds_direct {
    use super::*;
    use re_world::{MaterialRecipe, VoiceMode};

    #[test]
    fn the_five_pattern_kinds_parse_with_preset_defaults() {
        assert_eq!(
            parse_recipe("scanline"),
            Some(MaterialRecipe::Scanline {
                freq: 9.0,
                duty: 0.55,
                warp: 0.05,
                axis: 0
            })
        );
        assert_eq!(
            parse_recipe("truchet"),
            Some(MaterialRecipe::Truchet {
                seed: 0,
                density: 0.5,
                scale: 4.0
            })
        );
        assert_eq!(
            parse_recipe("fractal"),
            Some(MaterialRecipe::Fractal {
                arms: 0,
                octave_warp: 0.30,
                twist: 0.0
            })
        );
        assert_eq!(
            parse_recipe("substrate"),
            Some(MaterialRecipe::Substrate {
                scale: 3.0,
                held: 0.55,
                released: 0.30
            })
        );
        assert_eq!(
            parse_recipe("voice"),
            Some(MaterialRecipe::Voice {
                scale: 3.0,
                mode: VoiceMode::Hold
            })
        );
    }

    #[test]
    fn integer_knobs_are_read_as_integers_and_clamped_where_the_renderer_names_a_set() {
        let Some(MaterialRecipe::Truchet { seed, .. }) = parse_recipe("truchet seed=2147483647")
        else {
            panic!("truchet must parse")
        };
        assert_eq!(seed, i32::MAX, "a large seed keeps every bit (no f32 lap)");
        let Some(MaterialRecipe::Scanline { axis, .. }) = parse_recipe("scanline axis=9") else {
            panic!()
        };
        assert_eq!(axis, 2, "axis clamps to the three the renderer names");
        let Some(MaterialRecipe::Fractal { arms, .. }) = parse_recipe("fractal arms=300") else {
            panic!()
        };
        assert_eq!(arms, 255);
    }

    #[test]
    fn voice_mode_keyword_rejects_unknown_words_rather_than_guessing() {
        assert_eq!(
            parse_recipe("voice mode=release").map(|r| match r {
                MaterialRecipe::Voice { mode, .. } => mode,
                _ => unreachable!(),
            }),
            Some(VoiceMode::Release)
        );
        assert_eq!(parse_recipe("voice mode=whisper"), None);
    }

    #[test]
    fn unknown_recipe_kind_is_still_a_section_level_error() {
        assert_eq!(parse_recipe("plasma"), None);
        let src = "\
[meta]
name = t
[room]
size = 6 6 4
[material walls]
recipe = plasma
surface = wall_solid
";
        let err = load_string(src, "t").expect_err("unknown recipe must fail the load");
        assert!(
            format!("{err}").contains("[material] recipe: unknown"),
            "got: {err}"
        );
    }

    #[test]
    fn the_fixture_level_loads_with_all_five_kinds_bound() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../levels/91-recipe-kinds.lvl");
        let raw = std::fs::read_to_string(&path).expect("fixture exists");
        let lvl = load_string(&raw, "91").expect("fixture parses");
        let kinds: Vec<&'static str> = lvl
            .materials
            .iter()
            .filter_map(|m| m.recipe.as_ref())
            .map(|r| match r {
                MaterialRecipe::Scanline { .. } => "scanline",
                MaterialRecipe::Truchet { .. } => "truchet",
                MaterialRecipe::Fractal { .. } => "fractal",
                MaterialRecipe::Substrate { .. } => "substrate",
                MaterialRecipe::Voice { .. } => "voice",
                _ => "other",
            })
            .collect();
        for k in ["scanline", "truchet", "fractal", "substrate", "voice"] {
            assert!(kinds.contains(&k), "fixture must bind `{k}`; got {kinds:?}");
        }
    }
}

// ── `recipe = facade` — the structured bay×floor window grid. Same discipline
// as `recipe_kinds_direct` above: bare defaults, integer/keyword knobs, and an
// unknown style/flag word rejected rather than guessed.
mod recipe_facade {
    use super::*;
    use re_world::{FacadeStyle, MaterialRecipe};

    #[test]
    fn bare_facade_parses_with_office_defaults() {
        assert_eq!(
            parse_recipe("facade"),
            Some(MaterialRecipe::Facade {
                style: FacadeStyle::Office,
                bay: 3.0,
                floor_h: 4.0,
                lit: 0.35,
                seed: 0,
                ground_band: false,
                sign_row: false,
            })
        );
    }

    #[test]
    fn every_style_keyword_and_knob_parses() {
        let r = parse_recipe(
            "facade style=server_rack bay=6 floor=10 lit=0.6 seed=-17 ground_band=on sign_row=on",
        );
        assert_eq!(
            r,
            Some(MaterialRecipe::Facade {
                style: FacadeStyle::ServerRack,
                bay: 6.0,
                floor_h: 10.0,
                lit: 0.6,
                seed: -17,
                ground_band: true,
                sign_row: true,
            })
        );
    }

    #[test]
    fn unknown_style_keyword_is_rejected_rather_than_guessed() {
        assert_eq!(parse_recipe("facade style=chrome"), None);
    }

    #[test]
    fn unknown_ground_band_or_sign_row_value_is_rejected() {
        assert_eq!(parse_recipe("facade ground_band=maybe"), None);
        assert_eq!(parse_recipe("facade sign_row=maybe"), None);
    }

    #[test]
    fn all_five_styles_are_reachable() {
        for (kw, style) in [
            ("office", FacadeStyle::Office),
            ("residential", FacadeStyle::Residential),
            ("industrial", FacadeStyle::Industrial),
            ("server_rack", FacadeStyle::ServerRack),
            ("lockers", FacadeStyle::Lockers),
        ] {
            let Some(MaterialRecipe::Facade { style: got, .. }) =
                parse_recipe(&format!("facade style={kw}"))
            else {
                panic!("`facade style={kw}` must parse")
            };
            assert_eq!(got, style, "style keyword `{kw}`");
        }
    }
}

// ── `[texture] max_size` — the LOAD-time decode cap (engine spec §1.4) ───────
// An ASCII cell samples a handful of texels, so a huge source image costs
// `w*h*4` bytes of RAM for detail no glyph can show. `max_size` caps the longest
// edge at decode time; unset keeps the native size, byte-identical to before.
mod texture_max_size {
    use super::*;

    /// A `w`×`h` opaque-grey PNG in the temp dir, with a per-call unique name so
    /// parallel tests never share a file.
    fn write_png(w: u32, h: u32) -> PathBuf {
        let rgba: Vec<u8> = (0..w * h).flat_map(|_| [128u8, 128, 128, 255]).collect();
        let img: image::RgbaImage = image::ImageBuffer::from_raw(w, h, rgba).expect("rgba buffer");
        let mut path = std::env::temp_dir();
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        path.push(format!("ts_maxsize_{}_{}.png", std::process::id(), seq));
        img.save(&path).expect("write png fixture");
        path
    }

    fn level_with(file: &str, extra: &str) -> String {
        format!(
            "\
[meta]
name = maxsize
[room]
size = 6 4 4
ambient = solid
[texture pic]
file = {file}
{extra}
[material img_wall]
recipe = image
texture = pic
surface = wall_solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
"
        )
    }

    #[test]
    fn an_oversized_image_is_downsampled_on_load_keeping_its_aspect() {
        let path = write_png(64, 32);
        let file = path.to_string_lossy().replace('\\', "/");
        let lv = load_string(&level_with(&file, "max_size = 16"), "<ms>").expect("parses");
        assert_eq!(
            lv.textures[0].max_size,
            Some(16),
            "the cap reaches the level"
        );
        let map = lv.build_map();
        let tex = map.texture(0).expect("the image decoded");
        assert_eq!(
            (tex.width, tex.height),
            (16, 8),
            "64x32 capped at 16 → 16x8"
        );
        assert_eq!(
            tex.rgba.len(),
            16 * 8 * 4,
            "the buffer is the REDUCED size — the whole point of the cap"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_image_already_within_the_cap_is_left_alone() {
        let path = write_png(8, 8);
        let file = path.to_string_lossy().replace('\\', "/");
        let lv = load_string(&level_with(&file, "max_size = 64"), "<ms>").expect("parses");
        let map = lv.build_map();
        let tex = map.texture(0).expect("the image decoded");
        assert_eq!(
            (tex.width, tex.height),
            (8, 8),
            "a small image is not upscaled"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn no_cap_keeps_the_native_size() {
        let path = write_png(40, 24);
        let file = path.to_string_lossy().replace('\\', "/");
        let lv = load_string(&level_with(&file, ""), "<ms>").expect("parses");
        assert_eq!(lv.textures[0].max_size, None, "no cap authored");
        let map = lv.build_map();
        let tex = map.texture(0).expect("the image decoded");
        assert_eq!((tex.width, tex.height), (40, 24), "unset ⇒ untouched");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_bad_cap_is_a_parse_error_with_a_line_number() {
        let path = write_png(8, 8);
        let file = path.to_string_lossy().replace('\\', "/");
        for bad in ["max_size = 0", "max_size = wide", "max_size = -4"] {
            let err = load_string(&level_with(&file, bad), "<ms>")
                .expect_err(&format!("`{bad}` must be refused"));
            let msg = format!("{err}");
            assert!(msg.contains("max_size"), "the error names the key: {msg}");
            assert!(
                msg.contains("<ms>:8:"),
                "the error points at the offending line (source:line): {msg}"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unknown_texture_key_is_reported_instead_of_swallowed() {
        // `[texture]` used to silently ignore every key but `file`, so a typo
        // looked accepted and did nothing.
        let path = write_png(8, 8);
        let file = path.to_string_lossy().replace('\\', "/");
        let lv = load_string(&level_with(&file, "maxsize = 16"), "<ms>").expect("parses");
        assert!(
            lv.unknown_keys
                .iter()
                .any(|u| u.key == "maxsize" && u.section == "texture"),
            "a typo'd [texture] key must show up in the diagnostic: {:?}",
            lv.unknown_keys
        );
        let _ = std::fs::remove_file(&path);
    }
}

// -- `[texture] frames = clip.gif` + `fps` - ANIMATED image textures (spec 1.4) --
// A texture may carry N frames instead of 1, decoded from a GIF once at load. The
// shader indexes the set by the cast's `time`; a 1-frame texture behaves EXACTLY as
// a still one always has.
mod texture_frames {
    use super::*;

    /// The repo's committed, GENERATED 8-frame test GIF (48x24 —
    /// `re_voximport::texfixture::anim_gif`, replacing an earlier untracked,
    /// unknown-origin GIF that had no place in an open-source tree). Resolved
    /// from the MANIFEST dir, never the process CWD - the very thing asset
    /// resolution replaced.
    fn gif_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("assets")
            .join("texture")
            .join("gen_anim.gif")
    }

    /// A minimal level whose one `[texture pic]` carries `{extra}` verbatim.
    fn level_with(extra: &str) -> String {
        format!(
            "\
[meta]
name = texframes
[room]
size = 6 4 4
ambient = solid
[texture pic]
{extra}
[material img_wall]
recipe = image
texture = pic
surface = wall_solid
[node a]
pos = 1 1 2
is_input = yes
[node b]
pos = 4 1 2
is_output = yes
[pipe a->b]
path = (1,1,2) (4,1,2)
[pedestal]
center = 4 1 1
volume = 1 1 1
[door]
pos = 5 1 1
[player]
spawn = 3 1 1
"
        )
    }

    /// A `w`x`h` GIF of `n` frames in the temp dir (frame `i` a flat grey), with a
    /// per-call unique name so parallel tests never share a file.
    fn write_gif(w: u32, h: u32, n: u32) -> PathBuf {
        let mut path = std::env::temp_dir();
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        path.push(format!("ts_texframes_{}_{}.gif", std::process::id(), seq));
        let file = std::fs::File::create(&path).expect("create gif fixture");
        let mut enc = image::codecs::gif::GifEncoder::new(std::io::BufWriter::new(file));
        for i in 0..n {
            let v = (i * 20) as u8;
            let rgba: Vec<u8> = (0..w * h).flat_map(|_| [v, v, v, 255u8]).collect();
            let buf: image::RgbaImage =
                image::ImageBuffer::from_raw(w, h, rgba).expect("rgba buffer");
            enc.encode_frame(image::Frame::new(buf))
                .expect("encode frame");
        }
        drop(enc);
        path
    }

    /// PROTECTS: the repo's 8-frame GIF really decodes to 8 frames, and `max_size`
    /// applies to EVERY frame - the cap is the reason a large real-world animation
    /// (a downloaded clip, which this generated fixture stands in for) is affordable.
    #[test]
    fn the_repo_gif_decodes_every_frame_and_honours_max_size() {
        let file = gif_path().to_string_lossy().replace('\\', "/");
        let src = level_with(&format!("frames = {file}\nfps = 12\nmax_size = 24"));
        let lv = load_string(&src, "<gif>").expect("parses");
        assert_eq!(lv.textures[0].frames.as_deref(), Some(file.as_str()));
        assert_eq!(lv.textures[0].fps, Some(12.0));

        let map = lv.build_map();
        let set = map.texture_set(0).expect("the animation decoded");
        assert_eq!(set.len(), 8, "gen_anim.gif is an 8-frame clip");
        assert!(set.is_animated());
        assert!((set.fps() - 12.0).abs() < 1e-6);
        // 48x24 capped at 24 -> 24x12 (aspect preserved), for EVERY frame.
        for (i, f) in set.frames().iter().enumerate() {
            assert_eq!(
                (f.width, f.height),
                (24, 12),
                "frame {i} must obey max_size, not just frame 0"
            );
            assert_eq!(
                f.rgba.len(),
                24 * 12 * 4,
                "frame {i} buffer is the REDUCED size"
            );
        }
        assert_eq!(
            map.texture(0).map(|t| (t.width, t.height)),
            Some((24, 12)),
            "`Map::texture` still means frame 0"
        );
    }

    /// PROTECTS: the decoded set advances with time and WRAPS - the animation actually
    /// plays and loops, rather than freezing on frame 0.
    #[test]
    fn the_decoded_frame_index_advances_with_time_and_wraps() {
        let path = write_gif(8, 8, 4);
        let file = path.to_string_lossy().replace('\\', "/");
        let lv = load_string(&level_with(&format!("frames = {file}\nfps = 10")), "<gif>")
            .expect("parses");
        let map = lv.build_map();
        let set = map.texture_set(0).expect("decoded");
        assert_eq!(set.len(), 4);
        assert_eq!(set.index_at(0.0), 0);
        assert_eq!(set.index_at(0.15), 1);
        assert_eq!(set.index_at(0.35), 3);
        assert_eq!(set.index_at(0.40), 0, "the clip loops");
        let _ = std::fs::remove_file(&path);
    }

    /// PROTECTS: NO-REGRESSION - a texture with no `frames` is a ONE-frame set whose
    /// sampling is unchanged and time-independent.
    #[test]
    fn a_still_texture_is_a_one_frame_set_with_unchanged_sampling() {
        let path = write_gif(8, 8, 1);
        let file = path.to_string_lossy().replace('\\', "/");
        let lv = load_string(&level_with(&format!("file = {file}")), "<still>").expect("parses");
        assert_eq!(lv.textures[0].frames, None, "no frames authored");
        assert_eq!(lv.textures[0].fps, None);
        let map = lv.build_map();
        let set = map.texture_set(0).expect("decoded");
        assert_eq!(set.len(), 1);
        assert!(!set.is_animated(), "a still texture must not be animated");
        let a = set.frame_at(0.0).sample_nearest(0.5, 0.5);
        let b = set.frame_at(99.0).sample_nearest(0.5, 0.5);
        assert_eq!(a, b, "a still texture samples identically at any time");
        let _ = std::fs::remove_file(&path);
    }

    /// PROTECTS: `frames` pointing at a SINGLE-frame file is not an error - it yields
    /// a one-frame set, i.e. exactly a still texture.
    #[test]
    fn frames_on_a_single_frame_file_is_a_one_frame_set_not_an_error() {
        let path = write_gif(8, 8, 1);
        let file = path.to_string_lossy().replace('\\', "/");
        let lv = load_string(&level_with(&format!("frames = {file}")), "<one>").expect("parses");
        let map = lv.build_map();
        let set = map.texture_set(0).expect("decoded");
        assert_eq!(set.len(), 1);
        assert!(!set.is_animated());
        let _ = std::fs::remove_file(&path);
    }

    /// PROTECTS: an `fps` that could never play is a LINE-NUMBERED parse error, never
    /// a silent fallback to the default.
    #[test]
    fn a_bad_fps_is_a_parse_error_with_a_line_number() {
        for bad in [
            "fps = 0",
            "fps = -3",
            "fps = nope",
            "fps = inf",
            "fps = nan",
        ] {
            let err = load_string(&level_with(bad), "<fps>")
                .expect_err(&format!("`{bad}` must be refused"));
            let msg = format!("{err}");
            assert!(msg.contains("fps"), "the error names the key: {msg}");
            assert!(
                msg.contains("<fps>:7:"),
                "the error points at the offending line (source:line): {msg}"
            );
        }
    }

    /// PROTECTS: `fps` on its own (no `frames`) parses and is preserved - the two keys
    /// are independent, so authoring the rate first is not a load failure.
    #[test]
    fn fps_without_frames_parses_and_is_preserved() {
        let path = write_gif(8, 8, 1);
        let file = path.to_string_lossy().replace('\\', "/");
        let lv =
            load_string(&level_with(&format!("file = {file}\nfps = 24")), "<fo>").expect("parses");
        assert_eq!(lv.textures[0].fps, Some(24.0));
        assert_eq!(lv.textures[0].frames, None);
        let _ = std::fs::remove_file(&path);
    }

    /// PROTECTS: a MISSING `frames` file is reported rather than silently rendering
    /// nothing - the "authored asset that never appears" failure `asset_audit` exists
    /// to catch.
    #[test]
    fn a_missing_frames_file_is_reported_as_a_broken_asset() {
        let lv = load_string(
            &level_with("frames = assets/texture/definitely_not_a_clip.gif"),
            "<miss>",
        )
        .expect("parses");
        let missing = lv.missing_assets();
        assert!(
            missing
                .iter()
                .any(|i| i.asset.authored.contains("definitely_not_a_clip")),
            "the missing frames file must be reported: {missing:?}"
        );
    }
}

// ── Set-dressing prerequisites ──────────────────────────────────────────────

/// An 8x8x4 `[cells]` block over [`ZONE_FILL_BASE`]: floor slab at z=0, ceiling at
/// z=3, a perimeter wall ring at z=1..2, air inside — plus `extra` cells (`x,y,z,glyph`)
/// painted over it. Every interior block therefore exists ONLY in `[cells]`.
fn cells_room(extra: &[(usize, usize, usize, char)]) -> String {
    let (sx, sy, sz) = (8usize, 8usize, 4usize);
    let mut g = vec!['.'; sx * sy * sz];
    for z in 0..sz {
        for y in 0..sy {
            for x in 0..sx {
                let edge = x == 0 || y == 0 || x == sx - 1 || y == sy - 1;
                g[x + sx * (y + sy * z)] = match z {
                    0 => '_',
                    3 => '^',
                    _ if edge => '#',
                    _ => '.',
                };
            }
        }
    }
    for &(x, y, z, c) in extra {
        g[x + sx * (y + sy * z)] = c;
    }
    let mut out = String::from("[cells]\nsize = 8,8,4\n");
    for z in 0..sz {
        for y in 0..sy {
            out.extend((0..sx).map(|x| g[x + sx * (y + sy * z)]));
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

/// F1/E1: a `[zone]` over geometry that exists only in `[cells]` now skins it. Before
/// the fix the zone pass ran while those cells were still air and skipped them, so
/// the zone changed nothing (the doc's A/B: 0 of 12 000 cells).
#[test]
fn zone_skins_geometry_authored_only_in_cells() {
    let src = format!(
        "{ZONE_FILL_BASE}[zone shelf]\nregion = (3,4,1)-(4,4,2)\nmaterial = floor_tile\n{}",
        cells_room(&[
            (3, 4, 1, '#'),
            (3, 4, 2, '#'),
            (4, 4, 1, '/'),
            (5, 4, 1, '#')
        ])
    );
    let map = load_string(&src, "<zone-cells>")
        .expect("parses")
        .build_map();
    assert_eq!(map.cell_at(3, 4, 1), CellKind::WallSolid);
    assert_eq!(
        map.side_data_at(3, 4, 1),
        2,
        "cells-only block takes the zone material"
    );
    assert_eq!(map.side_data_at(3, 4, 2), 2);
    assert_eq!(map.side_data_at(4, 4, 1), 2, "a trace wall is geometry too");
    // Air inside the region stays air and untagged; a block outside it is untouched.
    assert_eq!(map.cell_at(4, 4, 2), CellKind::Empty);
    assert_eq!(map.side_data_at(4, 4, 2), 0);
    assert_eq!(
        map.side_data_at(5, 4, 1),
        0,
        "outside the zone => surface default"
    );
}

/// The second zone pass keeps the first pass's precedence rules: a later zone wins
/// an overlap, an explicit `[color_cell_ids]` entry beats both, and a node anchor
/// inside a zone keeps its node tag.
#[test]
fn zone_over_cells_geometry_keeps_precedence() {
    let src = format!(
        "{ZONE_FILL_BASE}\
[zone first]\nregion = (2,5,1)-(5,5,1)\nmaterial = floor_base\n\
[zone second]\nregion = (4,5,1)-(5,5,1)\nmaterial = floor_tile\n\
[zone over_node]\nregion = (5,1,1)-(5,1,1)\nmaterial = floor_base\n\
[color_cell_ids]\n5,5,1 = 1\n{}",
        cells_room(&[
            (2, 5, 1, '#'),
            (3, 5, 1, '#'),
            (4, 5, 1, '#'),
            (5, 5, 1, '#')
        ])
    );
    let map = load_string(&src, "<zone-prec>")
        .expect("parses")
        .build_map();
    assert_eq!(map.side_data_at(2, 5, 1), 1, "only `first` covers x=2");
    assert_eq!(
        map.side_data_at(4, 5, 1),
        2,
        "the later zone wins the overlap"
    );
    assert_eq!(
        map.side_data_at(5, 5, 1),
        1,
        "an explicit colour id beats both zones"
    );
    // Node b (index 1) at (5,1,1): `[cells]` cleared it, the anchor re-stamp restored
    // it, and the zone must not overwrite its node tag.
    assert_eq!(map.cell_at(5, 1, 1), CellKind::Object);
    assert_eq!(
        map.side_data_at(5, 1, 1),
        1,
        "the node keeps its node index"
    );
}

/// A level whose zones only cover the procedural shell gets exactly the tags it had
/// before the second pass existed (the pass only ever touches untagged cells).
#[test]
fn zone_second_pass_is_a_noop_without_cells_only_geometry() {
    let src = format!(
        "{ZONE_FILL_BASE}[zone tile]\nregion = (2,2,0)-(4,4,0)\nmaterial = floor_tile\n{}",
        cells_room(&[])
    );
    let map = load_string(&src, "<zone-shell>")
        .expect("parses")
        .build_map();
    assert_eq!(map.side_data_at(3, 3, 0), 2, "floor zone as before");
    assert_eq!(map.side_data_at(1, 1, 0), 0);
    assert_eq!(map.side_data_at(3, 3, 1), 0, "air above untouched");
}

/// `[cells]` sub-table order is irrelevant: `material_ids =` after `shapes =` used to
/// be swallowed as a malformed shape row (a parse error).
#[test]
fn cells_material_ids_after_shapes_parses() {
    // (0,1,1), not (2,1,1) — see the note on `cells_material_ids_round_trip` (that
    // coordinate is `cells_shell`'s door anchor, stamped with the DERIVED
    // `DOOR_OBJECT_TAG` after `[cells]`, RC5 fix).
    let body = "\
[cells]
size = 3,2,2
___
___

#.#
###
shapes =
0 0 1 wedgexy ne
material_ids =
0,1,1 = 7
shapes =
2 0 1 rampz posx
";
    let lv = load_string(&cells_shell(body), "<order>").expect("either order parses");
    let g = lv.cells.as_ref().unwrap();
    assert_eq!(g.material_ids, vec![(0, 1, 1, 7u16)]);
    assert_eq!(g.shapes.len(), 2, "both shape tables are kept");
    assert_eq!(lv.build_map().side_data_at(0, 1, 1), 7);
}

/// F2: a painted `O` no puzzle entity backs is tagged INERT (so it cannot read as
/// node 0), real node anchors keep their node tags, and the derived tag is never
/// persisted by the editor capture.
#[test]
fn unbacked_object_cells_are_tagged_inert() {
    let src = format!(
        "{ZONE_FILL_BASE}{}",
        cells_room(&[(1, 1, 1, 'O'), (3, 6, 1, 'O'), (4, 6, 1, 'O')])
    );
    let lv = load_string(&src, "<inert>").expect("parses");
    let map = lv.build_map();
    assert_eq!(map.cell_at(3, 6, 1), CellKind::Object);
    assert_eq!(
        map.side_data_at(3, 6, 1),
        INERT_OBJECT_TAG,
        "a bench is not a neuron"
    );
    assert_eq!(map.side_data_at(4, 6, 1), INERT_OBJECT_TAG);
    // Node a (index 0) painted as `O` at its own anchor keeps tag 0 — it IS node 0.
    assert_eq!(map.side_data_at(1, 1, 1), 0, "node 0's anchor is not inert");
    assert_eq!(map.side_data_at(5, 1, 1), 1, "node b keeps its index");
    // The editor capture drops the derived tag, so a save writes no bogus id.
    let g = CellGrid::from_map(&map);
    assert!(
        g.material_ids
            .iter()
            .all(|&(_, _, _, id)| id != INERT_OBJECT_TAG),
        "the inert tag must not be persisted: {:?}",
        g.material_ids
    );
}

/// RC5 fix regression: the door anchor is stamped `DOOR_OBJECT_TAG` (not `0`, which
/// used to collide with node 0's tag — see `Level::build_map`'s objects list) — and,
/// exactly like the INERT sentinel above, that DERIVED tag must never be captured by
/// the editor's `CellGrid::from_map` (it would otherwise be re-applied as a bogus
/// colour-material id on reload, corrupting the door's freshly-recomputed tag).
#[test]
fn door_anchor_tag_is_derived_not_persisted() {
    let lv = load_string(ZONE_FILL_BASE, "<door-tag>").expect("parses");
    let map = lv.build_map();
    assert_eq!(
        map.cell_at(6, 1, 1),
        CellKind::Object,
        "the door anchor from [door] pos = 6 1 1"
    );
    assert_eq!(
        map.side_data_at(6, 1, 1),
        re_core::DOOR_OBJECT_TAG,
        "the door must not carry tag 0 (node 0's tag)"
    );
    let g = CellGrid::from_map(&map);
    assert!(
        g.material_ids
            .iter()
            .all(|&(_, _, _, id)| id != re_core::DOOR_OBJECT_TAG),
        "the door's derived sentinel tag must not be persisted: {:?}",
        g.material_ids
    );
}

/// RC5 follow-up: when a `[cells]`-format level's dense grid repaints the door's own
/// position to a non-`Object` kind (the door lives baked into the room shell, per
/// `Level::build_map_base`'s "NODES ONLY" comment — doors are deliberately NOT
/// re-stamped as an anchor there), the door's `DOOR_OBJECT_TAG` from the earlier
/// procedural `stamp_geometry` pass must not survive as orphaned `side_data` on the
/// repainted cell. Before RC5 this exact leak was invisible (the door's tag was `0`,
/// indistinguishable from "no tag"); this pins the fix that clears it now that the tag
/// is a distinct nonzero sentinel.
#[test]
fn door_anchors_stale_tag_is_cleared_when_cells_repaints_it() {
    // cells_shell's door is at (2,1,1); its [cells] body paints a WallSolid ring at
    // z=1, so (2,1,1) — the last cell of "###" — ends up WallSolid, not Object.
    let body = "\
[cells]
size = 3,2,2
___
___

#.#
###
";
    let lv = load_string(&cells_shell(body), "<door-stale-tag>").expect("parses");
    let map = lv.build_map();
    assert_eq!(
        map.cell_at(2, 1, 1),
        CellKind::WallSolid,
        "the [cells] grid repaints the door's own position to a wall"
    );
    assert_eq!(
        map.side_data_at(2, 1, 1),
        0,
        "the door's stale DOOR_OBJECT_TAG must not leak onto the repainted cell"
    );
}

/// An authored tag on a bare `O` (the editor's inspect-`n`) still wins over the inert
/// default.
#[test]
fn authored_object_tag_beats_the_inert_default() {
    let src = format!(
        "{ZONE_FILL_BASE}{}material_ids =\n3,6,1 = 4\n",
        cells_room(&[(3, 6, 1, 'O')])
    );
    let map = load_string(&src, "<inert-auth>")
        .expect("parses")
        .build_map();
    assert_eq!(map.side_data_at(3, 6, 1), 4);
}

/// E0: each `[line]` gets its level-authored material's palette id on the map, and
/// binding a line to an authored material is enough to append the palette.
#[test]
fn line_material_resolves_to_its_palette_id() {
    let src = format!(
        "{ZONE_FILL_BASE}\
[line a]\npath = 2,2,1.1; 5,2,1.1\nmaterial = floor_tile\n\
[line b]\npath = 2,3,1.1; 5,3,1.1\nmaterial = rust\n\
[line c]\npath = 2,4,1.1; 5,4,1.1\n"
    );
    let map = load_string(&src, "<line-ids>").expect("parses").build_map();
    assert_eq!(
        map.materials().len(),
        7,
        "5 surface slots + the 2-entry palette"
    );
    assert_eq!(
        map.line_palette_id(0),
        2,
        "floor_tile is authored material #2"
    );
    assert_eq!(
        map.line_palette_id(1),
        0,
        "a catalog name is resolved by the renderer"
    );
    assert_eq!(map.line_palette_id(2), 0, "no material => unresolved");
    assert_eq!(map.line_palette_id(9), 0, "out of range => unresolved");
}

/// A level with no `[line]` binding an authored material keeps the historical
/// material list (no palette appended) — byte-identical.
#[test]
fn no_line_bindings_keep_the_five_slot_material_list() {
    let map = load_string(ZONE_FILL_BASE, "<no-lines>")
        .expect("parses")
        .build_map();
    assert_eq!(map.materials().len(), 5);
    assert_eq!(map.line_palette_id(0), 0);
}

/// A `[note]` authored on a zoned `[cells]`-only block keeps an UNTAGGED host cell at
/// the reader's eye layer (the placard binder paints only untagged masonry), while the
/// rest of the block takes the zone skin.
#[test]
fn zone_second_pass_leaves_the_note_host_wall_untagged() {
    let src = format!(
        "{ZONE_FILL_BASE}[zone block]\nregion = (4,2,1)-(4,6,2)\nmaterial = floor_tile\n\
         [note researcher pos=4,4,2 facing=-x]\n    hello\n{}",
        cells_room(&[
            (4, 2, 1, '#'),
            (4, 3, 1, '#'),
            (4, 4, 1, '#'),
            (4, 5, 1, '#'),
            (4, 6, 1, '#'),
            (4, 2, 2, '#'),
            (4, 3, 2, '#'),
            (4, 4, 2, '#'),
            (4, 5, 2, '#'),
            (4, 6, 2, '#'),
        ])
    );
    let map = load_string(&src, "<note-host>")
        .expect("parses")
        .build_map();
    // The note faces -x from (4,4): its wall is (4,4) itself; eye layer of storey 0 = 1.
    assert_eq!(
        map.side_data_at(4, 4, 1),
        0,
        "the placard's host stays paintable"
    );
    assert_eq!(map.side_data_at(4, 3, 1), 2, "its neighbours take the zone");
    assert_eq!(
        map.side_data_at(4, 4, 2),
        2,
        "the plane above is skinned too"
    );
}

/// Loose ends: a glyph `[cells]` paints OVER a node anchor no longer wins. The
/// anchor is restored as `Object` with the node's own tag, whatever was painted —
/// before, `stamp_geometry_into` kept the node index in `side_data` under a WallTrace
/// kind, so the renderer read the index as a palette colour (L10's junction `J` drew
/// as the researcher placard; every L11/L12 node was a `/`).
#[test]
fn a_glyph_painted_over_a_node_anchor_is_restored_to_the_node() {
    let src = format!(
        "{ZONE_FILL_BASE}{}",
        // node a (0) at (1,1,1) painted `/`, node b (1) at (5,1,1) painted `#`.
        cells_room(&[(1, 1, 1, '/'), (5, 1, 1, '#')])
    );
    let lv = load_string(&src, "<anchor-over>").expect("parses");
    for map in [lv.build_map_base(), lv.build_map()] {
        assert_eq!(
            map.cell_at(1, 1, 1),
            CellKind::Object,
            "node a is an Object again"
        );
        assert_eq!(map.side_data_at(1, 1, 1), 0, "node a keeps tag 0");
        assert_eq!(
            map.cell_at(5, 1, 1),
            CellKind::Object,
            "node b is an Object again"
        );
        assert_eq!(
            map.side_data_at(5, 1, 1),
            1,
            "node b keeps its index, not a colour"
        );
    }
    // The door is NOT forced: `[cells]` keeps whatever it put at the door anchor.
    let src = format!("{ZONE_FILL_BASE}{}", cells_room(&[(6, 1, 1, '#')]));
    let map = load_string(&src, "<door>").expect("parses").build_map();
    assert_eq!(
        map.cell_at(6, 1, 1),
        CellKind::WallSolid,
        "the door stays as authored"
    );
}

/// A level whose `[cells]` grid paints a glyph over BOTH of [`ZONE_FILL_BASE`]'s
/// nodes (`a` at (1,1,1), `b` at (5,1,1)) still restores every node anchor,
/// whatever was painted there — the general property the campaign's
/// `levels/L10-lying-door.lvl` / `L11-inner-circuit.lvl` / `L12-closing.lvl`
/// originally exposed (their `[cells]` grids each painted over a node). Three
/// glyph variants stand in for that diversity: a wall-trace, a wall-matrix, and
/// an `Object` glyph carrying the WRONG tag (`side_data = 0` from the raw
/// paint) — the last one is the sharpest case, since a naive "already an
/// Object" check would wrongly call it already-fixed. Re-pointed off the three
/// campaign files, since an engine test should not depend on shipped campaign
/// content.
#[test]
fn a_glyph_painted_over_either_node_anchor_is_restored_to_the_node() {
    for glyph in ['/', '%', 'O'] {
        let src = format!(
            "{ZONE_FILL_BASE}{}",
            cells_room(&[(1, 1, 1, glyph), (5, 1, 1, glyph)])
        );
        let lv = load_string(&src, "<anchors>").unwrap_or_else(|e| panic!("glyph {glyph}: {e}"));
        let map = lv.build_map();
        for (i, n) in lv.nodes.iter().enumerate() {
            assert_eq!(
                map.cell_at(n.x, n.y, n.z),
                CellKind::Object,
                "glyph {glyph}: node {}",
                n.name
            );
            assert_eq!(
                map.side_data_at(n.x, n.y, n.z),
                i as u16,
                "glyph {glyph}: node {} tag",
                n.name
            );
        }
    }
}

/// Loose ends: `[holo]` stamps survive a dense `[cells]` grid. The grid writes `.`
/// over the whole room, which used to erase every hologram voxel; holograms now stamp
/// again after `[cells]`, still only into air, so painted geometry keeps its cell.
#[test]
fn holo_survives_a_dense_cells_grid() {
    let src = format!(
        "{ZONE_FILL_BASE}[holo cube]\nshape = box\nregion = (2,3,1)-(3,4,2)\nstyle = 3\n{}",
        cells_room(&[(3, 4, 2, '#')])
    );
    let lv = load_string(&src, "<holo-cells>").expect("parses");
    let map = lv.build_map();
    assert_eq!(
        map.cell_at(2, 3, 1),
        CellKind::Hologram,
        "the holo is back in the air"
    );
    assert_eq!(map.side_data_at(2, 3, 1), 3, "with its style tag");
    assert_eq!(map.cell_at(3, 4, 1), CellKind::Hologram);
    assert_eq!(
        map.cell_at(3, 4, 2),
        CellKind::WallSolid,
        "painted geometry still wins"
    );
    assert_eq!(
        map.cell_at(4, 3, 1),
        CellKind::Empty,
        "outside the region stays air"
    );
    // Without the holo section the same grid is plain air there (the stamp is the
    // only thing that changed the cell).
    let bare = format!("{ZONE_FILL_BASE}{}", cells_room(&[(3, 4, 2, '#')]));
    let map = load_string(&bare, "<no-holo>").expect("parses").build_map();
    assert_eq!(map.cell_at(2, 3, 1), CellKind::Empty);
}

/// A minimal level with one `[part]` carrying `keys` (set-dressing pass 2, item 1).
fn part_look_level(keys: &str) -> Result<Level, ParseError> {
    let src = format!(
        "[meta]\nname = pl\n[room]\nsize = 6 4 4\n[part blk]\nsize = 1,1,1\n{keys}mask =\n#\n\
         [pedestal]\ncenter = 4 1 1\nvolume = 1 1 1\n[door]\npos = 5 1 1\n[player]\nspawn = 3 1 1\n"
    );
    load_string(&src, "<pl>")
}

/// `[part] height / top / top_from`: absent = the full-height one-colour default (so no
/// existing level changes); `half` / a fraction set the band; `top` defaults its split to
/// [`PART_TOP_FROM_DEFAULT`]; `top_from` is order-independent and without a `top` it is
/// inert; out-of-range values are hard errors.
#[test]
fn part_height_and_top_keys_parse() {
    let lv = part_look_level("").expect("plain part parses");
    assert_eq!(lv.part_look(0), PartLook::default());
    assert_eq!(PartLook::default().height, 1.0);
    assert_eq!(PartLook::default().top, None);

    let lv = part_look_level("height = half\n").unwrap();
    assert_eq!(lv.part_look(0).height, 0.5);
    let lv = part_look_level("height = 0.35\ntop = #FFD080\n").unwrap();
    assert_eq!(lv.part_look(0).height, 0.35);
    assert_eq!(lv.part_look(0).top, Some((0xFFD080, PART_TOP_FROM_DEFAULT)));

    for keys in [
        "top = #00FF00\ntop_from = 0.25\n",
        "top_from = 0.25\ntop = #00FF00\n",
    ] {
        let lv = part_look_level(keys).unwrap();
        assert_eq!(lv.part_look(0).top, Some((0x00FF00, 0.25)), "{keys:?}");
        assert_eq!(lv.part_look(0).height, 1.0, "height stays full: {keys:?}");
    }
    let lv = part_look_level("top_from = 0.3\n").unwrap();
    assert_eq!(lv.part_look(0).top, None, "top_from without top is inert");

    for bad in [
        "height = 0.05\n",
        "height = 1.5\n",
        "height = tall\n",
        "top = red\n",
        "top_from = 2\n",
    ] {
        assert!(part_look_level(bad).is_err(), "{bad:?} must be rejected");
    }
}
