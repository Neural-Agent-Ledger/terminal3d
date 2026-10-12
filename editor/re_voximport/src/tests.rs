//! In-memory derivation tests — no `.vox` file, no `dot_vox`.

use super::*;

/// Default-tag RGBs for building voxel grids by hand.
const C_NODE_IN: [u8; 3] = [0xFF, 0xD0, 0x00];
const C_NODE_OUT: [u8; 3] = [0xFF, 0x30, 0x30];
const C_PIPE: [u8; 3] = [0x80, 0x00, 0xFF];
const C_SWITCH: [u8; 3] = [0xFF, 0x00, 0xFF];
const C_DOOR: [u8; 3] = [0x00, 0xFF, 0xFF];
const C_PLAYER: [u8; 3] = [0xFF, 0xA5, 0x00];
const C_WALL_TRACE: [u8; 3] = [0x00, 0xA0, 0xFF];

fn vx(x: i32, y: i32, z: i32, rgb: [u8; 3]) -> Voxel {
    Voxel { x, y, z, rgb }
}

/// A minimal 2-node + 1-pipe + 1-switch room.
///
/// Layout in a 6x6x4 room at z=2:
///   in_a node at (1,2,2), out node at (4,2,2).
///   pipe along (2,2,2)-(3,2,2); switch adjacent at (2,3,2).
///   door (5,2,1), player (3,2,1).
fn minimal_voxels() -> Vec<Voxel> {
    vec![
        vx(1, 2, 2, C_NODE_IN),
        vx(4, 2, 2, C_NODE_OUT),
        vx(2, 2, 2, C_PIPE),
        vx(3, 2, 2, C_PIPE),
        vx(2, 3, 2, C_SWITCH), // 6-adjacent to pipe cell (2,2,2)
        vx(5, 2, 1, C_DOOR),
        vx(3, 2, 1, C_PLAYER),
    ]
}

fn minimal_cfg() -> Config {
    let toml = r#"
[meta]
name = "test01"
title = "TEST ROOM"

[[node]]
pos = [1, 2, 2]
name = "in_a"
glyph = "A"
threshold = 1

[[node]]
pos = [4, 2, 2]
name = "out"
glyph = "O"
threshold = 1

[inputs]
in_a = 1
"#;
    toml::from_str(toml).expect("cfg parse")
}

#[test]
fn minimal_room_derives_graph() {
    let v = minimal_voxels();
    let cfg = minimal_cfg();
    let level = derive_level(&v, (6, 6, 4), &cfg).expect("derive");

    assert_eq!(level.name, "test01");
    assert_eq!(level.size, (6, 6, 4));

    // Two nodes, named + role from tag.
    assert_eq!(level.nodes.len(), 2);
    let in_a = level.node_index("in_a").expect("in_a present");
    let out = level.node_index("out").expect("out present");
    assert!(level.nodes[in_a].is_input);
    assert!(!level.nodes[in_a].is_output);
    assert!(level.nodes[out].is_output);
    assert_eq!(
        (
            level.nodes[in_a].x,
            level.nodes[in_a].y,
            level.nodes[in_a].z
        ),
        (1, 2, 2)
    );
    assert_eq!(
        (level.nodes[out].x, level.nodes[out].y, level.nodes[out].z),
        (4, 2, 2)
    );

    // One pipe, oriented input → output, switch resolved onto the path.
    assert_eq!(level.pipes.len(), 1);
    let p = &level.pipes[0];
    assert_eq!(p.from_node, in_a);
    assert_eq!(p.to_node, out);
    assert_eq!(p.from, "in_a");
    assert_eq!(p.to, "out");
    // Path runs node→pipe cells→node, all 6-connected.
    assert_eq!(p.path.first(), Some(&(1, 2, 2)));
    assert_eq!(p.path.last(), Some(&(4, 2, 2)));
    for w in p.path.windows(2) {
        let d = (w[0].0 - w[1].0).abs() + (w[0].1 - w[1].1).abs() + (w[0].2 - w[1].2).abs();
        assert_eq!(d, 1, "path step must be 6-connected: {:?}", w);
    }
    let sidx = p.switch_cell.expect("switch resolved");
    assert_eq!(p.path[sidx], (2, 2, 2));

    // Door / player set from their tagged voxels.
    assert_eq!((level.door.x, level.door.y, level.door.z), (5, 2, 1));
    assert_eq!((level.player.x, level.player.y, level.player.z), (3, 2, 1));
    // target_parts auto-derives to the output node.
    assert_eq!(level.door.target_parts, vec![out]);

    // The world is not empty: stamped map has walls.
    let map = level.build_map();
    let walls = map.count(re_world::CellKind::WallTrace)
        + map.count(re_world::CellKind::WallSolid)
        + map.count(re_world::CellKind::WallMatrix);
    assert!(walls > 0, "stamped map must contain walls, got {walls}");
    // 2 node cells + 1 door anchor = 3 Object cells (all stamped as generic objects).
    assert_eq!(map.count(re_world::CellKind::Object), 3);
}

#[test]
fn round_trip_through_parser() {
    let v = minimal_voxels();
    let cfg = minimal_cfg();
    let level = derive_level(&v, (6, 6, 4), &cfg).expect("derive");

    let text = to_lvl_text(&level);
    let rt = re_level::load_string(&text, "rt").unwrap_or_else(|e| {
        panic!("round-trip parse failed: {e}\n---\n{text}\n---");
    });

    // Key fields survive the serialize→parse cycle.
    assert_eq!(rt.size, level.size, "size");
    assert_eq!(rt.nodes.len(), level.nodes.len(), "node count");
    for orig in &level.nodes {
        let ri = rt.node_index(&orig.name).expect("node by name");
        let rn = &rt.nodes[ri];
        assert_eq!(
            (rn.x, rn.y, rn.z),
            (orig.x, orig.y, orig.z),
            "node pos {}",
            orig.name
        );
        assert_eq!(rn.is_input, orig.is_input, "is_input {}", orig.name);
        assert_eq!(rn.is_output, orig.is_output, "is_output {}", orig.name);
    }

    assert_eq!(rt.pipes.len(), level.pipes.len(), "pipe count");
    for (op, rp) in level.pipes.iter().zip(&rt.pipes) {
        assert_eq!(rp.from_node, op.from_node, "pipe from_node");
        assert_eq!(rp.to_node, op.to_node, "pipe to_node");
        assert_eq!(rp.path, op.path, "pipe path");
        assert_eq!(rp.switch_cell, op.switch_cell, "pipe switch index");
    }

    assert_eq!(
        (rt.door.x, rt.door.y, rt.door.z),
        (level.door.x, level.door.y, level.door.z)
    );
    assert_eq!(
        (rt.player.x, rt.player.y, rt.player.z),
        (level.player.x, level.player.y, level.player.z)
    );
}

/// R-M1: the round-trip must also preserve the OPTIONAL data the minimal test
/// skipped — a `wall_trace` decal, `[inputs]`, a palette fog override, and the
/// pedestal volume. Build a level WITH all of them, then
/// vox→Level→to_lvl_text→re_level::load_string and assert they survive.
#[test]
fn round_trip_preserves_decals_inputs_palette_pedestal() {
    // minimal room + a 2x1x2 wall_trace box on the y=0 wall.
    let mut v = minimal_voxels();
    v.push(vx(1, 0, 1, C_WALL_TRACE));
    v.push(vx(2, 0, 1, C_WALL_TRACE));
    v.push(vx(1, 0, 2, C_WALL_TRACE));
    v.push(vx(2, 0, 2, C_WALL_TRACE));

    // cfg with inputs + a palette fog override.
    let cfg: Config = toml::from_str(
        r##"
[meta]
name = "rt_full"
title = "ROUND TRIP"
ambient = "trace"

[[node]]
pos = [1, 2, 2]
name = "in_a"
glyph = "A"
threshold = 1

[[node]]
pos = [4, 2, 2]
name = "out"
glyph = "O"
threshold = 1

[inputs]
in_a = 1

[palette]
fog_color = "#0A0F14"
fog_near = 2.5
fog_far = 9.0
"##,
    )
    .expect("cfg parse");

    let level = derive_level(&v, (6, 6, 4), &cfg).expect("derive");

    // Sanity: the derivation produced the optional data we want to track.
    assert!(
        level
            .decals
            .iter()
            .any(|d| d.surface_kind == re_world::CellKind::WallTrace),
        "expected a wall_trace decal"
    );
    assert!(!level.inputs.is_empty(), "expected inputs");
    assert!(level.palette.has_fog, "expected fog override");

    let text = to_lvl_text(&level);
    let rt = re_level::load_string(&text, "rt_full").unwrap_or_else(|e| {
        panic!("round-trip parse failed: {e}\n---\n{text}\n---");
    });

    // --- wall_trace decal survives (count + bbox) ---
    let orig_trace: Vec<_> = level
        .decals
        .iter()
        .filter(|d| d.surface_kind == re_world::CellKind::WallTrace)
        .collect();
    let rt_trace: Vec<_> = rt
        .decals
        .iter()
        .filter(|d| d.surface_kind == re_world::CellKind::WallTrace)
        .collect();
    assert_eq!(rt_trace.len(), orig_trace.len(), "wall_trace decal count");
    let od = orig_trace[0];
    let rd = rt_trace[0];
    assert_eq!(
        (rd.x0, rd.y0, rd.z0, rd.x1, rd.y1, rd.z1),
        (od.x0, od.y0, od.z0, od.x1, od.y1, od.z1),
        "wall_trace decal region"
    );

    // --- inputs survive ---
    assert_eq!(rt.inputs, level.inputs, "inputs");

    // --- palette fog override survives ---
    assert_eq!(rt.palette.has_fog, level.palette.has_fog, "has_fog");
    assert_eq!(rt.palette.fog_color, level.palette.fog_color, "fog_color");
    assert!(
        (rt.palette.fog_near - level.palette.fog_near).abs() < 1e-6,
        "fog_near"
    );
    assert!(
        (rt.palette.fog_far - level.palette.fog_far).abs() < 1e-6,
        "fog_far"
    );

    // --- pedestal volume survives (the clamp must not corrupt a real volume) ---
    assert_eq!(
        (rt.pedestal.cx, rt.pedestal.cy, rt.pedestal.cz),
        (level.pedestal.cx, level.pedestal.cy, level.pedestal.cz),
        "pedestal center"
    );
    assert_eq!(
        (rt.pedestal.w, rt.pedestal.h, rt.pedestal.d),
        (
            level.pedestal.w.max(1),
            level.pedestal.h.max(1),
            level.pedestal.d.max(1)
        ),
        "pedestal volume"
    );
}

/// The "color = material KEY" pipeline survives vox→Level→to_lvl_text→re_level.
/// A `[[material]]` + `[surface]` binding in the `.toml` becomes a `[material]`
/// section in the `.lvl`, round-trips through the parser, and reaches `build_map`.
#[test]
fn round_trip_preserves_materials() {
    let v = minimal_voxels();
    let cfg: Config = toml::from_str(
        r##"
[meta]
name = "mat_rt"
title = "MATERIAL RT"

[[node]]
pos = [1, 2, 2]
name = "in_a"

[[node]]
pos = [4, 2, 2]
name = "out"

[[material]]
name = "floor_neon"
fg = "#20FF80"
bg = "#04140C"
glyph = "+"
recipe = "grain scale=3.0 speck=0.5 tile_eps=0.08"
rank = 7

[[material]]
name = "wall_pulse"
fg = "#66CCFF"
bg = "#041018"
recipe = "brick aspect=2.0 mortar_w=0.16 grain=0.6"
animation = { kind = "pulse", speed = 0.5, amplitude = 0.18 }

[surface]
floor = "floor_neon"
wall_solid = "wall_pulse"
"##,
    )
    .expect("cfg parse");

    let level = derive_level(&v, (6, 6, 4), &cfg).expect("derive");
    // Derivation captured both materials + both bindings.
    assert_eq!(level.materials.len(), 2);
    assert_eq!(level.surface_materials.floor.as_deref(), Some("floor_neon"));
    assert_eq!(
        level.surface_materials.wall_solid.as_deref(),
        Some("wall_pulse")
    );

    let text = to_lvl_text(&level);
    let rt = re_level::load_string(&text, "mat_rt").unwrap_or_else(|e| {
        panic!("round-trip parse failed: {e}\n---\n{text}\n---");
    });

    // Materials survive the serialize→parse cycle (by name + key fields).
    assert_eq!(rt.materials.len(), 2, "material count survives");
    let neon = rt
        .materials
        .iter()
        .find(|m| m.name == "floor_neon")
        .expect("floor_neon");
    assert_eq!(neon.fg, 0x20FF80, "fg survives");
    assert_eq!(neon.bg, 0x04140C, "bg survives");
    assert_eq!(neon.glyph, Some(u32::from('+')), "glyph survives");
    assert_eq!(neon.rank, 7, "rank survives");
    assert!(
        matches!(neon.recipe, Some(re_world::MaterialRecipe::Grain { .. })),
        "recipe survives"
    );

    let pulse = rt
        .materials
        .iter()
        .find(|m| m.name == "wall_pulse")
        .expect("wall_pulse");
    assert_eq!(
        pulse.anim.kind,
        re_world::MaterialAnimKind::Pulse,
        "animation survives"
    );
    assert!((pulse.anim.speed - 0.5).abs() < 1e-6);

    // Bindings survive → the Map carries the authored, slot-ordered set.
    assert_eq!(rt.surface_materials.floor.as_deref(), Some("floor_neon"));
    assert_eq!(
        rt.surface_materials.wall_solid.as_deref(),
        Some("wall_pulse")
    );
    let map = rt.build_map();
    let mats = map.materials();
    assert_eq!(mats.len(), 5, "5 slot-ordered defs");
    assert_eq!(mats[0].name, "wall_pulse", "wall_solid slot is authored");
    assert_eq!(mats[3].name, "floor_neon", "floor slot is authored");
    assert!(
        mats[1].fallback && mats[2].fallback && mats[4].fallback,
        "unbound slots fall back"
    );
}

/// A `[surface]` binding naming an undefined `[[material]]` is a hard error
/// (a typo must not silently fall back).
#[test]
fn surface_binding_to_unknown_material_is_error() {
    let v = minimal_voxels();
    let cfg: Config = toml::from_str(
        r##"
[meta]
name = "bad_bind"

[[node]]
pos = [1, 2, 2]
name = "in_a"

[[node]]
pos = [4, 2, 2]
name = "out"

[surface]
floor = "does_not_exist"
"##,
    )
    .expect("cfg parse");
    let err = derive_level(&v, (6, 6, 4), &cfg).unwrap_err();
    match err {
        ImportError::Derive(m) => assert!(m.contains("undefined"), "got: {m}"),
        other => panic!("expected Derive error, got {other}"),
    }
}

#[test]
fn pipe_with_one_node_is_error() {
    // A pipe run that only touches ONE node → PipeEndpoints error.
    let v = vec![
        vx(1, 2, 2, C_NODE_IN),
        vx(2, 2, 2, C_PIPE),
        vx(3, 2, 2, C_PIPE), // dead-ends at no second node
        vx(5, 2, 1, C_DOOR),
        vx(3, 2, 1, C_PLAYER),
    ];
    let cfg = minimal_cfg();
    let err = derive_level(&v, (6, 6, 4), &cfg).unwrap_err();
    match err {
        ImportError::PipeEndpoints { found, cells } => {
            assert_eq!(found, 1, "should find exactly one endpoint");
            assert!(!cells.is_empty(), "offending cells reported");
        }
        other => panic!("expected PipeEndpoints, got {other}"),
    }
}

#[test]
fn switch_not_on_pipe_is_error() {
    // Switch placed far from any pipe path → SwitchUnattached.
    let mut v = minimal_voxels();
    // move the switch to an isolated cell.
    for vox in v.iter_mut() {
        if vox.rgb == C_SWITCH {
            vox.x = 5;
            vox.y = 5;
            vox.z = 3;
        }
    }
    let cfg = minimal_cfg();
    let err = derive_level(&v, (6, 6, 4), &cfg).unwrap_err();
    match err {
        ImportError::SwitchUnattached { touching, .. } => {
            assert_eq!(touching, 0);
        }
        other => panic!("expected SwitchUnattached, got {other}"),
    }
}

#[test]
fn unknown_color_becomes_wall_solid() {
    // An unknown colour is NOT an error — it becomes wall_solid (a decal).
    // Use a colour far from every tag.
    let unknown = [0x12, 0x34, 0x56];
    let v = vec![
        vx(1, 2, 2, C_NODE_IN),
        vx(4, 2, 2, C_NODE_OUT),
        vx(2, 2, 2, C_PIPE),
        vx(3, 2, 2, C_PIPE),
        vx(2, 3, 2, C_SWITCH),
        vx(5, 2, 1, C_DOOR),
        vx(3, 2, 1, C_PLAYER),
        vx(1, 4, 2, unknown), // mystery voxel
    ];
    let cfg = minimal_cfg();
    let level = derive_level(&v, (6, 6, 4), &cfg).expect("unknown colour must not error");
    // It is treated as a wall_solid decal.
    let has_solid_decal = level
        .decals
        .iter()
        .any(|d| d.surface_kind == re_world::CellKind::WallSolid);
    assert!(
        has_solid_decal,
        "unknown colour should produce a wall_solid decal"
    );
}

#[test]
fn wall_trace_box_becomes_decal() {
    // A 2x2 block of wall_trace voxels → one decal region (bbox of the run).
    let mut v = minimal_voxels();
    v.push(vx(1, 0, 1, C_WALL_TRACE));
    v.push(vx(2, 0, 1, C_WALL_TRACE));
    v.push(vx(1, 0, 2, C_WALL_TRACE));
    v.push(vx(2, 0, 2, C_WALL_TRACE));
    let cfg = minimal_cfg();
    let level = derive_level(&v, (6, 6, 4), &cfg).expect("derive");
    let trace: Vec<_> = level
        .decals
        .iter()
        .filter(|d| d.surface_kind == re_world::CellKind::WallTrace)
        .collect();
    assert_eq!(trace.len(), 1, "one connected trace run → one decal");
    let d = trace[0];
    assert_eq!((d.x0, d.x1), (1, 2));
    assert_eq!((d.z0, d.z1), (1, 2));
}

#[test]
fn missing_meta_name_is_error() {
    // An absent / empty [meta] name → MissingMeta (the one required field).
    let cfg: Config = toml::from_str("[meta]\ntitle = \"x\"\n").expect("parse");
    assert!(cfg.meta.name.is_empty());
    let err = derive_level(&minimal_voxels(), (6, 6, 4), &cfg).unwrap_err();
    match err {
        ImportError::MissingMeta(_) => {}
        other => panic!("expected MissingMeta, got {other}"),
    }
}

// ---------------------------------------------------------------------------
// lvl2vox / level_to_vox_bytes / level_to_toml round-trip tests
// ---------------------------------------------------------------------------

/// The key round-trip test:
///
///   derive_level(..)  ──►  Level
///        ↓ level_to_vox_bytes + level_to_toml
///   (.vox bytes, .toml str)
///        ↓ vox_to_level
///   Level′
///
/// Assert Level′ matches Level on the STRUCTURAL ESSENTIALS:
///   - room size
///   - node count, positions, roles
///   - pipe count, paths, switch cell indices
///   - door position
///   - player spawn position
#[test]
fn lvl2vox_round_trip_geometry_and_graph() {
    // Build a level using the minimal fixture.
    let v = minimal_voxels();
    let cfg = minimal_cfg();
    let orig = derive_level(&v, (6, 6, 4), &cfg).expect("forward derive");

    // Reverse: Level → .vox bytes + .toml text.
    let vox_bytes = level_to_vox_bytes(&orig);
    let toml_str = level_to_toml(&orig);

    // Verify the .vox is parse-clean.
    let parsed_data =
        dot_vox::load_bytes(&vox_bytes).expect("round-trip .vox must be parseable by dot_vox");
    assert!(!parsed_data.models.is_empty(), "at least one model in .vox");
    let vox_model = &parsed_data.models[0];
    assert!(
        !vox_model.voxels.is_empty(),
        "stamped .vox must contain voxels"
    );

    // Forward again: (.vox, .toml) → Level′.
    let rt = vox_to_level(&vox_bytes, &toml_str).unwrap_or_else(|e| {
        panic!("reverse round-trip failed: {e}\n--- .toml ---\n{toml_str}\n---")
    });

    // Room size.
    assert_eq!(rt.size, orig.size, "room size");

    // Node count, positions, roles.
    assert_eq!(rt.nodes.len(), orig.nodes.len(), "node count");
    for on in &orig.nodes {
        let ri = rt
            .node_index(&on.name)
            .unwrap_or_else(|| panic!("round-trip: node '{}' missing", on.name));
        let rn = &rt.nodes[ri];
        assert_eq!(
            (rn.x, rn.y, rn.z),
            (on.x, on.y, on.z),
            "node pos {}",
            on.name
        );
        assert_eq!(rn.is_input, on.is_input, "is_input {}", on.name);
        assert_eq!(rn.is_output, on.is_output, "is_output {}", on.name);
    }

    // Pipe count, paths, switch cells.
    assert_eq!(rt.pipes.len(), orig.pipes.len(), "pipe count");
    for (op, rp) in orig.pipes.iter().zip(rt.pipes.iter()) {
        assert_eq!(rp.from, op.from, "pipe from name");
        assert_eq!(rp.to, op.to, "pipe to name");
        assert_eq!(rp.path, op.path, "pipe path");
        assert_eq!(rp.switch_cell, op.switch_cell, "pipe switch_cell index");
    }

    // Door + player positions.
    assert_eq!(
        (rt.door.x, rt.door.y, rt.door.z),
        (orig.door.x, orig.door.y, orig.door.z),
        "door pos"
    );
    assert_eq!(
        (rt.player.x, rt.player.y, rt.player.z),
        (orig.player.x, orig.player.y, orig.player.z),
        "player pos"
    );

    // Verify the reconstructed level builds a non-empty map.
    let map = rt.build_map();
    let walls = map.count(re_world::CellKind::WallTrace)
        + map.count(re_world::CellKind::WallSolid)
        + map.count(re_world::CellKind::WallMatrix);
    assert!(walls > 0, "round-trip map must contain walls");
}

/// Round-trip preserves inputs + palette fog + decal regions.
#[test]
fn lvl2vox_round_trip_metadata() {
    let mut v = minimal_voxels();
    // Add a wall_trace patch to get a decal.
    v.push(vx(1, 0, 1, C_WALL_TRACE));
    v.push(vx(2, 0, 1, C_WALL_TRACE));

    let cfg: Config = toml::from_str(
        r##"
[meta]
name = "rt_meta"
title = "METADATA RT"
ambient = "trace"

[[node]]
pos = [1, 2, 2]
name = "in_a"
glyph = "A"
threshold = 1

[[node]]
pos = [4, 2, 2]
name = "out"
glyph = "O"
threshold = 1

[inputs]
in_a = 1

[palette]
fog_color = "#101826"
fog_near = 2.0
fog_far = 14.0
"##,
    )
    .expect("cfg parse");

    let orig = derive_level(&v, (6, 6, 4), &cfg).expect("forward derive");
    let vox_bytes = level_to_vox_bytes(&orig);
    let toml_str = level_to_toml(&orig);

    let rt = vox_to_level(&vox_bytes, &toml_str)
        .unwrap_or_else(|e| panic!("reverse round-trip failed: {e}"));

    // inputs preserved.
    assert_eq!(rt.inputs, orig.inputs, "inputs");

    // palette fog preserved.
    assert_eq!(rt.palette.has_fog, orig.palette.has_fog, "has_fog");
    assert_eq!(rt.palette.fog_color, orig.palette.fog_color, "fog_color");
    assert!(
        (rt.palette.fog_near - orig.palette.fog_near).abs() < 1e-5,
        "fog_near"
    );
    assert!(
        (rt.palette.fog_far - orig.palette.fog_far).abs() < 1e-5,
        "fog_far"
    );

    // ambient preserved.
    assert_eq!(rt.ambient, orig.ambient, "ambient");

    // wall_trace decals: after round-trip through .vox the exact bbox depends on
    // the re-derived runs (the vox stamps all cells in the decal bbox, so the
    // run is a single contiguous box → one decal). Count must survive.
    let orig_trace = orig
        .decals
        .iter()
        .filter(|d| d.surface_kind == re_world::CellKind::WallTrace)
        .count();
    let rt_trace = rt
        .decals
        .iter()
        .filter(|d| d.surface_kind == re_world::CellKind::WallTrace)
        .count();
    assert_eq!(rt_trace, orig_trace, "wall_trace decal count");
}

/// The .vox bytes produced by `level_to_vox_bytes` use the exact same 13-tag
/// palette RGBs as the forward path so `vox_to_level` re-derives without
/// needing `[tags]` overrides.
#[test]
fn lvl2vox_palette_uses_canonical_tag_rgbs() {
    let orig = derive_level(&minimal_voxels(), (6, 6, 4), &minimal_cfg()).expect("derive");
    let vox_bytes = level_to_vox_bytes(&orig);
    let data = dot_vox::load_bytes(&vox_bytes).expect("parse");

    // Palette slot 5 (node_input) must be #FFD000.
    let pal = &data.palette;
    assert!(pal.len() >= 13, "palette must have ≥13 entries");
    let inp = &pal[5]; // node_input index = 5
    assert_eq!(
        (inp.r, inp.g, inp.b),
        (0xFF, 0xD0, 0x00),
        "node_input palette RGB"
    );
    let out_c = &pal[6]; // node_output index = 6
    assert_eq!(
        (out_c.r, out_c.g, out_c.b),
        (0xFF, 0x30, 0x30),
        "node_output palette RGB"
    );
    let pipe_c = &pal[8]; // pipe index = 8
    assert_eq!(
        (pipe_c.r, pipe_c.g, pipe_c.b),
        (0x80, 0x00, 0xFF),
        "pipe palette RGB"
    );
}

/// `level_to_vox_bytes` is deterministic: same Level → identical bytes.
#[test]
fn lvl2vox_is_deterministic() {
    let orig = derive_level(&minimal_voxels(), (6, 6, 4), &minimal_cfg()).expect("derive");
    let a = level_to_vox_bytes(&orig);
    let b = level_to_vox_bytes(&orig);
    assert_eq!(a, b, "level_to_vox_bytes must be deterministic");
}

/// `level_to_toml` round-trips through the TOML parser (verifies the emitted
/// TOML is syntactically valid and the Config parses cleanly).
#[test]
fn lvl2vox_toml_parses_cleanly() {
    let orig = derive_level(&minimal_voxels(), (6, 6, 4), &minimal_cfg()).expect("derive");
    let toml_str = level_to_toml(&orig);
    let cfg: Config = toml::from_str(&toml_str).unwrap_or_else(|e| {
        panic!("level_to_toml emitted invalid TOML: {e}\n---\n{toml_str}\n---")
    });
    assert_eq!(cfg.meta.name, orig.name, "meta.name");
    assert_eq!(cfg.nodes.len(), orig.nodes.len(), "node count in toml");
    assert_eq!(
        cfg.edges.len(),
        orig.pipes.len(),
        "edge count == pipe count"
    );
}

// ---------------------------------------------------------------------------
// Hologram / Portal / Object dedicated-tag round-trip (R-NEW1)
// ---------------------------------------------------------------------------

/// The headline guarantee for the new dedicated tags:
///
///   voxel grid (Hologram + Portal + Object + surface kinds)
///        ↓ level_to_vox_bytes (Hologram→tag 13, Portal→14, Object→15)
///   .vox bytes
///        ↓ vox_to_level (tag 13→Hologram decal, 14→Portal decal, 15→Object decal)
///   Level
///        ↓ build_map
///   Map
///
/// After `build_map`, each Hologram/Portal/Object cell retains its kind;
/// wall/floor/ceiling surface kinds are unchanged.
#[test]
fn new_tags_round_trip_hologram_portal_object() {
    // Build a tiny level containing one cell of each new kind, plus the
    // existing three surface kinds, in a 10×10×5 room.
    const C_HOLOGRAM: [u8; 3] = [0x40, 0xFF, 0xFF];
    const C_PORTAL: [u8; 3] = [0xFF, 0x40, 0xC0];
    const C_OBJECT: [u8; 3] = [0xC0, 0x60, 0x00];
    const C_WALL: [u8; 3] = [0x80, 0x80, 0x80];
    const C_FLOOR: [u8; 3] = [0x40, 0x40, 0x40];
    const C_CEILING: [u8; 3] = [0x20, 0x20, 0x20];
    const C_NODE_IN: [u8; 3] = [0xFF, 0xD0, 0x00];
    const C_NODE_OUT: [u8; 3] = [0xFF, 0x30, 0x30];
    const C_PIPE: [u8; 3] = [0x80, 0x00, 0xFF];
    const C_DOOR: [u8; 3] = [0x00, 0xFF, 0xFF];
    const C_PLAYER: [u8; 3] = [0xFF, 0xA5, 0x00];

    let voxels = vec![
        // New geometry kinds.
        vx(2, 2, 1, C_HOLOGRAM),
        vx(3, 2, 1, C_HOLOGRAM),
        vx(2, 4, 1, C_PORTAL),
        vx(2, 6, 1, C_OBJECT),
        // Surface kinds (used in decals so they are visible in the map).
        vx(1, 0, 1, C_WALL),
        vx(1, 1, 0, C_FLOOR),
        vx(1, 1, 4, C_CEILING),
        // Graph elements (required for a valid level).
        vx(4, 2, 2, C_NODE_IN),
        vx(6, 2, 2, C_NODE_OUT),
        vx(5, 2, 2, C_PIPE),
        vx(7, 4, 1, C_DOOR),
        vx(5, 4, 1, C_PLAYER),
    ];

    let cfg: Config = toml::from_str(
        r#"
[meta]
name = "new_tags_rt"
title = "NEW TAGS ROUND-TRIP"

[[node]]
pos = [4, 2, 2]
name = "in_a"

[[node]]
pos = [6, 2, 2]
name = "out_b"
"#,
    )
    .expect("cfg parse");

    let level = derive_level(&voxels, (10, 10, 5), &cfg).expect("derive_level must succeed");

    // Forward: Level → .vox bytes.
    let vox_bytes = level_to_vox_bytes(&level);

    // Verify the raw palette: slots 13/14/15 must be the new tag RGBs.
    let parsed = dot_vox::load_bytes(&vox_bytes).expect("dot_vox parse");
    let pal = &parsed.palette;
    assert!(pal.len() >= 16, "palette must have ≥16 entries");
    assert_eq!(
        (pal[13].r, pal[13].g, pal[13].b),
        (0x40, 0xFF, 0xFF),
        "hologram palette RGB"
    );
    assert_eq!(
        (pal[14].r, pal[14].g, pal[14].b),
        (0xFF, 0x40, 0xC0),
        "portal palette RGB"
    );
    assert_eq!(
        (pal[15].r, pal[15].g, pal[15].b),
        (0xC0, 0x60, 0x00),
        "object palette RGB"
    );

    // The .vox model must contain voxels at the palette indices 13/14/15.
    let model = &parsed.models[0];
    let count = |idx: u8| model.voxels.iter().filter(|v| v.i == idx).count();
    assert!(
        count(13) >= 2,
        "at least 2 hologram cells in .vox (palette 13)"
    );
    assert_eq!(count(14), 1, "1 portal cell in .vox (palette 14)");
    assert_eq!(count(15), 1, "1 object cell in .vox (palette 15)");

    // Reverse: .vox bytes → Level′ → Map.
    let toml_str = level_to_toml(&level);
    let rt = vox_to_level(&vox_bytes, &toml_str).unwrap_or_else(|e| {
        panic!("reverse round-trip failed: {e}\n--- toml ---\n{toml_str}\n---")
    });

    let map = rt.build_map();

    // Hologram cells survive.
    assert_eq!(
        map.cell_at(2, 2, 1),
        re_world::CellKind::Hologram,
        "Hologram cell at (2,2,1) must survive round-trip"
    );
    assert_eq!(
        map.cell_at(3, 2, 1),
        re_world::CellKind::Hologram,
        "Hologram cell at (3,2,1) must survive round-trip"
    );

    // Portal cells survive.
    assert_eq!(
        map.cell_at(2, 4, 1),
        re_world::CellKind::Portal,
        "Portal cell at (2,4,1) must survive round-trip"
    );

    // Object cells survive (the generic painted Object, side_data=0).
    assert_eq!(
        map.cell_at(2, 6, 1),
        re_world::CellKind::Object,
        "Object cell at (2,6,1) must survive round-trip"
    );

    // Surface kinds are unchanged.
    assert_eq!(
        map.cell_at(1, 0, 1),
        re_world::CellKind::WallSolid,
        "WallSolid decal survives"
    );
    // Hologram flag is set on the map.
    assert!(
        map.has_holograms(),
        "map must report has_holograms after round-trip"
    );
}

/// The `cell_kind_to_palette_index` public helper is the single source of
/// truth: for every non-Empty CellKind it returns the palette index that
/// `vox_to_level` will re-derive to the same kind (lossless for the new tags).
#[test]
fn cell_kind_to_palette_index_is_canonical() {
    use re_world::CellKind;
    // Empty → None (no voxel emitted).
    assert_eq!(cell_kind_to_palette_index(CellKind::Empty, 0), None);
    // Surface kinds → their canonical indices.
    assert_eq!(cell_kind_to_palette_index(CellKind::WallSolid, 0), Some(0));
    assert_eq!(cell_kind_to_palette_index(CellKind::WallTrace, 0), Some(1));
    assert_eq!(cell_kind_to_palette_index(CellKind::WallMatrix, 0), Some(2));
    assert_eq!(cell_kind_to_palette_index(CellKind::Floor, 0), Some(3));
    assert_eq!(cell_kind_to_palette_index(CellKind::Ceiling, 0), Some(4));
    // New dedicated geometry tags.
    assert_eq!(cell_kind_to_palette_index(CellKind::Hologram, 0), Some(13));
    assert_eq!(cell_kind_to_palette_index(CellKind::Portal, 0), Some(14));
    // Object with side_data=0 (painted generic) → object tag (index 15).
    assert_eq!(cell_kind_to_palette_index(CellKind::Object, 0), Some(15));
    // Object with side_data≠0 (painted node) → node_hidden (index 7).
    assert_eq!(cell_kind_to_palette_index(CellKind::Object, 1), Some(7));
}

/// A `[portal]` section round-trips losslessly through
/// re_level::load_string → to_lvl_text → re_level::load_string: the regions,
/// rotate, and skin survive byte-for-byte (the editor depends on this).
#[test]
fn round_trip_preserves_portal_pair() {
    let src = "\
[meta]
name = wp_rt
[room]
size = 12 16 10
ambient = solid
[portal gate]
region_a = (1,0,1)-(10,0,8)
region_b = (0,1,1)-(0,10,8)
rotate = 90
skin = portal_spiral
[node a]
pos = 4,13,2
is_input = yes
[node b]
pos = 7,13,2
is_output = yes
threshold = 1
[pipe a->b]
path = (4,13,2) (7,13,2)
[pedestal]
center = 7 13 1
volume = 1 1 1
[door]
pos = 7 14 1
target_parts = b
[player]
spawn = 6 14 1
facing = -y
";
    let level = re_level::load_string(src, "wp_rt").expect("portal level must parse");
    assert_eq!(level.portals.len(), 1);

    let text = to_lvl_text(&level);
    assert!(
        text.contains("[portal gate]"),
        "emitted text must carry [portal gate]:\n{text}"
    );

    let rt = re_level::load_string(&text, "wp_rt2").unwrap_or_else(|e| {
        panic!("round-trip parse failed: {e}\n---\n{text}\n---");
    });
    assert_eq!(rt.portals.len(), 1, "portal survives round-trip");
    let (a, b) = (&level.portals[0], &rt.portals[0]);
    assert_eq!(a.name, b.name);
    assert_eq!(a.region_a(), b.region_a());
    assert_eq!(a.region_b(), b.region_b());
    assert_eq!(a.rotate, b.rotate);
    assert_eq!(a.skin, b.skin);

    // Idempotent: a second emit equals the first.
    assert_eq!(
        text,
        to_lvl_text(&rt),
        "to_lvl_text must be idempotent for portals"
    );

    // The rebuilt map carries the reciprocal pair.
    assert_eq!(rt.build_map().portals.len(), 2);
}

/// A `kind = GodRays` `[emitter]` survives re_level::load_string → to_lvl_text →
/// re_level::load_string: the kind (and the surrounding fields) round-trip. A
/// `Standard` emitter omits the `kind` key entirely (back-compat / compactness).
#[test]
fn round_trip_preserves_emitter_godrays_kind() {
    use re_world::EmitterKind;
    let src = "\
[meta]
name = gr_rt
[room]
size = 8 8 6
ambient = solid
[emitter shaft]
kind = godrays
origin_world = 3.5 2.0 4.0
direction = 0.2 0.6 -1.0
colors = 0xFFF6E0 0xFFD27A 0xC8841E
rate = 36.0
speed = 1.1
[emitter motes]
origin_world = 1.0 1.0 1.0
[node a]
pos = 3,6,2
is_input = yes
[node b]
pos = 5,6,2
is_output = yes
threshold = 1
[pipe a->b]
path = (3,6,2) (5,6,2)
[pedestal]
center = 5 6 1
volume = 1 1 1
[door]
pos = 5 7 1
target_parts = b
[player]
spawn = 4 4 1
facing = +y
";
    let level = re_level::load_string(src, "gr_rt").expect("godrays level must parse");
    assert_eq!(level.emitters.len(), 2);
    // The first emitter parsed as GodRays (case-insensitive key value), the second
    // defaulted to Standard.
    assert_eq!(level.emitters[0].kind, EmitterKind::GodRays);
    assert_eq!(level.emitters[1].kind, EmitterKind::Standard);

    let text = to_lvl_text(&level);
    // The GodRays emitter writes `kind = GodRays`; the Standard one omits the key.
    assert!(
        text.contains("kind = GodRays"),
        "emitted text must carry the GodRays kind:\n{text}"
    );
    assert_eq!(
        text.matches("kind = ").count(),
        1,
        "only the non-default (GodRays) emitter writes a kind key:\n{text}"
    );

    let rt = re_level::load_string(&text, "gr_rt2")
        .unwrap_or_else(|e| panic!("round-trip parse failed: {e}\n---\n{text}\n---"));
    assert_eq!(rt.emitters.len(), 2, "both emitters survive round-trip");
    assert_eq!(
        rt.emitters[0].kind,
        EmitterKind::GodRays,
        "GodRays kind survives"
    );
    assert_eq!(
        rt.emitters[1].kind,
        EmitterKind::Standard,
        "Standard default survives"
    );

    // Idempotent: a second emit equals the first.
    assert_eq!(
        text,
        to_lvl_text(&rt),
        "to_lvl_text must be idempotent for emitter kind"
    );
}

// ---------------------------------------------------------------------------
// [[color_material]] per-voxel material-id tests (Rule 9)
// ---------------------------------------------------------------------------

/// Two geometry voxels of different colours, each matching a distinct
/// `[[color_material]]` entry, must receive DIFFERENT per-cell material ids.
///
/// Layout: two wall_solid voxels at different positions, painted with
/// `#AA0000` and `#00AA00` respectively.  The .toml defines two
/// `[[color_material]]` entries keyed to those colours.  After
/// `derive_level_full` each voxel must carry its own 1-based id.
#[test]
fn color_material_two_colors_get_different_ids() {
    // Two wall_solid voxels at distinct positions with distinct colours.
    // Both colours are far from every tag colour (no tag-collision risk).
    let v1_rgb = [0xAA, 0x00, 0x00]; // deep red — no tag is this colour
    let v2_rgb = [0x00, 0xAA, 0x00]; // deep green — no tag is this colour
    let voxels = vec![
        // The two coloured walls.
        vx(1, 1, 1, v1_rgb),
        vx(3, 1, 1, v2_rgb),
        // Mandatory graph elements (minimal valid level).
        vx(1, 2, 2, C_NODE_IN),
        vx(4, 2, 2, C_NODE_OUT),
        vx(2, 2, 2, C_PIPE),
        vx(3, 2, 2, C_PIPE),
        vx(2, 3, 2, C_SWITCH),
        vx(5, 2, 1, C_DOOR),
        vx(3, 2, 1, C_PLAYER),
    ];

    let cfg: Config = toml::from_str(
        r##"
[meta]
name = "cm_test"
title = "COLOR MATERIAL TEST"

[[node]]
pos = [1, 2, 2]
name = "in_a"
threshold = 1

[[node]]
pos = [4, 2, 2]
name = "out"
threshold = 1

[[color_material]]
color = "#AA0000"
name = "red_wall"
fg = "#FF4444"
bg = "#200000"
recipe = "brick"

[[color_material]]
color = "#00AA00"
name = "green_wall"
fg = "#44FF44"
bg = "#002000"
recipe = "grain"
"##,
    )
    .expect("cfg parse");

    let result = derive_level_full(&voxels, (6, 6, 4), &cfg).expect("derive_level_full");

    // Both colour-materials were promoted into level.materials.
    assert_eq!(
        result.level.materials.len(),
        2,
        "two colour-materials become level materials"
    );
    let has_red = result.level.materials.iter().any(|m| m.name == "red_wall");
    let has_green = result
        .level
        .materials
        .iter()
        .any(|m| m.name == "green_wall");
    assert!(has_red, "red_wall material must be in level.materials");
    assert!(has_green, "green_wall material must be in level.materials");

    // Each geometry voxel carries its own 1-based id.
    let id1 = result.cell_material_ids.get(&(1, 1, 1)).copied();
    let id2 = result.cell_material_ids.get(&(3, 1, 1)).copied();
    assert!(
        id1.is_some(),
        "voxel at (1,1,1) must have a colour-material id"
    );
    assert!(
        id2.is_some(),
        "voxel at (3,1,1) must have a colour-material id"
    );
    assert_ne!(id1, id2, "the two voxels must have DIFFERENT material ids");

    // The ids must be 1-based (1 = first [[color_material]], 2 = second).
    // The first [[color_material]] (#AA0000 → "red_wall") is id=1;
    // the second (#00AA00 → "green_wall") is id=2.
    assert!(id1.unwrap() == 1 || id1.unwrap() == 2, "id must be 1 or 2");
    assert!(id2.unwrap() == 1 || id2.unwrap() == 2, "id must be 1 or 2");
    // They are different, so together they are {1, 2}.
    let id_set: std::collections::HashSet<u16> = [id1.unwrap(), id2.unwrap()].into_iter().collect();
    assert_eq!(
        id_set,
        [1u16, 2u16].into_iter().collect(),
        "ids must be exactly {{1, 2}}"
    );

    // A graph-element voxel (node_input at (1,2,2)) must NOT get a colour-material id.
    assert!(
        !result.cell_material_ids.contains_key(&(1, 2, 2)),
        "graph-element voxels (nodes) must not receive a colour-material id"
    );
}

/// The `to_lvl_text_with_ids` serializer emits a `[color_cell_ids]` section
/// that `re_level` silently ignores (round-trip: the Level survives intact) and
/// that contains the correct per-cell assignments.
#[test]
fn color_material_lvl_text_round_trip() {
    let v1_rgb = [0xAA, 0x00, 0x00];
    let v2_rgb = [0x00, 0xAA, 0x00];
    let voxels = vec![
        vx(1, 1, 1, v1_rgb),
        vx(3, 1, 1, v2_rgb),
        vx(1, 2, 2, C_NODE_IN),
        vx(4, 2, 2, C_NODE_OUT),
        vx(2, 2, 2, C_PIPE),
        vx(3, 2, 2, C_PIPE),
        vx(2, 3, 2, C_SWITCH),
        vx(5, 2, 1, C_DOOR),
        vx(3, 2, 1, C_PLAYER),
    ];

    let cfg: Config = toml::from_str(
        r##"
[meta]
name = "cm_lvl_rt"
title = "CM LVL ROUND TRIP"

[[node]]
pos = [1, 2, 2]
name = "in_a"
threshold = 1

[[node]]
pos = [4, 2, 2]
name = "out"
threshold = 1

[[color_material]]
color = "#AA0000"
name = "red_wall"
fg = "#FF4444"
bg = "#200000"

[[color_material]]
color = "#00AA00"
name = "green_wall"
fg = "#44FF44"
bg = "#002000"
"##,
    )
    .expect("cfg parse");

    let result = derive_level_full(&voxels, (6, 6, 4), &cfg).expect("derive_level_full");

    // Serialise with cell ids.
    let text = to_lvl_text_with_ids(&result.level, &result.cell_material_ids);

    // The emitted text must contain a [color_cell_ids] section.
    assert!(
        text.contains("[color_cell_ids]"),
        "emitted .lvl must contain [color_cell_ids] section:\n{text}"
    );
    // And it must mention both cell positions.
    assert!(
        text.contains("1,1,1 ="),
        "must contain assignment for (1,1,1):\n{text}"
    );
    assert!(
        text.contains("3,1,1 ="),
        "must contain assignment for (3,1,1):\n{text}"
    );

    // re_level must round-trip the Level successfully (the unknown section is ignored).
    let rt = re_level::load_string(&text, "cm_lvl_rt").unwrap_or_else(|e| {
        panic!("round-trip parse failed: {e}\n---\n{text}\n---");
    });

    // The Level data survives the round-trip.
    assert_eq!(rt.name, "cm_lvl_rt");
    assert_eq!(rt.size, result.level.size);
    assert_eq!(rt.nodes.len(), 2);

    // Both colour-materials survive as named [material] sections.
    assert_eq!(
        rt.materials.len(),
        2,
        "two named materials must survive round-trip"
    );
    let red = rt
        .materials
        .iter()
        .find(|m| m.name == "red_wall")
        .expect("red_wall");
    assert_eq!(red.fg, 0xFF4444);
    let green = rt
        .materials
        .iter()
        .find(|m| m.name == "green_wall")
        .expect("green_wall");
    assert_eq!(green.fg, 0x44FF44);
}

/// A voxel whose colour matches neither a tag nor a `[[color_material]]` entry
/// gets no material id (falls back to surface-kind default).
#[test]
fn color_material_no_match_gets_no_id() {
    let no_match_rgb = [0x12, 0x34, 0x56]; // far from all tags and color_materials
    let voxels = vec![
        vx(1, 1, 1, no_match_rgb), // unknown → wall_solid decal
        vx(1, 2, 2, C_NODE_IN),
        vx(4, 2, 2, C_NODE_OUT),
        vx(2, 2, 2, C_PIPE),
        vx(3, 2, 2, C_PIPE),
        vx(2, 3, 2, C_SWITCH),
        vx(5, 2, 1, C_DOOR),
        vx(3, 2, 1, C_PLAYER),
    ];

    // Single color_material that does NOT match no_match_rgb.
    let cfg: Config = toml::from_str(
        r##"
[meta]
name = "cm_no_match"

[[node]]
pos = [1, 2, 2]
name = "in_a"
threshold = 1

[[node]]
pos = [4, 2, 2]
name = "out"
threshold = 1

[[color_material]]
color = "#AA0000"
name = "red_wall"
fg = "#FF4444"
bg = "#200000"
"##,
    )
    .expect("cfg parse");

    let result = derive_level_full(&voxels, (6, 6, 4), &cfg).expect("derive_level_full");

    // The no_match voxel must not appear in cell_material_ids.
    assert!(
        !result.cell_material_ids.contains_key(&(1, 1, 1)),
        "voxel with no colour-material match must not have a material id"
    );
}

/// Duplicate `[[color_material]]` names (from identical colours) are a hard error.
#[test]
fn color_material_duplicate_name_is_error() {
    let voxels = minimal_voxels();
    let cfg: Config = toml::from_str(
        r##"
[meta]
name = "cm_dup"

[[node]]
pos = [1, 2, 2]
name = "in_a"

[[node]]
pos = [4, 2, 2]
name = "out"

[[color_material]]
color = "#AA0000"

[[color_material]]
color = "#AA0000"
"##,
    )
    .expect("cfg parse");

    let err = derive_level_full(&voxels, (6, 6, 4), &cfg).unwrap_err();
    match err {
        ImportError::Derive(m) => assert!(
            m.contains("duplicate"),
            "expected 'duplicate' in error, got: {m}"
        ),
        other => panic!("expected Derive error, got {other}"),
    }
}

/// Auto-name: when `[[color_material]]` has no explicit `name`, the material is
/// named `cm_RRGGBB` derived from the `color` hex string.
#[test]
fn color_material_auto_name() {
    let v_rgb = [0xBB, 0x22, 0x44]; // unique colour
    let voxels: Vec<Voxel> = vec![
        vx(2, 1, 1, v_rgb),
        vx(1, 2, 2, C_NODE_IN),
        vx(4, 2, 2, C_NODE_OUT),
        vx(2, 2, 2, C_PIPE),
        vx(3, 2, 2, C_PIPE),
        vx(2, 3, 2, C_SWITCH),
        vx(5, 2, 1, C_DOOR),
        vx(3, 2, 1, C_PLAYER),
    ];
    let cfg: Config = toml::from_str(
        r##"
[meta]
name = "cm_autoname"

[[node]]
pos = [1, 2, 2]
name = "in_a"
threshold = 1

[[node]]
pos = [4, 2, 2]
name = "out"
threshold = 1

[[color_material]]
color = "#BB2244"
fg = "#FF8899"
"##,
    )
    .expect("cfg parse");

    let result = derive_level_full(&voxels, (6, 6, 4), &cfg).expect("derive_level_full");

    // The auto-name must be cm_BB2244.
    assert!(
        result.level.materials.iter().any(|m| m.name == "cm_BB2244"),
        "auto-name must be cm_BB2244; materials: {:?}",
        result
            .level
            .materials
            .iter()
            .map(|m| &m.name)
            .collect::<Vec<_>>()
    );
    // The voxel at (2,1,1) has id=1.
    assert_eq!(result.cell_material_ids.get(&(2, 1, 1)).copied(), Some(1));
}

/// `to_lvl_text_with_ids` on a Level with no colour-material ids produces
/// identical output to `to_lvl_text` (back-compat: no `[color_cell_ids]`
/// section emitted).
#[test]
fn to_lvl_text_with_empty_ids_equals_to_lvl_text() {
    let level = derive_level(&minimal_voxels(), (6, 6, 4), &minimal_cfg()).expect("derive");
    let plain = to_lvl_text(&level);
    let with_ids = to_lvl_text_with_ids(&level, &std::collections::BTreeMap::new());
    assert_eq!(
        plain, with_ids,
        "empty ids must produce identical output to to_lvl_text"
    );
}

// ===========================================================================
// [cells] geometry section — lossless write round-trip (spec §5)
// ===========================================================================

fn cells_shell(body: &str) -> String {
    format!(
        "[meta]\nname = cells-test\n[room]\nsize = 3 2 2\nambient = solid\n\
         [pedestal]\ncenter = 1 1 1\nvolume = 1 1 1\n[door]\npos = 2 1 1\n\
         [player]\nspawn = 1 1 1\n{body}"
    )
}

/// load → to_lvl_text → load → to_lvl_text is byte-identical, and the grid
/// (kinds + ids) survives the round-trip.
#[test]
fn cells_write_round_trip_byte_identical() {
    let body = "\
[cells]
size = 3,2,2
_O_
.H.

#/#
%@%
material_ids =
1,0,0 = 2
2,1,1 = 9
";
    let lv = re_level::load_string(&cells_shell(body), "<rt>").expect("parses");
    let t1 = to_lvl_text(&lv);
    assert!(t1.contains("[cells]"), "[cells] emitted");
    let lv2 = re_level::load_string(&t1, "<rt2>").expect("re-parse serialized text");
    let t2 = to_lvl_text(&lv2);
    assert_eq!(t1, t2, "[cells] re-serialization is byte-identical");
    assert_eq!(lv.cells, lv2.cells, "CellGrid survives the round-trip");
}

/// A level with NO [cells] section serializes without one (back-compat).
#[test]
fn no_cells_serializes_without_section() {
    let level = derive_level(&minimal_voxels(), (6, 6, 4), &minimal_cfg()).expect("derive");
    assert!(level.cells.is_none());
    let t = to_lvl_text(&level);
    assert!(
        !t.contains("[cells]"),
        "no [cells] emitted for a cell-less level"
    );
}

/// The M3c `shapes =` sub-table (readable direction names) round-trips through
/// the writer: parse → to_lvl_text → parse yields identical shapes, and the
/// emitted text uses readable names.
#[test]
fn cells_shapes_write_round_trip() {
    use re_level::CellShape as Shape;
    use re_world::{DiagXY, RampDir};
    let body = "\
[cells]
size = 2,2,1
##
##
shapes =
0 0 0 wedgexy nw
1 1 0 rampz posy
";
    let lv = re_level::load_string(&cells_shell(body), "<sh>").expect("parses");
    let t1 = to_lvl_text(&lv);
    assert!(
        t1.contains("wedgexy nw"),
        "readable wedge dir emitted:\n{t1}"
    );
    assert!(
        t1.contains("rampz posy"),
        "readable ramp dir emitted:\n{t1}"
    );
    let lv2 = re_level::load_string(&t1, "<sh2>").expect("re-parse");
    assert_eq!(
        to_lvl_text(&lv2),
        t1,
        "shapes re-serialization byte-identical"
    );
    assert_eq!(lv.cells, lv2.cells);
    // The grid carries the exact orientations.
    let g = lv.cells.as_ref().unwrap();
    assert_eq!(
        g.shapes,
        vec![
            (0, 0, 0, Shape::WedgeXY(DiagXY::NW)),
            (1, 1, 0, Shape::RampZ(RampDir::PosY)),
        ]
    );
}

/// All four wedge orientations and all four ramp orientations survive
/// parse → to_lvl_text → parse with their exact `re_core::Shape` values.
#[test]
fn cells_shapes_all_orientations_round_trip() {
    use re_level::CellShape as Shape;
    use re_world::{DiagXY, RampDir};
    let body = "\
[cells]
size = 4,2,1
####
####
shapes =
0 0 0 wedgexy ne
1 0 0 wedgexy nw
2 0 0 wedgexy se
3 0 0 wedgexy sw
0 1 0 rampz posx
1 1 0 rampz negx
2 1 0 rampz posy
3 1 0 rampz negy
";
    let lv = re_level::load_string(&cells_shell(body), "<all>").expect("parses");
    let g = lv.cells.as_ref().unwrap();
    assert_eq!(
        g.shapes,
        vec![
            (0, 0, 0, Shape::WedgeXY(DiagXY::NE)),
            (1, 0, 0, Shape::WedgeXY(DiagXY::NW)),
            (2, 0, 0, Shape::WedgeXY(DiagXY::SE)),
            (3, 0, 0, Shape::WedgeXY(DiagXY::SW)),
            (0, 1, 0, Shape::RampZ(RampDir::PosX)),
            (1, 1, 0, Shape::RampZ(RampDir::NegX)),
            (2, 1, 0, Shape::RampZ(RampDir::PosY)),
            (3, 1, 0, Shape::RampZ(RampDir::NegY)),
        ]
    );
    let lv2 = re_level::load_string(&to_lvl_text(&lv), "<all2>").expect("re-parse");
    assert_eq!(
        lv.cells, lv2.cells,
        "all 8 orientations survive the round-trip"
    );
}

/// A shape-less [cells] level emits NO `shapes =` sub-table (back-compat).
#[test]
fn shapeless_cells_emit_no_shapes_table() {
    let body = "[cells]\nsize = 2,2,1\n##\n##\n";
    let lv = re_level::load_string(&cells_shell(body), "<noshape>").expect("parses");
    let t = to_lvl_text(&lv);
    assert!(t.contains("[cells]"), "[cells] present");
    assert!(
        !t.contains("shapes ="),
        "no shapes table for an all-Full grid:\n{t}"
    );
}

/// M5c: a `[vox]` stamp survives parse → to_lvl_text → parse with its file /
/// offset / material / scale intact. The writer is the exact inverse of the
/// `[vox]` parser (the gap that silently dropped vox stamps on save).
#[test]
fn vox_section_write_round_trip() {
    use re_level::VoxMaterial;
    let body = "\
[vox tower]
file = levels/ExampleTower.vox
offset = 3,4,1
material = trace
scale = 2
";
    let lv = re_level::load_string(&cells_shell(body), "<vox>").expect("parses");
    assert_eq!(lv.voxes.len(), 1, "one vox parsed");
    let t1 = to_lvl_text(&lv);
    assert!(t1.contains("[vox tower]"), "[vox] section emitted:\n{t1}");
    assert!(
        t1.contains("file = levels/ExampleTower.vox"),
        "file emitted"
    );
    assert!(t1.contains("offset = 3,4,1"), "offset emitted");
    assert!(
        t1.contains("material = trace"),
        "non-default material emitted"
    );
    assert!(t1.contains("scale = 2"), "non-unit scale emitted");

    let lv2 = re_level::load_string(&t1, "<vox2>").expect("re-parse");
    assert_eq!(lv2.voxes.len(), 1, "vox survives the round-trip");
    let v = &lv2.voxes[0];
    assert_eq!(v.name, "tower");
    assert_eq!(v.file, "levels/ExampleTower.vox");
    assert_eq!((v.off_x, v.off_y, v.off_z), (3, 4, 1));
    assert_eq!(v.material, VoxMaterial::Trace);
    assert_eq!(v.scale, 2);
    // And the serialization is idempotent.
    assert_eq!(to_lvl_text(&lv2), t1, "vox re-serialization byte-identical");
}

/// The two rungs added to the sub-cell ladder — `half` (2×1) and `sextant` (3×2) —
/// survive parse → `to_lvl_text` → parse.
///
/// The writer is an exhaustive `match`, so a variant added with NO arm is a compile
/// error. A variant written with the WRONG KEYWORD is not — it round-trips into a
/// different mode, or fails to re-parse, and either way the level silently changes
/// the next time the editor saves. That is the failure this test exists for; it is the
/// same class as the B1 drop the test below guards.
#[test]
fn subcell_half_and_sextant_round_trip() {
    use re_world::SubcellMode;
    let body = "\
[material img_a]
recipe = image
texture = t
subcell = half
surface = wall_solid
[material img_b]
recipe = image
texture = t
subcell = sextant
surface = floor
[texture t]
file = assets/texture/genplate_tex.png
";
    let lv = re_level::load_string(&cells_shell(body), "<ladder>").expect("parses");
    assert_eq!(lv.materials[0].image_subcell, SubcellMode::Half);
    assert_eq!(lv.materials[1].image_subcell, SubcellMode::Sextant);

    let t = to_lvl_text(&lv);
    assert!(t.contains("subcell = half"), "half emitted:\n{t}");
    assert!(t.contains("subcell = sextant"), "sextant emitted:\n{t}");

    let back = re_level::load_string(&t, "<ladder2>").expect("re-parse");
    assert_eq!(
        back.materials[0].image_subcell,
        SubcellMode::Half,
        "half must not drift through a write/read cycle"
    );
    assert_eq!(
        back.materials[1].image_subcell,
        SubcellMode::Sextant,
        "sextant must not drift through a write/read cycle"
    );
}

/// IMAGE TEXTURES: a `[texture NAME] file = …` section and an `[material] recipe =
/// image texture = NAME` survive `to_lvl_text` → parse with the path + binding
/// preserved (the decoded buffer is NOT serialized — only the path). Guards against
/// the "parses but never serializes" silent-drop bug — INCLUDING the M3a `span` and
/// M3b `subcell` keys (B1: previously parsed but never serialized, so a TX2-TX4
/// world-anchored / sub-cell textured wall reverted to legacy per-cell tiling on
/// load→save→load). One material authors `span` + `subcell = braille`; a second
/// authors `subcell = quad`; both must survive parse → to_lvl_text → parse.
#[test]
fn texture_section_and_image_material_round_trip() {
    use re_world::{MaterialRecipe, SubcellMode};
    let body = "\
[texture pic]
file = assets/texture/wall_tex.png
[material img_wall]
recipe = image
texture = pic
span = 2,7 3,2
subcell = braille
surface = wall_solid
[material img_floor]
recipe = image
texture = pic
subcell = quad
surface = floor
";
    let lv = re_level::load_string(&cells_shell(body), "<tex>").expect("parses");
    assert_eq!(lv.textures.len(), 1, "one [texture] parsed");
    assert_eq!(lv.materials.len(), 2, "two image materials parsed");
    // The span + subcell parsed onto the LevelMaterial fields.
    assert!(lv.materials[0].image_span.is_some(), "span parsed");
    assert_eq!(lv.materials[0].image_subcell, SubcellMode::Braille);
    assert_eq!(lv.materials[1].image_subcell, SubcellMode::Quad);

    let t1 = to_lvl_text(&lv);
    assert!(
        t1.contains("[texture pic]"),
        "[texture] section emitted:\n{t1}"
    );
    assert!(
        t1.contains("file = assets/texture/wall_tex.png"),
        "texture file path emitted:\n{t1}"
    );
    assert!(t1.contains("recipe = image"), "image recipe emitted:\n{t1}");
    assert!(
        t1.contains("texture = pic"),
        "texture binding emitted:\n{t1}"
    );
    // B1: span + subcell must now appear in the serialized text.
    assert!(t1.contains("span = 2,7 3,2"), "image span emitted:\n{t1}");
    assert!(
        t1.contains("subcell = braille"),
        "braille subcell emitted:\n{t1}"
    );
    assert!(t1.contains("subcell = quad"), "quad subcell emitted:\n{t1}");

    let lv2 = re_level::load_string(&t1, "<tex2>").expect("re-parse");
    assert_eq!(lv2.textures.len(), 1, "texture survives the round-trip");
    assert_eq!(lv2.textures[0].name, "pic");
    assert_eq!(lv2.textures[0].file, "assets/texture/wall_tex.png");
    let m = &lv2.materials[0];
    assert_eq!(
        m.texture.as_deref(),
        Some("pic"),
        "texture binding survives"
    );
    assert!(
        matches!(m.recipe, Some(MaterialRecipe::Image { .. })),
        "image recipe survives the round-trip"
    );
    // B1: span + subcell survive the round-trip (no revert to per-cell tiling).
    let span = m.image_span.expect("span survives the round-trip");
    assert_eq!(span.along_min, 2.0);
    assert_eq!(span.along_size, 7.0);
    assert_eq!(span.up_top, 3.0);
    assert_eq!(span.up_size, 2.0);
    assert_eq!(
        m.image_subcell,
        SubcellMode::Braille,
        "braille subcell survives"
    );
    assert_eq!(
        lv2.materials[1].image_subcell,
        SubcellMode::Quad,
        "quad subcell survives"
    );
    assert!(
        lv2.materials[1].image_span.is_none(),
        "a material with no span stays span-less (not defaulted to Some)"
    );
    assert_eq!(
        lv2.surface_materials.wall_solid.as_deref(),
        Some("img_wall"),
        "surface binding survives"
    );
    // The serialization is idempotent.
    assert_eq!(
        to_lvl_text(&lv2),
        t1,
        "texture re-serialization byte-identical"
    );
}

/// A level with NO `[texture]` section serializes without one (back-compat).
#[test]
fn no_texture_serializes_without_section() {
    let body = "[cells]\nsize = 2,2,1\n##\n##\n";
    let lv = re_level::load_string(&cells_shell(body), "<notex>").expect("parses");
    assert!(lv.textures.is_empty());
    let t = to_lvl_text(&lv);
    assert!(
        !t.contains("[texture"),
        "no [texture] emitted for a texture-less level:\n{t}"
    );
}

/// A default `[vox]` stamp (solid material, unit scale, zero offset) emits a
/// compact section — only the `file` line — and still round-trips.
#[test]
fn vox_default_fields_are_omitted_but_round_trip() {
    use re_level::VoxMaterial;
    let body = "[vox m]\nfile = a.vox\n";
    let lv = re_level::load_string(&cells_shell(body), "<voxd>").expect("parses");
    let t = to_lvl_text(&lv);
    assert!(t.contains("[vox m]"));
    assert!(t.contains("file = a.vox"));
    assert!(!t.contains("offset ="), "zero offset omitted:\n{t}");
    assert!(
        !t.contains("material ="),
        "default solid material omitted:\n{t}"
    );
    assert!(!t.contains("scale ="), "unit scale omitted:\n{t}");
    let lv2 = re_level::load_string(&t, "<voxd2>").expect("re-parse");
    let v = &lv2.voxes[0];
    assert_eq!(v.file, "a.vox");
    assert_eq!((v.off_x, v.off_y, v.off_z), (0, 0, 0));
    assert_eq!(v.material, VoxMaterial::Solid);
    assert_eq!(v.scale, 1);
}

/// A level with NO `[vox]` section serializes without one (back-compat).
#[test]
fn no_vox_serializes_without_section() {
    let body = "[cells]\nsize = 2,2,1\n##\n##\n";
    let lv = re_level::load_string(&cells_shell(body), "<novox>").expect("parses");
    assert!(lv.voxes.is_empty());
    let t = to_lvl_text(&lv);
    assert!(
        !t.contains("[vox"),
        "no [vox] emitted for a vox-less level:\n{t}"
    );
}

/// `CellGrid::from_map(&map)` then `stamp_into(&mut fresh)` preserves shapes
/// exactly — the editor save (capture) / load (apply) path is lossless for
/// 45° geometry.
#[test]
fn from_map_stamp_into_preserves_shapes() {
    use re_level::CellShape as Shape;
    use re_world::{CellKind, DiagXY, Map, RampDir};

    let mut map = Map::new(3, 3, 1);
    map.set_cell(0, 0, 0, CellKind::WallSolid, 0);
    map.set_shape(0, 0, 0, Shape::WedgeXY(DiagXY::SE));
    map.set_cell(2, 2, 0, CellKind::WallSolid, 0);
    map.set_shape(2, 2, 0, Shape::RampZ(RampDir::NegX));

    let grid = re_level::CellGrid::from_map(&map);
    assert_eq!(
        grid.shapes,
        vec![
            (0, 0, 0, Shape::WedgeXY(DiagXY::SE)),
            (2, 2, 0, Shape::RampZ(RampDir::NegX)),
        ]
    );

    let mut fresh = Map::new(3, 3, 1);
    grid.stamp_into(&mut fresh);
    assert_eq!(fresh.shape_at(0, 0, 0), Shape::WedgeXY(DiagXY::SE));
    assert_eq!(fresh.shape_at(2, 2, 0), Shape::RampZ(RampDir::NegX));
    assert_eq!(
        fresh.shape_at(1, 1, 0),
        Shape::Full,
        "unset cells reset to Full"
    );
}

// ───────────────────────────── B1: the five dropped sections ───────────────
//
// `to_lvl_text` previously emitted neither `[holo]`, `[zone]`, `[line]`,
// `[inset]`, nor `[zone_input]`, so a load→save through the editor SILENTLY
// dropped every holo/overlay/zone-input volume. These tests load the SHIPPED
// fixtures, serialize, reparse, serialize again, and assert (a) the second
// serialization is byte-identical (idempotent) and (b) the live fields survive
// with the right values.

/// Path to a repo-root level file (the workspace root is two dirs above this
/// crate's manifest).
fn lvl_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../levels")
        .join(name)
}

/// Path to a crate-local test fixture under `tests/fixtures/`. Used for engine
/// features (zone-input, sequence-lock round trip) that are convenient to
/// demonstrate on a real `.lvl` shape without depending on a campaign level
/// from the separate game repository.
fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Load a shipped level, serialize → reparse → serialize, assert idempotent.
/// Returns the reparsed level so the caller can check field survival.
fn round_trip_shipped(name: &str) -> re_level::Level {
    let lv = re_level::load(lvl_path(name)).unwrap_or_else(|e| panic!("{name} must load: {e}"));
    let t1 = to_lvl_text(&lv);
    let lv2 = re_level::load_string(&t1, name).unwrap_or_else(|e| panic!("{name} re-parse: {e}"));
    let t2 = to_lvl_text(&lv2);
    assert_eq!(
        t1, t2,
        "{name}: serialization must be idempotent (byte-identical)"
    );
    lv2
}

/// Same as [`round_trip_shipped`] but loads from `tests/fixtures/` — for tests
/// exercising an engine feature via a crate-local fixture rather than a
/// shipped/campaign level.
fn round_trip_fixture(name: &str) -> re_level::Level {
    let lv = re_level::load(fixture_path(name)).unwrap_or_else(|e| panic!("{name} must load: {e}"));
    let t1 = to_lvl_text(&lv);
    let lv2 = re_level::load_string(&t1, name).unwrap_or_else(|e| panic!("{name} re-parse: {e}"));
    let t2 = to_lvl_text(&lv2);
    assert_eq!(
        t1, t2,
        "{name}: serialization must be idempotent (byte-identical)"
    );
    lv2
}

#[test]
fn b1_holo_survives_round_trip() {
    // 13-holocube: one `[holo cube]`, shape=box, region (5,9,1)-(7,11,3), style 0.
    let lv = round_trip_shipped("13-holocube.lvl");
    assert_eq!(lv.holos.len(), 1, "the [holo] must survive the round-trip");
    let h = &lv.holos[0];
    assert_eq!(h.name, "cube");
    assert_eq!(h.shape, re_world::HoloShape::Box);
    assert_eq!((h.x0, h.y0, h.z0), (5, 9, 1));
    assert_eq!((h.x1, h.y1, h.z1), (7, 11, 3));
    assert_eq!(h.style, 0);
}

#[test]
fn b1_holomesh_multi_holo_survives_round_trip() {
    // 13b-holomesh exercises multiple holos / non-box shapes if present.
    let lv = round_trip_shipped("13b-holomesh.lvl");
    assert!(!lv.holos.is_empty(), "13b-holomesh must carry holos");
}

#[test]
fn b1_zone_survives_round_trip() {
    // A1-zone-materials: one `[zone center_tile]`, region (4,4,0)-(7,7,0),
    // material floor_tile, scale none.
    let lv = round_trip_shipped("A1-zone-materials.lvl");
    assert_eq!(
        lv.overlay.zones.len(),
        1,
        "the [zone] must survive the round-trip"
    );
    let z = &lv.overlay.zones[0];
    assert_eq!(z.name, "center_tile");
    assert_eq!(z.material.0, "floor_tile");
    assert_eq!(z.scale, re_level::OverlayScaleRule::None);
    let re_level::OverlayRegion::Box { min, max } = z.region else {
        panic!("zone region must be a Box");
    };
    assert_eq!(min, (4, 4, 0));
    assert_eq!(max, (7, 7, 0));
}

#[test]
fn b1_lines_survive_round_trip() {
    // A2-lines: two `[line]` polylines binding line_ink, scale none.
    let lv = round_trip_shipped("A2-lines.lvl");
    assert_eq!(lv.overlay.lines.len(), 2, "both [line]s must survive");
    let l0 = &lv.overlay.lines[0];
    assert_eq!(l0.name, "crisp_diagonal");
    assert_eq!(l0.material.0, "line_ink");
    assert_eq!(l0.scale, re_level::OverlayScaleRule::None);
    assert_eq!(l0.path.points.len(), 2, "polyline has two vertices");
    // Endpoints survive (authored in WORLD z: 6,6.5,3.8; 6.6,9.5,1.2).
    let p0 = l0.path.points[0];
    let p1 = l0.path.points[1];
    assert!((p0.x - 6.0).abs() < 1e-4 && (p0.y - 6.5).abs() < 1e-4 && (p0.z - 3.8).abs() < 1e-4);
    assert!((p1.x - 6.6).abs() < 1e-4 && (p1.y - 9.5).abs() < 1e-4 && (p1.z - 1.2).abs() < 1e-4);
}

#[test]
fn b1_inset_survives_round_trip() {
    // A5-inset: one self_similar `[zone north_portal]` referencing one `[inset
    // warp_window]` (at_octave 0, region (4,0,1)-(7,0,1), child A5-child.lvl).
    let lv = round_trip_shipped("A5-inset.lvl");
    assert_eq!(
        lv.overlay.insets.len(),
        1,
        "the [inset] must survive the round-trip"
    );
    let ins = &lv.overlay.insets[0];
    assert_eq!(ins.name, "warp_window");
    assert_eq!(ins.at_octave, (0, 0));
    let re_level::OverlayRegion::Box { min, max } = ins.region else {
        panic!("inset region must be a Box");
    };
    assert_eq!(min, (4, 0, 1));
    assert_eq!(max, (7, 0, 1));
    match &ins.content {
        re_level::OverlayInsetContent::ChildScene(c) => assert_eq!(c, "A5-child.lvl"),
    }
    // And its parent zone's SelfSimilar scale rule + inset reference survive.
    let z = lv
        .overlay
        .zones
        .iter()
        .find(|z| z.name == "north_portal")
        .expect("north_portal zone present");
    assert!(
        matches!(z.scale, re_level::OverlayScaleRule::SelfSimilar(_)),
        "self_similar scale rule survives"
    );
    assert_eq!(z.insets.len(), 1, "zone keeps its inset reference");
    assert_eq!(z.insets[0].0, "warp_window");
}

#[test]
fn b1_zone_input_survives_round_trip() {
    // `zone_input_sample.lvl`: one `[zone_input warm_square]` driving node
    // `present` — a neutral crate-local clone of a campaign floor-zone
    // shape, exercising the writer's `[zone_input]` round trip in isolation
    // from any campaign win-reachability concern (that lives in the host
    // game's own `campaign_walkable` suite, in its separate repository).
    let lv = round_trip_fixture("zone_input_sample.lvl");
    assert_eq!(lv.zone_inputs.len(), 1, "the [zone_input] must survive");
    let zi = &lv.zone_inputs[0];
    assert_eq!(zi.name, "warm_square");
    assert_eq!(zi.node, "present");
    assert_eq!((zi.x0, zi.y0, zi.z0), (1, 5, 1));
    assert_eq!((zi.x1, zi.y1, zi.z1), (2, 7, 1));
}

#[test]
fn b1_sequence_survives_round_trip() {
    // `sequence_lock_sample.lvl`: the `[sequence]` win mode, on a neutral
    // crate-local clone of a campaign "sequence-lock" graph shape. The
    // parser resolves `[sequence]` to a `SequenceSpec` whose `patterns`
    // discard the authored node NAMES and the observe window, so before the
    // authored-form fix the writer dropped the section and a save→load lost
    // the win sequence. `round_trip_fixture` already asserts the
    // serialization is byte-identical (idempotent); here we assert the
    // section was actually emitted and the authored + resolved forms survive
    // with the right values.
    let lv = round_trip_fixture("sequence_lock_sample.lvl");

    // The authored observe window + steps survived (the writer's source data).
    assert_eq!(
        lv.sequence.observe_names,
        vec!["out1".to_string(), "out2".to_string(), "out3".to_string()],
        "observe window node names survive the round-trip"
    );
    assert_eq!(
        lv.sequence.step_names,
        vec![
            vec!["out1".to_string()],
            vec!["out2".to_string()],
            vec!["out3".to_string()],
        ],
        "the three ordered step ON-sets survive the round-trip"
    );

    // The resolved patterns (what the win check consumes) re-resolved correctly:
    // three distinct ordered output states, BREAK→FALL→DARK.
    assert_eq!(
        lv.sequence.patterns.len(),
        3,
        "three resolved patterns survive"
    );
    assert_ne!(lv.sequence.patterns[0], lv.sequence.patterns[1]);
    assert_ne!(lv.sequence.patterns[1], lv.sequence.patterns[2]);
    assert_ne!(lv.sequence.patterns[0], lv.sequence.patterns[2]);

    // The emitted text actually carries the section (proves it is no longer dropped).
    let text = to_lvl_text(&lv);
    assert!(
        text.contains("[sequence]"),
        "emitted text carries [sequence]:\n{text}"
    );
    assert!(
        text.contains("observe = out1 out2 out3"),
        "observe line re-emitted"
    );
    assert!(text.contains("step = out1"), "step lines re-emitted");
    assert!(text.contains("step = out2"));
    assert!(text.contains("step = out3"));

    // And the fixture still SOLVES the same way: build the puzzle from the
    // ROUND-TRIPPED level (`lv`, the reparse of `to_lvl_text`) and drive the
    // lock P1→P2→P3 exactly as the mechanic's 3-step solve, then assert it
    // wins. This proves the writer did not perturb the win semantics — a
    // save→reload preserves the win.
    use re_puzzle::{puzzle_interact, PuzzleState};
    let mut s = PuzzleState::from_level(&lv).expect("round-tripped fixture builds");
    assert!(
        s.sequence.is_active(),
        "round-tripped fixture still drives a sequence lock"
    );
    assert_eq!(s.sequence.patterns.len(), 3);
    assert_eq!(s.sequence.progress, 0, "no auto-lock at the all-OFF start");
    assert!(!s.solved);

    // Resolve a switch cell by node-pair (from->to). `Pipe.from`/`.to` are node
    // INDICES in `re_puzzle`, so map names→indices via the level's name index.
    let idx = |n: &str| lv.node_index(n).expect("node present");
    let switch = |s: &PuzzleState, from: usize, to: usize| -> (i32, i32, i32) {
        s.pipes
            .iter()
            .find(|p| p.from == from && p.to == to)
            .and_then(|p| p.switch_coord())
            .expect("pipe with switch present")
    };
    let (in1, in2, in3, gate, out1, out2, out3) = (
        idx("in1"),
        idx("in2"),
        idx("in3"),
        idx("gate"),
        idx("out1"),
        idx("out2"),
        idx("out3"),
    );

    // `gate` (T2 = in1+in2) is load-bearing for both out2 and out3.
    // Pipes: A=in1->out1, B=in1->gate, C=in2->gate, D=in2->out2,
    // E=gate->out2, F=gate->out3, G=in3->out3.

    // P1 BREAK: +A (in1->out1) → output (1,0,0) locks P1.
    let (x, y, z) = switch(&s, in1, out1);
    assert!(puzzle_interact(&mut s, x, y, z));
    assert_eq!(s.sequence.progress, 1, "P1 locks");
    // P2 FALL: −A, +B, +C, +D, +E → gate = in1+in2 = 2, out2 = D+E = 2, locks P2.
    let (x, y, z) = switch(&s, in1, out1);
    assert!(puzzle_interact(&mut s, x, y, z));
    let (x, y, z) = switch(&s, in1, gate);
    assert!(puzzle_interact(&mut s, x, y, z));
    let (x, y, z) = switch(&s, in2, gate);
    assert!(puzzle_interact(&mut s, x, y, z));
    let (x, y, z) = switch(&s, in2, out2);
    assert!(puzzle_interact(&mut s, x, y, z));
    let (x, y, z) = switch(&s, gate, out2);
    assert!(puzzle_interact(&mut s, x, y, z));
    assert_eq!(s.sequence.progress, 2, "P2 locks");
    // P3 DARK: −D, −E, +F, +G → gate stays 2 (B,C untouched), out3 = F+G = 2 → WON.
    let (x, y, z) = switch(&s, in2, out2);
    assert!(puzzle_interact(&mut s, x, y, z));
    let (x, y, z) = switch(&s, gate, out2);
    assert!(puzzle_interact(&mut s, x, y, z));
    let (x, y, z) = switch(&s, gate, out3);
    assert!(puzzle_interact(&mut s, x, y, z));
    let (x, y, z) = switch(&s, in3, out3);
    assert!(puzzle_interact(&mut s, x, y, z));
    assert_eq!(s.sequence.progress, 3, "P3 locks");
    assert!(
        s.solved,
        "round-tripped fixture still WINS via the sequence lock"
    );
}

#[test]
fn b1_self_similar_scale_rule_round_trips_params() {
    // A self_similar zone's arms/twist/warp survive the writer's `key:val` form.
    let body = "\
[zone fract]
region   = (1,1,0)-(3,3,0)
material = m
scale    = self_similar arms:5 twist:0.9 warp:0.3

[material m]
fg = #FFFFFF
bg = #000000
";
    let lv = re_level::load_string(&cells_shell(body), "<ss>").expect("parses");
    let t1 = to_lvl_text(&lv);
    let lv2 = re_level::load_string(&t1, "<ss2>").expect("re-parse");
    assert_eq!(
        to_lvl_text(&lv2),
        t1,
        "self_similar re-serialization byte-identical"
    );
    let z = &lv2.overlay.zones[0];
    match &z.scale {
        re_level::OverlayScaleRule::SelfSimilar(p) => {
            assert_eq!(p.arms, 5);
            assert!((p.twist - 0.9).abs() < 1e-4);
            assert!((p.octave_warp - 0.3).abs() < 1e-4);
        }
        other => panic!("expected SelfSimilar, got {other:?}"),
    }
}

#[test]
fn b1_overlay_less_level_emits_no_overlay_sections() {
    // Back-compat: a level with none of the five new sections emits none of them.
    let lv = re_level::load(lvl_path("01.lvl")).expect("01.lvl loads");
    let t = to_lvl_text(&lv);
    for sec in [
        "[holo",
        "[zone ",
        "[zone]",
        "[line",
        "[inset",
        "[zone_input",
    ] {
        assert!(!t.contains(sec), "01.lvl must not emit {sec}:\n{t}");
    }
}

// ─────────────── storey_height omit-when-default (multi-floor spec §4) ────────
//
// `to_lvl_text` must emit `storey_height = N` ONLY when the effective height
// differs from DEFAULT_STOREY_HEIGHT (4).  Two invariants:
//   1. A level with a non-default height → the key appears and re-parses identically
//      (round-trip + idempotency).
//   2. A level with the default height (or no key at all) → no `storey_height` line
//      (byte-identical to a pre-feature save — existing levels are unaffected).

/// Round-trip: a level with `storey_height = 7` survives `to_lvl_text` →
/// `re_level::load_string` → `Level::storey_height() == 7`.  The key must appear
/// in the emitted text, and two passes are byte-identical (idempotency).
#[test]
fn storey_height_non_default_round_trips() {
    // Embed storey_height in a minimal well-formed level body.
    let body = "\
[meta]
name = multi-floor-rt

[room]
size          = 8 8 9
ambient       = solid
storey_height = 7

[pedestal]
center = 3 3 4
volume = 1 1 1

[door]
pos = 6 6 8

[player]
spawn = 2 2 1
";
    // First parse.
    let lv = re_level::load_string(body, "<sh7>").expect("storey_height=7 must parse");
    assert_eq!(lv.storey_height(), 7, "parsed storey_height must be 7");

    // Serialize.
    let t1 = to_lvl_text(&lv);
    assert!(
        t1.contains("storey_height = 7"),
        "emitted text must carry storey_height = 7:\n{t1}"
    );

    // Re-parse → the value survives.
    let lv2 = re_level::load_string(&t1, "<sh7-rt>")
        .unwrap_or_else(|e| panic!("re-parse of to_lvl_text output failed: {e}\n---\n{t1}\n---"));
    assert_eq!(
        lv2.storey_height(),
        7,
        "storey_height must survive the round-trip"
    );

    // Idempotency: a second serialization is byte-identical.
    let t2 = to_lvl_text(&lv2);
    assert_eq!(
        t1, t2,
        "storey_height re-serialization must be byte-identical"
    );
}

/// Omit-when-default: a level parsed without a `storey_height` key (or with the
/// default value 4) must NOT emit a `storey_height` line — keeping existing
/// single-storey levels byte-identical on re-save.
#[test]
fn storey_height_default_omitted_from_output() {
    // A level with no storey_height key → loaded value is the default (4).
    let body = "\
[meta]
name = single-storey-rt

[room]
size   = 6 6 4
ambient = solid

[pedestal]
center = 2 2 1
volume = 1 1 1

[door]
pos = 4 4 3

[player]
spawn = 1 1 1
";
    let lv = re_level::load_string(body, "<sh-def>").expect("must parse without storey_height");
    assert_eq!(
        lv.storey_height(),
        re_level::DEFAULT_STOREY_HEIGHT,
        "absent key → default height"
    );

    let t = to_lvl_text(&lv);
    assert!(
        !t.contains("storey_height"),
        "storey_height must NOT appear in output for default height:\n{t}"
    );

    // Explicit default (4) authored in the file is also omitted after the round-trip,
    // since the parser normalises it and the writer applies omit-when-default.
    let body_explicit = body.replace("ambient = solid", "ambient = solid\nstorey_height = 4");
    let lv_ex =
        re_level::load_string(&body_explicit, "<sh-def-ex>").expect("explicit default must parse");
    assert_eq!(lv_ex.storey_height(), 4, "explicit 4 → default height");
    let t_ex = to_lvl_text(&lv_ex);
    assert!(
        !t_ex.contains("storey_height"),
        "explicit default 4 must also be omitted in the output:\n{t_ex}"
    );
}

// ───────────────────────── B2: side_data namespace round-trip ──────────────
//
// `from_map` previously captured EVERY non-zero `side_data` as a colour-material
// id, then `[cells]` re-stamped those captured-stale tags LAST + authoritative —
// so a pipe/zone/node/reactive cell got MIS-TAGGED on the editor's load→save→load.
// The fix: `from_map` excludes procedural-tag cells, and `build_map` stamps the
// captured material ids EARLY (procedural passes win on their cells) while
// `[cells]` GEOMETRY still stamps last (preserving side_data). Invariant:
// procedurally-derived tags are never persisted-and-restamped; only authored
// geometry + authored material ids round-trip.

#[test]
fn b2_pipe_zone_node_reactive_tags_match_fresh_build_after_editor_round_trip() {
    use re_world::CellKind;

    // A level with: a render_pipes pipe (→ reactive bit-15 tagged WallTrace), an
    // overlay zone (→ zone-id-tagged solid floor), a node (→ Object node tag), AND
    // a user-painted colour id on a PLAIN wall (the authored-id-survives case).
    let body = "\
[meta]
name = b2-fixture

[room]
size         = 8 8 4
ambient      = solid
render_pipes = true

[material wall_paint]
fg = #AABBCC
bg = #112233
surface = wall_solid

[material zone_mat]
fg = #44CC88
bg = #061018

[zone z0]
region   = (2,2,0)-(4,4,0)
material = zone_mat
scale    = none

[node A]
pos      = 1,5,2
is_input = 1

[node B]
pos      = 6,5,2
is_output = 1

[inputs]
A = 1

[pipe A->B]
state    = ON
path     = (1,5,2) (6,5,2)
switch_at = (3,5,2)

# A genuine user-painted colour id on a PLAIN floor cell (not in the zone, not a
# pipe/node) — this MUST round-trip.
[color_cell_ids]
6,1,0 = 2

[pedestal]
center   = 6 5 2
volume   = 1 1 1

[door]
pos      = 0 5 1
target_parts = B

[player]
spawn    = 4 1 1
facing   = +x
";
    let lv = re_level::load_string(body, "<b2>").expect("fixture parses");
    let fresh = lv.build_map();

    // Sanity: the fresh build produced the procedural tags we mean to protect.
    // Pipe trace cells carry the bit-15 PIPE_TAG_FLAG.
    let pipe_cell = (3, 5, 2); // on the authored path, between the nodes
    assert_eq!(
        fresh.cell_at(3, 5, 2),
        CellKind::WallTrace,
        "pipe stamps a trace cell"
    );
    let fresh_pipe_sd = fresh.side_data_at(3, 5, 2);
    assert!(
        re_level::decode_pipe_tag(fresh_pipe_sd).is_some(),
        "fresh pipe cell carries the reactive bit-15 flag (sd=0x{fresh_pipe_sd:04X})"
    );
    // A node Object cell carries its node-index tag.
    assert_eq!(
        fresh.cell_at(1, 5, 2),
        CellKind::Object,
        "node A is an Object cell"
    );
    // A zone-stamped solid floor cell carries a non-zero colour id.
    let fresh_zone_sd = fresh.side_data_at(3, 3, 0);
    assert!(
        fresh_zone_sd != 0,
        "fresh zone cell carries a resolved colour id"
    );
    // The user-painted plain-wall id.
    let fresh_paint_sd = fresh.side_data_at(6, 1, 0);
    assert_eq!(
        fresh_paint_sd, 2,
        "fresh authored colour id present on the plain cell"
    );

    // ── The editor round-trip: capture → serialize → reparse → rebuild. ──
    let mut lv_saved = lv.clone();
    lv_saved.cells = Some(re_level::CellGrid::from_map(&fresh));
    let text = to_lvl_text(&lv_saved);
    let lv_reloaded = re_level::load_string(&text, "<b2-reload>").expect("reparse");
    let rebuilt = lv_reloaded.build_map();

    // INVARIANT: the procedurally-derived tags on the REBUILT map match a FRESH
    // build exactly — no mis-tag from a re-stamped stale capture.
    assert_eq!(
        rebuilt.side_data_at(3, 5, 2),
        fresh_pipe_sd,
        "pipe reactive tag must be RECOMPUTED, not the captured stale value"
    );
    assert!(
        re_level::decode_pipe_tag(rebuilt.side_data_at(3, 5, 2)).is_some(),
        "rebuilt pipe cell still carries the bit-15 flag"
    );
    assert_eq!(
        rebuilt.cell_at(3, 5, 2),
        CellKind::WallTrace,
        "pipe cell kind preserved across the round-trip"
    );
    assert_eq!(
        rebuilt.side_data_at(3, 3, 0),
        fresh_zone_sd,
        "zone id must be RECOMPUTED, not a captured stale id"
    );
    let (nx, ny, nz) = pipe_cell; // reuse to silence unused warning if any
    let _ = (nx, ny, nz);
    // The node tag survives (Object cells are excluded from capture, so the
    // node-index tag is recomputed by the geometry pass, not overwritten).
    assert_eq!(
        rebuilt.cell_at(1, 5, 2),
        CellKind::Object,
        "node cell kind preserved"
    );
    assert_eq!(
        rebuilt.side_data_at(1, 5, 2),
        fresh.side_data_at(1, 5, 2),
        "node-index tag must match the fresh build"
    );

    // And the GENUINELY authored colour id on the plain wall STILL survives.
    assert_eq!(
        rebuilt.side_data_at(6, 1, 0),
        2,
        "a user-painted material id on a plain cell must round-trip"
    );
}

#[test]
fn b2_from_map_captures_all_side_data_arbitration_is_on_load() {
    use re_world::{CellKind, Map};

    // Build a map by hand carrying each procedural tag kind, plus one authored
    // colour id on a plain wall.
    let mut map = Map::new(4, 1, 1);
    // (0,0,0): a node Object cell with a node-index tag (procedural-on-reload IFF a
    // backing [node] exists; here it does NOT, so this is a bare authored Object).
    map.set_cell(0, 0, 0, CellKind::Object, 7);
    // (1,0,0): a pipe WallTrace with the bit-15 flag (procedural).
    map.set_cell(
        1,
        0,
        0,
        CellKind::WallTrace,
        re_level::encode_pipe_side_data(3),
    );
    // (2,0,0): a Hologram cell with a style tag (procedural).
    map.set_cell(2, 0, 0, CellKind::Hologram, 5);
    // (3,0,0): a PLAIN wall with an AUTHORED colour id (must be captured).
    map.set_cell(3, 0, 0, CellKind::WallSolid, 4);

    let grid = re_level::CellGrid::from_map(&map);
    // B2-FIX CONTRACT: `from_map` now captures EVERY non-zero side_data — no value
    // is dropped at capture time. The procedural-vs-authored arbitration moved to
    // the LOAD side (`build_map` re-applies these ids only onto cells no procedural
    // pass claimed). This guarantees a bare authored Object/Hologram (no backing
    // entity) round-trips, while a real node/pipe/zone/holo cell still gets its
    // recomputed tag on reload (the re-apply skips a cell a procedural pass owns).
    let mut got = grid.material_ids.clone();
    got.sort();
    assert_eq!(
        got,
        vec![
            (0, 0, 0, 7),
            (1, 0, 0, re_level::encode_pipe_side_data(3)),
            (2, 0, 0, 5),
            (3, 0, 0, 4),
        ],
        "every non-zero side_data is captured (load-side arbitration)"
    );
    // Kinds are still captured densely (geometry round-trips).
    assert_eq!(grid.kind_at(0, 0, 0), CellKind::Object);
    assert_eq!(grid.kind_at(1, 0, 0), CellKind::WallTrace);
    assert_eq!(grid.kind_at(2, 0, 0), CellKind::Hologram);
    assert_eq!(grid.kind_at(3, 0, 0), CellKind::WallSolid);
}

/// TRIGGER-FIELDS: a `[trigger]` section (both `activate` and `event` actions)
/// survives load → `to_lvl_text` → reload UNCHANGED. This is the writer + parser
/// pair in one test — guarding the recent "section parses but never serializes"
/// gap that silently drops authored data on save.
#[test]
fn round_trip_preserves_triggers() {
    // A self-contained minimal level: one trigger-driven input node wired to one
    // output, plus TWO triggers (an ActivateNode and a generic Event).
    let src = "\
[meta]
name = trig-rt
version = 1
[room]
size = 12 6 4
[node gate]
pos = 8,3,1
glyph = G
is_input = yes
[node output]
pos = 2,3,1
glyph = O
is_output = yes
threshold = 1
[pipe gate->output]
state = ON
path = (8,3,1) (2,3,1)
[trigger trip]
region = (7,2,1)-(9,4,1)
action = activate gate
once = true
[trigger ping]
region = (3,2,1)-(4,3,1)
action = event entered_inner
once = false
[inputs]
gate = 1
[pedestal]
center = 4 3 1
volume = 3 3 2
[door]
pos = 1 3 1
target_parts = output
[player]
spawn = 10 3 1
facing = -x
";
    let level = re_level::load_string(src, "trig-src").expect("source level parses");
    assert_eq!(level.triggers.len(), 2, "two triggers parsed");

    // The ActivateNode's node name resolved to `gate`'s index in finalize.
    let gate_idx = level.node_index("gate").expect("gate node");
    assert_eq!(
        level.triggers[0].action,
        re_level::TriggerAction::ActivateNode {
            name: "gate".to_string(),
            node_index: gate_idx,
        },
    );
    assert!(level.triggers[0].once);
    assert_eq!(
        level.triggers[1].action,
        re_level::TriggerAction::event("entered_inner"),
    );
    assert!(!level.triggers[1].once);

    // Serialize → reparse. Every trigger field must survive (region + action +
    // once), and the ActivateNode re-resolves to the same node index.
    let text = to_lvl_text(&level);
    assert!(
        text.contains("[trigger]"),
        "writer emitted a [trigger] section"
    );
    let rt = re_level::load_string(&text, "trig-rt").unwrap_or_else(|e| {
        panic!("round-trip parse failed: {e}\n---\n{text}\n---");
    });

    assert_eq!(rt.triggers.len(), 2, "two triggers survive round-trip");
    assert_eq!(rt.triggers[0].region, level.triggers[0].region, "region 0");
    assert_eq!(rt.triggers[0].action, level.triggers[0].action, "action 0");
    assert_eq!(rt.triggers[0].once, level.triggers[0].once, "once 0");
    assert_eq!(rt.triggers[1].region, level.triggers[1].region, "region 1");
    assert_eq!(rt.triggers[1].action, level.triggers[1].action, "action 1");
    assert_eq!(rt.triggers[1].once, level.triggers[1].once, "once 1");

    // build_map carries the resolved triggers onto the Map for the firing system.
    let map = rt.build_map();
    assert_eq!(map.resolved_triggers().len(), 2, "triggers reach the Map");
}

// ───────────────────────────── object nesting ([hierarchy]) ─────────────────────────────

/// A small valid level (3 nodes + door/spawn/pedestal) for hierarchy round-trips.
fn hier_level() -> re_level::Level {
    let src = "[meta]
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
";
    re_level::load_string(src, "<h>").expect("parses")
}

#[test]
fn hierarchy_round_trips_through_lvl_text() {
    use re_level::{HierKind, HierRef};
    let mut lv = hier_level();
    let a = HierRef::new(HierKind::Node, 0);
    let b = HierRef::new(HierKind::Node, 1);
    let c = HierRef::new(HierKind::Node, 2);
    let door = HierRef::new(HierKind::Door, 0);
    lv.set_hier_parent(b, a);
    lv.set_hier_parent(c, b);
    lv.set_hier_parent(a, door); // a node nested under the door singleton.

    let text = to_lvl_text(&lv);
    assert!(
        text.contains("[hierarchy]"),
        "writer emitted a [hierarchy] section"
    );
    let rt = re_level::load_string(&text, "hier-rt").unwrap_or_else(|e| {
        panic!("round-trip parse failed: {e}\n---\n{text}\n---");
    });
    assert_eq!(rt.hier_parent_of(b), Some(a), "b->a survives");
    assert_eq!(rt.hier_parent_of(c), Some(b), "c->b survives (multi-level)");
    assert_eq!(
        rt.hier_parent_of(a),
        Some(door),
        "node-under-singleton survives"
    );
    assert_eq!(rt.hierarchy.len(), 3);
}

#[test]
fn hierarchy_absent_emits_no_section() {
    // A level with no nesting writes NO [hierarchy] section (byte-identical back-compat).
    let lv = hier_level();
    assert!(lv.hierarchy.is_empty());
    let text = to_lvl_text(&lv);
    assert!(
        !text.contains("[hierarchy]"),
        "no section when the level has no links"
    );
}

#[test]
fn hierarchy_idempotent_on_resave() {
    use re_level::{HierKind, HierRef};
    let mut lv = hier_level();
    lv.set_hier_parent(
        HierRef::new(HierKind::Node, 1),
        HierRef::new(HierKind::Node, 0),
    );
    // save → load → save must be a fixed point for the hierarchy section.
    let t1 = to_lvl_text(&lv);
    let lv2 = re_level::load_string(&t1, "rt1").expect("parses");
    let t2 = to_lvl_text(&lv2);
    assert_eq!(t1, t2, "serialization is idempotent");
}

// ===========================================================================
// SubcellMode extended variants + Fit field round-trips
// ===========================================================================

/// Helper: build a minimal level string that has one image material with the
/// given extra key-value lines inserted verbatim into the [material] section.
fn image_material_level_voximport(extra: &str) -> String {
    cells_shell(&format!(
        "[texture img]\nfile = assets/texture/test.png\n\
         [material wall]\nrecipe = image\ntexture = img\n{extra}"
    ))
}

/// SubcellMode::Ascii survives serialize → parse with the same enum variant.
#[test]
fn subcell_ascii_round_trips() {
    use re_world::SubcellMode;
    let src = image_material_level_voximport("subcell = ascii\n");
    let lv = re_level::load_string(&src, "<sc_ascii_in>").expect("initial parse");
    assert_eq!(
        lv.materials[0].image_subcell,
        SubcellMode::Ascii,
        "ascii parsed"
    );

    let t1 = to_lvl_text(&lv);
    assert!(t1.contains("subcell = ascii"), "ascii emitted:\n{t1}");

    let lv2 = re_level::load_string(&t1, "<sc_ascii_rt>").expect("re-parse");
    assert_eq!(
        lv2.materials[0].image_subcell,
        SubcellMode::Ascii,
        "ascii survives round-trip"
    );
    assert_eq!(to_lvl_text(&lv2), t1, "idempotent");
}

/// SubcellMode::Contour survives serialize → parse with the same enum variant.
#[test]
fn subcell_contour_round_trips() {
    use re_world::SubcellMode;
    let src = image_material_level_voximport("subcell = contour\n");
    let lv = re_level::load_string(&src, "<sc_contour_in>").expect("initial parse");
    assert_eq!(
        lv.materials[0].image_subcell,
        SubcellMode::Contour,
        "contour parsed"
    );

    let t1 = to_lvl_text(&lv);
    assert!(t1.contains("subcell = contour"), "contour emitted:\n{t1}");

    let lv2 = re_level::load_string(&t1, "<sc_contour_rt>").expect("re-parse");
    assert_eq!(
        lv2.materials[0].image_subcell,
        SubcellMode::Contour,
        "contour survives round-trip"
    );
    assert_eq!(to_lvl_text(&lv2), t1, "idempotent");
}

/// SubcellMode::Lines survives serialize → parse with the same enum variant.
#[test]
fn subcell_lines_round_trips() {
    use re_world::SubcellMode;
    let src = image_material_level_voximport("subcell = lines\n");
    let lv = re_level::load_string(&src, "<sc_lines_in>").expect("initial parse");
    assert_eq!(
        lv.materials[0].image_subcell,
        SubcellMode::Lines,
        "lines parsed"
    );

    let t1 = to_lvl_text(&lv);
    assert!(t1.contains("subcell = lines"), "lines emitted:\n{t1}");

    let lv2 = re_level::load_string(&t1, "<sc_lines_rt>").expect("re-parse");
    assert_eq!(
        lv2.materials[0].image_subcell,
        SubcellMode::Lines,
        "lines survives round-trip"
    );
    assert_eq!(to_lvl_text(&lv2), t1, "idempotent");
}

/// Fit::Stretch survives serialize → parse; `fit = stretch` is emitted.
#[test]
fn fit_stretch_round_trips() {
    use re_world::Fit;
    let src = image_material_level_voximport("fit = stretch\n");
    let lv = re_level::load_string(&src, "<fit_str_in>").expect("initial parse");
    assert_eq!(lv.materials[0].image_fit, Fit::Stretch, "stretch parsed");

    let t1 = to_lvl_text(&lv);
    assert!(t1.contains("fit = stretch"), "stretch emitted:\n{t1}");

    let lv2 = re_level::load_string(&t1, "<fit_str_rt>").expect("re-parse");
    assert_eq!(
        lv2.materials[0].image_fit,
        Fit::Stretch,
        "stretch survives round-trip"
    );
    assert_eq!(to_lvl_text(&lv2), t1, "idempotent");
}

/// Fit::Cover survives serialize → parse; `fit = cover` is emitted.
#[test]
fn fit_cover_round_trips() {
    use re_world::Fit;
    let src = image_material_level_voximport("fit = cover\n");
    let lv = re_level::load_string(&src, "<fit_cov_in>").expect("initial parse");
    assert_eq!(lv.materials[0].image_fit, Fit::Cover, "cover parsed");

    let t1 = to_lvl_text(&lv);
    assert!(t1.contains("fit = cover"), "cover emitted:\n{t1}");

    let lv2 = re_level::load_string(&t1, "<fit_cov_rt>").expect("re-parse");
    assert_eq!(
        lv2.materials[0].image_fit,
        Fit::Cover,
        "cover survives round-trip"
    );
    assert_eq!(to_lvl_text(&lv2), t1, "idempotent");
}

/// Fit::Contain (the default) is OMITTED from the serialized text so that
/// existing materials with no `fit` key stay byte-identical after a
/// load → save → load cycle.
#[test]
fn fit_contain_is_omitted_default() {
    use re_world::Fit;
    // A material with no explicit fit key should default to Contain.
    let src = image_material_level_voximport("");
    let lv = re_level::load_string(&src, "<fit_cont_in>").expect("initial parse");
    assert_eq!(
        lv.materials[0].image_fit,
        Fit::Contain,
        "absent fit key defaults to Contain"
    );

    let t1 = to_lvl_text(&lv);
    assert!(
        !t1.contains("fit ="),
        "Contain must not be emitted (omit-default rule):\n{t1}"
    );

    // Explicit `fit = contain` also serializes as omitted (same default).
    let src2 = image_material_level_voximport("fit = contain\n");
    let lv2 =
        re_level::load_string(&src2, "<fit_cont_ex>").expect("initial parse explicit contain");
    assert_eq!(
        lv2.materials[0].image_fit,
        Fit::Contain,
        "explicit contain parsed"
    );
    let t2 = to_lvl_text(&lv2);
    assert!(
        !t2.contains("fit ="),
        "explicit Contain must also be omitted:\n{t2}"
    );
}

/// Combined: subcell = contour + fit = stretch both survive a single
/// serialize → parse round-trip on the same material.
#[test]
fn subcell_contour_fit_stretch_combined_round_trip() {
    use re_world::{Fit, SubcellMode};
    let src = image_material_level_voximport("subcell = contour\nfit = stretch\n");
    let lv = re_level::load_string(&src, "<comb_in>").expect("initial parse");
    assert_eq!(lv.materials[0].image_subcell, SubcellMode::Contour);
    assert_eq!(lv.materials[0].image_fit, Fit::Stretch);

    let t1 = to_lvl_text(&lv);
    assert!(
        t1.contains("subcell = contour"),
        "contour in combined output:\n{t1}"
    );
    assert!(
        t1.contains("fit = stretch"),
        "stretch in combined output:\n{t1}"
    );

    let lv2 = re_level::load_string(&t1, "<comb_rt>").expect("re-parse");
    assert_eq!(
        lv2.materials[0].image_subcell,
        SubcellMode::Contour,
        "contour survives combined round-trip"
    );
    assert_eq!(
        lv2.materials[0].image_fit,
        Fit::Stretch,
        "stretch survives combined round-trip"
    );
    assert_eq!(to_lvl_text(&lv2), t1, "idempotent");
}

// ---------------------------------------------------------------------------
// [sky] round-trip: serialize → parse → same sky paths.
// ---------------------------------------------------------------------------

/// Helper: build and return a minimal `Level` that has sky paths set but no
/// real image files (the serializer only stores paths, so this is safe).
fn level_with_sky(
    cylinder: Option<&str>,
    cap_top: Option<&str>,
    cap_bottom: Option<&str>,
) -> re_level::Level {
    // Parse the minimal base level then inject the sky paths directly.
    let base = "\
[meta]
name = sky_rt
[room]
size = 6 4 4
ambient = solid
[node a]
pos = 1 2 2
is_input = yes
[node b]
pos = 4 2 2
is_output = yes
[pipe a->b]
path = (1,2,2) (4,2,2)
[pedestal]
center = 4 2 1
volume = 1 1 1
[door]
pos = 5 2 1
[player]
spawn = 3 2 1
";
    let mut lv = re_level::load_string(base, "<sky-rt-base>").expect("base parses");
    lv.sky.cylinder = cylinder.map(str::to_string);
    lv.sky.cap_top = cap_top.map(str::to_string);
    lv.sky.cap_bottom = cap_bottom.map(str::to_string);
    lv
}

/// ROUND-TRIP: a level with all three sky paths serializes and re-parses with the
/// same paths intact — the `[sky]` section survives save → load.
#[test]
fn sky_all_three_paths_round_trip() {
    let lv = level_with_sky(
        Some("sky/horizon.png"),
        Some("sky/top.png"),
        Some("sky/bottom.png"),
    );
    let text = to_lvl_text(&lv);
    // The emitted text must contain a [sky] section with all three keys.
    assert!(
        text.contains("[sky]"),
        "serialized text must contain [sky]:\n{text}"
    );
    assert!(
        text.contains("cylinder = sky/horizon.png"),
        "cylinder key:\n{text}"
    );
    assert!(
        text.contains("cap_top = sky/top.png"),
        "cap_top key:\n{text}"
    );
    assert!(
        text.contains("cap_bottom = sky/bottom.png"),
        "cap_bottom key:\n{text}"
    );

    let rt = re_level::load_string(&text, "<sky-rt-all>").unwrap_or_else(|e| {
        panic!("sky round-trip parse failed: {e}\n---\n{text}\n---");
    });
    assert_eq!(
        rt.sky.cylinder.as_deref(),
        Some("sky/horizon.png"),
        "cylinder survives"
    );
    assert_eq!(
        rt.sky.cap_top.as_deref(),
        Some("sky/top.png"),
        "cap_top survives"
    );
    assert_eq!(
        rt.sky.cap_bottom.as_deref(),
        Some("sky/bottom.png"),
        "cap_bottom survives"
    );
}

/// ROUND-TRIP: only `cylinder` + `cap_top` (no nadir cap) — `cap_bottom` stays
/// absent after serialize → parse, and the serialized text has no `cap_bottom` key.
#[test]
fn sky_cylinder_and_cap_top_only_round_trip() {
    let lv = level_with_sky(Some("assets/sky_cyl.png"), Some("assets/sky_cap.png"), None);
    let text = to_lvl_text(&lv);
    assert!(text.contains("[sky]"), "[sky] header must appear");
    assert!(
        text.contains("cylinder = assets/sky_cyl.png"),
        "cylinder key must appear"
    );
    assert!(
        text.contains("cap_top = assets/sky_cap.png"),
        "cap_top key must appear"
    );
    assert!(
        !text.contains("cap_bottom"),
        "absent cap_bottom must not appear"
    );

    let rt = re_level::load_string(&text, "<sky-rt-2>").unwrap_or_else(|e| {
        panic!("sky partial round-trip parse failed: {e}\n---\n{text}\n---");
    });
    assert_eq!(rt.sky.cylinder.as_deref(), Some("assets/sky_cyl.png"));
    assert_eq!(rt.sky.cap_top.as_deref(), Some("assets/sky_cap.png"));
    assert!(rt.sky.cap_bottom.is_none(), "cap_bottom must remain None");
}

/// ROUND-TRIP: a level with NO sky paths emits NO `[sky]` section — sky-less
/// levels stay byte-identical.
#[test]
fn sky_absent_emits_no_sky_section() {
    let lv = level_with_sky(None, None, None);
    let text = to_lvl_text(&lv);
    assert!(
        !text.contains("[sky]"),
        "a level with no sky paths must not emit a [sky] section:\n{text}"
    );
    // Re-parsing also yields no sky paths.
    let rt = re_level::load_string(&text, "<sky-rt-none>").unwrap_or_else(|e| {
        panic!("no-sky round-trip parse failed: {e}\n---\n{text}\n---");
    });
    assert!(rt.sky.is_empty(), "re-parsed level must have no sky paths");
}

/// IDEMPOTENT: serialize → parse → serialize yields the same text.
#[test]
fn sky_round_trip_is_idempotent() {
    let lv = level_with_sky(
        Some("levels/sky/horizon.png"),
        Some("levels/sky/zenith.png"),
        None,
    );
    let t1 = to_lvl_text(&lv);
    let lv2 = re_level::load_string(&t1, "<sky-idem>").expect("first parse");
    let t2 = to_lvl_text(&lv2);
    assert_eq!(t1, t2, "sky round-trip must be idempotent");
}

// ---------------------------------------------------------------------------
// [wall_decal] density / color_head / color_tail round-trip (fidelity fix)
// ---------------------------------------------------------------------------
//
// Previously `to_lvl_text` emitted only `surface` + `region` for decals,
// silently dropping `density`, `color_head`, and `color_tail` on save→reload.
// These tests guard the fix: every authorable field that `re_level` parses
// must now survive a full `to_lvl_text` → `re_level::load_string` round-trip.

/// A `[wall_decal]` with non-default `density`, `color_head`, and `color_tail`
/// survives `to_lvl_text` → `re_level::load_string` with the authored values
/// preserved.  Also confirms idempotency: a second serialization is byte-identical.
#[test]
fn wall_decal_density_and_colors_round_trip() {
    // Embed a wall_decal with all three non-default fields into a self-contained
    // minimal level so `re_level::load_string` can fully parse it.
    let src = cells_shell(
        "[wall_decal fx_streak]\n\
         surface  = trace\n\
         region   = (0,0,0)-(2,1,2)\n\
         density  = 0.75\n\
         color_head = #FF4040\n\
         color_tail = #2020FF\n",
    );

    // First parse — verify `re_level` reads all three fields correctly.
    let lv = re_level::load_string(&src, "<wd-in>").expect("initial parse");
    assert_eq!(lv.decals.len(), 1, "one decal parsed");
    let d = &lv.decals[0];
    assert_eq!(d.name, "fx_streak");
    assert!(
        (d.density - 0.75_f32).abs() < 1e-6,
        "density parsed as 0.75, got {}",
        d.density
    );
    assert_eq!(d.color_head, 0xFF4040, "color_head parsed as 0xFF4040");
    assert_eq!(d.color_tail, 0x2020FF, "color_tail parsed as 0x2020FF");

    // Serialize.
    let t1 = to_lvl_text(&lv);

    // The emitted text must carry all three keys.
    assert!(
        t1.contains("density = 0.75"),
        "emitted text must carry density = 0.75:\n{t1}"
    );
    assert!(
        t1.contains("color_head = #FF4040"),
        "emitted text must carry color_head = #FF4040:\n{t1}"
    );
    assert!(
        t1.contains("color_tail = #2020FF"),
        "emitted text must carry color_tail = #2020FF:\n{t1}"
    );

    // Re-parse → values survive the full round-trip.
    let rt = re_level::load_string(&t1, "<wd-rt>")
        .unwrap_or_else(|e| panic!("round-trip parse failed: {e}\n---\n{t1}\n---"));
    assert_eq!(rt.decals.len(), 1, "one decal survives round-trip");
    let rd = &rt.decals[0];
    assert_eq!(rd.name, "fx_streak", "name survives");
    assert!(
        (rd.density - 0.75_f32).abs() < 1e-6,
        "density 0.75 must survive round-trip, got {}",
        rd.density
    );
    assert_eq!(
        rd.color_head, 0xFF4040,
        "color_head 0xFF4040 must survive round-trip"
    );
    assert_eq!(
        rd.color_tail, 0x2020FF,
        "color_tail 0x2020FF must survive round-trip"
    );

    // Idempotency: a second serialization is byte-identical.
    let t2 = to_lvl_text(&rt);
    assert_eq!(t1, t2, "wall_decal re-serialization must be byte-identical");
}

/// A `[wall_decal]` with all-default values (density=0.0, colors=0) must NOT
/// emit the optional keys — preserving byte-identity for existing decal-only levels.
#[test]
fn wall_decal_default_fields_omitted() {
    let src = cells_shell(
        "[wall_decal plain]\n\
         surface = solid\n\
         region  = (1,0,0)-(3,2,2)\n",
    );
    let lv = re_level::load_string(&src, "<wd-def>").expect("parses");
    let decal = &lv.decals[0];
    assert_eq!(decal.density, 0.0, "no density key → default 0.0");
    assert_eq!(decal.color_head, 0, "no color_head key → default 0");
    assert_eq!(decal.color_tail, 0, "no color_tail key → default 0");

    let t = to_lvl_text(&lv);
    assert!(
        !t.contains("density"),
        "default density must not be emitted:\n{t}"
    );
    assert!(
        !t.contains("color_head"),
        "default color_head (0) must not be emitted:\n{t}"
    );
    assert!(
        !t.contains("color_tail"),
        "default color_tail (0) must not be emitted:\n{t}"
    );
}

// ===========================================================================
// [material] clarity serializer fidelity — reflective=false + non-default clarity
// ===========================================================================

/// A material with `reflective=false` and `clarity` set to a non-default value
/// (0.5) must survive `to_lvl_text` → `re_level::load_string` with its clarity
/// intact.  Prior to the fix, the writer only emitted `clarity` inside the
/// `if m.reflective` branch, so a non-reflective material with a non-default
/// clarity silently dropped it on save→reload.
#[test]
fn material_non_reflective_non_default_clarity_round_trips() {
    // Build a minimal level via the parser so we have a real Level with a
    // custom [material] section we can manipulate directly.
    let src = "\
[meta]\nname = clarity_rt\n\
[room]\nsize = 6 6 4\nambient = solid\n\
[material mat_a]\nfg = #AABBCC\nbg = #001122\nclarity = 0.5\n\
[node a]\npos = 1,2,2\nis_input = yes\n\
[node b]\npos = 4,2,2\nis_output = yes\nthreshold = 1\n\
[pipe a->b]\npath = (1,2,2) (4,2,2)\n\
[pedestal]\ncenter = 4 2 1\nvolume = 1 1 1\n\
[door]\npos = 4 3 1\ntarget_parts = b\n\
[player]\nspawn = 3 3 1\nfacing = -y\n";

    let level = re_level::load_string(src, "clarity_rt").expect("level must parse");

    // Confirm the parsed material has reflective=false and clarity=0.5.
    let mat = level
        .materials
        .iter()
        .find(|m| m.name == "mat_a")
        .expect("mat_a must be present after parse");
    assert!(!mat.reflective, "mat_a must NOT be reflective");
    assert_eq!(mat.clarity, 0.5, "parsed clarity must be 0.5");

    // Serialise → reload.
    let text = to_lvl_text(&level);

    // The emitted text must contain the clarity line.
    assert!(
        text.contains("clarity = 0.5"),
        "emitted .lvl must carry 'clarity = 0.5' for a non-reflective material:\n{text}"
    );
    // But it must NOT contain a `reflective` line (reflective is false → omitted).
    assert!(
        !text.contains("reflective"),
        "non-reflective material must NOT emit 'reflective':\n{text}"
    );

    let rt = re_level::load_string(&text, "clarity_rt2")
        .unwrap_or_else(|e| panic!("round-trip parse failed: {e}\n---\n{text}\n---"));
    let rt_mat = rt
        .materials
        .iter()
        .find(|m| m.name == "mat_a")
        .expect("mat_a must survive round-trip");
    assert!(
        !rt_mat.reflective,
        "reflective must remain false after round-trip"
    );
    assert_eq!(rt_mat.clarity, 0.5, "clarity 0.5 must survive save→reload");

    // Idempotent: re-serialising gives identical bytes.
    assert_eq!(
        text,
        to_lvl_text(&rt),
        "to_lvl_text must be idempotent for clarity"
    );
}

/// A material at the default clarity (0.85) must NOT emit a `clarity` line —
/// preserving byte-identity for existing materials that never set it.
#[test]
fn material_default_clarity_is_omitted() {
    use re_world::MaterialDef;

    let src = "\
[meta]\nname = clarity_default\n\
[room]\nsize = 6 6 4\nambient = solid\n\
[material mat_b]\nfg = #FFFFFF\nbg = #000000\n\
[node a]\npos = 1,2,2\nis_input = yes\n\
[node b]\npos = 4,2,2\nis_output = yes\nthreshold = 1\n\
[pipe a->b]\npath = (1,2,2) (4,2,2)\n\
[pedestal]\ncenter = 4 2 1\nvolume = 1 1 1\n\
[door]\npos = 4 3 1\ntarget_parts = b\n\
[player]\nspawn = 3 3 1\nfacing = -y\n";

    let level = re_level::load_string(src, "clarity_default").expect("level must parse");

    let mat = level
        .materials
        .iter()
        .find(|m| m.name == "mat_b")
        .expect("mat_b must be present");
    // The parser sets clarity to DEFAULT_CLARITY when the key is absent.
    assert_eq!(
        mat.clarity,
        MaterialDef::DEFAULT_CLARITY,
        "default clarity must be DEFAULT_CLARITY"
    );

    let text = to_lvl_text(&level);

    // No `clarity = …` key line must appear — the default must be omitted.
    // (Use `"clarity = "` not bare `"clarity"` to avoid matching the level name.)
    assert!(
        !text.contains("clarity = "),
        "default clarity ({}) must NOT be emitted:\n{text}",
        MaterialDef::DEFAULT_CLARITY
    );
}

// ── D1 round-trip: a placard body survives `.lvl` → Level → `.lvl` → Level ──

/// Dropping column-0 `#` lines from a `[note]` body must not cost the WRITER
/// anything: an authored body (indented, with a mid-line `#`) round-trips through
/// `to_lvl_text` byte-for-byte, and the comments that were never body text stay
/// out on the second pass too.
#[test]
fn note_body_round_trips_losslessly_without_comments() {
    let src = "\
[meta]
name = rt-note

[room]
size = 6 6 3

[player]
spawn = 2 2 1

[note researcher pos=1,2,1 facing=+x]
  MINISTRY CORPUS #9 — retained.
  second line, indented, kept verbatim.

# a trailing dev comment that must never be body text

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
    let first = re_level::load_string(src, "<rt-note>").expect("fixture parses");
    assert_eq!(first.notes.len(), 1);
    let body1 = first.notes[0].text.trim_end().to_string();
    assert!(body1.contains("MINISTRY CORPUS #9"));
    assert!(!body1.contains("dev comment"));

    let text = to_lvl_text(&first);
    let second = re_level::load_string(&text, "<rt-note-2>").expect("rewritten level parses");
    assert_eq!(second.notes.len(), 1, "the placard survives the write");
    assert_eq!(
        second.notes[0].text.trim_end(),
        body1,
        "note body must round-trip byte-for-byte"
    );
    assert_eq!(second.notes[0].kind, first.notes[0].kind);
    assert_eq!(
        (second.notes[0].x, second.notes[0].y, second.notes[0].z),
        (first.notes[0].x, first.notes[0].y, first.notes[0].z)
    );
}

/// **BUG 2 regression**: an AUTHORED note body line whose first character is `#`
/// (column 0) must survive `to_lvl_text` -> reparse. The editor's caret text
/// field deliberately allows a leading `#` in placard prose
/// (`ts_editor::entity`, the multi-line note-body field), but `to_lvl_text`
/// used to write bodies verbatim/unindented — so a reload would misread that
/// line as a level-file comment (the parser's `[note]` column-0 rule) and
/// silently drop it. Built directly on a `Level`, not from `.lvl` text, so this
/// exercises exactly the "in-memory authored content" path the editor produces.
#[test]
fn note_body_line_starting_with_hash_survives_save_and_reparse() {
    let mut level = re_level::Level {
        name: "hash-note".into(),
        size: (6, 6, 3),
        ..Default::default()
    };
    level.player.x = 2;
    level.player.y = 2;
    level.player.z = 1;
    level.notes.push(re_level::LevelNote {
        kind: re_level::NoteKind::Researcher,
        x: 1,
        y: 2,
        z: 1,
        facing: 0,
        text: "line one\n#9 MINISTRY\nline three".to_string(),
    });

    let text = to_lvl_text(&level);
    let reloaded = re_level::load_string(&text, "<hash-note>")
        .expect("a body line starting with # must still parse");
    assert_eq!(reloaded.notes.len(), 1, "the placard itself must survive");
    let body = &reloaded.notes[0].text;
    assert_eq!(
        body.lines().count(),
        3,
        "all three body lines must survive, got:\n{body:?}"
    );
    assert!(
        body.contains("line one"),
        "first line must survive:\n{body:?}"
    );
    assert!(
        body.contains("#9 MINISTRY"),
        "the '#'-led line must survive:\n{body:?}"
    );
    assert!(
        body.contains("line three"),
        "third line must survive:\n{body:?}"
    );

    // Idempotent from here on: a second save/reparse cycle changes nothing further.
    let text2 = to_lvl_text(&reloaded);
    let reloaded2 = re_level::load_string(&text2, "<hash-note-2>").expect("re-parses again");
    assert_eq!(
        reloaded2.notes[0].text, reloaded.notes[0].text,
        "a second save/reparse cycle must be a no-op"
    );
    assert_eq!(text, text2, "to_lvl_text must be idempotent once escaped");
}

/// Sibling case: a note body line beginning with `#` is not just single — a body
/// that is ENTIRELY `#`-led lines (no indentation at all) must also survive.
#[test]
fn note_body_all_lines_starting_with_hash_survive() {
    let mut level = re_level::Level {
        name: "hash-note-2".into(),
        size: (6, 6, 3),
        ..Default::default()
    };
    level.player.x = 2;
    level.player.y = 2;
    level.player.z = 1;
    level.notes.push(re_level::LevelNote {
        kind: re_level::NoteKind::SystemLog,
        x: 1,
        y: 2,
        z: 1,
        facing: 0,
        text: "#alpha\n#beta\n#gamma".to_string(),
    });

    let text = to_lvl_text(&level);
    let reloaded = re_level::load_string(&text, "<hash-note-all>").expect("re-parses");
    assert_eq!(reloaded.notes.len(), 1);
    let body = &reloaded.notes[0].text;
    assert_eq!(
        body.lines().count(),
        3,
        "all three lines must survive:\n{body:?}"
    );
    for needle in ["#alpha", "#beta", "#gamma"] {
        assert!(body.contains(needle), "{needle} must survive:\n{body:?}");
    }
}

/// `[room] power_budget` MUST round-trip. It is the rule that makes the campaign
/// losable, and this repo has a long history of "parses but never writes" data
/// loss — a writer that dropped it would silently turn every budgeted level back
/// into an unloseable one on the first editor save.
///
/// Also pins the omit-when-default rule: an UNLIMITED level (`0`, the absent-key
/// value) must emit NO `power_budget` line, so no existing level's bytes move.
#[test]
fn round_trip_preserves_power_budget() {
    const BODY: &str = "[meta]
name = pb_rt
[room]
size = 8 8 6
ambient = solid
[node a]
pos = 3,6,2
is_input = yes
[node b]
pos = 5,6,2
is_output = yes
threshold = 1
[pipe a->b]
path = (3,6,2) (5,6,2)
switch_at = (4,6,2)
[inputs]
a = 1
[pedestal]
center = 5 6 1
volume = 1 1 1
[door]
pos = 5 7 1
target_parts = b
[player]
spawn = 4 4 1
facing = +y
";

    // BUDGETED: the value survives write → reparse, and a second write is stable.
    let src = BODY.replacen(
        "ambient = solid
",
        "ambient = solid
power_budget = 4
",
        1,
    );
    let level = re_level::load_string(&src, "pb_rt").expect("budgeted level parses");
    assert_eq!(level.power_budget, 4);
    let text = to_lvl_text(&level);
    assert!(
        text.contains("power_budget = 4"),
        "the writer must emit the budget:
{text}"
    );
    let rt = re_level::load_string(&text, "pb_rt").expect("re-parses");
    assert_eq!(rt.power_budget, 4, "power_budget survives the round trip");
    assert_eq!(
        text,
        to_lvl_text(&rt),
        "to_lvl_text must be idempotent for power_budget"
    );

    // UNLIMITED: no line at all — an un-budgeted level's bytes are unchanged.
    let plain = re_level::load_string(BODY, "pb_rt0").expect("plain level parses");
    assert_eq!(plain.power_budget, 0);
    let plain_text = to_lvl_text(&plain);
    assert!(
        !plain_text.contains("power_budget"),
        "an unlimited level must not grow a power_budget line:
{plain_text}"
    );
}

// ── shipped-corpus sweeps ───────────────────────────────────────────────────
//
// TWO sweeps run over `levels/*.lvl`, and they catch DIFFERENT, non-overlapping
// classes of save bug. Do NOT consolidate them; neither subsumes the other.
//
// 1. `every_shipped_level_round_trips_idempotently` — TEXT vs TEXT.
//      t1 = write(parse(disk)); t2 = write(parse(t1)); assert t1 == t2.
//    Catches the **ADDITION** class (a save that GROWS the file: `[effects] snow`
//    re-expanding into a fresh emitter lattice every save) and the **MANGLING**
//    class (a save whose output re-parses to something else: a `[note]` body line
//    starting with `#` being eaten as a comment).
//    STRUCTURALLY BLIND to deletion: a key the parser reads and the writer never
//    emits is absent from t1 AND from t2, so the two texts still match.
//
// 2. `every_shipped_level_survives_a_save_with_no_structural_loss` — STRUCT vs
//    STRUCT. lv1 = parse(disk); lv2 = parse(write(lv1)); assert lv1 == lv2.
//    Catches the **DELETION** class the text sweep cannot see: any field the
//    parser populates that the writer drops shows up as a field that is set in
//    lv1 and empty/default in lv2. This is the sweep that would have caught
//    `[meta] version`, `[meta] spec_rev`, `[room] floor`, `[room] ceiling` and
//    `[door] target_model` being erased from up to 24 shipped levels on save.
//    It is blind to the addition class (both sides are post-write) and to pure
//    text formatting, which is why sweep 1 must stay.

/// Every file under `levels/` must serialize -> reparse -> serialize to the exact
/// same bytes twice in a row. This is the blanket regression guard for BOTH bugs
/// fixed alongside this test (the synthesized-emitter split and the note-body
/// escape): a narrow per-fixture test can pass while a shipped level still
/// regresses, so this sweeps the whole corpus (63 files as of this writing).
///
/// CLASS: addition + mangling. See the sweep note above — deletion is invisible
/// here and is covered by `every_shipped_level_survives_a_save_with_no_structural_loss`.
#[test]
fn every_shipped_level_round_trips_idempotently() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../levels");
    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("lvl") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let lv = match re_level::load(&path) {
            Ok(lv) => lv,
            Err(e) => {
                failures.push(format!("{name}: failed to load: {e}"));
                continue;
            }
        };
        checked += 1;
        let t1 = to_lvl_text(&lv);
        let lv2 = match re_level::load_string(&t1, &name) {
            Ok(lv2) => lv2,
            Err(e) => {
                failures.push(format!("{name}: re-parse of the written text failed: {e}"));
                continue;
            }
        };
        let t2 = to_lvl_text(&lv2);
        if t1 != t2 {
            failures.push(format!("{name}: to_lvl_text is not idempotent (t1 != t2)"));
        }
    }
    assert!(
        checked >= 60,
        "expected the ~63-level shipped corpus, only found {checked}"
    );
    assert!(
        failures.is_empty(),
        "{} shipped level(s) failed the round-trip sweep:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// A canonical, line-per-field rendering of a [`re_level::Level`] for the
/// structural sweep. Derived `Debug` is used deliberately: it covers EVERY field,
/// including any added later, so a future writer that forgets a new field is
/// caught without anyone remembering to extend this helper.
///
/// Exactly two normalizations are applied, and each is a characterized property of
/// the round-trip rather than a convenience:
///
/// * `source_path` — provenance, not content. `re_level::load` records the file it
///   came from; `load_string` cannot. Comparing it would only ever assert that one
///   side came from disk.
/// * `cells.shapes` / `cells.material_ids` — these are SPARSE MAPS keyed by
///   `(x,y,z)`, carried in a `Vec`. `CellGrid` documents that the writer emits them
///   in canonical `(z,y,x)` order, so a hand-authored `[cells]` block whose rows sit
///   in another order (`STAIRS-showcase.lvl`) comes back REORDERED — 0 entries lost,
///   0 gained. Sorting both sides compares them as the sets they are; a genuinely
///   lost or invented entry still fails, because sorting cannot change the count.
///
/// * `unknown_keys` — a DIAGNOSTIC about the source TEXT, not level content. Each
///   entry carries the line number it was found on, so comparing the vector across
///   two different texts compares line numbers of two different files; and the
///   writer deliberately never emits an unrecognized key, so a level that has any
///   would always "differ" here for a reason this sweep does not own. The finding
///   is not dropped — `re_level`'s `no_shipped_level_has_an_unknown_key` and
///   `unknown_section_findings_are_a_registered_defect_list` are the tests whose
///   whole job it is, and they assert the corpus contents exactly.
///
/// * `props[*].line` / `props[*].keys[*].line` — SOURCE-LINE PROVENANCE, kept only
///   so a prop parse/expansion error can name a line. The writer emits sections in
///   its canonical order (materials early, `[prop]` just before the notes), while
///   the shipped L1–L12 author their `[prop]` blocks after the notes and late
///   `[material]`s — so after a save the same props, with the same keys and values
///   in the same order, simply sit on different lines. Nothing downstream of the
///   parser reads these numbers except error messages. They are zeroed on both
///   sides; name, key order, keys and values are still compared EXACTLY, so a
///   dropped, reordered or altered prop key still fails.
///
/// * `field_defs` — same source-line provenance issue as `props`, PLUS an ordering
///   issue `props` does not have. An inline `field =` attach is parsed in
///   whatever position its host
///   section sits in the source text, but the writer re-embeds it inside that
///   host's own canonically-ordered section (`[node]`/`[pipe]`/`[door]`) while
///   standalone `[field]` sections are always re-emitted last. A level that mixes
///   an inline attach on an EARLY node with a standalone `[field]` authored
///   earlier in the source text (e.g. `L7-hidden-depths.lvl`: a `[field river]`
///   section precedes `[node growth]`'s inline attach on disk, but the growth
///   attach sits inside its node section — textually BEFORE the re-emitted
///   standalone fields — after a save) comes back with `field_defs` reordered: 0
///   entries lost, 0 gained, same reasoning as the `cells` sort above. `line` is
///   zeroed for the same reason as `props[*].line`; `field_defs` is then sorted by
///   name (host+keys are still compared EXACTLY per entry) so the comparison is
///   order-insensitive but content-exact, catching a genuinely dropped, altered or
///   retargeted field while ignoring where in the file it now sits.
///
/// Nothing else is normalized. In particular the `[note]` bodies are compared
/// EXACTLY: the one-shot trailing-`\n` drift the audit found was fixed at its
/// source (`re_level`'s note flush now canonicalizes the captured body), not
/// papered over here.
fn structural_fingerprint(level: &re_level::Level) -> String {
    let mut lv = level.clone();
    lv.source_path = None;
    lv.unknown_keys.clear();
    for p in &mut lv.props {
        p.line = 0;
        for k in &mut p.keys {
            k.line = 0;
        }
    }
    for f in &mut lv.field_defs {
        f.line = 0;
        for k in &mut f.keys {
            k.line = 0;
        }
    }
    lv.field_defs.sort_by(|a, b| a.name.cmp(&b.name));
    // `resolved_fields` (`Level::finalize`'s resolved mirror of `field_defs`, see
    // `Level::fields`) carries the SAME two provenance/ordering quirks: each entry
    // keeps its own `line` (for lvlcheck diagnostics), and the resolve pass walks
    // `field_defs` in its current order, so a reordered `field_defs` reorders this
    // too. Same fix, same reasoning.
    for f in &mut lv.resolved_fields {
        f.line = 0;
    }
    lv.resolved_fields.sort_by(|a, b| a.name.cmp(&b.name));
    if let Some(cells) = lv.cells.as_mut() {
        cells.shapes.sort_by_key(|&(x, y, z, _)| (z, y, x));
        cells.material_ids.sort_by_key(|&(x, y, z, _)| (z, y, x));
    }
    format!("{lv:#?}")
}

/// The first few differing lines between two fingerprints, with context, so a
/// failure names the field instead of dumping two whole levels.
fn fingerprint_diff(a: &str, b: &str) -> String {
    let (la, lb): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
    let mut out = Vec::new();
    let mut shown = 0usize;
    for i in 0..la.len().max(lb.len()) {
        let (x, y) = (
            la.get(i).copied().unwrap_or("<eof>"),
            lb.get(i).copied().unwrap_or("<eof>"),
        );
        if x != y {
            out.push(format!(
                "      line {i}: on disk `{}` | after save `{}`",
                x.trim(),
                y.trim()
            ));
            shown += 1;
            if shown >= 6 {
                out.push("      … (further differences suppressed)".to_string());
                break;
            }
        }
    }
    out.join("\n")
}

/// Every file under `levels/` must survive a save with NO STRUCTURAL LOSS: the
/// `Level` parsed from disk and the `Level` parsed back out of the serialized text
/// must be the same structure, field for field.
///
/// CLASS: **deletion**. This is the sweep `every_shipped_level_round_trips_idempotently`
/// cannot be — that one compares TEXT to TEXT, and a key the parser reads but the
/// writer never emits is missing from both texts, so it compares equal. Here the
/// BEFORE side still carries the key, so the drop is visible.
///
/// It found five silently-deleted keys when first written: `[meta] version` (24
/// levels), `[meta] spec_rev` (22), `[room] floor` (17), `[room] ceiling` (17) and
/// `[door] target_model` (1). Do not merge this into the idempotency sweep; do not
/// replace that one with this. See the sweep note above the two tests.
#[test]
fn every_shipped_level_survives_a_save_with_no_structural_loss() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../levels");
    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    let mut names: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("lvl"))
        .collect();
    names.sort(); // deterministic failure ordering
    for path in names {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let lv1 = match re_level::load(&path) {
            Ok(lv) => lv,
            Err(e) => {
                failures.push(format!("  {name}: failed to load: {e}"));
                continue;
            }
        };
        checked += 1;
        let text = to_lvl_text(&lv1);
        let lv2 = match re_level::load_string(&text, &name) {
            Ok(lv2) => lv2,
            Err(e) => {
                failures.push(format!(
                    "  {name}: re-parse of the written text failed: {e}"
                ));
                continue;
            }
        };
        let (f1, f2) = (structural_fingerprint(&lv1), structural_fingerprint(&lv2));
        if f1 != f2 {
            failures.push(format!(
                "  {name}: the save LOST or CHANGED authored data:\n{}",
                fingerprint_diff(&f1, &f2)
            ));
        }
    }
    assert!(
        checked >= 60,
        "expected the ~63-level shipped corpus, only found {checked}"
    );
    assert!(
        failures.is_empty(),
        "{} shipped level(s) did not survive a save intact — the writer drops \
         something the parser read:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ── D4 round-trip: `[effects] snow` must not COMPOUND across a save ─────────

/// `re_level` expands `[effects] snow` into a lattice of ambient emitters at load.
/// The writer must emit the `snow` key (it does) and NOT the expansion — otherwise a
/// save→load cycle would carry both, and every save would add another snowfall.
#[test]
fn snow_effect_does_not_compound_through_a_save_cycle() {
    let src = "\
[meta]
name = rt-snow

[room]
size = 10 10 4

[player]
spawn = 5 5 1

[node a]
pos = 2,2,1
is_input = yes

[node b]
pos = 6,2,1
is_output = yes

[pipe a->b]
state = OFF
path = (2,2,1) (6,2,1)

[pedestal]
center = 6 2 1
volume = 1 1 1

[door]
pos = 9 5 1

[emitter authored]
origin_world = 5 5 2
rate = 7

[effects]
snow = 0.5
";
    let first = re_level::load_string(src, "<rt-snow>").expect("fixture parses");
    let authored_count = first.emitters.iter().filter(|e| !e.synthesized).count();
    let synth_count = first.emitters.iter().filter(|e| e.synthesized).count();
    assert_eq!(authored_count, 1, "one authored [emitter] section");
    assert!(synth_count > 0, "snow must synthesize emitters");

    let text = to_lvl_text(&first);
    assert!(
        text.contains("snow = 0.5"),
        "the knob itself must be written"
    );
    assert_eq!(
        text.matches("[emitter ").count(),
        1,
        "only the AUTHORED emitter may be serialized:\n{text}"
    );

    let second = re_level::load_string(&text, "<rt-snow-2>").expect("rewritten level parses");
    assert_eq!(
        second.emitters.iter().filter(|e| !e.synthesized).count(),
        1,
        "the authored emitter survives the round trip"
    );
    assert_eq!(
        second.emitters.len(),
        first.emitters.len(),
        "the snowfall must not compound across a save cycle"
    );
    assert_eq!(second.emitters, first.emitters, "and it must be identical");
}

/// **BUG 1 regression**: `ts_editor` always APPENDS a newly authored emitter to
/// the END of `Level::emitters` — i.e. AFTER the synthesized snow tail on a level
/// that already has one (03.lvl ships `snow = 0.4` with zero authored `[emitter]`
/// sections, so its 48 emitters are ALL synthesized). A writer that assumed
/// "authored emitters are a leading prefix" would silently drop an
/// end-of-list authored emitter forever. The per-spec `synthesized` flag must
/// not care WHERE in the list an authored entry lands.
#[test]
fn authored_emitter_appended_after_snow_tail_survives_save() {
    let path = lvl_path("03.lvl");
    let mut level = re_level::load(&path).unwrap_or_else(|e| panic!("03.lvl must load: {e}"));
    assert!(
        level.emitters.iter().all(|e| e.synthesized),
        "fixture assumption: 03.lvl authors zero [emitter] sections, only snow"
    );
    let before = level.emitters.len();
    assert!(before > 0, "fixture assumption: snow synthesizes emitters");

    // Simulate `ts_editor::entity::place_at_cursor`'s `EntityKind::Emitter` arm
    // (`state.level.emitters.push(spec)`): append a fresh AUTHORED spec, with a
    // distinctive rate so it can be found after the round trip.
    level
        .emitters
        .push(re_world::EmitterSpec::new().with_rate(987.5));
    assert_eq!(level.emitters.len(), before + 1);

    let text = to_lvl_text(&level);
    assert_eq!(
        text.matches("[emitter ").count(),
        1,
        "exactly the ONE authored (appended) emitter must be serialized:\n{text}"
    );

    let reloaded = re_level::load_string(&text, "03-with-probe").expect("re-parses");
    let authored: Vec<&re_world::EmitterSpec> = reloaded
        .emitters
        .iter()
        .filter(|e| !e.synthesized)
        .collect();
    assert_eq!(authored.len(), 1, "the probe emitter must survive the save");
    assert!(
        (authored[0].rate - 987.5).abs() < 1e-3,
        "the probe's distinctive rate must survive, got {}",
        authored[0].rate
    );
    // The snow lattice regenerates identically (same density, same room), so the
    // total count matches: 48 synthesized + 1 authored probe.
    assert_eq!(
        reloaded.emitters.len(),
        before + 1,
        "snow lattice count unchanged, probe present"
    );
}

/// **BUG 1 regression (acceptance #3)**: a snow-bearing level's emitter count
/// must NOT grow across repeated load -> save -> load cycles (the synthesized
/// tail must never itself be treated as authored and re-serialized).
#[test]
fn snow_emitter_count_stable_across_two_save_cycles() {
    let path = lvl_path("03.lvl");
    let gen0 = re_level::load(&path).unwrap_or_else(|e| panic!("03.lvl must load: {e}"));
    let count0 = gen0.emitters.len();
    assert!(
        count0 > 0,
        "fixture assumption: 03.lvl synthesizes snow emitters"
    );

    let text1 = to_lvl_text(&gen0);
    let gen1 = re_level::load_string(&text1, "03-cycle-1").expect("cycle 1 re-parses");
    assert_eq!(
        gen1.emitters.len(),
        count0,
        "cycle 1 must not grow the emitter count"
    );

    let text2 = to_lvl_text(&gen1);
    let gen2 = re_level::load_string(&text2, "03-cycle-2").expect("cycle 2 re-parses");
    assert_eq!(
        gen2.emitters.len(),
        count0,
        "cycle 2 must not grow the emitter count"
    );

    assert_eq!(
        text1, text2,
        "the two save cycles must be byte-identical (idempotent)"
    );
}

// ---------------------------------------------------------------------------
// [wall_decal] surface: writer/parser keyword symmetry
// ---------------------------------------------------------------------------

/// A minimal legal level carrying six `[wall_decal]` sections, each with a
/// placeholder `surface = trace` (the only spelling the *old* parser knew for
/// every kind). The test then overwrites each decal's `surface_kind` in memory,
/// so the emitted text exercises all six writer spellings.
fn six_decal_shell() -> String {
    // Six decals, one per surface kind the writer can emit, each on its own
    // `y` row so the stamped regions never overlap.
    String::from(
        "[meta]\nname = decal_surface_rt\n[room]\nsize = 12 16 10\nambient = solid\n\
         [wall_decal dtrace]\nsurface = trace\nregion = (1,2,1)-(2,2,3)\n\
         [wall_decal dmatrix]\nsurface = trace\nregion = (1,3,1)-(2,3,3)\n\
         [wall_decal dsolid]\nsurface = trace\nregion = (1,4,1)-(2,4,3)\n\
         [wall_decal dholo]\nsurface = trace\nregion = (1,5,1)-(2,5,3)\n\
         [wall_decal dportal]\nsurface = trace\nregion = (1,6,1)-(2,6,3)\n\
         [wall_decal dobject]\nsurface = trace\nregion = (1,7,1)-(2,7,3)\n\
         [node a]\npos = 4,13,2\nis_input = yes\n\
         [node b]\npos = 7,13,2\nis_output = yes\nthreshold = 1\n\
         [pipe a->b]\npath = (4,13,2) (7,13,2)\n\
         [pedestal]\ncenter = 7 13 1\nvolume = 1 1 1\n\
         [door]\npos = 7 14 1\ntarget_parts = b\n\
         [player]\nspawn = 6 14 1\nfacing = -y\n",
    )
}

/// **ROUND-TRIP BREAK REGRESSION.**
///
/// `to_lvl_text` emits six `[wall_decal] surface` spellings — `trace`,
/// `matrix`, `solid`, `hologram`, `portal`, `object` — because the `.vox`
/// import path really produces all six `surface_kind`s (tag 13→Hologram,
/// 14→Portal, 15→Object; see `new_tags_round_trip_hologram_portal_object`).
/// The `re_level` parser used to accept only the first three and hard-error
/// with `[wall_decal] surface: unknown kind` on the other three, so a level
/// produced by the import path could be saved to a `.lvl` the game could not
/// load.
///
/// The decal surface is load-bearing, not cosmetic: `build_map` paints the
/// decal region with `kind: d.surface_kind`, so a dropped `hologram` spelling
/// would change the geometry, not just a label.
#[test]
fn every_decal_surface_kind_survives_to_lvl_text_and_reparse() {
    use re_world::CellKind;

    const KINDS: [CellKind; 6] = [
        CellKind::WallTrace,
        CellKind::WallMatrix,
        CellKind::WallSolid,
        CellKind::Hologram,
        CellKind::Portal,
        CellKind::Object,
    ];

    let mut level =
        re_level::load_string(&six_decal_shell(), "decal_rt").expect("shell level must parse");
    assert_eq!(level.decals.len(), 6, "fixture assumption: six decals");
    for (d, k) in level.decals.iter_mut().zip(KINDS) {
        d.surface_kind = k;
    }

    let text = to_lvl_text(&level);
    for word in ["trace", "matrix", "solid", "hologram", "portal", "object"] {
        assert!(
            text.contains(&format!("surface = {word}")),
            "writer must emit `surface = {word}`:\n{text}"
        );
    }

    let rt = re_level::load_string(&text, "decal_rt2")
        .unwrap_or_else(|e| panic!("emitted .lvl must re-parse: {e}\n---\n{text}\n---"));

    assert_eq!(rt.decals.len(), 6, "all six decals survive");
    for (i, (d, k)) in rt.decals.iter().zip(KINDS).enumerate() {
        assert_eq!(d.surface_kind, k, "decal #{i} ({}) surface_kind", d.name);
    }

    // Idempotent: a second emit equals the first.
    assert_eq!(
        text,
        to_lvl_text(&rt),
        "to_lvl_text must be idempotent for decal surfaces"
    );

    // The kinds are load-bearing: build_map paints them into the grid.
    let map = rt.build_map();
    assert_eq!(
        map.cell_at(1, 5, 1),
        CellKind::Hologram,
        "hologram decal paints Hologram cells"
    );
    assert_eq!(
        map.cell_at(1, 6, 1),
        CellKind::Portal,
        "portal decal paints Portal cells"
    );
    assert_eq!(
        map.cell_at(1, 7, 1),
        CellKind::Object,
        "object decal paints Object cells"
    );
}

// ── recipe round-trip: the five pattern recipes (spec 2026-09-02 §1.5) ─────────
//
// `re_level::parse_recipe` and `recipe_to_text` are the two halves of ONE
// vocabulary. A `.lvl` line must survive parse → serialize → parse unchanged, for
// every kind, with non-default knobs, so a load→save→load lap in the editor is the
// identity and the byte-stability sweep stays quiet.
mod recipe_kinds_round_trip {
    use crate::recipe_to_text;
    use re_level::parse_recipe;
    use re_world::{MaterialRecipe, VoiceMode};

    const LINES: [&str; 5] = [
        "scanline freq=11 duty=0.4 warp=0.2 axis=2",
        "truchet seed=-17 density=0.75 scale=6",
        "fractal arms=5 octave_warp=0.1 twist=0.9",
        "substrate scale=2.5 held=0.7 released=0.2",
        "voice scale=4 mode=release",
    ];

    #[test]
    fn every_pattern_recipe_parses_serializes_and_reparses_identically() {
        for line in LINES {
            let first = parse_recipe(line).unwrap_or_else(|| panic!("{line} must parse"));
            let text = recipe_to_text(first);
            let second = parse_recipe(&text)
                .unwrap_or_else(|| panic!("re-parse of `{text}` (from `{line}`)"));
            assert_eq!(first, second, "round-trip drift for `{line}` via `{text}`");
        }
    }

    #[test]
    fn knob_values_survive_the_lap_not_just_the_kind() {
        let r = parse_recipe("truchet seed=-17 density=0.75 scale=6").unwrap();
        assert_eq!(
            r,
            MaterialRecipe::Truchet {
                seed: -17,
                density: 0.75,
                scale: 6.0
            }
        );
        let r = parse_recipe("scanline freq=11 duty=0.4 warp=0.2 axis=2").unwrap();
        assert_eq!(
            r,
            MaterialRecipe::Scanline {
                freq: 11.0,
                duty: 0.4,
                warp: 0.2,
                axis: 2
            }
        );
        let r = parse_recipe("voice scale=4 mode=release").unwrap();
        assert_eq!(
            r,
            MaterialRecipe::Voice {
                scale: 4.0,
                mode: VoiceMode::Release
            }
        );
        assert_eq!(recipe_to_text(r), "voice scale=4 mode=release");
    }

    #[test]
    fn serializer_writes_every_knob_so_a_changed_default_cannot_drift_a_save() {
        for line in LINES {
            let text = recipe_to_text(parse_recipe(line).unwrap());
            let kind = line.split_whitespace().next().unwrap();
            let expected_knobs = line.split_whitespace().count() - 1;
            assert!(text.starts_with(kind), "{text}");
            assert_eq!(
                text.split_whitespace().count() - 1,
                expected_knobs,
                "`{text}` must name every knob of `{line}`"
            );
        }
    }
}

// ── recipe round-trip: `facade` — same discipline as `recipe_kinds_round_trip` above, kept separate
// because it authors a KEYWORD field (`style`) and two BOOLEAN flags neither of
// the five pattern recipes has.
mod recipe_facade_round_trip {
    use crate::recipe_to_text;
    use re_level::parse_recipe;
    use re_world::{FacadeStyle, MaterialRecipe};

    const LINES: [&str; 3] = [
        "facade style=office bay=3 floor=4 lit=0.35 seed=0 ground_band=off sign_row=off",
        "facade style=server_rack bay=8 floor=12 lit=0.6 seed=-42 ground_band=on sign_row=on",
        "facade style=lockers bay=2 floor=6 lit=0.2 seed=17 ground_band=off sign_row=on",
    ];

    #[test]
    fn every_line_parses_serializes_and_reparses_identically() {
        for line in LINES {
            let first = parse_recipe(line).unwrap_or_else(|| panic!("{line} must parse"));
            let text = recipe_to_text(first);
            let second = parse_recipe(&text)
                .unwrap_or_else(|| panic!("re-parse of `{text}` (from `{line}`)"));
            assert_eq!(first, second, "round-trip drift for `{line}` via `{text}`");
        }
    }

    #[test]
    fn every_style_and_flag_survives_the_lap() {
        let r = parse_recipe(
            "facade style=industrial bay=5 floor=7 lit=0.5 seed=3 ground_band=on sign_row=off",
        )
        .unwrap();
        assert_eq!(
            r,
            MaterialRecipe::Facade {
                style: FacadeStyle::Industrial,
                bay: 5.0,
                floor_h: 7.0,
                lit: 0.5,
                seed: 3,
                ground_band: true,
                sign_row: false,
            }
        );
        let text = recipe_to_text(r);
        assert!(text.contains("style=industrial"), "{text}");
        assert!(text.contains("ground_band=on"), "{text}");
        assert!(text.contains("sign_row=off"), "{text}");
    }

    #[test]
    fn serializer_writes_every_knob_so_a_changed_default_cannot_drift_a_save() {
        for line in LINES {
            let text = recipe_to_text(parse_recipe(line).unwrap());
            let expected_knobs = line.split_whitespace().count() - 1;
            assert!(text.starts_with("facade"), "{text}");
            assert_eq!(
                text.split_whitespace().count() - 1,
                expected_knobs,
                "`{text}` must name every knob of `{line}`"
            );
        }
    }
}

/// The `skyline` procedural sky layer (`crate::write_sky_layer` is the exact inverse
/// of `re_world::sky_spec::SkySpec::parse_layer`) must round-trip every knob through
/// a `[sky] layer = …` line, same shape as `recipe_facade_round_trip` above.
mod sky_skyline_round_trip {
    use crate::write_sky_layer;
    use re_world::sky_spec::SkySpec;

    const LINES: [&str; 3] = [
        "skyline blocks=96 density=0.75 height=0.12..0.55 style=office lit=0.3 landmark_rate=0.08",
        "skyline blocks=32 density=0.4 height=0.05..0.3 style=server_rack lit=0.6 landmark_rate=0.0",
        "skyline blocks=160 density=1 height=0.2..0.9 style=residential lit=0.45 landmark_rate=0.2",
    ];

    #[test]
    fn every_line_parses_serializes_and_reparses_identically() {
        for line in LINES {
            let first =
                SkySpec::parse_layer(line).unwrap_or_else(|e| panic!("{line} must parse: {e}"));
            let text = write_sky_layer(&first);
            let second = SkySpec::parse_layer(&text)
                .unwrap_or_else(|e| panic!("re-parse of `{text}` (from `{line}`): {e}"));
            assert_eq!(first, second, "round-trip drift for `{line}` via `{text}`");
        }
    }

    #[test]
    fn every_style_and_knob_survives_the_lap() {
        let first = SkySpec::parse_layer(
            "skyline blocks=48 density=0.9 height=0.1..0.4 style=lockers lit=0.55 landmark_rate=0.12",
        )
        .unwrap();
        let text = write_sky_layer(&first);
        for needle in [
            "blocks=48",
            "density=0.9",
            "height=0.1..0.4",
            "style=lockers",
            "lit=0.55",
            "landmark_rate=0.12",
        ] {
            assert!(text.contains(needle), "`{text}` must carry `{needle}`");
        }
    }
}

// ── [light] / [atmosphere] round-trip (spec 2026-09-02 §1.3) ───────────────────
//
// `to_lvl_text` must emit every `[light]` and the `[atmosphere]` lighting keys it
// reads off `Level::lights` / `Level::atmos` (the fog-DISTANCE keys stay owned by
// `[palette]` — see the emission site's comment), and must OMIT both sections
// entirely for a level that carries no lighting, so the existing byte-stability
// sweep over `levels/*.lvl` stays quiet.
mod lighting_round_trip {
    use super::*;
    use re_world::{HeightFog, PointLight};

    fn minimal_shell(body: &str) -> String {
        format!(
            "[meta]\nname = light-rt\n[room]\nsize = 4 4 3\nambient = solid\n\
             [pedestal]\ncenter = 1 1 1\nvolume = 1 1 1\n[door]\npos = 2 1 1\n\
             [player]\nspawn = 1 1 1\n{body}"
        )
    }

    /// Two `[light]` sections plus a fully-populated `[atmosphere]` survive
    /// re_level::load_string → to_lvl_text → re_level::load_string byte-for-byte
    /// in meaning (every field compares equal), and the emitted text is
    /// idempotent on a second lap.
    #[test]
    fn lights_and_atmosphere_round_trip() {
        let src = minimal_shell(
            "\
[light]
pos = 4.5, 2.0, 1.2
color = #FFB060
radius = 6
intensity = 1.0

[light]
pos = 0, 0, 0
color = #4080FF
radius = 3.5
intensity = 0.25

[atmosphere]
key_dir = -0.4, -1.0, 0.3
key_color = #C0C8FF
ambient = 0.18
fog_height = 0.0 0.6
postfx = clean
",
        );
        let level = re_level::load_string(&src, "light_rt").expect("level must parse");
        assert_eq!(level.lights.len(), 2, "both [light] sections parsed");
        assert!(level.atmos.key_dir.is_some());

        let text = to_lvl_text(&level);
        assert!(
            text.contains("[light]"),
            "emitted text must carry [light]:\n{text}"
        );
        assert!(
            text.contains("[atmosphere]"),
            "emitted text must carry [atmosphere]:\n{text}"
        );

        let rt = re_level::load_string(&text, "light_rt2").unwrap_or_else(|e| {
            panic!("round-trip parse failed: {e}\n---\n{text}\n---");
        });

        assert_eq!(level.lights, rt.lights, "PointLight vec survives the lap");
        assert_eq!(
            level.lights[0],
            PointLight {
                pos: glam::Vec3::new(4.5, 2.0, 1.2),
                color: 0xFFB060,
                radius: 6.0,
                intensity: 1.0,
            }
        );
        assert_eq!(
            level.lights[1],
            PointLight {
                pos: glam::Vec3::ZERO,
                color: 0x4080FF,
                radius: 3.5,
                intensity: 0.25,
            }
        );

        assert_eq!(level.atmos.key_dir, rt.atmos.key_dir);
        assert_eq!(level.atmos.key_color, rt.atmos.key_color);
        assert_eq!(level.atmos.ambient, rt.atmos.ambient);
        assert_eq!(level.atmos.fog_height, rt.atmos.fog_height);
        assert_eq!(level.atmos.postfx, rt.atmos.postfx);
        assert_eq!(
            level.atmos.fog_height,
            Some(HeightFog {
                base: 0.0,
                falloff: 0.6
            })
        );
        assert_eq!(level.atmos.postfx.as_deref(), Some("clean"));

        // Idempotent: a second emit equals the first.
        assert_eq!(
            text,
            to_lvl_text(&rt),
            "to_lvl_text must be idempotent for [light]/[atmosphere]"
        );
    }

    /// A level authored with no `[light]` and no `[atmosphere]` section emits
    /// neither — omit-when-default, so every unlit level in `levels/*.lvl` stays
    /// byte-identical across a save.
    #[test]
    fn unlit_level_emits_neither_section() {
        let src = minimal_shell("");
        let level = re_level::load_string(&src, "unlit_rt").expect("level must parse");
        assert!(level.lights.is_empty());
        assert_eq!(level.atmos, re_level::LevelAtmosphere::default());

        let text = to_lvl_text(&level);
        assert!(
            !text.contains("[light]"),
            "no lights ⇒ no [light] section:\n{text}"
        );
        assert!(
            !text.contains("[atmosphere]"),
            "default atmosphere ⇒ no [atmosphere] section:\n{text}"
        );
    }
}

/// `[texture] max_size` — the load-time decode cap (engine spec §1.4) — must
/// survive a save, or an editor round-trip would silently restore a 134 MB
/// decode the author had deliberately capped.
/// `[texture] height`, `[material] tile` and `[material] relief` (engine
/// temporal-stability + lighting spec 1.4) must survive a save. An editor round-trip
/// that dropped them would turn an authored lit-relief wall back into a flat decal
/// with no error anywhere - the same class of silent data loss `span`/`subcell` hit.
mod image_relief_round_trip {
    use super::*;

    fn shell(tex_body: &str, mat_body: &str) -> String {
        format!(
            "[meta]\nname = relief-rt\n[room]\nsize = 4 4 3\nambient = solid\n\
             [texture pic]\n{tex_body}\
             [material img]\nrecipe = image\ntexture = pic\n{mat_body}surface = wall_solid\n\
             [pedestal]\ncenter = 1 1 1\nvolume = 1 1 1\n[door]\npos = 2 1 1\n\
             [player]\nspawn = 1 1 1\n"
        )
    }

    #[test]
    fn an_authored_height_map_path_survives_the_writer() {
        let lv = re_level::load_string(
            &shell(
                "file = assets/texture/wall_tex.png\nheight = assets/texture/wall_h.png\n",
                "",
            ),
            "<h>",
        )
        .expect("parses");
        assert_eq!(
            lv.textures[0].height.as_deref(),
            Some("assets/texture/wall_h.png")
        );
        let text = crate::to_lvl_text(&lv);
        assert!(
            text.contains("height = assets/texture/wall_h.png"),
            "the relief path is written:\n{text}"
        );
        let back = re_level::load_string(&text, "<h2>").expect("re-parses");
        assert_eq!(
            back.textures[0].height.as_deref(),
            Some("assets/texture/wall_h.png"),
            "and comes back unchanged"
        );
    }

    #[test]
    fn an_authored_tile_survives_the_writer() {
        let lv = re_level::load_string(
            &shell("file = assets/texture/wall_tex.png\n", "tile = 4 2\n"),
            "<t>",
        )
        .expect("parses");
        assert_eq!(lv.materials[0].image_tile, Some((4.0, 2.0)));
        let text = crate::to_lvl_text(&lv);
        assert!(
            text.contains("tile = 4 2"),
            "the repeat is written:\n{text}"
        );
        let back = re_level::load_string(&text, "<t2>").expect("re-parses");
        assert_eq!(back.materials[0].image_tile, Some((4.0, 2.0)));
    }

    #[test]
    fn an_authored_relief_survives_the_writer_with_both_knobs() {
        let lv = re_level::load_string(
            &shell(
                "file = assets/texture/wall_tex.png\n",
                "relief = height_scale=1.25 normal_strength=0.5\n",
            ),
            "<r>",
        )
        .expect("parses");
        let text = crate::to_lvl_text(&lv);
        assert!(
            text.contains("relief = height_scale=1.25 normal_strength=0.5"),
            "both knobs are written in the parser's own token form:\n{text}"
        );
        let back = re_level::load_string(&text, "<r2>").expect("re-parses");
        let r = back.materials[0].image_relief.expect("relief survives");
        assert!((r.height_scale - 1.25).abs() < 1e-6);
        assert!((r.normal_strength - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_material_without_the_new_keys_emits_none_of_them() {
        let lv = re_level::load_string(&shell("file = assets/texture/wall_tex.png\n", ""), "<n>")
            .expect("parses");
        let text = crate::to_lvl_text(&lv);
        for key in ["height =", "tile =", "relief ="] {
            assert!(
                !text.contains(key),
                "an unauthored key must not appear (byte-stability): {key}\n{text}"
            );
        }
    }
}

mod texture_max_size_round_trip {
    use super::*;

    fn shell(tex_body: &str) -> String {
        format!(
            "[meta]\nname = ms-rt\n[room]\nsize = 4 4 3\nambient = solid\n\
             [texture pic]\n{tex_body}\
             [pedestal]\ncenter = 1 1 1\nvolume = 1 1 1\n[door]\npos = 2 1 1\n\
             [player]\nspawn = 1 1 1\n"
        )
    }

    #[test]
    fn an_authored_cap_survives_the_writer() {
        let lv = re_level::load_string(
            &shell("file = assets/texture/wall_tex.png\nmax_size = 512\n"),
            "<ms>",
        )
        .expect("parses");
        assert_eq!(lv.textures[0].max_size, Some(512));
        let text = crate::to_lvl_text(&lv);
        assert!(
            text.contains("max_size = 512"),
            "the cap is written:\n{text}"
        );
        let back = re_level::load_string(&text, "<ms2>").expect("re-parses");
        assert_eq!(
            back.textures[0].max_size,
            Some(512),
            "and comes back unchanged"
        );
    }

    #[test]
    fn an_uncapped_texture_emits_no_max_size_line() {
        let lv = re_level::load_string(&shell("file = assets/texture/wall_tex.png\n"), "<ms>")
            .expect("parses");
        let text = crate::to_lvl_text(&lv);
        assert!(
            !text.contains("max_size"),
            "a texture with no cap stays byte-identical:\n{text}"
        );
    }
}

/// `[texture] frames` + `fps` - the ANIMATED image texture (engine spec 1.4) - must
/// survive a save. An editor round-trip that dropped them would turn an authored
/// looping clip back into a still poster with no error anywhere.
mod texture_frames_round_trip {
    use super::*;

    fn shell(tex_body: &str) -> String {
        format!(
            "[meta]\nname = frames-rt\n[room]\nsize = 4 4 3\nambient = solid\n\
             [texture pic]\n{tex_body}\
             [pedestal]\ncenter = 1 1 1\nvolume = 1 1 1\n[door]\npos = 2 1 1\n\
             [player]\nspawn = 1 1 1\n"
        )
    }

    #[test]
    fn an_authored_frames_and_fps_survive_the_writer() {
        let lv = re_level::load_string(
            &shell(
                "file = assets/texture/wall_tex.png\nframes = assets/texture/clip.gif\nfps = 12\n",
            ),
            "<fr>",
        )
        .expect("parses");
        assert_eq!(
            lv.textures[0].frames.as_deref(),
            Some("assets/texture/clip.gif")
        );
        assert_eq!(lv.textures[0].fps, Some(12.0));

        let text = crate::to_lvl_text(&lv);
        assert!(
            text.contains("frames = assets/texture/clip.gif"),
            "the clip path is written:\n{text}"
        );
        assert!(text.contains("fps = 12"), "the rate is written:\n{text}");

        let back = re_level::load_string(&text, "<fr2>").expect("re-parses");
        assert_eq!(
            back.textures[0].frames.as_deref(),
            Some("assets/texture/clip.gif")
        );
        assert_eq!(back.textures[0].fps, Some(12.0));
        // Idempotent: a second emit equals the first.
        assert_eq!(
            text,
            crate::to_lvl_text(&back),
            "save -> load -> save is the identity"
        );
    }

    #[test]
    fn a_fractional_fps_round_trips_exactly() {
        let lv = re_level::load_string(
            &shell("frames = assets/texture/clip.gif\nfps = 7.5\n"),
            "<fr>",
        )
        .expect("parses");
        let text = crate::to_lvl_text(&lv);
        let back = re_level::load_string(&text, "<fr2>").expect("re-parses");
        assert_eq!(back.textures[0].fps, Some(7.5));
    }

    #[test]
    fn an_fps_without_frames_is_still_written() {
        // The two keys parse independently, so dropping a lone `fps` would be lossy
        // the moment an author set the rate before picking the clip.
        let lv = re_level::load_string(
            &shell("file = assets/texture/wall_tex.png\nfps = 24\n"),
            "<fo>",
        )
        .expect("parses");
        let text = crate::to_lvl_text(&lv);
        assert!(text.contains("fps = 24"), "a lone fps survives:\n{text}");
        assert!(
            !text.contains("frames ="),
            "no clip authored => no frames line:\n{text}"
        );
        let back = re_level::load_string(&text, "<fo2>").expect("re-parses");
        assert_eq!(back.textures[0].fps, Some(24.0));
        assert_eq!(back.textures[0].frames, None);
    }

    #[test]
    fn a_still_texture_emits_neither_key() {
        let lv = re_level::load_string(&shell("file = assets/texture/wall_tex.png\n"), "<st>")
            .expect("parses");
        let text = crate::to_lvl_text(&lv);
        for key in ["frames =", "fps ="] {
            assert!(
                !text.contains(key),
                "an unauthored key must not appear (byte-stability): {key}\n{text}"
            );
        }
    }
}

/// `[part] height / top / top_from` (the half-height, two-tone lit figures) must
/// survive an editor save. Before this, `to_lvl_text` wrote only `size / offset /
/// color / mask`, so one save flattened every reshaped figure back to storey-tall
/// single-colour columns.
#[test]
fn part_height_and_two_tone_top_round_trip() {
    // 04.lvl (archive, non-campaign): a convenient sample with an emit figure
    // of several parts — an engine example, not a campaign level.
    let mut lv = re_level::load(lvl_path("04.lvl")).expect("04.lvl loads");
    assert!(
        lv.parts.len() >= 2,
        "04.lvl has an emit figure with several parts"
    );
    lv.part_looks
        .resize(lv.parts.len(), re_level::PartLook::default());
    // A fraction height with a two-tone top at a non-default split.
    lv.part_looks[0] = re_level::PartLook {
        height: 0.35,
        top: Some((0xFF_F8D8, 0.7)),
    };
    // `half` with the default split (written without a `top_from` line).
    lv.part_looks[1] = re_level::PartLook {
        height: 0.5,
        top: Some((0x40_A0FF, re_level::PART_TOP_FROM_DEFAULT)),
    };
    // Every other part keeps the default look.
    for l in lv.part_looks.iter_mut().skip(2) {
        *l = re_level::PartLook::default();
    }

    let t1 = to_lvl_text(&lv);
    assert!(
        t1.contains("height = 0.35"),
        "fraction height written:\n{t1}"
    );
    assert!(t1.contains("top = #FFF8D8"), "top colour written");
    assert!(
        t1.contains("top_from = 0.7"),
        "non-default top_from written"
    );
    assert!(
        t1.contains("height = half"),
        "0.5 is written as the keyword"
    );
    assert_eq!(
        t1.matches("top_from =").count(),
        1,
        "a default split is not written"
    );

    let back = re_level::load_string(&t1, "<looks>").expect("re-parses");
    for i in 0..lv.parts.len() {
        assert_eq!(
            back.part_look(i),
            lv.part_look(i),
            "part {i} look survives the save"
        );
    }
    assert_eq!(t1, to_lvl_text(&back), "the save is idempotent");
}

/// A `top_from` with no `top` draws nothing (`Level::part_look` drops it), and a
/// level that never authored a look writes none of the three keys.
#[test]
fn part_look_defaults_write_no_keys() {
    let mut lv = re_level::load(lvl_path("04.lvl")).expect("04.lvl loads");
    lv.part_looks = vec![re_level::PartLook::default(); lv.parts.len()];
    lv.part_looks[0].top = Some((u32::MAX, 0.8)); // the parser's "top_from only" state
    let t = to_lvl_text(&lv);
    for key in ["height =", "top =", "top_from ="] {
        let in_parts = t
            .split("[part ")
            .skip(1)
            .map(|sec| sec.split("\n[").next().unwrap_or(""))
            .any(|sec| sec.contains(key));
        assert!(
            !in_parts,
            "an unauthored `{key}` must not appear in a [part]:\n{t}"
        );
    }
}

/// `[field]` sections: the real
/// `to_lvl_text` writer round-trips a full-control `[field]` byte-for-byte
/// through `load_string -> to_lvl_text -> load_string`.
#[test]
fn round_trip_preserves_field_sections() {
    let text = "\
[meta]
name = field_rt
[room]
size = 20 20 10
[pedestal]
center = 1,1,1
volume = 1 1 1
[door]
pos = 5 5 1
[player]
spawn = 2 2 1
[node growth]
pos = 4 4 2
[field strain]
kind = threshold_strain
at = 4, 4, 2
radius = 5.0
inner = 1.0
falloff = smooth
occlude = yes
depth = 0.7
bind = node:growth
invert = yes
drive = starve:growth
base = 0.2
phase = 0.6 1 1 1
dwell = 1.5
idle_boost = yes
priority = 8
variant = memory
seed = 42
";
    let level = re_level::load_string(text, "field_rt").expect("initial parse");
    let out = to_lvl_text(&level);
    let level2 = re_level::load_string(&out, "field_rt2").unwrap_or_else(|e| {
        panic!("round-trip parse failed: {e}\n---\n{out}\n---");
    });

    // Line numbers are SOURCE-LINE PROVENANCE (diagnostics only, like
    // `props[*].line`) and legitimately differ because the writer's canonical
    // section order differs from this fixture's. Normalise them to 0 before
    // comparing keys/values, exactly like the `[prop]` round-trip test does.
    let strip_def_lines = |defs: &[re_level::FieldDef]| -> Vec<re_level::FieldDef> {
        defs.iter()
            .map(|d| re_level::FieldDef {
                name: d.name.clone(),
                line: 0,
                host: d.host.clone(),
                keys: d
                    .keys
                    .iter()
                    .map(|k| re_level::FieldKey {
                        key: k.key.clone(),
                        value: k.value.clone(),
                        line: 0,
                    })
                    .collect(),
            })
            .collect()
    };
    let strip_resolved_lines = |fs: &[re_level::ResolvedField]| -> Vec<re_level::ResolvedField> {
        fs.iter()
            .map(|f| {
                let mut f = f.clone();
                f.line = 0;
                f
            })
            .collect()
    };

    assert_eq!(
        strip_def_lines(&level.field_defs),
        strip_def_lines(&level2.field_defs),
        "verbatim [field] keys survive the round trip"
    );
    assert_eq!(
        strip_resolved_lines(level.fields()),
        strip_resolved_lines(level2.fields()),
        "resolved fields survive the round trip"
    );
    assert_eq!(level2.fields().len(), 1);
}

/// §4.1 inline host-attach round-trips through the real writer for the three
/// host kinds it covers: `[node]`, `[pipe]` and `[door]`. The written text
/// carries `field = ARCHETYPE k=v…` INSIDE the host section (never a separate
/// `[field NAME]`), and re-parsing it must resolve to the identical
/// `ResolvedField` list (host-implied `at`/shape/drive included).
#[test]
fn round_trip_preserves_inline_field_attaches() {
    let text = "\
[meta]
name = inline_field_rt
[room]
size = 20 20 10
[pedestal]
center = 1,1,1
volume = 1 1 1
[door]
pos = 17,6,1
field = exit_detune radius=7
[player]
spawn = 2 2 1
[node growth]
pos = 4,4,2
field = threshold_strain depth=0.6 base=0.2
[node a]
pos = 2,12,2
[node b]
pos = 9,6,2
[pipe a->b]
path = (2,12,2) (5,12,2) (9,12,2) (9,6,2)
field = bit_rot depth=0.5
";
    let level = re_level::load_string(text, "inline_field_rt").expect("initial parse");
    assert_eq!(level.fields().len(), 3);
    let out = to_lvl_text(&level);

    // The written form carries the `field =` line INSIDE its host section —
    // never as a separate top-level `[field ...]` (that would double it on
    // the next load).
    assert!(
        !out.contains("[field "),
        "inline attaches must not also write a [field] section:\n{out}"
    );
    let node_growth = out
        .split("[node growth]")
        .nth(1)
        .unwrap()
        .split("\n[")
        .next()
        .unwrap();
    assert!(
        node_growth.contains("field = threshold_strain"),
        "{node_growth}"
    );
    let pipe_ab = out
        .split("[pipe a->b]")
        .nth(1)
        .unwrap()
        .split("\n[")
        .next()
        .unwrap();
    assert!(pipe_ab.contains("field = bit_rot"), "{pipe_ab}");
    let door = out
        .split("[door]")
        .nth(1)
        .unwrap()
        .split("\n[")
        .next()
        .unwrap();
    assert!(door.contains("field = exit_detune"), "{door}");

    let level2 = re_level::load_string(&out, "inline_field_rt2")
        .unwrap_or_else(|e| panic!("round-trip parse failed: {e}\n---\n{out}\n---"));
    assert_eq!(level2.fields().len(), 3);

    // Sorted by name: the writer's canonical section order (`[node]`s, then
    // `[pipe]`s, then `[door]`) legitimately differs from this fixture's
    // authored order (`[door]` first), so the resolved list's ORDER moves
    // even though its CONTENT does not — the same order-insensitivity the
    // shipped-corpus round-trip tests already rely on for `[prop]`/`[field]`.
    let strip = |fs: &[re_level::ResolvedField]| -> Vec<re_level::ResolvedField> {
        let mut v: Vec<re_level::ResolvedField> = fs
            .iter()
            .map(|f| {
                let mut f = f.clone();
                f.line = 0;
                f
            })
            .collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    };
    assert_eq!(
        strip(level.fields()),
        strip(level2.fields()),
        "resolved inline fields survive the round trip"
    );
}
