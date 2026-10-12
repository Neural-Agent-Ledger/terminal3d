//! End-to-end coverage of the SHIPPING entry point [`vox_to_level`] (R-H5).
//!
//! The in-crate unit tests only exercise the pure-data [`derive_level`]. This
//! integration test closes the gap: it AUTHORS a real binary `.vox` blob with
//! `dot_vox`'s writer (the same container an editor exports), writes a matching
//! companion `.toml`, and runs the full `vox_to_level` path —
//! `dot_vox::load_bytes` → palette indexing → multi-model flatten → derivation.
//!
//! It asserts the result equals the equivalent `derive_level` output, that the
//! stamped map is non-empty, and that the puzzle is solvable in principle (a
//! switch sits on the pipe), proving the dot_vox parse + palette path works.

use dot_vox::{Color, DotVoxData, Model, Size, Voxel as DotVoxel};
use re_voximport::{derive_level, vox_to_level, Config, Voxel};

// Default tag RGBs from docs/VOX_IMPORT.md.
const C_NODE_IN: [u8; 3] = [0xFF, 0xD0, 0x00];
const C_NODE_OUT: [u8; 3] = [0xFF, 0x30, 0x30];
const C_PIPE: [u8; 3] = [0x80, 0x00, 0xFF];
const C_SWITCH: [u8; 3] = [0xFF, 0x00, 0xFF];
const C_DOOR: [u8; 3] = [0x00, 0xFF, 0xFF];
const C_PLAYER: [u8; 3] = [0xFF, 0xA5, 0x00];

const ROOM: (u32, u32, u32) = (6, 6, 4);

/// The tiny-but-complete level, as `(x, y, z, rgb)` tuples:
/// a room with one input + one output node, a 2-cell pipe between them with a
/// switch voxel, a door voxel, and a player voxel.
fn cells() -> Vec<(i32, i32, i32, [u8; 3])> {
    vec![
        (1, 2, 2, C_NODE_IN),
        (4, 2, 2, C_NODE_OUT),
        (2, 2, 2, C_PIPE),
        (3, 2, 2, C_PIPE),
        (2, 3, 2, C_SWITCH), // 6-adjacent to pipe cell (2,2,2)
        (5, 2, 1, C_DOOR),
        (3, 2, 1, C_PLAYER),
    ]
}

/// The companion `.toml` (names the two nodes + an initial input).
const TOML: &str = r#"
[meta]
name = "vox_e2e"
title = "VOX END TO END"

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

/// Author a real `.vox` byte blob for `cells()` via `dot_vox`'s writer.
///
/// Each distinct tag colour gets its own palette slot; every voxel references
/// it by index. Palette index `k` (0-based, in memory) maps to `palette[k]`,
/// which is exactly what `vox_to_level` reads, so the colours round-trip.
fn author_vox() -> Vec<u8> {
    let cells = cells();

    // Distinct colours → palette slots (stable order of first appearance).
    let mut palette_colors: Vec<[u8; 3]> = Vec::new();
    let mut voxels: Vec<DotVoxel> = Vec::new();
    for &(x, y, z, rgb) in &cells {
        let idx = palette_colors
            .iter()
            .position(|c| *c == rgb)
            .unwrap_or_else(|| {
                palette_colors.push(rgb);
                palette_colors.len() - 1
            });
        voxels.push(DotVoxel {
            x: x as u8,
            y: y as u8,
            z: z as u8,
            i: idx as u8, // 0-based; the writer adds the +1 for the file format
        });
    }

    // Build a full 256-entry palette (standard size). Tag colours sit at the
    // low indices we referenced above; the rest are opaque black filler.
    let mut palette: Vec<Color> = Vec::with_capacity(256);
    for c in &palette_colors {
        palette.push(Color {
            r: c[0],
            g: c[1],
            b: c[2],
            a: 255,
        });
    }
    while palette.len() < 256 {
        palette.push(Color {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        });
    }

    let data = DotVoxData {
        version: 150,
        index_map: (0u8..=255).collect(),
        models: vec![Model {
            size: Size {
                x: ROOM.0,
                y: ROOM.1,
                z: ROOM.2,
            },
            voxels,
        }],
        palette,
        materials: vec![],
        scenes: vec![],
        layers: vec![],
    };

    let mut bytes = Vec::new();
    data.write_vox(&mut bytes).expect("write_vox");
    bytes
}

/// The equivalent in-memory voxel list for `derive_level`.
fn in_memory_voxels() -> Vec<Voxel> {
    cells()
        .into_iter()
        .map(|(x, y, z, rgb)| Voxel { x, y, z, rgb })
        .collect()
}

#[test]
fn vox_blob_round_trips_through_dot_vox() {
    // The hand-authored blob must parse back losslessly via dot_vox itself.
    let bytes = author_vox();
    let data = dot_vox::load_bytes(&bytes).expect("dot_vox must re-parse our blob");
    assert_eq!(data.models.len(), 1, "one model");
    assert_eq!(
        data.models[0].voxels.len(),
        cells().len(),
        "all voxels survive"
    );
    // Every voxel's palette colour resolves to the tag colour we authored.
    let cells = cells();
    for v in &data.models[0].voxels {
        let c = data.palette[v.i as usize];
        let found = (c.r, c.g, c.b);
        let want = cells
            .iter()
            .find(|(x, y, z, _)| (*x, *y, *z) == (v.x as i32, v.y as i32, v.z as i32))
            .map(|(_, _, _, rgb)| (rgb[0], rgb[1], rgb[2]))
            .expect("voxel position present");
        assert_eq!(found, want, "palette colour at ({},{},{})", v.x, v.y, v.z);
    }
}

#[test]
fn vox_to_level_matches_derive_level() {
    let bytes = author_vox();

    // The SHIPPING path: real .vox bytes + .toml → Level.
    let from_vox = vox_to_level(&bytes, TOML).expect("vox_to_level on authored blob");

    // The pure-data path, for equivalence.
    let cfg: Config = toml::from_str(TOML).expect("toml");
    let from_mem = derive_level(&in_memory_voxels(), ROOM, &cfg).expect("derive_level");

    // --- the two paths agree on the puzzle graph ---
    assert_eq!(from_vox.name, from_mem.name, "name");
    assert_eq!(from_vox.size, from_mem.size, "size");
    assert_eq!(from_vox.nodes.len(), from_mem.nodes.len(), "node count");
    for n in &from_mem.nodes {
        let vi = from_vox
            .node_index(&n.name)
            .expect("node by name in vox path");
        let vn = &from_vox.nodes[vi];
        assert_eq!((vn.x, vn.y, vn.z), (n.x, n.y, n.z), "node {} pos", n.name);
        assert_eq!(vn.is_input, n.is_input, "node {} is_input", n.name);
        assert_eq!(vn.is_output, n.is_output, "node {} is_output", n.name);
    }
    assert_eq!(from_vox.pipes.len(), from_mem.pipes.len(), "pipe count");
    for (vp, mp) in from_vox.pipes.iter().zip(&from_mem.pipes) {
        assert_eq!(vp.from, mp.from, "pipe from");
        assert_eq!(vp.to, mp.to, "pipe to");
        assert_eq!(vp.path, mp.path, "pipe path");
        assert_eq!(vp.switch_cell, mp.switch_cell, "pipe switch index");
    }
    assert_eq!(
        (from_vox.door.x, from_vox.door.y, from_vox.door.z),
        (from_mem.door.x, from_mem.door.y, from_mem.door.z),
        "door"
    );
    assert_eq!(
        (from_vox.player.x, from_vox.player.y, from_vox.player.z),
        (from_mem.player.x, from_mem.player.y, from_mem.player.z),
        "player"
    );
    assert_eq!(
        from_vox.door.target_parts, from_mem.door.target_parts,
        "target_parts"
    );

    // --- the world is not empty ---
    let map = from_vox.build_map();
    let walls = map.count(re_world::CellKind::WallTrace)
        + map.count(re_world::CellKind::WallSolid)
        + map.count(re_world::CellKind::WallMatrix);
    assert!(
        walls > 0,
        "stamped map from vox must contain walls, got {walls}"
    );
    // 2 node cells + 1 door anchor = 3 Object cells total.
    assert!(
        map.count(re_world::CellKind::Object) >= 3,
        "node + door cells stamped as Object"
    );

    // --- solvable in principle: the pipe carries a switch the player can flip ---
    assert_eq!(from_vox.pipes.len(), 1, "one pipe");
    let pipe = &from_vox.pipes[0];
    let switch_idx = pipe.switch_cell.expect("a switch sits on the pipe");
    assert!(switch_idx < pipe.path.len(), "switch index within the path");
    // The input feeds the pipe whose far end is the output node → a route exists.
    assert!(
        from_vox.nodes[pipe.from_node].is_input,
        "pipe starts at an input"
    );
    assert!(
        from_vox.nodes[pipe.to_node].is_output,
        "pipe ends at the output"
    );
}

/// Bonus: drive the BUILT `vox2lvl` binary against the fixture and confirm its
/// stdout is valid `.lvl` that `re_level` parses back to the same graph. This
/// exercises the actual CLI a level author would run.
#[test]
fn vox2lvl_binary_emits_parseable_lvl() {
    use std::io::Write;
    use std::process::Command;

    // Write the .vox + .toml fixture into the test's temp dir.
    let dir = std::env::temp_dir().join(format!("re_voximport_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir tmp");
    let vox_path = dir.join("fixture.vox");
    let toml_path = dir.join("fixture.toml");
    std::fs::File::create(&vox_path)
        .and_then(|mut f| f.write_all(&author_vox()))
        .expect("write .vox");
    std::fs::File::create(&toml_path)
        .and_then(|mut f| f.write_all(TOML.as_bytes()))
        .expect("write .toml");

    // Cargo exposes the built binary path via this env var.
    let bin = env!("CARGO_BIN_EXE_vox2lvl");
    let out = Command::new(bin)
        .arg(&vox_path)
        .arg(&toml_path)
        .output()
        .expect("run vox2lvl");
    assert!(
        out.status.success(),
        "vox2lvl exited non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let lvl_text = String::from_utf8(out.stdout).expect("stdout utf8");
    let parsed = re_level::load_string(&lvl_text, "vox2lvl-stdout")
        .unwrap_or_else(|e| panic!("vox2lvl stdout must parse: {e}\n---\n{lvl_text}\n---"));

    assert_eq!(parsed.name, "vox_e2e");
    assert_eq!(
        parsed.nodes.len(),
        2,
        "two nodes survive the CLI round-trip"
    );
    assert_eq!(parsed.pipes.len(), 1, "one pipe survives");
    assert!(
        parsed.pipes[0].switch_cell.is_some(),
        "switch survives the CLI round-trip"
    );
    assert!(
        parsed.build_map().count(re_world::CellKind::Object) >= 3,
        "map non-empty (nodes + door as Objects)"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
