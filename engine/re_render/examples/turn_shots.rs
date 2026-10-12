//! A live 2-frame ASCII dump pair showing a wall texture move WITH the wall
//! as the camera turns a few degrees — the direct visual evidence for the
//! turn/strafe tests in `re_render/src/turn_test.rs`.
//!
//! Casts the same small brick/circuit test room `turn_test.rs` uses through
//! the same LIVE near-cast entry point `composite_render_sys` calls
//! (`build_world_layers_fov_into`), once facing the south (brick) wall
//! dead-on and once turned by a few degrees, and writes both plain-glyph
//! ASCII grids to files so they can be diffed/eyeballed.
//!
//! Usage: `cargo run -p re_render --example turn_shots -- <out_dir>`
//! (defaults to the current directory if no `out_dir` is given).

use glam::Vec3;
use re_core::{CompositeStack, Framebuffer};
use re_render::{build_world_layers_fov_into, registry_for_map, Fov};
use re_world::{CameraPose, Map};

const ROOM_SX: usize = 12;
const ROOM_SY: usize = 12;
const ROOM_SZ: usize = 4;

/// The same room `re_render::lod::turn_test`'s tests use: a plain box with the
/// south wall (`y = 0`) left `WallSolid` (brick) and the north wall turned
/// into `WallTrace` (circuit).
fn turn_room() -> Map {
    let mut map = Map::new(ROOM_SX, ROOM_SY, ROOM_SZ);
    for z in 0..ROOM_SZ {
        for y in 0..ROOM_SY {
            for x in 0..ROOM_SX {
                let on_perimeter = x == 0 || x == ROOM_SX - 1 || y == 0 || y == ROOM_SY - 1;
                let kind = if y == ROOM_SY - 1 && x > 0 && x < ROOM_SX - 1 {
                    re_world::CellKind::WallTrace
                } else if on_perimeter {
                    re_world::CellKind::WallSolid
                } else if z == 0 {
                    re_world::CellKind::Floor
                } else if z == ROOM_SZ - 1 {
                    re_world::CellKind::Ceiling
                } else {
                    re_world::CellKind::Empty
                };
                map.set(x, y, z, kind);
            }
        }
    }
    map
}

fn grid_to_text(stack: &CompositeStack, cols: usize, rows: usize) -> String {
    let mut out = String::with_capacity((cols + 1) * rows);
    for row in 0..rows {
        for col in 0..cols {
            let ch = stack
                .opaque
                .get(col, row)
                .and_then(|c| char::from_u32(c.glyph))
                .unwrap_or(' ');
            out.push(ch);
        }
        out.push('\n');
    }
    out
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());

    let map = turn_room();
    let registry = registry_for_map(&map);
    let fov = Fov {
        h_preset_deg: 60.0,
        cell_aspect: 0.5,
        stretch: 1.0,
    };

    let cols = 100usize;
    let rows = 32usize;

    // Facing the south (brick) wall dead-on, turned by a few degrees over two
    // frames — exactly the reported symptom ("turning with the mouse").
    let base_yaw = std::f32::consts::PI;
    let turn_deg = 4.0f32;
    let cam0 = CameraPose {
        pos: Vec3::new(6.0, 6.0, re_world::STANDING_EYE_Z),
        yaw: base_yaw,
        pitch: 0.0,
    };
    let cam1 = CameraPose {
        pos: cam0.pos,
        yaw: base_yaw + turn_deg.to_radians(),
        pitch: 0.0,
    };

    let mut scratch = Framebuffer::new(cols, rows);
    let mut stack0 = CompositeStack::new(cols, rows);
    build_world_layers_fov_into(&map, &cam0, &registry, &fov, &mut scratch, &mut stack0);
    let mut stack1 = CompositeStack::new(cols, rows);
    build_world_layers_fov_into(&map, &cam1, &registry, &fov, &mut scratch, &mut stack1);

    let text0 = grid_to_text(&stack0, cols, rows);
    let text1 = grid_to_text(&stack1, cols, rows);

    std::fs::create_dir_all(&out_dir).ok();
    let p0 = format!("{out_dir}/turn_frame_0_yaw{:.4}.txt", cam0.yaw);
    let p1 = format!("{out_dir}/turn_frame_1_yaw{:.4}.txt", cam1.yaw);
    std::fs::write(&p0, &text0).expect("write frame 0");
    std::fs::write(&p1, &text1).expect("write frame 1");

    let changed = text0
        .lines()
        .zip(text1.lines())
        .filter(|(a, b)| a != b)
        .count();
    println!("wrote {p0}");
    println!("wrote {p1}");
    println!(
        "{changed}/{rows} rows differ between the {turn_deg} degree turn's two frames \
         (0 would mean the view never scrolled)"
    );
}
