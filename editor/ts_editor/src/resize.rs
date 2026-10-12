//! RESIZE-EXISTING-LEVEL — grow or shrink the live level's grid while preserving
//! content. A pure helper over [`EditorState`]: NO terminal, NO I/O — it only
//! reads/writes the editor's data (`map`, `level`, cursor/pan/z/cam), so the whole
//! module is unit-testable with no terminal present.
//!
//! ## What [`resize_level`] does
//!
//! 1. Builds a NEW [`re_world::Map`] of the requested `(sx, sy, sz)` and COPIES
//!    every cell — `CellKind` + per-cell [`re_core::Shape`] + `side_data`
//!    (material id) — from the old map for the OVERLAPPING region
//!    `min(old, new)` on each axis. New cells stay `Empty`/`Full`.
//! 2. Updates `level.size` (and clears the cached `[cells]` grid so the next save
//!    re-captures the resized map). The next save then writes the new `[room] size`
//!    and a `[cells]` grid of the new dimensions.
//! 3. CLAMPS every POINT entity (node / door / spawn / pedestal centre / part
//!    anchor / vox anchor / note / emitter origin) into the new bounds, and clamps
//!    every REGION entity's corners (trigger / zone-input). Clamping (not dropping)
//!    is the safe choice — no index is invalidated, so `re_level::finalize` never
//!    sees a dangling node/zone reference on reload.
//! 4. PIPE PATHS: a pipe's endpoints are nodes (by name/index), which survive the
//!    resize, so the LINK is always kept. The expanded per-cell `path`, however,
//!    may now exit the shrunken grid. If ANY path cell falls outside the new
//!    bounds, the path (and its `switch_cell`) is CLEARED so it can be re-drawn —
//!    the non-crashing option (a clamped path could self-intersect or detach from
//!    its endpoints). A fully in-bounds path is left untouched.
//! 5. Fixes up the editor VIEW state: `cursor`, `pan`, `z`, and the preview
//!    `cam.pos` are all clamped into the new bounds so nothing is out-of-range.
//!
//! The caller ([`crate::settings::resize`]) wraps the whole thing in ONE undo
//! checkpoint, so a resize is a single undoable edit.

use crate::settings::{MAX_DIM, MIN_DIM};
use crate::state::EditorState;
use re_world::{CellKind, Map};

/// Clamp an `i32` cell coordinate into `0..len` (inclusive of `0`, exclusive of
/// `len`). A zero-length axis (impossible after [`resize_level`]'s clamp, but
/// defensive) yields `0`.
#[inline]
fn clamp_coord(v: i32, len: usize) -> i32 {
    if len == 0 {
        return 0;
    }
    v.clamp(0, len as i32 - 1)
}

/// Build a NEW [`Map`] of `(nx, ny, nz)` and copy the overlapping region from
/// `old` — kind, shape, and side_data — for every cell in `min(old, new)` on each
/// axis. New cells stay `Empty` (kind) / `Full` (shape) / `0` (side_data). Pure;
/// allocates a fresh map.
fn remap_cells(old: &Map, nx: usize, ny: usize, nz: usize) -> Map {
    let (ox, oy, oz) = old.size();
    let mut new = Map::new(nx, ny, nz);
    let cx = ox.min(nx);
    let cy = oy.min(ny);
    let cz = oz.min(nz);
    for z in 0..cz {
        for y in 0..cy {
            for x in 0..cx {
                let (xi, yi, zi) = (x as i32, y as i32, z as i32);
                let kind = old.cell_at(xi, yi, zi);
                if kind != CellKind::Empty {
                    let side = old.side_data_at(xi, yi, zi);
                    new.set_cell(x, y, z, kind, side);
                }
                // Shapes are stored sparsely (default Full); copy any non-Full
                // entry so wedge/ramp geometry survives the remap.
                let shape = old.shape_at(xi, yi, zi);
                if shape != re_core::Shape::Full {
                    new.set_shape(x, y, z, shape);
                }
            }
        }
    }
    new
}

/// Clamp every POINT entity and REGION entity in `level` into a grid of
/// `(sx, sy, sz)`, and clear any pipe path that now exits the grid. Pure over the
/// level data. See the module doc for the per-entity policy.
fn clamp_entities(level: &mut re_level::Level, sx: usize, sy: usize, sz: usize) {
    // Point entities: nodes.
    for n in &mut level.nodes {
        n.x = clamp_coord(n.x, sx);
        n.y = clamp_coord(n.y, sy);
        n.z = clamp_coord(n.z, sz);
    }
    // Door anchor.
    level.door.x = clamp_coord(level.door.x, sx);
    level.door.y = clamp_coord(level.door.y, sy);
    level.door.z = clamp_coord(level.door.z, sz);
    // Player spawn.
    level.player.x = clamp_coord(level.player.x, sx);
    level.player.y = clamp_coord(level.player.y, sy);
    level.player.z = clamp_coord(level.player.z, sz);
    // Pedestal centre.
    level.pedestal.cx = clamp_coord(level.pedestal.cx, sx);
    level.pedestal.cy = clamp_coord(level.pedestal.cy, sy);
    level.pedestal.cz = clamp_coord(level.pedestal.cz, sz);
    // Part anchors.
    for p in &mut level.parts {
        p.off_x = clamp_coord(p.off_x, sx);
        p.off_y = clamp_coord(p.off_y, sy);
        p.off_z = clamp_coord(p.off_z, sz);
    }
    // Vox anchors.
    for v in &mut level.voxes {
        v.off_x = clamp_coord(v.off_x, sx);
        v.off_y = clamp_coord(v.off_y, sy);
        v.off_z = clamp_coord(v.off_z, sz);
    }
    // Notes.
    for note in &mut level.notes {
        note.x = clamp_coord(note.x, sx);
        note.y = clamp_coord(note.y, sy);
        note.z = clamp_coord(note.z, sz);
    }
    // Emitter origins (world-space cell coordinates; screen origins are viewport-
    // pinned and never out of grid bounds, so they are left alone).
    for e in &mut level.emitters {
        if let re_world::EmitterOrigin::World(p) = &mut e.origin {
            p.x = p.x.clamp(0.0, sx as f32);
            p.y = p.y.clamp(0.0, sy as f32);
            p.z = p.z.clamp(0.0, sz as f32);
        }
    }
    // Region entities: trigger fields (re_core::Trigger carries a Region::Box).
    for t in &mut level.triggers {
        if let re_core::Region::Box { min, max } = &mut t.region {
            min.0 = clamp_coord(min.0, sx);
            min.1 = clamp_coord(min.1, sy);
            min.2 = clamp_coord(min.2, sz);
            max.0 = clamp_coord(max.0, sx);
            max.1 = clamp_coord(max.1, sy);
            max.2 = clamp_coord(max.2, sz);
        }
    }
    // Region entities: zone inputs.
    for zi in &mut level.zone_inputs {
        zi.x0 = clamp_coord(zi.x0, sx);
        zi.y0 = clamp_coord(zi.y0, sy);
        zi.z0 = clamp_coord(zi.z0, sz);
        zi.x1 = clamp_coord(zi.x1, sx);
        zi.y1 = clamp_coord(zi.y1, sy);
        zi.z1 = clamp_coord(zi.z1, sz);
    }
    // Pipes: keep the LINK (endpoints are nodes by name/index, which survive). If
    // the expanded path now exits the grid, clear the path + switch so it can be
    // re-drawn — the non-crashing option (a clamped path could self-intersect).
    let in_bounds = |c: &(i32, i32, i32)| -> bool {
        c.0 >= 0
            && c.1 >= 0
            && c.2 >= 0
            && (c.0 as usize) < sx
            && (c.1 as usize) < sy
            && (c.2 as usize) < sz
    };
    for pipe in &mut level.pipes {
        if !pipe.path.iter().all(in_bounds) {
            pipe.path.clear();
            pipe.switch_cell = None;
            pipe.switch_at_raw = None;
        }
    }
}

/// RESIZE the current level to `(sx, sy, sz)`, preserving overlapping content and
/// clamping everything out-of-range. Pure over [`EditorState`] — does NOT push an
/// undo snapshot (the caller wraps it in one checkpoint) and does NOT touch the
/// dirty flag. Returns the CLAMPED `(sx, sy, sz)` actually applied (each axis is
/// clamped to `MIN_DIM..=MAX_DIM`).
///
/// Steps (see module doc): build + copy a new map, update `level.size`, drop the
/// cached `[cells]` grid, clamp entities + pipe paths, clamp the view state.
pub fn resize_level(
    state: &mut EditorState,
    sx: usize,
    sy: usize,
    sz: usize,
) -> (usize, usize, usize) {
    let nx = sx.clamp(MIN_DIM, MAX_DIM);
    let ny = sy.clamp(MIN_DIM, MAX_DIM);
    let nz = sz.clamp(MIN_DIM, MAX_DIM);

    // 1. New map with the overlapping region copied. The preview baseline is
    //    remapped IDENTICALLY so the rebuild diff-merge keeps comparing
    //    like against like at the new size.
    state.map = remap_cells(&state.map, nx, ny, nz);
    state.preview_baseline = state
        .preview_baseline
        .as_ref()
        .map(|b| remap_cells(b, nx, ny, nz));

    // 2. Level size + drop the stale cached cells grid (the next save re-captures
    //    the resized map, writing the new [room] size + [cells] grid). The
    //    as-loaded provenance is size-stale too.
    state.level.size = (nx, ny, nz);
    state.level.cells = None;
    state.loaded_cells = None;

    // 3. Clamp/clear entities + pipe paths into the new bounds.
    clamp_entities(&mut state.level, nx, ny, nz);

    // 4. View state: cursor / pan / z / preview cam into range.
    state.cursor.0 = state.cursor.0.min(nx.saturating_sub(1));
    state.cursor.1 = state.cursor.1.min(ny.saturating_sub(1));
    state.pan.0 = state.pan.0.min(nx.saturating_sub(1));
    state.pan.1 = state.pan.1.min(ny.saturating_sub(1));
    state.z = state.z.min(nz.saturating_sub(1));
    state.cam.pos.x = state.cam.pos.x.clamp(0.0, nx as f32);
    state.cam.pos.y = state.cam.pos.y.clamp(0.0, ny as f32);
    state.cam.pos.z = state.cam.pos.z.clamp(0.0, nz as f32);

    (nx, ny, nz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_world::CellKind;

    /// (a) GROW: overlapping cells (kind + shape + side_data) are preserved, new
    /// cells are Empty, and the size is updated.
    #[test]
    fn grow_preserves_overlap_and_adds_empty_cells() {
        let mut state = EditorState::new_empty(4, 4, 2);
        // Distinctive content inside the original bounds.
        state.map.set(3, 3, 1, CellKind::WallSolid);
        state.map.set_cell(2, 1, 0, CellKind::Object, 7); // tagged side_data
        state.map.set(1, 1, 0, CellKind::WallSolid);
        state
            .map
            .set_shape(1, 1, 0, re_core::Shape::WedgeXY(re_core::DiagXY::NE));

        resize_level(&mut state, 8, 8, 4);

        assert_eq!(state.map.size(), (8, 8, 4), "size updated");
        assert_eq!(state.level.size, (8, 8, 4), "level size updated");
        // Overlap preserved.
        assert_eq!(state.map.cell_at(3, 3, 1), CellKind::WallSolid);
        assert_eq!(state.map.cell_at(2, 1, 0), CellKind::Object);
        assert_eq!(state.map.side_data_at(2, 1, 0), 7, "side_data preserved");
        assert_eq!(
            state.map.shape_at(1, 1, 0),
            re_core::Shape::WedgeXY(re_core::DiagXY::NE),
            "shape preserved"
        );
        // A brand-new cell is Empty.
        assert_eq!(state.map.cell_at(7, 7, 3), CellKind::Empty);
        // Save re-captures the resized map.
        assert!(state.level.cells.is_none(), "stale cells grid dropped");
    }

    /// (b) SHRINK: cells inside the new bounds are preserved; an entity that was
    /// outside is clamped and leaves no out-of-bounds index.
    #[test]
    fn shrink_preserves_inner_cells_and_clamps_entities() {
        let mut state = EditorState::new_empty(10, 10, 6);
        state.map.set(1, 1, 0, CellKind::WallSolid); // stays in bounds
        state.map.set(9, 9, 5, CellKind::WallSolid); // dropped on shrink

        // An entity well outside the shrunk bounds.
        let ni = state.level.push_node(8, 9, 5);
        state.level.door.x = 9;
        state.level.door.y = 8;
        state.level.door.z = 5;
        state.level.player.x = 9;
        state.level.player.y = 9;
        state.level.player.z = 4;

        resize_level(&mut state, 4, 4, 2);

        assert_eq!(state.map.size(), (4, 4, 2));
        // Inner cell preserved; the far cell is simply gone (outside new bounds).
        assert_eq!(state.map.cell_at(1, 1, 0), CellKind::WallSolid);
        // Node clamped into range.
        let n = &state.level.nodes[ni];
        assert!(
            n.x < 4 && n.y < 4 && n.z < 2,
            "node clamped into bounds: {:?}",
            (n.x, n.y, n.z)
        );
        // Door + spawn clamped into range.
        assert!(state.level.door.x < 4 && state.level.door.y < 4 && state.level.door.z < 2);
        assert!(state.level.player.x < 4 && state.level.player.y < 4 && state.level.player.z < 2);
    }

    /// (c) View state: cursor / pan / z / cam are clamped into the new bounds.
    #[test]
    fn shrink_clamps_view_state() {
        let mut state = EditorState::new_empty(12, 12, 8);
        state.cursor = (11, 10);
        state.pan = (9, 8);
        state.z = 7;
        state.cam.pos = glam::Vec3::new(11.5, 10.5, 7.5);

        resize_level(&mut state, 4, 5, 3);

        assert!(state.cursor.0 < 4 && state.cursor.1 < 5, "cursor clamped");
        assert!(state.pan.0 < 4 && state.pan.1 < 5, "pan clamped");
        assert!(state.z < 3, "z clamped");
        assert!(
            state.cam.pos.x <= 4.0 && state.cam.pos.y <= 5.0 && state.cam.pos.z <= 3.0,
            "cam clamped: {:?}",
            state.cam.pos
        );
    }

    /// A pipe whose expanded path now exits the shrunken grid has its path cleared
    /// (so it can be re-drawn) but keeps its endpoints — no out-of-bounds path cell
    /// survives, and a fully in-bounds pipe is untouched.
    #[test]
    fn shrink_clears_out_of_bounds_pipe_path_but_keeps_endpoints() {
        let mut state = EditorState::new_empty(10, 10, 4);
        let a = state.level.push_node(1, 1, 0);
        let b = state.level.push_node(8, 8, 0);
        let a_name = state.level.nodes[a].name.clone();
        let b_name = state.level.nodes[b].name.clone();

        // An out-of-bounds-on-shrink pipe path.
        let mut p = re_level::Pipe {
            from: a_name.clone(),
            to: b_name.clone(),
            from_node: a,
            to_node: b,
            ..Default::default()
        };
        p.path = vec![(1, 1, 0), (5, 5, 0), (8, 8, 0)];
        p.switch_cell = Some(1);
        state.level.pipes.push(p);

        // A fully in-bounds pipe.
        let mut p2 = re_level::Pipe {
            from: a_name.clone(),
            to: a_name.clone(),
            from_node: a,
            to_node: a,
            ..Default::default()
        };
        p2.path = vec![(1, 1, 0), (2, 1, 0)];
        state.level.pipes.push(p2);

        resize_level(&mut state, 4, 4, 2);

        // First pipe: path cleared, endpoints retained.
        assert!(
            state.level.pipes[0].path.is_empty(),
            "out-of-bounds path cleared"
        );
        assert!(
            state.level.pipes[0].switch_cell.is_none(),
            "switch cleared with path"
        );
        assert_eq!(state.level.pipes[0].from, a_name, "endpoint name kept");
        assert_eq!(state.level.pipes[0].to, b_name, "endpoint name kept");
        // Second pipe: untouched (all cells in bounds).
        assert_eq!(state.level.pipes[1].path.len(), 2, "in-bounds path kept");
    }
}
