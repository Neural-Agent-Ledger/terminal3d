//! Staircase STAMP TOOL — one action lays a whole climbable staircase.
//!
//! This is the editor-side thin wrapper around the engine-pure prefab generator
//! [`re_level::staircase`] (the single source of truth for "what a walkable
//! staircase looks like"). The editor NEVER re-derives the geometry — it asks
//! the generator for the cell
//! stamps and writes them through the SAME `Map` cell+shape path every other
//! brush uses, so a stamped staircase round-trips to `[cells]` losslessly and is
//! undone as one step.
//!
//! Two sub-modes, selected by [`StairSub`]:
//! * **Straight** — anchored at the cursor cell, oriented by [`EditorState::stair_dir`]
//!   ([`RampDir`]), climbing exactly `level.storey_height()` cells
//!   ([`re_level::staircase::straight_staircase`]).
//! * **Spiral** — centred on the cursor, with editable `radius` + `turns`
//!   ([`re_level::staircase::spiral_staircase`]).
//!
//! Pure: no terminal, no I/O. The single mutation choke point
//! (`edit::apply_tool_at_cursor`) snapshots UNDO before calling [`stamp_at_cursor`],
//! so the whole staircase is one undoable, dirty-marking edit.
use crate::state::EditorState;
use re_core::{RampDir, Shape};
use re_level::staircase::{apply_stamps, spiral_staircase, straight_staircase, CellStamp};
use re_world::{CellKind, Map};

/// The staircase tool's two sub-modes. `Tab`-free toggle via `\` so it stays out
/// of the view-cycle key; `[`/`]` then re-purpose to direction (straight) and
/// `+`/`-` to the spiral params, mirroring the existing shape-brush convention.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StairSub {
    /// A straight 45° run climbing one storey, oriented by [`EditorState::stair_dir`].
    Straight,
    /// A spiral run wrapping a central column, sized by `radius` + `turns`.
    Spiral,
}

impl StairSub {
    /// Toggle Straight ↔ Spiral.
    pub fn toggled(self) -> StairSub {
        match self {
            StairSub::Straight => StairSub::Spiral,
            StairSub::Spiral => StairSub::Straight,
        }
    }
}

/// Cycle a [`RampDir`] (the straight staircase's rise direction) by `delta` (±1)
/// through the four cardinals, wrapping. Order: +X → +Y → -X → -Y (the same CW
/// walk `re_level::staircase::rotate_ramp_dir_cw` uses).
pub fn cycle_dir(dir: RampDir, delta: i32) -> RampDir {
    const DIRS: [RampDir; 4] = [RampDir::PosX, RampDir::PosY, RampDir::NegX, RampDir::NegY];
    let i = DIRS.iter().position(|&d| d == dir).unwrap_or(0) as i32;
    DIRS[(i + delta).rem_euclid(DIRS.len() as i32) as usize]
}

/// A short readable label for a [`RampDir`] (the staircase's rise direction).
pub fn dir_label(dir: RampDir) -> &'static str {
    match dir {
        RampDir::PosX => "+X",
        RampDir::NegX => "-X",
        RampDir::PosY => "+Y",
        RampDir::NegY => "-Y",
    }
}

/// A one-line summary of the current staircase tool settings for the panel,
/// including whether the stairwell auto-carve (`#`) is on.
pub fn summary(state: &EditorState) -> String {
    let carve = if state.auto_carve_stairwell {
        "carve ON"
    } else {
        "carve OFF"
    };
    match state.stair_sub {
        StairSub::Straight => format!(
            "straight {} H{} [{carve}]",
            dir_label(state.stair_dir),
            state.level.storey_height()
        ),
        StairSub::Spiral => format!(
            "spiral r{} t{} [{carve}]",
            state.stair_radius, state.stair_turns
        ),
    }
}

/// Stamp the active staircase at the cursor cell, writing the generator's cells
/// through `Map::set` / `Map::set_shape` (out-of-bounds cells are clipped by the
/// map, never escaping it). Marks dirty + sets a status line. Caller snapshots
/// UNDO first (see `edit::apply_tool_at_cursor`).
///
/// * Straight: `base = (cursor.x, cursor.y, z)`, oriented by `stair_dir`, climbing
///   `level.storey_height()`. The generator places tread 1 one step in + one up
///   from `base`, so the staircase grows out of the cursor cell's floor.
/// * Spiral: `center = (cursor.x, cursor.y, z)`, with `stair_radius` + `stair_turns`.
pub fn stamp_at_cursor(state: &mut EditorState) {
    let (x, y, z) = (state.cursor.0 as i32, state.cursor.1 as i32, state.z as i32);
    let base = glam::IVec3::new(x, y, z);
    let h = state.level.storey_height();
    match state.stair_sub {
        StairSub::Straight => {
            let stamps = straight_staircase(base, state.stair_dir, h);
            apply_stamps(&mut state.map, &stamps);
            if state.auto_carve_stairwell {
                carve_stairwell(&mut state.map, &stamps, h);
            }
            state.status = format!(
                "stamped straight stair {} H{} @ {},{},{} ({} cells)",
                dir_label(state.stair_dir),
                h,
                x,
                y,
                z,
                stamps.len()
            );
        }
        StairSub::Spiral => {
            let stamps = spiral_staircase(base, state.stair_radius, state.stair_turns, h);
            apply_stamps(&mut state.map, &stamps);
            if state.auto_carve_stairwell {
                carve_stairwell(&mut state.map, &stamps, h);
            }
            state.status = format!(
                "stamped spiral stair r{} t{} @ {},{},{} ({} cells)",
                state.stair_radius,
                state.stair_turns,
                x,
                y,
                z,
                stamps.len()
            );
        }
    }
    state.mark_dirty();
}

/// Punch a **stairwell hole** through the floor slab(s) above a stamped staircase
/// so the player can climb through into the storey above without manually erasing
/// the ceiling.
///
/// # The carve rule (which z's, which XY)
///
/// The run's cells are the [`Shape::RampZ`] treads (the climb) and the flat
/// [`Shape::Full`] landings (the upper-floor destination). For each **storey
/// floor plane** the run crosses — `base.z + k * storey_height` for `k = 1, 2, …`
/// (`base.z` recovered as the lowest ramp tread's z minus one, since tread 1 sits
/// one cell above the base) — we carve, **at that plane z**, the `(x, y)` of every
/// run cell whose OWN z is **strictly below** the plane. Those are exactly the
/// cells the climber passes *up through* the slab: the hole lets them emerge.
///
/// Run cells that already sit AT the plane (the top tread and the landing, which
/// ARE the upper-floor structure) are left solid — matching the manual
/// `levels/20-stairs-straight.lvl` pattern, where the stairwell shaft at z=4 is
/// open over x=4,5,6 (the treads climbing up from below) but the top tread (x=7)
/// and landing (x=8) at z=4 stay solid.
///
/// A run climbing several storeys (a multi-turn spiral) gets a hole punched at
/// EACH crossed storey floor — never a single slanted column following the treads.
/// Each carved cell is set to [`CellKind::Empty`] + [`Shape::Full`] (clean air).
/// Out-of-bounds writes are clipped by [`Map::set`].
///
/// `storey_height == 0` or a stamp list with no ramp treads carves nothing
/// (defined, never panics).
pub fn carve_stairwell(map: &mut Map, stamps: &[CellStamp], storey_height: u32) {
    if storey_height == 0 {
        return;
    }
    let h = storey_height as i32;

    // Collect the run cells (treads + landings) and the vertical span. A stamp
    // with no ramp treads is a degenerate run we leave untouched.
    let mut cells: Vec<(i32, i32, i32)> = Vec::new(); // (x, y, z)
    let mut min_ramp_z = i32::MAX;
    let mut max_z = i32::MIN;
    for s in stamps {
        match s.shape {
            Shape::RampZ(_) => {
                min_ramp_z = min_ramp_z.min(s.pos.z);
                max_z = max_z.max(s.pos.z);
                cells.push((s.pos.x, s.pos.y, s.pos.z));
            }
            Shape::Full => {
                // Landings: their XY is part of the footprint at lower planes, but
                // their own cell (at the upper floor) is never carved.
                max_z = max_z.max(s.pos.z);
                cells.push((s.pos.x, s.pos.y, s.pos.z));
            }
            Shape::WedgeXY(_) => {}
        }
    }
    if min_ramp_z == i32::MAX {
        return; // no ramp treads → nothing to climb through
    }

    // Recover the base floor z: tread 1 is one cell above the base.
    let base_z = min_ramp_z - 1;

    // Carve each storey floor plane the run crosses. At a plane we punch the XY of
    // every run cell strictly BELOW that plane (the cells climbing up through it),
    // de-duplicated. Cells at/above the plane are the upper-floor structure.
    let mut k = 1;
    loop {
        let slab_z = base_z + k * h;
        if slab_z > max_z {
            break;
        }
        if slab_z >= 0 {
            let mut carved: Vec<(i32, i32)> = Vec::new();
            for &(x, y, z) in &cells {
                if z < slab_z && x >= 0 && y >= 0 && !carved.contains(&(x, y)) {
                    carved.push((x, y));
                    map.set(x as usize, y as usize, slab_z as usize, CellKind::Empty);
                    map.set_shape(x as usize, y as usize, slab_z as usize, Shape::Full);
                }
            }
        }
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::Shape;
    use re_world::CellKind;

    #[test]
    fn cycle_dir_wraps_through_four() {
        let mut d = RampDir::PosX;
        let order = [RampDir::PosX, RampDir::PosY, RampDir::NegX, RampDir::NegY];
        for expected in order.iter().skip(1) {
            d = cycle_dir(d, 1);
            assert_eq!(d, *expected);
        }
        // One more wraps back to start; backward from start wraps to the last.
        d = cycle_dir(d, 1);
        assert_eq!(d, RampDir::PosX);
        assert_eq!(cycle_dir(RampDir::PosX, -1), RampDir::NegY);
    }

    #[test]
    fn stair_sub_toggles() {
        assert_eq!(StairSub::Straight.toggled(), StairSub::Spiral);
        assert_eq!(StairSub::Spiral.toggled(), StairSub::Straight);
    }

    /// A straight stamp lays exactly the generator's RampZ treads at the cursor
    /// (matching `re_level`'s proven climbable layout) plus the flat landing, and
    /// respects `level.storey_height()`.
    #[test]
    fn straight_stamp_matches_generator_and_respects_storey_height() {
        let mut s = EditorState::new_empty(12, 6, 12);
        // storey_height() defaults to 4 on a fresh level (raw 0 → DEFAULT).
        let h = s.level.storey_height();
        assert_eq!(h, 4, "fresh editor level reports the default storey height");
        s.stair_sub = StairSub::Straight;
        s.stair_dir = RampDir::PosX;
        s.cursor = (0, 2);
        s.z = 0;

        stamp_at_cursor(&mut s);

        // Treads at (n,2,n) for n=1..=H, WallSolid + RampZ(PosX) — the same layout
        // re_level::straight_staircase proves climbable from base=(0,2,0).
        for n in 1..=h as i32 {
            assert_eq!(
                s.map.cell_at(n, 2, n),
                CellKind::WallSolid,
                "tread {n} solid"
            );
            assert_eq!(
                s.map.shape_at(n, 2, n),
                Shape::RampZ(RampDir::PosX),
                "tread {n} is a +X ramp"
            );
        }
        // Landing one step past the top tread, flat solid on the upper floor.
        let hl = h as i32;
        assert_eq!(s.map.cell_at(hl + 1, 2, hl), CellKind::WallSolid);
        assert_eq!(
            s.map.shape_at(hl + 1, 2, hl),
            Shape::Full,
            "landing is flat"
        );
        assert!(s.dirty, "stamping marks the document dirty");
    }

    /// The pure carve helper, given a straight one-storey stamp, punches the
    /// stairwell shaft at the slab plane (`base.z + storey_height`) over the XY of
    /// the treads that climb UP THROUGH the plane, while leaving the top tread /
    /// landing AT the plane solid and the treads' own cells as RampZ. This mirrors
    /// the manual `levels/20-stairs-straight.lvl` shaft (open over x=4,5,6 at z=4,
    /// solid at x=7 (top tread) and x=8 (landing)).
    #[test]
    fn carve_opens_shaft_below_slab_plane() {
        const H: u32 = 4;
        // Map tall enough to hold the slab plane (base.z=0 → slab z=4).
        let mut map = Map::new(12, 6, 12);
        // Fill the whole slab plane row solid so we can prove what is carved.
        for x in 0..12 {
            map.set(x, 2, H as usize, CellKind::WallSolid);
        }
        let stamps = straight_staircase(glam::IVec3::new(0, 2, 0), RampDir::PosX, H);
        apply_stamps(&mut map, &stamps);
        carve_stairwell(&mut map, &stamps, H);

        let slab = H as i32;
        // Treads at z=1,2,3 (x=1,2,3) are below the slab → carved open at z=4.
        for n in 1..slab {
            assert_eq!(
                map.cell_at(n, 2, slab),
                CellKind::Empty,
                "shaft open over climbing tread x={n} at slab z={slab}"
            );
            assert_eq!(
                map.shape_at(n, 2, slab),
                Shape::Full,
                "carved cell is Full air"
            );
        }
        // Top tread (x=H, z=4) and landing (x=H+1, z=4) are AT the plane → stay solid.
        assert_eq!(
            map.cell_at(slab, 2, slab),
            CellKind::WallSolid,
            "top tread at the plane stays solid"
        );
        assert_eq!(
            map.cell_at(slab + 1, 2, slab),
            CellKind::WallSolid,
            "landing at the plane stays solid"
        );
        // Treads themselves remain RampZ.
        for n in 1..=H as i32 {
            assert_eq!(
                map.shape_at(n, 2, n),
                Shape::RampZ(RampDir::PosX),
                "tread {n} stays a ramp"
            );
        }
    }

    /// With auto-carve ON (the default), `stamp_at_cursor` opens the slab; with it
    /// OFF the slab cells stay solid. Treads remain RampZ in both cases.
    #[test]
    fn stamp_carves_only_when_toggle_on() {
        const H: u32 = 4;
        let slab = H as i32;
        let build = |carve: bool| -> EditorState {
            let mut s = EditorState::new_empty(12, 6, 12);
            // A solid ceiling slab over the run's row at the next storey floor.
            for x in 0..12 {
                s.map.set(x, 2, H as usize, CellKind::WallSolid);
            }
            s.stair_sub = StairSub::Straight;
            s.stair_dir = RampDir::PosX;
            s.cursor = (0, 2);
            s.z = 0;
            s.auto_carve_stairwell = carve;
            stamp_at_cursor(&mut s);
            s
        };

        // ON → the shaft over the climbing treads (x=1..slab-1) is carved to air.
        let on = build(true);
        for n in 1..slab {
            assert_eq!(
                on.map.cell_at(n, 2, slab),
                CellKind::Empty,
                "toggle ON carves shaft over tread x={n}"
            );
        }
        // Treads still ramps.
        for n in 1..=slab {
            assert_eq!(on.map.shape_at(n, 2, n), Shape::RampZ(RampDir::PosX));
        }

        // OFF → the slab stays solid over those same cells.
        let off = build(false);
        for n in 1..slab {
            assert_eq!(
                off.map.cell_at(n, 2, slab),
                CellKind::WallSolid,
                "toggle OFF leaves the slab solid at x={n}"
            );
        }
    }

    /// A multi-storey climb (spiral with several turns) carves a hole at EACH
    /// crossed storey-floor plane, not a slanted column.
    #[test]
    fn carve_multistorey_opens_each_storey_plane() {
        const H: u32 = 4;
        // A radius-1 / turns-2 spiral climbs 8*radius*turns = 16 cells, crossing
        // storey planes at z=4, 8, 12, 16.
        let mut map = Map::new(8, 8, 24);
        let stamps = spiral_staircase(glam::IVec3::new(3, 3, 0), 1, 2, H);
        apply_stamps(&mut map, &stamps);
        carve_stairwell(&mut map, &stamps, H);

        // Footprint = union of all tread/landing XY. At least one footprint cell
        // must be air at each storey plane within reach.
        let footprint: std::collections::HashSet<(i32, i32)> = stamps
            .iter()
            .filter(|s| matches!(s.shape, Shape::RampZ(_) | Shape::Full))
            .map(|s| (s.pos.x, s.pos.y))
            .collect();
        for &slab in &[4i32, 8, 12, 16] {
            let any_air = footprint.iter().any(|&(x, y)| {
                map.cell_at(x, y, slab) == CellKind::Empty
                    && map.shape_at(x, y, slab) == Shape::Full
            });
            assert!(
                any_air,
                "a footprint cell must be carved air at storey plane z={slab}"
            );
        }
    }

    /// A spiral stamp lays the generator's ramp ring (RampZ treads) + the solid
    /// central column at the cursor, sized by radius/turns.
    #[test]
    fn spiral_stamp_lays_ramp_ring_and_column() {
        let mut s = EditorState::new_empty(16, 16, 24);
        s.stair_sub = StairSub::Spiral;
        s.stair_radius = 1;
        s.stair_turns = 1;
        s.cursor = (6, 6);
        s.z = 0;

        stamp_at_cursor(&mut s);

        // Central column solid at the cursor for the base + above.
        assert_eq!(
            s.map.cell_at(6, 6, 0),
            CellKind::WallSolid,
            "column base solid"
        );
        // At least one RampZ tread was written somewhere around the ring.
        let mut found_ramp = false;
        for z in 0..24i32 {
            for y in 0..16i32 {
                for x in 0..16i32 {
                    if matches!(s.map.shape_at(x, y, z), Shape::RampZ(_)) {
                        found_ramp = true;
                    }
                }
            }
        }
        assert!(found_ramp, "spiral lays at least one ramp tread");
        assert!(s.dirty);
    }
}
