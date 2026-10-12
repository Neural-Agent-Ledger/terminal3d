//! Staircase **prefab generator** — the single source of truth for "what a
//! walkable staircase looks like".
//!
//! This module is a *pure* function library: it emits a `Vec<CellStamp>`
//! describing the cells of a climbable staircase. It performs **no** terminal
//! I/O, no rendering, and does not mutate any global state — every input is a
//! plain value and the output is owned data the caller (the editor stamp tool
//! or a demo-level builder) applies to a [`re_world::Map`].
//!
//! # Why these prefabs are authoritative
//!
//! `re_world`'s staircase unit tests
//! (`staircase_ascending_raises_z_and_stays_grounded`, …) prove a *specific*
//! cell layout is climbable: a run of [`Shape::RampZ`] cells where ramp cell
//! `n` sits at `(base.x + n, base.y, base.z + n)` (for `RampDir::PosX`), each
//! cell being a solid [`CellKind::WallSolid`] with the same `RampDir`. The
//! 45° ramp rises exactly **1 cell of height per 1 cell of horizontal travel**
//! ([`re_core::shape_solid_contains`] / `shape_surface_z`), so climbing one
//! storey of height `H` needs exactly `H` consecutive ramp cells plus a landing
//! on the upper floor.
//!
//! [`straight_staircase`] reproduces that layout **exactly** (generalised to all
//! four [`RampDir`] orientations), so anything stamped from this generator is
//! provably walkable by the same physics the `re_world` tests exercise.
//!
//! # Coordinate model
//!
//! All positions are integer cell coordinates ([`glam::IVec3`]), z-up. A ramp
//! cell's *low* edge touches the floor of its own cell and its *high* edge is a
//! full-height wall; stacking the next ramp one cell up and one cell along makes
//! the slope continuous across the seam (see the `re_world` seam test).

use glam::IVec3;
use re_core::{RampDir, Shape};
use re_world::CellKind;

/// One cell the prefab wants stamped into a [`re_world::Map`]: its position,
/// its surface [`CellKind`], and its geometry [`Shape`].
///
/// A `CellStamp` is a pure value — applying it (via [`apply_stamps`] or by hand)
/// is the only place the world is mutated. The generator never touches a `Map`
/// itself, which keeps it trivially testable and engine-pure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellStamp {
    /// Integer cell coordinate (z-up).
    pub pos: IVec3,
    /// Surface kind to stamp (ramp cells + landings are [`CellKind::WallSolid`];
    /// the central column of a spiral is also [`CellKind::WallSolid`]).
    pub kind: CellKind,
    /// Per-cell geometry. Ramp treads are [`Shape::RampZ`]; landings and the
    /// spiral column are [`Shape::Full`].
    pub shape: Shape,
}

impl CellStamp {
    /// A solid ramp tread with the given rise direction.
    #[inline]
    fn ramp(pos: IVec3, dir: RampDir) -> Self {
        CellStamp {
            pos,
            kind: CellKind::WallSolid,
            shape: Shape::RampZ(dir),
        }
    }

    /// A flat, full solid cell (landing tread or spiral column).
    #[inline]
    fn solid(pos: IVec3) -> Self {
        CellStamp {
            pos,
            kind: CellKind::WallSolid,
            shape: Shape::Full,
        }
    }
}

/// The unit horizontal step a [`RampDir`] climbs along: the cell-to-cell
/// advance that, paired with `+1` in z, keeps the 45° slope continuous.
///
/// `PosX → +x`, `NegX → −x`, `PosY → +y`, `NegY → −y`. This matches the
/// `re_world` proof: a `RampDir::PosX` run advances `(+1, 0, +1)` per tread.
#[inline]
pub fn ramp_step(dir: RampDir) -> IVec3 {
    match dir {
        RampDir::PosX => IVec3::new(1, 0, 0),
        RampDir::NegX => IVec3::new(-1, 0, 0),
        RampDir::PosY => IVec3::new(0, 1, 0),
        RampDir::NegY => IVec3::new(0, -1, 0),
    }
}

/// Rotate a [`RampDir`] 90° counter-clockwise (viewed top-down, +x east,
/// +y south on screen). Used by [`spiral_staircase`] to turn the rise axis at
/// each corner so the four runs wrap a central column.
///
/// Cycle: `PosX → PosY → NegX → NegY → PosX`. This walks the four sides of a
/// square footprint in order, each side climbing toward the next corner.
#[inline]
pub fn rotate_ramp_dir_cw(dir: RampDir) -> RampDir {
    match dir {
        RampDir::PosX => RampDir::PosY,
        RampDir::PosY => RampDir::NegX,
        RampDir::NegX => RampDir::NegY,
        RampDir::NegY => RampDir::PosX,
    }
}

/// Generate a **straight 45° staircase** rising exactly one storey.
///
/// Emits `storey_height` consecutive [`Shape::RampZ`]`(dir)` treads followed by
/// one flat landing on the upper floor. Tread `n` (1-based) is placed at
/// `base + n * (ramp_step(dir) + z_up)`, i.e. each tread advances one cell in
/// the rise direction **and** one cell up in z — the exact layout proven
/// climbable by `re_world`'s `staircase_ascending_raises_z_and_stays_grounded`
/// (which uses `base = (0, 2, 0)`, `dir = PosX`, so treads land at
/// `(1,2,1)..(H,2,H)`).
///
/// The landing is a flat [`Shape::Full`] cell one step past the top tread, on
/// the upper floor plane (`base.z + storey_height`), so the climber arrives on
/// solid ground.
///
/// `base` is the cell **below and behind** the first tread: tread 1 is at
/// `base + ramp_step(dir) + (0,0,1)`. Pick `base` as the floor cell the
/// staircase grows out of.
///
/// A `storey_height` of `0` yields just the landing at `base` height + 0 (an
/// empty climb — defined, never panics). Returns `storey_height + 1` stamps.
///
/// # Layout match
///
/// With `base = IVec3::new(0, 2, 0)`, `dir = RampDir::PosX`,
/// `storey_height = H`, the ramp treads are exactly
/// `(1,2,1), (2,2,2), …, (H,2,H)` — byte-identical to `re_world`'s
/// `staircase_map(H)`.
pub fn straight_staircase(base: IVec3, dir: RampDir, storey_height: u32) -> Vec<CellStamp> {
    let step = ramp_step(dir);
    let z_up = IVec3::new(0, 0, 1);
    let mut stamps = Vec::with_capacity(storey_height as usize + 1);

    // Treads: tread n is at base + n*(step + z_up), for n = 1..=H.
    for n in 1..=storey_height as i32 {
        let pos = base + (step + z_up) * n;
        stamps.push(CellStamp::ramp(pos, dir));
    }

    // Landing: flat solid one step past the top tread, on the upper floor
    // plane. Top tread (n=H) is at base + H*(step + z_up); the landing sits
    // one more step along at the same z (upper floor).
    let h = storey_height as i32;
    let landing = base + (step + z_up) * h + step;
    stamps.push(CellStamp::solid(landing));

    stamps
}

/// Generate a **spiral ("circular") staircase** rising up around a solid
/// central column, on a square footprint.
///
/// The footprint is a `(2*radius+1) × (2*radius+1)` square centred on `center`
/// (the column's base cell). The interior column (every cell within
/// Chebyshev distance `radius - 1` of the centre column line, i.e. a
/// `(2*radius-1)` square core) is stamped solid for the full climbed height;
/// for `radius == 1` that core is the single centre cell.
///
/// Around the column the generator lays four ramp **runs** per loop — one along
/// each side of the square — with the [`RampDir`] rotating 90°
/// ([`rotate_ramp_dir_cw`]) at each corner. Each ramp tread rises 1 cell, so a
/// run of `L = 2*radius` treads per side climbs `L` cells, and one full loop
/// (4 sides) climbs `8*radius` cells. Corner cells are flat landings.
///
/// # Rise per turn
///
/// **One full turn (4 sides) rises `4 * side_len = 4 * (2*radius) = 8*radius`
/// cells** (8 cells per turn at `radius = 1`). The staircase emits `turns`
/// full loops, so total rise = `turns * 8 * radius` cells = the number of ramp
/// treads emitted.
///
/// `radius < 1` or `turns == 0` yields just the (possibly empty) column and no
/// treads — defined, never panics.
///
/// The returned `Vec` lists the central column cells first, then the ramp/landing
/// path in ascending order, so a caller stamping them in order builds the column
/// then the connected rising ring.
pub fn spiral_staircase(
    center: IVec3,
    radius: i32,
    turns: u32,
    storey_height: u32,
) -> Vec<CellStamp> {
    // storey_height is accepted for API symmetry / validation hints; the spiral
    // rises by its tread count (8*radius per turn). It is not used to truncate
    // the geometry — the caller chooses `turns` for the desired height. We keep
    // the parameter so the editor can validate "turns * 8 * radius vs the map's
    // storey grid". Mark it read to avoid an unused warning while documenting it.
    let _ = storey_height;

    let mut stamps = Vec::new();
    if radius < 1 || turns == 0 {
        // Degenerate: emit just the column (a single cell at radius 1) at base
        // height, nothing to climb. Defined + non-panicking.
        if radius >= 1 {
            stamps.push(CellStamp::solid(center));
        }
        return stamps;
    }

    let side_len = 2 * radius; // treads per side run
    let total_treads = side_len * 4 * turns as i32;
    let top_z = center.z + total_treads; // column must reach the top landing

    // --- Central solid column ---
    // Core square of half-extent (radius-1) around the centre column line,
    // i.e. a (2*radius-1)² core (the single centre cell when radius==1),
    // raised the full climbed height + 1 (so the top loop has a column to hug).
    let core = radius - 1;
    for z in center.z..=top_z {
        for dy in -core..=core {
            for dx in -core..=core {
                stamps.push(CellStamp::solid(IVec3::new(
                    center.x + dx,
                    center.y + dy,
                    z,
                )));
            }
        }
    }

    // --- Ramp ring ---
    // Walk the perimeter of the (2*radius+1) square, side by side, rotating the
    // rise direction at each corner. Each side is `side_len` treads. Start at
    // the south-west-ish corner and go around; the exact start cell only needs
    // to (a) sit on the perimeter ring and (b) advance by ramp_step(dir) so the
    // run is climbable by the SAME per-side check as the straight prefab.
    //
    // We build each side as a straight-run of ramp treads. The first cell of
    // each side is the corner *landing* (flat) shared with the previous side's
    // end; the remaining `side_len - 1` cells plus the next corner are treads.
    //
    // To keep every side independently climbable (matching the straight prefab),
    // we lay each side as `side_len` treads of a single RampDir, then a flat
    // corner landing, then rotate. The perimeter ring radius is `radius` from
    // the centre line.
    let r = radius;
    // Perimeter corners (clockwise on screen, +x east / +y south), starting
    // from the NW-ish corner. Each side run advances along one edge.
    // Side directions cycle via rotate_ramp_dir_cw starting from PosX.
    let mut dir = RampDir::PosX;
    let mut z = center.z; // current floor z at the start of the current tread
                          // Start cell: the west edge mid is messy; instead anchor each side to a
                          // corner derived from the ring and the direction. We track an explicit
                          // cursor cell that walks the ring.
                          //
                          // Cursor starts at the corner that the first (PosX) run climbs FROM:
                          // the run advances +x, so it starts at the −x,−y corner of the ring.
    let mut cursor = IVec3::new(center.x - r, center.y - r, z);

    for _turn in 0..turns as i32 {
        for _side in 0..4 {
            let step = ramp_step(dir);
            // Lay `side_len` ramp treads along this edge, each rising 1.
            for _ in 0..side_len {
                cursor.z = z;
                stamps.push(CellStamp::ramp(cursor, dir));
                cursor += step;
                z += 1;
            }
            // Corner landing: flat tread at the new (turned) corner, same z.
            cursor.z = z;
            stamps.push(CellStamp::solid(cursor));
            // Rotate the rise direction 90° for the next side. The cursor is now
            // at the corner; the next side advances from here in the new dir.
            dir = rotate_ramp_dir_cw(dir);
        }
    }

    stamps
}

/// Apply a slice of [`CellStamp`]s to a [`re_world::Map`], in order. Each stamp
/// writes its cell kind then its shape; out-of-bounds positions are ignored by
/// [`re_world::Map::set`] / [`re_world::Map::set_shape`] (the prefab clips to the
/// map, never escaping it). This is the optional convenience path — callers that
/// want finer control (e.g. the editor's undo journal) can iterate the stamps
/// themselves.
pub fn apply_stamps(map: &mut re_world::Map, stamps: &[CellStamp]) {
    for s in stamps {
        if s.pos.x < 0 || s.pos.y < 0 || s.pos.z < 0 {
            continue;
        }
        map.set(s.pos.x as usize, s.pos.y as usize, s.pos.z as usize, s.kind);
        map.set_shape(
            s.pos.x as usize,
            s.pos.y as usize,
            s.pos.z as usize,
            s.shape,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use re_world::{
        stamp_geometry, CameraPose, GeometrySpec, Map, PlayerMotion, PLAYER_EYE_HEIGHT,
    };

    /// Build an empty room big enough to hold a staircase, with perimeter walls,
    /// floor, and ceiling, mirroring `re_world`'s `staircase_map` shell so the
    /// climb test runs in the same conditions.
    fn empty_room(size: (usize, usize, usize)) -> Map {
        let spec = GeometrySpec {
            size,
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        };
        stamp_geometry(&spec)
    }

    // ---- straight_staircase: layout matches the re_world proof ----

    #[test]
    fn straight_layout_matches_re_world_proof() {
        // re_world::staircase_map(H) places ramp cells at (n,2,n) for n=1..=H,
        // each WallSolid + RampZ(PosX). With base=(0,2,0), dir=PosX we must
        // reproduce exactly that (plus a landing).
        const H: u32 = 4;
        let stamps = straight_staircase(IVec3::new(0, 2, 0), RampDir::PosX, H);

        // H treads + 1 landing.
        assert_eq!(stamps.len(), H as usize + 1);

        for n in 1..=H as i32 {
            let s = stamps[(n - 1) as usize];
            assert_eq!(
                s.pos,
                IVec3::new(n, 2, n),
                "tread {n} must be at (n,2,n) like re_world::staircase_map"
            );
            assert_eq!(s.kind, CellKind::WallSolid);
            assert_eq!(s.shape, Shape::RampZ(RampDir::PosX));
        }

        // Landing: one step past top tread (H,2,H) → (H+1, 2, H), flat solid.
        let landing = *stamps.last().unwrap();
        assert_eq!(landing.pos, IVec3::new(H as i32 + 1, 2, H as i32));
        assert_eq!(landing.shape, Shape::Full);
        assert_eq!(landing.kind, CellKind::WallSolid);
    }

    /// Drive `re_world::step_gravity` up a stamped straight staircase: the eye
    /// z must rise by ~storey_height and `grounded` stay true throughout — the
    /// same pattern as `re_world::staircase_ascending_raises_z_and_stays_grounded`,
    /// but on cells emitted by THIS generator (the prefab is provably climbable).
    #[test]
    fn straight_staircase_is_climbable() {
        const H: u32 = 4;
        let mut map = empty_room((H as usize + 3, 5, H as usize + 3));
        let stamps = straight_staircase(IVec3::new(0, 2, 0), RampDir::PosX, H);
        apply_stamps(&mut map, &stamps);

        // Confirm the stamped cells equal the proven climbable layout.
        for n in 1..=H as i32 {
            assert_eq!(map.cell_at(n, 2, n), CellKind::WallSolid);
            assert_eq!(map.shape_at(n, 2, n), Shape::RampZ(RampDir::PosX));
        }

        // Start near the base of the first ramp (x=1.1), like the re_world test.
        let start_x: f32 = 1.1;
        let start_surface_z: f32 = 1.0 + (start_x - 1.0);
        let start_eye_z = start_surface_z + PLAYER_EYE_HEIGHT;
        let mut cam = CameraPose {
            pos: Vec3::new(start_x, 2.5, start_eye_z),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: false,
            ..Default::default()
        };
        for _ in 0..10 {
            cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
        }
        assert!(motion.grounded, "must start grounded on ramp 1");

        let x_end: f32 = (H as f32) + 0.9;
        let mut x = start_x;
        let mut seam_failures: Vec<f32> = Vec::new();
        while x < x_end {
            x = (x + 0.05).min(x_end);
            cam.pos.x = x;
            for _ in 0..3 {
                cam.step_gravity(&mut motion, 1.0 / 60.0, &map);
            }
            let local_x = x - x.floor();
            let near_seam = !(0.1..=0.9).contains(&local_x);
            if near_seam && !motion.grounded {
                seam_failures.push(x);
            }
        }
        assert!(
            seam_failures.is_empty(),
            "grounded flickered at seams: {seam_failures:?}"
        );
        assert!(motion.grounded, "must still be grounded at top");

        let rise = cam.pos.z - start_eye_z;
        // From x=1.1 to x=4.9 the surface rises ~ (H - 0.2) cells.
        let expected = (H as f32) - 0.2;
        assert!(
            (rise - expected).abs() < 0.25,
            "total rise={rise:.4} should be ≈{expected:.4} (H={H})"
        );
    }

    /// Every RampDir orientation produces a per-side climbable run: the treads
    /// advance by `ramp_step(dir)` and rise 1 each, exactly like the PosX proof.
    #[test]
    fn straight_staircase_all_directions_advance_and_rise() {
        const H: u32 = 3;
        for &dir in &[RampDir::PosX, RampDir::NegX, RampDir::PosY, RampDir::NegY] {
            let base = IVec3::new(10, 10, 0);
            let stamps = straight_staircase(base, dir, H);
            let step = ramp_step(dir);
            for n in 1..=H as i32 {
                let s = stamps[(n - 1) as usize];
                let expected = base + (step + IVec3::new(0, 0, 1)) * n;
                assert_eq!(s.pos, expected, "{dir:?} tread {n} position");
                assert_eq!(s.shape, Shape::RampZ(dir), "{dir:?} tread {n} shape");
                // Rises exactly 1 in z per tread.
                if n > 1 {
                    let prev = stamps[(n - 2) as usize];
                    assert_eq!(s.pos.z - prev.pos.z, 1, "{dir:?} must rise 1/tread");
                }
            }
        }
    }

    #[test]
    fn straight_staircase_zero_height_is_just_landing() {
        let stamps = straight_staircase(IVec3::new(0, 0, 0), RampDir::PosX, 0);
        assert_eq!(stamps.len(), 1, "H=0 → only the landing");
        assert_eq!(stamps[0].shape, Shape::Full);
    }

    // ---- spiral_staircase: geometric correctness ----

    #[test]
    fn spiral_central_column_is_solid() {
        let center = IVec3::new(20, 20, 0);
        let radius = 1;
        let turns = 2;
        let stamps = spiral_staircase(center, radius, turns, 4);
        // radius 1 → core is the single centre column line. Every z from base to
        // the top landing must have a solid centre cell.
        let side_len = 2 * radius;
        let total_treads = side_len * 4 * turns as i32;
        let top_z = center.z + total_treads;
        for z in center.z..=top_z {
            let want = IVec3::new(center.x, center.y, z);
            assert!(
                stamps.iter().any(|s| s.pos == want
                    && s.kind == CellKind::WallSolid
                    && s.shape == Shape::Full),
                "central column must be solid at z={z}"
            );
        }
    }

    #[test]
    fn spiral_rise_equals_ramp_tread_count() {
        // One full turn rises 8*radius cells (4 sides * 2*radius treads).
        let center = IVec3::new(0, 0, 0);
        let radius = 1;
        let turns = 3;
        let stamps = spiral_staircase(center, radius, turns, 4);

        let ramp_count = stamps
            .iter()
            .filter(|s| matches!(s.shape, Shape::RampZ(_)))
            .count() as i32;
        let expected = 8 * radius * turns as i32;
        assert_eq!(
            ramp_count, expected,
            "ramp tread count must equal rise per turn (8*radius) * turns"
        );

        // The ramp treads must climb monotonically: each ramp's z >= the
        // previous ramp's z (the ring rises continuously, never dips).
        let mut last_z = i32::MIN;
        for s in stamps.iter().filter(|s| matches!(s.shape, Shape::RampZ(_))) {
            assert!(
                s.pos.z >= last_z,
                "ramp ring must rise monotonically: z={} after {last_z}",
                s.pos.z
            );
            last_z = s.pos.z;
        }
    }

    #[test]
    fn spiral_sides_rotate_90_degrees() {
        // Across one turn the ramp dirs must cycle PosX→PosY→NegX→NegY (90° per
        // side), proving each side run is oriented for its edge.
        let stamps = spiral_staircase(IVec3::new(0, 0, 0), 1, 1, 4);
        // Collect the dir of the first tread of each side run. side_len=2 treads
        // + 1 landing per side. The ramp dirs of the 4 sides:
        let dirs: Vec<RampDir> = stamps
            .iter()
            .filter_map(|s| match s.shape {
                Shape::RampZ(d) => Some(d),
                _ => None,
            })
            .collect();
        // 8 ramp treads (2 per side * 4 sides). First two PosX, next two PosY, …
        assert_eq!(dirs.len(), 8);
        assert_eq!(&dirs[0..2], &[RampDir::PosX, RampDir::PosX]);
        assert_eq!(&dirs[2..4], &[RampDir::PosY, RampDir::PosY]);
        assert_eq!(&dirs[4..6], &[RampDir::NegX, RampDir::NegX]);
        assert_eq!(&dirs[6..8], &[RampDir::NegY, RampDir::NegY]);
    }

    #[test]
    fn spiral_each_side_run_advances_like_straight_prefab() {
        // Each side's treads must advance by ramp_step(dir) and rise 1 — the
        // SAME per-side climbable check as the straight prefab.
        let stamps = spiral_staircase(IVec3::new(0, 0, 0), 1, 1, 4);
        let ramps: Vec<CellStamp> = stamps
            .iter()
            .copied()
            .filter(|s| matches!(s.shape, Shape::RampZ(_)))
            .collect();
        // Walk consecutive treads of the SAME dir; each pair must differ by
        // ramp_step(dir) + (0,0,1).
        for w in ramps.windows(2) {
            let (a, b) = (w[0], w[1]);
            if let (Shape::RampZ(da), Shape::RampZ(db)) = (a.shape, b.shape) {
                if da == db {
                    let delta = b.pos - a.pos;
                    assert_eq!(
                        delta,
                        ramp_step(da) + IVec3::new(0, 0, 1),
                        "same-dir treads must advance like the straight prefab"
                    );
                }
            }
        }
    }

    #[test]
    fn spiral_degenerate_inputs_are_defined() {
        // radius < 1 → nothing (no panic).
        assert!(spiral_staircase(IVec3::ZERO, 0, 2, 4).is_empty());
        // turns == 0 → just the single column cell.
        let s = spiral_staircase(IVec3::new(5, 5, 0), 1, 0, 4);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].pos, IVec3::new(5, 5, 0));
        assert_eq!(s[0].shape, Shape::Full);
    }

    #[test]
    fn rotate_ramp_dir_cycles_through_all_four() {
        let mut d = RampDir::PosX;
        let mut seen = Vec::new();
        for _ in 0..4 {
            seen.push(d);
            d = rotate_ramp_dir_cw(d);
        }
        assert_eq!(d, RampDir::PosX, "4 rotations return to start");
        assert_eq!(
            seen,
            vec![RampDir::PosX, RampDir::PosY, RampDir::NegX, RampDir::NegY]
        );
    }
}
