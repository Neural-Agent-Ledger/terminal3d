//! TEST-ONLY diagnostic (`#[cfg(test)]`, never compiled into the shipped binary):
//! a trustworthy counter for the "dark strip at wall bases" defect class, extended to
//! cover the offslice / multi-plane path.
//!
//! ## Background
//!
//! `tests/golden/README.md` 2026-09-24 (c) fixed the PRIMARY case: the
//! classic per-row floor/ceiling cast in
//! `cast_column_depth` could compute a distance FARTHER than the column's own primary
//! DDA wall (`clamp_to_wall_depth`, keyed on `perp`). The (c)/(d) entries flagged, but
//! did not fix or trust-count, a second occurrence in the SEPARATE vertical/offslice
//! multi-plane resolver (`crate::offslice`): a floor/ceiling row can sit directly below
//! a wall the VERTICAL PASS painted (a different z-plane than the camera's own slice —
//! a stairwell rim, a balcony's underside, a raised interior wall) with no equivalent
//! clamp, and an earlier "quick, unvalidated" sweep (16,712 cells / 70 levels) was
//! explicitly distrusted for having no false-positive filter.
//!
//! ## The ground-truth test
//!
//! Rather than re-deriving "is the wall above this floor row on the SAME storey", this
//! module asks a strictly simpler, provably-correct question per candidate cell: does a
//! REAL, SOLID surface painted by the OFFSLICE / VERTICAL PASS — found by the exact
//! production ray-vs-geometry resolver it uses (`offslice::query_row`, the same function
//! `shade_subray`'s sub-cell refinement and `vertical_resolve` both call) — stand
//! strictly nearer along this column's own view ray than the depth the cell claims to
//! show? If yes, the cell is rendering a point the vertical pass's own geometry
//! occludes: geometrically impossible.
//!
//! **The primary DDA's own wall cell is out of scope** (excluded by (x, y) IDENTITY,
//! not by `Role` — see [`is_not_an_occluder_for_audit`]): that one already has its fix
//! (`clamp_to_wall_depth`) — a WINDOWED one, by deliberate, documented design
//! (`tests/golden/README.md` 2026-09-24 (d), "Item 2"): a wall close enough to the
//! camera keeps a long tail of rows at their pre-fix "impossible" depth on purpose, to
//! avoid flattening the floor into a wide plateau. Treating it as an occluder here
//! would re-flag that ALREADY-ACCEPTED trade-off on every level with a nearby wall —
//! pure noise for this diagnostic. A `Role::Slice` hit at a DIFFERENT (x, y) — a second wall
//! on the SAME plane as the camera, farther along the same ray — is NOT excluded and
//! DOES count: `clamp_to_wall_depth` only ever clamps against the nearest (`primary`)
//! wall, so a farther same-plane wall is exactly as unclamped as an off-slice one. Two
//! shipped fixtures (`TX6-subcell-ladder.lvl`, `TX7-edge-glyphs.lvl`, texture/glyph test
//! rooms, 1 cell each) hit exactly this: a genuine, un-clamped second same-plane wall,
//! not off-slice geometry at all. That is real and this counter is right to flag it,
//! but it is a DIFFERENT defect class than this module's target (the vertical/offslice
//! multi-plane resolver) — fixing it means extending `clamp_to_wall_depth` itself to
//! multiple same-plane walls, a primary-cast change, not an `offslice.rs` one — so it
//! is reported here as a known, scoped-out follow-up rather than folded into this fix.
//!
//! This one test gets the other false-positive exclusions FOR FREE, as a
//! consequence of reusing production's own occlusion semantics rather than
//! duplicating them by hand:
//!
//! * **A ledge / balcony edge revealing a farther, lower storey** is never flagged:
//!   by construction there is nothing solid between the camera and that farther floor,
//!   so `query_row` finds no nearer occluder and the check passes.
//! * **Holograms** are `offslice::Role::Air` in `offslice::role_of` — already
//!   transparent to `query_row`, no special case needed.
//! * **Sky** is the absence of any cell (`CellKind::Empty`, also `Role::Air`) — same
//!   mechanism.
//! * **Portals** re-base the ray in production, so the real destination legitimately
//!   reads farther than the portal's own near-face position — excluded explicitly.
//! * **A `Slab`'s horizontal lid, ANY face** (the ordinary floor/ceiling continuation,
//!   not a separate wall) is excluded — it is the SAME surface the classic backdrop
//!   already renders, under its own accepted formula, not a second occluding object.
//!
//! ## What is NOT re-derived
//!
//! This module never re-implements the caster's floor/ceiling formula, the vertical
//! pass's coverage bookkeeping, or `clamp_to_wall_depth` itself — it only re-asks
//! production's own `query_row` the ground-truth occlusion question, and compares the
//! answer against the depth the ACTUAL rendered `Cell` (from `raycast_with_time_fov_gameplay`
//! / `composite::composite_dump_once`, whichever the caller rendered) carries. A
//! disagreement is decisive: the same units, the same projection, the same map.

use crate::offslice::{self, VHit};
use crate::{
    draw_dist, gameplay_eye_ref_z, pitch_row_shift, vhit_opaque, view_params, vshift_per_perp, Fov,
    MaterialRegistry,
};
use re_core::{Framebuffer, SurfaceKind};
use re_world::{CameraPose, CellKind, Map};

/// One confirmed violation: a Floor/Ceiling cell, directly adjoining a WALL
/// run in the same column (the "wall-bottom → floor" / "ceiling → wall-top"
/// transition this module audits for), whose rendered depth is farther than
/// a real, strictly-nearer solid surface a ground-truth ray query proves
/// exists on the same view ray.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Violation {
    pub(crate) col: usize,
    pub(crate) row: usize,
    pub(crate) is_ceiling: bool,
    /// The depth the frame actually rendered at this cell.
    pub(crate) rendered_depth: f32,
    /// The nearer real occluder's own depth (`< rendered_depth − EPS`).
    pub(crate) occluder_depth: f32,
}

/// Depth slack (world/perp units) below which two independently-computed depths count
/// as "the same point" rather than a violation. Matches the primary-case regression
/// test's own tolerance (`crates/re_render/src/lib.rs`,
/// `floor_row_below_a_far_wall_is_owned_by_the_wall_not_a_farther_floor`), loosened
/// slightly because this check compares TWO different code paths (the rendered cell's
/// own formula vs. the independent `offslice::query_row` polygon resolve) rather than
/// one formula against itself.
const EPS: f32 = 0.08;

/// The exact per-frame view basis `raycast_with_time_fov_gameplay` /
/// `composite::composite_dump_once` derive from `(cam, fov, cols, rows)` — reproduced
/// here (not `SubViewBasis`, which deliberately pins its `cell_z` to the STANDING plane
/// for its own sub-cell-refinement purpose and would silently use the WRONG plane on
/// every elevated-storey pose, exactly the levels this audit targets: `22-tower`,
/// `21-stairs-spiral`, `L1-first-pulse`, ...). See `crate::raycast_full_memo`'s own
/// head, which this mirrors field-for-field.
struct GameplayBasis {
    dir_x: f32,
    dir_y: f32,
    plane_x: f32,
    plane_y: f32,
    pos_x: f32,
    pos_y: f32,
    cell_z: i32,
    horizon: i32,
    proj_plane_dist: f32,
    vshift_per_perp: f32,
    cols: usize,
}

impl GameplayBasis {
    fn new(cam: &CameraPose, fov: &Fov, cols: usize, rows: usize) -> Self {
        let view = view_params(cols.max(1), rows.max(1), fov);
        let (dir_x, dir_y) = cam.dir();
        let plane_x = cam.yaw.cos() * view.plane_scale;
        let plane_y = -cam.yaw.sin() * view.plane_scale;
        let eye_ref_z = gameplay_eye_ref_z(cam.pos.z);
        GameplayBasis {
            dir_x,
            dir_y,
            plane_x,
            plane_y,
            pos_x: cam.pos.x,
            pos_y: cam.pos.y,
            cell_z: eye_ref_z.floor() as i32,
            horizon: (rows.max(1) as i32 / 2) + pitch_row_shift(cam.pitch, rows.max(1)),
            proj_plane_dist: view.proj_plane_dist,
            vshift_per_perp: vshift_per_perp(cam.pos.z, eye_ref_z, view.proj_plane_dist),
            cols: cols.max(1),
        }
    }

    /// The (non-unit) ray direction for column `col`, matching
    /// `raycast_full_memo`'s own `camera_x` / `ray_dir_x` derivation exactly.
    fn ray_dir(&self, col: usize) -> (f32, f32) {
        let camera_x = 2.0 * (col as f32 + 0.5) / self.cols as f32 - 1.0;
        (
            self.dir_x + self.plane_x * camera_x,
            self.dir_y + self.plane_y * camera_x,
        )
    }
}

/// Is `h` — the FIRST hit production's own `vhit_opaque` predicate would have stopped
/// the search at (see [`nearer_real_occluder`]) — NOT a real, violation-relevant
/// occluder? Two cases (see the module doc's "Scope" note for the full rationale):
///
/// * **The SAME (x, y) grid cell as this column's PRIMARY DDA hit** (`primary_xy`) —
///   whatever z-plane bucket a specific row's ray happens to land the hit test in. A
///   full-height wall is solid at EVERY z, so the ray's horizontal crossing distance
///   into that (x, y) cell (`Step.t_in`, identical to the primary's own `perp` by
///   construction — both walk the SAME `GridWalk`) never depends on which integer z
///   bucket a given row's line happens to enter it through. Treating a different-z
///   bucket of the IDENTICAL wall cell as a separate "off-slice" occluder would just
///   re-flag the PRIMARY DDA's own wall (already the (c)-fix's territory —
///   `clamp_to_wall_depth`, windowed BY DESIGN, so a residual "impossible" tail on a
///   near wall, or a wide-FOV edge column's grazing angle, is an ALREADY-ACCEPTED
///   trade-off, not this module's target).
/// * **An off-slice `Portal` face** (`Role::Paint` on a `CellKind::Portal` cell) —
///   mirrors production's own `vertical_claims` exactly ("a portal face is not
///   claimed: its warped ray may escape, and then the row keeps the backdrop"): the
///   row falls back to the backdrop's own reading, which may legitimately be farther
///   than the portal's own near-face position, so the portal itself is never the
///   right thing to compare a rendered depth against.
/// * **A `Slab`'s lid, ANY face — `Top`, `Bottom`, AND `Near`.** `offslice`'s own
///   production match arm (`vertical_claims`/`shade_subray`,
///   `(Role::Slice, _) | (Role::Slab, Face::Top | Face::Bottom)`) only lists
///   `Top`/`Bottom` as "covers, does not paint" — but empirically (see this module's
///   git history: an earlier version keyed on exactly that pair and still mis-flagged
///   a PLAIN, FLAT, CONTINUOUS floor in a box room with no ledge, no dais, no rim at
///   all) a `Slab`'s `Face::Near` fires as an INCIDENTAL PRECISION ARTIFACT too: when
///   a floor row's view ray height already sits inside the slab's `[z, z+1)` band at
///   the EXACT moment it crosses into a farther grid column, `offslice::enter` reports
///   `Face::Near` (the "already inside the band at entry" branch) rather than
///   `Face::Top` (the "crosses the top line inside the column" branch) — same
///   surface, same ordinary floor, no actual vertical geometry, purely which side of a
///   column boundary the crossing height happened to land on. Production itself never
///   observes this for a real level's plain floors: `vertical_claims`'s `covered[]`
///   bookkeeping marks a row DONE at the FIRST column whose `resolve_in_column` finds
///   ANY `vhit_opaque`-true hit — for an ordinary flat floor that is virtually always
///   the earlier, nearer `Face::Top` crossing (excluded, but still "covers" the row) —
///   so a farther column's `Face::Near` artifact is normally never even reached. This
///   audit's [`nearer_real_occluder`] reproduces exactly that first-hit-wins order (a
///   single `query_row` call using the UNMODIFIED `vhit_opaque` predicate, classified
///   AFTER the fact — never a predicate that skips an excluded hit and keeps hunting
///   farther), so the same ordering guarantee holds here; excluding `Face::Near` too
///   is the correct, matching choice for a genuine dais/rim, which is authored in this
///   engine as a separate `WallSolid` cell (`Role::Paint`/`Role::Slice`, see
///   `offslice.rs`'s own `a_deck_overhead_shows_its_underside_and_hides_what_is_above`
///   test) — never as a `Floor`/`Ceiling` cell with a meaningful vertical face. A
///   `Role::Paint` hit (an off-slice wall on a different plane) is never excluded here
///   — it is exactly the class this audit targets.
///
#[inline]
fn is_not_an_occluder_for_audit(map: &Map, h: &VHit, primary_xy: Option<(i32, i32)>) -> bool {
    if primary_xy == Some((h.x, h.y)) {
        return true;
    }
    if h.role == offslice::Role::Paint && map.cell_at(h.x, h.y, h.plane) == CellKind::Portal {
        return true;
    }
    h.role == offslice::Role::Slab
}

/// Ground-truth "is the FIRST solid surface production's own vertical pass would stop
/// at on this row's view ray a violation-relevant occluder?" — reproduces
/// `offslice::vertical_claims`'s exact search order: ONE `query_row` call with the
/// UNMODIFIED `crate::vhit_opaque` predicate (so the search stops at the same first
/// hit production's `covered[]` row-bookkeeping would have locked in — never a
/// predicate that transparently skips an excluded hit and keeps hunting farther,
/// which would find geometry production itself never reaches for this row), then
/// classifies that single hit via [`is_not_an_occluder_for_audit`] (holograms/sky are
/// already `Role::Air`, invisible to `query_row` itself — no extra rule needed here).
/// `eye` is the caller's own `offslice::VProj::eye` — never recomputed here, so there
/// is exactly ONE place in this module that derives it (`GameplayBasis` + `VProj::new`
/// in [`sweep_frame`]), matching production's own single source. `primary_xy` is this
/// column's primary DDA hit cell (`None` on a primary miss), computed ONCE per column
/// by the caller.
#[allow(clippy::too_many_arguments)]
fn nearer_real_occluder(
    map: &Map,
    registry: &MaterialRegistry,
    cell_z: i32,
    pos_x: f32,
    pos_y: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    eye: f32,
    k: f32,
    max_steps: i32,
    primary_xy: Option<(i32, i32)>,
) -> Option<f32> {
    let hit = offslice::query_row(
        map,
        cell_z,
        pos_x,
        pos_y,
        ray_dir_x,
        ray_dir_y,
        eye,
        k,
        max_steps,
        |h| vhit_opaque(map, registry, h, pos_x, pos_y, ray_dir_x, ray_dir_y),
    )?;
    if is_not_an_occluder_for_audit(map, &hit, primary_xy) {
        None
    } else {
        Some(hit.s)
    }
}

/// Sweep an ALREADY-RENDERED frame (`fb`) for the residual dark-strip defect. `fb` must
/// be what `raycast_with_time_fov_gameplay` / `composite::composite_dump_once` produced
/// for this EXACT `(map, cam, fov)` at `fb`'s own `(cols, rows)` — the caller renders it
/// (so the same frame can be reused for other checks / inspected visually too).
///
/// For every column, tracks contiguous FLOOR runs that begin immediately below a WALL
/// row (`wall-bottom → floor`) and contiguous CEILING runs
/// that end immediately above a WALL row (the mirror-image `ceiling → wall-top` case
/// the same historical fix's ceiling branch covers) — the WINDOW is the whole
/// contiguous run, not just the first row, because the historical bug already showed a
/// 1–2 row extent and there is no reason to assume the offslice analogue is narrower.
/// Every candidate cell is checked via [`nearer_real_occluder`]; a hit strictly nearer
/// than the cell's own rendered depth (beyond [`EPS`]) is a confirmed [`Violation`].
pub(crate) fn sweep_frame(
    map: &Map,
    cam: &CameraPose,
    fov: &Fov,
    fb: &Framebuffer,
) -> Vec<Violation> {
    let cols = fb.cols();
    let rows = fb.rows();
    let registry = crate::registry_for_map(map);
    let basis = GameplayBasis::new(cam, fov, cols, rows);
    let max_steps = (draw_dist::active_max_dda_steps() as i32).min(offslice::MAX_STEPS);
    let vp = offslice::VProj::new(
        basis.cell_z,
        basis.horizon,
        basis.proj_plane_dist,
        basis.vshift_per_perp,
    );

    let mut out = Vec::new();
    for col in 0..cols {
        let (ray_dir_x, ray_dir_y) = basis.ray_dir(col);
        // This column's PRIMARY DDA hit cell (z-agnostic — the same 2D grid walk the
        // offslice resolver also runs), computed ONCE: excluded from the occluder
        // query so a full-height wall re-bucketed at a different z is never mistaken
        // for a separate off-slice object (see `is_not_an_occluder_for_audit`).
        let primary = crate::dda_hit(
            map,
            basis.cell_z,
            basis.pos_x,
            basis.pos_y,
            ray_dir_x,
            ray_dir_y,
            Some(&registry),
        );
        let primary_xy = primary.hit.then_some((primary.cell_x, primary.cell_y));

        // ── FLOOR runs starting right under a wall row ──
        let mut prev_is_wall = false;
        let mut run_from_wall = false;
        let mut prev_was_floor = false;
        for row in 0..rows {
            let Some(cell) = fb.get(col, row) else {
                continue;
            };
            let is_wall = cell.surface.is_wall();
            let is_floor = cell.surface == SurfaceKind::Floor;
            if is_floor {
                if !prev_was_floor {
                    run_from_wall = prev_is_wall;
                }
                if run_from_wall {
                    let k = vp.slope(row as f32 + 0.5);
                    if let Some(occ) = nearer_real_occluder(
                        map,
                        &registry,
                        basis.cell_z,
                        basis.pos_x,
                        basis.pos_y,
                        ray_dir_x,
                        ray_dir_y,
                        vp.eye,
                        k,
                        max_steps,
                        primary_xy,
                    ) {
                        if occ + EPS < cell.depth {
                            out.push(Violation {
                                col,
                                row,
                                is_ceiling: false,
                                rendered_depth: cell.depth,
                                occluder_depth: occ,
                            });
                        }
                    }
                }
            } else {
                run_from_wall = false;
            }
            prev_is_wall = is_wall;
            prev_was_floor = is_floor;
        }

        // ── CEILING runs ending right above a wall row (mirror image; walked
        // bottom-to-top so "the row below" plays the role "the row above" played
        // for the floor pass) ──
        let mut next_is_wall = false;
        let mut run_from_wall = false;
        let mut next_was_ceiling = false;
        for row in (0..rows).rev() {
            let Some(cell) = fb.get(col, row) else {
                continue;
            };
            let is_wall = cell.surface.is_wall();
            let is_ceiling = cell.surface == SurfaceKind::Ceiling;
            if is_ceiling {
                if !next_was_ceiling {
                    run_from_wall = next_is_wall;
                }
                if run_from_wall {
                    let k = vp.slope(row as f32 + 0.5);
                    if let Some(occ) = nearer_real_occluder(
                        map,
                        &registry,
                        basis.cell_z,
                        basis.pos_x,
                        basis.pos_y,
                        ray_dir_x,
                        ray_dir_y,
                        vp.eye,
                        k,
                        max_steps,
                        primary_xy,
                    ) {
                        if occ + EPS < cell.depth {
                            out.push(Violation {
                                col,
                                row,
                                is_ceiling: true,
                                rendered_depth: cell.depth,
                                occluder_depth: occ,
                            });
                        }
                    }
                }
            } else {
                run_from_wall = false;
            }
            next_is_wall = is_wall;
            next_was_ceiling = is_ceiling;
        }
    }
    out
}

#[cfg(test)]
mod validation {
    use super::*;
    use crate::{registry_for_map, Fov};
    use re_core::{Cell, Shape};
    use re_world::{stamp_geometry, GeometrySpec, STANDING_EYE_Z};

    /// A campaign-shaped room: floor z=0, habitable z=1..sz-2, ceiling z=sz-1 — the
    /// same shape `offslice.rs`'s own test helper uses.
    fn room(sx: usize, sy: usize, sz: usize) -> Map {
        stamp_geometry(&GeometrySpec {
            size: (sx, sy, sz),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        })
    }

    fn straight_cam(x: f32, y: f32) -> CameraPose {
        CameraPose {
            pos: glam::Vec3::new(x, y, STANDING_EYE_Z),
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
        }
    }

    /// The same `primary_xy` [`sweep_frame`] computes for a column — a small test
    /// helper so the validation tests exercise `nearer_real_occluder` exactly as the
    /// real sweep does, rather than a stand-in `None`.
    fn primary_xy_for(
        map: &Map,
        registry: &MaterialRegistry,
        basis: &GameplayBasis,
        ray_dir_x: f32,
        ray_dir_y: f32,
    ) -> Option<(i32, i32)> {
        let primary = crate::dda_hit(
            map,
            basis.cell_z,
            basis.pos_x,
            basis.pos_y,
            ray_dir_x,
            ray_dir_y,
            Some(registry),
        );
        primary.hit.then_some((primary.cell_x, primary.cell_y))
    }

    /// Build a `cols`×`rows` framebuffer with every cell defaulted to a harmless
    /// Ceiling-at-infinity so an untouched cell never accidentally satisfies a "wall
    /// run" or "floor run" predicate.
    fn blank_fb(cols: usize, rows: usize) -> Framebuffer {
        Framebuffer::new(cols, rows)
    }

    fn set_wall(fb: &mut Framebuffer, col: usize, row: usize, depth: f32) {
        let c = Cell {
            surface: SurfaceKind::WallSolid,
            depth,
            ..Cell::default()
        };
        fb.set(col, row, c);
    }

    fn set_floor(fb: &mut Framebuffer, col: usize, row: usize, depth: f32) {
        let c = Cell {
            surface: SurfaceKind::Floor,
            depth,
            ..Cell::default()
        };
        fb.set(col, row, c);
    }

    fn set_ceiling(fb: &mut Framebuffer, col: usize, row: usize, depth: f32) {
        let c = Cell {
            surface: SurfaceKind::Ceiling,
            depth,
            ..Cell::default()
        };
        fb.set(col, row, c);
    }

    // ── TRUE POSITIVE: a real, nearer solid stands where the fabricated Floor cell
    // claims nothing does ──
    #[test]
    fn flags_a_floor_cell_claiming_a_depth_a_real_wall_already_occludes() {
        let (sx, sy, sz) = (20usize, 3usize, 4usize);
        let mut map = room(sx, sy, sz);
        // An OFF-SLICE wall (plane 2, one above the camera's own slice plane 1): a
        // free-standing interior column blocking the corridor partway down, the same
        // shape a stairwell rim / raised interior wall takes in the shipped
        // multi-storey levels this audit targets.
        map.set_cell(10, 1, 2, CellKind::WallSolid, 0);

        let cam = straight_cam(2.5, 1.5);
        let (cols, rows) = (1usize, 32usize);
        let fov = Fov::default();
        let basis = GameplayBasis::new(&cam, &fov, cols, rows);
        assert_eq!(
            basis.cell_z, 1,
            "grounded standing camera must slice plane 1"
        );
        let vp = offslice::VProj::new(
            basis.cell_z,
            basis.horizon,
            basis.proj_plane_dist,
            basis.vshift_per_perp,
        );

        // Find a screen row whose ray actually meets the off-slice column's near face,
        // by scanning with the SAME ground-truth query the sweep itself uses.
        let (ray_dir_x, ray_dir_y) = basis.ray_dir(0);
        let registry = registry_for_map(&map);
        let primary_xy = primary_xy_for(&map, &registry, &basis, ray_dir_x, ray_dir_y);
        let max_steps = 256;
        let mut found_row = None;
        let mut found_depth = 0.0f32;
        for row in 1..rows {
            let k = vp.slope(row as f32 + 0.5);
            if let Some(occ) = nearer_real_occluder(
                &map,
                &registry,
                basis.cell_z,
                basis.pos_x,
                basis.pos_y,
                ray_dir_x,
                ray_dir_y,
                vp.eye,
                k,
                max_steps,
                primary_xy,
            ) {
                // Only accept a row whose occluder is the off-slice column, not the
                // corridor's own far end wall (x = sx-1).
                if occ < 8.0 {
                    found_row = Some(row);
                    found_depth = occ;
                    break;
                }
            }
        }
        let wall_row = found_row.expect("the off-slice column must occlude some row");

        // Fabricate the buggy frame by hand: the row above is (any) wall, and the
        // candidate row claims to be FLOOR reading far past the real occluder — the
        // exact shape of the historical bug, this time from an off-slice source.
        let mut fb = blank_fb(cols, rows);
        set_wall(&mut fb, 0, wall_row.saturating_sub(1), found_depth);
        set_floor(&mut fb, 0, wall_row, found_depth + 5.0);

        let violations = sweep_frame(&map, &cam, &fov, &fb);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].row, wall_row);
        assert!(!violations[0].is_ceiling);
        assert!(
            (violations[0].occluder_depth - found_depth).abs() < 0.2,
            "{violations:?}"
        );
    }

    // ── TRUE POSITIVE (ceiling mirror) ──
    #[test]
    fn flags_a_ceiling_cell_claiming_a_depth_a_real_wall_already_occludes() {
        let (sx, sy, sz) = (20usize, 3usize, 5usize);
        let mut map = room(sx, sy, sz);
        map.set_cell(10, 1, 2, CellKind::WallSolid, 0);

        // Pitch up so the near occluder shows on the CEILING side of the horizon.
        let cam = CameraPose {
            pos: glam::Vec3::new(2.5, 1.5, STANDING_EYE_Z),
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.5,
        };
        let (cols, rows) = (1usize, 32usize);
        let fov = Fov::default();
        let basis = GameplayBasis::new(&cam, &fov, cols, rows);
        let vp = offslice::VProj::new(
            basis.cell_z,
            basis.horizon,
            basis.proj_plane_dist,
            basis.vshift_per_perp,
        );
        let (ray_dir_x, ray_dir_y) = basis.ray_dir(0);
        let registry = registry_for_map(&map);
        let primary_xy = primary_xy_for(&map, &registry, &basis, ray_dir_x, ray_dir_y);
        let max_steps = 256;

        let mut found_row = None;
        let mut found_depth = 0.0f32;
        for row in 0..rows.saturating_sub(1) {
            if (row as i32) >= basis.horizon {
                continue; // ceiling side only
            }
            let k = vp.slope(row as f32 + 0.5);
            if let Some(occ) = nearer_real_occluder(
                &map,
                &registry,
                basis.cell_z,
                basis.pos_x,
                basis.pos_y,
                ray_dir_x,
                ray_dir_y,
                vp.eye,
                k,
                max_steps,
                primary_xy,
            ) {
                if occ < 8.0 {
                    found_row = Some(row);
                    found_depth = occ;
                    break;
                }
            }
        }
        let Some(wall_row) = found_row else {
            // The pitch/geometry combination did not line up a ceiling-side ray onto
            // the column in this environment; nothing to validate against, but do not
            // silently pass — fail loudly so the fixture gets adjusted, not ignored.
            panic!("expected a ceiling-side row to meet the off-slice column");
        };

        let mut fb = blank_fb(cols, rows);
        set_ceiling(&mut fb, 0, wall_row, found_depth + 5.0);
        set_wall(&mut fb, 0, wall_row + 1, found_depth);

        let violations = sweep_frame(&map, &cam, &fov, &fb);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].row, wall_row);
        assert!(violations[0].is_ceiling);
    }

    // ── FALSE POSITIVE: a genuine ledge — an ordinary, unbroken floor row (nothing
    // nearer stands in front of it) must NOT be flagged, however far it reads. Ground
    // truth here is `nearer_real_occluder` itself, matching `offslice::vertical_claims`'s
    // own first-hit-wins search order: for a PLAIN floor with no off-slice geometry at
    // all, the first (and only) surface the search meets is the floor's OWN lid
    // (`Role::Slab`, excluded — see `is_not_an_occluder_for_audit`), so the correct
    // ground-truth answer is "no occluder", not "the corridor's far wall" — proving
    // that is itself part of what this test checks, before using `None` as licence to
    // read an arbitrarily large depth (the general "no false alarm on a legitimately
    // farther surface, e.g. seen past a ledge onto a lower storey" shape). ──
    #[test]
    fn does_not_flag_a_genuinely_farther_floor_seen_past_a_ledge() {
        let (sx, sy, sz) = (20usize, 3usize, 4usize);
        let map = room(sx, sy, sz); // no off-slice geometry at all — every floor row is genuine
        let cam = straight_cam(2.5, 1.5);
        let (cols, rows) = (1usize, 32usize);
        let fov = Fov::default();

        let basis = GameplayBasis::new(&cam, &fov, cols, rows);
        let vp = offslice::VProj::new(
            basis.cell_z,
            basis.horizon,
            basis.proj_plane_dist,
            basis.vshift_per_perp,
        );
        let (ray_dir_x, ray_dir_y) = basis.ray_dir(0);
        let registry = registry_for_map(&map);
        let row = basis.horizon as usize + 5;
        let k = vp.slope(row as f32 + 0.5);
        let occ = nearer_real_occluder(
            &map,
            &registry,
            basis.cell_z,
            basis.pos_x,
            basis.pos_y,
            ray_dir_x,
            ray_dir_y,
            vp.eye,
            k,
            256,
            None,
        );
        assert_eq!(
            occ, None,
            "an ordinary, unbroken floor row must resolve to NO occluder, not the far wall"
        );
        // With no real occluder, an arbitrarily large claimed depth is legitimate —
        // exactly the "seen past a ledge onto a lower storey" shape.
        let legit_depth = 50.0f32;

        let mut fb = blank_fb(cols, rows);
        set_wall(&mut fb, 0, row - 1, 3.0);
        set_floor(&mut fb, 0, row, legit_depth);
        let violations = sweep_frame(&map, &cam, &fov, &fb);
        assert!(violations.is_empty(), "{violations:?}");
    }

    // ── FALSE POSITIVE: a portal must not count as an occluder — a floor reading the
    // depth of whatever is legitimately BEHIND the portal (farther than the portal's
    // own near-face position) must not be flagged ──
    #[test]
    fn does_not_flag_a_floor_read_through_a_portal() {
        let (sx, sy, sz) = (20usize, 3usize, 4usize);
        let mut map = room(sx, sy, sz);
        map.set_cell(6, 1, 2, CellKind::Portal, 0); // off-slice portal, plane 2
        let cam = straight_cam(2.5, 1.5);
        let (cols, rows) = (1usize, 32usize);
        let fov = Fov::default();
        let basis = GameplayBasis::new(&cam, &fov, cols, rows);
        let vp = offslice::VProj::new(
            basis.cell_z,
            basis.horizon,
            basis.proj_plane_dist,
            basis.vshift_per_perp,
        );
        let (ray_dir_x, ray_dir_y) = basis.ray_dir(0);
        let registry = registry_for_map(&map);
        let primary_xy = primary_xy_for(&map, &registry, &basis, ray_dir_x, ray_dir_y);

        // Locate a row whose RAW (non-audited, `opaque = |_| true`) resolve stops on
        // the portal plane — proving the geometry genuinely puts the portal in this
        // row's path, not just an empty ray.
        let mut found = None;
        for row in 1..rows {
            let k = vp.slope(row as f32 + 0.5);
            let raw = offslice::query_row(
                &map,
                basis.cell_z,
                basis.pos_x,
                basis.pos_y,
                ray_dir_x,
                ray_dir_y,
                vp.eye,
                k,
                256,
                |_| true,
            );
            if let Some(h) = raw {
                if map.cell_at(h.x, h.y, h.plane) == CellKind::Portal {
                    found = Some((row, h.s));
                    break;
                }
            }
        }
        let (row, portal_depth) =
            found.expect("expected a row whose raw resolve is the portal plane");

        // The AUDITED query must either find nothing, or find something strictly
        // FARTHER than the portal itself — proving the portal was skipped, not treated
        // as the occluder.
        let k = vp.slope(row as f32 + 0.5);
        let audited = nearer_real_occluder(
            &map,
            &registry,
            basis.cell_z,
            basis.pos_x,
            basis.pos_y,
            ray_dir_x,
            ray_dir_y,
            vp.eye,
            k,
            256,
            primary_xy,
        );
        if let Some(d) = audited {
            assert!(
                d > portal_depth + EPS,
                "the portal itself must not be the reported occluder"
            );
        }

        let mut fb = blank_fb(cols, rows);
        set_wall(&mut fb, 0, row - 1, 2.0);
        // The "destination" depth a real warp would legitimately report: farther than
        // the portal's own near-face position, and not exceeding whatever the audited
        // query actually found (so this is a LEGITIMATE reading, not a second bug).
        let upper = audited.unwrap_or(portal_depth + 50.0);
        let dest_depth = portal_depth + (upper - portal_depth) * 0.5;
        assert!(
            dest_depth > portal_depth,
            "fixture sanity: must read past the portal"
        );
        set_floor(&mut fb, 0, row, dest_depth);
        let violations = sweep_frame(&map, &cam, &fov, &fb);
        assert!(violations.is_empty(), "{violations:?}");
    }

    // ── FALSE POSITIVE: a hologram must not count as an occluder (already Role::Air,
    // proven here rather than assumed) ──
    #[test]
    fn does_not_flag_a_floor_read_through_a_hologram() {
        let (sx, sy, sz) = (20usize, 3usize, 4usize);
        let mut map = room(sx, sy, sz);
        map.set_cell(6, 1, 2, CellKind::Hologram, 0);
        let cam = straight_cam(2.5, 1.5);
        let (cols, rows) = (1usize, 32usize);
        let fov = Fov::default();
        let basis = GameplayBasis::new(&cam, &fov, cols, rows);
        let vp = offslice::VProj::new(
            basis.cell_z,
            basis.horizon,
            basis.proj_plane_dist,
            basis.vshift_per_perp,
        );
        let (ray_dir_x, ray_dir_y) = basis.ray_dir(0);
        let registry = registry_for_map(&map);
        let primary_xy = primary_xy_for(&map, &registry, &basis, ray_dir_x, ray_dir_y);

        // The hologram sits on plane 2 at x=6; find the row whose ray meets that exact
        // (x, plane) — the RAW resolve must pass straight through it (it is
        // `Role::Air`), landing on whatever real geometry is behind (the corridor's
        // far wall), never on the hologram cell itself.
        let mut hologram_row = None;
        for row in 1..rows {
            let k = vp.slope(row as f32 + 0.5);
            // Height the ray reaches at the hologram's x: z = eye + k * (6 - pos_x).
            let s_at_hologram = (6.0 - basis.pos_x) / ray_dir_x;
            let z_at_hologram = vp.eye + k * s_at_hologram;
            if (z_at_hologram - 2.5).abs() < 0.5 {
                hologram_row = Some(row);
                break;
            }
        }
        let row = hologram_row.expect("expected a row whose ray crosses the hologram's plane");
        let k = vp.slope(row as f32 + 0.5);
        let raw = offslice::query_row(
            &map,
            basis.cell_z,
            basis.pos_x,
            basis.pos_y,
            ray_dir_x,
            ray_dir_y,
            vp.eye,
            k,
            256,
            |_| true,
        );
        if let Some(h) = raw {
            assert_ne!(
                map.cell_at(h.x, h.y, h.plane),
                CellKind::Hologram,
                "a hologram must never itself be returned as a hit"
            );
        }
        let audited = nearer_real_occluder(
            &map,
            &registry,
            basis.cell_z,
            basis.pos_x,
            basis.pos_y,
            ray_dir_x,
            ray_dir_y,
            vp.eye,
            k,
            256,
            primary_xy,
        );

        let mut fb = blank_fb(cols, rows);
        set_wall(&mut fb, 0, row - 1, 2.0);
        set_floor(&mut fb, 0, row, audited.unwrap_or(60.0) - 0.5);
        let violations = sweep_frame(&map, &cam, &fov, &fb);
        assert!(violations.is_empty(), "{violations:?}");
    }

    // ── FALSE POSITIVE: genuinely open sky (no roof at all — the grid simply ends)
    // above a low wall must resolve to NO occluder, and a Ceiling cell reading an
    // arbitrarily large depth there must not be flagged ──
    #[test]
    fn does_not_flag_open_sky_past_a_low_wall() {
        // A short, roofless silo: perimeter walls only up to z=3, nothing at all
        // above that anywhere in a much taller grid — genuinely open sky, not just a
        // tall room (see the module note: `stamp_geometry` always caps every room
        // with an authored `Ceiling` cell, which is itself real, solid geometry and
        // the wrong fixture for "sky").
        let (sx, sy, sz) = (8usize, 3usize, 40usize);
        let mut map = Map::new(sx, sy, sz);
        for z in 0..4 {
            for y in 0..sy {
                for x in 0..sx {
                    let on_perimeter = x == 0 || x == sx - 1 || y == 0 || y == sy - 1;
                    if on_perimeter {
                        map.set(x, y, z, CellKind::WallSolid);
                    } else if z == 0 {
                        map.set(x, y, z, CellKind::Floor);
                    }
                }
            }
        }
        // z >= 4 stays entirely CellKind::Empty (default) — open sky over the whole silo.

        let cam = CameraPose {
            pos: glam::Vec3::new(2.5, 1.5, STANDING_EYE_Z),
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.6,
        };
        let (cols, rows) = (1usize, 32usize);
        let fov = Fov::default();
        let basis = GameplayBasis::new(&cam, &fov, cols, rows);
        let vp = offslice::VProj::new(
            basis.cell_z,
            basis.horizon,
            basis.proj_plane_dist,
            basis.vshift_per_perp,
        );
        let (ray_dir_x, ray_dir_y) = basis.ray_dir(0);
        let registry = registry_for_map(&map);
        let primary_xy = primary_xy_for(&map, &registry, &basis, ray_dir_x, ray_dir_y);

        let row = 0usize; // the topmost screen row: the steepest look-up this frame has
        let k = vp.slope(row as f32 + 0.5);
        assert!(
            nearer_real_occluder(
                &map,
                &registry,
                basis.cell_z,
                basis.pos_x,
                basis.pos_y,
                ray_dir_x,
                ray_dir_y,
                vp.eye,
                k,
                256,
                primary_xy
            )
            .is_none(),
            "a steep look-up through a genuinely open roof must find no occluder"
        );

        let mut fb = blank_fb(cols, rows);
        set_wall(&mut fb, 0, row + 1, 2.0);
        set_ceiling(&mut fb, 0, row, f32::INFINITY);
        let violations = sweep_frame(&map, &cam, &fov, &fb);
        assert!(violations.is_empty(), "{violations:?}");
    }

    // ── Regression sanity: the ALREADY-FIXED primary case must read clean through
    // the REAL renderer (proves the counter does not re-flag what `clamp_to_wall_depth`
    // already fixed — no noise on ordinary geometry) ──
    #[test]
    fn the_fixed_primary_case_sweeps_clean_through_the_real_renderer() {
        // A REALISTIC room width (shipped levels run 8-20+ cells wide; every campaign
        // room is at least this wide). A narrow 3-wide corridor combined with a wide
        // FOV puts the side perimeter wall within ~1 world unit of the camera, which
        // trips an UNRELATED, already-known acute-near-wall approximation edge (the
        // SAME wall re-seen at a different z-plane at extreme grazing incidence) that
        // has nothing to do with this module's offslice/multi-storey target — widened to
        // keep this regression test testing the FAR-wall case the pinned unit test
        // (`floor_row_below_a_far_wall_is_owned_by_the_wall_not_a_farther_floor`)
        // actually covers.
        let (sx, sy, sz) = (20usize, 9usize, 3usize);
        let map = room(sx, sy, sz);
        let cam = straight_cam(2.5, 4.5);
        let fov = Fov::default();
        let (cols, rows) = (100usize, 32usize);
        let mut fb = Framebuffer::new(cols, rows);
        let registry = registry_for_map(&map);
        crate::raycast_with_time_fov_gameplay(&map, &cam, &registry, &fov, 0.0, &mut fb);

        let violations = sweep_frame(&map, &cam, &fov, &fb);
        assert!(
            violations.is_empty(),
            "the fixed primary case must not be re-flagged: {violations:?}"
        );
    }

    /// A ramp must not be misread as a flat "wall then farther floor" violation — a
    /// slope's own floor-adjoining rows are real, continuous geometry.
    #[test]
    fn a_ramp_does_not_spuriously_trip_the_counter() {
        // Widened for the same reason as `the_fixed_primary_case_...` above — a
        // realistic room width so the side walls stay well clear of a wide-FOV frame's
        // edge columns.
        let (sx, sy, sz) = (10usize, 9usize, 4usize);
        let mut map = room(sx, sy, sz);
        map.set_cell(5, 4, 1, CellKind::WallSolid, 0);
        map.set_shape(5, 4, 1, Shape::RampZ(re_core::RampDir::PosX));
        let cam = CameraPose {
            pos: glam::Vec3::new(2.5, 4.5, STANDING_EYE_Z),
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
        };
        let fov = Fov::default();
        let (cols, rows) = (60usize, 32usize);
        let mut fb = Framebuffer::new(cols, rows);
        let registry = registry_for_map(&map);
        crate::raycast_with_time_fov_gameplay(&map, &cam, &registry, &fov, 0.0, &mut fb);

        let violations = sweep_frame(&map, &cam, &fov, &fb);
        assert!(violations.is_empty(), "{violations:?}");
    }
}

#[cfg(test)]
mod corpus_sweep {
    use super::*;
    use crate::{composite::composite_dump_once, registry_for_map, Emitter, Fov};
    use re_level::load;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    const COLS: usize = 100;
    const ROWS: usize = 32;

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    /// REGRESSION (real level, root-cause fix): an image-pane level's transparent
    /// margins used to permanently block the vertical pass from ever finding the
    /// genuine off-slice wall behind them (`crate::image_cutout_column`'s `pane_band`
    /// handed back the pane's FULL geometric slice — "legacy contract" — instead of
    /// only the rows its own opaque texel actually painted), so ceiling cells read
    /// the classic backdrop's own (too-far) depth straight through a nearer real wall.
    /// `TX5-genplate.lvl` exercises the `recipe = image` sub-cell pane path on a
    /// freely-licensed, generated texture. Not `#[ignore]`d — this is the
    /// pass/fail regression for the fix, run every time.
    #[test]
    fn tex_asuka_sweeps_clean_after_the_image_cutout_band_fix() {
        let path = workspace_root().join("levels").join("TX5-genplate.lvl");
        let lv = load(&path).unwrap_or_else(|e| panic!("cannot load {}: {e}", path.display()));
        let map = lv.build_map();
        let cam: CameraPose = lv.spawn_camera();
        let fov = Fov::default();
        let registry = registry_for_map(&map);
        let emitters: Vec<Emitter> = lv.emitters.iter().map(Emitter::from_spec).collect();
        let mut fb = Framebuffer::new(COLS, ROWS);
        composite_dump_once(&map, &cam, &registry, &fov, &emitters, &mut fb);
        let violations = sweep_frame(&map, &cam, &fov, &fb);
        assert!(violations.is_empty(), "{violations:?}");
    }

    /// Report per-level counts over every shipped level at its own
    /// spawn pose (the same pose + grid the golden harness uses,
    /// `crates/re_game/tests/golden_framebuffer.rs`). `#[ignore]`d — this is a reporting
    /// tool run on demand, not a pass/fail gate (a positive count here is a FINDING to
    /// investigate, not a build break).
    ///
    /// Run: `cargo test -p re_render --lib -- --ignored dark_strip_offslice_sweep_all_levels --nocapture`
    #[test]
    #[ignore]
    fn dark_strip_offslice_sweep_all_levels() {
        let levels_dir = workspace_root().join("levels");
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&levels_dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", levels_dir.display()))
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "lvl"))
            .collect();
        entries.sort();

        let mut per_level: BTreeMap<String, usize> = BTreeMap::new();
        let mut unreadable: Vec<String> = Vec::new();
        let mut total = 0usize;
        let mut levels_with_any = 0usize;

        for path in &entries {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let lv = match load(path) {
                Ok(l) => l,
                Err(e) => {
                    unreadable.push(format!("{name}: {e}"));
                    continue;
                }
            };
            let map = lv.build_map();
            let cam: CameraPose = lv.spawn_camera();
            let fov = Fov::default();
            let registry = registry_for_map(&map);
            let emitters: Vec<Emitter> = lv.emitters.iter().map(Emitter::from_spec).collect();
            let mut fb = Framebuffer::new(COLS, ROWS);
            composite_dump_once(&map, &cam, &registry, &fov, &emitters, &mut fb);

            let violations = sweep_frame(&map, &cam, &fov, &fb);
            if !violations.is_empty() {
                levels_with_any += 1;
            }
            total += violations.len();
            per_level.insert(name, violations.len());
        }

        println!("=== dark-strip offslice audit: per-level cell counts ===");
        for (name, n) in &per_level {
            if *n > 0 {
                println!("{name}: {n}");
            }
        }
        if !unreadable.is_empty() {
            println!("--- UNREADABLE (not counted) ---");
            for u in &unreadable {
                println!("{u}");
            }
        }
        println!(
            "TOTAL: {total} cells across {levels_with_any}/{} levels ({} unreadable)",
            per_level.len(),
            unreadable.len()
        );
    }
}
