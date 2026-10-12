//! FACE SPAN — how many terminal COLUMNS one world cell of a wall face covers.
//!
//! The number a text-fitting sampler needs in order to refuse to magnify. A placard
//! panel is fitted across a unit face (`u ∈ [0,1]`); when the face covers more screen
//! columns than the panel is wide, a nearest-neighbour stretch REPLICATES source
//! characters — `the lattice` renders as `thhe latttice`, and the replication pattern
//! re-deals as the player walks (the stutter). [`TextPanel::sample_fitted_clamped`]
//! already knows how to draw the block at integer 1:1 scale instead; it just needs the
//! span. This module is the seam that delivers it.
//!
//! ## The derivation
//!
//! The caster builds each column's ray as `dir + plane · camera_x` with
//! `camera_x ∈ [−1, 1]` mapped across `cols` and `plane` scaled by
//! `plane_scale = tan(hfov/2)`. So a LATERAL world offset `lat` at perpendicular depth
//! `d` lands at `camera_x = (lat/d)/plane_scale`, i.e.
//!
//! ```text
//! d(col)/d(lat) = cols / (2 · plane_scale · d)
//! ```
//!
//! One world unit of face width therefore covers `cols_per_unit / d` columns face-on,
//! where `cols_per_unit = cols / (2 · plane_scale)` is a per-cast constant. Seen at an
//! angle the face's 1-unit width projects to `cos θ = 1/foreshorten` of that, so
//!
//! ```text
//! face_cols = cols_per_unit / (depth · foreshorten)
//! ```
//!
//! — exactly the two values `SampleCtx` already carries. `foreshorten` is the wall's
//! `1/|cos θ|`, so a grazing face correctly reports a NARROW span (no clamp, the LOD's
//! business) and a face-on wall right in front of the player reports a wide one.
//!
//! ## Why a thread-local (mirrors [`crate::draw_dist`] / [`crate::lod`])
//!
//! `cols_per_unit` is known at the top of a cast ([`crate::view_params`]) and needed
//! DEEP in the shade dispatch (inside one recipe branch of
//! `ascii_shader::Material::sample`), where a plain argument cannot reach without
//! churning `CellGeom` + `SampleCtx` + ~50 construction sites for one scalar. So it
//! follows the established pattern: written once per cast, read in the hot path, no env
//! read anywhere, stable within a cast ⇒ deterministic and golden-safe.
//!
//! ## Default == span unknown == today
//!
//! The cell defaults to `0.0`, and [`face_cols`] returns `0.0` for any non-finite /
//! non-positive input. `0.0` is precisely the "span unknown" value
//! `sample_fitted_clamped` reads as "use the plain stretch mapping", so an un-applied
//! path (a focused unit test, a caller that never calls `view_params`) is
//! BYTE-IDENTICAL to the pre-clamp renderer.
//!
//! [`TextPanel::sample_fitted_clamped`]: crate::code_wall::TextPanel::sample_fitted_clamped

use std::cell::Cell as StdCell;

thread_local! {
    /// Screen columns per world unit at UNIT depth for the current cast:
    /// `cols / (2 · tan(hfov/2))`. `0.0` = not applied ⇒ span unknown.
    static COLS_PER_UNIT: StdCell<f32> = const { StdCell::new(0.0) };
    /// Screen ROWS per world unit at UNIT depth: the cast's `proj_plane_dist`
    /// (`(rows/2)/tan(vfov/2)`) — a unit-tall wall at depth `d` is `proj/d` rows tall,
    /// which is exactly the row span of one cell face. `0.0` ⇒ span unknown.
    static ROWS_PER_UNIT: StdCell<f32> = const { StdCell::new(0.0) };
    /// The current cast's camera basis, for projecting a WORLD point back to its
    /// screen column ([`apply_camera`]). `cols == 0.0` ⇒ not published.
    static CAMERA: StdCell<CamBasis> = const { StdCell::new(CamBasis::UNKNOWN) };
    /// The face anchor of the wall slice being shaded ([`set_face_anchor`]).
    static ANCHOR: StdCell<FaceAnchor> = const { StdCell::new(FaceAnchor::UNKNOWN) };
    /// The number of screen rows the face being shaded is DRAWN over in this column
    /// ([`set_drawn_rows`]). `0.0` = not published.
    static DRAWN_ROWS: StdCell<f32> = const { StdCell::new(0.0) };
    /// The wall-RUN coordinate of the slice being shaded ([`set_wall_run`]).
    static WALL_RUN: StdCell<Option<WallRun>> = const { StdCell::new(None) };
}

/// What a text surface needs to lay ONE continuous stream along a wall run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WallRun {
    /// The continuous, world-anchored coordinate along the wall plane's variable axis,
    /// signed so it increases with the face's `u` (see [`set_wall_run`]).
    pub u: f32,
    /// The EXACT local screen footprint: how many screen columns one world unit of this
    /// wall covers AT THIS CELL.
    ///
    /// Deliberately not [`face_cols`]. That returns
    /// `cols_per_unit · |cos(ray, face normal)| / depth`, which is the TEXTURE stretch
    /// — it is missing the fisheye factor `|ray| / depth` that relates a ray's own
    /// angle to its screen column. The two cancel exactly on a face-on wall (where the
    /// true footprint is a constant `cols_per_unit/depth` from screen edge to screen
    /// edge), so leaving the fisheye out overstates the footprint toward the edges of
    /// the screen and lets two adjacent columns both claim one source character —
    /// measured as `ssyynntt` at the frame edges with a clean middle.
    pub footprint_cols: f32,
}

/// Publish the WALL-RUN coordinate of the slice being shaded: the world position along
/// the wall plane's variable axis, SIGNED so it increases with the face's `u`.
///
/// ## Why the shader cannot derive this itself
///
/// A wall recipe is handed `coord.x = tex_u + <the face's CONSTANT axis>`
/// (`crate::sample_coord`), which de-duplicates procedural texture between parallel
/// wall planes — exactly right for brick. But `tex_u` WRAPS every world cell, so a
/// recipe that wants to flow ONE continuous text stream along a wall RUN sees the same
/// `[n, n+1)` interval on every cell of that run and draws the same characters over and
/// over. (That is what the code wall did: eight source columns, repeated on every cell
/// of every wall.)
///
/// The run coordinate is the raw world coordinate on the face's variable axis, negated
/// on the faces where the caster MIRRORS `tex_u` (so text is never drawn backwards).
/// Negating preserves `fract(run_u) == tex_u`, so it is the same coordinate the rest of
/// the wall path uses — just un-wrapped and continuous across cells.
///
/// View-INDEPENDENT by construction (pure world geometry), so every cast path — the
/// viewport cast, the near-ring bake, a `--dump` — publishes the same value for the
/// same world point and their text agrees.
#[inline]
pub fn set_wall_run(run: Option<WallRun>) {
    WALL_RUN.with(|c| {
        c.set(
            run.filter(|r| {
                r.u.is_finite() && r.footprint_cols.is_finite() && r.footprint_cols > 0.0
            }),
        )
    });
}

/// The wall run of the cell being shaded (`None` when the caster published none — a
/// focused unit test, or a non-wall surface).
#[inline]
pub fn wall_run() -> Option<WallRun> {
    WALL_RUN.with(|c| c.get())
}

/// The wall-run coordinate for an axis-aligned face hit, matching the caster's own
/// `tex_u` mirroring (`dda_hit`): an x-face is mirrored when the ray travels `+x`, a
/// y-face when it travels `−y`.
#[inline]
pub fn wall_run_for(side: u8, hit_x: f32, hit_y: f32, ray_dir_x: f32, ray_dir_y: f32) -> f32 {
    if side == 1 {
        if ray_dir_x > 0.0 {
            -hit_y
        } else {
            hit_y
        }
    } else if ray_dir_y < 0.0 {
        -hit_x
    } else {
        hit_x
    }
}

// ───────────────────────── the SCREEN-ANCHORED face span ────────────────────

/// Where the screen cell being shaded sits along its wall face, measured in SCREEN
/// COLUMNS — the exact replacement for the `u · face_cols` estimate.
///
/// ## Why `u · face_cols` is wrong on an OBLIQUE face
///
/// [`face_cols`] evaluates the projection at the shaded cell's OWN perpendicular
/// depth. Face-on that depth is constant across the face and `u · face_cols` is the
/// screen offset from the face's left edge — which is why the clamp measured clean at
/// 90°. Turn the face even 2° off and the depth VARIES across it, so `u · face_cols`
/// is a product of two quantities that both drift; the drift reaches half a cell
/// within a few columns and the `floor` then hands two adjacent screen columns the
/// SAME source index (a doubled letter) and skips the next (a dropped one) — the very
/// artifact the clamp exists to remove, re-dealt at every yaw.
///
/// ## What this is instead
///
/// The caster PROJECTS the face's two world endpoints (`u = 0` and `u = 1`) through
/// the same camera basis it builds its rays from, giving the face's exact screen
/// extent. `u_cols` is then simply `(this column's centre) − (the u = 0 edge)`, so it
/// advances by EXACTLY ONE per screen column — by construction, at any yaw, on any
/// face, with no dependence on the local depth at all. No duplicated column, no
/// dropped column, ever.
///
/// `span_cols <= 0.0` means "unknown" and keeps every historical mapping (see
/// [`FaceAnchor::UNKNOWN`]).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct FaceAnchor {
    /// Screen columns from the face's projected `u = 0` edge to this cell's centre.
    pub u_cols: f32,
    /// Screen columns the whole face (`u ∈ [0,1]`) covers. `<= 0` ⇒ unknown.
    pub span_cols: f32,
}

impl FaceAnchor {
    /// "No measured projection" — every consumer falls back to its historical
    /// stretch/estimate mapping, so an un-anchored cast is byte-identical to before.
    pub const UNKNOWN: FaceAnchor = FaceAnchor {
        u_cols: 0.0,
        span_cols: 0.0,
    };

    /// The LEGACY estimate, kept so callers that cannot measure the projection (a
    /// focused unit test, [`crate::code_wall::TextPanel::sample_fitted`]) reproduce
    /// exactly the pre-anchor arithmetic.
    #[inline]
    pub fn estimated(u: f32, face_cols: f32) -> FaceAnchor {
        FaceAnchor {
            u_cols: u * face_cols,
            span_cols: face_cols,
        }
    }

    /// Whether a real projection was measured.
    #[inline]
    pub fn is_known(&self) -> bool {
        self.span_cols.is_finite() && self.span_cols > 0.0 && self.u_cols.is_finite()
    }
}

/// The camera basis of the active cast, in the caster's OWN units: a world point `p`
/// projects to continuous column coordinate `(lat/depth + 1)/2 · cols`, where column
/// `x`'s CENTRE is `x + 0.5` (matching `raycast_full_memo`'s
/// `camera_x = 2·(x+0.5)/cols − 1`).
#[derive(Clone, Copy, Debug)]
struct CamBasis {
    pos_x: f32,
    pos_y: f32,
    dir_x: f32,
    dir_y: f32,
    plane_x: f32,
    plane_y: f32,
    /// Viewport width in columns. `0.0` ⇒ no basis published.
    cols: f32,
}

impl CamBasis {
    const UNKNOWN: CamBasis = CamBasis {
        pos_x: 0.0,
        pos_y: 0.0,
        dir_x: 0.0,
        dir_y: 0.0,
        plane_x: 0.0,
        plane_y: 0.0,
        cols: 0.0,
    };
}

/// Minimum perpendicular depth for a projected point to count. A face endpoint at or
/// behind the camera plane has no finite screen column, so the anchor degrades to
/// [`FaceAnchor::UNKNOWN`] rather than producing a garbage index.
const PROJ_MIN_DEPTH: f32 = 1.0e-3;

/// Minimum |determinant| of the camera basis before it is treated as degenerate.
const PROJ_MIN_DET: f32 = 1.0e-6;

/// Minimum projected face width (columns) worth anchoring. Below this the face is
/// edge-on, the clamp would not engage anyway, and the division is ill-conditioned.
const PROJ_MIN_SPAN: f32 = 1.0e-3;

/// How far (in screen columns) the shaded hit point may project from its own column
/// centre before the anchor is refused.
///
/// This is the RE-BASED-VIEW GUARD. A mirror reflection, a warp-portal hop, and an
/// image/ramp backdrop recast all shade geometry that does NOT live where the true
/// camera would see it, so projecting its world endpoints with the true camera basis
/// would anchor the text to the wrong columns. Rather than plumb a "is this view
/// re-based" flag through six call sites, the anchor VERIFIES itself: the hit point of
/// an honest primary ray projects back to the column that cast it (to float precision).
/// Anything else falls back to the historical mapping.
const ANCHOR_SELF_CHECK_COLS: f32 = 0.5;

/// Install the current cast's projection scales: `cols/(2·plane_scale)` columns and
/// `proj_plane_dist` rows per world unit at unit depth.
///
/// Called from [`crate::view_params`], so every cast entry — near, far ring, sub-cell
/// refine, `--dump` — publishes them with no extra plumbing. A non-finite or negative
/// value is stored as `0.0` (span unknown) so a poisoned view can never enable the
/// clamp on garbage.
#[inline]
pub fn apply(cols_per_unit: f32, rows_per_unit: f32) {
    let sane = |v: f32| if v.is_finite() && v > 0.0 { v } else { 0.0 };
    COLS_PER_UNIT.with(|c| c.set(sane(cols_per_unit)));
    ROWS_PER_UNIT.with(|c| c.set(sane(rows_per_unit)));
}

/// Publish the current cast's CAMERA BASIS so a wall face's world endpoints can be
/// projected back to exact screen columns ([`face_anchor_for`]).
///
/// Called from the viewport cast entries (`raycast_full_memo`,
/// `recast_special_columns_into`) — the ones whose `col` index IS a terminal column.
/// The near-ring bake casts one column at a time in its own angular metric, so it
/// deliberately does NOT publish: its cells keep the historical mapping.
///
/// A non-finite input, a degenerate basis, or `cols == 0` publishes "unknown".
#[inline]
pub fn apply_camera(
    pos_x: f32,
    pos_y: f32,
    dir_x: f32,
    dir_y: f32,
    plane_x: f32,
    plane_y: f32,
    cols: usize,
) {
    let finite = [pos_x, pos_y, dir_x, dir_y, plane_x, plane_y]
        .iter()
        .all(|v| v.is_finite());
    let det = dir_x * plane_y - plane_x * dir_y;
    let basis = if finite && cols > 0 && det.is_finite() && det.abs() > PROJ_MIN_DET {
        CamBasis {
            pos_x,
            pos_y,
            dir_x,
            dir_y,
            plane_x,
            plane_y,
            cols: cols as f32,
        }
    } else {
        CamBasis::UNKNOWN
    };
    CAMERA.with(|c| c.set(basis));
    ANCHOR.with(|a| a.set(FaceAnchor::UNKNOWN));
}

/// Whether a VIEWPORT camera basis is published — i.e. whether the `col` index this
/// cast is walking really is a terminal column.
///
/// The code wall's "one screen column per source character" rule needs this: the
/// near-ring bake casts in its own finer ANGULAR metric and is then RESAMPLED into
/// viewport columns, so a per-column decision taken during the bake would be decimated
/// by that resample. Ring-baked cells therefore keep the (world-anchored, and so
/// resample-safe) flow without the per-column de-duplication.
#[inline]
pub fn has_viewport_basis() -> bool {
    CAMERA.with(|c| c.get().cols > 0.0)
}

/// Forget the published camera basis (and any live face anchor).
#[inline]
pub fn clear_camera() {
    CAMERA.with(|c| c.set(CamBasis::UNKNOWN));
    ANCHOR.with(|a| a.set(FaceAnchor::UNKNOWN));
}

/// Continuous screen-column coordinate of world point `(wx, wy)` — column `x`'s
/// centre is `x + 0.5`. `None` when no basis is published or the point is at/behind
/// the camera plane.
#[inline]
fn project_col(b: &CamBasis, wx: f32, wy: f32) -> Option<f32> {
    if b.cols <= 0.0 {
        return None;
    }
    let (rx, ry) = (wx - b.pos_x, wy - b.pos_y);
    let det = b.dir_x * b.plane_y - b.plane_x * b.dir_y;
    // r = depth·dir + lat·plane (Cramer on the 2×2 camera basis).
    let depth = (rx * b.plane_y - b.plane_x * ry) / det;
    let lat = (b.dir_x * ry - rx * b.dir_y) / det;
    if !depth.is_finite() || depth <= PROJ_MIN_DEPTH || !lat.is_finite() {
        return None;
    }
    let col = (lat / depth + 1.0) * 0.5 * b.cols;
    col.is_finite().then_some(col)
}

/// Measure the [`FaceAnchor`] of an axis-aligned wall face for the column that just
/// hit it.
///
/// * `col` — the screen column being cast.
/// * `side` — `1` = x-face (variable axis is world Y), `0` = y-face (variable axis X).
/// * `cell_x`/`cell_y` — the hit cell's grid coordinates (the face spans one unit of
///   the variable axis, from the cell index to cell index + 1).
/// * `hit_x`/`hit_y` — the exact world hit point (fixes the face's constant axis and
///   drives the re-based-view self check).
/// * `ray_dir_x`/`ray_dir_y` — the casting ray, which selects WHICH endpoint is
///   `tex_u == 0` (the caster mirrors `tex_u` on opposing faces so text is never
///   drawn backwards; this reproduces exactly that flip).
///
/// Returns [`FaceAnchor::UNKNOWN`] for any view the projection cannot vouch for.
/// Pure; ~30 flops, and only on a wall column.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn face_anchor_for(
    col: usize,
    side: u8,
    cell_x: i32,
    cell_y: i32,
    hit_x: f32,
    hit_y: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
) -> FaceAnchor {
    let b = CAMERA.with(|c| c.get());
    if b.cols <= 0.0 {
        return FaceAnchor::UNKNOWN;
    }
    let centre = col as f32 + 0.5;
    // RE-BASED-VIEW GUARD: an honest primary hit projects back to its own column.
    match project_col(&b, hit_x, hit_y) {
        Some(c) if (c - centre).abs() <= ANCHOR_SELF_CHECK_COLS => {}
        _ => return FaceAnchor::UNKNOWN,
    }
    // The face's two world endpoints, ordered so `p0` is `tex_u == 0`. The flip
    // mirrors `dda_hit`: an x-face is flipped when the ray travels +x, a y-face when
    // it travels −y.
    let (p0, p1) = if side == 1 {
        let lo = (hit_x, cell_y as f32);
        let hi = (hit_x, cell_y as f32 + 1.0);
        if ray_dir_x > 0.0 {
            (hi, lo)
        } else {
            (lo, hi)
        }
    } else {
        let lo = (cell_x as f32, hit_y);
        let hi = (cell_x as f32 + 1.0, hit_y);
        if ray_dir_y < 0.0 {
            (hi, lo)
        } else {
            (lo, hi)
        }
    };
    let (Some(c0), Some(c1)) = (project_col(&b, p0.0, p0.1), project_col(&b, p1.0, p1.1)) else {
        return FaceAnchor::UNKNOWN;
    };
    let signed = c1 - c0;
    if !signed.is_finite() || signed.abs() < PROJ_MIN_SPAN {
        return FaceAnchor::UNKNOWN;
    }
    FaceAnchor {
        // Signed so `u_cols` grows with `u` whichever way the face runs on screen.
        u_cols: (centre - c0) * signed.signum(),
        span_cols: signed.abs(),
    }
}

/// Publish the face anchor for the wall slice about to be shaded. Scoped to the
/// slice's row loop; the caster clears it afterwards so no other surface can read a
/// stale value.
#[inline]
pub fn set_face_anchor(anchor: FaceAnchor) {
    ANCHOR.with(|a| a.set(anchor));
}

/// Publish how many screen rows the vertical face being shaded is drawn over in this
/// column (`None` clears). The caster maps its drawn rows `[first, first + n)` onto
/// `wall_v = (row − first) / n` (so a face's texture does not creep as its
/// sub-row extent drifts with depth), and a text sampler reconstructing the row index
/// from `wall_v` must multiply by that same `n` — see [`face_rows`].
#[inline]
pub fn set_drawn_rows(n: Option<usize>) {
    DRAWN_ROWS.with(|c| c.set(n.filter(|&n| n > 0).map_or(0.0, |n| n as f32)));
}

/// The face anchor of the cell being shaded ([`FaceAnchor::UNKNOWN`] when none).
#[inline]
pub fn face_anchor() -> FaceAnchor {
    ANCHOR.with(|a| a.get())
}

/// The active `cols_per_unit` (0.0 ⇒ unknown). Exposed for tests / diagnostics.
#[inline]
pub fn active_cols_per_unit() -> f32 {
    COLS_PER_UNIT.with(|c| c.get())
}

/// The active `rows_per_unit` (0.0 ⇒ unknown). Exposed for tests / diagnostics.
#[inline]
pub fn active_rows_per_unit() -> f32 {
    ROWS_PER_UNIT.with(|c| c.get())
}

/// Screen ROWS covered by ONE world cell of a wall face at `depth` — the wall slice's
/// own drawn height, so no view-angle term enters. `0.0` when the cast published no
/// scale or `depth` is degenerate.
///
/// When the caster published the face's DRAWN row count ([`set_drawn_rows`]) this is
/// that count: the caster hands the shader `wall_v = (row − first) / n`, so
/// `wall_v · face_rows` steps by exactly one per screen row — what a text sampler
/// reconstructing the row index needs. Otherwise it is the continuous projected span
/// `proj / depth` (every vertical surface projects world `[k, k+1]` over exactly that
/// many rows, `crate::offslice::VProj`).
#[inline]
pub fn face_rows(depth: f32) -> f32 {
    let drawn = DRAWN_ROWS.with(|c| c.get());
    if drawn > 0.0 {
        return drawn;
    }
    let scale = active_rows_per_unit();
    if scale <= 0.0 || !depth.is_finite() || depth <= 0.0 {
        return 0.0;
    }
    scale / depth
}

/// Screen columns covered by ONE world cell of a wall face at `depth` with the given
/// `foreshorten` (≥ 1). Returns `0.0` — "span unknown", the value that keeps a fitted
/// sampler on its plain stretch mapping — when the cast published no scale or the
/// inputs are degenerate.
#[inline]
pub fn face_cols(depth: f32, foreshorten: f32) -> f32 {
    let scale = active_cols_per_unit();
    if scale <= 0.0 || !depth.is_finite() || depth <= 0.0 {
        return 0.0;
    }
    let f = if foreshorten.is_finite() && foreshorten >= 1.0 {
        foreshorten
    } else {
        1.0
    };
    scale / (depth * f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Un-applied (or explicitly cleared) ⇒ span unknown ⇒ the fitted sampler keeps
    /// its historical stretch mapping.
    #[test]
    fn unapplied_reports_unknown_span() {
        apply(0.0, 0.0);
        assert_eq!(active_cols_per_unit(), 0.0);
        assert_eq!(active_rows_per_unit(), 0.0);
        assert_eq!(face_cols(2.0, 1.0), 0.0);
        assert_eq!(face_rows(2.0), 0.0);
    }

    /// The span shrinks with depth and with grazing angle — the two properties the
    /// clamp depends on (near+face-on magnifies; far or grazing does not).
    #[test]
    fn span_falls_off_with_depth_and_grazing() {
        apply(50.0, 30.0); // e.g. 100 cols at hfov 90°, proj_plane_dist 30
        assert!((face_cols(1.0, 1.0) - 50.0).abs() < 1e-4);
        assert!((face_cols(5.0, 1.0) - 10.0).abs() < 1e-4);
        // Grazing (foreshorten 2 ⇒ cos θ = 0.5) halves the projected width.
        assert!((face_cols(2.0, 2.0) - 12.5).abs() < 1e-4);
        // Degenerate inputs never poison the answer.
        assert_eq!(face_cols(f32::NAN, 1.0), 0.0);
        assert_eq!(face_cols(0.0, 1.0), 0.0);
        assert!((face_cols(2.0, f32::NAN) - 25.0).abs() < 1e-4);
        // Rows: the caster's continuous span, `proj / depth`.
        assert!((face_rows(2.0) - 15.0).abs() < 1e-4);
        assert!((face_rows(3.0) - 10.0).abs() < 1e-4);
        assert_eq!(face_rows(-1.0), 0.0);
        apply(0.0, 0.0);
    }

    /// `view_params` publishes the scale, and it matches the closed form.
    #[test]
    fn view_params_publishes_the_scale() {
        apply(0.0, 0.0);
        let view = crate::view_params(100, 32, &crate::Fov::default());
        let expect = 100.0 / (2.0 * view.plane_scale);
        assert!(
            (active_cols_per_unit() - expect).abs() < 1e-3,
            "view_params must publish cols/(2·plane_scale): got {}, want {expect}",
            active_cols_per_unit()
        );
        assert!(
            (active_rows_per_unit() - view.proj_plane_dist).abs() < 1e-3,
            "view_params must publish proj_plane_dist as rows_per_unit"
        );
        apply(0.0, 0.0);
    }
}
