//! `re_render::portal` — the non-euclidean WARP-PORTAL sub-system: the
//! render-side re-basing that makes a portal cell show the far region through
//! the opening instead of a wall.
//!
//! A [`re_world::CellKind::Portal`] cell is a solid the primary DDA STOPS on, but
//! instead of shading it as a wall the cast RE-BASES the ray by the portal's rigid
//! `delta` transform (new origin + direction) and continues the SAME grid-walk into
//! the destination region — so the slice shows the FAR region through the opening,
//! not a wall. This is the exact one-bounce structure [`crate::mirror`] uses for
//! reflections, with a general rigid transform substituted for the axis-aligned
//! mirror flip.
//!
//! ## The two correctness points `mirror.rs` does NOT get for free
//!
//! 1. **Accumulated depth.** The destination hit's depth/fog/LOD must
//!    read the TOTAL path length `dist_to_portal + second_segment_perp`, NOT the
//!    second segment alone — else the far side starts at depth 0 and shows a
//!    contrast seam at the opening. The cast threads the portal distance into the
//!    destination `RayHit`'s `perp` (the mirror path already does this for the
//!    reflected bounce; the portal does the same).
//! 2. **One warp only (MVP bound).** A `warp_budget` bounds re-basing to ONE hop:
//!    a portal seen through a portal shades as a plain wall ([`crate::surface_material`]
//!    `SurfaceKind::Portal` fallback), never re-warps. Bumping the budget to N later
//!    is a constant change, not a rewrite (the structure is graph-ready).
//!
//! Pure compute: no terminal, no allocation, deterministic (the re-based ray is a
//! fixed function of the hit point + `delta`). The geometry primitives (`dda_hit`)
//! and the `PortalView` construction stay in the cast where they belong, exactly as
//! the mirror module is structured.

use crate::RayHit;
use re_core::SurfaceKind;
use re_world::Map;

/// How far (world units, along the re-based ray) the warped DDA origin is nudged off
/// the portal's destination face, so the second walk does not immediately re-hit the
/// destination portal cell. The direct analogue of [`crate::mirror::MIRROR_ORIGIN_NUDGE`]
/// for the warp ray.
pub(crate) const PORTAL_ORIGIN_NUDGE: f32 = 1e-3;

/// Maximum number of warps a single primary ray may take. MVP = 1 (one hop): a
/// portal reached THROUGH a portal is shaded as a plain wall, never re-warped —
/// identical to the mirror "no re-bounce" rule and bounding cost to `+1 dda_hit`
/// per portal column. Threaded as a parameter so a later graph
/// model can raise it without touching the re-basing structure.
pub(crate) const PORTAL_WARP_BUDGET: u8 = 1;

/// If the primary ray hit a PORTAL cell, return the index into [`Map::portals`] for
/// the linked-pair entry (the sibling of [`crate::mirror::surface_is_mirror`]).
///
/// A portal hit is resolved from the integer hit cell ([`RayHit::cell_x`] /
/// `cell_y` + the cast plane `cell_z`) via the world's O(1) sparse lookup
/// [`Map::portal_for_cell`] — no grid re-walk, no per-cell allocation. Returns
/// `None` for any non-portal surface (the common path), so the warp branch is a
/// post-hit no-op on every ordinary column.
#[inline]
pub(crate) fn surface_is_portal(map: &Map, hit: &RayHit, cell_z: i32) -> Option<usize> {
    if !hit.hit || hit.surface != SurfaceKind::Portal {
        return None;
    }
    map.portal_for_cell(hit.cell_x, hit.cell_y, cell_z)
}

/// The re-based secondary ray for a warp: the destination-region origin (the portal
/// hit point mapped by `delta`, nudged off the destination face) and the
/// destination-region direction (the ray direction rotated by `delta`).
///
/// `delta` is a rigid transform (rotation + translation): a POINT is transformed by
/// the full affine map, a DIRECTION by the rotation only (`transform_vector3`). The
/// returned direction is renormalised so the second `dda_hit`'s perpendicular
/// distance stays in world units (the same scale as the primary segment), which is
/// what the accumulated-depth sum requires.
///
/// **MVP scope — vertical portals only.** `delta` must keep z invariant (a yaw +
/// horizontal translation; `euler.x == euler.z == 0`). The 2D DDA reads
/// only the transformed x/y and discards the transformed z, which is exact precisely
/// because z does not move; a pitched/rolled delta (z mixed into the in-plane image)
/// is out of MVP scope and tripped by the `debug_assert` below. So the cell-centre z
/// is fed as a stable representative — it does NOT make a non-vertical delta correct.
///
/// Returns `(origin', dir')` ready to feed a second `dda_hit` from the same plane.
/// Pure float maths, deterministic, no allocation.
#[inline]
pub(crate) fn rebase_ray(
    delta: &glam::Mat4,
    hit_x: f32,
    hit_y: f32,
    cell_z: i32,
    ray_dir_x: f32,
    ray_dir_y: f32,
) -> (f32, f32, f32, f32) {
    // MVP scope: VERTICAL portals only — `delta` is a yaw + translation in the
    // horizontal plane, with z DELTA-INVARIANT (no pitch/roll, no z scale;
    // `euler.x == euler.z == 0`). The 2D DDA marches on the cast plane and reads only
    // x/y of the transformed result, DISCARDING the transformed z. That is correct
    // *because* z is invariant under the MVP delta: the in-plane (x,y) image of the
    // hit point is independent of which z we feed, so the cast plane's cell centre is
    // a safe representative. (A delta that mixed z into x/y — a pitched portal — would
    // need the true hit z and a 3D march; out of MVP scope.) The debug_assert below
    // guards the invariant in debug builds: the delta's z basis column must map
    // z → z (identity-like z row), so feeding the cell-centre z is exact.
    debug_assert!(
        delta.x_axis.z.abs() < 1e-3
            && delta.y_axis.z.abs() < 1e-3
            && (delta.z_axis.z - 1.0).abs() < 1e-3
            && delta.z_axis.x.abs() < 1e-3
            && delta.z_axis.y.abs() < 1e-3,
        "rebase_ray is MVP vertical-portal only: delta's z must be invariant \
         (identity-like z row); got z_axis={:?}, x_axis.z={}, y_axis.z={}",
        delta.z_axis,
        delta.x_axis.z,
        delta.y_axis.z
    );
    let plane_z = cell_z as f32 + 0.5;
    let hit_pt = glam::Vec3::new(hit_x, hit_y, plane_z);
    let dir = glam::Vec3::new(ray_dir_x, ray_dir_y, 0.0);

    let mut o = delta.transform_point3(hit_pt);
    let d = delta.transform_vector3(dir);

    // Renormalise the (rotation-only) direction so the second segment's perp is in
    // the same world units as the first — required for the accumulated-depth sum to
    // be a true total path length (a non-unit dir would scale the second perp).
    let len = (d.x * d.x + d.y * d.y).sqrt().max(crate::PERP_MIN);
    let dx = d.x / len;
    let dy = d.y / len;

    // Nudge the destination origin a hair along the re-based direction so the second
    // walk does not immediately re-hit the destination portal cell (analogue of the
    // mirror origin nudge).
    o.x += dx * PORTAL_ORIGIN_NUDGE;
    o.y += dy * PORTAL_ORIGIN_NUDGE;

    (o.x, o.y, dx, dy)
}

/// Resolve a single warp hop into the `RayHit` the column shades from, given the
/// PRIMARY hit (the portal cell the ray stopped on) and the DESTINATION hit (the
/// re-based ray's first `dda_hit` in the destination region). This is the
/// load-bearing one-warp-bound + accumulated-depth decision, factored out of the
/// cast so it can be proven by a constructed fixture (synthetic hits) without
/// needing a registered portal pair in the world grid.
///
/// Three cases:
///
/// 1. **Destination MISS** — the warped ray escaped the finite destination grid:
///    it is looking into OPEN SPACE on the far side, so the column reads as
///    SKY/void exactly like an ordinary primary-ray miss. The un-modified `dest`
///    (a [`RayHit::miss`]: `hit=false`, `perp=DIST_INFINITY`, `surface=Empty`) is
///    returned verbatim — NO flat Portal brick punched to the portal's own depth
///    (that produced the reported depth-island seam).
/// 2. **Destination is itself a PORTAL** — a portal reached THROUGH a portal. The
///    MVP one-warp budget is already spent, so it is NOT re-warped: it shades as
///    the plain-wall fallback ([`SurfaceKind::Portal`] → the brick material) at the
///    ACCUMULATED one-hop depth (`dist_to_portal + dest.perp`). This is the
///    safety property that bounds cost to `+1 dda_hit` per portal column.
/// 3. **Destination is ordinary geometry** — shaded normally at the accumulated
///    depth so the far side recedes behind the opening (no depth-0 contrast seam).
///
/// `dist_to_portal` is the primary segment's perpendicular distance (the distance
/// the ray already travelled to reach the portal plane). Pure float maths,
/// deterministic, no allocation.
#[inline]
pub(crate) fn resolve_warp(dist_to_portal: f32, dest: RayHit) -> RayHit {
    if !dest.hit {
        // Case 1: warp into open space — a genuine SKY/void miss, not a brick.
        return dest;
    }
    // Cases 2 & 3 share the accumulated depth. A destination that is ITSELF a
    // portal keeps `surface == Portal`, so the ordinary wall path shades it as the
    // fallback brick (the budget is spent — no second warp). Ordinary geometry
    // shades from its own surface/material. Either way the slice sits at the total
    // one-hop path length.
    RayHit {
        perp: (dist_to_portal + dest.perp).max(crate::PERP_MIN),
        ..dest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `rebase_ray` with an IDENTITY delta is the identity (up to the tiny nudge and
    /// the renormalisation of the direction): the destination origin equals the hit
    /// point + nudge, and the direction is the input direction normalised.
    #[test]
    fn rebase_identity_is_identity_plus_nudge() {
        let delta = glam::Mat4::IDENTITY;
        let (ox, oy, dx, dy) = rebase_ray(&delta, 3.0, 4.0, 1, 0.6, 0.8);
        // Input dir (0.6, 0.8) is already unit, so direction is unchanged.
        assert!((dx - 0.6).abs() < 1e-5, "dx={dx}");
        assert!((dy - 0.8).abs() < 1e-5, "dy={dy}");
        // Origin = hit point nudged a hair along the (unit) direction.
        assert!(
            (ox - (3.0 + 0.6 * PORTAL_ORIGIN_NUDGE)).abs() < 1e-5,
            "ox={ox}"
        );
        assert!(
            (oy - (4.0 + 0.8 * PORTAL_ORIGIN_NUDGE)).abs() < 1e-5,
            "oy={oy}"
        );
    }

    /// A pure TRANSLATION delta shifts the origin but leaves the direction unchanged
    /// (the rotation part is identity → `transform_vector3` is the identity).
    #[test]
    fn rebase_translation_shifts_origin_only() {
        let delta = glam::Mat4::from_translation(glam::Vec3::new(10.0, -2.0, 0.0));
        let (ox, oy, dx, dy) = rebase_ray(&delta, 1.0, 1.0, 0, 1.0, 0.0);
        // Direction unchanged (already unit).
        assert!(
            (dx - 1.0).abs() < 1e-5 && dy.abs() < 1e-5,
            "dir=({dx},{dy})"
        );
        // Origin translated by +10/-2, then nudged +PORTAL_ORIGIN_NUDGE along +x.
        assert!((ox - (11.0 + PORTAL_ORIGIN_NUDGE)).abs() < 1e-5, "ox={ox}");
        assert!((oy - (-1.0)).abs() < 1e-5, "oy={oy}");
    }

    /// A 90°-about-Z rotation delta rotates the direction: +x → +y. The destination
    /// direction stays unit (renormalised).
    #[test]
    fn rebase_rotation_rotates_direction() {
        let delta = glam::Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let (_ox, _oy, dx, dy) = rebase_ray(&delta, 0.0, 0.0, 0, 1.0, 0.0);
        // +x rotated +90° about z → +y.
        assert!(dx.abs() < 1e-5, "dx={dx}");
        assert!((dy - 1.0).abs() < 1e-5, "dy={dy}");
        // Still unit length.
        let len = (dx * dx + dy * dy).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "len={len}");
    }

    /// The re-based direction is ALWAYS renormalised, so a non-unit input direction
    /// (the raw camera-plane ray the cast feeds) produces a UNIT second-segment dir —
    /// required for the accumulated-depth sum to be a true total path length.
    #[test]
    fn rebase_renormalises_nonunit_input() {
        let delta = glam::Mat4::IDENTITY;
        // A deliberately non-unit input direction (length 5).
        let (_ox, _oy, dx, dy) = rebase_ray(&delta, 0.0, 0.0, 0, 3.0, 4.0);
        let len = (dx * dx + dy * dy).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-5,
            "second-segment dir must be unit, len={len}"
        );
        // Direction preserved (3,4)/5 = (0.6,0.8).
        assert!(
            (dx - 0.6).abs() < 1e-5 && (dy - 0.8).abs() < 1e-5,
            "dir=({dx},{dy})"
        );
    }
}
