//! Shared WORLD/SURFACE anchor for post-process patterns that must travel WITH
//! the scene rather than staying glued to the screen.
//!
//! `film_grain` and `dither` were both keyed on screen `(col, row)` — the
//! "SCREEN" anchor that is appropriate only for lens-only effects (vignette,
//! CRT scanlines). On a low-contrast surface (typically a dark floor) that
//! screen-fixed pattern reads as a texture painted on the INSIDE of the
//! terminal window: turning the camera sweeps the surface's edges across the
//! screen while the grain/dither pattern stays put on the same columns/rows.
//! This module gives both effects the "WORLD / SURFACE" anchor instead: a
//! texel bucket that is a pure function of the cell's own world hit (floors)
//! or wall texture coordinate (walls), so the pattern pans WITH the surface
//! instead of re-rolling over the screen grid.
//!
//! Sky gets none at all — there is no surface for the pattern to anchor to,
//! so sky and screen-only cells get none.

use re_core::{Cell, SurfaceKind};

/// World-texel density for [`Floor`](SurfaceKind::Floor) cells: how many
/// grain/dither buckets per world unit of `Cell::hit`. Provisional — wiring in
/// `re_render::lod::Footprint{f_u,f_v,rho}` would let this vary with the
/// projected texel size so near and far cells get a correctly-scaled pattern.
/// Until then a fixed density is a flat, reasonable middle ground: fine enough
/// to still read as grain, coarse enough not to alias into per-cell noise.
pub(crate) const FLOOR_BUCKET_DENSITY: f32 = 5.0;

/// World-texel density for wall-like cells: `tex_u`/`tex_v` are each already
/// normalized to `[0,1]` across one face, so this multiplies up to a workable
/// bucket count per face. Higher than the floor density because a wall face
/// usually spans fewer world units than the floor a camera can see.
pub(crate) const WALL_BUCKET_DENSITY: f32 = 18.0;

/// `true` for the cells the spec says must carry NO grain/dither at all. Sky
/// (open above the horizon) and the ambient-shaded interior-ceiling fallback
/// share one predicate throughout this codebase — see e.g.
/// `re_render::lib.rs`'s repeated `matches!(surface, SurfaceKind::Ceiling |
/// SurfaceKind::Empty)` "is sky" checks — because `Ceiling` never carries a
/// distinct physical texture separate from the open-sky case here.
#[inline]
pub(crate) fn is_sky(surface: SurfaceKind) -> bool {
    matches!(surface, SurfaceKind::Ceiling | SurfaceKind::Empty)
}

/// The WORLD/SURFACE-anchored bucket for `cell`, or `None` on sky (the caller
/// draws no grain/dither there at all).
///
/// * Wall-like surfaces (`SurfaceKind::is_wall()`, plus `Portal` — which
///   "carries the same geometric data as a wall hit", per `Cell::surface`'s
///   own docs) use the FACE texture coordinate `(tex_u, tex_v)`: the same
///   physical point on the wall regardless of which screen column/row
///   currently happens to draw it, so it is stable under both panning and
///   camera rotation.
/// * Everything else physical (`Floor`) uses the world hit point
///   (`Cell::hit`) directly — the exact ground position the ray struck.
#[inline]
pub(crate) fn world_bucket(cell: &Cell) -> Option<(i64, i64)> {
    if is_sky(cell.surface) {
        return None;
    }
    let (bx, by) = if cell.surface.is_wall() || cell.surface == SurfaceKind::Portal {
        (
            cell.tex_u * WALL_BUCKET_DENSITY,
            cell.tex_v * WALL_BUCKET_DENSITY,
        )
    } else {
        (
            cell.hit.x * FLOOR_BUCKET_DENSITY,
            cell.hit.y * FLOOR_BUCKET_DENSITY,
        )
    };
    Some((bx.floor() as i64, by.floor() as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::glam::Vec2;
    use re_core::Cell;

    fn floor_cell(hit: Vec2) -> Cell {
        Cell {
            surface: SurfaceKind::Floor,
            hit,
            ..Default::default()
        }
    }

    fn wall_cell(tex_u: f32, tex_v: f32) -> Cell {
        Cell {
            surface: SurfaceKind::WallSolid,
            tex_u,
            tex_v,
            ..Default::default()
        }
    }

    #[test]
    fn sky_has_no_bucket() {
        let sky = Cell {
            surface: SurfaceKind::Empty,
            ..Default::default()
        };
        let ceiling = Cell {
            surface: SurfaceKind::Ceiling,
            ..Default::default()
        };
        assert_eq!(world_bucket(&sky), None);
        assert_eq!(world_bucket(&ceiling), None);
    }

    #[test]
    fn floor_bucket_is_keyed_on_world_hit_not_screen_position() {
        // Two cells at the SAME world point (e.g. reached by two different
        // screen columns/rows across a camera turn) must land in the same
        // bucket — that is the entire point of the anchor.
        let a = floor_cell(Vec2::new(3.4, 7.1));
        let b = floor_cell(Vec2::new(3.4, 7.1));
        assert_eq!(world_bucket(&a), world_bucket(&b));

        // A different world point must (generically) land in a different
        // bucket at this density.
        let c = floor_cell(Vec2::new(30.0, 70.0));
        assert_ne!(world_bucket(&a), world_bucket(&c));
    }

    #[test]
    fn wall_bucket_is_keyed_on_face_texel_not_screen_position() {
        let a = wall_cell(0.12, 0.5);
        let b = wall_cell(0.12, 0.5);
        assert_eq!(world_bucket(&a), world_bucket(&b));

        let c = wall_cell(0.92, 0.5);
        assert_ne!(world_bucket(&a), world_bucket(&c));
    }

    #[test]
    fn portal_uses_the_wall_texel_anchor() {
        let portal = Cell {
            surface: SurfaceKind::Portal,
            tex_u: 0.3,
            tex_v: 0.6,
            ..Default::default()
        };
        let wall = wall_cell(0.3, 0.6);
        assert_eq!(world_bucket(&portal), world_bucket(&wall));
    }
}
