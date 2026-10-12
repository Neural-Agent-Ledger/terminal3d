//! Dither (ordered Bayer 4×4): add a deterministic, ordered dither pattern to
//! fg + bg to smooth gradients (helps in 256-color mode) or as a stylistic
//! texture. No time term — the pattern is a pure function of the cell's
//! WORLD/SURFACE anchor (`effects::world_anchor`), so it is golden-stable.
//! Glyph and all other fields preserved.
//!
//! This effect used to be keyed on screen `(col, row)` — "the dither pattern
//! is a pure function of cell coordinates" was true, but of the wrong
//! coordinates. On a raking camera turn the Bayer threshold stayed fixed to
//! the screen grid while the underlying surface slid under it, which reads as
//! a texture glued to the inside of the terminal window. Keying on the
//! world/face texel instead means the SAME physical point on the surface
//! always gets the SAME Bayer threshold, so the pattern travels with the
//! geometry — a "dither stabilization" fix well known from other renderers
//! facing the same screen-vs-world anchoring problem.
//!
//! Sky gets none at all (`world_anchor::world_bucket` returns `None` there).

use re_core::Framebuffer;

use super::world_anchor::world_bucket;
use crate::color;
use crate::config::DitherFx;

/// The classic 4×4 Bayer ordered-dither matrix, normalized to `[-0.5, 0.5)`.
const BAYER4: [[f32; 4]; 4] = [
    [0.0, 8.0, 2.0, 10.0],
    [12.0, 4.0, 14.0, 6.0],
    [3.0, 11.0, 1.0, 9.0],
    [15.0, 7.0, 13.0, 5.0],
];

#[inline]
fn bayer(bx: i64, by: i64) -> f32 {
    // Normalize 0..15 → [-0.5, 0.5). `rem_euclid` keeps negative world
    // coordinates indexing the SAME 4×4 tile a positive one would (a plain `%`
    // on a negative i64 would wrap the sign and shift the tile).
    let row = by.rem_euclid(4) as usize;
    let col = bx.rem_euclid(4) as usize;
    BAYER4[row][col] / 16.0 - 0.5
}

#[inline]
fn dither_color(c: u32, offset: f32) -> u32 {
    let f = color::unpack_f(c);
    color::pack_f([f[0] + offset, f[1] + offset, f[2] + offset])
}

pub(crate) fn apply(fb: &mut Framebuffer, e: &DitherFx, master: f32) {
    if !e.enabled {
        return;
    }
    let amount = (e.amount * master).max(0.0);
    if amount <= 0.0 {
        return;
    }
    let cols = fb.cols();
    let rows = fb.rows();
    for row in 0..rows {
        for col in 0..cols {
            let Some(s) = fb.get(col, row) else { continue };
            let Some((bx, by)) = world_bucket(s) else {
                continue;
            }; // sky: no dither
            let offset = bayer(bx, by) * amount;
            if offset == 0.0 {
                continue;
            }
            let mut c = *s;
            c.fg = dither_color(c.fg, offset);
            c.bg = dither_color(c.bg, offset);
            fb.set(col, row, c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_core::glam::Vec2;
    use re_core::{Cell, SurfaceKind};

    fn fb_of(cells: &[(u32, SurfaceKind, Vec2, f32, f32)]) -> Framebuffer {
        let mut fb = Framebuffer::new(cells.len(), 1);
        for (i, &(bg, surface, hit, tex_u, tex_v)) in cells.iter().enumerate() {
            fb.set(
                i,
                0,
                Cell {
                    bg,
                    surface,
                    hit,
                    tex_u,
                    tex_v,
                    ..Default::default()
                },
            );
        }
        fb
    }

    fn on(amount: f32) -> DitherFx {
        DitherFx {
            enabled: true,
            amount,
        }
    }

    #[test]
    fn disabled_or_zero_amount_is_a_no_op() {
        let src = fb_of(&[(
            0x00_40_40_40,
            SurfaceKind::Floor,
            Vec2::new(1.0, 1.0),
            0.0,
            0.0,
        )]);
        let mut off = src.clone();
        apply(
            &mut off,
            &DitherFx {
                enabled: false,
                ..on(0.5)
            },
            1.0,
        );
        assert_eq!(off.get(0, 0), src.get(0, 0));

        let mut zero = src.clone();
        apply(&mut zero, &on(0.0), 1.0);
        assert_eq!(zero.get(0, 0), src.get(0, 0));
    }

    #[test]
    fn sky_cells_get_no_dither_at_all() {
        let src = fb_of(&[(0x00_40_40_40, SurfaceKind::Empty, Vec2::ZERO, 0.0, 0.0)]);
        let mut out = src.clone();
        apply(&mut out, &on(0.5), 1.0);
        assert_eq!(
            out.get(0, 0),
            src.get(0, 0),
            "sky must be untouched by dither"
        );
    }

    #[test]
    fn dither_travels_with_the_surface_across_a_simulated_camera_turn() {
        // Same shape as the film-grain acceptance test: two world floor points,
        // swapped across two screen columns (what a camera pan does). The
        // per-world-point dither result must be identical regardless of which
        // column currently carries it.
        let p0 = Vec2::new(5.0, 9.0);
        let p1 = Vec2::new(5.25, 9.0);
        let bg = 0x00_40_40_40;

        let mut frame_a = fb_of(&[
            (bg, SurfaceKind::Floor, p0, 0.0, 0.0),
            (bg, SurfaceKind::Floor, p1, 0.0, 0.0),
        ]);
        let mut frame_b = fb_of(&[
            (bg, SurfaceKind::Floor, p1, 0.0, 0.0),
            (bg, SurfaceKind::Floor, p0, 0.0, 0.0),
        ]);

        let cfg = on(0.3);
        apply(&mut frame_a, &cfg, 1.0);
        apply(&mut frame_b, &cfg, 1.0);

        assert_eq!(frame_a.get(0, 0).unwrap().bg, frame_b.get(1, 0).unwrap().bg);
        assert_eq!(frame_a.get(1, 0).unwrap().bg, frame_b.get(0, 0).unwrap().bg);
    }

    #[test]
    fn world_bucket_covers_the_full_bayer_tile() {
        // Regression guard for the rem_euclid fix: negative-leaning world
        // coordinates must still spread across all 4 rows/cols, not collapse
        // onto one row the way a naive `%` would for negative inputs.
        let mut seen_rows = std::collections::HashSet::new();
        let mut seen_cols = std::collections::HashSet::new();
        for i in -8_i64..8 {
            seen_rows.insert(i.rem_euclid(4));
            seen_cols.insert(i.rem_euclid(4));
        }
        assert_eq!(seen_rows.len(), 4);
        assert_eq!(seen_cols.len(), 4);
    }
}
