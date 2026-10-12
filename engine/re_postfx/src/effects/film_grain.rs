//! Film grain: a subtle luminance wobble keyed on the cell's own WORLD/SURFACE
//! anchor (`effects::world_anchor`) and a SLOW-clock `time_bucket` — NOT screen
//! `(col, row)`. So:
//!   * a STATIC surface texel at a FIXED `time_bucket` is byte-identical every
//!     call (no per-render-frame shimmer — the persistence principle: grain is
//!     anchored, it does not strobe), and
//!   * bumping `time_bucket` (the caller does this a few times a second) re-rolls
//!     the grain so it crawls gently like real film, and
//!   * turning or walking the camera reveals DIFFERENT world texels rather than
//!     re-rolling the ones already on screen — the pattern travels WITH the
//!     surface instead of staying glued to the screen grid (this replaced an
//!     earlier screen-anchored version that visibly glued the texture to the
//!     screen while turning).
//!
//! The blend is MULTIPLICATIVE, not additive: an additive `±amount` added to
//! every channel regardless of the cell's own colour meant that on a
//! near-black floor (`bg ≈ #0B0A0A`) a `cinematic`-preset amplitude of ~0.028
//! (0.035 master-scaled) was a ±60% swing — a clearly visible pattern on
//! exactly the darkest, lowest-contrast surfaces. Scaling by `(1 + n)` instead
//! makes the wobble proportionate: small in absolute terms on dark cells, more
//! visible on bright ones, which is what real film grain (and light-dependent
//! sensor noise) actually looks like.
//!
//! Sky gets none at all (`world_anchor::world_bucket` returns `None` there) —
//! sky and screen-only cells get none.
//!
//! Pure per-cell colour op on fg + bg; glyph and all other fields are preserved.
//! No global state, no RNG, no `std` time — deterministic and golden-stable.

use re_core::Framebuffer;

use super::world_anchor::world_bucket;
use crate::color;
use crate::config::FilmGrainFx;

/// SplitMix64 finalizer — a deterministic 64→64 avalanche mix (mirrors the integer
/// hash used in `re_glitch`/`ascii_shader`, so the look is reproducible).
#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

/// A signed `[-1, 1)` grain value for a world/surface bucket `(bx, by)` at a
/// given time bucket. The time bucket is folded into the hash so the grain is
/// STABLE within a bucket and re-rolls across buckets — never a per-frame term.
#[inline]
fn grain(bx: i64, by: i64, bucket: u32) -> f32 {
    // Distinct salts per lane so (bx,by,bucket) maps to a well-spread stream.
    // `as u64` on a negative i64 wraps to its two's-complement bit pattern,
    // which is fine here — we only need a stable, well-mixed key, not an
    // ordering.
    let h = mix64(
        (bx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (by as u64)
                .wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                .rotate_left(21)
            ^ (bucket as u64)
                .wrapping_mul(0x1656_67B1_9E37_79F9)
                .rotate_left(42),
    );
    // Top 24 bits → [0,1), then to [-1,1).
    let u = ((h >> 40) as f32) / ((1u64 << 24) as f32);
    u * 2.0 - 1.0
}

pub(crate) fn apply(fb: &mut Framebuffer, e: &FilmGrainFx, master: f32) {
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
            }; // sky: no grain
            let n = grain(bx, by, e.time_bucket) * amount;
            if n == 0.0 {
                continue;
            }
            let mut c = *s;
            // Luminance grain: the SAME multiplicative factor on every channel
            // (a brightness wobble, not a colour-noise hash), so it reads as
            // film grain and stays proportionate on dark cells.
            let k = 1.0 + n;
            c.fg = color::scale(c.fg, k);
            c.bg = color::scale(c.bg, k);
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

    fn on(amount: f32, bucket: u32) -> FilmGrainFx {
        FilmGrainFx {
            enabled: true,
            amount,
            time_bucket: bucket,
        }
    }

    #[test]
    fn disabled_or_zero_amount_is_a_no_op() {
        let src = fb_of(&[(
            0x00_10_10_10,
            SurfaceKind::Floor,
            Vec2::new(1.0, 1.0),
            0.0,
            0.0,
        )]);
        let mut off = src.clone();
        apply(
            &mut off,
            &FilmGrainFx {
                enabled: false,
                ..on(0.5, 3)
            },
            1.0,
        );
        assert_eq!(off.get(0, 0), src.get(0, 0));

        let mut zero = src.clone();
        apply(&mut zero, &on(0.0, 3), 1.0);
        assert_eq!(zero.get(0, 0), src.get(0, 0));
    }

    #[test]
    fn sky_cells_get_no_grain_at_all() {
        let src = fb_of(&[(0x00_10_10_10, SurfaceKind::Empty, Vec2::ZERO, 0.0, 0.0)]);
        let mut out = src.clone();
        apply(&mut out, &on(0.9, 1), 1.0);
        assert_eq!(
            out.get(0, 0),
            src.get(0, 0),
            "sky must be untouched by grain"
        );
    }

    #[test]
    fn grain_is_proportionate_not_a_fixed_absolute_delta() {
        // B36's core complaint: on a near-black floor, additive grain was a huge
        // RELATIVE swing. Prove the new multiplicative grain scales with the
        // cell's own brightness: a dark cell's absolute delta must be much
        // smaller than a bright cell's, at the SAME world bucket + amount (so
        // the same underlying grain sample `n` applies to both).
        let dark_hit = Vec2::new(11.0, 11.0);
        let dark = fb_of(&[(0x00_08_08_08, SurfaceKind::Floor, dark_hit, 0.0, 0.0)]);
        let bright = fb_of(&[(0x00_F0_F0_F0, SurfaceKind::Floor, dark_hit, 0.0, 0.0)]);

        let mut dark_out = dark.clone();
        let mut bright_out = bright.clone();
        let cfg = on(0.5, 7);
        apply(&mut dark_out, &cfg, 1.0);
        apply(&mut bright_out, &cfg, 1.0);

        let dark_delta = (color::unpack_f(dark_out.get(0, 0).unwrap().bg)[0]
            - color::unpack_f(dark.get(0, 0).unwrap().bg)[0])
            .abs();
        let bright_delta = (color::unpack_f(bright_out.get(0, 0).unwrap().bg)[0]
            - color::unpack_f(bright.get(0, 0).unwrap().bg)[0])
            .abs();
        assert!(
            bright_delta > dark_delta * 3.0,
            "multiplicative grain must scale with brightness: dark Δ={dark_delta}, bright Δ={bright_delta}"
        );
    }

    #[test]
    fn grain_travels_with_the_surface_across_a_simulated_camera_turn() {
        // B36's acceptance test in miniature: build two "frames" where the SAME
        // two world floor points are seen through DIFFERENT screen columns (as
        // a camera turn would do), at the SAME time_bucket. World-anchored
        // grain must produce the SAME per-world-point result regardless of
        // which screen column carries it — the old screen-keyed version could
        // not do this by construction.
        let p0 = Vec2::new(4.0, 9.0);
        let p1 = Vec2::new(4.0, 9.4);
        let bg = 0x00_20_20_20;

        // "Before the turn": p0 at column 0, p1 at column 1.
        let mut frame_a = fb_of(&[
            (bg, SurfaceKind::Floor, p0, 0.0, 0.0),
            (bg, SurfaceKind::Floor, p1, 0.0, 0.0),
        ]);
        // "After the turn": the same two world points now sampled by the
        // OPPOSITE screen columns (1 and 0) — exactly what panning does.
        let mut frame_b = fb_of(&[
            (bg, SurfaceKind::Floor, p1, 0.0, 0.0),
            (bg, SurfaceKind::Floor, p0, 0.0, 0.0),
        ]);

        let cfg = on(0.6, 2);
        apply(&mut frame_a, &cfg, 1.0);
        apply(&mut frame_b, &cfg, 1.0);

        // p0's shaded result must be identical whichever column it lands in.
        assert_eq!(frame_a.get(0, 0).unwrap().bg, frame_b.get(1, 0).unwrap().bg);
        // Same for p1.
        assert_eq!(frame_a.get(1, 0).unwrap().bg, frame_b.get(0, 0).unwrap().bg);
    }

    #[test]
    fn a_static_bucket_is_byte_identical_across_calls() {
        let src = fb_of(&[(
            0x00_40_30_20,
            SurfaceKind::Floor,
            Vec2::new(2.0, 2.0),
            0.0,
            0.0,
        )]);
        let mut a = src.clone();
        let mut b = src.clone();
        let cfg = on(0.4, 9);
        apply(&mut a, &cfg, 1.0);
        apply(&mut b, &cfg, 1.0);
        assert_eq!(a.get(0, 0), b.get(0, 0));
    }
}
