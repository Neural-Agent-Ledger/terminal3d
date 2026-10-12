//! Posterize / colour-quantize: snap each channel to `levels` evenly-spaced steps
//! for a flat, banded poster / ASCII-art look. A float-domain stylistic banding —
//! distinct from the final dither pass, which quantizes to the terminal PALETTE.
//! Pure per-cell colour op on fg + bg; glyph and all other fields are preserved.
//!
//! With `levels = N`, a channel value `v ∈ [0,1]` maps to `round(v·(N-1))/(N-1)`,
//! so the endpoints 0 and 1 are exactly preserved (true black and white survive).
//! `levels <= 1` would collapse the image, so it is clamped to `>= 2`.

use re_core::Framebuffer;

use crate::color;
use crate::config::PosterizeFx;

#[inline]
fn quantize(c: [f32; 3], steps: f32) -> [f32; 3] {
    let q = |v: f32| ((v.clamp(0.0, 1.0) * steps).round()) / steps;
    [q(c[0]), q(c[1]), q(c[2])]
}

pub(crate) fn apply(fb: &mut Framebuffer, e: &PosterizeFx, master: f32) {
    if !e.enabled || master <= 0.0 {
        return;
    }
    // `levels` evenly-spaced values ⇒ (levels-1) steps between endpoints.
    let levels = e.levels.clamp(2, 64);
    let steps = (levels - 1) as f32;
    let cols = fb.cols();
    let rows = fb.rows();
    for row in 0..rows {
        for col in 0..cols {
            let Some(s) = fb.get(col, row) else { continue };
            let mut c = *s;
            c.fg = color::pack_f(quantize(color::unpack_f(c.fg), steps));
            c.bg = color::pack_f(quantize(color::unpack_f(c.bg), steps));
            fb.set(col, row, c);
        }
    }
}
