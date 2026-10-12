//! Colour temperature / tint (white balance): a cheap per-channel multiply that
//! warms or cools the whole frame plus an orthogonal magenta↔green tint. Pure
//! per-cell colour op on fg + bg; glyph and all other fields are preserved.
//!
//! `temperature` > 0 warms (boost R, cut B); < 0 cools (cut R, boost B).
//! `tint` > 0 pushes green (boost G); < 0 pushes magenta (boost R+B). The push is
//! a gentle ±~30% gain so it tints rather than clips, and is energy-preserving
//! enough that a neutral grey stays roughly grey-valued.

use re_core::Framebuffer;

use crate::color;
use crate::config::ColorTempFx;

/// Per-channel white-balance gains for a given temperature/tint, both in [-1,1].
#[inline]
fn gains(temperature: f32, tint: f32) -> [f32; 3] {
    let temp = temperature.clamp(-1.0, 1.0);
    let tnt = tint.clamp(-1.0, 1.0);
    // Temperature: warm = +R / -B; the ±0.3 keeps it a tint, not a clip.
    let r = 1.0 + 0.30 * temp;
    let b = 1.0 - 0.30 * temp;
    // Tint: green = +G; magenta = -G (and a touch of +R/+B to balance luma).
    let g = 1.0 + 0.30 * tnt;
    let mag = if tnt < 0.0 { -0.15 * tnt } else { 0.0 }; // magenta lift on R,B
    [r + mag, g, b + mag]
}

pub(crate) fn apply(fb: &mut Framebuffer, e: &ColorTempFx, master: f32) {
    if !e.enabled || master <= 0.0 {
        return;
    }
    let mix = master.clamp(0.0, 1.0);
    let temperature = e.temperature * mix;
    let tint = e.tint * mix;
    if temperature == 0.0 && tint == 0.0 {
        return;
    }
    let [gr, gg, gb] = gains(temperature, tint);
    let cols = fb.cols();
    let rows = fb.rows();
    for row in 0..rows {
        for col in 0..cols {
            let Some(s) = fb.get(col, row) else { continue };
            let mut c = *s;
            let fg = color::unpack_f(c.fg);
            let bg = color::unpack_f(c.bg);
            c.fg = color::pack_f([fg[0] * gr, fg[1] * gg, fg[2] * gb]);
            c.bg = color::pack_f([bg[0] * gr, bg[1] * gg, bg[2] * gb]);
            fb.set(col, row, c);
        }
    }
}
