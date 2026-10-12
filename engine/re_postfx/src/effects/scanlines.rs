//! Scanlines / CRT shadow-mask: darken alternate rows (the scanline gap) and
//! optionally apply a 3-cell-period column mask (an aperture-grille feel). A
//! spatial, time-free effect — a pure function of cell coordinates, so it is
//! golden-stable. Multiplies fg + bg toward black; glyph and all other fields are
//! preserved.

use re_core::Framebuffer;

use crate::color;
use crate::config::ScanlineFx;

/// Per-column aperture mask weight (1 = full, <1 = darkened). A 3-phase RGB-stripe
/// mask: each column triad dims one phase, evoking the shadow-mask grille.
#[inline]
fn mask_weight(col: usize, mask: f32) -> f32 {
    // Phase 2 of every 3 columns is the "gap" between phosphor stripes.
    if col % 3 == 2 {
        1.0 - mask
    } else {
        1.0
    }
}

pub(crate) fn apply(fb: &mut Framebuffer, e: &ScanlineFx, master: f32) {
    if !e.enabled {
        return;
    }
    let strength = (e.strength * master).clamp(0.0, 1.0);
    let mask = (e.mask * master).clamp(0.0, 1.0);
    if strength <= 0.0 && mask <= 0.0 {
        return;
    }
    // period 1 ⇒ every other row is a scanline gap; period p ⇒ 1 dark row per p+1.
    let period = e.period.max(1) as usize;
    let cols = fb.cols();
    let rows = fb.rows();
    for row in 0..rows {
        // A row is a "gap" (darkened) on the off-phase of the period.
        let on_scanline = (row % (period + 1)) == period;
        let row_k = if on_scanline { 1.0 - strength } else { 1.0 };
        for col in 0..cols {
            let k = row_k
                * if mask > 0.0 {
                    mask_weight(col, mask)
                } else {
                    1.0
                };
            if k >= 1.0 {
                continue;
            }
            let Some(s) = fb.get(col, row) else { continue };
            let mut c = *s;
            c.fg = color::scale(c.fg, k);
            c.bg = color::scale(c.bg, k);
            fb.set(col, row, c);
        }
    }
}
