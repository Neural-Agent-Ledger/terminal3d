//! Vignette: radial darkening toward the frame edges. A normalized elliptical
//! radius drives a smooth falloff that multiplies fg + bg toward black near the
//! corners. Pure per-cell color op; glyph and all other fields are preserved.
//!
//! The cell aspect (terminal cells are ~2× tall) is folded in so the dark oval
//! reads roughly circular on screen rather than squashed.

use re_core::Framebuffer;

use crate::color;
use crate::config::VignetteFx;

pub(crate) fn apply(fb: &mut Framebuffer, e: &VignetteFx, master: f32) {
    if !e.enabled {
        return;
    }
    let amount = (e.amount * master).clamp(0.0, 1.0);
    if amount <= 0.0 {
        return;
    }
    let cols = fb.cols();
    let rows = fb.rows();
    if cols == 0 || rows == 0 {
        return;
    }
    // Falloff start radius (0..1): below this, no darkening; from here to 1 the
    // darkening ramps in. Clamp away from 1 so there's always some ramp.
    let start = e.softness.clamp(0.0, 0.95);
    let cx = (cols as f32 - 1.0) * 0.5;
    let cy = (rows as f32 - 1.0) * 0.5;
    // Normalize so a corner is radius ~1. Terminal cells are ~2:1, so weight the
    // vertical axis to keep the falloff visually round.
    let nx = cx.max(1e-3);
    let ny = cy.max(1e-3);

    for row in 0..rows {
        for col in 0..cols {
            let dx = (col as f32 - cx) / nx;
            let dy = (row as f32 - cy) / ny * 1.0; // grid-space round oval
            let r = (dx * dx + dy * dy).sqrt() / std::f32::consts::SQRT_2;
            // Quintic smootherstep from `start`..1.0 — a gentler shoulder than the
            // cubic smoothstep, so the darkening eases in without a visible ring.
            let t = ((r - start) / (1.0 - start)).clamp(0.0, 1.0);
            let smooth = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
            let dark = amount * smooth;
            if dark <= 0.0 {
                continue;
            }
            let k = 1.0 - dark;
            let Some(s) = fb.get(col, row) else { continue };
            let mut c = *s;
            c.fg = color::scale(c.fg, k);
            c.bg = color::scale(c.bg, k);
            fb.set(col, row, c);
        }
    }
}
