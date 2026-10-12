//! Chromatic aberration: a lens-fringe channel split. Near the frame edges the
//! red channel is sampled from a neighbour offset one way and the blue channel
//! from a neighbour offset the other way, leaving a cyan/red ghost on high-contrast
//! features — exactly the radial fringe of a real lens, strongest at the corners.
//!
//! Terminal-appropriate: there are no real subpixels, so we offset whole-cell
//! colour CHANNELS toward neighbour cells (a 1..n cell shift), not sub-pixel
//! samples. The offset magnitude grows with the radial distance from frame centre
//! (`edge_only`..corner), so the centre of frame stays crisp. Reads the clean
//! `src`; writes fg + bg. Glyph and all other fields are preserved.

use re_core::Framebuffer;

use crate::color;
use crate::config::ChromaticFx;

/// Sample one packed colour channel from `src` at `(col+off, row)`, clamped to the
/// frame edge. `chan`: 0=R, 1=G, 2=B.
#[inline]
fn sample_chan(src: &Framebuffer, col: i32, row: usize, off: i32, fg: bool, chan: usize) -> f32 {
    let cols = src.cols() as i32;
    let sx = (col + off).clamp(0, cols - 1) as usize;
    let Some(c) = src.get(sx, row) else {
        return 0.0;
    };
    color::unpack_f(if fg { c.fg } else { c.bg })[chan]
}

pub(crate) fn apply(fb: &mut Framebuffer, src: &Framebuffer, e: &ChromaticFx, master: f32) {
    if !e.enabled {
        return;
    }
    let amount = (e.amount * master).max(0.0);
    if amount <= 0.0 {
        return;
    }
    let cols = src.cols();
    let rows = src.rows();
    if cols == 0 || rows == 0 {
        return;
    }
    // Radial falloff: no fringe inside `edge_only`, ramping to full at the corner.
    let start = e.edge_only.clamp(0.0, 0.95);
    let cx = (cols as f32 - 1.0) * 0.5;
    let cy = (rows as f32 - 1.0) * 0.5;
    let nx = cx.max(1e-3);
    let ny = cy.max(1e-3);

    for row in 0..rows {
        for col in 0..cols {
            let dx = (col as f32 - cx) / nx;
            let dy = (row as f32 - cy) / ny;
            let r = (dx * dx + dy * dy).sqrt() / std::f32::consts::SQRT_2;
            let t = ((r - start) / (1.0 - start)).clamp(0.0, 1.0);
            // Smoothstep the radial ramp, then scale to the peak cell offset.
            let smooth = t * t * (3.0 - 2.0 * t);
            // Signed offset: split outward along the horizontal axis (sign follows
            // which side of centre we're on, so the fringe points away from centre).
            let dir = if (col as f32) < cx { -1.0 } else { 1.0 };
            let off = (amount * smooth * dir).round() as i32;
            if off == 0 {
                continue;
            }
            let Some(s) = src.get(col, row) else { continue };
            let mut c = *s;
            // R pulled from +off, B from -off, G stays — the classic split.
            let fg0 = color::unpack_f(c.fg);
            let bg0 = color::unpack_f(c.bg);
            c.fg = color::pack_f([
                sample_chan(src, col as i32, row, off, true, 0),
                fg0[1],
                sample_chan(src, col as i32, row, -off, true, 2),
            ]);
            c.bg = color::pack_f([
                sample_chan(src, col as i32, row, off, false, 0),
                bg0[1],
                sample_chan(src, col as i32, row, -off, false, 2),
            ]);
            fb.set(col, row, c);
        }
    }
}
