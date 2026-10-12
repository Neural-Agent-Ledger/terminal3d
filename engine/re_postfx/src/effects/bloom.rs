//! Bloom / glow: threshold the bright cells (fg luma above `threshold`) into a
//! bright-mask, box-blur that mask over a small radius, and ADD the blurred glow
//! back into every cell's fg + bg — bright geometry bleeds a soft halo onto its
//! neighbours.
//!
//! Two scratch buffers (one `Vec<[f32;3]>` for the source mask, one for the blur)
//! — no per-cell heap alloc. Reads fg from the clean `src`; writes fg + bg.
//! Glyph and all other fields are preserved.

use re_core::Framebuffer;

use crate::color;
use crate::config::BloomFx;

pub(crate) fn apply(fb: &mut Framebuffer, src: &Framebuffer, e: &BloomFx, master: f32) {
    if !e.enabled {
        return;
    }
    let intensity = (e.intensity * master).max(0.0);
    if intensity <= 0.0 {
        return;
    }
    let cols = src.cols();
    let rows = src.rows();
    if cols == 0 || rows == 0 {
        return;
    }
    let radius = e.radius.clamp(1, 2) as i32;
    let n = cols * rows;

    // 1. Bright-pass with a SOFT KNEE: per-cell mask = fg color weighted by a
    //    smooth ramp around `threshold`. A hard cutoff makes a bright cell pop the
    //    full glow on the frame it crosses the threshold (temporal shimmer); the
    //    knee fades it in over a `knee`-wide band BELOW the threshold instead.
    let knee = e.knee.clamp(0.0, 1.0);
    let mut mask = vec![[0.0f32; 3]; n];
    for row in 0..rows {
        for col in 0..cols {
            let Some(c) = src.get(col, row) else { continue };
            let l = color::luma(c.fg);
            // knee==0 → hard step at threshold (original behaviour); knee>0 → a
            // smoothstep from (threshold-knee) to (threshold), then a linear ramp
            // up to white past the threshold for the over-bright contribution.
            let w = if knee > 0.0 {
                let lo = e.threshold - knee;
                let t = ((l - lo) / knee.max(1e-3)).clamp(0.0, 1.0);
                let soft = t * t * (3.0 - 2.0 * t); // smoothstep across the knee
                let over = ((l - e.threshold) / (1.0 - e.threshold).max(1e-3)).clamp(0.0, 1.0);
                (soft * over).clamp(0.0, 1.0)
            } else if l > e.threshold {
                ((l - e.threshold) / (1.0 - e.threshold).max(1e-3)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            if w > 0.0 {
                let f = color::unpack_f(c.fg);
                mask[row * cols + col] = [f[0] * w, f[1] * w, f[2] * w];
            }
        }
    }

    // 2. Separable box blur of the mask (horizontal then vertical) into `blur`.
    let mut tmp = vec![[0.0f32; 3]; n];
    let win = (2 * radius + 1) as f32;
    // Horizontal.
    for row in 0..rows {
        for col in 0..cols {
            let mut acc = [0.0f32; 3];
            for dx in -radius..=radius {
                let sx = (col as i32 + dx).clamp(0, cols as i32 - 1) as usize;
                let m = mask[row * cols + sx];
                acc[0] += m[0];
                acc[1] += m[1];
                acc[2] += m[2];
            }
            tmp[row * cols + col] = [acc[0] / win, acc[1] / win, acc[2] / win];
        }
    }
    let mut blur = vec![[0.0f32; 3]; n];
    // Vertical.
    for row in 0..rows {
        for col in 0..cols {
            let mut acc = [0.0f32; 3];
            for dy in -radius..=radius {
                let sy = (row as i32 + dy).clamp(0, rows as i32 - 1) as usize;
                let m = tmp[sy * cols + col];
                acc[0] += m[0];
                acc[1] += m[1];
                acc[2] += m[2];
            }
            blur[row * cols + col] = [acc[0] / win, acc[1] / win, acc[2] / win];
        }
    }

    // 3. Add the glow back into fg (full) and bg (half — the halo lifts the
    //    surrounding darkness less than the lit glyphs).
    for row in 0..rows {
        for col in 0..cols {
            let g = blur[row * cols + col];
            if g[0] + g[1] + g[2] <= 0.0 {
                continue;
            }
            let Some(s) = src.get(col, row) else { continue };
            let mut c = *s;
            let glow = color::pack_f([g[0] * intensity, g[1] * intensity, g[2] * intensity]);
            c.fg = color::add(c.fg, glow);
            let glow_bg = color::pack_f([
                g[0] * intensity * 0.5,
                g[1] * intensity * 0.5,
                g[2] * intensity * 0.5,
            ]);
            c.bg = color::add(c.bg, glow_bg);
            fb.set(col, row, c);
        }
    }
}
