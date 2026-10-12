//! DoF / distance blur: cells beyond the focus distance soften. The blur amount
//! ramps from 0 at `focus_dist` to `max_blur` at `focus_dist + range`. A blurred
//! cell blends its color toward the 4-neighbour average and desaturates — cheap
//! atmospheric depth so the far scene recedes and the focal plane reads sharp.
//!
//! Samples the clean `src` for the neighbour average; writes fg + bg. Glyph and
//! all other fields are preserved. Sky / infinite-depth cells use the max blur
//! (they are the farthest thing in frame).

use re_core::Framebuffer;

use crate::color;
use crate::config::DofFx;

/// 4-neighbour (+self) average of a packed channel selector, from the clean src.
#[inline]
fn neighbour_avg(src: &Framebuffer, col: usize, row: usize, fg: bool) -> [f32; 3] {
    let cols = src.cols() as i32;
    let rows = src.rows() as i32;
    let c = col as i32;
    let r = row as i32;
    let mut acc = [0.0f32; 3];
    let mut count = 0.0f32;
    for (dx, dy) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)] {
        let nx = c + dx;
        let ny = r + dy;
        if nx < 0 || ny < 0 || nx >= cols || ny >= rows {
            continue;
        }
        let Some(cell) = src.get(nx as usize, ny as usize) else {
            continue;
        };
        let f = color::unpack_f(if fg { cell.fg } else { cell.bg });
        acc[0] += f[0];
        acc[1] += f[1];
        acc[2] += f[2];
        count += 1.0;
    }
    if count > 0.0 {
        [acc[0] / count, acc[1] / count, acc[2] / count]
    } else {
        acc
    }
}

/// Desaturate a float color toward its luma by `amt` (0 = none, 1 = greyscale).
#[inline]
fn desat(c: [f32; 3], amt: f32) -> [f32; 3] {
    let l = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    [
        c[0] + (l - c[0]) * amt,
        c[1] + (l - c[1]) * amt,
        c[2] + (l - c[2]) * amt,
    ]
}

pub(crate) fn apply(fb: &mut Framebuffer, src: &Framebuffer, e: &DofFx, master: f32) {
    if !e.enabled {
        return;
    }
    let max_blur = (e.max_blur * master).clamp(0.0, 1.0);
    if max_blur <= 0.0 {
        return;
    }
    let range = e.range.max(1e-3);
    let cols = src.cols();
    let rows = src.rows();

    for row in 0..rows {
        for col in 0..cols {
            let Some(s) = src.get(col, row) else { continue };
            // How far past the focus plane (0 at focus_dist, 1 at +range).
            let blur = if s.depth.is_finite() {
                ((s.depth - e.focus_dist) / range).clamp(0.0, 1.0) * max_blur
            } else {
                max_blur // sky is maximally out of focus
            };
            if blur <= 0.0 {
                continue;
            }
            let mut c = *s;
            // Blend toward neighbour average, then desaturate, scaled by blur.
            let fg_avg = neighbour_avg(src, col, row, true);
            let bg_avg = neighbour_avg(src, col, row, false);
            let mut fg = color::unpack_f(c.fg);
            let mut bg = color::unpack_f(c.bg);
            for i in 0..3 {
                fg[i] += (fg_avg[i] - fg[i]) * blur;
                bg[i] += (bg_avg[i] - bg[i]) * blur;
            }
            fg = desat(fg, blur * 0.6);
            bg = desat(bg, blur * 0.6);
            c.fg = color::pack_f(fg);
            c.bg = color::pack_f(bg);
            fb.set(col, row, c);
        }
    }
}
