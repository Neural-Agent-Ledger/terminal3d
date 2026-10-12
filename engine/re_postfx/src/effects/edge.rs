//! Edge-outline (toon ink): detect a depth OR surface-kind discontinuity versus
//! the 4-neighbours and darken the fg so geometry reads ink-outlined.
//!
//! For each cell we compare its depth + surface kind to its up/down/left/right
//! neighbours (sampled from the CLEAN `src` so an edge is detected against the
//! un-inked layout). A relative depth jump beyond `depth_threshold`, OR any
//! surface-kind change, marks the cell an edge; its fg is darkened by `strength`.
//! Pure fg modulation — bg, glyph, and every other field are preserved.

use re_core::{Framebuffer, SurfaceKind};

use crate::color;
use crate::config::EdgeFx;

/// True if `(col,row)` in `src` sits on a depth-, surface-, or (when
/// `luma_threshold > 0`) luma-discontinuity edge.
#[inline]
fn is_edge(
    src: &Framebuffer,
    col: usize,
    row: usize,
    depth_threshold: f32,
    luma_threshold: f32,
) -> bool {
    let here = match src.get(col, row) {
        Some(c) => c,
        None => return false,
    };
    let cols = src.cols() as i32;
    let rows = src.rows() as i32;
    let c = col as i32;
    let r = row as i32;
    // Local depth scale: edges are relative to how far this surface is, so a
    // distant wall and a near wall both outline at the same *relative* jump.
    let local = if here.depth.is_finite() {
        here.depth.max(0.001)
    } else {
        1.0
    };
    let limit = depth_threshold * local;
    let here_luma = if luma_threshold > 0.0 {
        color::luma(here.fg)
    } else {
        0.0
    };

    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let nx = c + dx;
        let ny = r + dy;
        if nx < 0 || ny < 0 || nx >= cols || ny >= rows {
            continue;
        }
        let Some(n) = src.get(nx as usize, ny as usize) else {
            continue;
        };
        // Surface-kind change is always an edge (e.g. wall→sky, wall→floor).
        if n.surface != here.surface {
            return true;
        }
        // Luma-discontinuity edge: a bright fg feature on a flat wall (a painted
        // line, a hot highlight) outlines even without a geometry seam.
        if luma_threshold > 0.0 && (here_luma - color::luma(n.fg)).abs() > luma_threshold {
            return true;
        }
        // A finite vs infinite depth boundary (silhouette against sky) is an edge.
        match (here.depth.is_finite(), n.depth.is_finite()) {
            (true, false) | (false, true) => return true,
            (true, true) => {
                if (here.depth - n.depth).abs() > limit {
                    return true;
                }
            }
            (false, false) => {}
        }
        // Edge between two different empty/sky surfaces caught by the kind check
        // above; nothing else to do.
        let _ = SurfaceKind::Empty;
    }
    false
}

pub(crate) fn apply(fb: &mut Framebuffer, src: &Framebuffer, e: &EdgeFx, master: f32) {
    if !e.enabled {
        return;
    }
    let strength = (e.strength * master).clamp(0.0, 1.0);
    if strength <= 0.0 {
        return;
    }
    let cols = src.cols();
    let rows = src.rows();
    for row in 0..rows {
        for col in 0..cols {
            if !is_edge(src, col, row, e.depth_threshold, e.luma_threshold) {
                continue;
            }
            let Some(s) = src.get(col, row) else { continue };
            let mut c = *s;
            // Ink: darken the fg toward black by `strength`.
            c.fg = color::scale(c.fg, 1.0 - strength);
            fb.set(col, row, c);
        }
    }
}
