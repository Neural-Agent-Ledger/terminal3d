//! SSAO / contact shadow: cheap screen-space ambient occlusion from the depth
//! buffer. A cell whose depth is notably GREATER than its neighbours' (a recess,
//! or the far side behind an edge) gets its background darkened proportionally.
//!
//! For each cell we accumulate, over a small radius, how much CLOSER each
//! neighbour is (neighbour in front ⇒ it occludes ambient light reaching us).
//! The summed, biased, normalized occlusion scales the bg toward black by up to
//! `strength`. Reads depth from the clean `src`; writes bg only — fg, glyph and
//! every other field are preserved.

use re_core::Framebuffer;

use crate::color;
use crate::config::SsaoFx;

pub(crate) fn apply(fb: &mut Framebuffer, src: &Framebuffer, e: &SsaoFx, master: f32) {
    if !e.enabled {
        return;
    }
    let strength = (e.strength * master).clamp(0.0, 1.0);
    if strength <= 0.0 {
        return;
    }
    let radius = e.radius.clamp(1, 2) as i32;
    let cols = src.cols() as i32;
    let rows = src.rows() as i32;
    if cols == 0 || rows == 0 {
        return;
    }

    for row in 0..rows {
        for col in 0..cols {
            let Some(here) = src.get(col as usize, row as usize) else {
                continue;
            };
            // Sky / infinite-depth cells receive no contact shadow.
            if !here.depth.is_finite() {
                continue;
            }
            let mut occ = 0.0f32;
            let mut samples = 0.0f32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = col + dx;
                    let ny = row + dy;
                    if nx < 0 || ny < 0 || nx >= cols || ny >= rows {
                        continue;
                    }
                    let Some(n) = src.get(nx as usize, ny as usize) else {
                        continue;
                    };
                    samples += 1.0;
                    if !n.depth.is_finite() {
                        continue; // a neighbour at infinity does not occlude
                    }
                    // Positive when the neighbour is CLOSER than us (it shadows us).
                    let diff = here.depth - n.depth - e.bias;
                    if diff > 0.0 {
                        // Saturate per-neighbour contribution so one very close
                        // neighbour doesn't blow out the whole sum.
                        occ += (diff).min(1.0);
                    }
                }
            }
            if samples <= 0.0 {
                continue;
            }
            let ao = (occ / samples).clamp(0.0, 1.0);
            if ao <= 0.0 {
                continue;
            }
            let k = 1.0 - strength * ao;
            let mut c = *here;
            c.bg = color::scale(c.bg, k);
            fb.set(col as usize, row as usize, c);
        }
    }
}
