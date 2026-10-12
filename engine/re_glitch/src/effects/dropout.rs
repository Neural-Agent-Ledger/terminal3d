//! Dropout / dead cells: cells intermittently blank to black (signal loss).
//!
//! Per cell, a hashed `[0,1)` value below the (intensity-scaled) `density`
//! blanks the cell — fg/bg forced black and the glyph set to `dead_glyph`
//! (space by default). The hash folds a time slice so the dead cells FLICKER at
//! `rate` Hz rather than sitting static.

use re_core::Framebuffer;

use super::DropoutEffect;
use crate::hash::{hash, tslice, unit};

pub(crate) fn apply(fb: &mut Framebuffer, e: &DropoutEffect, master: f32, time: f32, seed: u64) {
    if !e.enabled {
        return;
    }
    let strength = e.intensity * master;
    if strength <= 0.0 {
        return;
    }
    let density = (e.density * strength).clamp(0.0, 1.0);
    if density <= 0.0 {
        return;
    }
    let cols = fb.cols();
    let rows = fb.rows();
    if cols == 0 || rows == 0 {
        return;
    }
    let slot = tslice(time, e.rate);
    let salt = seed ^ e.seed_off;

    for row in 0..rows {
        for col in 0..cols {
            let h = hash(&[salt, col as u64, row as u64, slot]);
            if unit(h) >= density {
                continue;
            }
            let Some(src) = fb.get(col, row) else {
                continue;
            };
            let mut c = *src;
            c.fg = 0x0000_0000;
            c.bg = 0x0000_0000;
            c.glyph = e.dead_glyph;
            fb.set(col, row, c);
        }
    }
}
