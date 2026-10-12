//! Row shift / tearing: horizontal slices slide left/right (datamosh tears).
//!
//! The frame is cut into `slice_h`-row bands. A `tear_frac` fraction of bands
//! (hashed per time slice) shift horizontally by a coherent noise-driven amount
//! up to `max_shift`. Each shifted row samples the CLEAN `src` so the tear is a
//! clean offset, not a blur.

use re_core::Framebuffer;

use super::TearEffect;
use crate::hash::{hash, signed, tslice, unit};

pub(crate) fn apply(
    fb: &mut Framebuffer,
    src: &Framebuffer,
    e: &TearEffect,
    master: f32,
    time: f32,
    seed: u64,
) {
    if !e.enabled {
        return;
    }
    let strength = e.intensity * master;
    if strength <= 0.0 {
        return;
    }
    let slice_h = e.slice_h.max(1) as usize;
    let max_shift = ((e.max_shift as f32) * strength).round() as i32;
    if max_shift == 0 {
        return;
    }
    let cols = src.cols();
    let rows = src.rows();
    if cols == 0 || rows == 0 {
        return;
    }
    let tslot = tslice(time, e.rate);
    let tear_frac = (e.tear_frac * strength).clamp(0.0, 1.0);
    let salt = seed ^ e.seed_off;

    let n_slices = rows.div_ceil(slice_h);
    for s in 0..n_slices {
        let h = hash(&[salt, s as u64, tslot]);
        if unit(h) >= tear_frac {
            continue; // this band passes through clean
        }
        let shift = (signed(h.rotate_left(19)) * max_shift as f32).round() as i32;
        if shift == 0 {
            continue;
        }
        let y0 = s * slice_h;
        for row in y0..(y0 + slice_h).min(rows) {
            for col in 0..cols {
                let scol = (col as i32 - shift).rem_euclid(cols as i32) as usize;
                if let Some(c) = src.get(scol, row) {
                    fb.set(col, row, *c);
                }
            }
        }
    }
}
