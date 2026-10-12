//! Block jitter / displacement: rectangular macroblocks jump a few cells.
//!
//! The frame is tiled into `block_w × block_h` blocks. Each block, on a hashed
//! time slice, may "pop" (probability `jump_frac · intensity`) and displace by a
//! hashed offset up to `max_jump`. A displaced block samples the CLEAN `src` at
//! the shifted position, so blocks slide coherently rather than smear.

use re_core::Framebuffer;

use super::BlockEffect;
use crate::hash::{hash, signed, tslice, unit};

pub(crate) fn apply(
    fb: &mut Framebuffer,
    src: &Framebuffer,
    e: &BlockEffect,
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
    let bw = e.block_w.max(1) as usize;
    let bh = e.block_h.max(1) as usize;
    let max_jump = ((e.max_jump as f32) * strength).round() as i32;
    if max_jump == 0 {
        return;
    }
    let cols = src.cols();
    let rows = src.rows();
    if cols == 0 || rows == 0 {
        return;
    }
    let slice = tslice(time, e.rate);
    let jump_frac = (e.jump_frac * strength).clamp(0.0, 1.0);
    let salt = seed ^ e.seed_off;

    let n_bx = cols.div_ceil(bw);
    let n_by = rows.div_ceil(bh);

    for by in 0..n_by {
        for bx in 0..n_bx {
            let h = hash(&[salt, bx as u64, by as u64, slice]);
            // Does this block jump this slice?
            if unit(h) >= jump_frac {
                continue;
            }
            let jx = (signed(h.rotate_left(11)) * max_jump as f32).round() as i32;
            let jy = (signed(h.rotate_left(29)) * max_jump as f32).round() as i32;
            if jx == 0 && jy == 0 {
                continue;
            }
            let x0 = bx * bw;
            let y0 = by * bh;
            for row in y0..(y0 + bh).min(rows) {
                for col in x0..(x0 + bw).min(cols) {
                    let scol = (col as i32 - jx).rem_euclid(cols as i32) as usize;
                    let srow = (row as i32 - jy).rem_euclid(rows as i32) as usize;
                    if let Some(c) = src.get(scol, srow) {
                        fb.set(col, row, *c);
                    }
                }
            }
        }
    }
}
