//! Wave / warp: per-column vertical (and per-row horizontal) sine displacement.
//!
//! Each cell `(col,row)` samples the CLEAN source at a sine-offset position, so
//! the frame wobbles like a distorted signal. Deterministic: the phase comes
//! from `time * speed`, not a clock read. Reads `src` (a clean clone) and writes
//! `fb` so the displacement samples the un-warped image.

use re_core::Framebuffer;

use super::WarpEffect;

pub(crate) fn apply(
    fb: &mut Framebuffer,
    src: &Framebuffer,
    e: &WarpEffect,
    master: f32,
    time: f32,
    _seed: u64,
) {
    if !e.enabled {
        return;
    }
    let amp_y = e.amp_y * e.intensity * master;
    let amp_x = e.amp_x * e.intensity * master;
    if amp_y == 0.0 && amp_x == 0.0 {
        return;
    }
    let cols = src.cols();
    let rows = src.rows();
    if cols == 0 || rows == 0 {
        return;
    }
    let two_pi = std::f32::consts::TAU;
    let phase = time * e.speed * two_pi;

    for row in 0..rows {
        // Horizontal offset driven by the ROW position (a vertical-running wave).
        let rwave = ((row as f32 / rows as f32) * e.freq * two_pi + phase).sin();
        let off_x = (rwave * amp_x).round() as i32;
        for col in 0..cols {
            // Vertical offset driven by the COLUMN position (a horizontal wave).
            let cwave = ((col as f32 / cols as f32) * e.freq * two_pi + phase).sin();
            let off_y = (cwave * amp_y).round() as i32;

            let scol = (col as i32 - off_x).rem_euclid(cols as i32) as usize;
            let srow = (row as i32 - off_y).rem_euclid(rows as i32) as usize;
            if let Some(c) = src.get(scol, srow) {
                fb.set(col, row, *c);
            }
        }
    }
}
