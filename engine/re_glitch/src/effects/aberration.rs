//! Chromatic aberration / RGB split: red and blue channels sampled from offset
//! cells, leaving green in place — CRT colour fringing.
//!
//! For each cell, the red component is read from `(col+dx, row+dy)` of the clean
//! `src` and the blue from `(col-dx, row-dy)`; green stays. The glyph itself is
//! kept from the centre cell (we only ghost colour). Applied to fg always and to
//! bg when `split_bg`. With truecolor this makes visible red/blue edge ghosts.

use re_core::Framebuffer;

use super::{color, AberrationEffect};

#[inline]
fn split_channel(
    src: &Framebuffer,
    col: i32,
    row: i32,
    dx: i32,
    dy: i32,
    is_fg: bool,
    center: u32,
) -> u32 {
    let cols = src.cols() as i32;
    let rows = src.rows() as i32;
    let sample = |x: i32, y: i32| -> u32 {
        let cx = x.clamp(0, cols - 1);
        let cy = y.clamp(0, rows - 1);
        let cell = src.get(cx as usize, cy as usize).unwrap();
        if is_fg {
            cell.fg
        } else {
            cell.bg
        }
    };
    let (r, _, _) = color::unpack(sample(col + dx, row + dy));
    let (_, g, _) = color::unpack(center);
    let (_, _, b) = color::unpack(sample(col - dx, row - dy));
    color::pack(r, g, b)
}

pub(crate) fn apply(
    fb: &mut Framebuffer,
    src: &Framebuffer,
    e: &AberrationEffect,
    master: f32,
    _time: f32,
    _seed: u64,
) {
    if !e.enabled {
        return;
    }
    let strength = e.intensity * master;
    if strength <= 0.0 {
        return;
    }
    // Scale the integer offset by strength (round), so low intensity = no split.
    let dx = ((e.dx as f32) * strength).round() as i32;
    let dy = ((e.dy as f32) * strength).round() as i32;
    if dx == 0 && dy == 0 {
        return;
    }
    let cols = src.cols();
    let rows = src.rows();
    for row in 0..rows {
        for col in 0..cols {
            let center = *src.get(col, row).unwrap();
            let mut out = center;
            out.fg = split_channel(src, col as i32, row as i32, dx, dy, true, center.fg);
            if e.split_bg {
                out.bg = split_channel(src, col as i32, row as i32, dx, dy, false, center.bg);
            }
            fb.set(col, row, out);
        }
    }
}
