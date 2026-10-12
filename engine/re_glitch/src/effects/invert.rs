//! Invert / posterize flashes: pulsing full-frame or banded colour corruption.
//!
//! On a `rate`-Hz schedule, for the `duty` fraction of each period the flash is
//! ON: colours are inverted (`255 - c`) and, if `levels > 0`, posterized
//! (quantized to `levels` steps per channel). With `band_h > 0` only a
//! horizontal band (centred on a time-scrolled row) flashes; otherwise the whole
//! frame does. The pulse schedule is derived from `time` (deterministic), not a
//! clock read.

use re_core::Framebuffer;

use super::{color, InvertEffect};

#[inline]
fn posterize_ch(v: u8, levels: u32) -> u8 {
    if levels == 0 || levels >= 256 {
        return v;
    }
    let steps = levels.max(2) - 1;
    let q = ((v as u32 * steps + 127) / 255) * 255 / steps;
    q.min(255) as u8
}

#[inline]
fn corrupt_color(packed: u32, levels: u32) -> u32 {
    let (r, g, b) = color::unpack(packed);
    let (r, g, b) = (255 - r, 255 - g, 255 - b); // invert
    color::pack(
        posterize_ch(r, levels),
        posterize_ch(g, levels),
        posterize_ch(b, levels),
    )
}

pub(crate) fn apply(fb: &mut Framebuffer, e: &InvertEffect, master: f32, time: f32, _seed: u64) {
    if !e.enabled {
        return;
    }
    let strength = e.intensity * master;
    if strength <= 0.0 {
        return;
    }
    // Pulse schedule: phase in [0,1) of the current period; ON while phase < duty.
    let rate = e.rate.max(0.0);
    let on = if rate <= 0.0 {
        // rate 0 → a held-on flash (constant corruption) at this strength.
        true
    } else {
        let phase = (time * rate).rem_euclid(1.0);
        phase < e.duty.clamp(0.0, 1.0)
    };
    if !on {
        return;
    }

    let cols = fb.cols();
    let rows = fb.rows();
    if cols == 0 || rows == 0 {
        return;
    }

    // Banded vs full-frame.
    let (r0, r1) = if e.band_h > 0 {
        let bh = e.band_h.min(rows as u32) as usize;
        // Band centre scrolls down over time so successive flashes hit new rows.
        let center = (time * rate * 7.0).rem_euclid(rows as f32) as usize;
        let top = center.saturating_sub(bh / 2);
        (top, (top + bh).min(rows))
    } else {
        (0, rows)
    };

    let levels = e.levels;
    for row in r0..r1 {
        for col in 0..cols {
            let Some(src) = fb.get(col, row) else {
                continue;
            };
            let mut c = *src;
            c.fg = corrupt_color(c.fg, levels);
            c.bg = corrupt_color(c.bg, levels);
            fb.set(col, row, c);
        }
    }
}
