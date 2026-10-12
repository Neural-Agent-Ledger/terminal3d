//! Scanlines / CRT: periodic dim rows + a slow rolling bright/dark refresh band.
//!
//! Every `period`-th row is dimmed by `line_dim` (the scanline comb). On top, a
//! soft band `band_frac` of the frame tall rolls down at `roll_speed` rows/s; it
//! brightens just ahead of its centre and darkens just behind — the CRT refresh
//! roll. Pure brightness scaling of fg + bg; no glyph change.

use re_core::Framebuffer;

use super::{color, ScanlineEffect};

#[inline]
fn scale_rgb(packed: u32, k: f32) -> u32 {
    let (r, g, b) = color::unpack(packed);
    let s = |v: u8| (v as f32 * k).clamp(0.0, 255.0) as u8;
    color::pack(s(r), s(g), s(b))
}

pub(crate) fn apply(fb: &mut Framebuffer, e: &ScanlineEffect, master: f32, time: f32, _seed: u64) {
    if !e.enabled {
        return;
    }
    let strength = e.intensity * master;
    if strength <= 0.0 {
        return;
    }
    let rows = fb.rows();
    let cols = fb.cols();
    if rows == 0 || cols == 0 {
        return;
    }
    let period = e.period.max(1) as usize;
    let line_dim = (e.line_dim * strength).clamp(0.0, 1.0);
    let band_h = (e.band_frac * rows as f32).max(1.0);

    // Rolling band centre, wrapping over the frame height.
    let band_center = (time * e.roll_speed).rem_euclid(rows as f32);

    for row in 0..rows {
        // Scanline comb dim.
        let mut k = if row % period == 0 {
            1.0 - line_dim
        } else {
            1.0
        };

        // Rolling refresh band: a signed bump that brightens ahead of centre and
        // dims behind, falling off over `band_h`.
        let mut d = row as f32 - band_center;
        // Wrap to the nearest image of the band centre.
        if d > rows as f32 * 0.5 {
            d -= rows as f32;
        } else if d < -(rows as f32) * 0.5 {
            d += rows as f32;
        }
        let n = d / band_h;
        if n.abs() < 1.0 {
            // A smooth ±band: +brighten ahead (n<0), −dim behind (n>0).
            let bump = -n * (1.0 - n.abs()) * 2.0; // peak ±0.5 near edges
            k *= 1.0 + bump * 0.5 * strength;
        }
        k = k.max(0.0);

        if (k - 1.0).abs() < 1e-4 {
            continue; // untouched row, skip the write
        }
        for col in 0..cols {
            let Some(src) = fb.get(col, row) else {
                continue;
            };
            let mut c = *src;
            c.fg = scale_rgb(c.fg, k);
            c.bg = scale_rgb(c.bg, k);
            fb.set(col, row, c);
        }
    }
}
