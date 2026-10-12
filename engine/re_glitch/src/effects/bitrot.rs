//! Bit-rot / char corruption: a fraction of cells become garbage glyphs.
//!
//! Per cell, a hashed `[0,1)` value below the (intensity-scaled) `density` marks
//! the cell corrupt: its glyph is replaced by a hashed pick from a mojibake set
//! and its fg tinted toward a sick decay colour. With `spread > 0` the threshold
//! is RAISED for cells that already have corrupt 4-neighbours, giving corruption
//! a contagion / outward-decay feel.
//!
//! The spread reads the WORKING buffer left-to-right/top-to-bottom on purpose:
//! once a cell is corrupted, later neighbours see it, so rot bleeds across the
//! frame in a single deterministic pass (no RNG, no per-frame state).

use re_core::Framebuffer;

use super::{color, BitRotEffect};
use crate::hash::{hash, tslice, unit};

/// Mojibake / garbage glyph pool — a mix of box-drawing, blocks, and symbols
/// that read as "corrupted data" in a terminal. Kept ASCII-safe-ish + common
/// Unicode so any truecolor terminal renders them.
const GARBAGE: &[char] = &[
    '▓', '▒', '░', '█', '▚', '▞', '╳', '◊', '¶', '§', '※', '�', '#', '%', '@', '&', '$', '¤', '×',
    '╬', '╪', '╫', '■', '□', '▌', '▐', '?', '!', '0', '1',
];

/// A sick decay tint the corrupt fg is pulled toward (a dim necrotic green).
const DECAY_TINT: (u8, u8, u8) = (90, 130, 70);

pub(crate) fn apply(fb: &mut Framebuffer, e: &BitRotEffect, master: f32, time: f32, seed: u64) {
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
    let spread = e.spread.max(0.0);

    // Track corruption so spread can read already-decided neighbours this pass.
    let mut corrupt = vec![false; cols * rows];

    for row in 0..rows {
        for col in 0..cols {
            // Count already-corrupt 4-neighbours decided earlier this pass
            // (left + up are filled; right + down are not yet, so contagion
            // flows down-right — a directional bloom, which reads fine).
            let mut neigh = 0u32;
            if col > 0 && corrupt[row * cols + (col - 1)] {
                neigh += 1;
            }
            if row > 0 && corrupt[(row - 1) * cols + col] {
                neigh += 1;
            }
            let local_density = (density + spread * neigh as f32 * density).clamp(0.0, 1.0);

            let h = hash(&[salt, col as u64, row as u64, slot]);
            if unit(h) >= local_density {
                continue;
            }
            corrupt[row * cols + col] = true;

            let Some(src) = fb.get(col, row) else {
                continue;
            };
            let mut c = *src;
            let g = GARBAGE[(h >> 24) as usize % GARBAGE.len()];
            c.glyph = u32::from(g);
            // Pull fg toward the decay tint (half-blend) so corruption reads in
            // colour too, not only in glyph.
            let (r, gg, b) = color::unpack(c.fg);
            c.fg = color::pack(
                ((r as u16 + DECAY_TINT.0 as u16) / 2) as u8,
                ((gg as u16 + DECAY_TINT.1 as u16) / 2) as u8,
                ((b as u16 + DECAY_TINT.2 as u16) / 2) as u8,
            );
            fb.set(col, row, c);
        }
    }
}
