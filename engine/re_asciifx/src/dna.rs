//! DNA HELIX OF BITS — two strands of `0`/`1` glyphs winding around a
//! vertical axis, each row a fixed (per-seed) bit joined by a base-pair rung
//! every few rows, slowly rotating, with depth shading (the strand nearer
//! the viewer is brighter/bolder, the far one dimmer) and occasional
//! per-row "mutation" flashes — a bit briefly flips and flares a bright
//! accent colour before decaying back to normal.
//!
//! ## Geometry
//!
//! Row `r`'s helix angle `theta(r, t) = r * TURN_RATE + rotation_speed * t`
//! places strand A at `theta` and strand B at `theta + PI` (always exactly
//! opposite — a real double helix, not two independent wiggly lines). Each
//! strand's horizontal offset is `amplitude * cos(theta) * perspective(z)`
//! where `z = sin(theta)` is that strand's own depth (`+1` = nearest the
//! viewer, `-1` = farthest) — a small perspective narrowing on the far side
//! plus depth-based colour/weight is what reads as "winding in 3D" rather
//! than a flat sine wave. `paint` is a pure function of `(self, t)`: no
//! incremental state, so any `t` reproduces the same frame directly.
//!
//! One glyph sample per OUTPUT ROW (not sub-row, unlike the half-block
//! effects) — a strand point is a discrete bit character, not a colour
//! field, so half-block doubling would not add real detail here.

use crate::hash::{angle, chance, hash_unit, range};
use crate::{secs, set_clipped, AsciiEffect, Rect};
use re_core::oklab::{lerp_linear, oklch_to_packed, packed_to_oklch, unpack_linear};
use re_core::{Cell, CellStyle, Framebuffer};
use std::f32::consts::{PI, TAU};
use std::time::Duration;

// ─────────────────────────────── tuning ──────────────────────────────────

/// Full turns of the helix over the rect's own height — the "how tightly
/// wound" knob, its own independent per-seed frequency.
const TURNS_RANGE: (f32, f32) = (1.4, 3.2);
const ROTATION_SPEED_RANGE: (f32, f32) = (0.3, 1.0); // rad/s, continuous spin
const AMPLITUDE_FRAC_RANGE: (f32, f32) = (0.20, 0.36); // fraction of rect width
/// How much the far side of the helix narrows relative to the near side —
/// `0` = no perspective (flat), higher = a more pronounced 3D read.
const PERSPECTIVE_K_RANGE: (f32, f32) = (0.15, 0.40);
/// Rows between base-pair rungs.
const RUNG_SPACING_RANGE: (usize, usize) = (2, 4);
/// A row's own mutation-flash period/duration — independent per row (via a
/// per-row hash draw), so flashes never fall into a visible lockstep rhythm.
const MUTATION_PERIOD_RANGE: (f32, f32) = (2.5, 7.0);
const MUTATION_FLASH_HALF_WIDTH: f32 = 0.035; // fraction of the period
const MUTATION_CHANCE_PER_ROW: f32 = 0.55; // not every row ever mutates

const S_TURNS: u64 = 1;
const S_ROTATION: u64 = 2;
const S_AMPLITUDE: u64 = 3;
const S_PERSPECTIVE: u64 = 4;
const S_RUNG_SPACING: u64 = 5;
const S_HUE_BASE: u64 = 6;
const S_ROW_BIT: u64 = 100;
const S_ROW_MUTATES: u64 = 101;
const S_ROW_MUT_PERIOD: u64 = 102;
const S_ROW_MUT_PHASE: u64 = 103;
const S_RUNG_HUE: u64 = 104;

const FRONT_COLOR: u32 = 0x0060_F0FF;
const BACK_COLOR: u32 = 0x0014_3048;
const RUNG_COLOR_A: u32 = 0x00E0_60C0;
const RUNG_COLOR_B: u32 = 0x0060_E0A0;
const FLASH_COLOR: u32 = 0x00FF_F0A0;

/// A winding double helix of bit glyphs.
pub struct Dna {
    seed: u64,
    turns: f32,
    rotation_speed: f32,
    amplitude_frac: f32,
    perspective_k: f32,
    rung_spacing: usize,
    /// Every base colour below, hue-rotated by this seed's own draw — same
    /// technique `radar`/`oscilloscope` use so different seeds are a
    /// genuinely different palette, not just a phase-shifted copy of one.
    front_color: u32,
    back_color: u32,
    rung_color_a: u32,
    rung_color_b: u32,
    flash_color: u32,
}

fn perspective(z: f32, k: f32) -> f32 {
    1.0 - k * (1.0 - z) * 0.5
}

fn hue_rotate(base: u32, delta: f32) -> u32 {
    let mut oklch = packed_to_oklch(base);
    oklch.h += delta;
    oklch_to_packed(oklch)
}

impl Dna {
    /// Build a deterministic DNA-helix effect from `seed`. `_rect` is unused
    /// (the helix scales itself from its own turn/pitch parameters, not the
    /// target rect) but kept for a uniform `random(seed, rect)` constructor
    /// signature across every effect in this crate.
    #[must_use]
    pub fn random(seed: u64, _rect: Rect) -> Self {
        let hue_base = angle(seed, 0, S_HUE_BASE);
        Dna {
            seed,
            turns: range(seed, 0, S_TURNS, TURNS_RANGE.0, TURNS_RANGE.1),
            rotation_speed: range(
                seed,
                0,
                S_ROTATION,
                ROTATION_SPEED_RANGE.0,
                ROTATION_SPEED_RANGE.1,
            ),
            amplitude_frac: range(
                seed,
                0,
                S_AMPLITUDE,
                AMPLITUDE_FRAC_RANGE.0,
                AMPLITUDE_FRAC_RANGE.1,
            ),
            perspective_k: range(
                seed,
                0,
                S_PERSPECTIVE,
                PERSPECTIVE_K_RANGE.0,
                PERSPECTIVE_K_RANGE.1,
            ),
            rung_spacing: RUNG_SPACING_RANGE.0
                + (hash_unit(seed, 0, S_RUNG_SPACING)
                    * (RUNG_SPACING_RANGE.1 - RUNG_SPACING_RANGE.0 + 1) as f64)
                    as usize,
            front_color: hue_rotate(FRONT_COLOR, hue_base),
            back_color: hue_rotate(BACK_COLOR, hue_base),
            rung_color_a: hue_rotate(RUNG_COLOR_A, hue_base),
            rung_color_b: hue_rotate(RUNG_COLOR_B, hue_base),
            flash_color: hue_rotate(FLASH_COLOR, hue_base * 0.3),
        }
    }

    /// This row's helix angle at time `t`.
    fn theta(&self, row: i64, t: f32) -> f32 {
        row as f32 * (self.turns * TAU / 24.0) + self.rotation_speed * t
    }

    /// `(col_offset, z)` for one strand at `theta` — `col_offset` in cells
    /// from the axis, `z` in `[-1, 1]` (depth, `+1` nearest).
    fn strand_point(&self, theta: f32, half_width: f32) -> (f32, f32) {
        let z = theta.sin();
        let x = half_width * self.amplitude_frac * theta.cos() * perspective(z, self.perspective_k);
        (x, z)
    }

    /// Fixed (per-seed, per-row) base bit — `'0'` or `'1'` — before any
    /// mutation flash flips it for one instant.
    fn base_bit(&self, row: i64) -> char {
        if chance(self.seed, row.unsigned_abs(), S_ROW_BIT, 0.5) {
            '1'
        } else {
            '0'
        }
    }

    /// Whether this row ever mutates at all, and if so its flash brightness
    /// in `[0, 1]` at time `t` (`0` = quiescent, `1` = flash peak).
    fn mutation_brightness(&self, row: i64, t: f32) -> f32 {
        let id = row.unsigned_abs();
        if !chance(self.seed, id, S_ROW_MUTATES, MUTATION_CHANCE_PER_ROW) {
            return 0.0;
        }
        let period = range(
            self.seed,
            id,
            S_ROW_MUT_PERIOD,
            MUTATION_PERIOD_RANGE.0,
            MUTATION_PERIOD_RANGE.1,
        )
        .max(0.1);
        let phase0 = range(self.seed, id, S_ROW_MUT_PHASE, 0.0, 1.0);
        let phase = (t / period + phase0).rem_euclid(1.0);
        let dist = phase.min(1.0 - phase);
        if dist >= MUTATION_FLASH_HALF_WIDTH {
            0.0
        } else {
            let x = (1.0 - dist / MUTATION_FLASH_HALF_WIDTH).clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        }
    }
}

impl Dna {
    fn depth_color(&self, z: f32) -> u32 {
        let t = (z + 1.0) * 0.5;
        let front = unpack_linear(self.front_color);
        let back = unpack_linear(self.back_color);
        re_core::oklab::pack_linear(lerp_linear(back, front, t))
    }

    fn mix_flash(&self, base: u32, flash_brightness: f32) -> u32 {
        if flash_brightness <= 0.0 {
            return base;
        }
        let a = unpack_linear(base);
        let f = unpack_linear(self.flash_color);
        re_core::oklab::pack_linear(lerp_linear(a, f, flash_brightness))
    }
}

impl AsciiEffect for Dna {
    fn name(&self) -> &'static str {
        "dna"
    }

    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration) {
        let secs_t = secs(t);
        let (ccx, _) = rect.center();
        let half_width = rect.cols as f32 * 0.5;

        for row_u in rect.row0..(rect.row0 + rect.rows) {
            let row = row_u as i64;
            let theta = self.theta(row, secs_t);
            let (xa, za) = self.strand_point(theta, half_width);
            let (xb, zb) = self.strand_point(theta + PI, half_width);
            let col_a = (ccx + xa).round() as i64;
            let col_b = (ccx + xb).round() as i64;

            let flash = self.mutation_brightness(row, secs_t);
            let bit_a = self.base_bit(row);
            // Strand B reads the complementary bit — a real double helix
            // pairs complementary bases, and it also makes the two strands
            // visually distinguishable at a glance even where their depth
            // shading is similar.
            let bit_b = if bit_a == '0' { '1' } else { '0' };

            // Base-pair rung: every `rung_spacing`-th row, a connecting line
            // between the two strands (only when they are far enough apart
            // to draw one — near a crossing the strands nearly coincide).
            if row.rem_euclid(self.rung_spacing as i64) == 0 && (col_b - col_a).abs() >= 2 {
                let (lo, hi) = if col_a < col_b {
                    (col_a, col_b)
                } else {
                    (col_b, col_a)
                };
                let rung_hue_pick = chance(self.seed, row.unsigned_abs(), S_RUNG_HUE, 0.5);
                let rung_color = self.mix_flash(
                    if rung_hue_pick {
                        self.rung_color_a
                    } else {
                        self.rung_color_b
                    },
                    flash,
                );
                for c in (lo + 1)..hi {
                    set_clipped(
                        fb,
                        rect,
                        c,
                        row,
                        Cell {
                            glyph: u32::from('\u{2500}'),
                            fg: rung_color,
                            bg: 0x0000_0000,
                            ..Default::default()
                        },
                    );
                }
            }

            // Only the NEARER strand's bit actually flips during a flash
            // (a mutation "popping out" toward the viewer reads more clearly
            // than both strands changing at once); the farther strand just
            // tints toward the flash colour, dampened.
            let near_is_a = za >= zb;
            let disp_a = if near_is_a && flash > 0.5 {
                bit_b
            } else {
                bit_a
            };
            let disp_b = if !near_is_a && flash > 0.5 {
                bit_a
            } else {
                bit_b
            };

            // Depth ordering: draw the FARTHER strand first, the nearer one
            // second, so at a crossing the near strand visually occludes the
            // far one — real depth compositing, not an accident of loop order.
            let strand_a = (
                col_a,
                za,
                disp_a,
                if near_is_a { flash } else { flash * 0.5 },
            );
            let strand_b = (
                col_b,
                zb,
                disp_b,
                if near_is_a { flash * 0.5 } else { flash },
            );
            let (far, near) = if za <= zb {
                (strand_a, strand_b)
            } else {
                (strand_b, strand_a)
            };
            for (col, z, bit, own_flash) in [far, near] {
                let color = self.mix_flash(self.depth_color(z), own_flash);
                let style = if z > 0.15 {
                    CellStyle::NONE.with_bold()
                } else {
                    CellStyle::NONE
                };
                set_clipped(
                    fb,
                    rect,
                    col,
                    row,
                    Cell {
                        glyph: u32::from(bit),
                        fg: color,
                        bg: 0x0000_0000,
                        style,
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECT: Rect = Rect {
        col0: 0,
        row0: 0,
        cols: 60,
        rows: 40,
    };

    #[test]
    fn determinism_same_seed() {
        let a = Dna::random(21, RECT);
        let b = Dna::random(21, RECT);
        assert_eq!(a.turns, b.turns);
        assert_eq!(a.rotation_speed, b.rotation_speed);
        assert_eq!(a.rung_spacing, b.rung_spacing);
    }

    #[test]
    fn different_seeds_diverge_in_every_frequency() {
        let a = Dna::random(1, RECT);
        let b = Dna::random(2, RECT);
        assert_ne!(a.turns, b.turns);
        assert_ne!(a.rotation_speed, b.rotation_speed);
        assert_ne!(a.amplitude_frac, b.amplitude_frac);
        assert_ne!(a.perspective_k, b.perspective_k);
    }

    #[test]
    fn strands_are_always_exactly_opposite_in_depth() {
        let dna = Dna::random(9, RECT);
        let half_width = RECT.cols as f32 * 0.5;
        for row in 0..40i64 {
            let theta = dna.theta(row, 1.23);
            let (_, za) = dna.strand_point(theta, half_width);
            let (_, zb) = dna.strand_point(theta + PI, half_width);
            assert!(
                (za + zb).abs() < 1e-4,
                "row {row}: strand depths {za}/{zb} are not opposite"
            );
        }
    }

    #[test]
    fn mutation_brightness_stays_in_unit_range_and_some_row_flashes() {
        let dna = Dna::random(5, RECT);
        let mut saw_flash = false;
        for row in 0..40i64 {
            for ms in (0..8000u64).step_by(97) {
                let b = dna.mutation_brightness(row, ms as f32 / 1000.0);
                assert!(
                    (0.0..=1.0).contains(&b),
                    "row {row} t={ms}ms: brightness {b} out of range"
                );
                if b > 0.9 {
                    saw_flash = true;
                }
            }
        }
        assert!(
            saw_flash,
            "no row ever reached a near-peak mutation flash across an 8s sweep"
        );
    }

    #[test]
    fn not_every_row_mutates() {
        let dna = Dna::random(5, RECT);
        let mutates = (0..40i64)
            .filter(|&row| {
                chance(
                    dna.seed,
                    row.unsigned_abs(),
                    S_ROW_MUTATES,
                    MUTATION_CHANCE_PER_ROW,
                )
            })
            .count();
        assert!(mutates > 0, "no row ever mutates");
        assert!(
            mutates < 40,
            "every row mutates — no quiescent rows for contrast"
        );
    }

    #[test]
    fn paint_stays_inside_its_rect() {
        let mut fb = Framebuffer::new(70, 44);
        let rect = Rect::new(4, 2, 60, 40);
        let dna = Dna::random(6, rect);
        for ms in [0u64, 300, 2200, 7000] {
            dna.paint(&mut fb, rect, Duration::from_millis(ms));
        }
        for row in 0..44 {
            for col in 0..70 {
                if !rect.contains(col as i64, row as i64) {
                    assert_eq!(*fb.get(col, row).unwrap(), re_core::Cell::default());
                }
            }
        }
    }

    #[test]
    fn glyphs_used_are_all_reliable() {
        use re_render::glyph_family::is_reliable;
        for g in ['0', '1', '\u{2500}'] {
            assert!(is_reliable(g), "{g:?} not reliable");
        }
    }

    #[test]
    fn the_helix_lights_a_meaningful_area_of_the_rect() {
        let mut fb = Framebuffer::new(60, 40);
        let dna = Dna::random(11, RECT);
        dna.paint(&mut fb, RECT, Duration::from_millis(1700));
        let lit = (0..40)
            .flat_map(|r| (0..60).map(move |c| (c, r)))
            .filter(|&(c, r)| *fb.get(c, r).unwrap() != re_core::Cell::default())
            .count();
        assert!(
            lit > 80,
            "expected two winding strands + rungs to light a meaningful area, only {lit} cells lit"
        );
    }
}
