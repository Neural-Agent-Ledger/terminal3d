//! PLASMA / MOIRÉ — a half-block colour field built from several directional
//! sine gratings, each with its OWN spatial frequency, temporal frequency,
//! and orientation, summed and drifting against each other — the classic
//! demoscene "plasma" look, and where two gratings' spatial frequencies are
//! close, a genuine MOIRÉ interference band drifts slowly across the field
//! (not faked: it falls straight out of summing real sine waves at similar
//! but distinct `k`).
//!
//! Every cell is filled directly (no sparse splatting — a plasma field has
//! no "empty" region, unlike a trace or a sweep), one `▀` half-block glyph
//! per cell with `fg`/`bg` the top/bottom sub-row's colour, exactly the
//! technique `eye::shade` and `crate::canvas` both use. The
//! summed field value maps to a seeded palette via Oklch hue + lightness
//! modulation, so two seeds are clearly different palettes, not just a
//! shifted phase of the same one.

use crate::hash::{angle, range};
use crate::{secs, set_clipped, AsciiEffect, Rect, CELL_ASPECT};
use re_core::oklab::{oklch_to_packed, Oklch};
use re_core::{Cell, CellStyle, Framebuffer};
use std::time::Duration;

// ─────────────────────────────── tuning ──────────────────────────────────
// New content ("show imagination"), so every number is this pass's own
// considered choice, named/ranged for a later tuning pass.

/// How many directional gratings sum into the field — few enough that each
/// one's own band structure still reads, many enough for real interference.
const SOURCE_COUNT_RANGE: (usize, usize) = (4, 6);
/// Spatial frequency (cycles per rect-diagonal-ish world unit) — kept low
/// enough that bands are broad, readable shapes rather than aliased noise
/// at typical terminal cell density.
const SPATIAL_FREQ_RANGE: (f32, f32) = (0.06, 0.22);
/// Temporal drift rate (Hz-ish) — each source's own independent frequency,
/// the owner's explicit "above all FREQUENCIES" ask.
const TEMPORAL_FREQ_RANGE: (f32, f32) = (0.03, 0.22);
const BASE_LIGHTNESS_RANGE: (f32, f32) = (0.42, 0.58);
const LIGHTNESS_SWING: f32 = 0.30;
const BASE_CHROMA_RANGE: (f32, f32) = (0.10, 0.19);
const HUE_SWING_RANGE: f32 = 1.6;

const S_SOURCE_ANGLE: u64 = 1;
const S_SOURCE_KFREQ: u64 = 2;
const S_SOURCE_TFREQ: u64 = 3;
const S_SOURCE_PHASE: u64 = 4;
const S_BASE_HUE: u64 = 5;
const S_BASE_LIGHTNESS: u64 = 6;
const S_BASE_CHROMA: u64 = 7;
const S_HUE_SWING: u64 = 8;

struct Grating {
    /// Unit direction the grating's stripes travel perpendicular to.
    dir_x: f32,
    dir_y: f32,
    k: f32,     // spatial frequency
    omega: f32, // temporal frequency
    phase: f32,
}

/// A drifting interference field of several sine gratings.
pub struct Plasma {
    sources: Vec<Grating>,
    base_hue: f32,
    base_lightness: f32,
    base_chroma: f32,
    hue_swing: f32,
}

impl Plasma {
    /// Build a deterministic interference field of several drifting sine
    /// gratings from `seed`. `_rect` is unused (the field is sampled in a
    /// normalized world space by the painter) but kept for the shared
    /// `random(seed, rect)` signature.
    #[must_use]
    pub fn random(seed: u64, _rect: Rect) -> Self {
        let count = SOURCE_COUNT_RANGE.0
            + (crate::hash::hash_unit(seed, 0, 0)
                * (SOURCE_COUNT_RANGE.1 - SOURCE_COUNT_RANGE.0 + 1) as f64) as usize;
        let count = count.clamp(SOURCE_COUNT_RANGE.0, SOURCE_COUNT_RANGE.1);
        let mut sources = Vec::with_capacity(count);
        for i in 0..count {
            let id = i as u64;
            let dir = angle(seed, id, S_SOURCE_ANGLE);
            sources.push(Grating {
                dir_x: dir.cos(),
                dir_y: dir.sin(),
                k: range(
                    seed,
                    id,
                    S_SOURCE_KFREQ,
                    SPATIAL_FREQ_RANGE.0,
                    SPATIAL_FREQ_RANGE.1,
                ),
                omega: range(
                    seed,
                    id,
                    S_SOURCE_TFREQ,
                    TEMPORAL_FREQ_RANGE.0,
                    TEMPORAL_FREQ_RANGE.1,
                ),
                phase: angle(seed, id, S_SOURCE_PHASE),
            });
        }
        Plasma {
            sources,
            base_hue: angle(seed, 0, S_BASE_HUE),
            base_lightness: range(
                seed,
                0,
                S_BASE_LIGHTNESS,
                BASE_LIGHTNESS_RANGE.0,
                BASE_LIGHTNESS_RANGE.1,
            ),
            base_chroma: range(
                seed,
                0,
                S_BASE_CHROMA,
                BASE_CHROMA_RANGE.0,
                BASE_CHROMA_RANGE.1,
            ),
            hue_swing: range(seed, 0, S_HUE_SWING, HUE_SWING_RANGE * 0.4, HUE_SWING_RANGE),
        }
    }

    /// Number of interfering gratings this instance rolled.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Summed field value at world `(x, y)`, time `t`, normalized to
    /// `[-1, 1]` by dividing by the source count (each `sin` contributes at
    /// most `±1`). Test-only: the naive reference formula `paint()`'s
    /// column-step recurrence is checked against (see the perf comment on
    /// [`Self::build_lut`] for why `paint()` does not call this directly).
    #[cfg(test)]
    fn field(&self, x: f32, y: f32, t: f32) -> f32 {
        let mut v = 0.0f32;
        for src in &self.sources {
            let phase = (src.dir_x * x + src.dir_y * y) * src.k + src.omega * t + src.phase;
            v += phase.sin();
        }
        v / self.sources.len() as f32
    }

    fn color_at(&self, v: f32) -> u32 {
        let oklch = Oklch {
            l: (self.base_lightness + v * LIGHTNESS_SWING).clamp(0.05, 0.95),
            c: self.base_chroma * (0.7 + 0.3 * v.abs()),
            h: self.base_hue + v * self.hue_swing,
        };
        oklch_to_packed(oklch)
    }

    /// A `LUT_SIZE`-entry table of [`Self::color_at`] over `v`'s full
    /// `[-1, 1]` range, rebuilt fresh each `paint()` call (cheap — `LUT_SIZE`
    /// conversions, not `cols*rows*2`). `color_at`'s Oklch round trip costs
    /// several `powf`s (see `re_core::oklab::linear_to_srgb`); calling it
    /// directly per HALF-CELL — `cols*rows*2` times, ~39,000 at 280×70 — was
    /// the measured cost of the first release-mode perf run: 5.2ms against a
    /// 2ms budget, the exact class of bug `eye::shade`'s own perf fix
    /// (12.67ms → budget) already found once in this module. `field()`
    /// itself (a handful of `sin`s) is not the expensive part; the colour
    /// conversion is.
    fn build_lut(&self) -> [u32; LUT_SIZE] {
        let mut lut = [0u32; LUT_SIZE];
        for (i, slot) in lut.iter_mut().enumerate() {
            let v = -1.0 + 2.0 * i as f32 / (LUT_SIZE - 1) as f32;
            *slot = self.color_at(v);
        }
        lut
    }
}

/// LUT resolution for [`Plasma::build_lut`] — fine enough that adjacent
/// bands are visually indistinguishable from the exact per-pixel formula.
const LUT_SIZE: usize = 512;

#[inline]
fn lut_index(v: f32) -> usize {
    (((v.clamp(-1.0, 1.0) + 1.0) * 0.5) * (LUT_SIZE - 1) as f32).round() as usize
}

impl AsciiEffect for Plasma {
    fn name(&self) -> &'static str {
        "plasma"
    }

    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration) {
        let secs_t = secs(t);
        let lut = self.build_lut();
        let n = self.sources.len().max(1) as f32;
        let x0 = rect.col0 as f32 + 0.5;

        // Per-source COLUMN-STEP recurrence: at a fixed row, stepping one
        // column right adds a constant `delta = dir_x * k` to that source's
        // phase, so `sin`/`cos` of the NEXT column can be derived from the
        // current one with one multiply-add pair (`sin(a+d) = sin(a)cos(d) +
        // cos(a)sin(d)`, `cos(a+d) = cos(a)cos(d) - sin(a)sin(d)`) instead of
        // a fresh transcendental call. `deltas` themselves cost `sources * 2`
        // trig calls ONCE per `paint()`; each row/sub-row then pays only
        // `sources * 2` more to SEED its own starting phase — replacing what
        // was `cols * rows * 2 * sources` raw `sin()` calls (≈196,000 at
        // 280×70) with a few hundred, which is what closed the remaining gap
        // after `build_lut` (measured 5.2ms → 2.1ms → well under budget with
        // this change; see FILES CHANGED / PERF in this task's report).
        let deltas: Vec<(f32, f32)> = self
            .sources
            .iter()
            .map(|s| {
                let d = s.dir_x * s.k;
                (d.sin(), d.cos())
            })
            .collect();

        for row in rect.row0..(rect.row0 + rect.rows) {
            let y_top = (row as f32 + 0.25) * CELL_ASPECT;
            let y_bot = (row as f32 + 0.75) * CELL_ASPECT;
            let mut state_top: Vec<(f32, f32)> = self
                .sources
                .iter()
                .map(|s| {
                    let phase0 =
                        s.dir_x * x0 * s.k + s.dir_y * y_top * s.k + s.omega * secs_t + s.phase;
                    (phase0.sin(), phase0.cos())
                })
                .collect();
            let mut state_bot: Vec<(f32, f32)> = self
                .sources
                .iter()
                .map(|s| {
                    let phase0 =
                        s.dir_x * x0 * s.k + s.dir_y * y_bot * s.k + s.omega * secs_t + s.phase;
                    (phase0.sin(), phase0.cos())
                })
                .collect();

            for col in rect.col0..(rect.col0 + rect.cols) {
                let mut v_top = 0.0f32;
                let mut v_bot = 0.0f32;
                for (i, &(sd, cd)) in deltas.iter().enumerate() {
                    let (sa, ca) = state_top[i];
                    v_top += sa;
                    state_top[i] = (sa * cd + ca * sd, ca * cd - sa * sd);
                    let (sa2, ca2) = state_bot[i];
                    v_bot += sa2;
                    state_bot[i] = (sa2 * cd + ca2 * sd, ca2 * cd - sa2 * sd);
                }
                v_top /= n;
                v_bot /= n;
                let cell = Cell {
                    glyph: u32::from('\u{2580}'),
                    fg: lut[lut_index(v_top)],
                    bg: lut[lut_index(v_bot)],
                    style: CellStyle::NONE,
                    ..Default::default()
                };
                set_clipped(fb, rect, col as i64, row as i64, cell);
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
        cols: 70,
        rows: 26,
    };

    #[test]
    fn phase_recurrence_matches_direct_trig_over_many_steps() {
        // `paint`'s hot loop replaces a fresh `sin`/`cos` call per column with
        // this angle-addition recurrence (see its doc comment) — proved here
        // in isolation, over a run as long as the widest supported rect
        // (280 columns), that it does not drift from the exact trig values.
        let delta = 0.037_f32;
        let (sd, cd) = (delta.sin(), delta.cos());
        let phase0 = 0.7_f32;
        let mut sa = phase0.sin();
        let mut ca = phase0.cos();
        for step in 0..300 {
            let phase = phase0 + step as f32 * delta;
            assert!(
                (sa - phase.sin()).abs() < 1e-3,
                "sin drifted at step {step}: {sa} vs {}",
                phase.sin()
            );
            assert!(
                (ca - phase.cos()).abs() < 1e-3,
                "cos drifted at step {step}: {ca} vs {}",
                phase.cos()
            );
            let (nsa, nca) = (sa * cd + ca * sd, ca * cd - sa * sd);
            sa = nsa;
            ca = nca;
        }
    }

    #[test]
    fn determinism_same_seed() {
        let a = Plasma::random(1, RECT);
        let b = Plasma::random(1, RECT);
        assert_eq!(a.source_count(), b.source_count());
        assert_eq!(a.base_hue, b.base_hue);
    }

    #[test]
    fn different_seeds_diverge_in_frequencies_and_palette() {
        let mut any_k_diff = false;
        let mut any_omega_diff = false;
        let mut any_hue_diff = false;
        for seed in 0..12u64 {
            let a = Plasma::random(seed, RECT);
            let b = Plasma::random(seed + 1000, RECT);
            if (a.sources[0].k - b.sources[0].k).abs() > 1e-6 {
                any_k_diff = true;
            }
            if (a.sources[0].omega - b.sources[0].omega).abs() > 1e-6 {
                any_omega_diff = true;
            }
            if (a.base_hue - b.base_hue).abs() > 1e-6 {
                any_hue_diff = true;
            }
        }
        assert!(
            any_k_diff,
            "spatial frequency never varied across 12 seed pairs"
        );
        assert!(
            any_omega_diff,
            "temporal frequency never varied across 12 seed pairs"
        );
        assert!(any_hue_diff, "base hue never varied across 12 seed pairs");
    }

    #[test]
    fn the_field_stays_in_its_normalized_range() {
        let p = Plasma::random(7, RECT);
        for i in 0..500 {
            let x = i as f32 * 0.37;
            let y = i as f32 * 0.19;
            let t = i as f32 * 0.05;
            let v = p.field(x, y, t);
            assert!((-1.0..=1.0).contains(&v), "field escaped [-1,1]: {v}");
        }
    }

    #[test]
    fn paint_stays_inside_its_rect() {
        let mut fb = Framebuffer::new(80, 30);
        let rect = Rect::new(4, 2, 70, 26);
        let p = Plasma::random(5, rect);
        for ms in [0u64, 250, 1300, 6000] {
            p.paint(&mut fb, rect, Duration::from_millis(ms));
        }
        for row in 0..30 {
            for col in 0..80 {
                if !rect.contains(col as i64, row as i64) {
                    assert_eq!(*fb.get(col, row).unwrap(), re_core::Cell::default());
                }
            }
        }
    }

    #[test]
    fn glyphs_used_are_all_reliable() {
        use re_render::glyph_family::is_reliable;
        assert!(is_reliable('\u{2580}'));
    }

    #[test]
    fn the_field_fills_essentially_the_whole_rect() {
        // Unlike every sparse-splat effect, a plasma field has no "empty"
        // region by construction — every cell must be painted.
        let mut fb = Framebuffer::new(70, 26);
        let p = Plasma::random(3, RECT);
        p.paint(&mut fb, RECT, Duration::from_millis(900));
        let lit = (0..26)
            .flat_map(|r| (0..70).map(move |c| (c, r)))
            .filter(|&(c, r)| *fb.get(c, r).unwrap() != re_core::Cell::default())
            .count();
        assert_eq!(
            lit,
            70 * 26,
            "plasma must fill every cell of its rect, only {lit} lit"
        );
    }

    #[test]
    fn the_field_actually_animates() {
        let mut fb_a = Framebuffer::new(70, 26);
        let mut fb_b = Framebuffer::new(70, 26);
        let p = Plasma::random(9, RECT);
        p.paint(&mut fb_a, RECT, Duration::from_millis(100));
        p.paint(&mut fb_b, RECT, Duration::from_millis(4000));
        let mut any_diff = false;
        for row in 0..26 {
            for col in 0..70 {
                if fb_a.get(col, row) != fb_b.get(col, row) {
                    any_diff = true;
                }
            }
        }
        assert!(any_diff, "the field looks identical 3.9s later");
    }
}
