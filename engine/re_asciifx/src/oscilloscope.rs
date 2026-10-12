//! OSCILLOSCOPE — a Lissajous figure traced by two independent harmonic
//! frequencies (`freq_x`, `freq_y`, small-integer-ish ratios so the curve
//! closes into a recognizable knot rather than drifting), a base sweep
//! speed, and a fading phosphor trail sampled backward from the current
//! time — all pure `f(t)`, no incremental state, so the trail is just many
//! evaluations of the same closed-form curve at `t, t-dt, t-2dt, …`.
//!
//! Rendered onto a [`crate::canvas::SubrowCanvas`]: hundreds of
//! samples per frame, each a soft anti-aliased dot that brightest-wins
//! blends with its neighbours, so the curve reads as one continuous glowing
//! line rather than scattered points — plus a faint graticule grid behind it
//! (not just a bare crosshair) and a bright comet head.

use crate::canvas::SubrowCanvas;
use crate::hash::{angle, hash_unit, range};
use crate::{aspect_dist, secs, set_clipped, AsciiEffect, Rect};
use re_core::oklab::{unpack_linear, LinRgb};
use re_core::{Cell, Framebuffer};
use std::f32::consts::TAU;
use std::time::Duration;

const FREQ_CHOICES: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0];
const BASE_SPEED_RANGE: (f32, f32) = (0.6, 2.2); // rad/s multiplier applied to both freqs

/// How far back in time the phosphor trail reaches, in seconds. Long enough
/// that a slow Lissajous figure still closes into a recognizable knot.
const TRAIL_SECONDS: f32 = 1.4;
/// Sub-sample step: small enough that consecutive samples land under one
/// sub-pixel apart even at the fastest `BASE_SPEED_RANGE` — the "continuous
/// line, not dots" requirement is bought here, not with a thicker glyph.
const TRAIL_DT: f32 = 0.0035;
/// Exponential fade rate applied to `age / TRAIL_SECONDS` — >1 so the head
/// stays bright and the tail falls off quickly rather than lingering as a
/// uniform haze.
const FADE_POWER: f32 = 1.6;

const S_FREQ_X: u64 = 1;
const S_FREQ_Y: u64 = 2;
const S_PHASE_X: u64 = 3;
const S_PHASE_Y: u64 = 4;
const S_SPEED: u64 = 5;
const S_HUE: u64 = 6;

const HEAD_COLOR_BASE: u32 = 0x0060_FF90;
const GRATICULE_COLOR: u32 = 0x0012_2016;
const GRATICULE_AXIS_COLOR: u32 = 0x001C_3020;

/// A traced Lissajous figure.
pub struct Oscilloscope {
    freq_x: f32,
    freq_y: f32,
    phase_x: f32,
    phase_y: f32,
    speed: f32,
    head_color: u32,
}

fn sample(osc: &Oscilloscope, t: f32) -> (f32, f32) {
    let x = (TAU * osc.freq_x * osc.speed * t + osc.phase_x).sin();
    let y = (TAU * osc.freq_y * osc.speed * t + osc.phase_y).sin();
    (x, y)
}

impl Oscilloscope {
    /// Build a deterministic Lissajous-figure trace from `seed`. `_rect` is
    /// unused (the figure is drawn in a normalized `[-1, 1]` space centered
    /// on the target rect by the painter) but kept for the shared
    /// `random(seed, rect)` signature.
    #[must_use]
    pub fn random(seed: u64, _rect: Rect) -> Self {
        let pick = |salt: u64| -> f32 {
            let i = (hash_unit(seed, 0, salt) * FREQ_CHOICES.len() as f64) as usize;
            FREQ_CHOICES[i.min(FREQ_CHOICES.len() - 1)]
        };
        Oscilloscope {
            freq_x: pick(S_FREQ_X),
            freq_y: pick(S_FREQ_Y),
            phase_x: angle(seed, 0, S_PHASE_X),
            phase_y: angle(seed, 0, S_PHASE_Y),
            speed: range(seed, 0, S_SPEED, BASE_SPEED_RANGE.0, BASE_SPEED_RANGE.1),
            head_color: {
                let mut oklch = re_core::oklab::packed_to_oklch(HEAD_COLOR_BASE);
                oklch.h += angle(seed, 0, S_HUE);
                re_core::oklab::oklch_to_packed(oklch)
            },
        }
    }

    /// The `(x, y)` Lissajous frequency ratio this instance rolled.
    #[must_use]
    pub fn freqs(&self) -> (f32, f32) {
        (self.freq_x, self.freq_y)
    }
}

/// Draw a full graticule: a light grid at quarter-divisions of the rect plus
/// a brighter pair of centre axes — a real scope bezel, not a bare crosshair.
fn paint_graticule(fb: &mut Framebuffer, rect: Rect) {
    let (cx, cy) = rect.center();
    for i in 1..4 {
        let col = rect.col0 as f32 + rect.cols as f32 * (i as f32 / 4.0);
        for row in rect.row0..(rect.row0 + rect.rows) {
            set_clipped(
                fb,
                rect,
                col.round() as i64,
                row as i64,
                Cell {
                    glyph: u32::from('\u{2508}'),
                    fg: GRATICULE_COLOR,
                    bg: 0x0000_0000,
                    ..Default::default()
                },
            );
        }
        let row = rect.row0 as f32 + rect.rows as f32 * (i as f32 / 4.0);
        for col in rect.col0..(rect.col0 + rect.cols) {
            set_clipped(
                fb,
                rect,
                col as i64,
                row.round() as i64,
                Cell {
                    glyph: u32::from('\u{2504}'),
                    fg: GRATICULE_COLOR,
                    bg: 0x0000_0000,
                    ..Default::default()
                },
            );
        }
    }
    for col in rect.col0..(rect.col0 + rect.cols) {
        set_clipped(
            fb,
            rect,
            col as i64,
            cy.round() as i64,
            Cell {
                glyph: u32::from('\u{2500}'),
                fg: GRATICULE_AXIS_COLOR,
                bg: 0x0000_0000,
                ..Default::default()
            },
        );
    }
    for row in rect.row0..(rect.row0 + rect.rows) {
        set_clipped(
            fb,
            rect,
            cx.round() as i64,
            row as i64,
            Cell {
                glyph: u32::from('\u{2502}'),
                fg: GRATICULE_AXIS_COLOR,
                bg: 0x0000_0000,
                ..Default::default()
            },
        );
    }
}

impl AsciiEffect for Oscilloscope {
    fn name(&self) -> &'static str {
        "oscilloscope"
    }

    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration) {
        paint_graticule(fb, rect);

        let ax = rect.cols as f32 * 0.42;
        // Amplitude in SUB-ROW units (rows*2): the canvas is twice as tall in
        // sub-rows as the rect is in rows, so the same 0.42 fraction of the
        // rect's height becomes 0.42 of `rows*2`.
        let ay_subrows = rect.rows as f32 * 2.0 * 0.42;

        let head_lin = unpack_linear(self.head_color);
        let mut canvas = SubrowCanvas::new(
            rect,
            LinRgb {
                r: head_lin.r * 0.18,
                g: head_lin.g * 0.18,
                b: head_lin.b * 0.18,
            },
        );

        let secs_t = secs(t);
        let steps = (TRAIL_SECONDS / TRAIL_DT) as i64;
        let mut prev: Option<(f32, f32)> = None;
        for k in (0..steps).rev() {
            let sample_t = secs_t - k as f32 * TRAIL_DT;
            if sample_t < 0.0 {
                continue;
            }
            let (x, y) = sample(self, sample_t);
            let col_f = rect.cols as f32 * 0.5 + x * ax;
            let subrow_f = rect.rows as f32 + y * ay_subrows;
            let age = k as f32 * TRAIL_DT / TRAIL_SECONDS;
            let fade = (1.0 - age).max(0.0).powf(FADE_POWER);
            if let Some((pc, pr)) = prev {
                let d = aspect_dist(pc, pr * 0.5, col_f, subrow_f * 0.5);
                let n = (d.ceil() as i32).clamp(1, 24);
                for i in 0..=n {
                    let f = i as f32 / n as f32;
                    canvas.splat(
                        pc + (col_f - pc) * f,
                        pr + (subrow_f - pr) * f,
                        head_lin,
                        fade,
                    );
                }
            } else {
                canvas.splat(col_f, subrow_f, head_lin, fade);
            }
            prev = Some((col_f, subrow_f));
        }
        // The comet head itself, brightest of all.
        if let Some((pc, pr)) = prev {
            canvas.splat(pc, pr, head_lin, 1.35);
        }
        canvas.blit(
            fb,
            LinRgb {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
        );
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
    fn determinism_same_seed() {
        let a = Oscilloscope::random(31, RECT);
        let b = Oscilloscope::random(31, RECT);
        assert_eq!(a.freqs(), b.freqs());
        assert_eq!(a.speed, b.speed);
    }

    #[test]
    fn different_seeds_diverge_in_frequency() {
        let mut any_diff = false;
        for seed in 0..12u64 {
            let a = Oscilloscope::random(seed, RECT);
            let b = Oscilloscope::random(seed + 1000, RECT);
            if a.freqs() != b.freqs() || (a.speed - b.speed).abs() > 1e-6 {
                any_diff = true;
            }
        }
        assert!(
            any_diff,
            "no seed pair in a 12-sample sweep differed in frequency or speed"
        );
    }

    #[test]
    fn the_curve_stays_within_its_amplitude_bounds() {
        let osc = Oscilloscope::random(7, RECT);
        for ms in (0..8000u64).step_by(37) {
            let (x, y) = sample(&osc, secs(Duration::from_millis(ms)));
            assert!((-1.0..=1.0).contains(&x));
            assert!((-1.0..=1.0).contains(&y));
        }
    }

    #[test]
    fn glyphs_used_are_all_reliable() {
        use re_render::glyph_family::is_reliable;
        for g in ['\u{2500}', '\u{2502}', '\u{2508}', '\u{2504}', '\u{2580}'] {
            assert!(is_reliable(g), "{g:?} not reliable");
        }
    }

    #[test]
    fn paint_stays_inside_its_rect() {
        let mut fb = Framebuffer::new(80, 30);
        let rect = Rect::new(4, 2, 70, 26);
        let osc = Oscilloscope::random(5, rect);
        for ms in [0u64, 250, 1300, 6000] {
            osc.paint(&mut fb, rect, Duration::from_millis(ms));
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
    fn the_trace_fills_a_meaningful_fraction_of_the_rect_not_a_few_dots() {
        let mut fb = Framebuffer::new(70, 26);
        let osc = Oscilloscope::random(3, RECT);
        osc.paint(&mut fb, RECT, Duration::from_millis(3000));
        let lit = (0..26)
            .flat_map(|r| (0..70).map(move |c| (c, r)))
            .filter(|&(c, r)| *fb.get(c, r).unwrap() != re_core::Cell::default())
            .count();
        assert!(
            lit > 120,
            "expected a dense trace + graticule, only {lit} cells lit"
        );
    }
}
