//! HEARTBEAT MONITOR — a scrolling ECG trace: a closed-form PQRST waveform
//! (five Gaussian bumps at fixed relative phase offsets — the recognizable
//! "beat" shape) repeating at the instance's own `bpm`, scrolled across the
//! rect at an independently-randomized `scroll_speed` so the on-screen
//! period and the scroll rate are never simply proportional.
//!
//! `value_at(sample_t)` is a pure function — every sample is computed
//! directly from absolute time, not from a scrolled buffer — so a wider rect
//! just shows more history for free, and any `t` can be rendered in
//! isolation. Painted onto a [`crate::canvas::SubrowCanvas`] at
//! several sub-column samples per cell, connected into continuous
//! interpolated segments, with the trace fading from bright (the newest
//! sample, at the right edge) to dim (the oldest, scrolled off toward the
//! left) — real phosphor-persistence monitors hold the newest sweep
//! brightest and let earlier ones decay.

use crate::canvas::SubrowCanvas;
use crate::hash::range;
use crate::{put_str_clipped, secs, set_clipped, AsciiEffect, Rect};
use re_core::oklab::{unpack_linear, LinRgb};
use re_core::{Cell, Framebuffer};
use std::time::Duration;

const BPM_RANGE: (f32, f32) = (52.0, 128.0);
const SCROLL_SPEED_RANGE: (f32, f32) = (14.0, 34.0); // cells / second
const AMPLITUDE_ROWS_RANGE: (f32, f32) = (4.0, 9.0);

const S_BPM: u64 = 1;
const S_SCROLL: u64 = 2;
const S_AMP: u64 = 3;
const S_PHASE: u64 = 4;

const TRACE_COLOR: u32 = 0x0040_FF70;
const GRID_MAJOR: u32 = 0x0014_2818;
const GRID_MINOR: u32 = 0x000C_1810;
const LABEL_COLOR: u32 = 0x0060_FF90;

/// Sub-column samples per cell — the "dense sampling" that turns the trace
/// into a continuous line instead of one dot per integer column.
const SAMPLES_PER_COL: i32 = 3;
/// The oldest visible sample (left edge) is faded to this fraction of full
/// brightness; the newest (right edge) is always `1.0`.
const OLDEST_FADE: f32 = 0.25;

/// One PQRST bump: `(phase_center, width, amplitude)`. `phase` is a
/// fraction of one full beat cycle, `0..1`.
const BUMPS: [(f32, f32, f32); 5] = [
    (0.12, 0.020, 0.15),  // P
    (0.38, 0.010, -0.15), // Q
    (0.40, 0.012, 1.0),   // R
    (0.43, 0.012, -0.30), // S
    (0.65, 0.050, 0.25),  // T
];

fn ecg_waveform(phase: f32) -> f32 {
    let p = phase.rem_euclid(1.0);
    let mut v = 0.0f32;
    for &(center, width, amp) in &BUMPS {
        // Shortest signed distance on a unit circle, so a bump near phase
        // 0/1 still reads correctly for samples on the other side of the seam.
        let raw = p - center;
        let d = raw - raw.round();
        v += amp * (-(d * d) / (2.0 * width * width)).exp();
    }
    v
}

/// A scrolling ECG monitor.
pub struct Heartbeat {
    bpm: f32,
    scroll_speed: f32,
    amplitude_rows: f32,
    phase0: f32,
}

impl Heartbeat {
    /// Build a deterministic ECG-style trace from `seed`. `_rect` is unused
    /// (the trace scrolls at a fixed rows-per-second rate independent of the
    /// target size) but kept for the shared `random(seed, rect)` signature.
    #[must_use]
    pub fn random(seed: u64, _rect: Rect) -> Self {
        Heartbeat {
            bpm: range(seed, 0, S_BPM, BPM_RANGE.0, BPM_RANGE.1),
            scroll_speed: range(
                seed,
                0,
                S_SCROLL,
                SCROLL_SPEED_RANGE.0,
                SCROLL_SPEED_RANGE.1,
            ),
            amplitude_rows: range(
                seed,
                0,
                S_AMP,
                AMPLITUDE_ROWS_RANGE.0,
                AMPLITUDE_ROWS_RANGE.1,
            ),
            phase0: range(seed, 0, S_PHASE, 0.0, 1.0),
        }
    }

    /// Beats per minute this instance rolled at construction.
    #[must_use]
    pub fn bpm(&self) -> f32 {
        self.bpm
    }

    /// The trace's vertical offset (in rows, `0` = baseline, positive =
    /// below baseline) for the sample at absolute time `sample_t`.
    fn value_at(&self, sample_t: f32) -> f32 {
        let hz = self.bpm / 60.0;
        let phase = sample_t * hz + self.phase0;
        // ECG convention: R is a large UPWARD deflection -> negative row
        // offset (rows grow downward on screen).
        -ecg_waveform(phase) * self.amplitude_rows
    }
}

fn paint_grid(fb: &mut Framebuffer, rect: Rect, baseline_row: f32) {
    for row in rect.row0..(rect.row0 + rect.rows) {
        let glyph = if (row as f32 - baseline_row).abs() < 0.5 {
            '\u{2500}'
        } else {
            continue;
        };
        for col in rect.col0..(rect.col0 + rect.cols) {
            set_clipped(
                fb,
                rect,
                col as i64,
                row as i64,
                Cell {
                    glyph: u32::from(glyph),
                    fg: GRID_MAJOR,
                    bg: 0x0000_0000,
                    ..Default::default()
                },
            );
        }
    }
    // Minor vertical ticks every 8 columns, and a dim horizontal band 2 rows
    // above/below baseline — reads as ECG graph paper, not a bare line.
    for col in (rect.col0..(rect.col0 + rect.cols)).step_by(8) {
        for row in rect.row0..(rect.row0 + rect.rows) {
            if (row as f32 - baseline_row).abs() < 0.5 {
                continue;
            }
            set_clipped(
                fb,
                rect,
                col as i64,
                row as i64,
                Cell {
                    glyph: u32::from('\u{2506}'),
                    fg: GRID_MINOR,
                    bg: 0x0000_0000,
                    ..Default::default()
                },
            );
        }
    }
}

impl AsciiEffect for Heartbeat {
    fn name(&self) -> &'static str {
        "heartbeat"
    }

    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration) {
        let secs_t = secs(t);
        let baseline_row = rect.row0 as f32 + rect.rows as f32 * 0.62;
        paint_grid(fb, rect, baseline_row);

        let trace_lin = unpack_linear(TRACE_COLOR);
        let mut canvas = SubrowCanvas::new(
            rect,
            LinRgb {
                r: trace_lin.r * 0.12,
                g: trace_lin.g * 0.12,
                b: trace_lin.b * 0.12,
            },
        );

        let right_col = rect.cols;
        let total_samples = right_col * SAMPLES_PER_COL.max(1) as usize;
        let mut prev: Option<(f32, f32)> = None;
        for i in 0..=total_samples {
            let col_local = i as f32 / SAMPLES_PER_COL as f32;
            let cols_from_now = right_col as f32 - 1.0 - col_local;
            let sample_t = secs_t - cols_from_now / self.scroll_speed;
            if sample_t < 0.0 {
                prev = None;
                continue;
            }
            let row_local = (rect.rows as f32 * 0.62) + self.value_at(sample_t);
            let subrow_f = (row_local * 2.0).clamp(0.0, rect.rows as f32 * 2.0 - 1.0);
            // Fade toward the left (older) edge: newest (right, cols_from_now=0)
            // stays full brightness, oldest (left edge) fades to `OLDEST_FADE`.
            let age_frac = (cols_from_now / right_col as f32).clamp(0.0, 1.0);
            let fade = 1.0 - age_frac * (1.0 - OLDEST_FADE);

            if let Some((pc, pr)) = prev {
                let steps = ((col_local - pc)
                    .abs()
                    .max((subrow_f - pr).abs() * 0.5)
                    .ceil() as i32)
                    .clamp(1, 6);
                for s in 1..=steps {
                    let f = s as f32 / steps as f32;
                    canvas.splat(
                        pc + (col_local - pc) * f,
                        pr + (subrow_f - pr) * f,
                        trace_lin,
                        fade,
                    );
                }
            } else {
                canvas.splat(col_local, subrow_f, trace_lin, fade);
            }
            prev = Some((col_local, subrow_f));
        }
        // Bright comet head at the newest sample (right edge).
        if let Some((pc, pr)) = prev {
            canvas.splat(pc, pr, trace_lin, 1.4);
        }
        canvas.blit(
            fb,
            LinRgb {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
        );

        let label = format!("{:>3.0} BPM", self.bpm);
        put_str_clipped(
            fb,
            rect,
            rect.col0 as i64 + 1,
            rect.row0 as i64,
            &label,
            LABEL_COLOR,
            0x0000_0000,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECT: Rect = Rect {
        col0: 0,
        row0: 0,
        cols: 80,
        rows: 20,
    };

    #[test]
    fn determinism_same_seed() {
        let a = Heartbeat::random(51, RECT);
        let b = Heartbeat::random(51, RECT);
        assert_eq!(a.bpm, b.bpm);
        assert_eq!(a.scroll_speed, b.scroll_speed);
    }

    #[test]
    fn different_seeds_diverge_in_bpm_and_scroll() {
        let a = Heartbeat::random(1, RECT);
        let b = Heartbeat::random(2, RECT);
        assert_ne!(a.bpm, b.bpm);
        assert_ne!(a.scroll_speed, b.scroll_speed);
    }

    #[test]
    fn bpm_stays_within_physiological_range() {
        for seed in 0..30u64 {
            let hb = Heartbeat::random(seed * 13 + 1, RECT);
            assert!((BPM_RANGE.0..=BPM_RANGE.1).contains(&hb.bpm));
        }
    }

    #[test]
    fn the_r_spike_is_the_tallest_feature_of_one_beat() {
        let mut max_v = f32::MIN;
        let mut max_phase = 0.0f32;
        let mut steps = 0;
        while steps < 1000 {
            let phase = steps as f32 / 1000.0;
            let v = ecg_waveform(phase);
            if v > max_v {
                max_v = v;
                max_phase = phase;
            }
            steps += 1;
        }
        assert!(
            (max_phase - 0.40).abs() < 0.02,
            "peak should land on the R wave, found at {max_phase}"
        );
        assert!(max_v > 0.9, "R spike amplitude looks wrong: {max_v}");
    }

    #[test]
    fn glyphs_used_are_all_reliable() {
        use re_render::glyph_family::is_reliable;
        for g in ['\u{2500}', '\u{2506}', '\u{2580}'] {
            assert!(is_reliable(g), "{g:?} not reliable");
        }
    }

    #[test]
    fn paint_stays_inside_its_rect() {
        let mut fb = Framebuffer::new(90, 24);
        let rect = Rect::new(5, 2, 80, 20);
        let hb = Heartbeat::random(9, rect);
        for ms in [0u64, 400, 2200, 9000] {
            hb.paint(&mut fb, rect, Duration::from_millis(ms));
        }
        for row in 0..24 {
            for col in 0..90 {
                if !rect.contains(col as i64, row as i64) {
                    assert_eq!(*fb.get(col, row).unwrap(), re_core::Cell::default());
                }
            }
        }
    }

    #[test]
    fn the_trace_fills_a_meaningful_fraction_of_the_rect_not_a_few_dots() {
        let mut fb = Framebuffer::new(80, 20);
        let hb = Heartbeat::random(3, RECT);
        hb.paint(&mut fb, RECT, Duration::from_millis(3000));
        let lit = (0..20)
            .flat_map(|r| (0..80).map(move |c| (c, r)))
            .filter(|&(c, r)| *fb.get(c, r).unwrap() != re_core::Cell::default())
            .count();
        assert!(
            lit > 150,
            "expected a dense scrolling trace, only {lit} cells lit"
        );
    }
}
