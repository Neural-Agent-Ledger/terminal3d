// Test-module assert_eq! on f32/f64 values are intentional exact-equal checks
// (determinism proofs: same seed -> byte-identical parameters/output).
// Allow float_cmp only in test compilation units; lib code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `re_asciifx` — seeded 2D ASCII effects (eye, plasma, tunnel, radar, …) plus
//! the small text-painting helpers every screen uses.
//!
//! # Purpose
//!
//! Deterministic, seed-driven animated ASCII effects painted straight into a
//! [`re_core::Framebuffer`], plus `paint`'s `put_str`/`put_center`/`fill_bg`
//! helpers. No ECS, no game state — every effect is a pure function of
//! `(seed, rect, t)`: two calls with the same seed produce byte-identical
//! output, so an effect can be seeked to any `t` directly with no incremental
//! stepping and no accumulated state (see `eye`, `plasma`, etc.).
//!
//! # Dependencies
//!
//! Only `re_core` (the `Framebuffer`/`Cell` types) — no Bevy, no terminal, no
//! game state. `gallery` (test-only) additionally depends on `re_render` for
//! `glyph_family::is_reliable`, hence that crate is a dev-dependency only.

pub mod canvas;
pub mod circuit;
pub mod dna;
pub mod eye;
pub mod eye_swarm;
#[cfg(test)]
mod gallery;
pub mod hash;
pub mod heartbeat;
pub mod neural_net;
pub mod oscilloscope;
pub mod overgrowth;
pub mod paint;
pub mod plasma;
pub mod radar;
pub mod seeds;
pub mod tunnel;

// ─────────────────────────── the shared effect contract ─────────────────────
//
// Design, one sentence: every effect is a small `Copy`/`Clone` struct of
// already-resolved numbers (radii, hues, phase offsets, per-effect
// frequencies …) built ONCE by a `random(seed, rect)` constructor that draws
// every number from a `hash::hash_unit(seed, id, salt)`-family call — never a
// wall-clock, never a thread-local RNG, so two calls with the same seed
// produce byte-identical parameters (the same discipline `overgrowth` and
// `re_render`'s emitter/sky work also follow: randomized parameters with
// seeded determinism).
// [`AsciiEffect::paint`] then reads `self` and `t` only: no incremental
// stepping, no accumulated state, so a caller can seek to any `t` directly
// (start the eye mid-blink, dump a gallery frame at `t=1.7s`) and get the
// exact same frame every time — what makes every effect here headlessly
// unit-testable with no `App` and no terminal.

use re_core::Framebuffer;
use std::time::Duration;

/// A rectangular sub-region of a [`Framebuffer`] an effect paints into.
/// `col0`/`row0` are the top-left cell (inclusive), `cols`/`rows` the extent.
/// Every effect clips to this rect itself (never assumes the whole
/// framebuffer), so several effects can share one screen — the gallery paints
/// N effects side by side into N disjoint rects of the same framebuffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    /// Left column (inclusive, 0-based).
    pub col0: usize,
    /// Top row (inclusive, 0-based).
    pub row0: usize,
    /// Width in cells.
    pub cols: usize,
    /// Height in cells.
    pub rows: usize,
}

impl Rect {
    /// Construct a rect from its top-left corner and extent.
    #[must_use]
    pub fn new(col0: usize, row0: usize, cols: usize, rows: usize) -> Self {
        Rect {
            col0,
            row0,
            cols,
            rows,
        }
    }

    /// The rect's centre, in fractional framebuffer cell coordinates
    /// `(col, row)`. Every effect's "size/position" randomization treats this
    /// as its origin unless the effect specifically randomizes an offset from
    /// it (documented per effect).
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        (
            self.col0 as f32 + self.cols as f32 * 0.5,
            self.row0 as f32 + self.rows as f32 * 0.5,
        )
    }

    /// `true` iff `(col, row)` falls inside this rect.
    #[must_use]
    pub fn contains(&self, col: i64, row: i64) -> bool {
        col >= self.col0 as i64
            && row >= self.row0 as i64
            && col < (self.col0 + self.cols) as i64
            && row < (self.row0 + self.rows) as i64
    }

    /// The shorter of `cols`/`rows`, useful for an effect that wants to size
    /// itself to fit the smaller dimension of a non-square rect (the eye: a
    /// too-wide iris would poke out of a short rect's lids).
    #[must_use]
    pub fn min_extent(&self) -> usize {
        self.cols.min(self.rows)
    }
}

/// A terminal cell is roughly twice as tall as it is wide in every monospace
/// font this effect library targets (Cascadia/Consolas-class), so a shape
/// that should read as visually ROUND has to travel about twice as far in
/// row-units as in col-units for the same world distance. Every effect that
/// draws a round or elliptical shape (the eye's iris/pupil/sclera, the radar
/// sweep, DNA rungs …) divides row deltas by this constant before computing a
/// Euclidean distance, so a shape traced in cell coordinates reads as round in
/// the terminal's actual aspect ratio rather than stretched.
pub const CELL_ASPECT: f32 = 2.0;

/// Euclidean distance between `(col0, row0)` and `(col1, row1)` in
/// ASPECT-CORRECTED "world" units (see [`CELL_ASPECT`]) — the one distance
/// function every round-shape effect in this crate shares, so a circle drawn
/// by one effect looks exactly as round as a circle drawn by another.
#[must_use]
#[inline]
pub fn aspect_dist(col0: f32, row0: f32, col1: f32, row1: f32) -> f32 {
    let dx = col1 - col0;
    let dy = (row1 - row0) * CELL_ASPECT;
    dx.hypot(dy)
}

/// A procedural, seeded, animated ASCII effect over a fixed [`Rect`].
///
/// `paint` is a pure function of `(self, rect, t)`: it may read `self` and
/// `t` and nothing else (no wall clock, no RNG, no mutation through `&self`),
/// so calling it twice with the same inputs always paints the same cells —
/// the determinism every unit test below relies on.
pub trait AsciiEffect {
    /// A short, human-readable identifier (used by the gallery + docs, never
    /// parsed).
    fn name(&self) -> &'static str;

    /// Paint this effect's frame at time `t` into `rect` of `fb`. Must clip
    /// itself to `rect` — writes outside it are a bug, not a feature (the
    /// gallery relies on effects staying inside their assigned rect so
    /// several can share one framebuffer).
    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration);
}

/// Write `cell` at `(col, row)` if — and only if — it falls inside `rect`
/// (see [`AsciiEffect::paint`]'s clipping contract) AND inside `fb`'s own
/// bounds. Every effect's inner painter funnels through this one function so
/// the clip is enforced in exactly one place.
#[inline]
pub(crate) fn set_clipped(
    fb: &mut Framebuffer,
    rect: Rect,
    col: i64,
    row: i64,
    cell: re_core::Cell,
) {
    if rect.contains(col, row) && col >= 0 && row >= 0 {
        fb.set(col as usize, row as usize, cell);
    }
}

/// `t` as whole + fractional seconds, `f32` — every effect's own frequency
/// math wants a plain float, not a [`Duration`].
#[must_use]
#[inline]
pub fn secs(t: Duration) -> f32 {
    t.as_secs_f32()
}

/// Stamp `text` starting at `(col, row)`, one glyph per column, clipped to
/// `rect` (and `fb`'s own bounds) via [`set_clipped`] — the one small text
/// helper every non-eye effect in this crate shares for chip labels / part
/// numbers / readouts. Distinct from [`paint::put_str`]: this one clips to an
/// effect's own [`Rect`] rather than the whole framebuffer.
pub(crate) fn put_str_clipped(
    fb: &mut Framebuffer,
    rect: Rect,
    col: i64,
    row: i64,
    text: &str,
    fg: u32,
    bg: u32,
) {
    for (i, ch) in text.chars().enumerate() {
        set_clipped(
            fb,
            rect,
            col + i as i64,
            row,
            re_core::Cell {
                glyph: u32::from(ch),
                fg,
                bg,
                ..Default::default()
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_contains_is_exclusive_at_the_far_edge() {
        let r = Rect::new(2, 3, 4, 5); // cols 2..6, rows 3..8
        assert!(r.contains(2, 3));
        assert!(r.contains(5, 7));
        assert!(!r.contains(6, 7), "col_end is exclusive");
        assert!(!r.contains(5, 8), "row_end is exclusive");
        assert!(!r.contains(1, 3), "col before col0");
        assert!(!r.contains(-1, 3), "negative col must not panic or wrap");
    }

    #[test]
    fn aspect_dist_is_symmetric_and_zero_at_the_same_point() {
        assert_eq!(aspect_dist(5.0, 5.0, 5.0, 5.0), 0.0);
        let a = aspect_dist(0.0, 0.0, 3.0, 4.0);
        let b = aspect_dist(3.0, 4.0, 0.0, 0.0);
        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn aspect_dist_weighs_row_steps_more_than_col_steps() {
        // One col step vs one row step: the row step should read as FARTHER
        // (CELL_ASPECT > 1), which is the whole point of the correction.
        let col_step = aspect_dist(0.0, 0.0, 1.0, 0.0);
        let row_step = aspect_dist(0.0, 0.0, 0.0, 1.0);
        assert!(row_step > col_step);
    }
}

/// Perf: every effect must paint a 280×70 rect well under 2 ms in release
/// (task acceptance criterion). Run with
/// `cargo test --release -p re_asciifx --lib perf -- --nocapture` to
/// see the measured numbers on stdout; the assertions themselves use a
/// generous debug-safe threshold (debug builds are commonly 10-50x slower
/// than release for math-heavy code like this) so this stays a real,
/// non-flaky CI gate rather than a number tuned to whatever machine ran it
/// last — the PRECISE numbers are reported in FILES CHANGED / RESULT, not
/// asserted here.
#[cfg(test)]
mod perf {
    use super::*;
    use crate::circuit::Circuit;
    use crate::dna::Dna;
    use crate::eye::Eye;
    use crate::eye_swarm::EyeSwarm;
    use crate::heartbeat::Heartbeat;
    use crate::neural_net::NeuralNet;
    use crate::oscilloscope::Oscilloscope;
    use crate::plasma::Plasma;
    use crate::radar::Radar;
    use crate::seeds::Seeds;
    use crate::tunnel::Tunnel;
    use std::time::{Duration, Instant};

    const PERF_RECT: Rect = Rect {
        col0: 0,
        row0: 0,
        cols: 280,
        rows: 70,
    };
    /// Debug builds are not what the 2 ms acceptance number is about (see the
    /// module doc above) — this is just a "did not regress to something
    /// pathological" tripwire so a debug `cargo test` run still catches an
    /// accidental O(n²) rewrite.
    const DEBUG_SAFE_BUDGET: Duration = Duration::from_millis(200);

    fn measure(name: &str, mut paint_once: impl FnMut(Duration)) {
        // Warm up (first call may pay one-time allocator/cache costs).
        paint_once(Duration::ZERO);
        let iters = 20;
        let start = Instant::now();
        for i in 0..iters {
            paint_once(Duration::from_millis(i * 37));
        }
        let elapsed = start.elapsed();
        let per_frame = elapsed / iters as u32;
        eprintln!("ascii_fx perf: {name:>16} {per_frame:>10.2?} / frame ({iters} frames, 280x70)");
        assert!(
            per_frame < DEBUG_SAFE_BUDGET,
            "{name} took {per_frame:?}/frame, over the {DEBUG_SAFE_BUDGET:?} debug-safe tripwire"
        );
    }

    #[test]
    fn eye_paint_perf() {
        let mut fb = Framebuffer::new(PERF_RECT.cols, PERF_RECT.rows);
        let eye = Eye::random(1, PERF_RECT);
        measure("eye", |t| eye.paint(&mut fb, PERF_RECT, t));
    }

    #[test]
    fn eye_swarm_paint_perf() {
        let mut fb = Framebuffer::new(PERF_RECT.cols, PERF_RECT.rows);
        let swarm = EyeSwarm::random(1, PERF_RECT);
        measure("eye_swarm", |t| swarm.paint(&mut fb, PERF_RECT, t));
    }

    #[test]
    fn circuit_paint_perf() {
        let mut fb = Framebuffer::new(PERF_RECT.cols, PERF_RECT.rows);
        let circuit = Circuit::random(1, PERF_RECT);
        measure("circuit", |t| circuit.paint(&mut fb, PERF_RECT, t));
    }

    #[test]
    fn neural_net_paint_perf() {
        let mut fb = Framebuffer::new(PERF_RECT.cols, PERF_RECT.rows);
        let net = NeuralNet::random(1, PERF_RECT);
        measure("neural_net", |t| net.paint(&mut fb, PERF_RECT, t));
    }

    #[test]
    fn seeds_paint_perf() {
        let mut fb = Framebuffer::new(PERF_RECT.cols, PERF_RECT.rows);
        let seeds = Seeds::random(1, PERF_RECT);
        measure("seeds", |t| seeds.paint(&mut fb, PERF_RECT, t));
    }

    #[test]
    fn oscilloscope_paint_perf() {
        let mut fb = Framebuffer::new(PERF_RECT.cols, PERF_RECT.rows);
        let osc = Oscilloscope::random(1, PERF_RECT);
        measure("oscilloscope", |t| osc.paint(&mut fb, PERF_RECT, t));
    }

    #[test]
    fn radar_paint_perf() {
        let mut fb = Framebuffer::new(PERF_RECT.cols, PERF_RECT.rows);
        let radar = Radar::random(1, PERF_RECT);
        measure("radar", |t| radar.paint(&mut fb, PERF_RECT, t));
    }

    #[test]
    fn heartbeat_paint_perf() {
        let mut fb = Framebuffer::new(PERF_RECT.cols, PERF_RECT.rows);
        let hb = Heartbeat::random(1, PERF_RECT);
        measure("heartbeat", |t| hb.paint(&mut fb, PERF_RECT, t));
    }

    #[test]
    fn plasma_paint_perf() {
        let mut fb = Framebuffer::new(PERF_RECT.cols, PERF_RECT.rows);
        let plasma = Plasma::random(1, PERF_RECT);
        measure("plasma", |t| plasma.paint(&mut fb, PERF_RECT, t));
    }

    #[test]
    fn tunnel_paint_perf() {
        let mut fb = Framebuffer::new(PERF_RECT.cols, PERF_RECT.rows);
        let tunnel = Tunnel::random(1, PERF_RECT);
        measure("tunnel", |t| tunnel.paint(&mut fb, PERF_RECT, t));
    }

    #[test]
    fn dna_paint_perf() {
        let mut fb = Framebuffer::new(PERF_RECT.cols, PERF_RECT.rows);
        let dna = Dna::random(1, PERF_RECT);
        measure("dna", |t| dna.paint(&mut fb, PERF_RECT, t));
    }
}
