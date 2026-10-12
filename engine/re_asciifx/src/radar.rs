//! RADAR / SONAR SWEEP — a rotating beam with a smooth, colour-graded
//! trailing afterglow wedge around a fixed centre, thin continuous range
//! rings, a scatter of blips that FLARE (a bright core plus an expanding,
//! fading shockwave ring) and decay, and faint bearing ticks / range labels
//! around the rim.
//!
//! ## Rendering: half-block colour field, not glyph switching
//!
//! (Rewritten 2026-09-25 — the owner's brief: "the sweep reads blocky; needs
//! smooth thin rings, a crisp sweep line with a smooth fading trailing wedge
//! (half-block colour gradient), blips that flare and decay, faint range
//! labels/bearing ticks".) Rings, the sweep wedge, and blips all resolve
//! through the SAME [`crate::canvas::SubrowCanvas`]
//! `oscilloscope`/`heartbeat` already use: every sample is a soft
//! anti-aliased dot, brightest-wins blended, so the sweep's trailing wedge
//! reads as a continuous COLOUR gradient (bright at the leading edge, fading
//! smoothly to nothing at its tail) rather than the old three-glyph
//! (`█▓░`) step function, and a ring is one dense band of half-block pixels
//! rather than a scatter of `•` glyphs that under-sampled near the top/bottom
//! of the circle. Bearing ticks (N/E/S/W) and range labels (one faint number
//! per ring) are the one place this effect still uses plain glyphs — they
//! are text, not a colour field, and are painted BEFORE the canvas so a rim
//! label can never be blotted out by a ring passing through its cell (the
//! canvas only overwrites cells it actually touches — see `SubrowCanvas::
//! blit`'s doc).

use crate::canvas::SubrowCanvas;
use crate::hash::{angle, range};
use crate::{put_str_clipped, secs, set_clipped, AsciiEffect, Rect, CELL_ASPECT};
use re_core::oklab::{oklch_to_packed, packed_to_oklch, unpack_linear, LinRgb};
use re_core::{Cell, Framebuffer};
use std::f32::consts::TAU;
use std::time::Duration;

// ─────────────────────────────── tuning ──────────────────────────────────

const SWEEP_SPEED_RANGE: (f32, f32) = (0.6, 1.8); // rad/s — its own frequency
/// Radians of trailing wedge behind the beam — randomized per instance so a
/// wide sonar and a narrow lidar-style sweep both occur.
const AFTERGLOW_WIDTH_RANGE: (f32, f32) = (0.55, 1.3);
/// Shape of the wedge's fade: `intensity = (1 - behind/width) ^ POWER`.
/// `> 1` keeps the leading edge crisp and lets the tail fall away faster,
/// which is what reads as "smooth fading" rather than a linear ramp.
const AFTERGLOW_FADE_POWER: f32 = 1.6;
/// The disc's own overall size, as a fraction of the rect's shorter extent —
/// randomized per instance (the owner's "size" ask) rather than one fixed
/// `0.48` for every seed.
const RADIUS_FRAC_RANGE: (f32, f32) = (0.40, 0.49);
const RING_COUNT_RANGE: (usize, usize) = (2, 4);
const BLIP_COUNT_RANGE: (usize, usize) = (6, 13);
const BLIP_DECAY_RANGE: (f32, f32) = (0.3, 1.2); // 1/s brightness decay after a ping
/// How fast a blip's flare shockwave ring expands outward after a ping, in
/// world (row) units per second — its own independent frequency-ish knob.
const FLARE_EXPAND_SPEED_RANGE: (f32, f32) = (3.0, 7.0);
/// A whole-palette hue rotation, per instance, so different seeds don't all
/// read as the identical "submarine green" — same technique
/// `oscilloscope::Oscilloscope::random` uses for its head colour.
const HUE_ROTATE_RANGE: f32 = 0.9;

const S_BLIP_ANGLE: u64 = 1;
const S_BLIP_RADIUS: u64 = 2;
const S_BLIP_DECAY: u64 = 3;
const S_SWEEP_SPEED: u64 = 4;
const S_SWEEP_PHASE: u64 = 5;
const S_AFTERGLOW_WIDTH: u64 = 6;
const S_RADIUS_FRAC: u64 = 7;
const S_RING_COUNT: u64 = 8;
const S_HUE: u64 = 9;
const S_FLARE_SPEED: u64 = 10;

const RING_COLOR: u32 = 0x0030_7050;
const CROSSHAIR_COLOR: u32 = 0x0016_2E1E;
const CENTER_COLOR: u32 = 0x0080_FFB0;
const SWEEP_HEAD_COLOR: u32 = 0x00A0_FFC0;
const SWEEP_TAIL_COLOR: u32 = 0x0020_6040;
const BLIP_COLOR: u32 = 0x0040_FF60;
const BEARING_LABEL_COLOR: u32 = 0x0028_5038;
const RANGE_LABEL_COLOR: u32 = 0x0020_4030;

const BEARING_TICKS: [(&str, f32); 4] = [
    ("N", -std::f32::consts::FRAC_PI_2),
    ("E", 0.0),
    ("S", std::f32::consts::FRAC_PI_2),
    ("W", std::f32::consts::PI),
];

struct Blip {
    angle: f32,
    radius_frac: f32,
    decay: f32,
}

/// A rotating sweep with pinging blips.
pub struct Radar {
    sweep_speed: f32,
    sweep_phase: f32,
    afterglow_width: f32,
    radius_frac: f32,
    ring_count: usize,
    flare_expand_speed: f32,
    sweep_head: u32,
    sweep_tail: u32,
    ring_color: u32,
    blip_color: u32,
    blips: Vec<Blip>,
}

fn rotate_hue(base: u32, hue_delta: f32) -> u32 {
    let mut oklch = packed_to_oklch(base);
    oklch.h += hue_delta;
    oklch_to_packed(oklch)
}

impl Radar {
    /// Build a deterministic sweep radar from `seed`: a random blip count,
    /// ring count and rotating sweep-head hue. `_rect` is unused (the display
    /// is drawn in a normalized polar space centered on the target rect by
    /// the painter) but kept for the shared `random(seed, rect)` signature.
    #[must_use]
    pub fn random(seed: u64, _rect: Rect) -> Self {
        let count = BLIP_COUNT_RANGE.0
            + (crate::hash::hash_unit(seed, 0, 0)
                * (BLIP_COUNT_RANGE.1 - BLIP_COUNT_RANGE.0 + 1) as f64) as usize;
        let count = count.min(BLIP_COUNT_RANGE.1).max(BLIP_COUNT_RANGE.0);
        let mut blips = Vec::with_capacity(count);
        for i in 0..count {
            let id = i as u64;
            blips.push(Blip {
                angle: angle(seed, id, S_BLIP_ANGLE),
                radius_frac: range(seed, id, S_BLIP_RADIUS, 0.2, 0.95),
                decay: range(
                    seed,
                    id,
                    S_BLIP_DECAY,
                    BLIP_DECAY_RANGE.0,
                    BLIP_DECAY_RANGE.1,
                ),
            });
        }
        let ring_count = RING_COUNT_RANGE.0
            + (crate::hash::hash_unit(seed, 0, S_RING_COUNT)
                * (RING_COUNT_RANGE.1 - RING_COUNT_RANGE.0 + 1) as f64) as usize;
        let ring_count = ring_count.min(RING_COUNT_RANGE.1).max(RING_COUNT_RANGE.0);
        let hue_delta = (crate::hash::hash_signed(seed, 0, S_HUE) as f32) * HUE_ROTATE_RANGE;
        Radar {
            sweep_speed: range(
                seed,
                0,
                S_SWEEP_SPEED,
                SWEEP_SPEED_RANGE.0,
                SWEEP_SPEED_RANGE.1,
            ),
            sweep_phase: angle(seed, 0, S_SWEEP_PHASE),
            afterglow_width: range(
                seed,
                0,
                S_AFTERGLOW_WIDTH,
                AFTERGLOW_WIDTH_RANGE.0,
                AFTERGLOW_WIDTH_RANGE.1,
            ),
            radius_frac: range(
                seed,
                0,
                S_RADIUS_FRAC,
                RADIUS_FRAC_RANGE.0,
                RADIUS_FRAC_RANGE.1,
            ),
            ring_count,
            flare_expand_speed: range(
                seed,
                0,
                S_FLARE_SPEED,
                FLARE_EXPAND_SPEED_RANGE.0,
                FLARE_EXPAND_SPEED_RANGE.1,
            ),
            sweep_head: rotate_hue(SWEEP_HEAD_COLOR, hue_delta),
            sweep_tail: rotate_hue(SWEEP_TAIL_COLOR, hue_delta),
            ring_color: rotate_hue(RING_COLOR, hue_delta),
            blip_color: rotate_hue(BLIP_COLOR, hue_delta),
            blips,
        }
    }

    /// Number of blips on this radar's display.
    #[must_use]
    pub fn blip_count(&self) -> usize {
        self.blips.len()
    }

    fn sweep_angle(&self, t: f32) -> f32 {
        (self.sweep_speed * t + self.sweep_phase).rem_euclid(TAU)
    }

    /// Brightness in `[0, 1]` of `blip` at time `t`: `1.0` right as the beam
    /// crosses it, decaying exponentially at the blip's own `decay` rate
    /// until the beam comes back around. Also returns `dt`, the time since
    /// that last crossing (used to grow the flare shockwave ring).
    fn blip_state(&self, blip: &Blip, t: f32) -> (f32, f32) {
        let sweep = self.sweep_speed.max(1e-4);
        let raw = (sweep * t + self.sweep_phase - blip.angle) / TAU;
        let k = raw.floor();
        let crossing_t = (blip.angle + k * TAU - self.sweep_phase) / sweep;
        let dt = (t - crossing_t).max(0.0);
        ((-blip.decay * dt).exp(), dt)
    }

    /// Test-only convenience: brightness alone, dropping [`Self::blip_state`]'s
    /// `dt` half of the pair (production `paint()` calls `blip_state` directly
    /// since it needs both).
    #[cfg(test)]
    fn blip_brightness(&self, blip: &Blip, t: f32) -> f32 {
        self.blip_state(blip, t).0
    }
}

fn scale_lin(c: LinRgb, k: f32) -> LinRgb {
    LinRgb {
        r: c.r * k,
        g: c.g * k,
        b: c.b * k,
    }
}

/// Faint bearing ticks (N/E/S/W) just past the rim, plus one small dim range
/// label per ring along the NE diagonal — called by `paint` AFTER
/// `SubrowCanvas::blit`, so every label unconditionally wins its cell. It
/// used to run before the canvas on the theory that `blit` only touches
/// cells with a dot or glow, so text elsewhere would survive underneath —
/// true for the 'E' tick (the only one an earlier test actually checked),
/// false for 'N'/'S'/'W' (whose fixed clearance past `max_r`, divided by
/// `CELL_ASPECT`, lands inside the glow halo's 1-cell spread) and false for
/// every range label (deliberately placed ON a ring's own radius, which the
/// ring rendering is guaranteed to paint a dot at). Found via the 2026-09-25
/// gallery3 visual review.
fn paint_labels(fb: &mut Framebuffer, rect: Rect, cx: f32, cy: f32, max_r: f32, ring_count: usize) {
    // A very faint crosshair, dashed rather than solid so it reads as a
    // reticle rather than a hard line cutting the disc in two — painted
    // FIRST so the bearing/range text below always wins the same cell (a
    // dash landing exactly on a label's column would otherwise blot out the
    // one letter/digit that cell should show).
    for col in (rect.col0..(rect.col0 + rect.cols)).step_by(2) {
        set_clipped(
            fb,
            rect,
            col as i64,
            cy.round() as i64,
            Cell {
                glyph: u32::from('\u{2500}'),
                fg: CROSSHAIR_COLOR,
                bg: 0x0000_0000,
                ..Default::default()
            },
        );
    }
    for row in (rect.row0..(rect.row0 + rect.rows)).step_by(2) {
        set_clipped(
            fb,
            rect,
            cx.round() as i64,
            row as i64,
            Cell {
                glyph: u32::from('\u{2502}'),
                fg: CROSSHAIR_COLOR,
                bg: 0x0000_0000,
                ..Default::default()
            },
        );
    }
    for &(label, ang) in &BEARING_TICKS {
        let col = cx + (max_r + 1.5) * ang.cos();
        let row = cy + ((max_r + 1.5) * ang.sin()) / CELL_ASPECT;
        put_str_clipped(
            fb,
            rect,
            col.round() as i64,
            row.round() as i64,
            label,
            BEARING_LABEL_COLOR,
            0x0000_0000,
        );
    }
    // Range labels ride the NE diagonal (45°) rather than a cardinal bearing
    // — clear of every `BEARING_TICKS` entry (N/E/S/W) so the two label
    // kinds can never land on the same cell.
    let label_angle = std::f32::consts::FRAC_PI_4;
    let spacing = max_r / ring_count as f32;
    for ring_i in 1..=ring_count {
        let r = spacing * ring_i as f32;
        let range_units = ((r / max_r) * 100.0).round() as i64;
        let label = format!("{range_units}");
        let col = cx + r * label_angle.cos();
        let row = cy + (r * label_angle.sin()) / CELL_ASPECT;
        put_str_clipped(
            fb,
            rect,
            col.round() as i64,
            row.round() as i64,
            &label,
            RANGE_LABEL_COLOR,
            0x0000_0000,
        );
    }
}

impl AsciiEffect for Radar {
    fn name(&self) -> &'static str {
        "radar"
    }

    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration) {
        let (cx, cy) = rect.center();
        let max_r = rect.min_extent() as f32 * self.radius_frac;
        let sweep = self.sweep_angle(secs(t));

        let ring_lin = unpack_linear(self.ring_color);
        let head_lin = unpack_linear(self.sweep_head);
        let tail_lin = unpack_linear(self.sweep_tail);
        let blip_lin = unpack_linear(self.blip_color);
        let mut canvas = SubrowCanvas::new(rect, scale_lin(ring_lin, 0.10));

        // Range rings: dense parametric circles sampled directly in SUB-ROW
        // space, so the ring stays a continuous, evenly-lit band regardless
        // of aspect ratio (the old per-cell raster test under-sampled near
        // the top/bottom of each ring — see the 2026-09-24 "scattered dust"
        // regression this replaces).
        let ring_spacing = max_r / self.ring_count as f32;
        for ring_i in 1..=self.ring_count {
            let r = ring_spacing * ring_i as f32;
            let steps = ((r * TAU * 2.0) as usize).max(64);
            for step in 0..steps {
                let a = TAU * step as f32 / steps as f32;
                let col_f = (rect.cols as f32 * 0.5) + r * a.cos();
                let subrow_f = (rect.rows as f32) + (r * a.sin()) / CELL_ASPECT * 2.0;
                canvas.splat(col_f, subrow_f, ring_lin, 0.6);
            }
        }

        // The sweep's trailing wedge: dense per-sub-cell sampling so the
        // fade reads as a smooth COLOUR gradient (head → tail), not a
        // handful of glyph steps.
        let subrows = rect.rows * 2;
        for subrow in 0..subrows {
            // Physical (world-space) row position for the field math — the
            // two sub-rows of cell-row `r` sit at `r+0.25`/`r+0.75` (the same
            // convention `eye::shade` samples at).
            let local_row_phys = subrow as f32 * 0.5 + 0.25;
            let dy = (local_row_phys - rect.rows as f32 * 0.5) * CELL_ASPECT;
            for col in 0..rect.cols {
                let local_col_phys = col as f32 + 0.5;
                let dx = local_col_phys - rect.cols as f32 * 0.5;
                let dist = dx.hypot(dy);
                if dist > max_r || dist < 0.6 {
                    continue;
                }
                let cell_angle = dy.atan2(dx).rem_euclid(TAU);
                let behind = (sweep - cell_angle).rem_euclid(TAU);
                if behind >= self.afterglow_width {
                    continue;
                }
                let k = (1.0 - behind / self.afterglow_width)
                    .clamp(0.0, 1.0)
                    .powf(AFTERGLOW_FADE_POWER);
                if k <= 0.01 {
                    continue;
                }
                let color = re_core::oklab::lerp_linear(tail_lin, head_lin, k);
                // Bucket-EXACT target (integer col/subrow, zero fractional
                // remainder) rather than a physical continuous coordinate:
                // this loop already visits every output half-cell once, so a
                // fractional splat here would just blur each sample 50/50
                // into its neighbour for no benefit — the AA a moving point
                // (a ring, a blip) needs is irrelevant to a dense per-cell
                // fill that already IS at full resolution.
                canvas.splat(col as f32, subrow as f32, color, k);
            }
        }

        let secs_t = secs(t);
        for blip in &self.blips {
            let r = blip.radius_frac * max_r;
            let local_col = rect.cols as f32 * 0.5 + r * blip.angle.cos();
            let local_row = rect.rows as f32 * 0.5 + (r * blip.angle.sin()) / CELL_ASPECT;
            let (brightness, dt) = self.blip_state(blip, secs_t);
            if brightness < 0.02 {
                continue;
            }
            // Bright core, brightest right at the crossing.
            canvas.splat(local_col, local_row * 2.0, blip_lin, brightness * 1.3);
            // Expanding, fading shockwave ring — the "flare" the owner asked
            // for: a ping's flash visibly BLOOMS outward and shrinks away as
            // it decays, rather than one static pixel.
            let flare_r = 0.6 + dt * self.flare_expand_speed;
            if flare_r < max_r * 0.5 {
                let ring_steps = 16;
                for i in 0..ring_steps {
                    let a = TAU * i as f32 / ring_steps as f32;
                    let fc = local_col + flare_r * a.cos();
                    let fr = local_row * 2.0 + (flare_r * a.sin()) / CELL_ASPECT * 2.0;
                    canvas.splat(fc, fr, blip_lin, brightness * 0.5);
                }
            }
        }

        canvas.blit(
            fb,
            LinRgb {
                r: 0.0,
                g: 0.0,
                b: 0.0,
            },
        );

        // Labels painted AFTER the canvas blit, not before (see this
        // function's earlier 2026-09-24 version and `paint_labels`'s own
        // stale doc, both corrected here 2026-09-25): `SubrowCanvas::blit`
        // only SKIPS a cell with neither a direct dot nor meaningful glow,
        // and both range labels (deliberately placed ON a ring's own radius)
        // and the N/S bearing ticks (whose fixed world-unit clearance past
        // `max_r` becomes under one glow-halo cell once divided by
        // `CELL_ASPECT`) reliably fail that test — painting labels first let
        // the ring/glow silently blot every one of them out except 'E'
        // (found via the 2026-09-25 gallery3 visual review; the existing
        // test only ever asserted 'E' survived). Painting them last makes
        // every label unconditionally win, which is what a legible
        // annotation layer needs.
        paint_labels(fb, rect, cx, cy, max_r, self.ring_count);

        // Bright center dot — the sweep's pivot, drawn last so it always
        // reads clearly through everything underneath it.
        set_clipped(
            fb,
            rect,
            cx.round() as i64,
            cy.round() as i64,
            Cell {
                glyph: u32::from('\u{25CF}'),
                fg: CENTER_COLOR,
                bg: 0x0000_0000,
                ..Default::default()
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
        cols: 60,
        rows: 30,
    };

    #[test]
    fn determinism_same_seed() {
        let a = Radar::random(41, RECT);
        let b = Radar::random(41, RECT);
        assert_eq!(a.blip_count(), b.blip_count());
        assert_eq!(a.sweep_speed, b.sweep_speed);
        assert_eq!(a.radius_frac, b.radius_frac);
    }

    #[test]
    fn different_seeds_diverge_in_sweep_speed_and_blips() {
        let a = Radar::random(1, RECT);
        let b = Radar::random(2, RECT);
        assert_ne!(a.sweep_speed, b.sweep_speed);
    }

    #[test]
    fn different_seeds_diverge_in_size_and_afterglow_frequency() {
        // The owner's ask, verbatim: "size, direction, and above all
        // FREQUENCIES". `radius_frac` is the size knob, `afterglow_width` and
        // `flare_expand_speed` are two more independent per-instance rates.
        let mut any_radius_diff = false;
        let mut any_afterglow_diff = false;
        let mut any_flare_diff = false;
        for seed in 0..12u64 {
            let a = Radar::random(seed, RECT);
            let b = Radar::random(seed + 1000, RECT);
            if (a.radius_frac - b.radius_frac).abs() > 1e-6 {
                any_radius_diff = true;
            }
            if (a.afterglow_width - b.afterglow_width).abs() > 1e-6 {
                any_afterglow_diff = true;
            }
            if (a.flare_expand_speed - b.flare_expand_speed).abs() > 1e-6 {
                any_flare_diff = true;
            }
        }
        assert!(
            any_radius_diff,
            "radius_frac never varied across 12 seed pairs"
        );
        assert!(
            any_afterglow_diff,
            "afterglow_width never varied across 12 seed pairs"
        );
        assert!(
            any_flare_diff,
            "flare_expand_speed never varied across 12 seed pairs"
        );
    }

    #[test]
    fn brightness_peaks_right_at_a_sweep_crossing_and_decays_after() {
        let radar = Radar::random(3, RECT);
        let blip = &radar.blips[0];
        let sweep = radar.sweep_speed;
        // A hair after the exact crossing, not ON it — `raw` in
        // `blip_state` sits exactly on an integer boundary at the true
        // crossing instant, which floating-point `floor()` cannot be trusted
        // to round consistently.
        let t_cross = (blip.angle - radar.sweep_phase).rem_euclid(TAU) / sweep + 1e-4;
        let b_at = radar.blip_brightness(blip, t_cross);
        let b_after = radar.blip_brightness(blip, t_cross + 1.0 / blip.decay);
        assert!(
            b_at > 0.95,
            "brightness should be ~1.0 right at the crossing, got {b_at}"
        );
        assert!(
            b_after < b_at,
            "brightness must decay after the beam passes"
        );
    }

    #[test]
    fn flare_shockwave_grows_then_the_blip_fades_out() {
        let radar = Radar::random(12, RECT);
        let blip = &radar.blips[0];
        let sweep = radar.sweep_speed;
        let t_cross = (blip.angle - radar.sweep_phase).rem_euclid(TAU) / sweep;
        let (_, dt0) = radar.blip_state(blip, t_cross + 0.01);
        let (_, dt1) = radar.blip_state(blip, t_cross + 0.5);
        assert!(
            dt1 > dt0,
            "time-since-crossing must grow monotonically within one sweep period"
        );
    }

    #[test]
    fn glyphs_used_are_all_reliable() {
        use re_render::glyph_family::is_reliable;
        for g in ['\u{2500}', '\u{2502}', '\u{25CF}', '\u{2580}'] {
            assert!(
                is_reliable(g),
                "radar glyph {g:?} is not in the reliable set"
            );
        }
        for &(label, _) in &BEARING_TICKS {
            for g in label.chars() {
                assert!(
                    is_reliable(g),
                    "bearing label glyph {g:?} is not in the reliable set"
                );
            }
        }
    }

    #[test]
    fn paint_stays_inside_its_rect() {
        let mut fb = Framebuffer::new(70, 34);
        let rect = Rect::new(4, 2, 60, 30);
        let radar = Radar::random(6, rect);
        for ms in [0u64, 300, 2200, 7000] {
            radar.paint(&mut fb, rect, Duration::from_millis(ms));
        }
        for row in 0..34 {
            for col in 0..70 {
                if !rect.contains(col as i64, row as i64) {
                    assert_eq!(*fb.get(col, row).unwrap(), re_core::Cell::default());
                }
            }
        }
    }

    #[test]
    fn the_disc_reads_as_rings_and_blips_not_just_the_sweep_wedge() {
        // Regression for the 2026-09-24 gallery2 review: fraction-of-spacing
        // ring bands and single-pixel blips were so faint/thin they vanished
        // next to the sweep, leaving a near-empty disc. A representative
        // sample away from the sweep's own wedge must still light up a
        // meaningful number of cells from rings + blip flares + labels alone.
        let mut fb = Framebuffer::new(60, 30);
        let radar = Radar::random(11, RECT);
        radar.paint(&mut fb, RECT, Duration::from_millis(1700));
        let lit = (0..30)
            .flat_map(|r| (0..60).map(move |c| (c, r)))
            .filter(|&(c, r)| *fb.get(c, r).unwrap() != re_core::Cell::default())
            .count();
        assert!(lit > 250, "expected rings + crosshairs + blips + labels to light a meaningful area, only {lit} cells lit");
    }

    #[test]
    fn bearing_and_range_labels_are_actually_painted() {
        // Regression for the 2026-09-25 gallery3 visual review: this test
        // used to check ONLY 'E', which survived by coincidence (see
        // `paint_labels`'s doc) while 'N'/'S'/'W' and every range label were
        // silently erased by `SubrowCanvas::blit`'s glow halo / ring dots.
        // Checking all four bearing ticks plus one range label is what would
        // have caught it.
        let mut fb = Framebuffer::new(60, 30);
        let radar = Radar::random(5, RECT);
        radar.paint(&mut fb, RECT, Duration::from_millis(500));
        let (cx, cy) = RECT.center();
        let max_r = RECT.min_extent() as f32 * radar.radius_frac;
        for &(label, ang) in &BEARING_TICKS {
            let col = (cx + (max_r + 1.5) * ang.cos()).round() as usize;
            let row = (cy + ((max_r + 1.5) * ang.sin()) / CELL_ASPECT).round() as usize;
            let ch = char::from_u32(fb.get(col, row).unwrap().glyph);
            assert_eq!(
                ch,
                label.chars().next(),
                "{label} bearing tick missing at ({col},{row}), found {ch:?}"
            );
        }
        // The innermost ring's range label, on the NE diagonal.
        let label_angle = std::f32::consts::FRAC_PI_4;
        let spacing = max_r / radar.ring_count as f32;
        let r = spacing;
        let col = (cx + r * label_angle.cos()).round() as usize;
        let row = (cy + (r * label_angle.sin()) / CELL_ASPECT).round() as usize;
        let ch = fb.get(col, row).unwrap().glyph;
        assert!(
            char::from_u32(ch).is_some_and(|c| c.is_ascii_digit()),
            "range label missing at ({col},{row}), found glyph {ch}"
        );
    }
}
