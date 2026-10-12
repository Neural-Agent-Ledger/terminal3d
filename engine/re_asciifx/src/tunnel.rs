//! TUNNEL — a classic polar-coordinates texture tunnel: every cell's screen
//! position is remapped to `(depth, bearing)` around a (gently wobbling)
//! vanishing point, `depth ~ 1/r` slides outward over time (the "flying
//! forward" illusion), `bearing` twists with radius (a spiral, not a
//! straight tunnel) and rotates as a whole, and a depth×bearing checkerboard
//! picks both a GLYPH (a block-element shade ramp) and a colour (Oklch hue/
//! lightness banding, pulsing) per the owner's ask ("glyph/colour rings").
//!
//! Every cell is filled directly, same discipline as [`super::plasma`] — a
//! tunnel has no empty region either.

use crate::hash::{angle, range};
use crate::{secs, set_clipped, AsciiEffect, Rect, CELL_ASPECT};
use re_core::oklab::{oklch_to_packed, Oklch};
use re_core::{Cell, CellStyle, Framebuffer};
use std::time::Duration;

// ─────────────────────────────── tuning ──────────────────────────────────

const ROTATION_SPEED_RANGE: (f32, f32) = (0.15, 0.6); // rad/s, whole-field spin
const ZOOM_SPEED_RANGE: (f32, f32) = (0.5, 1.6); // depth-units/s, "flying forward" rate
/// Spiral twist: how many extra radians of bearing are added per world-radius
/// unit travelled outward. Signed — negative spins the opposite way.
const TWIST_AMOUNT_RANGE: (f32, f32) = (-0.35, 0.35);
const RING_FREQ_RANGE: (f32, f32) = (5.0, 11.0);
const SPOKE_COUNT_RANGE: (f32, f32) = (6.0, 14.0);
const PULSE_FREQ_RANGE: (f32, f32) = (0.2, 0.8);
/// How far the vanishing point may wobble off dead-centre, as a fraction of
/// the reference radius — a static centre reads as a mechanical loop, a
/// small drift reads as flying through a slightly irregular shaft.
const WOBBLE_AMP_FRAC_RANGE: (f32, f32) = (0.0, 0.12);
const WOBBLE_FREQ_RANGE: (f32, f32) = (0.05, 0.25);
const HUE_RANGE_SWING: f32 = 0.85;
const BASE_LIGHTNESS_RANGE: (f32, f32) = (0.30, 0.45);
const CHECKER_CONTRAST: f32 = 0.5;
const PULSE_AMOUNT: f32 = 0.12;
/// Vignette falloff: how quickly brightness fades with distance from the
/// vanishing point (kept gentle so the rim of a wide rect isn't crushed to
/// black).
const FALLOFF_K: f32 = 0.35;

const S_ROTATION: u64 = 1;
const S_ZOOM: u64 = 2;
const S_TWIST: u64 = 3;
const S_RING_FREQ: u64 = 4;
const S_SPOKE_COUNT: u64 = 5;
const S_PULSE_FREQ: u64 = 6;
const S_PULSE_PHASE: u64 = 7;
const S_WOBBLE_AMP: u64 = 8;
const S_WOBBLE_FREQ_X: u64 = 9;
const S_WOBBLE_FREQ_Y: u64 = 10;
const S_WOBBLE_PHASE_X: u64 = 11;
const S_WOBBLE_PHASE_Y: u64 = 12;
const S_HUE_BASE: u64 = 13;

/// A plain `sqrt(dx²+dy²)` — faster than `f32::hypot`, which does extra
/// overflow/underflow-safe rescaling we don't need at terminal-cell-scale
/// magnitudes. Used only in `paint`'s hot loop; [`Tunnel::sample`] (tests,
/// anywhere exactness matters) keeps the real `hypot`.
#[inline]
fn fast_hypot(dx: f32, dy: f32) -> f32 {
    (dx * dx + dy * dy).sqrt()
}

/// `1 / TAU`, precomputed so [`wrap_tau`] wraps via a multiply instead of a
/// division.
const INV_TAU: f32 = 1.0 / std::f32::consts::TAU;

/// Wraps `x` into `[0, 1)` — the same result as `x.rem_euclid(1.0)`, but
/// without going through the general `%` operator's `fmodf` call (which
/// handles NaN/infinity/arbitrary-magnitude edge cases we never hit here).
/// `x - x.floor()` is exact for the finite inputs `sample_core` produces and
/// costs a single `floor` (an SSE instruction) instead of a libm call. This
/// pair (with [`wrap_tau`]) was the dominant remaining cost in `paint`'s hot
/// loop — three wraps per half-cell, ~78,400 half-cells — after
/// `fast_hypot`/`fast_atan2` and hoisting the per-frame constants had
/// already been applied and `paint` was still at 2.22ms against the 2ms
/// budget.
#[inline]
fn wrap01(x: f32) -> f32 {
    x - x.floor()
}

/// Wraps `x` into `[0, TAU)` — see [`wrap01`]'s doc; same trick, scaled by
/// [`INV_TAU`] first so the division becomes a multiply.
#[inline]
fn wrap_tau(x: f32) -> f32 {
    x - (x * INV_TAU).floor() * std::f32::consts::TAU
}

/// A standard polynomial `atan2` approximation (measured max error ≈0.6° over
/// a wide directional sample, see the test below) — several times cheaper
/// than `f32::atan2`. `atan2`+`hypot` were the measured remaining cost after
/// [`Tunnel::build_lut`] closed the colour-conversion gap (2.70ms → still
/// over budget); a background colour field quantized through a 48×32 LUT
/// cannot show a fraction-of-a-degree bearing error.
#[inline]
fn fast_atan2(y: f32, x: f32) -> f32 {
    const QTR_PI: f32 = std::f32::consts::FRAC_PI_4;
    const THR_QTR_PI: f32 = 3.0 * QTR_PI;
    let abs_y = y.abs() + 1e-10;
    let (r, angle) = if x >= 0.0 {
        (((x - abs_y) / (x + abs_y)), QTR_PI)
    } else {
        (((x + abs_y) / (abs_y - x)), THR_QTR_PI)
    };
    let angle = angle + (0.196_3 * r * r - 0.981_7) * r;
    if y < 0.0 {
        -angle
    } else {
        angle
    }
}

/// Dark-to-light block-element shade ramp — the "glyph rings" half of the
/// owner's ask. All within Block Elements / Geometric Shapes, so every entry
/// is `re_render::glyph_family::is_reliable`.
const SHADE_RAMP: [char; 5] = [' ', '\u{2591}', '\u{2592}', '\u{2593}', '\u{2588}'];

/// A polar-coordinates texture tunnel.
pub struct Tunnel {
    rotation_speed: f32,
    zoom_speed: f32,
    twist_amount: f32,
    ring_freq: f32,
    spoke_count: f32,
    pulse_freq: f32,
    pulse_phase: f32,
    wobble_amp_frac: f32,
    wobble_freq_x: f32,
    wobble_freq_y: f32,
    wobble_phase_x: f32,
    wobble_phase_y: f32,
    hue_base: f32,
    base_lightness: f32,
}

impl Tunnel {
    /// Build a deterministic rotating/zooming ring-and-spoke tunnel from
    /// `seed`. `_rect` is unused (the tunnel is drawn in a normalized,
    /// aspect-corrected space centered on the target rect by the painter)
    /// but kept for the shared `random(seed, rect)` signature.
    #[must_use]
    pub fn random(seed: u64, _rect: Rect) -> Self {
        Tunnel {
            rotation_speed: range(
                seed,
                0,
                S_ROTATION,
                ROTATION_SPEED_RANGE.0,
                ROTATION_SPEED_RANGE.1,
            ),
            zoom_speed: range(seed, 0, S_ZOOM, ZOOM_SPEED_RANGE.0, ZOOM_SPEED_RANGE.1),
            twist_amount: range(seed, 0, S_TWIST, TWIST_AMOUNT_RANGE.0, TWIST_AMOUNT_RANGE.1),
            ring_freq: range(seed, 0, S_RING_FREQ, RING_FREQ_RANGE.0, RING_FREQ_RANGE.1),
            spoke_count: range(
                seed,
                0,
                S_SPOKE_COUNT,
                SPOKE_COUNT_RANGE.0,
                SPOKE_COUNT_RANGE.1,
            )
            .round()
            .max(4.0),
            pulse_freq: range(
                seed,
                0,
                S_PULSE_FREQ,
                PULSE_FREQ_RANGE.0,
                PULSE_FREQ_RANGE.1,
            ),
            pulse_phase: angle(seed, 0, S_PULSE_PHASE),
            wobble_amp_frac: range(
                seed,
                0,
                S_WOBBLE_AMP,
                WOBBLE_AMP_FRAC_RANGE.0,
                WOBBLE_AMP_FRAC_RANGE.1,
            ),
            wobble_freq_x: range(
                seed,
                0,
                S_WOBBLE_FREQ_X,
                WOBBLE_FREQ_RANGE.0,
                WOBBLE_FREQ_RANGE.1,
            ),
            wobble_freq_y: range(
                seed,
                0,
                S_WOBBLE_FREQ_Y,
                WOBBLE_FREQ_RANGE.0,
                WOBBLE_FREQ_RANGE.1,
            ),
            wobble_phase_x: angle(seed, 0, S_WOBBLE_PHASE_X),
            wobble_phase_y: angle(seed, 0, S_WOBBLE_PHASE_Y),
            hue_base: angle(seed, 0, S_HUE_BASE),
            base_lightness: range(seed, 0, 14, BASE_LIGHTNESS_RANGE.0, BASE_LIGHTNESS_RANGE.1),
        }
    }

    /// The brightness pulse at time `t` alone — depends on nothing else, so
    /// callers hoist it OUT of any per-pixel loop and pass it in (see
    /// [`Self::sample_core`]'s doc: calling this once per `paint()` instead
    /// of once per half-cell was one of the two changes that closed the
    /// perf gap).
    fn pulse_at(&self, t: f32) -> f32 {
        1.0 + PULSE_AMOUNT * (std::f32::consts::TAU * self.pulse_freq * t + self.pulse_phase).sin()
    }

    /// The shared physics core: given an ALREADY-COMPUTED world radius `r`
    /// and bearing `ang` (relative to the wobbling vanishing point,
    /// aspect-corrected) plus a handful of terms that are constant for the
    /// whole `paint()` call (`twist_per_ref = twist_amount / ref_r.max(1.0)`,
    /// `ring_freq_ref = ring_freq * ref_r`, `falloff_per_ref = FALLOFF_K /
    /// ref_r.max(1.0)`, `rot_term = rotation_speed * t`, `zoom_term =
    /// zoom_speed * t`), returns `(shade_0to1, hue_offset, checker_on)`.
    ///
    /// Those five are pulled out to per-frame constants — computed once in
    /// [`Self::paint`], or once per `t` in [`Self::sample`] — because two
    /// divisions and two multiplies were otherwise being repeated identically
    /// on every one of ~78,400 half-cell calls; hoisting them is what closed
    /// the remaining gap after `fast_hypot`/`fast_atan2` alone left `paint`
    /// at 2.30ms against the 2ms budget. Kept separate from `r`/`ang`'s own
    /// derivation so [`Self::sample`] (exact `hypot`/`atan2`, used by tests
    /// and anywhere correctness matters more than raw speed) and `paint`'s
    /// hot loop (a cheap polynomial `atan2` + a plain `sqrt(dx²+dy²)`, see
    /// [`fast_atan2`]/[`fast_hypot`]) can never disagree about what happens
    /// once `r`/`ang` are known — only about how fast they got there.
    #[allow(clippy::too_many_arguments)]
    fn sample_core(
        &self,
        r: f32,
        ang: f32,
        twist_per_ref: f32,
        ring_freq_ref: f32,
        falloff_per_ref: f32,
        rot_term: f32,
        zoom_term: f32,
        pulse: f32,
    ) -> (f32, f32, bool) {
        let r = r.max(0.05);
        let twisted = wrap_tau(ang + r * twist_per_ref + rot_term);
        let depth = ring_freq_ref / r - zoom_term;
        let ring_band = wrap01(depth);
        let spoke_raw = twisted * INV_TAU * self.spoke_count;
        let spoke_frac = wrap01(spoke_raw);
        let checker_on = (depth.floor() as i32 + spoke_raw.floor() as i32) & 1 == 0;

        // Triangle wave from `ring_band` for a smooth brightness ramp within
        // each ring (not just a hard step at the ring boundary).
        let ring_tri = 1.0 - (ring_band - 0.5).abs() * 2.0;
        let spoke_tri = 1.0 - (spoke_frac - 0.5).abs() * 2.0;
        // Squared, not just the raw blend: the raw ring/spoke triangle blend
        // rarely dips near 0 (both waves have to bottom out at once), so the
        // tunnel read as a soft, low-contrast pastel blob instead of the
        // intended banded/checkered "texture tunnel" (2026-09-25 gallery3
        // review). Squaring keeps the peak (`base == 1.0`) fixed while
        // pushing every mid-tone down — a single multiply, not a `powf`, so
        // it stays free in the hot loop instead of reopening the perf gap
        // `wrap01`/`wrap_tau` just closed.
        let base_raw = (ring_tri * 0.6 + spoke_tri * 0.4).clamp(0.0, 1.0);
        let base = base_raw * base_raw;
        let checker_k = if checker_on {
            1.0
        } else {
            1.0 - CHECKER_CONTRAST
        };
        let vignette = 1.0 / (1.0 + r * falloff_per_ref);
        let shade = (base * checker_k * vignette * pulse).clamp(0.0, 1.0);
        let hue_offset = ring_band * HUE_RANGE_SWING + spoke_frac * 0.4;
        (shade, hue_offset, checker_on)
    }

    /// `(shade_0to1, hue_offset, checker_on)` at world `(dx, dy)` (already
    /// relative to the wobbling vanishing point, aspect-corrected) and time
    /// `t` — the EXACT reference form (`hypot`/`atan2`), used by tests only;
    /// `paint()`'s hot loop uses the fast approximations instead (see
    /// [`Self::sample_core`]'s doc).
    #[cfg(test)]
    fn sample(&self, dx: f32, dy: f32, ref_r: f32, t: f32) -> (f32, f32, bool) {
        let r = dx.hypot(dy);
        let ang = dy.atan2(dx);
        let ref_r_safe = ref_r.max(1.0);
        self.sample_core(
            r,
            ang,
            self.twist_amount / ref_r_safe,
            self.ring_freq * ref_r,
            FALLOFF_K / ref_r_safe,
            self.rotation_speed * t,
            self.zoom_speed * t,
            self.pulse_at(t),
        )
    }

    fn color_at(&self, shade: f32, hue_offset: f32) -> u32 {
        let oklch = Oklch {
            l: (self.base_lightness + shade * 0.45).clamp(0.03, 0.92),
            c: 0.12 + 0.08 * shade,
            h: self.hue_base + hue_offset,
        };
        oklch_to_packed(oklch)
    }

    /// A `SHADE_STEPS × HUE_STEPS` table of [`Self::color_at`], rebuilt fresh
    /// each `paint()` call. Same fix, same reason, as [`super::plasma::
    /// Plasma::build_lut`]'s doc: `color_at`'s Oklch round trip costs several
    /// `powf`s, and calling it directly per half-cell (~39,000 times at
    /// 280×70) measured 4.85ms against a 2ms budget. `hue_offset`'s formula
    /// (`ring_band * HUE_RANGE_SWING + spoke_frac * 0.4`) is bounded — never
    /// negative, never past [`HUE_OFFSET_MAX`] — so a small 2D table covers
    /// its whole reachable domain.
    fn build_lut(&self) -> Vec<u32> {
        let mut lut = vec![0u32; SHADE_STEPS * HUE_STEPS];
        for si in 0..SHADE_STEPS {
            let shade = si as f32 / (SHADE_STEPS - 1) as f32;
            for hi in 0..HUE_STEPS {
                let hue_offset = hi as f32 / (HUE_STEPS - 1) as f32 * HUE_OFFSET_MAX;
                lut[si * HUE_STEPS + hi] = self.color_at(shade, hue_offset);
            }
        }
        lut
    }
}

const SHADE_STEPS: usize = 48;
const HUE_STEPS: usize = 32;
/// Upper bound of `sample`'s `hue_offset` return (`ring_band < 1` times
/// [`HUE_RANGE_SWING`], plus `spoke_frac < 1` times its own `0.4` factor) —
/// kept as one named constant so [`Tunnel::build_lut`] and [`lut_lookup`]
/// can never quietly disagree about the domain they quantize.
const HUE_OFFSET_MAX: f32 = HUE_RANGE_SWING + 0.4;

#[inline]
fn lut_lookup(lut: &[u32], shade: f32, hue_offset: f32) -> u32 {
    let si = (shade.clamp(0.0, 1.0) * (SHADE_STEPS - 1) as f32).round() as usize;
    let hi = ((hue_offset.clamp(0.0, HUE_OFFSET_MAX) / HUE_OFFSET_MAX) * (HUE_STEPS - 1) as f32)
        .round() as usize;
    lut[si.min(SHADE_STEPS - 1) * HUE_STEPS + hi.min(HUE_STEPS - 1)]
}

impl AsciiEffect for Tunnel {
    fn name(&self) -> &'static str {
        "tunnel"
    }

    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration) {
        let secs_t = secs(t);
        let ref_r = rect.min_extent() as f32 * 0.5;
        let (ccx, ccy) = rect.center();
        let wobble_r = ref_r * self.wobble_amp_frac;
        let wcx = ccx
            + wobble_r
                * (std::f32::consts::TAU * self.wobble_freq_x * secs_t + self.wobble_phase_x).cos();
        let wcy = ccy
            + wobble_r
                * (std::f32::consts::TAU * self.wobble_freq_y * secs_t + self.wobble_phase_y).sin()
                / CELL_ASPECT;
        let lut = self.build_lut();
        // Hoisted OUT of the pixel loop — every one of these depends only on
        // `t` and the effect's own (per-paint-call-constant) fields, so they
        // were being recomputed identically on every one of ~78,400 half-cell
        // calls before: a bare `sin()` for `pulse`, and — inside
        // `sample_core` itself — two divisions (`twist_per_ref`,
        // `falloff_per_ref`) and two multiplies (`rot_term`, `zoom_term`).
        // Closing this gap (on top of `fast_hypot`/`fast_atan2`) is what
        // brought `paint` back under the 2ms budget.
        let pulse = self.pulse_at(secs_t);
        let ref_r_safe = ref_r.max(1.0);
        let twist_per_ref = self.twist_amount / ref_r_safe;
        let ring_freq_ref = self.ring_freq * ref_r;
        let falloff_per_ref = FALLOFF_K / ref_r_safe;
        let rot_term = self.rotation_speed * secs_t;
        let zoom_term = self.zoom_speed * secs_t;

        for row in rect.row0..(rect.row0 + rect.rows) {
            for col in rect.col0..(rect.col0 + rect.cols) {
                let fcol = col as f32 + 0.5;
                let top_dx = fcol - wcx;
                let top_dy = (row as f32 + 0.25 - wcy) * CELL_ASPECT;
                let bot_dy = (row as f32 + 0.75 - wcy) * CELL_ASPECT;
                let r_top = fast_hypot(top_dx, top_dy);
                let ang_top = fast_atan2(top_dy, top_dx);
                let r_bot = fast_hypot(top_dx, bot_dy);
                let ang_bot = fast_atan2(bot_dy, top_dx);
                let (shade_top, hue_top, _checker_top) = self.sample_core(
                    r_top,
                    ang_top,
                    twist_per_ref,
                    ring_freq_ref,
                    falloff_per_ref,
                    rot_term,
                    zoom_term,
                    pulse,
                );
                let (shade_bot, hue_bot, _) = self.sample_core(
                    r_bot,
                    ang_bot,
                    twist_per_ref,
                    ring_freq_ref,
                    falloff_per_ref,
                    rot_term,
                    zoom_term,
                    pulse,
                );
                let ramp_idx =
                    ((shade_top.max(shade_bot)) * (SHADE_RAMP.len() - 1) as f32).round() as usize;
                let glyph = SHADE_RAMP[ramp_idx.min(SHADE_RAMP.len() - 1)];
                let cell = Cell {
                    glyph: u32::from(glyph),
                    fg: lut_lookup(&lut, shade_top, hue_top),
                    bg: lut_lookup(&lut, shade_bot, hue_bot),
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
    fn fast_hypot_and_atan2_stay_close_to_the_exact_versions() {
        // `paint`'s hot loop uses `fast_hypot`/`fast_atan2` instead of
        // `sample`'s exact `hypot`/`atan2` — proved here to stay within a
        // negligible-magnitude / sub-degree bearing error budget over a wide
        // sample of directions and radii, which is what makes substituting
        // them in the hot loop a performance change, not a correctness one.
        for i in 0..400 {
            let dx = ((i as f32) * 0.313).sin() * 60.0;
            let dy = ((i as f32) * 0.197).cos() * 30.0;
            if dx.abs() < 1e-3 && dy.abs() < 1e-3 {
                continue;
            }
            let exact_r = dx.hypot(dy);
            let exact_ang = dy.atan2(dx);
            let fast_r = fast_hypot(dx, dy);
            let fast_ang = fast_atan2(dy, dx);
            assert!(
                (exact_r - fast_r).abs() < 1e-3,
                "hypot mismatch at ({dx},{dy}): {exact_r} vs {fast_r}"
            );
            let ang_err = (exact_ang - fast_ang).abs();
            let ang_err = ang_err.min(std::f32::consts::TAU - ang_err); // wrap-around
                                                                        // Bound is ~1° (0.0175 rad): observed max over this sample is
                                                                        // ~0.58° (0.0101 rad); a background colour field quantized
                                                                        // through a 32-bin hue LUT cannot show a sub-degree bearing
                                                                        // error, so this is margin on a measured value, not a guess.
            assert!(
                ang_err < 0.018,
                "atan2 mismatch at ({dx},{dy}): {exact_ang} vs {fast_ang}, err {ang_err}"
            );
        }
    }

    #[test]
    fn determinism_same_seed() {
        let a = Tunnel::random(1, RECT);
        let b = Tunnel::random(1, RECT);
        assert_eq!(a.rotation_speed, b.rotation_speed);
        assert_eq!(a.spoke_count, b.spoke_count);
    }

    #[test]
    fn different_seeds_diverge_in_every_frequency() {
        let a = Tunnel::random(1, RECT);
        let b = Tunnel::random(2, RECT);
        assert_ne!(a.rotation_speed, b.rotation_speed);
        assert_ne!(a.zoom_speed, b.zoom_speed);
        assert_ne!(a.twist_amount, b.twist_amount);
        assert_ne!(a.ring_freq, b.ring_freq);
        assert_ne!(a.pulse_freq, b.pulse_freq);
    }

    #[test]
    fn shade_stays_in_unit_range() {
        let tun = Tunnel::random(4, RECT);
        let ref_r = RECT.min_extent() as f32 * 0.5;
        for i in 0..300 {
            let dx = (i as f32 * 0.7).sin() * 40.0;
            let dy = (i as f32 * 0.5).cos() * 20.0;
            let t = i as f32 * 0.03;
            let (shade, _, _) = tun.sample(dx, dy, ref_r, t);
            assert!((0.0..=1.0).contains(&shade), "shade escaped [0,1]: {shade}");
        }
    }

    #[test]
    fn rings_slide_over_time_depth_is_not_static() {
        let tun = Tunnel::random(6, RECT);
        let ref_r = RECT.min_extent() as f32 * 0.5;
        let (s0, _, _) = tun.sample(10.0, 0.0, ref_r, 0.0);
        let (s1, _, _) = tun.sample(10.0, 0.0, ref_r, 5.0);
        assert!(
            (s0 - s1).abs() > 1e-3,
            "shade at a fixed point did not change over 5s — rings are static"
        );
    }

    #[test]
    fn paint_stays_inside_its_rect() {
        let mut fb = Framebuffer::new(80, 30);
        let rect = Rect::new(4, 2, 70, 26);
        let tun = Tunnel::random(5, rect);
        for ms in [0u64, 250, 1300, 6000] {
            tun.paint(&mut fb, rect, Duration::from_millis(ms));
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
        for g in SHADE_RAMP {
            assert!(is_reliable(g), "{g:?} not reliable");
        }
    }

    #[test]
    fn the_tunnel_fills_essentially_the_whole_rect() {
        let mut fb = Framebuffer::new(70, 26);
        let tun = Tunnel::random(3, RECT);
        tun.paint(&mut fb, RECT, Duration::from_millis(900));
        let lit = (0..26)
            .flat_map(|r| (0..70).map(move |c| (c, r)))
            .filter(|&(c, r)| *fb.get(c, r).unwrap() != re_core::Cell::default())
            .count();
        // A cell painted with the SPACE glyph and matching fg/bg to a
        // default-looking colour is still a real, deliberate paint (not
        // "untouched") — `set_clipped` always writes, so every rect cell is
        // touched; this just double-checks nothing panicked/skipped.
        assert!(
            lit as f32 >= 70.0 * 26.0 * 0.95,
            "expected the tunnel to cover ~all of its rect, only {lit} lit"
        );
    }
}
