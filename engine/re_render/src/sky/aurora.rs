//! `re_render::sky::aurora` — a [`SkyLayer`] painting curtains of colour that
//! shift by altitude, with a rippling curly-underline hem where a curtain's
//! foot crosses the current row.
//!
//! Pipeline stage: one layer in the [`crate::sky::SkyProgram`] stack, sampled per
//! above-horizon ray via [`AuroraLayer::shade`](crate::sky::aurora::AuroraLayer::shade).
//! Inputs: [`SkyCtx`] (ray yaw/pitch, tick) + the cell shaded so far.
//! Output: the same cell with `bg` lifted by the curtain's emission colour and
//! (above a brightness threshold) a ray glyph + coloured underline hem.
//! Knobs: [`AuroraParams`](re_world::sky_spec::AuroraParams) (curtain count, band height/decay, drift, palette).
//!
//! Column-separable by construction: every curtain is a function of yaw and
//! time only, never of pitch, so there is no volumetric raymarch — a whole
//! vertical curtain band is priced (and updates) as one unit.

use std::f32::consts::TAU;

use re_core::{CellStyle, UnderlineStyle};
use re_world::sky_spec::{AuroraPalette, AuroraParams};

use super::{util, SkyCell, SkyCtx, SkyLayer};

/// A compiled aurora layer: the authored [`AuroraParams`] plus nothing else —
/// all per-ray state is computed fresh in [`SkyLayer::shade`].
#[derive(Clone, Copy, Debug)]
pub struct AuroraLayer {
    p: AuroraParams,
}

impl AuroraLayer {
    /// Wrap authored params into a layer ready for [`SkyLayer::shade`].
    pub fn new(p: AuroraParams) -> Self {
        AuroraLayer { p }
    }
}

/// Coarse YAW-only bucket grid for staggering the curtains' phase: every
/// driver of the curtain colour (`foot`, `ray`, `surge`, `drift_t`) is a
/// function of `u` (yaw fraction) and `qt` only — never of `v` — so a whole
/// vertical curtain-band updates in lockstep, which is exactly the right
/// granularity (freezing a curtain freezes the WHOLE band, not a scattering
/// of individual rows within it).
const AURORA_YAW_BUCKETS: u32 = 512;
/// Only ~1/period of the visible curtain bands recompute their phase on any
/// given tick. Tuned against `sky::budget_tests` at 322×86.
const AURORA_STAGGER_PERIOD: u32 = 36;

/// The colour-by-altitude ramp, grounded in the look of real emission lines
/// (green/red oxygen, violet nitrogen). `h` is the curtain-local height
/// fraction (0 = hem, 1 = decayed top).
fn altitude_ramp(palette: AuroraPalette, h: f32) -> u32 {
    let h = h.clamp(0.0, 1.0);
    let stops: [(f32, u32); 4] = match palette {
        AuroraPalette::Oxygen => [
            (0.0, 0xC04890),
            (0.05, 0x3DFF8C),
            (0.5, 0x20C8A0),
            (1.0, 0x7A2050),
        ],
        AuroraPalette::Violet => [
            (0.0, 0x5A2878),
            (0.05, 0x6A40C0),
            (0.5, 0x4A2890),
            (1.0, 0x281050),
        ],
        AuroraPalette::Mono => [
            (0.0, 0x0A3018),
            (0.05, 0x1A6030),
            (0.5, 0x30D060),
            (1.0, 0x60FF90),
        ],
    };
    for w in stops.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if h <= t1 || (t1 - t0).abs() < 1e-6 {
            let t = if t1 > t0 { (h - t0) / (t1 - t0) } else { 0.0 };
            return util::lerp_rgb(c0, c1, t.clamp(0.0, 1.0));
        }
    }
    stops[3].1
}

impl SkyLayer for AuroraLayer {
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, seed: u32, hz: u32) -> SkyCell {
        let mut cell = below;
        let v = super::dome_height(ctx.pitch);
        let u = util::wrap_tau(ctx.yaw) / TAU;
        // Byte budget: see the `AURORA_STAGGER_PERIOD` doc comment.
        let gx = util::frac_bucket(u, AURORA_YAW_BUCKETS);
        let qt = util::staggered_qt(util::hash2(seed, gx), ctx.tick, hz, AURORA_STAGGER_PERIOD);
        let p = &self.p;
        let n = p.curtains.max(1);

        let mut total_i = 0.0f32;
        let mut best_i = 0.0f32;
        let mut h_for_color = 0.0f32;
        let mut ray_val = 0.0f32;
        let mut min_abs_h = f32::MAX;

        for k in 0..n {
            let k_seed = seed.wrapping_add(k.wrapping_mul(97).wrapping_add(11));
            let f_k = 3.0 + (k as f32) * 1.3;
            let s_k = 0.015 + 0.01 * (k as f32);
            let phi_k = util::hash01(util::hash2(k_seed, 1)) * TAU;
            let foot = p.band_lo
                + (p.band_hi - p.band_lo) * util::value_noise1(u * f_k + qt * s_k + phi_k, k_seed);
            let ray = 0.55 + 0.45 * util::value_noise1(u * 40.0 + qt * 0.6, k_seed.wrapping_add(3));
            let height_k = p.height.max(0.02);
            let h = (v - foot) / height_k;
            if h.abs() < min_abs_h {
                min_abs_h = h.abs();
            }
            if h >= 0.0 {
                let i_k = (-h * p.decay.max(0.1)).exp() * ray * p.activity;
                let surge_phase = util::hash01(util::hash2(k_seed, 5));
                let u_surge = (qt * (0.05 + p.surge_rate) + surge_phase).fract();
                let du = (u - u_surge).abs();
                let du = du.min(1.0 - du);
                let surge = 1.0 + 1.5 * (-(du * du) / 0.012).exp();
                let i_k2 = i_k * surge;
                if i_k2 > best_i {
                    best_i = i_k2;
                    h_for_color = h;
                    ray_val = ray;
                }
                total_i += i_k2;
            }
        }

        let i = total_i.clamp(0.0, 1.6);
        if i > 0.02 {
            let drift_t = if p.drift_period > 1e-3 {
                (qt / p.drift_period).fract()
            } else {
                0.0
            };
            let colour = altitude_ramp(p.palette, h_for_color.clamp(0.0, 1.0));
            let colour = util::rotate_hue_rgb(colour, 15.0 * (drift_t * TAU).sin());
            cell.bg = util::add_rgb_clamped(
                cell.bg,
                ((colour >> 16) & 0xFF) as f32 * i * 0.55,
                ((colour >> 8) & 0xFF) as f32 * i * 0.55,
                (colour & 0xFF) as f32 * i * 0.55,
            );
            // Stars drawn under the aurora show through, dimmed by 1 - 0.5*I.
            if below.glyph != ' ' {
                cell.fg = util::scale_rgb(below.fg, (1.0 - 0.5 * i).clamp(0.2, 1.0));
                cell.glyph = below.glyph;
            }
            if i > 0.25 {
                let glyph = if ray_val > 0.85 {
                    '│'
                } else if ray_val > 0.65 {
                    '¦'
                } else if ray_val > 0.45 {
                    '∶'
                } else {
                    '\''
                };
                cell.glyph = glyph;
                cell.fg = util::scale_rgb(colour, 1.3);
                if i > 1.0 {
                    cell.style = cell.style.with_flags(CellStyle::BOLD);
                }
            }
        }

        // Hem: the rippling lower edge, wherever a curtain's foot crosses `v`.
        if min_abs_h < 0.10 {
            let shape = if best_i > 1.1 {
                UnderlineStyle::Double
            } else {
                UnderlineStyle::Curly
            };
            let hem_colour = altitude_ramp(p.palette, 0.0);
            cell.style = cell
                .style
                .with_underline_style(shape)
                .with_underline_color(hem_colour);
        }
        cell
    }

    fn is_time_animated(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_at(yaw: f32, pitch: f32, t: f32) -> SkyCtx {
        SkyCtx {
            yaw,
            pitch,
            tick: (t * 12.0) as u32,
            t,
            hz: 12.0,
            cam_xy: (0.0, 0.0),
            player_yaw: 0.0,
        }
    }

    #[test]
    fn aurora_lifts_bg_inside_the_band() {
        let layer = AuroraLayer::new(AuroraParams::default());
        let below = SkyCell::flat(0x04060E);
        // Sweep pitch to find a cell inside the authored band [0.18,0.55].
        let mut lifted = false;
        for i in 0..40 {
            let v = 0.18 + (i as f32 / 39.0) * (0.55 - 0.18);
            let pitch = super::super::V_LOW + v * (super::super::HALF_PI - super::super::V_LOW);
            let ctx = ctx_at(0.4, pitch, 1.0);
            let c = layer.shade(&ctx, below, 3, 12);
            if c.bg != below.bg {
                lifted = true;
                break;
            }
        }
        assert!(lifted, "some cell inside the aurora band must lift bg");
    }

    #[test]
    fn aurora_glyphs_are_reliable() {
        for g in ['│', '¦', '∶', '\''] {
            assert!(crate::glyph_family::is_reliable(g), "{g} must be reliable");
        }
    }

    #[test]
    fn altitude_ramp_hem_differs_from_top() {
        let hem = altitude_ramp(AuroraPalette::Oxygen, 0.0);
        let top = altitude_ramp(AuroraPalette::Oxygen, 1.0);
        assert_ne!(hem, top);
    }

    #[test]
    fn same_tick_is_byte_static() {
        let layer = AuroraLayer::new(AuroraParams::default());
        let below = SkyCell::flat(0x04060E);
        let mut a = ctx_at(
            0.7,
            super::super::V_LOW + 0.3 * (super::super::HALF_PI - super::super::V_LOW),
            10.0 / 12.0,
        );
        a.tick = 10;
        let mut b = a;
        b.t = 10.4 / 12.0;
        assert_eq!(layer.shade(&a, below, 5, 12), layer.shade(&b, below, 5, 12));
    }
}
