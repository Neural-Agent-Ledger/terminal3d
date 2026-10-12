//! `re_render::sky::clouds` — a [`SkyLayer`] of perspective cloud layers:
//! density maps to an edge glyph plus a background tint.
//!
//! Pipeline stage: one layer in the [`crate::sky::SkyProgram`] stack.
//! Inputs: [`SkyCtx`] (ray yaw/pitch, tick, camera ground position) + the cell
//! shaded so far. Output: `bg` tinted by cloud density, with an edge-band
//! glyph where density is thin enough to read as structure rather than mass.
//! Knobs: [`CloudsParams`](re_world::sky_spec::CloudsParams) (layer count, height, wind, cover/softness, scale,
//! haze falloff, shade/lit colours).
//!
//! Model: a flat sky PLANE (not a skybox texture) hit by the ray, so clouds
//! genuinely converge to the horizon and parallax under translation — the
//! plane intersection uses the camera's real WORLD position
//! ([`SkyCtx::cam_xy`]), so walking moves the cloud field the way a
//! world-anchored pattern should.

use re_world::sky_spec::CloudsParams;

use super::{util, SkyCell, SkyCtx, SkyLayer};

/// Coarse direction-bucket grid used ONLY to stagger the wind-driven world
/// position sample — see the comment in `shade()`. Resolution is chosen well
/// above a typical terminal's column/row count so a screen cell reliably
/// lands in its own bucket rather than several screen cells sharing one
/// (which would show visible seams).
const CLOUD_GRID_YAW: u32 = 480;
/// Vertical half of the same direction-bucket grid as [`CLOUD_GRID_YAW`].
const CLOUD_GRID_V: u32 = 120;
/// Only ~1/period of the covered dome recomputes its wind-driven position on
/// any given tick; the rest replay byte-identically. Tuned empirically
/// against `sky::budget_tests` at the 322×86 acceptance size — clouds' own
/// default `cover` leaves headroom, but `storm`'s 0.85-cover deck needs this
/// margin.
const CLOUD_STAGGER_PERIOD: u32 = 24;

/// A compiled cloud layer: the authored [`CloudsParams`] plus nothing else.
#[derive(Clone, Copy, Debug)]
pub struct CloudsLayer {
    p: CloudsParams,
}

impl CloudsLayer {
    /// Wrap authored params into a layer ready for [`SkyLayer::shade`].
    pub fn new(p: CloudsParams) -> Self {
        CloudsLayer { p }
    }

    /// Sample one cloud layer's density/lit/fade state at `ctx`, `qt` (the
    /// tick-quantized time), for a given layer index `li` (0 = cumulus
    /// low/slow, 1 = cirrus high/fast, giving the two layers independent
    /// parallax). `None` below the horizon or where density is negligible.
    fn sample_layer(&self, ctx: &SkyCtx, seed: u32, qt: f32, li: u32) -> Option<(f32, f32, f32)> {
        let height = self.p.height * if li == 0 { 1.0 } else { 3.0 };
        let wind = if li == 0 {
            self.p.wind
        } else {
            (self.p.wind.0 * 1.6, self.p.wind.1 * 1.6)
        };
        let sin_p = ctx.pitch.sin();
        if sin_p <= 0.02 {
            return None; // at/below the horizon: no plane hit within range.
        }
        let d = height / sin_p;
        let cos_p = ctx.pitch.cos();
        let (dir_x, dir_y) = (ctx.yaw.sin(), ctx.yaw.cos());
        let px = ctx.cam_xy.0 + dir_x * d * cos_p + wind.0 * qt;
        let py = ctx.cam_xy.1 + dir_y * d * cos_p + wind.1 * qt;
        let scale = (self.p.scale * 0.05).max(0.001);
        let n = util::fbm4(
            px * scale,
            py * scale,
            seed.wrapping_add(li.wrapping_mul(211)),
        );
        let soft = self.p.soft.max(0.001);
        let rho = util::smoothstep(self.p.cover, self.p.cover + soft, n);
        if rho <= 0.001 {
            return None;
        }
        // Silver-lining rim: one extra fbm4 evaluation at a nearby offset;
        // the density gradient between the two samples reads as a lit edge.
        let n2 = util::fbm4(
            (px + 3.0) * scale,
            (py + 3.0) * scale,
            seed.wrapping_add(li.wrapping_mul(211)),
        );
        let rho2 = util::smoothstep(self.p.cover, self.p.cover + soft, n2);
        let lit = ((rho - rho2) * 4.0).clamp(0.0, 1.0);
        let haze_d = self.p.haze.max(1.0);
        let fade = (-d / haze_d).exp();
        Some((rho, fade, lit))
    }
}

impl SkyLayer for CloudsLayer {
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, seed: u32, hz: u32) -> SkyCell {
        if ctx.pitch <= 0.02 {
            return below;
        }
        // Byte budget: the wind-driven world position is continuous in time,
        // so evaluating it at the LIVE tick would recolour nearly every
        // covered cell every single tick. Stagger it per direction bucket:
        // each bucket only actually advances once every `CLOUD_STAGGER_PERIOD`
        // ticks (offset by its own hash), so at most ~1/period of the covered
        // cells change on any given tick.
        let gx = util::frac_bucket(
            util::wrap_tau(ctx.yaw) / std::f32::consts::TAU,
            CLOUD_GRID_YAW,
        );
        let gy = util::frac_bucket(super::dome_height(ctx.pitch), CLOUD_GRID_V);
        let cell_hash = util::hash3(seed, gx, gy);
        let qt = util::staggered_qt(cell_hash, ctx.tick, hz, CLOUD_STAGGER_PERIOD);
        let mut cell = below;
        for li in 0..self.p.layers.clamp(1, 2) {
            let Some((rho, fade, lit)) = self.sample_layer(ctx, seed, qt, li) else {
                continue;
            };
            let base_col = util::lerp_rgb(self.p.shade, self.p.lit, lit);
            let alpha = (rho * 0.85 * fade).clamp(0.0, 1.0);
            cell.bg = util::lerp_rgb(cell.bg, base_col, alpha);
            // Occlusion: a cloud body hides whatever structure (stars/aurora)
            // the layer below drew, once density passes 0.3.
            if rho > 0.3 {
                cell.glyph = ' ';
            }
            // Edge band: structure goes in the glyph, mass in bg — only a thin
            // ρ window gets a glyph; the interior stays a flat bg fill.
            if (0.15..0.55).contains(&rho) {
                let edge_glyph = if rho < 0.25 {
                    '.'
                } else if rho < 0.35 {
                    '·'
                } else if rho < 0.45 {
                    ':'
                } else {
                    '░'
                };
                cell.glyph = edge_glyph;
                cell.fg = util::lerp_rgb(self.p.shade, self.p.lit, lit.max(0.4));
            }
        }
        cell
    }

    fn is_time_animated(&self) -> bool {
        self.p.wind.0.abs() > 1e-4 || self.p.wind.1.abs() > 1e-4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_at(yaw: f32, pitch: f32, tick: u32) -> SkyCtx {
        SkyCtx {
            yaw,
            pitch,
            tick,
            t: tick as f32 / 12.0,
            hz: 12.0,
            cam_xy: (5.0, 5.0),
            player_yaw: 0.0,
        }
    }

    #[test]
    fn dense_cover_lifts_bg_somewhere_in_the_sweep() {
        let layer = CloudsLayer::new(CloudsParams {
            cover: 0.1,
            soft: 0.4,
            ..CloudsParams::default()
        });
        let below = SkyCell::flat(0x203050);
        let mut changed = false;
        for i in 0..64 {
            let yaw = i as f32 * 0.1;
            let ctx = ctx_at(yaw, 0.5, 0);
            let c = layer.shade(&ctx, below, 11, 12);
            if c.bg != below.bg {
                changed = true;
                break;
            }
        }
        assert!(
            changed,
            "high cover must lift bg somewhere in a 64-column sweep"
        );
    }

    #[test]
    fn below_horizon_is_untouched() {
        let layer = CloudsLayer::new(CloudsParams::default());
        let below = SkyCell::flat(0x102030);
        let ctx = ctx_at(0.0, -0.2, 0);
        let c = layer.shade(&ctx, below, 1, 12);
        assert_eq!(c, below);
    }

    #[test]
    fn edge_glyphs_are_reliable() {
        for g in ['.', '·', ':', '░'] {
            assert!(crate::glyph_family::is_reliable(g), "{g} must be reliable");
        }
    }

    #[test]
    fn wind_marks_the_layer_animated() {
        let still = CloudsLayer::new(CloudsParams {
            wind: (0.0, 0.0),
            ..CloudsParams::default()
        });
        let windy = CloudsLayer::new(CloudsParams::default());
        assert!(!still.is_time_animated());
        assert!(windy.is_time_animated());
    }
}
