//! `re_render::sky::void` — four [`SkyLayer`]s for the surreal sky family:
//! [`VoidLayer`](crate::sky::void::VoidLayer) (a near-black emptiness with sparse drifting points),
//! [`FractalLayer`](crate::sky::void::FractalLayer) (an infinite log-polar zoom), [`WatcherLayer`](crate::sky::void::WatcherLayer) (a giant
//! iris whose pupil tracks the player's facing), and [`MirrorLayer`](crate::sky::void::MirrorLayer) (a dark
//! silhouette suggesting the level's floor plan seen from below).
//!
//! Pipeline stage: layers in the [`crate::sky::SkyProgram`] stack.
//! Inputs: [`SkyCtx`] (ray yaw/pitch, tick, player yaw) + the cell shaded so
//! far. Output: each layer either replaces or blends `bg`/`glyph`/`fg`; see
//! each layer's own doc for its exact contract.
//! Knobs: [`VoidParams`](re_world::sky_spec::VoidParams), [`FractalParams`](re_world::sky_spec::FractalParams), [`WatcherParams`](re_world::sky_spec::WatcherParams),
//! [`MirrorParams`](re_world::sky_spec::MirrorParams).

use std::f32::consts::TAU;

use re_core::{CellStyle, UnderlineStyle};
use re_world::sky_spec::{FractalParams, MirrorParams, VoidParams, WatcherParams};

use super::{util, SkyCell, SkyCtx, SkyLayer};

// ─────────────────────────────── void ───────────────────────────────────────

/// Fixed direction grid the sparse drifting points are hashed against.
const VOID_GRID_W: u32 = 300;
/// Vertical half of the same fixed grid as [`VOID_GRID_W`].
const VOID_GRID_H: u32 = 50;

/// A compiled void layer: the authored [`VoidParams`] plus nothing else.
#[derive(Clone, Copy, Debug)]
pub struct VoidLayer {
    p: VoidParams,
}

impl VoidLayer {
    /// Wrap authored params into a layer ready for [`SkyLayer::shade`].
    pub fn new(p: VoidParams) -> Self {
        VoidLayer { p }
    }
}

impl SkyLayer for VoidLayer {
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, seed: u32, hz: u32) -> SkyCell {
        let qt = ctx.tick as f32 / hz.max(1) as f32;
        let mut cell = below;
        cell.bg = util::lerp_rgb(cell.bg, 0x020204, 0.9);
        cell.fg = cell.bg;
        let yaw = util::wrap_tau(ctx.yaw);
        let v = super::dome_height(ctx.pitch);
        // "Drifts toward the view": points slide DOWN the dome (toward the
        // horizon, where the camera is) over time.
        let vv = (v - self.p.drift * qt * 0.05).rem_euclid(1.0);
        let gx = ((yaw / TAU) * VOID_GRID_W as f32).floor() as u32 % VOID_GRID_W;
        let gy = ((vv * VOID_GRID_H as f32).floor() as u32) % VOID_GRID_H;
        let hh = util::hash3(seed, gx, gy);
        if util::hash01(hh) < 0.01 {
            cell.glyph = '·';
            cell.fg = 0x303040;
        }
        cell
    }

    fn is_time_animated(&self) -> bool {
        self.p.drift.abs() > 1e-4
    }
}

// ────────────────────────────── fractal ─────────────────────────────────────

/// Coarse `(u,v)` bucket grid for staggering the fractal's spin-zoom phase —
/// see the comment in `shade()`.
const FRACTAL_GRID_U: u32 = 200;
/// Vertical half of the same bucket grid as [`FRACTAL_GRID_U`].
const FRACTAL_GRID_V: u32 = 200;
/// Only ~1/period of the cap recomputes its phase on any given tick. Tuned
/// against `sky::budget_tests` at 322×86.
const FRACTAL_STAGGER_PERIOD: u32 = 50;

/// A compiled fractal-zoom layer: the authored [`FractalParams`] plus the
/// [`noise::Perlin`] source and [`crate::fractal`] tuning it samples through.
pub struct FractalLayer {
    p: FractalParams,
    perlin: noise::Perlin,
    fparams: crate::fractal::FractalParams,
}

impl FractalLayer {
    /// Wrap authored params into a layer ready for [`SkyLayer::shade`], fixing
    /// the underlying [`crate::fractal`] field shape (arm count, warp, twist).
    pub fn new(p: FractalParams) -> Self {
        FractalLayer {
            p,
            perlin: noise::Perlin::new(0x0F4A_C7A1),
            fparams: crate::fractal::FractalParams {
                arms: 5,
                octave_warp: 0.35,
                twist: 0.5,
            },
        }
    }
}

impl std::fmt::Debug for FractalLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FractalLayer").field("p", &self.p).finish()
    }
}

impl SkyLayer for FractalLayer {
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, seed: u32, hz: u32) -> SkyCell {
        let r_norm =
            ((super::HALF_PI - ctx.pitch) / (super::HALF_PI - super::V_LOW)).clamp(0.0, 1.0);
        let angle = util::wrap_tau(ctx.yaw);
        let u = 0.5 + r_norm * 0.5 * angle.cos();
        let v = 0.5 + r_norm * 0.5 * angle.sin();
        // Byte budget: the spin-zoom phase is continuous in time over the
        // WHOLE cap, so sampling it at the live tick would recolour nearly
        // every cell every tick. Stagger it per (u,v) bucket — only
        // ~1/period of the cap recomputes its phase on any given tick.
        let gx = util::frac_bucket(u, FRACTAL_GRID_U);
        let gy = util::frac_bucket(v, FRACTAL_GRID_V);
        let qt = util::staggered_qt(
            util::hash3(seed, gx, gy),
            ctx.tick,
            hz,
            FRACTAL_STAGGER_PERIOD,
        );
        let zoom_scroll = qt * self.p.speed.max(0.0);
        let phase = crate::fractal::spinzoom_phase(zoom_scroll, self.fparams.twist);
        let sample = crate::fractal::log_polar_field(u, v, phase, &self.fparams, &self.perlin);
        let h = sample.height;
        let deep = util::lerp_rgb(0x05040A, 0x201030, sample.ring_phase);
        let bright = util::lerp_rgb(0x402868, 0x8858C0, sample.angle_idx);
        let colour = util::lerp_rgb(deep, bright, h);
        let mut cell = below;
        cell.bg = colour;
        let glyph = if h > 0.75 {
            '#'
        } else if h > 0.55 {
            '%'
        } else if h > 0.35 {
            '+'
        } else if h > 0.15 {
            '·'
        } else {
            ' '
        };
        if glyph != ' ' {
            cell.glyph = glyph;
            cell.fg = util::scale_rgb(colour, 1.4);
        } else {
            cell.glyph = ' ';
            cell.fg = colour;
        }
        cell
    }

    fn is_time_animated(&self) -> bool {
        self.p.speed.abs() > 1e-4
    }
}

// ────────────────────────────── watcher ─────────────────────────────────────

/// A compiled watcher layer: the authored [`WatcherParams`] plus nothing else.
#[derive(Clone, Copy, Debug)]
pub struct WatcherLayer {
    p: WatcherParams,
}

impl WatcherLayer {
    /// Wrap authored params into a layer ready for [`SkyLayer::shade`].
    pub fn new(p: WatcherParams) -> Self {
        WatcherLayer { p }
    }
}

/// Iris radius as a dome-cap fraction (`r_norm` from the zenith); everything
/// outside this radius passes `below` through unchanged.
const IRIS_R: f32 = 0.35;

impl SkyLayer for WatcherLayer {
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, _seed: u32, _hz: u32) -> SkyCell {
        let r_norm =
            ((super::HALF_PI - ctx.pitch) / (super::HALF_PI - super::V_LOW)).clamp(0.0, 1.0);
        if r_norm > IRIS_R {
            return below;
        }
        let mut cell = below;
        let angle = util::wrap_tau(ctx.yaw);
        let lx = r_norm * angle.cos();
        let ly = r_norm * angle.sin();
        let pupil_r = 0.30 * IRIS_R;
        let pupil_cx = pupil_r * ctx.player_yaw.cos();
        let pupil_cy = pupil_r * ctx.player_yaw.sin();
        let dpx = lx - pupil_cx;
        let dpy = ly - pupil_cy;
        let dp = (dpx * dpx + dpy * dpy).sqrt();
        if dp < 0.09 {
            cell.glyph = '@';
            cell.bg = 0x000000;
            cell.fg = 0xF0F0F0;
            cell.style = cell.style.with_flags(CellStyle::BOLD);
        } else {
            let ring = 1.0 - (r_norm / IRIS_R);
            cell.bg = util::lerp_rgb(0x0A0A10, self.p.iris, ring);
            cell.fg = cell.bg;
            if (IRIS_R - r_norm).abs() < 0.03 {
                cell.glyph = '·';
                cell.style = cell
                    .style
                    .with_underline_style(UnderlineStyle::Dotted)
                    .with_underline_color(self.p.iris);
            }
        }
        cell
    }

    fn is_time_animated(&self) -> bool {
        // Driven by the player's facing yaw, which already forces a structural
        // recast on its own (a yaw-bucket change) — no independent time drift.
        false
    }
}

// ─────────────────────────────── mirror ─────────────────────────────────────

/// A compiled mirror layer: the authored [`MirrorParams`] plus nothing else.
#[derive(Clone, Copy, Debug)]
pub struct MirrorLayer {
    p: MirrorParams,
}

impl MirrorLayer {
    /// Wrap authored params into a layer ready for [`SkyLayer::shade`].
    pub fn new(p: MirrorParams) -> Self {
        MirrorLayer { p }
    }
}

impl SkyLayer for MirrorLayer {
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, seed: u32, _hz: u32) -> SkyCell {
        // A literal reflection of the level's own floor plan would need the
        // `Map`'s cell grid, which `SkyLayer::shade` does not carry (it is a
        // pure function of `(ctx, seed)` only, with no `&Map`). This
        // approximates the "seen from below, rotating the wrong way" read
        // with a deterministic silhouette field keyed on the MIRRORED yaw, so
        // it is still direction-anchored and stable under rotation, just not
        // a literal reflection of the authored geometry.
        let mut cell = below;
        let yaw_mirror = -ctx.yaw;
        let v = super::dome_height(ctx.pitch);
        let gx = ((util::wrap_tau(yaw_mirror) / TAU) * 64.0).floor() as u32;
        let gy = (v * 24.0).floor() as u32;
        let h = util::hash01(util::hash3(seed, gx, gy));
        let dim = self.p.dim.clamp(0.0, 1.0);
        if h < 0.35 {
            cell.bg = util::lerp_rgb(cell.bg, 0x181820, 1.0 - dim * 0.3);
            cell.glyph = '▓';
            cell.fg = util::scale_rgb(cell.bg, 0.7);
        } else {
            cell.bg = util::scale_rgb(cell.bg, 1.0 - dim);
            cell.fg = util::scale_rgb(cell.fg, 1.0 - dim);
        }
        cell
    }

    fn is_time_animated(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_at(yaw: f32, pitch: f32, player_yaw: f32) -> SkyCtx {
        SkyCtx {
            yaw,
            pitch,
            tick: 3,
            t: 0.25,
            hz: 12.0,
            cam_xy: (0.0, 0.0),
            player_yaw,
        }
    }

    #[test]
    fn void_is_dark() {
        let layer = VoidLayer::new(VoidParams::default());
        let below = SkyCell::flat(0x102040);
        let c = layer.shade(&ctx_at(0.3, 0.2, 0.0), below, 5, 12);
        let luma = |x: u32| ((x >> 16) & 0xFF) + ((x >> 8) & 0xFF) + (x & 0xFF);
        assert!(luma(c.bg) < luma(below.bg), "void must darken the base");
    }

    #[test]
    fn fractal_zenith_is_finite_and_in_range() {
        let layer = FractalLayer::new(FractalParams::default());
        let below = SkyCell::flat(0x000000);
        let c = layer.shade(
            &ctx_at(0.0, super::super::HALF_PI - 0.01, 0.0),
            below,
            0,
            12,
        );
        assert!(c.bg <= 0x00FF_FFFF);
    }

    #[test]
    fn watcher_pupil_is_visible_near_zenith() {
        let layer = WatcherLayer::new(WatcherParams::default());
        let below = SkyCell::flat(0x000000);
        // At player_yaw = 0 the pupil centre sits at local-disc radius
        // `0.30 * IRIS_R` along angle 0 (see `WatcherLayer::shade`). Sweep a
        // ring of exactly that r_norm — at angle 0 it lands dead-centre on
        // the pupil, well inside the `dp < 0.09` hit test. Converting r_norm
        // back to `pitch` uses the SAME formula `shade` uses in reverse:
        // `r_norm = (HALF_PI - pitch) / (HALF_PI - V_LOW)`.
        let pupil_r_norm = 0.30 * IRIS_R;
        let mut saw_pupil = false;
        for i in 0..40 {
            let a = (i as f32 / 40.0) * TAU;
            let pitch = super::super::HALF_PI
                - pupil_r_norm * (super::super::HALF_PI - super::super::V_LOW);
            let yaw = a;
            let c = layer.shade(&ctx_at(yaw, pitch, 0.0), below, 0, 12);
            if c.glyph == '@' {
                saw_pupil = true;
                break;
            }
        }
        assert!(
            saw_pupil,
            "some ray tracing the pupil's own radius must hit it"
        );
    }

    #[test]
    fn mirror_is_dimmer_than_below() {
        let layer = MirrorLayer::new(MirrorParams { dim: 0.5 });
        let below = SkyCell::flat(0x404040);
        let c = layer.shade(&ctx_at(0.6, 0.3, 0.0), below, 2, 12);
        let luma = |x: u32| ((x >> 16) & 0xFF) + ((x >> 8) & 0xFF) + (x & 0xFF);
        assert!(luma(c.bg) <= luma(below.bg));
    }

    #[test]
    fn void_and_mirror_glyphs_are_reliable() {
        for g in ['·', '▓'] {
            assert!(crate::glyph_family::is_reliable(g), "{g} must be reliable");
        }
    }
}
