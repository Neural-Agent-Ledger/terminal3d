//! `re_render::sky::stars` — two [`SkyLayer`]s: a procedural starfield
//! ([`StarsLayer`](crate::sky::stars::StarsLayer): twinkle, a Milky Way band, meteors) and
//! [`ConstellationLayer`](crate::sky::stars::ConstellationLayer), a small fixed set of stars wired to puzzle state so
//! completing a puzzle lights up a named pattern in the sky.
//!
//! Pipeline stage: two layers in the [`crate::sky::SkyProgram`] stack.
//! Inputs: [`SkyCtx`] (ray yaw/pitch, tick) + the cell shaded so far;
//! `ConstellationLayer` additionally reads [`crate::state_bus`] for each
//! linked star's on/off state.
//! Output: star/meteor glyphs blended onto structure (`glyph`+`fg`), the
//! Milky Way band lifting `bg`.
//! Knobs: [`StarsParams`](re_world::sky_spec::StarsParams) (density, twinkle, meteor rate, Milky Way tilt) and
//! [`ConstellationParams`](re_world::sky_spec::ConstellationParams) (the three star directions + their state-bus tag
//! names).

use std::f32::consts::TAU;

use re_core::{CellStyle, UnderlineStyle};
use re_world::sky_spec::{ConstellationParams, StarsParams};

use super::{util, SkyCell, SkyCtx, SkyLayer};

/// Star grid resolution — much finer than any terminal so the field reads as
/// scattered points rather than a coarse checkerboard, but fixed (not
/// per-viewport), so the SAME world direction always picks the SAME grid cell
/// regardless of screen size (stable under rotation).
const GRID_W: u32 = 240;
/// Vertical half of the same fixed star grid as [`GRID_W`].
const GRID_H: u32 = 40;

/// Angular match radius (in the same `(yaw, pitch)` units as [`util::angle_diff`])
/// a constellation/meteor sample point must fall within to "hit" an authored
/// feature.
const FEATURE_RADIUS: f32 = 0.02;

/// A compiled procedural-starfield layer: the authored [`StarsParams`] plus
/// nothing else.
#[derive(Clone, Copy, Debug)]
pub struct StarsLayer {
    p: StarsParams,
}

impl StarsLayer {
    /// Wrap authored params into a layer ready for [`SkyLayer::shade`].
    pub fn new(p: StarsParams) -> Self {
        StarsLayer { p }
    }
}

/// A great-circle Milky Way mask: the signed distance of the ray direction
/// from a plane through the origin tilted by `pole_tilt_deg` off the vertical,
/// folded into a soft band in `[0,1]` (1 = on the band's centre line).
fn milky_mask(yaw: f32, pitch: f32, pole_tilt_deg: f32) -> f32 {
    let tilt = pole_tilt_deg.to_radians();
    let (nx, nz) = (tilt.sin(), tilt.cos());
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let rx = sy * cp;
    let ry = cy * cp;
    let rz = sp;
    let d = rx * nx + rz * nz + ry * 0.0;
    const BAND_HALF_WIDTH: f32 = 0.18;
    (1.0 - (d.abs() / BAND_HALF_WIDTH)).clamp(0.0, 1.0)
}

fn temp_color(h: f32) -> u32 {
    match ((h * 5.0) as u32).min(4) {
        0 => 0x9DB4FF,
        1 => 0xF8F7FF,
        2 => 0xFFF4E8,
        3 => 0xFFD2A1,
        _ => 0xFFB07A,
    }
}

fn glyph_for_intensity(inten: f32) -> char {
    if inten > 0.85 {
        '*'
    } else if inten > 0.65 {
        '+'
    } else if inten > 0.45 {
        '•'
    } else if inten > 0.25 {
        '∙'
    } else if inten > 0.08 {
        '·'
    } else {
        '.'
    }
}

/// Time-sliced meteor streak, overlaid on `below` (structure-only — `bg` is
/// left untouched, keeping "mass in B" for whatever base/aurora is under it).
fn meteor_overlay(ctx: &SkyCtx, below: SkyCell, seed: u32, meteors: f32, qt: f32) -> SkyCell {
    if meteors <= 0.0 {
        return below;
    }
    const SLOT_DUR: f32 = 0.5;
    const STREAK_DUR: f32 = 0.4;
    let slot = (qt / SLOT_DUR).floor().max(0.0);
    let slot_u = slot as u32;
    let slot_t = qt - slot * SLOT_DUR;
    if slot_t >= STREAK_DUR {
        return below;
    }
    let fire_hash = util::hash2(seed.wrapping_add(0x5EED_1234), slot_u);
    if util::hash01(fire_hash) >= meteors {
        return below;
    }
    let y0 = util::hash01(util::hash2(fire_hash, 1)) * TAU;
    let p0 = super::V_LOW
        + util::hash01(util::hash2(fire_hash, 2)) * (super::HALF_PI - super::V_LOW) * 0.8;
    let dyaw = (util::hash01(util::hash2(fire_hash, 3)) - 0.5) * 0.7;
    let dpitch = (util::hash01(util::hash2(fire_hash, 4)) - 0.5) * 0.4;
    let u = (slot_t / STREAK_DUR).clamp(0.0, 1.0);

    let mut best_d = f32::MAX;
    let mut best_along = 0.0f32;
    const STEPS: i32 = 12;
    for i in 0..=STEPS {
        let s = u * (i as f32 / STEPS as f32);
        let py = y0 + dyaw * s;
        let pp = p0 + dpitch * s;
        let dy = util::angle_diff(ctx.yaw, py);
        let dp = ctx.pitch - pp;
        let d = (dy * dy + dp * dp).sqrt();
        if d < best_d {
            best_d = d;
            best_along = i as f32 / STEPS as f32;
        }
    }
    if best_d > 0.018 {
        return below;
    }
    let head_dist = 1.0 - (best_along / u.max(1e-4)).min(1.0);
    let (glyph, style) = if head_dist < 0.15 {
        ('*', CellStyle::default().with_flags(CellStyle::BOLD))
    } else if head_dist < 0.5 {
        ('-', CellStyle::default())
    } else {
        ('.', CellStyle::default().with_flags(CellStyle::DIM))
    };
    SkyCell {
        glyph,
        fg: 0xF0F4FF,
        bg: below.bg,
        style,
    }
}

impl SkyLayer for StarsLayer {
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, seed: u32, hz: u32) -> SkyCell {
        // Derive the QUANTIZED animation time from `tick`/`hz` directly, never
        // from `ctx.t`: this is what guarantees "same tick -> same cell"
        // regardless of what continuous `t` the caller happens to pass.
        let qt = ctx.tick as f32 / hz.max(1) as f32;
        let mut cell = below;
        let h_norm = super::dome_height(ctx.pitch);
        let milky = milky_mask(ctx.yaw, ctx.pitch, self.p.pole_tilt_deg) * self.p.milkyway;
        if milky > 0.0 {
            cell.bg = util::add_rgb_clamped(cell.bg, milky * 10.0, milky * 8.0, milky * 14.0);
        }
        let yaw = util::wrap_tau(ctx.yaw + self.p.wheel * qt);
        let gx = ((yaw / TAU) * GRID_W as f32).floor() as u32 % GRID_W;
        let gy = ((h_norm * GRID_H as f32).floor() as u32).min(GRID_H - 1);
        let star_hash = util::hash3(seed, gx, gy);
        let density = self.p.density * (1.0 + milky * 2.0);
        if util::hash01(star_hash) < density {
            let mag_h = util::hash01(util::hash2(star_hash, 11));
            let bright = 1.0 - mag_h.powf(3.0);
            let temp_h = util::hash01(util::hash2(star_hash, 13));
            let fg = temp_color(temp_h);
            let stagger = 1 + (star_hash % 5);
            let eff_tick = util::staggered_tick(star_hash, ctx.tick, stagger);
            let phase = util::hash01(util::hash2(star_hash, 17)) * TAU;
            let f_s = (0.15 + 0.55 * (1.0 - h_norm)) * (0.4 + self.p.twinkle);
            let eff_t = eff_tick as f32 / hz.max(1) as f32;
            let osc = (eff_t * f_s * TAU + phase).sin();
            let inten = ((0.5 + 0.5 * osc) * bright).clamp(0.0, 1.0);
            cell.glyph = glyph_for_intensity(inten);
            cell.fg = fg;
            if inten > 0.72 {
                cell.style = cell.style.with_flags(CellStyle::BOLD);
            } else if inten < 0.22 {
                cell.style = cell.style.with_flags(CellStyle::DIM);
            }
        }
        meteor_overlay(ctx, cell, seed, self.p.meteors, qt)
    }

    fn is_time_animated(&self) -> bool {
        self.p.twinkle > 0.0 || self.p.meteors > 0.0 || self.p.wheel.abs() > 1e-6
    }
}

/// Hash a `state_bus` tag NAME into the `u16` id [`crate::state_bus`]
/// indexes by. The hook contract: whatever writer publishes a constellation
/// star's puzzle state must call this SAME function on the same tag name so
/// the id lines up.
pub fn tag_id_for_name(name: &str) -> u16 {
    let seed = name.bytes().fold(0x811C_9DC5u32, |acc, b| {
        (acc ^ b as u32).wrapping_mul(0x0100_0193)
    });
    (util::hash_u32(seed) & 0xFFFF) as u16
}

/// A compiled constellation layer: three star directions, each linked to a
/// [`crate::state_bus`] tag name resolved once via [`tag_id_for_name`] at
/// shade time.
#[derive(Clone, Debug)]
pub struct ConstellationLayer {
    p: ConstellationParams,
}

impl ConstellationLayer {
    /// Wrap authored params into a layer ready for [`SkyLayer::shade`].
    pub fn new(p: ConstellationParams) -> Self {
        ConstellationLayer { p }
    }

    /// World direction `(yaw, pitch)` of star `i`, from its authored
    /// normalized `(u, v)` position.
    fn star_dir(&self, i: usize) -> (f32, f32) {
        let (u, v) = self.p.stars[i];
        (u * TAU, super::V_LOW + v * (super::HALF_PI - super::V_LOW))
    }
}

impl SkyLayer for ConstellationLayer {
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, _seed: u32, _hz: u32) -> SkyCell {
        let mut cell = below;
        let states: [bool; 3] = [
            crate::state_bus::state_of(tag_id_for_name(&self.p.link[0])) != 0,
            crate::state_bus::state_of(tag_id_for_name(&self.p.link[1])) != 0,
            crate::state_bus::state_of(tag_id_for_name(&self.p.link[2])) != 0,
        ];

        let mut hit_idx = None;
        for i in 0..3 {
            let (sy, sp) = self.star_dir(i);
            let dy = util::angle_diff(ctx.yaw, sy);
            let dp = ctx.pitch - sp;
            if (dy * dy + dp * dp).sqrt() < FEATURE_RADIUS {
                hit_idx = Some(i);
                break;
            }
        }
        if let Some(i) = hit_idx {
            if states[i] {
                cell.glyph = '*';
                cell.fg = 0xFFFFFF;
                cell.style = cell.style.with_flags(CellStyle::BOLD);
                cell.bg = util::add_rgb_clamped(cell.bg, 6.0, 6.0, 10.0);
            } else {
                cell.glyph = '·';
                cell.fg = 0x8898B8;
                cell.style = cell.style.with_flags(CellStyle::DIM);
            }
            return cell;
        }

        // Link lines: only once every authored star is powered ("looks back").
        if states.iter().all(|&s| s) {
            for &(a, b) in &[(0usize, 1usize), (1, 2)] {
                let (ya, pa) = self.star_dir(a);
                let (yb, pb) = self.star_dir(b);
                let dyseg = util::angle_diff(yb, ya);
                let dpseg = pb - pa;
                let denom = dyseg * dyseg + dpseg * dpseg;
                if denom < 1e-8 {
                    continue;
                }
                let dyp = util::angle_diff(ctx.yaw, ya);
                let dpp = ctx.pitch - pa;
                let t = ((dyp * dyseg + dpp * dpseg) / denom).clamp(0.0, 1.0);
                let py = ya + dyseg * t;
                let pp = pa + dpseg * t;
                let ddy = util::angle_diff(ctx.yaw, py);
                let ddp = ctx.pitch - pp;
                if (ddy * ddy + ddp * ddp).sqrt() < FEATURE_RADIUS * 0.6 {
                    cell.glyph = '·';
                    cell.fg = 0x60A0D0;
                    cell.style = cell
                        .style
                        .with_underline_style(UnderlineStyle::Dotted)
                        .with_underline_color(0x60A0D0);
                    break;
                }
            }
        }
        cell
    }

    fn is_time_animated(&self) -> bool {
        // Driven by `state_bus` (a puzzle interaction), not the wall clock —
        // any camera move/recast already re-reads the current state.
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_at(yaw: f32, pitch: f32) -> SkyCtx {
        SkyCtx {
            yaw,
            pitch,
            tick: 0,
            t: 0.0,
            hz: 12.0,
            cam_xy: (0.0, 0.0),
            player_yaw: 0.0,
        }
    }

    #[test]
    fn stars_only_use_reliable_glyphs() {
        for g in ['*', '+', '•', '∙', '·', '.'] {
            assert!(crate::glyph_family::is_reliable(g), "{g} must be reliable");
        }
    }

    #[test]
    fn dense_sweep_produces_some_stars() {
        let layer = StarsLayer::new(StarsParams {
            density: 0.5,
            ..StarsParams::default()
        });
        let below = SkyCell::flat(0x04060E);
        let mut any_star = false;
        for i in 0..2000 {
            let yaw = (i as f32) * 0.017;
            let pitch = 0.2;
            let ctx = ctx_at(yaw, pitch);
            let c = layer.shade(&ctx, below, 42, 12);
            if c.glyph != ' ' {
                any_star = true;
                break;
            }
        }
        assert!(any_star, "a 0.5 density sweep must hit at least one star");
    }

    #[test]
    fn tag_id_for_name_is_deterministic_and_distinct() {
        assert_eq!(tag_id_for_name("star_a"), tag_id_for_name("star_a"));
        assert_ne!(tag_id_for_name("star_a"), tag_id_for_name("star_b"));
    }

    #[test]
    fn constellation_unpowered_star_is_dim() {
        let p = ConstellationParams::default();
        let (sy, sp) = (
            p.stars[0].0 * TAU,
            super::super::V_LOW + p.stars[0].1 * (super::super::HALF_PI - super::super::V_LOW),
        );
        let layer = ConstellationLayer::new(p);
        crate::state_bus::clear_active();
        let below = SkyCell::flat(0x04060E);
        let ctx = ctx_at(sy, sp);
        let c = layer.shade(&ctx, below, 0, 12);
        assert_eq!(c.glyph, '·');
        assert!(c.style.is_dim());
    }

    #[test]
    fn constellation_powered_star_is_bright() {
        let p = ConstellationParams::default();
        let (sy, sp) = (
            p.stars[0].0 * TAU,
            super::super::V_LOW + p.stars[0].1 * (super::super::HALF_PI - super::super::V_LOW),
        );
        let layer = ConstellationLayer::new(p.clone());
        let mut m = re_core::CellStateMap::new();
        m.set_state(tag_id_for_name(&p.link[0]), 1);
        let _scope = crate::state_bus::CellStateScope::new(&m);
        let below = SkyCell::flat(0x04060E);
        let ctx = ctx_at(sy, sp);
        let c = layer.shade(&ctx, below, 0, 12);
        assert_eq!(c.glyph, '*');
        assert!(c.style.is_bold());
    }
}
