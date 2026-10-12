//! `re_render::sky::skyline` — a [`SkyLayer`] distant-city horizon: a ring of
//! towers, one candidate per authored azimuth block, each with a seeded
//! height / hue / window rhythm.
//!
//! Pipeline stage: one layer in the [`crate::sky::SkyProgram`] stack.
//! Inputs: [`SkyCtx`] (ray yaw/pitch, tick) + the cell shaded so far. Output:
//! where a ray hits a tower, both `bg` (silhouette) and `glyph`+`fg` (window
//! unit) are fully replaced — a building is mass, occluding the sky below it;
//! a gap between towers passes `below` through unchanged.
//! Knobs: [`SkylineParams`](re_world::sky_spec::SkylineParams) (block count, density, height range, hue/style,
//! landmark rate + beacon blink).
//!
//! Reuses [`crate::facade`] for the window rhythm unchanged — a tower's
//! facade samples the SAME `(u, v, seed) -> FacadeTexel` function a nearby
//! `Facade` material wall would, so the skyline and a level's own facades
//! read as one look.
//!
//! ## Anchoring
//!
//! Sampled by ray DIRECTION only (`SkyCtx::yaw`/`pitch`), never
//! [`SkyCtx::cam_xy`] — PARALLAX-FREE, like a real skyline seen from far enough
//! that walking a few metres doesn't shift it. "Distant" and "horizon-anchored"
//! are the same property here: each block's tower height is a DOME-HEIGHT
//! fraction ([`crate::sky::dome_height`]), not a world distance, so the silhouette
//! sits at a fixed place against the horizon regardless of viewport/FOV.
//!
//! ## Composition
//!
//! Drawn like `clouds`: where a ray hits a tower, BOTH channels are replaced —
//! `bg` becomes the tower's silhouette shade (occluding whatever layer sits
//! below, e.g. `stars`), `glyph`+`fg` become its window unit — because a
//! building is mass, not a faint overlay. Where a ray does not hit a tower (a
//! gap between blocks, or above a roofline), `below` passes through UNCHANGED,
//! so `stars`/`aurora`/`clouds`/`datarain` and the level's fog all still show
//! through the gaps exactly as authored.

use re_core::CellStyle;
use re_world::sky_spec::SkylineParams;

use super::{dome_height, util, SkyCell, SkyCtx, SkyLayer};
use std::f32::consts::TAU;

/// How many "world cells" tall a tower's window grid is treated as, for
/// [`crate::facade::sample`]'s bay/floor-per-world-cell convention. A tower's
/// FULL height (base to roofline) maps to this many world-cell-equivalents, so
/// a `floor_h` of 4 gives roughly `4 * TOWER_WORLD_HEIGHT` window rows top to
/// bottom — enough for the rhythm to read at a tower's typical screen height.
const TOWER_WORLD_HEIGHT: f32 = 3.0;
/// How many "world cells" wide one azimuth BLOCK is treated as (so `bay=3`
/// gives 3 window columns across the block).
const TOWER_WORLD_WIDTH: f32 = 1.0;

/// Extra dome-height fraction a LANDMARK's spire adds above its roofline.
const SPIRE_HEIGHT_FRAC: f32 = 0.08;
/// Fraction of the spire band, from its base, that is the blinking ANTENNA TIP
/// (the rest is a plain dim mast glyph).
const SPIRE_TIP_FRAC: f32 = 0.7;
/// Antenna blink cycle, seconds.
const BLINK_PERIOD_SECS: f32 = 1.6;
/// Fraction of the blink cycle the light is ON.
const BLINK_ON_FRACTION: f32 = 0.25;
/// Landmarks stagger their blink phase over this many ticks (so a skyline of
/// several landmarks does not flash in unison — the byte-budget "hashed subset"
/// shape every other layer's animation follows).
const BLINK_STAGGER_TICKS: u32 = 7;

/// Two silhouette base tints a tower's hue hashes between — a cool steel-blue
/// and a warm brick tone, giving a mixed-hue skyline without needing a full
/// palette system for a background layer.
const SILHOUETTE_COOL: u32 = 0x121A24;
const SILHOUETTE_WARM: u32 = 0x241A12;
/// Lit-window colour, warm against either silhouette base — an "office
/// window at night" read.
const LIT_WINDOW: u32 = 0xFFE0A0;
/// Dim antenna mast colour (off state).
const MAST_DIM: u32 = 0x707880;
/// Bright antenna warning-light colour (on state).
const BEACON_ON: u32 = 0xFF3030;

/// A compiled skyline layer: the authored [`SkylineParams`] plus nothing else
/// — every tower is re-derived per ray from its azimuth block hash.
#[derive(Clone, Copy, Debug)]
pub struct SkylineLayer {
    p: SkylineParams,
}

impl SkylineLayer {
    /// Wrap authored params into a layer ready for [`SkyLayer::shade`].
    pub fn new(p: SkylineParams) -> Self {
        SkylineLayer { p }
    }
}

/// Per-block deal: does this azimuth block carry a tower, and if so its height /
/// hue / landmark-ness. Computed once per `shade` call from ONE hash so every
/// derived value is independent (distinct salts) yet stable for the block.
struct TowerDeal {
    present: bool,
    height_frac: f32,
    hue_t: f32,
    is_landmark: bool,
    seed: u32,
}

fn deal_for_block(seed: u32, gx: u32, p: &SkylineParams) -> TowerDeal {
    let h = util::hash2(seed.wrapping_add(0x5CA1_ABE1), gx);
    let present = util::hash01(h) < p.density.clamp(0.0, 1.0);
    let height_frac =
        p.height_lo + (p.height_hi - p.height_lo).max(0.0) * util::hash01(util::hash2(h, 1));
    let hue_t = util::hash01(util::hash2(h, 2));
    let is_landmark = util::hash01(util::hash2(h, 3)) < p.landmark_rate.clamp(0.0, 1.0);
    TowerDeal {
        present,
        height_frac: height_frac.clamp(0.0, 1.0),
        hue_t,
        is_landmark,
        seed: h,
    }
}

impl SkyLayer for SkylineLayer {
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, seed: u32, hz: u32) -> SkyCell {
        // Below the horizon band the dome has no towers to show (the ground/floor
        // owns that view); a skyline lives in the upward-looking dome like every
        // other sky layer.
        if ctx.pitch <= super::V_LOW {
            return below;
        }
        let blocks = self.p.blocks.max(1);
        let yaw = util::wrap_tau(ctx.yaw);
        let block_f = (yaw / TAU) * blocks as f32;
        let gx = (block_f.floor() as u32) % blocks;
        let fu = block_f - block_f.floor();

        let deal = deal_for_block(seed, gx, &self.p);
        if !deal.present {
            return below; // a gap between towers: whatever is behind shows.
        }

        let dh = dome_height(ctx.pitch);
        if dh > deal.height_frac {
            // Above the roofline: a plain tower shows nothing here; a LANDMARK's
            // spire occupies a thin band just above it.
            if deal.is_landmark && dh <= deal.height_frac + SPIRE_HEIGHT_FRAC {
                return spire_cell(ctx, hz, &deal);
            }
            return below;
        }

        // Inside the tower body: sample the SAME facade window-grid function a
        // `Facade` material wall uses, mapping this block's fractional azimuth
        // position and normalized height onto the recipe's world-cell-per-unit
        // convention. `world_per_cell = 0.0` (no LOD collapse): the sky's own
        // block/hz quantization already bounds detail, and there is no camera
        // distance to derive a footprint from for a parallax-free layer.
        let v_world = (dh / deal.height_frac.max(1e-4)) * TOWER_WORLD_HEIGHT;
        let u_world = fu * TOWER_WORLD_WIDTH;
        let params = crate::facade::FacadeParams {
            style: crate::facade_style_from_world(self.p.style),
            bay: 3.0,
            floor_h: 4.0,
            lit: self.p.lit,
            seed: deal.seed as i32,
            ground_band: false,
            sign_row: false,
        };
        let texel = crate::facade::sample(u_world, v_world, &params, 0.0);

        let silhouette = util::lerp_rgb(SILHOUETTE_COOL, SILHOUETTE_WARM, deal.hue_t);
        let fg = util::lerp_rgb(silhouette, LIT_WINDOW, texel.ink.clamp(0.0, 1.0));
        SkyCell {
            glyph: texel.glyph,
            fg,
            // A building is MASS: it occludes whatever sits under it (stars,
            // aurora, clouds), so `bg` is fully replaced, not blended.
            bg: silhouette,
            style: CellStyle::default(),
        }
    }

    fn is_time_animated(&self) -> bool {
        // Fully static (seeded, no time term) unless a landmark's beacon can
        // blink — matches `Substrate`'s "no time term ⇒ cacheable" reasoning on
        // the render side.
        self.p.landmark_rate > 0.0
    }
}

/// The spire band above a landmark's roofline: a dim mast glyph, with the tip
/// fraction blinking a bright warning light on a slow, per-landmark-staggered
/// cycle (so several landmarks do not flash in unison).
fn spire_cell(ctx: &SkyCtx, hz: u32, deal: &TowerDeal) -> SkyCell {
    let dh = dome_height(ctx.pitch);
    let band_t = ((dh - deal.height_frac) / SPIRE_HEIGHT_FRAC.max(1e-4)).clamp(0.0, 1.0);
    if band_t < SPIRE_TIP_FRAC {
        return SkyCell {
            glyph: '│',
            fg: MAST_DIM,
            bg: 0x000000,
            style: CellStyle::default(),
        };
    }
    let stagger = 1 + (deal.seed % BLINK_STAGGER_TICKS);
    let eff_tick = util::staggered_tick(deal.seed, ctx.tick, stagger);
    let t = eff_tick as f32 / hz.max(1) as f32;
    let phase = (t / BLINK_PERIOD_SECS).rem_euclid(1.0);
    let on = phase < BLINK_ON_FRACTION;
    SkyCell {
        glyph: if on { '●' } else { '·' },
        fg: if on { BEACON_ON } else { MAST_DIM },
        bg: 0x000000,
        style: if on {
            CellStyle::default().with_flags(CellStyle::BOLD)
        } else {
            CellStyle::default()
        },
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
    fn full_density_sweep_hits_a_tower_somewhere() {
        let layer = SkylineLayer::new(SkylineParams {
            density: 1.0,
            ..SkylineParams::default()
        });
        let below = SkyCell::flat(0x04060E);
        let mut hit = false;
        for i in 0..64 {
            let yaw = i as f32 * 0.1;
            let ctx = ctx_at(yaw, 0.05, 0);
            let c = layer.shade(&ctx, below, 11, 12);
            if c.bg != below.bg {
                hit = true;
                break;
            }
        }
        assert!(
            hit,
            "density=1.0 must hit a tower somewhere in a 64-column sweep"
        );
    }

    #[test]
    fn zero_density_never_occludes() {
        let layer = SkylineLayer::new(SkylineParams {
            density: 0.0,
            ..SkylineParams::default()
        });
        let below = SkyCell::flat(0x04060E);
        for i in 0..32 {
            let ctx = ctx_at(i as f32 * 0.2, 0.05, 0);
            assert_eq!(layer.shade(&ctx, below, 11, 12), below);
        }
    }

    #[test]
    fn below_the_dome_floor_is_untouched() {
        let layer = SkylineLayer::new(SkylineParams {
            density: 1.0,
            ..SkylineParams::default()
        });
        let below = SkyCell::flat(0x102030);
        let ctx = ctx_at(0.0, super::super::V_LOW - 0.05, 0);
        assert_eq!(layer.shade(&ctx, below, 1, 12), below);
    }

    #[test]
    fn high_enough_pitch_clears_every_roofline() {
        let layer = SkylineLayer::new(SkylineParams {
            density: 1.0,
            height_hi: 0.2,
            landmark_rate: 0.0,
            ..SkylineParams::default()
        });
        let below = SkyCell::flat(0x04060E);
        // Above the tallest possible roofline and no landmark spires: every
        // block must pass `below` through unchanged.
        let ctx = ctx_at(0.5, 1.3, 0);
        assert_eq!(layer.shade(&ctx, below, 7, 12), below);
    }

    #[test]
    fn static_without_landmarks_animated_with() {
        let still = SkylineLayer::new(SkylineParams {
            landmark_rate: 0.0,
            ..SkylineParams::default()
        });
        let blinking = SkylineLayer::new(SkylineParams {
            landmark_rate: 0.3,
            ..SkylineParams::default()
        });
        assert!(!still.is_time_animated());
        assert!(blinking.is_time_animated());
    }

    #[test]
    fn deterministic_same_ray_same_cell() {
        let layer = SkylineLayer::new(SkylineParams::default());
        let below = SkyCell::flat(0x04060E);
        let ctx = ctx_at(1.234, 0.3, 42);
        let a = layer.shade(&ctx, below, 9, 12);
        let b = layer.shade(&ctx, below, 9, 12);
        assert_eq!(a, b);
    }

    #[test]
    fn landmark_beacon_only_uses_reliable_glyphs() {
        for g in ['│', '●', '·'] {
            assert!(crate::glyph_family::is_reliable(g), "{g} must be reliable");
        }
    }
}
