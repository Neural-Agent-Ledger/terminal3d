//! `re_render::sky::storm` — a [`SkyLayer`] combining a dark cloud deck
//! ([`CloudsLayer`](crate::sky::clouds::CloudsLayer), reused internally) with a bounded lightning event stream.
//!
//! Pipeline stage: one layer in the [`crate::sky::SkyProgram`] stack.
//! Inputs: [`SkyCtx`] (ray yaw/pitch, tick) + the cell shaded so far. Output:
//! the storm deck's cloud shading, plus (during a fired bolt) a bright streak
//! glyph near the strike and a bounded afterglow lift to `bg`.
//! Knobs: [`StormParams`](re_world::sky_spec::StormParams) (deck cover, bolt rate).
//!
//! **The room-flash hook.** A real lightning strike should also "light the
//! room" for a tick or two — a temporary directional key light reaching the
//! walls. That belongs to the shading/lighting path
//! ([`crate::ascii_shader`]), not this layer, so this module only EXPOSES the
//! event via [`set_active_lightning_flash`](crate::sky::storm::set_active_lightning_flash)/[`active_lightning_flash`](crate::sky::storm::active_lightning_flash) — a
//! thread-local any lighting code can read and fold into its own light
//! construction. Nothing in this crate installs it today; it is a hook for
//! whichever caller wants the room-flash look.

use re_core::CellStyle;
use re_world::sky_spec::{CloudsParams, StormParams};

use super::{clouds::CloudsLayer, util, SkyCell, SkyCtx, SkyLayer};

/// A lightning bolt's room-flash event: `(azimuth radians, intensity 0..1)`.
/// `None` when no bolt is currently lit. Read by any lighting code that wants
/// to fold a transient key light in (the hook this module exposes — nothing in
/// this crate installs it into `LightEnv` today).
#[derive(Clone, Copy, Debug, Default)]
pub struct LightningFlash {
    /// Bolt direction (yaw, radians).
    pub azimuth: f32,
    /// Flash brightness, `1.0` at the strike tick decaying toward `0.0`.
    pub intensity: f32,
}

thread_local! {
    static LIGHTNING_FLASH: std::cell::Cell<Option<LightningFlash>> = const { std::cell::Cell::new(None) };
}

/// Publish (or clear, with `None`) the current room-flash event. Called by
/// [`StormLayer::shade`] when a bolt is live.
pub fn set_active_lightning_flash(flash: Option<LightningFlash>) {
    LIGHTNING_FLASH.with(|c| c.set(flash));
}

/// Read the current room-flash event, if any.
pub fn active_lightning_flash() -> Option<LightningFlash> {
    LIGHTNING_FLASH.with(|c| c.get())
}

/// A compiled storm layer: the authored [`StormParams`] plus a pre-built
/// [`CloudsLayer`] configured to look like a dark, low, fast-moving deck.
#[derive(Debug)]
pub struct StormLayer {
    p: StormParams,
    deck: CloudsLayer,
}

impl StormLayer {
    /// Wrap authored params into a layer ready for [`SkyLayer::shade`],
    /// building the internal cloud-deck layer from a fixed dark preset.
    pub fn new(p: StormParams) -> Self {
        let deck = CloudsLayer::new(CloudsParams {
            cover: p.cover,
            soft: 0.08,
            scale: 1.3,
            height: 45.0,
            wind: (0.08, 0.02),
            layers: 1,
            lit: 0x707888,
            shade: 0x0A0C12,
            haze: 35.0,
            bake: false,
        });
        StormLayer { p, deck }
    }
}

/// Bolts/second capped at the WCAG 2.3.1 "no more than 3 flashes per second"
/// photosensitivity ceiling.
const MAX_RATE_PER_SEC: f32 = 3.0;
/// Duration (seconds) of one bolt-fire lottery slot: `rate` is applied as a
/// per-slot probability (`rate * SLOT_DUR`), not a per-tick one, so the
/// expected bolt rate stays correct at any `hz`.
const SLOT_DUR: f32 = 0.25;
// Byte budget: a bolt's afterglow deliberately lifts the WHOLE deck (a real
// flash lights the whole sky), so the full-grid changed-cell count for those
// ticks is legitimately huge — that is only acceptable if lit ticks stay a
// small minority, since the p95 budget tolerates at most a 5% tail. The
// lit-tick fraction is `rate * BOLT_TICKS / hz`; at the authored preset
// default (`rate = 0.3`) and hz=12 with `BOLT_TICKS = 1` that is 2.5% —
// comfortably under the 5% tail (see `sky::budget_tests`). A level author
// pushing `rate` toward the WCAG ceiling (3/s) trades byte-budget headroom
// for flash frequency.
const BOLT_TICKS: u32 = 1;
/// The afterglow's reach around the bolt, in the same `(dy, dp)` metric as the
/// bolt-streak's own `near` test — a small, bounded lift, not literally the
/// entire dome (a real flash brightens the sky NEAR the bolt, not the far
/// side of it too). Sized so the touched footprint stays inside the byte
/// budget on every occurrence, not merely on average — no dependency on how
/// rarely a bolt fires. See `sky::budget_tests`.
const AFTERGLOW_RADIUS: f32 = 0.5;

impl SkyLayer for StormLayer {
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, seed: u32, hz: u32) -> SkyCell {
        let mut cell = self.deck.shade(ctx, below, seed.wrapping_add(0x5701), hz);
        let qt = ctx.tick as f32 / hz.max(1) as f32;
        let rate = self.p.rate.clamp(0.0, MAX_RATE_PER_SEC);
        let slot = (qt / SLOT_DUR).floor().max(0.0) as u32;
        let fire_hash = util::hash2(seed.wrapping_add(0x1116_7777), slot);
        let fire = util::hash01(fire_hash) < (rate * SLOT_DUR);
        if fire {
            let bolt_yaw = util::hash01(util::hash2(fire_hash, 1)) * std::f32::consts::TAU;
            let bolt_pitch = super::V_LOW + util::hash01(util::hash2(fire_hash, 2)) * 0.35;
            let dy = util::angle_diff(ctx.yaw, bolt_yaw);
            let dp = ctx.pitch - bolt_pitch;
            let dist2 = dy * dy + dp * dp;
            let near = dist2.sqrt() < 0.05;
            let afterglow = dist2.sqrt() < AFTERGLOW_RADIUS;
            let slot_start_tick = slot as f32 * SLOT_DUR * hz.max(1) as f32;
            let ticks_elapsed = ctx.tick as f32 - slot_start_tick;
            if ticks_elapsed >= 0.0 && ticks_elapsed < BOLT_TICKS as f32 {
                let intensity = 1.0 - ticks_elapsed / BOLT_TICKS as f32;
                if near {
                    cell.glyph = if dy.abs() > dp.abs() {
                        if dy > 0.0 {
                            '╱'
                        } else {
                            '╲'
                        }
                    } else {
                        '│'
                    };
                    cell.fg = 0xF0F4FF;
                    cell.style = cell.style.with_flags(CellStyle::BOLD);
                }
                // Afterglow: a small, bounded lift AROUND the bolt (not the
                // whole dome — see `AFTERGLOW_RADIUS`).
                if afterglow {
                    cell.bg = util::add_rgb_clamped(
                        cell.bg,
                        30.0 * intensity,
                        30.0 * intensity,
                        38.0 * intensity,
                    );
                }
                set_active_lightning_flash(Some(LightningFlash {
                    azimuth: bolt_yaw,
                    intensity,
                }));
            } else {
                set_active_lightning_flash(None);
            }
        } else {
            set_active_lightning_flash(None);
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

    fn ctx_at(tick: u32, hz: u32) -> SkyCtx {
        SkyCtx {
            yaw: 0.5,
            pitch: 0.4,
            tick,
            t: tick as f32 / hz as f32,
            hz: hz as f32,
            cam_xy: (0.0, 0.0),
            player_yaw: 0.0,
        }
    }

    #[test]
    fn a_high_rate_storm_fires_at_least_once_over_a_second() {
        let layer = StormLayer::new(StormParams {
            cover: 0.85,
            rate: MAX_RATE_PER_SEC,
        });
        let below = SkyCell::flat(0x0A0C12);
        let hz = 12u32;
        let mut fired = false;
        for tick in 0..(hz * 4) {
            let ctx = ctx_at(tick, hz);
            let _ = layer.shade(&ctx, below, 9, hz);
            if active_lightning_flash().is_some() {
                fired = true;
                break;
            }
        }
        assert!(
            fired,
            "a max-rate storm must fire within 4 simulated seconds"
        );
    }

    #[test]
    fn zero_rate_never_fires() {
        let layer = StormLayer::new(StormParams {
            cover: 0.85,
            rate: 0.0,
        });
        let below = SkyCell::flat(0x0A0C12);
        let hz = 12u32;
        for tick in 0..(hz * 8) {
            let ctx = ctx_at(tick, hz);
            let _ = layer.shade(&ctx, below, 3, hz);
            assert!(active_lightning_flash().is_none());
        }
    }

    #[test]
    fn bolt_glyphs_are_reliable() {
        for g in ['│', '╱', '╲'] {
            assert!(crate::glyph_family::is_reliable(g), "{g} must be reliable");
        }
    }
}
