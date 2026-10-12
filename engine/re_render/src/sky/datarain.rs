//! `re_render::sky::datarain` — a [`SkyLayer`] of falling glyph columns: the
//! matrix effect projected onto the sky rather than the screen.
//!
//! Pipeline stage: one layer in the [`crate::sky::SkyProgram`] stack.
//! Inputs: [`SkyCtx`] (ray yaw/pitch, tick) + the cell shaded so far. Output:
//! a fixed-width falling-head glyph column blended into `bg`/`fg`/`glyph`
//! where the column's trail currently covers this ray's height.
//! Knobs: [`DataRainParams`](re_world::sky_spec::DataRainParams) (fall speed range, trail length, glyph source).
//!
//! Columns are fixed in YAW (world/direction-anchored), so they turn WITH the
//! world — the opposite of a screen-fixed rain overlay, which would look
//! pinned to the viewport instead of belonging to the sky.

use std::f32::consts::TAU;

use re_core::CellStyle;
use re_world::sky_spec::{DataRainParams, DataRainSource};

use super::{util, SkyCell, SkyCtx, SkyLayer};

/// Number of falling-glyph columns wrapped around the full yaw turn.
const COLS: u32 = 180;
/// Only ~1/period of the [`COLS`] columns advance their fall state on any
/// given tick — see the comment in `shade()`. Tuned against
/// `sky::budget_tests` at 322×86.
const DATARAIN_STAGGER_PERIOD: u32 = 64;

/// A compiled data-rain layer: the authored [`DataRainParams`] plus nothing
/// else.
#[derive(Clone, Copy, Debug)]
pub struct DataRainLayer {
    p: DataRainParams,
}

impl DataRainLayer {
    /// Wrap authored params into a layer ready for [`SkyLayer::shade`].
    pub fn new(p: DataRainParams) -> Self {
        DataRainLayer { p }
    }
}

/// Pick the glyph for one falling cell, hashed from its column/row/tick.
fn glyph_for_source(source: DataRainSource, h: u32) -> char {
    match source {
        DataRainSource::Binary => {
            if h & 1 == 0 {
                '0'
            } else {
                '1'
            }
        }
        DataRainSource::Hex => {
            const HEX: [char; 16] = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
            ];
            HEX[(h % 16) as usize]
        }
        // `pool`/`shards` name the level's node-pool words / collected name
        // shards; wiring the actual text registry would couple this layer to
        // `re_render::scene_text`/game story data, so both fall back to a
        // wider deterministic letter set — visually distinct from
        // binary/hex, still reliable glyphs.
        DataRainSource::Pool | DataRainSource::Shards => {
            const LETTERS: [char; 20] = [
                'A', 'C', 'E', 'H', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W',
                'X', 'Y', 'Z', '#',
            ];
            LETTERS[(h % LETTERS.len() as u32) as usize]
        }
    }
}

impl SkyLayer for DataRainLayer {
    fn shade(&self, ctx: &SkyCtx, below: SkyCell, seed: u32, hz: u32) -> SkyCell {
        let mut cell = below;
        let v = super::dome_height(ctx.pitch);
        let vv = 1.0 - v; // falls from the zenith (1) toward the horizon (0).
        let c = ((util::wrap_tau(ctx.yaw) / TAU) * COLS as f32).floor() as u32 % COLS;
        // Byte budget: a column's fall position is continuous in time, so
        // sampling it at the live tick would recolour nearly the whole trail
        // band every tick. Stagger PER COLUMN instead — only ~1/period of
        // the columns advance their fall state on any given tick. Deriving
        // `glyph_tick` from the SAME `eff_tick` below (not `ctx.tick`) means
        // a frozen column is fully byte-identical between its own updates,
        // not just position-frozen.
        let col_hash = util::hash2(seed, c);
        let eff_tick = util::staggered_tick(col_hash, ctx.tick, DATARAIN_STAGGER_PERIOD);
        let qt = eff_tick as f32 / hz.max(1) as f32;
        let speed =
            self.p.speed0 + (self.p.speed1 - self.p.speed0) * util::hash01(util::hash2(seed, c));
        let phase = util::hash01(util::hash2(seed.wrapping_add(1), c));
        let head = (qt * speed + phase).fract();
        let trail = self.p.trail.max(0.01);
        let mut k = head - vv;
        if k < 0.0 {
            k += 1.0; // the column wraps: a head can loop past the top.
        }
        if k > trail {
            return cell;
        }
        let glyph_stagger = (hz.max(3) / 3).max(1);
        let glyph_tick = eff_tick / glyph_stagger;
        let glyph_hash = util::hash3(seed, c, ((vv * 40.0) as u32).wrapping_add(glyph_tick));
        cell.glyph = glyph_for_source(self.p.source, glyph_hash);
        let head_amt = 1.0 - (k / trail).clamp(0.0, 1.0);
        if head_amt > 0.85 {
            cell.fg = 0xE8FFE8;
            cell.style = cell.style.with_flags(CellStyle::BOLD);
        } else if head_amt > 0.4 {
            cell.fg = 0x40C060;
        } else {
            cell.fg = 0x186030;
            cell.style = cell.style.with_flags(CellStyle::DIM);
        }
        cell.bg = util::add_rgb_clamped(cell.bg, 0.0, head_amt * 6.0, 0.0);
        cell
    }

    fn is_time_animated(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_at(yaw: f32, pitch: f32, tick: u32, hz: u32) -> SkyCtx {
        SkyCtx {
            yaw,
            pitch,
            tick,
            t: tick as f32 / hz as f32,
            hz: hz as f32,
            cam_xy: (0.0, 0.0),
            player_yaw: 0.0,
        }
    }

    #[test]
    fn some_column_shows_a_falling_head_over_time() {
        let layer = DataRainLayer::new(DataRainParams::default());
        let below = SkyCell::flat(0x040608);
        let mut saw_head = false;
        'outer: for tick in 0..60 {
            for i in 0..COLS {
                let yaw = (i as f32 / COLS as f32) * TAU;
                let ctx = ctx_at(yaw, 0.3, tick, 12);
                let c = layer.shade(&ctx, below, 7, 12);
                if c.style.is_bold() {
                    saw_head = true;
                    break 'outer;
                }
            }
        }
        assert!(saw_head, "some column must show a bright falling head");
    }

    #[test]
    fn binary_source_only_emits_0_or_1() {
        let layer = DataRainLayer::new(DataRainParams {
            source: DataRainSource::Binary,
            trail: 2.0, // huge trail so nearly every sample is inside a column
            ..DataRainParams::default()
        });
        let below = SkyCell::flat(0x040608);
        for i in 0..40 {
            let yaw = (i as f32 / 40.0) * TAU;
            let ctx = ctx_at(yaw, 0.3, 5, 12);
            let c = layer.shade(&ctx, below, 2, 12);
            assert!(
                c.glyph == ' ' || c.glyph == '0' || c.glyph == '1',
                "{}",
                c.glyph
            );
        }
    }
}
