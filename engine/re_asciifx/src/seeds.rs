//! SEEDS — "a neural net escapes and scatters seeds" (the canon prompt this
//! pass was given): several independent escape points along the rect's
//! bottom edge each launch a cluster of seeds, every seed arcing outward on
//! its own randomized heading/distance/flight-time, landing and germinating
//! through a five-stage sprout ramp (grass tufts and, for about half the
//! seeds, a final flowering cluster) at its own growth rate. Every seed's
//! flight and growth are governed by independent per-seed frequencies, so a
//! burst reads as scattered rather than a single synchronized "poof" — and
//! every seed then LOOPS its own cycle indefinitely (flight → germinate →
//! grow → hold in full bloom → relaunch), each starting at its own random
//! phase within that loop, so the garden is already lush at `t=0` and never
//! settles into a single "done" frame (see the 2026-09-25 owner ask: "too
//! sparse — want a lush scatter … fill the rect", and
//! `menu_fx::backdrop`'s doc on why a one-shot effect used to be excluded
//! from the permanent menu backdrop pool).
//!
//! Flight is a closed-form arc, not a simulated trajectory: `paint`/
//! `stage_at` are pure functions of `(seed, t)`, so seeking to any `t`
//! directly reproduces the same frame — no stepping, no physics solver.

use crate::hash::{angle, chance, hash_unit, range};
use crate::{secs, set_clipped, AsciiEffect, Rect};
use re_core::oklab::{oklch_to_packed, packed_to_oklch};
use re_core::{Cell, Framebuffer};
use std::f32::consts::PI;
use std::time::Duration;

// ─────────────────────────────── tuning ──────────────────────────────────

/// Several escape points along the bottom edge, not one shared launch
/// column — this is what lets the scatter cover the FULL width of a wide
/// rect (a menu backdrop, a splash) rather than reading as one cluster with
/// empty margins either side.
const SOURCE_COUNT_RANGE: (usize, usize) = (3, 5);
/// Seeds per source — multiplied by [`SOURCE_COUNT_RANGE`] this is the
/// effective total ("many seeds" the owner asked for; a 280-wide rect at the
/// high end carries ~500).
const SEEDS_PER_SOURCE_RANGE: (usize, usize) = (18, 34);
const FLIGHT_TIME_RANGE: (f32, f32) = (0.6, 1.8);
/// Launch distance as a fraction of ONE SOURCE's own half-spread (itself
/// derived from the rect width and the number of sources — see
/// [`Seeds::random`]), not the whole rect — each source's own burst stays a
/// recognizable local cluster even as many sources together fill the rect.
const LAUNCH_DIST_FRAC_RANGE: (f32, f32) = (0.35, 1.05);
const ARC_HEIGHT_RANGE: (f32, f32) = (2.0, 7.0); // world (row) units of bulge
const GERMINATE_DELAY_RANGE: (f32, f32) = (0.1, 1.2);
const GROWTH_PERIOD_RANGE: (f32, f32) = (1.5, 4.5);
/// How long a fully-grown plant holds its bloom before the cycle relaunches
/// it — long enough to actually be admired, short enough that the garden
/// keeps visibly turning over.
const BLOOM_HOLD_RANGE: (f32, f32) = (2.5, 6.0);
/// Fraction of seeds whose growth culminates in a flowering cluster (stage 4,
/// plus the two side `*` blooms) rather than capping at a plain leafy tip
/// (stage 3) — a garden of all-identical flowers reads as wallpaper, a mix
/// reads as planted.
const BLOOM_CHANCE: f32 = 0.42;
/// Per-seed hue jitter (radians, Oklch) applied to both the seed-in-flight
/// and sprout colour ramps, so the garden is not perfectly monochrome.
const HUE_JITTER_RANGE: f32 = 0.35;

const S_ANGLE: u64 = 1;
const S_DIST: u64 = 2;
const S_FLIGHT: u64 = 3;
const S_ARC: u64 = 4;
const S_DELAY: u64 = 5;
const S_GROWTH: u64 = 6;
const S_BLOOM_HOLD: u64 = 7;
const S_PHASE0: u64 = 8;
const S_BLOOMS: u64 = 9;
const S_HUE: u64 = 10;
const S_SOURCE_PICK: u64 = 11;
const SRC_S_COL_JITTER: u64 = 101;
const SRC_S_ROW: u64 = 102;

/// Sprout growth-TIP stages, index 0 (just landed) through 4 (fully grown,
/// flowering). Every glyph is ASCII or the vetted extras list. `\u{2218}`
/// (∘ ring operator) — technically inside `re_render::glyph_family::
/// is_reliable`'s vetted list, which is scoped to broad terminal-font
/// compatibility — renders as tofu in the Consolas/Cascadia preview font this
/// project's own gallery/screenshot tooling uses. Swapped for `\u{2022}`
/// (• bullet), also reliable and present in both.
const SPROUT_GLYPHS: [char; 5] = ['.', ':', 'i', '\u{a6}', '*']; // . : i ¦ *  (stage 4 = flower)
const SEED_FLIGHT_GLYPH: char = '\u{2022}'; // • bullet — a small round seed in flight
const STEM_GLYPH: char = '\u{2502}'; // │
const LEAF_LEFT: char = '/';
const LEAF_RIGHT: char = '\\';

/// How many trailing echoes of a seed's own flight arc to draw behind its
/// current position — turns "a single moving dot" into a visible arced trail.
const FLIGHT_TRAIL_STEPS: i32 = 6;
const FLIGHT_TRAIL_DT: f32 = 0.045;

const SEED_COLOR: u32 = 0x00C8_B888;
const SPROUT_COLOR_YOUNG: u32 = 0x0060_A040;
const SPROUT_COLOR_GROWN: u32 = 0x0030_FF70;

struct Seed {
    launch_col: f32,
    launch_row: f32,
    land_col: f32,
    land_row: f32,
    flight_time: f32,
    arc_height: f32,
    germinate_delay: f32,
    growth_period: f32,
    bloom_hold: f32,
    phase0_frac: f32,
    blooms: bool,
    seed_color: u32,
    sprout_young: u32,
    sprout_grown: u32,
}

impl Seed {
    fn total_cycle(&self) -> f32 {
        (self.flight_time + self.germinate_delay + self.growth_period + self.bloom_hold).max(0.1)
    }

    /// The final tip stage this seed's growth caps at — 4 (flowering) or 3
    /// (a plain leafy tip), per [`BLOOM_CHANCE`].
    fn max_stage(&self) -> usize {
        if self.blooms {
            SPROUT_GLYPHS.len() - 1
        } else {
            SPROUT_GLYPHS.len() - 2
        }
    }
}

/// A germinating, endlessly-cycling seed garden.
pub struct Seeds {
    seeds: Vec<Seed>,
}

fn hue_rotate(base: u32, delta: f32) -> u32 {
    let mut oklch = packed_to_oklch(base);
    oklch.h += delta;
    oklch_to_packed(oklch)
}

impl Seeds {
    /// Build a deterministic "launch, arc, land, sprout, bloom" seed-plant
    /// effect from `seed`, laid out across `rect`: a handful of ground-level
    /// launch sources, each firing several seeds on overlapping ballistic
    /// arcs that land and grow independently.
    #[must_use]
    pub fn random(seed: u64, rect: Rect) -> Self {
        let source_count = SOURCE_COUNT_RANGE.0
            + (hash_unit(seed, 0, 0) * (SOURCE_COUNT_RANGE.1 - SOURCE_COUNT_RANGE.0 + 1) as f64)
                as usize;
        let source_count = source_count.clamp(SOURCE_COUNT_RANGE.0, SOURCE_COUNT_RANGE.1);

        // Evenly-spaced anchors across the width, each jittered so sources
        // don't look like a mechanically uniform row of pots.
        let slot_w = rect.cols as f32 / source_count as f32;
        let mut sources = Vec::with_capacity(source_count);
        for si in 0..source_count {
            let sid = si as u64;
            let anchor = rect.col0 as f32 + slot_w * (si as f32 + 0.5);
            let jitter = range(seed, sid, SRC_S_COL_JITTER, -0.3, 0.3) * slot_w;
            let launch_col =
                (anchor + jitter).clamp(rect.col0 as f32, (rect.col0 + rect.cols) as f32 - 1.0);
            let launch_row =
                rect.row0 as f32 + rect.rows as f32 * range(seed, sid, SRC_S_ROW, 0.72, 0.92);
            sources.push((launch_col, launch_row));
        }

        let per_source = SEEDS_PER_SOURCE_RANGE.0
            + (hash_unit(seed, 0, 1)
                * (SEEDS_PER_SOURCE_RANGE.1 - SEEDS_PER_SOURCE_RANGE.0 + 1) as f64)
                as usize;
        let per_source = per_source.clamp(SEEDS_PER_SOURCE_RANGE.0, SEEDS_PER_SOURCE_RANGE.1);
        let count = per_source * source_count;

        // Half-spread for ONE source's own burst — sized so neighbouring
        // sources' bursts comfortably overlap and cover the gaps between
        // anchors rather than leaving bare strips (see the module doc).
        let half_col = slot_w * 0.85;
        let half_row = rect.rows as f32 * 0.55;

        let mut seeds = Vec::with_capacity(count);
        for i in 0..count {
            let id = i as u64;
            // Shuffle which source a seed belongs to (rather than a strict
            // round-robin `i % source_count`) so two seeds launched from the
            // same source aren't suspiciously evenly interleaved in index
            // order — purely cosmetic, but avoids an accidental visible
            // pattern in which seed lands where relative to its neighbours.
            let src_idx = ((hash_unit(seed, id, S_SOURCE_PICK) * source_count as f64) as usize)
                .min(source_count - 1);
            let (launch_col, launch_row) = sources[src_idx];

            let a_raw = angle(seed, id, S_ANGLE);
            // Bias the scatter upward/outward (a launch, not a uniform ring)
            // by folding the angle into the upper half-plane: in [-PI, 0],
            // up and to either side.
            let a = -a_raw.abs().min(PI);
            let dist_frac = range(
                seed,
                id,
                S_DIST,
                LAUNCH_DIST_FRAC_RANGE.0,
                LAUNCH_DIST_FRAC_RANGE.1,
            );
            let land_col = (launch_col + dist_frac * half_col * a.cos())
                .clamp(rect.col0 as f32, (rect.col0 + rect.cols) as f32 - 1.0);
            let land_row = (launch_row + dist_frac * half_row * a.sin())
                .clamp(rect.row0 as f32, (rect.row0 + rect.rows) as f32 - 1.0);

            let hue = (crate::hash::hash_signed(seed, id, S_HUE) as f32) * HUE_JITTER_RANGE;
            let flight_time = range(seed, id, S_FLIGHT, FLIGHT_TIME_RANGE.0, FLIGHT_TIME_RANGE.1);
            let germinate_delay = range(
                seed,
                id,
                S_DELAY,
                GERMINATE_DELAY_RANGE.0,
                GERMINATE_DELAY_RANGE.1,
            );
            let growth_period = range(
                seed,
                id,
                S_GROWTH,
                GROWTH_PERIOD_RANGE.0,
                GROWTH_PERIOD_RANGE.1,
            );
            let bloom_hold = range(
                seed,
                id,
                S_BLOOM_HOLD,
                BLOOM_HOLD_RANGE.0,
                BLOOM_HOLD_RANGE.1,
            );
            seeds.push(Seed {
                launch_col,
                launch_row,
                land_col,
                land_row,
                flight_time,
                arc_height: range(seed, id, S_ARC, ARC_HEIGHT_RANGE.0, ARC_HEIGHT_RANGE.1),
                germinate_delay,
                growth_period,
                bloom_hold,
                phase0_frac: range(seed, id, S_PHASE0, 0.0, 1.0),
                blooms: chance(seed, id, S_BLOOMS, BLOOM_CHANCE),
                seed_color: hue_rotate(SEED_COLOR, hue),
                sprout_young: hue_rotate(SPROUT_COLOR_YOUNG, hue),
                sprout_grown: hue_rotate(SPROUT_COLOR_GROWN, hue),
            });
        }
        Seeds { seeds }
    }

    /// Total number of seeds across every source.
    #[must_use]
    pub fn count(&self) -> usize {
        self.seeds.len()
    }
}

/// What one seed looks like at time `t`: either still in flight (a position)
/// or germinating/grown (a stage fraction `0.0..=1.0`). Exposed for tests.
enum SeedState {
    Flight { col: f32, row: f32 },
    Sprout { stage_frac: f32 },
}

/// `t_local` is ALREADY the seed's own cycle-local time (post-
/// `rem_euclid(total_cycle)` — see [`Seeds::random`]'s doc and
/// [`AsciiEffect::paint`]'s call site). Kept as a free function taking the
/// resolved local time (rather than the absolute `t`) so a test can probe
/// any point in the cycle directly without re-deriving the modulo.
fn seed_state(s: &Seed, t_local: f32) -> SeedState {
    if t_local < s.flight_time {
        let frac = (t_local / s.flight_time.max(0.001)).clamp(0.0, 1.0);
        let col = s.launch_col + (s.land_col - s.launch_col) * frac;
        let row =
            s.launch_row + (s.land_row - s.launch_row) * frac - s.arc_height * (PI * frac).sin();
        SeedState::Flight { col, row }
    } else {
        let since_land = t_local - s.flight_time;
        if since_land < s.germinate_delay {
            SeedState::Sprout { stage_frac: 0.0 }
        } else {
            let grow_t = since_land - s.germinate_delay;
            let frac = (grow_t / s.growth_period.max(0.001)).clamp(0.0, 1.0);
            SeedState::Sprout { stage_frac: frac }
        }
    }
}

impl AsciiEffect for Seeds {
    fn name(&self) -> &'static str {
        "seeds"
    }

    fn paint(&self, fb: &mut Framebuffer, rect: Rect, t: Duration) {
        let secs_t = secs(t);
        for s in &self.seeds {
            let cycle = s.total_cycle();
            let t_local = (secs_t + s.phase0_frac * cycle).rem_euclid(cycle);
            match seed_state(s, t_local) {
                SeedState::Flight { col, row } => {
                    // Fading trail: a few earlier points on the SAME closed-
                    // form arc, dimmer the further back — a visible arc/trail
                    // instead of one bare travelling dot. Guarded against
                    // wrapping past the cycle seam (echo_t < 0 after the
                    // modulo, at the very start of a fresh flight).
                    for k in (1..=FLIGHT_TRAIL_STEPS).rev() {
                        let echo_t = t_local - k as f32 * FLIGHT_TRAIL_DT;
                        if echo_t < 0.0 {
                            continue;
                        }
                        if let SeedState::Flight { col: ec, row: er } = seed_state(s, echo_t) {
                            let fade = 1.0 - k as f32 / (FLIGHT_TRAIL_STEPS + 1) as f32;
                            set_clipped(
                                fb,
                                rect,
                                ec.round() as i64,
                                er.round() as i64,
                                Cell {
                                    glyph: u32::from('\u{b7}'),
                                    fg: scale_color(s.seed_color, fade),
                                    bg: 0x0000_0000,
                                    ..Default::default()
                                },
                            );
                        }
                    }
                    set_clipped(
                        fb,
                        rect,
                        col.round() as i64,
                        row.round() as i64,
                        Cell {
                            glyph: u32::from(SEED_FLIGHT_GLYPH),
                            fg: s.seed_color,
                            bg: 0x0000_0000,
                            ..Default::default()
                        },
                    );
                }
                SeedState::Sprout { stage_frac } => {
                    let max_stage = s.max_stage();
                    let stage = ((stage_frac * max_stage as f32).round() as usize).min(max_stage);
                    let color = lerp_color(s.sprout_young, s.sprout_grown, stage_frac);
                    let land_col = s.land_col.round() as i64;
                    let land_row = s.land_row.round() as i64;

                    // Lush sprouting: taller stages grow an actual STEM
                    // (rows of `│` climbing above the landing point) with a
                    // pair of leaf glyphs partway up once the stem is tall
                    // enough, topped by this stage's tip glyph — not just one
                    // escalating glyph sitting in place.
                    for h in 1..stage {
                        set_clipped(
                            fb,
                            rect,
                            land_col,
                            land_row - h as i64,
                            Cell {
                                glyph: u32::from(STEM_GLYPH),
                                fg: color,
                                bg: 0x0000_0000,
                                ..Default::default()
                            },
                        );
                    }
                    if stage >= 2 {
                        let leaf_row = land_row - (stage as i64 / 2).max(1);
                        set_clipped(
                            fb,
                            rect,
                            land_col - 1,
                            leaf_row,
                            Cell {
                                glyph: u32::from(LEAF_LEFT),
                                fg: color,
                                bg: 0x0000_0000,
                                ..Default::default()
                            },
                        );
                        set_clipped(
                            fb,
                            rect,
                            land_col + 1,
                            leaf_row,
                            Cell {
                                glyph: u32::from(LEAF_RIGHT),
                                fg: color,
                                bg: 0x0000_0000,
                                ..Default::default()
                            },
                        );
                    }
                    let top_row = land_row - stage as i64;
                    set_clipped(
                        fb,
                        rect,
                        land_col,
                        top_row,
                        Cell {
                            glyph: u32::from(SPROUT_GLYPHS[stage]),
                            fg: color,
                            bg: 0x0000_0000,
                            ..Default::default()
                        },
                    );
                    // Full bloom: a small flowering cluster either side of
                    // the tip — only for the ~42% of seeds `blooms`.
                    if s.blooms && stage == SPROUT_GLYPHS.len() - 1 {
                        set_clipped(
                            fb,
                            rect,
                            land_col - 1,
                            top_row,
                            Cell {
                                glyph: u32::from('*'),
                                fg: color,
                                bg: 0x0000_0000,
                                ..Default::default()
                            },
                        );
                        set_clipped(
                            fb,
                            rect,
                            land_col + 1,
                            top_row,
                            Cell {
                                glyph: u32::from('*'),
                                fg: color,
                                bg: 0x0000_0000,
                                ..Default::default()
                            },
                        );
                    }
                }
            }
        }
    }
}

fn lerp_color(a: u32, b: u32, t: f32) -> u32 {
    let la = re_core::oklab::unpack_linear(a);
    let lb = re_core::oklab::unpack_linear(b);
    re_core::oklab::pack_linear(re_core::oklab::lerp_linear(la, lb, t.clamp(0.0, 1.0)))
}

fn scale_color(c: u32, k: f32) -> u32 {
    let l = re_core::oklab::unpack_linear(c);
    re_core::oklab::pack_linear(re_core::oklab::LinRgb {
        r: l.r * k,
        g: l.g * k,
        b: l.b * k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECT: Rect = Rect {
        col0: 0,
        row0: 0,
        cols: 80,
        rows: 30,
    };

    #[test]
    fn determinism_same_seed() {
        let a = Seeds::random(13, RECT);
        let b = Seeds::random(13, RECT);
        assert_eq!(a.count(), b.count());
        for (x, y) in a.seeds.iter().zip(b.seeds.iter()) {
            assert_eq!((x.land_col, x.land_row), (y.land_col, y.land_row));
            assert_eq!(x.flight_time, y.flight_time);
        }
    }

    #[test]
    fn different_seeds_scatter_and_grow_differently() {
        let a = Seeds::random(1, RECT);
        let b = Seeds::random(2, RECT);
        assert!(a
            .seeds
            .iter()
            .zip(b.seeds.iter())
            .any(|(x, y)| x.land_col != y.land_col));
        assert!(a
            .seeds
            .iter()
            .zip(b.seeds.iter())
            .any(|(x, y)| x.growth_period != y.growth_period));
        assert_ne!(a.count(), 0);
    }

    #[test]
    fn different_seeds_diverge_in_source_count_and_bloom_hold() {
        // The owner's ask, verbatim: "size, direction, and above all
        // FREQUENCIES". `bloom_hold` is the new independent per-seed rate
        // that makes the whole effect loop; count varies with the number of
        // escape sources, which is itself randomized.
        let mut any_count_diff = false;
        let mut any_hold_diff = false;
        for seed in 0..12u64 {
            let a = Seeds::random(seed, RECT);
            let b = Seeds::random(seed + 1000, RECT);
            if a.count() != b.count() {
                any_count_diff = true;
            }
            if (a.seeds[0].bloom_hold - b.seeds[0].bloom_hold).abs() > 1e-6 {
                any_hold_diff = true;
            }
        }
        assert!(
            any_count_diff,
            "seed count never varied across 12 seed pairs"
        );
        assert!(
            any_hold_diff,
            "bloom_hold never varied across 12 seed pairs"
        );
    }

    #[test]
    fn every_seed_eventually_reaches_full_growth_within_one_cycle() {
        let s = Seeds::random(4, RECT);
        for seed in &s.seeds {
            let t_done = seed.flight_time + seed.germinate_delay + seed.growth_period + 0.01;
            match seed_state(seed, t_done) {
                SeedState::Sprout { stage_frac } => assert!(stage_frac >= 0.999),
                SeedState::Flight { .. } => panic!("seed still in flight past its own flight_time"),
            }
        }
    }

    #[test]
    fn the_cycle_relaunches_after_the_bloom_hold() {
        // Regression for the whole point of the loop: well past
        // flight+delay+growth+hold, the seed must be back in FLIGHT (a fresh
        // launch), not stuck holding its bloom forever.
        let s = Seeds::random(4, RECT);
        for seed in &s.seeds {
            let cycle = seed.total_cycle();
            let t_local = (cycle + 0.001).rem_euclid(cycle);
            match seed_state(seed, t_local) {
                SeedState::Flight { .. } => {}
                SeedState::Sprout { .. } => {
                    panic!("seed did not relaunch just past its own cycle boundary")
                }
            }
        }
    }

    #[test]
    fn about_a_minority_but_not_zero_or_all_seeds_bloom() {
        let s = Seeds::random(77, RECT);
        let bloomers = s.seeds.iter().filter(|sd| sd.blooms).count();
        assert!(bloomers > 0, "no seed blooms at all");
        assert!(
            bloomers < s.count(),
            "every seed blooms — no leafy-tip variety"
        );
    }

    #[test]
    fn landing_position_stays_within_the_rect() {
        let s = Seeds::random(6, RECT);
        for seed in &s.seeds {
            assert!(
                seed.land_col >= RECT.col0 as f32 && seed.land_col < (RECT.col0 + RECT.cols) as f32
            );
            assert!(
                seed.land_row >= RECT.row0 as f32 && seed.land_row < (RECT.row0 + RECT.rows) as f32
            );
        }
    }

    #[test]
    fn glyphs_used_are_all_reliable() {
        use re_render::glyph_family::is_reliable;
        for g in SPROUT_GLYPHS {
            assert!(
                is_reliable(g),
                "sprout glyph {g:?} is not in the reliable set"
            );
        }
        for g in [
            SEED_FLIGHT_GLYPH,
            STEM_GLYPH,
            LEAF_LEFT,
            LEAF_RIGHT,
            '\u{b7}',
        ] {
            assert!(
                is_reliable(g),
                "seeds glyph {g:?} is not in the reliable set"
            );
        }
    }

    #[test]
    fn seed_flight_glyph_is_not_the_tofu_prone_ring_operator() {
        // Regression: U+2218
        // is technically `is_reliable` but renders as tofu in this project's
        // own Consolas/Cascadia preview tooling.
        assert_ne!(SEED_FLIGHT_GLYPH, '\u{2218}');
    }

    #[test]
    fn the_burst_spreads_across_the_full_width_of_a_wide_rect() {
        // Regression for the 2026-09-24 gallery2 review: scatter distance was
        // scaled by `rect.min_extent()` (rows, for a wide tile), collapsing a
        // 70-wide rect's usable spread down to ~25 cells — a small off-centre
        // cluster instead of a burst that fills the rect. Now several
        // sources spread across the width, each with its own local scatter.
        const WIDE: Rect = Rect {
            col0: 0,
            row0: 0,
            cols: 70,
            rows: 26,
        };
        let s = Seeds::random(21, WIDE);
        let min_col = s
            .seeds
            .iter()
            .map(|sd| sd.land_col)
            .fold(f32::MAX, f32::min);
        let max_col = s
            .seeds
            .iter()
            .map(|sd| sd.land_col)
            .fold(f32::MIN, f32::max);
        assert!(
            max_col - min_col > 45.0,
            "landing spread only {} cells wide on a 70-wide rect",
            max_col - min_col
        );
    }

    #[test]
    fn paint_stays_inside_its_rect() {
        let mut fb = Framebuffer::new(90, 34);
        let rect = Rect::new(5, 2, 80, 30);
        let seeds = Seeds::random(8, rect);
        for ms in [0u64, 200, 900, 2500, 8000, 15_000] {
            seeds.paint(&mut fb, rect, Duration::from_millis(ms));
        }
        for row in 0..34 {
            for col in 0..90 {
                if !rect.contains(col as i64, row as i64) {
                    assert_eq!(*fb.get(col, row).unwrap(), re_core::Cell::default());
                }
            }
        }
    }

    #[test]
    fn the_garden_fills_a_meaningful_fraction_of_the_rect_at_t_zero() {
        // The owner's exact complaint: "too sparse". Every seed's phase0 is
        // spread across its own cycle, so even at t=0 many plants should
        // already be mid-growth or in flower, not all frozen at "just
        // launched" — a dense, lush frame from the very first paint.
        let mut fb = Framebuffer::new(80, 30);
        let seeds = Seeds::random(3, RECT);
        seeds.paint(&mut fb, RECT, Duration::ZERO);
        let lit = (0..30)
            .flat_map(|r| (0..80).map(move |c| (c, r)))
            .filter(|&(c, r)| *fb.get(c, r).unwrap() != re_core::Cell::default())
            .count();
        assert!(
            lit > 200,
            "expected a lush garden even at t=0, only {lit} cells lit"
        );
    }

    #[test]
    fn the_garden_keeps_turning_over_rather_than_freezing_on_one_frame() {
        // A snapshot minutes later must differ from an early one — the loop
        // is actually doing something, not settling into a static picture.
        let mut fb_early = Framebuffer::new(80, 30);
        let mut fb_late = Framebuffer::new(80, 30);
        let seeds = Seeds::random(9, RECT);
        seeds.paint(&mut fb_early, RECT, Duration::from_millis(500));
        seeds.paint(&mut fb_late, RECT, Duration::from_millis(47_000));
        let mut any_diff = false;
        for row in 0..30 {
            for col in 0..80 {
                if fb_early.get(col, row) != fb_late.get(col, row) {
                    any_diff = true;
                }
            }
        }
        assert!(
            any_diff,
            "the garden looks identical 46.5s later — the loop is not advancing"
        );
    }
}
