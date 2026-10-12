//! `re_render::detail_lod` — material DETAIL TIERS (tone / pattern / detail)
//! with a world-anchored stochastic cross-fade, plus the underline micro-detail
//! reveal gate.
//!
//! Instead of one glyph/colour sample per cell whatever the distance, a
//! surface shows three qualitatively different renderings depending on the
//! projected texel density ρ ([`crate::lod::Footprint`]):
//!
//! * **TONE** (ρ ≥ [`TONE_RHO`]) — the pattern is finer than a cell; show only its
//!   prefiltered MEAN (glyph often blank, colour carries everything).
//! * **PATTERN** ([`DETAIL_RHO`] ≤ ρ < [`TONE_RHO`]) — one glyph per texel; the
//!   material's own motif (brick bond, grid, circuit trace, …).
//! * **DETAIL** (ρ < [`DETAIL_RHO`]) — the next octave INSIDE each texel: fine
//!   grain, cracks, relief — plus optional underline micro-detail.
//!
//! The transition between tiers must never POP and must never SHIMMER while
//! turning: [`cross_fade_weight`] gives a continuous `[0,1]` blend weight from
//! `smoothstep`, and [`reveal_finer`] decides which INDIVIDUAL texel has
//! already "opened up" using a WORLD-anchored hash threshold (`ign(world texel
//! id) < w`) — never `(col, row)`. As the player approaches, texels cross
//! their own threshold one at a time, in world space, which is what reads as
//! the surface spreading out rather than the whole face snapping to a new
//! tier at once (the brick-wall example: bricks open up one by one as you
//! walk in).
//!
//! This module is PURE (no Bevy, no terminal, no material coupling) — see the
//! crate-level docs for why `re_render` is engine-agnostic. It publishes the
//! primitives; wiring a `Material` flag through the `.lvl` → `MaterialDef` →
//! `Material` pipeline to actually GATE a recipe on them is tracked separately
//! (the opt-in `[material] detail = on` key parses in `re_level`, and the field
//! reaches `re_world::MaterialDef`, but the last hop — into `re_render`'s live
//! registry-construction in `lib.rs` — is not yet wired).

use crate::ascii_shader::hash01;
use crate::lod::Footprint;

/// ρ boundary between TONE and PATTERN: "ρ ≥ 2 ⇒ TONE". Named separately from
/// [`crate::lod::LodConfig`]'s near/mid/far boundaries because the detail-tier
/// table is calibrated directly in ρ units, not derived from the (unrelated)
/// 6/13 world-cell distance bands [`crate::lod`] keeps byte-identical.
pub const TONE_RHO: f32 = 2.0;

/// ρ boundary between PATTERN and DETAIL: "ρ < 0.5 ⇒ DETAIL".
pub const DETAIL_RHO: f32 = 0.5;

/// The fraction of the DETAIL band (`0..DETAIL_RHO`) over which underline
/// micro-detail fades in, counted from the FINE end (small ρ = closer = deeper
/// into DETAIL). Micro-detail fades in only in the top 30% of the DETAIL band,
/// so it never flickers at the tier edge — i.e. only once a texel is solidly
/// inside DETAIL, not right at the ρ = `DETAIL_RHO` boundary where the glyph
/// itself is still cross-fading.
pub const MICRO_DETAIL_BAND_FRACTION: f32 = 0.3;

/// The three detail tiers, coarsest (least resolvable) to finest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetailTier {
    /// ρ ≥ [`TONE_RHO`]: pattern finer than a cell — mean colour only.
    Tone,
    /// [`DETAIL_RHO`] ≤ ρ < [`TONE_RHO`]: one glyph per texel, the material motif.
    Pattern,
    /// ρ < [`DETAIL_RHO`]: the next octave inside each texel, plus micro-detail.
    Detail,
}

/// Resolve the DETAIL tier for a footprint density `rho` (pattern texels per
/// cell — [`crate::lod::Footprint::rho`]). Non-finite `rho` (sky / a miss) is the
/// coarsest tier, matching [`crate::lod::LodConfig::tier_for`]'s non-finite rule.
#[inline]
pub fn resolve_tier(rho: f32) -> DetailTier {
    if !rho.is_finite() || rho >= TONE_RHO {
        DetailTier::Tone
    } else if rho >= DETAIL_RHO {
        DetailTier::Pattern
    } else {
        DetailTier::Detail
    }
}

/// The continuous cross-fade weight TOWARD FINER DETAIL as `rho` falls
/// (`w = smoothstep(ρ_lo, ρ_hi, ρ)`, expressed here as "how much of the finer
/// tier should show" so it rises monotonically as the surface is approached).
/// Used to lerp B/F colour continuously (`lerp` in whatever colour space the
/// caller works in) and as the stochastic threshold [`reveal_finer`] compares
/// individual texels against.
///
/// Implemented as ONE smooth ramp spanning the entire `[DETAIL_RHO, TONE_RHO]`
/// range in LOG space (ρ is a multiplicative quantity — halving it is "one
/// octave finer" regardless of where you start), rather than two independently
/// authored ramps that would have to be stitched together at each boundary by
/// hand. That gives continuity and monotonicity across BOTH the Tone/Pattern and
/// Pattern/Detail transitions for free instead of as a proven invariant.
///
/// Returns `0.0` at/above [`TONE_RHO`] (pure Tone — nothing finer to reveal),
/// `1.0` at/below [`DETAIL_RHO`] (already the finest tier), and a smooth ramp in
/// between. A non-finite `rho` (sky / a miss) returns `0.0` — no finer detail.
#[inline]
pub fn cross_fade_weight(rho: f32) -> f32 {
    if !rho.is_finite() {
        return 0.0;
    }
    let log_rho = rho.max(1e-6).log2();
    let log_tone = TONE_RHO.log2();
    let log_detail = DETAIL_RHO.log2();
    1.0 - crate::ascii_shader::smoothstep(log_detail, log_tone, log_rho)
}

/// The WORLD-anchored stochastic reveal test: bricks open up one by one as you
/// walk in. `world_texel_id` identifies ONE texel in world space
/// (e.g. a brick's `hash_cell` id, or `(floor(u), floor(v))` packed into a single
/// key) — never a screen coordinate. `seed` is the material's own seed so two
/// materials do not open their texels in lock-step. `w` is [`cross_fade_weight`]'s
/// output at the cell's CURRENT `rho`.
///
/// Because the per-texel threshold `ign(id)` is a pure hash of the WORLD id, a
/// given texel crosses over at exactly one `w` value every time any camera looks
/// at it, from any angle — the reveal cannot shimmer under a turn, and it
/// cannot flip back and forth from frame to frame the way a `(col,row)`-keyed
/// threshold would (the same screen-vs-world anchoring mistake restated at the
/// texel-detail level).
#[inline]
pub fn reveal_finer(world_texel_id: u64, seed: u32, w: f32) -> bool {
    let h = crate::ascii_shader::hash2i(
        (world_texel_id & 0xFFFF_FFFF) as i32,
        ((world_texel_id >> 32) as u32 ^ seed) as i32,
    );
    hash01(h) < w
}

/// The underline MICRO-DETAIL reveal weight: fades in only in the top 30% of
/// the DETAIL band. `0.0` outside [`DetailTier::Detail`] or in
/// its outer 70%, ramping to `1.0` as `rho → 0` (the texel is now filling most of
/// the view). Multiply this into whatever alpha/opacity the caller uses for the
/// mortar-hairline / wood-grain-dotted / crack-curly underline so it fades in
/// smoothly rather than switching on the instant the DETAIL tier is entered.
#[inline]
pub fn micro_detail_weight(rho: f32) -> f32 {
    if !rho.is_finite() || rho >= DETAIL_RHO {
        return 0.0;
    }
    let band_start = DETAIL_RHO * MICRO_DETAIL_BAND_FRACTION;
    // rho in [0, band_start] -> weight in [1, 0]; smoothstep(0, band_start, rho)
    // rises 0->1 as rho rises 0->band_start, so invert it.
    1.0 - crate::ascii_shader::smoothstep(0.0, band_start.max(1e-6), rho.max(0.0))
}

/// Convenience: resolve `(tier, cross_fade_weight)` in one call from a
/// [`Footprint`] the way a recipe would after reading the per-cell footprint.
#[inline]
pub fn tier_and_weight(fp: Footprint) -> (DetailTier, f32) {
    (resolve_tier(fp.rho), cross_fade_weight(fp.rho))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_match_the_literal_spec_table() {
        assert_eq!(resolve_tier(3.0), DetailTier::Tone, "rho >= 2 is TONE");
        assert_eq!(
            resolve_tier(TONE_RHO),
            DetailTier::Tone,
            "boundary itself is TONE"
        );
        assert_eq!(
            resolve_tier(1.0),
            DetailTier::Pattern,
            "0.5 <= rho < 2 is PATTERN"
        );
        assert_eq!(
            resolve_tier(DETAIL_RHO),
            DetailTier::Pattern,
            "boundary itself is PATTERN"
        );
        assert_eq!(resolve_tier(0.3), DetailTier::Detail, "rho < 0.5 is DETAIL");
        assert_eq!(resolve_tier(0.0), DetailTier::Detail);
    }

    #[test]
    fn non_finite_rho_is_the_coarsest_tier() {
        assert_eq!(resolve_tier(f32::INFINITY), DetailTier::Tone);
        assert_eq!(resolve_tier(f32::NAN), DetailTier::Tone);
        assert_eq!(cross_fade_weight(f32::NAN), 0.0);
    }

    #[test]
    fn cross_fade_weight_is_monotonic_decreasing_in_rho() {
        // As rho shrinks (closer / finer), the weight toward more detail rises.
        let samples = [10.0, 4.0, TONE_RHO, 1.5, 1.0, DETAIL_RHO, 0.3, 0.1, 0.0];
        let mut prev = -1.0f32;
        for &rho in &samples {
            let w = cross_fade_weight(rho);
            assert!(
                (0.0..=1.0).contains(&w),
                "w must stay in [0,1], got {w} at rho={rho}"
            );
            assert!(
                w >= prev - 1e-6,
                "weight must not DECREASE as rho falls: rho={rho} w={w} prev={prev}"
            );
            prev = w;
        }
        // And it actually reaches both ends over this range.
        assert!(cross_fade_weight(10.0) < 0.05, "far tone: weight ~0");
        assert!(cross_fade_weight(0.0) > 0.95, "deep detail: weight ~1");
    }

    #[test]
    fn cross_fade_weight_is_continuous_across_both_boundaries() {
        // No jump at TONE_RHO or DETAIL_RHO — approaching from either side gives
        // nearly the same weight (the whole point of a cross-fade, not a snap).
        for boundary in [TONE_RHO, DETAIL_RHO] {
            let just_above = cross_fade_weight(boundary + 0.001);
            let just_below = cross_fade_weight(boundary - 0.001);
            assert!(
                (just_above - just_below).abs() < 0.01,
                "boundary {boundary}: {just_above} vs {just_below} must be nearly equal"
            );
        }
    }

    #[test]
    fn reveal_finer_is_a_pure_function_of_world_id_not_of_call_order() {
        let a = reveal_finer(4242, 7, 0.5);
        let b = reveal_finer(4242, 7, 0.5);
        assert_eq!(a, b, "same world id + seed + weight must always agree");
        // A different seed can (and generally will) disagree — materials don't
        // open their texels in lock-step.
    }

    #[test]
    fn reveal_finer_never_reveals_at_zero_weight_and_always_at_full_weight() {
        for id in 0..64u64 {
            assert!(!reveal_finer(id, 1, 0.0), "w=0 must never reveal (id={id})");
            assert!(
                reveal_finer(id, 1, 1.0001),
                "w>1 must always reveal (id={id})"
            );
        }
    }

    #[test]
    fn reveal_finer_opens_texels_one_at_a_time_as_weight_rises() {
        // As `w` sweeps 0..1, the set of "already revealed" texel ids only GROWS
        // (a texel that opened at a lower w stays open at every higher w) — this
        // is what "opening up one by one" means: each texel crosses its OWN
        // threshold exactly once, never flickers open/closed as w rises.
        let ids: Vec<u64> = (0..200).collect();
        let mut prev_open: Vec<bool> = vec![false; ids.len()];
        let mut steps = 0u32;
        let mut w = 0.0f32;
        while w <= 1.0 {
            for (i, &id) in ids.iter().enumerate() {
                let open = reveal_finer(id, 99, w);
                assert!(
                    open || !prev_open[i],
                    "texel {id} closed again after opening (w={w}) — reveal is not monotone"
                );
                prev_open[i] = open;
            }
            w += 0.05;
            steps += 1;
        }
        assert!(steps > 15, "sanity: the sweep actually ran");
        // And SOME but not ALL ids opened at an intermediate weight — proving the
        // texels really do open at different weights, not all at once.
        let mid_open = ids.iter().filter(|&&id| reveal_finer(id, 99, 0.5)).count();
        assert!(
            mid_open > 0 && mid_open < ids.len(),
            "mid-weight opens a PARTIAL set, got {mid_open}/200"
        );
    }

    #[test]
    fn micro_detail_weight_is_zero_outside_the_inner_thirty_percent_of_detail() {
        assert_eq!(
            micro_detail_weight(DETAIL_RHO),
            0.0,
            "at the Pattern/Detail boundary itself"
        );
        assert_eq!(
            micro_detail_weight(DETAIL_RHO * 0.5),
            0.0,
            "outer 70% of the DETAIL band"
        );
        assert!(
            micro_detail_weight(DETAIL_RHO * 0.29) > 0.0,
            "inside the inner 30%"
        );
        assert_eq!(
            micro_detail_weight(0.0),
            1.0,
            "deepest into DETAIL: full micro-detail"
        );
        assert_eq!(
            micro_detail_weight(TONE_RHO),
            0.0,
            "well outside DETAIL entirely"
        );
    }

    #[test]
    fn micro_detail_weight_never_fires_in_tone_or_pattern() {
        for &rho in &[5.0, TONE_RHO, 1.0, DETAIL_RHO, DETAIL_RHO + 0.01] {
            assert_eq!(
                micro_detail_weight(rho),
                0.0,
                "rho={rho} is not inside DETAIL"
            );
        }
    }

    #[test]
    fn tier_and_weight_matches_the_split_calls() {
        let fp = Footprint::reference(6.0);
        let (tier, w) = tier_and_weight(fp);
        assert_eq!(tier, resolve_tier(fp.rho));
        assert_eq!(w, cross_fade_weight(fp.rho));
    }
}
