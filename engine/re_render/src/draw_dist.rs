//! Runtime DRAW-DISTANCE knobs — the two formerly-`const` caster bounds promoted to
//! per-cast thread-local values so a GRAPHICS-QUALITY setting can shorten / lengthen
//! the render reach at runtime WITHOUT a re-compile.
//!
//! Two bounds together define how far the caster looks:
//!
//! * **far-freeze distance** ([`active_far_freeze`]) — at/beyond it a cell is rendered
//!   fully quiet (the anti-noise "fog-death" cutoff, see
//!   [`crate::ascii_shader::FAR_FREEZE_DIST`]). The renderer also shortens fog
//!   (`fog_far`) and the LOD `far_start` alongside this, but those are their own
//!   resources; this is the one the deep shade dispatch reads.
//! * **DDA step cap** ([`active_max_dda_steps`]) — the hard upper bound on how many
//!   voxel boundaries the primary ray crosses before giving up (escaped → open sky).
//!   A shorter cap is a cheaper cast (fewer empty-air steps on a long corridor).
//!
//! ## Why thread-locals (mirrors [`crate::config`] / [`crate::lod`])
//!
//! Both values are read DEEP in the cast (the DDA loop, the `Material::sample`
//! dispatch) where a plain function argument cannot reach without a multi-file API
//! churn. So — exactly like the shading knobs and the LOD config — they are held in
//! thread-local cells written ONCE per cast via [`apply`] (from the live
//! [`crate::FarFreeze`] resource), and read back in the hot path. This is NOT an env
//! global: there is no `env::var` here, so a unit test sets the values directly and
//! asserts the result with zero environment dependence. Stable within a cast ⇒
//! golden-safe.
//!
//! ## Default == today
//!
//! The thread-locals default to the historical constants
//! ([`crate::ascii_shader::FAR_FREEZE_DIST`] = 32, `MAX_DDA_STEPS` = 256), so an
//! un-applied path (and a default-quality build) is BYTE-IDENTICAL to the pre-feature
//! renderer. Only an explicit [`apply`] of a non-default [`crate::FarFreeze`] changes
//! the reach.

use std::cell::Cell as StdCell;

/// The historical DDA step cap (the old `MAX_DDA_STEPS` const). The v0.1 maps are
/// tiny so this is comfortably above any walk; a quality setting may LOWER it to cut
/// the empty-air march on a long sight-line.
pub const DEFAULT_MAX_DDA_STEPS: u32 = 256;

thread_local! {
    /// Far-freeze distance (world cells) for the current cast. Defaults to the
    /// historical [`crate::ascii_shader::FAR_FREEZE_DIST`] so an un-applied path is
    /// byte-identical. `f32::INFINITY` disables the freeze.
    static ACTIVE_FAR_FREEZE: StdCell<f32> =
        const { StdCell::new(crate::ascii_shader::FAR_FREEZE_DIST) };
    /// DDA step cap for the current cast. Defaults to [`DEFAULT_MAX_DDA_STEPS`].
    static ACTIVE_MAX_DDA_STEPS: StdCell<u32> = const { StdCell::new(DEFAULT_MAX_DDA_STEPS) };
}

/// Install the active draw-distance bounds for the current thread. Called by the cast
/// from the live [`crate::FarFreeze`] resource, and by tests directly. No env is read.
///
/// A non-finite or `≤ 0` `max_dda_steps` is clamped to at least 1 (a cast must take
/// at least one step); a NaN `far_freeze` falls back to the default so a poisoned
/// setting never disables the freeze silently.
#[inline]
pub fn apply(far_freeze: f32, max_dda_steps: u32) {
    let ff = if far_freeze.is_nan() {
        crate::ascii_shader::FAR_FREEZE_DIST
    } else {
        far_freeze
    };
    ACTIVE_FAR_FREEZE.with(|c| c.set(ff));
    ACTIVE_MAX_DDA_STEPS.with(|c| c.set(max_dda_steps.max(1)));
}

/// The far-freeze distance (world cells) for the current cast. Read by the cast paths
/// that previously passed `FarFreeze::default().distance`. Default
/// [`crate::ascii_shader::FAR_FREEZE_DIST`].
#[inline]
pub fn active_far_freeze() -> f32 {
    ACTIVE_FAR_FREEZE.with(|c| c.get())
}

/// The DDA step cap for the current cast. Read by `dda_hit`'s march loop. Default
/// [`DEFAULT_MAX_DDA_STEPS`].
#[inline]
pub fn active_max_dda_steps() -> u32 {
    ACTIVE_MAX_DDA_STEPS.with(|c| c.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_historical_consts() {
        // A fresh thread (this test thread, un-applied) reads the historical values —
        // the byte-identical-to-today guarantee.
        apply(crate::ascii_shader::FAR_FREEZE_DIST, DEFAULT_MAX_DDA_STEPS);
        assert_eq!(active_far_freeze(), crate::ascii_shader::FAR_FREEZE_DIST);
        assert_eq!(active_max_dda_steps(), DEFAULT_MAX_DDA_STEPS);
    }

    #[test]
    fn apply_threads_the_values_without_env() {
        apply(18.0, 96);
        assert_eq!(active_far_freeze(), 18.0);
        assert_eq!(active_max_dda_steps(), 96);
        // Restore default for sibling tests on this thread.
        apply(crate::ascii_shader::FAR_FREEZE_DIST, DEFAULT_MAX_DDA_STEPS);
    }

    #[test]
    fn max_steps_is_clamped_to_at_least_one() {
        apply(crate::ascii_shader::FAR_FREEZE_DIST, 0);
        assert_eq!(
            active_max_dda_steps(),
            1,
            "a cast must take at least one step"
        );
        apply(crate::ascii_shader::FAR_FREEZE_DIST, DEFAULT_MAX_DDA_STEPS);
    }

    #[test]
    fn nan_far_freeze_falls_back_to_default() {
        apply(f32::NAN, DEFAULT_MAX_DDA_STEPS);
        assert_eq!(active_far_freeze(), crate::ascii_shader::FAR_FREEZE_DIST);
    }

    #[test]
    fn infinity_far_freeze_disables_the_freeze() {
        apply(f32::INFINITY, DEFAULT_MAX_DDA_STEPS);
        assert!(active_far_freeze().is_infinite());
        apply(crate::ascii_shader::FAR_FREEZE_DIST, DEFAULT_MAX_DDA_STEPS);
    }
}
