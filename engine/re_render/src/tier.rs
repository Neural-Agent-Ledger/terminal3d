//! `re_render::tier` — the temporal-tier scheduler.
//!
//! Pipeline stage: policy only — decides HOW OFTEN each render tier redoes
//! its expensive work; the actual re-bake/raycast lives in `composite.rs`.
//! Inputs: [`TierClock`] (a monotonic frame counter) + the camera's yaw/move
//! delta since the last bake. Output: a `bool` per tier ([`far_needs_rebake`])
//! the composite pass gates its rebake on.
//! Knobs: [`FAR_REBAKE_FRAMES`], [`ARC_BUCKET`], [`FAR_MOVE_CELLS`].
//!
//! The *how often* axis of the tiered-composite design. Three cadences feed
//! the SAME cheap composite fold:
//!
//! - **Far (rare):** re-bake the cylindrical panorama only when the camera turns
//!   past a yaw bucket, moves ≥1 cell, or the slow safety timer fires. Otherwise
//!   the far layer is filled by radial-sampling the cached panorama — zero rays.
//! - **Near (on-dirty):** raycast the opaque + transparent layers gated by the
//!   existing world dirty-flag (owned elsewhere).
//! - **Emitters (every frame):** integrate the particle pool unconditionally.
//!
//! This module owns only the FAR cadence policy (the part that is new and purely
//! arithmetic). [`far_needs_rebake`] is a pure helper unit-testable with no Bevy.

use bevy::prelude::Resource;

/// Slow safety re-bake of the far panorama: even if the camera neither turns past
/// a yaw bucket nor crosses a cell, re-cast the ring at least this often (frames)
/// so a slowly-evolving far material / lighting eventually refreshes. A large
/// value — the far plane is deliberately lazy.
pub const FAR_REBAKE_FRAMES: u64 = 240;

/// Coarse yaw-bucket width (radians). The far ring is re-baked when the camera's
/// yaw moves more than this from the bake's `base_yaw`; within a bucket a turn is
/// served purely by radial-sampling the cached panorama (0 rays — the economy
/// invariant this tier exists to hold). ~11.25° = a 32-bucket ring.
pub const ARC_BUCKET: f32 = std::f32::consts::TAU / 32.0;

/// Camera translation past this many world cells (Euclidean, xy) invalidates the
/// whole far ring — far parallax changed enough to re-cast. Below it, far parallax is
/// ignored ON PURPOSE (parallax ∝ 1/distance, so the distant backdrop barely moves).
///
/// Tuned to 4.0 cells rather than a tighter 1.0: the far ring is the
/// BACKGROUND tier, and a wall many cells away shifts < 1 glyph column over
/// several steps of walking, so re-baking it on every single cell crossing
/// is wasteful churn. At
/// 4 cells the worst-case angular parallax of a backdrop feature at the near/far seam
/// (~12 cells, [`crate::composite::SUBCELL_MAX_DEPTH`] band) is still ≲ 1–2 ring
/// columns — the same sub-column error the radial-sample's colour blend already
/// absorbs — so the backdrop is re-used across rotation always AND across many
/// translation steps, re-baking rarely. A genuinely large move (4 cells) or a yaw
/// bucket crossing still rebakes before the parallax error becomes visible.
pub const FAR_MOVE_CELLS: f32 = 4.0;

/// Frame counter driving the three cadences. Bumped once per render tick by the
/// Bevy render system; the pure policy helpers read it.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TierClock {
    /// Monotonic render-frame index.
    pub frame: u64,
}

impl TierClock {
    /// Advance one frame.
    #[inline]
    pub fn tick(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }
}

/// Whether the far cylindrical panorama must be re-baked this frame.
///
/// Pure: re-bake when ANY of —
/// - the absolute yaw delta from the bake exceeds [`ARC_BUCKET`] (turned out of
///   the cached bucket),
/// - the camera moved at least [`FAR_MOVE_CELLS`] cells since the bake (parallax),
/// - the slow safety timer elapsed (`frames_since_bake ≥ FAR_REBAKE_FRAMES`).
///
/// `dyaw` is the signed yaw delta (radians) since the last bake; `moved_cells` is
/// the Euclidean xy distance (in cells) the camera moved since the last bake;
/// `frames_since_bake` is `clock.frame - bake_frame`.
#[inline]
pub fn far_needs_rebake(frames_since_bake: u64, dyaw: f32, moved_cells: f32) -> bool {
    dyaw.abs() > ARC_BUCKET
        || moved_cells >= FAR_MOVE_CELLS
        || frames_since_bake >= FAR_REBAKE_FRAMES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_ticks() {
        let mut c = TierClock::default();
        assert_eq!(c.frame, 0);
        c.tick();
        c.tick();
        assert_eq!(c.frame, 2);
    }

    #[test]
    fn no_rebake_within_bucket_idle() {
        // Small turn, no move, recent bake → 0 rays.
        assert!(!far_needs_rebake(1, ARC_BUCKET * 0.5, 0.0));
        assert!(!far_needs_rebake(0, 0.0, 0.0));
    }

    #[test]
    fn rebake_on_yaw_past_bucket() {
        assert!(far_needs_rebake(0, ARC_BUCKET * 1.01, 0.0));
        // sign-independent.
        assert!(far_needs_rebake(0, -ARC_BUCKET * 1.01, 0.0));
        // exactly at the bucket is still inside (strict >).
        assert!(!far_needs_rebake(0, ARC_BUCKET, 0.0));
    }

    #[test]
    fn rebake_on_cell_move() {
        assert!(far_needs_rebake(0, 0.0, FAR_MOVE_CELLS));
        assert!(far_needs_rebake(0, 0.0, FAR_MOVE_CELLS + 1.0));
        // A move SHORT of the (raised) threshold must NOT rebake — the far backdrop is
        // reused across many translation steps now (distant parallax is sub-cell).
        assert!(!far_needs_rebake(0, 0.0, FAR_MOVE_CELLS - 0.01));
        assert!(
            !far_needs_rebake(0, 0.0, 2.5),
            "2.5 cells is below the 4-cell far threshold"
        );
    }

    #[test]
    fn rebake_on_safety_timer() {
        assert!(far_needs_rebake(FAR_REBAKE_FRAMES, 0.0, 0.0));
        assert!(far_needs_rebake(FAR_REBAKE_FRAMES + 5, 0.0, 0.0));
        assert!(!far_needs_rebake(FAR_REBAKE_FRAMES - 1, 0.0, 0.0));
    }
}
