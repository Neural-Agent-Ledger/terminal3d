//! `re_core::perf` — shared timing + UI-gate resources.
//!
//! Pure data resources that several layers read and write but none of them want
//! to define twice. They are engine-agnostic (`re_core` has no terminal, no
//! `bevy_render`) and carry no behaviour beyond a tiny EMA fold:
//!
//! * [`FrameTimingMetrics`] — per-stage render timing for the current frame, in
//!   microseconds. `re_render` writes the composite-stage times; `re_present`
//!   writes the present time; the perf overlay reads it.
//! * [`PerfOverlay`] — the on-screen timing overlay toggle (F3). Off by default.
//! * [`InputGate`] — a state-aware gate so `re_present` can freeze camera motion
//!   during pause/menus WITHOUT importing `GameState` (which lives in `re_game`).
//!
//! The `Resource` derive comes from `bevy::prelude::Resource` to match the rest
//! of `re_core` (see `Framebuffer` in `lib.rs`) — siblings should import the same
//! path, not `bevy_ecs::prelude::Resource`, so there is one derive source.

#[cfg(feature = "reflect")]
use bevy::prelude::ReflectResource;
use bevy::prelude::Resource;

/// Per-stage render timing for the CURRENT frame, in microseconds (`f32`).
///
/// Written each frame: `re_render` fills the composite-stage fields
/// (`far_bake_us` … `resolve_us`) plus `render_total_us`, and `re_present` fills
/// `present_us` and folds the wall-clock delta via [`record_frame`]. The perf
/// overlay only reads it. All fields default to `0`, so a frame that measures
/// nothing reads as zero rather than garbage.
///
/// [`record_frame`]: FrameTimingMetrics::record_frame
#[derive(Resource, Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "reflect", derive(bevy::reflect::Reflect))]
#[cfg_attr(feature = "reflect", reflect(Resource))]
pub struct FrameTimingMetrics {
    /// Time spent baking the far/horizon cache, in microseconds.
    pub far_bake_us: f32,
    /// Time spent on the near DDA cast, in microseconds.
    pub near_cast_us: f32,
    /// Time spent on sub-cell refinement, in microseconds.
    pub subcell_us: f32,
    /// Time spent compositing emitters, in microseconds.
    pub emitter_us: f32,
    /// Time spent resolving cells to glyph/colour, in microseconds.
    pub resolve_us: f32,
    /// Time spent presenting the framebuffer to the terminal, in microseconds.
    pub present_us: f32,
    /// Sum of the composite stages measured in `re_render`, in microseconds.
    pub render_total_us: f32,
    /// Wall-clock frame delta, in microseconds.
    pub frame_dt_us: f32,
    /// Exponential moving average of instantaneous FPS.
    pub fps_ema: f32,
    /// Total frames folded in via [`record_frame`](Self::record_frame).
    pub frame_count: u64,
}

impl FrameTimingMetrics {
    /// EMA smoothing factor: the weight given to the newest sample. ~0.1 keeps
    /// the FPS readout responsive but stable.
    const FPS_EMA_ALPHA: f32 = 0.1;

    /// Fold a new wall-clock frame delta (in SECONDS) into the timing.
    ///
    /// Sets [`frame_dt_us`](Self::frame_dt_us), updates the [`fps_ema`](Self::fps_ema)
    /// from the instantaneous FPS (`1 / dt`), and increments
    /// [`frame_count`](Self::frame_count). A non-positive `dt_secs` is a safe
    /// no-op (no division, no NaN, no counter change) — callers don't have to
    /// guard the first frame or a stalled clock.
    #[inline]
    pub fn record_frame(&mut self, dt_secs: f32) {
        // Accept only a finite, strictly-positive delta. `is_sign_positive`
        // alone would let +0.0 through, and a bare `dt <= 0.0` lets NaN through,
        // so require finiteness AND `> 0` explicitly. This rejects dt <= 0, NaN,
        // and infinities, leaving everything unchanged on a bad sample.
        if !dt_secs.is_finite() || dt_secs <= 0.0 {
            return;
        }
        self.frame_dt_us = dt_secs * 1.0e6;
        let fps_inst = 1.0 / dt_secs;
        if self.fps_ema <= 0.0 {
            // Seed the EMA on the first real sample so it converges immediately
            // instead of crawling up from zero.
            self.fps_ema = fps_inst;
        } else {
            self.fps_ema += Self::FPS_EMA_ALPHA * (fps_inst - self.fps_ema);
        }
        self.frame_count += 1;
    }
}

/// On-screen perf-overlay toggle (bound to F3 by the input layer).
///
/// MUST default to NOT visible — the overlay is opt-in. `re_present`/`re_game`
/// flip [`visible`](Self::visible) on the F3 action; the overlay draw system
/// `run_if`s on it.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct PerfOverlay {
    /// Whether the timing overlay is currently drawn. Default `false`.
    pub visible: bool,
}

/// State-aware input gate that lets `re_present` freeze camera motion during
/// pause/menus WITHOUT depending on `GameState`.
///
/// `GameState` lives in `re_game`, which sits ABOVE both `re_core` and
/// `re_present` in the dependency graph — so `re_present`'s movement systems
/// cannot `run_if(in_state(GameState::Playing))` without an illegal upward
/// dependency. Instead `re_game` owns the policy and writes it here:
/// it sets `camera_motion = true` on entering [`Playing`], and `false` on every
/// other state (Pause / menus / Won). `re_present`'s movement systems then
/// `run_if` on this gate, depending only on `re_core`.
///
/// The default is deliberately `false` (fail-safe): no camera motion until
/// `re_game` explicitly enables it on entering gameplay, so a missing or
/// mis-ordered setup never leaves the camera drivable in a menu.
///
/// [`Playing`]: # "GameState::Playing in re_game"
#[derive(Resource, Debug, Clone, Copy)]
pub struct InputGate {
    /// Whether camera-motion input is currently allowed. Default `false`.
    pub camera_motion: bool,
}

impl Default for InputGate {
    /// Fail-safe default: camera motion is OFF until `re_game` enables it on
    /// entering gameplay.
    fn default() -> Self {
        Self {
            camera_motion: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_frame_positive_dt_updates_ema_and_count() {
        let mut m = FrameTimingMetrics::default();
        assert_eq!(m.frame_count, 0);
        assert_eq!(m.fps_ema, 0.0);

        // 16.667 ms ≈ 60 FPS.
        m.record_frame(1.0 / 60.0);
        assert_eq!(m.frame_count, 1);
        assert!(m.fps_ema > 0.0, "fps_ema must become positive");
        assert!(m.frame_dt_us > 0.0, "frame_dt_us must be set");
        // First sample seeds the EMA exactly at the instantaneous FPS.
        assert!((m.fps_ema - 60.0).abs() < 1e-2);

        // A second frame folds in and bumps the count.
        m.record_frame(1.0 / 30.0);
        assert_eq!(m.frame_count, 2);
        // EMA stays between the two samples.
        assert!(m.fps_ema < 60.0 && m.fps_ema > 30.0);
    }

    #[test]
    fn record_frame_nonpositive_dt_is_noop() {
        let mut m = FrameTimingMetrics::default();
        for dt in [0.0_f32, -1.0, f32::NAN] {
            m.record_frame(dt);
            assert_eq!(m.frame_count, 0, "non-positive dt must not count");
            assert_eq!(m.fps_ema, 0.0, "non-positive dt must not move ema");
            assert_eq!(m.frame_dt_us, 0.0, "non-positive dt must not set dt");
        }
    }

    #[test]
    fn perf_overlay_defaults_hidden() {
        assert!(!PerfOverlay::default().visible);
    }

    #[test]
    fn input_gate_defaults_to_no_motion() {
        // Fail-safe: motion is off until re_game enables it.
        assert!(!InputGate::default().camera_motion);
    }

    #[test]
    #[cfg(feature = "reflect")]
    fn frame_timing_is_reflectable() {
        fn assert_reflect<T: bevy::reflect::Reflect>() {}
        assert_reflect::<FrameTimingMetrics>();
    }
}
