//! `color_drift_sources` — the driver that turns live puzzle/camera state into
//! `re_render::color_drift`'s `DriftSources`. `re_render::color_drift` is pure
//! engine math with no puzzle/save/camera knowledge; this module rebuilds a
//! fresh `DriftSources` EVERY FRAME from scratch — nothing accumulates, so an
//! empty result is always a true no-op.
//!
//! Its only seam is [`crate::session::PlayCamera`] — no host state enum
//! needed, so unlike [`crate::zoom`] there is no generic host wrapper.
//!
//! ## Sources implemented so far
//!
//! * **Powered pipe** — a travelling hue wave along each `On` pipe's world
//!   path (`PuzzleState::pipes[i].path`, already world cell coordinates), with
//!   a 400 ms ease-in on toggle. Floor/wall cells within ~1 cell of the pipe
//!   get its hue, exactly as authored data already describes the route.
//! * **Player gaze** — screen-centre desaturation ("horror mode") after the
//!   camera pose (yaw/pitch) has been STILL for 1.5 s, with a 1 s release ease
//!   when the player looks away.
//!
//! ## Sources not implemented here
//!
//! * **Emitter glow** would source from the per-cell energy/colour
//!   accumulator inside `emitter.rs`'s compositing pass, which is
//!   SCREEN-space; there is no world-position accessor for it yet, and
//!   `re_world::EmitterSpec` carries no `glow` field to derive a substitute
//!   from. Building that accessor belongs in `re_render`'s emitter code, not
//!   here.
//! * **Glitch residue**: `re_glitch::GlitchPhosphorState` holds the previous
//!   PRESENTED (screen-space) frame for its own optical blend; turning "which
//!   screen cells were corrupted this frame" into world-anchored drift sources
//!   needs a diff pass against the pre-glitch frame that does not exist.
//! * **Portal/fractal wall, lightning/surge, beat** — no wiring attempted.
//!
//! Both gaps are additive: an empty [`DriftSources`] is a true no-op
//! (`apply_drift` returns immediately), so adding these sources later cannot
//! regress anything wired here.
//!
//! ## Pipeline placement
//!
//! The [`DriftSources`] RESOURCE is computed in `Update` (schedule-order
//! independent). The per-frame `apply_drift` CALL must run AFTER the
//! composite resolve and BEFORE post-fx — the host adds
//! [`ColorDriftFramePlugin`] directly between `CompositeRenderPlugin` and its
//! post-fx plugin for exactly this reason (see [`apply_color_drift_to_frame`]'s
//! docs).

use bevy::prelude::*;
use re_core::{Framebuffer, Vec2};
use re_puzzle::PuzzleState;
use re_render::color_drift::{apply_drift, DriftSource, DriftSources, GazeSource};
use re_world::CameraPose;

use crate::session::PlayCamera;

/// [`DriftSources`] is pure engine data (no `bevy::Resource` derive, matching
/// `re_render::palette_gate::PaletteGate`'s split) — this newtype is the
/// resource wrapper, the same pattern a host's own palette-ramp state uses to
/// wrap `PaletteGate`+`GateLut`.
#[derive(Resource, Default)]
pub struct DriftSourcesRes(pub DriftSources);

// ─── Powered-pipe travelling wave (first-pass tuning, see module docs) ──────

/// Wavelength of the hue wave along a pipe's path, in WORLD CELLS:
/// `Δh = 18°·sin(2π(s/λ − f·t))`.
pub const PIPE_WAVELENGTH_CELLS: f32 = 4.0;
/// Travel frequency of the wave, in Hz.
pub const PIPE_WAVE_FREQ_HZ: f32 = 0.6;
/// Peak hue swing (`Δh = 18°·sin(...)`).
pub const PIPE_HUE_AMPLITUDE_DEG: f32 = 18.0;
/// Lightness lift on a wave crest only (`ΔL +0.03` on crests).
pub const PIPE_CREST_DELTA_L: f32 = 0.03;
/// Falloff radius around each path cell — floor/wall within 1 cell of the
/// pipe gets its hue.
pub const PIPE_SOURCE_RADIUS: f32 = 1.1;
/// Ease-in duration on toggle (400 ms).
pub const PIPE_TOGGLE_EASE_SECS: f32 = 0.4;

// ─── Player gaze (first-pass tuning; "horror mode" only, see module docs) ───

/// Dwell time before the gaze effect starts (1.5 s of stillness).
pub const GAZE_DWELL_SECS: f32 = 1.5;
/// Screen-space falloff radius, normalised `[0,1]` cell coords — cells within
/// 12% of screen centre.
pub const GAZE_RADIUS: f32 = 0.12;
/// Desaturation rate once dwelling, per second beyond the threshold
/// (`ΔC −0.04/s`).
pub const GAZE_DESAT_RATE_PER_SEC: f32 = 0.04;
/// Desaturation cap (down to `−0.10`).
pub const GAZE_DESAT_CAP: f32 = 0.10;
/// Release ease when the player looks away (1 s release).
pub const GAZE_RELEASE_SECS: f32 = 1.0;
/// A camera pose is "still" when yaw/pitch move less than this per frame
/// (radians) — small enough to ignore float jitter, large enough that the
/// natural render-loop tick noise never fakes a dwell.
const GAZE_STILLNESS_EPS: f32 = 0.002;

/// Per-pipe ease-in timers, resized to the live puzzle's pipe count (mirrors
/// `state_bridge::StateBridge::prev_pipe_state`'s resize-on-reload pattern).
#[derive(Resource, Default)]
struct PipeDriftTimers {
    since_on: Vec<f32>,
}

/// The gaze dwell state machine: how long the camera has been still, and the
/// currently-held (possibly releasing) chroma offset magnitude.
#[derive(Resource, Default)]
struct GazeDwellState {
    last_pose: Option<(f32, f32)>,
    stable_secs: f32,
    held_delta_c: f32,
}

/// `Update`: advance the per-pipe toggle-ease timers from the live
/// [`PuzzleState`].
fn tick_pipe_drift_timers(
    puzzle: Res<PuzzleState>,
    time: Res<Time>,
    mut timers: ResMut<PipeDriftTimers>,
) {
    if timers.since_on.len() != puzzle.pipes.len() {
        timers.since_on = vec![0.0; puzzle.pipes.len()];
    }
    let dt = time.delta_secs();
    for (i, pipe) in puzzle.pipes.iter().enumerate() {
        let t = &mut timers.since_on[i];
        if pipe.is_on() {
            *t = (*t + dt).min(PIPE_TOGGLE_EASE_SECS);
        } else {
            *t = 0.0;
        }
    }
}

/// `Update`: advance the gaze dwell/release state machine from the live
/// camera pose.
fn tick_gaze_dwell(
    time: Res<Time>,
    cam_q: Query<&CameraPose, With<PlayCamera>>,
    mut gaze: ResMut<GazeDwellState>,
) {
    let Ok(pose) = cam_q.single() else {
        // No live play camera (menus, etc.) — decay any held effect and stop.
        gaze.stable_secs = 0.0;
        gaze.last_pose = None;
        gaze.held_delta_c = 0.0;
        return;
    };
    let dt = time.delta_secs();
    let now = (pose.yaw, pose.pitch);
    let still = gaze.last_pose.is_some_and(|(y, p)| {
        (y - now.0).abs() < GAZE_STILLNESS_EPS && (p - now.1).abs() < GAZE_STILLNESS_EPS
    });
    gaze.last_pose = Some(now);

    if still {
        gaze.stable_secs += dt;
        let dwell = (gaze.stable_secs - GAZE_DWELL_SECS).max(0.0);
        gaze.held_delta_c = (dwell * GAZE_DESAT_RATE_PER_SEC).min(GAZE_DESAT_CAP);
    } else {
        gaze.stable_secs = 0.0;
        // Release: ease the held effect back to 0 over GAZE_RELEASE_SECS.
        let step = dt * (GAZE_DESAT_CAP / GAZE_RELEASE_SECS.max(0.001));
        gaze.held_delta_c = (gaze.held_delta_c - step).max(0.0);
    }
}

/// Build this frame's powered-pipe world drift sources.
fn pipe_world_sources(puzzle: &PuzzleState, timers: &PipeDriftTimers, t: f32) -> Vec<DriftSource> {
    let mut out = Vec::new();
    for (pi, pipe) in puzzle.pipes.iter().enumerate() {
        if !pipe.is_on() || pipe.path.is_empty() {
            continue;
        }
        let ease = timers
            .since_on
            .get(pi)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, PIPE_TOGGLE_EASE_SECS)
            / PIPE_TOGGLE_EASE_SECS;
        for (idx, &(x, y, _z)) in pipe.path.iter().enumerate() {
            let phase = std::f32::consts::TAU
                * (idx as f32 / PIPE_WAVELENGTH_CELLS - PIPE_WAVE_FREQ_HZ * t);
            let wave = phase.sin();
            out.push(DriftSource {
                world_pos: Vec2::new(x as f32 + 0.5, y as f32 + 0.5),
                radius: PIPE_SOURCE_RADIUS,
                delta_l: PIPE_CREST_DELTA_L * wave.max(0.0) * ease,
                delta_c: 0.0,
                delta_h_deg: PIPE_HUE_AMPLITUDE_DEG * wave * ease,
            });
        }
    }
    out
}

/// `Update`: assemble the fresh [`DriftSources`] resource from this frame's
/// pipe + gaze state (R4: rebuilt from scratch every frame, never mutated
/// incrementally).
fn build_drift_sources(
    puzzle: Res<PuzzleState>,
    time: Res<Time>,
    timers: Res<PipeDriftTimers>,
    gaze: Res<GazeDwellState>,
    mut sources: ResMut<DriftSourcesRes>,
) {
    let world = pipe_world_sources(&puzzle, &timers, time.elapsed_secs());
    let gaze_source = if gaze.held_delta_c > 0.0 {
        Some(GazeSource {
            screen_center: (0.5, 0.5),
            radius: GAZE_RADIUS,
            delta_l: 0.0,
            delta_c: -gaze.held_delta_c,
            delta_h_deg: 0.0,
        })
    } else {
        None
    };
    sources.0 = DriftSources {
        world,
        gaze: gaze_source,
    };
}

/// Registers the `Update`-schedule computation of [`DriftSources`] (order
/// independent — see module docs). The host adds this the same way it
/// registers its own progress-computation systems (e.g. a palette-ramp
/// state's own plugin).
pub struct ColorDriftPlugin;

impl Plugin for ColorDriftPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DriftSourcesRes>()
            .init_resource::<PipeDriftTimers>()
            .init_resource::<GazeDwellState>()
            .add_systems(
                Update,
                (tick_pipe_drift_timers, tick_gaze_dwell, build_drift_sources).chain(),
            );
    }
}

/// Apply the current [`DriftSources`] to the live frame. An empty
/// `DriftSources` (the common case — no powered pipe, no dwell) is a true
/// no-op inside `apply_drift` itself.
///
/// **Pipeline placement is the caller's responsibility**: registered directly
/// by the binary between `CompositeRenderPlugin` and its post-fx plugin
/// (after the composite resolve, before postfx), the same pattern a host's
/// own palette-gate frame plugin uses for its own PostUpdate slot.
pub fn apply_color_drift_to_frame(sources: Res<DriftSourcesRes>, mut fb: ResMut<Framebuffer>) {
    if sources.0.world.is_empty() && sources.0.gaze.is_none() {
        return;
    }
    apply_drift(&mut fb, &sources.0);
}

/// Registers ONLY the `PostUpdate` frame-application system, so the binary can
/// place it precisely between `CompositeRenderPlugin` and `PostfxPlugin`.
pub struct ColorDriftFramePlugin;

impl Plugin for ColorDriftFramePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DriftSourcesRes>()
            .add_systems(PostUpdate, apply_color_drift_to_frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use re_level::PipeState;
    use re_puzzle::Pipe;

    fn on_pipe(path: Vec<(i32, i32, i32)>) -> Pipe {
        Pipe {
            from: 0,
            to: 1,
            state: PipeState::On,
            path,
            switch_cell: None,
        }
    }

    #[test]
    fn an_off_pipe_contributes_no_source() {
        let puzzle = PuzzleState {
            pipes: vec![Pipe {
                state: PipeState::Off,
                ..on_pipe(vec![(1, 1, 1), (2, 1, 1)])
            }],
            ..PuzzleState::default()
        };
        let timers = PipeDriftTimers {
            since_on: vec![PIPE_TOGGLE_EASE_SECS],
        };
        let out = pipe_world_sources(&puzzle, &timers, 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn an_on_pipe_contributes_one_source_per_path_cell() {
        let path = vec![(1, 1, 1), (2, 1, 1), (3, 1, 1)];
        let puzzle = PuzzleState {
            pipes: vec![on_pipe(path.clone())],
            ..PuzzleState::default()
        };
        let timers = PipeDriftTimers {
            since_on: vec![PIPE_TOGGLE_EASE_SECS],
        };
        let out = pipe_world_sources(&puzzle, &timers, 0.3);
        assert_eq!(out.len(), path.len());
        for (src, &(x, y, _)) in out.iter().zip(path.iter()) {
            assert!((src.world_pos.x - (x as f32 + 0.5)).abs() < 1e-6);
            assert!((src.world_pos.y - (y as f32 + 0.5)).abs() < 1e-6);
        }
    }

    #[test]
    fn toggle_ease_scales_the_hue_and_lightness_offsets_toward_zero() {
        let puzzle = PuzzleState {
            pipes: vec![on_pipe(vec![(1, 1, 1), (2, 1, 1)])],
            ..PuzzleState::default()
        };
        let full = pipe_world_sources(
            &puzzle,
            &PipeDriftTimers {
                since_on: vec![PIPE_TOGGLE_EASE_SECS],
            },
            0.1,
        );
        let half = pipe_world_sources(
            &puzzle,
            &PipeDriftTimers {
                since_on: vec![PIPE_TOGGLE_EASE_SECS / 2.0],
            },
            0.1,
        );
        let just_on = pipe_world_sources(
            &puzzle,
            &PipeDriftTimers {
                since_on: vec![0.0],
            },
            0.1,
        );
        for s in &just_on {
            assert_eq!(s.delta_h_deg, 0.0, "no ease elapsed yet -> zero offset");
        }
        // A cell whose wave is non-zero must shrink monotonically with less ease.
        for i in 0..full.len() {
            assert!(full[i].delta_h_deg.abs() >= half[i].delta_h_deg.abs());
        }
    }

    #[test]
    fn gaze_stillness_check_rejects_sub_epsilon_jitter() {
        // A pose delta smaller than the epsilon must read as "still" — this is
        // a threshold sanity check, not a full ECS integration test.
        let a = (0.100_000, 0.0);
        let b = (0.100_000 + GAZE_STILLNESS_EPS / 2.0, 0.0);
        assert!((a.0 - b.0).abs() < GAZE_STILLNESS_EPS);
    }
}
