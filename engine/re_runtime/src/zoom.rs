//! `zoom` — the FOV-ZOOM "binoculars" toggle with a smooth radial REVEAL, plus
//! the live POST-FX cycle. Generic over the host's `live` state
//! ([`ZoomPlugin<S>`]) so any game can reuse it with any state enum: the host
//! wraps `ZoomPlugin { live: TheirState::Playing }`.
//!
//! # What this wires
//!
//! `re_present` emits two backend-neutral edges — [`re_present::ZoomToggle`] (`Z`)
//! and [`re_present::PostFxCycle`] (`G`). This module turns them into:
//!
//! 1. **FOV-ZOOM + REVEAL.** A toggle flips the raycaster between the player's
//!    *base* FOV (whatever `[`/`]` / settings last chose) and a narrow *zoom* FOV
//!    (default 30°, `TS_ZOOM_FOV`). The FOV value snaps instantly, but the picture
//!    does NOT: on each toggle we SNAPSHOT the current frame, and for a short
//!    transition (default 360 ms, `TS_ZOOM_MS`) we composite the new-FOV frame over
//!    the snapshot with [`re_render::composite_reveal_feathered`] — a disc growing
//!    from the centre when zooming IN ([`RevealMode::RadialOut`]) and a ring closing
//!    from the rim when zooming OUT ([`RevealMode::RadialIn`]). The radial wipe *is*
//!    the smooth transition.
//!
//! 2. **POST-FX cycle.** Each `G` press advances a small cycle
//!    (off → blur → grayscale → gameboy palette → off) and the active
//!    [`re_render::PostFxConfig`] is applied to the live frame.
//!
//! # System ordering (the contract)
//!
//! Everything runs in `Last`, `.before(PresentSet)`, gated to the host's `live` state:
//!
//! ```text
//! CompositeRenderPlugin (PostUpdate, writes Framebuffer)
//!   → reveal_transition_sys   (Last, before PresentSet)   — composites old↔new
//!   → postfx_apply_sys        (Last, before PresentSet, AFTER reveal)
//!   → PresentSet
//! ```
//!
//! `reveal_transition_sys` is explicitly `.before(postfx_apply_sys)` so the
//! post-fx grades the revealed image, never the raw new-FOV frame mid-wipe.
//!
//! # Purity
//!
//! No terminal, no rendering calls beyond `re_render`'s pure compositors. The state
//! machine ([`set_zoom_target_sys`]) is a pure function of the input edge + the live
//! `Fov`; the reveal/post-fx systems are pure functions of `(old, new, progress)` /
//! `(fb, cfg)`. All the per-cell math lives in `re_render` and is golden-tested
//! there; here we only own the *timeline* and the *config cycle*.

use bevy::prelude::*;
use re_core::Framebuffer;
use re_present::{PostFxCycle, ZoomToggle};
use re_render::{
    apply_posteffects, composite_reveal_feathered, BlurCfg, Fov, PaletteCfg, PostFxConfig,
    RenderForceDirty, RevealMode, DEFAULT_FEATHER,
};

/// Default narrow zoom FOV (degrees) when `TS_ZOOM_FOV` is unset. A clear "look
/// down the sight" narrowing from the ~60–65° base.
const DEFAULT_ZOOM_FOV: f32 = 30.0;

/// Default reveal transition duration (seconds) when `TS_ZOOM_MS` is unset.
/// 360 ms — slowed ~20% from the original 300 ms so the radial wipe reads as a
/// deliberate, unhurried reveal rather than a snap.
const DEFAULT_ZOOM_SECS: f32 = 0.36;

/// Lower / upper clamp for a configured zoom FOV, matching the raycaster's sane
/// `[`/`]` range (`re_render` casts any positive FOV, but this keeps a typo from
/// producing a degenerate pinhole or fisheye).
const ZOOM_FOV_MIN: f32 = 10.0;
const ZOOM_FOV_MAX: f32 = 120.0;

/// Feather-band width for the reveal frontier — a thin dissolving ribbon so the
/// wipe reads soft, not as a hard moving ring. Reuses `re_render`'s tuned default.
const DEFAULT_FEATHER_BAND: f32 = DEFAULT_FEATHER;

/// Static, env-derived zoom tuning: the narrow FOV, the transition duration, and
/// the reveal mode pair. Resolved once at startup so a per-frame system never hits
/// the environment.
#[derive(Resource, Debug, Clone, Copy)]
pub struct ZoomConfig {
    /// Narrow zoom FOV in degrees (`TS_ZOOM_FOV`, default 30°, clamped).
    pub zoom_fov_deg: f32,
    /// Reveal transition duration in seconds (`TS_ZOOM_MS` in milliseconds,
    /// default 360 ms). Clamped to a minimum so a 0 never divides-by-zero.
    pub duration_secs: f32,
    /// If true, use the spiral reveal pair (`Spiral` in / `SpiralIn` out) instead
    /// of the radial pair. Set by `TS_ZOOM_MODE=spiral`.
    pub spiral: bool,
}

impl Default for ZoomConfig {
    fn default() -> Self {
        ZoomConfig {
            zoom_fov_deg: DEFAULT_ZOOM_FOV,
            duration_secs: DEFAULT_ZOOM_SECS,
            spiral: false,
        }
    }
}

impl ZoomConfig {
    /// Resolve the zoom tuning from the environment, falling back to the defaults.
    ///
    /// * `TS_ZOOM_FOV` — narrow FOV in degrees (clamped to `[10,120]`).
    /// * `TS_ZOOM_MS`  — transition duration in milliseconds (≥ 1 ms enforced,
    ///   default 360 ms).
    /// * `TS_ZOOM_MODE=spiral` — use the spiral reveal pair.
    ///
    /// A malformed value is ignored (keeps the default), never panics.
    pub fn from_env() -> Self {
        let mut cfg = ZoomConfig::default();
        if let Ok(v) = std::env::var("TS_ZOOM_FOV") {
            if let Ok(f) = v.trim().parse::<f32>() {
                if f.is_finite() {
                    cfg.zoom_fov_deg = f.clamp(ZOOM_FOV_MIN, ZOOM_FOV_MAX);
                }
            }
        }
        if let Ok(v) = std::env::var("TS_ZOOM_MS") {
            if let Ok(ms) = v.trim().parse::<f32>() {
                if ms.is_finite() && ms >= 0.0 {
                    // Floor at 1 ms so `duration_secs` is always > 0.
                    cfg.duration_secs = (ms.max(1.0)) / 1000.0;
                }
            }
        }
        if let Ok(v) = std::env::var("TS_ZOOM_MODE") {
            if v.trim().eq_ignore_ascii_case("spiral") {
                cfg.spiral = true;
            }
        }
        cfg
    }

    /// The reveal mode for a transition in the given direction.
    ///
    /// * `narrowing == true` (zooming IN, FOV shrinking) → grow from the centre:
    ///   [`RevealMode::RadialOut`] / [`RevealMode::Spiral`].
    /// * `narrowing == false` (zooming OUT, FOV widening) → close from the rim:
    ///   [`RevealMode::RadialIn`] / [`RevealMode::SpiralIn`].
    #[inline]
    pub fn mode_for(&self, narrowing: bool) -> RevealMode {
        match (self.spiral, narrowing) {
            (false, true) => RevealMode::RadialOut,
            (false, false) => RevealMode::RadialIn,
            (true, true) => RevealMode::Spiral,
            (true, false) => RevealMode::SpiralIn,
        }
    }
}

/// The live FOV-zoom state machine.
///
/// `zoomed` is the LOGICAL target (false = base FOV, true = narrow zoom). `base_deg`
/// remembers the player's base FOV captured the instant we zoomed in, so zooming
/// out restores exactly that value (even if `[`/`]` changed the preset earlier).
/// `transition` is `Some` while the radial reveal is animating.
#[derive(Resource, Debug, Default)]
pub struct ZoomState {
    /// True when the current TARGET is the narrow zoom FOV.
    pub zoomed: bool,
    /// The base FOV (degrees) to restore on zoom-out — captured at zoom-in time.
    /// `None` until the first zoom-in (we read the live `Fov` then).
    pub base_deg: Option<f32>,
    /// The in-flight radial reveal, or `None` when idle.
    pub transition: Option<Transition>,
}

/// One in-flight reveal transition: the snapshot of the PRE-toggle frame, the
/// chosen reveal mode, and the `[0,1]` progress.
#[derive(Debug)]
pub struct Transition {
    /// The frame as it looked the instant the toggle fired (the reveal's OLD).
    pub old: Framebuffer,
    /// Centre-out (zoom-in) vs rim-in (zoom-out) reveal.
    pub mode: RevealMode,
    /// Reveal progress in `[0,1]`: `0` = all OLD, `1` = all NEW.
    pub progress: f32,
}

/// The live POST-FX cycle position. `G` advances it; the active config is applied
/// to the frame each live frame (a no-op at [`PostFxStage::Off`]).
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PostFxCycleState {
    /// Current cycle position.
    pub stage: PostFxStage,
}

/// The small, fixed POST-FX cycle the `G` key walks through. `Off` is the default
/// (a byte-identical no-op, so the shipped frames / golden hashes are unchanged
/// until the player opts in).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PostFxStage {
    /// No post-fx — the frame is presented untouched.
    #[default]
    Off,
    /// Radial focus-blur ([`BlurCfg::default`]).
    Blur,
    /// Grayscale (Rec.601 luma).
    Grayscale,
    /// Game Boy duotone palette.
    Gameboy,
}

impl PostFxStage {
    /// The next stage in the cycle (wraps `Gameboy` → `Off`).
    #[inline]
    pub fn next(self) -> Self {
        match self {
            PostFxStage::Off => PostFxStage::Blur,
            PostFxStage::Blur => PostFxStage::Grayscale,
            PostFxStage::Grayscale => PostFxStage::Gameboy,
            PostFxStage::Gameboy => PostFxStage::Off,
        }
    }

    /// The [`re_render::PostFxConfig`] this stage maps to. `Off` → the default
    /// no-op config.
    pub fn config(self) -> PostFxConfig {
        let mut cfg = PostFxConfig::default();
        match self {
            PostFxStage::Off => {}
            PostFxStage::Blur => cfg.blur = Some(BlurCfg::default()),
            PostFxStage::Grayscale => cfg.grayscale = true,
            PostFxStage::Gameboy => cfg.palette = Some(PaletteCfg::gameboy()),
        }
        cfg
    }

    /// A short HUD label for the active stage (empty for `Off`).
    #[inline]
    pub fn hud_label(self) -> &'static str {
        match self {
            PostFxStage::Off => "",
            PostFxStage::Blur => "BLUR",
            PostFxStage::Grayscale => "GRAY",
            PostFxStage::Gameboy => "GAMEBOY",
        }
    }
}

/// Plugin: registers the zoom + post-fx resources and the three systems, ordered
/// render → reveal → post-fx → present, gated to the host's `live` state.
///
/// The host supplies the state value that means "a level is live" — there is no
/// shared "generic Playing state" type here, only a value of the host's own
/// enum: `ZoomPlugin { live: GameState::Playing }`.
pub struct ZoomPlugin<S: States> {
    /// The host's state value meaning "a level is live". Every zoom/post-fx
    /// system is gated to `in_state(live)` (or its negation).
    pub live: S,
}

impl<S: States> Plugin for ZoomPlugin<S> {
    fn build(&self, app: &mut App) {
        let live = self.live.clone();
        app.insert_resource(ZoomConfig::from_env())
            .init_resource::<ZoomState>()
            .init_resource::<PostFxCycleState>()
            // The state machine runs in Update (it reads the input edge + sets the
            // live Fov + snapshots the frame at the START of the next render). It
            // does NOT need the just-rendered frame — it snapshots the PREVIOUS
            // frame (the one currently presented), which is what the player sees
            // when they press Z. Gated to `live`.
            // `force_render_during_transition_sys` runs AFTER `set_zoom_target_sys`
            // (which may START a transition this frame) so the force-redraw flag is set
            // for the SAME frame the toggle fires. It is NOT gated to `live`: when a
            // transition is in flight the flag is true; otherwise it CLEARS the flag —
            // and on a state exit the transition is gone, so the clear path still runs
            // and never leaves the cache jammed-dirty. (The reveal/recast systems below
            // are the `live`-gated ones.)
            .add_systems(
                Update,
                (
                    (set_zoom_target_sys, force_render_during_transition_sys)
                        .chain()
                        .run_if(in_state(live.clone())),
                    cycle_postfx_sys.run_if(in_state(live.clone())),
                    clear_force_render_when_not_playing_sys.run_if(not(in_state(live.clone()))),
                ),
            )
            // Reveal compositing + post-fx run in Last, after the composite render
            // produced this frame, before the present blit. reveal BEFORE post-fx
            // so the grade sees the revealed image. Both gated to `live`.
            .add_systems(
                Last,
                (reveal_transition_sys, postfx_apply_sys)
                    .chain()
                    .run_if(in_state(live))
                    .before(re_present::PresentSet),
            );
    }
}

/// `Update` (live): drain [`ZoomToggle`]. On the LAST toggle this frame (an odd
/// count flips the target; an even count is a no-op), capture the live base FOV if
/// needed, snapshot the currently-presented frame as the reveal OLD, snap the
/// raycaster `Fov` to the new target, and start the radial reveal.
///
/// The FOV change is what makes the NEXT composite render produce the NEW frame; the
/// snapshot taken here is the OLD frame the reveal wipes away.
fn set_zoom_target_sys(
    mut toggles: MessageReader<ZoomToggle>,
    cfg: Res<ZoomConfig>,
    mut state: ResMut<ZoomState>,
    mut fov: ResMut<Fov>,
    fb: Res<Framebuffer>,
) {
    // Collapse a burst of presses into a single net toggle (each press flips).
    let mut flips = 0u32;
    for _ in toggles.read() {
        flips += 1;
    }
    if flips.is_multiple_of(2) {
        // Even number of presses (incl. zero) → no net change this frame.
        return;
    }

    let narrowing = !state.zoomed; // about to zoom IN if we were not zoomed.
    if narrowing {
        // Zooming IN: remember the current base FOV so we can restore it later,
        // then snap to the narrow zoom FOV.
        state.base_deg = Some(fov.h_preset_deg);
        fov.h_preset_deg = cfg.zoom_fov_deg;
    } else {
        // Zooming OUT: restore the remembered base FOV (fall back to the live FOV
        // if we somehow never captured one — defensive, never panics).
        let base = state.base_deg.take().unwrap_or(fov.h_preset_deg);
        fov.h_preset_deg = base;
    }
    state.zoomed = narrowing;

    // Snapshot the frame the player currently sees as the reveal OLD, and start a
    // fresh transition at progress 0. If a transition was already mid-flight we
    // restart from the CURRENT frame (its OLD), so a rapid re-toggle wipes from
    // whatever is on screen rather than snapping.
    state.transition = Some(Transition {
        old: fb.clone(),
        mode: cfg.mode_for(narrowing),
        progress: 0.0,
    });
}

/// `Update` (live, AFTER [`set_zoom_target_sys`]): drive the renderer's
/// [`RenderForceDirty`] flag from whether a reveal is in flight.
///
/// THE STATIC-CAMERA FIX. The composite renderer skips the near cast (and the
/// rotation-pan) on an idle frame — so after the FOV snaps with a STATIC camera, the
/// live `Framebuffer` would keep the OLD-FOV geometry and the reveal would have
/// nothing new to wipe to. While a [`Transition`] is in flight we set `force = true`,
/// which makes [`re_render`]'s composite treat the frame as structurally dirty (a
/// fresh near cast at the live `Fov`, pan disabled) EVERY frame for the transition's
/// duration. When the transition ends the flag clears, restoring the idle economy.
///
/// Idempotent and self-clearing: it writes the flag every frame from the live
/// predicate, so a no-transition frame sets `false` and the cache is never left
/// jammed-on.
fn force_render_during_transition_sys(state: Res<ZoomState>, mut force: ResMut<RenderForceDirty>) {
    let want = state.transition.is_some();
    // Only touch the resource on an actual change so the renderer's `Changed`
    // tracking (if any) and the idle path stay quiet on a steady idle frame.
    if force.force != want {
        force.force = want;
    }
}

/// `Update` (NOT live): make sure the force-redraw flag is cleared whenever we are
/// outside the host's live state (menu/splash). A transition cannot survive a state
/// exit (the reveal systems are `live`-gated and the state is reset on teardown),
/// but clearing here is belt-and-suspenders so a menu frame can never be stuck
/// forcing a re-cast.
fn clear_force_render_when_not_playing_sys(mut force: ResMut<RenderForceDirty>) {
    if force.force {
        force.force = false;
    }
}

/// `Last` (live, before post-fx + present): if a reveal is in flight, composite
/// the freshly-rendered NEW frame over the snapshot OLD into a scratch buffer, copy
/// it back into the live [`Framebuffer`], advance progress, and clear the transition
/// at progress ≥ 1.
///
/// The scratch is a function-local buffer, kept DISTINCT from the live `fb` so the
/// `composite_reveal_feathered` no-alias contract holds (it reads `old`/`new` while
/// writing `out`).
fn reveal_transition_sys(
    time: Res<Time>,
    cfg: Res<ZoomConfig>,
    mut state: ResMut<ZoomState>,
    mut fb: ResMut<Framebuffer>,
) {
    let Some(t) = state.transition.as_mut() else {
        return;
    };

    // Dimension guard: if the frame resized mid-transition the snapshot no longer
    // matches; abandon the reveal cleanly (show the new frame) rather than skip.
    if t.old.cols() != fb.cols() || t.old.rows() != fb.rows() {
        state.transition = None;
        return;
    }

    // Composite OLD (snapshot) ↔ NEW (live fb) into a distinct scratch, then copy
    // back. `new` is the live frame; `out` must not alias it, hence the scratch.
    let mut scratch = Framebuffer::new(fb.cols(), fb.rows());
    composite_reveal_feathered(
        &t.old,
        &fb,
        &mut scratch,
        t.progress,
        t.mode,
        DEFAULT_FEATHER_BAND,
    );
    *fb = scratch;

    // Advance. `duration_secs` is floored > 0 in the config, so this never divides
    // by zero; a degenerate dt still makes monotonic progress.
    t.progress += time.delta_secs() / cfg.duration_secs.max(f32::EPSILON);
    if t.progress >= 1.0 {
        // Done — the live fb already holds the fully-revealed NEW frame next time
        // (progress ≥ 1 copies NEW), so just clear the transition.
        state.transition = None;
    }
}

/// `Update` (live): drain [`PostFxCycle`] and advance the cycle one step per
/// press (off → blur → gray → gameboy → off).
fn cycle_postfx_sys(mut presses: MessageReader<PostFxCycle>, mut state: ResMut<PostFxCycleState>) {
    let mut steps = 0u32;
    for _ in presses.read() {
        steps += 1;
    }
    for _ in 0..steps {
        state.stage = state.stage.next();
    }
}

/// `Last` (live, AFTER reveal, before present): apply the active post-fx config
/// to the live frame. A [`PostFxStage::Off`] config is a byte-identical no-op
/// ([`apply_posteffects`] early-returns), so the default path leaves the frame —
/// and every golden hash — unchanged.
///
/// SAFE IN-PLACE (no clean-snapshot needed, unlike `re_postfx`/`re_glitch`):
/// `composite_render_sys` (PostUpdate, before this `Last` system) resolves the live
/// frame UNCONDITIONALLY every tick — `fb.clear(); stack.resolve(&mut fb)` — so the
/// dirty-flag only skips the expensive near CAST, never the fb rebuild. This system
/// therefore always sees a freshly-resolved frame and a non-idempotent pass (the
/// focus blur averages neighbours) can NOT compound across idle ticks. If the
/// renderer is ever changed to dirty-skip the resolve itself, this must switch to the
/// clean-snapshot pattern the cinematic stack uses.
///
/// ORDERING vs `re_postfx`: this runs AFTER `PostfxDitherPlugin`'s "absolute-last"
/// dither (that plugin lives in PostUpdate). That is intentional — the player's
/// explicit G-cycle filter (B&W / gameboy palette / focus blur) is meant to be the
/// FINAL look, overriding the level's cinematic grade when both are active. It is not
/// a violation of the "dither last" contract: the chosen filter is itself the closing
/// recolour/quantise. When postfx is off (shipped default) there is no interaction.
fn postfx_apply_sys(state: Res<PostFxCycleState>, mut fb: ResMut<Framebuffer>) {
    if state.stage == PostFxStage::Off {
        return;
    }
    let cfg = state.stage.config();
    apply_posteffects(&mut fb, &cfg);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A local host-state stand-in, exactly the shape a host game supplies:
    /// `#[derive(States)] enum TestState { #[default] Idle, Live }`.
    #[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    enum TestState {
        #[default]
        Idle,
        Live,
    }

    /// Build a minimal headless app with just the zoom resources + systems, the
    /// messages they read, a `Framebuffer`, a `Fov`, and `Time`. Forced into
    /// `TestState::Live` so the `run_if` gate passes.
    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(bevy::state::app::StatesPlugin);
        app.init_state::<TestState>();
        app.add_message::<ZoomToggle>();
        app.add_message::<PostFxCycle>();
        app.insert_resource(ZoomConfig::default());
        app.init_resource::<ZoomState>();
        app.init_resource::<PostFxCycleState>();
        app.init_resource::<Time>();
        app.insert_resource(Fov::default());
        app.insert_resource(Framebuffer::new(40, 20));
        app.add_systems(
            Update,
            (set_zoom_target_sys, cycle_postfx_sys).run_if(in_state(TestState::Live)),
        );
        app.add_systems(
            Last,
            (reveal_transition_sys, postfx_apply_sys)
                .chain()
                .run_if(in_state(TestState::Live)),
        );
        // Enter Live.
        app.world_mut()
            .resource_mut::<NextState<TestState>>()
            .set(TestState::Live);
        app.update();
        app
    }

    /// Advance `Time` by a fixed dt and run one update (so reveal progress moves).
    fn step(app: &mut App, dt_secs: f32) {
        let mut time = app.world_mut().resource_mut::<Time>();
        time.advance_by(std::time::Duration::from_secs_f32(dt_secs));
        app.update();
    }

    fn send_zoom(app: &mut App) {
        app.world_mut()
            .resource_mut::<Messages<ZoomToggle>>()
            .write(ZoomToggle);
    }

    fn send_postfx(app: &mut App) {
        app.world_mut()
            .resource_mut::<Messages<PostFxCycle>>()
            .write(PostFxCycle);
    }

    #[test]
    fn mode_by_direction() {
        let radial = ZoomConfig {
            spiral: false,
            ..ZoomConfig::default()
        };
        assert_eq!(
            radial.mode_for(true),
            RevealMode::RadialOut,
            "zoom IN grows from centre"
        );
        assert_eq!(
            radial.mode_for(false),
            RevealMode::RadialIn,
            "zoom OUT closes from rim"
        );
        let spiral = ZoomConfig {
            spiral: true,
            ..ZoomConfig::default()
        };
        assert_eq!(spiral.mode_for(true), RevealMode::Spiral);
        assert_eq!(spiral.mode_for(false), RevealMode::SpiralIn);
    }

    #[test]
    fn toggle_sets_target_and_starts_transition() {
        let mut app = test_app();
        let base = app.world().resource::<Fov>().h_preset_deg;
        assert!(!app.world().resource::<ZoomState>().zoomed);

        // First Z: zoom IN. FOV snaps to the zoom FOV, a RadialOut transition starts.
        send_zoom(&mut app);
        app.update();
        {
            let st = app.world().resource::<ZoomState>();
            assert!(st.zoomed, "first toggle zooms in");
            assert_eq!(st.base_deg, Some(base), "base FOV captured for restore");
            let t = st.transition.as_ref().expect("transition started");
            assert_eq!(t.mode, RevealMode::RadialOut);
            assert_eq!(t.progress, 0.0);
        }
        assert_eq!(
            app.world().resource::<Fov>().h_preset_deg,
            ZoomConfig::default().zoom_fov_deg,
            "FOV snapped to the narrow zoom value"
        );

        // Let the transition finish so the next toggle starts clean.
        step(&mut app, 1.0);
        assert!(app.world().resource::<ZoomState>().transition.is_none());

        // Second Z: zoom OUT restores the base FOV with a RadialIn transition.
        // Advance a SMALL dt (well under the duration) so the freshly-started
        // reveal is mid-flight when we inspect it — a large lingering dt would
        // complete and clear it in the same frame.
        send_zoom(&mut app);
        step(&mut app, ZoomConfig::default().duration_secs * 0.25);
        {
            let st = app.world().resource::<ZoomState>();
            assert!(!st.zoomed, "second toggle zooms out");
            assert_eq!(st.transition.as_ref().unwrap().mode, RevealMode::RadialIn);
        }
        assert_eq!(
            app.world().resource::<Fov>().h_preset_deg,
            base,
            "FOV restored to the captured base"
        );
    }

    #[test]
    fn even_toggle_burst_is_a_noop() {
        let mut app = test_app();
        let base = app.world().resource::<Fov>().h_preset_deg;
        // Two presses in one frame cancel: no net zoom, no transition.
        send_zoom(&mut app);
        send_zoom(&mut app);
        app.update();
        let st = app.world().resource::<ZoomState>();
        assert!(!st.zoomed, "even press count is a no-op");
        assert!(st.transition.is_none(), "no transition for an even burst");
        assert_eq!(app.world().resource::<Fov>().h_preset_deg, base);
    }

    #[test]
    fn transition_advances_and_ends() {
        let mut app = test_app();
        send_zoom(&mut app);
        app.update();
        // Half the default duration → progress ~0.5, still transitioning.
        step(&mut app, ZoomConfig::default().duration_secs * 0.5);
        {
            let st = app.world().resource::<ZoomState>();
            let p = st
                .transition
                .as_ref()
                .expect("still transitioning")
                .progress;
            assert!(
                p > 0.0 && p < 1.0,
                "progress advanced to mid-flight, got {p}"
            );
        }
        // Past the full duration → cleared.
        step(&mut app, ZoomConfig::default().duration_secs);
        assert!(
            app.world().resource::<ZoomState>().transition.is_none(),
            "transition clears at progress >= 1"
        );
    }

    #[test]
    fn postfx_cycle_advances() {
        let mut app = test_app();
        assert_eq!(
            app.world().resource::<PostFxCycleState>().stage,
            PostFxStage::Off
        );
        for expected in [
            PostFxStage::Blur,
            PostFxStage::Grayscale,
            PostFxStage::Gameboy,
            PostFxStage::Off,
        ] {
            send_postfx(&mut app);
            app.update();
            assert_eq!(app.world().resource::<PostFxCycleState>().stage, expected);
        }
    }

    #[test]
    fn idle_frame_leaves_framebuffer_unchanged() {
        // No zoom, no post-fx: the reveal + post-fx systems must be byte-identical
        // no-ops (the default path that keeps the golden frames stable).
        let mut app = test_app();
        // Seed a recognisable frame.
        {
            let mut fb = app.world_mut().resource_mut::<Framebuffer>();
            for row in 0..fb.rows() {
                for col in 0..fb.cols() {
                    fb.set(
                        col,
                        row,
                        re_core::Cell {
                            glyph: u32::from('#'),
                            fg: 0x0012_3456,
                            bg: 0x0065_4321,
                            ..re_core::Cell::default()
                        },
                    );
                }
            }
        }
        let before = app.world().resource::<Framebuffer>().cells().to_vec();
        step(&mut app, 0.1);
        let after = app.world().resource::<Framebuffer>().cells().to_vec();
        assert_eq!(
            before, after,
            "idle (no zoom, no post-fx) must not touch the frame"
        );
    }

    /// Seed every cell of the live framebuffer with a recognisable coloured glyph.
    fn fill_frame(app: &mut App) {
        let mut fb = app.world_mut().resource_mut::<Framebuffer>();
        for row in 0..fb.rows() {
            for col in 0..fb.cols() {
                fb.set(
                    col,
                    row,
                    re_core::Cell {
                        glyph: u32::from('@'),
                        fg: 0x00C0_4020,
                        bg: 0x0010_2030,
                        ..re_core::Cell::default()
                    },
                );
            }
        }
    }

    #[test]
    fn postfx_off_stage_is_a_noop_but_active_stage_changes_frame() {
        // Reference: a seeded frame run through ONE Off-stage frame is unchanged.
        let mut app = test_app();
        fill_frame(&mut app);
        let seeded = app.world().resource::<Framebuffer>().cells().to_vec();
        step(&mut app, 0.01);
        assert_eq!(
            app.world().resource::<Framebuffer>().cells(),
            seeded.as_slice(),
            "Off stage must leave the frame byte-identical"
        );

        // Now advance to grayscale, RE-SEED to the same clean frame, run ONE frame,
        // and confirm the grayscale pass changed it (and equalised fg channels).
        send_postfx(&mut app); // Off → Blur
        send_postfx(&mut app); // Blur → Grayscale
        app.update(); // process the two cycle presses (no post-fx applied to seed yet)
        fill_frame(&mut app);
        let clean = app.world().resource::<Framebuffer>().cells().to_vec();
        step(&mut app, 0.01);
        let after = app.world().resource::<Framebuffer>().cells().to_vec();
        assert_ne!(clean, after, "an active post-fx stage must alter the frame");
        // Grayscale: fg channels must be equal on every cell.
        let c = app
            .world()
            .resource::<Framebuffer>()
            .get(0, 0)
            .copied()
            .unwrap();
        let (r, g, b) = ((c.fg >> 16) & 0xFF, (c.fg >> 8) & 0xFF, c.fg & 0xFF);
        assert!(r == g && g == b, "grayscale stage equalises fg channels");
    }

    // ───────────── static-camera FOV-zoom reveal (the regression) ─────────────

    /// Build a headless app with the REAL renderer (`CompositeRenderPlugin`) + the
    /// REAL generic `ZoomPlugin` (zoom target + force-redraw + reveal), a `test_room`
    /// map, a STATIC camera entity, and a `Framebuffer`. This is the end-to-end
    /// harness that reproduces the bug: with a static camera, the idle dirty-flag
    /// used to skip the new-FOV cast so the reveal had nothing to wipe to. Forced
    /// into `TestState::Live`.
    fn render_app() -> App {
        use re_render::CompositeRenderPlugin;
        use re_world::{test_room, CameraPose};

        let mut app = App::new();
        // TaskPoolPlugin: the composite's parallel iteration needs the task pools.
        app.add_plugins(bevy::app::TaskPoolPlugin::default());
        app.add_plugins(bevy::state::app::StatesPlugin);
        app.init_state::<TestState>();
        app.add_message::<ZoomToggle>();
        app.add_message::<PostFxCycle>();
        // MANUAL `Time` (NOT `TimePlugin`): the bevy time-system overwrites `Time`
        // with the real wall-clock each frame, which would clobber the fixed-dt
        // `advance_by` in `step` and make the reveal's per-frame progress
        // non-deterministic. A manually-inserted `Time<()>` honours `advance_by`, so
        // the composite + reveal both read the exact dt this test drives.
        app.init_resource::<Time>();
        // A real, modest framebuffer the composite writes into.
        app.insert_resource(Framebuffer::new(48, 24));
        // The world the renderer casts against + a STATIC camera entity.
        app.insert_resource(test_room());
        app.world_mut().spawn(CameraPose::spawn_in_test_room());
        // The renderer (PostUpdate) + the zoom systems (Update + Last).
        app.add_plugins(CompositeRenderPlugin);
        app.add_plugins(ZoomPlugin {
            live: TestState::Live,
        });
        // Enter Live and prime the renderer (first frame does the cold far bake +
        // the first near cast → a real base-FOV frame).
        app.world_mut()
            .resource_mut::<NextState<TestState>>()
            .set(TestState::Live);
        app.update();
        app.update();
        app
    }

    #[test]
    fn static_camera_zoom_reveal_animates_the_framebuffer() {
        // THE REGRESSION: pressing Z with a STATIC camera must visibly animate. We
        // confirm the `Framebuffer` CHANGES across consecutive transition frames even
        // though the camera never moves — proving the idle dirty-flag no longer skips
        // the new-FOV render/reveal.
        let mut app = render_app();

        // The reveal needs a non-uniform base frame to wipe (a blank frame would
        // composite blank→blank). The test_room cast fills real geometry, so the base
        // frame is already non-uniform; assert it to keep the test honest.
        let base = app.world().resource::<Framebuffer>().cells().to_vec();
        let uniform = base.iter().all(|c| *c == base[0]);
        assert!(
            !uniform,
            "base frame should hold real geometry to reveal against"
        );

        // Press Z: zoom IN. The FOV snaps to the narrow value and a RadialOut reveal
        // starts; with the fix the force-redraw flag goes true for the transition.
        send_zoom(&mut app);
        // Run the Update (zoom target + force flag set) without advancing time yet.
        app.update();
        assert!(
            app.world().resource::<ZoomState>().transition.is_some(),
            "Z must start a transition"
        );
        assert!(
            app.world().resource::<re_render::RenderForceDirty>().force,
            "an in-flight transition must force the renderer dirty (defeat the idle cache)"
        );

        // Now sample three consecutive controlled-dt frames (camera untouched) and
        // require the framebuffer to differ each step — the reveal frontier advancing
        // the disc through the frame. With the manual `Time`, each `step` advances the
        // reveal by exactly `dt/duration` of progress, so the composited frame moves.
        let dt = ZoomConfig::default().duration_secs * 0.2;
        step(&mut app, dt);
        let f1 = app.world().resource::<Framebuffer>().cells().to_vec();
        step(&mut app, dt);
        let f2 = app.world().resource::<Framebuffer>().cells().to_vec();
        step(&mut app, dt);
        let f3 = app.world().resource::<Framebuffer>().cells().to_vec();

        assert_ne!(
            f1, f2,
            "reveal frame 2 must differ from frame 1 (static camera)"
        );
        assert_ne!(
            f2, f3,
            "reveal frame 3 must differ from frame 2 (static camera)"
        );

        // Drive the transition to completion; the force flag must then clear so the
        // idle economy is restored.
        step(&mut app, ZoomConfig::default().duration_secs);
        // One more Update so the Update-stage force system observes `transition == None`.
        app.update();
        assert!(
            app.world().resource::<ZoomState>().transition.is_none(),
            "transition must complete"
        );
        assert!(
            !app.world().resource::<re_render::RenderForceDirty>().force,
            "force-redraw must clear once the transition ends (idle economy restored)"
        );
    }

    #[test]
    fn idle_static_frame_does_not_force_redraw() {
        // The idle economy must be intact: with NO transition, the force flag stays
        // false so the composite keeps caching a static idle frame (no regression to
        // the dirty-flag optimization).
        let mut app = render_app();
        assert!(
            !app.world().resource::<re_render::RenderForceDirty>().force,
            "no transition ⇒ no forced redraw"
        );
        // A few idle frames must keep it false.
        for _ in 0..3 {
            step(&mut app, 0.05);
            assert!(
                !app.world().resource::<re_render::RenderForceDirty>().force,
                "idle frames must never force a redraw"
            );
        }
    }
}
