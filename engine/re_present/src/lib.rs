// Test-module assert_eq! on f32 values are intentional exact-equal checks.
// Allow float_cmp only in test compilation units; lib code remains denied.
#![cfg_attr(test, allow(clippy::float_cmp))]
#![warn(missing_docs)]
//! `re_present` — the PRESENT layer (Layer B). **This is the only crate in the
//! workspace that touches a terminal.** Every other crate (world, puzzle,
//! camera, raycaster, framebuffer, `glyph_shade`) is engine-agnostic and
//! compiles and tests with no terminal present; only here do we depend on
//! `bevy_ratatui` / crossterm.
//!
//! It presents [`re_core::Framebuffer`] to the terminal via `bevy_ratatui`,
//! fully HEADLESS (`bevy` with `default-features = false`; no wgpu, no winit).
//! The crate re-exports the terminal plugin as [`PresentPlugin`]; `re_game`
//! adds exactly that.
//!
//! ## The present path: framebuffer → dirty diff → one write per frame
//!
//! Each frame, `present_sys` (in `terminal.rs`) is handed `&Framebuffer` and:
//!
//! 1. **Dirty-diffs it against the last frame's target** via [`PresentCache`]
//!    (in `present_cache.rs`): every cell's `{glyph, fg, bg, style}` is
//!    compared to what was last targeted, not what is currently on screen. If
//!    nothing changed and the terminal size is unchanged, the function returns
//!    before any terminal I/O — an idle frame (the common case for a player
//!    standing still) costs zero bytes and zero syscalls.
//! 2. If anything changed, colours pass through the cache's **deadband**
//!    first: a cell keeps the colour it last *emitted* while the new target
//!    stays within [`PresentConfig::color_deadband`] per channel, so the
//!    raycaster's continuous depth shade does not spam near-identical
//!    `SetColors` escapes on every cell after every small camera step. Glyph
//!    and style changes are never held back this way.
//! 3. The result is blitted into a ratatui `Buffer` once and handed to
//!    `bevy_ratatui`'s `RatatuiContext::draw`, which performs **one present**
//!    — ratatui's own internal diff against its last-drawn buffer decides the
//!    actual bytes written to stdout, wrapped in a single DEC 2026
//!    synchronized-update window so a partial frame is never visible mid-tear.
//!
//! ## The input path
//!
//! The terminal backend reads crossterm key/mouse/resize events each frame
//! (`input_sys`) and turns them into backend-neutral types the rest of the
//! workspace can subscribe to with no `crossterm` dependency: [`KeyIntent`]
//! (classifies a single keypress), [`HeldKeys`] (the set of movement/turn
//! keys currently down, driving smooth simultaneous WASD+arrow motion),
//! [`MenuNav`] / [`MenuMouseMove`] / [`MenuMouseClick`] (menu navigation and
//! mouse hit-testing), and the one-shot action messages [`InteractPressed`],
//! [`FovCycle`], [`ZoomToggle`], [`PostFxCycle`]. `re_game` consumes only
//! these types; it never sees a crossterm event.
//!
//! ## The seam
//!
//! The contract is intentionally tiny: the plugin each frame consumes
//! `&Framebuffer` and presents it. Color unpacking is centralized in [`rgb8`];
//! the backend adapts the triple to ratatui's `Color::Rgb`.

/// Per-EVENT camera move/turn step, used ONLY by the terminal fallback path on
/// terminals with no key-release reporting (one crossterm key event = one step).
/// Cell units for position, radians for yaw. The held-key path uses the
/// time-based [`MOVE_SPEED`] / [`TURN_SPEED`] instead (smooth, frame-rate
/// independent).
///
/// Reduced from 0.34 → **0.11** proportionally with the [`MOVE_SPEED`] reduction
/// (~3×), so the non-kitty discrete-step fallback also walks ~3× slower (walk
/// feel matches the smooth path). Shift-sprint on the fallback path multiplies
/// this by [`SPRINT_MULTIPLIER`] to restore the old fast feel.
pub(crate) const MOVE_STEP: f32 = 0.11;
// Used by the terminal fallback path (no key-release reporting) and its tests.
#[allow(dead_code)]
pub(crate) const TURN_STEP: f32 = 0.10;

/// Time-based movement SPEED in cells/second, applied as `MOVE_SPEED * dt` every
/// frame from the set of currently-held movement keys (W/A/S/D). This is the
/// primary, smooth, frame-rate-independent path: when the terminal reports key
/// release events (kitty protocol) we maintain a held-key set and drive this
/// path, so simultaneous + diagonal movement works (W+A = a smooth diagonal,
/// any press/release order).
///
/// Reduced from 7.0 → **2.3** (÷~3) so the default walk feels deliberate and
/// readable in the narrow FOV. Holding Shift (sprint) multiplies by
/// [`SPRINT_MULTIPLIER`] (2.6) to recover approximately the old 6 cells/sec feel
/// for players who want to move fast. At 60 fps `MOVE_SPEED * (1/60) ≈ 0.038`
/// cells/frame for walk, `≈ 0.10` cells/frame for sprint.
pub const MOVE_SPEED: f32 = 2.3;

/// Sprint multiplier applied to [`MOVE_SPEED`] (and to `MOVE_STEP` on the
/// non-kitty fallback path) while the Shift key is held.
///
/// `2.3 * 2.6 ≈ 5.98 cells/s` — close to the pre-reduction 7.0 speed.
/// Applied as a pure scaling factor in [`held_move_delta`]'s callers and in the
/// non-kitty per-event fallback; it NEVER clears or resets any movement key flag.
pub const SPRINT_MULTIPLIER: f32 = 2.6;

/// Time-based TURN speed in radians/second for the keyboard arrow turn (held),
/// applied as `TURN_SPEED * dt` per frame in the held paths. At 3.0 rad/s a
/// held arrow sweeps ~172°/s — responsive enough to be the primary look method,
/// covering a 180° turn in roughly 1 second. The per-event fallback keeps using
/// the small discrete `TURN_STEP` instead.
///
/// Raised from 2.4 → **3.0** so keyboard turning is a first-class alternative
/// to mouse-look. When the mouse pointer is pinned at the terminal edge and
/// mouse-look stalls, keyboard turning is seamless and unlimited — and the
/// higher speed makes it feel natural rather than sluggish.
pub const TURN_SPEED: f32 = 3.0;

/// The set of movement / turn keys currently held down, backend-neutral.
///
/// This is the heart of true **simultaneous** movement: every frame the terminal
/// backend fills this from the press/release-tracked set (kitty protocol) and
/// hands it to the pure [`held_move_delta`] / [`held_turn_delta`] helpers.
/// Because movement is derived from the SET, not from individual key events,
/// holding `W`+`A` yields a diagonal and releasing either one leaves the other
/// still driving — in any press/release order.
///
/// It carries the *continuous* (held) actions. One-shot actions (jump, interact,
/// FOV, menu-nav, Back) stay edge/event-driven so a held key never storms them.
/// `sprint` is a **pure modifier** — pressing/releasing Shift NEVER clears or
/// resets the movement flags; it only scales the speed.
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeldKeys {
    /// `W` — moving forward along facing.
    pub forward: bool,
    /// `S` — moving backward.
    pub backward: bool,
    /// `A` — strafing left.
    pub strafe_left: bool,
    /// `D` — strafing right.
    pub strafe_right: bool,
    /// `←` — turning yaw left.
    pub turn_left: bool,
    /// `→` — turning yaw right.
    pub turn_right: bool,
    /// `↑` — pitching up.
    pub pitch_up: bool,
    /// `↓` — pitching down.
    pub pitch_down: bool,
    /// `LeftShift` / `RightShift` — sprint modifier. When true the movement
    /// delta is multiplied by [`SPRINT_MULTIPLIER`]. Pure modifier: pressing or
    /// releasing Shift NEVER touches the WASD/arrow movement flags.
    pub sprint: bool,
}

/// Safety-net timeout for the kitty-protocol held-key set: if a held direction is
/// NOT refreshed by a Press or Repeat event within this many seconds, it is force-
/// cleared by [`HeldTimestamps::expire`].
///
/// ### Why this exists
///
/// On capable terminals (kitty, Windows Terminal, WezTerm) the kitty protocol
/// delivers explicit **Release** events, which clear `HeldKeys` immediately in
/// `input_sys` — the timeout NEVER fires on those paths, making this a true no-op.
///
/// On terminals that ignore the kitty bytes (legacy conhost, some SSH proxies) NO
/// Release events arrive, so `HeldKeys` would latch "on" forever once any movement
/// key is pressed. The timeout is the only safety net there.
///
/// ### The multi-key timing problem on non-Release terminals
///
/// `force_kitty_sys` always inserts `KittyEnabled` when its raw VT write to stdout
/// succeeds, even on conhost which silently ignores the bytes. The result: on
/// conhost `input_sys` takes the kitty branch and latches `held.forward = true` on
/// a W press — but Release events for W never arrive.
///
/// When the player presses A while holding W, the OS stops autorepeating W and
/// starts autorepeating A. W receives **no further events**, so `timestamps.forward`
/// is only as fresh as the last W autorepeat that arrived **before** A was pressed.
/// If the player pressed A before W's first autorepeat (initial repeat delay on
/// Windows ≈ 500 ms), `timestamps.forward` is as old as the original W press.
///
/// At the old 0.75 s timeout, W would expire just 0.25 s after A was pressed if A
/// was tapped before W's first autorepeat. That produced the user-observed bug:
/// "holding W then tapping Space/A stops forward movement."
///
/// The value must be:
/// * **Longer than (OS initial repeat delay + typical tap duration)**: on Windows
///   the default initial repeat delay is ~500 ms; a quick tap of another key is
///   ~200–500 ms. To survive a tap started before W's first autorepeat, we need
///   more than 500 ms + tap_duration + margin. **2.0 s** covers taps up to ~1.5 s,
///   which is far longer than any reasonable directional tap in gameplay.
/// * **Short enough that camera drift after true release feels bounded**: on conhost
///   the camera drifts for up to `HELD_EXPIRE_SECS` after the player fully releases
///   all keys and no more events arrive. 2.0 s is noticeable but still clearly
///   bounded and clearly documented as a conhost limitation.
///
/// ### What remains an unavoidable OS limitation
///
/// Even with a generous timeout, holding one movement key while another is pressed
/// **will stop the first key's movement for the duration the second is held** on
/// non-Release terminals (conhost), because the OS only autorepeats the last-pressed
/// key. The player must physically re-press W to resume before the timeout fires.
/// A Release-capable terminal (Windows Terminal, WezTerm, kitty, foot) never has
/// this problem — the kitty path clears each key on its own Release event only.
pub const HELD_EXPIRE_SECS: f64 = 2.0;

/// Per-direction last-refresh timestamps for the held-key safety expiry net.
///
/// Parallel to [`HeldKeys`]: one `f64` elapsed-seconds slot per direction, set to
/// the current [`bevy::prelude::Time`] elapsed time on every Press/Repeat of that
/// direction, and reset to `None` when that direction is Released (or never pressed).
///
/// [`expire`](Self::expire) is called by `expire_held_sys` each frame BEFORE
/// `held_move_sys`; it clears any `HeldKeys` flag whose timestamp is older than
/// [`HELD_EXPIRE_SECS`]. On kitty-capable terminals Release events clear `HeldKeys`
/// long before the timeout, so `expire` is effectively a no-op there.
///
/// `sprint` has a timestamp slot so the Shift key also goes through the expiry
/// safety net on legacy terminals. Pressing/releasing Shift is treated identically
/// to any other held key for lifecycle purposes — it just has no effect on
/// movement flag state when it fires.
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct HeldTimestamps {
    /// Last refresh time for `HeldKeys::forward` (`W`). `None` = not currently held.
    pub forward: Option<f64>,
    /// Last refresh time for `HeldKeys::backward` (`S`).
    pub backward: Option<f64>,
    /// Last refresh time for `HeldKeys::strafe_left` (`A`).
    pub strafe_left: Option<f64>,
    /// Last refresh time for `HeldKeys::strafe_right` (`D`).
    pub strafe_right: Option<f64>,
    /// Last refresh time for `HeldKeys::turn_left` (`←`).
    pub turn_left: Option<f64>,
    /// Last refresh time for `HeldKeys::turn_right` (`→`).
    pub turn_right: Option<f64>,
    /// Last refresh time for `HeldKeys::pitch_up` (`↑`).
    pub pitch_up: Option<f64>,
    /// Last refresh time for `HeldKeys::pitch_down` (`↓`).
    pub pitch_down: Option<f64>,
    /// Last refresh time for `HeldKeys::sprint` (`LeftShift` / `RightShift`).
    pub sprint: Option<f64>,
}

impl HeldTimestamps {
    /// Record the current elapsed time for `intent`'s slot (Press or Repeat).
    /// One-shot intents (same as [`HeldKeys::set`]) are ignored.
    #[inline]
    pub fn refresh(&mut self, intent: KeyIntent, now: f64) {
        match intent {
            KeyIntent::Forward => self.forward = Some(now),
            KeyIntent::Backward => self.backward = Some(now),
            KeyIntent::StrafeLeft => self.strafe_left = Some(now),
            KeyIntent::StrafeRight => self.strafe_right = Some(now),
            KeyIntent::TurnLeft => self.turn_left = Some(now),
            KeyIntent::TurnRight => self.turn_right = Some(now),
            KeyIntent::PitchUp => self.pitch_up = Some(now),
            KeyIntent::PitchDown => self.pitch_down = Some(now),
            KeyIntent::Sprint => self.sprint = Some(now),
            KeyIntent::Jump
            | KeyIntent::Interact
            | KeyIntent::Back
            | KeyIntent::Fov(_)
            | KeyIntent::ZoomToggle
            | KeyIntent::PostFxCycle
            | KeyIntent::TogglePerf
            | KeyIntent::MouseLookToggle
            | KeyIntent::Unmapped => {}
        }
    }

    /// Clear the timestamp slot for `intent` (Release edge or explicit reset).
    /// One-shot intents are ignored (they have no slot).
    #[inline]
    pub fn clear(&mut self, intent: KeyIntent) {
        match intent {
            KeyIntent::Forward => self.forward = None,
            KeyIntent::Backward => self.backward = None,
            KeyIntent::StrafeLeft => self.strafe_left = None,
            KeyIntent::StrafeRight => self.strafe_right = None,
            KeyIntent::TurnLeft => self.turn_left = None,
            KeyIntent::TurnRight => self.turn_right = None,
            KeyIntent::PitchUp => self.pitch_up = None,
            KeyIntent::PitchDown => self.pitch_down = None,
            KeyIntent::Sprint => self.sprint = None,
            KeyIntent::Jump
            | KeyIntent::Interact
            | KeyIntent::Back
            | KeyIntent::Fov(_)
            | KeyIntent::ZoomToggle
            | KeyIntent::PostFxCycle
            | KeyIntent::TogglePerf
            | KeyIntent::MouseLookToggle
            | KeyIntent::Unmapped => {}
        }
    }

    /// Expire any held direction whose last refresh is older than
    /// [`HELD_EXPIRE_SECS`] seconds before `now`, clearing both the timestamp slot
    /// and the corresponding flag in `held`.
    ///
    /// This is the safety-net call: on kitty-capable terminals Release events clear
    /// the flags long before this threshold, so the method is a no-op in practice
    /// there. On terminals without Release reporting it ensures the camera does not
    /// run forever after a key was physically released.
    pub fn expire(&mut self, held: &mut HeldKeys, now: f64) {
        macro_rules! check {
            ($ts:expr, $flag:expr) => {
                if let Some(t) = $ts {
                    if now - t >= HELD_EXPIRE_SECS {
                        $flag = false;
                        $ts = None;
                    }
                }
            };
        }
        check!(self.forward, held.forward);
        check!(self.backward, held.backward);
        check!(self.strafe_left, held.strafe_left);
        check!(self.strafe_right, held.strafe_right);
        check!(self.turn_left, held.turn_left);
        check!(self.turn_right, held.turn_right);
        check!(self.pitch_up, held.pitch_up);
        check!(self.pitch_down, held.pitch_down);
        check!(self.sprint, held.sprint);
    }
}

impl HeldKeys {
    /// Set/clear the held flag for a movement-or-turn [`KeyIntent`]. One-shot
    /// intents (jump/interact/FOV/back/unmapped) are ignored — they are not
    /// continuous, so they never enter the held set. Returns `true` if `intent`
    /// was a continuous (held-tracked) movement/turn/modifier intent.
    ///
    /// `Sprint` is a **pure modifier**: setting it to `true` or `false` NEVER
    /// touches the WASD/arrow flags. This is the architectural guarantee that
    /// pressing or releasing Shift cannot disrupt active movement.
    pub fn set(&mut self, intent: KeyIntent, down: bool) -> bool {
        match intent {
            KeyIntent::Forward => self.forward = down,
            KeyIntent::Backward => self.backward = down,
            KeyIntent::StrafeLeft => self.strafe_left = down,
            KeyIntent::StrafeRight => self.strafe_right = down,
            KeyIntent::TurnLeft => self.turn_left = down,
            KeyIntent::TurnRight => self.turn_right = down,
            KeyIntent::PitchUp => self.pitch_up = down,
            KeyIntent::PitchDown => self.pitch_down = down,
            // Sprint: pure modifier — only its OWN flag changes. Movement flags untouched.
            KeyIntent::Sprint => self.sprint = down,
            KeyIntent::Jump
            | KeyIntent::Interact
            | KeyIntent::Back
            | KeyIntent::Fov(_)
            | KeyIntent::ZoomToggle
            | KeyIntent::PostFxCycle
            | KeyIntent::TogglePerf
            | KeyIntent::MouseLookToggle
            | KeyIntent::Unmapped => return false,
        }
        true
    }

    /// True when no movement/turn key is held (so the per-frame apply can early-out).
    #[inline]
    pub fn is_empty(self) -> bool {
        self == HeldKeys::default()
    }
}

/// Combine all currently-held MOVE keys into one world-space `(dx, dy)` position
/// delta for this frame, given the camera facing `(dirx, diry)` (from
/// [`re_world::CameraPose::dir`]), a `speed` in cells/second, and the frame `dt`
/// in seconds. **Pure** — no ECS, no terminal — so the simultaneous-movement math
/// is unit-testable headlessly.
///
/// Forward/back scale the facing vector; strafe scales the facing rotated −90°
/// (`(diry, -dirx)`), matching the gamepad `stick_move` and the keyboard strafe
/// basis. Opposing keys (W+S or A+D) cancel. The result is **normalized** so a
/// diagonal (W+A) is not faster than a straight line — combined input is clamped
/// to unit length, then scaled by `speed * dt`. So:
///
/// * `W` held → forward at `speed`.
/// * `W`+`A` held → a 45° diagonal at `speed` (not `speed*√2`).
/// * release `A`, keep `W` → straight forward at `speed`, no hitch.
#[inline]
pub fn held_move_delta(held: HeldKeys, dirx: f32, diry: f32, speed: f32, dt: f32) -> (f32, f32) {
    // Net intent on each axis in facing-space: +1 forward, −1 back; +1 right, −1 left.
    let fwd = (held.forward as i32 - held.backward as i32) as f32;
    let strafe = (held.strafe_right as i32 - held.strafe_left as i32) as f32;
    if fwd == 0.0 && strafe == 0.0 {
        return (0.0, 0.0);
    }
    // World-space direction: forward along facing, strafe along facing rotated −90°.
    let mut wx = dirx * fwd + diry * strafe;
    let mut wy = diry * fwd - dirx * strafe;
    // Normalize so diagonals aren't faster than cardinals, then scale by speed*dt.
    let len = (wx * wx + wy * wy).sqrt();
    if len > 0.0 {
        let s = speed * dt / len;
        wx *= s;
        wy *= s;
    }
    (wx, wy)
}

/// Combine the held TURN keys into a `(d_yaw, d_pitch)` delta for this frame,
/// scaled by `speed` (rad/s) × `dt`. **Pure.** Left/right turn yaw (− / +),
/// up/down tilt pitch (+ / −); opposing keys cancel. Pitch clamping is the
/// caller's job (it owns the current pitch and `PITCH_LIMIT`); this returns only
/// the raw deltas so it stays a trivially-testable function of the held set.
#[inline]
pub fn held_turn_delta(held: HeldKeys, speed: f32, dt: f32) -> (f32, f32) {
    let yaw = (held.turn_right as i32 - held.turn_left as i32) as f32 * speed * dt;
    let pitch = (held.pitch_up as i32 - held.pitch_down as i32) as f32 * speed * dt;
    (yaw, pitch)
}

// --- Grounded movement glue (collision + gravity + jump). --------------------
//
// re_world owns the physics (see `re_world::CameraPose::move_with_collision`,
// `step_gravity`, `try_jump` + the `PlayerMotion` component and the
// `PLAYER_RADIUS` / `GRAVITY` / `JUMP_IMPULSE` consts). The present layer's job
// is only to WIRE that physics into the per-frame input/update flow:
//
//   * horizontal move keys/stick → `move_with_collision(dx, dy, &map, radius)`
//     (was a raw `pos += …` that walked through walls);
//   * every frame → `step_gravity(&mut motion, dt, &map)` so the camera is
//     grounded by default and falls;
//   * Space (key) / a gamepad face button → `try_jump(&mut motion)`.
//
// FREE-FLY DISPOSITION: the old free-fly rise/fall (`x`/`z`, PageUp/PageDown,
// and the gamepad shoulders/triggers driving `apply_height`) is RETIRED in favor
// of the grounded + jump model. Running both `step_gravity` and `apply_height`
// on the same entity each frame would have them fight over `pos.z`; rather than
// gate a noclip mode we simply drop free-fly. Reaching a switch one z-layer up
// (e.g. level 01's z=2 switch from the z=1 floor) is now done by JUMPING.

pub use re_world::{PlayerMotion, PLAYER_RADIUS};

use bevy::ecs::entity::Entity;
use bevy::ecs::query::{With, Without};
use bevy::ecs::schedule::SystemSet;
use bevy::ecs::system::{Commands, Query};

/// Insert-if-missing system: give every [`re_world::CameraPose`] entity a
/// [`PlayerMotion`] component so the gravity tick has somewhere to store its
/// vertical state. Runs in both backends so `re_present` is self-sufficient —
/// it does NOT rely on `re_game` (or anyone upstream) adding `PlayerMotion`.
///
/// The query filter `Without<PlayerMotion>` makes this a no-op once the
/// component exists, so it costs one archetype scan per frame and nothing more.
pub(crate) fn ensure_player_motion(
    mut commands: Commands,
    q: Query<Entity, (With<re_world::CameraPose>, Without<PlayerMotion>)>,
) {
    for e in &q {
        commands.entity(e).insert(PlayerMotion::default());
    }
}

/// Vertical step per height-key press, in cell units. Retained only because the
/// (now-removed) free-fly path referenced it; kept as a constant so a future
/// noclip/debug toggle could reuse it. **Currently unused by the live input.**
#[allow(dead_code)]
pub(crate) const HEIGHT_STEP: f32 = 1.0;

/// All framebuffer-to-terminal presentation runs in this set. Any system that
/// WRITES the [`re_core::Framebuffer`] (HUD, menus, overlays, post-fx) must be
/// ordered `.before(PresentSet)` so its writes are visible in the presented
/// frame.
///
/// `re_game` (and any crate that paints overlays onto the framebuffer in
/// `Last`) should declare:
/// ```ignore
/// .add_systems(Last, my_overlay_sys.before(PresentSet))
/// ```
/// This eliminates the race where `present_sys` runs before overlay painters
/// in an unordered `Last` schedule, causing overlays to flicker or vanish
/// on some frames.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PresentSet;

/// Base mouse-look sensitivity (radians of camera rotation per one cell of
/// relative mouse motion), shared by both backends so they feel identical. The
/// retired C build turned the camera with relative mouse motion; we reproduce
/// that. These are the BASE values for the `Medium` sensitivity preset; the
/// live [`MouseSettings`] resource scales them by a runtime multiplier so the
/// settings menu can change the feel without recompiling.
///
/// * `MOUSE_YAW_SENS` — horizontal: yaw is unbounded (wraps).
/// * `MOUSE_PITCH_SENS` — vertical: slightly gentler, and clamped to
///   `PITCH_LIMIT` so the player can't look past straight up/down. The y axis
///   is inverted (mouse-down → look down) to match the C build / standard FPS.
///
/// Halved from 0.10 / 0.08 → **0.05 / 0.04** (playtest: mouse-look was
/// reported as too fast). The preset multipliers (Low=0.5, Med=1.0, High=1.75)
/// are unchanged — every preset now feels half as fast as before.
pub const MOUSE_YAW_SENS: f32 = 0.05;
/// Vertical companion to [`MOUSE_YAW_SENS`] — see that constant's doc for the
/// shared rationale and the invert/clamp behaviour applied on top of it.
pub const MOUSE_PITCH_SENS: f32 = 0.04;

/// Pitch clamp (radians). `re_world::CameraPose` carries a `pitch` field but
/// exposes no clamp helper / limit constant, so we own one here: ±π/3 (60°),
/// the user-approved compromise for the 2.5D shear engine (widened from ±π/4).
/// Keeping the camera within this band stops it tilting past the horizon-readable
/// range the raycaster shades for.
///
/// CONTRACT: this MUST equal `re_render::PITCH_LIMIT_RAD` — the pitch the input can
/// produce is exactly the pitch the raycaster's horizon shift saturates at, so the
/// near-tier sky/floor fill covers the whole view to the clamp with no frozen-look
/// dead band and no black band. (`re_present` does not depend on `re_render`, so the
/// value is duplicated, not imported; the compile-time guard in `re_render` and the
/// `apply_mouse_look` saturation test pin both sides to π/3.)
pub(crate) const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_3;

/// Apply a relative mouse motion (in terminal cells) to a camera's `(yaw,
/// pitch)`, returning the new pair, scaled by a sensitivity `mult`. **Pure** —
/// no ECS, no terminal — so the feel math (sensitivity + invert + pitch clamp)
/// is unit-testable headlessly. The live mouse-look systems pass
/// [`MouseSettings::sensitivity`] as `mult`; tests can pass `1.0` for the base
/// feel.
///
/// * `dx` (columns moved this step) turns yaw by `dx * MOUSE_YAW_SENS * mult`;
///   yaw is left unbounded (it wraps naturally through `sin`/`cos`).
/// * `dy` (rows moved this step) tilts pitch by `-dy * MOUSE_PITCH_SENS * mult`
///   (screen y grows downward, so moving the mouse down should look down), then
///   clamps the result to ±`PITCH_LIMIT`.
#[inline]
pub fn apply_mouse_look(yaw: f32, pitch: f32, dx: f32, dy: f32, mult: f32) -> (f32, f32) {
    let new_yaw = yaw + dx * MOUSE_YAW_SENS * mult;
    let new_pitch = (pitch - dy * MOUSE_PITCH_SENS * mult).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    (new_yaw, new_pitch)
}

/// The player-tunable mouse-look settings the present layer reads each frame.
///
/// Inserted as a Bevy [`bevy::ecs::resource::Resource`] by `re_game` (from the
/// persisted settings, or defaulted). Both backends' mouse-look systems read it:
/// when [`enabled`](Self::enabled) is false they skip applying any motion, and
/// when true they scale the base [`MOUSE_YAW_SENS`] / [`MOUSE_PITCH_SENS`] by
/// [`sensitivity`](Self::sensitivity). This replaces the old compile-time
/// `pub(crate)` sensitivity constants — the constants are now just the base feel.
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct MouseSettings {
    /// Master on/off for mouse-look. When false the mouse-look systems are inert
    /// (keyboard turn still works — mouse-look was always additive).
    pub enabled: bool,
    /// Multiplier on the base sensitivity. `1.0` = the historical feel; the
    /// settings menu offers Low (0.5) / Medium (1.0) / High (1.75) presets.
    pub sensitivity: f32,
}

impl Default for MouseSettings {
    fn default() -> Self {
        // The historical feel: mouse-look on, base (Medium) sensitivity.
        MouseSettings {
            enabled: true,
            sensitivity: 1.0,
        }
    }
}

/// Backend-neutral classification of a single key press. Every downstream
/// decision — what moves, what turns, what navigates a menu — is driven from
/// THIS enum. It is also what makes the mapping unit-testable headlessly
/// (no live terminal required).
///
/// ## The live key map (post player-bug-fix)
///
/// * **WASD = movement** (collision-checked): `W` forward, `S` back,
///   `A` strafe LEFT, `D` strafe RIGHT — NONE of them turn.
/// * **Arrows = turn**: ←/→ yaw left/right, ↑/↓ pitch up/down (the
///   always-available keyboard turn; the mouse turns too but edge-stalls).
/// * **Space** = jump, **E/Enter** = interact, **Esc** = [`MenuNav::Back`].
/// * **`q` is unmapped** — it no longer quits (was an accidental-exit footgun).
/// * `[` / `]` = FOV narrow / widen.
///
/// In menus (no camera) the move/turn arms are inert and only [`Self::menu_nav`]
/// matters; in `Playing` the camera consumes the move/turn intent and `re_game`
/// ignores `MenuNav`. The two are emitted together, additively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyIntent {
    /// `W` — move forward along facing (collision-checked).
    Forward,
    /// `S` — move backward (collision-checked).
    Backward,
    /// `A` — strafe LEFT (perpendicular to facing, collision-checked). NOT a turn.
    StrafeLeft,
    /// `D` — strafe RIGHT (perpendicular to facing, collision-checked). NOT a turn.
    StrafeRight,
    /// `←` — turn yaw left (keyboard turn; menus read it as `MenuNav::Left`).
    TurnLeft,
    /// `→` — turn yaw right (keyboard turn; menus read it as `MenuNav::Right`).
    TurnRight,
    /// `↑` — pitch up, clamped (menus read it as `MenuNav::Up`).
    PitchUp,
    /// `↓` — pitch down, clamped (menus read it as `MenuNav::Down`).
    PitchDown,
    /// `Space` — jump (only when grounded); menus read it as `MenuNav::Confirm`.
    Jump,
    /// `E` / `Enter` — interact; menus read it as `MenuNav::Confirm`.
    Interact,
    /// `Esc` or `q`/`Q` — back / cancel / pause. (Both keys map here; Back is never
    /// a surprise hard-quit — it opens Pause in play, goes up one level in menus,
    /// and quits only at the top menu, decided by `re_game`.)
    Back,
    /// `[` (−1) / `]` (+1) — cycle the FOV preset.
    Fov(i8),
    /// `Z` — toggle the FOV-ZOOM (base ↔ narrow zoom). One-shot edge action; the
    /// `re_game` zoom state machine flips its target and starts the reveal. NOT a
    /// nav intent.
    ZoomToggle,
    /// `G` — cycle the live POST-FX stack (off → blur → grayscale → palette → …).
    /// One-shot edge action consumed by `re_game`. NOT a nav intent.
    PostFxCycle,
    /// `F3` — toggle the on-screen performance overlay. NOT a nav intent.
    TogglePerf,
    /// `LeftShift` / `RightShift` — sprint modifier (held). Scales movement speed
    /// by [`SPRINT_MULTIPLIER`] while held. PURE modifier: pressing/releasing Shift
    /// NEVER clears, resets, or interferes with any WASD/arrow movement flag.
    /// Emits NO [`MenuNav`] — Shift is movement-only, never navigation.
    Sprint,
    /// `M` — toggle mouse-look on/off. A one-shot press that flips
    /// [`MouseSettings::enabled`]. When mouse-look is OFF the free OS pointer
    /// cannot accidentally drag the view, and keyboard turning becomes the sole
    /// look method (always edge-free). Emits NO [`MenuNav`].
    MouseLookToggle,
    /// A key we do not bind (e.g. `q`): no movement, no nav, no action.
    Unmapped,
}

impl KeyIntent {
    /// The [`MenuNav`] this intent emits in a menu state, if any. The directional
    /// mapping is the player-expected one: WASD navigate (W/S = Up/Down, A/D =
    /// Left/Right) AND arrows navigate (↑/↓ = Up/Down, ←/→ = Left/Right), so both
    /// the movement keys and the turn keys still drive menus. `Unmapped` (e.g.
    /// `q`) returns `None` — it never navigates and never backs out.
    #[inline]
    pub fn menu_nav(self) -> Option<MenuNav> {
        match self {
            KeyIntent::Forward | KeyIntent::PitchUp => Some(MenuNav::Up),
            KeyIntent::Backward | KeyIntent::PitchDown => Some(MenuNav::Down),
            KeyIntent::StrafeLeft | KeyIntent::TurnLeft => Some(MenuNav::Left),
            KeyIntent::StrafeRight | KeyIntent::TurnRight => Some(MenuNav::Right),
            KeyIntent::Jump | KeyIntent::Interact => Some(MenuNav::Confirm),
            KeyIntent::Back => Some(MenuNav::Back),
            // Sprint (Shift) is a movement modifier only — it never navigates menus.
            // Zoom + post-fx are one-shot gameplay toggles, never menu nav.
            // MouseLookToggle is a one-shot display setting, never menu nav.
            KeyIntent::Fov(_)
            | KeyIntent::ZoomToggle
            | KeyIntent::PostFxCycle
            | KeyIntent::TogglePerf
            | KeyIntent::Sprint
            | KeyIntent::MouseLookToggle
            | KeyIntent::Unmapped => None,
        }
    }
}

/// Backend-neutral input action surfaced by the present layer for gameplay the
/// binary owns. Today the only variant is [`InteractPressed`] (the player
/// pressed the interact key); modelled as its own Bevy [`bevy::ecs::message::Message`]
/// so downstream crates subscribe without any terminal dependency.
///
/// The present layer deliberately does NOT compute *what* is interacted with —
/// it only reports that the interact action fired. `re_game` owns the
/// faced-cell math and turns this into a `re_puzzle::InteractAt`.
#[derive(bevy::ecs::message::Message, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InteractPressed;

/// Backend-neutral **menu-navigation** action surfaced by the present layer for
/// the binary's state-machine menus. Modelled as its own Bevy
/// [`bevy::ecs::message::Message`] so the flow / menu code subscribes without any
/// terminal dependency — exactly like [`InteractPressed`].
///
/// ## Why this exists (vs. the old press-and-pause hack)
///
/// Before this message the only neutral signal a menu could observe was
/// [`InteractPressed`], so `re_game` faked directional movement with a
/// "press steps the highlight, then a short idle commits it" model. That was
/// imprecise (you couldn't go *up*, and a deliberate pause was required to pick).
/// `MenuNav` gives the menus crisp Up/Down/Left/Right/Confirm/Back independent of
/// which backend is active.
///
/// ## Coexistence with gameplay movement
///
/// `MenuNav` is **additive**: it is emitted *alongside* the existing camera
/// motion / height / [`InteractPressed`] outputs, on the same keypress. Both
/// fire — that is fine, because the consumers are state-gated:
///
/// * in the menu states (`Splash`/`Menu`/`LevelSelect`/`Won`) no `CameraPose`
///   entity exists, so the camera-motion writes are inert, and the flow reads
///   `MenuNav`;
/// * in `Playing` the flow ignores `MenuNav` and the camera reads the motion.
///
/// Emitted **once per keypress** (key-release edges are dropped on the terminal
/// backend; only press edges fire), so navigation never storms from autorepeat.
#[derive(bevy::ecs::message::Message, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuNav {
    /// Previous item (Up arrow / `W`).
    Up,
    /// Next item (Down arrow / `S`).
    Down,
    /// Previous group / left (Left arrow / `A`).
    Left,
    /// Next group / right (Right arrow / `D`).
    Right,
    /// Commit the highlighted item (Enter / Space / `E`).
    Confirm,
    /// Go back / cancel (Esc OR `q`/`Q`). `re_game` turns this into pause
    /// (gameplay), up-one-level (sub-menus), or quit (top menu); the present layer
    /// never writes `AppExit` itself.
    Back,
}

/// Backend-neutral **mouse hover** position for menu hit-testing, in terminal
/// CELL coordinates (column, row) — additive alongside [`MenuNav`] exactly the
/// same way [`MenuNav`] is additive alongside camera motion: `re_present`
/// EMITS it every frame the mouse moves, unconditionally, and `re_game`'s
/// per-state menu systems (state-gated via `run_if(in_state(..))`) decide
/// whether it means anything.
///
/// Emitted by the terminal backend's `emit_menu_mouse_sys` from crossterm's
/// `Moved`/`Drag` mouse events — the same event kinds
/// [`mouse_motion_cell`] extracts for the look path — but suppressed while
/// [`OsCursorGrab::active`] is true (i.e. mouse-look engaged in `Playing`):
/// see that system's doc for why checking the grab, not `GameState` (which
/// `re_present` cannot see — the host's state type is outside the engine's
/// dependency direction, see `docs/LAYERING.md`), is the correct gate here.
#[derive(bevy::ecs::message::Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct MenuMouseMove {
    /// Terminal column (0-based).
    pub col: u16,
    /// Terminal row (0-based).
    pub row: u16,
}

/// Backend-neutral **mouse left-click** at a terminal cell position, emitted
/// ONLY on the press edge (never `Up`/release, never a held-drag repeat) —
/// the same press-edge debounce [`MenuNav`] uses so a held button can never
/// storm a menu. `re_game`'s per-state menu systems hit-test `(col, row)`
/// against that screen's own painted item rects and, on a hit, treat it
/// exactly like a keyboard `Enter`: move [`MenuCursor`] to the hit item and
/// write a [`MenuNav::Confirm`] — reusing the existing commit pipeline rather
/// than duplicating any state-transition logic.
///
/// [`MenuCursor`]: # "re_game::flow::MenuCursor"
#[derive(bevy::ecs::message::Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct MenuMouseClick {
    /// Terminal column (0-based).
    pub col: u16,
    /// Terminal row (0-based).
    pub row: u16,
}

/// Backend-neutral **field-of-view cycle** request surfaced by the present
/// layer, emitted by the gamepad's FOV button and a keyboard binding. The
/// payload is a signed step: `+1` widens to the next FOV preset, `-1` narrows to
/// the previous one (the consumer in `re_game` owns the actual preset list +
/// clamping and applies it to the live raycaster FOV).
///
/// Like [`InteractPressed`] / [`MenuNav`] this is a backend-neutral
/// [`bevy::ecs::message::Message`] so downstream crates subscribe with no
/// terminal dependency. The present layer only EMITS it (gamepad
/// Y/Triangle → `FovCycle(1)`; keyboard `[` → `FovCycle(-1)`, `]` →
/// `FovCycle(1)`); it never reads the current FOV or computes the new one.
#[derive(bevy::ecs::message::Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct FovCycle(pub i8);

/// Backend-neutral **FOV-ZOOM toggle** request, emitted once per `Z` keypress.
///
/// This is the "binoculars" zoom, distinct from [`FovCycle`] (which steps the
/// player's *base* FOV preset). It is a **TOGGLE**, not a hold: each press flips
/// the zoom target between the base FOV and the narrow zoom FOV. Toggle was chosen
/// over hold-to-zoom because the terminal backend's key-release reporting is only
/// available under the kitty protocol (absent on Windows conhost and many SSH
/// proxies), so a reliable, identically-behaving-on-both-backends, trivially-
/// testable signal is a toggle edge. `re_game` owns the zoom FOV value, the
/// transition, and the radial reveal; the present layer only reports the press.
///
/// Like [`InteractPressed`] / [`FovCycle`] it carries no terminal types.
#[derive(bevy::ecs::message::Message, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZoomToggle;

/// Backend-neutral **POST-FX cycle** request, emitted once per `G` keypress.
///
/// Each press advances the live screen-space post-effect stack by one step
/// (off → blur → grayscale → palette → … → off). `re_game` owns the cycle order
/// and the `re_render::PostFxConfig` it maps to; the present layer only reports
/// the press. Backend-neutral, exactly like [`ZoomToggle`].
#[derive(bevy::ecs::message::Message, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PostFxCycle;

/// Unpack a packed `0x00RRGGBB` color into `(r, g, b)` bytes. The single point
/// where our `u32` color convention is decoded; each backend then adapts the
/// triple to its own color type (ratatui `Color::Rgb`, bevy `LinearRgba`).
#[inline]
pub fn rgb8(c: u32) -> (u8, u8, u8) {
    (
        ((c >> 16) & 0xFF) as u8,
        ((c >> 8) & 0xFF) as u8,
        (c & 0xFF) as u8,
    )
}

/// A framebuffer glyph (`u32` scalar value) as a `char`, falling back to space
/// for any non-scalar value AND for any control character. Shared decode for
/// both backends.
///
/// ratatui's `cell_width` debug-asserts that no ASCII control char reaches it —
/// `Cell::set_char` does not filter, so the caller (us, the terminal boundary) must.
/// A default/unwritten framebuffer cell has glyph 0, and `char::from_u32(0)` is
/// `Some('\0')` (NUL is a valid scalar, so `unwrap_or` does NOT catch it). Passing
/// that to the present panics in debug and emits raw control bytes in release, so any
/// control char (C0 `0x00..=0x1F`, DEL, C1) is mapped to a blank space.
#[inline]
pub fn glyph_char(g: u32) -> char {
    let c = char::from_u32(g).unwrap_or(' ');
    if c.is_control() {
        ' '
    } else {
        c
    }
}

// --- Shared color policy (runtime truecolor vs. 256-color selector). ---

mod color;
pub use color::{ansi256_to_rgb, rgb_to_ansi256, ColorMode};

// --- Present-layer temporal memory: idle-frame skip + colour deadband. ---

mod present_cache;
pub use present_cache::{deadband_color, PresentCache, PresentConfig, TargetCell, DEADBAND_ENV};

// --- Cross-platform gamepad (gilrs), wired into BOTH backends. ---

mod gamepad;
pub use gamepad::{
    add_gamepad, apply_deadzone, stick_look, stick_move, trigger_axis, EdgeDetect, RepeatGate,
    LOOK_GAIN, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL, NAV_STICK_THRESHOLD, RESPONSE_CURVE,
    STICK_DEADZONE, TRIGGER_THRESHOLD,
};

// --- OS cursor grab (real FPS mouse-look via Win32/X11). ---
// Compiled unconditionally so the OsCursorGrab / OsCursorSimpleHide resource
// types are always available regardless of which backend is active. The
// platform-specific implementations inside are all no-ops on unsupported OSes.
mod os_cursor;
pub use os_cursor::{OsCursorGrab, OsCursorSimpleHide};

// --- Render configuration (cursor hide switch + grid size override). ---
// A single Bevy Resource that binaries insert before adding PresentPlugin.
// PresentPlugin reads it in Startup to drive the simple-hide and the
// best-effort terminal window resize. Both defaults reproduce today's exact
// behaviour (no hide for the editor, no fixed size for the game).
pub mod render_config;
pub use render_config::RenderConfig;

// --- Backend module + the re-exported PresentPlugin. ---

#[cfg(feature = "backend-terminal")]
mod terminal;
#[cfg(feature = "backend-terminal")]
pub use terminal::TerminalBackendPlugin as PresentPlugin;
/// Suspend/resume the terminal presentation for a playtest child-process
/// handoff (e.g. `ts_editor`'s F5 launching the game binary on the same
/// terminal). See the doc comments on each function — in particular the
/// drop-order contract — before calling either.
#[cfg(feature = "backend-terminal")]
pub use terminal::{resume_terminal, suspend_terminal};

// --- Phase 1 of the opening boot sequence: tear the real pre-existing main-
// screen text (VT editing sequences), before the alternate screen exists. ---
#[cfg(feature = "backend-terminal")]
mod intro_tear;
#[cfg(feature = "backend-terminal")]
pub use intro_tear::{
    finalize_terminal_teardown, generate_tear_ops, render_ops, run_screen_tear, should_run_tear,
    tty_and_vt_capable, TearColor, TearConfig, TearOp, SHADE_GLYPHS,
};

/// Query the terminal's current size in cells as `(cols, rows)`.
///
/// Thin wrapper over `crossterm::terminal::size()` (via `bevy_ratatui`) so binaries
/// that don't depend on crossterm directly (e.g. `ts_editor`) can pick a full-screen
/// initial framebuffer size. Returns `None` if the size cannot be determined (e.g. a
/// piped / non-tty stdout); callers fall back to a sane default.
#[cfg(feature = "backend-terminal")]
pub fn terminal_size() -> Option<(usize, usize)> {
    bevy_ratatui::crossterm::terminal::size()
        .ok()
        .map(|(w, h)| (w as usize, h as usize))
}

/// Re-export of `bevy_ratatui`'s mouse-event message type.
///
/// Downstream consumers (e.g. the level editor `ts_editor`) that replicate the
/// game's mouse-look path can name this type without adding a direct
/// `bevy_ratatui` or `crossterm` dependency. The crossterm `MouseEventKind` enum
/// and the cell coordinates are accessible on it via [`mouse_motion_cell`] —
/// keeping the crossterm surface confined to `re_present`.
#[cfg(feature = "backend-terminal")]
pub use bevy_ratatui::event::MouseMessage;

/// Extract a cell-coordinate mouse-motion position from a [`MouseMessage`].
///
/// Returns `Some((column, row))` — in terminal-cell units — when the event is a
/// `crossterm::event::MouseEventKind::Moved` or
/// `crossterm::event::MouseEventKind::Drag` (any button). Returns `None` for
/// every other kind (button press/release, scroll).
///
/// This mirrors the private `match ev.kind { Moved | Drag(_) => … }` guard in
/// `terminal_present_sys`'s mouse-look path, factored out as a pure,
/// terminal-free helper so the editor can reuse the same filtering logic without
/// importing `crossterm` directly. Field access on [`MouseMessage`] uses the
/// `Deref` impl to reach the inner `crossterm::event::MouseEvent`.
///
/// # Example (inside `re_present` or a consumer)
/// ```ignore
/// if let Some((col, row)) = re_present::mouse_motion_cell(&msg) {
///     // replicate the mouse-look delta logic here
/// }
/// ```
#[cfg(feature = "backend-terminal")]
#[must_use]
pub fn mouse_motion_cell(ev: &MouseMessage) -> Option<(u16, u16)> {
    use bevy_ratatui::crossterm::event::MouseEventKind;
    match ev.kind {
        MouseEventKind::Moved | MouseEventKind::Drag(_) => Some((ev.column, ev.row)),
        _ => None,
    }
}

// --- Editor input seam (backend-neutral raw keys for ts_editor). ---

#[cfg(feature = "editor")]
pub mod editor_input;
#[cfg(feature = "editor")]
pub use editor_input::{EditorKey, EditorKeyCode, EditorMouse, EditorMouseButton, EditorMouseKind};

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use re_world::{stamp_geometry, CameraPose, CellKind, GeometrySpec};

    /// Open 5×5×4 room: perimeter walls, floor z=0, ceiling z=3, air at z=1..=2.
    fn open_room() -> re_world::Map {
        stamp_geometry(&GeometrySpec {
            size: (5, 5, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        })
    }

    /// The exact horizontal-move call the input systems make: forward by
    /// `MOVE_STEP` along facing, COLLISION-CHECKED. Mirrors the per-frame wiring
    /// so the integration (not the physics math) is what's under test.
    fn step_forward(cam: &mut CameraPose, map: &re_world::Map) -> (bool, bool) {
        let (dx, dy) = cam.dir();
        cam.move_with_collision(dx * MOVE_STEP, dy * MOVE_STEP, map, PLAYER_RADIUS)
    }

    #[test]
    fn forward_move_into_wall_is_blocked_by_collision() {
        // Stand just inside the east wall (perimeter at x=4) facing +x and walk
        // into it. With the OLD raw `pos += dir*step` this would tunnel through;
        // the wired `move_with_collision` must block the x axis.
        let map = open_room();
        // yaw for facing +x: dir = (sin yaw, cos yaw) = (1,0) → yaw = PI/2.
        let mut cam = CameraPose {
            pos: Vec3::new(3.5, 2.5, 1.5),
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
        };
        let (dx, _) = cam.dir();
        assert!(dx > 0.9, "test setup: camera must face +x, dx={dx}");

        // Walk forward several frames; the wall must stop us short of x=4.
        for _ in 0..20 {
            step_forward(&mut cam, &map);
        }
        assert!(
            cam.pos.x + PLAYER_RADIUS <= 4.0 + 1e-4,
            "x={} tunnelled into the east wall — collision not wired",
            cam.pos.x
        );
    }

    #[test]
    fn forward_move_in_open_space_advances() {
        // The same wiring must still let the camera move when nothing blocks it.
        // Start at the room CENTRE (2.5, 2.5) so a single MOVE_STEP toward the
        // −y wall stays well clear of it (the wall blocks at y < 1+PLAYER_RADIUS).
        let map = open_room();
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 1.5),
            yaw: std::f32::consts::PI, // face -y (+PI faces -y; the y=1 wall is far)
            pitch: 0.0,
        };
        let y0 = cam.pos.y;
        let (mx, my) = step_forward(&mut cam, &map);
        assert!(
            my,
            "open-space forward move should advance y, got moved_y=false"
        );
        assert!(
            (cam.pos.y - (y0 - MOVE_STEP)).abs() < 1e-5,
            "y should step by MOVE_STEP"
        );
        let _ = mx;
    }

    #[test]
    fn jump_only_when_grounded_integration() {
        // The exact jump wiring: try_jump on a Space/face-button press. It must
        // set an upward velocity when grounded, and NOT grant a second jump
        // to an immediate second press mid-air (re_world's coyote-time /
        // jump-buffer envelope — Swaggerfall-study item 5 — is exercised in
        // `re_world`'s own test suite; this integration test only pins the
        // call site's unconditional, ungated wiring).
        let mut motion = PlayerMotion {
            vz: 0.0,
            grounded: true,
            ..Default::default()
        };
        assert!(CameraPose::try_jump(&mut motion), "grounded → jump fires");
        assert!(
            motion.vz > 0.0,
            "jump must set upward velocity, vz={}",
            motion.vz
        );
        assert!(!motion.grounded, "jump clears grounded");

        // A second press immediately mid-air does not double-jump (blocked by
        // `jumped_airborne`, not by `since_grounded` — that is still inside
        // the coyote window right after a jump).
        let vz_before = motion.vz;
        assert!(
            !CameraPose::try_jump(&mut motion),
            "mid-air → no double jump"
        );
        assert_eq!(
            motion.vz, vz_before,
            "denied press must not change velocity"
        );
    }

    #[test]
    fn gravity_tick_grounds_a_dropped_camera() {
        // Wiring the per-frame `step_gravity` makes the camera grounded by
        // default: drop it from an air layer and it settles on the floor.
        let map = open_room();
        let mut cam = CameraPose {
            pos: Vec3::new(2.5, 2.5, 2.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut motion = PlayerMotion::default();
        for _ in 0..200 {
            cam.step_gravity(&mut motion, 0.01, &map);
        }
        assert!(motion.grounded, "step_gravity should ground the camera");
        assert!(
            (cam.pos.z - 1.5).abs() < 0.05,
            "settled at floor height, z={}",
            cam.pos.z
        );
    }

    #[test]
    fn rgb8_unpacks_channels() {
        assert_eq!(rgb8(0x00FF_8040), (0xFF, 0x80, 0x40));
        assert_eq!(rgb8(0x0000_0000), (0, 0, 0));
        assert_eq!(rgb8(0x00FF_FFFF), (0xFF, 0xFF, 0xFF));
    }

    #[test]
    fn glyph_char_decodes_and_falls_back() {
        assert_eq!(glyph_char(u32::from('#')), '#');
        assert_eq!(glyph_char(u32::from(' ')), ' ');
        // 0xD800 is a lone surrogate — not a valid scalar → space.
        assert_eq!(glyph_char(0xD800), ' ');
        // Control characters must NOT pass through (ratatui's cell_width panics on
        // them). glyph 0 (NUL — the default/unwritten cell) is the real-world case
        // that crashed the live present; tab/CR/DEL/C1 are covered too.
        assert_eq!(glyph_char(0), ' ', "NUL (default cell) → space, not '\\0'");
        assert_eq!(glyph_char(u32::from('\t')), ' ');
        assert_eq!(glyph_char(u32::from('\r')), ' ');
        assert_eq!(glyph_char(0x7F), ' ', "DEL → space");
        assert_eq!(glyph_char(0x9B), ' ', "C1 control → space");
    }

    #[test]
    fn mouse_look_turns_yaw_by_sensitivity() {
        // Positive dx turns yaw right by dx * MOUSE_YAW_SENS; dy=0 leaves pitch.
        let (yaw, pitch) = apply_mouse_look(1.0, 0.0, 3.0, 0.0, 1.0);
        assert!((yaw - (1.0 + 3.0 * MOUSE_YAW_SENS)).abs() < 1e-6);
        assert_eq!(pitch, 0.0);
        // Negative dx turns the other way.
        let (yaw, _) = apply_mouse_look(0.0, 0.0, -2.0, 0.0, 1.0);
        assert!((yaw + 2.0 * MOUSE_YAW_SENS).abs() < 1e-6);
    }

    #[test]
    fn mouse_look_scales_by_sensitivity_multiplier() {
        // A 2.0 multiplier turns yaw exactly twice as far as the base feel.
        let (base, _) = apply_mouse_look(0.0, 0.0, 1.0, 0.0, 1.0);
        let (fast, _) = apply_mouse_look(0.0, 0.0, 1.0, 0.0, 2.0);
        assert!((fast - 2.0 * base).abs() < 1e-6, "2x mult = 2x yaw");
        // A 0.0 multiplier (e.g. a hypothetical zero-sens) freezes the camera.
        let (frozen_y, frozen_p) = apply_mouse_look(0.3, 0.1, 5.0, 5.0, 0.0);
        assert_eq!((frozen_y, frozen_p), (0.3, 0.1));
    }

    #[test]
    fn mouse_settings_default_is_historical_feel() {
        let s = MouseSettings::default();
        assert!(s.enabled, "mouse-look defaults on");
        assert_eq!(s.sensitivity, 1.0, "default = base (Medium) feel");
    }

    #[test]
    fn mouse_look_pitch_is_inverted() {
        // Mouse DOWN (dy > 0, screen-y grows downward) looks DOWN (pitch < 0).
        let (_, pitch) = apply_mouse_look(0.0, 0.0, 0.0, 2.0, 1.0);
        assert!(pitch < 0.0, "mouse-down should look down, got {pitch}");
        // Mouse UP (dy < 0) looks up (pitch > 0).
        let (_, pitch) = apply_mouse_look(0.0, 0.0, 0.0, -2.0, 1.0);
        assert!(pitch > 0.0, "mouse-up should look up, got {pitch}");
    }

    #[test]
    fn mouse_look_clamps_pitch_to_limit() {
        // A huge downward drag saturates at -PITCH_LIMIT, never beyond.
        let (_, pitch) = apply_mouse_look(0.0, 0.0, 0.0, 10_000.0, 1.0);
        assert!((pitch + PITCH_LIMIT).abs() < 1e-6, "pitch={pitch}");
        // A huge upward drag saturates at +PITCH_LIMIT.
        let (_, pitch) = apply_mouse_look(0.0, 0.0, 0.0, -10_000.0, 1.0);
        assert!((pitch - PITCH_LIMIT).abs() < 1e-6, "pitch={pitch}");
        // And the band is exactly ±π/3 (60°) — the widened, user-approved compromise,
        // matching `re_render::PITCH_LIMIT_RAD` so the look saturates exactly where the
        // raycaster's horizon shift does (no frozen-look dead band, full fill to clamp).
        assert!((PITCH_LIMIT - std::f32::consts::FRAC_PI_3).abs() < 1e-6);
    }

    // --- KeyIntent mapping (the player-bug-fix key map, backend-neutral) ---

    #[test]
    fn wasd_are_all_movement_not_turn() {
        // A/D must be STRAFE intents (movement), NEVER a turn. This is the core of
        // player bug #1 — A/D used to change yaw.
        assert_eq!(KeyIntent::StrafeLeft.menu_nav(), Some(MenuNav::Left));
        assert_eq!(KeyIntent::StrafeRight.menu_nav(), Some(MenuNav::Right));
        // And they are distinct from the turn intents.
        assert_ne!(KeyIntent::StrafeLeft, KeyIntent::TurnLeft);
        assert_ne!(KeyIntent::StrafeRight, KeyIntent::TurnRight);
        // W/S are forward/back movement.
        assert_eq!(KeyIntent::Forward.menu_nav(), Some(MenuNav::Up));
        assert_eq!(KeyIntent::Backward.menu_nav(), Some(MenuNav::Down));
    }

    #[test]
    fn arrows_turn_and_still_navigate_menus() {
        // Turn/pitch intents (the arrow keys) ALSO drive MenuNav so menus work.
        assert_eq!(KeyIntent::TurnLeft.menu_nav(), Some(MenuNav::Left));
        assert_eq!(KeyIntent::TurnRight.menu_nav(), Some(MenuNav::Right));
        assert_eq!(KeyIntent::PitchUp.menu_nav(), Some(MenuNav::Up));
        assert_eq!(KeyIntent::PitchDown.menu_nav(), Some(MenuNav::Down));
    }

    #[test]
    fn back_and_confirm_map_to_nav() {
        assert_eq!(KeyIntent::Back.menu_nav(), Some(MenuNav::Back));
        assert_eq!(KeyIntent::Interact.menu_nav(), Some(MenuNav::Confirm));
        assert_eq!(KeyIntent::Jump.menu_nav(), Some(MenuNav::Confirm));
    }

    #[test]
    fn unmapped_and_fov_emit_no_nav() {
        // An unmapped key → no nav. (`q`/`Q` are now BOUND to Back at the
        // backend key-table level — see `terminal::tests::q_maps_to_back` — so they
        // are no longer Unmapped; this only asserts the Unmapped→None invariant.)
        assert_eq!(KeyIntent::Unmapped.menu_nav(), None);
        // FOV is an action, not navigation.
        assert_eq!(KeyIntent::Fov(1).menu_nav(), None);
        assert_eq!(KeyIntent::Fov(-1).menu_nav(), None);
    }

    #[test]
    fn mouse_look_accumulates_over_steps() {
        // Applying two successive deltas equals applying their sum (pre-clamp),
        // proving the per-step relative-delta model composes as expected.
        let (y1, p1) = apply_mouse_look(0.5, 0.0, 1.0, 1.0, 1.0);
        let (y2, p2) = apply_mouse_look(y1, p1, 2.0, 1.0, 1.0);
        let (ys, ps) = apply_mouse_look(0.5, 0.0, 3.0, 2.0, 1.0);
        assert!((y2 - ys).abs() < 1e-6);
        assert!((p2 - ps).abs() < 1e-6);
    }

    // --- Held-set simultaneous movement (the kitty held-key path) ---

    /// `HeldKeys::set` populates exactly the movement/turn flag for each
    /// continuous intent and IGNORES the one-shot intents (so a held key never
    /// latches jump/interact/etc into a "continuous" state).
    #[test]
    fn held_set_tracks_only_continuous_intents() {
        let mut h = HeldKeys::default();
        assert!(h.set(KeyIntent::Forward, true));
        assert!(h.set(KeyIntent::StrafeLeft, true));
        assert!(h.forward && h.strafe_left);
        // One-shots are not continuous → set() reports false and changes nothing.
        let before = h;
        assert!(!h.set(KeyIntent::Jump, true));
        assert!(!h.set(KeyIntent::Interact, true));
        assert!(!h.set(KeyIntent::Fov(1), true));
        assert!(!h.set(KeyIntent::Back, true));
        assert!(!h.set(KeyIntent::MouseLookToggle, true));
        assert!(!h.set(KeyIntent::Unmapped, true));
        assert_eq!(h, before, "one-shot intents must not touch the held set");
        // Clearing works (the Release edge).
        assert!(h.set(KeyIntent::Forward, false));
        assert!(!h.forward);
    }

    /// TWO held keys (W+A) combine into a single DIAGONAL vector, and it is
    /// normalized to `speed*dt` — a diagonal is NOT √2 faster than a cardinal.
    #[test]
    fn two_held_keys_make_a_normalized_diagonal() {
        // Face +y: dir=(0,1). Forward=+y, strafe-left=−x.
        let (dirx, diry) = (0.0_f32, 1.0_f32);
        let speed = 7.0;
        let dt = 0.1;
        let h = HeldKeys {
            forward: true,
            strafe_left: true,
            ..Default::default()
        };
        let (dx, dy) = held_move_delta(h, dirx, diry, speed, dt);
        // Diagonal: −x (strafe-left) and +y (forward), equal magnitude.
        assert!(dx < 0.0, "W+A should carry a −x (left) component, dx={dx}");
        assert!(
            dy > 0.0,
            "W+A should carry a +y (forward) component, dy={dy}"
        );
        assert!(
            (dx.abs() - dy.abs()).abs() < 1e-6,
            "45° diagonal: |dx|==|dy|"
        );
        // Magnitude is exactly speed*dt (normalized) — same as a single key.
        let mag = (dx * dx + dy * dy).sqrt();
        assert!(
            (mag - speed * dt).abs() < 1e-5,
            "diagonal speed == cardinal speed, mag={mag}"
        );

        // A single forward key alone has the SAME magnitude (not slower/faster).
        let fwd_only = HeldKeys {
            forward: true,
            ..Default::default()
        };
        let (sx, sy) = held_move_delta(fwd_only, dirx, diry, speed, dt);
        let smag = (sx * sx + sy * sy).sqrt();
        assert!(
            (smag - mag).abs() < 1e-5,
            "cardinal and diagonal speeds match"
        );
    }

    /// Releasing ONE of two held keys leaves the OTHER still driving movement —
    /// the core "any press/release order" guarantee.
    #[test]
    fn releasing_one_key_keeps_the_other_moving() {
        let (dirx, diry) = (0.0_f32, 1.0_f32); // facing +y
        let speed = 7.0;
        let dt = 0.1;
        // Hold W+A (diagonal), then release A → pure forward (+y), no left component.
        let mut h = HeldKeys {
            forward: true,
            strafe_left: true,
            ..Default::default()
        };
        h.set(KeyIntent::StrafeLeft, false);
        let (dx, dy) = held_move_delta(h, dirx, diry, speed, dt);
        assert!(
            dx.abs() < 1e-6,
            "after releasing A, no lateral motion, dx={dx}"
        );
        assert!(
            (dy - speed * dt).abs() < 1e-5,
            "forward continues at full speed, dy={dy}"
        );
        // Now release W too → fully stopped.
        h.set(KeyIntent::Forward, false);
        assert_eq!(held_move_delta(h, dirx, diry, speed, dt), (0.0, 0.0));
    }

    /// Opposing held keys (W+S, A+D) cancel to zero net movement.
    #[test]
    fn opposing_held_keys_cancel() {
        let (dirx, diry) = (0.0_f32, 1.0_f32);
        let wsad = HeldKeys {
            forward: true,
            backward: true,
            strafe_left: true,
            strafe_right: true,
            ..Default::default()
        };
        assert_eq!(held_move_delta(wsad, dirx, diry, 7.0, 0.1), (0.0, 0.0));
    }

    /// The strafe BASIS: held strafe-right moves perpendicular to facing (facing
    /// rotated −90°), with no forward component; strafe-left is the opposite.
    #[test]
    fn held_strafe_basis_is_perpendicular_to_facing() {
        // Face +y: dir=(0,1). Strafe-right basis (diry,-dirx)=(1,0) → +x.
        let (dirx, diry) = (0.0_f32, 1.0_f32);
        let right = HeldKeys {
            strafe_right: true,
            ..Default::default()
        };
        let (rx, ry) = held_move_delta(right, dirx, diry, 7.0, 0.1);
        assert!(rx > 0.0, "strafe-right → +x, rx={rx}");
        assert!(ry.abs() < 1e-6, "strafe has no forward component, ry={ry}");
        // Strafe-left is the mirror.
        let left = HeldKeys {
            strafe_left: true,
            ..Default::default()
        };
        let (lx, _) = held_move_delta(left, dirx, diry, 7.0, 0.1);
        assert!(lx < 0.0, "strafe-left → −x, lx={lx}");
        assert!((lx + rx).abs() < 1e-6, "left and right are exact opposites");
    }

    /// Movement is TIME-BASED: the per-frame delta scales linearly with both
    /// `speed` and `dt` (frame-rate independent), so a faster `MOVE_SPEED` moves
    /// proportionally faster and a 2× longer frame moves 2× as far.
    #[test]
    fn held_move_scales_with_speed_and_dt() {
        let (dirx, diry) = (0.0_f32, 1.0_f32);
        let h = HeldKeys {
            forward: true,
            ..Default::default()
        };
        let (_, base) = held_move_delta(h, dirx, diry, 7.0, 0.1);
        // Double the speed → double the distance.
        let (_, fast) = held_move_delta(h, dirx, diry, 14.0, 0.1);
        assert!((fast - 2.0 * base).abs() < 1e-5, "2× speed = 2× distance");
        // Double the dt → double the distance (frame-rate independence).
        let (_, long_frame) = held_move_delta(h, dirx, diry, 7.0, 0.2);
        assert!(
            (long_frame - 2.0 * base).abs() < 1e-5,
            "2× dt = 2× distance"
        );
        // Walk speed is deliberately deliberate; sprint reaches ~SPRINT_MULTIPLIER×.
        let ms = MOVE_SPEED;
        assert!(ms > 1.0, "MOVE_SPEED should be at least 1 cell/s, got {ms}");
        assert!(
            ms < 6.0,
            "MOVE_SPEED (walk) reduced from 7.0 → 2.3; got {ms}"
        );
    }

    /// Held turn: ←/→ yaw (− / +), ↑/↓ pitch (+ / −), scaled by `speed*dt`;
    /// opposing keys cancel. (Pitch CLAMP is the caller's job — this returns the
    /// raw delta.)
    #[test]
    fn held_turn_delta_signs_and_scaling() {
        let speed = 2.4;
        let dt = 0.1;
        // Turn right → +yaw; turn left → −yaw.
        let r = HeldKeys {
            turn_right: true,
            ..Default::default()
        };
        let (yaw_r, _) = held_turn_delta(r, speed, dt);
        assert!((yaw_r - speed * dt).abs() < 1e-6, "turn-right = +speed*dt");
        let l = HeldKeys {
            turn_left: true,
            ..Default::default()
        };
        let (yaw_l, _) = held_turn_delta(l, speed, dt);
        assert!((yaw_l + speed * dt).abs() < 1e-6, "turn-left = −speed*dt");
        // Pitch up → +; down → −.
        let u = HeldKeys {
            pitch_up: true,
            ..Default::default()
        };
        let (_, p_up) = held_turn_delta(u, speed, dt);
        assert!(p_up > 0.0, "pitch-up positive");
        let d = HeldKeys {
            pitch_down: true,
            ..Default::default()
        };
        let (_, p_dn) = held_turn_delta(d, speed, dt);
        assert!(p_dn < 0.0, "pitch-down negative");
        // Opposing cancel.
        let both = HeldKeys {
            turn_left: true,
            turn_right: true,
            pitch_up: true,
            pitch_down: true,
            ..Default::default()
        };
        assert_eq!(held_turn_delta(both, speed, dt), (0.0, 0.0));
    }

    /// `is_empty` is the per-frame early-out: true with nothing held, false once
    /// any movement/turn key is down.
    #[test]
    fn held_is_empty_gate() {
        assert!(HeldKeys::default().is_empty());
        let h = HeldKeys {
            forward: true,
            ..Default::default()
        };
        assert!(!h.is_empty());
    }

    // --- HeldTimestamps expiry safety-net (deterministic, no ECS, no terminal) ---

    /// Simulates the "legacy terminal" runaway scenario: a key-Press sets a held
    /// flag and refreshes its timestamp, but NO Release ever arrives. After
    /// `HELD_EXPIRE_SECS` has elapsed (we advance `now` manually — no real clock,
    /// deterministic), `expire` MUST clear the held flag. This is the exact defect
    /// the safety-net fixes: on a terminal that ignores kitty bytes, holding `W`
    /// would latch `forward = true` forever without this.
    #[test]
    fn held_expires_after_timeout_when_no_release_arrives() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();

        // Simulate a Press at t=0.0.
        let press_time: f64 = 0.0;
        held.set(KeyIntent::Forward, true);
        ts.refresh(KeyIntent::Forward, press_time);
        assert!(held.forward, "forward must be set after the press");

        // At t=0.5 s (within the window) — flag is still live.
        ts.expire(&mut held, 0.5);
        assert!(
            held.forward,
            "forward must NOT be expired at 0.5 s (window is {HELD_EXPIRE_SECS} s)"
        );

        // At t = press_time + HELD_EXPIRE_SECS (exactly on the threshold) — still live
        // (the check is `>=`, so we need strictly past to expire ... actually
        // our macro uses `>=` so the threshold itself is expired).
        ts.expire(&mut held, press_time + HELD_EXPIRE_SECS);
        assert!(
            !held.forward,
            "forward MUST be expired at exactly HELD_EXPIRE_SECS ({HELD_EXPIRE_SECS} s)"
        );
        assert_eq!(
            ts.forward, None,
            "timestamp slot must be cleared after expiry"
        );
    }

    /// Verifies that a refresh within the window (simulating OS autorepeat Press
    /// events that a non-kitty terminal sends during an active hold) keeps the
    /// flag alive. This is the "no stutter on active holds" guarantee: the timer
    /// is pushed back by each Press/Repeat, so a sustained hold never expires.
    #[test]
    fn held_does_not_expire_when_refreshed_within_window() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();

        // Press at t=0.0.
        held.set(KeyIntent::StrafeLeft, true);
        ts.refresh(KeyIntent::StrafeLeft, 0.0);

        // OS autorepeat fires at t=0.5 (within the 0.75 s window) — refresh it.
        ts.refresh(KeyIntent::StrafeLeft, 0.5);

        // At t=1.1 s: 1.1 - 0.5 = 0.6 s since last refresh → still inside window.
        ts.expire(&mut held, 1.1);
        assert!(
            held.strafe_left,
            "strafe_left must NOT expire if last refresh was only 0.6 s ago (window {HELD_EXPIRE_SECS} s)"
        );

        // A second autorepeat at t=1.1 s pushes the window forward again.
        ts.refresh(KeyIntent::StrafeLeft, 1.1);

        // At t=1.8 s: 1.8 - 1.1 = 0.7 s — still inside the window.
        ts.expire(&mut held, 1.8);
        assert!(
            held.strafe_left,
            "strafe_left must still be alive at 0.7 s since last refresh"
        );

        // Now no more refreshes. At t=1.1 + HELD_EXPIRE_SECS the timeout fires.
        ts.expire(&mut held, 1.1 + HELD_EXPIRE_SECS);
        assert!(
            !held.strafe_left,
            "strafe_left must expire after HELD_EXPIRE_SECS ({HELD_EXPIRE_SECS} s) without refresh"
        );
    }

    /// Confirms that on the kitty path a Release immediately clears both the flag
    /// and the timestamp, making expire a no-op on subsequent calls — which is
    /// exactly the "no-op on capable terminals" invariant.
    #[test]
    fn release_clears_timestamp_and_expire_is_noop_afterwards() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();

        // Press at t=0.0, Release at t=0.1 (well within the window).
        held.set(KeyIntent::TurnRight, true);
        ts.refresh(KeyIntent::TurnRight, 0.0);
        // Release: clear flag + timestamp (mirroring what input_sys does on kitty).
        held.set(KeyIntent::TurnRight, false);
        ts.clear(KeyIntent::TurnRight);

        // Even far past the timeout, expire does NOTHING (already cleared).
        ts.expire(&mut held, 0.0 + HELD_EXPIRE_SECS + 10.0);
        assert!(
            !held.turn_right,
            "turn_right must stay clear after Release + expire"
        );
        assert_eq!(
            ts.turn_right, None,
            "timestamp slot stays None after Release"
        );
        // And HeldKeys is entirely empty — no runaway.
        assert!(
            held.is_empty(),
            "HeldKeys must be empty after Release + expire"
        );
    }

    /// One-shot intents (Jump, Interact, Back, Fov, TogglePerf, Unmapped) must NOT
    /// populate any timestamp slot — they are not continuous held actions.
    #[test]
    fn one_shot_intents_do_not_populate_timestamps() {
        let mut ts = HeldTimestamps::default();
        let one_shots = [
            KeyIntent::Jump,
            KeyIntent::Interact,
            KeyIntent::Back,
            KeyIntent::Fov(1),
            KeyIntent::Fov(-1),
            KeyIntent::TogglePerf,
            KeyIntent::MouseLookToggle,
            KeyIntent::Unmapped,
        ];
        for intent in one_shots {
            ts.refresh(intent, 99.0);
        }
        assert_eq!(
            ts,
            HeldTimestamps::default(),
            "one-shot intents must not touch any timestamp slot"
        );
    }

    // -----------------------------------------------------------------------
    // Sprint modifier (Task 1 & 2) — pure held flag, never disrupts WASD.
    // -----------------------------------------------------------------------

    /// Sprint is a CONTINUOUS intent: `HeldKeys::set(Sprint, true)` latches the
    /// sprint flag, and `set(Sprint, false)` clears it. Returns `true` from `set`
    /// so the timestamp is refreshed exactly like WASD keys.
    #[test]
    fn sprint_is_tracked_in_held_set() {
        let mut h = HeldKeys::default();
        // Sprint is a continuous intent → set() returns true and latches the flag.
        assert!(h.set(KeyIntent::Sprint, true), "Sprint must be continuous");
        assert!(h.sprint, "sprint flag must be set");
        // Releasing clears it.
        assert!(h.set(KeyIntent::Sprint, false));
        assert!(!h.sprint, "sprint flag must clear on Release");
    }

    /// Pressing Sprint (Shift) NEVER clears or modifies WASD/arrow movement flags.
    /// This is the core non-interference guarantee for Task 2.
    #[test]
    fn sprint_press_does_not_clear_movement_flags() {
        let mut h = HeldKeys {
            forward: true,
            strafe_left: true,
            ..Default::default()
        };
        // Press Shift (sprint) while W+A are held.
        h.set(KeyIntent::Sprint, true);
        assert!(h.forward, "W must still be held after pressing Shift");
        assert!(h.strafe_left, "A must still be held after pressing Shift");
        assert!(h.sprint, "sprint flag must now be set");
        // Release Shift.
        h.set(KeyIntent::Sprint, false);
        assert!(h.forward, "W must still be held after releasing Shift");
        assert!(h.strafe_left, "A must still be held after releasing Shift");
        assert!(!h.sprint, "sprint flag must be clear after Release");
    }

    /// Pressing Space (Jump) on the kitty path is a one-shot and NEVER clears
    /// movement flags — the `set()` call for Jump returns false and is a no-op
    /// in the held set, so W/A/S/D remain live.
    #[test]
    fn space_press_does_not_clear_movement_flags() {
        let mut h = HeldKeys {
            forward: true,
            strafe_right: true,
            ..Default::default()
        };
        // Simulate the kitty path: Jump is not continuous so set() is a no-op.
        let changed = h.set(KeyIntent::Jump, true);
        assert!(
            !changed,
            "Jump must NOT enter the held set (it is a one-shot)"
        );
        assert!(h.forward, "W must still be held after Jump press");
        assert!(h.strafe_right, "D must still be held after Jump press");
    }

    /// Sprint multiplies the effective speed: `held_move_delta` called with
    /// `speed * SPRINT_MULTIPLIER` gives exactly `SPRINT_MULTIPLIER` times the
    /// walk distance. (The caller is responsible for applying the multiplier
    /// when `held.sprint` is true — this test verifies the math is correct.)
    #[test]
    fn sprint_multiplier_scales_move_delta() {
        let (dirx, diry) = (0.0_f32, 1.0_f32); // facing +y
        let h = HeldKeys {
            forward: true,
            ..Default::default()
        };
        let dt = 1.0 / 60.0;
        // Walk speed.
        let (_, walk_dy) = held_move_delta(h, dirx, diry, MOVE_SPEED, dt);
        // Sprint speed (caller applies multiplier to speed before passing it in).
        let (_, sprint_dy) = held_move_delta(h, dirx, diry, MOVE_SPEED * SPRINT_MULTIPLIER, dt);
        assert!(
            (sprint_dy / walk_dy - SPRINT_MULTIPLIER).abs() < 1e-5,
            "sprint delta must be exactly SPRINT_MULTIPLIER × walk delta, \
             walk={walk_dy} sprint={sprint_dy} ratio={}",
            sprint_dy / walk_dy
        );
    }

    /// Sprint flag has a timestamp slot — it can be expired by the safety-net
    /// like any movement key. Pressing Sprint refreshes the slot; on a legacy
    /// terminal without Release events the flag will auto-clear after HELD_EXPIRE_SECS.
    #[test]
    fn sprint_timestamp_expires_like_movement_keys() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();

        held.set(KeyIntent::Sprint, true);
        ts.refresh(KeyIntent::Sprint, 0.0);
        assert!(held.sprint, "sprint must be set after press");

        // Within the window — still alive.
        ts.expire(&mut held, 0.5);
        assert!(
            held.sprint,
            "sprint must not expire at 0.5 s (window = {HELD_EXPIRE_SECS} s)"
        );

        // At exactly the threshold — expires.
        ts.expire(&mut held, HELD_EXPIRE_SECS);
        assert!(
            !held.sprint,
            "sprint must expire after {HELD_EXPIRE_SECS} s without refresh"
        );
        assert_eq!(
            ts.sprint, None,
            "timestamp slot must be cleared after expiry"
        );
    }

    /// Sprint emits NO MenuNav: Shift must never navigate menus.
    #[test]
    fn sprint_emits_no_menu_nav() {
        assert_eq!(
            KeyIntent::Sprint.menu_nav(),
            None,
            "Sprint must not emit any MenuNav"
        );
    }

    // -----------------------------------------------------------------------
    // Mouse sensitivity halved (Task 3) — the new base constants.
    // -----------------------------------------------------------------------

    /// Verify the new MOUSE_YAW_SENS and MOUSE_PITCH_SENS are exactly half the
    /// previous values (0.10→0.05, 0.08→0.04).
    #[test]
    fn mouse_sensitivity_constants_are_halved() {
        assert!(
            (MOUSE_YAW_SENS - 0.05).abs() < 1e-6,
            "MOUSE_YAW_SENS must be 0.05 (halved from 0.10), got {MOUSE_YAW_SENS}"
        );
        assert!(
            (MOUSE_PITCH_SENS - 0.04).abs() < 1e-6,
            "MOUSE_PITCH_SENS must be 0.04 (halved from 0.08), got {MOUSE_PITCH_SENS}"
        );
    }

    /// At the new base sensitivity, a 1-cell horizontal drag turns yaw by 0.05 rad
    /// (half the previous 0.10), confirming the playtest fix is applied end-to-end.
    #[test]
    fn mouse_yaw_sensitivity_at_medium_preset_is_halved() {
        // Medium preset multiplier = 1.0 (unchanged).
        let (yaw, _) = apply_mouse_look(0.0, 0.0, 1.0, 0.0, 1.0);
        assert!(
            (yaw - 0.05).abs() < 1e-6,
            "1-cell rightward drag at Medium must yield 0.05 rad yaw, got {yaw}"
        );
    }

    // -----------------------------------------------------------------------
    // Repro-scenario tests: cross-key independence on the kitty held-key path.
    //
    // These drive the pure HeldKeys / HeldTimestamps model with synthetic
    // kitty Press / Release sequences — no ECS, no live terminal. Each test
    // corresponds to a documented player repro scenario and verifies that
    // pressing or releasing a DIFFERENT key never clears or expires the held
    // movement flag of the ORIGINAL key.
    //
    // On the kitty path:
    //   Press  X → held.set(X, true)  + ts.refresh(X, now)
    //   Repeat X → held.set(X, true)  + ts.refresh(X, now)
    //   Release X → held.set(X, false) + ts.clear(X)
    //
    // Jump (Space) and Sprint (Shift) are one-shots / pure modifiers on the
    // held-set side: held.set(Jump/Sprint, …) for Jump is a no-op; Sprint
    // only touches the sprint flag. ts.refresh/clear for both is also a no-op
    // (no timestamp slot). So neither can touch the Forward/StrafeLeft/…
    // flags that movement depends on.
    // -----------------------------------------------------------------------

    /// REPRO 1: Hold W (forward), tap Space (jump) then release it.
    /// → W must KEEP moving (jump must not stop forward).
    ///
    /// Kitty path simulation:
    ///   Press W → held.forward = true, ts.forward refreshed
    ///   Press Space → held.set(Jump, true) = no-op, ts.refresh(Jump) = no-op
    ///   Release Space → held.set(Jump, false) = no-op, ts.clear(Jump) = no-op
    ///   assert held.forward = true, ts.forward unchanged
    #[test]
    fn kitty_hold_w_tap_space_w_stays_held() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();
        let t0: f64 = 1.0;

        // Press W.
        held.set(KeyIntent::Forward, true);
        ts.refresh(KeyIntent::Forward, t0);
        assert!(held.forward, "W must be held after press");
        assert_eq!(ts.forward, Some(t0), "forward timestamp set");

        // Tap Space (jump) — kitty Press edge.
        held.set(KeyIntent::Jump, true); // no-op: Jump is a one-shot
        ts.refresh(KeyIntent::Jump, t0); // no-op: no slot for Jump

        // Release Space — kitty Release edge.
        held.set(KeyIntent::Jump, false); // no-op
        ts.clear(KeyIntent::Jump); // no-op

        // W must still be held; its timestamp must be unchanged.
        assert!(
            held.forward,
            "forward must still be set after Space press+release (jump must not stop W)"
        );
        assert_eq!(
            ts.forward,
            Some(t0),
            "forward timestamp must be unchanged by the Space press+release"
        );
        assert!(
            !held.sprint && !held.strafe_left && !held.backward,
            "no other movement flags must be dirty"
        );
    }

    /// REPRO 2: Hold W, tap A (strafe), release A.
    /// → after releasing A, W must STILL be moving (strafe release must not clear W).
    ///
    /// Kitty path simulation:
    ///   Press W → held.forward = true
    ///   Press A → held.strafe_left = true
    ///   Release A → held.strafe_left = false, ts.strafe_left = None
    ///   assert held.forward = true
    #[test]
    fn kitty_hold_w_tap_a_release_a_w_stays_held() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();
        let t_w: f64 = 0.0;
        let t_a: f64 = 0.1;
        let t_rel: f64 = 0.3;

        // Press W.
        held.set(KeyIntent::Forward, true);
        ts.refresh(KeyIntent::Forward, t_w);

        // Press A.
        held.set(KeyIntent::StrafeLeft, true);
        ts.refresh(KeyIntent::StrafeLeft, t_a);
        assert!(
            held.forward && held.strafe_left,
            "both W and A must be held"
        );

        // Release A — only clears strafe_left.
        held.set(KeyIntent::StrafeLeft, false);
        ts.clear(KeyIntent::StrafeLeft);
        let _ = t_rel; // available for expire checks below

        // W must still be held; strafe must be cleared.
        assert!(
            held.forward,
            "forward must still be set after releasing A — strafe release must not stop W"
        );
        assert!(
            !held.strafe_left,
            "strafe_left must be cleared after A release"
        );
        assert_eq!(
            ts.forward,
            Some(t_w),
            "forward timestamp must be unchanged by A press+release"
        );
        assert_eq!(
            ts.strafe_left, None,
            "strafe_left timestamp must be cleared after A release"
        );
    }

    /// REPRO 3: Hold W + A (forward-left diagonal). Release A. → W keeps going.
    ///
    /// Also verifies the held_move_delta output: after releasing A, the delta
    /// must be pure forward (no lateral component).
    #[test]
    fn kitty_diagonal_release_a_w_keeps_going() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();
        let t0: f64 = 0.0;

        // Hold W + A.
        held.set(KeyIntent::Forward, true);
        ts.refresh(KeyIntent::Forward, t0);
        held.set(KeyIntent::StrafeLeft, true);
        ts.refresh(KeyIntent::StrafeLeft, t0);

        // Release A.
        held.set(KeyIntent::StrafeLeft, false);
        ts.clear(KeyIntent::StrafeLeft);

        // W must still be held; movement delta must be pure forward.
        assert!(held.forward, "forward must survive A release");
        assert!(!held.strafe_left, "strafe must be cleared");

        // Compute the movement delta — should be forward only (no lateral).
        let (dirx, diry) = (0.0_f32, 1.0_f32); // facing +y
        let (dx, dy) = held_move_delta(held, dirx, diry, MOVE_SPEED, 1.0 / 60.0);
        assert!(
            dx.abs() < 1e-6,
            "after releasing A, lateral component must be zero, dx={dx}"
        );
        assert!(
            dy > 0.0,
            "forward component must remain after releasing A, dy={dy}"
        );
    }

    /// REPRO 4: Hold W, press Shift (sprint) then release it.
    /// → W must KEEP moving; sprint flag must clear on release; no movement flags
    ///    must be affected.
    ///
    /// Kitty path simulation:
    ///   Press W → held.forward = true
    ///   Press LeftShift → held.sprint = true
    ///   Release LeftShift → held.sprint = false
    ///   assert held.forward = true
    #[test]
    fn kitty_hold_w_tap_shift_w_stays_held() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();
        let t0: f64 = 0.0;
        let t1: f64 = 0.1;

        // Press W.
        held.set(KeyIntent::Forward, true);
        ts.refresh(KeyIntent::Forward, t0);

        // Press Shift (sprint) — kitty sends a dedicated modifier Press event.
        held.set(KeyIntent::Sprint, true);
        ts.refresh(KeyIntent::Sprint, t1);
        assert!(held.forward && held.sprint, "W and Shift must both be held");

        // Release Shift — kitty Release for LeftShift/RightShift.
        held.set(KeyIntent::Sprint, false);
        ts.clear(KeyIntent::Sprint);

        // W must still be held; sprint must be cleared.
        assert!(
            held.forward,
            "forward must still be set after releasing Shift — sprint release must not stop W"
        );
        assert!(!held.sprint, "sprint must be cleared after Shift release");
        assert_eq!(
            ts.forward,
            Some(t0),
            "forward timestamp must be unchanged by Shift press+release"
        );
        assert_eq!(
            ts.sprint, None,
            "sprint timestamp must be cleared after Shift release"
        );
    }

    /// REPRO 5 (complete multi-key sequence): W+A held, A released, then W also
    /// released. Verify each step transitions the held set correctly and no
    /// spurious flags are set or cleared.
    #[test]
    fn kitty_full_wa_press_release_sequence() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();

        // Press W, then A (diagonal).
        held.set(KeyIntent::Forward, true);
        ts.refresh(KeyIntent::Forward, 0.0);
        held.set(KeyIntent::StrafeLeft, true);
        ts.refresh(KeyIntent::StrafeLeft, 0.05);

        // Sanity: diagonal held set.
        assert!(
            held.forward && held.strafe_left,
            "W+A must be simultaneously held"
        );

        // Release A: only strafe clears.
        held.set(KeyIntent::StrafeLeft, false);
        ts.clear(KeyIntent::StrafeLeft);
        assert!(held.forward, "W must survive A release");
        assert!(!held.strafe_left, "A must be cleared by its own release");
        assert_eq!(ts.strafe_left, None, "strafe_left ts cleared");
        assert_eq!(ts.forward, Some(0.0), "forward ts must be unchanged");

        // Release W: forward clears.
        held.set(KeyIntent::Forward, false);
        ts.clear(KeyIntent::Forward);
        assert!(!held.forward, "W cleared by its own release");
        assert_eq!(ts.forward, None, "forward ts cleared");

        // All clear.
        assert!(
            held.is_empty(),
            "HeldKeys must be empty after both releases"
        );
    }

    // -----------------------------------------------------------------------
    // Non-kitty (conhost with forced KittyEnabled) timeout tests.
    //
    // On conhost, force_kitty_sys inserts KittyEnabled (the raw VT write
    // succeeds), so input_sys takes the KITTY branch and sets held.forward =
    // true on a W press. But Release events never arrive on conhost, so
    // timestamps are only refreshed by W's OS autorepeat events.
    //
    // When A is pressed while W is held:
    //  - OS stops autorepeating W, starts autorepeating A.
    //  - W gets no further events → timestamps.forward freezes at the last W
    //    autorepeat.
    //  - After HELD_EXPIRE_SECS, expire_held_sys clears held.forward.
    //
    // WHAT IS FIXED IN CODE: HELD_EXPIRE_SECS is 2.0 s (up from 0.75 s),
    // giving 1.5 s headroom above the 500 ms Windows initial repeat delay.
    // A typical directional tap (< 1.5 s) will not expire W before A is
    // released and W's autorepeat resumes.
    //
    // WHAT REMAINS AN OS LIMITATION (unavoidable):
    //  - If A is held for > HELD_EXPIRE_SECS (2.0 s) while W autorepeat is
    //    suppressed, W WILL expire and movement stops. There is no way to know
    //    W is still physically held without Release reporting.
    //  - The 2.0 s drift after a true key release on conhost is also an OS
    //    limitation — on proper kitty terminals the Release event clears it
    //    immediately.
    // -----------------------------------------------------------------------

    /// Confirms that on the non-kitty/conhost path (with forced KittyEnabled),
    /// pressing A while W is held does NOT directly clear W's timestamp or flag.
    /// W's timestamp is only as fresh as the last W event — A's press only
    /// refreshes A's own slot (strafe_left).
    #[test]
    fn non_kitty_a_press_does_not_touch_w_timestamp() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();

        // Simulate W press (force-kitty path: held.set IS called).
        held.set(KeyIntent::Forward, true);
        ts.refresh(KeyIntent::Forward, 0.0);

        // Simulate one W autorepeat (typical: arrives at ~500 ms).
        ts.refresh(KeyIntent::Forward, 0.5);

        // Simulate A press (OS suppresses further W autorepeat after this).
        held.set(KeyIntent::StrafeLeft, true);
        ts.refresh(KeyIntent::StrafeLeft, 0.6);

        // Verify: A press only touched strafe_left, not forward.
        assert_eq!(
            ts.forward,
            Some(0.5),
            "A press must NOT modify forward's timestamp — it stays at the last W event (0.5 s)"
        );
        assert_eq!(
            ts.strafe_left,
            Some(0.6),
            "A press sets strafe_left timestamp to its own event time"
        );
        assert!(held.forward, "W must still be flagged held");
        assert!(held.strafe_left, "A must be flagged held");
    }

    /// Within the new HELD_EXPIRE_SECS window (2.0 s), a quick tap of A while
    /// W is held must NOT cause W to expire — even if no W autorepeat arrived
    /// between the W press and A's press (e.g., W was pressed at t=0 and A
    /// pressed at t=0.4 before W's first autorepeat at ~0.5 s).
    #[test]
    fn non_kitty_quick_a_tap_does_not_expire_w() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();

        // W pressed at t=0 (no autorepeat yet — initial delay ~500 ms).
        held.set(KeyIntent::Forward, true);
        ts.refresh(KeyIntent::Forward, 0.0);

        // A pressed at t=0.4 s (before W's first autorepeat).
        held.set(KeyIntent::StrafeLeft, true);
        ts.refresh(KeyIntent::StrafeLeft, 0.4);

        // Simulate quick tap: A released at t=0.6 s (on non-kitty no Release
        // arrives, so A's flag stays until expire clears it; we just stop
        // refreshing A's timestamp). At t=0.6+HELD_EXPIRE_SECS, A expires.
        // But for W: last refreshed at t=0.0; expires at t=0.0+HELD_EXPIRE_SECS.

        // Check at t=1.0 s (1.0 s since W was pressed, well within 2.0 s window).
        ts.expire(&mut held, 1.0);
        assert!(
            held.forward,
            "W must NOT expire at t=1.0 s (HELD_EXPIRE_SECS={HELD_EXPIRE_SECS} s); \
             this was the bug at the old 0.75 s timeout where W expired at t=0.75 s"
        );
        // A is also still alive at t=1.0 s (last refreshed at 0.4, expires at 2.4).
        assert!(held.strafe_left, "A must also still be alive at t=1.0 s");

        // At t=0.0 + HELD_EXPIRE_SECS exactly: W expires (safety-net fires).
        ts.expire(&mut held, HELD_EXPIRE_SECS);
        assert!(
            !held.forward,
            "W must expire at t=HELD_EXPIRE_SECS ({HELD_EXPIRE_SECS} s) — \
             this is the unavoidable non-kitty OS limitation: once OS stops \
             autorepeating W, it can only survive for HELD_EXPIRE_SECS from its \
             last autorepeat event"
        );
    }

    /// On the kitty path (proper Release events), expire is a no-op — a
    /// Release clears the flag and timestamp long before the timeout fires.
    /// This test confirms the timeout increase does NOT change kitty behaviour.
    #[test]
    fn kitty_release_still_clears_before_expire_with_new_timeout() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();

        // W held via kitty path.
        held.set(KeyIntent::Forward, true);
        ts.refresh(KeyIntent::Forward, 0.0);

        // Proper kitty Release for W at t=0.3 s.
        held.set(KeyIntent::Forward, false);
        ts.clear(KeyIntent::Forward);

        // expire at t=1.0 s (past old 0.75 s, well within new 2.0 s) → still a no-op.
        ts.expire(&mut held, 1.0);
        assert!(
            !held.forward,
            "W must still be cleared after proper Release, even at t=1.0 s; \
             the larger timeout must not resurrect a Released key"
        );
        assert_eq!(
            ts.forward, None,
            "forward ts must remain None after Release"
        );

        // Even far past the new timeout: still a no-op.
        ts.expire(&mut held, HELD_EXPIRE_SECS + 10.0);
        assert!(
            !held.forward,
            "Released key must never come back from expire"
        );
    }

    /// Full independence: multiple keys held simultaneously, each with its own
    /// timestamp. expire only clears keys whose OWN timestamp expired, never a
    /// key whose timestamp was refreshed more recently. Verifies that the expiry
    /// macro in HeldTimestamps::expire is truly per-key and never cross-key.
    #[test]
    fn expire_is_per_key_never_cross_key() {
        let mut held = HeldKeys::default();
        let mut ts = HeldTimestamps::default();

        // W: last refreshed at t=0.0 (expires at HELD_EXPIRE_SECS).
        held.set(KeyIntent::Forward, true);
        ts.refresh(KeyIntent::Forward, 0.0);

        // A: last refreshed at t=1.5 s (expires at 1.5 + HELD_EXPIRE_SECS).
        held.set(KeyIntent::StrafeLeft, true);
        ts.refresh(KeyIntent::StrafeLeft, 1.5);

        // At t = HELD_EXPIRE_SECS: W expires, A does NOT (A's window runs to 3.5 s).
        ts.expire(&mut held, HELD_EXPIRE_SECS);
        assert!(
            !held.forward,
            "W must expire at its own threshold (t=HELD_EXPIRE_SECS)"
        );
        assert!(
            held.strafe_left,
            "A must NOT expire at t=HELD_EXPIRE_SECS — its window runs to 1.5+{HELD_EXPIRE_SECS} s"
        );

        // At t = 1.5 + HELD_EXPIRE_SECS: A now expires.
        ts.expire(&mut held, 1.5 + HELD_EXPIRE_SECS);
        assert!(!held.strafe_left, "A must expire at its own threshold");
        assert!(
            held.is_empty(),
            "HeldKeys must be empty after both expirations"
        );
    }

    // -----------------------------------------------------------------------
    // mouse_motion_cell — pure helper for the crossterm cell-event mouse-look
    // path. Confined to backend-terminal because MouseMessage lives there.
    // -----------------------------------------------------------------------

    /// `mouse_motion_cell` uses crossterm types that clippy's `disallowed_types`
    /// lint would flag in engine crates. Inside re_present itself the use is
    /// the intentional one — suppress the expected false-positive exactly as the
    /// `terminal` module's own tests do.
    #[cfg(feature = "backend-terminal")]
    #[allow(clippy::disallowed_types)]
    mod mouse_motion_cell_tests {
        use super::*;
        use bevy_ratatui::crossterm::event::{
            KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
        };

        /// Wrap a `MouseEvent` into a `MouseMessage` the same way
        /// `bevy_ratatui`'s event system does: `MouseMessage(event)`.
        fn make(kind: MouseEventKind, column: u16, row: u16) -> MouseMessage {
            MouseMessage(MouseEvent {
                kind,
                column,
                row,
                modifiers: KeyModifiers::NONE,
            })
        }

        #[test]
        fn moved_returns_coords() {
            let msg = make(MouseEventKind::Moved, 12, 34);
            assert_eq!(
                mouse_motion_cell(&msg),
                Some((12, 34)),
                "Moved must return Some((column, row))"
            );
        }

        #[test]
        fn drag_left_returns_coords() {
            let msg = make(MouseEventKind::Drag(MouseButton::Left), 5, 7);
            assert_eq!(
                mouse_motion_cell(&msg),
                Some((5, 7)),
                "Drag(Left) must return Some((column, row))"
            );
        }

        #[test]
        fn drag_right_returns_coords() {
            let msg = make(MouseEventKind::Drag(MouseButton::Right), 0, 0);
            assert_eq!(
                mouse_motion_cell(&msg),
                Some((0, 0)),
                "Drag(Right) must return Some((column, row))"
            );
        }

        #[test]
        fn button_down_returns_none() {
            let msg = make(MouseEventKind::Down(MouseButton::Left), 3, 3);
            assert_eq!(
                mouse_motion_cell(&msg),
                None,
                "Down must return None — not a motion event"
            );
        }

        #[test]
        fn button_up_returns_none() {
            let msg = make(MouseEventKind::Up(MouseButton::Left), 3, 3);
            assert_eq!(
                mouse_motion_cell(&msg),
                None,
                "Up must return None — not a motion event"
            );
        }

        #[test]
        fn scroll_up_returns_none() {
            let msg = make(MouseEventKind::ScrollUp, 1, 1);
            assert_eq!(
                mouse_motion_cell(&msg),
                None,
                "ScrollUp must return None — not a motion event"
            );
        }

        #[test]
        fn scroll_down_returns_none() {
            let msg = make(MouseEventKind::ScrollDown, 1, 1);
            assert_eq!(
                mouse_motion_cell(&msg),
                None,
                "ScrollDown must return None — not a motion event"
            );
        }
    }
}
