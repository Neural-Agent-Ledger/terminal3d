//! Cross-platform **gamepad** support for the present layer, via the
//! [`gilrs`](https://docs.rs/gilrs) crate (XInput/DInput on Windows, evdev on
//! Linux, IOKit on macOS). It works **headless** — gilrs talks to the OS input
//! stack directly, no window required — so it is wired into BOTH backends
//! (terminal and window) identically.
//!
//! ## Design
//!
//! The gamepad is **purely additive**: it drives the SAME outputs the keyboard /
//! mouse paths already produce — it writes [`CameraPose`] directly the way the
//! keyboard move does, applies look through the shared [`apply_mouse_look`]
//! exactly like mouse-look, and emits the same [`InteractPressed`] / [`MenuNav`]
//! / [`FovCycle`] messages. There is no parallel input channel; a pad press is
//! indistinguishable downstream from the equivalent keypress.
//!
//! ## Why `NonSend`
//!
//! `gilrs::Gilrs` is **not `Send`** on every platform (the macOS IOKit backend in
//! particular), so it is held as a Bevy **non-send resource**
//! ([`GamepadCtx`]) — its polling system therefore runs on the main thread, which
//! is correct for OS input pumping anyway. Zero connected pads is a graceful
//! no-op; hot-plug (connect/disconnect mid-run) is tolerated because we re-read
//! the live analog state from whichever pad is currently connected each frame and
//! drain the event queue (which also keeps gilrs's internal state current).
//!
//! ## Layering of the file
//!
//! All the *feel math* — deadzone, response curve, stick→move vector,
//! stick→look delta, trigger→rise/fall, button edge-detection and D-pad/stick
//! repeat — is factored into **pure functions** (`apply_deadzone`,
//! `stick_move`, `stick_look`, `trigger_axis`, [`EdgeDetect`], [`RepeatGate`])
//! that take plain numbers and are unit-tested without any pad present. The
//! gilrs-touching [`poll_gamepad_sys`] is a thin shell that reads live axis /
//! button values and feeds them to those functions; it is the only part that
//! cannot be unit-tested without physical hardware.

use bevy::ecs::message::MessageWriter;
use bevy::prelude::*;

use gilrs::{Axis, Button, Gilrs};

use re_core::InputGate;
use re_world::CameraPose;
use re_world::Map;
use re_world::PlayerMotion;

use crate::{
    apply_mouse_look, FovCycle, InteractPressed, MenuNav, MouseSettings, MOVE_STEP, PLAYER_RADIUS,
};

// --------------------------------------------------------------------------
// Tunables (deadzone / curve / repeat). Chosen for a standard FPS pad feel.
// --------------------------------------------------------------------------

/// Per-axis stick deadzone: stick magnitudes at or below this read as zero, so a
/// resting (drifting) stick never creeps the camera. ~0.15 is the conventional
/// XInput-ish value — small enough to keep fine aim, large enough to swallow
/// hardware drift.
pub const STICK_DEADZONE: f32 = 0.15;

/// Trigger (LT/RT) press threshold: an analog trigger past this counts as "held"
/// for the edge-triggered height step. Triggers rest at 0.0 and travel to 1.0.
pub const TRIGGER_THRESHOLD: f32 = 0.5;

/// Response-curve exponent applied to the *deadzone-compensated* stick magnitude.
/// `1.0` is linear; `>1` makes small tilts gentler (finer control) while still
/// reaching full speed at full deflection. `2.0` = a square curve, a common FPS
/// choice for analog look/move.
pub const RESPONSE_CURVE: f32 = 2.0;

/// Right-stick look gain: how many "mouse cells" of [`apply_mouse_look`] one full
/// stick deflection produces per frame, before the [`MouseSettings::sensitivity`]
/// multiplier. Tuned so a full push turns at a brisk-but-controllable rate
/// comparable to a fast mouse flick sustained over a second.
pub const LOOK_GAIN: f32 = 1.5;

/// Initial delay (seconds) before a held D-pad / menu-stick direction starts to
/// auto-repeat, and the interval between repeats thereafter. Keeps menu nav from
/// storming while still allowing a comfortable held-scroll.
pub const NAV_REPEAT_DELAY: f32 = 0.35;
/// Interval (seconds) between auto-repeats once [`NAV_REPEAT_DELAY`] elapses.
pub const NAV_REPEAT_INTERVAL: f32 = 0.12;

/// Magnitude past which a menu-navigation **stick** flick counts as a direction
/// (separate from the gameplay [`STICK_DEADZONE`]; menus want a deliberate push).
pub const NAV_STICK_THRESHOLD: f32 = 0.6;

// --------------------------------------------------------------------------
// Pure feel-math (unit-tested without hardware).
// --------------------------------------------------------------------------

/// Apply a symmetric deadzone + response curve to a single signed stick axis in
/// `[-1, 1]`, returning a re-normalized signed value in `[-1, 1]`.
///
/// * `|v| <= deadzone` → `0.0` (kills resting drift).
/// * Above the deadzone the remaining travel `(|v| - dz)/(1 - dz)` is rescaled to
///   the full `[0, 1]` range (so the very edge of the deadzone is 0 speed and a
///   full push is full speed — no discontinuous jump), then raised to `curve`
///   (so small pushes are gentler when `curve > 1`), and the sign is restored.
#[inline]
pub fn apply_deadzone(v: f32, deadzone: f32, curve: f32) -> f32 {
    let mag = v.abs();
    if mag <= deadzone {
        return 0.0;
    }
    let scaled = ((mag - deadzone) / (1.0 - deadzone)).clamp(0.0, 1.0);
    let curved = scaled.powf(curve);
    curved.copysign(v)
}

/// Map a left-stick `(x, y)` (raw, in `[-1, 1]` each, y already in "up = +1"
/// convention) into a per-frame world move delta, given the camera facing
/// `(dirx, diry)` (from [`CameraPose::dir`]) and the per-step `move_step`.
///
/// Returns `(d_pos_x, d_pos_y)` to ADD to `CameraPose.pos`, computed the SAME way
/// the keyboard move does: forward/back scales the facing vector, strafe scales
/// the facing vector rotated 90°. Each axis is deadzoned+curved independently;
/// the result is scaled by `move_step` so a full forward push equals one keyboard
/// forward tap per frame, and partial tilts move proportionally slower.
///
/// `y > 0` (stick up) moves **forward** (along `+dir`); `x > 0` (stick right)
/// **strafes right** (along `dir` rotated −90°: `(diry, -dirx)`).
#[inline]
pub fn stick_move(
    x: f32,
    y: f32,
    dirx: f32,
    diry: f32,
    move_step: f32,
    deadzone: f32,
    curve: f32,
) -> (f32, f32) {
    let fwd = apply_deadzone(y, deadzone, curve);
    let strafe = apply_deadzone(x, deadzone, curve);
    // Forward along facing; strafe along facing rotated -90° (right-hand).
    let dx = (dirx * fwd + diry * strafe) * move_step;
    let dy = (diry * fwd - dirx * strafe) * move_step;
    (dx, dy)
}

/// Map a right-stick `(x, y)` (raw, in `[-1, 1]`) into the `(dx, dy)` "mouse
/// cells" delta consumed by [`apply_mouse_look`].
///
/// Each axis is deadzoned+curved then scaled by `gain`. The y axis is returned as
/// `-y` so that **stick up looks up**: [`apply_mouse_look`] inverts its `dy`
/// (screen-y grows downward), so feeding it `-stick_y` cancels to "up = up",
/// matching a standard (non-inverted) FPS right stick.
#[inline]
pub fn stick_look(x: f32, y: f32, gain: f32, deadzone: f32, curve: f32) -> (f32, f32) {
    let dx = apply_deadzone(x, deadzone, curve) * gain;
    // Negate so stick-up → look-up after apply_mouse_look's own y inversion.
    let dy = -apply_deadzone(y, deadzone, curve) * gain;
    (dx, dy)
}

/// Whether an analog trigger value (`[0, 1]`) counts as pressed for the height
/// step, given `threshold`. Pure helper so the threshold is testable.
#[inline]
pub fn trigger_axis(value: f32, threshold: f32) -> bool {
    value >= threshold
}

/// Rising-edge detector for a single boolean button: `fire(now)` returns `true`
/// only on the frame the button transitions from released to pressed, so a held
/// button fires exactly once. Reset-free; the previous state lives in the struct.
#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeDetect {
    prev: bool,
}

impl EdgeDetect {
    /// Feed the current pressed state; returns `true` on the press edge only.
    #[inline]
    pub fn fire(&mut self, now: bool) -> bool {
        let edge = now && !self.prev;
        self.prev = now;
        edge
    }
}

/// Edge-plus-repeat gate for a held direction (D-pad / menu stick): fires once on
/// the press edge, then — only while still held — again after `delay`, then every
/// `interval`. Releasing resets it so the next press fires immediately again.
///
/// This is what stops a held D-pad from storming menu navigation while still
/// allowing a comfortable held auto-scroll.
#[derive(Clone, Copy, Debug)]
pub struct RepeatGate {
    held_for: f32,
    fired_once: bool,
    last_repeat: f32,
}

impl Default for RepeatGate {
    fn default() -> Self {
        RepeatGate {
            held_for: 0.0,
            fired_once: false,
            last_repeat: 0.0,
        }
    }
}

impl RepeatGate {
    /// Advance by `dt` seconds with the current `held` state; returns `true` on
    /// the frames a navigation step should fire (press edge + delayed repeats).
    #[inline]
    pub fn tick(&mut self, held: bool, dt: f32, delay: f32, interval: f32) -> bool {
        if !held {
            // Released: reset so the next press fires instantly.
            self.held_for = 0.0;
            self.fired_once = false;
            self.last_repeat = 0.0;
            return false;
        }
        if !self.fired_once {
            // Press edge: fire immediately, start the hold timer.
            self.fired_once = true;
            self.held_for = 0.0;
            self.last_repeat = 0.0;
            return true;
        }
        self.held_for += dt;
        if self.held_for < delay {
            return false;
        }
        // Past the initial delay: fire every `interval` of accumulated hold.
        if self.held_for - self.last_repeat >= interval {
            self.last_repeat = self.held_for;
            return true;
        }
        false
    }
}

// --------------------------------------------------------------------------
// The non-send gilrs context + per-frame poll system.
// --------------------------------------------------------------------------

/// Non-send wrapper around the live [`Gilrs`] context plus the persistent
/// edge/repeat state for every gamepad control. `Gilrs` is `!Send` on some
/// platforms, so this is inserted with `insert_non_send` and its poll
/// system runs on the main thread. `ctx` is `Option` so a failed `Gilrs::new()`
/// (no input subsystem available) degrades to an inert no-op rather than panics.
pub struct GamepadCtx {
    /// The gilrs context, or `None` if initialization failed (then the pad is a
    /// silent no-op — keyboard/mouse are unaffected).
    ctx: Option<Gilrs>,
    // Edge detectors for the one-shot buttons.
    interact_edge: EdgeDetect, // A / Cross
    back_edge: EdgeDetect,     // B / Circle
    start_edge: EdgeDetect,    // Start / Menu
    fov_edge: EdgeDetect,      // Y / Triangle
    jump_edge: EdgeDetect,     // X / Square — jump (only when grounded)
    // Repeat gates for directional menu nav (D-pad + left-stick flick).
    nav_up: RepeatGate,
    nav_down: RepeatGate,
    nav_left: RepeatGate,
    nav_right: RepeatGate,
}

impl GamepadCtx {
    /// Build the context, attempting to initialize gilrs. Any error (no input
    /// backend, permission, etc.) is swallowed into `ctx = None` so the rest of
    /// the app is unaffected.
    fn new() -> Self {
        let ctx = Gilrs::new().ok();
        GamepadCtx {
            ctx,
            interact_edge: EdgeDetect::default(),
            back_edge: EdgeDetect::default(),
            start_edge: EdgeDetect::default(),
            fov_edge: EdgeDetect::default(),
            jump_edge: EdgeDetect::default(),
            nav_up: RepeatGate::default(),
            nav_down: RepeatGate::default(),
            nav_left: RepeatGate::default(),
            nav_right: RepeatGate::default(),
        }
    }
}

/// Register the gamepad context + poll system on an app. Called by BOTH backends'
/// plugins so the pad behaves identically headless and windowed. Inserting the
/// `NonSend` resource here (App build, main thread) is the correct place for a
/// `!Send` type; the poll system is added to `Update` so it runs every frame
/// alongside the keyboard input system.
pub fn add_gamepad(app: &mut App) {
    app.insert_non_send(GamepadCtx::new())
        .add_systems(Update, poll_gamepad_sys);
}

/// Per-frame gilrs poll: drain the event queue (keeps gilrs state current + is
/// the hot-plug signal), then read the live analog/button state of the first
/// connected pad and translate it into the SAME outputs the keyboard path emits.
///
/// LIVE PATH — this is the only part that cannot be unit-tested without a
/// physical pad; the feel math it calls is all pure-tested above. With zero pads
/// connected the whole body is a no-op.
///
/// ## Gating
///
/// Menu-nav edges (D-pad, B/Circle back, Start) and the interact/confirm edge
/// (A/Cross) are always live — a controller must still drive the pause menu.
/// The camera-coupled arms (left-stick move, right-stick look, X/Square jump)
/// are gated on [`InputGate::camera_motion`]: they are skipped while paused or
/// in menus, matching the keyboard path at `terminal.rs:~294-316`. The interact
/// edge is similarly gated (it fires a gameplay action) while nav stays ungated.
#[allow(clippy::too_many_arguments)]
fn poll_gamepad_sys(
    mut gp: NonSendMut<GamepadCtx>,
    gate: Res<InputGate>,
    time: Res<Time>,
    settings: Res<MouseSettings>,
    map: Res<Map>,
    mut cam_q: Query<(&mut CameraPose, &mut PlayerMotion)>,
    mut interact: MessageWriter<InteractPressed>,
    mut nav: MessageWriter<MenuNav>,
    mut fov: MessageWriter<FovCycle>,
) {
    // Drain the event queue first. This advances gilrs's internal state and is
    // how connect/disconnect (hot-plug) is observed; we don't need the events
    // themselves because we read live state below, but draining is required.
    {
        let Some(ctx) = gp.ctx.as_mut() else {
            return; // gilrs unavailable → inert.
        };
        while ctx.next_event().is_some() {}
    }

    // Pick the first currently-connected pad. Zero pads → graceful no-op. We
    // re-resolve this every frame so a hot-plugged pad is picked up automatically
    // and an unplugged one simply stops contributing.
    let dt = time.delta_secs();
    let gp = &mut *gp;
    let Some(ctx) = gp.ctx.as_ref() else {
        return;
    };
    let Some((_id, pad)) = ctx.gamepads().next() else {
        return; // no pad connected this frame.
    };

    // --- Read live analog state. ---
    let lx = pad.value(Axis::LeftStickX);
    let ly = pad.value(Axis::LeftStickY);
    let rx = pad.value(Axis::RightStickX);
    let ry = pad.value(Axis::RightStickY);

    // Buttons (digital). Triggers double as digital via is_pressed but we also
    // read their analog axis so shoulder OR trigger drives height.
    let a = pad.is_pressed(Button::South); // A / Cross
    let b = pad.is_pressed(Button::East); // B / Circle
    let yb = pad.is_pressed(Button::North); // Y / Triangle
    let xb = pad.is_pressed(Button::West); // X / Square — jump
    let start = pad.is_pressed(Button::Start);
    let dpad_up = pad.is_pressed(Button::DPadUp);
    let dpad_down = pad.is_pressed(Button::DPadDown);
    let dpad_left = pad.is_pressed(Button::DPadLeft);
    let dpad_right = pad.is_pressed(Button::DPadRight);

    // --- One-shot edge-triggered messages (fire once per press). ---
    // A / Cross → interact AND menu-confirm (additive, like Enter/Space on kbd).
    // Interact is a gameplay action and is gated; MenuNav::Confirm is always live
    // so the player can confirm a pause-menu selection with A.
    if gp.interact_edge.fire(a) {
        if gate.camera_motion {
            interact.write(InteractPressed);
        }
        nav.write(MenuNav::Confirm);
    }
    // B / Circle → menu back (always live — drives the pause menu).
    if gp.back_edge.fire(b) {
        nav.write(MenuNav::Back);
    }
    // Start / Menu → menu-back, consistent with the keyboard Esc path: emit ONLY
    // `MenuNav::Back` and let `re_game` decide (PAUSE in gameplay, quit at the top
    // menu). It no longer writes `AppExit` directly — that hard-quit was the same
    // accidental-exit footgun removed from the keyboard path (player bug #3/#5).
    // Always live (so Start can resume from pause via the menu).
    if gp.start_edge.fire(start) {
        nav.write(MenuNav::Back);
    }
    // Y / Triangle → cycle FOV up by one notch (always available).
    if gp.fov_edge.fire(yb) {
        fov.write(FovCycle(1));
    }
    // X / Square → JUMP (edge-triggered). Detect the edge here (always), but only
    // apply the jump when the gate is open (camera-coupled section below).
    let jump_edge = gp.jump_edge.fire(xb);

    // --- Directional menu nav: D-pad OR a left-stick flick, with repeat. ---
    let up_held = dpad_up || ly > NAV_STICK_THRESHOLD;
    let down_held = dpad_down || ly < -NAV_STICK_THRESHOLD;
    let left_held = dpad_left || lx < -NAV_STICK_THRESHOLD;
    let right_held = dpad_right || lx > NAV_STICK_THRESHOLD;
    if gp
        .nav_up
        .tick(up_held, dt, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL)
    {
        nav.write(MenuNav::Up);
    }
    if gp
        .nav_down
        .tick(down_held, dt, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL)
    {
        nav.write(MenuNav::Down);
    }
    if gp
        .nav_left
        .tick(left_held, dt, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL)
    {
        nav.write(MenuNav::Left);
    }
    if gp
        .nav_right
        .tick(right_held, dt, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL)
    {
        nav.write(MenuNav::Right);
    }

    // --- Camera-coupled outputs (move / look / jump). ---
    // These are gated on `camera_motion`: while paused the camera entity is kept
    // alive (the scene stays rendered) but must not be driven by the stick or
    // jump button. Menu-nav (above) is already done and stays live regardless.
    // Mirror the keyboard gate at terminal.rs:~314-316.
    if !gate.camera_motion {
        return;
    }

    let Ok((mut cam, mut motion)) = cam_q.single_mut() else {
        return; // no camera entity → movement/look/jump inert (nav done above).
    };

    // Left stick → move, now COLLISION-CHECKED via move_with_collision (slides
    // along walls, never tunnels) — same wiring as the keyboard move. The pure
    // stick_move feel math is unchanged; only the apply step gained collision.
    let (dirx, diry) = cam.dir();
    let (mdx, mdy) = stick_move(
        lx,
        ly,
        dirx,
        diry,
        MOVE_STEP,
        STICK_DEADZONE,
        RESPONSE_CURVE,
    );
    if mdx != 0.0 || mdy != 0.0 {
        cam.move_with_collision(mdx, mdy, &map, PLAYER_RADIUS);
    }

    // Right stick → look (same shared apply_mouse_look path as the mouse, scaled
    // by the live sensitivity; respects the on/off toggle like mouse-look does).
    if settings.enabled {
        let (ldx, ldy) = stick_look(rx, ry, LOOK_GAIN, STICK_DEADZONE, RESPONSE_CURVE);
        if ldx != 0.0 || ldy != 0.0 {
            let (yaw, pitch) = apply_mouse_look(cam.yaw, cam.pitch, ldx, ldy, settings.sensitivity);
            cam.yaw = yaw;
            cam.pitch = pitch;
        }
    }

    // JUMP (X / Square) — call unconditionally on the press edge; `try_jump`
    // is its own gate (grounded, coyote-time grace window, or buffered for
    // landing — Swaggerfall-study item 5). The backend's per-frame
    // `gravity_sys` integrates the impulse (and fires any buffered press on
    // landing); the gamepad never ticks gravity itself (that would
    // double-apply it).
    if jump_edge {
        CameraPose::try_jump(&mut motion);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadzone_zeroes_small_input() {
        // Anything inside the deadzone reads as exactly zero (no drift creep).
        assert_eq!(apply_deadzone(0.0, STICK_DEADZONE, RESPONSE_CURVE), 0.0);
        assert_eq!(apply_deadzone(0.10, STICK_DEADZONE, RESPONSE_CURVE), 0.0);
        assert_eq!(apply_deadzone(-0.14, STICK_DEADZONE, RESPONSE_CURVE), 0.0);
        // Exactly at the deadzone edge is still zero (boundary is inclusive).
        assert_eq!(
            apply_deadzone(STICK_DEADZONE, STICK_DEADZONE, RESPONSE_CURVE),
            0.0
        );
    }

    #[test]
    fn deadzone_full_push_is_full_magnitude() {
        // A full deflection always reaches ±1.0 regardless of the curve, so top
        // speed is never lost.
        assert!((apply_deadzone(1.0, STICK_DEADZONE, RESPONSE_CURVE) - 1.0).abs() < 1e-6);
        assert!((apply_deadzone(-1.0, STICK_DEADZONE, RESPONSE_CURVE) + 1.0).abs() < 1e-6);
        // Linear curve, no deadzone → identity.
        assert!((apply_deadzone(0.5, 0.0, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn deadzone_preserves_sign_and_is_monotonic() {
        let a = apply_deadzone(0.4, STICK_DEADZONE, RESPONSE_CURVE);
        let bgr = apply_deadzone(0.8, STICK_DEADZONE, RESPONSE_CURVE);
        assert!(a > 0.0 && bgr > 0.0);
        assert!(bgr > a, "larger push → larger output ({bgr} > {a})");
        // Sign is preserved.
        assert!(apply_deadzone(-0.8, STICK_DEADZONE, RESPONSE_CURVE) < 0.0);
    }

    #[test]
    fn curve_softens_small_pushes() {
        // With curve > 1 a half push yields LESS than half output (gentle near
        // center), but full push is still full.
        let half = apply_deadzone(0.5 + STICK_DEADZONE * 0.5, 0.0, 2.0);
        assert!(half < 0.5 + STICK_DEADZONE, "curve should soften mid-range");
    }

    #[test]
    fn stick_move_forward_follows_facing() {
        // Facing +y (yaw=0 → dir=(0,1)). Full stick up (y=+1) moves +y by MOVE_STEP.
        let (dx, dy) = stick_move(
            0.0,
            1.0,
            0.0,
            1.0,
            MOVE_STEP,
            STICK_DEADZONE,
            RESPONSE_CURVE,
        );
        assert!(dx.abs() < 1e-6, "no lateral drift, dx={dx}");
        assert!(
            (dy - MOVE_STEP).abs() < 1e-6,
            "full forward = one MOVE_STEP, dy={dy}"
        );
        // Full stick down (y=-1) moves backward.
        let (_, dyb) = stick_move(
            0.0,
            -1.0,
            0.0,
            1.0,
            MOVE_STEP,
            STICK_DEADZONE,
            RESPONSE_CURVE,
        );
        assert!((dyb + MOVE_STEP).abs() < 1e-6);
    }

    #[test]
    fn stick_move_strafe_is_perpendicular() {
        // Facing +y; full stick right (x=+1) strafes along +x (dir rotated -90°).
        let (dx, dy) = stick_move(
            1.0,
            0.0,
            0.0,
            1.0,
            MOVE_STEP,
            STICK_DEADZONE,
            RESPONSE_CURVE,
        );
        assert!((dx - MOVE_STEP).abs() < 1e-6, "strafe right = +x, dx={dx}");
        assert!(
            dy.abs() < 1e-6,
            "pure strafe has no forward component, dy={dy}"
        );
    }

    #[test]
    fn stick_move_deadzone_is_still() {
        // A resting stick produces no movement at all.
        let (dx, dy) = stick_move(
            0.05,
            -0.05,
            0.0,
            1.0,
            MOVE_STEP,
            STICK_DEADZONE,
            RESPONSE_CURVE,
        );
        assert_eq!((dx, dy), (0.0, 0.0));
    }

    #[test]
    fn stick_look_magnitude_and_inversion() {
        // Full right (x=+1) yaws right (dx>0) at LOOK_GAIN before apply_mouse_look.
        let (dx, _) = stick_look(1.0, 0.0, LOOK_GAIN, STICK_DEADZONE, RESPONSE_CURVE);
        assert!((dx - LOOK_GAIN).abs() < 1e-6, "full right = +gain, dx={dx}");
        // Stick UP (y=+1) must look UP. apply_mouse_look inverts dy, so we feed it
        // a NEGATIVE dy; feeding our dy through apply_mouse_look must raise pitch.
        let (_, dy) = stick_look(0.0, 1.0, LOOK_GAIN, STICK_DEADZONE, RESPONSE_CURVE);
        assert!(
            dy < 0.0,
            "stick-up yields negative look-dy (so pitch rises), dy={dy}"
        );
        let (_, pitch) = apply_mouse_look(0.0, 0.0, 0.0, dy, 1.0);
        assert!(pitch > 0.0, "stick-up must look up (pitch>0), got {pitch}");
    }

    #[test]
    fn stick_look_deadzone_is_zero() {
        let (dx, dy) = stick_look(0.1, -0.1, LOOK_GAIN, STICK_DEADZONE, RESPONSE_CURVE);
        assert_eq!((dx, dy), (0.0, -0.0));
    }

    #[test]
    fn trigger_threshold() {
        assert!(!trigger_axis(0.0, TRIGGER_THRESHOLD));
        assert!(!trigger_axis(0.49, TRIGGER_THRESHOLD));
        assert!(trigger_axis(0.5, TRIGGER_THRESHOLD));
        assert!(trigger_axis(1.0, TRIGGER_THRESHOLD));
    }

    #[test]
    fn edge_detect_fires_once_per_press() {
        let mut e = EdgeDetect::default();
        assert!(!e.fire(false), "released → no fire");
        assert!(e.fire(true), "press edge fires");
        assert!(!e.fire(true), "held does NOT re-fire");
        assert!(!e.fire(true), "still held, still silent");
        assert!(!e.fire(false), "release does not fire");
        assert!(e.fire(true), "next press fires again");
    }

    #[test]
    fn repeat_gate_edge_then_delayed_repeat() {
        let mut g = RepeatGate::default();
        // Press edge fires immediately.
        assert!(g.tick(true, 0.0, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL));
        // Holding within the initial delay does NOT storm.
        assert!(!g.tick(true, 0.1, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL));
        assert!(!g.tick(true, 0.1, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL));
        // Cross the delay → one repeat fires.
        assert!(g.tick(true, 0.2, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL)); // held_for=0.4 > 0.35
                                                                           // Immediately after, not yet another interval elapsed → silent.
        assert!(!g.tick(true, 0.01, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL));
        // After another interval → fires again.
        assert!(g.tick(
            true,
            NAV_REPEAT_INTERVAL,
            NAV_REPEAT_DELAY,
            NAV_REPEAT_INTERVAL
        ));
        // Release resets: next press fires immediately again.
        assert!(!g.tick(false, 0.5, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL));
        assert!(g.tick(true, 0.0, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL));
    }

    #[test]
    fn repeat_gate_release_before_delay_fires_only_once() {
        // A quick tap (press then release before the delay) fires exactly once.
        let mut g = RepeatGate::default();
        assert!(g.tick(true, 0.0, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL));
        assert!(!g.tick(false, 0.05, NAV_REPEAT_DELAY, NAV_REPEAT_INTERVAL));
    }
}
