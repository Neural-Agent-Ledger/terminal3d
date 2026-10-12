//! `backend-terminal` — the DEFAULT present backend, on `bevy_ratatui` 0.11.
//!
//! HEADLESS: `bevy_ratatui` depends on `bevy` with `default-features = false`,
//! so this build never pulls wgpu or winit. The plugin group
//! [`RatatuiPlugins`] owns the raw-mode / alt-screen lifecycle and emits
//! crossterm-sourced [`KeyMessage`] / [`ResizeMessage`] events; we consume
//! them.
//!
//! Pipeline per frame:
//! * `Update`  — [`input_sys`] maps key messages → camera move/turn + quit;
//!   [`resize_sys`] maps resize messages → [`Framebuffer::resize`].
//! * `Last`    — [`present_sys`] blits the [`Framebuffer`] into the ratatui
//!   [`Buffer`] in one `RatatuiContext::draw`.
//!
//! ## Kitty keyboard protocol — Windows / reliable-push fix
//!
//! crossterm's `supports_keyboard_enhancement()` **always returns `false` on
//! Windows** (hardcoded in `crossterm::terminal::sys::windows`). As a result,
//! `bevy_ratatui`'s `KittyPlugin` never inserts `KittyEnabled` on Windows
//! Terminal or WezTerm, even though both fully support the protocol. Without
//! `KittyEnabled`, `input_sys` takes the per-event fallback: one key-down event
//! = one step, with the OS ~500 ms repeat delay before autorepeat, producing the
//! classic hold-then-stutter symptom.
//!
//! Additionally, `PushKeyboardEnhancementFlags` / `PopKeyboardEnhancementFlags`
//! override `is_ansi_code_supported()` to return `false` on Windows, so even a
//! direct `execute!(stdout, PushKeyboardEnhancementFlags(...))` call goes to the
//! WinAPI path, which returns `Err(Unsupported)` and sends nothing.
//!
//! **The fix** is [`force_kitty_sys`]: if `bevy_ratatui`'s kitty setup
//! succeeded (Linux / non-Windows path) we leave it alone. If it did NOT
//! (Windows where the probe always returns false), we **bypass the probe** and
//! write the push VT sequence directly to stdout via [`push_kitty_raw`],
//! bypassing the command dispatch. On success we insert both `KittyEnabled` (so
//! `input_sys` takes the held-key smooth path) and [`KittyGuard`] (whose `Drop`
//! writes the pop sequence the same raw way). The panic hook is also extended to
//! pop the flags so a panic never leaves the terminal in kitty mode.
//!
//! Terminals that do NOT support the protocol (legacy conhost, some SSH
//! connections) will simply ignore the unrecognised escape sequence, and the pop
//! in `KittyGuard::drop` is likewise a no-op. The per-event fallback in
//! `input_sys` (`if !kitty`) remains active for those terminals — smooth held
//! movement is impossible there because no release events arrive, and that is an
//! honest limitation, not a bug.

use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::prelude::*;

use bevy_ratatui::crossterm::event::{
    KeyCode, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags, ModifierKeyCode, MouseEventKind,
    PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use bevy_ratatui::event::{KeyMessage, MouseMessage, ResizeMessage};
use bevy_ratatui::kitty::KittyEnabled;
use bevy_ratatui::RatatuiContext;
use bevy_ratatui::RatatuiPlugins;

use ratatui::style::{Color, Modifier, Style};

use re_core::{CellStyle, FrameTimingMetrics, Framebuffer, InputGate, PerfOverlay};
use re_world::CameraPose;
use re_world::PlayerMotion;

use re_world::Map;

use crate::{
    add_gamepad, apply_mouse_look, ensure_player_motion, glyph_char, held_move_delta,
    held_turn_delta, rgb8, rgb_to_ansi256, ColorMode, FovCycle, HeldKeys, HeldTimestamps,
    InteractPressed, KeyIntent, MenuMouseClick, MenuMouseMove, MenuNav, MouseSettings,
    OsCursorGrab, OsCursorSimpleHide, PostFxCycle, PresentCache, PresentConfig, PresentSet,
    RenderConfig, ZoomToggle, MOVE_SPEED, MOVE_STEP, PITCH_LIMIT, PLAYER_RADIUS, SPRINT_MULTIPLIER,
    TURN_SPEED, TURN_STEP,
};

// ---------------------------------------------------------------------------
// Kitty keyboard protocol — reliable cross-platform push/pop
// ---------------------------------------------------------------------------

/// Write the kitty keyboard-enhancement push sequence directly to stdout as
/// raw bytes, bypassing crossterm's command-dispatch guard that returns
/// `Err(Unsupported)` on Windows for `PushKeyboardEnhancementFlags`.
///
/// We format the VT escape via [`Command::write_ansi`] (which does know the
/// correct bytes — `\x1b[>Xu`), collect it into a `String`, and write it as
/// raw UTF-8. This works on any VT-capable terminal regardless of OS.
///
/// Returns `Ok(())` if the write + flush succeeded. An `Err` means the bytes
/// could not be sent (stdout not a tty, etc.) and the caller should NOT insert
/// `KittyEnabled` or `KittyGuard`.
fn push_kitty_raw() -> std::io::Result<()> {
    use bevy_ratatui::crossterm::Command as _;
    use std::io::Write as _;

    let flags = KeyboardEnhancementFlags::REPORT_EVENT_TYPES
        | KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES;
    let mut buf = String::with_capacity(16);
    PushKeyboardEnhancementFlags(flags)
        .write_ansi(&mut buf)
        .map_err(std::io::Error::other)?;
    let mut stdout = std::io::stdout();
    stdout.write_all(buf.as_bytes())?;
    stdout.flush()
}

/// Write the kitty pop sequence directly to stdout as raw bytes.
///
/// Mirrors [`push_kitty_raw`]. Best-effort: errors are silently discarded
/// because this is called from `Drop` and panic hooks where we cannot
/// propagate errors.
fn pop_kitty_raw() {
    use bevy_ratatui::crossterm::Command as _;
    use std::io::Write as _;

    #[cfg(test)]
    test_hooks::record_pop_kitty_raw_call();

    let mut buf = String::with_capacity(8);
    if PopKeyboardEnhancementFlags.write_ansi(&mut buf).is_ok() {
        let mut stdout = std::io::stdout();
        let _ = stdout.write_all(buf.as_bytes());
        let _ = stdout.flush();
    }
}

/// Test-only instrumentation for [`pop_kitty_raw`]'s call count.
///
/// Real raw stdout writes are not independently observable in a headless
/// `cargo test` run (no tty, and the write is best-effort/swallowed), so the
/// push/pop BOOKKEEPING — how many times [`pop_kitty_raw`] actually fired —
/// is what the M2 regression tests assert instead. A `thread_local` keeps
/// counts isolated per test when `cargo test`'s thread pool runs cases
/// concurrently; each test resets the counter before exercising the code
/// under test so a reused worker thread's stale count never leaks in.
#[cfg(test)]
mod test_hooks {
    use std::cell::Cell;

    thread_local! {
        static POP_KITTY_RAW_CALLS: Cell<u32> = const { Cell::new(0) };
    }

    pub(super) fn record_pop_kitty_raw_call() {
        POP_KITTY_RAW_CALLS.with(|c| c.set(c.get() + 1));
    }

    /// Reset this thread's call counter to zero. Call before the code under
    /// test so a prior test's count (same worker thread) can't leak in.
    pub(super) fn reset_pop_kitty_raw_calls() {
        POP_KITTY_RAW_CALLS.with(|c| c.set(0));
    }

    /// Read this thread's call counter since the last reset.
    pub(super) fn pop_kitty_raw_calls() -> u32 {
        POP_KITTY_RAW_CALLS.with(Cell::get)
    }
}

/// A RAII guard that pops the kitty keyboard-enhancement flags on `Drop` using
/// the raw-write path ([`pop_kitty_raw`]).
///
/// Inserted by [`force_kitty_sys`] ONLY when WE pushed the flags (i.e. when
/// `bevy_ratatui`'s own `KittyPlugin` did not succeed, which is always the case
/// on Windows). On Linux, `bevy_ratatui`'s `KittyEnabled::Drop` handles the pop
/// via its own path, so we set `we_pushed = false` and this guard is a no-op.
///
/// `we_pushed = false` is also inserted on Linux (where `KittyEnabled` was
/// already there) so the resource lifecycle is always tracked, but no extra pop
/// is emitted.
#[derive(Resource)]
struct KittyGuard {
    we_pushed: bool,
}

impl Drop for KittyGuard {
    fn drop(&mut self) {
        if self.we_pushed {
            pop_kitty_raw();
        }
    }
}

/// Whether a [`KittyGuard`] inserted alongside a MANUAL [`push_kitty_raw`]
/// push (i.e. one not routed through `bevy_ratatui`'s own Startup probe) must
/// carry `we_pushed: true` to guarantee the matching pop actually happens.
///
/// `KittyEnabled`'s own `Drop` (`bevy_ratatui::crossterm_context::kitty`)
/// unconditionally calls `disable_kitty_protocol()`, which dispatches
/// `PopKeyboardEnhancementFlags` through crossterm's `.execute()` — the SAME
/// command-dispatch path whose `is_ansi_code_supported()` guard is hardcoded
/// `false` on Windows (see the module-level doc), so on Windows that call
/// silently sends nothing. Everywhere else it is a real write. So: `true` on
/// Windows, where `KittyGuard`'s own raw pop is the ONLY one that actually
/// clears the terminal state; `false` elsewhere, where `KittyEnabled::Drop`'s
/// pop already suffices and a second raw pop would double-pop the kitty flag
/// stack.
///
/// Used by [`resume_terminal`], which — unlike [`force_kitty_sys`] — has no
/// pre-existing `KittyEnabled` resource to probe for presence (the playtest
/// handoff removed it in [`suspend_terminal`]), so it cannot reuse
/// `force_kitty_sys`'s `kitty.is_some()` check and must decide the flag
/// directly. Deliberately a compile-time platform check rather than a fresh
/// `supports_keyboard_enhancement()` probe: that function blocks reading a
/// terminal response (see crossterm's unix `query_keyboard_enhancement_flags`)
/// and would race the live crossterm event stream `input_sys` is draining —
/// unacceptable to call outside `bevy_ratatui`'s own one-shot Startup probe.
const fn kitty_manual_pop_required() -> bool {
    cfg!(windows)
}

/// Startup system: ensure the kitty keyboard protocol is active on capable
/// terminals even when `bevy_ratatui`'s auto-detection failed.
///
/// On Windows, `crossterm::terminal::supports_keyboard_enhancement()` always
/// returns `false` (hardcoded), so `bevy_ratatui`'s `kitty_setup` never inserts
/// `KittyEnabled`. We detect that here and push the flags ourselves via
/// [`push_kitty_raw`], then insert both `KittyEnabled` (so `input_sys` takes the
/// smooth held-key path) and [`KittyGuard`] (so the pop fires on teardown).
///
/// On Linux where `bevy_ratatui` already succeeded, `KittyEnabled` is present —
/// we just insert `KittyGuard { we_pushed: false }` so the resource exists but
/// does not emit a spurious extra pop.
fn force_kitty_sys(kitty: Option<Res<KittyEnabled>>, mut commands: Commands) {
    if kitty.is_some() {
        // bevy_ratatui already pushed on this terminal (Linux / kitty / foot).
        // No second push. Install a no-op guard so the resource is always present.
        commands.insert_resource(KittyGuard { we_pushed: false });
        return;
    }

    // bevy_ratatui's probe returned false (always on Windows) or errored.
    // Attempt a direct raw push.
    if push_kitty_raw().is_ok() {
        // Protocol accepted. Mark as active for input_sys and install the pop guard.
        commands.insert_resource(KittyEnabled);
        commands.insert_resource(KittyGuard { we_pushed: true });
    }
    // If push_kitty_raw also failed (stdout not a tty, etc.), we leave both
    // KittyEnabled and KittyGuard absent. input_sys will take the per-event
    // fallback — the correct, honest degradation for a non-capable terminal.
}

// ---------------------------------------------------------------------------
// Terminal suspend/resume — playtest child-process handoff
// ---------------------------------------------------------------------------
//
// `ts_editor`'s F5 playtest spawns the actual game binary as a child process
// that inherits stdio and owns the SAME terminal the editor itself is
// presenting through. Before the spawn, the editor's terminal presentation
// must get out of the way (leave the alt screen, restore raw mode + cursor,
// disable mouse capture, pop the kitty keyboard-enhancement flags) so the
// child's own `RatatuiPlugins` startup can take over cleanly; after the child
// exits, the editor must resume exactly where it left off.
//
// ## The drop-order contract (read before touching this pair)
//
// bevy_ratatui's `TerminalContext::restore()` (called by both
// `RatatuiContext::restore()` and `RatatuiContext`'s own `Drop` impl) is a
// **static** passthrough — it performs the terminal I/O
// (`LeaveAlternateScreen`, cursor `Show`, `disable_raw_mode()`) with NO
// reference to `self`. That means calling it directly while the
// `RatatuiContext` *resource* is still alive restores the terminal without
// retiring the resource: the stale resource's own `Drop` will restore the
// terminal AGAIN, arbitrarily later — including, in the worst case, right
// after [`resume_terminal`] has already re-entered the alt screen, landing
// the user back outside it (bare shell prompt over a "restored" editor). The
// identical hazard applies to `KittyEnabled` (its `Drop` pops the kitty
// flags) and `MouseEnabled` (its `Drop` disables mouse capture) — both are
// zero-sized but their `Drop` impls have real side effects.
//
// The contract these two functions uphold:
// 1. [`suspend_terminal`] REMOVES the `RatatuiContext` / [`KittyEnabled`] /
//    [`KittyGuard`] / `bevy_ratatui::mouse::MouseEnabled` resources, in that
//    order — mirroring bevy_ratatui's own `CleanupPlugin::cleanup` teardown
//    order on `AppExit` (kitty, then mouse, then the terminal context last).
//    Removing a resource drops it in place, so their `Drop` impls perform the
//    actual restore; `RatatuiContext` is removed LAST so its alt-screen exit
//    is the final terminal write. Kitty's actual pop is entirely
//    [`KittyGuard`]'s job: its `Drop` fires [`pop_kitty_raw`] IFF
//    `we_pushed` — i.e. iff `KittyEnabled::Drop`'s standard crossterm pop
//    (which just ran, per the removal order above) will silently no-op on
//    this platform (Windows — see the module-level doc). There is
//    deliberately NO separate unconditional raw pop after the removals: one
//    was tried and it double-popped the kitty flag stack on Linux, where
//    `KittyEnabled::Drop`'s pop is real (M2 fix).
// 2. [`resume_terminal`] re-creates every one of those resources and inserts
//    them BEFORE returning, in bevy_ratatui's own Startup order (context →
//    mouse capture → kitty) — so at no point after this call does a stale
//    resource exist that could later drop and undo the fresh state. The
//    `KittyGuard` it inserts alongside a successful manual push carries
//    `we_pushed: `[`kitty_manual_pop_required`]`()` — true only on Windows,
//    matching `force_kitty_sys`'s own `kitty.is_some()` policy — so the SAME
//    balanced-pop rule applies whether the flags were pushed at app Startup
//    or re-pushed here after a playtest handoff.
//
// Both functions are plain `fn(&mut World)`, not an RAII guard: after
// `suspend_terminal` there are NO terminal-owning resources left in the
// `World` at all (by design — the child process is expected to fully own the
// terminal for its lifetime, which may be many seconds if it has to compile
// first), so there is nothing for a guard's `Drop` to safely reach without
// capturing a `World` reference across a blocking child spawn (which would
// also block the caller from doing anything else with the `World` in the
// meantime — no better than the explicit call). The caller owns pairing
// them: call [`resume_terminal`] once the child exits, on EVERY exit path
// including a spawn failure, so the editor is never left stuck outside the
// alt screen.

/// Suspend terminal presentation for a playtest child-process handoff.
///
/// Leaves the alt screen, disables raw mode, shows the cursor, disables mouse
/// capture, and pops the kitty keyboard-enhancement flags (best-effort; works
/// on Windows too via the raw-write path, unlike the standard crossterm pop
/// command). After this call the caller's process no longer touches the
/// terminal in any way — it is now safe to spawn a child that inherits stdio
/// and takes over the same terminal.
///
/// Idempotent: calling this twice in a row (no [`resume_terminal`] between)
/// is a harmless no-op the second time — the resources are already absent, so
/// no `Drop` fires and nothing is written.
///
/// **Balanced pop, not a double pop:** removing `KittyEnabled` runs its own
/// `Drop` (a real pop on non-Windows, a silent no-op on Windows); removing
/// `KittyGuard` then runs `pop_kitty_raw` IFF `we_pushed` is set — i.e.
/// only on the platform where `KittyEnabled::Drop` just no-opped. There is no
/// additional unconditional pop after these two removals: that used to exist
/// as a "safety net" and, on Linux, fired a genuine SECOND real pop on top of
/// `KittyEnabled::Drop`'s already-real one (M2). The guard's own
/// `we_pushed`-gated `Drop` is the single source of truth for whether a raw
/// pop is still owed.
///
/// See the drop-order contract in the module doc above this function; this
/// MUST be paired with [`resume_terminal`] once the child exits.
pub fn suspend_terminal(world: &mut World) {
    // Order mirrors bevy_ratatui's own `CleanupPlugin::cleanup`
    // (crossterm_context/cleanup.rs): kitty, then mouse, then the terminal
    // context last — each removal drops the resource in place, running its
    // restore-on-Drop side effect. `KittyEnabled::Drop` and `KittyGuard::Drop`
    // (gated on `we_pushed`) together perform EXACTLY one real pop between
    // them; see the module doc and the doc comment above for why there is
    // deliberately no extra `pop_kitty_raw()` call here.
    world.remove_resource::<KittyEnabled>();
    world.remove_resource::<KittyGuard>();
    world.remove_resource::<bevy_ratatui::mouse::MouseEnabled>();
    world.remove_resource::<RatatuiContext>();
}

/// Resume terminal presentation after [`suspend_terminal`] and the child
/// process exits.
///
/// Re-enters the alt screen, re-enables raw mode, re-enables mouse capture,
/// and re-pushes the kitty keyboard-enhancement flags — mirroring
/// `bevy_ratatui`'s own Startup system order (context → mouse → kitty) so the
/// resulting resource set is indistinguishable from a fresh app start.
///
/// Returns `false` if the terminal context could not be re-initialized (rare
/// — e.g. stdout is no longer a tty). On `false` the `World` is left with NO
/// `RatatuiContext` resource, i.e. still suspended; the caller must surface
/// this to the user rather than silently limping on with no presentation
/// (a system reading `ResMut<RatatuiContext>` would panic on a missing
/// resource, which is the correct fail-loud behaviour here).
pub fn resume_terminal(world: &mut World) -> bool {
    let ctx = match RatatuiContext::init() {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("re_present: failed to resume terminal presentation: {err}");
            return false;
        }
    };
    // RatatuiContext first: everything else assumes the alt screen + raw mode
    // are already active (matching bevy_ratatui's own Startup ordering).
    world.insert_resource(ctx);

    {
        use bevy_ratatui::crossterm::event::EnableMouseCapture;
        use bevy_ratatui::crossterm::ExecutableCommand as _;
        let _ = std::io::stdout().execute(EnableMouseCapture);
    }
    world.insert_resource(bevy_ratatui::mouse::MouseEnabled);

    // Kitty last, mirroring `force_kitty_sys` (PostStartup): try the raw push
    // and only claim `KittyEnabled` if the terminal actually accepted it.
    // `we_pushed: kitty_manual_pop_required()` (NOT unconditional `true`,
    // see M2): a hardcoded `true` here meant `KittyEnabled::Drop`'s own real
    // pop (which runs on non-Windows) AND `KittyGuard`'s raw pop both fired
    // at teardown — a genuine double pop of the kitty flag stack on Linux.
    if push_kitty_raw().is_ok() {
        world.insert_resource(KittyEnabled);
        world.insert_resource(KittyGuard {
            we_pushed: kitty_manual_pop_required(),
        });
    } else {
        world.insert_resource(KittyGuard { we_pushed: false });
    }
    true
}

/// Classify a crossterm [`KeyCode`] into the backend-neutral [`KeyIntent`].
///
/// This is the ONE place the live terminal binds raw keys; everything downstream
/// (movement, turn, menu nav, quit) is driven from the returned intent, so the
/// map is a single, unit-testable table. **`q`/`Q` AND `Esc` both map to
/// [`KeyIntent::Back`]** so the on-screen "q back" / "q quit" hints actually work.
/// This is safe because `Back` is NO LONGER a hard quit during gameplay — it opens
/// the Pause overlay — so the old accidental-quit footgun is gone; `re_game`
/// decides what Back means per state (pause in play, up-one-level in menus, quit
/// only at the top menu). WASD = movement (A/D strafe), arrows = turn; both still
/// feed menus via [`KeyIntent::menu_nav`].
///
/// **Sprint (Shift):** `LeftShift` and `RightShift` map to [`KeyIntent::Sprint`].
/// On a non-kitty terminal crossterm normally reports modifier bits on a `Char`
/// event (e.g. `Char('W')` with SHIFT set) rather than a dedicated `KeyCode::LeftShift`
/// / `KeyCode::RightShift` event. The explicit `KeyCode::LeftShift` /
/// `KeyCode::RightShift` events only arrive when the **kitty protocol** is active
/// (because kitty sends dedicated press/release events for modifier keys). On
/// non-kitty terminals Shift-held is detected via the `KeyModifiers::SHIFT` check in
/// `input_sys` instead (see the modifier path there), so sprint still works on
/// legacy terminals.
fn key_intent(code: KeyCode) -> KeyIntent {
    match code {
        KeyCode::Char('w') | KeyCode::Char('W') => KeyIntent::Forward,
        KeyCode::Char('s') | KeyCode::Char('S') => KeyIntent::Backward,
        KeyCode::Char('a') | KeyCode::Char('A') => KeyIntent::StrafeLeft,
        KeyCode::Char('d') | KeyCode::Char('D') => KeyIntent::StrafeRight,
        KeyCode::Left => KeyIntent::TurnLeft,
        KeyCode::Right => KeyIntent::TurnRight,
        KeyCode::Up => KeyIntent::PitchUp,
        KeyCode::Down => KeyIntent::PitchDown,
        KeyCode::Char(' ') => KeyIntent::Jump,
        KeyCode::Char('e') | KeyCode::Char('E') | KeyCode::Enter => KeyIntent::Interact,
        // `Esc` AND `q`/`Q` back out / cancel (matching the "q back" / "q quit"
        // hints). No longer a hard quit in gameplay — it opens Pause — so binding
        // `q` reintroduces no footgun.
        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => KeyIntent::Back,
        KeyCode::Char('[') => KeyIntent::Fov(-1),
        KeyCode::Char(']') => KeyIntent::Fov(1),
        // `Z` — toggle the FOV-ZOOM (binoculars); `G` — cycle the live post-fx
        // stack. Both are one-shot gameplay edges consumed by `re_game`.
        KeyCode::Char('z') | KeyCode::Char('Z') => KeyIntent::ZoomToggle,
        KeyCode::Char('g') | KeyCode::Char('G') => KeyIntent::PostFxCycle,
        // F3 toggles the on-screen perf overlay. It must NOT also produce any
        // movement or menu-nav intent — `TogglePerf` maps to None in menu_nav.
        KeyCode::F(3) => KeyIntent::TogglePerf,
        // Shift = sprint modifier. On the kitty path these arrive as dedicated
        // `KeyCode::Modifier(ModifierKeyCode::LeftShift / RightShift)` events
        // (press/release) so the held-set can track them cleanly. On non-kitty
        // terminals sprint is detected via KeyModifiers::SHIFT on Char events instead.
        KeyCode::Modifier(ModifierKeyCode::LeftShift | ModifierKeyCode::RightShift) => {
            KeyIntent::Sprint
        }
        // `M` — toggle mouse-look on/off. One-shot: flips MouseSettings::enabled.
        // When the OS pointer drifts to the terminal edge and stalls, the player
        // presses M to disable mouse-look and rely entirely on keyboard turning
        // (always edge-free). A second M re-enables it. The free pointer never
        // fights keyboard look when mouse-look is off.
        KeyCode::Char('m') | KeyCode::Char('M') => KeyIntent::MouseLookToggle,
        // Everything else: no binding.
        _ => KeyIntent::Unmapped,
    }
}

/// The DEFAULT (terminal) present backend. Re-exported from the crate root as
/// `PresentPlugin` when `backend-terminal` is active.
pub struct TerminalBackendPlugin;

impl Plugin for TerminalBackendPlugin {
    fn build(&self, app: &mut App) {
        // RatatuiPlugins installs raw mode + alt-screen + the crossterm event
        // pump (KeyMessage / MouseMessage / ResizeMessage / ...). It restores
        // the terminal on drop, so we need no manual teardown system.
        //
        // `enable_mouse_capture: true` turns on bevy_ratatui's MousePlugin,
        // which emits `EnableMouseCapture` at startup and — via the
        // `MouseEnabled` resource's `Drop` impl — `DisableMouseCapture` on
        // teardown. So mouse capture's lifecycle is tied to app shutdown exactly
        // like raw mode: it is always restored on quit (including ESC/q).
        //
        // KITTY KEYBOARD PROTOCOL — we KEEP the default `enable_kitty_protocol:
        // true`. On Linux (kitty, foot, …) bevy_ratatui's `KittyPlugin` probes
        // with `supports_keyboard_enhancement()` and, if supported, pushes
        // `KeyboardEnhancementFlags::all()` (incl. `REPORT_EVENT_TYPES`) at
        // startup and pops it on teardown via `KittyEnabled::Drop`. On Windows
        // that probe always returns `false` (hardcoded in crossterm's
        // windows sys module), so `KittyEnabled` is never inserted by bevy_ratatui
        // even on Windows Terminal / WezTerm. `force_kitty_sys` (PostStartup)
        // detects this and bypasses the probe — see the module-level doc and
        // the `force_kitty_sys` / `KittyGuard` code below. `enable_input_-
        // forwarding` stays off (we consume the crossterm messages directly).

        // Install a panic hook that restores the terminal cursor AND pops any
        // kitty keyboard-enhancement flags before the default hook prints the
        // panic message. Without this, a panic while the cursor is hidden leaves
        // the user's shell with no visible cursor — even though RatatuiPlugins's
        // Drop restores raw-mode/alt-screen, it does NOT guarantee a cursor Show
        // if the stack unwinds past it before its Drop runs. Similarly,
        // KittyGuard::drop runs during the Bevy world teardown that follows a
        // graceful shutdown, but does NOT run when a panic short-circuits the
        // normal shutdown. We therefore pop the kitty flags here (best-effort)
        // so the terminal is not left in kitty mode after a crash.
        //
        // We chain our hook in FRONT of whatever hook is already installed
        // (the default), so cursor + kitty are restored FIRST, then the normal
        // panic message appears.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // FIRST — before anything else, and before any `World`/resource
            // `Drop` runs (that only happens once unwinding actually proceeds,
            // AFTER this hook returns): force the terminal back to the main
            // screen and CLEAR it. This is the OPENING boot sequence's "exit
            // clears the main screen" contract (`intro_tear` module doc)
            // extended to the panic path. We cannot rely on
            // `RatatuiContext::Drop` (which also leaves the alt screen) having
            // run before this hook, nor on ITS drop racing ours in any
            // particular order relative to a hypothetical teardown resource —
            // Bevy's `World` resource drop order is unspecified. Calling this
            // explicitly, here, at the earliest possible moment, sidesteps
            // that entirely: every operation it performs (leave alt screen,
            // disable raw mode, clear, show cursor) is idempotent, so the
            // later `Drop`s repeating the same steps are harmless no-ops.
            crate::intro_tear::finalize_terminal_teardown();
            // Best-effort: if stdout can't be written, we can't do much.
            use bevy_ratatui::crossterm::cursor::Show;
            use bevy_ratatui::crossterm::execute;
            let _ = execute!(std::io::stdout(), Show);
            // Release the OS cursor grab (ShowCursor(TRUE) on Windows, no-op
            // elsewhere). Must run before the default hook prints the panic message
            // so the user's cursor is visible when they read the backtrace.
            OsCursorGrab::release_on_panic();
            // Also restore any OS cursor hidden by the simple-hide path (Feature A).
            // A second ShowCursor(TRUE) on a cursor that was already visible is
            // harmless — the counter goes above 0 and the cursor stays visible.
            OsCursorSimpleHide::release_on_panic();
            // Pop kitty flags via the raw path so the pop works on Windows too
            // (the standard crossterm execute path returns Err(Unsupported) there).
            pop_kitty_raw();
            prev_hook(info);
        }));

        app.add_plugins(RatatuiPlugins {
            enable_mouse_capture: true,
            ..Default::default()
        })
        // `ColorMode::default()` auto-detects from the environment
        // (`ColorMode::resolve(None)`); `re_game` overrides via the `--color`
        // flag / the saved setting. `init_resource` (insert-if-absent) keeps a
        // flagless build correct even with no `re_game` layer at all.
        .init_resource::<ColorMode>()
        // Player-tunable mouse-look (on/off + sensitivity). `re_game`
        // overrides this from persisted settings; `init_resource` keeps a
        // flagless / settings-less build correct (historical feel).
        .init_resource::<MouseSettings>()
        // Gate resources (insert-if-absent so plugin order doesn't matter).
        // `InputGate` defaults to camera_motion=false (fail-safe: no motion
        // until re_game enables it on entering gameplay).
        .init_resource::<InputGate>()
        // `PerfOverlay` defaults to hidden (off by default; F3 toggles it).
        .init_resource::<PerfOverlay>()
        // `FrameTimingMetrics` defaults to all-zeros; present_sys fills it.
        .init_resource::<FrameTimingMetrics>()
        // Backend-neutral interact action, consumed by `re_game`.
        .add_message::<InteractPressed>()
        // Backend-neutral menu-navigation action (additive, alongside the
        // camera motion above). Consumed by `re_game`'s flow in the menu
        // states; ignored in `Playing`.
        .add_message::<MenuNav>()
        // Backend-neutral mouse HOVER/CLICK at a terminal cell position, so
        // `re_game`'s menu screens can be mouse-driven (hover moves the
        // highlight, a left-click commits it) alongside the existing
        // keyboard `MenuNav` path. See `emit_menu_mouse_sys`'s doc for why
        // these are suppressed while the OS cursor grab is active (i.e.
        // during `Playing` mouse-look) rather than gated on `GameState`,
        // which `re_present` cannot see.
        .add_message::<MenuMouseMove>()
        .add_message::<MenuMouseClick>()
        // Backend-neutral FOV-cycle action, emitted by the gamepad FOV button
        // and the keyboard `[` / `]` bindings; consumed by `re_game`.
        .add_message::<FovCycle>()
        // Backend-neutral FOV-ZOOM toggle (`Z`) and POST-FX cycle (`G`),
        // consumed by `re_game`'s zoom/post-fx systems.
        .add_message::<ZoomToggle>()
        .add_message::<PostFxCycle>()
        // Per-frame cursor position, for deriving relative mouse motion.
        .init_resource::<MouseTracker>()
        // The currently-held WASD/arrow set, maintained from crossterm
        // press/release events when the kitty protocol is active (see
        // [`input_sys`] / [`held_move_sys`]). Empty + unused on terminals
        // without release reporting (those use the per-event fallback).
        .init_resource::<HeldKeys>()
        // Per-direction last-refresh timestamps, used by `expire_held_sys` to
        // auto-clear any held flag that was never released (safety net for
        // terminals that ignore kitty bytes and thus never send Release events).
        // On kitty-capable terminals Release clears the flag before the timeout,
        // making `expire_held_sys` a no-op in practice on those paths.
        .init_resource::<HeldTimestamps>()
        // OS cursor grab for real FPS mouse-look (Win32 on Windows, X11 on
        // Linux). Starts inactive; activated by the M toggle when mouse-look
        // is turned on. Defaults to inactive (no OS grab until M is pressed).
        .init_resource::<OsCursorGrab>()
        // Simple OS-cursor hide (Feature A): hides the system arrow pointer
        // WITHOUT the recenter/poll logic of the grab. Starts not-hidden;
        // `startup_render_config_sys` hides it in Startup if `RenderConfig`
        // requests it. Its Drop restores the pointer on normal exit; the
        // panic hook above is a belt-and-suspenders restore on crash.
        .init_resource::<OsCursorSimpleHide>()
        // RenderConfig carries the hide-cursor switch + optional fixed grid
        // size. `init_resource` supplies the safe default (no hide, auto-size)
        // when the binary did not insert one. Binaries insert their own before
        // adding PresentPlugin so that insert_resource wins over init_resource.
        .init_resource::<RenderConfig>()
        // Present-layer temporal memory (idle-frame skip + colour deadband).
        // `PresentConfig` is seeded ONCE from `TS_PRESENT_DEADBAND` here; the
        // per-frame path never touches the environment.
        .init_resource::<PresentCache>()
        .insert_resource(PresentConfig::from_env())
        // `ensure_player_motion` gives the camera a `PlayerMotion` so the
        // gravity tick has state; `gravity_sys` runs the per-frame
        // grounded physics. Ordered after `input_sys` so a jump pressed this
        // frame is integrated by gravity in the SAME frame.
        .add_systems(
            Update,
            (
                ensure_player_motion,
                input_sys,
                // Safety-net expiry: clears any held direction not refreshed
                // within HELD_EXPIRE_SECS. Runs after input_sys (which may have
                // just refreshed timestamps this frame) and before held_move_sys
                // (which reads the held set). On kitty-capable terminals this is
                // always a no-op because Release events clear flags immediately.
                expire_held_sys.after(input_sys),
                // Continuous held-key movement (kitty path only). Gated by
                // `InputGate.camera_motion` so it freezes during pause/menus.
                // Ordered after `expire_held_sys` (which may have cleared stale
                // flags this frame) and before `gravity_sys` (which integrates
                // the vertical step on the resulting pose).
                held_move_sys
                    .after(expire_held_sys)
                    .run_if(|g: Res<InputGate>| g.camera_motion),
                gravity_sys
                    .after(held_move_sys)
                    .run_if(|g: Res<InputGate>| g.camera_motion),
                // mouse_look_sys runs ALWAYS in the game (ungated at the
                // schedule level) because it must keep updating
                // `tracker.last` even during pause — so that re-entering
                // Playing doesn't see a stale position delta and jump the
                // camera. The gate is checked INSIDE the system: it
                // consumes motion events (updates the tracker) but skips
                // writing cam.yaw / cam.pitch when `!camera_motion`.
                //
                // In the editor (`RenderConfig.is_editor = true`) the
                // system is SUPPRESSED entirely via the run-condition below.
                // Without this guard, the system hits its no-camera
                // early-return (the editor has no CameraPose entity) and
                // calls `os_grab.deactivate()` every frame — thrashing the
                // OS grab the editor's own preview-look system relies on.
                //
                // `sync_os_grab_sys` runs BEFORE this so `os_grab.active`
                // already reflects the current state by the time this reads
                // it (see that system's doc for the full fix rationale).
                sync_os_grab_sys
                    .before(mouse_look_sys)
                    .run_if(|cfg: Res<RenderConfig>| !cfg.is_editor),
                mouse_look_sys.run_if(|cfg: Res<RenderConfig>| !cfg.is_editor),
                // Menu mouse (hover/click), suppressed in the editor for the
                // same reason as the two systems above (no `InputGate`/
                // `MouseSettings` policy applies there — `ts_editor` owns its
                // own mouse path via `EditorMouse`).
                emit_menu_mouse_sys.run_if(|cfg: Res<RenderConfig>| !cfg.is_editor),
                resize_sys,
            ),
        );
        // Cross-platform gamepad (gilrs) — additive, drives the same outputs.
        add_gamepad(app);
        // Hide the terminal cursor while gameplay runs (best-effort); restored on
        // exit by the alt-screen teardown RatatuiPlugins owns.
        app.add_systems(Startup, hide_terminal_cursor);
        // Stamp the console title with the marker `os_cursor::find_host_window`
        // (Windows) looks for, so the OS cursor grab can find the REAL, visible
        // host window even under Windows Terminal (see that module's doc for
        // why `GetConsoleWindow()` alone is unreliable there). No-op elsewhere.
        app.add_systems(Startup, stamp_console_title_sys);
        // Apply RenderConfig: hide the OS mouse pointer if requested, and
        // attempt a best-effort terminal window resize for a fixed grid size.
        // Runs in Startup (after resources are inserted by the binary, before
        // the first frame). The OS pointer restore is handled by
        // OsCursorSimpleHide::Drop (normal exit) and the panic hook (crash).
        app.add_systems(Startup, startup_render_config_sys);
        // Force-enable the kitty keyboard protocol on terminals where
        // bevy_ratatui's auto-probe failed (always on Windows — crossterm's
        // supports_keyboard_enhancement() is hardcoded to return false there,
        // even on Windows Terminal / WezTerm). We run in `PostStartup` so that
        // ALL `Startup` schedule systems — including bevy_ratatui's `kitty_setup`
        // — have already completed. This guarantees `force_kitty_sys` sees the
        // final value of `Option<Res<KittyEnabled>>` and never races with or
        // double-pushes over bevy_ratatui's own successful push. The KittyGuard
        // resource it inserts handles the pop on teardown; the panic hook above
        // handles the pop on crash.
        app.add_systems(PostStartup, force_kitty_sys);
        app.add_systems(Last, present_sys.in_set(PresentSet));

        // Editor input seam: emit backend-neutral EditorKey/EditorMouse events
        // (ts_editor only).
        #[cfg(feature = "editor")]
        {
            app.add_message::<crate::EditorKey>();
            app.add_message::<crate::EditorMouse>();
            app.add_systems(Update, (emit_editor_keys, emit_editor_mouse));
        }
    }
}

/// Map ratatui (crossterm) key events → the held-key SET + the one-shot actions
/// (jump / interact / FOV) + the backend-neutral [`MenuNav`].
///
/// ## Simultaneous movement: the kitty-protocol held-key path vs. the fallback
///
/// Most terminals send only key-PRESS events (and OS autorepeat re-presses) with
/// **no key-release**, so you cannot know which keys are *currently held* → true
/// simultaneous diagonal movement (hold W+A, release one, keep going on the other)
/// is impossible by default. The fix is the **kitty keyboard protocol**: when the
/// terminal supports it, `bevy_ratatui`'s `RatatuiPlugins` (with the default
/// `enable_kitty_protocol: true`, which we keep) pushes
/// `KeyboardEnhancementFlags::all()` — including `REPORT_EVENT_TYPES` — at startup
/// and pops it on teardown (via the `KittyEnabled` resource's `Drop`), so the
/// terminal is restored cleanly on exit with NO manual flag handling here. We then
/// receive explicit Press / Repeat / **Release** events.
///
/// * **Kitty active** (`KittyEnabled` present — kitty, WezTerm, recent Windows
///   Terminal, foot, …): we maintain the [`HeldKeys`] set — Press/Repeat of a
///   WASD/arrow key SETS its flag, Release CLEARS it — and [`held_move_sys`]
///   applies continuous, time-based movement for ALL held keys every frame. So
///   W+A = a smooth diagonal, and releasing either leaves the other moving, in any
///   order. No per-event stepping happens here for movement.
/// * **Kitty unsupported** (`KittyEnabled` absent — no release events): we FALL
///   BACK to per-event stepping — one Press (incl. autorepeat-as-Press) of a
///   movement/turn key = one [`MOVE_STEP`] / [`TURN_STEP`]. Still responsive (the
///   step was bumped), but it cannot do true simultaneous diagonals (the terminal
///   never tells us two keys are down at once). This degrades honestly.
///
/// ## The map (unchanged from the prior player-bug fix)
///
/// **WASD are ALL movement** — W forward, S back, A strafe LEFT, D strafe RIGHT —
/// COLLISION-CHECKED. **Arrows turn**: ←/→ yaw, ↑/↓ pitch (clamped to
/// ±[`crate::PITCH_LIMIT`]). **Space → jump** (grounded-only). **E/Enter →**
/// [`InteractPressed`]. **Esc → [`MenuNav::Back`]** only (`q` UNMAPPED — no quit).
///
/// ## Menu nav stays debounced (a held key must not storm the menu)
///
/// One-shot actions and [`MenuNav`] fire ONLY on the **Press** edge (never on
/// Repeat or Release), so holding a movement key produces continuous *gameplay*
/// motion but exactly one menu step per physical tap. On a non-kitty terminal the
/// OS autorepeat arrives as Press events, preserving the prior held-to-scroll feel
/// in menus; with kitty, autorepeat is a distinct Repeat kind we deliberately skip
/// for nav. `MenuNav` is state-gated downstream (inert in gameplay, read in menus).
// Bevy systems take their dependencies as params; this one needs the event reader,
// the kitty-support flag, the held set + timestamps, the camera, the map, and three
// message writers — one over clippy's arg cap. Same idiomatic allow as
// `poll_gamepad_sys`.
#[allow(clippy::too_many_arguments)]
fn input_sys(
    mut keys: MessageReader<KeyMessage>,
    kitty: Option<Res<KittyEnabled>>,
    gate: Res<InputGate>,
    mut held: ResMut<HeldKeys>,
    mut timestamps: ResMut<HeldTimestamps>,
    time: Res<Time>,
    mut cam_q: Query<(&mut CameraPose, &mut PlayerMotion)>,
    map: Res<Map>,
    mut interact: MessageWriter<InteractPressed>,
    mut nav: MessageWriter<MenuNav>,
    mut fov: MessageWriter<FovCycle>,
    mut zoom: MessageWriter<ZoomToggle>,
    mut postfx: MessageWriter<PostFxCycle>,
    mut perf_overlay: ResMut<PerfOverlay>,
    mut mouse_settings: ResMut<MouseSettings>,
    mut os_grab: ResMut<OsCursorGrab>,
) {
    // Does this terminal report key-release events? `KittyEnabled` is inserted by
    // `bevy_ratatui` ONLY when the kitty protocol was actually enabled (terminal
    // supports it), so its presence is our single, authoritative support signal —
    // no second `supports_keyboard_enhancement()` probe needed.
    let kitty = kitty.is_some();

    // The camera may be absent (menu states). We still process MenuNav / interact;
    // only the camera-motion/turn arms are skipped when it's missing.
    let mut cam = cam_q.single_mut().ok();

    for key in keys.read() {
        // KeyMessage derefs to crossterm::event::KeyEvent.
        let intent = key_intent(key.code);

        // --- Held-set maintenance (kitty path only). ---
        // A Release CLEARS the held flag AND its timestamp (so the expiry safety-
        // net never fires spuriously after a normal release). A Press or Repeat
        // SETS the flag AND refreshes the timestamp (so the expiry timer resets
        // on every OS-autorepeat event, keeping the flag alive during an active
        // hold on capable terminals).
        //
        // On a non-kitty terminal no Release ever arrives, so we leave the held
        // set untouched there and use the per-event fallback below instead. The
        // expire_held_sys safety net clears any latched flags on those terminals.
        let now = time.elapsed_secs_f64();
        if kitty {
            match key.kind {
                KeyEventKind::Release => {
                    held.set(intent, false);
                    timestamps.clear(intent);
                    // Release never navigates / acts / steps.
                    continue;
                }
                KeyEventKind::Press | KeyEventKind::Repeat => {
                    held.set(intent, true);
                    timestamps.refresh(intent, now);
                }
            }
        } else if key.kind == KeyEventKind::Release {
            // Defensive: a stray Release on a non-kitty terminal triggers nothing.
            continue;
        } else {
            // Non-kitty Press/Repeat: refresh the timestamp so expire_held_sys
            // can track the last-activity time even on terminals without Release
            // reporting. (The held flag itself stays false on this path — the
            // per-event fallback below drives movement directly without using
            // HeldKeys — but we still update the timestamp so that IF a legacy
            // terminal somehow sets a flag, the expiry can clear it.)
            timestamps.refresh(intent, now);
        }

        // --- Menu nav + one-shot actions: PRESS edge ONLY (debounced). ---
        // Skipping Repeat keeps a held key from storming the menu (kitty repeats
        // come as a distinct Repeat kind; non-kitty autorepeat arrives as Press,
        // preserving the historical held-scroll in menus).
        if key.kind == KeyEventKind::Press {
            // Backend-neutral menu navigation (additive — see [`MenuNav`]). `q` is
            // Unmapped → emits nothing (no accidental Back/quit).
            // MenuNav is NOT gated: menus/pause must still respond regardless of
            // whether camera motion is allowed.
            if let Some(n) = intent.menu_nav() {
                nav.write(n);
            }
            // One-shot actions. Some are gated (only when camera motion is
            // permitted); others are always active.
            match intent {
                // InteractPressed is a gameplay action: gated so it cannot fire
                // while paused or in menus where the camera is frozen.
                KeyIntent::Interact => {
                    if gate.camera_motion {
                        interact.write(InteractPressed);
                    }
                }
                // FOV cycle is always available (can be used in menus too).
                KeyIntent::Fov(step) => {
                    fov.write(FovCycle(step));
                }
                // FOV-ZOOM toggle + POST-FX cycle: gameplay actions, gated like
                // interact so they cannot fire while paused / in menus where the
                // camera is frozen. `re_game` gates the consumers to Playing too.
                KeyIntent::ZoomToggle => {
                    if gate.camera_motion {
                        zoom.write(ZoomToggle);
                    }
                }
                KeyIntent::PostFxCycle => {
                    if gate.camera_motion {
                        postfx.write(PostFxCycle);
                    }
                }
                // F3 toggles the perf overlay — always available, no gate.
                KeyIntent::TogglePerf => {
                    perf_overlay.visible = !perf_overlay.visible;
                }
                // M toggles mouse-look on/off — always available, no gate.
                //
                // When toggling ON: attempt to activate the OS cursor grab (Win32
                // GetCursorPos/SetCursorPos recenter, ShowCursor(FALSE)). If the
                // grab succeeds, `os_grab.active` is true and `mouse_look_sys` will
                // use pixel-resolution OS deltas instead of cell-resolution crossterm
                // events — this removes the edge-stall problem entirely. If the grab
                // fails (console not foreground, OS call failed), mouse-look still
                // works via the crossterm cell-event fallback.
                //
                // When toggling OFF: release the OS grab (ShowCursor(TRUE)) and fall
                // back to keyboard turning — always edge-free.
                //
                // When disabled, mouse events are still consumed each frame
                // (by mouse_look_sys) so the tracker.last stays current: toggling
                // back ON never produces a stale-position jump.
                KeyIntent::MouseLookToggle => {
                    mouse_settings.enabled = !mouse_settings.enabled;
                    if mouse_settings.enabled {
                        // Attempt to activate the OS cursor grab. The row count for
                        // the cell-size estimate is 0 here (we don't read the
                        // framebuffer in this system to stay under Bevy's 16-param
                        // limit); the resize_sys updates it on the next resize event,
                        // and the fallback DEFAULT_PIXELS_PER_CELL is used until then.
                        os_grab.activate(0);
                        // (If activate returns false, the crossterm fallback is used.)
                    } else {
                        os_grab.deactivate();
                    }
                }
                _ => {}
            }
        }

        // Camera-motion arms are gated: skip if the gate is closed (pause/menus).
        // MenuNav and TogglePerf are already handled above; only the physical
        // camera movement / jump must be frozen during pause.
        if !gate.camera_motion {
            continue;
        }

        // Camera-coupled arms need a live camera.
        let Some((cam, motion)) = cam.as_mut() else {
            continue;
        };

        // JUMP: Space on the PRESS edge only (one impulse per tap; never on Repeat
        // or Release). Call unconditionally — `try_jump` is its own gate
        // (grounded, coyote-time grace window, or buffered for landing); no
        // external `grounded` check is needed here (Swaggerfall-study item 5:
        // coyote time + jump buffer). `gravity_sys` integrates the result and
        // fires any buffered press automatically on landing.
        if key.kind == KeyEventKind::Press && intent == KeyIntent::Jump {
            CameraPose::try_jump(motion);
        }

        // --- Per-event movement/turn FALLBACK (non-kitty terminals only). ---
        // With release reporting we drive movement from the held set in
        // `held_move_sys` instead, so skip the discrete step here. Without it, one
        // Press (incl. autorepeat-as-Press) = one collision-checked step — the
        // responsive repeat path. (Repeat/Release are already filtered above.)
        //
        // Sprint on the fallback path: crossterm does NOT send dedicated LeftShift /
        // RightShift key events on non-kitty terminals — instead the shift bit is
        // carried as a modifier flag on the Char event (e.g. Char('W') with
        // KeyModifiers::SHIFT set). We read it here and multiply MOVE_STEP by
        // SPRINT_MULTIPLIER when Shift is held, matching the kitty-path behaviour
        // from `held_move_sys`. Pressing Shift alone (without a movement key) emits
        // KeyIntent::Sprint which falls through the `_` arm and does nothing — safe.
        if !kitty {
            let sprint_active = key.modifiers.contains(KeyModifiers::SHIFT);
            let step = if sprint_active {
                MOVE_STEP * SPRINT_MULTIPLIER
            } else {
                MOVE_STEP
            };
            let (dx, dy) = cam.dir();
            // Strafe basis: facing rotated −90° (right-hand) = (dy, -dx).
            let (sx, sy) = (dy, -dx);
            match intent {
                KeyIntent::Forward => {
                    cam.move_with_collision(dx * step, dy * step, &map, PLAYER_RADIUS);
                }
                KeyIntent::Backward => {
                    cam.move_with_collision(-dx * step, -dy * step, &map, PLAYER_RADIUS);
                }
                KeyIntent::StrafeLeft => {
                    cam.move_with_collision(-sx * step, -sy * step, &map, PLAYER_RADIUS);
                }
                KeyIntent::StrafeRight => {
                    cam.move_with_collision(sx * step, sy * step, &map, PLAYER_RADIUS);
                }
                KeyIntent::TurnLeft => cam.yaw -= TURN_STEP,
                KeyIntent::TurnRight => cam.yaw += TURN_STEP,
                KeyIntent::PitchUp => {
                    cam.pitch = (cam.pitch + TURN_STEP).clamp(-PITCH_LIMIT, PITCH_LIMIT);
                }
                KeyIntent::PitchDown => {
                    cam.pitch = (cam.pitch - TURN_STEP).clamp(-PITCH_LIMIT, PITCH_LIMIT);
                }
                _ => {}
            }
        }
    }
}

/// Safety-net expiry for the [`HeldKeys`] set: clears any held direction that has
/// not been refreshed within [`crate::HELD_EXPIRE_SECS`] seconds.
///
/// ### Why this exists
///
/// When `force_kitty_sys` force-enables the kitty keyboard protocol on Windows
/// via a raw VT write, terminals that do NOT understand the protocol simply ignore
/// the bytes. No Release events ever arrive from those terminals, so without this
/// system any key-Press would latch a `HeldKeys` flag "on" forever → the camera
/// would drift endlessly after a single tap.
///
/// ### No-op on capable terminals
///
/// On kitty-capable terminals (kitty, Windows Terminal, WezTerm, foot, …) every
/// key-Release event arrives and clears the held flag — and its timestamp — in
/// `input_sys` immediately. By the time `expire_held_sys` runs, the slot is already
/// `None`, so the `expire` call exits the macro body for each field without touching
/// anything. The system costs one `Time` read + eight `Option::is_some()` checks per
/// frame: effectively zero overhead on the fast path.
///
/// ### No stutter on active holds
///
/// On terminals without Release reporting the timestamp is refreshed on every
/// Press/Repeat event (OS autorepeat fires every ~30–50 ms after the initial ~500 ms
/// delay). A sustained hold therefore keeps resetting the timer well within
/// `HELD_EXPIRE_SECS = 0.75 s`, so the expiry never fires during active holding —
/// only after the player has physically released the key and no further events arrive.
fn expire_held_sys(
    time: Res<Time>,
    mut held: ResMut<HeldKeys>,
    mut timestamps: ResMut<HeldTimestamps>,
) {
    let now = time.elapsed_secs_f64();
    timestamps.expire(&mut held, now);
}

/// Continuous, time-based movement from the [`HeldKeys`] set — the kitty-protocol
/// path that makes WASD work SIMULTANEOUSLY and faster on the terminal backend.
///
/// Runs every frame (after [`input_sys`] has updated the held set this frame, and
/// before [`gravity_sys`]). It is a no-op when the held set is empty — which is
/// ALWAYS the case on non-kitty terminals (their movement goes through the
/// per-event fallback in `input_sys`, and the set is never populated), so this
/// system costs one early-out there and changes nothing.
///
/// Movement is `MOVE_SPEED * dt` along the combined held direction (so W+A is a
/// normalized diagonal, not √2 faster), COLLISION-CHECKED via
/// [`CameraPose::move_with_collision`] so it still slides along walls. Turning is
/// `TURN_SPEED * dt` from the held arrows, with pitch clamped to ±[`PITCH_LIMIT`].
fn held_move_sys(
    time: Res<Time>,
    held: Res<HeldKeys>,
    map: Res<Map>,
    mut cam_q: Query<&mut CameraPose>,
) {
    if held.is_empty() {
        return; // nothing held (or non-kitty terminal) → nothing to do.
    }
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let Ok(mut cam) = cam_q.single_mut() else {
        return; // menu state: no camera.
    };

    // Continuous move from ALL held WASD keys, normalized + collision-checked.
    // Apply SPRINT_MULTIPLIER when Shift is held (pure modifier — the sprint flag
    // changes only speed, never the movement direction or which keys are active).
    let (dirx, diry) = cam.dir();
    let speed = if held.sprint {
        MOVE_SPEED * SPRINT_MULTIPLIER
    } else {
        MOVE_SPEED
    };
    let (mdx, mdy) = held_move_delta(*held, dirx, diry, speed, dt);
    if mdx != 0.0 || mdy != 0.0 {
        cam.move_with_collision(mdx, mdy, &map, PLAYER_RADIUS);
    }

    // Continuous turn from the held arrows (yaw unbounded, pitch clamped).
    let (dyaw, dpitch) = held_turn_delta(*held, TURN_SPEED, dt);
    if dyaw != 0.0 {
        cam.yaw += dyaw;
    }
    if dpitch != 0.0 {
        cam.pitch = (cam.pitch + dpitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }
}

/// Per-frame grounded-physics tick: gravity + floor/ceiling snap, applied to the
/// camera every frame so it falls and stays grounded by default. Uses the real
/// frame delta from [`Time`]. No-op in menu states (no camera entity).
///
/// This is the present-layer wiring of [`CameraPose::step_gravity`]; all the
/// physics math lives in `re_world`. It runs AFTER `input_sys` so a jump pressed
/// this frame gets its first integration step immediately.
fn gravity_sys(
    time: Res<Time>,
    map: Res<Map>,
    mut cam_q: Query<(&mut CameraPose, &mut PlayerMotion)>,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    if let Ok((mut cam, mut motion)) = cam_q.single_mut() {
        cam.step_gravity(&mut motion, dt, &map);
    }
}

/// Hide the terminal cursor at startup so it doesn't blink over the 3D view.
/// This is the FIRST hide; afterwards ratatui's `draw` re-hides it on every
/// presented frame (its frame sets no cursor position), and [`present_sys`] flips
/// ratatui's internal `hidden_cursor` flag on the first present — the protocols
/// enabled after startup can re-show it (player bug #4), and the first real draw
/// puts it back.
///
/// ## What `crossterm::cursor::Hide` actually hides
///
/// `crossterm::cursor::Hide` hides the terminal **text insertion caret** — the
/// blinking bar/block that marks where the next character will appear. It does NOT
/// hide the **OS mouse pointer** (the arrow/hand cursor controlled by the window
/// manager). Hiding the OS pointer from a TUI app is not possible via any portable
/// terminal escape sequence; it requires an OS-level or window-manager API that is
/// outside crossterm's scope.
///
/// ## Mouse capture and the OS pointer
///
/// We enable `enable_mouse_capture: true` in `RatatuiPlugins` (see the plugin's
/// `build`). `EnableMouseCapture` causes many terminals — including **Windows
/// Terminal** and **WezTerm** — to automatically hide the OS pointer while the
/// mouse is inside the window and mouse-capture events are being consumed. This is
/// a terminal emulator feature, not a TUI escape. In practice:
///
/// * **Windows Terminal ≥ 1.14, WezTerm**: the pointer auto-hides while captured.
///   Re-activating the window re-shows it, but it hides again on next movement.
/// * **kitty, foot**: same auto-hide behaviour.
/// * **Legacy conhost (Windows old cmd.exe)**: the OS pointer stays visible. No
///   workaround exists there; this is a known TUI limitation.
/// * **SSH sessions**: depends on the client terminal, not the server.
///
/// ## Residual limitation (accepted, documented, not fixable in this crate)
///
/// If a player's terminal does NOT auto-hide the pointer on mouse capture, the OS
/// arrow cursor will remain visible over the 3D view for the entire session. There
/// is no crossterm API to force-hide it, no portable VT escape for it, and
/// implementing a platform-specific workaround (e.g. calling `SetCursor(NULL)` on
/// Windows) is outside `re_present`'s scope and would require `unsafe`. Players on
/// affected terminals should use keyboard turn (arrows) as the always-available
/// fallback — mouse-look was always additive and optional, never required.
///
/// TERMINAL LIMITATION (mouse-look infinite-look): a terminal reports only ABSOLUTE
/// cell coordinates and offers NO pointer-warp, so true infinite mouse-look (like
/// the window backend's locked+hidden OS cursor) is also IMPOSSIBLE here — the
/// pointer can reach the window edge and stall. crossterm restores the cursor on
/// alt-screen teardown that `RatatuiPlugins` owns, so exit stays clean.
fn hide_terminal_cursor() {
    use bevy_ratatui::crossterm::cursor::Hide;
    use bevy_ratatui::crossterm::execute;
    let _ = execute!(std::io::stdout(), Hide);
}

/// Startup: apply [`RenderConfig`] — hide the OS mouse pointer if requested
/// (Feature A) and attempt a best-effort terminal window resize for a fixed
/// grid size (Feature B).
///
/// ## Feature A — OS cursor hide (decoupled from mouse-look)
///
/// When [`RenderConfig::hide_os_cursor`] is `true`, calls
/// [`OsCursorSimpleHide::hide`] which issues a platform `ShowCursor(FALSE)` on
/// Windows or the X11 blank-cursor equivalent. The hide is reversed by:
/// * [`OsCursorSimpleHide::Drop`] on a normal exit (Bevy world teardown).
/// * The panic hook installed in [`TerminalBackendPlugin::build`] on a crash
///   (calls [`OsCursorSimpleHide::release_on_panic`]).
///
/// The `OsCursorGrab` (mouse-look `M` toggle) is fully independent: if the
/// grab activates while the plain hide is live, the grab issues its OWN
/// `ShowCursor(FALSE)` (tracked by its own `cursor_hidden` bool) and reverses
/// only that on deactivation. The net Win32 counter remains balanced.
///
/// ## Feature B — Best-effort terminal window resize
///
/// When [`RenderConfig::fixed_size`] is `Some((cols, rows))`, sends
/// `crossterm::terminal::SetSize(cols, rows)` to the terminal. Many terminal
/// emulators honour this (Windows Terminal, WezTerm, xterm); others silently
/// ignore it. Failure is discarded — we never panic on a size request.
///
/// The framebuffer was already created at the fixed size by the binary before
/// `PresentPlugin` was added, so this call only tries to make the window match
/// the framebuffer, not the other way around.
fn startup_render_config_sys(cfg: Res<RenderConfig>, mut simple_hide: ResMut<OsCursorSimpleHide>) {
    if cfg.hide_os_cursor {
        simple_hide.hide();
    }
    if let Some((cols, rows)) = cfg.fixed_size {
        use bevy_ratatui::crossterm::execute;
        use bevy_ratatui::crossterm::terminal::SetSize;
        // Best-effort: ignore the result. A terminal that doesn't support
        // SetSize returns an error; we continue with whatever size it has.
        let _ = execute!(std::io::stdout(), SetSize(cols as u16, rows as u16));
    }
}

/// Last-seen mouse cell position, so successive [`MouseMessage`] events can be
/// turned into a **relative** motion delta.
///
/// The terminal reports ABSOLUTE cell coordinates (column, row) with no pointer
/// warp — there is no "lock cursor to center" like a windowed FPS. We synthesize
/// a relative delta by differencing consecutive positions. `last` is `None`
/// until the first event arrives (and is reset whenever the camera is absent),
/// so the very first sample of a fresh stream produces no jump.
#[derive(Resource, Default)]
struct MouseTracker {
    last: Option<(u16, u16)>,
}

/// Mouse-look: relative cursor motion → camera yaw/pitch, the always-additive
/// companion to keyboard turn (A/D / arrows), reproducing the retired C build.
///
/// ## Two delta sources (priority order)
///
/// 1. **OS cursor grab** (Win32 / X11, when `os_grab.active`): reads the OS
///    pointer position in screen pixels, computes the delta from a fixed anchor,
///    warps the pointer back to the anchor via `SetCursorPos` / `XWarpPointer`.
///    This removes the terminal-edge stall entirely: the pointer never reaches
///    the edge because it is recentered every frame. Pixel-resolution deltas are
///    converted to cell units (px / px_per_cell) so sensitivity feels the same.
///
/// 2. **Crossterm cell events** (fallback when grab inactive or OS call fails):
///    each `Moved`/`Drag` event carries an absolute `(column, row)`; we difference
///    it against `tracker.last` to get a per-step delta. The terminal gives only
///    absolute cell coords and cannot warp the pointer, so the cursor CAN hit the
///    window edge and stall. Keyboard turn is the always-available fallback there.
///
/// Applied ONLY when a camera exists (gameplay, not menus). With no camera the
/// tracker is reset so re-entering gameplay doesn't see a stale-position jump.
fn mouse_look_sys(
    mut mouse: MessageReader<MouseMessage>,
    mut tracker: ResMut<MouseTracker>,
    settings: Res<MouseSettings>,
    gate: Res<InputGate>,
    mut os_grab: ResMut<OsCursorGrab>,
    mut cam_q: Query<&mut CameraPose>,
) {
    let Ok(mut cam) = cam_q.single_mut() else {
        // No camera (menu state): drop crossterm events and reset position
        // so the first gameplay sample starts a fresh, jump-free delta stream.
        // Also deactivate the OS grab — no camera means no look to apply.
        mouse.clear();
        tracker.last = None;
        os_grab.deactivate();
        return;
    };

    let apply = gate.camera_motion && settings.enabled;

    // ── Path 1: OS cursor grab (Win32 / X11 pixel-resolution) ────────────────
    // When the grab is active, consume OS pointer deltas here and skip the
    // crossterm cell-event path for the look application. We still drain the
    // crossterm mouse event queue so it doesn't pile up (below), but we don't
    // apply those deltas when the OS path is live.
    if os_grab.active {
        if apply {
            if let Some((dx, dy)) = os_grab.poll_delta() {
                if dx != 0.0 || dy != 0.0 {
                    let (yaw, pitch) =
                        apply_mouse_look(cam.yaw, cam.pitch, dx, dy, settings.sensitivity);
                    cam.yaw = yaw;
                    cam.pitch = pitch;
                }
            }
        } else {
            // Gate closed or mouse-look disabled: still call poll_delta to issue
            // the recenter warp (keeps the pointer pinned even while paused) but
            // discard the returned delta.
            let _ = os_grab.poll_delta();
        }
        // Drain crossterm mouse events so they don't accumulate. We don't update
        // tracker.last here — on re-enabling the crossterm path we want a fresh
        // start (set tracker.last to None so the first event is anchor-only).
        mouse.clear();
        tracker.last = None;
        return;
    }

    // ── Path 2: Crossterm cell events (fallback) ──────────────────────────────
    // Consume all motion events this frame, updating the tracker regardless of
    // the gate or the settings toggle. The anti-jump invariant: keeping
    // `tracker.last` current while paused means re-entering Playing won't
    // accumulate a stale position delta and lurch the camera. The gate and the
    // settings toggle only suppress the yaw/pitch WRITE.
    for ev in mouse.read() {
        match ev.kind {
            MouseEventKind::Moved | MouseEventKind::Drag(_) => {}
            _ => continue,
        }
        let pos = (ev.column, ev.row);
        if apply {
            if let Some((px, py)) = tracker.last {
                let dx = pos.0 as f32 - px as f32;
                let dy = pos.1 as f32 - py as f32;
                if dx != 0.0 || dy != 0.0 {
                    let (yaw, pitch) =
                        apply_mouse_look(cam.yaw, cam.pitch, dx, dy, settings.sensitivity);
                    cam.yaw = yaw;
                    cam.pitch = pitch;
                }
            }
        }
        // Always update the tracker, gated or not, so position stays current.
        tracker.last = Some(pos);
    }
}

/// `Startup`: stamp the console/window title marker (see
/// `os_cursor::stamp_console_title`'s doc). One-shot, best-effort, a no-op on
/// non-Windows.
fn stamp_console_title_sys() {
    crate::os_cursor::stamp_console_title();
}

/// State-driven OS cursor grab sync — the fix for the grab only engaging
/// after two `M` presses (`OsCursorGrab.active` used to be
/// flipped ONLY inside the `KeyIntent::MouseLookToggle` arm of `input_sys`, so
/// a fresh entry into `Playing` — or a resume from Pause/menus — left the grab
/// wherever the toggle last set it, with no system reconciling the mismatch
/// until the player happened to press `M` twice).
///
/// Runs every `Update` frame, ordered `.before(mouse_look_sys)` so
/// `os_grab.active` already reflects the CURRENT frame's state by the time
/// `mouse_look_sys` reads it. `os_grab.active` becomes the pure function
/// [`os_cursor::should_be_active`] of two already-live resources:
///
/// * [`MouseSettings::enabled`] — the player's mouse-look on/off setting (the
///   `M` toggle still flips this directly in `input_sys`, unchanged);
/// * [`InputGate::camera_motion`] — `re_game::flow` sets this `true` ONLY in
///   `GameState::Playing` (see that type's doc comment), so it is exactly the
///   "are we actually playing" signal this system needs without depending on
///   `re_game`'s `GameState` (which would be an illegal upward dependency —
///   see `docs/LAYERING.md`).
///
/// This covers every transition the fix needs to cover for free, because both
/// inputs are resources `re_game` already drives on every relevant edge:
/// entering `Playing` (gate flips true), leaving it for Pause/any menu/Won
/// (gate flips false — releasing the grab so those screens can show the OS
/// cursor for mouse-driven navigation), and a live settings-screen toggle of
/// MOUSE-LOOK (flips `MouseSettings.enabled` directly).
///
/// The RISING edge (grab turning ON) additionally requires the console/game
/// window to be the OS foreground window: activating (warping the pointer +
/// hiding the OS cursor) while some OTHER window has focus would steal that
/// window's pointer. The FALLING edge (turning OFF) is unconditional — a
/// release must never be blocked by a foreground check, or alt-tabbing away
/// mid-game would leave the grab stuck, and a menu needs its cursor back
/// immediately regardless of focus history.
fn sync_os_grab_sys(
    gate: Res<InputGate>,
    settings: Res<MouseSettings>,
    mut os_grab: ResMut<OsCursorGrab>,
) {
    let want = crate::os_cursor::should_be_active(settings.enabled, gate.camera_motion);
    if want && !os_grab.active {
        if OsCursorGrab::console_is_foreground() {
            os_grab.activate(0);
        }
        // Else: not foreground yet — try again next frame (e.g. the OS hasn't
        // finished raising our window after a state transition). No `active`
        // change, so this branch costs nothing beyond the one focus check.
    } else if !want && os_grab.active {
        os_grab.deactivate();
    }
}

/// Menu mouse (hover + click): turn crossterm mouse events into the
/// backend-neutral [`MenuMouseMove`]/[`MenuMouseClick`] `re_game` consumes to
/// drive hover-highlight + click-to-commit on every list menu (main menu,
/// category/level select, settings, pause, won).
///
/// ## Why gate on the OS cursor grab, not `GameState`
///
/// `re_present` cannot see `re_game::GameState` (upward dependency — see
/// `docs/LAYERING.md`), so it cannot `run_if(in_state(Menu))`
/// itself. [`OsCursorGrab::active`] is, after [`sync_os_grab_sys`]'s fix, an
/// exact proxy for "are we in `Playing` with mouse-look on": suppressing menu
/// mouse events while the grab is active means they are never emitted during
/// gameplay look (where cell-granularity clicks would be meaningless — there
/// is no menu on screen to hit-test against), and ARE emitted in every menu /
/// pause / splash / won screen, which is exactly the set of states that
/// registers a menu-mouse consumer system in `re_game`. A frame or two right
/// at a state transition where the grab hasn't caught up yet is harmless: the
/// state that has no menu-mouse consumer simply never reads the message, and
/// `MenuMouseMove`/`MenuMouseClick` are ordinary double-buffered messages, not
/// state.
///
/// ## Press-edge only for clicks
///
/// Only `MouseEventKind::Down(Left)` emits a click (never `Up`, never a held
/// `Drag`), mirroring [`MenuNav`]'s press-edge debounce so a held button can
/// never storm a menu with repeated commits.
fn emit_menu_mouse_sys(
    mut mouse: MessageReader<MouseMessage>,
    os_grab: Res<OsCursorGrab>,
    mut hover: MessageWriter<MenuMouseMove>,
    mut click: MessageWriter<MenuMouseClick>,
) {
    if os_grab.active {
        // Gameplay mouse-look owns the pointer; drain without interpreting as
        // menu input (there is no menu to hit-test in this state).
        mouse.clear();
        return;
    }
    for ev in mouse.read() {
        match ev.kind {
            MouseEventKind::Moved | MouseEventKind::Drag(_) => {
                hover.write(MenuMouseMove {
                    col: ev.column,
                    row: ev.row,
                });
            }
            MouseEventKind::Down(bevy_ratatui::crossterm::event::MouseButton::Left) => {
                click.write(MenuMouseClick {
                    col: ev.column,
                    row: ev.row,
                });
            }
            _ => {}
        }
    }
}

/// On a terminal resize, resize the framebuffer to the new cols×rows and
/// refresh the OS cursor grab anchor (so the center stays current after resize).
///
/// Full-window, ANY ratio — the new grid is exactly the terminal's reported
/// size. `Size` is `{ width, height }` in cells; width = columns, height =
/// rows. Only the last resize this tick matters, so we coalesce.
///
/// **Fixed-size mode (Feature B):** when [`RenderConfig::fixed_size`] is
/// `Some(…)`, this system skips the framebuffer resize entirely — the grid
/// stays at the declared size regardless of what the terminal reports. The
/// present layer already clamps its drawing to `min(fb_size, terminal_area)`,
/// so a smaller terminal than the fixed grid simply clips gracefully. The OS
/// grab anchor is still updated (a resize means the window moved).
fn resize_sys(
    mut resizes: MessageReader<ResizeMessage>,
    mut fb: ResMut<Framebuffer>,
    mut os_grab: ResMut<OsCursorGrab>,
    cfg: Res<RenderConfig>,
) {
    let mut latest: Option<(usize, usize)> = None;
    for ResizeMessage(size) in resizes.read() {
        latest = Some((size.width as usize, size.height as usize));
    }
    if let Some((cols, rows)) = latest {
        if cfg.fixed_size.is_none() {
            // Auto-size: follow the terminal's reported dimensions (default).
            fb.resize(cols, rows);
        }
        // Always update the OS grab anchor — a terminal resize also moves the
        // window, so the center (and pixel-per-cell estimate) needs refreshing
        // regardless of whether we resized the framebuffer.
        os_grab.on_resize(rows);
    }
}

/// Pure, terminal-free mapping from a raw crossterm key event to a
/// backend-neutral [`crate::EditorKey`].  Factored out so the mapping is
/// unit-testable without a live terminal or a running Bevy app.
///
/// # Letter normalisation
///
/// For ASCII letters (`c.is_ascii_alphabetic()`), crossterm may deliver either
/// the shifted glyph (`'S'`) with no SHIFT modifier (`CapsLock` is on) or the
/// unshifted glyph (`'s'`) with the SHIFT modifier set (some terminals in
/// extended-key mode).  We normalise to a **consistent contract**:
///
/// - SHIFT modifier set   → uppercase letter, `shift = true`
/// - SHIFT modifier clear → lowercase letter, `shift = false`
///
/// `CapsLock` is therefore invisible to the editor: a `'S'` delivered without the
/// SHIFT modifier (`CapsLock` only) is treated as plain unshifted `'s'`, so the
/// editor's `Char('S')` arms fire only when the user really presses Shift+S.
///
/// Non-letter printable characters (digits, punctuation, space, …) pass through
/// exactly as crossterm delivers them; `shift` still reflects the modifier bit.
///
/// # Return value
///
/// `Some(EditorKey)` for every key the editor cares about — including all
/// function keys (`KeyCode::F(n)` → `EditorKeyCode::F(n)`, unconditionally;
/// the editor decides which `n` it binds), `Delete`/`Home`/`End` (forward-delete
/// and caret-to-line-start/end, driving `ts_editor::widget::{delete,home,end}`)
/// — and `None` for keys with no mapping (e.g. `Insert`, media keys).
#[cfg(feature = "editor")]
#[allow(clippy::disallowed_types)] // re_present is the authorised terminal crate
const fn map_editor_key(
    code: KeyCode,
    modifiers: bevy_ratatui::crossterm::event::KeyModifiers,
) -> Option<crate::EditorKey> {
    use crate::{EditorKey, EditorKeyCode as E};
    use bevy_ratatui::crossterm::event::KeyModifiers;

    let shift = modifiers.contains(KeyModifiers::SHIFT);
    let ecode = match code {
        KeyCode::Up => E::Up,
        KeyCode::Down => E::Down,
        KeyCode::Left => E::Left,
        KeyCode::Right => E::Right,
        KeyCode::PageUp => E::PageUp,
        KeyCode::PageDown => E::PageDown,
        KeyCode::Enter => E::Enter,
        KeyCode::Esc => E::Esc,
        KeyCode::Tab => E::Tab,
        KeyCode::Backspace => E::Backspace,
        KeyCode::Delete => E::Delete,
        KeyCode::Home => E::Home,
        KeyCode::End => E::End,
        // Function keys pass through verbatim (F2 rename, F5 playtest, …). No
        // case normalisation applies — `shift` still reflects the modifier bit
        // exactly as crossterm delivered it (e.g. Shift+F5 arrives as `F(5)`
        // with `shift=true`, letting the editor distinguish chorded variants).
        KeyCode::F(n) => E::F(n),
        KeyCode::Char(c) => {
            // Normalise ASCII letters: the emitted char's case is determined
            // SOLELY by whether the SHIFT modifier is held — CapsLock is ignored.
            // shift=true  → uppercase (e.g. `'S'`), so `Char('S')` editor arms fire.
            // shift=false → lowercase (e.g. `'s'`), so `Char('s')` editor arms fire.
            // CapsLock (uppercase char, no SHIFT modifier) → lowercase, shift=false:
            // treated as the unshifted key.
            let normalised = if c.is_ascii_alphabetic() {
                if shift {
                    c.to_ascii_uppercase()
                } else {
                    c.to_ascii_lowercase()
                }
            } else {
                c
            };
            E::Char(normalised)
        }
        _ => return None,
    };
    Some(EditorKey { code: ecode, shift })
}

/// Map the active backend's raw key events → backend-neutral [`crate::EditorKey`]
/// for `ts_editor`. Gated on the `editor` feature so the game build never compiles
/// it. Keeps `bevy_ratatui`/crossterm confined to `re_present`.
#[cfg(feature = "editor")]
fn emit_editor_keys(mut keys: MessageReader<KeyMessage>, mut out: MessageWriter<crate::EditorKey>) {
    for k in keys.read() {
        if k.kind == KeyEventKind::Release {
            continue;
        }
        if let Some(ek) = map_editor_key(k.code, k.modifiers) {
            out.write(ek);
        }
    }
}

/// Pure translation of a crossterm [`MouseEvent`] → backend-neutral
/// [`crate::EditorMouse`]. Returns `None` for the kinds the editor ignores
/// (plain `Moved` with no button, and the horizontal `ScrollLeft`/`ScrollRight`
/// touchpad gestures). Factored out so the mapping is unit-testable without a
/// terminal or a running app.
// re_present is the intentional consumer of crossterm types — allow the
// disallowed_types lint here (clippy.toml is workspace-global and cannot
// exclude the one crate that is supposed to use these types).
#[cfg(feature = "editor")]
#[allow(clippy::disallowed_types)]
fn map_mouse_event(ev: &bevy_ratatui::crossterm::event::MouseEvent) -> Option<crate::EditorMouse> {
    use crate::{EditorMouse, EditorMouseButton as B, EditorMouseKind as K};
    use bevy_ratatui::crossterm::event::{MouseButton, MouseEventKind};

    let btn = |b: MouseButton| match b {
        MouseButton::Left => B::Left,
        MouseButton::Right => B::Right,
        MouseButton::Middle => B::Middle,
    };
    let (kind, button) = match ev.kind {
        MouseEventKind::Down(b) => (K::Press, Some(btn(b))),
        MouseEventKind::Drag(b) => (K::Drag, Some(btn(b))),
        MouseEventKind::Up(b) => (K::Release, Some(btn(b))),
        MouseEventKind::ScrollUp => (K::ScrollUp, None),
        MouseEventKind::ScrollDown => (K::ScrollDown, None),
        // Plain motion (no button) and horizontal scroll carry no edit intent.
        MouseEventKind::Moved | MouseEventKind::ScrollLeft | MouseEventKind::ScrollRight => {
            return None
        }
    };
    Some(EditorMouse {
        col: ev.column,
        row: ev.row,
        kind,
        button,
    })
}

/// Map the active backend's raw mouse events → backend-neutral
/// [`crate::EditorMouse`] for `ts_editor`. Mirrors [`emit_editor_keys`]: gated on
/// the `editor` feature, confining `bevy_ratatui`/crossterm to re_present. The
/// editor consumes these and hit-tests its panes; this is pure event translation
/// (no terminal writes).
#[cfg(feature = "editor")]
fn emit_editor_mouse(
    mut mouse: MessageReader<MouseMessage>,
    mut out: MessageWriter<crate::EditorMouse>,
) {
    for ev in mouse.read() {
        // `MouseMessage` derefs to crossterm::event::MouseEvent.
        if let Some(m) = map_mouse_event(ev) {
            out.write(m);
        }
    }
}

/// Convert a packed `0x00RRGGBB` color into a ratatui color honoring `mode`.
///
/// `Truecolor` → `Color::Rgb`; `Ansi256` → `Color::Indexed`, which ratatui
/// emits as crossterm `Color::AnsiValue` (an SGR 38/48;5;N sequence).
#[inline]
fn color(c: u32, mode: ColorMode) -> Color {
    let (r, g, b) = rgb8(c);
    match mode {
        ColorMode::Truecolor => Color::Rgb(r, g, b),
        ColorMode::Ansi256 => Color::Indexed(rgb_to_ansi256(r, g, b)),
    }
}

/// Map an additive [`CellStyle`] onto a ratatui [`Style`]: the attribute bitset
/// becomes a [`Modifier`] set, and the colored-underline channel becomes
/// `Style::underline_color` (ratatui emits it as the SGR `58;2;r;g;b` — a real,
/// INDEPENDENT third color, separate from fg/bg).
///
/// `mode` honours 256-color quantization for the underline color exactly like
/// fg/bg, so an Ansi256 terminal still gets a valid underline color (emitted as
/// `58;5;N`). An empty style yields an empty `Style` (no SGR change). Pure — no
/// allocation, called per cell only when a cell actually carries a style.
///
/// UNDERLINE-SUBSTYLE LIMITATION (best-effort live): the smart-material underline
/// LAYER ([`re_core::UnderlineStyle`]: curly / dotted / dashed / double) is emitted
/// FAITHFULLY only in the `--ansi` dump (we own those bytes) and shown by a capable
/// terminal. The LIVE path here goes through `bevy_ratatui`'s `CrosstermBackend`,
/// and ratatui's public `Modifier` models only a SINGLE `UNDERLINED` bit — it does
/// not expose crossterm's `Undercurled` / `Underdotted` / `Underdashed` /
/// `DoubleUnderlined` attributes, and we write through ratatui's diffing `Buffer`
/// (not raw crossterm attributes), so there is no clean per-cell injection point
/// for the substyle without bypassing the diff. We therefore FALL BACK to a single
/// underline live: any non-`None` substyle still draws a colored underline rule
/// (via [`CellStyle::effective_underline_style`] / the implied `UNDERLINE` flag),
/// just not the curly/dotted/dashed SHAPE. The colour (`58;…`) IS honoured live, so
/// the holo scan-glow and ridge glint still read as a colored under-rule — only the
/// squiggle/dash texture is dump-only. This degrades honestly and never breaks the
/// backend.
#[inline]
fn ratatui_style(style: CellStyle, mode: ColorMode) -> Style {
    let mut m = Modifier::empty();
    if style.is_bold() {
        m |= Modifier::BOLD;
    }
    if style.is_dim() {
        m |= Modifier::DIM;
    }
    if style.is_italic() {
        m |= Modifier::ITALIC;
    }
    // A colored underline OR any underline substyle implies the underline attribute
    // (so the rule + colour draw). ratatui only models the single UNDERLINED bit, so
    // every substyle degrades to a single colored underline live — see the
    // fn-level UNDERLINE-SUBSTYLE LIMITATION note. The dump emits the true shape.
    if style.effective_underline_style().is_some() {
        m |= Modifier::UNDERLINED;
    }
    if style.is_reverse() {
        m |= Modifier::REVERSED;
    }
    if style.is_strike() {
        m |= Modifier::CROSSED_OUT;
    }

    let mut out = Style::new().add_modifier(m);
    if let Some(uc) = style.underline_color() {
        // The third color channel: independent of fg/bg.
        out = out.underline_color(color(uc, mode));
    }
    out
}

/// Blit the framebuffer into the ratatui buffer in one `draw` call, wrapped in a
/// **synchronized output** update so the frame draws atomically (tear-free).
///
/// We write directly into the frame's [`Buffer`] cell-by-cell — ratatui diffs
/// it against the previous frame and emits the minimal terminal update, so this
/// is the single batched write per frame. The `BeginSynchronizedUpdate` /
/// `EndSynchronizedUpdate` pair (DEC private mode `?2026`) brackets the whole
/// flush: a terminal that supports it buffers every byte ratatui writes and
/// presents them in one shot (no mid-frame tearing); a terminal that doesn't
/// understand `?2026` silently ignores the two control sequences, so this is a
/// safe, gracefully-degrading no-op there. crossterm writes to the same
/// `stdout()` the ratatui `CrosstermBackend` flushes through, so bracketing the
/// `draw` is the correct, single wiring point.
///
/// ## Idle frames cost nothing (2026-09-02)
///
/// Before touching the terminal the system asks [`PresentCache::needs_present`]
/// whether the framebuffer differs from the last one presented (or the terminal /
/// framebuffer size changed). If not, it returns immediately: no cursor escape, no
/// `?2026` markers, no ratatui `draw`, no SGR resets, **zero bytes, zero flushes**.
/// A puzzle player stands still most of the time, so this is the common frame.
///
/// When a frame does present, per-cell colours pass through the cache's
/// **deadband** ([`PresentCache::resolve_fg`] / [`resolve_bg`](PresentCache::resolve_bg)):
/// a cell keeps the colour it last showed while the new colour is within
/// [`PresentConfig::color_deadband`] per channel, so the continuous depth shade
/// does not re-emit a near-identical `SetColors` on every cell after every step.
/// Glyph and style changes are never held. See `present_cache.rs` for the drift
/// bound and the Ansi256 rule.
///
/// The former per-frame `cursor::Hide` re-assert (player bug #4) is gone: ratatui's
/// `draw` hides the cursor itself on every draw whose frame sets no cursor position
/// (ratatui-core `render.rs`, test `draw_hides_cursor_when_frame_cursor_is_not_set`),
/// so on presented frames it was a duplicate escape + an extra flush, and on idle
/// frames there is no draw that could re-show it. We flip ratatui's internal
/// `hidden_cursor` flag once, on the first present.
fn present_sys(
    mut ctx: ResMut<RatatuiContext>,
    fb: Res<Framebuffer>,
    mode: Res<ColorMode>,
    time: Res<Time>,
    cfg: Res<PresentConfig>,
    mut cache: ResMut<PresentCache>,
    mut metrics: ResMut<FrameTimingMetrics>,
) {
    use bevy_ratatui::crossterm::terminal::{BeginSynchronizedUpdate, EndSynchronizedUpdate};
    use bevy_ratatui::crossterm::{execute, queue};
    use std::time::Instant;

    // Record the real frame cadence FIRST (before the blit timer), so FPS
    // reflects the full loop period even when the frame is a dirty-diff no-op.
    metrics.record_frame(time.delta_secs());

    let mode = *mode;
    let deadband = cfg.color_deadband;

    // Time the present: on an idle frame this measures only the cell comparison.
    let blit_start = Instant::now();

    // The terminal's own size is part of the key: ratatui's `autoresize` clears its
    // buffers on a size change, so we must draw then even if our cells match.
    let term_size = ctx.size().map(|s| (s.width, s.height)).unwrap_or((0, 0));
    if !cache.needs_present(&fb, term_size) {
        metrics.present_us = blit_start.elapsed().as_micros() as f32;
        return;
    }

    // Flip ratatui's OWN cursor-visibility state once so its `draw` keeps the
    // cursor hidden at the end of every frame (it tracks this internally).
    if !cache.cursor_hidden {
        let _ = ctx.hide_cursor();
        cache.cursor_hidden = true;
    }

    // Open the synchronized-update window. QUEUED, not executed: it rides the same
    // `stdout` ratatui flushes through, so Begin … cells … End leave in ONE write
    // instead of three. Best-effort: if the write fails we still draw.
    let _ = queue!(std::io::stdout(), BeginSynchronizedUpdate);
    let cache = &mut *cache;
    let _ = ctx.draw(|frame| {
        let area = frame.area();
        let buf = frame.buffer_mut();

        // Clamp the blit to whichever is smaller — the framebuffer or the
        // actual terminal area — so a momentarily-stale fb size can't panic.
        let cols = (fb.cols() as u16).min(area.width);
        let rows = (fb.rows() as u16).min(area.height);

        for row in 0..rows {
            for col in 0..cols {
                let (c, r) = (col as usize, row as usize);
                let Some(src) = fb.get(c, r) else {
                    continue;
                };
                cache.record_target(c, r, src);
                let cell = &mut buf[(area.x + col, area.y + row)];
                cell.set_char(glyph_char(src.glyph));
                // Both fg AND bg are emitted every cell, honoring the mode — through
                // the deadband, so an imperceptible recolour keeps last frame's bytes.
                let fg = cache.resolve_fg(c, r, src.fg, mode, deadband);
                let bg = cache.resolve_bg(c, r, src.bg, mode, deadband);
                cell.set_fg(color(fg, mode));
                cell.set_bg(color(bg, mode));
                // Additive text style (attributes + the colored-underline 3rd
                // channel). Skip the work entirely for an unstyled cell so the
                // overwhelming majority of cells cost nothing.
                if !src.style.is_empty() {
                    cell.set_style(ratatui_style(src.style, mode));
                }
            }
        }
    });
    // Close the window so the terminal presents the buffered frame atomically.
    let _ = execute!(std::io::stdout(), EndSynchronizedUpdate);

    // Record the blit duration in microseconds.
    metrics.present_us = blit_start.elapsed().as_micros() as f32;
}

// The disallowed-types lint guards against terminal crates leaking into engine
// crates. Inside re_present itself the use is intentional and correct — clippy.toml
// documents this as an expected false-positive (see the comment at the top of
// clippy.toml: "False positives in re_present itself are expected and acceptable").
#[cfg(test)]
#[allow(clippy::disallowed_types)]
mod tests {
    use super::*;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;
    use re_core::Cell as FbCell;

    // -------------------------------------------------------------------
    // suspend_terminal / resume_terminal — headless-testable surface only.
    // -------------------------------------------------------------------
    // `resume_terminal` calls `RatatuiContext::init()`, which performs real
    // terminal I/O (`EnterAlternateScreen` + `enable_raw_mode` on the actual
    // process stdout). Its success/failure — and therefore any assertion
    // about it — depends on whether the test process has a real tty attached
    // (false under `cargo test` in CI, possibly true when run interactively),
    // so exercising it here would be either a no-op (non-tty) or would
    // visibly flicker the developer's own terminal into the alt screen
    // (interactive) — neither is an honest, portable unit test. It is
    // exercised by manual testing instead: `F5` on both Windows Terminal and
    // Linux via the editor's playtest handoff.
    //
    // `suspend_terminal` has no such dependency: every step is either a
    // resource removal (safe and deterministic on any `World`, present or
    // not) or a raw stdout write ([`pop_kitty_raw`]) that succeeds whether or
    // not stdout is a tty — so it is fully testable headless.

    /// `suspend_terminal` on a `World` with NONE of the terminal resources
    /// present must not panic — it is meant to be safely idempotent (e.g. a
    /// caller invoking it twice because a prior `resume_terminal` failed).
    #[test]
    fn suspend_terminal_on_empty_world_is_a_safe_noop() {
        let mut world = World::new();
        suspend_terminal(&mut world);
        assert!(world.get_resource::<RatatuiContext>().is_none());
        assert!(world.get_resource::<KittyEnabled>().is_none());
        assert!(world.get_resource::<KittyGuard>().is_none());
    }

    /// `suspend_terminal` actually RETIRES a present `KittyGuard` (runs its
    /// `Drop`, which pops the kitty flags via the raw-write path) rather than
    /// leaving it in the `World` — the drop-order contract this pair exists
    /// to uphold depends on nothing stale surviving a suspend.
    #[test]
    fn suspend_terminal_removes_kitty_guard_if_present() {
        let mut world = World::new();
        world.insert_resource(KittyGuard { we_pushed: false });
        suspend_terminal(&mut world);
        assert!(world.get_resource::<KittyGuard>().is_none());
    }

    // -------------------------------------------------------------------
    // M2 — kitty push/pop bookkeeping balance.
    // -------------------------------------------------------------------
    // `suspend_terminal` used to follow the two resource removals with an
    // UNCONDITIONAL raw `pop_kitty_raw()` "safety net" call. On a platform
    // where `KittyGuard { we_pushed: true }` is present, `KittyGuard::Drop`
    // (fired by `world.remove_resource::<KittyGuard>()` above it) ALREADY
    // performs that exact raw pop — so the trailing call double-popped.
    // These tests assert the guard's own `Drop` is the ONLY thing that pops,
    // gated correctly on `we_pushed`, with no extra call after it.

    /// `we_pushed: true` (the "we must raw-pop ourselves" case — Windows, or
    /// any manual push where `KittyEnabled::Drop`'s standard pop won't run
    /// for real): removing the guard fires `pop_kitty_raw` EXACTLY once. Two
    /// or zero would both be regressions (leaked push / double pop).
    #[test]
    fn suspend_terminal_pops_kitty_exactly_once_when_we_pushed() {
        let mut world = World::new();
        world.insert_resource(KittyGuard { we_pushed: true });
        test_hooks::reset_pop_kitty_raw_calls();
        suspend_terminal(&mut world);
        assert_eq!(
            test_hooks::pop_kitty_raw_calls(),
            1,
            "KittyGuard::drop must fire exactly one raw pop when we_pushed=true, \
             and suspend_terminal must not add a second unconditional one (M2)"
        );
    }

    /// `we_pushed: false` (`bevy_ratatui`'s own `KittyEnabled::Drop` already
    /// performs a real pop, or nothing was ever pushed): removing the guard
    /// must NOT also fire a raw pop — that would double the real pop that
    /// `KittyEnabled::Drop` already ran (Linux case in the M2 finding).
    #[test]
    fn suspend_terminal_does_not_raw_pop_when_bevy_ratatui_owns_it() {
        let mut world = World::new();
        world.insert_resource(KittyGuard { we_pushed: false });
        test_hooks::reset_pop_kitty_raw_calls();
        suspend_terminal(&mut world);
        assert_eq!(
            test_hooks::pop_kitty_raw_calls(),
            0,
            "we_pushed=false must be a fully no-op guard drop (M2)"
        );
    }

    /// No `KittyGuard` in the `World` at all (nothing was ever pushed, or a
    /// prior `suspend_terminal` already retired it): no raw pop of any kind.
    #[test]
    fn suspend_terminal_pops_nothing_when_no_guard_present() {
        let mut world = World::new();
        test_hooks::reset_pop_kitty_raw_calls();
        suspend_terminal(&mut world);
        assert_eq!(test_hooks::pop_kitty_raw_calls(), 0);
    }

    /// [`kitty_manual_pop_required`] — the flag `resume_terminal` now uses
    /// instead of a hardcoded `we_pushed: true` — is exactly the compile-time
    /// Windows check the module doc for [`KittyGuard`] documents. This is the
    /// pure, headless-testable half of the M2 `resume_terminal` fix;
    /// `resume_terminal` itself calls `RatatuiContext::init()` (real terminal
    /// I/O) so it is exercised by the manual F5 smoke test instead (see the
    /// comment above the `suspend_terminal` / `resume_terminal` test section).
    #[test]
    fn kitty_manual_pop_required_matches_windows_cfg() {
        assert_eq!(kitty_manual_pop_required(), cfg!(windows));
    }

    #[cfg(feature = "editor")]
    #[test]
    fn mouse_event_maps_to_editor_mouse() {
        use crate::{EditorMouseButton as B, EditorMouseKind as K};
        use bevy_ratatui::crossterm::event::{
            KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
        };
        let at = |kind| MouseEvent {
            kind,
            column: 4,
            row: 9,
            modifiers: KeyModifiers::NONE,
        };

        // Down/Drag/Up carry the button; col/row pass through unchanged.
        let m = map_mouse_event(&at(MouseEventKind::Down(MouseButton::Left))).unwrap();
        assert_eq!((m.col, m.row), (4, 9));
        assert_eq!(m.kind, K::Press);
        assert_eq!(m.button, Some(B::Left));

        let m = map_mouse_event(&at(MouseEventKind::Drag(MouseButton::Right))).unwrap();
        assert_eq!(m.kind, K::Drag);
        assert_eq!(m.button, Some(B::Right));

        let m = map_mouse_event(&at(MouseEventKind::Up(MouseButton::Middle))).unwrap();
        assert_eq!(m.kind, K::Release);
        assert_eq!(m.button, Some(B::Middle));

        // Scroll has no button.
        let m = map_mouse_event(&at(MouseEventKind::ScrollUp)).unwrap();
        assert_eq!(m.kind, K::ScrollUp);
        assert_eq!(m.button, None);
        let m = map_mouse_event(&at(MouseEventKind::ScrollDown)).unwrap();
        assert_eq!(m.kind, K::ScrollDown);

        // Plain motion + horizontal scroll are dropped (no edit intent).
        assert!(map_mouse_event(&at(MouseEventKind::Moved)).is_none());
        assert!(map_mouse_event(&at(MouseEventKind::ScrollLeft)).is_none());
        assert!(map_mouse_event(&at(MouseEventKind::ScrollRight)).is_none());
    }

    /// Bytes-per-frame fixture: a real ratatui `Terminal` over a `CrosstermBackend`
    /// whose writer is a `Vec<u8>`, driven through the SAME cache + blit logic as
    /// `present_sys` (minus the stdout-only `?2026` markers, which are a fixed
    /// 8 + 8 bytes on presented frames and absent on idle ones). Measures what the
    /// terminal actually receives for an idle frame, a one-cell change, and a
    /// full-frame change.
    /// A `Write` sink the test can read back while the backend owns it
    /// (`CrosstermBackend::writer` is not public in the pinned ratatui).
    #[derive(Clone, Default)]
    struct SharedSink(std::rc::Rc<std::cell::RefCell<Vec<u8>>>);

    impl SharedSink {
        fn len(&self) -> usize {
            self.0.borrow().len()
        }
    }

    impl std::io::Write for SharedSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    type VecTerminal = ratatui::Terminal<ratatui::backend::CrosstermBackend<SharedSink>>;

    fn present_into(
        (term, sink): &mut (VecTerminal, SharedSink),
        cache: &mut PresentCache,
        fb: &Framebuffer,
        mode: ColorMode,
        deadband: u8,
    ) -> usize {
        let before = sink.len();
        // Fixed viewport ⇒ the terminal size is constant; pass the fb dims.
        if !cache.needs_present(fb, (fb.cols() as u16, fb.rows() as u16)) {
            return 0;
        }
        term.draw(|frame| {
            let area = frame.area();
            let buf = frame.buffer_mut();
            let cols = (fb.cols() as u16).min(area.width);
            let rows = (fb.rows() as u16).min(area.height);
            for row in 0..rows {
                for col in 0..cols {
                    let (c, r) = (col as usize, row as usize);
                    let Some(src) = fb.get(c, r) else { continue };
                    cache.record_target(c, r, src);
                    let cell = &mut buf[(area.x + col, area.y + row)];
                    cell.set_char(glyph_char(src.glyph));
                    let fg = cache.resolve_fg(c, r, src.fg, mode, deadband);
                    let bg = cache.resolve_bg(c, r, src.bg, mode, deadband);
                    cell.set_fg(color(fg, mode));
                    cell.set_bg(color(bg, mode));
                    if !src.style.is_empty() {
                        cell.set_style(ratatui_style(src.style, mode));
                    }
                }
            }
        })
        .expect("draw into Vec<u8> backend");
        sink.len() - before
    }

    fn gradient_fb(cols: usize, rows: usize, shift: u32) -> Framebuffer {
        let mut fb = Framebuffer::new(cols, rows);
        for r in 0..rows {
            for c in 0..cols {
                // A smooth depth-like gradient across the grid, offset by `shift`
                // units per channel — the shape of a one-cell camera step.
                let v = ((c * 200 / cols.max(1)) as u32 + shift).min(255);
                let w = ((r * 120 / rows.max(1)) as u32 + shift / 2).min(255);
                fb.set(
                    c,
                    r,
                    FbCell {
                        glyph: b"#=+-."[(c + r) % 5] as u32,
                        fg: (v << 16) | (v << 8) | v,
                        bg: (w << 16) | (w << 8) | (w / 2),
                        ..FbCell::default()
                    },
                );
            }
        }
        fb
    }

    fn vec_terminal(cols: u16, rows: u16) -> (VecTerminal, SharedSink) {
        use ratatui::{backend::CrosstermBackend, Terminal, TerminalOptions, Viewport};
        let sink = SharedSink::default();
        let term = Terminal::with_options(
            CrosstermBackend::new(sink.clone()),
            TerminalOptions {
                viewport: Viewport::Fixed(Rect::new(0, 0, cols, rows)),
            },
        )
        .expect("fixed-viewport terminal needs no tty");
        (term, sink)
    }

    #[test]
    fn present_bytes_idle_frame_is_zero_and_one_cell_is_small() {
        let (cols, rows) = (120usize, 40usize);
        let mut term = vec_terminal(cols as u16, rows as u16);
        let mut cache = PresentCache::default();
        let fb = gradient_fb(cols, rows, 0);

        let first = present_into(&mut term, &mut cache, &fb, ColorMode::Truecolor, 6);
        assert!(first > 10_000, "first frame paints every cell: {first} B");

        let idle = present_into(&mut term, &mut cache, &fb, ColorMode::Truecolor, 6);
        assert_eq!(idle, 0, "identical frame must not reach the terminal");

        let mut one = fb.clone();
        let mut cell = *one.get(17, 9).unwrap();
        cell.glyph = '@' as u32;
        one.set(17, 9, cell);
        let one_b = present_into(&mut term, &mut cache, &one, ColorMode::Truecolor, 6);
        // MoveTo + SetColors + glyph + ratatui's trailing SGR/cursor resets:
        // measured 64 B on the pinned ratatui; budget leaves headroom for a
        // wider escape encoding but must stay far below a row.
        assert!(
            one_b > 0 && one_b <= 96,
            "one changed cell must cost at most 96 bytes, got {one_b}"
        );
        eprintln!("PRESENT_BYTES first={first} idle={idle} one_cell={one_b} grid={cols}x{rows}");
    }

    #[test]
    fn present_bytes_deadband_absorbs_a_sub_threshold_recolour() {
        let (cols, rows) = (120usize, 40usize);
        for mode in [ColorMode::Truecolor, ColorMode::Ansi256] {
            let mut with = vec_terminal(cols as u16, rows as u16);
            let mut without = vec_terminal(cols as u16, rows as u16);
            let mut cache_with = PresentCache::default();
            let mut cache_without = PresentCache::default();
            let a = gradient_fb(cols, rows, 0);
            let b = gradient_fb(cols, rows, 3); // every colour moves by 3 (fg) / 1 (bg)
            present_into(&mut with, &mut cache_with, &a, mode, 6);
            present_into(&mut without, &mut cache_without, &a, mode, 0);
            let held = present_into(&mut with, &mut cache_with, &b, mode, 6);
            let raw = present_into(&mut without, &mut cache_without, &b, mode, 0);
            eprintln!("PRESENT_BYTES {mode:?} step3: deadband6={held} B, deadband0={raw} B");
            // The frame DID change (targets differ) so it is presented, but every
            // colour is held ⇒ ratatui's diff finds nothing to write.
            assert!(
                held * 10 < raw.max(1),
                "{mode:?}: deadband must remove ≥90% of the recolour bytes ({held} vs {raw})"
            );
            // A step past the deadband is emitted in full again.
            let c = gradient_fb(cols, rows, 20);
            let snapped = present_into(&mut with, &mut cache_with, &c, mode, 6);
            assert!(
                snapped > raw / 2,
                "a real recolour must still be emitted ({snapped} vs {raw})"
            );
        }
    }

    /// The pure cell-write the draw closure performs, factored out so it can be
    /// exercised against a standalone `Buffer` without a live terminal.
    fn blit_cell(buf: &mut Buffer, x: u16, y: u16, src: &FbCell, mode: ColorMode) {
        let cell = &mut buf[(x, y)];
        cell.set_char(glyph_char(src.glyph));
        cell.set_fg(color(src.fg, mode));
        cell.set_bg(color(src.bg, mode));
        if !src.style.is_empty() {
            cell.set_style(ratatui_style(src.style, mode));
        }
    }

    // --- The live crossterm key → intent table (player-bug-fix map) ---

    #[test]
    fn ad_keys_classify_as_strafe_not_turn() {
        // Player bug #1: A/D must be STRAFE (movement), never a yaw turn.
        assert_eq!(key_intent(KeyCode::Char('a')), KeyIntent::StrafeLeft);
        assert_eq!(key_intent(KeyCode::Char('A')), KeyIntent::StrafeLeft);
        assert_eq!(key_intent(KeyCode::Char('d')), KeyIntent::StrafeRight);
        assert_eq!(key_intent(KeyCode::Char('D')), KeyIntent::StrafeRight);
        // W/S are forward/back movement.
        assert_eq!(key_intent(KeyCode::Char('w')), KeyIntent::Forward);
        assert_eq!(key_intent(KeyCode::Char('s')), KeyIntent::Backward);
    }

    #[test]
    fn arrow_keys_classify_as_turn() {
        // Player bug #2: arrows TURN (yaw/pitch), and still nav menus.
        assert_eq!(key_intent(KeyCode::Left), KeyIntent::TurnLeft);
        assert_eq!(key_intent(KeyCode::Right), KeyIntent::TurnRight);
        assert_eq!(key_intent(KeyCode::Up), KeyIntent::PitchUp);
        assert_eq!(key_intent(KeyCode::Down), KeyIntent::PitchDown);
    }

    #[test]
    fn q_maps_to_back() {
        // The on-screen "q back" / "q quit" hints must actually work: `q`/`Q` AND
        // `Esc` all classify to `Back`, and Back drives `MenuNav::Back`. This is
        // safe because Back is no longer a surprise hard-quit during gameplay (it
        // opens the Pause overlay); `re_game` decides what Back means per state.
        assert_eq!(key_intent(KeyCode::Char('q')), KeyIntent::Back);
        assert_eq!(key_intent(KeyCode::Char('Q')), KeyIntent::Back);
        assert_eq!(
            key_intent(KeyCode::Char('q')).menu_nav(),
            Some(MenuNav::Back)
        );
        assert_eq!(
            key_intent(KeyCode::Char('Q')).menu_nav(),
            Some(MenuNav::Back)
        );
        // Esc still backs out too.
        assert_eq!(key_intent(KeyCode::Esc), KeyIntent::Back);
        assert_eq!(key_intent(KeyCode::Esc).menu_nav(), Some(MenuNav::Back));
    }

    #[test]
    fn space_and_interact_and_fov_classify() {
        assert_eq!(key_intent(KeyCode::Char(' ')), KeyIntent::Jump);
        assert_eq!(key_intent(KeyCode::Char('e')), KeyIntent::Interact);
        assert_eq!(key_intent(KeyCode::Enter), KeyIntent::Interact);
        assert_eq!(key_intent(KeyCode::Char('[')), KeyIntent::Fov(-1));
        assert_eq!(key_intent(KeyCode::Char(']')), KeyIntent::Fov(1));
    }

    #[test]
    fn zoom_and_postfx_keys_classify_and_emit_no_nav() {
        // `Z` (either case) → FOV-ZOOM toggle; `G` → POST-FX cycle. Both are
        // one-shot gameplay edges that never navigate menus and never latch as
        // held movement keys.
        for code in [KeyCode::Char('z'), KeyCode::Char('Z')] {
            assert_eq!(key_intent(code), KeyIntent::ZoomToggle);
            assert_eq!(key_intent(code).menu_nav(), None);
            let mut h = crate::HeldKeys::default();
            assert!(
                !h.set(key_intent(code), true),
                "ZoomToggle must not latch in HeldKeys"
            );
            assert_eq!(h, crate::HeldKeys::default());
        }
        for code in [KeyCode::Char('g'), KeyCode::Char('G')] {
            assert_eq!(key_intent(code), KeyIntent::PostFxCycle);
            assert_eq!(key_intent(code).menu_nav(), None);
            let mut h = crate::HeldKeys::default();
            assert!(
                !h.set(key_intent(code), true),
                "PostFxCycle must not latch in HeldKeys"
            );
            assert_eq!(h, crate::HeldKeys::default());
        }
    }

    #[test]
    fn f3_maps_to_toggle_perf_and_emits_no_nav() {
        // F3 must map to TogglePerf and NOT produce any menu-navigation intent
        // (it is a toggle, not a direction or confirm/back).
        assert_eq!(key_intent(KeyCode::F(3)), KeyIntent::TogglePerf);
        assert_eq!(key_intent(KeyCode::F(3)).menu_nav(), None);
        // F3 must not be in the held-set (it is a one-shot, not a continuous key).
        let mut h = crate::HeldKeys::default();
        assert!(!h.set(KeyIntent::TogglePerf, true));
        assert_eq!(
            h,
            crate::HeldKeys::default(),
            "TogglePerf must not latch in HeldKeys"
        );
    }

    #[test]
    fn strafe_moves_perpendicular_to_facing_and_keeps_yaw() {
        use glam::Vec3;
        use re_world::{stamp_geometry, CameraPose, CellKind, GeometrySpec};

        // Open room so the strafe isn't blocked by a wall.
        let map = stamp_geometry(&GeometrySpec {
            size: (9, 9, 4),
            wall_kind: CellKind::WallSolid,
            decals: &[],
            objects: &[],
        });
        // Face +y (yaw=0 → dir=(0,1)); strafe basis (dy,-dx) = (1,0) → +x.
        let mut cam = CameraPose {
            pos: Vec3::new(4.5, 4.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let yaw_before = cam.yaw;
        let (x0, y0) = (cam.pos.x, cam.pos.y);

        // Reproduce the exact strafe-right wiring from `input_sys`.
        let (dx, dy) = cam.dir();
        let (sx, sy) = (dy, -dx);
        cam.move_with_collision(sx * MOVE_STEP, sy * MOVE_STEP, &map, PLAYER_RADIUS);

        // Yaw is UNCHANGED (this is movement, not a turn — player bug #1).
        assert_eq!(cam.yaw, yaw_before, "strafe must not turn the camera");
        // Moved along +x (perpendicular to the +y facing), not along y.
        assert!(
            cam.pos.x > x0 + 1e-4,
            "strafe-right should advance +x, x={}",
            cam.pos.x
        );
        assert!(
            (cam.pos.y - y0).abs() < 1e-4,
            "strafe must not move forward, y={}",
            cam.pos.y
        );

        // Strafe LEFT goes the other way (−x) from a fresh start.
        let mut cam2 = CameraPose {
            pos: Vec3::new(4.5, 4.5, 1.5),
            yaw: 0.0,
            pitch: 0.0,
        };
        let (dx2, dy2) = cam2.dir();
        let (sx2, sy2) = (dy2, -dx2);
        cam2.move_with_collision(-sx2 * MOVE_STEP, -sy2 * MOVE_STEP, &map, PLAYER_RADIUS);
        assert!(
            cam2.pos.x < 4.5 - 1e-4,
            "strafe-left should advance -x, x={}",
            cam2.pos.x
        );
    }

    #[test]
    fn arrow_turn_changes_yaw_and_clamps_pitch() {
        use glam::Vec3;
        use re_world::CameraPose;
        // ←/→ change yaw by ±TURN_STEP (the exact `input_sys` wiring).
        let mut cam = CameraPose {
            pos: Vec3::ZERO,
            yaw: 1.0,
            pitch: 0.0,
        };
        cam.yaw -= TURN_STEP; // TurnLeft
        assert!((cam.yaw - (1.0 - TURN_STEP)).abs() < 1e-6);
        cam.yaw += 2.0 * TURN_STEP; // TurnRight twice
        assert!((cam.yaw - (1.0 + TURN_STEP)).abs() < 1e-6);
        // ↑/↓ pitch is clamped to ±PITCH_LIMIT.
        cam.pitch = (PITCH_LIMIT + 5.0).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        assert!((cam.pitch - PITCH_LIMIT).abs() < 1e-6, "pitch clamps up");
    }

    #[test]
    fn blit_sets_both_fg_and_bg_truecolor() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 1, 1));
        let src = FbCell {
            glyph: u32::from('#'),
            fg: 0x00FF_8040,
            bg: 0x0010_2030,
            ..Default::default()
        };
        blit_cell(&mut buf, 0, 0, &src, ColorMode::Truecolor);

        let cell = &buf[(0, 0)];
        assert_eq!(cell.symbol(), "#");
        assert_eq!(cell.fg, Color::Rgb(0xFF, 0x80, 0x40));
        // bg must NOT be the default — it carries the cell's real background.
        assert_eq!(cell.bg, Color::Rgb(0x10, 0x20, 0x30));
        assert_ne!(cell.bg, Color::Reset);
    }

    #[test]
    fn blit_quantizes_both_fg_and_bg_in_ansi256() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 1, 1));
        let src = FbCell {
            glyph: u32::from('@'),
            fg: 0x00FF_0000, // pure red → 196
            bg: 0x0000_FF00, // pure green → 46
            ..Default::default()
        };
        blit_cell(&mut buf, 0, 0, &src, ColorMode::Ansi256);

        let cell = &buf[(0, 0)];
        assert_eq!(cell.fg, Color::Indexed(196));
        assert_eq!(cell.bg, Color::Indexed(46));
    }

    #[test]
    fn styled_cell_sets_modifiers_and_underline_color() {
        // A cell with bold + an independent colored underline must blit to a
        // ratatui cell whose Modifier carries BOLD + UNDERLINED and whose
        // underline_color is the cell's third color channel (NOT fg/bg).
        let mut buf = Buffer::empty(Rect::new(0, 0, 1, 1));
        let src = FbCell {
            glyph: u32::from('@'),
            fg: 0x00FF_FFFF,
            bg: 0x0000_0000,
            style: CellStyle::default()
                .with_bold()
                .with_underline_color(0x0011_2233),
            ..Default::default()
        };
        blit_cell(&mut buf, 0, 0, &src, ColorMode::Truecolor);

        let cell = &buf[(0, 0)];
        assert!(cell.modifier.contains(Modifier::BOLD), "BOLD must be set");
        assert!(
            cell.modifier.contains(Modifier::UNDERLINED),
            "a colored underline implies UNDERLINED"
        );
        // The colored underline is a real, independent third channel.
        assert_eq!(cell.underline_color, Color::Rgb(0x11, 0x22, 0x33));
        assert_ne!(cell.underline_color, cell.fg);
        assert_ne!(cell.underline_color, cell.bg);
    }

    #[test]
    fn underline_substyle_degrades_to_single_underline_live() {
        use re_core::UnderlineStyle;
        // A curly scan-glow accent: live ratatui can't draw the curl, but it MUST
        // still draw a single underline + the colour (best-effort fallback).
        let style = CellStyle::default()
            .with_underline_style(UnderlineStyle::Curly)
            .with_underline_color(0x0050_E8FF);
        let st = ratatui_style(style, ColorMode::Truecolor);
        assert!(
            st.add_modifier.contains(Modifier::UNDERLINED),
            "a substyle still lights the single UNDERLINED bit live"
        );
        // The colour is honoured even though the shape is not.
        assert_eq!(st.underline_color, Some(Color::Rgb(0x50, 0xE8, 0xFF)));
    }

    #[test]
    fn all_six_style_flags_map_to_modifiers() {
        let style = CellStyle::default().with_flags(
            CellStyle::BOLD
                | CellStyle::DIM
                | CellStyle::ITALIC
                | CellStyle::UNDERLINE
                | CellStyle::REVERSE
                | CellStyle::STRIKE,
        );
        let m = ratatui_style(style, ColorMode::Truecolor).add_modifier;
        assert!(m.contains(Modifier::BOLD));
        assert!(m.contains(Modifier::DIM));
        assert!(m.contains(Modifier::ITALIC));
        assert!(m.contains(Modifier::UNDERLINED));
        assert!(m.contains(Modifier::REVERSED));
        assert!(m.contains(Modifier::CROSSED_OUT));
    }

    #[test]
    fn unstyled_cell_leaves_modifier_empty() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 1, 1));
        let src = FbCell {
            glyph: u32::from('#'),
            fg: 0x00FF_8040,
            bg: 0x0010_2030,
            ..Default::default()
        };
        blit_cell(&mut buf, 0, 0, &src, ColorMode::Truecolor);
        let cell = &buf[(0, 0)];
        assert_eq!(cell.modifier, Modifier::empty());
        // No colored underline → ratatui's default (Reset).
        assert_eq!(cell.underline_color, Color::Reset);
    }

    #[test]
    fn underline_color_quantizes_in_ansi256() {
        // The 3rd channel honours the color mode like fg/bg.
        let style = CellStyle::default().with_underline_color(0x0000_FF00); // green → 46
        let st = ratatui_style(style, ColorMode::Ansi256);
        assert_eq!(st.underline_color, Some(Color::Indexed(46)));
    }

    #[test]
    fn color_helper_respects_mode() {
        assert_eq!(
            color(0x00FF_FFFF, ColorMode::Truecolor),
            Color::Rgb(255, 255, 255)
        );
        assert_eq!(color(0x00FF_FFFF, ColorMode::Ansi256), Color::Indexed(231));
    }

    // -----------------------------------------------------------------------
    // C2a gate regression: mouse_look_sys must NOT move cam.yaw/cam.pitch when
    // InputGate.camera_motion is false, but MUST still keep tracker.last
    // current (so re-entering Playing doesn't apply a stale accumulated delta).
    // -----------------------------------------------------------------------

    /// Simulate the exact logic of `mouse_look_sys` without any ECS machinery:
    /// a pure extraction that mirrors the gate logic the live system implements.
    ///
    /// Arguments:
    /// * `events`   — sequence of (column, row) absolute mouse positions.
    /// * `tracker`  — mutable last-seen position, updated on every event.
    /// * `cam_yaw`  / `cam_pitch` — the camera pose to (maybe) mutate.
    /// * `enabled`  — `MouseSettings::enabled`.
    /// * `gate`     — `InputGate::camera_motion`.
    ///
    /// Returns the new `(yaw, pitch)` after all events.
    fn simulate_mouse_look(
        events: &[(u16, u16)],
        tracker: &mut Option<(u16, u16)>,
        cam_yaw: f32,
        cam_pitch: f32,
        enabled: bool,
        gate: bool,
    ) -> (f32, f32) {
        let apply = gate && enabled;
        let sensitivity = 1.0_f32;
        let mut yaw = cam_yaw;
        let mut pitch = cam_pitch;

        for &(col, row) in events {
            // Mirror the Moved/Drag arm: only motion events enter.
            let pos = (col, row);
            if apply {
                if let Some((px, py)) = *tracker {
                    let dx = pos.0 as f32 - px as f32;
                    let dy = pos.1 as f32 - py as f32;
                    if dx != 0.0 || dy != 0.0 {
                        let (y, p) = apply_mouse_look(yaw, pitch, dx, dy, sensitivity);
                        yaw = y;
                        pitch = p;
                    }
                }
            }
            // Always update the tracker (gate-independent).
            *tracker = Some(pos);
        }
        (yaw, pitch)
    }

    #[test]
    fn mouse_look_gate_closed_does_not_move_camera() {
        use glam::Vec3;
        use re_core::InputGate;
        use re_world::CameraPose;

        // ---- Setup ----
        let initial = CameraPose {
            pos: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
        };
        let mut tracker: Option<(u16, u16)> = Some((10, 10)); // pre-seeded so delta is non-zero
        let events: &[(u16, u16)] = &[(15, 12), (20, 14)]; // two motion events
        let gate = InputGate {
            camera_motion: false,
        };

        // ---- Gate closed: tracker stays current but camera MUST NOT move. ----
        let (yaw_after, pitch_after) = simulate_mouse_look(
            events,
            &mut tracker,
            initial.yaw,
            initial.pitch,
            true, // settings.enabled = true
            gate.camera_motion,
        );
        assert_eq!(
            (yaw_after, pitch_after),
            (initial.yaw, initial.pitch),
            "gate closed: cam.yaw/cam.pitch must be UNCHANGED, got yaw={yaw_after} pitch={pitch_after}"
        );
        // The tracker MUST still have been updated (anti-jump invariant).
        assert_eq!(
            tracker,
            Some(*events.last().unwrap()),
            "tracker.last must be updated even when the gate is closed"
        );

        // ---- Gate open: the same events MUST move the camera. ----
        let gate_open = InputGate {
            camera_motion: true,
        };
        // Feed the same events starting from the last tracker position so the
        // delta is well-defined (non-zero).
        let mut tracker2: Option<(u16, u16)> = Some((10, 10));
        let (yaw_open, pitch_open) = simulate_mouse_look(
            events,
            &mut tracker2,
            initial.yaw,
            initial.pitch,
            true,
            gate_open.camera_motion,
        );
        assert!(
            yaw_open != initial.yaw || pitch_open != initial.pitch,
            "gate open: at least one of yaw/pitch must have changed (yaw={yaw_open} pitch={pitch_open})"
        );
        // In particular a rightward motion (col increasing) must turn yaw right.
        assert!(
            yaw_open > initial.yaw,
            "rightward mouse motion must increase yaw, got {yaw_open}"
        );
    }

    #[test]
    fn mouse_look_gate_open_then_closed_no_accumulated_jump() {
        // If the gate closes mid-stream, tracker.last is kept current, so
        // re-opening the gate the very next frame applies ONLY the most recent
        // inter-frame delta — not an accumulated stale one.

        // Simulate: gate open for 2 events, then gate closes for 2 more events
        // (tracker still advances), then gate re-opens for 1 event.

        let start_pos: (u16, u16) = (20, 20);
        let mut tracker: Option<(u16, u16)> = Some(start_pos);
        let mut yaw = 0.0_f32;
        let mut pitch = 0.0_f32;

        // Phase 1 — gate open: move from (20,20) to (25,20) [+5 cols].
        let events_open = [(25u16, 20u16)];
        (yaw, pitch) = simulate_mouse_look(&events_open, &mut tracker, yaw, pitch, true, true);
        let yaw_after_phase1 = yaw;
        assert!(
            yaw > 0.0,
            "rightward motion while gate open must advance yaw"
        );

        // Phase 2 — gate closed: 3 large moves (50,20 → 80,20 → 110,20).
        // Camera must not change, but tracker.last must end at (110,20).
        let events_closed = [(50u16, 20u16), (80u16, 20u16), (110u16, 20u16)];
        (yaw, pitch) = simulate_mouse_look(&events_closed, &mut tracker, yaw, pitch, true, false);
        assert_eq!(
            (yaw, pitch),
            (yaw_after_phase1, 0.0),
            "gate closed: camera must not have moved during the 3 large events"
        );
        assert_eq!(
            tracker,
            Some((110, 20)),
            "tracker must be current at (110,20)"
        );

        // Phase 3 — gate re-opens: one small move (112,20) [+2 cols from 110].
        let events_reopen = [(112u16, 20u16)];
        let (yaw_final, _) =
            simulate_mouse_look(&events_reopen, &mut tracker, yaw, pitch, true, true);
        // The jump from 20 to 110 must NOT have been applied; only the +2
        // inter-frame delta fires.
        let expected_delta = 2.0_f32 * crate::MOUSE_YAW_SENS; // +2 cols at sens=1
        assert!(
            (yaw_final - yaw_after_phase1 - expected_delta).abs() < 1e-5,
            "re-opening the gate must apply only the +2 col delta, not the stale 90-col accumulation; \
             got yaw_final={yaw_final} yaw_after_phase1={yaw_after_phase1} expected_delta={expected_delta}"
        );
    }

    // -----------------------------------------------------------------------
    // Sprint key classification in the crossterm key table (Task 1).
    // -----------------------------------------------------------------------

    /// LeftShift and RightShift must classify to `KeyIntent::Sprint` in the live
    /// crossterm key table. These codes only arrive on the kitty path (dedicated
    /// `KeyCode::Modifier(ModifierKeyCode::LeftShift / RightShift)` events),
    /// but the lookup itself is always testable without a live terminal.
    #[test]
    fn shift_keys_classify_as_sprint() {
        assert_eq!(
            key_intent(KeyCode::Modifier(ModifierKeyCode::LeftShift)),
            KeyIntent::Sprint,
            "LeftShift must map to Sprint"
        );
        assert_eq!(
            key_intent(KeyCode::Modifier(ModifierKeyCode::RightShift)),
            KeyIntent::Sprint,
            "RightShift must map to Sprint"
        );
        // Sprint emits NO menu navigation.
        assert_eq!(
            key_intent(KeyCode::Modifier(ModifierKeyCode::LeftShift)).menu_nav(),
            None,
            "Sprint must not navigate menus"
        );
        // Sprint IS a continuous (held-tracked) intent.
        let mut h = crate::HeldKeys::default();
        assert!(
            h.set(KeyIntent::Sprint, true),
            "Sprint must be continuous (set() should return true)"
        );
        assert!(h.sprint, "sprint flag must be latched");
    }

    /// Verify that the mouse sensitivity constants used in the live terminal path
    /// are the halved values (0.05 / 0.04). The `simulate_mouse_look` helper above
    /// relies on `crate::MOUSE_YAW_SENS`; this test verifies the constant itself
    /// from the terminal module's perspective.
    #[test]
    fn terminal_mouse_sensitivity_constants_are_halved() {
        assert!(
            (crate::MOUSE_YAW_SENS - 0.05).abs() < 1e-6,
            "MOUSE_YAW_SENS must be 0.05, got {}",
            crate::MOUSE_YAW_SENS
        );
        assert!(
            (crate::MOUSE_PITCH_SENS - 0.04).abs() < 1e-6,
            "MOUSE_PITCH_SENS must be 0.04, got {}",
            crate::MOUSE_PITCH_SENS
        );
    }

    // -----------------------------------------------------------------------
    // M key — mouse-look toggle binding.
    // -----------------------------------------------------------------------

    /// `M` (both cases) must classify to `KeyIntent::MouseLookToggle`.
    /// It must emit NO menu-nav and must NOT enter the held set (it is a one-shot).
    #[test]
    fn m_key_classifies_as_mouse_look_toggle() {
        for code in [KeyCode::Char('m'), KeyCode::Char('M')] {
            assert_eq!(
                key_intent(code),
                KeyIntent::MouseLookToggle,
                "{code:?} must map to MouseLookToggle"
            );
            // A toggle is never a menu-navigation intent.
            assert_eq!(
                key_intent(code).menu_nav(),
                None,
                "MouseLookToggle must not emit any MenuNav"
            );
            // It must NOT latch in HeldKeys (it is a one-shot).
            let mut h = crate::HeldKeys::default();
            assert!(
                !h.set(key_intent(code), true),
                "MouseLookToggle must not latch in HeldKeys"
            );
            assert_eq!(
                h,
                crate::HeldKeys::default(),
                "HeldKeys unchanged after MouseLookToggle"
            );
        }
    }

    /// Mouse-look edge-graceful: when consecutive events carry the SAME position
    /// (pointer pinned at the terminal edge), the delta is (0, 0) and the camera
    /// must not move. No NaN, no spin, no jitter.
    #[test]
    fn mouse_look_edge_pin_produces_zero_delta() {
        let edge_pos = (0u16, 0u16); // pointer pinned at top-left corner
        let events: &[(u16, u16)] = &[edge_pos, edge_pos, edge_pos, edge_pos];
        let mut tracker: Option<(u16, u16)> = Some(edge_pos);
        let (yaw_after, pitch_after) = simulate_mouse_look(
            events,
            &mut tracker,
            0.5, // initial yaw
            0.1, // initial pitch
            true,
            true,
        );
        // Camera must be UNCHANGED — a zero delta produces no rotation.
        assert!(
            (yaw_after - 0.5).abs() < 1e-6,
            "pinned-at-edge events must not change yaw (got {yaw_after})"
        );
        assert!(
            (pitch_after - 0.1).abs() < 1e-6,
            "pinned-at-edge events must not change pitch (got {pitch_after})"
        );
        // No NaN.
        assert!(!yaw_after.is_nan(), "yaw must never be NaN after edge-pin");
        assert!(
            !pitch_after.is_nan(),
            "pitch must never be NaN after edge-pin"
        );
    }

    /// After the pointer leaves the edge (delta non-zero), mouse-look resumes
    /// immediately — no hysteresis, no dead-band.
    #[test]
    fn mouse_look_resumes_after_leaving_edge() {
        // Pointer pinned, then moves right.
        let events: &[(u16, u16)] = &[(5, 5), (5, 5), (5, 5), (8, 5)]; // last = +3 cols
        let mut tracker: Option<(u16, u16)> = Some((5, 5));
        let (yaw_after, _) = simulate_mouse_look(events, &mut tracker, 0.0, 0.0, true, true);
        // The +3-column move must register as a positive yaw change.
        assert!(
            yaw_after > 0.0,
            "after leaving the edge, a rightward move must increase yaw (got {yaw_after})"
        );
    }

    /// TURN_SPEED is the primary keyboard-look constant; verify it is at least
    /// the raised 3.0 rad/s (fast enough to be a reliable first-class look method).
    #[test]
    fn turn_speed_is_responsive() {
        const {
            assert!(
                crate::TURN_SPEED >= 3.0,
                "TURN_SPEED must be at least 3.0 rad/s"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Editor key mapping — pure map_editor_key unit tests.
    // These exercise the crossterm KeyCode → EditorKey translation without a
    // live terminal or a running Bevy app.
    // -----------------------------------------------------------------------

    #[cfg(feature = "editor")]
    mod editor_key_tests {
        use super::super::{map_editor_key, KeyCode};
        use crate::EditorKeyCode as E;
        use bevy_ratatui::crossterm::event::KeyModifiers;

        fn no_mod() -> KeyModifiers {
            KeyModifiers::NONE
        }
        fn shift() -> KeyModifiers {
            KeyModifiers::SHIFT
        }

        // --- Backspace ---

        /// Backspace maps to `EditorKeyCode::Backspace` with shift=false.
        #[test]
        fn backspace_maps_through() {
            let ek = map_editor_key(KeyCode::Backspace, no_mod()).unwrap();
            assert_eq!(ek.code, E::Backspace);
            assert!(!ek.shift);
        }

        /// Backspace with Shift modifier still maps to `Backspace` (shift=true).
        #[test]
        fn backspace_with_shift_maps_through() {
            let ek = map_editor_key(KeyCode::Backspace, shift()).unwrap();
            assert_eq!(ek.code, E::Backspace);
            assert!(ek.shift);
        }

        // --- Letter normalisation (CapsLock-independence) ---

        /// Plain `s` with no modifier → `Char('s')`, shift=false.
        #[test]
        fn plain_s_no_modifier_maps_lowercase() {
            let ek = map_editor_key(KeyCode::Char('s'), no_mod()).unwrap();
            assert_eq!(ek.code, E::Char('s'));
            assert!(!ek.shift);
        }

        /// Shift+S — most terminals deliver `'S'` with SHIFT modifier set.
        /// Must emit `Char('S')` + shift=true so editor `Char('S')` arms fire.
        #[test]
        fn shift_plus_s_uppercase_char_with_shift_mod() {
            let ek = map_editor_key(KeyCode::Char('S'), shift()).unwrap();
            assert_eq!(ek.code, E::Char('S'), "SHIFT+letter must be uppercase");
            assert!(ek.shift, "shift flag must be true");
        }

        /// Shift+S — some terminals (extended key mode) deliver `'s'` (lowercase)
        /// with SHIFT modifier.  We must still emit `Char('S')` + shift=true so
        /// the editor's `Char('S')` arms fire correctly.
        #[test]
        fn shift_plus_s_lowercase_char_with_shift_mod_normalises_to_uppercase() {
            let ek = map_editor_key(KeyCode::Char('s'), shift()).unwrap();
            assert_eq!(
                ek.code,
                E::Char('S'),
                "lowercase char + SHIFT modifier must be normalised to uppercase"
            );
            assert!(ek.shift, "shift flag must be true");
        }

        /// CapsLock-on, no Shift held: terminal delivers `'S'` with NO SHIFT
        /// modifier.  We normalise to `Char('s')` + shift=false — the editor sees
        /// it as plain `s`, not as a Shift command.  CapsLock is invisible.
        #[test]
        fn capslock_uppercase_no_shift_mod_normalises_to_lowercase() {
            let ek = map_editor_key(KeyCode::Char('S'), no_mod()).unwrap();
            assert_eq!(
                ek.code,
                E::Char('s'),
                "CapsLock uppercase without SHIFT modifier must normalise to lowercase"
            );
            assert!(
                !ek.shift,
                "shift must be false when only CapsLock (not Shift) is on"
            );
        }

        /// CapsLock-on AND Shift held: some terminals deliver `'s'` (lowercase)
        /// with SHIFT modifier (Shift cancels CapsLock).  Must emit `Char('S')` +
        /// shift=true.
        #[test]
        fn capslock_on_shift_held_lowercase_char_normalises_to_uppercase() {
            let ek = map_editor_key(KeyCode::Char('s'), shift()).unwrap();
            assert_eq!(ek.code, E::Char('S'), "SHIFT modifier → uppercase");
            assert!(ek.shift, "shift modifier present → shift=true");
        }

        // --- Non-letter characters pass through unchanged ---

        /// Non-letter chars pass through unchanged. Digits and punctuation are
        /// forwarded exactly — only ASCII letters are normalised.
        #[test]
        fn non_letter_chars_pass_through_unchanged() {
            // Digit
            let ek = map_editor_key(KeyCode::Char('3'), no_mod()).unwrap();
            assert_eq!(ek.code, E::Char('3'));
            // Punctuation with shift (e.g. Shift+1 → '!' on US keyboard)
            let ek2 = map_editor_key(KeyCode::Char('!'), shift()).unwrap();
            assert_eq!(ek2.code, E::Char('!'));
            assert!(ek2.shift);
            // Space
            let ek3 = map_editor_key(KeyCode::Char(' '), no_mod()).unwrap();
            assert_eq!(ek3.code, E::Char(' '));
        }

        // --- Navigation keys ---

        /// All navigation keys map to their EditorKeyCode variants.
        #[test]
        fn navigation_keys_map() {
            assert_eq!(map_editor_key(KeyCode::Up, no_mod()).unwrap().code, E::Up);
            assert_eq!(
                map_editor_key(KeyCode::Down, no_mod()).unwrap().code,
                E::Down
            );
            assert_eq!(
                map_editor_key(KeyCode::Left, no_mod()).unwrap().code,
                E::Left
            );
            assert_eq!(
                map_editor_key(KeyCode::Right, no_mod()).unwrap().code,
                E::Right
            );
            assert_eq!(
                map_editor_key(KeyCode::PageUp, no_mod()).unwrap().code,
                E::PageUp
            );
            assert_eq!(
                map_editor_key(KeyCode::PageDown, no_mod()).unwrap().code,
                E::PageDown
            );
            assert_eq!(
                map_editor_key(KeyCode::Enter, no_mod()).unwrap().code,
                E::Enter
            );
            assert_eq!(map_editor_key(KeyCode::Esc, no_mod()).unwrap().code, E::Esc);
            assert_eq!(map_editor_key(KeyCode::Tab, no_mod()).unwrap().code, E::Tab);
        }

        /// `Delete`/`Home`/`End` map to their `EditorKeyCode` variants (M5:
        /// these were previously unmapped, leaving the shipped
        /// `ts_editor::widget::{delete,home,end}` caret ops dead — no crossterm
        /// key could ever reach them).
        #[test]
        fn delete_home_end_map() {
            assert_eq!(
                map_editor_key(KeyCode::Delete, no_mod()).unwrap().code,
                E::Delete
            );
            assert_eq!(
                map_editor_key(KeyCode::Home, no_mod()).unwrap().code,
                E::Home
            );
            assert_eq!(map_editor_key(KeyCode::End, no_mod()).unwrap().code, E::End);
        }

        // --- Function keys ---

        /// Function keys map through to `EditorKeyCode::F(n)` verbatim — they
        /// unblock the editor's F2-rename and F5-playtest bindings (§5 row A,
        /// 2026-07-30 editor-overhaul plan). `shift=false` when no modifier is
        /// held.
        #[test]
        fn f_keys_map_through() {
            let f2 = map_editor_key(KeyCode::F(2), no_mod()).unwrap();
            assert_eq!(f2.code, E::F(2));
            assert!(!f2.shift);

            let f5 = map_editor_key(KeyCode::F(5), no_mod()).unwrap();
            assert_eq!(f5.code, E::F(5));
            assert!(!f5.shift);

            // High function-key numbers (e.g. F12) map through unchanged too.
            let f12 = map_editor_key(KeyCode::F(12), no_mod()).unwrap();
            assert_eq!(f12.code, E::F(12));
        }

        /// A Shift-held function key still maps to `F(n)`; `shift=true` lets
        /// the editor distinguish Shift+F2 / Shift+F5 chords from the bare key.
        #[test]
        fn f_keys_with_shift_set_shift_flag() {
            let f2 = map_editor_key(KeyCode::F(2), shift()).unwrap();
            assert_eq!(f2.code, E::F(2));
            assert!(f2.shift);

            let f5 = map_editor_key(KeyCode::F(5), shift()).unwrap();
            assert_eq!(f5.code, E::F(5));
            assert!(f5.shift);
        }

        // --- Unknown keys ---

        /// Unknown keys (e.g. Insert) return None — never crash, never a
        /// default action.
        #[test]
        fn unknown_keys_return_none() {
            assert!(map_editor_key(KeyCode::Insert, no_mod()).is_none());
        }
    }
}
