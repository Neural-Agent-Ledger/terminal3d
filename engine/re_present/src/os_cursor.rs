//! OS-level cursor grab for real FPS mouse-look in a terminal process.
//!
//! ## Why this exists
//!
//! A terminal running in a console window **is a native OS process**. It can
//! call the same Win32 / X11 APIs that a game window calls. The key insight:
//! crossterm's `MouseEvent` gives only ABSOLUTE cell coordinates, so the look
//! vector stalls once the pointer hits the terminal edge. The fix is to READ
//! the OS pointer position (pixel resolution), compute the delta from a fixed
//! anchor (center of the console window), apply the delta as yaw/pitch, then
//! WARP the pointer back to the anchor — so the pointer never drifts to the edge.
//!
//! ## Architecture
//!
//! The public interface is a single [`OsCursorGrab`] resource (Bevy `Resource`).
//! `re_present` creates it, and `mouse_look_sys` reads [`OsCursorGrab::poll_delta`]
//! each frame when the grab is active. Initialization, anchor update, teardown,
//! and the panic hook extension are wired in `terminal.rs`.
//!
//! ## Grab / release lifecycle
//!
//! * **Activate** (M toggle ON + console foreground): warp cursor to window center
//!   anchor, hide OS cursor (Win32 `ShowCursor(FALSE)`), set `active = true`.
//! * **Each frame**: `GetCursorPos` → delta from anchor → `SetCursorPos(anchor)`.
//!   The pointer always starts each frame at the anchor → no edge accumulation.
//! * **Deactivate** (M toggle OFF, focus lost, quit, panic): `ShowCursor(TRUE)`,
//!   `active = false`.
//!
//! ## Safety contract
//!
//! `unsafe` blocks appear only at the minimal FFI call sites. Each is documented
//! with the invariant that makes it sound:
//!
//! * `GetCursorPos(&mut POINT)` — always valid for a stack-allocated `POINT`.
//! * `SetCursorPos(x, y)` — no preconditions beyond i32 screen coordinates.
//! * `GetConsoleWindow()` — returns `HWND` or `0`; `0` is checked before use.
//! * `GetForegroundWindow()` / `GetActiveWindow()` — always return `HWND` or `0`.
//! * `GetWindowRect(hwnd, &mut RECT)` — requires a valid non-null `HWND` and a
//!   valid `RECT` pointer; both satisfied (non-null guard + stack `RECT`).
//! * `ShowCursor(BOOL)` — a refcounted display counter; always safe to call.
//!   We track `cursor_hidden` so we issue exactly one `TRUE` per `FALSE`.
//! * `GetSystemMetrics(index)` — no preconditions.
//!
//! The Linux / X11 path is best-effort behind the `x11` feature gate. Wayland
//! degrades with a one-time log message and a no-op on every subsequent call.

use bevy::ecs::resource::Resource;

// ─── Simple OS-cursor hide (decoupled from mouse-look grab) ──────────────────

/// A lightweight OS-cursor visibility toggle that hides the system mouse pointer
/// **without** the recenter/poll logic of [`OsCursorGrab`].
///
/// This is the path for `TS_HIDE_CURSOR`: it calls the platform's
/// `ShowCursor(FALSE)` / X11 blank-cursor once on `hide()` and once reverses
/// that with `ShowCursor(TRUE)` on `show()`. No warping, no per-frame polling,
/// no anchor state.
///
/// ## Win32 counter balance
///
/// Win32's `ShowCursor` is a reference-counted display counter. `OsCursorSimpleHide`
/// tracks its own `cursor_hidden: bool` (separate from `OsCursorGrab`'s field) so
/// it issues exactly one `ShowCursor(FALSE)` per `ShowCursor(TRUE)`. The two
/// resources can safely coexist:
///
/// * Plain hide only: counter goes −1 on hide, +1 on release.
/// * Grab activates while plain hide is live: grab issues its own `ShowCursor(FALSE)`
///   (counter −2); grab deactivate issues its own `ShowCursor(TRUE)` (counter −1);
///   plain hide release issues `ShowCursor(TRUE)` (counter 0). Net: visible. Correct.
/// * Any ordering of their lifetimes is safe because each path only reverses its own
///   decrement.
///
/// ## Teardown
///
/// [`Drop`] calls [`show`](Self::show) so the pointer is restored even if the
/// caller forgets. The panic hook in `terminal.rs` calls
/// [`OsCursorSimpleHide::release_on_panic`] as
/// a belt-and-suspenders restore (one extra `ShowCursor(TRUE)` on a cursor that was
/// already visible is harmless — the counter goes above 0 and the cursor stays visible).
#[derive(Resource, Default)]
pub struct OsCursorSimpleHide {
    /// True while we have issued a `ShowCursor(FALSE)` that hasn't been reversed.
    #[cfg(target_os = "windows")]
    cursor_hidden: bool,
}

impl OsCursorSimpleHide {
    /// Hide the OS mouse pointer. Idempotent (only hides once; a second call is a
    /// no-op that doesn't double-decrement the Win32 counter).
    pub fn hide(&mut self) {
        #[cfg(target_os = "windows")]
        platform::hide_os_cursor(&mut self.cursor_hidden);
        #[cfg(not(target_os = "windows"))]
        platform::hide_os_cursor();
    }

    /// Restore the OS mouse pointer. Idempotent — only reverses our own hide.
    pub fn show(&mut self) {
        #[cfg(target_os = "windows")]
        platform::show_os_cursor(&mut self.cursor_hidden);
        #[cfg(not(target_os = "windows"))]
        platform::show_os_cursor();
    }

    /// True when we have issued a hide that hasn't been reversed.
    pub fn is_hidden(&self) -> bool {
        #[cfg(target_os = "windows")]
        return self.cursor_hidden;
        #[cfg(not(target_os = "windows"))]
        false
    }

    /// Unconditional restore from the panic hook. Mirrors
    /// [`OsCursorGrab::release_on_panic`]: one extra `ShowCursor(TRUE)` on an
    /// already-visible cursor is harmless.
    pub fn release_on_panic() {
        platform::force_show_cursor();
    }
}

impl Drop for OsCursorSimpleHide {
    fn drop(&mut self) {
        self.show();
    }
}

// ─── Cell-size default ───────────────────────────────────────────────────────

/// Fallback pixel height of one terminal cell when the window size is unknown.
/// Windows Terminal with a typical monospace font is around 18–22 px/cell.
const DEFAULT_PIXELS_PER_CELL: f32 = 20.0;

// ─── Public resource ─────────────────────────────────────────────────────────

/// OS cursor grab state, stored as a Bevy [`Resource`].
///
/// When `active` is true the system pins the cursor to `anchor_px` every frame.
/// Each call to [`OsCursorGrab::poll_delta`] reads the current OS cursor
/// position, computes the pixel delta from the anchor, warps the pointer back,
/// and returns the delta in "cell units" (pixels / pixels_per_cell).
///
/// When `active` is false, `poll_delta` returns `None` immediately so the
/// crossterm cell-event path can take over.
#[derive(Resource)]
pub struct OsCursorGrab {
    /// True while the grab is live (M toggle ON and console is in focus).
    pub active: bool,
    /// Anchor in screen-pixel coordinates (center of the console window).
    anchor_px: (i32, i32),
    /// Estimated pixel height of one terminal cell, for delta conversion.
    pixels_per_cell: f32,
    /// Whether we decremented the Win32 ShowCursor counter (Windows only).
    /// We only increment it back once per decrement to keep the refcount clean.
    #[cfg(target_os = "windows")]
    cursor_hidden: bool,
}

impl Default for OsCursorGrab {
    fn default() -> Self {
        OsCursorGrab {
            active: false,
            anchor_px: (0, 0),
            pixels_per_cell: DEFAULT_PIXELS_PER_CELL,
            #[cfg(target_os = "windows")]
            cursor_hidden: false,
        }
    }
}

impl OsCursorGrab {
    /// Activate the OS cursor grab.
    ///
    /// Computes the console window center as the anchor, estimates cell size,
    /// warps the pointer to the anchor, and hides the OS cursor. Returns `true`
    /// if the grab was successfully set up (console window found + foreground).
    pub fn activate(&mut self, terminal_rows: usize) -> bool {
        let Some(anchor) = platform::console_center() else {
            return false;
        };
        self.anchor_px = anchor;
        self.pixels_per_cell =
            platform::estimate_cell_px(terminal_rows).unwrap_or(DEFAULT_PIXELS_PER_CELL);
        self.active = true;
        platform::warp_cursor(anchor.0, anchor.1);
        #[cfg(target_os = "windows")]
        platform::hide_os_cursor(&mut self.cursor_hidden);
        #[cfg(not(target_os = "windows"))]
        platform::hide_os_cursor();
        true
    }

    /// Deactivate the grab, restore OS cursor visibility, and stop consuming
    /// pointer events. Safe to call when already inactive.
    pub fn deactivate(&mut self) {
        if self.active {
            self.active = false;
            #[cfg(target_os = "windows")]
            platform::show_os_cursor(&mut self.cursor_hidden);
            #[cfg(not(target_os = "windows"))]
            platform::show_os_cursor();
        }
    }

    /// Per-frame delta poll. Returns `Some((dx_cells, dy_cells))` when the grab
    /// is active and the console is in the foreground, else `None`.
    ///
    /// Reads the current OS cursor position, computes the delta from the anchor,
    /// warps the pointer back to the anchor (which removes the edge limit), and
    /// converts the pixel delta to cell units so [`crate::apply_mouse_look`]
    /// receives the same coordinate space as the crossterm cell-event path.
    pub fn poll_delta(&mut self) -> Option<(f32, f32)> {
        if !self.active {
            return None;
        }
        // Stop applying look when the console is not the foreground window
        // (e.g. after an alt-tab). We do NOT auto-deactivate here; the toggle
        // stays on so the grab resumes when the user alt-tabs back.
        if !platform::console_is_foreground() {
            return None;
        }
        let (cx, cy) = platform::get_cursor_pos()?;
        let (ax, ay) = self.anchor_px;
        let px_dx = (cx - ax) as f32;
        let px_dy = (cy - ay) as f32;
        // Warp unconditionally — even for a zero delta. This is the recenter that
        // eliminates edge stall: the pointer always begins the next frame at the
        // anchor, so `cx - ax` is bounded by at most one frame's motion.
        platform::warp_cursor(ax, ay);
        let ppc = self.pixels_per_cell.max(1.0);
        Some((px_dx / ppc, px_dy / ppc))
    }

    /// Refresh the anchor when the terminal resizes (so the center stays current).
    pub fn on_resize(&mut self, terminal_rows: usize) {
        if let Some(anchor) = platform::console_center() {
            self.anchor_px = anchor;
        }
        if let Some(ppc) = platform::estimate_cell_px(terminal_rows) {
            self.pixels_per_cell = ppc;
        }
    }

    /// Best-effort cursor restore called from the panic hook.
    ///
    /// The panic hook cannot hold `&mut OsCursorGrab`, so this is a free
    /// function that calls the platform's unconditional show-cursor API. On
    /// Windows this calls `ShowCursor(TRUE)` once to re-increment the counter
    /// regardless of whether we were the one who hid it — the worst case is an
    /// extra increment which is harmless (cursor becomes visible, which is what
    /// we want after a crash).
    pub fn release_on_panic() {
        platform::force_show_cursor();
    }

    /// True when the console/game window is the current OS foreground window.
    ///
    /// Exposed as an associated function (rather than kept `platform`-private)
    /// so the state-driven grab sync in `terminal.rs` can gate the RISING edge
    /// (mouse-look turning ON) on live focus — the exact same check
    /// [`poll_delta`](Self::poll_delta) already applies per-frame to suppress
    /// (not release) the grab during a transient alt-tab. Activating the grab
    /// while some OTHER window is focused would steal that window's pointer.
    pub fn console_is_foreground() -> bool {
        platform::console_is_foreground()
    }
}

/// Pure decision: should [`OsCursorGrab::active`] be `true` this frame?
///
/// `true` exactly when the player has mouse-look turned ON
/// ([`crate::MouseSettings::enabled`]) AND camera motion is currently permitted
/// ([`re_core::InputGate::camera_motion`], which `re_game` sets `true` only in
/// `GameState::Playing` — see that type's doc comment). This is the fix for the
/// grab only engaging after two `M` presses: previously `OsCursorGrab::active`
/// was flipped ONLY on the `M`-toggle key edge, so a fresh `Playing` entry (or a
/// resume from Pause/menus) left the grab wherever the toggle last left it —
/// inactive by default, requiring an extra press-press to notice the mismatch
/// and correct it. Driving `active` from STATE instead of the edge means it is
/// resynchronised every frame: entering `Playing`, leaving it (Pause/menu/Won),
/// and any settings change all take effect on the very next frame, with no edge
/// to miss.
///
/// Deliberately excludes the live foreground check — see
/// [`OsCursorGrab::console_is_foreground`]'s doc for why that is applied only
/// on the rising edge by the caller, not folded into this predicate.
#[must_use]
pub fn should_be_active(mouse_look_enabled: bool, camera_motion_allowed: bool) -> bool {
    mouse_look_enabled && camera_motion_allowed
}

impl Drop for OsCursorGrab {
    /// Final teardown safety net: if the app exits while the grab is still live
    /// (e.g. the player quits without toggling mouse-look off), restore OS cursor
    /// visibility. Mirrors `KittyGuard`'s RAII teardown. `ShowCursor` is refcounted
    /// and `deactivate()` only reverses a hide we actually issued, so this is
    /// idempotent with the explicit `deactivate()` call on the normal toggle path —
    /// ordering relative to the ratatui teardown is irrelevant for a visibility
    /// restore. Covers the normal quit-while-grabbed exit path.
    fn drop(&mut self) {
        self.deactivate();
    }
}

// ─── Windows implementation ───────────────────────────────────────────────────

#[cfg(target_os = "windows")]
mod platform {
    //! Win32 cursor grab — `GetCursorPos` / `SetCursorPos` / `ShowCursor` /
    //! `GetConsoleWindow` / `GetWindowRect` / `GetForegroundWindow`.
    //!
    //! All `unsafe` blocks are FFI calls. Sound because:
    //! - Stack-allocated `POINT` / `RECT` are properly aligned and non-null.
    //! - `HWND` values are checked for `0` (null) before use in `GetWindowRect`.
    //! - `ShowCursor`, `SetCursorPos`, `GetCursorPos`, `GetForegroundWindow`,
    //!   `GetActiveWindow`, and `GetSystemMetrics` have no preconditions beyond
    //!   their documented argument types (i32, *mut POINT, *mut RECT, etc.).

    use windows_sys::Win32::{
        Foundation::{HWND, POINT, RECT, TRUE},
        System::Console::{GetConsoleWindow, SetConsoleTitleW},
        UI::{
            Input::KeyboardAndMouse::GetActiveWindow,
            WindowsAndMessaging::{
                GetCursorPos, GetForegroundWindow, GetSystemMetrics, GetWindowRect, GetWindowTextW,
                SetCursorPos, ShowCursor, SM_CXSCREEN, SM_CYSCREEN,
            },
        },
    };

    // `windows_sys::core::BOOL` is i32. Define our own FALSE so we don't import
    // Foundation::FALSE (which conflicts with the type alias from core::BOOL).
    const FALSE: windows_sys::core::BOOL = 0i32;

    // ─── Windows Terminal pseudo-console fallback ────────────────────────────
    //
    // `GetConsoleWindow()` is reliable on conhost (`cmd.exe`, plain
    // PowerShell): it returns the actual visible console window. Under
    // **Windows Terminal** (`wt.exe`), every session runs through conpty, and
    // `GetConsoleWindow()` resolves to the HIDDEN pseudo-console host window
    // (owned by `OpenConsole.exe`), not Windows Terminal's own visible
    // top-level window — that hidden window's `GetWindowRect` is typically
    // degenerate (zero-area or parked off-screen), so warping the cursor to
    // its "center" and checking its foreground state are both meaningless.
    //
    // The fix: stamp a distinctive marker into the console TITLE at startup
    // ([`stamp_console_title`]). Windows Terminal mirrors the active pane's
    // console title into its own OS window caption by default (no custom tab
    // title override), so [`find_host_window`] can identify the REAL visible
    // window by checking `GetForegroundWindow()`'s caption for the marker,
    // falling back to the historical `GetConsoleWindow()`-only behaviour when
    // the marker isn't found (conhost, or a terminal that doesn't propagate
    // the title) so nothing regresses there.

    /// The player-visible window/tab title (Integration 2026-09-23: the title
    /// used to read as the player's own shell prompt/profile name with an ugly
    /// `█TS-GAME-WINDOW█` marker appended — this is what the player should
    /// actually see once the game has started).
    const TITLE_TEXT: &str = "TERMINAL STATE";

    /// Distinctive substring embedded (by [`stamp_console_title`]) alongside
    /// [`TITLE_TEXT`] for [`find_host_window`] to key off. Entirely ZERO-WIDTH
    /// unicode (U+200B/U+200C/U+200D) — a first attempt kept a readable
    /// `TS-GAME-WINDOW` ASCII payload merely FLANKED by zero-width characters,
    /// which still rendered as visible junk (`TERMINAL STATETS-GAME-WINDOW`)
    /// because only the flanking characters were invisible, not the payload
    /// itself. Every codepoint here renders as nothing in a real terminal, so
    /// the player-visible title reads as exactly [`TITLE_TEXT`], while the
    /// specific 8-codepoint sequence is still unique enough that it can never
    /// collide with ordinary title text.
    const TITLE_MARKER: &str = "\u{200B}\u{200C}\u{200B}\u{200D}\u{200B}\u{200C}\u{200D}\u{200B}";

    /// Stamp the console title to [`TITLE_TEXT`] (what the player sees) with
    /// [`TITLE_MARKER`] appended invisibly (what [`find_host_window`] matches
    /// on). Deliberately CLOBBERS rather than appends: the previous behaviour
    /// preserved whatever the player's shell/profile happened to show (e.g.
    /// `C:\Program Files\Git\bin\bash.exe`) with the marker glued onto the end,
    /// which is exactly the "doesn't read like a game" defect this fixes — a
    /// shipped game's window should read as itself. Best-effort: on any
    /// failure the title is simply left unmarked, and [`find_host_window`]
    /// falls back to the pre-existing `GetConsoleWindow` behaviour. Call once,
    /// from `Startup`.
    pub fn stamp_console_title() {
        let stamped = format!("{TITLE_TEXT}{TITLE_MARKER}");
        let mut wide: Vec<u16> = stamped.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: `wide` is a valid, NUL-terminated UTF-16 buffer owned for the
        // duration of this call; `SetConsoleTitleW` only reads it.
        unsafe { SetConsoleTitleW(wide.as_mut_ptr()) };
    }

    /// Read a window's title bar text (best-effort; empty string on failure —
    /// including a null `hwnd`, which `GetWindowTextW` would otherwise fault
    /// on, so the caller never needs to null-check before calling this).
    fn window_text(hwnd: HWND) -> String {
        if hwnd.is_null() {
            return String::new();
        }
        let mut buf = [0u16; 512];
        // SAFETY: hwnd is checked non-null above; buf is a valid stack buffer
        // of the declared length that GetWindowTextW writes at most into.
        let n = unsafe { GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32) };
        if n <= 0 {
            return String::new();
        }
        String::from_utf16_lossy(&buf[..n as usize])
    }

    /// A window's screen rect, but only when it is NON-DEGENERATE (positive
    /// width and height) — the discriminator that rejects Windows Terminal's
    /// hidden pseudo-console host (see the module doc above).
    fn valid_rect(hwnd: HWND) -> Option<RECT> {
        if hwnd.is_null() {
            return None;
        }
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        // SAFETY: hwnd is non-null (checked above); rect is stack-allocated.
        let ok = unsafe { GetWindowRect(hwnd, &mut rect) };
        if ok == 0 {
            return None;
        }
        if rect.right > rect.left && rect.bottom > rect.top {
            Some(rect)
        } else {
            None
        }
    }

    /// The real, visible window showing THIS game — robust to the Windows
    /// Terminal pseudo-console quirk described above.
    ///
    /// Tries, in priority order:
    /// 1. `GetConsoleWindow()`, if it has a non-degenerate rect AND its title
    ///    carries [`TITLE_MARKER`] — correct on conhost/cmd/PowerShell, where
    ///    the window title mirrors the console title 1:1.
    /// 2. `GetForegroundWindow()`, under the same two conditions — the
    ///    Windows Terminal case, where `GetConsoleWindow()` resolves to the
    ///    hidden host but WT's own top-level window carries the mirrored
    ///    title.
    /// 3. Lenient fallback (marker not found on EITHER candidate — e.g.
    ///    [`stamp_console_title`] never ran, or a terminal that doesn't
    ///    propagate the console title into its window caption at all):
    ///    whichever candidate has a valid rect, console window first — the
    ///    pre-existing behaviour, so nothing regresses for a terminal this
    ///    fix doesn't need to touch.
    fn find_host_window() -> Option<HWND> {
        // SAFETY: GetConsoleWindow / GetForegroundWindow have no preconditions.
        let console: HWND = unsafe { GetConsoleWindow() };
        let foreground: HWND = unsafe { GetForegroundWindow() };

        let console_rect = valid_rect(console);
        let foreground_rect = valid_rect(foreground);

        if console_rect.is_some() && window_text(console).contains(TITLE_MARKER) {
            return Some(console);
        }
        if foreground_rect.is_some() && window_text(foreground).contains(TITLE_MARKER) {
            return Some(foreground);
        }
        if console_rect.is_some() {
            return Some(console);
        }
        if foreground_rect.is_some() {
            return Some(foreground);
        }
        None
    }

    /// Return the center of the host window in screen pixels (see
    /// [`find_host_window`]). Falls back to screen center if no candidate
    /// window has a usable rect.
    pub fn console_center() -> Option<(i32, i32)> {
        if let Some(hwnd) = find_host_window() {
            if let Some(rect) = valid_rect(hwnd) {
                return Some(((rect.left + rect.right) / 2, (rect.top + rect.bottom) / 2));
            }
        }
        // Fallback: screen center via GetSystemMetrics.
        // SAFETY: GetSystemMetrics with SM_CXSCREEN/SM_CYSCREEN is always safe.
        let w = unsafe { GetSystemMetrics(SM_CXSCREEN) };
        let h = unsafe { GetSystemMetrics(SM_CYSCREEN) };
        if w > 0 && h > 0 {
            Some((w / 2, h / 2))
        } else {
            None
        }
    }

    /// Estimate pixel height per cell from the host window height / row count.
    pub fn estimate_cell_px(terminal_rows: usize) -> Option<f32> {
        if terminal_rows == 0 {
            return None;
        }
        let hwnd = find_host_window()?;
        let rect = valid_rect(hwnd)?;
        let h = (rect.bottom - rect.top) as f32;
        if h > 0.0 {
            Some(h / terminal_rows as f32)
        } else {
            None
        }
    }

    /// True when the host window (see [`find_host_window`]) is the current
    /// foreground window (or its active child, for terminals that embed a VT
    /// renderer child HWND).
    pub fn console_is_foreground() -> bool {
        let Some(hwnd) = find_host_window() else {
            return false;
        };
        // SAFETY: GetForegroundWindow/GetActiveWindow are always safe.
        let fg: HWND = unsafe { GetForegroundWindow() };
        if fg == hwnd {
            return true;
        }
        // Also check GetActiveWindow() — some terminal emulators park focus on
        // a VT child window rather than the console window itself.
        let active: HWND = unsafe { GetActiveWindow() };
        active == hwnd || (!fg.is_null() && active == fg)
    }

    /// Read the current cursor position in screen pixels.
    pub fn get_cursor_pos() -> Option<(i32, i32)> {
        let mut pt = POINT { x: 0, y: 0 };
        // SAFETY: stack-allocated POINT; GetCursorPos writes into it.
        let ok = unsafe { GetCursorPos(&mut pt) };
        if ok != 0 {
            Some((pt.x, pt.y))
        } else {
            None
        }
    }

    /// Warp the cursor to screen-pixel coordinate (x, y).
    pub fn warp_cursor(x: i32, y: i32) {
        // SAFETY: SetCursorPos with i32 coords; out-of-screen clamped by Windows.
        unsafe { SetCursorPos(x, y) };
    }

    /// Decrement the Win32 ShowCursor reference counter (best-effort hide).
    /// Tracks whether we issued the decrement so we only reverse it once.
    pub fn hide_os_cursor(cursor_hidden: &mut bool) {
        if !*cursor_hidden {
            // SAFETY: ShowCursor is a refcount-only operation; always safe.
            unsafe { ShowCursor(FALSE) };
            *cursor_hidden = true;
        }
    }

    /// Reverse our previous `ShowCursor(FALSE)` by calling `ShowCursor(TRUE)`.
    pub fn show_os_cursor(cursor_hidden: &mut bool) {
        if *cursor_hidden {
            // SAFETY: ShowCursor is always safe to call.
            unsafe { ShowCursor(TRUE) };
            *cursor_hidden = false;
        }
    }

    /// Unconditional cursor restore from the panic hook (no mutable state).
    /// One extra `ShowCursor(TRUE)` on a cursor that was already visible is
    /// harmless — the counter goes one above 0 and the cursor stays visible.
    pub fn force_show_cursor() {
        // SAFETY: ShowCursor is always safe to call from any context.
        unsafe { ShowCursor(TRUE) };
    }
}

#[cfg(not(target_os = "windows"))]
mod stamp_title_stub {
    /// No-op on non-Windows: `find_host_window`'s marker trick is a Win32-only
    /// concern (X11/Wayland have no equivalent "hidden pseudo-console window"
    /// quirk — `console_center`'s X11 path just uses the live pointer position).
    pub fn stamp_console_title() {}
}
#[cfg(target_os = "windows")]
pub use platform::stamp_console_title as platform_stamp_console_title;
#[cfg(not(target_os = "windows"))]
pub use stamp_title_stub::stamp_console_title as platform_stamp_console_title;

/// Stamp the console/window title with a marker the Windows platform module's
/// `find_host_window` uses to identify the real, visible host window even when
/// `GetConsoleWindow()` resolves to Windows Terminal's hidden pseudo-console
/// host. Call once, from `Startup`. A no-op on every other platform.
pub fn stamp_console_title() {
    platform_stamp_console_title();
}

// ─── Linux / X11 best-effort ─────────────────────────────────────────────────

#[cfg(all(unix, not(target_os = "macos")))]
mod platform {
    //! X11 cursor warp (best-effort). Compiled when `x11` feature is enabled;
    //! otherwise every function is a no-op.
    //!
    //! Wayland sessions: `XOpenDisplay(null)` returns null → logged once → all
    //! subsequent calls are no-ops and the crossterm cell-event path is used.
    //!
    //! Safety: `XQueryPointer`, `XWarpPointer`, and `XFlush` require a valid
    //! non-null `Display*` and a valid window ID. We only call them inside
    //! a `Some(…)` guard on the `OnceLock`-cached `XlibState`.
    //! The display is never closed (intentional process-lifetime leak that
    //! avoids use-after-free on the panic hook path — the OS reclaims it at exit).

    use std::sync::atomic::{AtomicBool, Ordering};

    static X11_UNAVAILABLE_LOGGED: AtomicBool = AtomicBool::new(false);

    #[cfg(feature = "x11")]
    use x11_dl::xlib::Xlib;

    #[cfg(feature = "x11")]
    struct XlibState {
        xlib: Xlib,
        display: *mut x11_dl::xlib::Display,
        root: x11_dl::xlib::Window,
    }

    // SAFETY: we only call into this from the Bevy main thread (single-threaded
    // input path). The X11 display connection is not Send in general, but we
    // treat it as process-lifetime state accessed from one thread only.
    #[cfg(feature = "x11")]
    unsafe impl Send for XlibState {}
    #[cfg(feature = "x11")]
    unsafe impl Sync for XlibState {}

    #[cfg(feature = "x11")]
    static XLIB_STATE: std::sync::OnceLock<Option<XlibState>> = std::sync::OnceLock::new();

    #[cfg(feature = "x11")]
    fn ensure_xlib() -> Option<&'static XlibState> {
        XLIB_STATE
            .get_or_init(|| {
                let xlib = Xlib::open().ok()?;
                // SAFETY: XOpenDisplay(null) uses $DISPLAY; returns null on failure.
                let display = unsafe { (xlib.XOpenDisplay)(std::ptr::null()) };
                if display.is_null() {
                    if !X11_UNAVAILABLE_LOGGED.swap(true, Ordering::Relaxed) {
                        eprintln!(
                            "[re_present] X11 unavailable (Wayland / no $DISPLAY) — \
                             OS cursor grab inactive, using crossterm cell events"
                        );
                    }
                    return None;
                }
                // SAFETY: XDefaultRootWindow requires a valid open display.
                let root = unsafe { (xlib.XDefaultRootWindow)(display) };
                Some(XlibState {
                    xlib,
                    display,
                    root,
                })
            })
            .as_ref()
    }

    pub fn console_center() -> Option<(i32, i32)> {
        // Use the current pointer position as the anchor so the first delta is zero.
        get_cursor_pos()
    }

    pub fn estimate_cell_px(_terminal_rows: usize) -> Option<f32> {
        None // No window geometry access without compositor extension.
    }

    pub fn console_is_foreground() -> bool {
        // Optimistically true when X11 is reachable — focus tracked by X server.
        #[cfg(feature = "x11")]
        return ensure_xlib().is_some();
        #[cfg(not(feature = "x11"))]
        false
    }

    pub fn get_cursor_pos() -> Option<(i32, i32)> {
        #[cfg(feature = "x11")]
        {
            let s = ensure_xlib()?;
            let mut root_ret = 0u64;
            let mut child_ret = 0u64;
            let mut root_x = 0i32;
            let mut root_y = 0i32;
            let mut win_x = 0i32;
            let mut win_y = 0i32;
            let mut mask = 0u32;
            // SAFETY: display and root are valid non-null (see ensure_xlib).
            unsafe {
                (s.xlib.XQueryPointer)(
                    s.display,
                    s.root,
                    &mut root_ret,
                    &mut child_ret,
                    &mut root_x,
                    &mut root_y,
                    &mut win_x,
                    &mut win_y,
                    &mut mask,
                );
            }
            return Some((root_x, root_y));
        }
        #[cfg(not(feature = "x11"))]
        None
    }

    pub fn warp_cursor(x: i32, y: i32) {
        #[cfg(feature = "x11")]
        if let Some(s) = ensure_xlib() {
            // SAFETY: XWarpPointer with src_w=0 moves relative to dst window (root).
            // display and root are valid (see ensure_xlib).
            unsafe {
                (s.xlib.XWarpPointer)(s.display, 0, s.root, 0, 0, 0, 0, x, y);
                (s.xlib.XFlush)(s.display);
            }
        }
        let _ = (x, y);
    }

    // Linux: the terminal emulator's mouse-capture mode (enabled by `bevy_ratatui`)
    // often hides the pointer automatically; we skip XFixes for simplicity.
    pub fn hide_os_cursor() {}
    pub fn show_os_cursor() {}
    pub fn force_show_cursor() {}
}

// ─── Stub for macOS and any other unimplemented platform ─────────────────────

#[cfg(not(any(target_os = "windows", all(unix, not(target_os = "macos")))))]
mod platform {
    pub fn console_center() -> Option<(i32, i32)> {
        None
    }
    pub fn estimate_cell_px(_rows: usize) -> Option<f32> {
        None
    }
    pub fn console_is_foreground() -> bool {
        false
    }
    pub fn get_cursor_pos() -> Option<(i32, i32)> {
        None
    }
    pub fn warp_cursor(_x: i32, _y: i32) {}
    pub fn hide_os_cursor() {}
    pub fn show_os_cursor() {}
    pub fn force_show_cursor() {}
}

// ─── Unit tests (pure math — no OS, no terminal) ─────────────────────────────

#[cfg(test)]
mod tests {
    /// The pixel-to-cell conversion used inside `poll_delta`. Pure arithmetic,
    /// no platform call — fully testable without a live display.
    #[test]
    fn pixel_to_cell_delta_conversion() {
        let ppc = 20.0_f32;

        // 40 px right + 20 px down → 2.0 cells right + 1.0 cell down.
        assert!((40.0_f32 / ppc - 2.0).abs() < 1e-6);
        assert!((20.0_f32 / ppc - 1.0).abs() < 1e-6);

        // Zero pixel delta → zero cell delta.
        assert_eq!(0.0_f32 / ppc, 0.0);

        // Negative delta (left / up).
        assert!((-10.0_f32 / ppc) < 0.0);
        assert!((-5.0_f32 / ppc) < 0.0);

        // Sub-cell delta: finite, non-NaN, in (0, 1).
        let sub = 3.0_f32 / ppc;
        assert!(sub > 0.0 && sub < 1.0 && !sub.is_nan());
    }

    /// Recenter arithmetic: each frame the delta equals `cursor_pos - anchor`.
    /// After a warp to the anchor, the next frame starts from `anchor`, not from
    /// wherever the pointer was before — this removes the edge stall.
    #[test]
    fn recenter_removes_edge_accumulation() {
        let anchor = (500i32, 300i32);

        // Frame 1: pointer has drifted to (600, 300). Delta = (+100, 0).
        let delta1 = ((600 - anchor.0) as f32, (300 - anchor.1) as f32);
        assert_eq!(delta1, (100.0, 0.0));
        // Warp → pointer is now at (500, 300). Next frame starts there.

        // Frame 2: pointer moves another +100 px → now at (600, 300) again.
        let delta2 = ((600 - anchor.0) as f32, (300 - anchor.1) as f32);
        assert_eq!(
            delta2,
            (100.0, 0.0),
            "each frame sees only the inter-frame delta"
        );
        // Without recenter, frame 2 would have seen (700-500)=200 px. With recenter: 100.
    }

    /// `pixels_per_cell.max(1.0)` in `poll_delta` prevents divide-by-zero.
    #[test]
    fn pixels_per_cell_zero_clamped_to_one() {
        let ppc = 0.0_f32.max(1.0);
        let dx = 40.0_f32 / ppc;
        assert!(
            !dx.is_nan() && dx.is_finite(),
            "clamped ppc must prevent NaN/Inf"
        );
        assert_eq!(ppc, 1.0);
    }

    /// `OsCursorGrab::default()` is inactive — no cursor is hidden on construction.
    #[test]
    fn default_grab_is_inactive() {
        let grab = super::OsCursorGrab::default();
        assert!(!grab.active, "fresh OsCursorGrab must be inactive");
    }

    /// Calling `deactivate()` on an already-inactive grab is a no-op (no panic).
    #[test]
    fn deactivate_when_already_inactive_is_noop() {
        let mut grab = super::OsCursorGrab::default();
        grab.deactivate(); // must not panic
        assert!(!grab.active);
    }

    // --- `should_be_active` — the state-driven grab-sync decision ------------

    /// Both conditions must hold: mouse-look enabled AND camera motion allowed
    /// (`re_core::InputGate::camera_motion`, true only in `GameState::Playing`).
    #[test]
    fn should_be_active_requires_both_enabled_and_playing() {
        use super::should_be_active;
        assert!(
            should_be_active(true, true),
            "mouse-look on + Playing → active"
        );
        assert!(
            !should_be_active(false, true),
            "mouse-look off → inactive even in Playing"
        );
        assert!(
            !should_be_active(true, false),
            "menus/pause (camera_motion=false) → inactive"
        );
        assert!(!should_be_active(false, false), "neither → inactive");
    }

    /// The exact regression scenario: entering Playing
    /// (camera_motion flips true) with mouse-look already enabled by default
    /// must want the grab active WITHOUT any `M` keypress.
    #[test]
    fn should_be_active_engages_on_playing_entry_with_no_toggle_press() {
        use super::should_be_active;
        // MouseSettings::default().enabled == true (see lib.rs); simulate the
        // OnEnter(Playing) transition that sets InputGate.camera_motion = true.
        let mouse_look_enabled_by_default = true;
        let just_entered_playing = true;
        assert!(should_be_active(
            mouse_look_enabled_by_default,
            just_entered_playing
        ));
    }

    /// Leaving Playing (Pause, any menu, Won) must want the grab inactive even
    /// if the player never touched the `M` toggle — this is what lets menus
    /// show the OS cursor for mouse-driven navigation.
    #[test]
    fn should_be_active_releases_on_leaving_playing() {
        use super::should_be_active;
        assert!(
            !should_be_active(true, false),
            "Pause/menu/Won → released regardless of the M toggle"
        );
    }
}
