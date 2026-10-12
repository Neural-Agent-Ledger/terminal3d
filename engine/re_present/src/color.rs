//! Color emission policy shared by both present backends.
//!
//! Upstream every framebuffer [`re_core::Cell`] carries packed `0x00RRGGBB`
//! foreground and background colors. The terminal a player runs in may or may
//! not speak 24-bit truecolor, so the present layer exposes a runtime
//! [`ColorMode`] resource that decides, per frame, whether each cell color is
//! emitted as full RGB or quantized to the xterm 256-color palette.
//!
//! This module is backend-agnostic: it only converts colors. Each backend
//! consumes [`ColorMode`] and maps the resulting choice onto its own color type
//! (ratatui `Color`, bevy `LinearRgba`).
//!
//! ## Selector precedence (highest wins)
//!
//! 1. An explicit `--color=truecolor|256` CLI flag (wired in `re_game`).
//! 2. Auto-detect from the environment: if no flag is given, [`ColorMode::Truecolor`]
//!    is used when `COLORTERM` advertises it (`truecolor` / `24bit`, case-insensitive).
//!    Otherwise — including when `COLORTERM` is simply unset — **on Windows** we
//!    probe the console's actual VT capability (`ENABLE_VIRTUAL_TERMINAL_PROCESSING`
//!    via `GetConsoleMode`/`SetConsoleMode`) rather than assuming the absence of
//!    `COLORTERM` means no truecolor: Windows consoles (conhost, Windows Terminal)
//!    have supported 24-bit VT sequences since Windows 10 1511 but never set
//!    `COLORTERM`, so treating "unset" as "no truecolor" downgraded every
//!    Windows player to [`ColorMode::Ansi256`] regardless of what their console
//!    could actually render. On non-Windows platforms (or if the Windows probe
//!    fails — an old conhost predating VT support), we downgrade to
//!    [`ColorMode::Ansi256`] exactly as before.
//!
//! A manual flag therefore always overrides auto-detect, and an explicit
//! `COLORTERM` value always overrides the Windows probe.

use bevy::prelude::Resource;

/// How the present layer emits cell colors at runtime.
///
/// Inserted as a Bevy resource by `re_game`; both backends read it each frame.
/// `Default` resolves through [`ColorMode::resolve`]'s env auto-detect (`None` —
/// no explicit choice) rather than hardcoding [`ColorMode::Truecolor`]: any
/// consumer that reaches this via `init_resource::<ColorMode>()` (a
/// flagless/settings-less build, or a plugin that inits it before `re_game`'s
/// settings-driven `insert_resource` overrides it) still gets the terminal-aware
/// answer instead of an unconditional truecolor guess.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    /// Emit full 24-bit RGB. Requires a truecolor-capable terminal.
    Truecolor,
    /// Quantize every fg/bg to the nearest xterm 256-color palette index.
    Ansi256,
}

impl Default for ColorMode {
    /// Auto-detects via [`Self::resolve`]`(None)` — see the type docs.
    fn default() -> Self {
        Self::resolve(None)
    }
}

impl ColorMode {
    /// Resolve the effective color mode from an optional explicit choice plus
    /// the environment, applying the documented precedence:
    ///
    /// * `Some(mode)` — an explicit `--color` flag — is returned verbatim.
    /// * `None` — no flag — defaults to [`ColorMode::Truecolor`], downgraded to
    ///   [`ColorMode::Ansi256`] when `COLORTERM` is not `truecolor`/`24bit`.
    pub fn resolve(explicit: Option<ColorMode>) -> ColorMode {
        if let Some(mode) = explicit {
            return mode;
        }
        Self::auto_from_env()
    }

    /// Auto-detect the effective color mode from `COLORTERM` plus, on Windows
    /// when `COLORTERM` is absent, a real console VT-capability probe.
    ///
    /// This is a thin wrapper around the pure, testable `Self::decide_auto`:
    /// it only gathers the three inputs (are we on Windows, what did
    /// `COLORTERM` say, did the Windows probe succeed) and hands them off. The
    /// probe itself is skipped whenever it cannot change the answer — non-Windows
    /// builds, or an explicit `COLORTERM` — so this never pays for a
    /// `GetConsoleMode`/`SetConsoleMode` round-trip it doesn't need.
    pub fn auto_from_env() -> ColorMode {
        let colorterm = std::env::var("COLORTERM").ok();
        let is_windows = cfg!(target_os = "windows");
        let vt_probe_ok = is_windows && colorterm.is_none() && platform::supports_truecolor_vt();
        Self::decide_auto(is_windows, colorterm.as_deref(), vt_probe_ok)
    }

    /// The pure decision behind [`Self::auto_from_env`], factored out so the
    /// auto-detect policy is unit-testable with no live console:
    ///
    /// * `is_windows` — target-platform flag (normally `cfg!(target_os = "windows")`).
    /// * `colorterm` — the raw `COLORTERM` env value, if the variable was set at all.
    /// * `vt_probe_ok` — the result of the Windows `ENABLE_VIRTUAL_TERMINAL_PROCESSING`
    ///   capability probe. Ignored on non-Windows and whenever `colorterm` is
    ///   `Some` (an explicit `COLORTERM` always wins over the probe).
    ///
    /// Precedence: an explicit `COLORTERM` classification always wins; failing
    /// that, Windows trusts a successful VT probe; failing that (non-Windows, or
    /// a Windows console too old to support VT processing), downgrade to
    /// [`ColorMode::Ansi256`] — the pre-existing behavior.
    pub(crate) fn decide_auto(
        is_windows: bool,
        colorterm: Option<&str>,
        vt_probe_ok: bool,
    ) -> ColorMode {
        if let Some(v) = colorterm {
            return Self::from_colorterm(v);
        }
        if is_windows && vt_probe_ok {
            ColorMode::Truecolor
        } else {
            ColorMode::Ansi256
        }
    }

    /// Classify a `COLORTERM` value. Only `truecolor` / `24bit` (any case) keep
    /// truecolor; everything else (including empty) downgrades to 256.
    pub fn from_colorterm(colorterm: &str) -> ColorMode {
        let v = colorterm.trim().to_ascii_lowercase();
        if v == "truecolor" || v == "24bit" {
            ColorMode::Truecolor
        } else {
            ColorMode::Ansi256
        }
    }

    /// Parse a `--color` flag value (`truecolor` | `256`). Accepts a few common
    /// spellings; returns `None` for anything unrecognized so the caller can
    /// report a usage error.
    pub fn parse_flag(value: &str) -> Option<ColorMode> {
        match value.trim().to_ascii_lowercase().as_str() {
            "truecolor" | "true" | "24bit" | "rgb" => Some(ColorMode::Truecolor),
            "256" | "ansi256" | "ansi" | "8bit" => Some(ColorMode::Ansi256),
            _ => None,
        }
    }
}

// ─── Windows console VT-capability probe ─────────────────────────────────────
//
// Confined `unsafe` FFI, mirroring the pattern in `os_cursor.rs`: `unsafe`
// blocks are minimal, each documented with the invariant that makes it sound.
// A stub with the same signature exists on every other platform so
// `auto_from_env` never needs a platform `#[cfg]` at its call site.

#[cfg(target_os = "windows")]
mod platform {
    //! Win32 console VT probe — `GetStdHandle` / `GetConsoleMode` /
    //! `SetConsoleMode`.
    //!
    //! Safety: `GetStdHandle(STD_OUTPUT_HANDLE)` is a well-known pseudo-handle
    //! request with no preconditions. Its result is checked against both `NULL`
    //! and `INVALID_HANDLE_VALUE` before use. `GetConsoleMode`/`SetConsoleMode`
    //! require a valid, non-null/non-invalid console handle (checked) and a
    //! stack-allocated `CONSOLE_MODE` output pointer (always the case here).

    use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Console::{
        GetConsoleMode, GetStdHandle, SetConsoleMode, CONSOLE_MODE,
        ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_OUTPUT_HANDLE,
    };

    /// True when stdout's console handle can run in
    /// `ENABLE_VIRTUAL_TERMINAL_PROCESSING` mode — the flag that makes conhost
    /// (and Windows Terminal, which already runs with it on) interpret 24-bit
    /// VT color escapes instead of ignoring or mis-rendering them. Available
    /// since Windows 10 1511; a `SetConsoleMode` failure here means an older
    /// console that predates VT support.
    ///
    /// This is a CAPABILITY CHECK, not a lasting enable: if the flag was not
    /// already set, we set it just long enough to confirm the call succeeds,
    /// then restore the original mode immediately. The terminal backend (not
    /// this probe) is responsible for any permanent mode change it needs when
    /// it actually sets up raw/VT mode.
    ///
    /// Returns `false` (never panics) for every failure path: no console
    /// attached (e.g. output redirected to a file/pipe with no console
    /// backing), a closed/invalid handle, or a `GetConsoleMode`/`SetConsoleMode`
    /// failure.
    pub fn supports_truecolor_vt() -> bool {
        // SAFETY: STD_OUTPUT_HANDLE is a fixed pseudo-handle constant;
        // GetStdHandle has no preconditions and never panics.
        let handle: HANDLE = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            return false;
        }

        let mut mode: CONSOLE_MODE = 0;
        // SAFETY: handle checked non-null/non-invalid above; `mode` is a valid
        // stack-allocated output pointer.
        if unsafe { GetConsoleMode(handle, &mut mode) } == 0 {
            return false;
        }

        if mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING != 0 {
            // Already on (e.g. the terminal host itself enabled it) — no
            // round-trip needed to prove the capability exists.
            return true;
        }

        let probed_mode = mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        // SAFETY: handle is valid (checked above); probed_mode is a
        // well-formed CONSOLE_MODE bitmask (existing bits + one documented flag).
        let enabled = unsafe { SetConsoleMode(handle, probed_mode) } != 0;
        if enabled {
            // Restore: this function is a probe, not a permanent mode change.
            // SAFETY: handle is valid; `mode` is the exact value read back above.
            unsafe { SetConsoleMode(handle, mode) };
        }
        enabled
    }
}

#[cfg(not(target_os = "windows"))]
mod platform {
    /// Non-Windows stub: always `false`. Non-Windows terminals are detected
    /// purely through `COLORTERM` (unchanged pre-existing behavior); there is
    /// no equivalent probe wired here.
    pub fn supports_truecolor_vt() -> bool {
        false
    }
}

/// Whether the CURRENT stdout console can run VT/ANSI escape processing at
/// all — the broader capability [`intro_tear`](crate::intro_tear) needs
/// before writing raw CSI sequences (delete/insert-char, insert/delete-line,
/// scroll-region) that have no crossterm `Command` wrapper to fall back on.
///
/// Every real terminal on every non-Windows platform is ANSI-native by
/// construction (the same assumption [`ColorMode::auto_from_env`] already
/// makes for ITS non-Windows path), so this is `true` there. On Windows it
/// reuses the exact same [`platform::supports_truecolor_vt`] probe
/// `ColorMode` uses for truecolor — a console that can enable
/// `ENABLE_VIRTUAL_TERMINAL_PROCESSING` can run every sequence this crate
/// emits, not just truecolor SGR.
pub(crate) fn vt_capable() -> bool {
    !cfg!(target_os = "windows") || platform::supports_truecolor_vt()
}

/// The six intensity levels used by the xterm 6×6×6 color cube.
const CUBE_LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];

/// Map an 8-bit channel to its nearest cube level INDEX (0..=5).
#[inline]
fn nearest_cube_index(c: u8) -> u8 {
    let mut best = 0u8;
    let mut best_d = u16::MAX;
    for (i, &level) in CUBE_LEVELS.iter().enumerate() {
        let d = (c as i16 - level as i16).unsigned_abs();
        if d < best_d {
            best_d = d;
            best = i as u8;
        }
    }
    best
}

/// Squared distance between two RGB triples (kept in `u32`, no overflow).
#[inline]
fn dist2(a: (u8, u8, u8), b: (u8, u8, u8)) -> u32 {
    let dr = a.0 as i32 - b.0 as i32;
    let dg = a.1 as i32 - b.1 as i32;
    let db = a.2 as i32 - b.2 as i32;
    (dr * dr + dg * dg + db * db) as u32
}

/// Chroma (max channel − min channel) at or below which a color is treated as
/// effectively neutral for palette selection, letting the 24-step grayscale
/// ramp compete for it even when a cube candidate is numerically closer.
///
/// Without this gate, `rgb_to_ansi256` compared raw RGB distance only: the
/// grayscale ramp's dense 10-unit steps are almost always closer in Euclidean
/// terms than the cube's coarse ~40-unit steps for any *dark, low-saturation*
/// color — even a clearly tinted one — so tinted darks (e.g. `#1A2A40`, a dark
/// blue) were quantized to flat gray, discarding hue entirely. Gating the ramp
/// candidate on chroma keeps genuinely neutral/near-neutral colors (chroma ≤ 12,
/// about 4.7% of the 0..=255 range — below the tint a normal display makes
/// visually distinguishable from true gray) on the fine ramp, while anything
/// with a real tint is forced into the cube, which preserves hue even at the
/// cost of coarser brightness steps.
const NEUTRAL_CHROMA_THRESHOLD: i32 = 12;

/// Quantize an RGB color to the nearest **xterm 256-color** palette index.
///
/// Considers two candidate families and picks the nearest, with one gate:
/// * the 6×6×6 color cube (indices 16..=231) is always a candidate;
/// * the 24-step grayscale ramp (indices 232..=255) is a candidate ONLY when
///   the input's chroma is at or below `NEUTRAL_CHROMA_THRESHOLD` — see its
///   doc comment for why. A tinted color therefore always lands in the cube,
///   never the ramp, regardless of raw RGB distance.
///
/// The system colors 0..=15 are intentionally NOT used as targets: they are
/// terminal-theme-dependent and unreliable, so we always land in the
/// deterministic cube / grayscale region. Known anchors: pure red → 196, pure
/// green → 46, pure blue → 21, pure white → 231, pure black → 16.
pub fn rgb_to_ansi256(r: u8, g: u8, b: u8) -> u8 {
    let target = (r, g, b);

    // Candidate 1: nearest entry in the 6×6×6 cube.
    let (ci, cg, cb) = (
        nearest_cube_index(r),
        nearest_cube_index(g),
        nearest_cube_index(b),
    );
    let cube_rgb = (
        CUBE_LEVELS[ci as usize],
        CUBE_LEVELS[cg as usize],
        CUBE_LEVELS[cb as usize],
    );
    let cube_idx = 16 + 36 * ci + 6 * cg + cb;
    let cube_d = dist2(target, cube_rgb);

    // Candidate 2: nearest entry in the 24-step grayscale ramp (232..=255),
    // whose levels are 8, 18, ..., 238 (8 + 10*i).
    let gray_avg = (r as u16 + g as u16 + b as u16) / 3;
    // Closest ramp step i to `gray_avg`: round((avg - 8) / 10), clamped 0..=23.
    let step = if gray_avg < 8 {
        0i32
    } else {
        (((gray_avg as i32 - 8) * 2 + 10) / 20).clamp(0, 23)
    };
    let gray_level = (8 + 10 * step) as u8;
    let gray_rgb = (gray_level, gray_level, gray_level);
    let gray_idx = 232 + step as u8;
    let gray_d = dist2(target, gray_rgb);

    // The ramp only competes when the input is near-neutral; a genuine tint
    // is always routed to the cube even if the ramp is numerically closer.
    let chroma = i32::from(r.max(g).max(b)) - i32::from(r.min(g).min(b));
    if chroma <= NEUTRAL_CHROMA_THRESHOLD && gray_d < cube_d {
        gray_idx
    } else {
        cube_idx
    }
}

/// The RGB an xterm 256-color INDEX renders as, for the deterministic region we
/// emit (cube 16..=231 and grayscale 232..=255). Indices 0..=15 — which we
/// never produce — fall back to black, since their true value is theme-defined.
///
/// Used by the window backend, which has no palette of its own: it quantizes to
/// an index, then maps the index back to a concrete RGB for the tile.
pub fn ansi256_to_rgb(idx: u8) -> (u8, u8, u8) {
    match idx {
        16..=231 => {
            let i = idx - 16;
            let r = CUBE_LEVELS[(i / 36) as usize];
            let g = CUBE_LEVELS[((i / 6) % 6) as usize];
            let b = CUBE_LEVELS[(i % 6) as usize];
            (r, g, b)
        }
        232..=255 => {
            let level = 8 + 10 * (idx - 232);
            (level, level, level)
        }
        // System colors: theme-dependent; we never emit these as targets.
        _ => (0, 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_colors_hit_known_indices() {
        assert_eq!(rgb_to_ansi256(255, 0, 0), 196, "pure red");
        assert_eq!(rgb_to_ansi256(0, 255, 0), 46, "pure green");
        assert_eq!(rgb_to_ansi256(0, 0, 255), 21, "pure blue");
        assert_eq!(rgb_to_ansi256(255, 255, 255), 231, "pure white");
        assert_eq!(rgb_to_ansi256(0, 0, 0), 16, "pure black");
    }

    #[test]
    fn mid_gray_lands_in_grayscale_ramp() {
        // 0x80 = 128: the cube only has 135/95 nearby, the gray ramp has 128.
        let idx = rgb_to_ansi256(128, 128, 128);
        assert!(
            (232..=255).contains(&idx),
            "mid-gray should pick the grayscale ramp, got {idx}"
        );
        // And it should be the nearest gray step (128 → step 12 → idx 244).
        assert_eq!(idx, 244);
    }

    #[test]
    fn quantize_roundtrip_is_stable() {
        // Quantizing an already-on-palette color and re-quantizing its RGB must
        // be idempotent for every emitted index.
        for idx in 16u8..=255 {
            let (r, g, b) = ansi256_to_rgb(idx);
            let requant = rgb_to_ansi256(r, g, b);
            assert_eq!(requant, idx, "roundtrip drift at index {idx}");
        }
    }

    #[test]
    fn roundtrip_rgb_is_on_palette() {
        // An arbitrary color quantized then expanded must equal expanding the
        // index it maps to (the two-step path the window backend takes).
        for &(r, g, b) in &[(200u8, 30u8, 90u8), (12, 200, 240), (77, 77, 77)] {
            let idx = rgb_to_ansi256(r, g, b);
            let rgb = ansi256_to_rgb(idx);
            assert_eq!(rgb_to_ansi256(rgb.0, rgb.1, rgb.2), idx);
        }
    }

    #[test]
    fn dark_tinted_colors_keep_hue_not_gray() {
        // #1A2A40 — a dark, clearly blue-tinted navy. Chroma = 0x40-0x1A = 38,
        // well above NEUTRAL_CHROMA_THRESHOLD, so it must land in the cube
        // (16..=231), never the flat grayscale ramp (232..=255), even though the
        // ramp is numerically closer in raw RGB distance for a color this dark.
        let idx = rgb_to_ansi256(0x1A, 0x2A, 0x40);
        assert!(
            (16..=231).contains(&idx),
            "dark navy must quantize into the cube (kept hue), got gray idx {idx}"
        );
        assert_eq!(idx, 17, "exact nearest-cube index for #1A2A40");

        // #402418 — a dark, clearly red/brown-tinted color. Chroma = 0x40-0x18 = 40.
        let idx = rgb_to_ansi256(0x40, 0x24, 0x18);
        assert!(
            (16..=231).contains(&idx),
            "dark brown must quantize into the cube (kept hue), got gray idx {idx}"
        );
        assert_eq!(idx, 52, "exact nearest-cube index for #402418");
    }

    #[test]
    fn exact_gray_still_lands_on_the_ramp() {
        // #808080 (chroma 0, perfectly neutral) must still resolve to the
        // grayscale ramp — the chroma gate must not defeat genuinely neutral
        // colors, only tinted ones. Same anchor as `mid_gray_lands_in_grayscale_ramp`.
        let idx = rgb_to_ansi256(0x80, 0x80, 0x80);
        assert!(
            (232..=255).contains(&idx),
            "exact gray must still pick the grayscale ramp, got {idx}"
        );
        assert_eq!(idx, 244);
    }

    #[test]
    fn near_neutral_low_chroma_still_uses_ramp() {
        // A barely-off-neutral color (chroma = 4, well under the threshold)
        // should still be allowed to compete for the ramp — the gate is meant to
        // exclude actual tints, not any non-zero channel difference.
        let idx = rgb_to_ansi256(130, 128, 126);
        assert!(
            (232..=255).contains(&idx),
            "near-neutral chroma=4 color should still be eligible for the ramp, got {idx}"
        );
    }

    // --- Windows auto-detect: pure decision function, no live console needed ---

    #[test]
    fn decide_auto_explicit_colorterm_wins_over_everything() {
        // An explicit COLORTERM classification wins regardless of platform or
        // probe result, on every platform.
        assert_eq!(
            ColorMode::decide_auto(false, Some("truecolor"), false),
            ColorMode::Truecolor
        );
        assert_eq!(
            ColorMode::decide_auto(true, Some("truecolor"), false),
            ColorMode::Truecolor
        );
        assert_eq!(
            ColorMode::decide_auto(true, Some("256color"), true),
            ColorMode::Ansi256,
            "an explicit non-truecolor COLORTERM overrides even a successful probe"
        );
    }

    #[test]
    fn decide_auto_windows_trusts_a_successful_vt_probe() {
        // No COLORTERM, Windows, probe succeeded: this is the fix — Windows
        // consoles that support VT processing get truecolor even with no
        // COLORTERM set (which Windows consoles never set).
        assert_eq!(
            ColorMode::decide_auto(true, None, true),
            ColorMode::Truecolor
        );
    }

    #[test]
    fn decide_auto_windows_downgrades_on_failed_probe() {
        // No COLORTERM, Windows, probe failed (old conhost predating VT
        // support): downgrade to 256, same as the pre-fix behavior for that case.
        assert_eq!(
            ColorMode::decide_auto(true, None, false),
            ColorMode::Ansi256
        );
    }

    #[test]
    fn decide_auto_non_windows_ignores_probe_result() {
        // Non-Windows platforms never run the probe; a stray `true` here must
        // not leak through and must not change non-Windows behavior at all —
        // no COLORTERM still downgrades to 256 exactly as before this fix.
        assert_eq!(
            ColorMode::decide_auto(false, None, true),
            ColorMode::Ansi256
        );
        assert_eq!(
            ColorMode::decide_auto(false, None, false),
            ColorMode::Ansi256
        );
    }

    #[test]
    fn explicit_flag_overrides_env() {
        // Even with a non-truecolor COLORTERM, an explicit Truecolor flag wins.
        assert_eq!(
            ColorMode::resolve(Some(ColorMode::Truecolor)),
            ColorMode::Truecolor
        );
        assert_eq!(
            ColorMode::resolve(Some(ColorMode::Ansi256)),
            ColorMode::Ansi256
        );
    }

    #[test]
    fn colorterm_classification() {
        assert_eq!(ColorMode::from_colorterm("truecolor"), ColorMode::Truecolor);
        assert_eq!(ColorMode::from_colorterm("24bit"), ColorMode::Truecolor);
        assert_eq!(ColorMode::from_colorterm("TrueColor"), ColorMode::Truecolor);
        assert_eq!(ColorMode::from_colorterm(""), ColorMode::Ansi256);
        assert_eq!(ColorMode::from_colorterm("256color"), ColorMode::Ansi256);
    }

    #[test]
    fn flag_parsing() {
        assert_eq!(
            ColorMode::parse_flag("truecolor"),
            Some(ColorMode::Truecolor)
        );
        assert_eq!(ColorMode::parse_flag("256"), Some(ColorMode::Ansi256));
        assert_eq!(ColorMode::parse_flag("ANSI256"), Some(ColorMode::Ansi256));
        assert_eq!(ColorMode::parse_flag("nonsense"), None);
    }
}
