//! [`RenderConfig`] — per-binary startup knobs for the `re_present` layer.
//!
//! Binaries insert this resource before adding [`crate::PresentPlugin`]. The
//! plugin reads it in its `Startup` systems to:
//!
//! * Optionally hide the OS mouse pointer (system arrow) via
//!   [`crate::OsCursorSimpleHide`] — independent of mouse-look grab, and fully
//!   restored on every exit path including panics.
//! * Optionally fix the render grid to a given `cols × rows` and ask the
//!   terminal emulator to resize its window to match (best-effort; many
//!   terminals ignore `crossterm::terminal::SetSize`).
//!
//! If the resource is absent, [`RenderConfig::default()`] is used:
//! * `hide_os_cursor = false`  — OS pointer stays visible.
//! * `fixed_size = None`       — grid auto-sizes from the terminal's reported dimensions.
//!
//! ## Environment variables and CLI flags
//!
//! Both binaries (`re_game` and `ts_editor`) parse these before inserting
//! the resource so that environment and CLI override the per-binary default:
//!
//! | Source | Variable / flag | Meaning |
//! |--------|----------------|---------|
//! | env | `TS_HIDE_CURSOR=1` / `true` | Hide OS pointer. `0`/`false` → show. |
//! | env | `TS_COLS=120`, `TS_ROWS=40` | Fix grid width / height. |
//! | CLI | `--size <cols>x<rows>` | Fix both at once (e.g. `--size 120x40`). |
//!
//! Precedence for cursor: CLI is not wired (a flag would be Feature-B scope),
//! so env → binary default. Precedence for size: CLI flag > env > binary default.
//!
//! ## `--size` parsing
//!
//! [`parse_size_arg`] is a pure function (testable without a terminal) that
//! accepts `"<cols>x<rows>"` (e.g. `"120x40"`) and returns
//! `Some((cols, rows))` or `None` on any parse error. Used by both binaries.
//!
//! ## Cursor hide default: GAME on, EDITOR off
//!
//! The game hides the OS pointer so the system arrow doesn't overlay the 3D
//! view. The editor keeps it visible because the user clicks palette buttons.
//! Both respect the `TS_HIDE_CURSOR` env override.
//!
//! ## Mouse-look system gating: `is_editor`
//!
//! `PresentPlugin` registers `mouse_look_sys` with a run-condition that checks
//! `!RenderConfig.is_editor`. In the editor (`is_editor = true`) the system is
//! completely suppressed so the editor's own preview-look system has sole
//! ownership of the OS cursor grab. In the game (`is_editor = false`, the
//! default) behaviour is unchanged.

use bevy::ecs::resource::Resource;

/// Per-binary startup configuration for the `re_present` layer.
///
/// Insert before [`crate::PresentPlugin`]:
/// ```ignore
/// app.insert_resource(RenderConfig { hide_os_cursor: true, is_editor: false, fixed_size: None });
/// app.add_plugins(PresentPlugin);
/// ```
///
/// If not inserted, [`Default`] is used (hide=false, is_editor=false, auto-size).
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RenderConfig {
    /// Hide the OS system mouse pointer (the arrow / hand cursor) while the
    /// app is running. Restored on every exit path including panics.
    ///
    /// Default: `false` (pointer visible; the editor always needs it).
    ///
    /// Override: `TS_HIDE_CURSOR=1` / `0` in the environment.
    pub hide_os_cursor: bool,

    /// Set to `true` when this binary is the level editor (`ts_editor`).
    ///
    /// When `true`, `PresentPlugin`'s `mouse_look_sys` is completely suppressed
    /// via a run-condition so the editor's own preview-look system has sole
    /// ownership of the OS cursor grab. Without this gate, `mouse_look_sys`
    /// hits its no-camera early-return (the editor has no `CameraPose` entity)
    /// and calls `os_grab.deactivate()` every frame — thrashing the grab the
    /// editor relies on for smooth pixel-look.
    ///
    /// Default: `false` (game behavior; `mouse_look_sys` runs normally).
    ///
    /// Set via [`RenderConfig::for_editor`]; game uses [`RenderConfig::for_game`].
    pub is_editor: bool,

    /// Fix the render grid to exactly these `(cols, rows)` and attempt to
    /// resize the terminal window to match at startup (best-effort via
    /// `crossterm::terminal::SetSize`; many terminals ignore it silently).
    ///
    /// When `Some`: `resize_sys` ignores subsequent terminal-resize events so
    /// the framebuffer stays at the declared size (the present layer already
    /// clamps drawing to `min(fb, area)`).
    ///
    /// When `None` (default): the framebuffer auto-sizes from the terminal's
    /// reported dimensions on every resize event — today's exact behaviour.
    ///
    /// Override: `TS_COLS` + `TS_ROWS` env vars (both required for env
    /// override), or CLI `--size <cols>x<rows>`. CLI takes precedence over env.
    pub fixed_size: Option<(usize, usize)>,
}

impl RenderConfig {
    /// Resolve the final config for the **game** binary:
    /// * cursor hide DEFAULT = `true` (game hides the system arrow by default).
    /// * `is_editor = false` — `mouse_look_sys` runs normally.
    /// * size default = `None` (auto-size from terminal).
    ///
    /// Env + CLI overrides are applied on top; the caller passes the already-
    /// parsed `cli_size` from [`parse_size_arg`].
    pub fn for_game(cli_size: Option<(usize, usize)>) -> Self {
        let hide_os_cursor = parse_hide_cursor_env().unwrap_or(true);
        let fixed_size = cli_size
            .or_else(parse_size_env)
            .map(|(c, r)| (c.max(1), r.max(1)));
        RenderConfig {
            hide_os_cursor,
            is_editor: false,
            fixed_size,
        }
    }

    /// Resolve the final config for the **editor** binary:
    /// * cursor hide DEFAULT = `false` (editor keeps the pointer visible for clicking).
    /// * `is_editor = true` — suppresses `mouse_look_sys` so the editor's own
    ///   preview-look system has sole ownership of the OS cursor grab.
    /// * size default = `None` (auto-size to the terminal so the editor fills the
    ///   screen and tracks resizes — the responsive `layout::compute` does the rest).
    ///
    /// Env + CLI overrides are applied on top: a CLI `--size` or `TS_COLS`/`TS_ROWS`
    /// still pins the grid to a fixed size; without them the editor auto-sizes.
    pub fn for_editor(cli_size: Option<(usize, usize)>) -> Self {
        let hide_os_cursor = parse_hide_cursor_env().unwrap_or(false);
        let fixed_size = cli_size
            .or_else(parse_size_env)
            .map(|(c, r)| (c.max(1), r.max(1)));
        RenderConfig {
            hide_os_cursor,
            is_editor: true,
            fixed_size,
        }
    }
}

// ─── Env-var parsers (pure, testable) ────────────────────────────────────────

/// Parse `TS_HIDE_CURSOR` from the environment.
/// Accepts `"1"` / `"true"` / `"yes"` (case-insensitive) → `Some(true)`;
/// `"0"` / `"false"` / `"no"` → `Some(false)`; absent or unrecognised → `None`.
pub fn parse_hide_cursor_env() -> Option<bool> {
    let v = std::env::var("TS_HIDE_CURSOR").ok()?;
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Some(true),
        "0" | "false" | "no" => Some(false),
        _ => None,
    }
}

/// Parse `TS_COLS` and `TS_ROWS` from the environment. Both must be present
/// and parseable as `usize`; if either is missing or invalid, returns `None`.
pub fn parse_size_env() -> Option<(usize, usize)> {
    let cols: usize = std::env::var("TS_COLS").ok()?.trim().parse().ok()?;
    let rows: usize = std::env::var("TS_ROWS").ok()?.trim().parse().ok()?;
    Some((cols, rows))
}

/// Parse a `--size <cols>x<rows>` argument from a CLI args slice.
///
/// Accepts both `--size=120x40` and `--size 120x40` forms. Returns `None` if
/// the flag is absent, the value is missing, or it doesn't parse cleanly.
/// This is a **pure function** — testable without a live terminal.
///
/// ```
/// # use re_present::render_config::parse_size_arg;
/// let args: Vec<String> = ["--size", "120x40"].iter().map(|s| s.to_string()).collect();
/// assert_eq!(parse_size_arg(&args), Some((120, 40)));
///
/// let args2: Vec<String> = ["--size=80x24"].iter().map(|s| s.to_string()).collect();
/// assert_eq!(parse_size_arg(&args2), Some((80, 24)));
///
/// let empty: Vec<String> = Vec::new();
/// assert_eq!(parse_size_arg(&empty), None);
/// ```
pub fn parse_size_arg(args: &[String]) -> Option<(usize, usize)> {
    for (i, a) in args.iter().enumerate() {
        if let Some(v) = a.strip_prefix("--size=") {
            return parse_size_value(v);
        }
        if a == "--size" {
            return args.get(i + 1).and_then(|v| parse_size_value(v));
        }
    }
    None
}

/// Parse a `"<cols>x<rows>"` string (e.g. `"120x40"`). Returns `None` on any
/// format error. Pure — no IO.
///
/// ```
/// # use re_present::render_config::parse_size_value;
/// assert_eq!(parse_size_value("120x40"), Some((120, 40)));
/// assert_eq!(parse_size_value("80x24"),  Some((80, 24)));
/// assert_eq!(parse_size_value("120"),    None);          // missing 'x'
/// assert_eq!(parse_size_value("axb"),    None);          // not numbers
/// assert_eq!(parse_size_value("0x40"),   Some((0, 40))); // caller clamps to >=1
/// ```
pub fn parse_size_value(s: &str) -> Option<(usize, usize)> {
    let (l, r) = s.split_once('x')?;
    let cols: usize = l.trim().parse().ok()?;
    let rows: usize = r.trim().parse().ok()?;
    Some((cols, rows))
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_size_value_happy_paths() {
        assert_eq!(parse_size_value("120x40"), Some((120, 40)));
        assert_eq!(parse_size_value("80x24"), Some((80, 24)));
        assert_eq!(parse_size_value("1x1"), Some((1, 1)));
        // Leading/trailing spaces in each component.
        assert_eq!(parse_size_value(" 100 x 30 "), Some((100, 30)));
    }

    #[test]
    fn parse_size_value_bad_input() {
        assert_eq!(parse_size_value("120"), None); // no 'x'
        assert_eq!(parse_size_value("axb"), None); // not numbers
        assert_eq!(parse_size_value("120x"), None); // missing rows
        assert_eq!(parse_size_value("x40"), None); // missing cols
        assert_eq!(parse_size_value(""), None); // empty
        assert_eq!(parse_size_value("-1x40"), None); // negative (not usize)
    }

    #[test]
    fn parse_size_arg_long_form() {
        let args: Vec<String> = ["--size", "120x40"].iter().map(|s| s.to_string()).collect();
        assert_eq!(parse_size_arg(&args), Some((120, 40)));
    }

    #[test]
    fn parse_size_arg_equals_form() {
        let args: Vec<String> = ["--size=80x24"].iter().map(|s| s.to_string()).collect();
        assert_eq!(parse_size_arg(&args), Some((80, 24)));
    }

    #[test]
    fn parse_size_arg_absent() {
        let args: Vec<String> = vec!["--color=truecolor".to_string()];
        assert_eq!(parse_size_arg(&args), None);
    }

    #[test]
    fn parse_size_arg_after_other_flags() {
        let args: Vec<String> = ["--color", "truecolor", "--size", "100x30"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(parse_size_arg(&args), Some((100, 30)));
    }

    #[test]
    fn parse_size_arg_missing_value() {
        // `--size` at the end with no following arg → None.
        let args: Vec<String> = ["--size"].iter().map(|s| s.to_string()).collect();
        assert_eq!(parse_size_arg(&args), None);
    }

    #[test]
    fn parse_size_arg_bad_value() {
        let args: Vec<String> = ["--size=invalid"].iter().map(|s| s.to_string()).collect();
        assert_eq!(parse_size_arg(&args), None);
    }

    #[test]
    fn render_config_default_is_no_hide_no_size() {
        let cfg = RenderConfig::default();
        assert!(!cfg.hide_os_cursor);
        assert!(!cfg.is_editor);
        assert_eq!(cfg.fixed_size, None);
    }

    #[test]
    fn is_editor_flag_for_editor_true() {
        let cfg = RenderConfig::for_editor(None);
        assert!(cfg.is_editor, "for_editor must set is_editor = true");
    }

    #[test]
    fn is_editor_flag_for_game_false() {
        let cfg = RenderConfig::for_game(None);
        assert!(!cfg.is_editor, "for_game must set is_editor = false");
    }

    #[test]
    fn is_editor_flag_default_false() {
        let cfg = RenderConfig::default();
        assert!(
            !cfg.is_editor,
            "Default must leave is_editor = false (game behavior)"
        );
    }

    #[test]
    fn render_config_for_editor_default_auto_size() {
        // With no CLI size and no env override, the editor now AUTO-SIZES (fixed_size
        // = None) so it fills the terminal and tracks resizes. Skip if env pins a size.
        if std::env::var("TS_COLS").is_ok() || std::env::var("TS_ROWS").is_ok() {
            return; // env already set; skip to avoid false failure
        }
        let cfg = RenderConfig::for_editor(None);
        assert_eq!(cfg.fixed_size, None);
    }

    #[test]
    fn render_config_for_editor_cli_overrides_default() {
        let cfg = RenderConfig::for_editor(Some((200, 50)));
        assert_eq!(cfg.fixed_size, Some((200, 50)));
    }

    #[test]
    fn render_config_for_game_default_no_size() {
        if std::env::var("TS_COLS").is_ok() || std::env::var("TS_ROWS").is_ok() {
            return;
        }
        let cfg = RenderConfig::for_game(None);
        assert_eq!(cfg.fixed_size, None);
    }

    #[test]
    fn render_config_for_game_cli_size() {
        let cfg = RenderConfig::for_game(Some((160, 48)));
        assert_eq!(cfg.fixed_size, Some((160, 48)));
    }
}
