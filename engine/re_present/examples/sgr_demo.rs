//! `sgr_demo` — write a cat-able ANSI file exercising every per-cell text-style
//! channel the present layer can emit, so the new attributes can be SEEN.
//!
//! Run: `cargo run -p re_present --example sgr_demo`
//! View: `cat docs/sgr_demo.ansi` in a truecolor terminal (Windows Terminal,
//! xterm, kitty, wezterm, …).
//!
//! Each demo line builds a [`re_core::CellStyle`] — the SAME additive style type
//! the renderer stamps onto a [`re_core::Cell`] — and renders it as the raw SGR
//! sequence the terminal backend's ratatui mapping produces:
//!
//!   * `1` bold, `2` dim, `3` italic, `4` underline, `7` reverse, `9` strike
//!   * `58;2;r;g;b` — the COLORED UNDERLINE (the independent 3rd color channel)
//!   * `38;2;r;g;b` / `48;2;r;g;b` — truecolor fg / bg (the existing two channels)
//!
//! A terminal lacking any one of these attributes simply ignores its code
//! (graceful degradation) — exactly how the live backend behaves.

use std::fmt::Write as _;
use std::path::PathBuf;

use re_core::CellStyle;

/// Build the SGR parameter list for a cell with packed `fg`/`bg` and a
/// [`CellStyle`] — the raw-byte equivalent of the terminal backend's ratatui
/// `Modifier` + `underline_color` mapping. Returns just the numeric params
/// (joined by `;`), ready to wrap in `ESC[ … m`.
fn sgr_params(fg: u32, bg: u32, style: CellStyle) -> String {
    let mut parts: Vec<String> = Vec::new();
    if style.is_bold() {
        parts.push("1".into());
    }
    if style.is_dim() {
        parts.push("2".into());
    }
    if style.is_italic() {
        parts.push("3".into());
    }
    // A colored underline implies the underline attribute (matches
    // `CellStyle::with_underline_color`).
    if style.is_underline() || style.has_underline_color() {
        parts.push("4".into());
    }
    if style.is_reverse() {
        parts.push("7".into());
    }
    if style.is_strike() {
        parts.push("9".into());
    }
    // Existing two channels: truecolor fg + bg.
    let (fr, fg_, fb) = (fg >> 16 & 0xFF, fg >> 8 & 0xFF, fg & 0xFF);
    let (br, bg_, bb) = (bg >> 16 & 0xFF, bg >> 8 & 0xFF, bg & 0xFF);
    parts.push(format!("38;2;{fr};{fg_};{fb}"));
    parts.push(format!("48;2;{br};{bg_};{bb}"));
    // The THIRD channel: colored underline (58;2;r;g;b).
    if let Some(uc) = style.underline_color() {
        let (ur, ug, ub) = (uc >> 16 & 0xFF, uc >> 8 & 0xFF, uc & 0xFF);
        parts.push(format!("58;2;{ur};{ug};{ub}"));
    }
    parts.join(";")
}

/// Wrap `text` in the SGR sequence for `(fg, bg, style)`, resetting after.
fn styled(fg: u32, bg: u32, style: CellStyle, text: &str) -> String {
    format!("\x1b[{}m{text}\x1b[0m", sgr_params(fg, bg, style))
}

fn main() {
    let fg = 0x00E6_F0FF; // near-white
    let bg = 0x000A_0E18; // deep navy

    let rows: &[(&str, CellStyle)] = &[
        ("plain      ", CellStyle::NONE),
        ("bold       ", CellStyle::NONE.with_bold()),
        ("dim        ", CellStyle::NONE.with_flags(CellStyle::DIM)),
        ("italic     ", CellStyle::NONE.with_flags(CellStyle::ITALIC)),
        ("underline  ", CellStyle::NONE.with_underline()),
        (
            "reverse    ",
            CellStyle::NONE.with_flags(CellStyle::REVERSE),
        ),
        ("strike     ", CellStyle::NONE.with_flags(CellStyle::STRIKE)),
        (
            "colored-underline (3rd channel, magenta glow)",
            CellStyle::NONE.with_underline_color(0x00FF_30C0),
        ),
        (
            "bold + colored-underline (the shipped ridge+glow accent)",
            CellStyle::NONE
                .with_bold()
                .with_underline_color(0x0040_FF90),
        ),
        (
            "everything at once",
            CellStyle::NONE
                .with_flags(
                    CellStyle::BOLD
                        | CellStyle::DIM
                        | CellStyle::ITALIC
                        | CellStyle::REVERSE
                        | CellStyle::STRIKE,
                )
                .with_underline_color(0x0030_C0FF),
        ),
    ];

    let mut out = String::new();
    let _ = writeln!(
        out,
        "TERMINAL STATE per-cell SGR demo — bold/dim/italic/underline/reverse/strike \
         + colored underline (3rd channel)\n"
    );
    for (label, style) in rows {
        let _ = writeln!(out, "  {}", styled(fg, bg, *style, &format!(" {label} ")));
    }
    out.push('\n');

    // Resolve docs/sgr_demo.ansi relative to the crate, then the workspace root.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let docs = manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("docs"))
        .unwrap_or_else(|| PathBuf::from("docs"));
    let path = docs.join("sgr_demo.ansi");
    std::fs::write(&path, out.as_bytes()).expect("write docs/sgr_demo.ansi");
    println!("wrote {}", path.display());

    // Echo one styled cell's raw escape sequence so it's visible in the run log.
    let sample = CellStyle::NONE
        .with_bold()
        .with_underline_color(0x0040_FF90);
    let esc = sgr_params(fg, bg, sample);
    println!("sample 'bold + colored-underline' cell SGR: ESC[{esc}m");
}
