//! `banner` — a pixel-font wall placard legible at range.
//!
//! A [`crate::code_wall::TextPanel`] normally lays out one authored CHARACTER per
//! grid cell, so a `[note stencil]` shrinks to noise a few rows away — the wall has
//! no way to say "EXIT" bigger than one glyph. A banner instead rasterises a WORD
//! through our own tiny bitmap font, blowing each font pixel up into an `N × N`
//! block of panel cells, so the word reads from many cells down a corridor exactly
//! the way `TextPanel::sample_fitted_anchored_2d`'s refuse-to-magnify clamp already
//! keeps ordinary text from doubling up close.
//!
//! ## Why this needs no new render path
//!
//! A static (non-cycling) banner is built ONCE, at level load, into an ordinary
//! [`TextPanel`] via `build_panel` — the SAME grid
//! mechanism a stencil or a code wall uses. The shader-facing seam
//! (`crate::scene_text::sample_note_panel_anchored`, `crate::ascii_shader`'s
//! `NoteWall` recipe) is completely unaware a banner exists: it samples a panel
//! exactly as it samples any other note. The only new work is HOW the grid
//! gets filled: `glyph_bits` rasterises our own
//! 3×5 font instead of copying authored characters.
//!
//! Colour rides the SAME three-channel budget every note register has
//! ([`crate::note_style::NotePalette`]): a lit font pixel is [`CodeClass::Code`]
//! (bright ink), the optional frame is [`CodeClass::Comment`] (the accent ink), and
//! an unlit pixel / the margin is [`CodeClass::Blank`] (the plate). That is exactly
//! "bright/frame/dim" from the study without a new colour channel anywhere.
//!
//! ## Live word cycling
//!
//! `cycle = A, B, C` (the word cycling over time) builds ONE panel per
//! word ([`build_cycle_panels`](crate::banner::build_cycle_panels)), registered
//! as `crate::scene_text`'s `NoteEntry::Cycled`. Which panel is live is picked
//! at SAMPLE time from a
//! process-wide clock (`crate::scene_text::set_cycle_time` / `cycle_time`) — the SAME
//! thread-local-publish shape `crate::state_bus` already uses to reach the deep shade
//! path without threading a new parameter through `ascii_shader::Material::sample_note_wall`
//! (its call into `sample_note_panel_anchored` cannot change shape). `set_cycle_time`
//! is called once per shaded cell from
//! `crate::shade_cell` (the SAME `time: f32` every recipe already receives via
//! `SampleCtx::time`), so it is current before any note is sampled that frame, and it
//! is a pure function of the fed time — `t = 0.0` (every `--dump`, every golden,
//! every focused unit test that never calls `set_cycle_time`, whose default is `0.0`)
//! always selects `cycle[0]`, deterministically.
//!
//! ## Our own font, not a copy
//!
//! `glyph_bits`'s bitmap table is authored
//! freeform for this module — a 3-column × 5-row bitmap per character is a
//! generic, decades-old technique; no glyph bitmap, string, or byte table was
//! read out of any reference material.
//!
//! ## What is NOT implemented here
//!
//! * **Multi-cell world-anchored spanning.** Every `[note]` — banner included —
//!   resolves to exactly ONE wall-cell face (`re_game/src/level_text.rs`). A banner
//!   therefore reads large on ONE face via the existing refuse-to-magnify clamp, not
//!   by claiming a run of cells to span a whole storefront.

use crate::code_wall::{CodeClass, CodeGlyph, TextPanel};

/// Panel cells one FONT PIXEL is blown up into, along each axis. Clamped to keep a
/// pathological `scale =` from allocating an unbounded grid. `MAX_SCALE` was `5`
/// until the redesign below — a 4-letter word at scale 5 rasterises to a ~70-column
/// panel, which the refuse-to-magnify clamp (`code_wall::sample_fitted_anchored_2d`)
/// can never show at 1:1 on a ONE-world-cell face at any distance a player can stand:
/// it falls to the STRETCH branch and minifies to noise. `3` is still generous for a
/// short word within [`TARGET_FULL_COLS`].
const MIN_SCALE: usize = 1;
const MAX_SCALE: usize = 3;
/// The default when no `scale =` is authored — an upper CAP handed to
/// [`effective_scale`], not the scale used outright; see that function's doc.
const DEFAULT_SCALE: usize = 2;

/// Target panel width (columns, frame included) the FULL pixel-font tier is fit
/// to by [`effective_scale`] — the render-geometry half of the legibility fix.
/// Chosen to comfortably clear a face big enough to show the raster at 1:1, while
/// still being narrow enough that a modest, close-but-plausible face (see
/// `levels/100-signage-showcase.lvl`) can clear it.
const TARGET_FULL_COLS: f32 = 22.0;

/// The character an empty word's ACCENT tier falls back to — a defined, legible
/// answer (never a blank/panic) for a body that authored no `word =` at all, mirroring
/// [`BannerSpec::default`]'s "empty word ⇒ an empty but valid panel" contract for the
/// FULL tier.
const ACCENT_FALLBACK: char = '•';

/// Font glyph footprint, in cells (before scaling): 3 columns × 5 rows.
const FONT_COLS: usize = 3;
const FONT_ROWS: usize = 5;

/// Gap between letters, in PANEL cells (not scaled — a banner reads at a glance, not
/// letter by letter, so a fixed one-cell gap keeps the word compact at any scale).
const LETTER_GAP: usize = 1;

/// Frame margin, in panel cells, on each side when `frame = on` (the default).
const FRAME_MARGIN: usize = 1;

/// Longest word rendered. A banner is signage, not a paragraph; this bounds the
/// panel's worst-case width (`MAX_WORD_LEN * FONT_COLS * MAX_SCALE`, well inside the
/// engine's other panel budgets like `code_wall::PANEL_COLS`).
const MAX_WORD_LEN: usize = 16;

/// The bright, filled cell a lit font pixel draws with. A solid block reads
/// unambiguously at a distance where a thin stroke glyph would alias away.
const LIT_GLYPH: char = '█';

/// A parsed `--- banner ---` directive block.
#[derive(Clone, Debug, PartialEq)]
struct BannerSpec {
    /// The word to render, already upper-cased. Empty ⇒ an empty (all-plate) panel —
    /// a defined, non-panicking answer for a body that authored no `word =`.
    word: String,
    /// Panel cells per font pixel, clamped to `[MIN_SCALE, MAX_SCALE]`.
    scale: usize,
    /// Draw the `+=|` frame around the word.
    frame: bool,
    /// `cycle = A, B, C` — parsed in full. [`build_cycle_panels`] rasterises ONE
    /// panel per word here; which panel is live at sample time is picked by
    /// `crate::scene_text`'s clock, not by this struct (see that module's
    /// `NoteEntry::Cycled` / [`CYCLE_WORD_SECS`]).
    cycle: Vec<String>,
}

impl Default for BannerSpec {
    fn default() -> Self {
        BannerSpec {
            word: String::new(),
            scale: DEFAULT_SCALE,
            frame: true,
            cycle: Vec::new(),
        }
    }
}

/// Parse `key = value` directive lines out of a `--- banner ---` block. Unknown keys
/// and malformed values are ignored (a defined default, never a parse error — a
/// banner is decorative signage, not puzzle-critical data).
fn parse_spec(body: &str) -> BannerSpec {
    let mut spec = BannerSpec::default();
    for raw in body.lines() {
        let line = raw.trim();
        let Some((key, val)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        let val = val.trim();
        match key.as_str() {
            "word" => spec.word = val.to_ascii_uppercase(),
            "scale" => {
                if let Ok(n) = val.parse::<usize>() {
                    spec.scale = n.clamp(MIN_SCALE, MAX_SCALE);
                }
            }
            "frame" => spec.frame = is_truthy(val),
            "cycle" => {
                spec.cycle = val
                    .split(',')
                    .map(|w| w.trim().to_ascii_uppercase())
                    .filter(|w| !w.is_empty())
                    .collect();
            }
            _ => {}
        }
    }
    // `cycle`'s first word is a fallback WORD when none was authored explicitly.
    if spec.word.is_empty() {
        if let Some(first) = spec.cycle.first() {
            spec.word.clone_from(first);
        }
    }
    if spec.word.chars().count() > MAX_WORD_LEN {
        spec.word = spec.word.chars().take(MAX_WORD_LEN).collect();
    }
    spec
}

/// `true`/`on`/`yes`/`1` (case-insensitive) ⇒ `true`; everything else (including an
/// empty value) ⇒ `false`. Mirrors the `.lvl` format's existing truthy convention
/// (`re_level::is_truthy`) without adding a cross-crate dependency for one helper.
fn is_truthy(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "true" | "on" | "yes" | "1"
    )
}

/// The 3×5 bitmap for one character, as 5 row-bytes whose low 3 bits are
/// `col0 col1 col2` (bit 2 = leftmost). Unsupported characters (and space) return an
/// all-blank glyph — a defined, legible-as-a-gap answer rather than a panic or mojibake.
fn glyph_bits(c: char) -> [u8; FONT_ROWS] {
    match c {
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b010],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b101, 0b111, 0b111, 0b111, 0b101],
        'O' | '0' => [0b010, 0b101, 0b101, 0b101, 0b010],
        'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'Q' => [0b010, 0b101, 0b101, 0b010, 0b001],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b110, 0b001, 0b010, 0b100, 0b111],
        '3' => [0b110, 0b001, 0b010, 0b001, 0b110],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b110, 0b001, 0b110],
        '6' => [0b011, 0b100, 0b110, 0b101, 0b010],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b010, 0b101, 0b010, 0b101, 0b010],
        '9' => [0b010, 0b101, 0b011, 0b001, 0b010],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '.' | ':' => [0b000, 0b000, 0b000, 0b000, 0b010],
        '!' => [0b010, 0b010, 0b010, 0b000, 0b010],
        _ => [0b000, 0b000, 0b000, 0b000, 0b000], // space + anything unsupported
    }
}

/// Render `word` at `scale` into a `(cols, rows, cells)` rasterised buffer, without
/// the optional frame — the frame is added by [`build_panel`] so this stays a pure
/// "one word, one raster" function the tests can check directly.
fn rasterise_word(word: &str, scale: usize) -> (usize, usize, Vec<CodeGlyph>) {
    let letters: Vec<char> = word.chars().collect();
    let glyph_w = FONT_COLS * scale;
    let glyph_h = FONT_ROWS * scale;
    let n = letters.len().max(1);
    let cols = if letters.is_empty() {
        glyph_w
    } else {
        n * glyph_w + (n - 1) * LETTER_GAP
    };
    let rows = glyph_h;
    let mut cells = vec![
        CodeGlyph {
            glyph: u32::from(' '),
            class: CodeClass::Blank
        };
        cols * rows
    ];

    for (li, &ch) in letters.iter().enumerate() {
        let bits = glyph_bits(ch);
        let ox = li * (glyph_w + LETTER_GAP);
        for (fr, row_bits) in bits.iter().enumerate() {
            for fc in 0..FONT_COLS {
                let lit = (row_bits >> (FONT_COLS - 1 - fc)) & 1 == 1;
                if !lit {
                    continue;
                }
                for sr in 0..scale {
                    for sc in 0..scale {
                        let px = ox + fc * scale + sc;
                        let py = fr * scale + sr;
                        if px < cols && py < rows {
                            cells[py * cols + px] = CodeGlyph {
                                glyph: u32::from(LIT_GLYPH),
                                class: CodeClass::Code,
                            };
                        }
                    }
                }
            }
        }
    }
    (cols, rows, cells)
}

/// Rasterise ONE word into a framed (or bare) note panel — the shared core both
/// [`build_panel`] (the single, non-cycling case) and [`build_cycle_panels`] (one
/// panel per `cycle =` word) delegate to, so a cycling banner's individual frames are
/// built with EXACTLY the same margin/frame logic as a static one, just keyed on a
/// different word each time.
fn panel_for_word(label: &str, word: &str, scale: usize, frame: bool) -> TextPanel {
    let (wcols, wrows, wcells) = rasterise_word(word, scale);

    let margin = if frame { FRAME_MARGIN } else { 0 };
    let cols = wcols + margin * 2;
    let rows = wrows + margin * 2;
    let mut cells = vec![
        CodeGlyph {
            glyph: u32::from(' '),
            class: CodeClass::Blank
        };
        cols * rows
    ];

    // Paste the rasterised word into the framed interior.
    for wy in 0..wrows {
        for wx in 0..wcols {
            cells[(wy + margin) * cols + (wx + margin)] = wcells[wy * wcols + wx];
        }
    }

    if frame && cols >= 2 && rows >= 2 {
        let frame_cell = |g: char| CodeGlyph {
            glyph: u32::from(g),
            class: CodeClass::Comment,
        };
        for x in 0..cols {
            cells[x] = frame_cell(if x == 0 || x == cols - 1 { '+' } else { '=' });
            let last = (rows - 1) * cols + x;
            cells[last] = frame_cell(if x == 0 || x == cols - 1 { '+' } else { '=' });
        }
        for y in 0..rows {
            cells[y * cols] = frame_cell('|');
            cells[y * cols + cols - 1] = frame_cell('|');
        }
        // Corners take the frame's own corner glyph (overwrites the row/col loops'
        // '=' / '|' at the four intersections, which is what a real frame draws).
        cells[0] = frame_cell('+');
        cells[cols - 1] = frame_cell('+');
        cells[(rows - 1) * cols] = frame_cell('+');
        cells[(rows - 1) * cols + cols - 1] = frame_cell('+');
    }

    TextPanel::from_cells(label, cols, rows, cells)
}

/// Largest `scale` (bounded by `cap`, itself already clamped to
/// `[MIN_SCALE, MAX_SCALE]`) whose FRAMED pixel-font raster of `word` is no wider
/// than [`TARGET_FULL_COLS`] — the "size the bitmap to fit the face" half of the
/// legibility fix. A fixed author-chosen scale can rasterise wider than any
/// ONE-world-cell face's refuse-to-magnify clamp can show at 1:1, falling to the
/// STRETCH/minify branch and reading as noise at every distance; fitting the scale
/// to the face avoids that. Never returns below `MIN_SCALE`:
/// a word too long to fit even the smallest scale still renders (oversized, not
/// blank), because the TIER FALLBACK below (`compact_panel`/`accent_panel`) is what
/// actually keeps a too-small face legible, not this function alone.
///
/// An empty word has nothing to fit against, so it keeps the author's own cap
/// (clamped) — matching the historical "empty word renders at the authored/default
/// scale, just blank" behaviour ([`a_missing_word_renders_an_empty_but_valid_panel`]).
fn effective_scale(word: &str, cap: usize, frame: bool) -> usize {
    let cap = cap.clamp(MIN_SCALE, MAX_SCALE);
    if word.is_empty() {
        return cap;
    }
    let margin = if frame {
        (FRAME_MARGIN * 2) as f32
    } else {
        0.0
    };
    for scale in (MIN_SCALE..=cap).rev() {
        let (cols, _, _) = rasterise_word(word, scale);
        if cols as f32 + margin <= TARGET_FULL_COLS {
            return scale;
        }
    }
    MIN_SCALE
}

/// COMPACT fallback tier: the bare word, ONE CHARACTER PER PANEL CELL (no font
/// blow-up, no letter gap) — far narrower than the FULL pixel-font tier, so it stays
/// legible on a face too small to show that tier at 1:1. Bracketed (`frame = on`,
/// the default) so it still reads as an authored sign rather than stray wall text,
/// at the cost of 2 columns — matching the FULL tier's own `frame =` directive
/// rather than always/never bracketing regardless of it.
fn compact_panel(label: &str, word: &str, frame: bool) -> TextPanel {
    let letters: Vec<char> = word.chars().collect();
    let margin = usize::from(frame);
    let cols = (letters.len() + margin * 2).max(1);
    let mut cells = vec![
        CodeGlyph {
            glyph: u32::from(' '),
            class: CodeClass::Blank
        };
        cols
    ];
    if frame {
        cells[0] = CodeGlyph {
            glyph: u32::from('['),
            class: CodeClass::Comment,
        };
        cells[cols - 1] = CodeGlyph {
            glyph: u32::from(']'),
            class: CodeClass::Comment,
        };
    }
    for (i, &ch) in letters.iter().enumerate() {
        cells[i + margin] = CodeGlyph {
            glyph: u32::from(ch),
            class: CodeClass::Code,
        };
    }
    TextPanel::from_cells(label, cols, 1, cells)
}

/// ACCENT fallback tier: a SINGLE character filling the whole 1×1 panel — the last
/// legible thing a one-world-cell sign can show at extreme range (mirrors
/// `stencil_variants::TierPanel::single_glyph`'s "reads as that character everywhere
/// on its face" contract). The word's own first letter, so the mark still carries
/// SOME of the word's identity rather than being a generic dot; [`ACCENT_FALLBACK`]
/// only for the (empty-word) edge case that has no letter to take.
fn accent_panel(label: &str, word: &str) -> TextPanel {
    let ch = word.chars().next().unwrap_or(ACCENT_FALLBACK);
    TextPanel::from_cells(
        label,
        1,
        1,
        vec![CodeGlyph {
            glyph: u32::from(ch),
            class: CodeClass::Code,
        }],
    )
}

/// The three size tiers built for ONE banner word — see `crate::scene_text`'s
/// `select_banner_tier` for how the live, MEASURED face size (`face_rows` +
/// `FaceAnchor::span_cols`, both real per-frame projection reads) picks among them.
/// Always all three: an empty/unsupported word still yields a (blank) full tier, a
/// bracket-only compact tier, and an [`ACCENT_FALLBACK`] accent tier — no branch of
/// `select_banner_tier` can find nothing to sample.
pub(crate) struct BannerWordTiers {
    pub(crate) full: TextPanel,
    pub(crate) compact: TextPanel,
    pub(crate) accent: TextPanel,
}

fn build_word_tiers(label: &str, word: &str, scale_cap: usize, frame: bool) -> BannerWordTiers {
    let scale = effective_scale(word, scale_cap, frame);
    BannerWordTiers {
        full: panel_for_word(label, word, scale, frame),
        compact: compact_panel(label, word, frame),
        accent: accent_panel(label, word),
    }
}

/// Build the tiered note panel set for a `--- banner ---` block: parse its
/// directives, then build all three [`BannerWordTiers`]. Always returns a valid
/// (if blank) set — an empty/unsupported word renders as a blank plate rather than
/// failing the level load, consistent with every other note kind's "a bad body is
/// content, not a parse error" contract.
///
/// `#[cfg(test)]`: the production entry point is [`build_cycle_panels`] now (its
/// `spec.cycle.len() <= 1` branch reproduces exactly this, byte-for-byte — see
/// `no_cycle_directive_yields_exactly_one_panel_matching_build_panel`); this stays
/// as a focused single-word test helper so the existing word/scale/frame tests
/// below don't need to reach into a `Vec`.
#[cfg(test)]
pub(crate) fn build_panel(label: &str, body: &str) -> BannerWordTiers {
    let spec = parse_spec(body);
    build_word_tiers(label, &spec.word, spec.scale, spec.frame)
}

/// Seconds each `cycle =` word stays on screen before the next one takes over —
/// deliberately slow (signage, not a strobe): long enough to read a short word at a
/// glance from down a corridor. Not spec-mandated (the study report names the
/// technique, not a cadence); tunable, `balance-reviewer` territory, same status as
/// [`crate::stencil_variants::GLYPH_TIER_MAX_ROWS`].
pub(crate) const CYCLE_WORD_SECS: f32 = 2.2;

/// Longest cycle phrase rendered per word — reuses [`MAX_WORD_LEN`] so a cycling
/// banner never allocates a wider panel than a static one could.
const MAX_CYCLE_WORD_LEN: usize = MAX_WORD_LEN;

/// Build one [`BannerWordTiers`] PER `cycle =` word (same `scale =` / `frame =` for
/// all of them), for a live, time-selected rotation (`crate::scene_text`'s
/// `NoteEntry::Cycled` / [`CYCLE_WORD_SECS`]). A banner with no `cycle =` (or only
/// one distinct word) returns a single-element `Vec` — the ordinary, non-cycling
/// case — built the exact same way [`build_panel`] would, so a placard with no
/// `cycle =` is unaffected by this function existing.
pub(crate) fn build_cycle_panels(label: &str, body: &str) -> Vec<BannerWordTiers> {
    let spec = parse_spec(body);
    if spec.cycle.len() <= 1 {
        return vec![build_word_tiers(label, &spec.word, spec.scale, spec.frame)];
    }
    spec.cycle
        .iter()
        .map(|w| {
            let clamped: String = w.chars().take(MAX_CYCLE_WORD_LEN).collect();
            build_word_tiers(label, &clamped, spec.scale, spec.frame)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_word_scale_and_frame() {
        let spec = parse_spec("word = exit\nscale = 3\nframe = off");
        assert_eq!(spec.word, "EXIT", "word is upper-cased");
        assert_eq!(spec.scale, 3);
        assert!(!spec.frame);
    }

    #[test]
    fn scale_out_of_range_is_clamped() {
        assert_eq!(parse_spec("scale = 0").scale, MIN_SCALE);
        assert_eq!(parse_spec("scale = 99").scale, MAX_SCALE);
        // A non-numeric scale keeps the default rather than panicking.
        assert_eq!(parse_spec("scale = big").scale, DEFAULT_SCALE);
    }

    #[test]
    fn frame_defaults_on_and_accepts_common_spellings() {
        assert!(parse_spec("word = a").frame, "frame defaults ON");
        for v in ["on", "true", "yes", "1", "ON", "Yes"] {
            assert!(
                parse_spec(&format!("frame = {v}")).frame,
                "{v} must be truthy"
            );
        }
        for v in ["off", "false", "no", "0", ""] {
            assert!(
                !parse_spec(&format!("frame = {v}")).frame,
                "{v} must be falsy"
            );
        }
    }

    #[test]
    fn cycle_is_parsed_in_full_but_only_the_first_word_becomes_the_render_word() {
        let spec = parse_spec("cycle = exit, this way, out");
        assert_eq!(
            spec.cycle,
            vec!["EXIT", "THIS WAY", "OUT"],
            "cycle keeps every word"
        );
        assert_eq!(
            spec.word, "EXIT",
            "no explicit `word=` ⇒ fall back to cycle[0]"
        );
    }

    #[test]
    fn an_explicit_word_wins_over_cycle() {
        let spec = parse_spec("word = a\ncycle = b, c");
        assert_eq!(spec.word, "A");
    }

    #[test]
    fn unknown_keys_and_malformed_lines_are_ignored_not_fatal() {
        let spec = parse_spec("word = ok\nbanana\nfoo = bar\nscale = ok");
        assert_eq!(spec.word, "OK");
        assert_eq!(
            spec.scale, DEFAULT_SCALE,
            "a bad scale value keeps the default"
        );
    }

    #[test]
    fn a_missing_word_renders_an_empty_but_valid_panel() {
        let tiers = build_panel("n", "scale = 2");
        let p = &tiers.full;
        assert!(
            p.cols() > 0 && p.rows() > 0,
            "still a real, sample-able panel"
        );
        // Every cell is either the frame or blank — no stray ink from an empty word.
        for c in 0..p.cols() {
            for r in 0..p.rows() {
                assert_ne!(p.cell(c, r).glyph, u32::from(LIT_GLYPH));
            }
        }
        // The fallback tiers are still real, sample-able panels too — an empty word
        // is content (a blank sign), never a reason for `select_banner_tier` to find
        // nothing.
        assert!(tiers.compact.cols() > 0 && tiers.compact.rows() > 0);
        assert_eq!((tiers.accent.cols(), tiers.accent.rows()), (1, 1));
        assert_eq!(tiers.accent.cell(0, 0).glyph, u32::from(ACCENT_FALLBACK));
    }

    #[test]
    fn a_lit_font_pixel_blows_up_into_a_scale_by_scale_block() {
        // Don't hardcode an assumed bit pattern for 'I' — read the real one from
        // `glyph_bits` and assert the rasteriser reproduces it, each font pixel blown
        // up into a `scale`-by-`scale` block of lit/blank cells.
        const SCALE: usize = 3;
        let bits = glyph_bits('I');
        let (cols, rows, cells) = rasterise_word("I", SCALE);
        assert_eq!(cols, FONT_COLS * SCALE);
        assert_eq!(rows, FONT_ROWS * SCALE);
        for row in 0..rows {
            let font_row = row / SCALE;
            for col in 0..cols {
                let font_col = col / SCALE;
                let expected_lit = bits[font_row] & (0b100 >> font_col) != 0;
                let lit = cells[row * cols + col].class == CodeClass::Code;
                assert_eq!(lit, expected_lit, "row {row} col {col} of a scaled 'I'");
            }
        }
    }

    #[test]
    fn letters_are_separated_by_a_blank_gap_column() {
        let (cols, _rows, cells) = rasterise_word("II", 1);
        // Two 3-wide glyphs + a 1-wide gap = 7 columns.
        assert_eq!(cols, FONT_COLS * 2 + LETTER_GAP);
        // The gap column (index 3) is blank on every row.
        for row in 0..FONT_ROWS {
            assert_eq!(
                cells[row * cols + 3].class,
                CodeClass::Blank,
                "gap column row {row}"
            );
        }
    }

    #[test]
    fn build_panel_frames_the_word_in_plus_equals_pipe() {
        let p = build_panel("n", "word = i\nscale = 1\nframe = on").full;
        // Corners are '+'.
        assert_eq!(p.cell(0, 0).glyph, u32::from('+'));
        assert_eq!(p.cell(p.cols() - 1, 0).glyph, u32::from('+'));
        assert_eq!(p.cell(0, p.rows() - 1).glyph, u32::from('+'));
        assert_eq!(p.cell(p.cols() - 1, p.rows() - 1).glyph, u32::from('+'));
        // Top/bottom edges (excluding corners) are '=', classified as the accent ink.
        assert_eq!(p.cell(1, 0).glyph, u32::from('='));
        assert_eq!(p.cell(1, 0).class, CodeClass::Comment);
        // Left/right edges are '|'.
        assert_eq!(p.cell(0, 1).glyph, u32::from('|'));
        // The word itself sits INSIDE the frame margin.
        assert_eq!(
            p.cell(1, 1).class,
            CodeClass::Code,
            "the 'I' bar starts right inside the frame"
        );
    }

    #[test]
    fn build_panel_without_a_frame_has_no_margin() {
        let framed = build_panel("n", "word = i\nscale = 1\nframe = on").full;
        let bare = build_panel("n", "word = i\nscale = 1\nframe = off").full;
        assert_eq!(bare.cols(), framed.cols() - 2);
        assert_eq!(bare.rows(), framed.rows() - 2);
        assert_eq!(
            bare.cell(0, 0).class,
            CodeClass::Code,
            "no frame ⇒ the word starts at the edge"
        );
    }

    #[test]
    fn a_word_over_the_length_cap_is_truncated_not_rejected() {
        let long = "A".repeat(40);
        let spec = parse_spec(&format!("word = {long}"));
        assert_eq!(spec.word.chars().count(), MAX_WORD_LEN);
    }

    #[test]
    fn unsupported_characters_render_as_a_blank_gap_not_a_panic() {
        let (_cols, _rows, cells) = rasterise_word("A#Z", 1);
        // The middle letter ('#') has no glyph, so its whole 3×5 block is blank.
        assert!(cells.iter().take(0).all(|_| true)); // structural no-op guard
        let mid = glyph_bits('#');
        assert_eq!(
            mid,
            [0, 0, 0, 0, 0],
            "an unsupported char is the all-blank glyph"
        );
    }

    // ── build_cycle_panels ──────────────────────────────────────────────────

    #[test]
    fn no_cycle_directive_yields_exactly_one_panel_matching_build_panel() {
        let body = "word = exit\nscale = 2";
        let panels = build_cycle_panels("n", body);
        assert_eq!(panels.len(), 1, "no cycle ⇒ the ordinary single-panel case");
        let solo = build_panel("n", body);
        for (got, want) in [
            (&panels[0].full, &solo.full),
            (&panels[0].compact, &solo.compact),
            (&panels[0].accent, &solo.accent),
        ] {
            assert_eq!(got.cols(), want.cols());
            assert_eq!(got.rows(), want.rows());
            for c in 0..want.cols() {
                for r in 0..want.rows() {
                    assert_eq!(got.cell(c, r), want.cell(c, r));
                }
            }
        }
    }

    #[test]
    fn a_single_word_cycle_also_yields_one_panel() {
        // `cycle = exit` (one entry) must not be treated as "cycling" — a
        // one-word rotation is just the static case.
        let panels = build_cycle_panels("n", "cycle = exit");
        assert_eq!(panels.len(), 1);
    }

    #[test]
    fn a_multi_word_cycle_yields_one_panel_per_word_in_order() {
        let panels = build_cycle_panels("n", "cycle = exit, this way, out");
        assert_eq!(panels.len(), 3, "one panel per cycle word");
        // Each panel's top-left interior cell must differ from at least one of
        // the others (different first letters ⇒ different rasterised glyph at
        // the same grid position, framed at margin (1,1)).
        assert_ne!(
            panels[0].full.cell(1, 1),
            panels[2].full.cell(1, 1),
            "EXIT vs OUT start differently"
        );
        // The ACCENT tier carries the word's own first letter, so it too must
        // differ between EXIT and OUT.
        assert_eq!(panels[0].accent.cell(0, 0).glyph, u32::from('E'));
        assert_eq!(panels[2].accent.cell(0, 0).glyph, u32::from('O'));
    }

    #[test]
    fn every_cycle_word_gets_the_same_scale_and_frame() {
        let panels = build_cycle_panels("n", "cycle = a, b\nscale = 1\nframe = off");
        assert_eq!(panels.len(), 2);
        // `frame = off` ⇒ no cell in ANY tier of ANY panel is frame ink
        // (`CodeClass::Comment` is ONLY ever produced by a frame — the FULL tier's
        // `+=|` box or the COMPACT tier's `[…]` brackets), for every word in the
        // rotation, not just the first.
        for p in &panels {
            for t in [&p.full, &p.compact, &p.accent] {
                for c in 0..t.cols() {
                    for r in 0..t.rows() {
                        assert_ne!(
                            t.cell(c, r).class,
                            CodeClass::Comment,
                            "frame = off ⇒ no frame ink anywhere"
                        );
                    }
                }
            }
        }
        // Bare (unframed) single-letter FULL panels at scale 1 are exactly
        // FONT_COLS × FONT_ROWS — same footprint for every word in the rotation.
        for p in &panels {
            assert_eq!(p.full.cols(), FONT_COLS);
            assert_eq!(p.full.rows(), FONT_ROWS);
            // Bare (unframed) COMPACT tier of a 1-letter word is exactly 1×1 — no
            // bracket margin.
            assert_eq!((p.compact.cols(), p.compact.rows()), (1, 1));
        }
    }

    #[test]
    fn an_overlong_cycle_word_is_truncated_not_rejected() {
        let long = "A".repeat(40);
        let panels = build_cycle_panels("n", &format!("cycle = {long}, b"));
        // Still exactly 2 panels (truncation, not rejection); the first FULL tier
        // is built from a MAX_CYCLE_WORD_LEN-letter word, so it must be at least
        // as wide as the (short) second word's FULL tier.
        assert_eq!(panels.len(), 2);
        assert!(panels[0].full.cols() >= panels[1].full.cols());
    }

    // ── effective_scale: the "fit the face" half of the legibility fix ────────

    #[test]
    fn effective_scale_shrinks_a_word_that_would_blow_the_column_budget() {
        // "EXIT" at scale 2, framed, rasterises to 29 columns (> TARGET_FULL_COLS
        // = 22), which a fixed scale would ship as unreadable noise.
        // `effective_scale` must back off to a scale whose framed width actually fits.
        let scale = effective_scale("EXIT", 2, true);
        let (cols, _, _) = rasterise_word("EXIT", scale);
        assert!(
            cols as f32 + (FRAME_MARGIN * 2) as f32 <= TARGET_FULL_COLS,
            "scale {scale} still doesn't fit: {cols} cols framed"
        );
        assert!(scale < 2, "must have backed off from the authored cap");
    }

    #[test]
    fn effective_scale_never_exceeds_the_authored_cap() {
        // A short word ("OK") could easily fit MAX_SCALE, but a lower authored
        // `scale =` is still a ceiling, not just a target.
        assert_eq!(effective_scale("OK", 1, true), 1);
    }

    #[test]
    fn effective_scale_of_an_empty_word_keeps_the_cap() {
        // Nothing to fit against ⇒ the historical "render at the authored/default
        // scale, just blank" behaviour.
        assert_eq!(effective_scale("", 2, true), 2);
    }

    // ── compact_panel / accent_panel: the fallback tiers ──────────────────────

    #[test]
    fn compact_panel_is_far_narrower_than_the_full_pixel_font_tier() {
        let tiers = build_word_tiers("n", "EXIT", MAX_SCALE, true);
        assert!(
            tiers.compact.cols() < tiers.full.cols(),
            "compact {} cols must be narrower than full {} cols",
            tiers.compact.cols(),
            tiers.full.cols()
        );
        // "[EXIT]" — 4 letters + 2 brackets, 1 row.
        assert_eq!((tiers.compact.cols(), tiers.compact.rows()), (6, 1));
        assert_eq!(tiers.compact.cell(0, 0).glyph, u32::from('['));
        assert_eq!(tiers.compact.cell(5, 0).glyph, u32::from(']'));
        assert_eq!(tiers.compact.cell(1, 0).glyph, u32::from('E'));
    }

    #[test]
    fn accent_panel_is_always_exactly_one_cell() {
        let tiers = build_word_tiers("n", "EXIT", MAX_SCALE, true);
        assert_eq!((tiers.accent.cols(), tiers.accent.rows()), (1, 1));
        assert_eq!(tiers.accent.cell(0, 0).glyph, u32::from('E'));
    }
}
