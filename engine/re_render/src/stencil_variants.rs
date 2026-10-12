//! `stencil_variants` — state-swapped art and authored LOD tiers for `[note
//! stencil]` placards.
//!
//! ## The syntax this module owns
//!
//! A stencil body (the text captured VERBATIM by `re_level` under `[note stencil …]`
//! or `[note banner …]`, see `re_level::parse_note_header`'s doc comment on why
//! `banner` is a keyword alias for `NoteKind::Stencil` and not a new variant) may
//! open with up to two DIRECTIVE lines, then zero or more `--- LABEL ---`-delimited
//! sections:
//!
//! ```text
//! [note stencil bind=0 pos=6,3,2 facing=-x]
//!     colors = [:#e0a040 E:#40ff80
//!     --- 0 ---
//!     +--[E]--+
//!     |  OFF  |
//!     +-------+
//!     --- far ---
//!     [OFF]
//!     --- glyph o ---
//!     --- 1 ---
//!     +--[E]--+
//!     |  ON   |
//!     +-------+
//! ```
//!
//! * **`bind = N`** — a state-bus TAG (`crate::state_bus::state_of`),
//!   which for a puzzle node is its 0-based declaration index in `[node …]` order
//!   (`re_game::state_bridge`'s `map.set_state(idx as u16, …)` — see the module docs
//!   below for why the tag must be authored as a raw number rather than a node NAME).
//!   Absent ⇒ the note is unbound: the FIRST group renders regardless of any state.
//! * **`colors = GLYPH:#hex …`** — a glyph→accent key. Every character named
//!   here is retinted from the body ink to the register's ACCENT ink
//!   (`crate::note_style`'s `NotePalette::mark_fg`) wherever it appears in ANY tier of
//!   this note. See the module docs' "Colour key: a two-tone approximation" section
//!   for why this is not a full per-glyph colour table.
//! * **`--- N ---`** (a section whose trimmed label parses as a plain `u32`) opens a
//!   STATE GROUP selected when `state_of(bind) == N`. A non-numeric label (`--- off
//!   ---`) is accepted too and assigned the POSITIONAL value (0, 1, 2, … in the order
//!   its group appears) — so `--- off ---` then `--- on ---` behaves exactly like
//!   `--- 0 ---` then `--- 1 ---`. Mixing numeric and positional labels in one note is
//!   allowed but not recommended (the positional counter only advances past STATE
//!   sections, numeric or not).
//! * **`--- far ---`** / **`--- glyph X ---`** attach a COMPACT tier / a
//!   single-glyph tier to the group currently open (the most recently opened `---
//!   N ---`, or an implicit group 0 seeded from the PREAMBLE if a tier marker appears
//!   before any state section — the shape a tiered-but-unbound stencil uses).
//!
//! A body with NO `--- … ---` markers at all is the LEGACY shape and is untouched:
//! [`parse`](crate::stencil_variants::parse) returns [`ParsedNote::Legacy`](crate::stencil_variants::ParsedNote::Legacy) with the body byte-identical (once any
//! `bind`/`colors` directive lines — which no shipped stencil authors today — are
//! peeled off), so every stencil placard shipped before this module existed renders
//! exactly as before.
//!
//! **Exception — a markerless BANNER.** `[note banner]` is documented (see
//! [`crate::banner`]) as `word = … / scale = … / frame = … / cycle = …` directive
//! lines with NO `--- banner ---` wrapper — the level author never writes that
//! marker themselves (`re_level` maps the `banner` keyword straight onto
//! `NoteKind::Stencil`, and every shipped `[note banner]` body is bare directive
//! lines, e.g. `levels/100-signage-showcase.lvl`'s `word = EXIT`). So a markerless
//! body whose every non-blank line is one of those four directive keys is NOT the
//! legacy ascii-art shape: [`parse`](crate::stencil_variants::parse) recognises it (see `looks_like_banner_body`)
//! and returns [`ParsedNote::Banner`](crate::stencil_variants::ParsedNote::Banner) directly, exactly as if it had been wrapped in
//! an explicit `--- banner ---` section. A body mixing ordinary art with a stray
//! `word =`-shaped line (vanishingly unlikely — real stencil art is box-drawing, not
//! `key = value` prose) still needs EVERY line to match to be treated as a banner,
//! so authored ASCII art is never misread.
//!
//! ## Why `bind` takes a raw tag, not a node NAME
//!
//! `re_level::LevelNote` carries no field for it. Adding one would be a breaking
//! struct-literal change to every OTHER crate that constructs a `LevelNote` by full
//! literal (`re_game`, `ts_editor`, `re_voximport`, `re_imgqa`, `re_lvlcheck` tests).
//! Resolving a node NAME to its index also needs the level's full `[node]` table,
//! which is only visible to `re_level`/`re_game`, never to `re_render` (engine-agnostic
//! by design — only `re_present` touches the terminal, and `re_render` carries no
//! puzzle/game knowledge either). So `bind` is parsed HERE, from the body text that
//! already reaches `re_render` untouched via `re_game::level_text::build_level_text`'s
//! existing `registry.push_note_text(label, &note.text)` call — no cross-crate
//! plumbing, no new struct field, and the existing pipe/lighting tag space
//! (`re_core::CellStateMap`, keyed by small dense `u16`s) already expects a raw
//! integer tag, not a name. An editor wanting to hide this arithmetic behind a node
//! picker is a follow-up, not a blocker.
//!
//! ## Colour key: per-glyph, via a side channel
//!
//! Every stencil cell that reaches the shader carries only a
//! `crate::code_wall::CodeClass` (`Blank | Code | Comment`) plus the note's
//! REGISTER — `CodeGlyph` itself (owned by `code_wall`) has no per-cell colour
//! field, so a *class*-only lookup can select at
//! most one shared accent colour for every keyed glyph (`reclass_accent` still marks
//! them `Comment`, which is what a caller with no colour table falls back to).
//!
//! To give each keyed glyph its OWN authored colour without widening `CodeGlyph`,
//! [`parse`](crate::stencil_variants::parse) keeps the full `Vec<ColorKeyEntry>` (glyph *and* hex, not just the
//! glyph set) on both [`ParsedNote::Legacy`](crate::stencil_variants::ParsedNote::Legacy) and [`StencilVariants`](crate::stencil_variants::StencilVariants) (via
//! [`StencilVariants::colors`](crate::stencil_variants::StencilVariants::colors)). `crate::scene_text` reads it once at PUSH time and
//! stores it on the `RegisteredNote`; `NoteSample::color_override` is filled in by
//! matching the just-sampled glyph CHARACTER against that table
//! (`scene_text::sample_note_panel_anchored`), and
//! `ascii_shader::Material::sample_note_wall` applies it as a final override AFTER
//! its class-based colour pick — so an unkeyed glyph is completely unaffected
//! (`color_override` is `None`) and a keyed glyph gets its own literal `#RRGGBB`
//! regardless of which other glyphs share its class.

use crate::code_wall::{CodeGlyph, TextPanel, NOTE_COLS, NOTE_ROWS};
use crate::note_style;

/// Projected face height (screen rows) at or below which a group's `glyph` tier is
/// chosen over `far`/`full` — a "projected height ≤ 2 rows" rule. Tunable.
pub const GLYPH_TIER_MAX_ROWS: f32 = 2.0;
/// Projected face height at or below which a group's `far` (compact) tier is chosen
/// over `full`, when no `glyph` tier applies. Comfortably above [`GLYPH_TIER_MAX_ROWS`]
/// and well under [`crate::code_wall::NOTE_ROWS`] so a note walked up to still shows
/// its full body.
pub const FAR_TIER_MAX_ROWS: f32 = 8.0;

/// One glyph named in an authored `colors =` key, with its full authored colour kept
/// for completeness (see the module docs' "Colour key" section for why only the glyph
/// itself drives today's render).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ColorKeyEntry {
    pub glyph: char,
    pub rgb: u32,
}

/// What [`parse`] found in a stencil/banner body.
pub(crate) enum ParsedNote {
    /// No `--- … ---` markers: the historical single-body stencil. `body` is the
    /// original text with only `bind`/`colors` directive lines removed (empty on
    /// every shipped stencil, since none author those keys), so a caller building the
    /// legacy way (`NoteRegister::Stencil.body(..)` → `TextPanel::build_verbatim`) is
    /// byte-identical to before this module existed. `accent` is the colour-key glyph
    /// set (empty unless a `colors =` line was authored); `colors` is the same key
    /// with its authored RGB kept, for the per-glyph override
    /// (`crate::scene_text::NoteSample::color_override`; see the module docs' "Colour
    /// key" section).
    Legacy {
        body: String,
        accent: Vec<char>,
        colors: Vec<ColorKeyEntry>,
    },
    /// The FIRST section's label was `banner` (case-insensitive): `block` is that
    /// section's raw text, handed to [`crate::banner::build_panel`].
    Banner(String),
    /// One or more state/tier sections: a live, sampled [`StencilVariants`].
    Variants(StencilVariants),
}

/// A precomputed panel plus its cached `used_rows` (mirrors
/// `scene_text::Registered`'s existing field — recomputing `used_rows()` per SAMPLE
/// would be an O(cols·rows) scan on every rendered cell).
#[derive(Debug)]
struct TierPanel {
    panel: TextPanel,
    used_rows: usize,
}

impl TierPanel {
    fn build(label: &str, body: &str, accent: &[char]) -> TierPanel {
        let dedented = note_style::dedent(body);
        let mut panel = TextPanel::build_verbatim(label, dedented.as_ref(), NOTE_COLS, NOTE_ROWS);
        panel.reclass_accent(accent);
        let used_rows = panel.used_rows();
        TierPanel { panel, used_rows }
    }

    /// A `--- glyph X ---` tier: a single-CELL grid holding just `ch`, so the glyph
    /// fills the WHOLE face wherever it is sampled — a far object represented by one
    /// character reads as that character everywhere on its face, not just in one
    /// corner of an otherwise-blank 26-column plate.
    fn single_glyph(label: &str, ch: char, accent: &[char]) -> TierPanel {
        let mut panel = TextPanel::build_verbatim(label, &ch.to_string(), 1, 1);
        panel.reclass_accent(accent);
        TierPanel {
            panel,
            used_rows: 1,
        }
    }
}

/// One STATE of a bound stencil (or the sole group of an unbound but tiered one).
#[derive(Debug)]
struct StencilGroup {
    /// The `state_of(bind)` value that selects this group.
    value: u32,
    full: TierPanel,
    far: Option<TierPanel>,
    glyph: Option<TierPanel>,
}

/// A parsed, sample-ready state-swapped / tiered stencil.
#[derive(Debug)]
pub(crate) struct StencilVariants {
    bind: Option<u16>,
    groups: Vec<StencilGroup>,
    /// The note's `colors =` key, RGB kept, for the per-glyph override
    /// (`crate::scene_text` reads this via [`StencilVariants::colors`] at push time).
    colors: Vec<ColorKeyEntry>,
}

impl StencilVariants {
    /// The note's authored colour key (glyph + RGB), empty if it authored none. Read
    /// once by `crate::scene_text::push_note_text_styled`'s `Variants` branch — see
    /// the module docs' "Colour key" section for why this cannot live on `CodeGlyph`
    /// itself.
    pub(crate) fn colors(&self) -> &[ColorKeyEntry] {
        &self.colors
    }
}

/// Sample `variants` at wall coordinate `(u, v)`, exactly like
/// `TextPanel::sample_fitted_anchored_2d` — picks the live STATE group (via
/// `crate::state_bus::state_of`) and the LOD tier (via `face_rows`), then delegates.
/// `variants.groups` is never empty for a value produced by [`parse`], but a defined
/// blank answer is returned rather than panicking if it ever were.
pub(crate) fn sample(
    variants: &StencilVariants,
    u: f32,
    v: f32,
    anchor: crate::face_span::FaceAnchor,
    face_rows: f32,
) -> CodeGlyph {
    let Some(group) = select_group(variants) else {
        return CodeGlyph::BLANK;
    };
    let tier = select_tier(group, face_rows);
    tier.panel
        .sample_fitted_anchored_2d(u, v, tier.used_rows, anchor, face_rows)
}

/// Which group is live right now: unbound ⇒ always the first; bound ⇒ the group whose
/// `value` matches the live state word, falling back to the LAST group (rather than
/// blank) when the live state names no group — an authored stencil should always show
/// SOMETHING, and "the last declared state" is a safer default than a hole in the wall.
fn select_group(v: &StencilVariants) -> Option<&StencilGroup> {
    match v.bind {
        None => v.groups.first(),
        Some(tag) => {
            let live = crate::state_bus::state_of(tag);
            v.groups
                .iter()
                .find(|g| g.value == live)
                .or_else(|| v.groups.last())
        }
    }
}

fn select_tier(g: &StencilGroup, face_rows: f32) -> &TierPanel {
    if face_rows > 0.0 && face_rows <= GLYPH_TIER_MAX_ROWS {
        if let Some(t) = &g.glyph {
            return t;
        }
    }
    if face_rows > 0.0 && face_rows <= FAR_TIER_MAX_ROWS {
        if let Some(t) = &g.far {
            return t;
        }
    }
    &g.full
}

/// Parse `raw_body` (the note's captured, un-dedented text) into a [`ParsedNote`].
/// `label` is forwarded to every built [`TextPanel`] (diagnostics only).
pub(crate) fn parse(label: &str, raw_body: &str) -> ParsedNote {
    let (bind, colors, rest) = strip_directives(raw_body);
    let accent: Vec<char> = colors.iter().map(|e| e.glyph).collect();
    let (preamble, sections) = split_markers(&rest);

    if sections.is_empty() {
        if looks_like_banner_body(&preamble) {
            return ParsedNote::Banner(preamble);
        }
        return ParsedNote::Legacy {
            body: preamble,
            accent,
            colors,
        };
    }
    if sections[0].0.eq_ignore_ascii_case("banner") {
        return ParsedNote::Banner(sections[0].1.clone());
    }

    let mut groups: Vec<StencilGroup> = Vec::new();
    let mut positional = 0u32;
    for (marker, block) in &sections {
        let m = marker.trim();
        if let Some(rest) = strip_ci_prefix(m, "glyph") {
            let ch = rest.trim().chars().next().unwrap_or(' ');
            let g = ensure_group(&mut groups, label, &preamble, &accent);
            g.glyph = Some(TierPanel::single_glyph(label, ch, &accent));
        } else if m.eq_ignore_ascii_case("far") {
            let g = ensure_group(&mut groups, label, &preamble, &accent);
            g.far = Some(TierPanel::build(label, block, &accent));
        } else {
            let value = m.parse::<u32>().unwrap_or(positional);
            positional += 1;
            groups.push(StencilGroup {
                value,
                full: TierPanel::build(label, block, &accent),
                far: None,
                glyph: None,
            });
        }
    }

    ParsedNote::Variants(StencilVariants {
        bind,
        groups,
        colors,
    })
}

/// `true` when EVERY non-blank line of a markerless body is a banner directive
/// (`word =` / `scale =` / `frame =` / `cycle =`, case-insensitive on the key) — the
/// shape a `[note banner]` body actually authors (see the module docs' "Exception —
/// a markerless BANNER" section). At least one such line must be present (an
/// all-blank body is not a banner). A line that doesn't split on `=`, or whose key
/// isn't one of the four, fails the whole body — so real ASCII art (box-drawing,
/// no `=` at all) is never misread.
fn looks_like_banner_body(body: &str) -> bool {
    const BANNER_KEYS: [&str; 4] = ["word", "scale", "frame", "cycle"];
    let mut any = false;
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, _)) = line.split_once('=') else {
            return false;
        };
        if !BANNER_KEYS.contains(&key.trim().to_ascii_lowercase().as_str()) {
            return false;
        }
        any = true;
    }
    any
}

/// Return the LAST group, creating an implicit group 0 (seeded from `preamble`) first
/// if none exists yet — the shape an unbound-but-tiered stencil uses (`--- far ---`
/// with no preceding `--- N ---`).
fn ensure_group<'a>(
    groups: &'a mut Vec<StencilGroup>,
    label: &str,
    preamble: &str,
    accent: &[char],
) -> &'a mut StencilGroup {
    if groups.is_empty() {
        groups.push(StencilGroup {
            value: 0,
            full: TierPanel::build(label, preamble, accent),
            far: None,
            glyph: None,
        });
    }
    groups.last_mut().expect("just ensured non-empty")
}

/// `s` with a case-insensitive `prefix` removed, IF `s` starts with `prefix` followed
/// by nothing or whitespace (so `"glyph"` matches `"glyph x"` but not `"glyphon"`).
fn strip_ci_prefix<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() < prefix.len()
        || !s.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
    {
        return None;
    }
    let rest = &s[prefix.len()..];
    if rest.is_empty() || rest.starts_with(char::is_whitespace) {
        Some(rest)
    } else {
        None
    }
}

/// Peel `bind = N` / `colors = …` directive lines off the TOP of `raw_body` (in
/// either order, either present, either absent). Stops at the first line that is
/// neither — including a `--- … ---` marker or ordinary stencil content — so a
/// directive-shaped line of AUTHORED ART past the first couple of lines is never
/// mistaken for one.
fn strip_directives(raw_body: &str) -> (Option<u16>, Vec<ColorKeyEntry>, String) {
    let mut bind = None;
    let mut colors = Vec::new();
    let lines: Vec<&str> = raw_body.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim();
        if let Some(rest) =
            strip_ci_prefix(t, "bind").and_then(|r| r.trim_start().strip_prefix('='))
        {
            if let Ok(v) = rest.trim().parse::<u16>() {
                bind = Some(v);
                i += 1;
                continue;
            }
        }
        if let Some(rest) =
            strip_ci_prefix(t, "colors").and_then(|r| r.trim_start().strip_prefix('='))
        {
            colors = parse_colors(rest.trim());
            i += 1;
            continue;
        }
        break;
    }
    (bind, colors, lines[i..].join("\n"))
}

/// Parse `colors = ` tokens of the shape `GLYPH:#RRGGBB`, space-separated. A malformed
/// token is silently skipped — a colour key is decoration, never a load-bearing parse.
fn parse_colors(rest: &str) -> Vec<ColorKeyEntry> {
    let mut out = Vec::new();
    for tok in rest.split_whitespace() {
        let Some((g, hex)) = tok.split_once(':') else {
            continue;
        };
        let Some(glyph) = g.chars().next() else {
            continue;
        };
        let hex = hex.strip_prefix('#').unwrap_or(hex);
        if let Ok(rgb) = u32::from_str_radix(hex, 16) {
            out.push(ColorKeyEntry { glyph, rgb });
        }
    }
    out
}

/// A `--- LABEL ---` marker line: `LABEL` trimmed, or `None` when `line` is not one —
/// including a pure-dash divider (`"--------"`), which would otherwise strip to an
/// all-dash "label" and be mistaken for a marker.
fn marker_label(line: &str) -> Option<&str> {
    let t = line.trim();
    let inner = t.strip_prefix("---")?.strip_suffix("---")?.trim();
    if inner.is_empty() || inner.chars().all(|c| c == '-') {
        None
    } else {
        Some(inner)
    }
}

/// Split `body` into its PREAMBLE (lines before the first marker) and an ordered list
/// of `(label, block_text)` sections. No markers at all ⇒ `sections` is empty and
/// `preamble` is `body` itself, byte-identical (see [`ParsedNote::Legacy`]).
fn split_markers(body: &str) -> (String, Vec<(String, String)>) {
    let mut preamble: Vec<&str> = Vec::new();
    let mut sections: Vec<(String, Vec<&str>)> = Vec::new();
    for line in body.lines() {
        if let Some(label) = marker_label(line) {
            sections.push((label.to_string(), Vec::new()));
        } else if let Some(last) = sections.last_mut() {
            last.1.push(line);
        } else {
            preamble.push(line);
        }
    }
    let sections = sections
        .into_iter()
        .map(|(l, ls)| (l, ls.join("\n")))
        .collect();
    (preamble.join("\n"), sections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_wall::CodeClass;
    use crate::state_bus::CellStateScope;
    use re_core::CellStateMap;

    // ── marker_label / split_markers ──

    #[test]
    fn marker_label_recognises_a_dashed_header_and_rejects_a_bare_divider() {
        assert_eq!(marker_label("--- off ---"), Some("off"));
        assert_eq!(marker_label("   --- 0 ---   "), Some("0"));
        assert_eq!(marker_label("--- glyph X ---"), Some("glyph X"));
        assert_eq!(
            marker_label("--------"),
            None,
            "a pure-dash divider is not a marker"
        );
        assert_eq!(marker_label("-- 0 --"), None, "needs 3 dashes each side");
        assert_eq!(marker_label("not a marker"), None);
    }

    #[test]
    fn split_markers_with_no_markers_returns_the_whole_body_as_preamble() {
        let (pre, sections) = split_markers("+--+\n| a |\n+--+");
        assert_eq!(pre, "+--+\n| a |\n+--+");
        assert!(sections.is_empty());
    }

    #[test]
    fn split_markers_groups_lines_under_their_marker() {
        let (pre, sections) = split_markers("intro\n--- 0 ---\nline a\nline b\n--- far ---\nAB");
        assert_eq!(pre, "intro");
        assert_eq!(
            sections,
            vec![
                ("0".to_string(), "line a\nline b".to_string()),
                ("far".to_string(), "AB".to_string()),
            ]
        );
    }

    // ── strip_directives ──

    #[test]
    fn strip_directives_reads_bind_and_colors_in_either_order() {
        let (bind, colors, rest) = strip_directives("bind = 3\ncolors = X:#ff0000\nbody text");
        assert_eq!(bind, Some(3));
        assert_eq!(
            colors,
            vec![ColorKeyEntry {
                glyph: 'X',
                rgb: 0xff0000
            }]
        );
        assert_eq!(rest, "body text");

        let (bind2, colors2, rest2) = strip_directives("colors = Y:#00ff00\nbind = 5\nrest");
        assert_eq!(bind2, Some(5));
        assert_eq!(
            colors2,
            vec![ColorKeyEntry {
                glyph: 'Y',
                rgb: 0x00ff00
            }]
        );
        assert_eq!(rest2, "rest");
    }

    #[test]
    fn strip_directives_stops_at_the_first_non_directive_line() {
        // A body that legitimately contains the word "bind" past the header is left
        // alone — only LEADING directive lines are consumed.
        let (bind, _colors, rest) = strip_directives("+--[bind]--+\n|  ON  |\n+------+");
        assert_eq!(bind, None);
        assert_eq!(rest, "+--[bind]--+\n|  ON  |\n+------+");
    }

    #[test]
    fn strip_directives_on_a_plain_body_is_a_lossless_identity() {
        let body = "+--------------------+\n|   GALLERY WING C   |\n|  [E] TO OPERATE    |\n|  MIND THE GLASS    |\n+--------------------+";
        let (bind, colors, rest) = strip_directives(body);
        assert_eq!(bind, None);
        assert!(colors.is_empty());
        assert_eq!(
            rest, body,
            "a shipped stencil with no directives is untouched"
        );
    }

    #[test]
    fn a_malformed_bind_value_is_left_for_the_body_not_silently_dropped() {
        // `bind = five` doesn't parse as u16, so the line is NOT a directive and
        // becomes ordinary (if odd) stencil content instead of vanishing.
        let (bind, _colors, rest) = strip_directives("bind = five\ntext");
        assert_eq!(bind, None);
        assert_eq!(rest, "bind = five\ntext");
    }

    // ── parse_colors ──

    #[test]
    fn parse_colors_reads_glyph_hex_pairs_and_skips_junk() {
        let out = parse_colors("[:#e0a040 E:#40ff80 bad junk X:#zzz");
        assert_eq!(
            out,
            vec![
                ColorKeyEntry {
                    glyph: '[',
                    rgb: 0xe0a040
                },
                ColorKeyEntry {
                    glyph: 'E',
                    rgb: 0x40ff80
                },
            ]
        );
    }

    // ── parse(): end-to-end shapes ──

    #[test]
    fn a_body_with_no_markers_is_legacy_and_byte_identical() {
        let body = "    +--[E]--+\n    |  ON   |\n    +-------+";
        match parse("n", body) {
            ParsedNote::Legacy {
                body: got,
                accent,
                colors,
            } => {
                assert_eq!(got, body);
                assert!(accent.is_empty());
                assert!(colors.is_empty());
            }
            _ => panic!("expected Legacy"),
        }
    }

    #[test]
    fn a_colors_directive_on_an_otherwise_plain_body_still_yields_legacy_with_an_accent_set() {
        let body = "colors = [:#e0a040\n+--[E]--+\n|  ON   |\n+-------+";
        match parse("n", body) {
            ParsedNote::Legacy {
                body: got,
                accent,
                colors,
            } => {
                assert_eq!(got, "+--[E]--+\n|  ON   |\n+-------+");
                assert_eq!(accent, vec!['[']);
                assert_eq!(
                    colors,
                    vec![ColorKeyEntry {
                        glyph: '[',
                        rgb: 0xe0a040
                    }]
                );
            }
            _ => panic!("expected Legacy"),
        }
    }

    #[test]
    fn a_leading_banner_marker_yields_the_banner_variant() {
        match parse("n", "--- banner ---\nword = exit\nscale = 2") {
            ParsedNote::Banner(block) => {
                assert_eq!(block, "word = exit\nscale = 2");
            }
            _ => panic!("expected Banner"),
        }
    }

    // ── parse(): a markerless `[note banner]` body (BUG-101) ──

    #[test]
    fn a_markerless_body_of_only_banner_directives_is_the_banner_variant() {
        // The exact shape `levels/100-signage-showcase.lvl` authors: no
        // `--- banner ---` wrapper, just the directive lines.
        match parse("n", "word = EXIT\nscale = 3\nframe = on") {
            ParsedNote::Banner(block) => {
                assert_eq!(block, "word = EXIT\nscale = 3\nframe = on");
            }
            _ => panic!("expected Banner, got a variant that is not Banner"),
        }
    }

    #[test]
    fn a_markerless_body_of_one_banner_directive_is_still_the_banner_variant() {
        match parse("n", "cycle = exit, this way") {
            ParsedNote::Banner(block) => assert_eq!(block, "cycle = exit, this way"),
            _ => panic!("expected Banner"),
        }
    }

    #[test]
    fn a_markerless_body_mixing_art_and_a_banner_looking_line_stays_legacy() {
        // Real stencil art is box-drawing, not `key = value` prose — but if a line
        // happens to shape like one, the WHOLE body must still fail the banner test
        // unless every line matches, so ASCII art is never misread.
        let body = "+--[E]--+\nword = ON\n+-------+";
        match parse("n", body) {
            ParsedNote::Legacy { body: got, .. } => assert_eq!(got, body),
            _ => panic!("expected Legacy: a mixed body is not a banner"),
        }
    }

    #[test]
    fn an_all_blank_markerless_body_stays_legacy_not_banner() {
        match parse("n", "   \n\n  ") {
            ParsedNote::Legacy { .. } => {}
            _ => panic!("expected Legacy: no directive lines at all ⇒ not a banner"),
        }
    }

    #[test]
    fn a_colors_directive_ahead_of_a_markerless_banner_body_still_yields_banner() {
        // `bind =`/`colors =` are stripped as directives before the banner check
        // ever sees the body, so a (rare, not authored today) `colors =` line ahead
        // of `word =` does not defeat the detection.
        match parse("n", "colors = X:#ff0000\nword = exit") {
            ParsedNote::Banner(block) => assert_eq!(block, "word = exit"),
            _ => panic!("expected Banner"),
        }
    }

    #[test]
    fn numeric_state_sections_build_groups_keyed_by_their_literal_value() {
        let body = "bind = 0\n--- 0 ---\nAAA\n--- 1 ---\nBBB";
        match parse("n", body) {
            ParsedNote::Variants(v) => {
                assert_eq!(v.bind, Some(0));
                assert_eq!(v.groups.len(), 2);
                assert_eq!(v.groups[0].value, 0);
                assert_eq!(v.groups[1].value, 1);
            }
            _ => panic!("expected Variants"),
        }
    }

    #[test]
    fn non_numeric_state_labels_get_positional_values() {
        let body = "--- off ---\nAAA\n--- on ---\nBBB\n--- decoy ---\nCCC";
        match parse("n", body) {
            ParsedNote::Variants(v) => {
                assert_eq!(
                    v.groups.iter().map(|g| g.value).collect::<Vec<_>>(),
                    vec![0, 1, 2]
                );
            }
            _ => panic!("expected Variants"),
        }
    }

    #[test]
    fn far_and_glyph_markers_attach_to_the_most_recently_opened_state() {
        let body =
            "--- 0 ---\nfull0\n--- far ---\nf0\n--- glyph o ---\n--- 1 ---\nfull1\n--- far ---\nf1";
        match parse("n", body) {
            ParsedNote::Variants(v) => {
                assert_eq!(v.groups.len(), 2);
                assert!(v.groups[0].far.is_some());
                assert!(v.groups[0].glyph.is_some());
                assert!(v.groups[1].far.is_some());
                assert!(
                    v.groups[1].glyph.is_none(),
                    "group 1 authored no glyph tier"
                );
            }
            _ => panic!("expected Variants"),
        }
    }

    #[test]
    fn a_tier_marker_before_any_state_section_seeds_an_implicit_group_from_the_preamble() {
        let body = "PREAMBLE BODY\n--- far ---\nCOMPACT";
        match parse("n", body) {
            ParsedNote::Variants(v) => {
                assert_eq!(v.groups.len(), 1);
                assert_eq!(v.groups[0].value, 0);
                assert!(v.groups[0].far.is_some());
            }
            _ => panic!("expected Variants"),
        }
    }

    // ── sample(): group selection by live state ──

    fn state_map(pairs: &[(u16, u32)]) -> CellStateMap {
        let mut m = CellStateMap::new();
        for &(tag, val) in pairs {
            m.set_state(tag, val);
        }
        m
    }

    #[test]
    fn unbound_variants_always_sample_the_first_group_regardless_of_live_state() {
        let body = "--- 0 ---\nAAA\n--- 1 ---\nBBB";
        let ParsedNote::Variants(v) = parse("n", body) else {
            panic!("expected Variants")
        };
        let map = state_map(&[(0, 1)]); // irrelevant: v.bind is None
        let _scope = CellStateScope::new(&map);
        let g = sample(&v, 0.02, 0.02, crate::face_span::FaceAnchor::UNKNOWN, 0.0);
        assert_eq!(char::from_u32(g.glyph), Some('A'));
    }

    #[test]
    fn a_bound_stencil_switches_body_with_the_live_state() {
        let body = "bind = 2\n--- 0 ---\nAAA\n--- 1 ---\nBBB";
        let ParsedNote::Variants(v) = parse("n", body) else {
            panic!("expected Variants")
        };

        let off = state_map(&[(2, 0)]);
        {
            let _scope = CellStateScope::new(&off);
            let g = sample(&v, 0.02, 0.02, crate::face_span::FaceAnchor::UNKNOWN, 0.0);
            assert_eq!(
                char::from_u32(g.glyph),
                Some('A'),
                "state 0 selects the first group"
            );
        }
        let on = state_map(&[(2, 1)]);
        {
            let _scope = CellStateScope::new(&on);
            let g = sample(&v, 0.02, 0.02, crate::face_span::FaceAnchor::UNKNOWN, 0.0);
            assert_eq!(
                char::from_u32(g.glyph),
                Some('B'),
                "state 1 selects the second group"
            );
        }
    }

    #[test]
    fn an_out_of_range_live_state_falls_back_to_the_last_group() {
        let body = "bind = 9\n--- 0 ---\nAAA\n--- 1 ---\nBBB";
        let ParsedNote::Variants(v) = parse("n", body) else {
            panic!("expected Variants")
        };
        let map = state_map(&[(9, 77)]); // no group has value 77
        let _scope = CellStateScope::new(&map);
        let g = sample(&v, 0.02, 0.02, crate::face_span::FaceAnchor::UNKNOWN, 0.0);
        assert_eq!(
            char::from_u32(g.glyph),
            Some('B'),
            "unmatched state ⇒ last group, never blank"
        );
    }

    // ── sample(): tier selection by face span ──

    #[test]
    fn tier_selection_prefers_glyph_then_far_then_full_as_the_face_shrinks() {
        // Distinct leading letters per tier so the sampled glyph unambiguously names
        // which tier answered. Each tier's body is ONE line, so — per the
        // refuse-to-magnify clamp (`TextPanel::sample_fitted_anchored_2d`) — a
        // `face_rows` taller than that one line vertically CENTRES it; sampling at
        // `v = 0.5` hits it regardless of how tall the face is.
        let body = "--- 0 ---\nZFULL\n--- far ---\nFAR\n--- glyph G ---";
        let ParsedNote::Variants(v) = parse("n", body) else {
            panic!("expected Variants")
        };
        let anchor = crate::face_span::FaceAnchor::UNKNOWN;

        let near = sample(&v, 0.02, 0.5, anchor, 20.0);
        assert_eq!(
            char::from_u32(near.glyph),
            Some('Z'),
            "above the far threshold ⇒ the full body"
        );

        let mid = sample(&v, 0.02, 0.5, anchor, 5.0);
        assert_eq!(
            char::from_u32(mid.glyph),
            Some('F'),
            "within the far band ⇒ the far tier"
        );

        let tiny = sample(&v, 0.5, 0.5, anchor, 1.0);
        assert_eq!(
            char::from_u32(tiny.glyph),
            Some('G'),
            "at/under the glyph threshold ⇒ the single glyph"
        );
    }

    #[test]
    fn zero_or_unknown_face_rows_keeps_the_full_body() {
        let body = "--- 0 ---\nFULL\n--- far ---\nF\n--- glyph G ---";
        let ParsedNote::Variants(v) = parse("n", body) else {
            panic!("expected Variants")
        };
        let g = sample(&v, 0.02, 0.02, crate::face_span::FaceAnchor::UNKNOWN, 0.0);
        assert_eq!(
            char::from_u32(g.glyph),
            Some('F'),
            "face_rows=0 (unknown span) ⇒ the historical full tier"
        );
    }

    // ── colour key reaches the panel as a Comment reclass ──

    #[test]
    fn a_colors_key_retints_the_named_glyph_to_comment_in_every_tier() {
        let body = "colors = X:#e0a040\n--- 0 ---\nX AND Y\n--- far ---\nXY";
        let ParsedNote::Variants(v) = parse("n", body) else {
            panic!("expected Variants")
        };
        assert_eq!(
            v.groups[0].full.panel.cell(0, 0).class,
            CodeClass::Comment,
            "'X' is keyed"
        );
        // 'Y' (not keyed) stays Code in the same row.
        let y_col = (0..NOTE_COLS)
            .find(|&c| v.groups[0].full.panel.cell(c, 0).glyph == u32::from('Y'))
            .unwrap();
        assert_eq!(v.groups[0].full.panel.cell(y_col, 0).class, CodeClass::Code);
        // The far tier gets the SAME key.
        assert_eq!(
            v.groups[0].far.as_ref().unwrap().panel.cell(0, 0).class,
            CodeClass::Comment
        );
    }

    #[test]
    fn variants_colors_exposes_the_full_glyph_and_rgb_key_for_the_per_glyph_override() {
        let body = "colors = X:#e0a040 Y:#40ff80\n--- 0 ---\nX AND Y";
        let ParsedNote::Variants(v) = parse("n", body) else {
            panic!("expected Variants")
        };
        assert_eq!(
            v.colors(),
            &[
                ColorKeyEntry {
                    glyph: 'X',
                    rgb: 0xe0a040
                },
                ColorKeyEntry {
                    glyph: 'Y',
                    rgb: 0x40ff80
                },
            ]
        );
    }

    #[test]
    fn a_body_with_no_colors_directive_exposes_an_empty_key() {
        let body = "--- 0 ---\nAAA";
        let ParsedNote::Variants(v) = parse("n", body) else {
            panic!("expected Variants")
        };
        assert!(v.colors().is_empty());
    }
}
