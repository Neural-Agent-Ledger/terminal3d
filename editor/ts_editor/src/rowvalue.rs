//! TYPED INSPECTOR ROW VALUES — the structural unblock.
//!
//! An [`crate::chrome::InspectorRow`] used to carry a pre-formatted `String`,
//! so a row knew its label and its rendered text but NOT its type. A click
//! could therefore only ever focus the field: it could not toggle a bool, open
//! an enum list, step a number, or edit a colour. Every direct-manipulation
//! widget was blocked behind that one missing piece of type information.
//!
//! [`RowValue`] restores it. Each row now carries the datum plus the metadata a
//! direct-manipulation widget needs (an enum's option list; a number's clamp
//! bounds and step), and this module owns BOTH halves of the resulting widget:
//!
//! * [`row_text_in`] — what the row renders as.
//! * [`zones_in`] — where each clickable part of that rendering sits.
//!
//! The two are computed by the SAME function ([`zones_in`] lays the row out and
//! [`row_text_in`] is its concatenation), so the painter and the hit-test can never
//! disagree about which column belongs to which widget. This mirrors the
//! `panel_buttons`/`button_span` discipline already established in `chrome.rs`:
//! ONE layout pass, consumed twice.

/// Which part of a row's WIDGET a column belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WidgetPart {
    /// A [`RowValue::Bool`] checkbox — `[x]` / `[ ]`. Click toggles.
    Toggle,
    /// A [`RowValue::Int`]/[`RowValue::Float`] decrement button — `[-]`.
    Dec,
    /// A [`RowValue::Int`]/[`RowValue::Float`] increment button — `[+]`.
    Inc,
    /// A numeric row's VALUE text. Click opens the inline numeric editor.
    Value,
    /// A [`RowValue::Enum`]'s `[< >]` affordance. Click opens the option popup.
    Options,
    /// A [`RowValue::Color`]'s `#RRGGBB ██`. Click opens the colour overlay.
    Swatch,
}

/// Which ZONE of an inspector row a column belongs to. A click on the LABEL
/// keeps the pre-existing behaviour (focus the field); a click on a widget part
/// ACTS.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RowZone {
    /// The label, the ` = ` separator, and any non-interactive value text.
    Label,
    /// One interactive part of the row's widget.
    Widget(WidgetPart),
}

/// One laid-out run of a row's rendering: the text of the run and the zone it
/// belongs to. Concatenating every `text` in order gives the row's full line.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RowSpan {
    /// The characters of this run.
    pub text: String,
    /// What clicking inside this run does.
    pub zone: RowZone,
}

/// A typed inspector-row value.
///
/// `Text` is the fallback and renders exactly as the old `String` did, so every
/// field that has not (yet) been given a richer type keeps byte-identical
/// rendering — that is what makes the row-model refactor verifiable against a
/// parity snapshot.
#[derive(Clone, PartialEq, Debug)]
pub enum RowValue {
    /// Display-only or caret-editable free text. Renders as the bare value.
    Text(String),
    /// A two-state flag. Renders as a clickable `[x]` / `[ ]` checkbox.
    Bool(bool),
    /// One of a named set. Renders as `value [< >]`; the affordance opens a
    /// scrollable popup of `options`.
    Enum {
        /// The currently selected option (need not be present in `options`
        /// when the underlying datum is unset or stale).
        current: String,
        /// Every option the popup offers, in cycle order.
        options: Vec<String>,
    },
    /// A clamped integer. Renders as `[-] value [+]`.
    ///
    /// `i128`, not `i64`, because the editor has integer fields across the FULL
    /// `u64` range — the emitter `seed` is one, and a randomly generated seed
    /// lands above `i64::MAX` about half the time. A narrower type would quietly
    /// demote exactly those rows to read-only text.
    Int {
        /// Current value.
        v: i128,
        /// Inclusive lower clamp.
        min: i128,
        /// Inclusive upper clamp.
        max: i128,
        /// What one `[-]`/`[+]` press moves by.
        step: i128,
    },
    /// A clamped float. Renders as `[-] value [+]`.
    Float {
        /// Current value.
        v: f32,
        /// Inclusive lower clamp.
        min: f32,
        /// Inclusive upper clamp.
        max: f32,
        /// What one `[-]`/`[+]` press moves by.
        step: f32,
        /// Decimal places the value renders with (matching the pre-existing
        /// per-field format, so parity is preserved).
        dp: usize,
    },
    /// An `0x00RRGGBB` colour. Renders as `#RRGGBB ██`, the block painted in
    /// the colour itself.
    Color(u32),
}

/// The filled block the colour swatch is drawn with, twice (a 2-cell swatch
/// reads as a colour patch; one cell reads as a glyph).
pub const SWATCH: &str = "\u{2588}\u{2588}";

/// ONE cell of the swatch block, for the painter to repaint in the row's own
/// colour (`draw_str` writes a string, so the tint loop needs a single-cell one).
pub const SWATCH_CELL: &str = "\u{2588}";

impl RowValue {
    /// The value's own rendered text, WITHOUT the label or the widget
    /// affordances — what a status line quotes.
    #[must_use]
    pub fn display(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Bool(b) => b.to_string(),
            Self::Enum { current, .. } => current.clone(),
            Self::Int { v, .. } => v.to_string(),
            Self::Float { v, dp, .. } => format!("{v:.*}", *dp),
            Self::Color(c) => format!("#{:06X}", c & 0xFF_FFFF),
        }
    }

    /// True when the value has a direct-manipulation widget (so `Enter` on the
    /// focused row opens its editor, and a click can act rather than focus).
    #[must_use]
    pub const fn is_widget(&self) -> bool {
        !matches!(self, Self::Text(_))
    }
}

/// The affordance FORM a row's value is drawn with.
///
/// The natural form is what a comfortable panel shows; the COMPACT form trades
/// the bracket chrome for single characters when the panel is narrow. Both
/// forms keep every zone clickable — narrowing must never make a control
/// unreachable, which is exactly what plain end-clipping used to do (the
/// default 30-column panel clipped `[Rates] max_particles = [-] 256 [+]` at 30,
/// cutting the number in half and dropping `[+]` off the row entirely).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Form {
    /// `[-] v [+]`, `v [< >]`, `#RRGGBB ██`.
    Natural,
    /// `- v +`, `v >`, `#RRGGBB`.
    Compact,
}

/// Lay out ONE field row as an ordered list of spans, in the given `form`.
fn build(label: &str, value: &RowValue, form: Form) -> Vec<RowSpan> {
    let lab = |t: String| RowSpan {
        text: t,
        zone: RowZone::Label,
    };
    let wid = |t: &str, p: WidgetPart| RowSpan {
        text: t.to_string(),
        zone: RowZone::Widget(p),
    };
    let compact = form == Form::Compact;
    let mut v = vec![lab(format!("{label} = "))];
    match value {
        RowValue::Text(s) => v.push(lab(s.clone())),
        RowValue::Bool(b) => {
            // Already three cells; there is nothing to compact.
            v.push(wid(if *b { "[x]" } else { "[ ]" }, WidgetPart::Toggle));
        }
        RowValue::Enum { current, .. } => {
            v.push(lab(format!("{current} ")));
            v.push(wid(
                if compact { ">" } else { "[< >]" },
                WidgetPart::Options,
            ));
        }
        RowValue::Int { .. } | RowValue::Float { .. } => {
            v.push(wid(if compact { "-" } else { "[-]" }, WidgetPart::Dec));
            v.push(lab(" ".to_string()));
            v.push(RowSpan {
                text: value.display(),
                zone: RowZone::Widget(WidgetPart::Value),
            });
            v.push(lab(" ".to_string()));
            v.push(wid(if compact { "+" } else { "[+]" }, WidgetPart::Inc));
        }
        RowValue::Color(c) => {
            // Compact drops the block, not the hex: the hex IS the value, and
            // the run stays one Swatch zone so the click target survives.
            let t = if compact {
                format!("#{:06X}", c & 0xFF_FFFF)
            } else {
                format!("#{:06X} {SWATCH}", c & 0xFF_FFFF)
            };
            v.push(RowSpan {
                text: t,
                zone: RowZone::Widget(WidgetPart::Swatch),
            });
        }
    }
    v
}

/// Total rendered width of a span list.
fn span_len(v: &[RowSpan]) -> usize {
    v.iter().map(|s| s.text.chars().count()).sum()
}

/// Lay out ONE field row to fit within `width` cells.
///
/// Three stages, in order of what is cheapest to lose:
/// 1. the NATURAL form, when it fits;
/// 2. the COMPACT form, trading bracket chrome for single characters;
/// 3. the compact form with a TRUNCATED LABEL — the label is the redundant
///    half (the focused row is named in full in the status line), while the
///    value and its controls are the part the row exists for.
///
/// The value's own text is never truncated, so a number is never shown cut in
/// half, and every widget zone stays on the row and therefore clickable.
#[must_use]
pub fn zones_in(label: &str, value: &RowValue, width: usize) -> Vec<RowSpan> {
    let natural = build(label, value, Form::Natural);
    if span_len(&natural) <= width {
        return natural;
    }
    let compact = build(label, value, Form::Compact);
    if span_len(&compact) <= width {
        return compact;
    }
    // Give the label whatever the value's rendering leaves over.
    let value_len = span_len(&compact) - (label.chars().count() + 3);
    let budget = width.saturating_sub(value_len + 3);
    let short: String = label.chars().take(budget).collect();
    build(&short, value, Form::Compact)
}

/// The rendered line of a field row laid out to fit `width` cells.
#[must_use]
pub fn row_text_in(label: &str, value: &RowValue, width: usize) -> String {
    zones_in(label, value, width)
        .into_iter()
        .map(|s| s.text)
        .collect()
}

/// Resolve a CHAR OFFSET within a row's rendered line to its zone, for a row
/// laid out at `width`. Offsets past the end of the line resolve to
/// [`RowZone::Label`] (clicking the blank tail of a row focuses it, matching
/// the pre-existing whole-row click target).
#[must_use]
pub fn zone_at_in(label: &str, value: &RowValue, width: usize, offset: usize) -> RowZone {
    let mut at = 0usize;
    for s in zones_in(label, value, width) {
        let n = s.text.chars().count();
        if offset < at + n {
            return s.zone;
        }
        at += n;
    }
    RowZone::Label
}

/// The `(start, len)` CHAR span of the colour swatch block within a colour
/// row's line laid out at `width`, for the painter to tint. `None` for every
/// other value type, and for the COMPACT colour form, which drops the block
/// (there is then nothing to tint).
#[must_use]
pub fn swatch_span_in(label: &str, value: &RowValue, width: usize) -> Option<(usize, usize)> {
    if !matches!(value, RowValue::Color(_)) {
        return None;
    }
    let blocks = SWATCH.chars().count();
    let mut at = 0usize;
    for s in zones_in(label, value, width) {
        if s.zone == RowZone::Widget(WidgetPart::Swatch) {
            let n = s.text.chars().count();
            // The block sits at the END of the `#RRGGBB ██` run; the compact
            // form has no block at all.
            return s.text.ends_with(SWATCH).then(|| (at + n - blocks, blocks));
        }
        at += s.text.chars().count();
    }
    None
}

/// Clamp and round a parsed number into an [`RowValue::Int`] row's bounds.
#[must_use]
pub fn clamp_int(v: i128, min: i128, max: i128) -> i128 {
    v.clamp(min, max)
}

/// Clamp a parsed number into a [`RowValue::Float`] row's bounds. A NaN input
/// resolves to `min` (a NaN can never satisfy a clamp, and silently writing one
/// into the level would poison the `.lvl`).
#[must_use]
pub const fn clamp_float(v: f32, min: f32, max: f32) -> f32 {
    if v.is_nan() {
        return min;
    }
    v.clamp(min, max)
}

/// Parse a `#RRGGBB` / `RRGGBB` hex colour (case-insensitive, optional `#`).
/// `None` when the text is not exactly six hex digits — the colour overlay
/// REFUSES to commit rather than writing a half-parsed value.
#[must_use]
pub fn parse_hex(text: &str) -> Option<u32> {
    let t = text.trim();
    let t = t.strip_prefix('#').unwrap_or(t);
    if t.len() != 6 || !t.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    u32::from_str_radix(t, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_renders_exactly_as_the_old_string_did() {
        let v = RowValue::Text("hello".into());
        assert_eq!(row_text_in("label", &v, usize::MAX), "label = hello");
        // Every offset is Label — a Text row is click-to-focus, as before.
        for i in 0..20 {
            assert_eq!(zone_at_in("label", &v, usize::MAX, i), RowZone::Label);
        }
    }

    #[test]
    fn bool_renders_a_checkbox_and_hits_the_toggle() {
        assert_eq!(
            row_text_in("once", &RowValue::Bool(true), usize::MAX),
            "once = [x]"
        );
        assert_eq!(
            row_text_in("once", &RowValue::Bool(false), usize::MAX),
            "once = [ ]"
        );
        let v = RowValue::Bool(true);
        // "once = " is 7 chars; the checkbox is 7..10.
        assert_eq!(zone_at_in("once", &v, usize::MAX, 6), RowZone::Label);
        for i in 7..10 {
            assert_eq!(
                zone_at_in("once", &v, usize::MAX, i),
                RowZone::Widget(WidgetPart::Toggle),
                "col {i}"
            );
        }
        assert_eq!(
            zone_at_in("once", &v, usize::MAX, 10),
            RowZone::Label,
            "past the end focuses"
        );
    }

    #[test]
    fn enum_renders_the_option_affordance() {
        let v = RowValue::Enum {
            current: "box".into(),
            options: vec!["box".into(), "sphere".into()],
        };
        assert_eq!(row_text_in("shape", &v, usize::MAX), "shape = box [< >]");
        // "shape = box " is 12 chars; "[< >]" is 12..17.
        assert_eq!(zone_at_in("shape", &v, usize::MAX, 11), RowZone::Label);
        for i in 12..17 {
            assert_eq!(
                zone_at_in("shape", &v, usize::MAX, i),
                RowZone::Widget(WidgetPart::Options),
                "col {i}"
            );
        }
    }

    #[test]
    fn numeric_renders_steppers_around_a_clickable_value() {
        let v = RowValue::Int {
            v: 42,
            min: 0,
            max: 99,
            step: 1,
        };
        assert_eq!(row_text_in("rank", &v, usize::MAX), "rank = [-] 42 [+]");
        assert_eq!(
            zone_at_in("rank", &v, usize::MAX, 7),
            RowZone::Widget(WidgetPart::Dec)
        );
        assert_eq!(
            zone_at_in("rank", &v, usize::MAX, 9),
            RowZone::Widget(WidgetPart::Dec)
        );
        assert_eq!(
            zone_at_in("rank", &v, usize::MAX, 10),
            RowZone::Label,
            "the gap"
        );
        assert_eq!(
            zone_at_in("rank", &v, usize::MAX, 11),
            RowZone::Widget(WidgetPart::Value)
        );
        assert_eq!(
            zone_at_in("rank", &v, usize::MAX, 12),
            RowZone::Widget(WidgetPart::Value)
        );
        assert_eq!(
            zone_at_in("rank", &v, usize::MAX, 14),
            RowZone::Widget(WidgetPart::Inc)
        );
        assert_eq!(
            zone_at_in("rank", &v, usize::MAX, 16),
            RowZone::Widget(WidgetPart::Inc)
        );
    }

    #[test]
    fn float_keeps_its_field_specific_decimal_places() {
        let v = RowValue::Float {
            v: 0.4,
            min: 0.0,
            max: 1.0,
            step: 0.1,
            dp: 2,
        };
        assert_eq!(
            row_text_in("density", &v, usize::MAX),
            "density = [-] 0.40 [+]"
        );
        let v0 = RowValue::Float {
            v: 40.0,
            min: 0.0,
            max: 180.0,
            step: 5.0,
            dp: 0,
        };
        assert_eq!(v0.display(), "40");
    }

    #[test]
    fn colour_renders_hex_plus_a_block_swatch() {
        let v = RowValue::Color(0x00FF_4040);
        assert_eq!(
            row_text_in("color", &v, usize::MAX),
            "color = #FF4040 \u{2588}\u{2588}"
        );
        // The whole `#RRGGBB ██` run opens the overlay.
        assert_eq!(
            zone_at_in("color", &v, usize::MAX, 8),
            RowZone::Widget(WidgetPart::Swatch)
        );
        assert_eq!(
            zone_at_in("color", &v, usize::MAX, 16),
            RowZone::Widget(WidgetPart::Swatch)
        );
        // The painter tints exactly the two block cells.
        let (start, len) =
            swatch_span_in("color", &v, usize::MAX).expect("colour row has a swatch");
        assert_eq!(len, 2);
        let line: Vec<char> = row_text_in("color", &v, usize::MAX).chars().collect();
        assert_eq!(line[start], '\u{2588}');
        assert_eq!(line[start + 1], '\u{2588}');
        assert!(swatch_span_in("color", &RowValue::Bool(true), usize::MAX).is_none());
    }

    /// The alpha byte never leaks into the rendering or into a comparison.
    #[test]
    fn colour_display_masks_the_high_byte() {
        assert_eq!(RowValue::Color(0xFF_123456).display(), "#123456");
    }

    #[test]
    fn zones_concatenate_to_the_rendered_line() {
        for v in [
            RowValue::Text("t".into()),
            RowValue::Bool(false),
            RowValue::Enum {
                current: "a".into(),
                options: vec!["a".into()],
            },
            RowValue::Int {
                v: -3,
                min: -9,
                max: 9,
                step: 1,
            },
            RowValue::Float {
                v: 1.5,
                min: 0.0,
                max: 2.0,
                step: 0.5,
                dp: 2,
            },
            RowValue::Color(0x00_00FF00),
        ] {
            let joined: String = zones_in("lab", &v, usize::MAX)
                .into_iter()
                .map(|s| s.text)
                .collect();
            assert_eq!(joined, row_text_in("lab", &v, usize::MAX), "{v:?}");
        }
    }

    /// Every column of the rendered line resolves to SOME zone, and the widget
    /// zones are contiguous — no off-by-one hole a click could fall through.
    #[test]
    fn every_column_of_every_widget_resolves() {
        for v in [
            RowValue::Bool(true),
            RowValue::Enum {
                current: "opt".into(),
                options: vec![],
            },
            RowValue::Int {
                v: 0,
                min: 0,
                max: 1,
                step: 1,
            },
            RowValue::Color(0x00_ABCDEF),
        ] {
            let n = row_text_in("lab", &v, usize::MAX).chars().count();
            let hits: Vec<RowZone> = (0..n)
                .map(|i| zone_at_in("lab", &v, usize::MAX, i))
                .collect();
            assert!(
                hits.iter().any(|z| matches!(z, RowZone::Widget(_))),
                "{v:?} has no clickable widget"
            );
        }
    }

    #[test]
    fn hex_parsing_accepts_both_spellings_and_refuses_junk() {
        assert_eq!(parse_hex("#FF4040"), Some(0x00FF_4040));
        assert_eq!(parse_hex("ff4040"), Some(0x00FF_4040));
        assert_eq!(parse_hex("  #00Ff40 "), Some(0x0000_FF40));
        assert_eq!(parse_hex("#FF404"), None, "five digits");
        assert_eq!(parse_hex("#FF40400"), None, "seven digits");
        assert_eq!(parse_hex("#GGGGGG"), None, "not hex");
        assert_eq!(parse_hex(""), None);
    }

    #[test]
    fn clamping_pins_into_bounds_and_survives_nan() {
        assert_eq!(clamp_int(500, 0, 99), 99);
        assert_eq!(clamp_int(-500, 0, 99), 0);
        assert_eq!(clamp_int(7, 0, 99), 7);
        assert!((clamp_float(9.0, 0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((clamp_float(-9.0, 0.0, 1.0)).abs() < 1e-6);
        assert!((clamp_float(f32::NAN, 0.25, 1.0) - 0.25).abs() < 1e-6);
    }
}
