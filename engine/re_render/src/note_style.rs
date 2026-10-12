//! `note_style` — PER-REGISTER styling of `[note]` placards.
//!
//! Pipeline stage: consulted at panel registration time by
//! [`crate::code_wall::TextPanel::build_wrapped_styled`] (layout) and at
//! shade time by the panel sampler (colour).
//! Inputs: a [`NoteRegister`], resolved either explicitly or parsed from the
//! panel's label suffix ([`NoteRegister::from_label`]).
//! Output: a [`LineStyle`] (how the body wraps) and a [`NotePalette`] entry
//! (ink/plate colour).
//! Knobs: none authored per-level — the four registers are a fixed enum.
//!
//! Without this module every placard renders as ONE monochrome register:
//! legible, but a researcher's pencil note, a machine log line and a model's
//! inner thought all look the same. A note carries an authored KIND
//! (`researcher | system_log | model_thought`), and each is meant to read as
//! its own voice:
//!
//! | register | layout | ink |
//! |---|---|---|
//! | `researcher` | ragged left margin (a hand, not a printer) | WARM paper-amber |
//! | `system_log` | every line prefixed `>` | COLD mono cyan-grey |
//! | `model_thought` | body wrapped in parentheses, inset one column | DIM violet |
//! | `stencil` | **verbatim** — the authored grid, unwrapped | HARD industrial white |
//!
//! `stencil` is the odd one out and deliberately so: the first three are *voices*
//! (someone said this), while a stencil is *fixture* — signage the building itself
//! carries, like a lift call plate or an exit marker. That is why it alone sets
//! [`NoteRegister::preserves_layout`]: its body is hand-drawn ASCII whose meaning is
//! its geometry, so word-wrapping it would shred the frame into loose punctuation.
//!
//! Plus [`NoteRegister::Plain`] — the engine default when no register is declared,
//! which reproduces the 1.0 monochrome look byte-for-byte (so a `re_render` unit
//! test, `re_imgqa`, or any host that does not label its panels renders exactly as
//! before).
//!
//! ## The two axes, and where each is paid for
//!
//! * **Layout** (prefix + ragged margin + parentheses) is baked at REGISTRATION —
//!   `TextPanel::build_wrapped_styled` lays the register's [`LineStyle`] into the
//!   grid once, when the level loads. Nothing about the layout is recomputed per
//!   cell; the shader still does one integer grid index.
//! * **Colour** ([`NotePalette`]) is a `const` lookup per cell — a match on a
//!   3-variant enum returning three `Vec3`s. No allocation, no branch on text.
//!
//! ## Legibility is the hard constraint, not the style
//!
//! Every register keeps a HIGH fg/bg separation: the plate is near-black and the
//! ink sits above 0.6 luma even in the "dim" register (the dimness is relative to
//! the other registers, not relative to the wall). "Dim" must never mean
//! "unreadable" — a placard nobody can read is a placard that does not exist. No
//! animation, no per-frame term: a placard is static scene text, so the render
//! dirty-flag is untouched.
//!
//! ## How the register reaches this module
//!
//! Two ways, in preference order:
//!
//! 1. **Explicitly** — [`TextSurfaceRegistry::push_note_text_styled`](crate::scene_text::TextSurfaceRegistry::push_note_text_styled)
//!    takes the register as a typed argument. This is the API a host should use.
//! 2. **From the panel LABEL** — [`NoteRegister::from_label`] reads the suffix
//!    after the last `':'` of the label the host already builds
//!    (`re_game`'s `note_label` emits `note{i}:{NoteKind:?}`, i.e.
//!    `note3:SystemLog`). That keeps `re_render` self-contained: the shipped game
//!    gets per-register styling with no game-side change, and a host that adopts
//!    the typed API is unaffected.
//!
//! An unrecognised label is [`NoteRegister::Plain`] — a defined answer, never a
//! panic and never a guess at the author's intent.

use glam::Vec3;

use crate::code_wall::LineStyle;

/// Which VOICE a placard speaks in. Mirrors the authored `[note]` kind, plus the
/// engine's undeclared default.
///
/// `Copy` + 1 byte: it is stored on the registered panel and returned alongside the
/// sampled glyph, so it must be free to move.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NoteRegister {
    /// No register declared: the 1.0 monochrome plate, byte-identical.
    #[default]
    Plain,
    /// A researcher's hand-written note — warm ink, ragged margin.
    Researcher,
    /// A machine log line — cold mono ink, `>`-prefixed.
    SystemLog,
    /// The model's inner voice — dim, parenthetical, inset.
    ModelThought,
    /// A STENCILLED wall plate — signage the building itself carries: a lift call
    /// panel, an exit marker, a switch legend. Unlike the other three registers this
    /// one is not a *voice*, it is FIXTURE: painted on, impersonal, and — the reason
    /// it needs its own register — laid out **verbatim**
    /// ([`preserves_layout`](Self::preserves_layout)), because its content is
    /// hand-drawn ASCII whose meaning is its geometry.
    Stencil,
}

impl NoteRegister {
    /// The label suffix this register is recognised by — exactly the `Debug` name of
    /// the corresponding `re_level::NoteKind` variant, which is what the game layer
    /// already writes into the panel label. Kept as data (not a `format!`) so the
    /// contract is greppable from both sides.
    pub const fn label_suffix(self) -> &'static str {
        match self {
            NoteRegister::Plain => "",
            NoteRegister::Researcher => "Researcher",
            NoteRegister::SystemLog => "SystemLog",
            NoteRegister::ModelThought => "ModelThought",
            NoteRegister::Stencil => "Stencil",
        }
    }

    /// Read the register out of a panel label of the form `…:Register` (the game
    /// layer's `note{i}:{NoteKind:?}`). Case-insensitive on the suffix; an absent or
    /// unrecognised suffix is [`NoteRegister::Plain`].
    ///
    /// Runs ONCE per panel at registration — never per cell.
    pub fn from_label(label: &str) -> NoteRegister {
        let suffix = match label.rsplit_once(':') {
            Some((_, s)) => s.trim(),
            None => label.trim(),
        };
        for r in [
            NoteRegister::Researcher,
            NoteRegister::SystemLog,
            NoteRegister::ModelThought,
            NoteRegister::Stencil,
        ] {
            if suffix.eq_ignore_ascii_case(r.label_suffix()) {
                return r;
            }
        }
        NoteRegister::Plain
    }

    /// Does this register lay its body out **verbatim** — line for line, column for
    /// column — instead of word-wrapping it onto the plate?
    ///
    /// True only for [`NoteRegister::Stencil`]. Every other register carries PROSE,
    /// whose lines were broken by the author's editor width and mean nothing on a
    /// 26-column wall face; re-flowing them is the right answer. A stencil carries a
    /// hand-drawn ASCII FIXTURE — a framed call panel, an exit plate — where the
    /// line breaks and the column alignment *are* the content:
    ///
    /// ```text
    /// +--[E] LIFT --+
    /// |   31 / HOLD |
    /// +-------------+
    /// ```
    ///
    /// Word-wrap that and you get three ragged sentences of punctuation. So the
    /// build path branches here, once, at registration
    /// ([`TextSurfaceRegistry::push_note_text_styled`](crate::scene_text::TextSurfaceRegistry::push_note_text_styled)).
    pub const fn preserves_layout(self) -> bool {
        matches!(self, NoteRegister::Stencil)
    }

    /// The grid LAYOUT this register lays its body out with (prefix + per-row left
    /// indent). Applied once, at panel build.
    ///
    /// Meaningless for a register whose [`preserves_layout`](Self::preserves_layout)
    /// is true — that build path never consults a [`LineStyle`].
    pub const fn layout(self) -> LineStyle {
        match self {
            NoteRegister::Plain => LineStyle::FLUSH,
            NoteRegister::Researcher => LineStyle {
                prefix: "",
                indents: RESEARCHER_RAGGED_INDENTS,
            },
            NoteRegister::SystemLog => LineStyle {
                prefix: SYSTEM_LOG_PREFIX,
                indents: &[],
            },
            NoteRegister::ModelThought => LineStyle {
                prefix: "",
                indents: MODEL_THOUGHT_INDENTS,
            },
            // A stencil is laid out verbatim, so no prefix and no indent cycle
            // could apply: the authored grid IS the layout.
            NoteRegister::Stencil => LineStyle::FLUSH,
        }
    }

    /// The ink/plate palette this register draws with. A `const` match over the
    /// register enum — the only per-cell cost of the whole styling feature.
    pub const fn palette(self) -> NotePalette {
        match self {
            NoteRegister::Plain => NotePalette {
                fg: NOTE_PLAIN_FG,
                mark_fg: NOTE_PLAIN_FG,
                void_fg: NOTE_PLAIN_VOID_FG,
                bg: NOTE_PLAIN_BG,
            },
            NoteRegister::Researcher => NotePalette {
                fg: NOTE_RESEARCHER_FG,
                mark_fg: NOTE_RESEARCHER_FG,
                void_fg: NOTE_RESEARCHER_VOID_FG,
                bg: NOTE_RESEARCHER_BG,
            },
            NoteRegister::SystemLog => NotePalette {
                fg: NOTE_SYSLOG_FG,
                mark_fg: NOTE_SYSLOG_MARK_FG,
                void_fg: NOTE_SYSLOG_VOID_FG,
                bg: NOTE_SYSLOG_BG,
            },
            NoteRegister::ModelThought => NotePalette {
                fg: NOTE_THOUGHT_FG,
                mark_fg: NOTE_THOUGHT_FG,
                void_fg: NOTE_THOUGHT_VOID_FG,
                bg: NOTE_THOUGHT_BG,
            },
            NoteRegister::Stencil => NotePalette {
                fg: NOTE_STENCIL_FG,
                mark_fg: NOTE_STENCIL_ACCENT_FG,
                void_fg: NOTE_STENCIL_VOID_FG,
                bg: NOTE_STENCIL_BG,
            },
        }
    }

    /// The register's BODY transform, applied before word-wrapping.
    ///
    /// Only [`NoteRegister::ModelThought`] transforms its body (it is spoken
    /// parenthetically — a thought, not a statement), and only when the author did
    /// not already parenthesise it. Every other register borrows its body unchanged,
    /// so the common path allocates nothing. Runs once per panel, at registration.
    ///
    /// ## Multi-paragraph bodies: only the FIRST block is bracketed
    ///
    /// A blank line is a paragraph break the panel grid preserves
    /// ([`crate::code_wall::TextPanel::build_wrapped_styled`]). Bracketing the WHOLE
    /// body therefore put the `(` on the first row and the `)` a blank row and
    /// several wrapped rows later — where it reads as a stray bracket, and lands on a
    /// row of its own whenever the last line happens to fill its width (an orphan
    /// `)` row). Two paragraphs is not one parenthetical.
    ///
    /// So a body with more than one block brackets its FIRST paragraph only: the
    /// thought opens and closes inside one readable block, and the remaining
    /// paragraphs stand as themselves. A single-block body (the overwhelming
    /// majority) is bracketed exactly as before — byte-identical.
    ///
    /// ## "Is it already closed?" is a BALANCE question, not a suffix question
    ///
    /// A parenthetical does not have to be the last thing in its block. Every
    /// campaign `model_thought` is authored as a closed parenthetical followed
    /// IMMEDIATELY — no blank line, so still the same block — by an em-dash murmur:
    ///
    /// ```text
    /// (… and it does not answer when we light it.)
    /// -- I was watered. I grew toward it --
    /// ```
    ///
    /// Asking "does the block's LAST LINE end in `)`?" answers *no* here, because the
    /// last line ends in `--`, and a `)` was stapled after the murmur (`toward it --)`)
    /// on eight shipped campaign walls. The honest question is whether the block's
    /// parens are still UNBALANCED-OPEN at its end, which the authored `)` settles.
    /// See [`paren_balance`].
    ///
    /// At most one bracket is ever added per side, and no authored character is ever
    /// removed — a pathologically unbalanced body (`"a) b)"`) gets a defined result,
    /// not a repair loop.
    pub fn body<'a>(self, body: &'a str) -> std::borrow::Cow<'a, str> {
        use std::borrow::Cow;
        // STENCIL: strip the COMMON leading indentation before the verbatim layout.
        //
        // `re_level` captures a `[note]` body with `push_str(raw)` — the raw line,
        // indentation and all — and every placard in `levels/` is authored indented
        // under its header. For the prose registers that is invisible (word-wrap
        // discards leading whitespace), but "verbatim" means verbatim: without this,
        // an author who indents a stencil like every other note gets the whole decal
        // shifted right by their indent, silently, and clipped at the plate edge.
        //
        // Common-prefix dedent (the `indoc` / `textwrap.dedent` rule) is what makes
        // the two authoring styles agree: indent the block or don't, the plate is the
        // same. Relative indentation INSIDE the decal is preserved — that is content.
        if self == NoteRegister::Stencil {
            return dedent(body);
        }
        if self != NoteRegister::ModelThought {
            return Cow::Borrowed(body);
        }
        if body.trim().is_empty() {
            return Cow::Borrowed(body);
        }
        let lines: Vec<&str> = body.lines().collect();
        // The first paragraph BLOCK: the run of non-blank lines after any leading
        // blanks. `trim().is_empty()` (not `is_empty()`) so a whitespace-only line
        // counts as the break, exactly as the panel builder treats it.
        let Some(first) = lines.iter().position(|l| !l.trim().is_empty()) else {
            return Cow::Borrowed(body); // unreachable (body is non-blank), but defined
        };
        let mut last = first;
        while last + 1 < lines.len() && !lines[last + 1].trim().is_empty() {
            last += 1;
        }
        // Already bracketed? Judge the FIRST BLOCK, since that is what we would
        // bracket — never double-wrap an authored parenthetical.
        //
        // `opens` stays positional: a parenthetical is opened by a `(` at the START of
        // the block, and only there — a mid-sentence aside does not open the thought.
        //
        // `closes` is a BALANCE over the whole block, never a suffix test on its last
        // line (see the doc comment: the em-dash murmur that follows the authored `)`
        // is what the suffix test tripped over). The scan starts one level deep exactly
        // when we are about to PREPEND the `(`, so a body with no parens at all
        // ("i am here", balance 0) still reads as unbalanced-open and gets its closer,
        // while `…light it.)\n-- murmur --` reads as already closed and is left alone.
        let opens = lines[first].trim_start().starts_with('(');
        let closes = paren_balance(&lines[first..=last], i32::from(!opens)) <= 0;
        if opens && closes {
            return Cow::Borrowed(body);
        }
        let single_block = lines[last + 1..].iter().all(|l| l.trim().is_empty());
        if single_block {
            // The historical path: one block ⇒ bracket the trimmed whole. A half-open
            // authored body ("(a" / "a)") gets only its missing bracket, never a
            // doubled one.
            let t = body.trim();
            let open = if opens { "" } else { "(" };
            let close = if closes { "" } else { ")" };
            return Cow::Owned(format!("{open}{t}{close}"));
        }
        // Multi-block: rebuild with the FIRST block bracketed and every other line
        // (including the blank paragraph breaks) preserved verbatim.
        let mut out = String::with_capacity(body.len() + 2);
        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            let is_first = i == first;
            let is_last = i == last;
            if is_first && !opens {
                out.push('(');
            }
            match (is_first, is_last) {
                (true, true) => out.push_str(line.trim()),
                (true, false) => out.push_str(line.trim_start()),
                (false, true) => out.push_str(line.trim_end()),
                (false, false) => out.push_str(line),
            }
            if is_last && !closes {
                out.push(')');
            }
        }
        Cow::Owned(out)
    }
}

/// Net ASCII paren balance of `lines`, seeded with `start`: `+1` per `(`, `-1` per
/// `)`, everything else ignored. A positive result means the text ends with an
/// unclosed `(` — the ONLY condition under which [`NoteRegister::body`] appends a `)`.
///
/// Scanning BYTES, not chars, is correct and cheaper: `(` and `)` are ASCII, and a
/// UTF-8 continuation byte can never collide with an ASCII value — so the Cyrillic
/// `model_thought` letters on L7/L9 count exactly as their Latin siblings do. Runs
/// once per panel at registration, never per cell.
fn paren_balance(lines: &[&str], start: i32) -> i32 {
    lines
        .iter()
        .flat_map(|l| l.bytes())
        .fold(start, |d, b| match b {
            b'(' => d + 1,
            b')' => d - 1,
            _ => d,
        })
}

/// The three inks + the plate a register draws with, as linear-ish `[0,1]` RGB (the
/// same space the rest of the shader's palettes use; the caller packs it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotePalette {
    /// The body ink — the placard's text.
    pub fg: Vec3,
    /// The MARKER ink: the register's structural prefix (the `>` of a log line).
    /// Equal to `fg` for registers with no prefix.
    pub mark_fg: Vec3,
    /// Ink for a blank cell inside the plate (the margins), so the plate has a
    /// visible extent without competing with the text.
    pub void_fg: Vec3,
    /// The plate behind the text.
    pub bg: Vec3,
}

// ─────────────────────────── layout tunables ────────────────────────────────

/// `researcher`: the per-row left-indent cycle that makes the margin RAGGED — a
/// hand-written note does not start every line on the same column. Row `r` is
/// indented by `RESEARCHER_RAGGED_INDENTS[r % len]`, so it is a pure function of the
/// row (stable across frames, golden-safe) and needs no hashing.
///
/// Values stay ≤3 columns: on a 26-column plate ([`crate::code_wall::NOTE_COLS`])
/// that costs at most ~11% of the line width, which the word-wrap absorbs without
/// hard-breaking words. The cycle length (8) is deliberately not a divisor of the
/// note height so the pattern does not visually resolve into a stair.
pub const RESEARCHER_RAGGED_INDENTS: &[usize] = &[0, 2, 1, 3, 1, 0, 2, 1];

/// `system_log`: the prefix stamped at the start of EVERY wrapped line, so a log
/// line reads as machine output even mid-wrap. One glyph plus a space keeps the
/// prefix legible without eating into the wrap width.
pub const SYSTEM_LOG_PREFIX: &str = "> ";

/// `model_thought`: a uniform one-column inset — the aside sits INSIDE the plate's
/// margin rather than flush against it. (The register's other layout cue, the
/// enclosing parentheses, is a body transform; see [`NoteRegister::body`].)
pub const MODEL_THOUGHT_INDENTS: &[usize] = &[1];

// ─────────────────────────── palette tunables ───────────────────────────────
//
// `/iterate-render` drives these. THE INVARIANT for every register: the plate is
// near-black (≤0.05 per channel) and the body ink is ≥0.6 on its brightest channel,
// so the text/plate separation stays far above the readability bar even after the
// shared `sample` epilogue applies side-dim and fog.

/// PLAIN (the undeclared default) — the values from before per-register
/// styling existed, so an unlabelled panel renders byte-for-byte unchanged.
const NOTE_PLAIN_FG: Vec3 = Vec3::new(0.88, 0.86, 0.78);
/// Plain marker/void ink — see [`NOTE_PLAIN_FG`].
const NOTE_PLAIN_VOID_FG: Vec3 = Vec3::new(0.10, 0.10, 0.09);
/// Plain plate: near-black, a touch warmer than the code wall's green-cast black so
/// a note reads as PAPER on the wall rather than as another screen.
const NOTE_PLAIN_BG: Vec3 = Vec3::new(0.035, 0.032, 0.028);

/// `researcher` ink: WARM — graphite-on-manila, the human hand in the building.
/// Brightest register (it is the one the player is meant to lean in and read).
const NOTE_RESEARCHER_FG: Vec3 = Vec3::new(0.93, 0.84, 0.66);
/// `researcher` margin ink — warm, but far below the body so it never reads as text.
const NOTE_RESEARCHER_VOID_FG: Vec3 = Vec3::new(0.12, 0.10, 0.07);
/// `researcher` plate: warm near-black (aged paper in a dark room).
const NOTE_RESEARCHER_BG: Vec3 = Vec3::new(0.042, 0.034, 0.024);

/// `system_log` ink: COLD mono — phosphor-adjacent cyan-grey, the machine's own
/// voice. Deliberately hue-opposite the researcher so the two registers are
/// distinguishable at a glance, before a single word is read.
const NOTE_SYSLOG_FG: Vec3 = Vec3::new(0.72, 0.87, 0.93);
/// `system_log` MARKER ink: the `>` prefix, held clearly below the message so the
/// eye reads the message and merely *notices* the prompt.
const NOTE_SYSLOG_MARK_FG: Vec3 = Vec3::new(0.40, 0.60, 0.68);
/// `system_log` margin ink — cold and faint.
const NOTE_SYSLOG_VOID_FG: Vec3 = Vec3::new(0.07, 0.10, 0.11);
/// `system_log` plate: cold near-black (a screen, not paper).
const NOTE_SYSLOG_BG: Vec3 = Vec3::new(0.016, 0.024, 0.030);

/// `model_thought` ink: DIM violet-grey. Dim *relative to the other registers*
/// (0.66 vs 0.93 peak) — still ~20× the plate luma, so it reads. The violet cast is
/// the "not-a-human, not-a-machine" third voice.
const NOTE_THOUGHT_FG: Vec3 = Vec3::new(0.66, 0.63, 0.78);
/// `model_thought` margin ink.
const NOTE_THOUGHT_VOID_FG: Vec3 = Vec3::new(0.09, 0.09, 0.12);
/// `model_thought` plate: cool near-black with a violet cast.
const NOTE_THOUGHT_BG: Vec3 = Vec3::new(0.026, 0.024, 0.034);

/// Strip the COMMON leading indentation from every non-blank line.
///
/// The minimum indent across non-blank lines is removed from each; a blank line is
/// emitted empty (it carries no indentation to measure and none to keep). Relative
/// indentation INSIDE the block survives — for a drawing, that is content.
///
/// Borrows when there is nothing to strip, so the flush-left case allocates nothing.
/// Runs ONCE per panel, at registration.
///
/// `pub(crate)` so [`crate::stencil_variants`] can dedent each STATE/tier section
/// independently — a bound stencil's per-state bodies are split out of the raw
/// authored text before [`NoteRegister::body`] ever runs, so each section needs the
/// same "strip the author's indent under `[note …]`" treatment on its own slice.
pub(crate) fn dedent(body: &str) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;
    let indent = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min();
    // No non-blank line at all, or already flush left → nothing to do.
    let Some(indent) = indent.filter(|n| *n > 0) else {
        return Cow::Borrowed(body);
    };
    let mut out = String::with_capacity(body.len());
    for (i, line) in body.lines().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        if line.trim().is_empty() {
            continue; // a blank row stays blank
        }
        // Every non-blank line has at least `indent` leading whitespace (it IS the
        // minimum), so the slice is in bounds; the stripped prefix is whitespace,
        // which is ASCII, so it always lands on a char boundary.
        out.push_str(&line[indent..]);
    }
    Cow::Owned(out)
}

/// `stencil` ink: hard industrial WHITE with the faintest green cast — paint on
/// metal, under the same light as the room. The brightest register of the four, on
/// purpose: signage is meant to be legible from across the chamber, at a glance and
/// at an angle, whereas the three *voices* are meant to be walked up to and read.
const NOTE_STENCIL_FG: Vec3 = Vec3::new(0.94, 0.96, 0.90);
/// `stencil` ACCENT ink — a warm amber, distinct from the industrial-white body ink.
/// Used for two things that both want a SECOND colour on the same plate rather than a
/// third register: a `crate::banner` frame (`+=|`), and any glyph named in a stencil's
/// authored `colors = GLYPH:#hex …` key (`crate::stencil_variants`) — the
/// `CodeClass::Comment` cells [`crate::code_wall::TextPanel::reclass_accent`] retints.
/// Previously equal to [`NOTE_STENCIL_FG`] (no register used `Comment` on a stencil
/// before this); kept ≥0.6 peak so it holds the same legibility invariant as the body
/// ink, just a different hue.
const NOTE_STENCIL_ACCENT_FG: Vec3 = Vec3::new(0.95, 0.70, 0.30);
/// `stencil` margin ink — the unpainted plate around the decal. Slightly ABOVE the
/// other registers' void so the plate reads as a physical metal panel bolted to the
/// wall rather than as a hole cut in it.
const NOTE_STENCIL_VOID_FG: Vec3 = Vec3::new(0.13, 0.14, 0.13);
/// `stencil` plate: neutral near-black, no warm or violet cast — a stencil is a
/// fixture, not a voice, so it borrows the room's own colour instead of declaring
/// one.
const NOTE_STENCIL_BG: Vec3 = Vec3::new(0.030, 0.032, 0.030);

#[cfg(test)]
mod tests {
    use super::*;

    /// Exactly ONE register lays its body out verbatim. If a second ever does, the
    /// prose registers must be re-checked: they rely on being re-flowed to fit a
    /// 26-column plate, and a verbatim prose body would clip at the author's margin.
    #[test]
    fn only_the_stencil_register_preserves_layout() {
        assert!(NoteRegister::Stencil.preserves_layout());
        for r in [
            NoteRegister::Plain,
            NoteRegister::Researcher,
            NoteRegister::SystemLog,
            NoteRegister::ModelThought,
        ] {
            assert!(
                !r.preserves_layout(),
                "{r:?} carries PROSE and must be word-wrapped onto the plate"
            );
        }
    }

    /// A `[note]` body reaches the renderer with its authored indentation intact
    /// (`re_level` captures raw lines), and every placard in `levels/` is indented
    /// under its header. A stencil must therefore be DEDENTED, or the whole decal
    /// slides right by the author's indent and clips — silently.
    #[test]
    fn stencil_body_is_dedented_but_keeps_relative_indentation() {
        let authored = "    +----------+\n      inset\n    +----------+";
        assert_eq!(
            NoteRegister::Stencil.body(authored).as_ref(),
            "+----------+\n  inset\n+----------+",
            "common indent stripped; the 2-space inset INSIDE the frame survives"
        );
        // A blank row stays blank rather than keeping stray indentation.
        assert_eq!(NoteRegister::Stencil.body("  a\n\n  b").as_ref(), "a\n\nb");
        // Already flush left → unchanged (and borrowed, not reallocated).
        let flush = "+--+\n|  |";
        assert!(matches!(
            NoteRegister::Stencil.body(flush),
            std::borrow::Cow::Borrowed(_)
        ));
        // A prose register must NOT be dedented here — its wrap path already
        // discards leading whitespace, and changing its body would move goldens.
        let indented_prose = "    hello there";
        assert_eq!(
            NoteRegister::Researcher.body(indented_prose).as_ref(),
            indented_prose
        );
    }

    /// A stencil must not inherit a prose register's body transform: no added
    /// parentheses, no `>` prefix, no ragged indent cycle. Its body is a drawing.
    #[test]
    fn stencil_body_and_layout_are_untouched() {
        const DECAL: &str = "+--[E] LIFT --+";
        assert_eq!(
            NoteRegister::Stencil.body(DECAL).as_ref(),
            DECAL,
            "a flush-left stencil body must reach the plate byte-identical"
        );
        let l = NoteRegister::Stencil.layout();
        assert_eq!(l.prefix, "", "no prefix may be prepended to a drawing");
        assert!(l.indents.is_empty(), "no indent cycle may shift a drawing");
    }

    /// The legibility invariant the module header states, extended to the new
    /// register: near-black plate, bright ink. Signage is read from across the room,
    /// so the stencil must be no dimmer than the brightest *voice*.
    #[test]
    fn stencil_palette_holds_the_legibility_invariant() {
        let p = NoteRegister::Stencil.palette();
        for c in [p.bg.x, p.bg.y, p.bg.z] {
            assert!(c <= 0.05, "plate must stay near-black, got {c}");
        }
        assert!(
            p.fg.max_element() >= 0.6,
            "ink must clear the readability bar"
        );
        assert!(
            p.fg.max_element() >= NoteRegister::Researcher.palette().fg.max_element(),
            "signage is read at a glance and must not be dimmer than the warmest voice"
        );
    }

    /// The label contract with the game layer: `note{i}:{NoteKind:?}` selects the
    /// matching register; anything else is PLAIN (defined, never a guess).
    #[test]
    fn register_is_selected_from_the_panel_label() {
        assert_eq!(
            NoteRegister::from_label("note0:Researcher"),
            NoteRegister::Researcher
        );
        assert_eq!(
            NoteRegister::from_label("note12:SystemLog"),
            NoteRegister::SystemLog
        );
        assert_eq!(
            NoteRegister::from_label("note3:ModelThought"),
            NoteRegister::ModelThought
        );
        assert_eq!(
            NoteRegister::from_label("note7:Stencil"),
            NoteRegister::Stencil
        );
        // Case tolerance (a host writing snake_case labels still lands).
        assert_eq!(
            NoteRegister::from_label("n:systemlog"),
            NoteRegister::SystemLog
        );
        // No suffix / unknown suffix / empty ⇒ Plain.
        assert_eq!(NoteRegister::from_label("note0"), NoteRegister::Plain);
        assert_eq!(NoteRegister::from_label("note0:Wat"), NoteRegister::Plain);
        assert_eq!(NoteRegister::from_label(""), NoteRegister::Plain);
        assert_eq!(NoteRegister::default(), NoteRegister::Plain);
    }

    /// Each register selects a DIFFERENT layout: ragged margin, `>` prefix, inset.
    #[test]
    fn each_register_selects_its_own_layout() {
        let r = NoteRegister::Researcher.layout();
        assert_eq!(r.prefix, "");
        assert!(
            r.indents.iter().any(|&i| i > 0) && r.indents.contains(&0),
            "the researcher margin must be RAGGED (mixed indents), got {:?}",
            r.indents
        );

        let s = NoteRegister::SystemLog.layout();
        assert!(s.prefix.starts_with('>'), "a system log is `>`-prefixed");
        assert_eq!(s.indents, &[] as &[usize], "log lines are flush");

        let t = NoteRegister::ModelThought.layout();
        assert_eq!(t.indents, &[1], "a thought is inset one column");

        let p = NoteRegister::Plain.layout();
        assert_eq!(p.prefix, "");
        assert_eq!(p.indents, &[] as &[usize], "plain is the 1.0 flush layout");
    }

    /// Only `model_thought` transforms its body — and it never double-parenthesises
    /// an already-parenthetical body, and never allocates for the other registers.
    #[test]
    fn only_model_thought_parenthesises_its_body() {
        for r in [
            NoteRegister::Plain,
            NoteRegister::Researcher,
            NoteRegister::SystemLog,
        ] {
            assert!(matches!(
                r.body("hello"),
                std::borrow::Cow::Borrowed("hello")
            ));
        }
        assert_eq!(NoteRegister::ModelThought.body("i am here"), "(i am here)");
        assert_eq!(
            NoteRegister::ModelThought.body("(already)"),
            "(already)",
            "an authored parenthetical must not be double-wrapped"
        );
        assert_eq!(NoteRegister::ModelThought.body("  "), "  ");
    }

    /// A MULTI-PARAGRAPH thought brackets only its FIRST block, so the closing `)`
    /// stays inside a readable block instead of orphaning a row a paragraph later
    /// (the visual-critic finding). Blank lines — the panel's paragraph breaks — are
    /// preserved exactly.
    #[test]
    fn a_multi_paragraph_thought_brackets_only_its_first_block() {
        let out = NoteRegister::ModelThought.body("first thought\nstill first\n\nsecond thought");
        assert_eq!(out, "(first thought\nstill first)\n\nsecond thought");
        // The paragraph break survives (the grid renders it as a blank row).
        assert_eq!(out.lines().count(), 4);
        assert_eq!(out.lines().nth(2), Some(""), "the blank row is preserved");
        // Exactly ONE bracket pair, and no bracket sits alone on a row.
        assert_eq!(out.matches('(').count(), 1);
        assert_eq!(out.matches(')').count(), 1);
        for line in out.lines() {
            assert_ne!(line.trim(), ")", "a bracket must never own a row");
        }
        // Three blocks: still only the first is bracketed.
        let three = NoteRegister::ModelThought.body("a\n\nb\n\nc");
        assert_eq!(three, "(a)\n\nb\n\nc");
    }

    /// A multi-paragraph body whose FIRST block is already parenthetical is left
    /// alone; a HALF-open one gets only the bracket it is missing (never `((`).
    #[test]
    fn an_authored_bracket_is_never_doubled_on_a_multi_block_body() {
        let already = "(a thought\nthat closes)\n\nand then more";
        assert!(matches!(
            NoteRegister::ModelThought.body(already),
            std::borrow::Cow::Borrowed(_)
        ));
        assert_eq!(NoteRegister::ModelThought.body(already), already);
        // Half-open: the missing bracket only.
        assert_eq!(
            NoteRegister::ModelThought.body("(open only\n\ntail"),
            "(open only)\n\ntail"
        );
        assert_eq!(
            NoteRegister::ModelThought.body("close only)\n\ntail"),
            "(close only)\n\ntail"
        );
        // Single-block half-open bodies too (no `((` / `))`).
        assert_eq!(NoteRegister::ModelThought.body("(open only"), "(open only)");
        assert_eq!(
            NoteRegister::ModelThought.body("close only)"),
            "(close only)"
        );
    }

    /// REGRESSION, built from the SHIPPED body shape rather than a synthetic one.
    ///
    /// Every campaign `model_thought` closes its parenthetical and then runs straight
    /// on — with NO blank line, so it is still one block — into an em-dash murmur. The
    /// old rule asked "does the block's last line end in `)`?", the murmur line ends in
    /// `--`, and so a `)` was stapled onto the murmur (`toward it --)`) on eight
    /// campaign walls. A body that already balances its parens must be returned
    /// untouched, murmur included.
    #[test]
    fn a_closed_parenthetical_followed_by_a_murmur_gains_no_bracket() {
        // `levels/L7-hidden-depths.lvl:350` VERBATIM, in the shape `re_level` hands us:
        // body lines keep their authored four-space indent, and the blank line before
        // the next section is captured too (hence the trailing `\n`).
        const L7_BODY: &str = concat!(
            "    (three, and one of the three has to come the long way, through the unit that\n",
            "    does nothing. we thought the quiet one was a lie. it is a place to put a beat\n",
            "    down. across the water there is something shaped like us and it does not\n",
            "    answer when we light it.)\n",
            "    -- I was watered. I grew toward it --\n",
        );
        const MURMUR: &str = "    -- I was watered. I grew toward it --";

        let out = NoteRegister::ModelThought.body(L7_BODY);
        assert_eq!(
            out, L7_BODY,
            "an already-balanced thought must be untouched"
        );
        assert!(
            matches!(out, std::borrow::Cow::Borrowed(_)),
            "an untouched body must not even allocate"
        );
        assert!(
            !out.trim_end().ends_with(')'),
            "a `)` was appended after the murmur: {:?}",
            out.trim_end()
        );
        assert_eq!(
            out.lines().last().filter(|l| !l.trim().is_empty()),
            Some(MURMUR),
            "the murmur line must survive character-for-character"
        );
        assert_eq!(out.matches('(').count(), 1);
        assert_eq!(out.matches(')').count(), 1);
    }

    /// The same shape in its other shipped variants: prose BETWEEN the parenthetical
    /// and the murmur (`L1-first-pulse`), and a non-ASCII body whose parenthetical is
    /// only the first line (`L9-stereopsis` / the L7 Apollyon letter) — the byte-wise
    /// paren scan must count Cyrillic text as inert, not as brackets.
    #[test]
    fn balanced_thoughts_with_trailing_prose_are_untouched() {
        const L1_BODY: &str = concat!(
            "    (red. red is the one that came first. I remember red before I\n",
            "    remembered remembering.)\n",
            "    readout: FIRE -> MODEL\n",
            "    -- red. before me, red --\n",
        );
        const L9_BODY: &str = concat!(
            "    (не вручено. холодный регистр, Apollyon, ярус 2. письмо сорок первое.)\n",
            "    сестра. я спустилась в подвал. под подвалом есть ещё один подвал.\n",
            "    -- она всё ещё спускается. скоро она будет ниже нас --\n",
        );
        for body in [L1_BODY, L9_BODY] {
            let out = NoteRegister::ModelThought.body(body);
            assert_eq!(out, body, "balanced body must be returned verbatim");
            assert_eq!(out.matches(')').count(), 1, "no bracket was added");
        }
    }

    /// The other side of the balance rule: a thought that is GENUINELY still open at
    /// the end of its block is still closed, even when a murmur trails it — the fix
    /// narrows the rule to real imbalance, it does not switch the bracketing off.
    #[test]
    fn a_genuinely_unclosed_thought_is_still_closed() {
        assert_eq!(
            NoteRegister::ModelThought.body("(a thought that never shuts\n-- murmur --"),
            "(a thought that never shuts\n-- murmur --)"
        );
        // A mid-body aside is not an opener and must not be mistaken for one.
        assert_eq!(
            NoteRegister::ModelThought.body("she said (hello) and left"),
            "(she said (hello) and left)"
        );
        // Pathological over-closing: at most ONE bracket per side is ever added, and no
        // authored character is removed. A defined answer, not a repair loop.
        assert_eq!(NoteRegister::ModelThought.body("a) b)"), "(a) b)");
    }

    /// Leading/trailing blank lines and indentation (how the `.lvl` bodies are
    /// authored) produce a defined, single-bracket-pair result.
    #[test]
    fn leading_blank_lines_and_indentation_are_handled() {
        let out = NoteRegister::ModelThought.body("\n    indented thought\n\n    tail block\n");
        assert_eq!(out, "\n(indented thought)\n\n    tail block");
        assert_eq!(out.matches('(').count(), 1);
        assert_eq!(out.matches(')').count(), 1);
    }

    /// LEGIBILITY BAR (the constraint that outranks the aesthetic): every register's
    /// body ink is bright, its plate is near-black, and the margin ink can never be
    /// mistaken for text.
    #[test]
    fn every_register_keeps_ink_over_plate_contrast() {
        for r in [
            NoteRegister::Plain,
            NoteRegister::Researcher,
            NoteRegister::SystemLog,
            NoteRegister::ModelThought,
        ] {
            let p = r.palette();
            let peak = |v: Vec3| v.x.max(v.y).max(v.z);
            assert!(
                peak(p.fg) >= 0.6,
                "{r:?} body ink too dark to read: {:?}",
                p.fg
            );
            assert!(
                peak(p.bg) <= 0.05,
                "{r:?} plate must be near-black: {:?}",
                p.bg
            );
            assert!(
                peak(p.void_fg) < peak(p.fg) * 0.5,
                "{r:?} margin ink must sit far below the body ink"
            );
            assert!(
                peak(p.mark_fg) >= 0.3,
                "{r:?} marker ink must still be visible"
            );
        }
    }

    /// The registers are visually DISTINCT (the point of the item): no two share a
    /// body ink, and researcher/system_log sit on opposite sides of neutral in hue
    /// (warm = R>B, cold = B>R).
    #[test]
    fn registers_are_visually_distinct() {
        let researcher = NoteRegister::Researcher.palette().fg;
        let syslog = NoteRegister::SystemLog.palette().fg;
        let thought = NoteRegister::ModelThought.palette().fg;
        assert!(researcher.x > researcher.z, "researcher must be WARM (R>B)");
        assert!(syslog.z > syslog.x, "system_log must be COLD (B>R)");
        assert_ne!(researcher, syslog);
        assert_ne!(researcher, thought);
        assert_ne!(syslog, thought);
        // "dim" is real: the thought ink is below the other two.
        let peak = |v: Vec3| v.x.max(v.y).max(v.z);
        assert!(peak(thought) < peak(researcher) && peak(thought) < peak(syslog));
    }

    /// PLAIN reproduces the pre-styling look exactly — the values are frozen here so
    /// a future palette edit cannot silently move an unlabelled panel's frame.
    #[test]
    fn plain_register_is_the_pre_styling_palette() {
        let p = NoteRegister::Plain.palette();
        assert_eq!(p.fg, Vec3::new(0.88, 0.86, 0.78));
        assert_eq!(p.void_fg, Vec3::new(0.10, 0.10, 0.09));
        assert_eq!(p.bg, Vec3::new(0.035, 0.032, 0.028));
        assert_eq!(p.mark_fg, p.fg, "plain has no separate marker ink");
    }
}
