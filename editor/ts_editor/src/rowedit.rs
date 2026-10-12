//! DIRECT-MANIPULATION EDITORS for typed inspector rows.
//!
//! Once a row carries a [`RowValue`] instead of a formatted `String`, a click or
//! an `Enter` can do something better than "focus this field". This module owns
//! the four resulting widgets:
//!
//! | Row type | Affordance | Editor |
//! |----------|-----------|--------|
//! | `Bool`   | `[x]` / `[ ]` | toggles in place (no overlay) |
//! | `Int`/`Float` | `[-] v [+]` | steps in place, or the INLINE NUMERIC editor |
//! | `Enum`   | `[< >]` | the OPTION POPUP (a scrollable list) |
//! | `Color`  | `#RRGGBB` + swatch | the COLOUR OVERLAY (hex + palette + used) |
//!
//! ## Why the editors drive the EXISTING steppers
//!
//! Every commit path here reaches the level through the same
//! `entity::adjust_property` / `entity::cycle_property` / `material::adjust_field`
//! / `material::cycle_enum` functions the keyboard already uses, rather than
//! through a second set of per-field setters. Two reasons, both about safety:
//!
//! * Those steppers already carry each field's clamp, its `validate()` call, its
//!   `mark_dirty`, and its side effects (resizing a part's mask, keeping a
//!   region's min ≤ max). A parallel setter table would have to re-derive all of
//!   that for ~190 fields and would drift the moment one stepper changed.
//! * They are the tested path. `set_numeric` converts "I want 4096" into "press
//!   `+` 240 times" arithmetically — one call with a computed delta — and then
//!   VERIFIES the result, iterating if a clamp intervened. So an editor can
//!   never write a value the keyboard could not have produced.
//!
//! Colour is the one exception: no sequence of palette steps reaches an
//! arbitrary `#RRGGBB`, so [`set_color`] writes the ten colour fields directly
//! from an explicit table.
//!
//! ## Undo
//!
//! Each editor commits as ONE undo step, using the editor's established
//! `push_undo_checkpoint` → mutate → `drop_undo_if_unchanged` pattern, so a
//! commit that changes nothing leaves no dead entry on the stack.
//!
//! NO terminal here: this module is pure state + geometry, painted by
//! `chrome::draw_*` and hit-tested by `mouse.rs`, exactly as `picker.rs` is.

use crate::chrome::RowAction;
use crate::rowvalue::RowValue;
use crate::state::EditorState;
use crate::widget::Caret;
use re_present::{EditorKey, EditorKeyCode as K};

// ── the three overlay states ────────────────────────────────────────────────

/// The INLINE NUMERIC EDITOR (`Enter` on a focused numeric row, or a click on
/// its value): a caret buffer over the row's current number.
///
/// The buffer is seeded from the row's rendered value, so what the user starts
/// editing is exactly what the row showed. `Enter` parses, CLAMPS into the row's
/// own `min`/`max`, and commits; `Esc` restores by simply dropping the state —
/// nothing was written until commit, so a cancel needs no snapshot.
#[derive(Clone, PartialEq, Debug)]
pub struct NumEdit {
    /// Which field is being edited (also names the focus register to restore).
    pub target: RowAction,
    /// The row's label, for the editor's own caption.
    pub label: String,
    /// The digits being typed.
    pub buf: String,
    /// Caret within [`NumEdit::buf`].
    pub caret: Caret,
    /// The row's clamp bounds, in the row's OWN numeric domain.
    pub bounds: NumBounds,
}

/// An inline numeric editor's clamp bounds, kept in the row's own domain.
///
/// Integer rows carry `i128` bounds, not `f64` ones: `RowValue::Int` is `i128`
/// so that a `u64` emitter seed is representable, and storing the bounds as
/// floats would reintroduce the 53-bit truncation at the clamp step.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum NumBounds {
    /// Inclusive integer bounds.
    Int {
        /// Inclusive lower clamp.
        min: i128,
        /// Inclusive upper clamp.
        max: i128,
    },
    /// Inclusive float bounds.
    Float {
        /// Inclusive lower clamp.
        min: f64,
        /// Inclusive upper clamp.
        max: f64,
    },
}

impl NumBounds {
    /// True when the row is a float — the editor then accepts a decimal point.
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(self, Self::Float { .. })
    }

    /// The `min .. max` caption the overlay shows.
    #[must_use]
    pub fn caption(&self) -> String {
        match self {
            Self::Int { min, max } => format!("range {min} .. {max}"),
            Self::Float { min, max } => format!("range {min} .. {max}"),
        }
    }
}

/// The ENUM OPTION POPUP (`Enter` on a focused enum row, or a click on `[< >]`):
/// a scrollable list of every option with the current one marked.
///
/// This is the direct answer to a real usability problem: before it, choosing one of 192
/// materials meant pressing `c` up to 192 times with no idea what was coming.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EnumPick {
    /// Which field is being picked for.
    pub target: RowAction,
    /// The row's label, for the popup caption.
    pub label: String,
    /// Every option, in the order the `c` cycle visits them.
    pub options: Vec<String>,
    /// Highlighted row index into [`EnumPick::options`].
    pub sel: usize,
}

/// Which part of the COLOUR OVERLAY has the keyboard.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorFocus {
    /// The `#RRGGBB` hex entry field.
    Hex,
    /// The 12-swatch [`crate::material::PALETTE`] grid.
    Palette,
    /// The "used in this level" row of colours already present in the level.
    Used,
}

/// The COLOUR OVERLAY (`Enter` on a focused colour row, or a click on its
/// swatch).
///
/// Three ways to reach a colour, because the 12-swatch stepper alone could not
/// reach the 234 distinct colours the shipped levels actually use:
///
/// 1. **Hex entry** — type any `#RRGGBB`. Validated on commit, never on the way.
/// 2. **The palette grid** — the same 12 swatches `+`/`-` steps, now visible and
///    clickable rather than blind.
/// 3. **Used in this level** — every distinct colour already present in the
///    level. THIS is what makes existing content reproducible: after the old
///    stepper snapped a value to white, there was no way to get the authored
///    colour back; now it is one click away as long as anything else still uses
///    it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ColorEdit {
    /// Which colour field is being edited.
    pub target: RowAction,
    /// The row's label, for the overlay caption.
    pub label: String,
    /// The hex text being typed (without the leading `#`).
    pub buf: String,
    /// Caret within [`ColorEdit::buf`].
    pub caret: Caret,
    /// The colour the overlay opened on — restored on `Esc`, and shown as the
    /// "before" swatch.
    pub original: u32,
    /// Distinct colours already present in the level, sorted (see
    /// [`used_colors`]).
    pub used: Vec<u32>,
    /// Which sub-widget has the keyboard.
    pub focus: ColorFocus,
    /// Highlighted index into [`crate::material::PALETTE`].
    pub pal_sel: usize,
    /// Highlighted index into [`ColorEdit::used`].
    pub used_sel: usize,
}

/// Columns the palette / used-colour swatch grids wrap at. 6 × 2 lays the 12
/// palette swatches out as two readable rows at every panel width we support.
pub const SWATCH_COLS: usize = 6;

/// Cells one swatch occupies in a grid, including its trailing gap.
pub const SWATCH_W: usize = 3;

// ── reading + focusing a row ────────────────────────────────────────────────

/// The typed value of the field a [`RowAction`] names, or `None` for a
/// non-field row. The ONE place the three per-family classifiers are dispatched.
#[must_use]
pub fn value_of(state: &EditorState, action: RowAction) -> Option<RowValue> {
    match action {
        RowAction::None => None,
        RowAction::FocusEntity(f) => Some(crate::rowtype::entity_value(state, f)),
        RowAction::FocusEmitter(f) => Some(crate::rowtype::emitter_value(state, f)),
        RowAction::FocusMaterial(f) => Some(crate::rowtype::material_value(state, f)),
    }
}

/// The label the field a [`RowAction`] names renders with.
#[must_use]
pub fn label_of(action: RowAction) -> String {
    match action {
        RowAction::None => String::new(),
        RowAction::FocusEntity(f) => f.label().to_string(),
        RowAction::FocusEmitter(f) => f.label().to_string(),
        RowAction::FocusMaterial(f) => f.label().to_string(),
    }
}

/// Point the matching focus register at this field. Every widget does this
/// FIRST, because the shared steppers all operate on the focused field — so a
/// click acts on exactly the row that was clicked, and the focus it leaves
/// behind is the same one a `;`/`'` page-through would have produced.
pub const fn focus(state: &mut EditorState, action: RowAction) {
    match action {
        RowAction::None => {}
        RowAction::FocusEntity(f) => state.entity_field = f,
        RowAction::FocusEmitter(f) => state.emitter_field = f,
        RowAction::FocusMaterial(f) => state.material_field = f,
    }
}

/// Run the field's `+`/`-` stepper `delta` steps.
fn step_field(state: &mut EditorState, action: RowAction, delta: i32) {
    match action {
        RowAction::None => {}
        RowAction::FocusEntity(_) | RowAction::FocusEmitter(_) => {
            crate::entity::adjust_property(state, delta);
        }
        RowAction::FocusMaterial(_) => crate::material::adjust_field(state, delta),
    }
}

/// Run the field's `c` cycle once.
fn cycle_field(state: &mut EditorState, action: RowAction) {
    match action {
        RowAction::None => {}
        RowAction::FocusEntity(_) | RowAction::FocusEmitter(_) => {
            crate::entity::cycle_property(state);
        }
        RowAction::FocusMaterial(_) => crate::material::cycle_enum(state, 1),
    }
}

/// The row's current numeric value and step, when it is an `Int`/`Float`.
fn numeric_of(state: &EditorState, action: RowAction) -> Option<(f64, f64)> {
    match value_of(state, action)? {
        RowValue::Int { v, step, .. } => Some((v as f64, step as f64)),
        RowValue::Float { v, step, .. } => Some((f64::from(v), f64::from(step))),
        RowValue::Text(_) | RowValue::Bool(_) | RowValue::Enum { .. } | RowValue::Color(_) => None,
    }
}

// ── applying a value through the existing steppers ──────────────────────────

/// Drive the field to `target` by computing how many `+`/`-` steps that is and
/// taking them in ONE stepper call, then verifying and refining.
///
/// The refine loop matters because a stepper may clamp, round, or (for a float)
/// accumulate a little error: after each attempt the row is re-read and the
/// remaining distance recomputed. It stops as soon as the value stops moving,
/// so a clamp pins cleanly instead of spinning. Bounded, so a pathological
/// stepper can never hang the editor.
///
/// Returns the value actually reached.
pub fn set_numeric(state: &mut EditorState, action: RowAction, target: f64) -> f64 {
    let Some((mut last, _)) = numeric_of(state, action) else {
        return f64::NAN;
    };
    for _ in 0..8 {
        let Some((cur, step)) = numeric_of(state, action) else {
            return last;
        };
        if step == 0.0 {
            return cur;
        }
        let d = ((target - cur) / step).round();
        if d == 0.0 {
            return cur;
        }
        // The steppers take an i32 delta; a jump wider than that is taken in
        // as large a bite as fits and refined on the next pass.
        let d = d.clamp(f64::from(i32::MIN + 1), f64::from(i32::MAX - 1)) as i32;
        step_field(state, action, d);
        let after = numeric_of(state, action).map_or(cur, |(v, _)| v);
        if (after - cur).abs() < f64::EPSILON {
            return after; // clamped — no further progress is possible
        }
        last = after;
    }
    last
}

/// Fields whose value the `+`/`-` stepper CANNOT reach in a usable number of
/// steps, written directly instead.
///
/// The only member today is the emitter `seed`: it is a `u64` stepped ±1, so
/// reaching a specific 19-digit seed through the stepper would take on the
/// order of 10^19 presses — which is exactly the "setting a specific seed is
/// not practical" finding this work exists to fix. Everything else stays on
/// the stepper path, where the field's own clamp and side effects live.
///
/// Returns `true` when the field was written here (the generic path is then
/// skipped).
fn set_numeric_direct(state: &mut EditorState, action: RowAction, want: i128) -> bool {
    if action != RowAction::FocusEmitter(crate::state::EmitterField::Seed) {
        return false;
    }
    let Some(i) = state
        .selected_emitter
        .filter(|&i| i < state.level.emitters.len())
    else {
        return false;
    };
    // A seed is an opaque bit pattern, so clamp into the u64 range rather than
    // wrapping — a typed value out of range is a typo, not an intent to alias.
    // `i128` throughout: a `u64` seed does not survive a trip through `f64`.
    let seed = want.clamp(0, i128::from(u64::MAX)) as u64;
    // Committing an UNMODIFIED buffer must be a no-op: Enter on a row the user
    // only looked at should not mark the document dirty or rebuild the preview.
    if state.level.emitters[i].seed == seed {
        return true;
    }
    state.level.emitters[i].seed = seed;
    state.level.emitters[i].validate();
    state.mark_dirty();
    state.schedule_preview_rebuild();
    true
}

/// Drive an enum field to the option named `want` by cycling it, bounded by the
/// option count so an unreachable name cannot loop forever. Returns `true` when
/// the field ended on `want`.
pub fn set_enum(state: &mut EditorState, action: RowAction, want: &str) -> bool {
    let current = |s: &EditorState| match value_of(s, action) {
        Some(RowValue::Enum { current, .. }) => current,
        Some(v) => v.display(),
        None => String::new(),
    };
    let n = match value_of(state, action) {
        Some(RowValue::Enum { options, .. }) => options.len(),
        _ => return false,
    };
    for _ in 0..=n {
        if current(state) == want {
            return true;
        }
        cycle_field(state, action);
    }
    current(state) == want
}

/// Write an arbitrary `0x00RRGGBB` into the colour field a [`RowAction`] names.
///
/// The ONE place the editor writes a colour directly rather than through a
/// stepper — no sequence of 12-swatch steps reaches an arbitrary colour, which
/// is precisely the hole that made the old snap-to-white destructive. The table
/// is exhaustive over the ten colour-bearing fields; anything else returns
/// `false` and writes nothing.
pub fn set_color(state: &mut EditorState, action: RowAction, rgb: u32) -> bool {
    use crate::state::{EmitterField as EF, EntityField as F};
    let rgb = rgb & 0xFF_FFFF;
    match action {
        RowAction::FocusEntity(F::NodeColor) => {
            let Some(i) = state.selected_node.filter(|&i| i < state.level.nodes.len()) else {
                return false;
            };
            state.level.nodes[i].color_rgb = rgb;
        }
        RowAction::FocusEntity(F::PartColor) => {
            let Some(i) = state.selected_part.filter(|&i| i < state.level.parts.len()) else {
                return false;
            };
            state.level.parts[i].color_rgb = rgb;
        }
        RowAction::FocusEntity(F::PipeColor) => {
            let Some(i) = state.selected_pipe.filter(|&i| i < state.level.pipes.len()) else {
                return false;
            };
            state.level.pipes[i].color_rgb = rgb;
        }
        RowAction::FocusEntity(F::DecalColorHead) => {
            let Some(i) = state
                .selected_decal
                .filter(|&i| i < state.level.decals.len())
            else {
                return false;
            };
            state.level.decals[i].color_head = rgb;
        }
        RowAction::FocusEntity(F::DecalColorTail) => {
            let Some(i) = state
                .selected_decal
                .filter(|&i| i < state.level.decals.len())
            else {
                return false;
            };
            state.level.decals[i].color_tail = rgb;
        }
        RowAction::FocusEmitter(f @ (EF::Color0 | EF::Color1 | EF::Color2)) => {
            let Some(i) = state
                .selected_emitter
                .filter(|&i| i < state.level.emitters.len())
            else {
                return false;
            };
            let slot = match f {
                EF::Color0 => 0,
                EF::Color1 => 1,
                _ => 2,
            };
            state.level.emitters[i].colors[slot] = rgb;
            state.level.emitters[i].validate();
        }
        RowAction::FocusMaterial(crate::material::MaterialField::Fg) => {
            let Some(i) = crate::material::sel_index(state) else {
                return false;
            };
            state.level.materials[i].fg = rgb;
        }
        RowAction::FocusMaterial(crate::material::MaterialField::Bg) => {
            let Some(i) = crate::material::sel_index(state) else {
                return false;
            };
            state.level.materials[i].bg = rgb;
        }
        _ => return false,
    }
    state.mark_dirty();
    // Every stepper path this replaces schedules a rebuild (`entity::adjust_property`,
    // `material::adjust_field`); without it a colour committed through the overlay
    // leaves the 3D preview showing the old one until an unrelated edit.
    state.schedule_preview_rebuild();
    true
}

/// Every DISTINCT colour already present in the level, sorted ascending.
///
/// Sweeps every colour-bearing field in the document — node, part and pipe
/// colours, both decal ends, every material's fg/bg, and all three emitter
/// ramp stops — so the colour overlay can offer the level's own vocabulary
/// back to the author. Sorted (rather than in discovery order) so the row is
/// stable across edits and the tests are deterministic.
#[must_use]
pub fn used_colors(level: &re_level::Level) -> Vec<u32> {
    let mut v: Vec<u32> = Vec::new();
    let mut push = |c: u32| v.push(c & 0xFF_FFFF);
    for n in &level.nodes {
        push(n.color_rgb);
    }
    for p in &level.parts {
        push(p.color_rgb);
    }
    for p in &level.pipes {
        push(p.color_rgb);
    }
    for d in &level.decals {
        push(d.color_head);
        push(d.color_tail);
    }
    for m in &level.materials {
        push(m.fg);
        push(m.bg);
    }
    for e in &level.emitters {
        for &c in &e.colors {
            push(c);
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

// ── opening an editor ───────────────────────────────────────────────────────

/// Open the editor appropriate to the row `action` names, focusing the field
/// first. Returns `true` when an overlay opened (a `Bool` toggles in place and
/// a `Text` row has no editor, so both return `false`).
///
/// This is the shared entry point for BOTH `Enter` on a focused row and a click
/// on a row's widget, so the two can never diverge.
pub fn open_for(state: &mut EditorState, action: RowAction) -> bool {
    let Some(value) = value_of(state, action) else {
        return false;
    };
    // Exactly ONE row editor is open at a time. `focus.rs` walks its stack in a
    // fixed order while `main::quit_on_esc` closes whichever `rowedit::cancel`
    // finds first; enforcing single-occupancy here is what keeps that
    // divergence unobservable instead of merely unlikely.
    state.num_edit = None;
    state.enum_pick = None;
    state.color_edit = None;
    focus(state, action);
    let label = label_of(action);
    match value {
        RowValue::Int { v, min, max, .. } => {
            let buf = v.to_string();
            state.num_edit = Some(NumEdit {
                target: action,
                label,
                caret: crate::widget::caret_at_end(&buf),
                buf,
                bounds: NumBounds::Int { min, max },
            });
            true
        }
        RowValue::Float { min, max, .. } => {
            let buf = value.display();
            state.num_edit = Some(NumEdit {
                target: action,
                label,
                caret: crate::widget::caret_at_end(&buf),
                buf,
                bounds: NumBounds::Float {
                    min: f64::from(min),
                    max: f64::from(max),
                },
            });
            true
        }
        RowValue::Enum {
            current,
            mut options,
        } => {
            if options.is_empty() {
                state.status = format!("{label}: no options");
                return false;
            }
            // THE POPUP MUST NOT LIE. It used to `unwrap_or(0)`, so a value the
            // option list did not carry silently highlighted row 0 — the author was
            // shown a different value from the one the level holds, and `Enter` on
            // the highlight then WROTE that value. Every option list now splices the
            // live value in (`crate::vocab`), so this is unreachable; when it is
            // reached anyway (a row whose readout is a `"-"`/`"(none)"` sentinel, or
            // a future list that forgets), the true value is APPENDED and selected
            // rather than misrepresented.
            let sel = if let Some(i) = options.iter().position(|o| *o == current) {
                i
            } else {
                options.push(current);
                options.len() - 1
            };
            state.enum_pick = Some(EnumPick {
                target: action,
                label,
                options,
                sel,
            });
            true
        }
        RowValue::Color(c) => {
            let buf = format!("{:06X}", c & 0xFF_FFFF);
            let used = used_colors(&state.level);
            let pal_sel = crate::material::nearest_palette(c, 1);
            state.color_edit = Some(ColorEdit {
                target: action,
                label,
                caret: crate::widget::caret_at_end(&buf),
                buf,
                original: c & 0xFF_FFFF,
                used_sel: used.iter().position(|&u| u == c & 0xFF_FFFF).unwrap_or(0),
                used,
                focus: ColorFocus::Hex,
                pal_sel,
            });
            true
        }
        RowValue::Bool(_) | RowValue::Text(_) => false,
    }
}

/// Toggle a `Bool` row in place as ONE undo step, by running the field's own
/// `c` cycle (so the checkbox and the `c` key are literally the same code path).
pub fn toggle_bool(state: &mut EditorState, action: RowAction) {
    focus(state, action);
    state.push_undo_checkpoint();
    cycle_field(state, action);
    state.drop_undo_if_unchanged();
}

/// Run a `[-]`/`[+]` stepper press as ONE undo step — identical to the `-`/`+`
/// keys, which is why it reuses their stepper rather than reimplementing it.
pub fn step_click(state: &mut EditorState, action: RowAction, delta: i32) {
    focus(state, action);
    state.push_undo_checkpoint();
    step_field(state, action, delta);
    state.drop_undo_if_unchanged();
}

// ── the NUMERIC editor ──────────────────────────────────────────────────────

/// Keys the inline numeric editor consumes. Digits, `.` (floats only) and a
/// leading `-` type; Backspace/Delete/arrows/Home/End navigate through the
/// shared `widget` ops; `Enter` commits; `Esc` cancels.
///
/// Bare digits are deliberately NOT bound at the root — `1`-`6`, `9` and `0`
/// are geometry brushes there — so this editor is only ever reachable through
/// `Enter` on a focused numeric row or a click on its value.
pub fn num_key(state: &mut EditorState, key: EditorKey) {
    let Some(mut ed) = state.num_edit.take() else {
        return;
    };
    match key.code {
        K::Enter => {
            commit_num(state, &ed);
            return;
        }
        K::Backspace => {
            crate::widget::backspace(&mut ed.buf, &mut ed.caret);
        }
        K::Delete => {
            crate::widget::delete(&mut ed.buf, &mut ed.caret);
        }
        K::Left => crate::widget::move_left(&ed.buf, &mut ed.caret),
        K::Right => crate::widget::move_right(&ed.buf, &mut ed.caret),
        K::Home => crate::widget::home(&ed.buf, &mut ed.caret),
        K::End => crate::widget::end(&ed.buf, &mut ed.caret),
        K::Char(c) if c.is_ascii_digit() || c == '-' || (c == '.' && ed.bounds.is_float()) => {
            crate::widget::insert_char(&mut ed.buf, &mut ed.caret, c);
        }
        _ => {}
    }
    state.num_edit = Some(ed);
}

/// Parse, CLAMP into the row's bounds, and write — one undo step. A buffer that
/// does not parse is REFUSED (the editor stays open with a status hint) rather
/// than committing a garbage value into the level.
///
/// INT ROWS NEVER TOUCH `f64`. `RowValue::Int` is `i128` precisely so a `u64`
/// emitter seed is representable; routing the commit through `f64` would throw
/// that away again at 53 bits, committing a DIFFERENT value than the one typed
/// and displayed — and, because the direct path reports `got == want`, claiming
/// it landed exactly. Integers stay in integer space end to end; `f64` is used
/// only where the row really is a float.
fn commit_num(state: &mut EditorState, ed: &NumEdit) {
    let typed = ed.buf.trim();
    state.push_undo_checkpoint();
    let msg = if let NumBounds::Float { min, max } = ed.bounds {
        let Ok(raw) = typed.parse::<f64>() else {
            state.drop_undo_if_unchanged();
            state.status = format!("{}: '{typed}' is not a number", ed.label);
            state.num_edit = Some(ed.clone());
            return;
        };
        let want = f64::from(crate::rowvalue::clamp_float(
            raw as f32, min as f32, max as f32,
        ));
        let got = set_numeric(state, ed.target, want);
        if (got - want).abs() < 1e-4 {
            format!("{} = {}", ed.label, trim_num(got))
        } else {
            format!("{} = {} (nearest reachable)", ed.label, trim_num(got))
        }
    } else {
        let NumBounds::Int { min, max } = ed.bounds else {
            unreachable!("bounds are Int or Float")
        };
        // An integer row parses as `i128`, so a 20-digit seed survives verbatim.
        let Ok(raw) = typed.parse::<i128>() else {
            state.drop_undo_if_unchanged();
            state.status = format!("{}: '{typed}' is not a whole number", ed.label);
            state.num_edit = Some(ed.clone());
            return;
        };
        let want = crate::rowvalue::clamp_int(raw, min, max);
        if set_numeric_direct(state, ed.target, want) {
            format!("{} = {want}", ed.label)
        } else {
            let got = set_numeric(state, ed.target, want as f64);
            let landed = current_int(state, ed.target).unwrap_or(got as i128);
            if landed == want {
                format!("{} = {want}", ed.label)
            } else {
                format!("{} = {landed} (nearest reachable)", ed.label)
            }
        }
    };
    state.drop_undo_if_unchanged();
    state.status = msg;
}

/// The row's current value as an exact `i128`, when it is an `Int` row.
fn current_int(state: &EditorState, action: RowAction) -> Option<i128> {
    match value_of(state, action)? {
        RowValue::Int { v, .. } => Some(v),
        _ => None,
    }
}

/// Render a committed number without a trailing `.0` on a whole value.
fn trim_num(v: f64) -> String {
    if (v - v.round()).abs() < 1e-6 {
        format!("{}", v.round() as i64)
    } else {
        format!("{v:.3}")
    }
}

// ── the ENUM popup ──────────────────────────────────────────────────────────

/// Keys the enum option popup consumes: Up/Down (and PgUp/PgDn) move the
/// highlight, `Enter` picks, `Esc` cancels.
pub fn enum_key(state: &mut EditorState, key: EditorKey) {
    let Some(mut p) = state.enum_pick.take() else {
        return;
    };
    let last = p.options.len().saturating_sub(1);
    match key.code {
        K::Enter => {
            // `open_for` refuses to open an empty popup, but this is a `pub fn`
            // and must not index on that assumption holding.
            let Some(want) = p.options.get(p.sel.min(last)).cloned() else {
                state.status = format!("{}: no options", p.label);
                return;
            };
            state.push_undo_checkpoint();
            let ok = set_enum(state, p.target, &want);
            state.drop_undo_if_unchanged();
            state.status = if ok {
                format!("{} = {want}", p.label)
            } else {
                format!("{}: could not reach '{want}'", p.label)
            };
            return;
        }
        K::Up => p.sel = p.sel.saturating_sub(1),
        K::Down => p.sel = (p.sel + 1).min(last),
        K::PageUp => p.sel = p.sel.saturating_sub(8),
        K::PageDown => p.sel = (p.sel + 8).min(last),
        _ => {}
    }
    state.enum_pick = Some(p);
}

// ── the COLOUR overlay ──────────────────────────────────────────────────────

/// Keys the colour overlay consumes. `Tab` moves between the hex field, the
/// palette grid and the used-colour row; within a grid the arrows move the
/// highlight; `Enter` commits whatever the focused sub-widget selects; `Esc`
/// cancels and leaves the original colour untouched.
pub fn color_key(state: &mut EditorState, key: EditorKey) {
    let Some(mut e) = state.color_edit.take() else {
        return;
    };
    match key.code {
        K::Enter => {
            let picked = match e.focus {
                ColorFocus::Hex => {
                    let Some(c) = crate::rowvalue::parse_hex(&e.buf) else {
                        state.status =
                            format!("{}: '#{}' is not a 6-digit hex colour", e.label, e.buf);
                        state.color_edit = Some(e);
                        return;
                    };
                    c
                }
                ColorFocus::Palette => {
                    crate::material::PALETTE[e.pal_sel.min(crate::material::PALETTE.len() - 1)]
                }
                ColorFocus::Used => {
                    let Some(&c) = e.used.get(e.used_sel) else {
                        state.status = format!("{}: no used colours", e.label);
                        state.color_edit = Some(e);
                        return;
                    };
                    c
                }
            };
            commit_color(state, &e, picked);
            return;
        }
        K::Tab => {
            e.focus = match e.focus {
                ColorFocus::Hex => ColorFocus::Palette,
                ColorFocus::Palette if !e.used.is_empty() => ColorFocus::Used,
                _ => ColorFocus::Hex,
            };
        }
        K::Backspace if e.focus == ColorFocus::Hex => {
            crate::widget::backspace(&mut e.buf, &mut e.caret);
        }
        K::Delete if e.focus == ColorFocus::Hex => {
            crate::widget::delete(&mut e.buf, &mut e.caret);
        }
        K::Home if e.focus == ColorFocus::Hex => crate::widget::home(&e.buf, &mut e.caret),
        K::End if e.focus == ColorFocus::Hex => crate::widget::end(&e.buf, &mut e.caret),
        K::Left | K::Right | K::Up | K::Down => move_color_sel(&mut e, key.code),
        // A typed hex digit always goes to the hex field, and pulls focus there
        // — typing is unambiguous intent, and hunting for Tab first is friction.
        K::Char(c) if c.is_ascii_hexdigit() => {
            if e.focus != ColorFocus::Hex {
                e.focus = ColorFocus::Hex;
            }
            if e.buf.chars().count() < 6 {
                crate::widget::insert_char(&mut e.buf, &mut e.caret, c.to_ascii_uppercase());
            }
        }
        _ => {}
    }
    state.color_edit = Some(e);
}

/// Arrow-key movement inside the colour overlay: the hex field moves its caret,
/// the two swatch grids move their highlight (wrapping by row).
fn move_color_sel(e: &mut ColorEdit, code: K) {
    match e.focus {
        ColorFocus::Hex => {
            // Up/Down mean nothing in a single-line field; only the horizontal
            // arrows move the caret.
            if code == K::Left {
                crate::widget::move_left(&e.buf, &mut e.caret);
            } else if code == K::Right {
                crate::widget::move_right(&e.buf, &mut e.caret);
            }
        }
        ColorFocus::Palette => {
            let n = crate::material::PALETTE.len();
            e.pal_sel = grid_move(e.pal_sel, n, code);
        }
        ColorFocus::Used => {
            let n = e.used.len();
            if n > 0 {
                e.used_sel = grid_move(e.used_sel, n, code);
            }
        }
    }
}

/// Move a selection index inside a [`SWATCH_COLS`]-wide grid of `n` swatches.
/// Left/Right step by one and clamp; Up/Down move a whole row.
#[must_use]
pub fn grid_move(sel: usize, n: usize, code: K) -> usize {
    if n == 0 {
        return 0;
    }
    let last = n - 1;
    match code {
        K::Left => sel.saturating_sub(1),
        K::Right => (sel + 1).min(last),
        K::Up => sel.saturating_sub(SWATCH_COLS),
        K::Down => (sel + SWATCH_COLS).min(last),
        _ => sel,
    }
}

/// Commit a colour as ONE undo step and close the overlay.
fn commit_color(state: &mut EditorState, e: &ColorEdit, rgb: u32) {
    state.push_undo_checkpoint();
    let ok = set_color(state, e.target, rgb);
    state.drop_undo_if_unchanged();
    state.status = if ok {
        format!("{} = #{:06X}", e.label, rgb & 0xFF_FFFF)
    } else {
        format!("{}: no object selected", e.label)
    };
}

/// Close whichever row editor is open, writing nothing. Returns `true` when one
/// was open (so the caller can CONSUME the Esc).
///
/// Cancellation lives here rather than in the three key handlers because
/// `main::quit_on_esc` owns the editor's Esc close-precedence chain — the same
/// reason `focus.rs` documents that Esc is not routed through the focus stack.
/// Keeping one entry point makes the behaviour independent of which system sees
/// the keypress first: whoever gets there closes it, and the other finds
/// nothing to do instead of falling through to the quit guard.
pub fn cancel(state: &mut EditorState) -> bool {
    if let Some(ed) = state.num_edit.take() {
        state.status = format!("{}: cancelled", ed.label);
        return true;
    }
    if let Some(p) = state.enum_pick.take() {
        state.status = format!("{}: cancelled", p.label);
        return true;
    }
    if let Some(e) = state.color_edit.take() {
        state.status = format!("{}: cancelled (#{:06X})", e.label, e.original);
        return true;
    }
    false
}

// ── geometry (shared by the painters in `chrome` and the hit-tests in `mouse`) ─

/// The centred box of an overlay `pw` × `ph` inside a `w` × `h` framebuffer, or
/// `None` when it does not fit — matching the existing modal geometry helpers in
/// `chrome.rs` so every overlay clips the same way on a small terminal.
#[must_use]
pub fn centred_box(
    w: usize,
    h: usize,
    pw: usize,
    ph: usize,
) -> Option<(usize, usize, usize, usize)> {
    let pw = pw.min(w);
    let ph = ph.min(h);
    if pw < 12 || ph < 4 || w < 12 || h < 4 {
        return None;
    }
    Some(((w - pw) / 2, (h - ph) / 2, pw, ph))
}

/// The NUMERIC editor's box: `(x0, y0, pw, ph)`.
#[must_use]
pub fn num_box(state: &EditorState, w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let ed = state.num_edit.as_ref()?;
    let want = (ed.label.chars().count() + 24).max(34);
    centred_box(w, h, want, 7)
}

/// The ENUM popup's box: `(x0, y0, pw, ph)`. Height grows with the option count
/// up to a cap, so a 2-option enum does not paint a tall empty list and a
/// 192-material list still fits and scrolls.
#[must_use]
pub fn enum_box(state: &EditorState, w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let p = state.enum_pick.as_ref()?;
    let widest = p
        .options
        .iter()
        .map(|o| o.chars().count())
        .max()
        .unwrap_or(0);
    let want_w = (widest + 8).max(p.label.chars().count() + 12).max(24);
    let want_h = (p.options.len() + 5).min(h.saturating_sub(2)).max(7);
    centred_box(w, h, want_w, want_h)
}

/// How many option rows the enum popup shows inside a box of height `ph`.
#[must_use]
pub fn enum_visible(ph: usize) -> usize {
    ph.saturating_sub(5).max(1)
}

/// The resolved geometry of the COLOUR overlay: the box, its inner text
/// column, and the absolute row each sub-widget occupies.
///
/// THE geometry SSOT for that overlay — `chrome::draw_color_edit` paints from
/// it and `mouse.rs` hit-tests against it, so a clicked swatch is always the
/// swatch drawn under the pointer.
///
/// EVERY row, including the hint line and the box height itself, comes from the
/// one running cursor in [`color_layout`]. That is deliberate: the first cut of
/// this overlay derived the section rows here but computed the hint row
/// separately in the painter as `rect.3 - 2`, and the two expressions collided
/// exactly — the hint was painted over the last row of used-colour swatches,
/// which the hit-test still resolved to a colour commit. A click on what read
/// as help text silently wrote a colour into the level. Deriving the height
/// FROM the cursor rather than the cursor from the height makes that class of
/// off-by-one unrepresentable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ColorLayout {
    /// The overlay box as `(x0, y0, pw, ph)`.
    pub rect: (usize, usize, usize, usize),
    /// Column the inner content starts at.
    pub x: usize,
    /// Row of the `#RRGGBB` hex entry field.
    pub hex_y: usize,
    /// Row of the `palette` section caption.
    pub pal_label_y: usize,
    /// Row the palette swatch grid starts on.
    pub pal_y: usize,
    /// Rows the palette grid occupies.
    pub pal_rows: usize,
    /// Row of the `used in this level` caption, when that section is shown.
    pub used_label_y: Option<usize>,
    /// Row the used-colour grid starts on, when shown.
    pub used_y: Option<usize>,
    /// Rows the used-colour grid occupies (capped by [`MAX_USED_ROWS`]).
    pub used_rows: usize,
    /// How many used colours actually fit in those rows.
    pub used_shown: usize,
    /// Row of the key hint line.
    pub hint_y: usize,
}

/// Resolve the colour overlay's geometry, or `None` when it cannot be drawn.
///
/// One cursor walks down the box and every row falls out of it in order; the
/// box height is whatever the cursor reached. Nothing downstream may compute a
/// row from `rect.3`.
#[must_use]
pub fn color_layout(state: &EditorState, w: usize, h: usize) -> Option<ColorLayout> {
    let edit = state.color_edit.as_ref()?;
    let want_w = (SWATCH_COLS * SWATCH_W + 8).max(edit.label.chars().count() + 16);
    let pal_rows = grid_rows(crate::material::PALETTE.len());
    let used_shown = edit.used.len().min(SWATCH_COLS * MAX_USED_ROWS);
    let used_rows = grid_rows(used_shown);

    // Rows RELATIVE to the box top. 0 = top border, 1 = title.
    let mut dy = 2usize;
    let hex_dy = dy;
    dy += 2; // the hex field, then a blank separator
    let pal_label_dy = dy;
    dy += 1;
    let pal_dy = dy;
    dy += pal_rows;
    let (used_label_dy, used_dy) = if used_rows > 0 {
        let l = dy;
        dy += 1;
        let g = dy;
        dy += used_rows;
        (Some(l), Some(g))
    } else {
        (None, None)
    };
    dy += 1; // blank separator above the hint
    let hint_dy = dy;
    dy += 1;
    let ph = dy + 1; // + the bottom border

    let rect = centred_box(w, h, want_w, ph)?;
    let (x0, y0, _pw, _ph) = rect;
    Some(ColorLayout {
        rect,
        x: x0 + 2,
        hex_y: y0 + hex_dy,
        pal_label_y: y0 + pal_label_dy,
        pal_y: y0 + pal_dy,
        pal_rows,
        used_label_y: used_label_dy.map(|d| y0 + d),
        used_y: used_dy.map(|d| y0 + d),
        used_rows,
        used_shown,
        hint_y: y0 + hint_dy,
    })
}

/// What a click at `(col, row)` inside the COLOUR overlay means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorHit {
    /// The hex entry field — move the keyboard there.
    HexField,
    /// A [`crate::material::PALETTE`] swatch by index.
    Palette(usize),
    /// A "used in this level" swatch by index.
    Used(usize),
}

/// Resolve a click inside the colour overlay to a [`ColorHit`], or `None` when
/// it landed on inert chrome. Reads the SAME [`color_layout`] the painter drew
/// from, and the same [`grid_hit`] inverse of [`grid_cell`].
#[must_use]
pub fn color_hit(
    state: &EditorState,
    w: usize,
    h: usize,
    col: usize,
    row: usize,
) -> Option<ColorHit> {
    let l = color_layout(state, w, h)?;
    let e = state.color_edit.as_ref()?;
    if col < l.x {
        return None;
    }
    let dx = col - l.x;
    if row == l.hex_y {
        return Some(ColorHit::HexField);
    }
    if row >= l.pal_y && row < l.pal_y + l.pal_rows {
        return grid_hit(dx, row - l.pal_y, crate::material::PALETTE.len()).map(ColorHit::Palette);
    }
    if let Some(uy) = l.used_y {
        if row >= uy && row < uy + l.used_rows {
            return grid_hit(dx, row - uy, l.used_shown.min(e.used.len())).map(ColorHit::Used);
        }
    }
    None
}

/// Apply a colour-overlay click.
///
/// A click on a SWATCH commits it and closes the overlay: a pointer landing on
/// a specific colour is unambiguous intent, and the write is one undo step, so
/// making the user then reach for Enter would be friction with no safety gain.
/// A click on the hex field just moves the keyboard there.
pub fn apply_color_hit(state: &mut EditorState, hit: ColorHit) {
    let Some(e) = state.color_edit.clone() else {
        return;
    };
    match hit {
        ColorHit::HexField => {
            if let Some(cur) = state.color_edit.as_mut() {
                cur.focus = ColorFocus::Hex;
            }
        }
        ColorHit::Palette(i) => {
            let Some(&c) = crate::material::PALETTE.get(i) else {
                return;
            };
            state.color_edit = None;
            commit_color(state, &e, c);
        }
        ColorHit::Used(i) => {
            let Some(&c) = e.used.get(i) else {
                return;
            };
            state.color_edit = None;
            commit_color(state, &e, c);
        }
    }
}

/// Rows of "used in this level" swatches the overlay will show at most, so a
/// level with 234 distinct colours cannot paint a box taller than the terminal.
pub const MAX_USED_ROWS: usize = 4;

/// Rows a [`SWATCH_COLS`]-wide grid of `n` swatches needs.
#[must_use]
pub const fn grid_rows(n: usize) -> usize {
    n.div_ceil(SWATCH_COLS)
}

/// The `(col, row)` cell offset of swatch `i` within a grid, relative to the
/// grid's top-left. THE geometry both the swatch painter and the swatch
/// hit-test use, so a click always lands on the swatch drawn under it.
#[must_use]
pub const fn grid_cell(i: usize) -> (usize, usize) {
    ((i % SWATCH_COLS) * SWATCH_W, i / SWATCH_COLS)
}

/// Resolve a click at grid-relative `(dx, dy)` back to a swatch index, or
/// `None` when it fell in a gap or past the last swatch. The exact inverse of
/// [`grid_cell`].
#[must_use]
pub fn grid_hit(dx: usize, dy: usize, n: usize) -> Option<usize> {
    let col = dx / SWATCH_W;
    // The trailing gap column of each swatch is not part of it.
    if col >= SWATCH_COLS || dx % SWATCH_W >= SWATCH_W - 1 {
        return None;
    }
    let i = dy * SWATCH_COLS + col;
    (i < n).then_some(i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{EntityField, EntityKind};

    fn node_state() -> EditorState {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(EntityKind::Node);
        crate::entity::place_at_cursor(&mut s);
        s
    }

    fn key(c: char) -> EditorKey {
        EditorKey {
            code: K::Char(c),
            shift: false,
        }
    }

    fn code(c: K) -> EditorKey {
        EditorKey {
            code: c,
            shift: false,
        }
    }

    // ── the enum popup must show the TRUE current value ─────────────────────

    /// An OFF-LIST authored value opens the popup with THAT value highlighted —
    /// not row 0.
    ///
    /// The popup used to do `options.position(current).unwrap_or(0)`, so a node
    /// whose glyph the candidate list did not carry (`levels/01.lvl` authors
    /// `glyph = R`) opened with `N` highlighted and no indication. Pressing Enter
    /// on that highlight then WROTE `N`. Two things now prevent it: every option
    /// list splices the live value in (`crate::vocab`), and `open_for` appends +
    /// selects the true value if a list ever forgets.
    #[test]
    fn the_enum_popup_highlights_the_authored_value_even_when_it_is_off_list() {
        let mut s = node_state();
        let i = s.selected_node.expect("a node was placed");
        s.level.nodes[i].glyph = 'R'; // not on NODE_GLYPHS
        let action = RowAction::FocusEntity(EntityField::NodeGlyph);
        assert!(open_for(&mut s, action));
        let p = s.enum_pick.as_ref().expect("popup opened");
        assert_eq!(
            p.options[p.sel], "R",
            "the popup highlighted {:?} while the node holds 'R'",
            p.options[p.sel]
        );
    }

    /// And committing that highlight is a NO-OP, not a rewrite: `Enter` on the
    /// true current value leaves the level untouched.
    #[test]
    fn committing_the_popups_own_highlight_never_changes_the_value() {
        let mut s = node_state();
        let i = s.selected_node.expect("a node was placed");
        s.level.nodes[i].glyph = 'R';
        let action = RowAction::FocusEntity(EntityField::NodeGlyph);
        assert!(open_for(&mut s, action));
        enum_key(&mut s, code(K::Enter));
        assert_eq!(
            s.level.nodes[i].glyph, 'R',
            "Enter on the highlight rewrote the glyph"
        );
    }

    /// A row whose readout is a SENTINEL (`"-"` — nothing selected) still opens a
    /// popup that names what it is showing, rather than silently pointing at row 0.
    #[test]
    fn a_sentinel_readout_is_shown_rather_than_misrepresented() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.place = Some(EntityKind::Node); // no node selected -> readout is "-"
        let action = RowAction::FocusEntity(EntityField::NodeGlyph);
        assert!(open_for(&mut s, action));
        let p = s.enum_pick.as_ref().expect("popup opened");
        assert_eq!(
            p.options[p.sel], "-",
            "the popup must show the sentinel it is on"
        );
    }

    // ── numeric ─────────────────────────────────────────────────────────────

    /// The headline B2 fix: a value that took 199 presses to reach is typed.
    #[test]
    fn numeric_editor_types_a_value_the_stepper_would_take_hundreds_of_presses_to_reach() {
        let mut s = node_state();
        s.entity_field = EntityField::NodeThreshold;
        let act = RowAction::FocusEntity(EntityField::NodeThreshold);
        assert!(open_for(&mut s, act), "numeric row opens the inline editor");
        // Retype the buffer as 137.
        for _ in 0..8 {
            num_key(&mut s, code(K::Backspace));
        }
        for c in "137".chars() {
            num_key(&mut s, key(c));
        }
        num_key(&mut s, code(K::Enter));
        assert!(s.num_edit.is_none(), "Enter closed the editor");
        assert_eq!(s.level.nodes[0].threshold, 137);
    }

    /// Esc restores: nothing is written until commit.
    #[test]
    fn numeric_editor_esc_cancels_without_writing() {
        let mut s = node_state();
        let before = s.level.nodes[0].threshold;
        let act = RowAction::FocusEntity(EntityField::NodeThreshold);
        open_for(&mut s, act);
        for c in "99".chars() {
            num_key(&mut s, key(c));
        }
        assert!(cancel(&mut s), "Esc closes the editor");
        assert!(s.num_edit.is_none());
        assert_eq!(s.level.nodes[0].threshold, before, "cancel wrote nothing");
    }

    /// A commit CLAMPS into the row's own bounds instead of writing out of range.
    #[test]
    fn numeric_commit_clamps_into_the_rows_bounds() {
        let mut s = node_state();
        // Node x is clamped to the map extent (0..=15 on a 16-wide map).
        let act = RowAction::FocusEntity(EntityField::NodePosX);
        open_for(&mut s, act);
        for _ in 0..4 {
            num_key(&mut s, code(K::Backspace));
        }
        for c in "999".chars() {
            num_key(&mut s, key(c));
        }
        num_key(&mut s, code(K::Enter));
        assert_eq!(s.level.nodes[0].x, 15, "clamped to the map extent, not 999");
    }

    /// Junk is refused, not committed — the editor stays open to be fixed.
    #[test]
    fn numeric_commit_refuses_a_non_number() {
        let mut s = node_state();
        let before = s.level.nodes[0].threshold;
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeThreshold));
        for _ in 0..8 {
            num_key(&mut s, code(K::Backspace));
        }
        // 'z' is not accepted as input at all; an empty buffer then fails to parse.
        num_key(&mut s, key('z'));
        num_key(&mut s, code(K::Enter));
        assert!(s.num_edit.is_some(), "editor stayed open for a fix");
        assert_eq!(s.level.nodes[0].threshold, before, "nothing was written");
        // An INT row says "whole number" (a float row says "number") — the
        // message names the domain the row actually parses in.
        assert!(
            s.status.contains("is not a whole number"),
            "status explains: {}",
            s.status
        );
    }

    /// A typed commit is ONE undo step.
    #[test]
    fn numeric_commit_is_one_undo_step() {
        let mut s = node_state();
        let before = s.level.nodes[0].threshold;
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeThreshold));
        for c in "42".chars() {
            num_key(&mut s, key(c));
        }
        num_key(&mut s, code(K::Enter));
        let after = s.level.nodes[0].threshold;
        assert_ne!(after, before);
        s.undo();
        assert_eq!(
            s.level.nodes[0].threshold, before,
            "ONE undo restores the value"
        );
    }

    /// A specific 19-digit emitter SEED can be typed. The `+`/`-` stepper moves
    /// a `u64` seed by 1, which is impractical for entering a specific large
    /// value — this is the one case that bypasses the stepper.
    #[test]
    fn a_specific_emitter_seed_can_be_typed() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(EntityKind::Emitter);
        crate::entity::place_at_cursor(&mut s);
        let act = RowAction::FocusEmitter(EmitterField::Seed);
        let before = s.level.emitters[0].seed;
        assert!(open_for(&mut s, act), "seed opens the numeric editor");
        for _ in 0..30 {
            num_key(&mut s, code(K::Backspace));
        }
        // A 19-digit seed WELL above 2^53 — the whole reason `RowValue::Int` is
        // `i128` and the commit path never touches `f64`. Routed through an
        // `f64` this lands on a different number while still claiming success.
        let want: u64 = 1_234_567_890_123_456_789;
        assert!(
            want as f64 as u64 != want,
            "the test value is f64-lossy by construction"
        );
        for c in want.to_string().chars() {
            num_key(&mut s, key(c));
        }
        num_key(&mut s, code(K::Enter));
        assert!(s.num_edit.is_none(), "committed");
        assert_eq!(
            s.level.emitters[0].seed, want,
            "the typed seed survived verbatim"
        );
        assert!(
            !s.status.contains("nearest reachable"),
            "landed exactly: {}",
            s.status
        );
        assert!(
            s.status.contains(&want.to_string()),
            "status quotes it: {}",
            s.status
        );
        s.undo();
        assert_eq!(s.level.emitters[0].seed, before, "ONE undo step");
    }

    // ── bool ────────────────────────────────────────────────────────────────

    /// Clicking a checkbox toggles the flag, as one undo step.
    #[test]
    fn bool_toggle_flips_and_undoes_in_one_step() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(EntityKind::Trigger);
        crate::entity::place_at_cursor(&mut s);
        let act = RowAction::FocusEntity(EntityField::TrigOnce);
        let before = s.level.triggers[0].once;
        toggle_bool(&mut s, act);
        assert_ne!(s.level.triggers[0].once, before, "checkbox toggled");
        s.undo();
        assert_eq!(s.level.triggers[0].once, before, "ONE undo restores it");
    }

    // ── enum ────────────────────────────────────────────────────────────────

    /// The popup opens on the CURRENT option and picking writes it.
    #[test]
    fn enum_popup_opens_on_current_and_picks() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(EntityKind::Holo);
        crate::entity::place_at_cursor(&mut s);
        let act = RowAction::FocusEntity(EntityField::HoloShapeField);
        assert!(open_for(&mut s, act), "enum row opens the popup");
        let p = s.enum_pick.clone().expect("popup open");
        assert!(p.options.len() >= 2, "holo shape has options");
        assert_eq!(
            p.options[p.sel],
            crate::rowtype::entity_value(&s, EntityField::HoloShapeField).display(),
            "opens on the CURRENT option"
        );
        // Move to a different option and pick it.
        enum_key(&mut s, code(K::Down));
        let want = s.enum_pick.as_ref().unwrap().options[1].clone();
        enum_key(&mut s, code(K::Enter));
        assert!(s.enum_pick.is_none(), "Enter closed the popup");
        assert_eq!(
            crate::rowtype::entity_value(&s, EntityField::HoloShapeField).display(),
            want,
            "the picked option was written"
        );
    }

    /// Esc leaves the value alone.
    #[test]
    fn enum_popup_esc_cancels() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(EntityKind::Holo);
        crate::entity::place_at_cursor(&mut s);
        let before = crate::rowtype::entity_value(&s, EntityField::HoloShapeField).display();
        open_for(&mut s, RowAction::FocusEntity(EntityField::HoloShapeField));
        enum_key(&mut s, code(K::Down));
        assert!(cancel(&mut s), "Esc closes the popup");
        assert!(s.enum_pick.is_none());
        assert_eq!(
            crate::rowtype::entity_value(&s, EntityField::HoloShapeField).display(),
            before
        );
    }

    /// The highlight clamps at both ends — no panic, no wrap past the list.
    #[test]
    fn enum_popup_navigation_clamps() {
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(EntityKind::Holo);
        crate::entity::place_at_cursor(&mut s);
        open_for(&mut s, RowAction::FocusEntity(EntityField::HoloShapeField));
        for _ in 0..50 {
            enum_key(&mut s, code(K::Down));
        }
        let p = s.enum_pick.as_ref().unwrap();
        assert_eq!(p.sel, p.options.len() - 1, "clamped at the end");
        for _ in 0..50 {
            enum_key(&mut s, code(K::Up));
        }
        assert_eq!(s.enum_pick.as_ref().unwrap().sel, 0, "clamped at the start");
    }

    // ── colour ──────────────────────────────────────────────────────────────

    /// The whole point of STEP 3: an ARBITRARY colour can be typed back in.
    #[test]
    fn colour_overlay_commits_typed_hex() {
        let mut s = node_state();
        let act = RowAction::FocusEntity(EntityField::NodeColor);
        assert!(open_for(&mut s, act), "colour row opens the overlay");
        for _ in 0..8 {
            color_key(&mut s, code(K::Backspace));
        }
        for c in "C0FFEE".chars() {
            color_key(&mut s, key(c));
        }
        color_key(&mut s, code(K::Enter));
        assert!(s.color_edit.is_none(), "Enter closed the overlay");
        assert_eq!(s.level.nodes[0].color_rgb & 0xFF_FFFF, 0x00_C0FFEE);
    }

    /// An invalid hex is REFUSED — the level is never written a half-parsed value.
    #[test]
    fn colour_overlay_refuses_bad_hex() {
        let mut s = node_state();
        let before = s.level.nodes[0].color_rgb;
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        for _ in 0..8 {
            color_key(&mut s, code(K::Backspace));
        }
        for c in "ABC".chars() {
            color_key(&mut s, key(c));
        }
        color_key(&mut s, code(K::Enter));
        assert!(s.color_edit.is_some(), "overlay stayed open");
        assert_eq!(s.level.nodes[0].color_rgb, before, "nothing written");
    }

    /// Esc restores the original colour.
    #[test]
    fn colour_overlay_esc_cancels() {
        let mut s = node_state();
        let before = s.level.nodes[0].color_rgb;
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        for c in "FF00FF".chars() {
            color_key(&mut s, key(c));
        }
        assert!(cancel(&mut s), "Esc closes the overlay");
        assert!(s.color_edit.is_none());
        assert_eq!(s.level.nodes[0].color_rgb, before);
    }

    /// A palette swatch can be picked with the keyboard.
    #[test]
    fn colour_overlay_picks_a_palette_swatch() {
        let mut s = node_state();
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        // Tab from the hex field onto the palette grid, move, pick.
        color_key(&mut s, code(K::Tab));
        assert_eq!(s.color_edit.as_ref().unwrap().focus, ColorFocus::Palette);
        color_key(&mut s, code(K::Right));
        let want = crate::material::PALETTE[s.color_edit.as_ref().unwrap().pal_sel];
        color_key(&mut s, code(K::Enter));
        assert_eq!(s.level.nodes[0].color_rgb & 0xFF_FFFF, want & 0xFF_FFFF);
    }

    /// The "used in this level" row offers back a colour the level already has —
    /// the recovery path for a value the old stepper would have destroyed.
    #[test]
    fn colour_overlay_offers_the_levels_own_colours() {
        let mut s = node_state();
        // Author an off-palette colour on the node, then place a second node.
        s.level.nodes[0].color_rgb = 0x00_123456;
        s.cursor = (4, 4);
        crate::entity::place_at_cursor(&mut s);
        let used = used_colors(&s.level);
        assert!(
            used.contains(&0x00_123456),
            "the authored colour is offered back"
        );
        assert!(used.windows(2).all(|w| w[0] < w[1]), "sorted + deduped");
    }

    /// A colour commit is ONE undo step.
    #[test]
    fn colour_commit_is_one_undo_step() {
        let mut s = node_state();
        let before = s.level.nodes[0].color_rgb;
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        for _ in 0..8 {
            color_key(&mut s, code(K::Backspace));
        }
        for c in "10203040".chars().take(6) {
            color_key(&mut s, key(c));
        }
        color_key(&mut s, code(K::Enter));
        assert_ne!(s.level.nodes[0].color_rgb, before);
        s.undo();
        assert_eq!(s.level.nodes[0].color_rgb, before, "ONE undo restores it");
    }

    /// `set_color` covers every colour-bearing field and refuses anything else.
    #[test]
    fn set_color_covers_every_colour_field() {
        use crate::state::EmitterField as EF;
        let mut s = node_state();
        assert!(set_color(
            &mut s,
            RowAction::FocusEntity(EntityField::NodeColor),
            0x0011_1111
        ));
        assert_eq!(s.level.nodes[0].color_rgb, 0x0011_1111);
        // A non-colour field is refused, and writes nothing.
        assert!(!set_color(
            &mut s,
            RowAction::FocusEntity(EntityField::NodeThreshold),
            0x0022_2222
        ));
        assert!(!set_color(
            &mut s,
            RowAction::FocusEmitter(EF::Rate),
            0x0022_2222
        ));
        assert!(!set_color(&mut s, RowAction::None, 0x0022_2222));
    }

    // ── the COLOUR overlay's clickable grid ─────────────────────────────────

    /// Every palette swatch the overlay PAINTS is clickable at the position it
    /// was painted — the painter and the hit-test read one `color_layout`.
    #[test]
    fn every_painted_palette_swatch_is_clickable_where_it_is_drawn() {
        let mut s = node_state();
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        let (w, h) = (120usize, 40usize);
        let l = color_layout(&s, w, h).expect("overlay fits at 120x40");
        for i in 0..crate::material::PALETTE.len() {
            let (dx, dy) = grid_cell(i);
            assert_eq!(
                color_hit(&s, w, h, l.x + dx, l.pal_y + dy),
                Some(ColorHit::Palette(i)),
                "palette swatch {i}"
            );
        }
        // The hex field row resolves to the hex field.
        assert_eq!(color_hit(&s, w, h, l.x, l.hex_y), Some(ColorHit::HexField));
        // A row of inert chrome resolves to nothing.
        assert_eq!(
            color_hit(&s, w, h, l.x, l.rect.1),
            None,
            "the border is inert"
        );
    }

    /// Clicking a palette swatch COMMITS that colour and closes the overlay.
    #[test]
    fn clicking_a_palette_swatch_commits_it() {
        let mut s = node_state();
        let before = s.level.nodes[0].color_rgb;
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        apply_color_hit(&mut s, ColorHit::Palette(5));
        assert!(s.color_edit.is_none(), "the overlay closed");
        assert_eq!(
            s.level.nodes[0].color_rgb & 0xFF_FFFF,
            crate::material::PALETTE[5]
        );
        s.undo();
        assert_eq!(s.level.nodes[0].color_rgb, before, "ONE undo step");
    }

    /// Clicking a "used in this level" swatch commits that colour — the
    /// recovery path for an authored value the old stepper would have snapped
    /// to white.
    #[test]
    fn clicking_a_used_colour_restores_it() {
        let mut s = node_state();
        // An off-palette authored colour lives on a SECOND node.
        s.cursor = (5, 5);
        crate::entity::place_at_cursor(&mut s);
        s.level.nodes[1].color_rgb = 0x00_C0FFEE;
        s.selected_node = Some(0);
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        let idx = s
            .color_edit
            .as_ref()
            .unwrap()
            .used
            .iter()
            .position(|&c| c == 0x00_C0FFEE)
            .expect("the authored colour is offered back");
        apply_color_hit(&mut s, ColorHit::Used(idx));
        assert_eq!(s.level.nodes[0].color_rgb & 0xFF_FFFF, 0x00_C0FFEE);
    }

    /// Clicking the hex field only moves the keyboard there — it never commits.
    #[test]
    fn clicking_the_hex_field_only_moves_focus() {
        let mut s = node_state();
        let before = s.level.nodes[0].color_rgb;
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        color_key(&mut s, code(K::Tab)); // onto the palette
        apply_color_hit(&mut s, ColorHit::HexField);
        assert_eq!(
            s.color_edit.as_ref().expect("still open").focus,
            ColorFocus::Hex
        );
        assert_eq!(s.level.nodes[0].color_rgb, before, "nothing was committed");
    }

    /// The overlay's used-colour section is CAPPED, so a level with hundreds of
    /// distinct colours cannot paint a box taller than the terminal.
    #[test]
    fn used_colour_section_is_capped_so_the_box_always_fits() {
        let mut s = node_state();
        // 300 distinct colours, far more than the section can show.
        for i in 0..300u32 {
            s.cursor = ((i % 15) as usize, (i / 15) as usize % 15);
            crate::entity::place_at_cursor(&mut s);
            let n = s.level.nodes.len() - 1;
            s.level.nodes[n].color_rgb = 0x0001_0000 + i * 37;
        }
        s.selected_node = Some(0);
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        assert!(
            s.color_edit.as_ref().unwrap().used.len() > 100,
            "many colours"
        );
        let l = color_layout(&s, 80, 24).expect("still fits at 80x24");
        assert!(l.used_rows <= MAX_USED_ROWS, "used section capped");
        assert!(
            l.rect.1 + l.rect.3 <= 24,
            "the box stays inside the terminal: {:?}",
            l.rect
        );
    }

    /// `set_numeric` reaches a distant target through the EXISTING stepper —
    /// the mechanism the inline editor relies on — for a field whose step is
    /// not 1 (`max_particles` steps by 16).
    #[test]
    fn set_numeric_reaches_a_distant_target_through_a_coarse_stepper() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(EntityKind::Emitter);
        crate::entity::place_at_cursor(&mut s);
        let act = RowAction::FocusEmitter(EmitterField::MaxParticles);
        focus(&mut s, act);
        assert_eq!(s.level.emitters[0].max_particles, 256, "the default");
        let got = set_numeric(&mut s, act, 4096.0);
        assert!((got - 4096.0).abs() < 1e-6, "landed on 4096, got {got}");
        assert_eq!(s.level.emitters[0].max_particles, 4096);
    }

    /// `set_numeric` stops cleanly at a clamp instead of spinning.
    #[test]
    fn set_numeric_stops_at_the_stepper_clamp() {
        let mut s = node_state();
        let act = RowAction::FocusEntity(EntityField::NodePosX);
        focus(&mut s, act);
        let got = set_numeric(&mut s, act, 10_000.0);
        assert!(
            (got - 15.0).abs() < 1e-6,
            "pinned at the map extent, got {got}"
        );
        assert_eq!(s.level.nodes[0].x, 15);
    }

    /// B1 REGRESSION — the test that would have caught the overpainting.
    ///
    /// The earlier `every_painted_palette_swatch_is_clickable_where_it_is_drawn`
    /// compared `color_hit` against `grid_cell` WITHOUT ever rendering, so it
    /// could not see that the hint line was painted over the last row of
    /// used-colour swatches while the hit-test still resolved that row to a
    /// colour commit — a click on what read as help text silently wrote a
    /// colour into the level.
    ///
    /// This renders for real and asserts the two agree CELL BY CELL: wherever
    /// `color_hit` says `Used(i)` / `Palette(i)` lives, the framebuffer must
    /// actually carry that swatch's block glyph in that swatch's own colour.
    #[test]
    fn every_swatch_the_overlay_paints_survives_to_the_framebuffer() {
        for n_used in [1usize, 3, 7, 13, 24, 60] {
            let mut st = node_state();
            // Author `n_used` distinct colours so the used section takes 1..4 rows.
            for idx in 0..n_used {
                st.cursor = (idx % 15, idx / 15);
                crate::entity::place_at_cursor(&mut st);
                let last = st.level.nodes.len() - 1;
                st.level.nodes[last].color_rgb = 0x0001_0000 + (idx as u32) * 0x0001_0101;
            }
            st.selected_node = Some(0);
            open_for(&mut st, RowAction::FocusEntity(EntityField::NodeColor));
            let (w, h) = (100usize, 40usize);
            let l = color_layout(&st, w, h).expect("overlay fits");
            let used_list = st.color_edit.as_ref().expect("open").used.clone();

            let mut fb = re_core::Framebuffer::new(w, h);
            crate::chrome::draw_color_edit(&st, &mut fb, w, h);

            let block = '\u{2588}' as u32;
            // Every USED swatch the hit-test claims is really drawn there.
            for (i, &want_rgb) in used_list.iter().enumerate().take(l.used_shown) {
                let (dx, dy) = grid_cell(i);
                let (cx, cy) = (l.x + dx, l.used_y.expect("used section shown") + dy);
                assert_eq!(
                    color_hit(&st, w, h, cx, cy),
                    Some(ColorHit::Used(i)),
                    "n_used={n_used}: used swatch {i} is clickable at ({cx},{cy})"
                );
                let cell = fb.get(cx, cy).expect("inside the framebuffer");
                assert_eq!(
                    cell.glyph,
                    block,
                    "n_used={n_used}: used swatch {i} at ({cx},{cy}) was OVERPAINTED with {:?}",
                    char::from_u32(cell.glyph)
                );
                assert_eq!(
                    cell.fg & 0xFF_FFFF,
                    want_rgb & 0xFF_FFFF,
                    "n_used={n_used}: used swatch {i} shows the wrong colour"
                );
            }
            // And every PALETTE swatch likewise.
            for i in 0..crate::material::PALETTE.len() {
                let (dx, dy) = grid_cell(i);
                let (cx, cy) = (l.x + dx, l.pal_y + dy);
                assert_eq!(color_hit(&st, w, h, cx, cy), Some(ColorHit::Palette(i)));
                let cell = fb.get(cx, cy).expect("inside the framebuffer");
                assert_eq!(
                    cell.glyph, block,
                    "n_used={n_used}: palette swatch {i} overpainted"
                );
                assert_eq!(cell.fg & 0xFF_FFFF, crate::material::PALETTE[i] & 0xFF_FFFF);
            }
        }
    }

    /// The hint line has a row of its own, below every swatch, and clicking it
    /// commits NOTHING. (The bug: `hint_y` and the last used-swatch row were the
    /// same cell for every palette size and every terminal size.)
    #[test]
    fn the_hint_line_never_shares_a_row_with_a_swatch() {
        for n_used in [1usize, 6, 7, 12, 25, 60] {
            let mut s = node_state();
            for i in 0..n_used {
                s.cursor = (i % 15, i / 15);
                crate::entity::place_at_cursor(&mut s);
                let k = s.level.nodes.len() - 1;
                s.level.nodes[k].color_rgb = 0x0002_0000 + (i as u32) * 0x0001_0101;
            }
            s.selected_node = Some(0);
            open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
            let (w, h) = (100usize, 40usize);
            let l = color_layout(&s, w, h).expect("overlay fits");

            let last_used = l
                .used_y
                .map_or(l.pal_y + l.pal_rows - 1, |uy| uy + l.used_rows - 1);
            assert!(
                l.hint_y > last_used,
                "n_used={n_used}: hint row {} collides with the last swatch row {last_used}",
                l.hint_y
            );
            assert!(
                l.hint_y < l.rect.1 + l.rect.3 - 1,
                "n_used={n_used}: hint row is on the bottom border"
            );
            // A click anywhere along the hint row is inert.
            for cx in l.x..(l.rect.0 + l.rect.2) {
                assert_eq!(
                    color_hit(&s, w, h, cx, l.hint_y),
                    None,
                    "n_used={n_used}: clicking the hint line at col {cx} would commit a colour"
                );
            }
        }
    }

    /// The selection marker sits in the swatch's own gap cell, so it can never
    /// overpaint another swatch, a caption, or the border.
    #[test]
    fn the_selection_marker_never_overpaints_a_swatch() {
        let mut s = node_state();
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        if let Some(e) = s.color_edit.as_mut() {
            e.focus = ColorFocus::Palette;
            e.pal_sel = 0;
        }
        let (w, h) = (100usize, 40usize);
        let l = color_layout(&s, w, h).expect("fits");
        let mut fb = re_core::Framebuffer::new(w, h);
        crate::chrome::draw_color_edit(&s, &mut fb, w, h);
        // Swatch 1 sits immediately after swatch 0's gap; it must be intact.
        let (dx, dy) = grid_cell(1);
        let cell = fb.get(l.x + dx, l.pal_y + dy).expect("inside");
        assert_eq!(
            cell.glyph, '\u{2588}' as u32,
            "the marker ate the next swatch"
        );
        assert_eq!(cell.fg & 0xFF_FFFF, crate::material::PALETTE[1] & 0xFF_FFFF);
        // The marker really is drawn, in swatch 0's gap column.
        let (g0, _) = grid_cell(0);
        let marker = fb.get(l.x + g0 + SWATCH_W - 1, l.pal_y).expect("inside");
        assert_eq!(marker.glyph, '<' as u32, "the selection marker is drawn");
    }

    /// An overlay commit must refresh the 3D preview, exactly
    /// like the stepper paths it bypasses (`entity::adjust_property`,
    /// `material::adjust_field`). Without it a colour or seed committed through
    /// an overlay leaves the preview showing the OLD value until some unrelated
    /// edit happens to schedule a rebuild.
    #[test]
    fn an_overlay_commit_schedules_a_preview_rebuild() {
        use crate::state::EmitterField;
        // Colour.
        let mut s = node_state();
        s.preview_rebuild_delay = None;
        assert!(set_color(
            &mut s,
            RowAction::FocusEntity(EntityField::NodeColor),
            0x00AB_CDEF
        ));
        assert!(
            s.preview_rebuild_delay.is_some(),
            "a colour commit schedules a preview rebuild"
        );

        // Seed (the direct numeric path).
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(EntityKind::Emitter);
        crate::entity::place_at_cursor(&mut s);
        let act = RowAction::FocusEmitter(EmitterField::Seed);
        focus(&mut s, act);
        let before = s.level.emitters[0].seed;
        s.preview_rebuild_delay = None;
        s.dirty = false;
        assert!(set_numeric_direct(&mut s, act, i128::from(before) + 1));
        assert!(
            s.preview_rebuild_delay.is_some(),
            "a seed commit schedules a rebuild"
        );
        assert!(s.dirty, "and marks the document dirty");
    }

    /// Committing an UNMODIFIED buffer is a no-op: Enter on a row the user only
    /// looked at must not dirty the document or churn the preview.
    #[test]
    fn committing_an_unchanged_value_does_not_dirty_the_document() {
        use crate::state::EmitterField;
        let mut s = EditorState::new_empty(16, 16, 4);
        s.cursor = (2, 2);
        s.z = 1;
        s.place = Some(EntityKind::Emitter);
        crate::entity::place_at_cursor(&mut s);
        let act = RowAction::FocusEmitter(EmitterField::Seed);
        assert!(open_for(&mut s, act), "seed opens the editor");
        let before = s.level.emitters[0].seed;
        s.dirty = false;
        s.preview_rebuild_delay = None;
        // Enter with the buffer exactly as it was seeded.
        num_key(&mut s, code(K::Enter));
        assert_eq!(s.level.emitters[0].seed, before, "unchanged");
        assert!(!s.dirty, "an unchanged commit does not dirty the document");
        assert!(s.preview_rebuild_delay.is_none(), "nor churn the preview");
    }

    /// Only ONE row editor is ever open — opening a second closes the first.
    #[test]
    fn opening_a_row_editor_closes_any_other() {
        let mut s = node_state();
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        assert!(s.color_edit.is_some());
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeThreshold));
        assert!(s.num_edit.is_some(), "the numeric editor opened");
        assert!(s.color_edit.is_none(), "and the colour overlay closed");
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeIo));
        assert!(s.enum_pick.is_some());
        assert!(
            s.num_edit.is_none() && s.color_edit.is_none(),
            "single occupancy"
        );
        // So one Esc always leaves no editor behind.
        assert!(cancel(&mut s));
        assert!(s.num_edit.is_none() && s.enum_pick.is_none() && s.color_edit.is_none());
    }

    /// An empty option list can never be indexed, even though `open_for` guards
    /// against opening one — `enum_key` is `pub` and must stand on its own.
    #[test]
    fn an_empty_option_popup_never_panics() {
        let mut s = node_state();
        s.enum_pick = Some(EnumPick {
            target: RowAction::FocusEntity(EntityField::NodeIo),
            label: "io".into(),
            options: Vec::new(),
            sel: 0,
        });
        enum_key(&mut s, code(K::Enter));
        assert!(s.enum_pick.is_none(), "it bails instead of indexing");
        assert!(
            s.status.contains("no options"),
            "and says why: {}",
            s.status
        );
    }

    // ── grid geometry ───────────────────────────────────────────────────────

    /// The swatch grid's painter and hit-test are exact inverses.
    #[test]
    fn swatch_grid_hit_inverts_the_layout() {
        let n = crate::material::PALETTE.len();
        for i in 0..n {
            let (dx, dy) = grid_cell(i);
            assert_eq!(grid_hit(dx, dy, n), Some(i), "swatch {i} round-trips");
            // The second cell of the 2-wide swatch also hits it.
            assert_eq!(
                grid_hit(dx + 1, dy, n),
                Some(i),
                "swatch {i} is 2 cells wide"
            );
            // The trailing gap belongs to nothing.
            assert_eq!(grid_hit(dx + 2, dy, n), None, "gap after swatch {i}");
        }
        // Past the last swatch.
        assert_eq!(grid_hit(0, 99, n), None);
    }

    #[test]
    fn grid_rows_wrap_at_the_column_count() {
        assert_eq!(grid_rows(0), 0);
        assert_eq!(grid_rows(1), 1);
        assert_eq!(grid_rows(SWATCH_COLS), 1);
        assert_eq!(grid_rows(SWATCH_COLS + 1), 2);
        assert_eq!(grid_rows(12), 2);
    }

    #[test]
    fn grid_move_clamps_at_both_ends() {
        assert_eq!(grid_move(0, 12, K::Left), 0);
        assert_eq!(grid_move(11, 12, K::Right), 11);
        assert_eq!(grid_move(0, 12, K::Up), 0);
        assert_eq!(grid_move(11, 12, K::Down), 11);
        assert_eq!(grid_move(0, 12, K::Down), SWATCH_COLS);
        assert_eq!(grid_move(SWATCH_COLS, 12, K::Up), 0);
        assert_eq!(grid_move(0, 0, K::Right), 0, "empty grid is safe");
    }

    /// Every overlay declines to draw rather than panicking on a tiny terminal.
    #[test]
    fn overlays_degrade_on_a_tiny_framebuffer() {
        let mut s = node_state();
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeColor));
        assert!(color_layout(&s, 10, 4).is_none(), "too narrow to draw");
        assert!(color_layout(&s, 80, 24).is_some(), "fits at 80x24");
        s.color_edit = None;
        open_for(&mut s, RowAction::FocusEntity(EntityField::NodeThreshold));
        assert!(num_box(&s, 8, 3).is_none());
        assert!(num_box(&s, 80, 24).is_some(), "fits at 80x24");
    }
}
