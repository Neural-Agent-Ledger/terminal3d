//! Backend-neutral raw-key + mouse events for the level editor (`ts_editor`).
//! Gated on the `editor` feature; keeps terminal/bevy_ratatui details inside
//! re_present.
use bevy::prelude::*;

/// A backend-neutral key, mapped from the active terminal backend's raw events.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct EditorKey {
    pub code: EditorKeyCode,
    pub shift: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditorKeyCode {
    Up,
    Down,
    Left,
    Right,
    PageUp,
    PageDown,
    Enter,
    Esc,
    Tab,
    /// Delete the character to the LEFT of the cursor (standard Backspace
    /// semantics). Mapped from `crossterm::KeyCode::Backspace`.
    Backspace,
    /// Delete the character AT the cursor (forward-delete). Mapped from
    /// `crossterm::KeyCode::Delete`. Drives `ts_editor::widget::delete`.
    Delete,
    /// Move the caret to the start of its line. Mapped from
    /// `crossterm::KeyCode::Home`. Drives `ts_editor::widget::home`.
    Home,
    /// Move the caret to the end of its line. Mapped from
    /// `crossterm::KeyCode::End`. Drives `ts_editor::widget::end`.
    End,
    /// A printable character.
    ///
    /// For ASCII letters (`c.is_ascii_alphabetic()`), the char's case is derived
    /// **exclusively from the Shift modifier**, not from `CapsLock`:
    ///
    /// - SHIFT modifier set   → uppercase letter (e.g. `'S'`), `shift = true`
    /// - SHIFT modifier clear → lowercase letter (e.g. `'s'`), `shift = false`
    ///
    /// `CapsLock` is therefore invisible: a terminal delivering `'S'` without the
    /// SHIFT modifier (`CapsLock` only) is normalised to `Char('s')` + `shift=false`
    /// — the unshifted key. The editor's `Char('S')` match arms therefore fire
    /// only when the user truly presses Shift+S, on every terminal and regardless
    /// of `CapsLock` state.
    ///
    /// Non-letter printable characters (digits, punctuation, space, …) are
    /// forwarded exactly as crossterm delivers them; `shift` reflects the modifier.
    Char(char),
    /// A function key, `F1`..`F(u8::MAX)`, mapped verbatim from crossterm's
    /// `KeyCode::F(u8)`. The editor uses these for chorded commands (e.g. `F2`
    /// rename, `F5` playtest); `shift` on the enclosing [`EditorKey`] reflects
    /// the SHIFT modifier exactly as delivered (function keys are not subject
    /// to the letter-case normalisation above — there is no natural case for
    /// `F5`).
    F(u8),
}

/// Which mouse button an [`EditorMouse`] event carries. Mirrors crossterm's
/// `MouseButton` so the editor can branch (paint vs. erase, say) without
/// depending on crossterm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditorMouseButton {
    Left,
    Right,
    Middle,
}

/// The kind of mouse interaction, flattened from crossterm's `MouseEventKind`
/// into the four the editor cares about. Button-bearing kinds (`Press`/`Drag`/
/// `Release`) carry the button in [`EditorMouse::button`]; the scroll kinds set
/// it to `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditorMouseKind {
    /// A button went down at this cell.
    Press,
    /// The cursor moved with a button held (a paint stroke).
    Drag,
    /// A button was released at this cell.
    Release,
    /// Wheel scrolled up (away from the user).
    ScrollUp,
    /// Wheel scrolled down (toward the user).
    ScrollDown,
}

/// A backend-neutral mouse event, mapped from the active terminal backend's raw
/// `MouseEvent`. `col`/`row` are absolute terminal cells (the same coordinate
/// space the editor's `layout::compute` and `chrome::draw_grid` paint into), so
/// the consumer can hit-test panes and invert the grid projection directly.
///
/// Plain `Moved` events (no button held) are NOT surfaced — they carry no edit
/// intent and would flood the channel; only the five [`EditorMouseKind`] cases
/// are emitted.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct EditorMouse {
    /// Absolute terminal column the event occurred on.
    pub col: u16,
    /// Absolute terminal row the event occurred on.
    pub row: u16,
    pub kind: EditorMouseKind,
    /// The button for `Press`/`Drag`/`Release`; `None` for scroll.
    pub button: Option<EditorMouseButton>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn editorkey_is_constructible() {
        let k = EditorKey {
            code: EditorKeyCode::Char('w'),
            shift: false,
        };
        assert_eq!(k.code, EditorKeyCode::Char('w'));
    }

    #[test]
    fn f_key_variant_is_constructible_and_distinct() {
        let k = EditorKey {
            code: EditorKeyCode::F(2),
            shift: false,
        };
        assert_eq!(k.code, EditorKeyCode::F(2));
        assert_ne!(k.code, EditorKeyCode::F(5));
        assert_ne!(k.code, EditorKeyCode::Char('2'));
    }

    #[test]
    fn backspace_variant_exists_and_is_distinct() {
        // Smoke-test: Backspace is a real, matchable variant, not a Char.
        let k = EditorKey {
            code: EditorKeyCode::Backspace,
            shift: false,
        };
        assert_eq!(k.code, EditorKeyCode::Backspace);
        assert_ne!(k.code, EditorKeyCode::Char('\x08')); // not a raw byte alias
    }

    #[test]
    fn char_letter_contract_case_from_shift() {
        // shift=false → Char holds the lowercase letter.
        let k = EditorKey {
            code: EditorKeyCode::Char('s'),
            shift: false,
        };
        assert!(matches!(k.code, EditorKeyCode::Char('s')));
        assert!(!k.shift);
        // shift=true → Char holds the uppercase letter (Shift+S).
        // The editor's `Char('S')` match arms rely on this.
        let k2 = EditorKey {
            code: EditorKeyCode::Char('S'),
            shift: true,
        };
        assert!(matches!(k2.code, EditorKeyCode::Char('S')));
        assert!(k2.shift);
    }

    #[test]
    fn editormouse_is_constructible() {
        let m = EditorMouse {
            col: 5,
            row: 7,
            kind: EditorMouseKind::Press,
            button: Some(EditorMouseButton::Left),
        };
        assert_eq!(m.col, 5);
        assert_eq!(m.kind, EditorMouseKind::Press);
        assert_eq!(m.button, Some(EditorMouseButton::Left));
    }
}
