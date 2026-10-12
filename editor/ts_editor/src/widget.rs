//! Reusable TEXT-EDITING primitives: a real caret over a `String`.
//!
//! The editor's text fields keep their content in the DOCUMENT (a note's body,
//! a node's pool words, a trigger's event label) — there is no shadow buffer to
//! desynchronize. What the widget adds is a [`Caret`] and a set of PURE
//! operations over `(&mut String, &mut Caret)`: mid-string insert/delete,
//! newline handling, and line/column navigation. Callers that want a
//! single-line field simply never call [`insert_newline`] (every other op
//! degrades naturally to one line).
//!
//! Coordinates are `(line, column)` in CHARACTERS (not bytes) so multi-byte
//! glyphs move as one unit. Every operation clamps the caret into the text
//! first, so a stale caret (the selection changed under it) can never panic or
//! splice at a wrong byte index.

/// Caret position within a text: `line` (0-based, `\n`-separated) and `col`
/// (0-based CHAR offset within that line; `col == line_len` means "after the
/// last char").
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Caret {
    /// 0-based line index.
    pub line: usize,
    /// 0-based char column within the line (may equal the line length).
    pub col: usize,
}

/// The text's lines. `split('\n')` yields at least one (possibly empty) line,
/// so every text has a valid caret position.
fn lines(text: &str) -> Vec<&str> {
    text.split('\n').collect()
}

/// Char length of one line.
fn line_len(line: &str) -> usize {
    line.chars().count()
}

/// Clamp `caret` into `text`: line into the line count, column into that
/// line's char length.
#[must_use]
pub fn clamp(text: &str, caret: Caret) -> Caret {
    let ls = lines(text);
    let line = caret.line.min(ls.len() - 1);
    let col = caret.col.min(line_len(ls[line]));
    Caret { line, col }
}

/// The caret position after the LAST char of `text` (where a fresh focus
/// lands, matching the old append-only behaviour).
#[must_use]
pub fn caret_at_end(text: &str) -> Caret {
    let ls = lines(text);
    Caret {
        line: ls.len() - 1,
        col: line_len(ls.last().unwrap_or(&"")),
    }
}

/// The BYTE offset of `caret` within `text` (for splicing). Assumes a clamped
/// caret.
fn byte_offset(text: &str, caret: Caret) -> usize {
    let mut off = 0usize;
    for (i, l) in lines(text).iter().enumerate() {
        if i == caret.line {
            return off + l.char_indices().nth(caret.col).map_or(l.len(), |(b, _)| b);
        }
        off += l.len() + 1; // + the '\n'
    }
    text.len()
}

/// Insert one char at the caret (mid-string capable) and advance the caret
/// past it.
pub fn insert_char(text: &mut String, caret: &mut Caret, c: char) {
    *caret = clamp(text, *caret);
    let at = byte_offset(text, *caret);
    text.insert(at, c);
    if c == '\n' {
        caret.line += 1;
        caret.col = 0;
    } else {
        caret.col += 1;
    }
}

/// Insert a NEWLINE at the caret, splitting the current line (multi-line
/// text-area behaviour — Enter).
pub fn insert_newline(text: &mut String, caret: &mut Caret) {
    insert_char(text, caret, '\n');
}

/// Delete the char BEFORE the caret (Backspace). At a line start this joins
/// the line onto the previous one. Returns `true` when the text changed.
pub fn backspace(text: &mut String, caret: &mut Caret) -> bool {
    *caret = clamp(text, *caret);
    if caret.line == 0 && caret.col == 0 {
        return false;
    }
    // The caret target after deletion is the position of the removed char.
    let before = if caret.col > 0 {
        Caret {
            line: caret.line,
            col: caret.col - 1,
        }
    } else {
        let ls = lines(text);
        Caret {
            line: caret.line - 1,
            col: line_len(ls[caret.line - 1]),
        }
    };
    let at = byte_offset(text, before);
    text.remove(at);
    *caret = before;
    true
}

/// Delete the char AT the caret (Delete). At a line end this joins the next
/// line up. Returns `true` when the text changed. Bound to
/// `EditorKeyCode::Delete` in `entity::text_entry_key`.
pub fn delete(text: &mut String, caret: &mut Caret) -> bool {
    *caret = clamp(text, *caret);
    let at = byte_offset(text, *caret);
    if at >= text.len() {
        return false;
    }
    text.remove(at);
    true
}

/// Move the caret one char left (wrapping to the previous line end).
pub fn move_left(text: &str, caret: &mut Caret) {
    *caret = clamp(text, *caret);
    if caret.col > 0 {
        caret.col -= 1;
    } else if caret.line > 0 {
        caret.line -= 1;
        caret.col = line_len(lines(text)[caret.line]);
    }
}

/// Move the caret one char right (wrapping to the next line start).
pub fn move_right(text: &str, caret: &mut Caret) {
    *caret = clamp(text, *caret);
    let ls = lines(text);
    if caret.col < line_len(ls[caret.line]) {
        caret.col += 1;
    } else if caret.line + 1 < ls.len() {
        caret.line += 1;
        caret.col = 0;
    }
}

/// Move the caret one line up (column clamped into the shorter line).
pub fn move_up(text: &str, caret: &mut Caret) {
    *caret = clamp(text, *caret);
    if caret.line > 0 {
        caret.line -= 1;
        *caret = clamp(text, *caret);
    }
}

/// Move the caret one line down (column clamped into the shorter line).
pub fn move_down(text: &str, caret: &mut Caret) {
    *caret = clamp(text, *caret);
    if caret.line + 1 < lines(text).len() {
        caret.line += 1;
        *caret = clamp(text, *caret);
    }
}

/// Move the caret to the start of its line (Home). Bound to
/// `EditorKeyCode::Home` in `entity::text_entry_key`.
pub fn home(text: &str, caret: &mut Caret) {
    *caret = clamp(text, *caret);
    caret.col = 0;
}

/// Move the caret to the end of its line (End). Bound to
/// `EditorKeyCode::End` in `entity::text_entry_key`.
pub fn end(text: &str, caret: &mut Caret) {
    *caret = clamp(text, *caret);
    caret.col = line_len(lines(text)[caret.line]);
}

/// Render ONE line of `text` — the caret's line — with a visible `‸` caret
/// marker spliced in at the caret column, plus a `line i/N` suffix when the
/// text is multi-line. The inspector row readout for a focused text field.
#[must_use]
pub fn caret_line_readout(text: &str, caret: Caret) -> String {
    let caret = clamp(text, caret);
    let ls = lines(text);
    let l = ls[caret.line];
    let mut shown = String::with_capacity(l.len() + 3);
    for (i, ch) in l.chars().enumerate() {
        if i == caret.col {
            shown.push('\u{2038}'); // ‸
        }
        shown.push(ch);
    }
    if caret.col >= line_len(l) {
        shown.push('\u{2038}');
    }
    if ls.len() > 1 {
        format!("{shown}  (line {}/{})", caret.line + 1, ls.len())
    } else {
        shown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_mid_string_and_advance() {
        let mut t = String::from("hlo");
        let mut c = Caret { line: 0, col: 1 };
        insert_char(&mut t, &mut c, 'e');
        assert_eq!(t, "helo");
        assert_eq!(c, Caret { line: 0, col: 2 });
        insert_char(&mut t, &mut c, 'l');
        assert_eq!(t, "hello");
        assert_eq!(c, Caret { line: 0, col: 3 });
    }

    #[test]
    fn newline_splits_line_and_moves_caret() {
        let mut t = String::from("ab");
        let mut c = Caret { line: 0, col: 1 };
        insert_newline(&mut t, &mut c);
        assert_eq!(t, "a\nb");
        assert_eq!(c, Caret { line: 1, col: 0 });
        // Typing lands on the new line.
        insert_char(&mut t, &mut c, 'x');
        assert_eq!(t, "a\nxb");
    }

    #[test]
    fn backspace_mid_line_and_line_join() {
        let mut t = String::from("ab\ncd");
        let mut c = Caret { line: 1, col: 1 };
        assert!(backspace(&mut t, &mut c));
        assert_eq!(t, "ab\nd");
        assert_eq!(c, Caret { line: 1, col: 0 });
        // At line start: join onto the previous line, caret at the join point.
        assert!(backspace(&mut t, &mut c));
        assert_eq!(t, "abd");
        assert_eq!(c, Caret { line: 0, col: 2 });
        // At the very start: nothing to remove.
        c = Caret { line: 0, col: 0 };
        assert!(!backspace(&mut t, &mut c));
        assert_eq!(t, "abd");
    }

    #[test]
    fn delete_at_caret_and_join_next_line() {
        let mut t = String::from("ab\ncd");
        let mut c = Caret { line: 0, col: 2 };
        assert!(delete(&mut t, &mut c), "deletes the newline");
        assert_eq!(t, "abcd");
        let mut c2 = caret_at_end(&t);
        assert!(!delete(&mut t, &mut c2), "nothing after the end");
    }

    #[test]
    fn arrow_navigation_wraps_and_clamps() {
        let t = String::from("long line\nab");
        let mut c = Caret { line: 0, col: 0 };
        move_left(&t, &mut c);
        assert_eq!(c, Caret { line: 0, col: 0 }, "left at origin is a no-op");
        end(&t, &mut c);
        assert_eq!(c.col, 9);
        move_right(&t, &mut c);
        assert_eq!(c, Caret { line: 1, col: 0 }, "right at line end wraps down");
        move_left(&t, &mut c);
        assert_eq!(c, Caret { line: 0, col: 9 }, "left at line start wraps up");
        // Down clamps the column into the shorter line.
        move_down(&t, &mut c);
        assert_eq!(c, Caret { line: 1, col: 2 });
        move_up(&t, &mut c);
        assert_eq!(c, Caret { line: 0, col: 2 });
        home(&t, &mut c);
        assert_eq!(c.col, 0);
    }

    #[test]
    fn clamp_handles_stale_carets_and_multibyte() {
        let t = String::from("héllo\nwörld");
        let c = clamp(&t, Caret { line: 9, col: 99 });
        assert_eq!(
            c,
            Caret { line: 1, col: 5 },
            "clamped to the last line end (chars)"
        );
        // Insert after a multi-byte char splices at the char boundary.
        let mut t2 = String::from("hé");
        let mut c2 = Caret { line: 0, col: 2 };
        insert_char(&mut t2, &mut c2, '!');
        assert_eq!(t2, "hé!");
    }

    #[test]
    fn caret_at_end_lands_after_last_char() {
        assert_eq!(caret_at_end(""), Caret { line: 0, col: 0 });
        assert_eq!(caret_at_end("ab"), Caret { line: 0, col: 2 });
        assert_eq!(caret_at_end("ab\ncde"), Caret { line: 1, col: 3 });
    }

    #[test]
    fn readout_marks_caret_and_line_position() {
        assert_eq!(
            caret_line_readout("ab", Caret { line: 0, col: 1 }),
            "a\u{2038}b"
        );
        assert_eq!(
            caret_line_readout("ab\ncd", Caret { line: 1, col: 2 }),
            "cd\u{2038}  (line 2/2)"
        );
    }
}
