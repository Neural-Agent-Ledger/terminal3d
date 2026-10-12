//! VALIDATION DRAWER: a read-only, toggleable
//! drawer listing the LAST SAVE's full `re_lvlcheck` report — every check with
//! its severity, not just the one-line verdict the status bar shows. The lines
//! are captured by `io::save_lvl_text` into `EditorState::last_check`; this
//! module owns the toggle, the scroll, the painter, and the mouse hit-test
//! (one shared geometry — painter/hit-test SSOT).
//!
//! JUMP-TO-FINDING is deliberately ABSENT: `re_lvlcheck::Check` carries no
//! object/coordinate reference to jump to.
use crate::state::EditorState;
use re_core::{text::draw_str, Cell, Framebuffer};
use re_present::{EditorKey, EditorKeyCode as E};

/// Toggle the drawer (the `V` chord / palette action).
pub fn toggle(state: &mut EditorState) {
    state.validation_open = !state.validation_open;
    state.validation_scroll = 0;
    state.status = if state.validation_open {
        if state.last_check.is_empty() {
            "validation: no report yet — save (s) to lint".into()
        } else {
            "validation report (Up/Dn scroll, V/Esc close)".into()
        }
    } else {
        "validation drawer closed".into()
    };
}

/// The drawer's report lines: the captured last-save report, or a hint line
/// when no save has run yet. Shared by the painter and the scroll clamp.
pub fn report_lines(state: &EditorState) -> Vec<String> {
    if state.last_check.is_empty() {
        vec!["no report yet — save (s) to run lvlcheck".to_string()]
    } else {
        state.last_check.clone()
    }
}

/// Focus-frame key handling while the drawer is open. Returns `true` when the
/// key was CONSUMED (scroll / close); everything else falls through to the
/// chain below — the drawer is a NON-BLOCKING overlay, editing keeps working.
pub fn key(state: &mut EditorState, key: EditorKey) -> bool {
    if !state.validation_open {
        return false;
    }
    let total = report_lines(state).len();
    match key.code {
        E::Up => {
            state.validation_scroll = state.validation_scroll.saturating_sub(1);
            true
        }
        E::Down => {
            state.validation_scroll = (state.validation_scroll + 1).min(total.saturating_sub(1));
            true
        }
        E::PageUp => {
            state.validation_scroll = state.validation_scroll.saturating_sub(8);
            true
        }
        E::PageDown => {
            state.validation_scroll = (state.validation_scroll + 8).min(total.saturating_sub(1));
            true
        }
        E::Char('V' | 'q') => {
            toggle(state);
            true
        }
        _ => false,
    }
}

/// Geometry of the drawer box: `(x0, y0, pw, ph)` — a right-side drawer over
/// the body (below the toolbar, above the status row), or `None` when the
/// framebuffer can't host it. THE shared source for the painter and the mouse
/// hit-test.
pub fn box_geom(w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    if w < 30 || h < 8 {
        return None;
    }
    let pw = (w * 2 / 5).clamp(28, 60).min(w);
    let ph = h.saturating_sub(3); // toolbar row + status row + one margin
    let x0 = w - pw;
    let y0 = 1;
    Some((x0, y0, pw, ph))
}

/// The `[X]` close-button span on the drawer's top border: `(row, x0, x1)`.
pub fn close_button(w: usize, h: usize) -> Option<(usize, usize, usize)> {
    let (x0, y0, pw, _) = box_geom(w, h)?;
    let bx = x0 + pw.saturating_sub(4);
    Some((y0, bx, bx + 3))
}

/// `true` when `(col, row)` lands inside the open drawer (the click is the
/// drawer's — never grid paint under it).
pub fn hit(w: usize, h: usize, col: usize, row: usize) -> bool {
    box_geom(w, h)
        .is_some_and(|(x0, y0, pw, ph)| col >= x0 && col < x0 + pw && row >= y0 && row < y0 + ph)
}

/// Severity colour for a report line (keyed off its `PASS`/`WARN`/`FAIL` tag).
fn line_fg(line: &str) -> u32 {
    if line.starts_with("FAIL") {
        0x00_FF6060
    } else if line.starts_with("WARN") {
        0x00_FFC040
    } else if line.starts_with("PASS") {
        0x00_60D080
    } else {
        0x00_C8D4E0
    }
}

/// Paint the drawer: bordered box, verdict header, one row per check line
/// (scrolled), ▲/▼ overflow markers. Read-only — rows are not clickable.
pub fn draw(state: &EditorState, fb: &mut Framebuffer, w: usize, h: usize) {
    if !state.validation_open {
        return;
    }
    let Some((x0, y0, pw, ph)) = box_geom(w, h) else {
        return;
    };
    let bg = 0x00_0C1018;
    let border = 0x00_77AADD;
    for ry in 0..ph {
        for rx in 0..pw {
            let edge = rx == 0 || ry == 0 || rx == pw - 1 || ry == ph - 1;
            fb.set(
                x0 + rx,
                y0 + ry,
                Cell {
                    glyph: if edge { '#' as u32 } else { ' ' as u32 },
                    fg: border,
                    bg,
                    ..Cell::default()
                },
            );
        }
    }
    draw_str(fb, x0 + 2, y0, " VALIDATION (last save) ", border, bg);
    if let Some((brow, bx, _)) = close_button(w, h) {
        draw_str(fb, bx, brow, "[X]", 0x00_FFC0C0, bg);
    }
    let lines = report_lines(state);
    let inner_h = ph.saturating_sub(3);
    let inner_w = pw.saturating_sub(4);
    let scroll = state.validation_scroll.min(lines.len().saturating_sub(1));
    for (i, line) in lines.iter().skip(scroll).take(inner_h).enumerate() {
        let shown: String = line.chars().take(inner_w).collect();
        draw_str(fb, x0 + 2, y0 + 1 + i, &shown, line_fg(line), bg);
    }
    // Overflow markers on the right border.
    if scroll > 0 {
        draw_str(fb, x0 + pw - 1, y0 + 1, "\u{25B2}", border, bg);
    }
    if scroll + inner_h < lines.len() {
        draw_str(
            fb,
            x0 + pw - 1,
            y0 + ph.saturating_sub(2),
            "\u{25BC}",
            border,
            bg,
        );
    }
    let hint: String = "Up/Dn scroll  V close".chars().take(inner_w).collect();
    draw_str(
        fb,
        x0 + 2,
        y0 + ph.saturating_sub(2),
        &hint,
        0x00_8899AA,
        bg,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_of(code: E) -> EditorKey {
        EditorKey { code, shift: false }
    }

    #[test]
    fn toggle_opens_and_closes_with_scroll_reset() {
        let mut s = EditorState::new_empty(4, 4, 2);
        toggle(&mut s);
        assert!(s.validation_open);
        s.validation_scroll = 5;
        toggle(&mut s);
        assert!(!s.validation_open);
        toggle(&mut s);
        assert_eq!(s.validation_scroll, 0, "reopen starts at the top");
    }

    #[test]
    fn drawer_consumes_only_its_keys() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.validation_open = true;
        s.last_check = (0..20).map(|i| format!("PASS check{i} — ok")).collect();
        assert!(key(&mut s, key_of(E::Down)), "Down scrolls");
        assert_eq!(s.validation_scroll, 1);
        assert!(key(&mut s, key_of(E::Up)), "Up scrolls back");
        assert_eq!(s.validation_scroll, 0);
        assert!(
            !key(&mut s, key_of(E::Char(' '))),
            "space is NOT consumed — the drawer never blocks editing"
        );
        assert!(key(&mut s, key_of(E::Char('V'))), "V closes");
        assert!(!s.validation_open);
    }

    #[test]
    fn scroll_clamps_to_report_length() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.validation_open = true;
        s.last_check = vec!["PASS a".into(), "WARN b".into()];
        for _ in 0..10 {
            key(&mut s, key_of(E::Down));
        }
        assert_eq!(s.validation_scroll, 1, "scroll clamps to the last line");
    }

    #[test]
    fn draw_lists_report_lines_and_close_button_hits() {
        let mut s = EditorState::new_empty(4, 4, 2);
        s.validation_open = true;
        s.last_check = vec![
            "verdict: WARN".into(),
            "PASS notes — 0 note placards".into(),
            "WARN thresholds — node t out of range".into(),
        ];
        let (w, h) = (120usize, 40usize);
        let mut fb = Framebuffer::new(w, h);
        draw(&s, &mut fb, w, h);
        let (x0, y0, pw, ph) = box_geom(w, h).unwrap();
        let grab_row = |ry: usize| -> String {
            (x0..x0 + pw)
                .filter_map(|x| fb.get(x, y0 + ry).and_then(|c| char::from_u32(c.glyph)))
                .collect()
        };
        assert!(grab_row(1).contains("verdict: WARN"));
        assert!(grab_row(2).contains("notes"), "notes check listed");
        assert!(
            grab_row(3).contains("thresholds"),
            "thresholds check listed"
        );
        // The [X] close button sits on the top border inside the box.
        let (brow, bx, be) = close_button(w, h).unwrap();
        assert!(hit(w, h, bx, brow) && be <= x0 + pw);
        assert!(
            hit(w, h, x0 + 1, y0 + 1),
            "clicks inside the drawer are the drawer's"
        );
        assert!(
            !hit(w, h, x0.saturating_sub(2), y0 + 1),
            "outside the drawer misses"
        );
        let _ = ph;
    }
}
